"""Evaluate the experimental M4.0 agent route with real model requests.

The planner and structured adapter are exercised through the ASGI boundary, while
the glossary adapter receives the configured ``OpenAICompatibleModel``. Requests
are deliberately made one at a time so latency and provider failures remain
attributable to a single manifest case.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig, load_provider_config
from scripts.tracing import configured_tracer, flush_tracer

CASES = ROOT / "metadata/m4_0_live_agent_cases_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"
DEFAULT_TOTAL_TIMEOUT_SECONDS = 15.0
RUNTIME_CODES = frozenset({
    "provider_failed", "model_timeout", "model_unconfigured", "database_unavailable",
    "tool_timeout", "tool_failed", "evidence_timeout", "knowledge_unavailable",
    "knowledge_unconfigured", "knowledge_provider_failed", "adapter_failed",
    "agent_timeout",
})


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9,
        )
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _same(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def validate_live_cases(document: Mapping[str, Any], *, source_snapshot: str) -> list[dict[str, Any]]:
    """Validate a manifest without executing a model or opening a database."""
    if document.get("source_snapshot") != source_snapshot:
        raise ValueError("M4 live cases source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("M4 live case set must contain cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("M4 live case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"invalid or duplicate M4 case ID: {case_id}")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"M4 case {case_id} has no question")
        if case.get("category") not in {"answerable", "clarification", "unsupported"}:
            raise ValueError(f"invalid category for {case_id}")
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"M4 case {case_id} has no expected contract")
        if expected.get("route") not in {"structured", "mixed", "clarification", "unsupported"}:
            raise ValueError(f"invalid expected route for {case_id}")
        if case["category"] == "answerable" and not isinstance(expected.get("structured"), dict):
            raise ValueError(f"answerable M4 case {case_id} has no structured oracle")
        if case["category"] != "answerable" and expected.get("no_executed_steps") is not True:
            raise ValueError(f"refusal M4 case {case_id} must require no executed steps")
    return cases


def select_live_cases(cases: Sequence[dict[str, Any]], case_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    """Select requested IDs while preserving manifest order."""
    if case_ids is None:
        return list(cases)
    requested = list(dict.fromkeys(case_ids))
    if not requested:
        raise ValueError("case selection must contain at least one case ID")
    available = {case["case_id"] for case in cases}
    unknown = [case_id for case_id in requested if case_id not in available]
    if unknown:
        raise ValueError(f"unknown M4 case ID(s): {', '.join(unknown)}")
    wanted = set(requested)
    return [case for case in cases if case["case_id"] in wanted]


def _reason_codes(body: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    if isinstance(body.get("reason_code"), str):
        codes.append(body["reason_code"])
    for step in body.get("steps", []) if isinstance(body.get("steps"), list) else []:
        if not isinstance(step, Mapping):
            continue
        result = step.get("result")
        if isinstance(result, Mapping) and isinstance(result.get("reason_code"), str):
            codes.append(result["reason_code"])
        if isinstance(step.get("error"), str):
            text = step["error"].lower()
            if "timeout" in text:
                codes.append("tool_timeout")
            if "provider" in text or "model" in text:
                codes.append("provider_failed")
    return codes


def classify_failure(body: Mapping[str, Any], *, http_status: int, exception: BaseException | None = None) -> str | None:
    """Classify provider/runtime failures before capability scoring."""
    if exception is not None or http_status >= 500:
        return "provider_runtime"
    if any(code in RUNTIME_CODES for code in _reason_codes(body)):
        return "provider_runtime"
    if not isinstance(body.get("status"), str) or not isinstance(body.get("route"), str):
        return "evaluation_contract"
    return None


def _step(body: Mapping[str, Any], capability: str) -> Mapping[str, Any] | None:
    for step in body.get("steps", []) if isinstance(body.get("steps"), list) else []:
        if isinstance(step, Mapping) and step.get("capability") == capability:
            return step
    return None


def _structured_evidence_ok(body: Mapping[str, Any]) -> bool:
    step = _step(body, "player_efg")
    evidence = step.get("evidence") if step else None
    structured = evidence.get("structured") if isinstance(evidence, Mapping) else None
    return isinstance(structured, Mapping) and all(
        bool(structured.get(key)) for key in ("source_sql", "formula", "units", "precision")
    )


def _citation_count(body: Mapping[str, Any]) -> int:
    total = 0
    for evidence in body.get("evidence", []) if isinstance(body.get("evidence"), list) else []:
        knowledge = evidence.get("knowledge") if isinstance(evidence, Mapping) else None
        citations = knowledge.get("citations") if isinstance(knowledge, Mapping) else None
        if isinstance(citations, list):
            total += sum(isinstance(item, Mapping) and bool(item.get("citation_id") and item.get("excerpt")) for item in citations)
    return total


def score_live_case(body: Mapping[str, Any], case: Mapping[str, Any], *, http_status: int,
                    exception: BaseException | None = None) -> dict[str, Any]:
    """Score independent route, numeric, evidence, and refusal dimensions."""
    failure_class = classify_failure(body, http_status=http_status, exception=exception)
    if failure_class == "provider_runtime":
        return {"outcome": "provider_runtime", "failure_class": failure_class, "provider_runtime": True,
                "evaluation_contract": False, "checks": {}, "reason": next(iter(_reason_codes(body)), "request_failed")}
    if failure_class == "evaluation_contract":
        return {"outcome": "evaluation_contract", "failure_class": failure_class, "provider_runtime": False,
                "evaluation_contract": True, "checks": {}, "reason": "malformed_response"}

    expected = case["expected"]
    checks: dict[str, bool] = {
        # Route selection is a planning dimension; downstream completion is
        # scored separately so an adapter failure is not misreported here.
        "planning": body.get("route") == expected.get("route"),
    }
    if case["category"] == "answerable":
        checks["completion_status"] = body.get("status") == expected.get("status")
        structured = _step(body, "player_efg")
        result = structured.get("result") if structured else None
        oracle = expected["structured"]
        actual_inputs = result.get("inputs") if isinstance(result, Mapping) else None
        checks["numeric_fidelity"] = isinstance(result, Mapping) and _same(result.get("efg_percentage"), oracle.get("efg_percentage")) and _same(actual_inputs, oracle.get("inputs"))
        checks["structured_evidence"] = _structured_evidence_ok(body)
        if expected.get("knowledge_required"):
            knowledge = _step(body, "knowledge_glossary")
            checks["knowledge_step_status"] = isinstance(knowledge, Mapping) and knowledge.get("status") == "complete"
            checks["citation_evidence_shape"] = _citation_count(body) >= int(expected.get("min_citations", 1))
        else:
            checks["knowledge_step_status"] = not any(step.get("capability") == "knowledge_glossary" for step in body.get("steps", []) if isinstance(step, Mapping))
    else:
        checks["reason_code"] = body.get("reason_code") == expected.get("reason_code")
        checks["no_executed_steps"] = expected.get("no_executed_steps") is True and not body.get("steps")
    passed = all(checks.values())
    if passed:
        reason = None
    elif case["category"] == "answerable" and expected.get("knowledge_required") and not checks.get("knowledge_step_status", True):
        knowledge = _step(body, "knowledge_glossary")
        result = knowledge.get("result") if knowledge else None
        code = result.get("reason_code") if isinstance(result, Mapping) else None
        reason = f"knowledge_{code}" if code else "knowledge_step_status"
    else:
        reason = next(name for name, value in checks.items() if not value)
    return {"outcome": "passed" if passed else "failed", "failure_class": None,
            "provider_runtime": False, "evaluation_contract": False, "checks": checks,
            "reason": reason}


def summarize_live_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    capability = [row for row in records if not row.get("provider_runtime") and not row.get("evaluation_contract")]
    dimensions = sorted({name for row in capability for name in row.get("checks", {})})
    category_counts: dict[str, dict[str, int]] = {}
    for row in records:
        category = str(row.get("category", "unknown"))
        bucket = category_counts.setdefault(category, {"total": 0, "passed": 0, "failed": 0, "runtime": 0, "contract": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row.get("outcome") == "passed")
        bucket["failed"] += int(row.get("outcome") == "failed")
        bucket["runtime"] += int(bool(row.get("provider_runtime")))
        bucket["contract"] += int(bool(row.get("evaluation_contract")))
    return {
        "total": total,
        "passed": sum(row.get("outcome") == "passed" for row in records),
        "failed": sum(row.get("outcome") == "failed" for row in records),
        "provider_runtime_failures": sum(bool(row.get("provider_runtime")) for row in records),
        "evaluation_contract_failures": sum(bool(row.get("evaluation_contract")) for row in records),
        "full_request_success_rate": round(sum(row.get("outcome") == "passed" for row in records) / total, 4) if total else None,
        "capability_denominator": len(capability),
        "capability_success_rate": round(sum(row.get("outcome") == "passed" for row in capability) / len(capability), 4) if capability else None,
        "category_counts": category_counts,
        "dimension_passes": {name: sum(bool(row.get("checks", {}).get(name)) for row in capability) for name in dimensions},
        "dimension_denominators": {name: sum(name in row.get("checks", {}) for row in capability) for name in dimensions},
        "mean_latency_ms": round(sum(float(row.get("latency_ms", 0)) for row in records) / total, 3) if total else None,
    }


async def run_live_evaluation(db_path: Path, *, index_path: Path | None = None,
                              cases_path: Path = CASES, case_ids: Sequence[str] | None = None,
                              output_path: Path | None = None, total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
                              model: OpenAICompatibleModel | Any | None = None,
                              provider_config: ProviderConfig | None = None) -> dict[str, Any]:
    if total_timeout_seconds <= 0:
        raise ValueError("total timeout must be positive")
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    cases = select_live_cases(validate_live_cases(document, source_snapshot=catalog["source_snapshot"]), case_ids)
    provider_config = provider_config or (load_provider_config() if model is None else ProviderConfig.empty())
    search = None
    if index_path is not None:
        from scripts.knowledge_search import configured_search
        search = configured_search(index_path, provider_config=provider_config)
        if search is None:
            raise ValueError(f"knowledge index is unavailable: {index_path}")
    selected_model = model or OpenAICompatibleModel(timeout=total_timeout_seconds, config=provider_config)
    tracer = configured_tracer(provider_config)
    app = create_app(model=selected_model, db_path=db_path, knowledge_search=search,
                     provider_config=provider_config, tracer=tracer,
                     ask_model_timeout_seconds=total_timeout_seconds,
                     ask_total_timeout_seconds=total_timeout_seconds)
    records: list[dict[str, Any]] = []
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=total_timeout_seconds) as client:
            for case in cases:
                started = time.monotonic()
                exception: BaseException | None = None
                try:
                    response = await asyncio.wait_for(
                        client.post("/agent/answer", json={"question": case["question"]}),
                        timeout=total_timeout_seconds,
                    )
                    http_status = response.status_code
                    try:
                        body = response.json()
                    except (ValueError, json.JSONDecodeError):
                        body = {"status": "incomplete", "route": "unknown", "reason_code": "malformed_response"}
                except Exception as exc:  # keep every request visible in the report
                    exception = exc
                    http_status = 599
                    body = {"status": "incomplete", "route": "unknown", "reason_code": "provider_failed"}
                elapsed = round((time.monotonic() - started) * 1000, 3)
                scored = score_live_case(body, case, http_status=http_status, exception=exception)
                citation_count = _citation_count(body)
                evidence_present = bool(body.get("evidence"))
                records.append({
                    "case_id": case["case_id"], "category": case["category"], "question": case["question"],
                    "latency_ms": elapsed, "elapsed_ms": elapsed, "http_status": http_status, "route": body.get("route"),
                    "status": body.get("status"), "step_statuses": [step.get("status") for step in body.get("steps", []) if isinstance(step, Mapping)],
                    "citation_count": citation_count, "evidence_present": evidence_present,
                    "trace_id": body.get("trace_id"),
                    "failure_class": scored["failure_class"], "runtime_reason": scored.get("reason"),
                    "provider_runtime": scored["provider_runtime"], "evaluation_contract": scored["evaluation_contract"],
                    "outcome": scored["outcome"], "reason": scored["reason"], "checks": scored["checks"],
                    "response_payload": body, "response": body,
                })
    finally:
        if search is not None and getattr(search, "dense", None) is not None:
            search.dense.close()
    tracing_flushed = flush_tracer() if tracer is not None else False
    report = {
        "case_set_version": document.get("case_set_version"), "mode": "live_agent_m4_1_phoenix_smoke",
        "source_snapshot": catalog["source_snapshot"], "catalog_version": catalog.get("catalog_version"),
        "evaluation": {"case_ids": [case["case_id"] for case in cases], "total_timeout_seconds": total_timeout_seconds,
                        "model": getattr(selected_model, "model", "configured_model")},
        "summary": summarize_live_records(records), "cases": records,
        "tracing": {
            "enabled": tracer is not None,
            "flushed": tracing_flushed,
            "trace_ids": [row["trace_id"] for row in records if row.get("trace_id")],
            "trace_available": sum(bool(row.get("trace_id")) for row in records),
        },
        "provenance": {"route": "agent_answer", "cases": str(cases_path), "catalog": str(CATALOG), "index": str(index_path) if index_path else None},
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".env.local")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--total-timeout-seconds", type=float, default=DEFAULT_TOTAL_TIMEOUT_SECONDS)
    args = parser.parse_args()
    try:
        provider_config = load_provider_config(args.config)
        report = asyncio.run(run_live_evaluation(args.db, index_path=args.index, cases_path=args.cases,
                                                  case_ids=args.case_ids, output_path=args.output,
                                                  total_timeout_seconds=args.total_timeout_seconds,
                                                  provider_config=provider_config))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"M4 live evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["summary"]["failed"] > 0 or report["summary"]["provider_runtime_failures"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
