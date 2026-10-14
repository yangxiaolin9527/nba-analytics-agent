"""Evaluate live model routing through the structured ``/answer`` endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from scripts.evaluation_contract import report_provenance
from scripts.model_adapter import OpenAICompatibleModel, TextToSQLModel
from scripts.relational_baseline import CAPABILITY_MATRIX

CASES = ROOT / "metadata/m3_2_live_answer_cases_v2.json"
CATALOG = ROOT / "metadata/schema_catalog.json"
DEFAULT_ASK_MODEL_TIMEOUT_SECONDS = 7.0
DEFAULT_ASK_TOTAL_TIMEOUT_SECONDS = 10.0
RUNTIME_CODES = frozenset({
    "provider_failed", "model_timeout", "model_unconfigured", "database_unavailable",
    "tool_timeout", "tool_failed", "evidence_timeout",
})


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-9,
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
    if document.get("source_snapshot") != source_snapshot:
        raise ValueError("live answer cases source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("live answer case set must contain cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("live answer case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"invalid or duplicate live case ID: {case_id}")
        seen.add(case_id)
        if "plan" in case or "simulate_timeout" in case:
            raise ValueError(f"live case {case_id} must not contain plan or timeout injection")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"live case {case_id} has no question")
        if case.get("category") not in {"answerable", "clarification", "unsupported"}:
            raise ValueError(f"invalid category for {case_id}")
        if not isinstance(case.get("expected"), dict):
            raise ValueError(f"live case {case_id} has no expected contract")
    return cases


def select_live_cases(cases: Sequence[dict[str, Any]], case_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    """Select cases in manifest order while rejecting unknown IDs."""
    if case_ids is None:
        return list(cases)
    requested = list(dict.fromkeys(case_ids))
    if not requested:
        raise ValueError("case selection must contain at least one case ID")
    available = {case["case_id"] for case in cases}
    unknown = [case_id for case_id in requested if case_id not in available]
    if unknown:
        raise ValueError(f"unknown live answer case ID(s): {', '.join(unknown)}")
    selected = set(requested)
    return [case for case in cases if case["case_id"] in selected]


def _is_runtime(body: Mapping[str, Any], http_status: int) -> bool:
    return http_status >= 500 or body.get("reason_code") in RUNTIME_CODES


def score_live_case(body: Mapping[str, Any], case: Mapping[str, Any], *, http_status: int) -> dict[str, Any]:
    """Score one response while preserving independent failure dimensions."""
    expected = case.get("expected", {})
    if not isinstance(body, Mapping) or not isinstance(body.get("status"), str):
        return {
            "outcome": "evaluation_contract", "provider_runtime": False,
            "evaluation_contract": True, "checks": {}, "reason": "malformed_response",
        }
    if _is_runtime(body, http_status):
        return {
            "outcome": "provider_runtime", "provider_runtime": True,
            "evaluation_contract": False, "checks": {}, "reason": body.get("reason_code"),
        }

    target_tool = expected.get("tool")
    is_refusal = target_tool is None
    checks: dict[str, bool] = {
        "planning": body.get("status") == expected.get("status")
        and body.get("tool") == target_tool
        and body.get("tool_calls_used") == (0 if is_refusal else 1),
    }
    if "reason_code" in expected:
        checks["reason_code"] = body.get("reason_code") == expected["reason_code"]

    if is_refusal:
        checks["no_tool_call"] = body.get("tool_calls_used") == 0 and body.get("tool_result") is None
        checks["grounding"] = True
        checks["deterministic_result"] = True
        checks["renderer_evidence"] = isinstance(body.get("answer"), str) and bool(body["answer"].strip())
    else:
        expected_arguments = expected.get("arguments")
        checks["grounding"] = expected_arguments is not None and _same(body.get("arguments"), expected_arguments)
        result = body.get("tool_result")
        checks["deterministic_result"] = (
            isinstance(result, Mapping)
            and result.get("status") in {"complete", "partial"}
            and _same(result, expected.get("tool_result", {}))
        )
        answer = body.get("answer")
        checks["renderer_evidence"] = (
            isinstance(answer, str) and bool(answer.strip())
            and all(fragment in answer for fragment in expected.get("answer_contains", []))
            and (not expected.get("require_evidence") or isinstance(body.get("evidence"), Mapping)
                 and bool(body["evidence"]))
        )
    passed = all(checks.values())
    return {
        "outcome": "passed" if passed else "failed",
        "provider_runtime": False,
        "evaluation_contract": False,
        "checks": checks,
        "reason": None if passed else next((name for name, value in checks.items() if not value), "check_failed"),
    }


def summarize_live_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return rates with explicit denominators for runtime-safe reporting."""
    total = len(records)
    passed = sum(row.get("outcome") == "passed" for row in records)
    runtime = sum(bool(row.get("provider_runtime")) for row in records)
    contract = sum(bool(row.get("evaluation_contract")) for row in records)
    capability = [row for row in records if not row.get("provider_runtime") and not row.get("evaluation_contract")]
    category_counts: dict[str, dict[str, int]] = {}
    for row in records:
        category = str(row.get("category", "unknown"))
        bucket = category_counts.setdefault(category, {"total": 0, "passed": 0, "runtime": 0, "contract": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row.get("outcome") == "passed")
        bucket["runtime"] += int(bool(row.get("provider_runtime")))
        bucket["contract"] += int(bool(row.get("evaluation_contract")))
    return {
        "total": total,
        "passed": passed,
        "failed": sum(row.get("outcome") == "failed" for row in records),
        "provider_runtime_failures": runtime,
        "provider_runtime_rate": round(runtime / total, 4) if total else None,
        "evaluation_contract_failures": contract,
        "full_request_success_rate": round(passed / total, 4) if total else None,
        "capability_denominator": len(capability),
        "capability_success_rate": round(sum(row.get("outcome") == "passed" for row in capability) / len(capability), 4)
        if capability else None,
        "category_counts": category_counts,
        "dimension_passes": {name: sum(bool(row.get("checks", {}).get(name)) for row in capability) for name in {"planning", "grounding", "deterministic_result", "renderer_evidence", "reason_code", "no_tool_call"}},
        "dimension_denominators": {name: sum(name in row.get("checks", {}) for row in capability) for name in {"planning", "grounding", "deterministic_result", "renderer_evidence", "reason_code", "no_tool_call"}},
        "mean_latency_ms": round(sum(float(row.get("elapsed_ms", 0)) for row in records) / total, 3) if total else None,
    }


async def run_live_evaluation(
    db_path: Path,
    *,
    cases_path: Path = CASES,
    matrix_path: Path | None = None,
    model: TextToSQLModel | None = None,
    model_id: str | None = None,
    case_ids: Sequence[str] | None = None,
    ask_model_timeout_seconds: float = DEFAULT_ASK_MODEL_TIMEOUT_SECONDS,
    ask_total_timeout_seconds: float = DEFAULT_ASK_TOTAL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    cases = select_live_cases(
        validate_live_cases(document, source_snapshot=catalog["source_snapshot"]), case_ids,
    )
    if ask_model_timeout_seconds <= 0 or ask_total_timeout_seconds <= 0:
        raise ValueError("evaluation timeouts must be positive")
    selected_model = model or OpenAICompatibleModel()
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.monotonic()
        app = create_app(
            db_path=db_path, model=selected_model,
            quality_matrix_path=matrix_path or CAPABILITY_MATRIX,
            ask_model_timeout_seconds=ask_model_timeout_seconds,
            ask_total_timeout_seconds=ask_total_timeout_seconds,
        )
        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/answer", json={"question": case["question"]})
            http_status = response.status_code
            try:
                body = response.json()
            except (ValueError, json.JSONDecodeError):
                body = {"status": "incomplete", "reason_code": "malformed_response", "answer": ""}
        except Exception as exc:  # Keep provider/runtime failures visible per case.
            http_status = 599
            body = {"status": "incomplete", "reason_code": "provider_failed", "answer": "", "error_type": type(exc).__name__}
        scored = score_live_case(body, case, http_status=http_status)
        records.append({
            "case_id": case["case_id"], "category": case["category"], "question": case["question"],
            "http_status": http_status, "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "outcome": scored["outcome"], "reason": scored["reason"], "checks": scored["checks"],
            "provider_runtime": scored["provider_runtime"], "evaluation_contract": scored["evaluation_contract"],
            "status": body.get("status"), "tool": body.get("tool"), "arguments": body.get("arguments"),
            "tool_calls_used": body.get("tool_calls_used", 0), "tool_result": body.get("tool_result"),
            "answer": body.get("answer", ""), "evidence": body.get("evidence"),
            "reason_code": body.get("reason_code"), "timings": body.get("timings", {}),
        })

    summary = summarize_live_records(records)
    return {
        "case_set_version": document.get("case_set_version"),
        "mode": "live",
        "source_snapshot": catalog["source_snapshot"],
        "catalog_version": catalog["catalog_version"],
        "evaluation": {
            "case_ids": [case["case_id"] for case in cases],
            "ask_model_timeout_seconds": ask_model_timeout_seconds,
            "ask_total_timeout_seconds": ask_total_timeout_seconds,
        },
        "summary": summary,
        "cases": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=catalog["source_snapshot"],
            model_id=model_id or getattr(selected_model, "model", "configured_model"),
            route="answer_live_model_one_tool_deterministic_render",
            input_paths={"cases": cases_path, "catalog": CATALOG, "route": ROOT / "app/answer_routes.py",
                         "ask_flow": ROOT / "app/ask_routes.py", "grounding": ROOT / "scripts/ask_grounding.py",
                         "renderer": ROOT / "scripts/answer_renderer.py", "evaluator": ROOT / "scripts/evaluate_answer_live.py"},
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-ids", nargs="+", help="evaluate only these manifest case IDs")
    parser.add_argument("--model-timeout-seconds", type=float, default=DEFAULT_ASK_MODEL_TIMEOUT_SECONDS)
    parser.add_argument("--total-timeout-seconds", type=float, default=DEFAULT_ASK_TOTAL_TIMEOUT_SECONDS)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_live_evaluation(
            args.db,
            cases_path=args.cases,
            case_ids=args.case_ids,
            ask_model_timeout_seconds=args.model_timeout_seconds,
            ask_total_timeout_seconds=args.total_timeout_seconds,
        ))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"live evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(report["summary"]["failed"] > 0 or report["summary"]["provider_runtime_failures"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
