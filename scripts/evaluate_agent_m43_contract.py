"""Run the deterministic M4.3 Agent contract through the real ASGI route."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_routes import AgentCapabilityError, CapabilityRegistry, StructuredToolAdapter
from app.main import create_app
from tests.test_ask import make_ask_database, make_matrix

CASES = ROOT / "metadata/m4_3_agent_http_cases_v1.json"
RUNTIME_CODES = {
    "database_unavailable", "tool_timeout", "tool_failed", "knowledge_unavailable",
    "provider_failed", "generation_timeout", "knowledge_provider_failed",
    "agent_timeout",
}
STRUCTURED = {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg"}


def _git_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"code_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_revision": None, "git_dirty": None}


class ContractKnowledgeAdapter:
    name = "knowledge_glossary"
    route = "knowledge"

    def __init__(self, scenario: str):
        self.scenario = scenario

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question, arguments
        if self.scenario == "knowledge_failure":
            return {"status": "partial", "answer": "eFG% 反映投篮效率。", "claims": [], "unresolved": ["limitations"]}
        return {
            "status": "answered",
            "answer": "eFG% 在调整三分球价值后衡量投篮效率。",
            "claims": [{"citation_id": "fixture-efg", "excerpt": "Effective Field Goal Percentage", "source_id": "fixture-glossary", "source_version": "fixture-v1", "validation": "validated"}],
            "unresolved": [],
        }


def load_cases(path: Path = CASES) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("M4.3 case manifest is empty")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("M4.3 case IDs must be unique strings")
    return cases


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(key in actual and _same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_same(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _same_exact(actual: Any, expected: Any) -> bool:
    """Compare grounded arguments exactly while result oracles may be subsets."""
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(_same_exact(actual[key], expected[key]) for key in expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_exact(left, right) for left, right in zip(actual, expected))
    return _same(actual, expected)


def _formatted(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _answer_fidelity(answer: Any, expected_result: Mapping[str, Any]) -> bool:
    """Require independently expected scalar values to appear in the rendered answer."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    def scalars(value: Any):
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from scalars(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from scalars(nested)
        else:
            yield value

    for value in scalars(expected_result):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and _formatted(value) not in answer:
            return False
    if "efg_percentage" in expected_result and "%" not in answer:
        return False
    return True


def _reason_codes(body: Mapping[str, Any]) -> list[str]:
    codes = [body["reason_code"]] if isinstance(body.get("reason_code"), str) else []
    for step in body.get("steps", []) if isinstance(body.get("steps"), list) else []:
        result = step.get("result") if isinstance(step, Mapping) else None
        if isinstance(result, Mapping) and isinstance(result.get("reason_code"), str):
            codes.append(result["reason_code"])
    return codes


def _trace_parent_ok(exporter: InMemorySpanExporter, body: Mapping[str, Any]) -> bool:
    spans = list(exporter.get_finished_spans())
    root = next((item for item in spans if item.name == "nba.agent.request"), None)
    if root is None or not body.get("trace_id"):
        return False
    children = [item for item in spans if item.name != "nba.agent.request"]
    return all(item.context.trace_id == root.context.trace_id and item.parent and item.parent.span_id == root.context.span_id for item in children)


def _route_arguments(body: Mapping[str, Any], capability: str | None) -> dict[str, Any] | None:
    route = (body.get("task_state") or {}).get("route") if isinstance(body.get("task_state"), Mapping) else None
    for step in route.get("steps", []) if isinstance(route, Mapping) else []:
        if step.get("capability") == capability:
            return step.get("arguments") if isinstance(step.get("arguments"), Mapping) else None
    return None


def _step(body: Mapping[str, Any], capability: str | None) -> Mapping[str, Any] | None:
    for step in body.get("steps", []) if isinstance(body.get("steps"), list) else []:
        if isinstance(step, Mapping) and step.get("capability") == capability:
            return step
    return None


def _evidence_ok(step: Mapping[str, Any], capability: str) -> bool:
    evidence = step.get("evidence") if isinstance(step.get("evidence"), Mapping) else {}
    structured = evidence.get("structured") if isinstance(evidence, Mapping) else None
    if capability in STRUCTURED:
        return isinstance(structured, Mapping) and all(structured.get(key) for key in ("source_sql", "snapshot_id", "catalog_version", "precision"))
    knowledge = evidence.get("knowledge") if isinstance(evidence, Mapping) else None
    citations = knowledge.get("citations") if isinstance(knowledge, Mapping) else None
    return isinstance(citations, list) and bool(citations) and all(
        isinstance(item, Mapping) and item.get("citation_id") and item.get("excerpt")
        and item.get("source_id") and item.get("source_version")
        for item in citations
    )


def score_http_case(body: Mapping[str, Any], case: Mapping[str, Any], *, http_status: int, trace_parent_ok: bool = False) -> dict[str, Any]:
    if http_status >= 500 or not isinstance(body.get("status"), str) or not isinstance(body.get("route"), str):
        return {"outcome": "evaluation_contract", "failure_class": "evaluation_contract", "provider_runtime": False, "evaluation_contract": True, "capability_denominator": False, "checks": {}, "reason": "malformed_response"}
    if any(code in RUNTIME_CODES for code in _reason_codes(body)):
        return {"outcome": "provider_runtime", "failure_class": "provider_runtime", "provider_runtime": True, "evaluation_contract": False, "capability_denominator": False, "checks": {}, "reason": next(code for code in _reason_codes(body) if code in RUNTIME_CODES)}
    expected = case["expected"]
    checks: dict[str, bool] = {
        "route": body.get("route") == expected.get("route"),
        "status": body.get("status") == expected.get("status"),
        "trace": bool(body.get("trace_id")) and trace_parent_ok,
    }
    if expected.get("reason_code"):
        checks["reason_code"] = body.get("reason_code") == expected["reason_code"]
    if expected.get("no_steps"):
        checks["no_steps"] = not body.get("steps")
    capability = expected.get("capability")
    step = _step(body, capability)
    if capability:
        checks["capability"] = step is not None
        if step is not None:
            expected_args = expected.get("arguments")
            if expected_args is not None:
                checks["arguments"] = _same_exact(_route_arguments(body, capability), expected_args)
            checks["evidence"] = _evidence_ok(step, capability) if expected.get("status") == "complete" else True
            for key, value in (expected.get("result") or {}).items():
                checks[f"result_{key}"] = _same((step.get("result") or {}).get(key), value)
            if expected.get("status") == "complete":
                checks["answer_fidelity"] = _answer_fidelity(body.get("answer"), expected.get("result") or {})
            if expected.get("knowledge"):
                knowledge_step = _step(body, "knowledge_glossary")
                expected_knowledge_status = expected["knowledge"] if isinstance(expected["knowledge"], str) else "complete"
                checks["knowledge"] = knowledge_step is not None and knowledge_step.get("status") == expected_knowledge_status
                if expected_knowledge_status == "complete" and knowledge_step is not None:
                    checks["knowledge_evidence"] = _evidence_ok(knowledge_step, "knowledge_glossary")
    passed = all(checks.values())
    return {"outcome": "passed" if passed else "failed", "failure_class": None if passed else "capability", "provider_runtime": False, "evaluation_contract": False, "capability_denominator": True, "checks": checks, "reason": None if passed else next(key for key, value in checks.items() if not value)}


async def _run_case(case: Mapping[str, Any], database: Path, matrix: Path) -> dict[str, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    scenario = str(case.get("scenario", ""))
    adapters = [StructuredToolAdapter(name, database=database, quality_matrix_path=matrix) for name in STRUCTURED]
    adapters.append(ContractKnowledgeAdapter(scenario))
    app = create_app(db_path=database, quality_matrix_path=matrix, tracer=provider.get_tracer("m4.3-contract"), agent_registry=CapabilityRegistry(adapters))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/agent/answer", json={"question": case["question"]})
    body = response.json()
    scored = score_http_case(body, case, http_status=response.status_code, trace_parent_ok=_trace_parent_ok(exporter, body))
    return {"case_id": case["case_id"], "category": case["category"], "question": case["question"], "outcome": scored["outcome"], "failure_class": scored["failure_class"], "provider_runtime": scored["provider_runtime"], "evaluation_contract": scored["evaluation_contract"], "capability_denominator": scored["capability_denominator"], "reason": scored["reason"], "checks": scored["checks"], "status": body.get("status"), "route": body.get("route"), "trace_id": body.get("trace_id"), "response": body}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capability = [item for item in records if item["capability_denominator"]]
    dimensions = sorted({key for item in capability for key in item["checks"]})
    return {
        "total": len(records), "passed": sum(item["outcome"] == "passed" for item in records), "failed": sum(item["outcome"] == "failed" for item in records),
        "provider_runtime_failures": sum(item["provider_runtime"] for item in records), "evaluation_contract_failures": sum(item["evaluation_contract"] for item in records),
        "full_request_success_rate": round(sum(item["outcome"] == "passed" for item in records) / len(records), 4) if records else None,
        "capability_denominator": len(capability), "capability_success_rate": round(sum(item["outcome"] == "passed" for item in capability) / len(capability), 4) if capability else None,
        "dimension_passes": {key: sum(bool(item["checks"].get(key)) for item in capability) for key in dimensions},
        "dimension_denominators": {key: sum(key in item["checks"] for item in capability) for key in dimensions},
    }


def run_evaluation(*, cases_path: Path = CASES, output_path: Path | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        database, matrix = Path(directory) / "agent.sqlite", Path(directory) / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        records = asyncio.run(_run_all(cases, database, matrix))
    report = {
        "case_set_version": json.loads(cases_path.read_text(encoding="utf-8"))["case_set_version"],
        "mode": "deterministic_agent_http_contract_m4_3",
        "provenance": _git_provenance(),
        "summary": summarize(records),
        "cases": records,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


async def _run_all(cases, database, matrix):
    records = []
    for case in cases:
        records.append(await _run_case(case, database, matrix))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_evaluation(cases_path=args.cases, output_path=args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(bool(report["summary"]["failed"] or
                    report["summary"]["provider_runtime_failures"] or
                    report["summary"]["evaluation_contract_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
