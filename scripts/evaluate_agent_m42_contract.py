"""Deterministic M4.2 Agent contract evaluator."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_routes import CapabilityRegistry, KnowledgeGlossaryAdapter, StructuredToolAdapter, execute_agent_plan, plan_question
from scripts.ask_grounding import ground_ask_question
from scripts.query_context import QueryLinks
from tests.test_ask import make_ask_database, make_matrix

CASES = ROOT / "metadata/m4_2_agent_contract_cases_v1.json"
RUNTIME_CODES = {"database_unavailable", "tool_timeout", "tool_failed", "provider_failed", "knowledge_unavailable"}


def load_cases(path: Path = CASES) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("M4.2 case manifest is empty")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("M4.2 case IDs must be unique")
    return cases


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(key in actual and _same(actual[key], value) for key, value in expected.items())
    return actual == expected


def score_response(body: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    code = body.get("reason_code")
    runtime = code in RUNTIME_CODES or any((step.get("result") or {}).get("reason_code") in RUNTIME_CODES for step in body.get("steps", []) if isinstance(step, Mapping))
    if runtime:
        return {"outcome": "provider_runtime", "failure_class": "provider_runtime", "capability_denominator": False, "checks": {}}
    checks = {"route": body.get("route") == expected.get("route"), "status": body.get("status") == expected.get("status")}
    if expected.get("reason_code"):
        checks["reason_code"] = code == expected["reason_code"]
    steps = body.get("steps") if isinstance(body.get("steps"), list) else []
    step = next((item for item in steps if item.get("capability") == expected.get("capability")), None)
    if expected.get("capability"):
        checks["capability"] = step is not None
        result = step.get("result") if step else None
        for key, value in (expected.get("result") or {}).items():
            checks[f"result_{key}"] = _same((result or {}).get(key), value)
        evidence = (step or {}).get("evidence", {}).get("structured", {})
        checks["structured_evidence"] = all(evidence.get(key) for key in ("source_sql", "precision"))
    if expected.get("knowledge"):
        knowledge = next((item for item in steps if item.get("capability") == "knowledge_glossary"), None)
        checks["knowledge"] = bool(knowledge and knowledge.get("status") == "complete")
    if case["category"] in {"clarification", "unsupported"}:
        checks["no_steps"] = not steps
    passed = all(checks.values())
    return {"outcome": "passed" if passed else "failed", "failure_class": None if passed else "capability", "capability_denominator": True, "checks": checks}


class _FakeKnowledge:
    name = "knowledge_glossary"
    route = "knowledge"

    async def execute(self, question, arguments):
        del question, arguments
        return {"status": "answered", "answer": "eFG% reflects shooting efficiency.", "claims": [{"citation_id": "fixture", "excerpt": "eFG", "validation": "validated"}]}


async def _run_case(case: Mapping[str, Any], database: Path, matrix: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        links, _ = ground_ask_question(connection, case["question"])
    plan = plan_question(case["question"], links=links)
    adapters = [StructuredToolAdapter(name, database=database, quality_matrix_path=matrix) for name in ("season_summary", "single_game_score", "window_query", "season_compare", "player_efg")]
    adapters.append(_FakeKnowledge())
    response = await execute_agent_plan(case["question"], plan, CapabilityRegistry(adapters), request_id=case["case_id"], deadline_seconds=10.0)
    body = response.model_dump(mode="json")
    scored = score_response(body, case)
    return {"case_id": case["case_id"], "question": case["question"], **scored, "response": body}


def run_evaluation(*, cases_path: Path = CASES, output_path: Path | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        database, matrix = Path(directory) / "agent.sqlite", Path(directory) / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        records = asyncio.run(_run_all(cases, database, matrix))
    capability = [row for row in records if row["capability_denominator"]]
    report = {
        "case_set_version": json.loads(cases_path.read_text())["case_set_version"],
        "mode": "deterministic_agent_contract_m4_2",
        "summary": {
            "total": len(records),
            "passed": sum(row["outcome"] == "passed" for row in records),
            "failed": sum(row["outcome"] == "failed" for row in records),
            "provider_runtime_failures": sum(row["failure_class"] == "provider_runtime" for row in records),
            "evaluation_contract_failures": 0,
            "capability_denominator": len(capability),
            "capability_success_rate": round(sum(row["outcome"] == "passed" for row in capability) / len(capability), 4) if capability else None,
        },
        "cases": records,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


async def _run_all(cases, database, matrix):
    records = []
    for case in cases:
        records.append(await _run_case(case, database, matrix))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_evaluation(output_path=args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
