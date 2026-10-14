"""Evaluate the rules-based Agent on a frozen multi-turn case manifest."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from tests.test_ask import make_ask_database, make_matrix

DEFAULT_CASES = ROOT / "metadata/agent_v2_cases_v1.json"
RUNTIME_CODES = {
    "database_unavailable", "tool_timeout", "tool_failed", "knowledge_unavailable",
    "knowledge_unconfigured", "provider_failed", "generation_timeout", "agent_timeout",
}


class FixtureKnowledge:
    name = "knowledge_glossary"
    route = "knowledge"

    def __init__(self, scenario: str = ""):
        self.scenario = scenario

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question, arguments
        if self.scenario == "knowledge_partial":
            return {
                "status": "partial",
                "answer": "eFG% 反映投篮效率。",
                "claims": [],
                "unresolved": ["limitations"],
            }
        if self.scenario in {"knowledge_failure", "knowledge_unavailable"}:
            raise RuntimeError("fixture knowledge unavailable")
        return {
            "status": "answered",
            "answer": "eFG% 在调整三分球价值后衡量投篮效率。",
            "claims": [{
                "citation_id": "fixture-efg",
                "excerpt": "Effective Field Goal Percentage",
                "source_id": "fixture-glossary",
                "source_version": "fixture-v1",
                "validation": "validated",
            }],
            "unresolved": [],
        }


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("case_set_version") != "1.0.0":
        raise ValueError("unsupported Agent v2 case set version")
    if document.get("source_snapshot") != "fixture:eoin-v515-shape-v1":
        raise ValueError("Agent v2 source snapshot is not pinned")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 60:
        raise ValueError("Agent v2 case set must contain exactly 60 cases")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Agent v2 case IDs must be unique strings")
    if sum(case.get("split") == "development" for case in cases) != 40:
        raise ValueError("Agent v2 must contain 40 development cases")
    if sum(case.get("split") == "holdout" for case in cases) != 20:
        raise ValueError("Agent v2 must contain 20 holdout cases")
    for case in cases:
        if not isinstance(case.get("turns"), list) or not 1 <= len(case["turns"]) <= 3:
            raise ValueError(f"invalid turn count for {case.get('case_id')}")
        if any(not isinstance(turn, dict) or not str(turn.get("question", "")).strip() for turn in case["turns"]):
            raise ValueError(f"invalid turn in {case.get('case_id')}")
        expected = case.get("expected")
        if not isinstance(expected, dict) or expected.get("status") not in {
            "complete", "partial", "clarification_required", "unsupported", "not_computable", "incomplete",
        }:
            raise ValueError(f"invalid expected contract for {case.get('case_id')}")
        if expected.get("route") not in {"structured", "knowledge", "mixed", "comparison", "clarification", "unsupported"}:
            raise ValueError(f"invalid expected route for {case.get('case_id')}")
    return cases


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


def _steps_for(body: Mapping[str, Any], capability: str) -> list[Mapping[str, Any]]:
    return [
        step for step in body.get("steps", [])
        if isinstance(step, Mapping) and step.get("capability") == capability
    ]


def _all_reason_codes(body: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    if isinstance(body.get("reason_code"), str):
        values.add(body["reason_code"])
    for step in body.get("steps", []) if isinstance(body.get("steps"), list) else []:
        result = step.get("result") if isinstance(step, Mapping) else None
        if isinstance(result, Mapping) and isinstance(result.get("reason_code"), str):
            values.add(result["reason_code"])
    return values


def score_case_result(case: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    runtime_codes = _all_reason_codes(body) & RUNTIME_CODES
    if runtime_codes:
        return {
            "outcome": "provider_runtime", "failure_class": "provider_runtime",
            "provider_runtime": True, "capability_denominator": False,
            "checks": {}, "reason": sorted(runtime_codes)[0],
        }
    if not isinstance(body.get("status"), str) or not isinstance(body.get("route"), str):
        return {
            "outcome": "evaluation_contract", "failure_class": "evaluation_contract",
            "provider_runtime": False, "capability_denominator": False,
            "checks": {}, "reason": "malformed_response",
        }
    checks: dict[str, bool] = {
        "status": body["status"] == expected["status"],
        "route": body["route"] == expected["route"],
    }
    if expected.get("require_trace"):
        checks["trace"] = bool(body.get("trace_id"))
    if expected.get("reason_code"):
        checks["reason_code"] = body.get("reason_code") == expected["reason_code"]
    if expected.get("no_steps"):
        checks["no_steps"] = not body.get("steps")
    expected_capabilities = expected.get("capabilities")
    if expected_capabilities is not None:
        actual_capabilities = [step.get("capability") for step in body.get("steps", [])]
        checks["capabilities"] = actual_capabilities == expected_capabilities
    numeric_checks: list[bool] = []
    for capability, oracle in (expected.get("numeric_oracle") or {}).items():
        candidates = _steps_for(body, capability)
        numeric_check = bool(candidates) and all(
            _same((candidates[0].get("result") or {}).get(key), value)
            for key, value in oracle.items()
        )
        checks[f"numeric_{capability}"] = numeric_check
        numeric_checks.append(numeric_check)
    if numeric_checks:
        checks["numeric_fidelity"] = all(numeric_checks)
    required = expected.get("required_evidence") or []
    if required:
        evidence_bundles: list[Mapping[str, Any]] = []
        for step in body.get("steps", []):
            structured = (step.get("evidence") or {}).get("structured") if isinstance(step, Mapping) else None
            if isinstance(structured, Mapping):
                evidence_bundles.append(structured)
        checks["evidence"] = all(
            any(bool(bundle.get(key)) for bundle in evidence_bundles)
            for key in required
        )
    required_citations = expected.get("required_citations") or []
    if required_citations:
        citation_ids = {
            citation.get("citation_id")
            for step in body.get("steps", [])
            for citation in ((step.get("evidence") or {}).get("knowledge") or {}).get("citations", [])
            if isinstance(citation, Mapping)
        }
        checks["citations"] = all(citation in citation_ids for citation in required_citations)
    required_terms = expected.get("required_answer_terms") or []
    if required_terms:
        answer = body.get("answer")
        answer_text = answer.casefold() if isinstance(answer, str) else ""
        checks["answer_terms"] = all(str(term).casefold() in answer_text for term in required_terms)
    passed = all(checks.values())
    return {
        "outcome": "passed" if passed else "failed",
        "failure_class": None if passed else "capability",
        "provider_runtime": False,
        "capability_denominator": True,
        "checks": checks,
        "reason": None if passed else next(key for key, value in checks.items() if not value),
    }


def summarize_records(records: Sequence[Mapping[str, Any]], *, split: str = "all") -> dict[str, Any]:
    selected = [record for record in records if split == "all" or record.get("split") == split]
    capability = [
        record for record in selected
        if record.get("capability_denominator")
        or record.get("outcome") in {"passed", "failed"}
    ]
    passed = sum(record.get("outcome") == "passed" for record in selected)
    return {
        "split": split,
        "total": len(selected),
        "passed": passed,
        "failed": sum(record.get("outcome") == "failed" for record in selected),
        "provider_runtime_failures": sum(record.get("outcome") == "provider_runtime" for record in selected),
        "evaluation_contract_failures": sum(record.get("outcome") == "evaluation_contract" for record in selected),
        "capability_denominator": len(capability),
        "task_success_rate": round(passed / len(capability), 4) if capability else None,
    }


async def _run_case(case: Mapping[str, Any], database: Path, matrix: Path) -> dict[str, Any]:
    scenario = str(case.get("scenario", ""))
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_knowledge_adapter=FixtureKnowledge(scenario),
    )
    responses: list[dict[str, Any]] = []
    conversation_id: str | None = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for turn in case["turns"]:
            payload = {"question": turn["question"]}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            response = await client.post("/agent/answer", json=payload)
            if response.status_code >= 500:
                return {
                    "case_id": case["case_id"], "split": case["split"],
                    "outcome": "evaluation_contract", "failure_class": "evaluation_contract",
                    "provider_runtime": False, "capability_denominator": False,
                    "checks": {}, "reason": f"http_{response.status_code}",
                }
            body = response.json()
            responses.append(body)
            conversation_id = body.get("conversation_id", conversation_id)
    scored = score_case_result(case, responses[-1])
    return {
        "case_id": case["case_id"], "split": case["split"],
        "outcome": scored["outcome"], "failure_class": scored["failure_class"],
        "provider_runtime": scored["provider_runtime"],
        "capability_denominator": scored["capability_denominator"],
        "reason": scored["reason"], "checks": scored["checks"],
        "turn_count": len(responses),
        "conversation_id": responses[-1].get("conversation_id"),
        "turn_index": responses[-1].get("turn_index"),
        "response": responses[-1],
    }


async def _run_all(cases: Sequence[Mapping[str, Any]], database: Path, matrix: Path) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        records.append(await _run_case(case, database, matrix))
    return records


def _provenance() -> dict[str, Any]:
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


def run_evaluation(*, cases_path: Path = DEFAULT_CASES, split: str = "all", output_path: Path | None = None) -> dict[str, Any]:
    cases = [case for case in load_cases(cases_path) if split == "all" or case["split"] == split]
    if not cases:
        raise ValueError(f"no cases selected for split {split}")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="nba-agent-v2-") as directory:
        database, matrix = Path(directory) / "agent.sqlite", Path(directory) / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        records = asyncio.run(_run_all(cases, database, matrix))
    report = {
        "case_set_version": "1.0.0",
        "mode": "fixture_agent_v2_rules_baseline",
        "provenance": _provenance(),
        "summary": summarize_records(records, split=split),
        "cases": records,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="all")
    parser.add_argument("--mode", choices=("fixture",), default="fixture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_evaluation(cases_path=args.cases, split=args.split, output_path=args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
