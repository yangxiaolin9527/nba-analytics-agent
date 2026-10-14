"""Evaluate the fixture contract for bounded loop conversation context."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import subprocess
from typing import Any

from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_actions import AgentAction
from app.main import create_app
from tests.test_ask import make_ask_database, make_matrix
from tests.test_agent_routes import FakeKnowledge

DEFAULT_CASES = ROOT / "metadata/agent_loop_conversation_v1.json"
RUNTIME_CODES = {
    "knowledge_unavailable", "knowledge_unconfigured", "tool_timeout", "tool_failed",
    "adapter_failed", "decision_timeout", "decision_failed", "deadline_exceeded",
    "verification_timeout", "retrieval_timeout", "generation_timeout",
}


class FixtureDecider:
    async def __call__(self, context: dict[str, Any]):
        observed = {item.get("requirement_id") for item in context.get("observations", [])}
        for requirement in context["task"]["requirements"]:
            if requirement["id"] not in observed:
                arguments = dict(requirement["arguments"])
                if requirement["capability"] == "efg_compare":
                    arguments = {"left_observation_id": "o1", "right_observation_id": "o2"}
                return AgentAction("call", requirement["capability"], arguments)
        return AgentAction("finish")


class UnavailableKnowledge:
    name, route = "knowledge_glossary", "knowledge"
    async def execute(self, question, arguments):
        from app.agent_routes import AgentCapabilityError
        raise AgentCapabilityError("knowledge_unavailable", "fixture knowledge unavailable")


def load_cases(path: Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("case_set_version") != "1.0.0" or document.get("source_snapshot") != "fixture:eoin-v515-shape-v1":
        raise ValueError("unsupported conversation case set")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != 8:
        raise ValueError("conversation case set must contain eight cases")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise ValueError("conversation case IDs must be unique")
    if {case.get("split") for case in cases} != {"development", "holdout"}:
        raise ValueError("conversation cases must contain development and holdout splits")
    return cases


async def run_case(case: dict[str, Any], database: Path, matrix: Path) -> dict[str, Any]:
    adapter = UnavailableKnowledge() if case.get("scenario") == "knowledge_unavailable" else FakeKnowledge()
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_loop_enabled=True,
                     agent_loop_decider=FixtureDecider(), agent_knowledge_adapter=adapter)
    responses = []
    conversation_id = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for question in case["turns"]:
            payload = {"question": question}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            response = await client.post("/agent/loop", json=payload)
            body = response.json()
            responses.append(body)
            conversation_id = body.get("conversation_id", conversation_id)
    expected = case["expected"]
    turn_records = []
    runtime_failures = []
    for body in responses:
        reasons = set()
        for observation in body.get("observations", []):
            if observation.get("reason_code") in RUNTIME_CODES:
                reasons.add(observation["reason_code"])
        for warning in body.get("warnings", []):
            if warning in RUNTIME_CODES:
                reasons.add(warning)
        if body.get("reason_code") in RUNTIME_CODES:
            reasons.add(body["reason_code"])
        runtime_failures.extend(sorted(reasons))
        turn_records.append({
            "status": body.get("status"),
            "route": body.get("route"),
            "turn_index": body.get("turn_index"),
            "termination_reason": body.get("termination_reason"),
            "conversation_id": body.get("conversation_id"),
            "inherited": body.get("context", {}).get("inherited", {}),
            "runtime_failures": sorted(reasons),
        })
    checks = {
        "statuses": [body.get("status") for body in responses] == expected["statuses"],
        "conversation_id": bool(conversation_id) and len({body.get("conversation_id") for body in responses}) == 1,
        "turn_indices": [body.get("turn_index") for body in responses] == list(range(1, len(responses) + 1)),
    }
    if expected.get("reason_code"):
        checks["reason_code"] = responses[-1].get("reason_code") == expected["reason_code"]
    if expected.get("inherit_player"):
        checks["inherit_player"] = expected["inherit_player"] in responses[-1].get("context", {}).get("inherited", {}).get("player_ids", [])
    if expected.get("saved_metric"):
        snapshot = app.state.agent_conversations.get(conversation_id)
        checks["saved_metric"] = snapshot is not None and expected["saved_metric"] in snapshot.metric_terms
    return {
        "case_id": case["case_id"], "split": case["split"],
        "scenario": case.get("scenario"),
        "outcome": "passed" if all(checks.values()) else "failed",
        "checks": checks, "runtime_failures": sorted(set(runtime_failures)),
        "turns": turn_records,
    }


def run_evaluation(*, cases_path: Path = DEFAULT_CASES, output_path: Path | None = None) -> dict[str, Any]:
    cases = load_cases(cases_path)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="nba-loop-context-") as directory:
        database, matrix = Path(directory) / "agent.sqlite", Path(directory) / "matrix.json"
        make_ask_database(database); make_matrix(matrix)
        records = asyncio.run(_run_all(cases, database, matrix))
    summary = _summarize(cases, records)
    report = {
        "case_set_version": "1.0.0",
        "source_snapshot": "fixture:eoin-v515-shape-v1",
        "provenance": _provenance(),
        "summary": summary,
        "cases": records,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _summarize(cases, records):
    turns = [turn for record in records for turn in record["turns"]]
    scope_cases = [record for record, case in zip(records, cases) if case["expected"].get("inherit_player")]
    scope_passed = sum(record["checks"].get("inherit_player", False) for record in scope_cases)
    runtime = [code for record in records for code in record["runtime_failures"]]
    by_split = {}
    for split in ("development", "holdout"):
        selected = [record for record in records if record["split"] == split]
        by_split[split] = {
            "total": len(selected),
            "passed": sum(record["outcome"] == "passed" for record in selected),
            "success_rate": round(sum(record["outcome"] == "passed" for record in selected) / len(selected), 4) if selected else None,
        }
    injected = [code for record in records if record.get("scenario") for code in record["runtime_failures"]]
    return {
        "total": len(records), "passed": sum(record["outcome"] == "passed" for record in records),
        "turns_total": len(turns), "turns_passed": sum(turn["status"] in {"complete", "partial", "clarification_required", "unsupported"} for turn in turns),
        "scope_checks": {"total": len(scope_cases), "passed": scope_passed},
        "runtime_failures": len(runtime), "injected_runtime_failures": len(injected),
        "natural_runtime_failures": len(runtime) - len(injected), "splits": by_split,
    }


def _provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"code_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"code_revision": None, "git_dirty": None}


async def _run_all(cases, database, matrix):
    return [await run_case(case, database, matrix) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(cases_path=args.cases, output_path=args.output)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
