"""Evaluate the opt-in bounded loop on the locked Agent v2 conversation set."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app
from scripts.evaluate_agent_loop_conversation import FixtureDecider
from scripts.evaluate_agent_v2 import FixtureKnowledge, load_cases, score_case_result
from tests.test_ask import make_ask_database, make_matrix

DEFAULT_CASES = ROOT / "metadata/agent_v2_cases_v1.json"


async def _run_case(case: Mapping[str, Any], database: Path, matrix: Path) -> dict[str, Any]:
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_loop_enabled=True,
        agent_loop_decider=FixtureDecider(),
        agent_knowledge_adapter=FixtureKnowledge(str(case.get("scenario", ""))),
    )
    responses: list[dict[str, Any]] = []
    conversation_id: str | None = None
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for turn in case["turns"]:
            payload = {"question": turn["question"]}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            response = await client.post("/agent/loop", json=payload)
            body = response.json()
            responses.append(body)
            conversation_id = body.get("conversation_id", conversation_id)
    scored = score_case_result(case, responses[-1])
    return {
        "case_id": case["case_id"], "split": case["split"],
        "outcome": scored["outcome"], "reason": scored["reason"],
        "checks": scored["checks"], "turn_count": len(responses),
        "conversation_id": responses[-1].get("conversation_id"),
        "turn_index": responses[-1].get("turn_index"),
        "response": responses[-1],
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    splits = {}
    for split in ("development", "holdout"):
        selected = [record for record in records if record["split"] == split]
        passed = sum(record["outcome"] == "passed" for record in selected)
        splits[split] = {
            "total": len(selected), "passed": passed,
            "success_rate": round(passed / len(selected), 4) if selected else None,
        }
    return {
        "total": len(records), "passed": sum(record["outcome"] == "passed" for record in records),
        "failed": sum(record["outcome"] == "failed" for record in records),
        "provider_runtime_failures": sum(record["outcome"] == "provider_runtime" for record in records),
        "evaluation_contract_failures": sum(record["outcome"] == "evaluation_contract" for record in records),
        "splits": splits,
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


def run_evaluation(*, cases_path: Path = DEFAULT_CASES, split: str = "all", output_path: Path | None = None) -> dict[str, Any]:
    cases = [case for case in load_cases(cases_path) if split == "all" or case["split"] == split]
    if not cases:
        raise ValueError(f"no cases selected for split {split}")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="nba-loop-v2-") as directory:
        database, matrix = Path(directory) / "agent.sqlite", Path(directory) / "matrix.json"
        make_ask_database(database)
        make_matrix(matrix)
        records = asyncio.run(_run_all(cases, database, matrix))
    report = {
        "case_set_version": "1.0.0", "mode": "fixture_agent_v2_loop",
        "scoring_scope": "final_turn_only",
        "provenance": _provenance(), "summary": _summary(records), "cases": records,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


async def _run_all(cases, database, matrix):
    return [await _run_case(case, database, matrix) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(cases_path=args.cases, split=args.split, output_path=args.output)["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
