"""Run the bounded Agent against the pinned Eoin v515 SQLite snapshot.

This smoke is deliberately provider-free: the rule planner and deterministic
structured adapters are exercised through the real ASGI boundary.  Expected
values are calculated by independent read-only SQL queries in this runner and
stored beside every response, so fixture values cannot masquerade as live data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient
from opentelemetry.trace import NoOpTracerProvider

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import create_app


DEFAULT_DB = Path("/tmp/nba-m1.sqlite")
DEFAULT_MATRIX = ROOT / "data/profile/eoin-v515-capability-matrix.json"
DEFAULT_OUTPUT = ROOT / "reports/m4/agent-m44-structured-smoke.json"
SNAPSHOT = "eoinamoore/historical-nba-data-and-player-box-scores@515"
CATALOG_VERSION = "0.5.2"

CASES = [
    {"case_id": "M44-S-001", "capability": "season_summary", "question": "库里 2023-24 常规赛总得分？"},
    {"case_id": "M44-S-002", "capability": "single_game_score", "question": "2023-10-24 勇士对太阳谁赢了？"},
    {"case_id": "M44-S-003", "capability": "window_query", "question": "库里 2023-24 常规赛最近两次出场的场均得分？"},
    {"case_id": "M44-S-004", "capability": "season_compare", "question": "库里 2023-24 常规赛总得分比上一季增加多少？"},
    {"case_id": "M44-S-005", "capability": "player_efg", "question": "库里 2023-24 常规赛 eFG% 是多少？"},
]


def _oracle(connection: sqlite3.Connection, capability: str) -> tuple[dict[str, Any], str]:
    if capability == "season_summary":
        query = "SELECT points_total FROM player_season_stat WHERE person_id = ? AND season_id = ? AND competition_phase = ?"
        value = connection.execute(query, ("201939", "2023-24", "regular_season")).fetchone()[0]
        return {"points_total": value}, query
    if capability == "single_game_score":
        query = "SELECT game_id, home_score, away_score FROM game WHERE game_date = ? AND home_team_id = ? AND away_team_id = ?"
        row = connection.execute(query, ("2023-10-24", "1610612744", "1610612756")).fetchone()
        return {"game_id": row[0], "home_score": row[1], "away_score": row[2], "winner_team_id": "1610612756"}, query
    if capability == "window_query":
        query = """SELECT p.points FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
            WHERE p.person_id = ? AND g.season_id = ? AND g.competition_phase = ?
              AND p.appearance_status IN ('played', 'played_inferred')
            ORDER BY g.game_date DESC, p.game_id DESC LIMIT 2"""
        values = [row[0] for row in connection.execute(query, ("201939", "2023-24", "regular_season")).fetchall()]
        return {"actual_sample_count": len(values), "points_per_appearance": sum(values) / len(values)}, query
    if capability == "season_compare":
        query = """SELECT season_id, SUM(p.points) FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
            WHERE p.person_id = ? AND g.season_id IN (?, ?) AND g.competition_phase = ?
              AND p.appearance_status IN ('played', 'played_inferred') GROUP BY season_id"""
        rows = dict(connection.execute(query, ("201939", "2022-23", "2023-24", "regular_season")).fetchall())
        return {"absolute_delta": rows["2023-24"] - rows["2022-23"], "baseline": rows["2022-23"], "comparison": rows["2023-24"]}, query
    query = """SELECT SUM(field_goals_made), SUM(three_pointers_made), SUM(field_goals_attempted), COUNT(*)
        FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
        WHERE p.person_id = ? AND g.season_id = ? AND g.competition_phase = ?
          AND p.appearance_status IN ('played', 'played_inferred')"""
    fgm, threes, fga, count = connection.execute(query, ("201939", "2023-24", "regular_season")).fetchone()
    return {"efg_percentage": (fgm + 0.5 * threes) / fga * 100, "inputs": {
        "field_goals_made": fgm, "three_pointers_made": threes,
        "field_goals_attempted": fga, "sample_count": count,
    }}, query


def _matches(capability: str, result: Any, oracle: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or result.get("status") != "complete":
        return False
    if capability == "season_summary":
        return result.get("points_total") == oracle["points_total"]
    if capability == "single_game_score":
        return all(result.get(key) == oracle[key] for key in ("home_score", "away_score", "winner_team_id"))
    if capability == "window_query":
        return result.get("actual_sample_count") == oracle["actual_sample_count"] and math.isclose(
            result.get("values", {}).get("points_per_appearance", -1), oracle["points_per_appearance"], rel_tol=1e-9,
        )
    if capability == "season_compare":
        return result.get("absolute_delta") == oracle["absolute_delta"]
    return result.get("inputs") == oracle["inputs"] and math.isclose(
        result.get("efg_percentage", -1), oracle["efg_percentage"], rel_tol=1e-9,
    )


def _git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


async def run(*, db_path: Path, matrix_path: Path, output_path: Path, timeout_seconds: float = 60.0) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(f"database is unavailable: {db_path}")
    if not matrix_path.is_file():
        raise FileNotFoundError(f"quality matrix is unavailable: {matrix_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        oracle_values = {}
        for case in CASES:
            value, query = _oracle(connection, case["capability"])
            oracle_values[case["case_id"]] = {"values": value, "query": query}

    tracer = NoOpTracerProvider().get_tracer("nba-m44-real-smoke")
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path, tracer=tracer)
    records = []
    started_all = time.monotonic()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://smoke") as client:
        for case in CASES:
            started = time.monotonic()
            exception = None
            try:
                response = await asyncio.wait_for(client.post("/agent/answer", json={"question": case["question"]}), timeout=timeout_seconds)
                body = response.json()
                http_status = response.status_code
            except Exception as exc:
                exception = type(exc).__name__
                body, http_status = {}, 599
            step = next((item for item in body.get("steps", []) if item.get("capability") == case["capability"]), None)
            result = step.get("result") if isinstance(step, dict) else None
            oracle = oracle_values[case["case_id"]]
            passed = exception is None and http_status == 200 and body.get("status") == "complete" and body.get("route") == "structured" and _matches(case["capability"], result, oracle["values"])
            records.append({
                **case, "outcome": "passed" if passed else ("provider_runtime" if exception else "failed"),
                "http_status": http_status, "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "status": body.get("status"), "route": body.get("route"), "trace_id": body.get("trace_id"),
                "oracle": oracle, "response": body, "exception": exception,
            })
    report = {
        "mode": "real_agent_structured_smoke_m4_4", "case_set_version": "m4.4-structured-v1",
        "source_snapshot": SNAPSHOT, "catalog_version": CATALOG_VERSION,
        "database": {"path": str(db_path), "snapshot_id": "eoin-v515"},
        "quality_matrix": str(matrix_path), "code_revision": _git_revision(),
        "evaluation": {"timeout_seconds": timeout_seconds, "elapsed_ms": round((time.monotonic() - started_all) * 1000, 3), "oracle": "independent read-only SQL in this runner"},
        "summary": {"total": len(records), "passed": sum(item["outcome"] == "passed" for item in records), "failed": sum(item["outcome"] == "failed" for item in records), "provider_runtime_failures": sum(item["outcome"] == "provider_runtime" for item in records)},
        "cases": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--capability-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = asyncio.run(run(db_path=args.db, matrix_path=args.capability_matrix, output_path=args.output, timeout_seconds=args.timeout_seconds))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return int(report["summary"]["failed"] or report["summary"]["provider_runtime_failures"])


if __name__ == "__main__":
    raise SystemExit(main())
