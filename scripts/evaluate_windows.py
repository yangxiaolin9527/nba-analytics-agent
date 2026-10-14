"""Score typed window queries against independently checked Eoin v515 cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluation_contract import report_provenance
from scripts.relational_baseline import CAPABILITY_MATRIX


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "metadata/m1_10a_window_cases_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))
    return actual == expected


def _source_matches(body: dict[str, Any], request: dict[str, Any], table: str) -> bool:
    sql = body.get("source_sql") or ""
    alias = "p" if request["subject"] == "player" else "t"
    required = [
        f"FROM {table} {alias}", "JOIN game g", f"{alias}." +
        ("person_id" if alias == "p" else "team_id") + " = ?",
        "g.season_id = ?", "g.competition_phase = ?",
        f"ORDER BY g.game_date ASC, {alias}.game_id ASC",
    ]
    if alias == "p":
        required.append("appearance_status IN ('played', 'played_inferred')")
    else:
        required.append("JOIN game_team_participation")
    params = [request["entity_id"], request["season_id"], request["competition_phase"]]
    for field, predicate in (("date_from", "g.game_date >= ?"), ("date_to", "g.game_date <= ?")):
        if field in request:
            required.append(predicate)
            params.append(request[field])
    if "opponent_team_id" in request:
        required.append(f"{alias}.opponent_team_id = ?")
        params.append(request["opponent_team_id"])
    return (all(fragment in sql for fragment in required) and sql.count("?") == len(params)
            and body.get("source_parameters") == params)


def score_window_response(
    body: dict[str, Any], case: dict[str, Any], *, expected_source_snapshot: str,
) -> dict[str, Any]:
    """Require exact scope, selected games, values, quality, and replay evidence."""
    expected = case.get("expected")
    request = case.get("request")
    if not isinstance(expected, dict) or not expected or not isinstance(request, dict):
        return {"outcome": "unscorable", "reason": "missing_exact_expectation"}
    evidence = body.get("evidence") or {}
    evidence_match = (
        evidence.get("source_snapshot") == expected_source_snapshot
        and isinstance(evidence.get("snapshot_id"), str) and bool(evidence["snapshot_id"])
        and isinstance(evidence.get("catalog_version"), str) and bool(evidence["catalog_version"])
        and evidence.get("requested_scope") == request
        and evidence.get("selection_rule") == "filter_then_date_game_order_then_last_n"
        and all(evidence.get(key) == request.get(key) for key in (
            "subject", "entity_id", "season_id", "competition_phase",
        ))
    )
    result_match = body.get("status") == expected.get("status")
    if expected.get("status") == "not_computable":
        if any(key not in expected for key in ("reason_code", "quality_status", "limitations")):
            return {"outcome": "unscorable", "reason": "incomplete_exact_expectation"}
        result_match &= (
            body.get("reason_code") == expected.get("reason_code")
            and body.get("values") == {} and body.get("selected_game_ids") == []
            and body.get("actual_sample_count") == 0
            and body.get("quality_status") == expected["quality_status"]
            and set(body.get("limitations", [])) == set(expected["limitations"])
        )
    else:
        required = (
            "values", "actual_sample_count", "available_sample_count", "selected_game_ids",
            "actual_date_from", "actual_date_to", "quality_status", "limitations", "source_table",
        )
        if any(key not in expected for key in required):
            return {"outcome": "unscorable", "reason": "incomplete_exact_expectation"}
        result_match &= all(
            _equal(body.get(key), expected[key]) if key != "values" else
            isinstance(body.get("values"), dict) and body["values"].keys() == expected["values"].keys()
            and all(_equal(body["values"][metric], value) for metric, value in expected["values"].items())
            for key in required if key != "limitations"
        )
        result_match &= (
            isinstance(body.get("limitations"), list)
            and len(body["limitations"]) == len(set(body["limitations"]))
            and set(body["limitations"]) == set(expected["limitations"])
        )
        evidence_match &= _source_matches(body, request, expected["source_table"])
        result_match &= (body.get("actual_sample_count") == len(body.get("selected_game_ids", []))
                         and body.get("available_sample_count", -1) >= body.get("actual_sample_count", 0))
    if not result_match:
        reason = "result_or_quality_mismatch"
    elif not evidence_match:
        reason = "evidence_mismatch"
    else:
        reason = None
    return {
        "outcome": "passed" if reason is None else "failed", "reason": reason,
        "result_match": bool(result_match), "evidence_match": bool(evidence_match),
    }


async def run_evaluation(
    db_path: Path, *, cases_path: Path = CASES, matrix_path: Path = CAPABILITY_MATRIX,
) -> dict[str, Any]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if document.get("source_snapshot") != catalog["source_snapshot"]:
        raise ValueError("window cases source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("window case set must be non-empty")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("window case IDs must be unique and non-empty")
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)
    records = []
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for case in cases:
            started = time.monotonic()
            response = await client.post("/query/window", json=case["request"])
            body = response.json()
            scored = score_window_response(body, case, expected_source_snapshot=catalog["source_snapshot"])
            records.append({
                "case_id": case["case_id"], "http_status": response.status_code,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "outcome": scored["outcome"] if response.status_code == 200 else "failed",
                "reason": scored["reason"] if response.status_code == 200 else "http_status_mismatch",
                "checks": {"result": scored.get("result_match"), "evidence": scored.get("evidence_match")},
                "request": case["request"], "response": body,
            })
    return {
        "case_set_version": document["case_set_version"],
        "source_snapshot": catalog["source_snapshot"], "catalog_version": catalog["catalog_version"],
        "summary": {name: sum(row["outcome"] == name for row in records)
                    for name in ("passed", "failed", "unscorable")} | {"total": len(records)},
        "cases": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=catalog["source_snapshot"], model_id="deterministic",
            route="window_query", input_paths={
                "cases": cases_path, "matrix": matrix_path, "catalog": CATALOG,
                "window_query": ROOT / "scripts/window_query.py",
                "window_routes": ROOT / "app/window_routes.py",
                "schemas": ROOT / "app/schemas.py",
                "query_quality": ROOT / "scripts/query_quality.py",
                "evaluator": ROOT / "scripts/evaluate_windows.py",
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--capability-matrix", type=Path, default=CAPABILITY_MATRIX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_evaluation(
            args.db, cases_path=args.cases, matrix_path=args.capability_matrix,
        ))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(report["summary"]["passed"] != report["summary"]["total"])


if __name__ == "__main__":
    raise SystemExit(main())
