"""Evaluate player eFG% against versioned exact cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluation_contract import report_provenance
from scripts.relational_baseline import CAPABILITY_MATRIX


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "metadata/m1_9b_efg_cases_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"


def score_efg_response(body: dict[str, Any], case: dict[str, Any], *, expected_source_snapshot: str) -> dict[str, Any]:
    expected = dict(case.get("expected") or {})
    expected.setdefault("source_table", "player_game_stat")
    expected.setdefault("unit", "percent")
    expected.setdefault("formula", "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100")
    inputs = body.get("inputs") or {}
    actual = {
        "status": body.get("status"), "efg_percentage": body.get("efg_percentage"),
        "field_goals_made": inputs.get("field_goals_made"),
        "three_pointers_made": inputs.get("three_pointers_made"),
        "field_goals_attempted": inputs.get("field_goals_attempted"),
        "sample_count": inputs.get("sample_count"), "quality_status": body.get("quality_status"),
        "limitations": body.get("limitations", []), "source_table": body.get("source_table"),
        "unit": (body.get("evidence") or {}).get("unit"),
        "formula": (body.get("evidence") or {}).get("formula"),
    }
    value_keys = tuple(key for key in expected if key not in {"source_table", "unit", "formula"})
    result_match = all(_equal(actual.get(key), expected.get(key)) for key in value_keys)
    evidence = body.get("evidence") or {}
    evidence_match = (
        evidence.get("source_snapshot") == expected_source_snapshot
        and evidence.get("entity_id") == case.get("entity_id")
        and evidence.get("season_id") == case.get("season_id")
        and evidence.get("competition_phase") == case.get("competition_phase")
        and _equal(actual.get("source_table"), expected.get("source_table"))
        and _equal(actual.get("unit"), expected.get("unit"))
        and _equal(actual.get("formula"), expected.get("formula"))
        and bool(body.get("source_sql"))
        and "player_game_stat" in body.get("source_sql", "")
        and "person_id = ?" in body.get("source_sql", "")
        and "season_id = ?" in body.get("source_sql", "")
        and "competition_phase = ?" in body.get("source_sql", "")
    )
    return {"outcome": "passed" if result_match and evidence_match else "failed",
            "reason": None if result_match and evidence_match else ("result_mismatch" if not result_match else "evidence_mismatch"),
            "result_match": result_match, "evidence_match": evidence_match, "actual": actual}


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


async def evaluate_case(app: FastAPI, case: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/calculate/efg", json={
            key: case[key] for key in ("entity_id", "season_id", "competition_phase")
        })
    body = response.json()
    scored = score_efg_response(body, case, expected_source_snapshot=json.loads(CATALOG.read_text(encoding="utf-8"))["source_snapshot"])
    checks = {"result": scored["result_match"], "evidence": scored["evidence_match"]}
    return {
        "case_id": case["case_id"],
        "http_status": response.status_code,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "checks": checks,
        "outcome": "passed" if response.status_code == 200 and scored["outcome"] == "passed" else "failed",
        "reason": scored["reason"],
        "request": {key: case[key] for key in ("entity_id", "season_id", "competition_phase")},
        "response": body,
        "reason_code": body.get("reason_code"),
    }


async def run_evaluation(
    db_path: Path,
    *,
    cases_path: Path = CASES,
    matrix_path: Path | None = None,
) -> dict[str, Any]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if document.get("source_snapshot") != catalog["source_snapshot"]:
        raise ValueError("eFG cases source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("eFG cases must be non-empty")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("eFG case IDs must be unique and non-empty")
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path or CAPABILITY_MATRIX)
    records = [await evaluate_case(app, case) for case in cases]
    return {
        "case_set_version": document.get("case_set_version"),
        "source_snapshot": catalog["source_snapshot"],
        "catalog_version": catalog["catalog_version"],
        "summary": {
            "passed": sum(record["outcome"] == "passed" for record in records),
            "failed": sum(record["outcome"] == "failed" for record in records),
            "total": len(records),
        },
        "cases": records,
        "provenance": report_provenance(
            root=ROOT,
            source_snapshot=catalog["source_snapshot"],
            model_id="deterministic",
            route="efg_calculation",
            input_paths={
                "cases": cases_path,
                "capability_matrix": matrix_path or CAPABILITY_MATRIX,
                "catalog": CATALOG,
                "calculator": ROOT / "scripts/efg_calculation.py",
                "api": ROOT / "app/main.py",
                "routes": ROOT / "app/calculation_routes.py",
                "schemas": ROOT / "app/schemas.py",
                "context": ROOT / "app/context.py",
                "evaluator": ROOT / "scripts/evaluate_efg.py",
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--capability-matrix", type=Path)
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
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
