"""Evaluate one-tool natural-language routing with fixed plans and optional model calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluate_comparisons import score_comparison_response
from scripts.evaluate_efg import score_efg_response
from scripts.evaluate_windows import score_window_response
from scripts.evaluation_contract import report_provenance
from scripts.model_adapter import FakeModel
from scripts.relational_baseline import CAPABILITY_MATRIX


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "metadata/m1_11_ask_cases_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"
ORACLE_PATHS = {
    "window": ROOT / "metadata/m1_10a_window_cases_v1.json",
    "comparison": ROOT / "metadata/m1_9a_comparison_cases_v1.json",
    "efg": ROOT / "metadata/m1_9b_efg_cases_v1.json",
}


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _equal(actual[key], value)
                                                for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _equal(left, right) for left, right in zip(actual, expected))
    return actual == expected


def score_ask_response(
    body: dict[str, Any], case: dict[str, Any], *, expected_source_snapshot: str,
    oracle_cases: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    expected = case.get("expected")
    if not isinstance(expected, dict) or any(key not in expected for key in (
        "status", "tool", "arguments", "tool_calls_used",
    )):
        return {"outcome": "unscorable", "reason": "missing_exact_expectation"}
    route_match = body.get("status") == expected["status"] and body.get("tool") == expected["tool"]
    grounding_match = body.get("arguments") == expected["arguments"]
    budget_match = body.get("tool_calls_used") == expected["tool_calls_used"]
    evidence = body.get("evidence") or {}
    evidence_match = (
        evidence.get("source_name", "") + "@" + evidence.get("source_version", "") == expected_source_snapshot
        and isinstance(evidence.get("snapshot_id"), str) and bool(evidence["snapshot_id"])
        and isinstance(evidence.get("catalog_version"), str) and bool(evidence["catalog_version"])
    )
    raw_result = body.get("tool_result")
    result = raw_result if isinstance(raw_result, dict) else {}
    if expected["tool"] is None:
        result_match = (raw_result is None and expected["tool_calls_used"] == 0
                        and body.get("reason_code") == expected.get("reason_code"))
    else:
        result_match = isinstance(raw_result, dict) and result.get("status") == expected["status"]
        result_match &= _equal(result, expected.get("result", {}))
        reference = case.get("oracle_ref")
        if reference:
            if oracle_cases is None or reference.get("kind") not in oracle_cases:
                return {"outcome": "unscorable", "reason": "oracle_unavailable"}
            oracle = oracle_cases[reference["kind"]].get(reference.get("case_id"))
            if oracle is None:
                return {"outcome": "unscorable", "reason": "oracle_case_missing"}
            if reference["kind"] == "window":
                grounding_match &= expected["arguments"] == oracle["request"]
                score = score_window_response(result, oracle, expected_source_snapshot=expected_source_snapshot)
            elif reference["kind"] == "comparison":
                grounding_match &= all(expected["arguments"].get(key) == oracle.get(key) for key in (
                    "subject", "entity_id", "metric", "baseline_season", "comparison_season", "competition_phase",
                ))
                score = score_comparison_response(result, oracle, expected_source_snapshot=expected_source_snapshot)
            else:
                grounding_match &= all(expected["arguments"].get(key) == oracle.get(key) for key in (
                    "entity_id", "season_id", "competition_phase",
                ))
                score = score_efg_response(result, oracle, expected_source_snapshot=expected_source_snapshot)
            result_match &= score["outcome"] == "passed"
            inner_evidence = result.get("evidence") or {}
            evidence_match &= (inner_evidence.get("snapshot_id") == evidence.get("snapshot_id")
                               and inner_evidence.get("catalog_version") == evidence.get("catalog_version"))
        else:
            if not isinstance(expected.get("result"), dict) or not expected["result"]:
                return {"outcome": "unscorable", "reason": "missing_exact_result"}
            result_match &= (
                isinstance(result.get("sql"), str) and result["sql"].startswith("SELECT ")
                and result.get("source_table", "") in result["sql"]
                and result.get("source_snapshot") == expected_source_snapshot
                and isinstance(result.get("quality_status"), str)
                and isinstance(result.get("limitations"), list)
            )
    checks = {
        "route": bool(route_match), "grounding": bool(grounding_match),
        "budget": bool(budget_match), "result": bool(result_match), "evidence": bool(evidence_match),
    }
    reason = next((name + "_mismatch" for name, passed in checks.items() if not passed), None)
    return {"outcome": "passed" if reason is None else "failed", "reason": reason, "checks": checks}


def _load_oracles(source_snapshot: str) -> dict[str, dict[str, dict[str, Any]]]:
    loaded = {}
    for kind, path in ORACLE_PATHS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("source_snapshot") != source_snapshot:
            raise ValueError(f"{kind} oracle snapshot does not match catalog")
        loaded[kind] = {case["case_id"]: case for case in document["cases"]}
    return loaded


async def run_evaluation(
    db_path: Path, *, mode: str = "fixture", cases_path: Path = CASES,
    matrix_path: Path = CAPABILITY_MATRIX, include_holdout: bool = False,
) -> dict[str, Any]:
    if mode not in {"fixture", "model"}:
        raise ValueError("mode must be fixture or model")
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    snapshot = catalog["source_snapshot"]
    if document.get("source_snapshot") != snapshot:
        raise ValueError("ask cases source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("ask case set must be non-empty")
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("ask case IDs must be unique and non-empty")
    if mode == "model" and not include_holdout:
        cases = [case for case in cases if case.get("split") == "development"]
    oracles = _load_oracles(snapshot)
    records = []
    for case in cases:
        model_output = case.get("model_output")
        if isinstance(model_output, dict):
            model_output = json.dumps(model_output, ensure_ascii=False)
        if mode == "fixture" and not isinstance(model_output, str):
            raise ValueError(f"missing fixed model output for {case['case_id']}")
        app = create_app(
            db_path=db_path, quality_matrix_path=matrix_path,
            model=FakeModel({None: model_output}) if mode == "fixture" else None,
        )
        started = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ask", json={"question": case["question"]})
        body = response.json()
        scored = score_ask_response(
            body, case, expected_source_snapshot=snapshot, oracle_cases=oracles,
        )
        records.append({
            "case_id": case["case_id"], "split": case.get("split"),
            "http_status": response.status_code,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "outcome": scored["outcome"] if response.status_code == 200 else "failed",
            "reason": scored["reason"] if response.status_code == 200 else "http_status_mismatch",
            "checks": scored.get("checks"), "response": body,
        })
    return {
        "case_set_version": document["case_set_version"], "mode": mode,
        "source_snapshot": snapshot, "catalog_version": catalog["catalog_version"],
        "summary": {name: sum(row["outcome"] == name for row in records)
                    for name in ("passed", "failed", "unscorable")} | {"total": len(records)},
        "cases": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=snapshot,
            model_id="fixture" if mode == "fixture" else "configured_model",
            route="ask_one_tool", input_paths={
                "cases": cases_path, "window_oracles": ORACLE_PATHS["window"],
                "comparison_oracles": ORACLE_PATHS["comparison"], "efg_oracles": ORACLE_PATHS["efg"],
                "matrix": matrix_path, "catalog": CATALOG,
                "prompt": ROOT / "scripts/ask_plan.py", "grounding": ROOT / "scripts/ask_grounding.py",
                "routes": ROOT / "app/ask_routes.py", "tools": ROOT / "app/ask_tools.py",
                "schemas": ROOT / "app/schemas.py", "model_adapter": ROOT / "scripts/model_adapter.py",
                "evaluator": ROOT / "scripts/evaluate_ask.py",
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "model"), default="fixture")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--capability-matrix", type=Path, default=CAPABILITY_MATRIX)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_evaluation(
            args.db, mode=args.mode, cases_path=args.cases,
            matrix_path=args.capability_matrix, include_holdout=args.include_holdout,
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
