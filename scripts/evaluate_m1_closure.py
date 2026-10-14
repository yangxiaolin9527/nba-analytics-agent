"""Run all deterministic M1 evaluations and summarize closure evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.evaluate_api import run_evaluation as run_query_evaluation
from scripts.evaluate_ask import run_evaluation as run_ask_evaluation
from scripts.evaluate_comparisons import run_evaluation as run_comparison_evaluation
from scripts.evaluate_efg import run_evaluation as run_efg_evaluation
from scripts.evaluate_query_plans import run as run_plan_evaluation
from scripts.evaluate_windows import run_evaluation as run_window_evaluation
from scripts.evaluation_contract import report_provenance
from scripts.relational_baseline import CAPABILITY_MATRIX

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "metadata/schema_catalog.json"
MATRIX = ROOT / "metadata/m1_12_requirements_evidence_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("summary", {})
    if not isinstance(value, dict):
        raise ValueError("component report has no summary")
    required = {"passed", "failed", "total"}
    if not required <= value.keys():
        raise ValueError(f"component summary missing keys: {sorted(required - value.keys())}")
    return {key: value.get(key, 0) for key in ("passed", "failed", "unscorable", "contract_only", "total")}


def _validate_snapshot(report: dict[str, Any], snapshot: str) -> None:
    if report.get("source_snapshot") != snapshot:
        raise ValueError("component report source snapshot does not match catalog")
    _summary(report)


async def run_closure(db_path: Path, *, matrix_path: Path = CAPABILITY_MATRIX) -> dict[str, Any]:
    catalog = _load(CATALOG)
    snapshot = catalog["source_snapshot"]
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)

    reports: dict[str, dict[str, Any]] = {}
    reports["query_direct_fixture"] = await run_query_evaluation(
        db_path, mode="fixture", execution_mode="direct", matrix_path=matrix_path
    )
    reports["query_plan_development_fixture"] = await run_plan_evaluation(
        db_path, ROOT / "metadata/query_plan_cases_v1.json", "development", mode="fixed", matrix_path=matrix_path
    )
    reports["query_plan_holdout_fixture"] = await run_plan_evaluation(
        db_path, ROOT / "metadata/query_plan_cases_v1.json", "holdout", mode="fixed", matrix_path=matrix_path
    )
    reports["window_query"] = await run_window_evaluation(db_path, matrix_path=matrix_path)
    reports["season_comparison"] = await run_comparison_evaluation(db_path, matrix_path=matrix_path)
    reports["efg_calculation"] = await run_efg_evaluation(db_path, matrix_path=matrix_path)
    reports["ask_fixture"] = await run_ask_evaluation(db_path, mode="fixture", matrix_path=matrix_path)
    for report in reports.values():
        _validate_snapshot(report, snapshot)

    failed = sum(_summary(report)["failed"] for report in reports.values())
    unscorable = sum(_summary(report)["unscorable"] for report in reports.values())
    passed = sum(_summary(report)["passed"] for report in reports.values())
    return {
        "audit_version": "1.0.0",
        "source_snapshot": snapshot,
        "catalog_version": catalog["catalog_version"],
        "matrix_path": str(matrix_path),
        "summary": {"passed": passed, "failed": failed, "unscorable": unscorable,
                    "components": len(reports)},
        "components": {name: {"summary": _summary(report), "provenance": report.get("provenance")}
                       for name, report in reports.items()},
        "provenance": report_provenance(
            root=ROOT, source_snapshot=snapshot, model_id="fixed-fixtures", route="m1_closure",
            input_paths={"catalog": CATALOG, "matrix": matrix_path,
                         "requirements": MATRIX, "evaluator": Path(__file__)},
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--capability-matrix", type=Path, default=CAPABILITY_MATRIX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_closure(args.db, matrix_path=args.capability_matrix))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"closure evaluation failed: {exc}")
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
