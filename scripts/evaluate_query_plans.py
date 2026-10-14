"""Evaluate deterministic M1.8B plans through the real ASGI endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluation_contract import (
    report_provenance,
    score_plan_response,
    validate_case_document,
    validate_quality_expectations,
)
from scripts.model_adapter import ModelRequest, ModelResponse, OpenAICompatibleModel, TextToSQLModel
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.sql_gateway import DEFAULT_CATALOG

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "metadata/query_plan_cases_v1.json"
QUALITY_EXPECTATIONS = ROOT / "metadata/query_quality_expectations_v1.json"


class FixedPlanModel:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(json.dumps(self.plan, ensure_ascii=False), "fixed-plan")


async def evaluate_case(
    db_path: Path,
    case: dict[str, Any],
    *,
    model: TextToSQLModel | None = None,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> dict[str, Any]:
    selected_model = model or FixedPlanModel(case["plan"])
    started = time.monotonic()
    app = create_app(model=selected_model, db_path=db_path, quality_matrix_path=matrix_path,
                     execution_mode="plan", context_mode="schema_value_retrieval",
                     route="query_plan_model" if model is not None else "query_plan_fixed")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"question": case["question"]})
    request_wall_ms = round((time.monotonic() - started) * 1000, 3)
    body = response.json()
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    source_name, source_version = catalog["source_snapshot"].rsplit("@", 1)
    expected_evidence = {
        "source_name": source_name, "source_version": source_version,
        "snapshot_id": "eoin-v515", "catalog_version": catalog["catalog_version"],
    }
    scored = score_plan_response(body, case, expected_evidence=expected_evidence)
    if response.status_code != 200:
        scored.update(outcome="failed", reason="http_error", result_match=False)
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        **scored,
        "status": body.get("status"),
        "result_status": body.get("result_status"),
        "quality_status": body.get("quality_status"),
        "quality_match": scored["quality_match"],
        "limitations": body.get("limitations", []),
        "plan_error_code": body.get("plan_error_code"),
        "sql": body.get("sql", ""),
        "request_wall_ms": request_wall_ms,
        "timings_ms": {
            "total": body.get("elapsed_ms"), "model": body.get("model_elapsed_ms"),
            "gateway": body.get("gateway_elapsed_ms"), "plan": body.get("plan_elapsed_ms"),
            "compile": body.get("compile_elapsed_ms"),
        },
        "response": body,
    }


async def run(
    db_path: Path,
    cases_path: Path,
    split: str,
    *,
    mode: str = "fixed",
    model: TextToSQLModel | None = None,
    quality_path: Path = QUALITY_EXPECTATIONS,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> dict[str, Any]:
    if mode not in {"fixed", "model"}:
        raise ValueError("mode must be fixed or model")
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    all_cases = validate_case_document(document, source_snapshot=catalog["source_snapshot"])
    quality_document = json.loads(quality_path.read_text(encoding="utf-8"))
    quality_expectations = validate_quality_expectations(
        quality_document, source_snapshot=catalog["source_snapshot"]
    )
    if not any(case.get("split") == split for case in all_cases):
        raise ValueError(f"no cases in split: {split}")
    cases = [case for case in all_cases if case["split"] == split]
    for case in cases:
        if case["case_id"] not in quality_expectations:
            raise ValueError(f"missing quality expectation: {case['case_id']}")
        case["expected_quality"] = quality_expectations[case["case_id"]]
    selected_model = model if mode == "model" else None
    if mode == "model" and selected_model is None:
        selected_model = OpenAICompatibleModel()
    records = [await evaluate_case(db_path, case, model=selected_model, matrix_path=matrix_path) for case in cases]
    passed = sum(item["outcome"] == "passed" for item in records)
    failed = sum(item["outcome"] == "failed" for item in records)
    unscorable = sum(item["outcome"] == "unscorable" for item in records)
    elapsed = [item["request_wall_ms"] for item in records]
    summary = {
        "total": len(records), "passed": passed, "failed": failed, "unscorable": unscorable,
        "pass_rate": passed / len(records) if records else math.nan,
        "quality_conditional": sum(item["quality_status"] == "conditional" for item in records),
        "quality_mismatches": sum(item["quality_match"] is False for item in records),
        "evidence_mismatches": sum(not item["evidence_match"] for item in records),
        "latency_ms": {
            "mean": round(statistics.mean(elapsed), 3) if elapsed else None,
            "p50": round(statistics.median(elapsed), 3) if elapsed else None,
            "max": round(max(elapsed), 3) if elapsed else None,
        },
    }
    return {
        "mode": mode, "model_id": getattr(selected_model, "model", "fixed-plan"),
        "split": split, "source_snapshot": document["source_snapshot"],
        "total": len(records), "passed": passed, "summary": summary, "records": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=document["source_snapshot"],
            model_id=str(getattr(selected_model, "model", "fixed-plan")),
            route="query_plan_model" if mode == "model" else "query_plan_fixed",
            input_paths={
                "cases": cases_path,
                "quality_expectations": quality_path,
                "capability_matrix": matrix_path,
                "catalog": DEFAULT_CATALOG,
                "prompt_builder": ROOT / "scripts/prompt_builder.py",
                "evaluation_contract": ROOT / "scripts/evaluation_contract.py",
                "evaluator": ROOT / "scripts/evaluate_query_plans.py",
                "query_plan_compiler": ROOT / "scripts/query_plan_compiler.py",
                "quality_assessor": ROOT / "scripts/query_quality.py",
                "api": ROOT / "app/main.py",
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--mode", choices=("fixed", "model"), default="fixed")
    parser.add_argument("--quality-expectations", type=Path, default=QUALITY_EXPECTATIONS)
    parser.add_argument("--capability-matrix", type=Path, default=CAPABILITY_MATRIX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run(
        args.db, args.cases, args.split, mode=args.mode,
        quality_path=args.quality_expectations, matrix_path=args.capability_matrix,
    ))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2, allow_nan=False))
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    raise SystemExit(main())
