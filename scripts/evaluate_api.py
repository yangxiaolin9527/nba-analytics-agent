"""Evaluate the structured query API against versioned M1 cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI
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


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "metadata/agent_evaluation_baseline_v1.json"
ORACLES = ROOT / "metadata/agent_evaluation_oracles_v1.json"
FIXTURES = ROOT / "metadata/sql_gateway_fixtures_v1.json"
QUALITY_EXPECTATIONS = ROOT / "metadata/query_quality_expectations_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"


class FixedSQLModel:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(self.sql, "fixed-sql")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_evidence() -> dict[str, str]:
    catalog = _load_json(CATALOG)
    name, version = catalog["source_snapshot"].rsplit("@", 1)
    return {
        "source_name": name,
        "source_version": version,
        "snapshot_id": "eoin-v515",
        "catalog_version": catalog["catalog_version"],
    }


def _equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


async def evaluate_case(app: FastAPI, case: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"question": case["question"]})
    request_wall_ms = round((time.monotonic() - started) * 1000, 3)
    body = response.json()
    record: dict[str, Any] = {
        "case_id": case["case_id"],
        "source_case": case["source_case"],
        "http_status": response.status_code,
        "request_wall_ms": request_wall_ms,
        "gateway_status": body.get("status"),
        "route": body.get("route"),
        "result_status": body.get("result_status"),
        "timings_ms": {
            "total": body.get("elapsed_ms"),
            "model": body.get("model_elapsed_ms"),
            "gateway": body.get("gateway_elapsed_ms"),
        },
        "sql": body.get("sql"),
        "result_match": None,
        "evidence_match": body.get("evidence") == _expected_evidence(),
        "quality_status": body.get("quality_status"),
        "limitations": body.get("limitations", []),
        "outcome": "failed",
        "reason": None,
        "response": body,
    }
    if response.status_code != 200:
        record["reason"] = body.get("detail", "http_error")
        return record
    if case["source_case"] == "AG-017":
        record["result_match"] = body.get("status") == "rejected" and body.get("rows") == []
        record["quality_match"] = None
        record["outcome"] = "passed" if record["result_match"] and record["evidence_match"] and bool(body.get("reason")) else "failed"
        record["reason"] = None if record["outcome"] == "passed" else "safety_rejection_mismatch"
        return record
    if body.get("status") != "ok":
        record["reason"] = "gateway_not_ok"
        return record
    if body.get("result_status") != "complete":
        record["reason"] = "result_not_complete"
        return record
    expected = case.get("expected")
    if not isinstance(expected, dict):
        record["outcome"] = "contract_only"
        record["reason"] = "no_exact_oracle"
        return record
    scored = score_plan_response(body, case, expected_evidence=_expected_evidence())
    record.update(scored)
    return record


def _load_cases(
    cases_path: Path,
    *,
    quality_path: Path = QUALITY_EXPECTATIONS,
) -> tuple[list[dict[str, Any]], list[str]]:
    baseline = _load_json(BASELINE)
    oracles = _load_json(ORACLES)
    snapshot = _load_json(CATALOG)["source_snapshot"]
    fixtures = _load_json(cases_path)
    if any(source["source_snapshot"] != snapshot for source in (baseline, oracles, fixtures)):
        raise ValueError("evaluation metadata source snapshots disagree")
    baseline_cases = baseline.get("cases")
    if not isinstance(baseline_cases, list) or not baseline_cases:
        raise ValueError("baseline must contain cases")
    baseline_ids = [case.get("case_id") for case in baseline_cases]
    if len(baseline_ids) != len(set(baseline_ids)) or any(not value for value in baseline_ids):
        raise ValueError("baseline case IDs must be unique and non-empty")
    by_id = {case["case_id"]: case for case in baseline_cases}
    validated_fixtures = validate_case_document(
        fixtures, source_snapshot=snapshot,
        allow_missing_expected=frozenset(case["case_id"] for case in fixtures.get("cases", [])
                                         if case.get("source_case") == "AG-017"),
    )
    quality_document = _load_json(quality_path)
    quality_expectations = validate_quality_expectations(quality_document, source_snapshot=snapshot)
    scored: list[dict[str, Any]] = []
    for fixture in validated_fixtures:
        source_id = fixture["source_case"]
        if source_id not in by_id:
            raise ValueError(f"unknown evaluation case: {source_id}")
        case = {**fixture, "question": fixture.get("question", by_id[source_id]["query"])}
        oracle = oracles["cases"].get(source_id)
        if oracle is not None:
            expected = case.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise ValueError(f"missing exact expectation: {source_id}")
            if any(not _equal(oracle.get(key), value) for key, value in expected.items()):
                raise ValueError(f"fixture disagrees with oracle: {source_id}")
            if source_id not in quality_expectations:
                raise ValueError(f"missing quality expectation: {source_id}")
            case["expected_quality"] = quality_expectations[source_id]
        if oracle is not None or source_id == "AG-017":
            scored.append(case)
    exact_ids = set(oracles["cases"])
    contract_only = sorted(set(by_id) - exact_ids - {"AG-017"})
    return scored, contract_only


async def run_evaluation(
    db_path: Path,
    *,
    mode: str = "fixture",
    execution_mode: str = "direct",
    cases_path: Path = FIXTURES,
    quality_path: Path = QUALITY_EXPECTATIONS,
    matrix_path: Path | None = None,
    model: TextToSQLModel | None = None,
) -> dict[str, Any]:
    if mode not in {"fixture", "model"}:
        raise ValueError("mode must be fixture or model")
    if execution_mode not in {"plan", "direct"}:
        raise ValueError("execution_mode must be plan or direct")
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    selected_matrix_path = matrix_path or CAPABILITY_MATRIX
    cases, contract_only = _load_cases(cases_path, quality_path=quality_path)
    if mode == "model":
        cases = [case for case in cases if case["source_case"] != "AG-017"]
        model = model if model is not None else OpenAICompatibleModel()
    if not cases:
        raise ValueError("no scorable cases for selected mode")
    records = []
    for case in cases:
        selected_model = model if mode == "model" else FixedSQLModel(case["sql"])
        records.append(await evaluate_case(
            create_app(
                model=selected_model,
                db_path=db_path,
                execution_mode=execution_mode,
                context_mode="schema_value_retrieval",
                quality_matrix_path=selected_matrix_path,
                route="direct_schema_value" if execution_mode == "direct" else "query_plan_model",
            ), case
        ))
    summary = {name: sum(record["outcome"] == name for record in records)
               for name in ("passed", "failed", "unscorable")}
    summary["contract_only"] = len(contract_only)
    summary["total"] = len(records)
    summary["quality_mismatches"] = sum(record.get("quality_match") is False for record in records)
    catalog = _load_json(CATALOG)
    model_id = "fixed-sql" if mode == "fixture" else str(getattr(model, "model", type(model).__name__))
    return {
        "mode": mode,
        "model_id": model_id,
        "source_snapshot": catalog["source_snapshot"],
        "catalog_version": catalog["catalog_version"],
        "summary": summary,
        "contract_only_case_ids": contract_only,
        "cases": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=catalog["source_snapshot"], model_id=model_id,
            route=f"{execution_mode}_schema_value",
            input_paths={
                "cases": cases_path,
                "quality_expectations": quality_path,
                "capability_matrix": selected_matrix_path,
                "catalog": CATALOG,
                "prompt_builder": ROOT / "scripts/prompt_builder.py",
                "evaluation_contract": ROOT / "scripts/evaluation_contract.py",
                "evaluator": ROOT / "scripts/evaluate_api.py",
                "quality_assessor": ROOT / "scripts/query_quality.py",
                "api": ROOT / "app/main.py",
            },
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixture", "model"), default="fixture")
    parser.add_argument("--execution-mode", choices=("plan", "direct"), default="direct")
    parser.add_argument("--cases", type=Path, default=FIXTURES)
    parser.add_argument("--quality-expectations", type=Path, default=QUALITY_EXPECTATIONS)
    parser.add_argument("--capability-matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_evaluation(
            args.db, mode=args.mode, execution_mode=args.execution_mode, cases_path=args.cases,
            quality_path=args.quality_expectations, matrix_path=args.capability_matrix,
        ))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(report["summary"]["failed"] > 0 or report["summary"]["unscorable"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
