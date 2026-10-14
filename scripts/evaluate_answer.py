"""Evaluate the deterministic structured answer API with fixed or live planning."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.evaluation_contract import report_provenance
from scripts.model_adapter import FakeModel, ModelResponse
from scripts.relational_baseline import CAPABILITY_MATRIX

CASES = ROOT / "metadata/m3_1_answer_cases_v1.json"
CATALOG = ROOT / "metadata/schema_catalog.json"


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def score_response(body: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    checks = {
        "status": body.get("status") == expected["status"],
        "tool": body.get("tool") == expected["tool"],
        "arguments": _same(body.get("arguments"), expected["arguments"]),
        "tool_calls": body.get("tool_calls_used") == expected["tool_calls_used"],
    }
    if "reason_code" in expected:
        checks["reason_code"] = body.get("reason_code") == expected["reason_code"]
    if expected["tool"] is not None:
        result = body.get("tool_result") or {}
        checks["result_status"] = result.get("status") in {"complete", "partial"}
        checks["oracle_result"] = _same(result, expected["result"])
        checks["answer_present"] = isinstance(body.get("answer"), str) and bool(body["answer"].strip())
        checks["answer_values"] = all(
            fragment in body.get("answer", "") for fragment in expected["answer_contains"]
        )
        checks["query_metadata"] = bool(body.get("query_metadata", {}).get("snapshot_id"))
        checks["evidence"] = isinstance(body.get("evidence"), dict) and bool(
            body["evidence"].get("source_table") or body["evidence"].get("formula")
            or body["evidence"].get("source_snapshot")
        )
        quality = expected.get("quality_status")
        if quality:
            checks["quality"] = body.get("quality_status") == quality and bool(body.get("limitations"))
        if result.get("status") == "partial":
            checks["partial_disclosed"] = body.get("status") == "complete" and "partial_result" in body.get("warnings", [])
    else:
        checks["no_tool_result"] = body.get("tool_result") is None and body.get("tool_calls_used") == 0
    return {"outcome": "passed" if all(checks.values()) else "failed", "checks": checks}


async def run_evaluation(
    db_path: Path, *, mode: str = "fixture", cases_path: Path = CASES,
    matrix_path: Path = CAPABILITY_MATRIX, include_holdout: bool = False,
) -> dict[str, Any]:
    if mode not in {"fixture", "model"}:
        raise ValueError("mode must be fixture or model")
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if document.get("source_snapshot") != catalog.get("source_snapshot"):
        raise ValueError("answer cases source snapshot does not match catalog")
    cases = document.get("cases")
    ids = [case.get("case_id") for case in cases or []]
    if not cases or len(ids) != len(set(ids)) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("answer case set must have unique non-empty IDs")
    if mode == "model" and not include_holdout:
        cases = [case for case in cases if case.get("split") == "development"]
    records = []
    for case in cases:
        model_output = case["plan"]
        if isinstance(model_output, dict):
            model_output = json.dumps(model_output, ensure_ascii=False)
        selected_model = None
        timeout = 7.0
        if mode == "fixture":
            if case.get("simulate_timeout"):
                class SlowModel:
                    def generate(self, request):
                        time.sleep(0.1)
                        return ModelResponse("{}", "slow-fixture")
                selected_model = SlowModel()
                timeout = 0.01
            else:
                selected_model = FakeModel({None: model_output})
        app = create_app(
            db_path=db_path, quality_matrix_path=matrix_path, model=selected_model,
            ask_model_timeout_seconds=timeout,
        )
        started = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/answer", json={"question": case["question"]})
        elapsed = round((time.monotonic() - started) * 1000, 3)
        body = response.json()
        scored = score_response(body, case) if response.status_code == 200 else {"outcome": "failed", "checks": {"http": False}}
        records.append({
            "case_id": case["case_id"], "split": case["split"], "http_status": response.status_code,
            "elapsed_ms": elapsed, "outcome": scored["outcome"], "checks": scored["checks"],
            "failure_reason": body.get("reason_code") if scored["outcome"] != "passed" else None,
            "tool_calls_used": body.get("tool_calls_used"), "response": body,
        })
    summary = {
        "total": len(records), "passed": sum(row["outcome"] == "passed" for row in records),
        "failed": sum(row["outcome"] == "failed" for row in records),
        "mean_latency_ms": round(sum(row["elapsed_ms"] for row in records) / len(records), 3),
        "tool_calls": sum(row["tool_calls_used"] or 0 for row in records),
    }
    return {
        "case_set_version": document["case_set_version"], "mode": mode,
        "source_snapshot": catalog["source_snapshot"], "catalog_version": catalog["catalog_version"],
        "summary": summary, "cases": records,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=catalog["source_snapshot"],
            model_id="fixture" if mode == "fixture" else "configured_model",
            route="answer_one_tool_deterministic_render", input_paths={
                "cases": cases_path, "catalog": CATALOG, "matrix": matrix_path,
                "route": ROOT / "app/answer_routes.py", "renderer": ROOT / "scripts/answer_renderer.py",
                "evidence": ROOT / "app/answer_evidence.py", "ask_flow": ROOT / "app/ask_routes.py",
                "tools": ROOT / "app/ask_tools.py", "evaluator": ROOT / "scripts/evaluate_answer.py",
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
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
