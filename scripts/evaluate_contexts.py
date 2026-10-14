"""Compare the three M1.8a Text-to-SQL contexts against verified development oracles."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from app.main import create_app
from scripts.evaluate_api import _equal, evaluate_case
from scripts.model_adapter import ModelRequest, ModelResponse, OpenAICompatibleModel, TextToSQLModel
from scripts.query_context import MODES
from scripts.sql_gateway import DEFAULT_CATALOG, GatewayStatus, SqlGateway
from scripts.provider_config import ProviderConfig, load_provider_config
from scripts.tracing import configured_tracer, span


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "metadata" / "context_experiment_cases_v1.json"


class RecordingModel:
    def __init__(self, delegate: TextToSQLModel) -> None:
        self.delegate = delegate
        self.prompt = ""
        self.output = ""

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.prompt = request.prompt
        response = self.delegate.generate(request)
        self.output = response.text
        return response


def load_cases(cases_path: Path, db_path: Path) -> list[dict[str, Any]]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    if document["source_snapshot"] != catalog["source_snapshot"]:
        raise ValueError("experiment snapshot does not match catalog")
    cases = document["cases"]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("experiment cases must have unique IDs")
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        gateway = SqlGateway(connection, max_rows=2)
        for case in cases:
            expected = case.get("expected")
            if not isinstance(expected, dict) or not expected:
                raise ValueError(f"case has no exact oracle: {case['case_id']}")
            result = gateway.execute(case["oracle_sql"])
            if result.status is not GatewayStatus.OK or len(result.rows) != 1:
                raise ValueError(f"oracle query failed: {case['case_id']}: {result.reason}")
            actual = dict(zip(result.columns, result.rows[0]))
            if not expected.keys() <= actual.keys() or any(
                not _equal(actual[key], value) for key, value in expected.items()
            ):
                raise ValueError(f"oracle mismatch: {case['case_id']}")
    return cases


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 3)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in MODES:
        subset = [record for record in records if record["context_mode"] == mode]
        durations = {
            key: [record["timings_ms"][key] for record in subset
                  if isinstance(record["timings_ms"].get(key), (int, float))]
            for key in ("total", "context", "model", "gateway")
        }
        hits = [record["linking_hit"] for record in subset if record["linking_hit"] is not None]
        summary[mode] = {
            "total": len(subset),
            "passed": sum(record["outcome"] == "passed" for record in subset),
            "failed": sum(record["outcome"] == "failed" for record in subset),
            "unscorable": sum(record["outcome"] == "unscorable" for record in subset),
            "linking_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
            "average_prompt_chars": round(statistics.mean(record["prompt_chars"] for record in subset)) if subset else None,
            "failure_categories": dict(Counter(record["reason"] or "passed" for record in subset)),
            "timings_ms": {key: {"p50": _percentile(values, 0.5),
                                 "p95": _percentile(values, 0.95)}
                           for key, values in durations.items()},
        }
    return summary


async def run_experiment(
    db_path: Path, *, cases_path: Path = CASES, model: TextToSQLModel | None = None,
    repeats: int = 2, modes: tuple[str, ...] = MODES,
    case_ids: tuple[str, ...] | None = None, tracer=None,
    provider_config: ProviderConfig | None = None,
) -> dict[str, Any]:
    if repeats < 1 or not modes or any(mode not in MODES for mode in modes):
        raise ValueError("invalid experiment repetitions or context modes")
    cases = load_cases(cases_path, db_path)
    if case_ids is not None:
        selected = set(case_ids)
        if not selected <= {case["case_id"] for case in cases}:
            raise ValueError("unknown experiment case ID")
        cases = [case for case in cases if case["case_id"] in selected]
    config = provider_config or (ProviderConfig.empty() if model is not None else load_provider_config())
    selected_model = model if model is not None else OpenAICompatibleModel(config=config)
    query_tracer = tracer if tracer is not None else configured_tracer(config)
    records: list[dict[str, Any]] = []
    for case in cases:
        for mode in modes:
            for repetition in range(1, repeats + 1):
                recording_model = RecordingModel(selected_model)
                application = create_app(
                    model=recording_model, db_path=db_path, context_mode=mode, tracer=query_tracer
                )
                with span(query_tracer, "nba.evaluate_case") as evaluation_span:
                    evaluation_span.set_attribute("input.value", case["question"])
                    evaluation_span.set_attribute("nba.case_id", case["case_id"])
                    evaluation_span.set_attribute("nba.repetition", repetition)
                    record = await evaluate_case(application, {**case, "source_case": case["case_id"]})
                    evaluation_span.set_attribute("output.value", record["outcome"])
                    evaluation_span.set_attribute("nba.reason", record["reason"] or "passed")
                response = record["response"]
                expected_links = case.get("expected_links", {})
                resolved = response.get("resolved_values", {})
                linking_hit = None
                if mode == "schema_value_retrieval" and response.get("status") is not None:
                    linking_hit = all(
                        set(expected_links.get(kind, [])) <= {
                            entity["id"] for entity in resolved.get(kind, [])
                        } for kind in ("players", "teams")
                    ) and set(expected_links.get("phases", [])) <= set(resolved.get("phases", []))
                record.update({
                    "question": case["question"],
                    "context_mode": mode,
                    "repetition": repetition,
                    "prompt": recording_model.prompt,
                    "model_output": recording_model.output,
                    "prompt_chars": response.get("prompt_chars", len(recording_model.prompt)),
                    "resolved_values": resolved,
                    "linking_hit": linking_hit,
                })
                record["timings_ms"]["context"] = response.get("context_elapsed_ms")
                records.append(record)
    valid_experiment = any(record["outcome"] == "passed" for record in records) or any(
        record["gateway_status"] in {"ok", "rejected", "timeout", "truncated", "error"}
        for record in records
    )
    return {
        "source_snapshot": json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))["source_snapshot"],
        "model_id": str(getattr(selected_model, "model", type(selected_model).__name__)),
        "repeats": repeats,
        "oracle_verified": len(cases),
        "valid_experiment": valid_experiment,
        "summary": _summary(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(run_experiment(
            args.db, cases_path=args.cases, repeats=args.repeats,
            modes=tuple(args.mode) if args.mode else MODES,
            case_ids=tuple(args.case_id) if args.case_id else None,
        ))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"experiment setup failed: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"oracle_verified": report["oracle_verified"],
                      "model_id": report["model_id"], "summary": report["summary"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
