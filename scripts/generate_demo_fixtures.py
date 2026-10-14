"""Generate reproducible demo fixtures by replaying the structured answer flow."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_answer import run_evaluation

SOURCE_CASES = ROOT / "metadata/m3_1_answer_cases_v1.json"
OUTPUT = ROOT / "metadata/m3_4_demo_cases_v2.json"
CATALOG = ROOT / "metadata/schema_catalog.json"

TOOL_ORDER = (
    "season_summary", "window_query", "season_compare", "player_efg", "single_game_score",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_cases() -> list[dict[str, Any]]:
    document = _load_json(SOURCE_CASES)
    selected: list[dict[str, Any]] = []
    for tool in TOOL_ORDER:
        match = next(
            case for case in document["cases"]
            if isinstance(case.get("plan"), dict)
            and case["plan"].get("tool") == tool
            and case.get("expected", {}).get("status") == "complete"
            and not case.get("simulate_timeout")
        )
        selected.append(match)
    return selected


async def generate_demo_catalog(*, db_path: Path, matrix_path: Path, output_path: Path = OUTPUT) -> dict[str, Any]:
    source_document = _load_json(SOURCE_CASES)
    catalog = _load_json(CATALOG)
    selected = _select_cases()
    selected_ids = {case["case_id"] for case in selected}
    cases_path = output_path.with_suffix(".source.json")
    cases_path.write_text(json.dumps({
        "case_set_version": source_document["case_set_version"],
        "source_snapshot": source_document["source_snapshot"],
        "cases": selected,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = await run_evaluation(
        db_path, mode="fixture", cases_path=cases_path,
        matrix_path=matrix_path, include_holdout=True,
    )
    cases = []
    source_by_id = {case["case_id"]: case for case in selected}
    for record in report["cases"]:
        if record["case_id"] not in selected_ids:
            continue
        source_case = source_by_id[record["case_id"]]
        response = record["response"]
        cases.append({
            "case_id": f"M34-FIX-{len(cases) + 1:03d}",
            "source_case_id": record["case_id"],
            "label": source_case["question"],
            "tool": response["tool"],
            "question": source_case["question"],
            "response": response,
        })
    document = {
        "version": "2.0.0",
        "fixture_mode": "answer_flow_replay",
        "source_snapshot": catalog["source_snapshot"],
        "catalog_version": catalog["catalog_version"],
        "cases": cases,
        "provenance": {
            "generator": "scripts/generate_demo_fixtures.py",
            "source_case_file": str(SOURCE_CASES.relative_to(ROOT)),
            "source_case_ids": [case["case_id"] for case in selected],
            "evaluation_route": "answer_one_tool_deterministic_render",
            "evaluation_report_summary": report["summary"],
            "input_provenance": report["provenance"],
        },
    }
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases_path.unlink(missing_ok=True)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--capability-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    document = asyncio.run(generate_demo_catalog(
        db_path=args.db, matrix_path=args.capability_matrix, output_path=args.output,
    ))
    print(json.dumps({"output": str(args.output), "cases": len(document["cases"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
