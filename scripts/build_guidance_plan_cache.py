"""Materialize a fixed planner-plan cache from an audited guided report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def cases_hash(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite plan cache")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    report_cases = {row["id"]: row for row in json.loads(args.report.read_text(encoding="utf-8"))["cases"]}
    selected = [case for case in cases if case["id"] in report_cases]
    if not selected:
        raise ValueError("report does not cover any selected cases")
    plans = {}
    for case in selected:
        row = report_cases[case["id"]]
        rewritten = row.get("rewritten_query")
        if case["answerable"] and not rewritten:
            raise ValueError(f"answerable case {case['id']} has no rewritten query")
        if not case["answerable"] or not rewritten:
            plans[case["question"]] = {"action": "abstain", "intent": "outside", "english_query": None}
            continue
        source_ids = {ref["source_id"] for ref in case.get("relevant", [])}
        intent = "glossary" if source_ids == {"nba-stats-glossary"} else "rules"
        plans[case["question"]] = {"action": "search", "intent": intent, "english_query": rewritten}
    payload = {
        "schema_version": 1,
        "cases_sha256": cases_hash(selected),
        "source_report": str(args.report),
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
