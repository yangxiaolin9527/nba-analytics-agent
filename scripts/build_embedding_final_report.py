"""Build the final paired embedding report and an auditable review worksheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.embedding_final_analysis import latency_summary, stable_hash


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(report):
    return {row["id"]: row for row in report["cases"]}


def build_worksheet(cases, reports):
    mapped = {name: _case_map(report) for name, report in reports.items()}
    rows = []
    for case in cases:
        for candidate in ("ali", "bge"):
            row = mapped[candidate][case["id"]]
            result = row["result"]
            rows.append({
                "case_id": case["id"], "candidate": candidate,
                "question": case["question"], "language": case["language"],
                "expected_outcome": case["expected_outcome"],
                "answerable": case["answerable"],
                "expected_facts": case.get("expected_facts", []),
                "fact_evidence": case.get("fact_evidence", []),
                "question_scope": case.get("question_scope"),
                "status": result.get("status"),
                "reason_code": result.get("reason_code") or result.get("error_code"),
                "claims": result.get("claims", []),
                "unresolved": result.get("unresolved", []),
                "citation_integrity_auto": None,
                "semantic_support": None,
                "claim_complete": None,
                "correct_abstention": None,
                "runtime_failure": result.get("status") == "error",
                "review_notes": "",
            })
    return rows


def build_report(cases, reports, worksheet, retrieval_reports, plan_cache):
    summaries = {}
    for candidate, report in reports.items():
        rows = [row for row in worksheet if row["candidate"] == candidate]
        summaries[candidate] = {
            "case_count": len(rows),
            "answerable_count": sum(row["answerable"] for row in rows),
            "answered_count": sum(row["status"] == "answered" for row in rows),
            "correct_refusal_or_denial_auto": sum(
                (not row["answerable"] and row["status"] == "abstained") or
                (row["expected_outcome"] == "premise_denial" and row["status"] == "answered")
                for row in rows),
            "runtime_failure_count": sum(row["runtime_failure"] for row in rows),
            "failure_codes": sorted({row["reason_code"] for row in rows if row["reason_code"]}),
            "latency": latency_summary(
                [row["estimated_live_endpoint_ms"] for row in report["cases"]],
                [row["result"].get("status") == "error" for row in report["cases"]]),
            "downstream_latency": latency_summary(
                [row["wall_ms"] for row in report["cases"]],
                [row["result"].get("status") == "error" for row in report["cases"]]),
        }
    return {
        "schema_version": 1,
        "experiment": "embedding-final-paired-m24",
        "case_count": len(cases),
        "case_sha256": stable_hash(cases),
        "planner_cache": str(plan_cache),
        "planner_cache_sha256": stable_hash(_load(plan_cache)),
        "arms": summaries,
        "retrieval_reports": retrieval_reports,
        "answer_level": {"status": "pending_manual_worksheet", "row_count": len(worksheet)},
        "promotion": {"status": "blocked_pending_answer_review", "promote_bge": False},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--ali", type=Path, required=True)
    parser.add_argument("--bge", type=Path, required=True)
    parser.add_argument("--plan-cache", type=Path, required=True)
    parser.add_argument("--retrieval", nargs="+", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.worksheet, args.output):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    cases = _load(args.cases)["cases"]
    reports = {"ali": _load(args.ali), "bge": _load(args.bge)}
    worksheet = build_worksheet(cases, reports)
    args.worksheet.write_text(json.dumps({"schema_version": 1, "status": "pending_manual_review",
                                           "rubric": {"semantic_support": "Every expected fact and claim is supported by the cited excerpt.",
                                                       "claim_complete": "The answer covers every required fact without unsupported additions.",
                                                       "correct_abstention": "Refusal/premise denial matches expected outcome.",
                                                       "citation_integrity_auto": "Citation id, source, page/anchor and excerpt match the pinned node."},
                                           "rows": worksheet}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    retrieval = {path.stem: _load(path) for path in args.retrieval}
    result = build_report(cases, reports, worksheet, retrieval, args.plan_cache)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
