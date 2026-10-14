"""Bind completed review artifacts or prepare pending review for a new RAG run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.score_revised_m24 import json_hash, score_report


def bind_review(cases, run_cases, report, reviews):
    binding = {"source_report_sha256": json_hash(report), "label_case_sha256": json_hash(cases),
               "evaluation_case_sha256": json_hash(run_cases)}
    for key, value in binding.items():
        if key in reviews and reviews[key] != value:
            raise ValueError(f"review {key} mismatch; cannot rebind existing judgments")
    return {**deepcopy(reviews), "schema_version": 2, **binding,
            "hash_policy": "sha256 of sorted-key UTF-8 JSON with ensure_ascii=False"}


def prepare_review(cases, run_cases, report):
    by_id = {case["id"]: case for case in cases}
    rows = report["cases"]
    packet = {"reviewer": "pending", "reviewer_type": "pending", "human_review_confirmed": False,
              "status": "pending_human_confirmation", "cases": {}, "review_items": []}
    for row in rows:
        case = by_id[row["id"]]
        result = row.get("result", row)
        packet["cases"][row["id"]] = {"review_status": "pending", "facts_present": None,
                                     "claim_support": None, "refusal_correct": None}
        packet["review_items"].append({
            "id": row["id"], "question": case["question"], "expected_outcome": case["expected_outcome"],
            "refusal_category": case.get("refusal_category"), "expected_facts": case["expected_facts"],
            "fact_evidence": case.get("fact_evidence", []), "acceptable_evidence": case.get("acceptable_evidence", []),
            "result": result,
            "verification_candidates": row.get("observation", {}).get("events", {}).get("verification_input", {}).get("candidates"),
        })
    return bind_review(cases, run_cases, report, packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bind", "prepare"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run-cases", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--packet", type=Path, help="existing completed human-review packet, required for bind")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    run_cases = json.loads(args.run_cases.read_text(encoding="utf-8"))["cases"] if args.run_cases else cases
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.action == "bind":
        if not args.packet:
            parser.error("bind requires --packet for confirmed historical provenance")
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        if packet.get("status") != "completed" or packet.get("human_review_confirmed") is not True:
            parser.error("packet must already be human-confirmed")
        if Path(packet["source_report"]).resolve() != args.report.resolve():
            parser.error("packet source_report mismatch")
        review_path = Path(packet.get("review_metadata") or packet["revised_human_review"])
        reviews = json.loads(review_path.read_text(encoding="utf-8"))
        # Refusal judgments live in the completed packet, rather than the older
        # per-answer review metadata. Copy those explicitly confirmed judgments.
        for row in packet["cases"]:
            if row["expected_outcome"] == "refusal":
                reviews["cases"][row["id"]] = {
                    "facts_present": [], "claim_support": [], "refusal_correct": True,
                    "notes": row["human_review"].get("notes", ""),
                }
        payload = bind_review(cases, run_cases, report, reviews)
        payload["source_packet"] = str(args.packet)
        payload["source_review"] = str(review_path)
    else:
        payload = prepare_review(cases, run_cases, report)
    from scripts.knowledge_search import load_nodes
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    score_report(cases, report, payload, nodes, run_cases=run_cases)
    payload["source_report"] = str(args.report)
    payload["label_manifest"] = str(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Review: {args.output}; human_review_confirmed={payload['human_review_confirmed']}")


if __name__ == "__main__":
    main()
