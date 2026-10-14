"""Summarize final paired reports and apply the embedding promotion gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.embedding_final_analysis import (
    latency_summary, mcnemar_exact, paired_bootstrap, promotion_decision, stable_hash,
    summarize_answer_reviews,
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--ali-report", type=Path, required=True)
    parser.add_argument("--bge-report", type=Path, required=True)
    parser.add_argument("--ali-retrieval", type=Path)
    parser.add_argument("--bge-retrieval", type=Path)
    parser.add_argument("--readiness", type=Path, help="explicit local readiness evidence; unknown when omitted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.promotion_output.exists():
        parser.error("refusing to overwrite final analysis artifacts")
    summary = load(args.summary)
    worksheet = load(args.worksheet)
    rows = worksheet["rows"]
    by_candidate = {candidate: {row["case_id"]: row for row in rows if row["candidate"] == candidate}
                    for candidate in ("ali", "bge")}
    ali_ids = set(by_candidate["ali"])
    bge_ids = set(by_candidate["bge"])
    if ali_ids != bge_ids:
        raise ValueError("worksheet candidates are not paired")
    if len(rows) != len(ali_ids) * 2:
        raise ValueError("worksheet contains duplicate candidate/case rows")
    try:
        answer_summary = summarize_answer_reviews(rows)
    except ValueError:
        answer_summary = None
    labels_ready = answer_summary is not None
    ali_report = load(args.ali_report)
    bge_report = load(args.bge_report)
    ali_rows = {row["id"]: row for row in ali_report["cases"]}
    bge_rows = {row["id"]: row for row in bge_report["cases"]}
    if ali_ids != set(ali_rows) or ali_ids != set(bge_rows):
        raise ValueError("runtime reports and worksheet are not paired")
    ids = sorted(ali_ids)
    ali_endpoint = [ali_rows[i]["estimated_live_endpoint_ms"] for i in ids]
    bge_endpoint = [bge_rows[i]["estimated_live_endpoint_ms"] for i in ids]
    def is_timeout(row):
        code = row["result"].get("reason_code") or row["result"].get("error_code") or ""
        return "timeout" in code
    latency = {
        "ali": latency_summary(ali_endpoint, [is_timeout(ali_rows[i]) for i in ids]),
        "bge": latency_summary(bge_endpoint, [is_timeout(bge_rows[i]) for i in ids]),
        "paired_difference_ali_minus_bge": paired_bootstrap(ali_endpoint, bge_endpoint),
    }
    bge_failure = [bge_rows[i]["result"].get("status") != "error" for i in ids]
    ali_failure = [ali_rows[i]["result"].get("status") != "error" for i in ids]
    auto = {
        "successful_execution_mcnemar": mcnemar_exact(ali_failure, bge_failure),
        "status_pairs": {},
    }
    for i in ids:
        pair = (ali_rows[i]["result"].get("status"), bge_rows[i]["result"].get("status"))
        auto["status_pairs"]["|".join(str(x) for x in pair)] = auto["status_pairs"].get("|".join(str(x) for x in pair), 0) + 1
    quality = None
    if labels_ready:
        q = {key: answer_summary["bge"][key] - answer_summary["ali"][key]
             for key in ("semantic_support_rate", "claim_completeness_rate", "citation_integrity_rate",
                         "correct_abstention_rate", "runtime_failure_rate")}
        false_acceptance = [answer_summary[name]["false_acceptance_rate"] for name in ("ali", "bge")]
        quality = {"semantic_support_difference": q["semantic_support_rate"],
                   "completeness_difference": q["claim_completeness_rate"],
                   "citation_integrity_difference": q["citation_integrity_rate"],
                   "correct_abstention_difference": q["correct_abstention_rate"],
                   "false_acceptance_difference": (false_acceptance[1] - false_acceptance[0]
                                                   if all(x is not None for x in false_acceptance) else None),
                   "runtime_failure_difference": q["runtime_failure_rate"]}
    readiness = load(args.readiness) if args.readiness else {"passed": None, "details": "not supplied"}
    latency_gate = {
        "p50_relative_improvement": (latency["ali"]["p50_ms"] - latency["bge"]["p50_ms"]) / latency["ali"]["p50_ms"],
        "p95_relative_change": (latency["bge"]["p95_ms"] - latency["ali"]["p95_ms"]) / latency["ali"]["p95_ms"],
        "timeout_rate_difference": latency["bge"]["timeout_rate"] - latency["ali"]["timeout_rate"],
    }
    gate = promotion_decision({"quality": quality, "latency": latency_gate, "readiness": readiness})
    gate["owner_review_confirmed"] = worksheet.get("status") == "owner_review_confirmed"
    gate["experiment_frozen"] = gate["owner_review_confirmed"] and not gate["promote_bge"]
    gate["known_runtime_failure_rate_difference"] = (
        sum(not x for x in bge_failure) - sum(not x for x in ali_failure)) / len(ids)
    retrieval = None
    if bool(args.ali_retrieval) != bool(args.bge_retrieval):
        raise ValueError("both paired retrieval reports must be supplied")
    if args.ali_retrieval:
        retrieval_reports = {name: load(path) for name, path in
                             (("ali", args.ali_retrieval), ("bge", args.bge_retrieval))}
        retrieval_rows = {name: {row["id"]: row for row in document["cases"]}
                          for name, document in retrieval_reports.items()}
        if any(set(candidate_rows) != set(ids) for candidate_rows in retrieval_rows.values()):
            raise ValueError("retrieval and endpoint reports are not paired")
        retrieval = {"source_hashes": {name: stable_hash(document)
                                      for name, document in retrieval_reports.items()}, "metrics": {}}
        for key in ("location_recall_at_5", "location_recall_at_10", "mrr_at_5", "ndcg_at_5", "support_recall_at_5"):
            scored = [i for i in ids if all(retrieval_rows[name][i][key] is not None for name in ("ali", "bge"))]
            values = {name: [retrieval_rows[name][i][key] for i in scored] for name in ("ali", "bge")}
            a, b = (sum(values[name]) / len(scored) for name in ("ali", "bge"))
            retrieval["metrics"][key] = {"count": len(scored), "ali": a, "bge": b,
                "difference_bge_minus_ali": b - a, "relative_difference": (b - a) / a if a else None,
                "paired_mean_ali_minus_bge": paired_bootstrap(values["ali"], values["bge"])}
    report = {"schema_version": 1, "experiment": "embedding-final-paired-m24",
              "source_summary_sha256": stable_hash(summary), "answer_review_ready": gate["owner_review_confirmed"] or labels_ready,
              "numeric_answer_labels_ready": labels_ready, "retrieval_paired": retrieval,
              "worksheet_status": worksheet.get("status"), "paired_case_ids": ids,
              "source_hashes": {"ali_report": stable_hash(ali_report), "bge_report": stable_hash(bge_report),
                                "worksheet": stable_hash(worksheet)},
              "latency_gate": latency_gate, "readiness": readiness,
              "auto_execution": auto, "latency": latency, "answer_level": answer_summary,
              "quality_differences": quality, "promotion": gate}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.promotion_output.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
