"""Scope-aware scoring for the revised M2.4 contract.

This scorer is deliberately separate from the historical M2.4 scorer. It treats
gold locations as diagnostics, while allowing explicitly pinned equivalent
locations to support the same fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.evaluate_knowledge_answer import validate_cases
from scripts.knowledge_answer import canonical_excerpt

RUNTIME_CODES = {
    "planner_timeout", "embedding_timeout", "embedding_error", "retrieval_timeout",
    "generation_timeout", "verification_timeout", "total_timeout", "runner_error",
    "run_cancelled",
}
EXECUTION_CODES = {
    "invalid_plan", "planner_error", "retrieval_error", "generation_error",
    "invalid_generation", "verification_error", "verification_failed",
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _check_identity(case, other):
    for field in ("question", "language", "expected_outcome"):
        expected = _outcome(case) if field == "expected_outcome" else case.get(field)
        actual = _outcome(other) if field == "expected_outcome" else other.get(field)
        if expected != actual:
            raise ValueError(f"{case['id']}: {field} mismatch")


def _validate_review(case, claims, review, *, runtime=False):
    if review is None or review.get("review_status") == "pending":
        return False
    for field, count in (("facts_present", len(case["expected_facts"])),
                         ("claim_support", len(claims))):
        if field == "facts_present" and "facts_present" not in review and isinstance(review.get("facts_complete"), bool):
            continue
        values = review.get(field)
        # A failed request has no answer whose completeness could be judged.
        allowed_counts = {0, count} if runtime or not claims else {count}
        if not isinstance(values, list) or len(values) not in allowed_counts or any(type(v) is not bool for v in values):
            raise ValueError(f"{case['id']}: invalid {field}")
    if "refusal_correct" in review and type(review["refusal_correct"]) is not bool:
        raise ValueError(f"{case['id']}: invalid refusal_correct")
    return True


def _outcome(case):
    return case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal")


def _location(metadata):
    return metadata.get("id"), metadata.get("page") if metadata.get("page") is not None else metadata.get("anchor")


def _provenance(claim, nodes):
    node = nodes.get(claim.get("citation_id"))
    if node is None:
        return False
    metadata = node.metadata
    fields = {
        "source_id": "id", "source_url": "source_url", "source_version": "version",
        "sha256": "sha256", "page": "page", "anchor": "anchor",
    }
    if any(claim.get(left) != metadata.get(right) for left, right in fields.items()):
        return False
    excerpt = claim.get("excerpt")
    return isinstance(excerpt, str) and bool(excerpt) and canonical_excerpt(node.text, excerpt) is not None


def _acceptable_locations(case):
    locations = set()
    for evidence in case.get("acceptable_evidence", []):
        for location in evidence.get("locations", []):
            locations.add((location.get("source_id"), location.get("page")
                           if location.get("page") is not None else location.get("anchor")))
    return locations


def _fact_rows(case, claims, review, nodes):
    facts = case.get("expected_facts", [])
    acceptable = case.get("acceptable_evidence", [])
    if isinstance(review, dict) and "facts_present" in review:
        present = review["facts_present"]
    elif isinstance(review, dict) and isinstance(review.get("facts_complete"), bool):
        # Compatibility adapter for the old agent-review artifact. This is
        # explicitly provisional and never marks a human review as confirmed.
        present = [review["facts_complete"]] * len(facts)
    else:
        present = []
    support = review.get("claim_support", []) if isinstance(review, dict) else []
    rows = []
    claim_support = [value is True for value in support]
    for index, fact in enumerate(facts):
        evidence = acceptable[index] if index < len(acceptable) else {}
        citation_ids = set(evidence.get("citation_ids", []))
        locations = {
            (location.get("source_id"), location.get("page")
             if location.get("page") is not None else location.get("anchor"))
            for location in evidence.get("locations", [])
        }
        def supports_fact(claim):
            citation_id = claim.get("citation_id")
            if citation_id in citation_ids:
                return True
            node = nodes.get(citation_id)
            return node is not None and _location(node.metadata) in locations
        fact_claims = [claim for claim in claims if supports_fact(claim)]
        fact_claim_indexes = [i for i, claim in enumerate(claims) if supports_fact(claim)]
        rows.append({
            "fact": fact,
            "present": present[index] is True if index < len(present) else False,
            "supported": any(claim_support[i] for i in fact_claim_indexes if i < len(claim_support)),
            "claim_count": len(fact_claims),
        })
    return rows


def validate_revised_cases(cases):
    validate_cases(cases)
    for case in cases:
        outcome = _outcome(case)
        if outcome != "refusal":
            if not isinstance(case.get("question_scope"), str) or not case["question_scope"].strip():
                raise ValueError(f"question_scope required: {case['id']}")
            evidence = case.get("acceptable_evidence")
            if not isinstance(evidence, list) or len(evidence) != len(case["expected_facts"]):
                raise ValueError(f"acceptable_evidence must align with facts: {case['id']}")
            for item in evidence:
                if not item.get("fact") or not item.get("citation_ids") or not item.get("locations"):
                    raise ValueError(f"incomplete acceptable evidence: {case['id']}")
        elif case.get("acceptable_evidence"):
            raise ValueError(f"refusal cannot have evidence: {case['id']}")


def score_report(cases, report, reviews, nodes, *, run_cases=None):
    validate_revised_cases(cases)
    if not reviews.get("reviewer") or not isinstance(reviews.get("cases"), dict):
        raise ValueError("reviewer and cases required")
    rows = report.get("cases", [])
    by_id = {case["id"]: case for case in cases}
    if len(rows) != len(cases) or {row.get("id") for row in rows} != set(by_id):
        raise ValueError("report cases do not match labels")
    run_cases = cases if run_cases is None else run_cases
    validate_cases(run_cases)
    run_by_id = {case["id"]: case for case in run_cases}
    if set(run_by_id) != set(by_id):
        raise ValueError("run cases do not match labels")
    case_hash, run_hash, report_hash = json_hash(cases), json_hash(run_cases), json_hash(report)
    if report.get("case_sha256") not in (None, run_hash):
        raise ValueError("report case hash mismatch; use --run-cases for revised labels")
    for case in cases:
        _check_identity(case, run_by_id[case["id"]])
    if report.get("case_sha256") is not None:
        for row in rows:
            _check_identity(run_by_id[row["id"]], row)
    if set(reviews["cases"]) - set(by_id):
        raise ValueError("review contains unknown case IDs")
    binding = {
        "source_report_sha256": report_hash,
        "label_case_sha256": case_hash,
        "evaluation_case_sha256": run_hash,
    }
    for field, expected in binding.items():
        if field in reviews and reviews[field] != expected:
            raise ValueError(f"review {field} mismatch")
    bound = all(reviews.get(field) == expected for field, expected in binding.items())
    human = bound and reviews.get("reviewer_type") == "human" and reviews.get("human_review_confirmed") is True
    out = {
        "schema_version": 2, "case_sha256": case_hash, **binding,
        "hash_policy": "sha256 of sorted-key UTF-8 JSON with ensure_ascii=False",
        "review_provenance": reviews.get("reviewer_type", "unverified"),
        "review_bound": bound, "review_complete": True, "human_review_confirmed": human,
        "evaluation_kind": "diagnostic", "promotion_ready": False,
        "reviewer": reviews["reviewer"], "answer_total": 0,
        "answer_semantic_supported": 0, "refusal_total": 0, "refusal_correct": 0,
        "premise_denial_total": 0, "premise_denial_supported": 0,
        "exact_gold_location_match_cases": 0, "equivalent_location_match_cases": 0,
        "unsupported_claim_count": 0, "provider_runtime_failures": 0,
        "failure_counts": {}, "outcome_counts": {}, "cases": {}, "quality_gate_pass": False,
    }
    for row in rows:
        case = by_id[row["id"]]
        outcome = _outcome(case)
        out[f"{outcome}_total"] += 1
        result = row.get("result", row)
        claims = result.get("claims", [])
        review = reviews["cases"].get(row["id"])
        code = result.get("error_code") or result.get("reason_code")
        runtime = code in RUNTIME_CODES
        execution_error = runtime or result.get("status") == "error" or code in EXECUTION_CODES
        if review is not None and not isinstance(review, dict):
            raise ValueError(f"{row['id']}: review must be an object")
        reviewed = _validate_review(case, claims, review, runtime=runtime)
        out["review_complete"] &= reviewed
        provenance = bool(claims) and all(_provenance(claim, nodes) for claim in claims)
        exact = bool(claims) and all(
            _location(nodes[claim["citation_id"]].metadata) in {
                _location(nodes[cid].metadata) for cid in case.get("supporting_citation_ids", []) if cid in nodes
            } if claim.get("citation_id") in nodes else False for claim in claims
        )
        equivalent_locations = _acceptable_locations(case)
        equivalent = bool(claims) and all(
            _location(nodes[claim["citation_id"]].metadata) in equivalent_locations
            if claim.get("citation_id") in nodes else False for claim in claims
        )
        fact_rows = _fact_rows(case, claims, review if reviewed else None, nodes)
        unsupported = sum(not value for value in review.get("claim_support", [])) if reviewed else 0
        out["unsupported_claim_count"] += unsupported
        out["exact_gold_location_match_cases"] += int(exact)
        out["equivalent_location_match_cases"] += int(equivalent)
        complete = result.get("status") == "answered" and not result.get("unresolved") and bool(claims)
        facts_supported = bool(fact_rows) and all(item["present"] and item["supported"] for item in fact_rows)
        semantic = (complete and provenance and equivalent and facts_supported and unsupported == 0
                    and not execution_error) if outcome != "refusal" else False
        if outcome != "refusal" and (runtime or (claims and not reviewed)):
            semantic = None
        if runtime:
            classification = "provider_runtime_failure"
            out["provider_runtime_failures"] += 1
        elif execution_error:
            classification = "execution_failure"
        elif unsupported:
            classification = "unsupported_claim"
        elif semantic:
            classification = (
                "premise_denial_supported"
                if outcome == "premise_denial"
                else "answer_semantic_supported"
            )
        elif outcome == "refusal":
            correct = result.get("status") == "abstained" and not claims
            if reviewed and "refusal_correct" in review:
                correct &= review["refusal_correct"]
            classification = "correct_refusal" if correct else "incorrect_refusal"
        elif claims and not reviewed:
            classification = "pending_review"
        else:
            classification = "answer_incomplete"
        out["outcome_counts"][classification] = out["outcome_counts"].get(classification, 0) + 1
        if classification not in {"answer_semantic_supported", "premise_denial_supported", "correct_refusal", "pending_review"}:
            out["failure_counts"][classification] = out["failure_counts"].get(classification, 0) + 1
        out["cases"][row["id"]] = {
            "classification": classification, "semantic_supported": semantic,
            "provenance_valid": provenance, "equivalent_location_match": equivalent,
            "exact_gold_location_match": exact, "facts": fact_rows,
            "runtime_unknown": runtime,
            "review_status": "completed" if reviewed else "pending",
            "execution_failure": execution_error,
            "error_code": result.get("error_code"), "error_stage": result.get("error_stage"),
        }
        if outcome == "answer":
            out["answer_semantic_supported"] += int(semantic is True)
        elif outcome == "premise_denial":
            out["premise_denial_supported"] += int(semantic is True)
        elif outcome == "refusal":
            out["refusal_correct"] += int(classification == "correct_refusal")
    out["required_supported_answers"] = math.ceil(out["answer_total"] * 0.85)
    out["quality_gate_pass"] = (
        out["answer_total"] > 0
        and bound and out["review_complete"]
        and human
        and report.get("case_sha256") == run_hash and report.get("run_status") == "completed"
        and out["answer_semantic_supported"] >= out["required_supported_answers"]
        and out["refusal_correct"] == out["refusal_total"]
        and out["premise_denial_supported"] == out["premise_denial_total"]
        and out["provider_runtime_failures"] == 0
        and out["unsupported_claim_count"] == 0
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run-cases", type=Path, help="original labels used for this run")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve archived scoring")
    from scripts.knowledge_search import load_nodes
    cases = json.loads(args.cases.read_text())["cases"]
    run_cases = json.loads(args.run_cases.read_text())["cases"] if args.run_cases else None
    report = json.loads(args.report.read_text())
    reviews = json.loads(args.reviews.read_text())
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    index_hash = hashlib.sha256((args.index / "nodes.json").read_bytes()).hexdigest()
    if report.get("manifest", {}).get("index_nodes_sha256") not in (None, index_hash):
        parser.error("index nodes hash mismatch")
    scored = score_report(cases, report, reviews, nodes, run_cases=run_cases)
    scored["index_nodes_sha256"] = index_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(scored, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}; quality_gate_pass={scored['quality_gate_pass']}")


if __name__ == "__main__":
    main()
