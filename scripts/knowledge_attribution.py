"""Attribute observed RAG failures without issuing queries or semantic judgments."""
from __future__ import annotations

STAGES = (
    "original_dense_top50", "rewritten_dense_top50", "bm25_top20", "rrf_ranking",
    "dense_threshold", "domain_filter", "dedup", "final_top_k", "supplemented_hits",
    "expanded_evidence", "bounded_evidence",
)


def _location(node):
    meta = node.metadata
    return meta.get("id"), meta.get("page") if meta.get("page") is not None else meta.get("anchor")


def _stage_summary(hits, case, nodes):
    facts = case.get("acceptable_evidence", [])
    coverage = []
    ranks = []
    for evidence in facts:
        locations = {
            (loc.get("source_id"), loc.get("page") if loc.get("page") is not None else loc.get("anchor"))
            for loc in evidence["locations"]
        }
        matches = [hit for hit in hits if hit.get("citation_id") in nodes and (
            hit["citation_id"] in evidence["citation_ids"] or _location(nodes[hit["citation_id"]]) in locations)]
        coverage.append(bool(matches))
        ranks.extend(hit["rank"] for hit in matches if isinstance(hit.get("rank"), int))
    gold = set(case.get("supporting_citation_ids", []))
    return {"fact_location_coverage": coverage, "best_rank": min(ranks) if ranks else None,
            "exact_citation_hit": any(hit.get("citation_id") in gold for hit in hits),
            "candidate_count": len(hits)}


def attribute_case(case, row, nodes, *, reviewed_semantic=None, unsupported=False):
    result = row.get("result", row)
    observation = row.get("observation", {})
    valid = observation.get("valid", True) and not observation.get("errors")
    events = observation.get("events", {}) if valid else {}
    stages = {stage: _stage_summary(events[stage], case, nodes)
              if isinstance(events.get(stage), list) else None for stage in STAGES}
    code = result.get("error_code") or result.get("reason_code")
    verdicts = events.get("verification_result", {}).get("verdicts", [])
    diagnostics = {
        "stages": stages, "semantic_supported": reviewed_semantic,
        "stage": result.get("error_stage"), "error_code": result.get("error_code"),
        "reason_code": result.get("reason_code"), "wall_ms": row.get("wall_ms"),
        "observation_valid": bool(valid),
        "verification_verdicts": verdicts,
        "citation_validation": events.get("citation_validation"),
    }

    def output(primary, stage=None):
        return {**diagnostics, "primary": primary, "stage": stage or diagnostics["stage"]}

    if code and (code.endswith("_timeout") or code in {"embedding_error", "runner_error", "run_cancelled"}):
        return output("runtime_failure")
    if result.get("status") == "error":
        return output("execution_failure")
    if not valid:
        return output("invalid_observation")
    if unsupported:
        accepted = bool(verdicts) and all(v.get("supported") is True for v in verdicts)
        return output("verification_false_accept" if accepted else "unsupported_claim", "verification")
    if reviewed_semantic is True:
        return output("semantic_supported")
    if result.get("claims"):
        return output("pending_semantic_review" if reviewed_semantic is None else "answer_incomplete")
    if code in {"invalid_plan", "planner_error"}:
        return output("planner_failure", "planner")
    if code in {"invalid_generation", "generation_error"}:
        return output("generation_failure", "generation")
    if code in {"verification_failed", "verification_error"}:
        return output("verification_failure", "verification")
    if case.get("expected_outcome") == "refusal":
        if result.get("status") == "abstained" and code in {"outside_corpus_scope", "uncertain_intent"}:
            return output("refusal_observed")
        return output("unexpected_refusal_result")
    if events.get("planner", {}).get("action") == "abstain":
        return output("planner_abstention", "planner")
    if not stages["rrf_ranking"] and not stages["bounded_evidence"]:
        return output("unobserved")
    for before, after, primary in (
        ("rrf_ranking", "dense_threshold", "dense_threshold_failure"),
        ("dense_threshold", "domain_filter", "domain_filter_failure"),
        ("domain_filter", "dedup", "dedup_failure"),
        ("dedup", "final_top_k", "ranking_cutoff_failure"),
        ("expanded_evidence", "bounded_evidence", "evidence_budget_failure"),
    ):
        left, right = stages[before], stages[after]
        if left and right and any(a and not b for a, b in zip(left["fact_location_coverage"], right["fact_location_coverage"])):
            return output(primary, after)
    rrf = stages["rrf_ranking"]
    if rrf and not all(rrf["fact_location_coverage"]):
        return output("candidate_pool_miss", "rrf_ranking")
    final, bounded = stages["final_top_k"], stages["bounded_evidence"]
    if final and bounded and any(a and not b for a, b in zip(final["fact_location_coverage"], bounded["fact_location_coverage"])):
        return output("evidence_assembly_failure", "bounded_evidence")
    validation = events.get("citation_validation", {})
    if validation.get("rejected_claim_count", 0):
        return output("citation_validation_failure", "citation_validation")
    if verdicts and any(v.get("supported") is False for v in verdicts):
        return output("verification_rejection", "verification")
    if "generation_draft" in events:
        return output("generation_no_supported_claim", "generation")
    return output("unobserved")
