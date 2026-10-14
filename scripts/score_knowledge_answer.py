"""Score a completed knowledge-answer run with explicit human claim review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.evaluate_knowledge_answer import validate_cases
from scripts.knowledge_answer import canonical_excerpt
from scripts.knowledge_search import load_nodes


def _outcome(case: dict) -> str:
    return case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal")


def _metadata_valid(claim: dict, case: dict, nodes: dict) -> bool:
    node = nodes.get(claim.get("citation_id"))
    if node is None:
        return False
    meta = node.metadata
    fields = {"source_id": "id", "source_url": "source_url", "source_version": "version",
              "sha256": "sha256", "page": "page", "anchor": "anchor"}
    if any(claim.get(output) != meta.get(source) for output, source in fields.items()):
        return False
    excerpt = claim.get("excerpt")
    if not isinstance(excerpt, str) or not excerpt or canonical_excerpt(node.text, excerpt) is None:
        return False
    for citation_id in case["supporting_citation_ids"]:
        gold = nodes.get(citation_id)
        if gold is None:
            continue
        source = gold.metadata
        if (meta.get("id"), meta.get("version"), meta.get("sha256")) != (
                source.get("id"), source.get("version"), source.get("sha256")):
            continue
        if meta.get("page") is not None and meta.get("page") == source.get("page"):
            return True
        if meta.get("anchor") is not None and meta.get("anchor") == source.get("anchor"):
            return True
    return False


def _execution_failure(row: dict) -> bool:
    """Return whether a response did not complete its answer contract."""
    return bool(row.get("error_code") or row.get("failure_class") or row.get("status") == "error")


def _runtime_failure(row: dict) -> bool:
    """Return whether a response failed in provider/runtime execution."""
    failure_class = row.get("failure_class")
    error_code = str(row.get("error_code") or "")
    if failure_class == "provider_runtime_failure":
        return True
    if error_code.endswith("_timeout"):
        return True
    return row.get("status") == "error" and bool(row.get("error_stage"))


def score_report(cases: list[dict], report: dict, reviews: dict, nodes: dict) -> dict:
    """Combine mechanical provenance checks with per-claim human judgments."""
    validate_cases(cases)
    if not reviews.get("reviewer") or not isinstance(reviews.get("cases"), dict):
        raise ValueError("human review needs a reviewer and cases mapping")
    case_by_id = {case["id"]: case for case in cases}
    rows = report["cases"]
    if len(rows) != len(cases) or {row["id"] for row in rows} != set(case_by_id):
        raise ValueError("report cases do not match the labelled case file")
    expected_hash = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if report.get("case_sha256") is not None and report["case_sha256"] != expected_hash:
        raise ValueError("report case hash does not match the labelled case file")
    result = {"case_sha256": expected_hash, "reviewer": reviews["reviewer"],
              "answer_total": 0, "answer_semantic_supported": 0, "answer_exact_citation": 0,
              "refusal_total": 0, "refusal_correct": 0,
              "premise_denial_total": 0, "premise_denial_supported": 0,
              "error_count": 0, "execution_failure_count": 0,
              "unsupported_claim_count": 0, "metadata_invalid_count": 0,
              "cases": {}}
    for row in rows:
        case = case_by_id[row["id"]]
        outcome = _outcome(case)
        result[f"{outcome}_total"] += 1
        claims = row.get("claims", [])
        exact = bool(claims) and set(case["supporting_citation_ids"]) <= {
            claim.get("citation_id") for claim in claims}
        metadata_valid = all(_metadata_valid(claim, case, nodes) for claim in claims)
        execution_failure = _execution_failure(row)
        runtime_failure = _runtime_failure(row)
        if execution_failure:
            result["execution_failure_count"] += 1
        if runtime_failure:
            result["error_count"] += 1
        review = reviews["cases"].get(row["id"])
        if claims and outcome in {"answer", "premise_denial"}:
            if not isinstance(review, dict) or not isinstance(review.get("facts_complete"), bool):
                raise ValueError(f"facts_complete review missing for {row['id']}")
            support = review.get("claim_support")
            if not isinstance(support, list) or len(support) != len(claims) or any(
                    not isinstance(value, bool) for value in support):
                raise ValueError(f"claim_support review missing for {row['id']}")
            result["unsupported_claim_count"] += sum(not value for value in support)
            if not metadata_valid:
                result["metadata_invalid_count"] += 1
        elif review is not None:
            raise ValueError(f"unexpected review for {row['id']}")
        complete = bool(claims) and row.get("status") == "answered" and not row.get("unresolved")
        semantic_supported = (complete and metadata_valid and bool(review and review["facts_complete"])
                              and all(review["claim_support"]) if outcome != "refusal" else False)
        if outcome == "answer":
            result["answer_semantic_supported"] += int(semantic_supported)
            result["answer_exact_citation"] += int(semantic_supported and exact)
        elif outcome == "premise_denial":
            result["premise_denial_supported"] += int(semantic_supported)
        else:
            result["refusal_correct"] += int(
                not execution_failure and row.get("status") == "abstained" and not claims
            )
            result["unsupported_claim_count"] += len(claims)
        result["cases"][row["id"]] = {
            "expected_outcome": outcome,
            "exact_citation": exact,
            "metadata_valid": metadata_valid,
            "semantic_supported": semantic_supported,
            "execution_failure": execution_failure,
            "runtime_failure": runtime_failure,
            "grade": row.get("grade"),
            "error_code": row.get("error_code"),
            "failure_class": row.get("failure_class"),
            "error_stage": row.get("error_stage"),
            "trace_id": row.get("trace_id"),
        }
    result["promotion_ready"] = (result["answer_total"] == 20 and result["answer_semantic_supported"] >= 17
        and result["refusal_total"] + result["premise_denial_total"] == 10
        and result["refusal_correct"] == result["refusal_total"]
        and result["premise_denial_supported"] == result["premise_denial_total"]
        and result["error_count"] <= 2 and result["unsupported_claim_count"] == 0
        and result["metadata_invalid_count"] == 0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    result = score_report(cases, report, reviews, nodes)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}; promotion_ready={result['promotion_ready']}")


if __name__ == "__main__":
    main()
