"""Score an Oracle Context generation run with explicit human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.knowledge_answer import canonical_excerpt
from scripts.knowledge_search import load_nodes
from scripts.evaluate_knowledge_answer import validate_cases


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
    gold_ids = set(case.get("supporting_citation_ids", []))
    if claim.get("citation_id") in gold_ids:
        return True
    for citation_id in gold_ids:
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


def _raw_result(row: dict) -> dict:
    result = row.get("result")
    return result if isinstance(result, dict) else row


def _failure_class(result: dict) -> str | None:
    failure = result.get("failure_class")
    code = str(result.get("error_code") or "")
    stage = result.get("error_stage")
    if failure == "provider_runtime_failure" or code.endswith("_timeout"):
        return "provider_runtime_failure"
    if stage == "verification" and code in {"verification_failed", "no_supported_claims"}:
        return "verification_failure"
    if failure == "generation_failure":
        return "generation_failure"
    if failure == "citation_validation_failure":
        return "citation_validation_failure"
    if failure in {"runner_failure", "provider_runtime_failure"}:
        return "provider_runtime_failure"
    return None


def score_oracle_report(cases: list[dict], report: dict, reviews: dict, nodes: dict) -> dict:
    """Score all answer cases and classify the first observable failure stage."""
    validate_cases(cases)
    if not reviews.get("reviewer") or not isinstance(reviews.get("cases"), dict):
        raise ValueError("human review needs a reviewer and cases mapping")
    case_by_id = {case["id"]: case for case in cases}
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != len(cases) or {row.get("id") for row in rows} != set(case_by_id):
        raise ValueError("report cases do not match the labelled answer case file")
    expected_hash = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if report.get("case_sha256") is not None and report["case_sha256"] != expected_hash:
        raise ValueError("report case hash does not match the labelled case file")
    if set(reviews["cases"]) != set(case_by_id):
        raise ValueError("human review must cover every answer case exactly once")

    result = {
        "case_sha256": expected_hash,
        "reviewer": reviews["reviewer"],
        "answer_total": len(cases),
        "semantic_supported": 0,
        "facts_complete": 0,
        "citation_valid": 0,
        "verification_supported": 0,
        "generation_failures": 0,
        "citation_validation_failures": 0,
        "verification_failures": 0,
        "provider_runtime_failures": 0,
        "incomplete_answers": 0,
        "unsupported_claim_count": 0,
        "promotion_ready": False,
        "cases": {},
    }
    failure_keys = {
        "generation_failure": "generation_failures",
        "citation_validation_failure": "citation_validation_failures",
        "verification_failure": "verification_failures",
        "provider_runtime_failure": "provider_runtime_failures",
    }
    for row in rows:
        case = case_by_id[row["id"]]
        review = reviews["cases"][row["id"]]
        if not isinstance(review, dict) or not isinstance(review.get("facts_complete"), bool):
            raise ValueError(f"facts_complete review missing for {row['id']}")
        raw = _raw_result(row)
        claims = raw.get("claims", [])
        support = review.get("claim_support")
        if not isinstance(support, list) or len(support) != len(claims) or any(not isinstance(v, bool) for v in support):
            raise ValueError(f"claim_support review missing for {row['id']}")
        result["facts_complete"] += int(review["facts_complete"])
        result["unsupported_claim_count"] += sum(not value for value in support)
        metadata_valid = bool(claims) and all(_metadata_valid(claim, case, nodes) for claim in claims)
        citation_valid = bool(claims) and metadata_valid and raw.get("failure_class") not in {
            "citation_validation_failure"
        }
        verified = bool(claims) and raw.get("status") in {"answered", "partial"}
        failure = _failure_class(raw)
        if failure:
            result[failure_keys[failure]] += 1
        complete = bool(claims) and raw.get("status") == "answered" and not raw.get("unresolved")
        result["citation_valid"] += int(citation_valid)
        result["verification_supported"] += int(verified)
        semantic = complete and metadata_valid and review["facts_complete"] and all(support)
        result["semantic_supported"] += int(semantic)
        incomplete = not semantic and not failure
        if incomplete:
            result["incomplete_answers"] += 1
        result["cases"][row["id"]] = {
            "classification": failure or ("answer_semantic_supported" if semantic else "answer_incomplete"),
            "facts_complete": review["facts_complete"],
            "citation_valid": citation_valid,
            "verification_supported": verified,
            "metadata_valid": metadata_valid,
            "semantic_supported": semantic,
            "unsupported_claim_count": sum(not value for value in support),
            "status": raw.get("status"),
            "error_code": raw.get("error_code"),
            "error_stage": raw.get("error_stage"),
        }
    result["promotion_ready"] = (
        result["answer_total"] == 20 and result["semantic_supported"] >= 17
        and result["unsupported_claim_count"] == 0
        and result["provider_runtime_failures"] <= 2
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [case for case in json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
             if case.get("expected_outcome") == "answer"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    scored = score_oracle_report(cases, report, reviews, nodes)
    args.output.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}; promotion_ready={scored['promotion_ready']}")


if __name__ == "__main__":
    main()
