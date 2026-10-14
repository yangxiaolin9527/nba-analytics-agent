"""Paired answer-level evaluation for offline retrieval ranking candidates."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts.evaluate_knowledge_answer import grade_response, validate_cases
from scripts.knowledge_answer import (
    bounded_evidence,
    expand_evidence,
    supplement_hits,
)
from scripts.knowledge_search import KnowledgeSearch, load_nodes
from scripts.knowledge_vector import APIEmbedding, DenseIndex
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import DEFAULT_ENV_PATH, load_provider_config
from scripts.retrieval_lab import RetrievalLabError, diagnose_retrieval, oracle_context_answer


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def _domain_ok(item: dict, intent: str) -> bool:
    if intent == "rules":
        return item.get("page") is not None
    if intent == "glossary":
        return item.get("anchor") is not None
    return True


def _location(item: dict) -> tuple[str | None, object]:
    return item.get("source_id"), item.get("page") if item.get("page") is not None else item.get("anchor")


def _dedup(items: list[dict], intent: str) -> list[dict]:
    seen = set()
    selected = []
    for item in items:
        if not _domain_ok(item, intent):
            continue
        location = _location(item)
        if location in seen:
            continue
        seen.add(location)
        selected.append(item)
    return selected


def _lexical_rerank(items: list[dict], query: str, nodes: dict) -> list[dict]:
    query_tokens = _tokens(query)
    scored = []
    for base_rank, item in enumerate(items, 1):
        node = nodes[item["citation_id"]]
        text = node.text if hasattr(node, "text") else node.get("text", "")
        overlap = len(query_tokens & _tokens(text)) / max(len(query_tokens), 1)
        scored.append((overlap, float(item.get("score", 0.0)), -base_rank, item))
    return [item for _, _, _, item in sorted(scored, key=lambda row: (-row[0], -row[1], -row[2]))]


def _node_hit(item: dict, nodes: dict) -> dict:
    node = nodes[item["citation_id"]]
    meta = node.metadata if hasattr(node, "metadata") else node
    return {
        "citation_id": meta["citation_id"],
        "excerpt": node.text if hasattr(node, "text") else meta.get("text", ""),
        "source_id": meta["id"], "source_url": meta["source_url"],
        "source_version": meta["version"], "sha256": meta["sha256"],
        "page": meta.get("page"), "heading": meta.get("heading"), "anchor": meta.get("anchor"),
        "score": item.get("score", 0.0),
    }


def select_variant_hits(diagnostic: dict, nodes: dict, variant: str, *, top_k: int = 5) -> list[dict]:
    """Select baseline or candidate hits from one immutable retrieval trace."""
    if variant == "baseline":
        items = list(diagnostic.get("stages", {}).get("final_top_k", []))[:top_k]
    elif variant == "candidate":
        planner = diagnostic.get("planner", {})
        intent = planner.get("intent", "rules")
        dense_ids = {
            item["citation_id"] for item in diagnostic.get("stages", {}).get("dense_threshold", [])
        }
        ranked = [item for item in diagnostic.get("stages", {}).get("rrf_ranking", [])
                  if item["citation_id"] in dense_ids]
        ranked = _dedup(ranked, intent)[:10]
        query = planner.get("english_query") or diagnostic.get("question", "")
        items = _dedup(_lexical_rerank(ranked, query, nodes), intent)[:top_k]
    else:
        raise ValueError(f"unknown ranking variant: {variant}")
    return [_node_hit(item, nodes) for item in items]


def _variant_evidence(search, question: str, english_query: str, hits: list[dict]) -> list[dict]:
    supplemented = supplement_hits(search, hits, english_query)
    return bounded_evidence(question, expand_evidence(search, supplemented))


async def evaluate_pair(answerer, search, question: str, diagnostic: dict) -> dict:
    """Run generation and verification twice while sharing one retrieval trace."""
    nodes = search.nodes
    english_query = diagnostic.get("planner", {}).get("english_query") or question
    output = {}
    for variant in ("baseline", "candidate"):
        hits = select_variant_hits(diagnostic, nodes, variant)
        evidence = _variant_evidence(search, question, english_query, hits)
        if not evidence:
            result = {"status": "abstained", "answer": None, "claims": [],
                      "unresolved": [question], "reason_code": "no_evidence"}
        else:
            result = await oracle_context_answer(answerer, question, evidence)
        result["evidence_citation_ids"] = [item["citation_id"] for item in evidence]
        result["ranked_citation_ids"] = [item["citation_id"] for item in hits]
        result["variant"] = variant
        output[variant] = result
    return output


def _error_variant(error: Exception, question: str) -> dict:
    return {"status": "error", "answer": None, "claims": [], "unresolved": [question],
            "grade": "error", "error_type": type(error).__name__,
            "error_code": getattr(error, "reason_code", "retrieval_error"),
            "error_stage": getattr(error, "stage", "retrieval"),
            "failure_class": getattr(error, "failure_class", "provider_runtime_failure"),
            "original_type": getattr(error, "original_type", type(error).__name__),
            "original_message": str(getattr(error, "original", error))[:300]}


async def evaluate_cases(answerer, search, guided, cases: list[dict], *, inter_case_delay: float = 2.0) -> dict:
    validate_cases(cases, search)
    rows = []
    for index, case in enumerate(cases):
        started = perf_counter()
        try:
            diagnostic = diagnose_retrieval(guided, case["question"], top_k=5)
            pair = await evaluate_pair(answerer, search, case["question"], diagnostic)
            for variant in pair.values():
                variant["grade"] = grade_response(case, variant)
        except Exception as exc:
            diagnostic = {"question": case["question"], "stages": {}, "errors": []}
            pair = {variant: _error_variant(exc, case["question"])
                    for variant in ("baseline", "candidate")}
        row = {"id": case["id"], "question": case["question"], "language": case["language"],
               "expected_outcome": case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal"),
               "expected_facts": case["expected_facts"],
               "expected_citation_ids": case["supporting_citation_ids"],
               "retrieval": diagnostic, **pair,
               "wall_ms": round((perf_counter() - started) * 1000, 3)}
        rows.append(row)
        print(f"{case['id']} baseline={pair['baseline'].get('grade')} candidate={pair['candidate'].get('grade')}", flush=True)
        if index < len(cases) - 1 and inter_case_delay:
            await asyncio.sleep(inter_case_delay)
    return {
        "evaluation_mode": "paired_answer_level",
        "case_count": len(rows),
        "case_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "concurrency": 1,
        "inter_case_delay_seconds": inter_case_delay,
        "variant_order": ["baseline", "candidate"],
        "cases": rows,
    }


def _variant_report(report: dict, variant: str) -> dict:
    return {"cases": [{"id": row["id"], "status": row[variant].get("status"),
                         "claims": row[variant].get("claims", []),
                         "unresolved": row[variant].get("unresolved", []),
                         "grade": row[variant].get("grade", "error"),
                         "error_code": row[variant].get("error_code"),
                         "failure_class": row[variant].get("failure_class"),
                         "error_stage": row[variant].get("error_stage"),
                         "error_type": row[variant].get("error_type")}
                        for row in report["cases"]],
            "case_sha256": report.get("case_sha256")}


def _reviews_for_variant(reviews: dict, variant: str) -> dict:
    return {
        "reviewer": reviews.get("reviewer", ""),
        "cases": {
            case_id: value[variant]
            for case_id, value in reviews.get("cases", {}).items()
            if isinstance(value, dict) and variant in value
        },
    }


def score_pair_report(cases: list[dict], report: dict, reviews: dict, nodes: dict) -> dict:
    """Score both variants with the existing semantic contract and compare them."""
    from scripts.score_knowledge_answer import score_report

    baseline = score_report(cases, _variant_report(report, "baseline"),
                            _reviews_for_variant(reviews, "baseline"), nodes)
    candidate = score_report(cases, _variant_report(report, "candidate"),
                            _reviews_for_variant(reviews, "candidate"), nodes)
    non_regression = all(candidate[key] >= baseline[key] for key in (
        "answer_semantic_supported", "answer_exact_citation", "refusal_correct",
        "premise_denial_supported")) and candidate["unsupported_claim_count"] <= baseline["unsupported_claim_count"]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_non_regression": non_regression,
        "promotion_ready": bool(candidate["promotion_ready"] and non_regression),
        "transitions": {
            row["id"]: {"baseline": row["baseline"].get("grade"), "candidate": row["candidate"].get("grade")}
            for row in report["cases"]
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-timeout", type=float, default=11.0)
    parser.add_argument("--planner-timeout", type=float, default=10.0)
    parser.add_argument("--embedding-timeout", type=float, default=12.0)
    parser.add_argument("--generation-timeout", type=float, default=10.0)
    parser.add_argument("--verification-timeout", type=float, default=10.0)
    parser.add_argument("--inter-case-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_PATH)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.ids:
        requested = set(args.ids)
        cases = [case for case in cases if case["id"] in requested]
        if len(cases) != len(requested):
            parser.error("unknown or duplicate case ID")
    if args.limit is not None:
        cases = cases[:args.limit]
    provider_config = load_provider_config(args.config)
    embedding = APIEmbedding(provider_config)
    dense = DenseIndex.open(args.index, embedding)
    payload = json.loads((args.index / "nodes.json").read_text(encoding="utf-8"))
    search = KnowledgeSearch(load_nodes(args.index), dense=dense, manifest_hash=payload["manifest_hash"])
    model = OpenAICompatibleModel(config=provider_config, timeout=args.model_timeout)
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    guided = GuidedKnowledgeSearch(search, model, planner_timeout=args.planner_timeout,
                                   embedding_timeout=args.embedding_timeout)
    from scripts.knowledge_answer import KnowledgeAnswer
    answerer = KnowledgeAnswer(search, model, guided=guided,
                               generation_timeout=args.generation_timeout,
                               verification_timeout=args.verification_timeout,
                               total_timeout=30.0)
    try:
        report = asyncio.run(evaluate_cases(answerer, search, guided, cases,
                                            inter_case_delay=args.inter_case_delay))
        report.update({"planner_model": model.model, "embedding_model": embedding.model,
                       "index_manifest_hash": payload["manifest_hash"],
                       "timeout_budgets_seconds": {
                           "model_request": args.model_timeout, "planner": args.planner_timeout,
                           "embedding": args.embedding_timeout, "generation": args.generation_timeout,
                           "verification": args.verification_timeout, "inter_case_delay": args.inter_case_delay,
                       }})
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    finally:
        dense.close()


if __name__ == "__main__":
    main()
