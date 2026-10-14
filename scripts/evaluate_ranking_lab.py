"""Offline fusion and ranking experiments over a frozen retrieval-lab report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter


STAGES = {
    "original_dense": "original_dense_top50",
    "rewritten_dense": "rewritten_dense_top50",
    "bm25": "bm25_top20",
    "rrf": "rrf_ranking",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def _domain_ok(item: dict, intent: str) -> bool:
    if intent == "rules":
        return item.get("page") is not None
    if intent == "glossary":
        return item.get("anchor") is not None
    return True


def _dedup(items: list[dict], intent: str) -> list[dict]:
    seen = set()
    output = []
    for item in items:
        if not _domain_ok(item, intent):
            continue
        location = (item.get("source_id"), item.get("page")
                    if item.get("page") is not None else item.get("anchor"))
        if location in seen:
            continue
        seen.add(location)
        output.append(item)
    return output


def _rerank(items: list[dict], query: str, nodes: dict[str, dict]) -> list[dict]:
    query_tokens = _tokens(query)
    scored = []
    for base_rank, item in enumerate(items, 1):
        text = nodes.get(item["citation_id"], {}).get("text", "")
        overlap = len(query_tokens & _tokens(text)) / max(len(query_tokens), 1)
        scored.append((overlap, float(item.get("score", 0.0)), -base_rank, item))
    return [item for _, _, _, item in sorted(scored, key=lambda row: (-row[0], -row[1], -row[2]))]


def _gold_rank(items: list[dict], gold: set[str]) -> int | None:
    for rank, item in enumerate(items, 1):
        if item["citation_id"] in gold:
            return rank
    return None


def _metrics(ranks: list[int | None], *, cutoff: int = 5) -> dict:
    if not ranks:
        return {"cases": 0, "recall": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "gold_ranks": [],
                "recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0}
    hits = [rank for rank in ranks if rank is not None]
    recall = len(hits) / len(ranks) if ranks else 0.0
    reciprocal = [1 / rank if rank is not None else 0.0 for rank in ranks]
    ndcg = []
    for rank in ranks:
        ndcg.append(1 / __import__("math").log2(rank + 1) if rank is not None and rank <= cutoff else 0.0)
    return {"cases": len(ranks), "recall": round(recall, 4), "mrr": round(sum(reciprocal) / len(ranks), 4),
            "ndcg_at_5": round(sum(ndcg) / len(ranks), 4), "gold_ranks": ranks,
            "recall_at_1": round(sum(rank == 1 for rank in ranks) / len(ranks), 4),
            "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / len(ranks), 4),
            "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks), 4),
            "recall_at_10": round(sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks), 4)}


def run(report: dict, nodes: dict[str, dict], *, candidate_sizes=(5, 10, 20, 50)) -> dict:
    answer_cases = [case for case in report["cases"]
                    if case.get("expected_outcome") == "answer"]
    experiments = {}
    for fusion_name, stage_name in STAGES.items():
        for candidate_size in candidate_sizes:
            for reranker in ("none", "lexical_overlap"):
                key = f"{fusion_name}_top{candidate_size}_{reranker}"
                ranks = []
                elapsed = []
                for case in answer_cases:
                    started = perf_counter()
                    intent = case.get("planner", {}).get("intent", "rules")
                    candidates = _dedup(case.get("stages", {}).get(stage_name, [])[:candidate_size], intent)
                    if reranker == "lexical_overlap":
                        query = case.get("planner", {}).get("english_query") or case["question"]
                        candidates = _rerank(candidates, query, nodes)
                    ranked = _dedup(candidates, intent)
                    ranks.append(_gold_rank(ranked, set(case["gold_citation_ids"])))
                    elapsed.append((perf_counter() - started) * 1000)
                metric = _metrics(ranks)
                metric.update({"reranker": reranker, "fusion": fusion_name,
                               "candidate_top_n": candidate_size,
                               "mean_reranker_ms": round(sum(elapsed) / len(elapsed), 4) if elapsed else 0.0,
                               "external_model_calls": 0, "estimated_external_cost_usd": 0.0})
                experiments[key] = metric
    return {"version": 1, "runner": "m2.2-ranking-lab", "answer_case_count": len(answer_cases),
            "source_retrieval_report": report.get("runner"), "experiments": experiments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True,
                        help="knowledge index nodes.json used for lexical reranking")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    payload = json.loads(args.nodes.read_text(encoding="utf-8"))
    nodes = {item["metadata"]["citation_id"]: {"text": item["text"]}
             for item in payload["nodes"]}
    output = run(report, nodes)
    output["index_manifest_hash"] = payload["manifest_hash"]
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
