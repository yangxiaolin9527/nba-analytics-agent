"""Evaluate lexical and Cross-Encoder reranking on the current guided trace.

This is an offline experiment. It captures the production guided retriever's
location-aware candidate list, then reranks immutable candidate prefixes. It
does not change the online answer path or the configured Dense index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from scripts.knowledge_guidance import GuidedKnowledgeSearch
from scripts.knowledge_observation import RequestObservation
from scripts.knowledge_search import KnowledgeSearch, load_nodes
from scripts.knowledge_vector import APIEmbedding, DenseIndex
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import DEFAULT_ENV_PATH, load_provider_config


ROOT = Path(__file__).resolve().parents[1]


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def location(item: dict) -> tuple[str | None, object]:
    return item.get("source_id"), item.get("page") if item.get("page") is not None else item.get("anchor")


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokens(query)
    return len(query_tokens & tokens(text)) / max(len(query_tokens), 1)


def rank_items(items: list[dict], query: str, nodes: dict, method: str, cross_encoder=None) -> tuple[list[dict], float]:
    if method == "none":
        return list(items), 0.0
    started = perf_counter()
    if method == "lexical_overlap":
        scored = [(lexical_score(query, nodes[item["citation_id"]].text),
                   float(item.get("score", 0.0)), -index, item)
                  for index, item in enumerate(items)]
    elif method == "cross_encoder":
        pairs = [(query, nodes[item["citation_id"]].text) for item in items]
        values = cross_encoder.predict(pairs, batch_size=8, show_progress_bar=False)
        scored = [(float(value), float(item.get("score", 0.0)), -index, item)
                  for index, (value, item) in enumerate(zip(values, items))]
    else:
        raise ValueError(f"unknown reranker: {method}")
    ranked = [item for _, _, _, item in sorted(scored, key=lambda row: (-row[0], -row[1], -row[2]))]
    return ranked, (perf_counter() - started) * 1000


def gold_rank(items: list[dict], gold: set[str]) -> int | None:
    for rank, item in enumerate(items, 1):
        if item["citation_id"] in gold:
            return rank
    return None


def metric(ranks: list[int | None], *, final_k: int) -> dict:
    count = len(ranks)
    hits = [rank for rank in ranks if rank is not None]
    return {
        "cases": count,
        "recall": round(len(hits) / count, 4) if count else 0.0,
        "recall_at_final_k": round(sum(rank is not None and rank <= final_k for rank in ranks) / count, 4) if count else 0.0,
        "mrr": round(sum(1 / rank if rank is not None else 0.0 for rank in ranks) / count, 4) if count else 0.0,
        "ndcg_at_final_k": round(sum(1 / __import__("math").log2(rank + 1)
                                     if rank is not None and rank <= final_k else 0.0
                                     for rank in ranks) / count, 4) if count else 0.0,
        "gold_ranks": ranks,
    }


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case.get("expected_outcome") == "answer"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "metadata/knowledge_ranking_holdout_m22rlb_v1.json")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--candidate-depths", default="5,10,20,30,50")
    parser.add_argument("--final-k", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite report")
    depths = [int(value) for value in args.candidate_depths.split(",") if value.strip()]
    if any(depth <= 0 for depth in depths) or args.final_k <= 0:
        raise ValueError("candidate depths and final-k must be positive")

    config = load_provider_config(args.config)
    config.require_model(); config.require_embedding()
    payload = json.loads((args.index / "nodes.json").read_text(encoding="utf-8"))
    embedding = APIEmbedding(config)
    dense = DenseIndex.open(args.index, embedding)
    search = KnowledgeSearch(load_nodes(args.index), dense=dense, manifest_hash=payload.get("manifest_hash"))
    model = OpenAICompatibleModel(config=config)
    cases = load_cases(args.cases)
    cross_encoder = None
    try:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder(args.model, device=args.device, max_length=args.max_length)
        all_traces = []
        rows = []
        methods = ("none", "lexical_overlap", "cross_encoder")
        for case in cases:
            observation = RequestObservation()
            guided = GuidedKnowledgeSearch(search, model,
                                           min_dense_score=config.knowledge_min_dense_score,
                                           planner_timeout=config.knowledge_planner_timeout_seconds,
                                           embedding_timeout=config.knowledge_embedding_timeout_seconds,
                                           observer=observation)
            started = perf_counter()
            result = guided.search(case["question"], top_k=max(depths))
            observation.close()
            snapshot = observation.snapshot()
            dedup = snapshot["events"].get("dedup", [])
            planner = snapshot["events"].get("planner", {})
            nodes = search.nodes
            gold = set(case["supporting_citation_ids"])
            candidates = [item for item in dedup if item["citation_id"] in nodes]
            trace = {"id": case["id"], "question": case["question"],
                     "gold_citation_ids": sorted(gold), "planner": planner,
                     "retrieval_status": result.get("retrieval_status"),
                     "candidate_locations": candidates,
                     "retrieval_elapsed_ms": round((perf_counter() - started) * 1000, 3)}
            all_traces.append(trace)
            for depth in depths:
                prefix = candidates[:depth]
                candidate_recall = bool(gold & {item["citation_id"] for item in prefix})
                for method in methods:
                    query = planner.get("english_query") or case["question"]
                    ranked, rerank_ms = rank_items(prefix, query, nodes, method, cross_encoder)
                    rows.append({"case_id": case["id"], "depth": depth, "method": method,
                                 "candidate_recall": candidate_recall,
                                 "gold_rank": gold_rank(ranked, gold),
                                 "rerank_ms": round(rerank_ms, 4)})
            print(case["id"], result.get("retrieval_status"), len(candidates), flush=True)
        summary = {}
        for depth in depths:
            for method in methods:
                selected = [row for row in rows if row["depth"] == depth and row["method"] == method]
                summary[f"{method}_top{depth}"] = {
                    **metric([row["gold_rank"] for row in selected], final_k=args.final_k),
                    "candidate_recall": round(sum(row["candidate_recall"] for row in selected) / len(selected), 4),
                    "mean_rerank_ms": round(mean(row["rerank_ms"] for row in selected), 4),
                    "p95_rerank_ms": round(sorted(row["rerank_ms"] for row in selected)[max(0, int(len(selected) * .95) - 1)], 4),
                    "method": method, "candidate_depth": depth,
                }
        report = {
            "schema_version": 1,
            "experiment": "current-guided-cross-encoder-reranker-v1",
            "complete": True,
            "review_type": "gold citation retrieval proxy; no human answer labels",
            "provider_config": config.summary(),
            "index": str(args.index),
            "index_manifest_hash": payload.get("manifest_hash"),
            "cases_file": str(args.cases),
            "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "cross_encoder": {"model": args.model, "device": args.device, "max_length": args.max_length},
            "candidate_depths": depths, "final_k": args.final_k,
            "controls": {"online_path_changed": False, "retries": 0, "concurrency": 1,
                         "candidate_source": "current GuidedKnowledgeSearch dedup stage"},
            "gpu_note": "CUDA availability must be verified separately; this run records the requested device.",
            "summary": summary, "rows": rows, "traces": all_traces,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}", flush=True)
    finally:
        dense.close()


if __name__ == "__main__":
    main()
