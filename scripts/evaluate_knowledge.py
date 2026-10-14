"""Deterministic ranked-retrieval scoring for pinned knowledge questions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def _matches(hit: dict, reference: dict) -> bool:
    return hit["source_id"] == reference["source_id"] and all(
        hit.get(key) == reference[key] for key in ("page", "heading", "anchor") if key in reference
    )


def validate_cases(search, cases: list[dict]) -> None:
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("knowledge case IDs must be unique")
    for case in cases:
        if case["split"] not in {"development", "holdout"}:
            raise ValueError(f"case {case['id']} has invalid split")
        if case["answerable"] != bool(case["relevant"]):
            raise ValueError(f"case {case['id']} has incomplete relevance labels")
        for reference in case["relevant"]:
            if not any(_matches({"source_id": node.metadata["id"],
                                 "page": node.metadata.get("page"),
                                 "heading": node.metadata.get("heading"),
                                 "anchor": node.metadata.get("anchor")}, reference)
                       for node in search.nodes.values()):
                raise ValueError(f"case {case['id']} references a missing source location: {reference}")


def evaluate_cases(search, cases: list[dict], *, split: str, mode: str, top_k: int = 10,
                   min_score: float | None = None) -> dict:
    if hasattr(search, "nodes"):
        validate_cases(search, cases)
    selected = [case for case in cases if case["split"] == split]
    if not selected:
        raise ValueError(f"no cases in {split} split")
    rows = []
    for case in selected:
        result = search.search(case["question"], mode=mode, top_k=top_k)
        hits = [hit for hit in result["hits"] if min_score is None or hit["score"] >= min_score]
        references = case["relevant"]
        seen_references: set[int] = set()
        ranks = []
        for rank, hit in enumerate(hits, 1):
            match = next((index for index, ref in enumerate(references)
                          if index not in seen_references and _matches(hit, ref)), None)
            if match is not None:
                seen_references.add(match)
                ranks.append(rank)
        dcg = sum(1 / math.log2(rank + 1) for rank in ranks[:10])
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(references), 10) + 1))
        resolved_hits = [
            hit["citation_id"] in search.nodes
            and search.nodes[hit["citation_id"]].metadata["id"] == hit["source_id"]
            and search.nodes[hit["citation_id"]].metadata["page"] == hit.get("page")
            and search.nodes[hit["citation_id"]].metadata["heading"] == hit.get("heading")
            and search.nodes[hit["citation_id"]].metadata.get("anchor") == hit.get("anchor")
            for hit in hits
        ] if hasattr(search, "nodes") else [any(_matches(hit, ref) for ref in references) for hit in hits]
        resolved = mean(resolved_hits) if resolved_hits else None
        rows.append({"id": case["id"], "answerable": case["answerable"], "ranks": ranks,
                     "hit_count": len(hits), "recall_at_5": sum(rank <= 5 for rank in ranks) / len(references) if references else None,
                     "mrr_at_10": 1 / ranks[0] if ranks and ranks[0] <= 10 else 0,
                     "ndcg_at_10": dcg / ideal if ideal else None,
                     "citation_resolved": resolved, "resolved_hit_count": sum(resolved_hits),
                     "elapsed_ms": result["elapsed_ms"],
                     "embedding_ms": result.get("embedding_ms", 0.0),
                     "vector_query_ms": result.get("vector_query_ms", 0.0)})
    answerable = [row for row in rows if row["answerable"]]
    no_answer = [row for row in rows if not row["answerable"]]
    return {
        "split": split, "mode": mode, "case_count": len(rows), "answerable_count": len(answerable),
        "recall_at_5": mean(row["recall_at_5"] for row in answerable) if answerable else None,
        "mrr_at_10": mean(row["mrr_at_10"] for row in answerable) if answerable else None,
        "ndcg_at_10": mean(row["ndcg_at_10"] for row in answerable) if answerable else None,
        "citation_location_accuracy": sum(row["resolved_hit_count"] for row in rows) / sum(row["hit_count"] for row in rows) if any(row["hit_count"] for row in rows) else None,
        "citation_checked_hit_count": sum(row["hit_count"] for row in rows),
        "no_answer_false_acceptance_rate": mean(row["hit_count"] > 0 for row in no_answer) if no_answer else None,
        "mean_retrieval_ms": mean(row["elapsed_ms"] for row in rows),
        "mean_embedding_ms": mean(row["embedding_ms"] for row in rows),
        "mean_vector_query_ms": mean(row["vector_query_ms"] for row in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "holdout"], required=True)
    parser.add_argument("--mode", choices=["bm25", "dense", "rrf"], default="bm25")
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--query-cache", type=Path, help="precomputed question vectors; excludes API latency")
    parser.add_argument("--write-query-cache", type=Path, help="batch-embed all benchmark questions and exit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.write_query_cache:
        from scripts.knowledge_vector import APIEmbedding
        embedding = APIEmbedding()
        questions = list(dict.fromkeys(case["question"] for case in cases))
        vectors = embedding.embed(questions)
        args.write_query_cache.write_text(json.dumps({"model": embedding.model,
                                                       "vectors": dict(zip(questions, vectors))}), encoding="utf-8")
        return
    from scripts.knowledge_search import configured_search
    search = configured_search(args.index)
    if search is None:
        parser.error("knowledge index is unavailable")
    if args.query_cache:
        if search.dense is None or args.mode == "bm25":
            parser.error("query cache requires a dense index and dense or rrf mode")
        cache = json.loads(args.query_cache.read_text(encoding="utf-8"))
        if cache["model"] != search.dense.embedding.model:
            parser.error("query cache model differs from dense index")
        if any(question not in cache["vectors"] or len(cache["vectors"][question]) != search.dense.dimension
               for question in (case["question"] for case in cases)):
            parser.error("query cache has missing questions or inconsistent dimensions")

        class CachedEmbedding:
            def embed(self, texts):
                return [cache["vectors"][text] for text in texts]

        search.dense.embedding = CachedEmbedding()
    report = evaluate_cases(search, cases, split=args.split, mode=args.mode, min_score=args.min_score)
    index_metadata = json.loads((args.index / "nodes.json").read_text(encoding="utf-8"))
    report["index"] = {key: index_metadata[key] for key in ("manifest_hash", "chunk_size", "overlap")}
    report["index"]["source_hashes"] = dict(sorted({node.metadata["id"]: node.metadata["sha256"]
                                                      for node in search.nodes.values()}.items()))
    if (args.index / "dense.json").exists():
        report["index"]["embedding"] = json.loads((args.index / "dense.json").read_text(encoding="utf-8"))
    report["query_embedding_cache"] = bool(args.query_cache)
    report["min_score"] = args.min_score
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
