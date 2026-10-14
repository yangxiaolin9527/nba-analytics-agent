"""Run M2.2-RL retrieval diagnostics without changing the answer endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.knowledge_guidance import GuidedKnowledgeSearch
from scripts.knowledge_search import KnowledgeSearch, load_nodes
from scripts.knowledge_vector import APIEmbedding, DenseIndex
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import DEFAULT_ENV_PATH, load_provider_config
from scripts.retrieval_lab import RetrievalLabError, diagnose_retrieval, gold_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("metadata/knowledge_answer_cases_m21d_v1.json"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_PATH)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    provider_config = load_provider_config(args.config)
    embedding = APIEmbedding(provider_config)
    dense = DenseIndex.open(args.index, embedding)
    payload = json.loads((args.index / "nodes.json").read_text(encoding="utf-8"))
    search = KnowledgeSearch(load_nodes(args.index), dense=dense, manifest_hash=payload["manifest_hash"])
    try:
        model = OpenAICompatibleModel(config=provider_config, timeout=30.0)
        guided = GuidedKnowledgeSearch(search, model, min_dense_score=0.43)
        rows = []
        for case in cases:
            row = {"id": case["id"], "question": case["question"],
                   "expected_outcome": case.get("expected_outcome"),
                   "gold_citation_ids": case["supporting_citation_ids"]}
            try:
                diagnostic = diagnose_retrieval(guided, case["question"], top_k=args.top_k)
                row.update(diagnostic)
                row["gold_trace"] = gold_trace(diagnostic, case["supporting_citation_ids"])
            except Exception as exc:
                row["final_status"] = "error"
                row["failure_class"] = getattr(exc, "failure_class", "provider_runtime_failure" if getattr(exc, "stage", "") == "planner" else "retrieval_failure")
                row["error"] = {"stage": getattr(exc, "stage", "unknown"),
                                 "type": getattr(exc, "original_type", type(exc).__name__),
                                 "message": str(exc)}
            rows.append(row)
            print(f"{case['id']} {row.get('final_status')}", flush=True)
        output = {"version": 1, "runner": "m2.2-retrieval-lab", "cases_file": str(args.cases),
                  "index": str(args.index), "index_manifest_hash": payload["manifest_hash"],
                  "planner_model": model.model, "embedding_model": embedding.model,
                  "top_k": args.top_k, "dense_threshold": 0.43, "cases": rows}
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dense.close()


if __name__ == "__main__":
    main()
