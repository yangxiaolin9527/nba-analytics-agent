"""Compare dense/RRF retrieval across independently built embedding indexes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from scripts.evaluate_guided_knowledge import evaluate_guided_cases
from scripts.knowledge_guidance import GuidedKnowledgeSearch
from scripts.knowledge_search import KnowledgeSearch, load_nodes
from scripts.knowledge_vector import DenseIndex
from scripts.model_adapter import ModelResponse


def _cases_hash(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class CachedPlanner:
    model = "cached-guidance-plan"

    def __init__(self, plans: dict[str, dict]):
        self.plans = plans

    def generate(self, request):
        question = request.prompt.rsplit("Question: ", 1)[-1].strip()
        try:
            question = json.loads(question)
        except json.JSONDecodeError:
            pass
        return ModelResponse(json.dumps(self.plans[question], ensure_ascii=False), self.model)


def load_plan_cache(cases: list[dict], path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cases_sha256") != _cases_hash(cases):
        raise ValueError("plan cache belongs to different cases")
    plans = payload.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("plan cache must contain a plans object")
    if set(plans) != {case["question"] for case in cases}:
        raise ValueError("plan cache does not cover exactly the selected cases")
    return plans


def run(index: Path, cases_path: Path, split: str, threshold: float, plan_cache: Path,
        provider: str | None = None, model: str | None = None, dimension: int | None = None,
        normalize: bool | None = None, device: str | None = None) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    selected = cases if split == "all" else [case for case in cases if case["split"] == split]
    plans = load_plan_cache(selected, plan_cache)
    selected = [dict(case, answerable=(case["answerable"] or case.get("expected_outcome") == "premise_denial"), relevant=case.get("relevant", [
        {key: item[key] for key in ("source_id", "page", "anchor") if key in item}
        for item in case.get("fact_evidence", [])
    ])) for case in selected]
    from scripts.embedding_adapters import load_embedding_adapter
    from scripts.provider_config import load_provider_config
    config = load_provider_config().with_embedding_overrides(
        embedding_provider=provider, embedding_model=model, embedding_dimension=dimension,
        embedding_normalize=normalize, embedding_local_device=device,
    )
    dense = DenseIndex.open(index, load_embedding_adapter(config))
    search = KnowledgeSearch(load_nodes(index), dense=dense)
    guided = GuidedKnowledgeSearch(search, CachedPlanner(plans), min_dense_score=threshold)
    try:
        report = evaluate_guided_cases(guided, selected, split=split, top_k=10)
        report.update({
            "index": str(index), "split": split, "plan_cache": str(plan_cache),
            "plan_cache_sha256": hashlib.sha256(plan_cache.read_bytes()).hexdigest(),
            "dense": json.loads((index / "dense.json").read_text(encoding="utf-8")),
            "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        })
        return report
    finally:
        dense.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("metadata/knowledge_cases_v1.json"))
    parser.add_argument("--split", choices=("development", "holdout", "validation", "all"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--plan-cache", type=Path, required=True)
    parser.add_argument("--embedding-provider", choices=("ali_bailian", "local_sentence_transformers"), required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--embedding-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite report")
    args.output.write_text(json.dumps(run(args.index, args.cases, args.split, args.threshold, args.plan_cache,
                                          args.embedding_provider, args.embedding_model,
                                          args.embedding_dimension, args.embedding_normalize,
                                          args.embedding_device),
                                      ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
