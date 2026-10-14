"""Evaluate guided retrieval with answer-bearing citations and abstention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from math import log2

from scripts.model_adapter import ModelRequest, ModelResponse


def validate_guided_cases(search, cases: list[dict]) -> None:
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("guided case IDs must be unique")
    for case in cases:
        support_ids = case.get("supporting_citation_ids")
        if case["answerable"] != bool(case["relevant"]) or (
            support_ids is not None and case["answerable"] != bool(support_ids)
        ):
            raise ValueError(f"case {case['id']} has incomplete labels")
        for citation_id in support_ids or []:
            node = search.nodes.get(citation_id)
            if node is None:
                raise ValueError(f"case {case['id']} references a missing supporting citation: {citation_id}")
            meta = node.metadata
            if not any(meta["id"] == ref["source_id"] and all(
                meta.get(key) == value for key, value in ref.items() if key != "source_id"
            ) for ref in case["relevant"]):
                raise ValueError(f"case {case['id']} has a supporting citation outside its labeled location")


def evaluate_guided_cases(search, cases: list[dict], *, split: str, top_k: int = 5) -> dict:
    validate_guided_cases(search, cases)
    selected = cases if split == "all" else [case for case in cases if case["split"] == split]
    if not selected:
        raise ValueError(f"no cases in {split} split")
    rows = []
    for case in selected:
        # Fact annotations may share a page/anchor. Ranking evaluates unique
        # source locations, not the number of facts on one location.
        references = list({json.dumps(ref, sort_keys=True): ref for ref in case["relevant"]}.values())
        case = dict(case, relevant=references)
        result = search.search(case["question"], top_k=top_k)
        hits = result["hits"]
        hits_at_5 = hits[:5]
        matched_locations = set()
        for hit in hits_at_5:
            for index, ref in enumerate(case["relevant"]):
                if hit["source_id"] == ref["source_id"] and all(
                    hit.get(key) == value for key, value in ref.items() if key != "source_id"
                ):
                    matched_locations.add(index)
        support_ids = case.get("supporting_citation_ids")
        support_hits = {hit["citation_id"] for hit in hits_at_5} & set(support_ids or [])
        relevant_ranks = []
        ranked_locations = set()
        for rank, hit in enumerate(hits_at_5, 1):
            matches = {index for index, ref in enumerate(references)
                       if hit["source_id"] == ref["source_id"] and all(
                           hit.get(key) == value for key, value in ref.items() if key != "source_id")}
            if matches - ranked_locations:
                relevant_ranks.append(rank)
                ranked_locations.update(matches)
        reciprocal_rank = 1 / min(relevant_ranks) if relevant_ranks else 0.0
        ideal = min(len(case["relevant"]), 5)
        dcg = sum(1 / log2(rank + 1) for rank in relevant_ranks)
        idcg = sum(1 / log2(rank + 1) for rank in range(1, ideal + 1))
        integrity = []
        for hit in hits_at_5:
            node = search.nodes.get(hit["citation_id"])
            integrity.append(node is not None and node.metadata["id"] == hit["source_id"]
                             and node.metadata.get("page") == hit.get("page")
                             and node.metadata.get("anchor") == hit.get("anchor"))
        rows.append({
            "id": case["id"], "language": case.get("language", "unknown"), "answerable": case["answerable"],
            "retrieval_status": result["retrieval_status"], "reason_code": result["reason_code"],
            "rewritten_query": result["rewritten_query"],
            "location_recall_at_5": len(matched_locations) / len(case["relevant"]) if case["answerable"] else None,
            "location_recall_at_10": len({
                index for hit in hits[:10] for index, ref in enumerate(case["relevant"])
                if hit["source_id"] == ref["source_id"] and all(
                    hit.get(key) == value for key, value in ref.items() if key != "source_id"
                )
            }) / len(case["relevant"]) if case["answerable"] else None,
            "mrr_at_5": reciprocal_rank if case["answerable"] else None,
            "ndcg_at_5": dcg / idcg if case["answerable"] and idcg else None,
            "support_recall_at_5": len(support_hits) / len(support_ids) if case["answerable"] and support_ids else None,
            "hit_count": len(hits),
            "hits": [{key: value for key, value in hit.items() if key != "excerpt"} for hit in hits],
            "integrity": integrity,
            "elapsed_ms": result["elapsed_ms"], "planning_ms": result["planning_ms"],
            "embedding_ms": result["embedding_ms"], "vector_query_ms": result["vector_query_ms"],
        })
    answerable = [row for row in rows if row["answerable"]]
    annotated = [row for row in answerable if row["support_recall_at_5"] is not None]
    no_answer = [row for row in rows if not row["answerable"]]
    all_integrity = [value for row in rows for value in row["integrity"]]
    by_language = {}
    for language in sorted({row["language"] for row in rows}):
        language_rows = [row for row in answerable if row["language"] == language]
        by_language[language] = {
            "answerable_count": len(language_rows),
            "location_recall_at_5": mean(row["location_recall_at_5"] for row in language_rows) if language_rows else None,
            "support_recall_at_5": mean(row["support_recall_at_5"] for row in language_rows
                                        if row["support_recall_at_5"] is not None) if any(
                                            row["support_recall_at_5"] is not None for row in language_rows) else None,
        }
    return {
        "split": split, "case_count": len(rows), "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "location_recall_at_5": mean(row["location_recall_at_5"] for row in answerable) if answerable else None,
        "location_recall_at_10": mean(row["location_recall_at_10"] for row in answerable) if answerable else None,
        "mrr_at_5": mean(row["mrr_at_5"] for row in answerable) if answerable else None,
        "ndcg_at_5": mean(row["ndcg_at_5"] for row in answerable) if answerable else None,
        "support_recall_at_5": mean(row["support_recall_at_5"] for row in annotated) if annotated else None,
        "negative_candidate_rate": mean(bool(row["hit_count"]) for row in no_answer) if no_answer else None,
        # Legacy key retained for report compatibility; this measures retrieval candidates, not answers.
        "no_answer_false_acceptance_rate": mean(bool(row["hit_count"]) for row in no_answer) if no_answer else None,
        "citation_integrity": mean(all_integrity) if all_integrity else None,
        "citation_checked_hit_count": len(all_integrity),
        "mean_elapsed_ms": mean(row["elapsed_ms"] for row in rows),
        "mean_planning_ms": mean(row["planning_ms"] for row in rows),
        "by_language": by_language, "cases": rows,
    }


def calibrate_threshold(search, cases: list[dict], *, thresholds: list[float]) -> dict:
    if not thresholds or any(value < 0 or value > 1 for value in thresholds):
        raise ValueError("threshold candidates must be in [0, 1]")
    candidates = []
    for threshold in sorted(set(thresholds)):
        search.min_dense_score = threshold
        report = evaluate_guided_cases(search, cases, split="development")
        if report["no_answer_false_acceptance_rate"] == 0:
            metric = report["support_recall_at_5"]
            candidates.append((metric if metric is not None else report["location_recall_at_5"], threshold, report))
    if not candidates:
        raise ValueError("no threshold has zero development false acceptance")
    _, threshold, report = max(candidates, key=lambda item: (item[0], item[1]))
    search.min_dense_score = threshold
    return {"selected_threshold": threshold, "development": report}


def _cases_hash(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare_cache(cases: list[dict], cache_dir: Path, model, embedding) -> None:
    from scripts.knowledge_guidance import build_guidance_prompt, parse_guidance_plan

    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.iterdir()):
        raise ValueError("cache directory must be empty")
    plans = {}
    query_texts = []
    for case in cases:
        question = case["question"]
        prompt = build_guidance_prompt(question)
        response = model.generate(ModelRequest(prompt=prompt, response_format="json_object"))
        plan = parse_guidance_plan(response.text, question)
        plans[prompt] = response.text
        if plan.action == "search":
            query_texts.extend((question, plan.english_query))
    unique_texts = list(dict.fromkeys(query_texts))
    vectors = dict(zip(unique_texts, embedding.embed(unique_texts))) if unique_texts else {}
    stamp = _cases_hash(cases)
    (cache_dir / "plans.json").write_text(json.dumps({"cases_sha256": stamp, "model": model.model,
                                                     "plans": plans}, ensure_ascii=False), encoding="utf-8")
    (cache_dir / "vectors.json").write_text(json.dumps({"cases_sha256": stamp, "model": embedding.model,
                                                       "vectors": vectors}), encoding="utf-8")


def load_cache(cases: list[dict], cache_dir: Path):
    plans = json.loads((cache_dir / "plans.json").read_text(encoding="utf-8"))
    vectors = json.loads((cache_dir / "vectors.json").read_text(encoding="utf-8"))
    expected = _cases_hash(cases)
    if plans["cases_sha256"] != expected or vectors["cases_sha256"] != expected:
        raise ValueError("cached guidance belongs to different cases")

    class CachedModel:
        model = plans["model"]

        def generate(self, request):
            if request.prompt not in plans["plans"]:
                raise ValueError("guidance prompt differs from cached version")
            return ModelResponse(plans["plans"][request.prompt], self.model)

    class CachedEmbedding:
        model = vectors["model"]

        def embed(self, texts):
            return [vectors["vectors"][text] for text in texts]

        async def aembed(self, texts):
            return self.embed(texts)

    return CachedModel(), CachedEmbedding()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from scripts.knowledge_search import KnowledgeSearch, load_nodes
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.model_adapter import OpenAICompatibleModel
    from scripts.knowledge_vector import APIEmbedding, DenseIndex

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    selected = [case for case in cases if case["split"] == args.split]
    if not selected:
        parser.error("selected split has no cases")
    if args.prepare_cache:
        prepare_cache(selected, args.cache_dir, OpenAICompatibleModel(timeout=5.0), APIEmbedding())
        print(f"Cached {len(selected)} guidance plans and query vectors at {args.cache_dir}")
        return
    model, embedding = load_cache(selected, args.cache_dir)
    dense = DenseIndex.open(args.index, embedding)
    index = KnowledgeSearch(load_nodes(args.index), dense=dense)
    search = GuidedKnowledgeSearch(index, model, min_dense_score=args.threshold)
    try:
        if args.calibrate:
            if args.split != "development":
                parser.error("calibration is restricted to development")
            calibration = calibrate_threshold(search, selected, thresholds=[value / 100 for value in range(81)])
            report = calibration["development"]
            report["threshold"] = calibration["selected_threshold"]
        else:
            report = evaluate_guided_cases(search, selected, split=args.split)
            report["threshold"] = args.threshold
        report["planner_model"] = search.model.model
        report["embedding_model"] = index.dense.embedding.model
        report["query_embedding_cache"] = True
        report["source_hashes"] = dict(sorted({node.metadata["id"]: node.metadata["sha256"]
                                               for node in index.nodes.values()}.items()))
        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output)
    finally:
        index.dense.close()


if __name__ == "__main__":
    main()
