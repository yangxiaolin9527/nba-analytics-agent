"""Evaluate generation and verification with manually selected evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts.knowledge_answer import KnowledgeAnswer
from scripts.knowledge_search import KnowledgeSearch, load_nodes
from scripts.knowledge_vector import APIEmbedding, DenseIndex
from scripts.model_adapter import OpenAICompatibleModel
from scripts.retrieval_lab import oracle_context_answer, oracle_context_prompt


def select_answer_cases(cases: list[dict], ids: list[str] | None = None) -> list[dict]:
    selected = [case for case in cases if case.get("expected_outcome") == "answer"]
    if ids is not None:
        requested = set(ids)
        if len(requested) != len(ids):
            raise ValueError("duplicate case ID")
        selected = [case for case in selected if case["id"] in requested]
        if len(selected) != len(requested):
            raise ValueError("unknown or non-answer case ID")
    return selected


async def evaluate_cases(answerer, search, cases: list[dict], *, inter_case_delay: float = 0.0) -> dict:
    rows = []
    for index, case in enumerate(cases):
        started = perf_counter()
        context = oracle_context_prompt(search, case["supporting_citation_ids"], case["question"])
        try:
            result = await asyncio.wait_for(
                oracle_context_answer(answerer, case["question"], context["evidence"]),
                timeout=getattr(answerer, "total_timeout", 30.0),
            )
        except asyncio.TimeoutError:
            result = {"status": "abstained", "claims": [], "unresolved": [case["question"]],
                      "error_code": "total_timeout", "failure_class": "provider_runtime_failure",
                      "error_stage": "total", "elapsed_ms": round((perf_counter() - started) * 1000, 3)}
        except Exception as exc:
            result = {"status": "error", "claims": [], "unresolved": [case["question"]],
                      "error_code": type(exc).__name__, "failure_class": "runner_failure",
                      "error_stage": "runner", "elapsed_ms": 0.0}
        rows.append({"id": case["id"], "question": case["question"],
                     "expected_facts": case["expected_facts"],
                     "gold_citation_ids": case["supporting_citation_ids"],
                     "injected_evidence_ids": [item["citation_id"] for item in context["evidence"]],
                     "evidence_chars": context["evidence_chars"],
                     "evidence_count": context["evidence_count"],
                     "result": result,
                     "wall_ms": round((perf_counter() - started) * 1000, 3)})
        if index < len(cases) - 1 and inter_case_delay:
            await asyncio.sleep(inter_case_delay)
    elapsed = sorted(row["wall_ms"] for row in rows)
    return {"case_count": len(rows), "concurrency": 1,
            "inter_case_delay_seconds": inter_case_delay,
            "mean_wall_ms": round(mean(elapsed), 3) if elapsed else 0.0,
            "p95_wall_ms": elapsed[max(0, (95 * len(elapsed) + 99) // 100 - 1)] if elapsed else 0.0,
            "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("metadata/knowledge_answer_cases_m21d_v1.json"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--inter-case-delay", type=float, default=0.0)
    parser.add_argument("--model-timeout", type=float, default=11.0)
    parser.add_argument("--generation-timeout", type=float, default=10.0)
    parser.add_argument("--verification-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    selected = select_answer_cases(cases, args.ids)
    embedding = APIEmbedding()
    dense = DenseIndex.open(args.index, embedding)
    payload = json.loads((args.index / "nodes.json").read_text(encoding="utf-8"))
    search = KnowledgeSearch(load_nodes(args.index), dense=dense, manifest_hash=payload["manifest_hash"])
    try:
        model = OpenAICompatibleModel(timeout=args.model_timeout)
        answerer = KnowledgeAnswer(search, model, generation_timeout=args.generation_timeout,
                                   verification_timeout=args.verification_timeout,
                                   total_timeout=args.request_timeout)
        summary = asyncio.run(evaluate_cases(answerer, search, selected,
                                              inter_case_delay=args.inter_case_delay))
        summary.update({"version": 1, "runner": "m2.3-oracle-context",
                        "cases_file": str(args.cases),
                        "case_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True,
                                                                  ensure_ascii=False).encode()).hexdigest(),
                        "index_manifest_hash": payload["manifest_hash"],
                        "model": model.model,
                        "timeout_budgets_seconds": {
                            "model_request": args.model_timeout,
                            "generation": args.generation_timeout,
                            "verification": args.verification_timeout,
                            "request": args.request_timeout,
                        }})
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dense.close()


if __name__ == "__main__":
    main()
