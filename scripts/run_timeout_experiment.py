"""Small serial timeout experiment without a model-call ledger or call cap."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError
from scripts.knowledge_guidance import GuidedKnowledgeSearch, POLICY_PATH
from scripts.knowledge_search import configured_search
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import load_provider_config


async def run_case(*, question: str, config, generation_timeout: float,
                   total_timeout: float, thinking: str | None,
                   verification_thinking: str | None,
                   candidate_top_k: int | None = None) -> dict:
    search = configured_search(provider_config=config)
    if search is None or search.dense is None:
        raise RuntimeError("knowledge dense index unavailable")
    model = OpenAICompatibleModel(config=config)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    guided = GuidedKnowledgeSearch(
        search, model, min_dense_score=policy["min_dense_score"],
        planner_timeout=config.knowledge_planner_timeout_seconds,
        embedding_timeout=config.knowledge_embedding_timeout_seconds,
    )
    answerer = KnowledgeAnswer(
        search, model, guided=guided,
        retrieval_timeout=config.knowledge_retrieval_timeout_seconds,
        generation_timeout=generation_timeout,
        verification_timeout=config.knowledge_verification_timeout_seconds,
        total_timeout=total_timeout,
        generation_thinking=thinking,
        verification_thinking=verification_thinking,
        generation_prompt_strategy=config.knowledge_generation_prompt_strategy,
        generation_policy="fixed",
        candidate_top_k=candidate_top_k or config.knowledge_candidate_top_k,
    )
    started = perf_counter()
    try:
        result = await answerer.aanswer(question)
        row = {"status": result.get("retrieval_status", result.get("status")),
               "reason_code": result.get("reason_code"), "error_stage": result.get("error_stage"),
               "result": result}
    except KnowledgeAnswerError as exc:
        row = {"status": "failed", "reason_code": exc.reason_code,
               "error_stage": exc.stage, "timings": exc.timings,
               "model_calls": getattr(exc, "model_calls", {})}
    finally:
        search.dense.close()
    row.update({"question": question, "generation_thinking": thinking or "provider_default",
                "generation_timeout": generation_timeout, "total_timeout": total_timeout,
                "wall_ms": round((perf_counter() - started) * 1000, 3)})
    return row


async def main_async(args) -> None:
    config = load_provider_config(args.config)
    config.require_model(); config.require_embedding()
    rows = []
    for generation_timeout, total_timeout, thinking, verification_thinking in (
        (args.generation_timeout, args.total_timeout, "enabled", None),
        (args.generation_timeout, args.total_timeout, "disabled", "disabled"),
        (args.long_generation_timeout, args.long_total_timeout, "enabled", None),
        (args.long_generation_timeout, args.long_total_timeout, "disabled", "disabled"),
    ):
        row = await run_case(question=args.question, config=config,
                             generation_timeout=generation_timeout,
                             total_timeout=total_timeout, thinking=thinking,
                             verification_thinking=verification_thinking,
                             candidate_top_k=args.candidate_top_k)
        row["verification_thinking"] = verification_thinking or "provider_default"
        rows.append(row)
        print(row["generation_timeout"], row["generation_thinking"],
              row["status"], row.get("reason_code"), row["wall_ms"], flush=True)
    payload = {"experiment": "targeted-timeout-v1", "question": args.question,
               "controls": {"serial": True, "call_cap": None, "ledger": None},
               "interpretation": "provider failures are not timeout evidence; rerun after provider connectivity is restored",
               "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="什么时候吹罚带球撞人犯规？")
    parser.add_argument("--generation-timeout", type=float, default=10)
    parser.add_argument("--total-timeout", type=float, default=30)
    parser.add_argument("--long-generation-timeout", type=float, default=20)
    parser.add_argument("--long-total-timeout", type=float, default=40)
    parser.add_argument("--config", type=Path, default=Path(".env.local"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
