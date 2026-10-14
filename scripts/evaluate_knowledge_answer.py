"""Live, answer-level M2.1 evaluation; semantic labels require human review."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from scripts.provider_config import DEFAULT_ENV_PATH, load_provider_config


def validate_cases(cases: list[dict], search=None) -> None:
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("answer case IDs must be unique")
    for case in cases:
        outcome = case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal")
        if outcome not in {"answer", "refusal", "premise_denial"}:
            raise ValueError(f"unknown expected outcome: {case['id']}")
        citations = case["supporting_citation_ids"]
        if outcome == "answer" and (not case["answerable"] or not citations or not case["expected_facts"]):
            raise ValueError(f"incomplete answer labels: {case['id']}")
        if outcome == "refusal" and (case["answerable"] or citations or case["expected_facts"]):
            raise ValueError(f"incomplete refusal labels: {case['id']}")
        if outcome == "premise_denial" and (case["answerable"] or not citations or not case["expected_facts"]):
            raise ValueError(f"incomplete answer labels: {case['id']}")
        if search is not None and any(citation not in search.nodes for citation in citations):
            raise ValueError(f"missing citation in pinned index: {case['id']}")


def grade_response(case: dict, result: dict) -> str:
    claims = result.get("claims", [])
    outcome = case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal")
    if outcome == "refusal":
        return "false_answer" if claims else "correct_refusal"
    if outcome == "premise_denial":
        if not claims:
            return "premise_denial_missing"
        if not set(case["supporting_citation_ids"]) <= {claim["citation_id"] for claim in claims}:
            return "premise_denial_wrong_citation"
        return "premise_denial_review"
    if result.get("status") != "answered" or result.get("unresolved") or not claims:
        return "incomplete"
    if not set(case["supporting_citation_ids"]) <= {claim["citation_id"] for claim in claims}:
        return "wrong_citation"
    return "manual_review"


async def evaluate(answerer, cases: list[dict], *, inter_case_delay: float = 0.0) -> dict:
    from scripts.knowledge_answer import KnowledgeAnswer
    if inter_case_delay < 0:
        raise ValueError("inter_case_delay must be non-negative")
    validate_cases(cases)
    rows = []
    for case_index, case in enumerate(cases):
        started = perf_counter()
        if hasattr(answerer.model, "outputs"):
            answerer.model.outputs.clear()
        if hasattr(answerer.guided, "last_hits"):
            answerer.guided.last_hits = []
        try:
            # The real answerer owns its deadline and stage diagnostics. A second
            # equal-budget wait races its error translation and discards timings.
            if isinstance(answerer, KnowledgeAnswer):
                result = await answerer.aanswer(case["question"])
            else:
                result = await asyncio.wait_for(
                    answerer.aanswer(case["question"]), timeout=getattr(answerer, "total_timeout", 20.0),
                )
            grade = grade_response(case, result)
            row = {"id": case["id"], "question": case["question"], "language": case["language"],
                   "answerable": case["answerable"],
                   "expected_outcome": case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal"),
                   "expected_facts": case["expected_facts"],
                   "expected_citation_ids": case["supporting_citation_ids"], "grade": grade,
                   **result}
        except asyncio.TimeoutError:
            row = {"id": case["id"], "question": case["question"], "language": case["language"],
                   "answerable": case["answerable"],
                   "expected_outcome": case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal"),
                   "expected_facts": case["expected_facts"],
                   "expected_citation_ids": case["supporting_citation_ids"],
                   "grade": "error", "error_type": "TimeoutError",
                   "error_code": "total_timeout", "error_stage": "total"}
        except Exception as exc:
            row = {"id": case["id"], "question": case["question"], "language": case["language"],
                   "answerable": case["answerable"],
                   "expected_outcome": case.get("expected_outcome") or ("answer" if case["answerable"] else "refusal"),
                   "expected_facts": case["expected_facts"],
                   "expected_citation_ids": case["supporting_citation_ids"],
                   "grade": "error", "error_type": type(exc).__name__}
            if hasattr(exc, "reason_code"):
                row["error_code"] = exc.reason_code
                row["error_stage"] = exc.stage
                row.update(getattr(exc, "timings", {}))
                if getattr(exc, "trace_id", None):
                    row["trace_id"] = exc.trace_id
        row["wall_ms"] = round((perf_counter() - started) * 1000, 3)
        if hasattr(answerer.model, "outputs"):
            row["model_outputs"] = list(answerer.model.outputs)
        if hasattr(answerer.guided, "last_hits"):
            row["retrieval_hits"] = list(answerer.guided.last_hits)
        rows.append(row)
        print(f"{case['id']} {row['grade']} {row['wall_ms']:.0f}ms", flush=True)
        if case_index < len(cases) - 1 and inter_case_delay:
            await asyncio.sleep(inter_case_delay)
    elapsed = sorted(row["wall_ms"] for row in rows)
    error_codes = {
        code: sum(row.get("error_code") == code for row in rows)
        for code in sorted({row.get("error_code") for row in rows if row.get("error_code")})
    }
    return {"evaluation_mode": "live", "case_count": len(rows),
            "case_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "mean_wall_ms": round(mean(elapsed), 3),
            "p95_wall_ms": elapsed[max(0, (95 * len(elapsed) + 99) // 100 - 1)],
            "concurrency": 1, "inter_case_delay_seconds": inter_case_delay,
            "error_count": sum(row["grade"] == "error" for row in rows),
            "error_codes": error_codes,
            "outcome_counts": {
                outcome: sum(row.get("expected_outcome") == outcome for row in rows)
                for outcome in sorted({row.get("expected_outcome") for row in rows})
            },
            "grades": {grade: sum(row["grade"] == grade for row in rows)
                       for grade in sorted({row["grade"] for row in rows})}, "cases": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("metadata/knowledge_answer_cases_m21a_v1.json"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--trace-model-output", action="store_true")
    parser.add_argument("--model-timeout", type=float)
    parser.add_argument("--planner-timeout", type=float, default=5.0)
    parser.add_argument("--retrieval-timeout", type=float, default=7.0)
    parser.add_argument("--embedding-timeout", type=float, default=8.0)
    parser.add_argument("--generation-timeout", type=float, default=6.0)
    parser.add_argument("--verification-timeout", type=float, default=6.0)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--generation-thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--inter-case-delay", type=float, default=0.0)
    parser.add_argument("--allow-model-mismatch", action="store_true",
                        help="use the pinned retrieval threshold for an uncalibrated planner model")
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_PATH)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from scripts.knowledge_answer import KnowledgeAnswer
    from scripts.knowledge_guidance import GuidedKnowledgeSearch, POLICY_PATH
    from scripts.knowledge_search import KnowledgeSearch, load_nodes
    from scripts.knowledge_vector import APIEmbedding, DenseIndex
    from scripts.model_adapter import OpenAICompatibleModel
    from scripts.tracing import configured_tracer, flush_tracer

    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.ids:
        cases = [case for case in cases if case["id"] in set(args.ids)]
        if len(cases) != len(set(args.ids)):
            parser.error("unknown or duplicate case ID")
    if args.limit is not None:
        cases = cases[:args.limit]
    provider_config = load_provider_config(args.config)
    embedding = APIEmbedding(provider_config)
    dense = DenseIndex.open(args.index, embedding)
    search = KnowledgeSearch(load_nodes(args.index), dense=dense,
                             manifest_hash=json.loads((args.index / "nodes.json").read_text())["manifest_hash"])
    validate_cases(cases, search)
    try:
        tracer = configured_tracer(provider_config)
        model = OpenAICompatibleModel(
            config=provider_config,
            timeout=args.model_timeout or 6.0,
        )
        if args.trace_model_output:
            class RecordingModel:
                def __init__(self, underlying):
                    self.underlying = underlying
                    self.model = underlying.model
                    self.outputs = []

                def generate(self, request):
                    response = self.underlying.generate(request)
                    self.outputs.append(response.text)
                    return response

            model = RecordingModel(model)
        if args.allow_model_mismatch:
            policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
            guided = GuidedKnowledgeSearch(
                search, model, min_dense_score=policy["min_dense_score"],
                planner_timeout=args.planner_timeout, embedding_timeout=args.embedding_timeout,
                tracer=tracer,
            )
        else:
            guided = GuidedKnowledgeSearch(
                search, model, planner_timeout=args.planner_timeout,
                embedding_timeout=args.embedding_timeout, tracer=tracer,
            )
        answerer = KnowledgeAnswer(
            search, model, guided=guided, retrieval_timeout=args.retrieval_timeout,
            generation_timeout=args.generation_timeout,
            verification_timeout=args.verification_timeout,
            total_timeout=args.request_timeout, tracer=tracer,
            generation_thinking=("disabled" if args.generation_thinking == "disabled" else None),
        )
        if args.trace_model_output:
            class RecordingGuided:
                def __init__(self, underlying):
                    self.underlying = underlying
                    self.last_hits = []

                async def asearch(self, question, *, top_k):
                    result = await self.underlying.asearch(question, top_k=top_k)
                    self.last_hits = [{"citation_id": hit["citation_id"], "page": hit["page"],
                                       "anchor": hit["anchor"]} for hit in result["hits"]]
                    return result

            answerer.guided = RecordingGuided(answerer.guided)
        report = asyncio.run(evaluate(answerer, cases, inter_case_delay=args.inter_case_delay))
        report["planner_model"] = answerer.model.model
        report["embedding_model"] = embedding.model
        report["generation_thinking"] = args.generation_thinking
        report["timeout_budgets_seconds"] = {
            "model_request": args.model_timeout or 6.0,
            "planner": args.planner_timeout,
            "embedding": args.embedding_timeout,
            "generation": args.generation_timeout,
            "verification": args.verification_timeout,
            "retrieval": args.retrieval_timeout,
            "total_request": args.request_timeout,
        }
        report["source_hashes"] = dict(sorted({node.metadata["id"]: node.metadata["sha256"]
                                               for node in search.nodes.values()}.items()))
        report["tracing"] = {"enabled": tracer is not None, "flushed": False}
        if tracer is not None:
            report["tracing"]["flushed"] = flush_tracer()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}")
    finally:
        dense.close()


if __name__ == "__main__":
    main()
