"""Stage-by-stage retrieval diagnostics and oracle-context evaluation helpers."""

from __future__ import annotations

import json
from time import perf_counter

from llama_index.core.schema import NodeWithScore

from scripts.knowledge_answer import (
    AnswerDraft,
    KnowledgeAnswerError,
    Verification,
    bounded_evidence,
    canonical_excerpt,
    generation_prompt,
    verification_prompt,
)
from scripts.knowledge_guidance import build_guidance_prompt, parse_guidance_plan
from scripts.model_adapter import ModelRequest


class RetrievalLabError(RuntimeError):
    def __init__(self, stage: str, failure_class: str, original: BaseException):
        super().__init__(str(original))
        self.stage = stage
        self.failure_class = failure_class
        self.original_type = type(original).__name__


def _candidate(item: NodeWithScore, rank: int, *, list_name: str) -> dict:
    meta = item.node.metadata
    return {"citation_id": meta["citation_id"], "rank": rank,
            "score": float(item.score or 0.0), "list": list_name,
            "source_id": meta["id"], "page": meta["page"], "anchor": meta.get("anchor")}


def _stage_hit(node, *, score: float, rank: int, stage: str, dense_score: float | None = None,
               rrf_score: float | None = None, production_dense_score: float | None = None) -> dict:
    meta = node.metadata
    return {"citation_id": meta["citation_id"], "rank": rank, "score": round(float(score), 8),
            "dense_score": None if dense_score is None else round(float(dense_score), 8),
            "production_dense_score": (None if production_dense_score is None else
                                       round(float(production_dense_score), 8)),
            "rrf_score": None if rrf_score is None else round(float(rrf_score), 8),
            "stage": stage, "source_id": meta["id"], "page": meta["page"], "anchor": meta.get("anchor")}


def diagnose_retrieval(guided, question: str, *, top_k: int = 5) -> dict:
    """Run the existing guided recipe while retaining every candidate boundary."""
    started = perf_counter()
    try:
        planner = guided.model.generate(ModelRequest(
            prompt=build_guidance_prompt(question), response_format="json_object",
            timeout=getattr(guided, "planner_timeout", None)))
    except Exception as exc:
        raise RetrievalLabError("planner", "provider_runtime_failure", exc) from exc
    try:
        plan = parse_guidance_plan(planner.text, question)
    except Exception as exc:
        raise RetrievalLabError("planner", "retrieval_failure", exc) from exc
    result = {"question": question, "planner": {"raw": planner.text},
              "stages": {}, "timings_ms": {}, "errors": []}
    result["planner"]["action"] = plan.action
    result["planner"]["intent"] = plan.intent
    result["planner"]["english_query"] = plan.english_query
    if plan.action == "abstain":
        result["final_status"] = "abstained"
        result["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 3)
        return result

    queries = list(dict.fromkeys([question, plan.english_query]))
    try:
        vectors = guided.search_index.dense.embedding.embed(queries)
    except Exception as exc:
        raise RetrievalLabError("embedding", "provider_runtime_failure", exc) from exc
    dense_lists = []
    for index, vector in enumerate(vectors):
        name = "original_dense_top50" if index == 0 else "rewritten_dense_top50"
        try:
            ranked = guided.search_index.dense.retrieve_vector(vector, min(50, len(guided.nodes)))
        except Exception as exc:
            raise RetrievalLabError("dense", "retrieval_failure", exc) from exc
        dense_lists.append(ranked)
        result["stages"][name] = [_candidate(item, rank, list_name=name) for rank, item in enumerate(ranked, 1)]
    try:
        lexical = guided.bm25.retrieve(plan.english_query)
    except Exception as exc:
        raise RetrievalLabError("bm25", "retrieval_failure", exc) from exc
    result["stages"]["bm25_top20"] = [_candidate(item, rank, list_name="bm25_top20")
                                      for rank, item in enumerate(lexical[:20], 1) if (item.score or 0) > 0]

    lexical_positive = [item for item in lexical if (item.score or 0) > 0]
    ranked_lists = dense_lists + [lexical_positive]
    true_dense_scores: dict[str, float] = {}
    production_dense_scores: dict[str, float] = {}
    rrf_scores: dict[str, float] = {}
    for ranked in dense_lists:
        for item in ranked:
            cid = item.node.metadata["citation_id"]
            true_dense_scores[cid] = max(true_dense_scores.get(cid, 0), float(item.score or 0))
    # Mirror the online route's score provenance: only dense retrieval lists
    # participate in the dense threshold; BM25 contributes to RRF only.
    production_dense_scores.update(true_dense_scores)
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            cid = item.node.metadata["citation_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (60 + rank)
    ordered = sorted(rrf_scores, key=lambda cid: (-rrf_scores[cid], -true_dense_scores.get(cid, 0), cid))
    result["stages"]["rrf_ranking"] = [_stage_hit(guided.nodes[cid], score=rrf_scores[cid], rank=rank,
                                                     stage="rrf", dense_score=true_dense_scores.get(cid),
                                                     rrf_score=rrf_scores[cid],
                                                     production_dense_score=production_dense_scores.get(cid))
                                       for rank, cid in enumerate(ordered, 1)]
    thresholded = [cid for cid in ordered if cid in production_dense_scores and production_dense_scores[cid] >= guided.min_dense_score]
    result["stages"]["dense_threshold"] = [_stage_hit(guided.nodes[cid], score=production_dense_scores[cid], rank=rank,
                                                        stage="dense_threshold", dense_score=true_dense_scores.get(cid),
                                                        production_dense_score=production_dense_scores[cid],
                                                        rrf_score=rrf_scores[cid])
                                             for rank, cid in enumerate(thresholded, 1)]
    domain = [cid for cid in thresholded if not ((plan.intent == "rules" and guided.nodes[cid].metadata["page"] is None) or
                                                  (plan.intent == "glossary" and guided.nodes[cid].metadata.get("anchor") is None))]
    result["stages"]["domain_filter"] = [_stage_hit(guided.nodes[cid], score=rrf_scores[cid], rank=rank,
                                                      stage="domain_filter", dense_score=true_dense_scores.get(cid),
                                                      production_dense_score=production_dense_scores.get(cid),
                                                      rrf_score=rrf_scores[cid])
                                           for rank, cid in enumerate(domain, 1)]
    dedup, seen = [], set()
    for cid in domain:
        meta = guided.nodes[cid].metadata
        location = (meta["id"], meta["page"] if meta["page"] is not None else meta.get("anchor"))
        if location in seen:
            continue
        seen.add(location); dedup.append(cid)
    result["stages"]["dedup"] = [_stage_hit(guided.nodes[cid], score=rrf_scores[cid], rank=rank,
                                               stage="dedup", dense_score=true_dense_scores.get(cid),
                                               production_dense_score=production_dense_scores.get(cid),
                                               rrf_score=rrf_scores[cid])
                                  for rank, cid in enumerate(dedup, 1)]
    final = dedup[:top_k]
    result["stages"]["final_top_k"] = [_stage_hit(guided.nodes[cid], score=rrf_scores[cid], rank=rank,
                                                    stage="final_top_k", dense_score=true_dense_scores.get(cid),
                                                    production_dense_score=production_dense_scores.get(cid),
                                                    rrf_score=rrf_scores[cid])
                                       for rank, cid in enumerate(final, 1)]
    result["final_status"] = "candidates_found" if final else "abstained"
    result["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 3)
    return result


def gold_trace(diagnostic: dict, citation_ids: list[str]) -> dict:
    stages = diagnostic.get("stages", {})
    trace = {}
    for citation in citation_ids:
        trace[citation] = {}
        for stage, hits in stages.items():
            matches = [hit for hit in hits if hit["citation_id"] == citation]
            trace[citation][stage] = matches[0] if matches else None
        ordered_stages = ["original_dense_top50", "rewritten_dense_top50", "bm25_top20",
                          "rrf_ranking", "dense_threshold", "domain_filter", "dedup", "final_top_k"]
        present = [stage for stage in ordered_stages if trace[citation].get(stage) is not None]
        trace[citation]["first_seen_stage"] = present[0] if present else None
        if trace[citation].get("final_top_k") is None and present:
            last = present[-1]
            trace[citation]["removed_after_stage"] = last
            trace[citation]["removal_reason"] = "top_k_cutoff" if last == "dedup" else "stage_boundary"
        elif not present:
            trace[citation]["removed_after_stage"] = None
            trace[citation]["removal_reason"] = "not_in_candidate_pool"
    return trace


def oracle_context_prompt(search, citation_ids: list[str], question: str) -> dict:
    hits = []
    for cid in citation_ids:
        node = search.nodes[cid]
        meta = node.metadata
        hits.append({"citation_id": cid, "excerpt": node.text, "source_id": meta["id"],
                     "source_url": meta["source_url"], "source_version": meta["version"],
                     "sha256": meta["sha256"], "page": meta["page"], "anchor": meta.get("anchor")})
    evidence = bounded_evidence(question, hits)
    return {"evidence": evidence, "generation_prompt": generation_prompt(question, evidence),
            "verification_prompt": None,
            "evidence_chars": sum(len(item.get("excerpt", "")) for item in evidence),
            "evidence_count": len(evidence)}


def validate_oracle_generation(search, question: str, evidence: list[dict], generated: str,
                               verified: str | None = None) -> dict:
    result = {"evidence_citation_ids": [item["citation_id"] for item in evidence],
              "generation_raw": generated, "claims": [], "unresolved": [], "citation_valid": False}
    try:
        draft = AnswerDraft.model_validate(json.loads(generated))
    except (ValueError, TypeError):
        result["failure_class"] = "generation_failure"
        return result
    by_id = {item["citation_id"]: item for item in evidence}
    valid = []
    for part in draft.parts:
        if part.claim is None:
            result["unresolved"].append(part.question); continue
        hit = by_id.get(part.citation_id)
        quote = canonical_excerpt(hit["excerpt"], part.excerpt) if hit else None
        if quote is None:
            result["unresolved"].append(part.question); continue
        valid.append((part, hit)); result["claims"].append({"question": part.question, "claim": part.claim,
                                                             "citation_id": part.citation_id, "excerpt": quote})
    result["citation_valid"] = bool(valid)
    if verified is not None:
        result["verification_raw"] = verified
        try:
            result["verdicts"] = [item.model_dump() for item in Verification.model_validate(json.loads(verified)).verdicts]
        except (ValueError, TypeError):
            result["failure_class"] = "citation_validation_failure"
    return result


async def oracle_context_answer(answerer, question: str, evidence: list[dict]) -> dict:
    """Run the existing generation/validation/verification contracts on fixed evidence."""
    started = perf_counter()
    result = {"status": "abstained", "answer": None, "claims": [], "unresolved": [],
              "evidence_citation_ids": [item["citation_id"] for item in evidence],
              "generation_raw": None, "verification_raw": None,
              "generation_ms": 0.0, "verification_ms": 0.0}
    generation_started = perf_counter()
    try:
        generation_input = generation_prompt(question, evidence)
        raw = await answerer._generate(generation_input, answerer.generation_timeout)
        result["generation_raw"] = raw
    except TimeoutError as exc:
        result.update({"failure_class": "provider_runtime_failure", "error_code": "generation_timeout",
                       "error_stage": "generation"})
        result["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    except Exception as exc:
        result.update({"failure_class": "generation_failure", "error_code": type(exc).__name__,
                       "error_stage": "generation"})
        result["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    result["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
    try:
        draft = AnswerDraft.model_validate(json.loads(result["generation_raw"]))
    except (ValueError, TypeError):
        result.update({"failure_class": "generation_failure", "error_code": "invalid_generation",
                       "error_stage": "generation"})
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    by_id = {item["citation_id"]: item for item in evidence}
    valid, unresolved = [], []
    for part in draft.parts:
        if part.claim is None:
            unresolved.append(part.question)
            continue
        hit = by_id.get(part.citation_id)
        quote = canonical_excerpt(hit["excerpt"], part.excerpt) if hit else None
        if quote is None:
            unresolved.append(part.question)
            continue
        part.excerpt = quote
        valid.append((part, hit))
    result["unresolved"] = unresolved
    result["citation_valid_count"] = len(valid)
    if not valid:
        result.update({"failure_class": "citation_validation_failure", "error_code": "no_verified_claims",
                       "error_stage": "citation_validation"})
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    candidates = [{"index": index, "question": part.question, "claim": part.claim,
                   "excerpt": part.excerpt, "source_excerpt": hit["excerpt"]}
                  for index, (part, hit) in enumerate(valid)]
    verification_started = perf_counter()
    try:
        result["verification_raw"] = await answerer._generate(
            verification_prompt(question, candidates), answerer.verification_timeout)
    except TimeoutError:
        result.update({"failure_class": "provider_runtime_failure", "error_code": "verification_timeout",
                       "error_stage": "verification"})
        result["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    except Exception as exc:
        result.update({"failure_class": "generation_failure", "error_code": type(exc).__name__,
                       "error_stage": "verification"})
        result["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    result["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
    try:
        checked = Verification.model_validate(json.loads(result["verification_raw"]))
        verdicts = {item.index: item.supported for item in checked.verdicts}
        if len(checked.verdicts) != len(valid) or set(verdicts) != set(range(len(valid))):
            raise ValueError("missing or duplicate verdict")
    except (ValueError, TypeError):
        result.update({"failure_class": "citation_validation_failure", "error_code": "verification_failed",
                       "error_stage": "verification"})
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    for index, (part, hit) in enumerate(valid):
        if not verdicts[index]:
            unresolved.append(part.question)
            continue
        result["claims"].append({"question": part.question, "text": part.claim,
                                 "citation_id": part.citation_id, "excerpt": part.excerpt,
                                 "source_id": hit["source_id"], "source_url": hit["source_url"],
                                 "source_version": hit["source_version"], "sha256": hit["sha256"],
                                 "page": hit["page"], "anchor": hit["anchor"]})
    result["unresolved"] = unresolved
    if not result["claims"]:
        result.update({"failure_class": "citation_validation_failure", "error_code": "no_supported_claims",
                       "error_stage": "verification"})
        result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
        return result
    result["status"] = "partial" if unresolved else "answered"
    result["answer"] = "\n".join(item["text"] for item in result["claims"])
    result["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
    return result
