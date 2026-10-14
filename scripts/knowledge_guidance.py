"""Bounded model guidance for corpus-scoped, bilingual evidence retrieval."""

from __future__ import annotations

import asyncio
import json
import re
import httpx
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Literal

from llama_index.retrievers.bm25 import BM25Retriever
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.model_adapter import ModelRequest
from scripts.tracing import mark_error, mark_success, set_attribute, set_text, span


POLICY_PATH = Path(__file__).resolve().parents[1] / "metadata/knowledge_guided_policy_v1.json"


from scripts.knowledge_observation import observe, candidate_rows, record_ranking
from scripts.knowledge_runtime import current_runtime, embedding_deadline, stage_budget, record_model_call
from app.execution_events import emit_execution_event


class GuidancePlanError(ValueError):
    pass


def _planner_error_code(exc: Exception) -> str:
    message = str(exc)
    if "authentication_failed" in message:
        return "planner_authentication_failed"
    if "rate_limited" in message:
        return "planner_rate_limited"
    if "provider_server_error" in message:
        return "planner_server_error"
    if "invalid_provider_response" in message:
        return "planner_invalid_provider_response"
    if "provider_connection_error" in message:
        return "planner_connection_error"
    return "planner_provider_error"


class GuidancePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["search", "abstain"]
    intent: Literal["rules", "glossary", "outside", "uncertain"]
    english_query: str | None = Field(default=None, max_length=500)
    question_scope: Literal["single", "multipart"] | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "GuidancePlan":
        if self.action == "search":
            if self.intent not in {"rules", "glossary"} or not self.english_query or not self.english_query.strip():
                raise ValueError("search requires a rules/glossary intent and an English query")
            # Preserve compatibility with older planner responses while keeping
            # the fast path limited to the glossary domain.
            if self.question_scope is None and self.intent == "glossary":
                self.question_scope = "single"
        elif self.intent not in {"outside", "uncertain"} or self.english_query is not None:
            raise ValueError("abstain requires outside/uncertain intent and no query")
        return self


def parse_guidance_plan(raw: str, question: str) -> GuidancePlan:
    try:
        plan = GuidancePlan.model_validate(json.loads(raw))
    except (ValueError, TypeError) as exc:
        raise GuidancePlanError("invalid knowledge guidance plan") from exc
    if plan.action == "search":
        source_numbers = re.findall(r"\d+", question)
        rewritten_numbers = re.findall(r"\d+", plan.english_query or "")
        if any(rewritten_numbers.count(number) < source_numbers.count(number) for number in set(source_numbers)):
            raise GuidancePlanError("English query dropped a number from the question")
    return plan


def build_guidance_prompt(question: str) -> str:
    return (
        "Classify the user's question against exactly two sources: the 2025-26 Official NBA "
        "Playing Rules and the NBA Stats Glossary of metric definitions. Return one JSON object "
        "with exactly action, intent, english_query, question_scope. For a rule question use action=search, "
        "intent=rules; for a metric-definition question use action=search, intent=glossary. "
        "Set question_scope=single for one canonical metric definition, including that metric's "
        "formula, or one factual rule question; use multipart for multiple metrics, independent "
        "conditions, or subquestions. "
        "Write a concise English search query that preserves the full meaning, qualifiers, "
        "names, dates, season numbers, and negation. Do not answer the question. "
        "For player/team statistical facts, rankings, game outcomes, current events, salaries, "
        "other leagues or rulebook editions use action=abstain, intent=outside, english_query=null, question_scope=null. "
        "For a question without enough detail to identify a single rule outcome use "
        "action=abstain, intent=uncertain, english_query=null, question_scope=null. "
        "A question asking what 3P% means is glossary search; a question asking who led "
        "the league in 3P% is outside. Return no prose or extra fields.\n"
        f"Question: {json.dumps(question, ensure_ascii=False)}"
    )


def _hit(node, score: float, dense_score: float) -> dict:
    meta = node.metadata
    return {
        "citation_id": meta["citation_id"], "excerpt": node.text,
        "source_id": meta["id"], "source_url": meta["source_url"],
        "source_version": meta["version"], "sha256": meta["sha256"],
        "page": meta["page"], "heading": meta["heading"], "anchor": meta.get("anchor"),
        "score": round(score, 8), "dense_score": round(dense_score, 8),
    }


class GuidedKnowledgeSearch:
    def __init__(self, search, model, *, min_dense_score: float | None = None,
                 planner_timeout: float = 5.0, embedding_timeout: float = 8.0,
                 tracer=None, observer=None):
        if search.dense is None:
            raise ValueError("guided retrieval requires a dense index")
        if min_dense_score is None:
            policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
            embedding_model = getattr(search.dense.embedding, "model", None)
            if model.model != policy["planner_model"] or embedding_model != policy["embedding_model"]:
                raise ValueError("guided policy model differs from configured model")
            if search.manifest_hash is not None and search.manifest_hash != policy["index_manifest_sha256"]:
                raise ValueError("guided policy corpus differs from configured index")
            min_dense_score = policy["min_dense_score"]
        if not 0 <= min_dense_score <= 1:
            raise ValueError("guided threshold must be between zero and one")
        self.search_index = search
        self.nodes = search.nodes
        self.model = model
        self.min_dense_score = min_dense_score
        self.planner_timeout = planner_timeout
        self.embedding_timeout = embedding_timeout
        self.tracer = tracer
        self.observer = observer
        self.bm25 = BM25Retriever.from_defaults(
            nodes=list(search.nodes.values()), similarity_top_k=min(20, len(search.nodes)),
        )

    def _abstained(self, reason_code: str, *, rewritten_query: str | None,
                   planning_ms: float, started: float) -> dict:
        return {
            "mode": "guided", "retrieval_status": "abstained", "reason_code": reason_code,
            "rewritten_query": rewritten_query, "planner_model": str(self.model.model),
            "intent": None, "question_scope": None,
            "scope_recovery": None,
            "score_type": "rrf", "hits": [],
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "planning_ms": round(planning_ms, 3), "embedding_ms": 0.0, "vector_query_ms": 0.0,
        }

    def _rank(self, english_query: str, intent: str, vectors: list[list[float]],
              *, question_scope: str | None = None, top_k: int, planning_ms: float,
              embedding_ms: float, started: float, scope_recovery: str | None = None) -> dict:
        dense = self.search_index.dense
        vector_query_ms = 0.0
        dense_lists = []
        dense_scores: dict[str, float] = {}
        for query_index, vector in enumerate(vectors):
            ranked = dense.retrieve_vector(vector, min(50, len(self.search_index.nodes)))
            vector_query_ms += dense.last_vector_query_ms
            dense_lists.append(ranked)
            if self.observer is not None:
                observe(self.observer, "original_dense_top50" if query_index == 0 else "rewritten_dense_top50",
                        candidate_rows(ranked, "dense"))
            for item in ranked:
                citation_id = item.node.metadata["citation_id"]
                dense_scores[citation_id] = max(dense_scores.get(citation_id, 0.0), item.score or 0.0)
        lexical = self.bm25.retrieve(english_query)
        if self.observer is not None:
            observe(self.observer, "bm25_top20", candidate_rows(lexical, "bm25"))
        ranked_lists = dense_lists + [[item for item in lexical if (item.score or 0.0) > 0]]
        # Deduplicate each ranked list by source location before RRF. Chunking is
        # an evidence representation detail; it must not give a long page more
        # votes than a short page containing the answer.
        dense_location_scores: dict[tuple, float] = {}
        lexical_location_scores: dict[tuple, float] = {}
        location_best: dict[tuple, str] = {}
        scores: dict[str, float] = {}
        dense_candidate_ids = set(dense_scores)
        list_weights = [1.0] + [0.5] * (len(dense_lists) - 1) + [0.25]
        for list_index, ranked in enumerate(ranked_lists):
            seen_locations: set[tuple] = set()
            for rank, item in enumerate(ranked, 1):
                meta = item.node.metadata
                citation_id = meta["citation_id"]
                location = (meta["id"], meta["page"] if meta["page"] is not None else meta.get("anchor"))
                if location in seen_locations:
                    continue
                seen_locations.add(location)
                contribution = list_weights[list_index] / (60 + rank)
                target_scores = (lexical_location_scores if list_index == len(ranked_lists) - 1
                                 else dense_location_scores)
                # Original and translated dense queries describe the same user
                # intent. Keep the strongest rank for that location instead of
                # counting query variants as independent votes.
                target_scores[location] = max(target_scores.get(location, 0.0), contribution)
                best = location_best.get(location)
                if citation_id in dense_candidate_ids and (
                        best is None or best not in dense_candidate_ids
                        or dense_scores[citation_id] > dense_scores.get(best, 0.0)):
                    location_best[location] = citation_id
                scores[citation_id] = max(scores.get(citation_id, 0.0),
                                          dense_location_scores.get(location, 0.0)
                                          + lexical_location_scores.get(location, 0.0))
        ranked_ids = sorted(
            (citation_id for citation_id in scores
             if citation_id in dense_candidate_ids and dense_scores[citation_id] >= self.min_dense_score),
            key=lambda citation_id: (-scores[citation_id], -dense_scores[citation_id], citation_id),
        )
        # Choose the strongest dense chunk as the citation representative for
        # every retained location. A lexical-only chunk must never replace the
        # dense hit that made that location eligible.
        for location in set(dense_location_scores) | set(lexical_location_scores):
            candidates = [cid for cid in dense_candidate_ids
                          if (self.search_index.nodes[cid].metadata["id"],
                              self.search_index.nodes[cid].metadata["page"]
                              if self.search_index.nodes[cid].metadata["page"] is not None
                              else self.search_index.nodes[cid].metadata.get("anchor")) == location]
            if candidates:
                location_best[location] = max(candidates,
                                              key=lambda cid: (dense_scores[cid], scores.get(cid, 0.0), cid))
        # Fuse at the source location before applying top_k. A page/entry may be
        # split into several chunks; summing their independent rank contributions
        # prevents one location's context from losing to many unrelated locations.
        filtered_locations: set[tuple] = set()
        for citation_id in ranked_ids:
            meta = self.search_index.nodes[citation_id].metadata
            location = (meta["id"], meta["page"] if meta["page"] is not None else meta.get("anchor"))
            filtered_locations.add(location)
        location_scores = {
            location: dense_location_scores.get(location, 0.0)
            + 0.25 * lexical_location_scores.get(location, 0.0)
            for location in filtered_locations
        }
        for location, citation_id in location_best.items():
            if location in location_scores:
                scores[citation_id] = location_scores[location]
        ranked_locations = sorted(filtered_locations,
                                  key=lambda location: (-dense_scores[location_best[location]],
                                                        -location_scores[location], location))
        hits = []
        for location in ranked_locations:
            citation_id = location_best[location]
            node = self.search_index.nodes[citation_id]
            meta = node.metadata
            if (intent == "rules" and meta["page"] is None) or (intent == "glossary" and meta["anchor"] is None):
                continue
            hits.append(_hit(node, scores[citation_id], dense_scores[citation_id]))
            if len(hits) == top_k:
                break
        record_ranking(self.observer, self.search_index.nodes, scores, dense_scores,
                       ranked_ids, intent, hits)
        return {
            "mode": "guided", "retrieval_status": "candidates_found" if hits else "abstained",
            "reason_code": None if hits else "low_relevance", "rewritten_query": english_query,
            "planner_model": str(self.model.model), "intent": intent,
            "question_scope": question_scope, "score_type": "rrf", "hits": hits,
            "scope_recovery": scope_recovery,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "planning_ms": round(planning_ms, 3), "embedding_ms": round(embedding_ms, 3),
            "vector_query_ms": round(vector_query_ms, 3),
        }

    def search(self, question: str, *, top_k: int = 5) -> dict:
        started = perf_counter()
        response = self.model.generate(ModelRequest(
            prompt=build_guidance_prompt(question), response_format="json_object",
        ))
        planning_ms = (perf_counter() - started) * 1000
        plan = parse_guidance_plan(response.text, question)
        observe(self.observer, "planner", plan.model_dump())
        if plan.action == "abstain":
            reason = "outside_corpus_scope" if plan.intent == "outside" else "uncertain_intent"
            return self._abstained(reason, rewritten_query=None, planning_ms=planning_ms, started=started)
        retrieval_query = plan.english_query.strip()
        queries = list(dict.fromkeys([question, retrieval_query]))
        question_scope = plan.question_scope or ("single" if plan.intent == "glossary" else None)
        observe(self.observer, "embedding_queries", queries)
        embedding_started = perf_counter()
        vectors = self.search_index.dense.embedding.embed(queries)
        embedding_ms = (perf_counter() - embedding_started) * 1000
        return self._rank(retrieval_query, plan.intent, vectors,
                          question_scope=question_scope, top_k=top_k,
                          planning_ms=planning_ms, embedding_ms=embedding_ms, started=started)

    async def asearch(self, question: str, *, top_k: int = 5) -> dict:
        with span(self.tracer, "nba.knowledge.retrieval", "RETRIEVER") as retrieval_span:
            set_text(retrieval_span, "input.value", question)
            set_attribute(retrieval_span, "input.mime_type", "text/plain")
            set_attribute(retrieval_span, "retrieval.top_k", top_k)
            try:
                result = await self._asearch(question, top_k=top_k)
                set_attribute(retrieval_span, "retrieval.status", result.get("retrieval_status"))
                set_attribute(retrieval_span, "retrieval.reason_code", result.get("reason_code"))
                set_text(retrieval_span, "retrieval.rewritten_query", result.get("rewritten_query") or "")
                set_attribute(retrieval_span, "retrieval.citation_ids",
                              [hit.get("citation_id") for hit in result.get("hits", [])])
                set_attribute(retrieval_span, "retrieval.scores",
                              [hit.get("score") for hit in result.get("hits", [])])
                mark_success(retrieval_span)
                return result
            except Exception as exc:
                mark_error(retrieval_span, exc)
                raise

    async def _asearch(self, question: str, *, top_k: int = 5) -> dict:
        started = perf_counter()
        planner_started = started
        emit_execution_event("planning", "running", "查询规划")
        planner_budget = stage_budget(self.planner_timeout)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-knowledge-model")
        planner_prompt = build_guidance_prompt(question)
        with span(self.tracer, "nba.knowledge.planner", "LLM") as planner_span:
            set_text(planner_span, "input.value", planner_prompt)
            set_attribute(planner_span, "input.mime_type", "text/plain")
            set_text(planner_span, "nba.knowledge.original_question", question)
            set_attribute(planner_span, "llm.model_name", self.model.model)
            set_attribute(planner_span, "llm.provider", getattr(self.model, "base_url", ""))
            future = executor.submit(self.model.generate, ModelRequest(
                prompt=planner_prompt, response_format="json_object",
                timeout=planner_budget,
            ))
            try:
                while not future.done():
                    remaining = planner_budget - (perf_counter() - planner_started)
                    if remaining <= 0:
                        set_attribute(planner_span, "error.type", "TimeoutError")
                        emit_execution_event("planning", "failed", "查询规划失败", metadata={"reason_code": "planner_timeout"})
                        return self._stage_error("planner", "planner_timeout", started,
                                                 planner_started=planner_started)
                    await asyncio.sleep(min(0.01, remaining))
                response = future.result()
                metadata = record_model_call("planner", self.model, response,
                                             timeout=planner_budget, prompt_chars=len(planner_prompt))
                observe(self.observer, "planner_usage", metadata)
                set_text(planner_span, "output.value", response.text)
                set_attribute(planner_span, "output.mime_type", "application/json")
                set_attribute(planner_span, "llm.finish_reason", getattr(response, "finish_reason", None) or "")
                if response.usage:
                    for key, value in response.usage.items():
                        if isinstance(value, (int, float)):
                            set_attribute(planner_span, f"llm.usage.{key}", value)
                mark_success(planner_span)
            except GuidancePlanError:
                set_attribute(planner_span, "error.type", "GuidancePlanError")
                mark_error(planner_span, GuidancePlanError("invalid knowledge guidance plan"), reason_code="invalid_plan")
                emit_execution_event("planning", "failed", "查询规划失败", metadata={"reason_code": "invalid_plan"})
                return self._stage_error("planner", "invalid_plan", started,
                                         planner_started=planner_started)
            except TimeoutError:
                set_attribute(planner_span, "error.type", "TimeoutError")
                mark_error(planner_span, TimeoutError("planner timeout"), reason_code="planner_timeout")
                emit_execution_event("planning", "failed", "查询规划失败", metadata={"reason_code": "planner_timeout"})
                return self._stage_error("planner", "planner_timeout", started,
                                         planner_started=planner_started)
            except Exception as exc:
                set_attribute(planner_span, "error.type", type(exc).__name__)
                reason_code = "planner_timeout" if isinstance(exc, TimeoutError) else _planner_error_code(exc)
                set_attribute(planner_span, "error.detail", str(exc)[:500])
                mark_error(planner_span, exc, reason_code=reason_code)
                emit_execution_event("planning", "failed", "查询规划失败", metadata={"reason_code": reason_code})
                return self._stage_error("planner", reason_code, started,
                                         planner_started=planner_started)
            finally:
                runtime = current_runtime.get()
                if runtime is not None:
                    runtime.timings["planning_ms"] = round((perf_counter() - planner_started) * 1000, 3)
                future.cancel()
                executor.shutdown(wait=future.done(), cancel_futures=True)
        observe(self.observer, "planner_response", {"raw": response.text})
        planning_ms = (perf_counter() - planner_started) * 1000
        try:
            plan = parse_guidance_plan(response.text, question)
        except GuidancePlanError:
            return self._stage_error("planner", "invalid_plan", started,
                                     planner_started=planner_started)
        observe(self.observer, "planner", plan.model_dump())
        emit_execution_event("planning", "complete", "查询规划完成", elapsed_ms=round(planning_ms, 3),
                             metadata={"intent": plan.intent, "action": plan.action})
        if plan.action == "abstain":
            recovered = await self._recover_abstained_scope(
                question, plan.intent, top_k=top_k, planning_ms=planning_ms, started=started,
            )
            if recovered is not None:
                return recovered
            reason = "outside_corpus_scope" if plan.intent == "outside" else "uncertain_intent"
            return self._abstained(reason, rewritten_query=None, planning_ms=planning_ms, started=started)
        retrieval_query = plan.english_query.strip()
        queries = list(dict.fromkeys([question, retrieval_query]))
        question_scope = plan.question_scope or ("single" if plan.intent == "glossary" else None)
        observe(self.observer, "embedding_queries", queries)
        embedding_started = perf_counter()
        embedding_budget = stage_budget(self.embedding_timeout)
        embedding_token = embedding_deadline.set(embedding_started + embedding_budget)
        with span(self.tracer, "nba.knowledge.embedding", "EMBEDDING") as embedding_span:
            set_text(embedding_span, "input.value", queries)
            set_attribute(embedding_span, "input.mime_type", "application/json")
            set_attribute(embedding_span, "embedding.query_labels", ["original_query", "rewritten_query"])
            set_attribute(embedding_span, "embedding.input_count", len(queries))
            set_attribute(embedding_span, "embedding.model_name",
                          getattr(self.search_index.dense.embedding, "model", None))
            try:
                vectors = await asyncio.wait_for(
                    self.search_index.dense.embedding.aembed(queries), timeout=embedding_budget,
                )
                set_attribute(embedding_span, "output.vector_count", len(vectors))
                set_attribute(embedding_span, "output.vector_dimension", len(vectors[0]) if vectors else 0)
                mark_success(embedding_span)
            except (asyncio.TimeoutError, httpx.TimeoutException):
                set_attribute(embedding_span, "error.type", "TimeoutError")
                mark_error(embedding_span, TimeoutError("embedding timeout"), reason_code="embedding_timeout")
                return self._stage_error("embedding", "embedding_timeout", started,
                                         planning_ms=planning_ms,
                                         embedding_started=embedding_started)
            except Exception as exc:
                set_attribute(embedding_span, "error.type", type(exc).__name__)
                mark_error(embedding_span, exc, reason_code="embedding_error")
                return self._stage_error("embedding", "embedding_error", started,
                                         planning_ms=planning_ms,
                                         embedding_started=embedding_started)
            finally:
                embedding_deadline.reset(embedding_token)
                runtime = current_runtime.get()
                if runtime is not None:
                    runtime.timings["embedding_ms"] = round((perf_counter() - embedding_started) * 1000, 3)
        embedding_ms = (perf_counter() - embedding_started) * 1000
        with span(self.tracer, "nba.knowledge.ranking", "RETRIEVER") as ranking_span:
            set_text(ranking_span, "input.query", retrieval_query)
            set_text(ranking_span, "retrieval.original_query", question)
            set_attribute(ranking_span, "retrieval.intent", plan.intent)
            set_attribute(ranking_span, "retrieval.vector_query_count", len(vectors))
            set_attribute(ranking_span, "retrieval.fusion_method", "rrf")
            set_attribute(ranking_span, "retrieval.rrf_k", 60)
            set_attribute(ranking_span, "retrieval.top_k", top_k)
            result = self._rank(retrieval_query, plan.intent, vectors, question_scope=question_scope, top_k=top_k,
                                planning_ms=planning_ms, embedding_ms=embedding_ms, started=started)
            set_attribute(ranking_span, "retrieval.citation_ids",
                          [hit.get("citation_id") for hit in result.get("hits", [])])
            set_attribute(ranking_span, "retrieval.dense_scores",
                          [hit.get("dense_score") for hit in result.get("hits", [])])
            set_attribute(ranking_span, "retrieval.final_candidate_count", len(result.get("hits", [])))
            mark_success(ranking_span)
            return result

    async def _recover_abstained_scope(self, question: str, planner_intent: str, *, top_k: int,
                                       planning_ms: float, started: float) -> dict | None:
        """Recover an in-domain request without trusting an unsupported planner abstention.

        The probe uses the existing dense index and accepts a scope only when the
        thresholded candidates agree on rules versus glossary metadata. It does
        not infer scope from user vocabulary or add a query-specific branch.
        """
        embedding_started = perf_counter()
        embedding_budget = stage_budget(self.embedding_timeout)
        embedding_token = embedding_deadline.set(embedding_started + embedding_budget)
        try:
            vectors = await asyncio.wait_for(
                self.search_index.dense.embedding.aembed([question]), timeout=embedding_budget,
            )
        except (asyncio.TimeoutError, httpx.TimeoutException):
            return None
        except Exception:
            return None
        finally:
            embedding_deadline.reset(embedding_token)
        if not vectors:
            return None
        candidates = self.search_index.dense.retrieve_vector(
            vectors[0], min(50, len(self.search_index.nodes)),
        )
        domain_scores = {"rules": 0.0, "glossary": 0.0}
        for item in candidates:
            score = float(item.score or 0.0)
            if score < self.min_dense_score:
                continue
            metadata = item.node.metadata
            if metadata.get("page") is not None:
                domain_scores["rules"] = max(domain_scores["rules"], score)
            elif metadata.get("anchor") is not None:
                domain_scores["glossary"] = max(domain_scores["glossary"], score)
        if not any(domain_scores.values()):
            return None
        intent, best = max(domain_scores.items(), key=lambda item: item[1])
        other = domain_scores["glossary" if intent == "rules" else "rules"]
        if best <= 0 or (other > 0 and best == other):
            return None
        embedding_ms = (perf_counter() - embedding_started) * 1000
        result = self._rank(
            question, intent, vectors, question_scope="single", top_k=top_k,
            planning_ms=planning_ms, embedding_ms=embedding_ms, started=started,
            scope_recovery="corpus_probe",
        )
        result["planner_intent"] = planner_intent
        observe(self.observer, "scope_recovery", {
            "planner_intent": planner_intent, "intent": intent,
            "domain_scores": domain_scores,
        })
        return result

    def _stage_error(self, stage: str, reason_code: str, started: float, *,
                     planner_started: float | None = None,
                     planning_ms: float | None = None,
                     embedding_started: float | None = None) -> dict:
        planner_ms = (planning_ms if planning_ms is not None else
                      ((perf_counter() - planner_started) * 1000
                       if planner_started is not None else 0.0))
        embedding_ms = ((perf_counter() - embedding_started) * 1000
                        if embedding_started is not None else 0.0)
        return {
            "mode": "guided", "retrieval_status": "error", "reason_code": reason_code,
            "error_stage": stage, "rewritten_query": None,
            "planner_model": str(self.model.model), "intent": None, "question_scope": None,
            "score_type": "rrf", "hits": [],
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "planning_ms": round(planner_ms, 3), "planner_ms": round(planner_ms, 3),
            "embedding_ms": round(embedding_ms, 3), "vector_query_ms": 0.0,
        }
