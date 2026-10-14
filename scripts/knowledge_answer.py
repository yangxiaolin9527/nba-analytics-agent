"""Evidence-bound answers over the pinned knowledge corpus."""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.model_adapter import ModelRequest
from scripts.answer_language import AnswerLanguage
from scripts.tracing import mark_error, mark_success, set_attribute, set_text, span, trace_id


from scripts.knowledge_observation import observe
from app.execution_events import emit_execution_event
from scripts.knowledge_runtime import (
    KnowledgeRuntime, current_runtime, record_model_call, stage_budget, total_expired,
)


class KnowledgeAnswerError(RuntimeError):
    """A bounded answer-pipeline failure with a stable public reason code."""

    def __init__(self, stage: str, reason_code: str, *, retryable: bool,
                 timings: dict[str, float] | None = None) -> None:
        super().__init__(reason_code)
        self.stage = stage
        self.reason_code = reason_code
        self.retryable = retryable
        self.timings = timings or {}


def _model_error_code(stage: str, exc: Exception) -> str:
    message = str(exc)
    if "authentication_failed" in message:
        return f"{stage}_authentication_failed"
    if "rate_limited" in message:
        return f"{stage}_rate_limited"
    if "provider_server_error" in message:
        return f"{stage}_server_error"
    if "invalid_provider_response" in message:
        return f"{stage}_invalid_provider_response"
    if "provider_connection_error" in message:
        return f"{stage}_connection_error"
    return f"{stage}_error"


class AnswerPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    claim: str | None = Field(max_length=1000)
    citation_id: str | None = None
    excerpt: str | None = Field(max_length=1500)

    @model_validator(mode="after")
    def complete_claim(self):
        if bool(self.claim) != bool(self.citation_id) or bool(self.claim) != bool(self.excerpt):
            raise ValueError("a claim requires a citation ID and excerpt")
        return self


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[AnswerPart] = Field(min_length=1, max_length=6)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    supported: bool


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict]


def canonical_excerpt(source: str, quote: str) -> str | None:
    if quote in source:
        return quote
    parts = re.split(r"\s+", quote.strip())
    if not parts or not all(parts):
        return None
    match = re.search(r"\s+".join(re.escape(part) for part in parts), source)
    return match.group() if match else None


def supplement_hits(search, guided_hits: list[dict], english_query: str) -> list[dict]:
    if not guided_hits or guided_hits[0].get("anchor") is None:
        return guided_hits
    seen = {hit["citation_id"] for hit in guided_hits}
    hits = list(guided_hits)
    normalized_query = re.sub(r"\s+", " ", english_query.lower())
    for node in getattr(search, "nodes", {}).values():
        meta = node.metadata
        heading = re.sub(r"\s+", " ", (meta.get("heading") or "").lower())
        anchor = meta.get("anchor") or ""
        if not anchor or not heading or not (
            re.search(rf"\b{re.escape(heading)}\b", normalized_query)
            or re.search(rf"\b{re.escape(anchor.lower())}\b", normalized_query)
        ):
            continue
        citation_id = meta["citation_id"]
        if citation_id in seen:
            continue
        hits.append({"citation_id": citation_id, "excerpt": node.text,
                     "source_id": meta["id"], "source_url": meta["source_url"],
                     "source_version": meta["version"], "sha256": meta["sha256"],
                     "page": meta["page"], "heading": meta["heading"], "anchor": anchor})
        seen.add(citation_id)
        if len(hits) >= 8:
            return hits
    for hit in search.search(english_query, mode="bm25", top_k=5)["hits"]:
        if hit.get("anchor") is not None and hit["citation_id"] not in seen:
            hits.append(hit)
            seen.add(hit["citation_id"])
            if len(hits) >= 8:
                break
    return hits


def expand_evidence(search, hits: list[dict], *, max_pages: int = 5, max_chunks: int = 12,
                    max_chars: int = 24000) -> list[dict]:
    """Keep ranked hits, then add original chunks from their highest-ranked rule pages."""
    pages = list(dict.fromkeys((hit["source_id"], hit["page"]) for hit in hits if hit.get("page") is not None))[:max_pages]
    allowed = set(pages)
    selected = []
    seen = set()
    selected_chars = 0

    def append(node):
        nonlocal selected_chars
        meta = node.metadata
        citation_id = meta["citation_id"]
        if citation_id in seen or len(selected) >= max_chunks:
            return False
        text = node.text
        if selected_chars + len(text) > max_chars:
            return False
        selected.append({"citation_id": citation_id, "excerpt": node.text,
                         "source_id": meta["id"], "source_url": meta["source_url"],
                         "source_version": meta["version"], "sha256": meta["sha256"],
                         "page": meta["page"], "heading": meta["heading"],
                         "anchor": meta.get("anchor")})
        seen.add(citation_id)
        selected_chars += len(text)
        return True

    for hit in hits:
        if hit.get("page") is not None and (hit["source_id"], hit["page"]) not in allowed:
            continue
        node = search.nodes.get(hit["citation_id"])
        if node is not None and node.metadata["id"] == hit["source_id"] and node.metadata["page"] == hit.get("page"):
            append(node)
    # Round-robin keeps location coverage while adding nearby chunks.
    page_nodes = [[node for node in search.nodes.values()
                   if (node.metadata["id"], node.metadata["page"]) == page] for page in pages]
    for index in range(max((len(group) for group in page_nodes), default=0)):
        for group in page_nodes:
            if index < len(group):
                append(group[index])
    return selected


_ANSWER_SCOPE_POLICY = (
    "Answer requested subquestions first. Allow brief, directly relevant explanatory context; "
    "respect explicit scope and brevity instructions. Keep conditions and exceptions needed for "
    "accuracy. Optional context must not replace requested answers or introduce unrelated details. "
)


def generation_prompt(question: str, evidence: list[dict], *, strategy: str = 'baseline') -> str:
    excerpts = [{"citation_id": hit["citation_id"], "excerpt": hit["excerpt"]} for hit in evidence]
    language = AnswerLanguage.from_question(question)
    language_instruction = (
        f"Answer prose language: {'Chinese' if language.expected == 'zh' else 'English'}. "
        "This applies to claim, not verbatim excerpt, requested source text, names or formulas. "
        if language.expected else ''
    )
    if strategy == 'concise':
        return (
            "Extract an evidence-bound answer to this NBA question. Use ONLY the supplied English "
            "2025-26 NBA rules or NBA Stats Glossary excerpts. For each subquestion, locate a single "
            "VERBATIM source passage covering the facts in the proposed claim. "
            f"{_ANSWER_SCOPE_POLICY}"
            "Copy it without translation into excerpt, use its citation_id, and express only supported "
            "facts as one concise claim in the user's language. Split independently supported facts "
            "into separate parts and leave an unsupported part unresolved. Preserve every subquestion. "
            "If a requested fact is unsupported, set claim, citation_id and excerpt to null. Do not invent "
            "statistics, outcomes or facts "
            "about other seasons or leagues. If rules and glossary are mixed, leave all parts null. "
            "Return only JSON: {\"parts\":[{\"question\":\"...\",\"claim\":\"...\","
            "\"citation_id\":\"...\",\"excerpt\":\"...\"}]}.\n"
            f"{language_instruction}Question: {json.dumps(question, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(excerpts, ensure_ascii=False)}"
        )
    if strategy != 'baseline':
        raise ValueError('generation_prompt_strategy must be baseline or concise')
    return (
        "Answer the user's question in its language using ONLY these English source excerpts. "
        "The sources cover the 2025-26 NBA rules and NBA Stats Glossary definitions. "
        "Handle multiple subquestions only within one source domain. "
        f"{_ANSWER_SCOPE_POLICY}"
        "For each subquestion, make "
        "a concise claim containing only facts supported by one cited passage, or set claim, citation_id "
        "and excerpt to null. Split independently supported facts into separate parts and leave the "
        "unsupported part unresolved. "
        "Do not infer player statistics, game outcomes, other seasons or leagues. "
        "Use a citation whose source names the exact metric or rule condition asked about. "
        "Quote a VERBATIM substring covering every independent factual clause in that claim. When explaining a "
        "formula, quote the definition and formula only when the question asks for both. "
        "Do not translate the quote. "
        "Return JSON only: {\"parts\":[{\"question\":\"...\",\"claim\":\"...\","
        "\"citation_id\":\"...\",\"excerpt\":\"...\"}]}. "
        "If the question mixes rules and glossary, mark all parts unresolved. "
        "Keep every subquestion, including unsupported ones.\n"
        f"{language_instruction}Question: {json.dumps(question, ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(excerpts, ensure_ascii=False)}"
    )


def bounded_evidence(question: str, evidence: list[dict], *, max_prompt_chars: int = 24000) -> list[dict]:
    selected = []
    for hit in evidence:
        candidate = [*selected, hit]
        if len(generation_prompt(question, candidate)) <= max_prompt_chars:
            selected.append(hit)
        else:
            break
    return selected


def verification_prompt(question: str, claims: list[dict]) -> str:
    return (
        "Independently check each proposed answer. A claim is supported only if its exact factual "
        "meaning, qualifiers and scope follow from the cited excerpt itself, and it contributes to "
        "answering the original question within the allowed scope. A claim may answer part of a "
        "subquestion or provide permitted explanatory context; it need not answer the whole "
        "subquestion alone. Assess relevance against the original question, not only the proposed "
        "subquestion. "
        f"{_ANSWER_SCOPE_POLICY}"
        "The full source excerpt provides context for qualifiers, but the "
        "cited excerpt itself must support every independent factual clause; do not use omitted source "
        "context to approve an extra penalty, condition, or result. "
        "Reject unsupported additions, irrelevant context, violations of explicit scope instructions, "
        "overgeneralization, wrong conditions, or an answer that names a metric "
        "which the cited source does not define. A rule giving a category of eligible players "
        "does not identify a specific named player. "
        "Return JSON only: {\"verdicts\":[{\"index\":0,\"supported\":true}]}, "
        "one verdict for every indexed claim.\n"
        f"Original question: {json.dumps(question, ensure_ascii=False)}\n"
        f"Claims: {json.dumps(claims, ensure_ascii=False)}"
    )


class KnowledgeAnswer:
    def __init__(self, search, model, *, guided=None, retrieval_timeout=7.0,
                 generation_timeout=6.0, verification_timeout=6.0, total_timeout=20.0,
                 tracer=None, observer=None, generation_thinking=None, verification_thinking=None,
                 generation_prompt_strategy='baseline',
                 generation_policy='fixed', candidate_top_k=10):
        if search.dense is None:
            raise ValueError("knowledge answer requires a dense index")
        if generation_thinking not in {None, "enabled", "disabled"}:
            raise ValueError("generation_thinking must be enabled or disabled")
        if generation_prompt_strategy not in {'baseline', 'concise'}:
            raise ValueError('generation_prompt_strategy must be baseline or concise')
        if generation_policy not in {'fixed', 'adaptive'}:
            raise ValueError('generation_policy must be fixed or adaptive')
        if not isinstance(candidate_top_k, int) or not 1 <= candidate_top_k <= 50:
            raise ValueError('candidate_top_k must be between 1 and 50')
        if guided is None:
            from scripts.knowledge_guidance import GuidedKnowledgeSearch
            guided = GuidedKnowledgeSearch(search, model, observer=observer)
        self.search = search
        self.model = model
        self.guided = guided
        self.retrieval_timeout = retrieval_timeout
        self.generation_timeout = generation_timeout
        self.verification_timeout = verification_timeout
        self.total_timeout = total_timeout
        self.tracer = tracer
        self.observer = observer
        self.generation_thinking = generation_thinking
        self.verification_thinking = verification_thinking
        self.generation_prompt_strategy = generation_prompt_strategy
        self.generation_policy = generation_policy
        self.candidate_top_k = candidate_top_k

    def _generation_options(self, retrieval: dict, evidence: list[dict]) -> tuple[str, str | None, str]:
        """Select the low-risk policy only from explicit planner and evidence signals."""
        strategy = self.generation_prompt_strategy
        thinking = self.generation_thinking
        policy = 'fixed'
        glossary_only = bool(evidence) and all(
            hit.get('source_id') == 'nba-stats-glossary' and hit.get('page') is None
            for hit in evidence
        )
        if (self.generation_policy == 'adaptive'
                and retrieval.get('intent') == 'glossary'
                and retrieval.get('question_scope') == 'single'
                and glossary_only):
            strategy, thinking, policy = 'concise', 'disabled', 'adaptive_glossary'
        return strategy, thinking, policy

    async def _generate(self, prompt: str, timeout: float, *, thinking=None) -> str:
        runtime = current_runtime.get()
        if runtime is not None:
            record_model_call(runtime.stage, self.model, timeout=timeout, prompt_chars=len(prompt), status='running')
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-knowledge-answer")
        future = executor.submit(
            self.model.generate,
            ModelRequest(prompt=prompt, response_format="json_object", timeout=timeout,
                         thinking=thinking),
        )
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            while not future.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("knowledge answer model call timed out")
                await asyncio.sleep(min(0.01, remaining))
            response = future.result()
            runtime = current_runtime.get()
            if runtime is not None:
                record_model_call(runtime.stage, self.model, response,
                                  timeout=timeout, prompt_chars=len(prompt))
        except BaseException:
            if runtime is not None:
                record_model_call(runtime.stage, self.model, timeout=timeout, prompt_chars=len(prompt), status='failed')
            raise
        finally:
            future.cancel()
            executor.shutdown(wait=future.done(), cancel_futures=True)
        return response.text

    async def aanswer(self, question: str) -> dict:
        started = perf_counter()
        runtime = KnowledgeRuntime(deadline=started + self.total_timeout, started=started)
        token = current_runtime.set(runtime)
        try:
            return await self._bounded_answer(question, runtime)
        finally:
            current_runtime.reset(token)

    async def _bounded_answer(self, question, runtime):
        with span(self.tracer, "nba.knowledge.answer", "CHAIN") as answer_span:
            set_text(answer_span, "input.value", question)
            set_attribute(answer_span, "input.mime_type", "text/plain")
            set_attribute(answer_span, "llm.model_name", getattr(self.model, "model", None))
            try:
                result = await asyncio.wait_for(self._aanswer(question), timeout=stage_budget(self.total_timeout))
            except asyncio.TimeoutError as exc:
                runtime.fail_active()
                runtime.emit_skipped_stages()
                emit_execution_event("result", "failed", "知识回答失败", metadata={"reason_code": "total_timeout"})
                error = KnowledgeAnswerError(
                    "total", "total_timeout", retryable=True,
                    timings=runtime.snapshot_timings(),
                )
                error.model_calls = dict(runtime.model_calls)
                error.stage_statuses = dict(runtime.stage_statuses)
                observe(self.observer, "answer_error", {"error_type": type(error).__name__,
                        "error_code": error.reason_code, "error_stage": error.stage,
                        "timings": error.timings})
                set_attribute(answer_span, "error.stage", error.stage)
                set_attribute(answer_span, "error.reason_code", error.reason_code)
                set_text(answer_span, "output.value", {
                    "status": "failed", "reason_code": error.reason_code,
                    "error_stage": error.stage, "timings": error.timings,
                })
                set_attribute(answer_span, "output.mime_type", "application/json")
                error.trace_id = trace_id() if self.tracer is not None else None
                raise error from exc
            except asyncio.CancelledError:
                observe(self.observer, "answer_error", {"error_type": "CancelledError"})
                raise
            except KnowledgeAnswerError as exc:
                runtime.fail_active()
                runtime.emit_skipped_stages()
                emit_execution_event("result", "failed", "知识回答失败", metadata={"reason_code": exc.reason_code})
                exc.timings = {**runtime.snapshot_timings(), **exc.timings,
                               "total_ms": round((perf_counter() - runtime.started) * 1000, 3)}
                exc.model_calls = dict(runtime.model_calls)
                exc.stage_statuses = dict(runtime.stage_statuses)
                observe(self.observer, "answer_error", {"error_type": type(exc).__name__,
                        "error_code": exc.reason_code, "error_stage": exc.stage,
                        "timings": exc.timings})
                set_attribute(answer_span, "error.stage", exc.stage)
                set_attribute(answer_span, "error.reason_code", exc.reason_code)
                set_text(answer_span, "output.value", {
                    "status": "failed", "reason_code": exc.reason_code,
                    "error_stage": exc.stage, "timings": exc.timings,
                })
                set_attribute(answer_span, "output.mime_type", "application/json")
                exc.trace_id = trace_id() if self.tracer is not None else None
                raise
            except Exception as exc:
                observe(self.observer, "answer_error", {"error_type": type(exc).__name__})
                set_attribute(answer_span, "error.type", type(exc).__name__)
                raise
            set_attribute(answer_span, "answer.status", result.get("status"))
            set_attribute(answer_span, "answer.reason_code", result.get("reason_code"))
            set_attribute(answer_span, "answer.citation_ids",
                          [claim.get("citation_id") for claim in result.get("claims", [])])
            set_attribute(answer_span, "answer.unresolved", result.get("unresolved", []))
            set_attribute(answer_span, "answer.evidence_chunk_count", result.get("evidence_chunk_count", 0))
            set_attribute(answer_span, "answer.evidence_char_count", result.get("evidence_char_count", 0))
            result["model_calls"] = dict(runtime.model_calls)
            result["stage_statuses"] = dict(runtime.stage_statuses)
            result["trace_id"] = trace_id() if self.tracer is not None else None
            set_text(answer_span, "output.value", result)
            set_attribute(answer_span, "output.mime_type", "application/json")
            mark_success(answer_span)
            runtime.emit_skipped_stages()
            emit_execution_event("result", "complete", "知识回答完成", metadata={"status": result.get("status")})
            observe(self.observer, "answer_result", result)
            return result

    async def _aanswer(self, question: str) -> dict:
        started = perf_counter()
        runtime = current_runtime.get()
        timings = {"retrieval_ms": 0.0, "planning_ms": 0.0, "embedding_ms": 0.0,
                   "vector_query_ms": 0.0, "generation_ms": 0.0, "verification_ms": 0.0}
        runtime.timings = timings
        evidence_metrics = {"evidence_chunk_count": 0, "evidence_char_count": 0,
                            "evidence_page_count": 0, "evidence_citation_ids": []}
        generation_meta = {
            'generation_policy': 'fixed',
            'generation_thinking': self.generation_thinking,
            'generation_prompt_strategy': self.generation_prompt_strategy,
        }
        language = AnswerLanguage.from_question(question)
        language_check = {'expected': language.expected, 'source': language.source,
                          'kind': 'conservative_script_check', 'rejected_count': 0, 'parts': []}

        def response(status, reason, claims=None, unresolved=None):
            claims = claims or []
            elapsed_ms = round((perf_counter() - started) * 1000, 3)
            return {"status": status, "answer": "\n".join(item["text"] for item in claims) or None,
                    "claims": claims, "unresolved": unresolved or [], "reason_code": reason,
                    "answer_language": language_check,
                    **timings, **evidence_metrics, **generation_meta,
                    "elapsed_ms": elapsed_ms, "total_ms": elapsed_ms}

        retrieval_started = perf_counter()
        runtime.begin("retrieval")
        try:
            result = await asyncio.wait_for(
                self.guided.asearch(question, top_k=self.candidate_top_k),
                timeout=stage_budget(self.retrieval_timeout),
            )
        except asyncio.TimeoutError as exc:
            timings["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 3)
            if total_expired():
                raise KnowledgeAnswerError("total", "total_timeout", retryable=True,
                                           timings=runtime.snapshot_timings()) from exc
            raise KnowledgeAnswerError("retrieval", "retrieval_timeout", retryable=True,
                                       timings=timings) from exc
        timings["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 3)
        observe(self.observer, "retrieval_result", result)
        for key in ("planning_ms", "embedding_ms", "vector_query_ms"):
            timings[key] = float(result.get(key, 0.0) or 0.0)
        if result.get("retrieval_status") == "error":
            stage = result.get("error_stage") or "retrieval"
            retryable = result.get("reason_code", "").endswith("timeout")
            raise KnowledgeAnswerError(stage, result.get("reason_code") or "retrieval_error",
                                       retryable=retryable, timings=timings)
        runtime.finish('retrieval')
        if result["retrieval_status"] != "candidates_found":
            return response("abstained", result.get("reason_code") or "no_evidence", unresolved=[question])
        hits = supplement_hits(self.search, result["hits"], result.get("rewritten_query") or question)
        observe(self.observer, "supplemented_hits", hits)
        emit_execution_event("evidence", "running", "整理相关证据")
        with span(self.tracer, "nba.knowledge.evidence", "RETRIEVER") as evidence_span:
            set_attribute(evidence_span, "input.citation_ids",
                          [hit.get("citation_id") for hit in hits])
            expanded = expand_evidence(self.search, hits)
            observe(self.observer, "expanded_evidence", expanded)
            evidence = bounded_evidence(question, expanded)
            observe(self.observer, "bounded_evidence", evidence)
            set_attribute(evidence_span, "output.citation_ids",
                          [hit.get("citation_id") for hit in evidence])
            set_attribute(evidence_span, "output.pages",
                          [hit.get("page") for hit in evidence])
            evidence_metrics["evidence_chunk_count"] = len(evidence)
            evidence_metrics["evidence_char_count"] = sum(len(hit["excerpt"]) for hit in evidence)
            evidence_metrics["evidence_page_count"] = len({
                (hit.get("source_id"), hit.get("page")) for hit in evidence
            })
            evidence_metrics["evidence_citation_ids"] = [
                hit.get("citation_id") for hit in evidence
            ]
            set_attribute(evidence_span, "evidence.candidate_count", len(hits))
            set_attribute(evidence_span, "evidence.expanded_count", len(expanded))
            set_attribute(evidence_span, "evidence.final_count", len(evidence))
            set_attribute(evidence_span, "evidence.char_count", evidence_metrics["evidence_char_count"])
            set_attribute(evidence_span, "evidence.page_count", evidence_metrics["evidence_page_count"])
            set_attribute(evidence_span, "evidence.max_chars", 24000)
            set_attribute(evidence_span, "evidence.max_pages", 5)
            set_attribute(evidence_span, "evidence.max_chunks", 12)
            mark_success(evidence_span)
        emit_execution_event("evidence", "complete", "整理相关证据完成")
        if not evidence:
            return response("abstained", "no_evidence", unresolved=[question])
        generation_started = perf_counter()
        runtime.begin("generation")
        generation_strategy, generation_thinking, generation_policy = self._generation_options(result, evidence)
        generation_meta.update({
            'generation_policy': generation_policy,
            'generation_thinking': generation_thinking,
            'generation_prompt_strategy': generation_strategy,
        })
        generation_input = generation_prompt(question, evidence, strategy=generation_strategy)
        observe(self.observer, "generation_input", {"prompt": generation_input})
        try:
            with span(self.tracer, "nba.knowledge.generation", "LLM") as generation_span:
                set_text(generation_span, "input.value", generation_input)
                set_attribute(generation_span, "input.mime_type", "text/plain")
                set_attribute(generation_span, "llm.model_name", getattr(self.model, "model", None))
                set_attribute(generation_span, "llm.provider", getattr(self.model, "base_url", ""))
                set_attribute(generation_span, "generation.prompt_strategy", generation_strategy)
                set_attribute(generation_span, "generation.thinking", generation_thinking or "default")
                set_attribute(generation_span, "generation.evidence_count", len(evidence))
                set_attribute(generation_span, "generation.evidence_chars", evidence_metrics["evidence_char_count"])
                options = {"thinking": {"type": "disabled"}} if generation_thinking == "disabled" else {}
                raw = await self._generate(generation_input, stage_budget(self.generation_timeout), **options)
                observe(self.observer, "generation_response", {"raw": raw})
                observe(self.observer, "generation_usage", runtime.model_calls.get("generation"))
                set_text(generation_span, "output.value", raw)
                set_attribute(generation_span, "output.mime_type", "application/json")
                usage = runtime.model_calls.get("generation") or {}
                for key, value in (usage.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        set_attribute(generation_span, f"llm.usage.{key}", value)
                set_attribute(generation_span, "llm.reasoning_tokens", usage.get("reasoning_tokens"))
                set_attribute(generation_span, "llm.finish_reason", usage.get("finish_reason") or "")
                mark_success(generation_span)
        except TimeoutError as exc:
            timings["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
            mark_error(generation_span, exc)
            if total_expired():
                raise KnowledgeAnswerError("total", "total_timeout", retryable=True,
                                           timings=runtime.snapshot_timings()) from exc
            raise KnowledgeAnswerError("generation", "generation_timeout", retryable=True,
                                       timings=timings) from exc
        except Exception as exc:
            timings["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
            mark_error(generation_span, exc)
            raise KnowledgeAnswerError("generation", _model_error_code("generation", exc), retryable=False,
                                       timings=timings) from exc
        timings["generation_ms"] = round((perf_counter() - generation_started) * 1000, 3)
        runtime.finish('generation')
        try:
            draft = AnswerDraft.model_validate(json.loads(raw))
            observe(self.observer, "generation_draft", draft.model_dump())
        except (ValueError, TypeError):
            return response("abstained", "invalid_generation", unresolved=[question])
        by_id = {hit["citation_id"]: hit for hit in evidence}
        valid = []
        unresolved = []
        rejected_language = set()
        with span(self.tracer, 'nba.knowledge.answer_language_validation', 'CHAIN') as language_span:
            for index, part in enumerate(draft.parts):
                if part.claim is None:
                    continue
                # The complete question owns output policy; per-part wording may only
                # authorize literal output for a subquestion the user actually asked.
                literal_question = part.question if part.question in question else question
                status = language.check(part.claim, literal_question, part.excerpt)
                language_check['parts'].append({'index': index, 'status': status})
                if status == 'mismatch':
                    rejected_language.add(index)
            language_check['rejected_count'] = len(rejected_language)
            set_text(language_span, 'output.value', language_check)
            set_attribute(language_span, 'validation.kind', 'conservative_script_check')
            set_attribute(language_span, 'validation.rejected_count', len(rejected_language))
            mark_success(language_span)
        observe(self.observer, 'answer_language_validation', language_check)
        runtime.begin('citation_validation')
        with span(self.tracer, "nba.knowledge.citation_validation", "CHAIN") as validation_span:
            set_attribute(validation_span, "input.part_count", len(draft.parts))
            for index, part in enumerate(draft.parts):
                if index in rejected_language:
                    unresolved.append(part.question)
                    continue
                if part.claim is None:
                    unresolved.append(part.question)
                    continue
                hit = by_id.get(part.citation_id)
                quote = canonical_excerpt(hit["excerpt"], part.excerpt) if hit is not None else None
                if quote is None:
                    unresolved.append(part.question)
                    continue
                part.excerpt = quote
                valid.append((part, hit))
            set_attribute(validation_span, "output.valid_count", len(valid))
            set_attribute(validation_span, "output.unresolved", unresolved)
            set_attribute(validation_span, "validation.kind", "deterministic_citation")
            set_attribute(validation_span, "validation.rejected_count", len(draft.parts) - len(valid))
            mark_success(validation_span)
        timings['citation_validation_ms'] = round((perf_counter() - runtime.stage_started) * 1000, 3)
        runtime.finish('citation_validation')
        observe(self.observer, "citation_validation", {
            "valid_citation_ids": [hit["citation_id"] for _, hit in valid],
            "proposed_claim_count": sum(part.claim is not None for part in draft.parts),
            "rejected_claim_count": sum(part.claim is not None for part in draft.parts) - len(valid),
            "unresolved": unresolved,
        })
        if not valid:
            return response("abstained", 'answer_language_mismatch' if rejected_language else "no_verified_claims",
                            unresolved=unresolved or [question])
        candidates = [{"index": index, "question": part.question, "claim": part.claim,
                       "excerpt": part.excerpt, "source_excerpt": hit["excerpt"]}
                      for index, (part, hit) in enumerate(valid)]
        remaining = stage_budget(self.verification_timeout)
        if remaining <= 0:
            raise KnowledgeAnswerError("total", "total_timeout", retryable=True,
                                       timings=runtime.snapshot_timings())
        observe(self.observer, "verification_input", {"prompt": verification_prompt(question, candidates),
                                                    "candidates": candidates})
        verification_started = perf_counter()
        runtime.begin("verification")
        verification_input = verification_prompt(question, candidates)
        try:
            with span(self.tracer, "nba.knowledge.verification", "LLM") as verification_span:
                set_text(verification_span, "input.value", verification_input)
                set_attribute(verification_span, "input.mime_type", "text/plain")
                set_attribute(verification_span, "llm.model_name", getattr(self.model, "model", None))
                set_attribute(verification_span, "llm.provider", getattr(self.model, "base_url", ""))
                set_attribute(verification_span, "verification.claim_count", len(candidates))
                set_attribute(verification_span, "verification.input_chars", len(verification_input))
                verification_options = ({"thinking": {"type": self.verification_thinking}}
                                        if self.verification_thinking in {"enabled", "disabled"} else {})
                raw = await self._generate(verification_input, remaining, **verification_options)
                observe(self.observer, "verification_response", {"raw": raw})
                observe(self.observer, "verification_usage", runtime.model_calls.get("verification"))
                set_text(verification_span, "output.value", raw)
                set_attribute(verification_span, "output.mime_type", "application/json")
                usage = runtime.model_calls.get("verification") or {}
                for key, value in (usage.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        set_attribute(verification_span, f"llm.usage.{key}", value)
                set_attribute(verification_span, "llm.reasoning_tokens", usage.get("reasoning_tokens"))
                set_attribute(verification_span, "llm.finish_reason", usage.get("finish_reason") or "")
                mark_success(verification_span)
        except TimeoutError as exc:
            timings["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
            mark_error(verification_span, exc)
            if total_expired():
                raise KnowledgeAnswerError("total", "total_timeout", retryable=True,
                                           timings=runtime.snapshot_timings()) from exc
            raise KnowledgeAnswerError("verification", "verification_timeout", retryable=True,
                                       timings=timings) from exc
        except Exception as exc:
            timings["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
            mark_error(verification_span, exc)
            raise KnowledgeAnswerError("verification", _model_error_code("verification", exc), retryable=False,
                                       timings=timings) from exc
        timings["verification_ms"] = round((perf_counter() - verification_started) * 1000, 3)
        runtime.finish('verification')
        try:
            checked = Verification.model_validate(json.loads(raw))
            verdicts = {item.index: item.supported for item in checked.verdicts}
            if len(checked.verdicts) != len(valid) or set(verdicts) != set(range(len(valid))):
                raise ValueError("missing or duplicate verdict")
        except (ValueError, TypeError):
            return response("abstained", "verification_failed", unresolved=[question])
        observe(self.observer, "verification_result", checked.model_dump())
        claims = []
        for index, (part, hit) in enumerate(valid):
            if not verdicts[index]:
                unresolved.append(part.question)
                continue
            claims.append({"question": part.question, "text": part.claim, "citation_id": part.citation_id,
                           "excerpt": part.excerpt, "source_id": hit["source_id"],
                           "source_url": hit["source_url"], "source_version": hit["source_version"],
                           "sha256": hit["sha256"], "page": hit["page"], "anchor": hit["anchor"]})
        if not claims:
            return response("abstained", 'answer_language_mismatch' if rejected_language else "no_verified_claims",
                            unresolved=unresolved or [question])
        return response("partial" if unresolved else "answered",
                        'answer_language_mismatch' if rejected_language else None,
                        claims=claims, unresolved=unresolved)
