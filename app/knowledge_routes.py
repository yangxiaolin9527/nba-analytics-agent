"""Independent evidence retrieval endpoint."""

from __future__ import annotations

import asyncio
import inspect
import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig
from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError
from scripts.knowledge_guidance import GuidedKnowledgeSearch
from app.execution_events import ExecutionEventSink, execution_event_context
from app.live_stream import execution_sse


logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = "bm25"
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


def register_knowledge_routes(app: FastAPI, search, model=None, tracer=None,
                               provider_config: ProviderConfig | None = None):
    config = provider_config or ProviderConfig.empty()
    guidance_options = ({"min_dense_score": config.knowledge_min_dense_score}
                        if config.knowledge_min_dense_score is not None else {})

    async def execute_answer(request: AnswerRequest, sink: ExecutionEventSink):
        if search is None:
            raise HTTPException(503, detail="knowledge index unavailable; run knowledge ingest")
        if search.dense is None:
            raise HTTPException(503, detail="knowledge answering requires a dense index")
        try:
            answer_model = model if model is not None else OpenAICompatibleModel(
                timeout=config.knowledge_model_timeout_seconds, config=config,
            )
            from scripts import knowledge_guidance
            guided = knowledge_guidance.GuidedKnowledgeSearch(search, answer_model, **guidance_options)
            guided.planner_timeout = config.knowledge_planner_timeout_seconds
            guided.embedding_timeout = config.knowledge_embedding_timeout_seconds
            guided.tracer = tracer
            answerer = KnowledgeAnswer(
                search, answer_model, guided=guided,
                retrieval_timeout=config.knowledge_retrieval_timeout_seconds,
                generation_timeout=config.knowledge_generation_timeout_seconds,
                verification_timeout=config.knowledge_verification_timeout_seconds,
                total_timeout=config.knowledge_total_timeout_seconds,
                tracer=tracer,
                generation_thinking=config.knowledge_generation_thinking,
                verification_thinking=config.knowledge_verification_thinking,
                generation_prompt_strategy=config.knowledge_generation_prompt_strategy,
                generation_policy=config.knowledge_generation_policy,
                candidate_top_k=config.knowledge_candidate_top_k,
            )
        except ValueError as exc:
            raise HTTPException(503, detail="knowledge answer configuration is unavailable") from exc
        result = await answerer.aanswer(request.question)
        result["request_id"] = sink.request_id
        result["execution_events"] = sink.snapshot()
        return result

    @app.post("/knowledge/answer")
    async def knowledge_answer(request: AnswerRequest):
        sink = ExecutionEventSink(request_id=uuid4().hex)
        try:
            with execution_event_context(sink):
                return await execute_answer(request, sink)
        except HTTPException:
            raise
        except KnowledgeAnswerError as exc:
            status_code = 504 if exc.retryable else 502
            raise HTTPException(
                status_code,
                detail={"code": exc.reason_code, "stage": exc.stage, "timings": exc.timings,
                        "stage_statuses": getattr(exc, 'stage_statuses', {}),
                        "model_calls": getattr(exc, 'model_calls', {}),
                        "request_id": getattr(exc, 'request_id', None),
                        "execution_events": sink.snapshot() if 'sink' in locals() else []},
            ) from exc
        except TimeoutError as exc:
            raise HTTPException(504, detail={"code": "answer_timeout", "stage": "total"}) from exc
        except Exception as exc:
            logger.error("knowledge answering failed: %s", type(exc).__name__)
            raise HTTPException(502, detail="knowledge answering failed") from exc

    @app.post("/knowledge/answer/stream")
    async def knowledge_answer_stream(request: AnswerRequest):
        if search is None or search.dense is None:
            raise HTTPException(503, detail="knowledge answering requires a dense index")
        return execution_sse(
            lambda sink: execute_answer(request, sink), request_id=uuid4().hex,
        )

    @app.post("/knowledge/search")
    async def knowledge_search(request: SearchRequest):
        if search is None:
            raise HTTPException(503, detail="knowledge index unavailable; run knowledge ingest")
        if request.mode == "guided":
            if search.dense is None:
                raise HTTPException(503, detail="guided retrieval requires a dense index")
            try:
                planner_timeout = config.knowledge_planner_timeout_seconds
                embedding_timeout = config.knowledge_embedding_timeout_seconds
                planner = model if model is not None else OpenAICompatibleModel(
                    timeout=config.knowledge_model_timeout_seconds, config=config,
                )
                guided = GuidedKnowledgeSearch(
                    search, planner, planner_timeout=planner_timeout,
                    embedding_timeout=embedding_timeout, tracer=tracer, **guidance_options,
                )
            except ValueError as exc:
                raise HTTPException(503, detail="knowledge guidance configuration is unavailable") from exc
            try:
                result = await asyncio.wait_for(
                    guided.asearch(request.question, top_k=request.top_k),
                    timeout=config.knowledge_retrieval_timeout_seconds,
                )
                if result.get("retrieval_status") == "error":
                    reason_code = result.get("reason_code") or "retrieval_error"
                    if reason_code.endswith("timeout"):
                        raise HTTPException(
                            504,
                            detail={"code": reason_code, "stage": result.get("error_stage", "retrieval"),
                                    "timings": {key: result.get(key, 0.0)
                                                for key in ("planning_ms", "embedding_ms", "vector_query_ms")}},
                        )
                    raise HTTPException(502, detail="guided retrieval failed")
                return result
            except HTTPException:
                raise
            except TimeoutError as exc:
                raise HTTPException(504, detail="guided retrieval timed out") from exc
            except Exception as exc:
                logger.error("guided knowledge search failed: %s", type(exc).__name__)
                raise HTTPException(502, detail="guided retrieval failed") from exc
        try:
            result = search.asearch(request.question, mode=request.mode, top_k=request.top_k) if hasattr(search, "asearch") else search.search(request.question, mode=request.mode, top_k=request.top_k)
            return await result if inspect.isawaitable(result) else result
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
