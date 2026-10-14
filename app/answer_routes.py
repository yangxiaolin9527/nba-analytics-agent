"""Presentation adapter over the shared single-tool ask flow."""
from __future__ import annotations

import logging
import sqlite3
import time
from uuid import uuid4

from fastapi import FastAPI

from app.ask_routes import AskFlowError, run_ask_flow
from app.answer_evidence import build_answer_evidence
from app.config import AppContext
from app.schemas import AnswerRequest, AnswerResponse
from scripts.answer_renderer import render_failure, render_tool_answer

logger = logging.getLogger(__name__)


def register_answer_routes(application: FastAPI, settings: AppContext) -> None:
    @application.post('/answer', response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse:
        started = time.monotonic()
        request_id = uuid4().hex
        try:
            raw = await run_ask_flow(request, settings, request_id=request_id)
        except AskFlowError as exc:
            raw = exc.response
        status, code = raw.status, raw.reason_code
        error_stage = raw.error_stage
        reason = raw.reason
        if status == 'plan_invalid':
            status, code = 'incomplete', 'invalid_plan'
            reason = f'plan validation failed: {raw.reason_code}'
        if status == 'partial' and not raw.routing_mode:
            status = 'complete'
        result = raw.tool_result
        evidence = raw.evidence.model_dump() if raw.evidence else None
        metadata = dict(evidence or {})
        metadata['planner_reason_code'] = raw.reason_code if raw.status == 'plan_invalid' else None
        text, warnings = None, []
        render_started = time.monotonic()
        try:
            if result is not None:
                evidence = build_answer_evidence(raw, settings.database,
                                                deadline=started + settings.ask_total_timeout_seconds)
            if status in {'complete', 'partial'}:
                rendered = render_tool_answer(raw.tool, raw.arguments, result, evidence=evidence)
                text, warnings = rendered.text, rendered.warnings
            if time.monotonic() - started > settings.ask_total_timeout_seconds and status in {'complete', 'partial'}:
                raise TimeoutError('answer deadline reached')
        except (TimeoutError, sqlite3.OperationalError):
            status, code = 'incomplete', 'evidence_timeout'
            error_stage = 'evidence'
            text, reason = None, 'evidence collection did not complete'
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
            status, code = 'incomplete', 'answer_invalid_result'
            error_stage = 'rendering'
            text, reason = None, 'result or evidence cannot support a complete answer'
        total_ms = round((time.monotonic() - started) * 1000, 3)
        logger.info('answer request_id=%s status=%s tool=%s calls=%s reason_code=%s elapsed_ms=%s',
                    request_id, status, raw.tool, raw.tool_calls_used, code, total_ms)
        return AnswerResponse(
            status=status, request_id=request_id, answer=text or render_failure(status, code, reason),
            error_stage=error_stage,
            conversation_id=raw.conversation_id, turn_index=raw.turn_index,
            conversation_context_status=raw.conversation_context_status,
            routing_mode=raw.routing_mode, planning=raw.planning,
            reason_code=code, reason=reason, tool=raw.tool, arguments=raw.arguments,
            tool_result=result, evidence=evidence, query_metadata=metadata, warnings=warnings,
            quality_status=(evidence or {}).get('quality_status'),
            limitations=(evidence or {}).get('limitations', []), tool_calls_used=raw.tool_calls_used,
            timings={'total_ms': total_ms, 'model_ms': raw.model_elapsed_ms,
                     'tool_ms': raw.tool_elapsed_ms,
                     'render_ms': round((time.monotonic()-render_started)*1000, 3)},
        )
