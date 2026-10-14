"""Bounded natural-language router over registered deterministic tools."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Event
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.ask_tools import AskToolError, execute_registered_tool
from app.config import AppContext
from app.context import database_evidence, query_context
from app.schemas import AskRequest, AskResponse
from scripts.ask_grounding import (
    AskGroundingError, clarification_gap, ground_ask_question, preflight_rejection,
    validate_ask_call, verified_prompt_values,
)
from scripts.ask_plan import build_ask_prompt, parse_ask_plan
from scripts.model_adapter import ModelRequest, OpenAICompatibleModel


logger = logging.getLogger(__name__)


class AskFlowError(HTTPException):
    """Preserve legacy HTTP errors and measurements for the answer adapter."""

    def __init__(self, status_code: int, detail: str, response: AskResponse) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.response = response


def register_ask_routes(application: FastAPI, settings: AppContext) -> None:
    @application.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        return await run_ask_flow(request, settings)


async def run_ask_flow(
    request: AskRequest, settings: AppContext, *, request_id: str | None = None,
) -> AskResponse:
    if request.routing_mode or settings.routing_mode:
        from app.routing_service import run_routed_ask
        return await run_routed_ask(request, settings, request_id=request_id)
    started = time.monotonic()
    deadline = started + settings.ask_total_timeout_seconds
    request_id = request_id or uuid4().hex
    model_elapsed_ms = 0.0
    tool_elapsed_ms = 0.0
    links = None
    evidence = None

    def respond(status: str, *, reason_code: str | None = None, reason: str | None = None,
                tool: str | None = None, arguments: dict | None = None,
                tool_result: dict | None = None, tool_calls_used: int = 0) -> AskResponse:
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        logger.info(
            "ask request_id=%s status=%s tool=%s calls=%s reason_code=%s elapsed_ms=%s",
            request_id, status, tool, tool_calls_used, reason_code, elapsed_ms,
        )
        return AskResponse(
            status=status, reason_code=reason_code, reason=reason,
            error_stage=None if status in {'complete', 'partial'} else 'validation' if status == 'plan_invalid' else 'execution' if tool_calls_used else 'planning',
            tool=tool, arguments=arguments, tool_result=tool_result,
            resolved_values=links, evidence=evidence, tool_calls_used=tool_calls_used,
            elapsed_ms=elapsed_ms, model_elapsed_ms=model_elapsed_ms,
            tool_elapsed_ms=tool_elapsed_ms,
        )

    if settings.database is None or not settings.database.is_file():
        raise AskFlowError(503, "query database is unavailable", respond("incomplete", reason_code="database_unavailable"))
    try:
        context, source_snapshot = query_context()
        with closing(sqlite3.connect(
            f"{settings.database.resolve().as_uri()}?mode=ro", uri=True,
        )) as connection:
            evidence = database_evidence(connection, context, source_snapshot)
            links, normalized_question = ground_ask_question(connection, request.question)
    except AskGroundingError as exc:
        return respond(exc.status, reason_code=exc.code, reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        logger.warning("ask request_id=%s status=unavailable reason=%s", request_id, type(exc).__name__)
        raise AskFlowError(503, "query database is unavailable", respond("incomplete", reason_code="database_unavailable")) from exc

    rejection = preflight_rejection(request.question, links)
    if rejection is not None:
        return respond(rejection.status, reason_code=rejection.code, reason=str(rejection))
    if settings.ask_max_tool_calls < 1:
        return respond("incomplete", reason_code="tool_budget_exceeded", reason="tool call budget is zero")
    try:
        selected_model = settings.model if settings.model is not None else OpenAICompatibleModel(
            timeout=settings.ask_model_timeout_seconds,
            config=settings.provider_config,
        )
    except ValueError as exc:
        raise AskFlowError(503, "ask model is not configured", respond("incomplete", reason_code="model_unconfigured")) from exc
    prompt = build_ask_prompt(normalized_question, verified_prompt_values(links, normalized_question))
    model_started = time.monotonic()
    remaining = min(settings.ask_model_timeout_seconds, deadline - model_started)
    if remaining <= 0:
        return respond("incomplete", reason_code="model_timeout", reason="request deadline reached before planning")
    model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-ask-model")
    try:
        model_response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                model_executor, selected_model.generate,
                ModelRequest(prompt=prompt, response_format="json_object"),
            ), timeout=remaining,
        )
    except TimeoutError:
        model_elapsed_ms = round((time.monotonic() - model_started) * 1000, 3)
        return respond("incomplete", reason_code="model_timeout", reason="model planning exceeded its timeout")
    except Exception as exc:
        logger.error("ask request_id=%s status=model_error type=%s", request_id, type(exc).__name__)
        model_elapsed_ms = round((time.monotonic() - model_started) * 1000, 3)
        raise AskFlowError(502, "ask model planning failed", respond("incomplete", reason_code="provider_failed")) from exc
    finally:
        model_executor.shutdown(wait=False, cancel_futures=True)
    model_elapsed_ms = round((time.monotonic() - model_started) * 1000, 3)
    try:
        plan = parse_ask_plan(model_response.text)
    except (ValueError, TypeError) as exc:
        return respond("plan_invalid", reason_code="invalid_plan", reason=str(exc))
    if plan.action == "clarify":
        code, reason = clarification_gap(normalized_question, links)
        return respond("clarification_required", reason_code=code, reason=reason)
    if plan.action == "unsupported":
        return respond("unsupported", reason_code="unsupported_request",
                       reason="the question requires a capability outside registered tools")
    try:
        arguments = validate_ask_call(plan, links, normalized_question)
    except AskGroundingError as exc:
        return respond(exc.status, reason_code=exc.code, reason=str(exc))

    tool_started = time.monotonic()
    remaining = min(settings.ask_tool_timeout_seconds, deadline - tool_started)
    if remaining <= 0:
        return respond("incomplete", reason_code="tool_timeout", reason="request deadline reached before tool execution")
    tool_deadline = tool_started + remaining
    stop_tool = Event()

    def run_tool() -> dict:
        if stop_tool.is_set() or time.monotonic() >= tool_deadline:
            raise TimeoutError("tool deadline reached before execution")
        with closing(sqlite3.connect(
            f"{settings.database.resolve().as_uri()}?mode=ro", uri=True,
        )) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_progress_handler(
                lambda: int(stop_tool.is_set() or time.monotonic() >= tool_deadline), 1000,
            )
            return execute_registered_tool(
                connection, tool=plan.tool, arguments=arguments,
                matrix_path=settings.quality_matrix_path,
                max_seconds=max(0.0, tool_deadline - time.monotonic()),
            )

    tool_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-ask-tool")
    tool_future = tool_executor.submit(run_tool)
    try:
        # Poll the worker Future directly: on Python 3.14 a completed SQLite
        # worker did not reliably wake an asyncio executor Future here.
        while not tool_future.done():
            wait = tool_deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError("tool execution exceeded its timeout")
            await asyncio.sleep(min(0.01, wait))
        result = tool_future.result()
    except TimeoutError:
        tool_elapsed_ms = round((time.monotonic() - tool_started) * 1000, 3)
        return respond("incomplete", reason_code="tool_timeout", reason="tool execution exceeded its timeout",
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    except AskToolError as exc:
        tool_elapsed_ms = round((time.monotonic() - tool_started) * 1000, 3)
        return respond(exc.status, reason_code=exc.code, reason=str(exc),
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    except sqlite3.OperationalError as exc:
        tool_elapsed_ms = round((time.monotonic() - tool_started) * 1000, 3)
        code = "tool_timeout" if "interrupted" in str(exc).lower() else "tool_failed"
        return respond("incomplete", reason_code=code, reason="tool query did not complete",
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        logger.error("ask request_id=%s status=tool_error type=%s", request_id, type(exc).__name__)
        tool_elapsed_ms = round((time.monotonic() - tool_started) * 1000, 3)
        return respond("incomplete", reason_code="tool_failed", reason="registered tool failed",
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    finally:
        stop_tool.set()
        tool_future.cancel()
        tool_executor.shutdown(wait=False, cancel_futures=True)
    tool_elapsed_ms = round((time.monotonic() - tool_started) * 1000, 3)
    if time.monotonic() > deadline or tool_elapsed_ms > remaining * 1000:
        return respond("incomplete", reason_code="tool_timeout", reason="tool execution exceeded its timeout",
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    status = result["status"]
    if status not in {"complete", "partial", "not_computable"}:
        return respond("incomplete", reason_code="tool_invalid_response", reason="tool returned an invalid status",
                       tool=plan.tool, arguments=arguments, tool_calls_used=1)
    return respond(status, reason_code=result.get("reason_code"), reason=result.get("reason"),
                   tool=plan.tool, arguments=arguments, tool_result=result, tool_calls_used=1)
