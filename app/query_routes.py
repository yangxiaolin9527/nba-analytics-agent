"""Natural-language structured query HTTP route."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from scripts.model_adapter import ModelRequest, OpenAICompatibleModel
from scripts.prompt_builder import build_plan_prompt, build_prompt
from scripts.query_context import build_query_context
from scripts.query_plan import QueryPlan, parse_query_plan
from scripts.query_plan_compiler import PlanCompileError, compile_query_plan
from scripts.query_quality import assess_scope, infer_sql_scope, read_database_snapshot
from scripts.sql_gateway import GatewayStatus, SqlGateway
from scripts.tracing import span

from app.config import AppContext
from app.context import database_evidence
from app.schemas import PlanGroundingError, QueryRequest, QueryResponse
from app.main_grounding import verify_direct_sql_grounding, verify_plan_grounding


logger = logging.getLogger(__name__)


def register_query_routes(application: FastAPI, settings: AppContext) -> None:
    @application.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        started = time.monotonic()
        request_id = uuid4().hex
        if settings.database is None or not settings.database.is_file():
            logger.warning("query request_id=%s status=unavailable reason=database", request_id)
            raise HTTPException(status_code=503, detail="query database is unavailable")

        try:
            with span(settings.tracer, "nba.query") as query_span:
                query_span.set_attribute("input.value", request.question)
                query_span.set_attribute("nba.request_id", request_id)
                query_span.set_attribute("nba.context_mode", settings.context_mode)
                query_span.set_attribute("nba.execution_mode", settings.execution_mode)
                query_span.set_attribute("nba.route", settings.route)
                context_started = time.monotonic()
                with span(settings.tracer, "nba.context", "RETRIEVER") as context_span:
                    query_context = build_query_context(
                        settings.database, request.question, settings.context_mode
                    )
                    context_span.set_attribute("input.value", request.question)
                    context_span.set_attribute("nba.route", settings.route)
                    context_span.set_attribute("nba.context_mode", settings.context_mode)
                    context_span.set_attribute(
                        "output.value", query_context.links.model_dump_json(ensure_ascii=False)
                    )
                    context_span.set_attribute("nba.selected_tables", list(query_context.context["tables"]))
                context_elapsed_ms = round((time.monotonic() - context_started) * 1000, 3)
                with closing(sqlite3.connect(f"{settings.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                    evidence = database_evidence(connection, query_context.context, query_context.source_snapshot)
                    gateway = SqlGateway(connection)
                    if len(query_context.links.seasons) > 1:
                        reason = "cross-season queries are deferred to M1.9"
                        query_span.set_attribute("nba.status", "unsupported_scope")
                        return _early_response(
                            "unsupported_scope", reason, query_context.links, evidence, settings,
                            started=started, context_elapsed_ms=context_elapsed_ms,
                            code="unsupported_scope",
                        )
                    if query_context.links.ambiguous_entities:
                        reason = "entity resolution is ambiguous: " + ", ".join(query_context.links.ambiguous_entities)
                        query_span.set_attribute("nba.status", "ambiguous_entity")
                        return _early_response(
                            "ambiguous_entity", reason, query_context.links, evidence, settings,
                            started=started, context_elapsed_ms=context_elapsed_ms,
                            code="ambiguous_entity",
                        )
                    try:
                        selected_model = settings.model if settings.model is not None else OpenAICompatibleModel(
                            config=settings.provider_config,
                        )
                    except ValueError as exc:
                        logger.warning("query request_id=%s status=unavailable reason=model_config", request_id)
                        raise HTTPException(status_code=503, detail="SQL model is not configured") from exc

                    prompt = (
                        build_plan_prompt(request.question, query_context.context, links=query_context.links)
                        if settings.execution_mode == "plan"
                        else build_prompt(
                            request.question,
                            query_context.context,
                            links=query_context.links if settings.context_mode == "schema_value_retrieval" else None,
                        )
                    )
                    try:
                        model_started = time.monotonic()
                        with span(settings.tracer, "nba.model", "LLM") as model_span:
                            model_span.set_attribute("input.value", prompt)
                            model_span.set_attribute("nba.route", settings.route)
                            model_span.set_attribute("nba.execution_mode", settings.execution_mode)
                            model_span.set_attribute("nba.context_mode", settings.context_mode)
                            model_span.set_attribute("nba.model_name", str(getattr(selected_model, "model", type(selected_model).__name__)))
                            model_span.set_attribute("nba.prompt_chars", len(prompt))
                            model_span.set_attribute("nba.response_format", "json_object" if settings.execution_mode == "plan" else "text")
                            model_response = selected_model.generate(ModelRequest(
                                prompt=prompt,
                                response_format="json_object" if settings.execution_mode == "plan" else None,
                            ))
                            model_output = model_response.text
                            model_span.set_attribute("output.value", model_output)
                        model_elapsed_ms = round((time.monotonic() - model_started) * 1000, 3)
                    except Exception as exc:
                        logger.error("query request_id=%s status=model_error error_type=%s", request_id, type(exc).__name__)
                        raise HTTPException(status_code=502, detail="SQL generation failed") from exc

                    plan, sql, quality_status, quality_limitations, plan_error_code, plan_error, plan_elapsed_ms, compile_elapsed_ms = _compile_or_assess(
                        settings, model_output, query_context, connection, request.question
                    )
                    if plan_error_code:
                        query_span.set_attribute("nba.plan_error_code", plan_error_code)
                        return QueryResponse(
                            status="unsupported_data" if plan_error_code == "unsupported_data" else "plan_invalid",
                            result_status="not_executed", result_reason=plan_error,
                            sql="", columns=[], rows=[], reason=plan_error,
                            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                            model_elapsed_ms=model_elapsed_ms, gateway_elapsed_ms=0.0,
                            context_elapsed_ms=context_elapsed_ms, context_mode=settings.context_mode,
                            prompt_chars=len(prompt), resolved_values=query_context.links, evidence=evidence,
                            execution_mode=settings.execution_mode, route=settings.route,
                            plan=plan.model_dump(mode="json") if plan else None,
                            plan_error_code=plan_error_code, plan_error=plan_error,
                            quality_status=quality_status, limitations=quality_limitations,
                            plan_elapsed_ms=plan_elapsed_ms, compile_elapsed_ms=compile_elapsed_ms,
                        )

                    with span(settings.tracer, "nba.gateway", "TOOL") as gateway_span:
                        gateway_span.set_attribute("input.value", sql)
                        gateway_span.set_attribute("nba.route", settings.route)
                        result = gateway.execute(sql)
                        gateway_span.set_attribute("nba.status", result.status.value)
                        gateway_span.set_attribute("output.value", json.dumps({
                            "columns": result.columns, "rows": result.rows[:10], "reason": result.reason,
                        }, ensure_ascii=False))
                    query_span.set_attribute("output.value", result.status.value)
                    if result.status is not GatewayStatus.OK:
                        quality_status = "unverified"
                        quality_limitations = [*quality_limitations, "query_not_successful"]
                    elif not result.rows:
                        quality_status = "unverified"
                        quality_limitations = [*quality_limitations, "empty_result_not_quality_checked"]
                    elif any(value is None for row in result.rows for value in row):
                        quality_status = "unverified"
                        quality_limitations = [*quality_limitations, "null_values_not_quality_checked"]
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            logger.warning("query request_id=%s status=unavailable reason=%s", request_id, type(exc).__name__)
            raise HTTPException(status_code=503, detail="query database is unavailable") from exc

        elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        if result.status is GatewayStatus.OK:
            if not result.rows:
                result_status, result_reason = "empty", "query returned no rows"
            elif any(value is None for row in result.rows for value in row):
                result_status, result_reason = "contains_null", "query returned NULL values"
            else:
                result_status, result_reason = "complete", None
        else:
            result_status = "incomplete" if result.status is GatewayStatus.TRUNCATED else "not_executed"
            result_reason = result.reason
        logger.info(
            "query request_id=%s status=%s elapsed_ms=%s model_elapsed_ms=%s gateway_elapsed_ms=%s",
            request_id, result.status.value, elapsed_ms, model_elapsed_ms, result.elapsed_ms,
        )
        return QueryResponse(
            status=result.status.value, result_status=result_status, result_reason=result_reason,
            sql=sql, columns=result.columns, rows=[list(row) for row in result.rows], reason=result.reason,
            elapsed_ms=elapsed_ms, model_elapsed_ms=model_elapsed_ms,
            gateway_elapsed_ms=result.elapsed_ms, context_elapsed_ms=context_elapsed_ms,
            context_mode=settings.context_mode, prompt_chars=len(prompt),
            resolved_values=query_context.links, evidence=evidence,
            execution_mode=settings.execution_mode, route=settings.route,
            plan=plan.model_dump(mode="json") if settings.execution_mode == "plan" and plan else None,
            quality_status=quality_status, limitations=quality_limitations,
            plan_elapsed_ms=plan_elapsed_ms, compile_elapsed_ms=compile_elapsed_ms,
        )


def _early_response(status, reason, links, evidence, settings, *, started, context_elapsed_ms, code):
    return QueryResponse(
        status=status, result_status="not_executed", result_reason=reason,
        sql="", columns=[], rows=[], reason=reason,
        elapsed_ms=round((time.monotonic() - started) * 1000, 3),
        model_elapsed_ms=0.0, gateway_elapsed_ms=0.0,
        context_elapsed_ms=context_elapsed_ms, context_mode=settings.context_mode,
        prompt_chars=0,
        resolved_values=links,
        evidence=evidence, execution_mode=settings.execution_mode, route=settings.route,
        plan_error_code=code, plan_error=reason,
    )


def _compile_or_assess(settings, model_output, query_context, connection, question):
    plan = None
    quality_status = None
    limitations: list[str] = []
    error_code = error = None
    plan_elapsed_ms = compile_elapsed_ms = 0.0
    sql = model_output if settings.execution_mode == "direct" else ""
    if settings.execution_mode == "plan":
        started = time.monotonic()
        try:
            plan = parse_query_plan(model_output)
            verify_plan_grounding(plan, query_context.links, question)
        except PlanGroundingError as exc:
            error_code, error = exc.code, str(exc)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            error_code, error = "invalid_plan", str(exc)
        plan_elapsed_ms = round((time.monotonic() - started) * 1000, 3)
        if plan is not None:
            started = time.monotonic()
            try:
                compiled = compile_query_plan(connection, plan, matrix_path=settings.quality_matrix_path)
                quality_status, limitations = compiled.quality_status, list(compiled.limitations)
                sql = compiled.render()
            except PlanCompileError as exc:
                error_code, error = exc.code, exc.message
                if exc.code == "unsupported_data":
                    quality_status = "unsupported"
                    limitations = [item.strip() for item in exc.message.split(",")]
            compile_elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    else:
        try:
            verify_direct_sql_grounding(model_output, question, query_context.links)
        except PlanGroundingError as exc:
            error_code, error = exc.code, str(exc)
            return plan, sql, quality_status, limitations, error_code, error, plan_elapsed_ms, compile_elapsed_ms
        inferred = infer_sql_scope(model_output, connection)
        if inferred.scope is None:
            quality_status, limitations = "unverified", [inferred.reason or "quality_scope_unverified"]
        else:
            assessment = assess_scope(
                inferred.scope,
                matrix_path=settings.quality_matrix_path,
                source_snapshot=read_database_snapshot(connection) or "",
            )
            quality_status, limitations = assessment.status, list(assessment.limitations)
            if assessment.status == "unsupported":
                error_code, error = "unsupported_data", ", ".join(assessment.limitations)
    return plan, sql, quality_status, limitations, error_code, error, plan_elapsed_ms, compile_elapsed_ms
