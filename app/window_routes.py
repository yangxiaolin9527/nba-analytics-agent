"""Typed deterministic window-query route."""

from __future__ import annotations

import sqlite3
from contextlib import closing

from fastapi import FastAPI, HTTPException

from app.config import AppContext
from app.context import database_evidence, query_context
from app.schemas import WindowQueryRequest, WindowQueryResponse, WindowEvidence
from scripts.window_query import WindowQueryError, query_window


def register_window_routes(application: FastAPI, settings: AppContext) -> None:
    @application.post("/query/window", response_model=WindowQueryResponse)
    async def window_query_endpoint(request: WindowQueryRequest) -> WindowQueryResponse:
        if settings.database is None or not settings.database.is_file():
            raise HTTPException(status_code=503, detail="query database is unavailable")
        window_evidence = None
        try:
            context, source_snapshot = query_context()
            with closing(sqlite3.connect(f"{settings.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                evidence = database_evidence(connection, context, source_snapshot)
                window_evidence = WindowEvidence(
                    source_snapshot=source_snapshot, snapshot_id=evidence.snapshot_id,
                    catalog_version=evidence.catalog_version, subject=request.subject,
                    entity_id=request.entity_id, season_id=request.season_id,
                    competition_phase=request.competition_phase,
                    requested_scope=request.model_dump(exclude_none=True),
                )
                result = query_window(connection, **request.model_dump(), matrix_path=settings.quality_matrix_path)
                return WindowQueryResponse(
                    status=result.status, values=result.values,
                    actual_sample_count=result.actual_sample_count,
                    available_sample_count=result.available_sample_count,
                    selected_game_ids=list(result.selected_game_ids),
                    actual_date_from=result.actual_date_from, actual_date_to=result.actual_date_to,
                    source_table=result.source_table, source_sql=result.source_sql,
                    source_parameters=list(result.source_parameters),
                    quality_status=result.quality_status, limitations=list(result.limitations),
                    evidence=window_evidence,
                )
        except WindowQueryError as exc:
            return WindowQueryResponse(
                status="not_computable", reason=str(exc), reason_code=exc.reason_code,
                quality_status=exc.quality_status, limitations=list(exc.limitations),
                evidence=window_evidence,
            )
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="window query database is unavailable") from exc
