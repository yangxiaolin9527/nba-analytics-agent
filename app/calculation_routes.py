"""Deterministic calculation HTTP routes."""

from __future__ import annotations

import sqlite3
from contextlib import closing

from fastapi import FastAPI, HTTPException

from scripts.efg_calculation import (
    EfgCalculationError,
    calculate_efg,
    load_source_efg_statistic,
)
from scripts.season_comparison import (
    ComparisonError,
    calculate_comparison,
    load_source_statistic,
    metric_definition,
)

from app.config import AppContext
from app.context import database_evidence, query_context
from app.schemas import (
    ComparisonEvidence,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonSource,
    EfgEvidence,
    EfgInputsResponse,
    EfgRequest,
    EfgResponse,
)


def register_calculation_routes(application: FastAPI, settings: AppContext) -> None:
    @application.post("/calculate/efg", response_model=EfgResponse)
    async def calculate_efg_endpoint(request: EfgRequest) -> EfgResponse:
        if settings.database is None or not settings.database.is_file():
            raise HTTPException(status_code=503, detail="query database is unavailable")
        evidence = None
        try:
            context, source_snapshot = query_context()
            with closing(sqlite3.connect(f"{settings.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                evidence = database_evidence(connection, context, source_snapshot)
                source = load_source_efg_statistic(
                    connection,
                    entity_id=request.entity_id,
                    season_id=request.season_id,
                    competition_phase=request.competition_phase,
                    matrix_path=settings.quality_matrix_path,
                )
                calculation = calculate_efg(
                    source.inputs,
                    quality_status=source.quality_status,
                    limitations=source.limitations,
                )
                limitations = list(calculation.limitations)
                if calculation.calculation_limitation and calculation.calculation_limitation not in limitations:
                    limitations.append(calculation.calculation_limitation)
                return EfgResponse(
                    status=calculation.status,
                    efg_percentage=calculation.efg_percentage,
                    inputs=EfgInputsResponse(
                        field_goals_made=source.inputs.field_goals_made,
                        three_pointers_made=source.inputs.three_pointers_made,
                        field_goals_attempted=source.inputs.field_goals_attempted,
                        sample_count=source.sample_count,
                    ),
                    source_table=source.source_table,
                    source_sql=source.source_sql,
                    quality_status=calculation.quality_status,
                    limitations=limitations,
                    calculation_limitation=calculation.calculation_limitation,
                    reason_code=calculation.reason_code,
                    evidence=EfgEvidence(
                        source_snapshot=source_snapshot,
                        snapshot_id=evidence.snapshot_id,
                        catalog_version=evidence.catalog_version,
                        entity_id=request.entity_id,
                        season_id=request.season_id,
                        competition_phase=request.competition_phase,
                    ),
                )
        except EfgCalculationError as exc:
            return EfgResponse(
                status="not_computable",
                reason=str(exc),
                reason_code=exc.reason_code,
                evidence=(
                    EfgEvidence(
                        source_snapshot=f"{evidence.source_name}@{evidence.source_version}",
                        snapshot_id=evidence.snapshot_id,
                        catalog_version=evidence.catalog_version,
                        entity_id=request.entity_id,
                        season_id=request.season_id,
                        competition_phase=request.competition_phase,
                    )
                    if evidence is not None else None
                ),
            )
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="eFG calculation database is unavailable") from exc

    @application.post("/calculate/compare", response_model=ComparisonResponse)
    async def compare_seasons(request: ComparisonRequest) -> ComparisonResponse:
        if request.baseline_season == request.comparison_season:
            raise HTTPException(
                status_code=422,
                detail="baseline_season and comparison_season must differ",
            )
        try:
            metric_definition(request.subject, request.metric)
        except ComparisonError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if settings.database is None or not settings.database.is_file():
            raise HTTPException(status_code=503, detail="query database is unavailable")
        try:
            context, source_snapshot = query_context()
            with closing(sqlite3.connect(f"{settings.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                evidence = database_evidence(connection, context, source_snapshot)
                definition = metric_definition(request.subject, request.metric)
                baseline = load_source_statistic(
                    connection,
                    subject=request.subject,
                    entity_id=request.entity_id,
                    metric=request.metric,
                    season_id=request.baseline_season,
                    competition_phase=request.competition_phase,
                    matrix_path=settings.quality_matrix_path,
                )
                comparison = load_source_statistic(
                    connection,
                    subject=request.subject,
                    entity_id=request.entity_id,
                    metric=request.metric,
                    season_id=request.comparison_season,
                    competition_phase=request.competition_phase,
                    matrix_path=settings.quality_matrix_path,
                )
                calculation = calculate_comparison(
                    definition, baseline.observation, comparison.observation
                )
                return ComparisonResponse(
                    status=calculation.status,
                    baseline=ComparisonSource(
                        season_id=baseline.season_id,
                        competition_phase=baseline.competition_phase,
                        value=baseline.observation.value,
                        sample_count=baseline.observation.sample_count,
                        source_table=baseline.source_table,
                        source_sql=baseline.source_sql,
                    ),
                    comparison=ComparisonSource(
                        season_id=comparison.season_id,
                        competition_phase=comparison.competition_phase,
                        value=comparison.observation.value,
                        sample_count=comparison.observation.sample_count,
                        source_table=comparison.source_table,
                        source_sql=comparison.source_sql,
                    ),
                    absolute_delta=calculation.absolute_delta,
                    relative_change_pct=calculation.relative_change_pct,
                    relative_change_status=calculation.relative_change_status,
                    calculation_limitation=calculation.limitation,
                    quality_status=calculation.quality_status,
                    limitations=list(calculation.limitations),
                    evidence=ComparisonEvidence(
                        source_snapshot=source_snapshot,
                        snapshot_id=evidence.snapshot_id,
                        catalog_version=evidence.catalog_version,
                        entity_id=request.entity_id,
                        subject=request.subject,
                        metric=request.metric,
                        unit=definition.unit,
                        absolute_delta_formula=calculation.absolute_delta_formula,
                        relative_change_formula=calculation.relative_change_formula,
                    ),
                )
        except ComparisonError as exc:
            return ComparisonResponse(status="not_computable", reason=str(exc), reason_code=exc.reason_code)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise HTTPException(status_code=503, detail="comparison database is unavailable") from exc
