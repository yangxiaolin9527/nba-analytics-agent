"""Registered read-only tools used by the single-call /ask orchestrator."""

from __future__ import annotations

import sqlite3
from typing import Any

from app.context import database_evidence, query_context
from app.schemas import (
    ComparisonEvidence, ComparisonResponse, ComparisonSource, EfgEvidence, EfgInputsResponse,
    EfgResponse, WindowEvidence, WindowQueryResponse,
)
from scripts.efg_calculation import EfgCalculationError, calculate_efg, load_source_efg_statistic
from scripts.query_plan import QueryFamily, QueryPlan
from scripts.query_plan_compiler import PlanCompileError, compile_query_plan
from scripts.season_comparison import ComparisonError, calculate_comparison, load_source_statistic, metric_definition
from scripts.sql_gateway import GatewayStatus, SqlGateway
from scripts.window_query import WindowQueryError, query_window


class AskToolError(ValueError):
    def __init__(self, code: str, message: str, status: str = "incomplete") -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def execute_registered_tool(
    connection: sqlite3.Connection, *, tool: str, arguments: dict[str, Any],
    matrix_path, max_seconds: float,
) -> dict[str, Any]:
    context, source_snapshot = query_context()
    dataset = database_evidence(connection, context, source_snapshot)
    if tool == "window_query":
        window_evidence = WindowEvidence(
            source_snapshot=source_snapshot, snapshot_id=dataset.snapshot_id,
            catalog_version=dataset.catalog_version, subject=arguments["subject"],
            entity_id=arguments["entity_id"], season_id=arguments["season_id"],
            competition_phase=arguments["competition_phase"], requested_scope=arguments,
        )
        try:
            result = query_window(connection, **arguments, matrix_path=matrix_path)
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
            ).model_dump(mode="json")
        except WindowQueryError as exc:
            return WindowQueryResponse(
                status="not_computable", reason=str(exc), reason_code=exc.reason_code,
                quality_status=exc.quality_status, limitations=list(exc.limitations),
                evidence=window_evidence,
            ).model_dump(mode="json")
    if tool == "season_compare":
        try:
            definition = metric_definition(arguments["subject"], arguments["metric"])
            sources = [load_source_statistic(
                connection, subject=arguments["subject"], entity_id=arguments["entity_id"],
                metric=arguments["metric"], season_id=arguments[key],
                competition_phase=arguments["competition_phase"], matrix_path=matrix_path,
            ) for key in ("baseline_season", "comparison_season")]
            calculation = calculate_comparison(
                definition, sources[0].observation, sources[1].observation,
            )
            return ComparisonResponse(
                status=calculation.status,
                baseline=ComparisonSource(
                    season_id=sources[0].season_id, competition_phase=sources[0].competition_phase,
                    value=sources[0].observation.value, sample_count=sources[0].observation.sample_count,
                    source_table=sources[0].source_table, source_sql=sources[0].source_sql,
                ),
                comparison=ComparisonSource(
                    season_id=sources[1].season_id, competition_phase=sources[1].competition_phase,
                    value=sources[1].observation.value, sample_count=sources[1].observation.sample_count,
                    source_table=sources[1].source_table, source_sql=sources[1].source_sql,
                ),
                absolute_delta=calculation.absolute_delta,
                relative_change_pct=calculation.relative_change_pct,
                relative_change_status=calculation.relative_change_status,
                calculation_limitation=calculation.limitation,
                quality_status=calculation.quality_status,
                limitations=list(calculation.limitations),
                evidence=ComparisonEvidence(
                    source_snapshot=source_snapshot, snapshot_id=dataset.snapshot_id,
                    catalog_version=dataset.catalog_version, entity_id=arguments["entity_id"],
                    subject=arguments["subject"], metric=arguments["metric"], unit=definition.unit,
                    absolute_delta_formula=calculation.absolute_delta_formula,
                    relative_change_formula=calculation.relative_change_formula,
                ),
            ).model_dump(mode="json")
        except ComparisonError as exc:
            return ComparisonResponse(
                status="not_computable", reason=str(exc), reason_code=exc.reason_code,
            ).model_dump(mode="json")
    if tool == "player_efg":
        try:
            source = load_source_efg_statistic(
                connection, entity_id=arguments["entity_id"], season_id=arguments["season_id"],
                competition_phase=arguments["competition_phase"], matrix_path=matrix_path,
            )
            calculation = calculate_efg(
                source.inputs, quality_status=source.quality_status, limitations=source.limitations,
            )
            limitations = list(calculation.limitations)
            if calculation.calculation_limitation and calculation.calculation_limitation not in limitations:
                limitations.append(calculation.calculation_limitation)
            return EfgResponse(
                status=calculation.status, efg_percentage=calculation.efg_percentage,
                inputs=EfgInputsResponse(
                    field_goals_made=source.inputs.field_goals_made,
                    three_pointers_made=source.inputs.three_pointers_made,
                    field_goals_attempted=source.inputs.field_goals_attempted,
                    sample_count=source.sample_count,
                ),
                source_table=source.source_table, source_sql=source.source_sql,
                quality_status=calculation.quality_status, limitations=limitations,
                calculation_limitation=calculation.calculation_limitation,
                reason_code=calculation.reason_code,
                evidence=EfgEvidence(
                    source_snapshot=source_snapshot, snapshot_id=dataset.snapshot_id,
                    catalog_version=dataset.catalog_version, entity_id=arguments["entity_id"],
                    season_id=arguments["season_id"], competition_phase=arguments["competition_phase"],
                ),
            ).model_dump(mode="json")
        except EfgCalculationError as exc:
            return EfgResponse(
                status="not_computable", reason=str(exc), reason_code=exc.reason_code,
                evidence=EfgEvidence(
                    source_snapshot=source_snapshot, snapshot_id=dataset.snapshot_id,
                    catalog_version=dataset.catalog_version, entity_id=arguments["entity_id"],
                    season_id=arguments["season_id"], competition_phase=arguments["competition_phase"],
                ),
            ).model_dump(mode="json")
    if tool in {"season_summary", "single_game_score"}:
        if tool == "season_summary":
            plan = QueryPlan(
                query_family=(QueryFamily.PLAYER_SEASON_SUMMARY if arguments["subject"] == "player"
                              else QueryFamily.TEAM_SEASON_SUMMARY),
                player_id=arguments["entity_id"] if arguments["subject"] == "player" else None,
                team_id=arguments["entity_id"] if arguments["subject"] == "team" else None,
                season_id=arguments["season_id"], competition_phase=arguments["competition_phase"],
                requested_metrics=arguments["metrics"],
            )
        else:
            plan = QueryPlan(
                query_family=QueryFamily.SINGLE_GAME_SCORE, game_id=arguments["game_id"],
                requested_metrics=["home_score", "away_score"],
            )
        try:
            compiled = compile_query_plan(connection, plan, matrix_path=matrix_path)
        except PlanCompileError as exc:
            raise AskToolError(exc.code, exc.message, "unsupported" if exc.code == "unsupported_data" else "not_computable") from exc
        sql = compiled.render()
        result = SqlGateway(connection, max_seconds=max_seconds, max_rows=1).execute(sql)
        if result.status is not GatewayStatus.OK:
            raise AskToolError("query_" + result.status.value, result.reason or "query did not complete")
        if len(result.rows) != 1 or any(value is None for value in result.rows[0]):
            raise AskToolError("query_result_incomplete", "query did not return one complete row", "not_computable")
        values = dict(zip(result.columns, result.rows[0]))
        if tool == "single_game_score":
            row = connection.execute(
                "SELECT home_team_id, away_team_id FROM game WHERE game_id = ?", (arguments["game_id"],),
            ).fetchone()
            if row is None or values["home_score"] == values["away_score"]:
                raise AskToolError("winner_not_computable", "winner cannot be determined", "not_computable")
            values["winner_team_id"] = row[0] if values["home_score"] > values["away_score"] else row[1]
        return {
            "status": "complete", **values, "sql": sql, "source_table": (
                "game" if tool == "single_game_score" else
                "player_season_stat" if arguments["subject"] == "player" else "team_season_stat"
            ), "quality_status": compiled.quality_status,
            "limitations": list(compiled.limitations), "source_snapshot": source_snapshot,
        }
    raise AskToolError("unregistered_tool", "tool is not registered", "unsupported")
