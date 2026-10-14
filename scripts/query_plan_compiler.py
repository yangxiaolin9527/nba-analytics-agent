"""Compile validated M1.8B plans into bounded, deterministic SQLite SQL."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.query_plan import (
    GAME_METRICS,
    PLAYER_METRICS,
    TEAM_METRICS,
    METRIC_CAPABILITIES,
    QueryFamily,
    QueryPlan,
)
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.query_quality import QualityScope, assess_scope, read_database_snapshot


class PlanCompileError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    parameters: tuple[Any, ...]
    quality_status: str = "validated"
    limitations: tuple[str, ...] = ()

    def render(self) -> str:
        rendered = self.sql
        for value in self.parameters:
            literal = str(value) if isinstance(value, (int, float)) else "'" + str(value).replace("'", "''") + "'"
            rendered = rendered.replace("?", literal, 1)
        return rendered


def _disclosure(
    connection: sqlite3.Connection,
    season_id: str,
    phase: str,
    requirements: list[tuple[str, str]],
    *,
    aggregate_scope: bool,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> tuple[str, tuple[str, ...]]:
    """Use the same snapshot-aware quality policy as the direct SQL route."""
    subject = requirements[0][0]
    metrics = tuple(metric for _, metric in requirements
                    if metric not in {"team_participation", "team_opponent_filter"})
    assessment = assess_scope(
        QualityScope(
            season_id=season_id,
            competition_phase=phase,
            subject=subject,
            metrics=metrics,
            aggregate_scope=aggregate_scope,
            opponent_filter=any(metric == "team_opponent_filter" for _, metric in requirements),
            team_participation=any(metric == "team_participation" for _, metric in requirements),
        ),
        matrix_path=matrix_path,
        source_snapshot=read_database_snapshot(connection) or "",
    )
    if assessment.status == "unsupported":
        raise PlanCompileError("unsupported_data", ", ".join(assessment.limitations))
    return assessment.status, assessment.limitations


def _require_row(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...], label: str) -> tuple[Any, ...]:
    row = connection.execute(sql, params).fetchone()
    if row is None:
        raise PlanCompileError("entity_not_found", f"{label} was not found")
    return row


def _require_scope(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> None:
    if connection.execute(sql, params).fetchone() is None:
        raise PlanCompileError("data_unavailable", "no data is available for the requested scope")


def _columns(metrics: list[str], allowed: frozenset[str]) -> list[str]:
    unknown = set(metrics) - allowed
    if unknown:
        raise PlanCompileError("unsupported_metric", f"unsupported metrics: {', '.join(sorted(unknown))}")
    return metrics


def _require_metric_inputs(connection: sqlite3.Connection, plan: QueryPlan, metrics: list[str], *, subject: str) -> None:
    # Season marts use SUM, which silently skips NULL. Check eligible facts first.
    columns = {'points_total': 'points' if subject == 'player' else 'team_score',
               'points_per_appearance': 'points', 'assists_total': 'assists', 'rebounds_total': 'rebounds_total'}
    required = list(dict.fromkeys(columns[m] for m in metrics if m in columns))
    if not required:
        return
    where = [f'p.{"person_id" if subject == "player" else "team_id"} = ?', 'g.season_id = ?', 'g.competition_phase = ?']
    params = [plan.player_id if subject == 'player' else plan.team_id, plan.season_id, plan.competition_phase.value]
    if subject == 'player':
        where.append("p.appearance_status IN ('played', 'played_inferred')")
    if plan.opponent_team_id:
        where.append('p.opponent_team_id = ?')
        params.append(plan.opponent_team_id)
    sql = ('SELECT COUNT(*), ' + ', '.join(f'COUNT(p.{column})' for column in required) +
           f' FROM {subject}_game_stat p JOIN game g ON g.game_id = p.game_id WHERE ' + ' AND '.join(where))
    row = connection.execute(sql, params).fetchone()
    if any(count != row[0] for count in row[1:]):
        raise PlanCompileError('query_inputs_missing', 'requested statistic input is missing')


def compile_query_plan(
    connection: sqlite3.Connection,
    plan: QueryPlan,
    *,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> CompiledQuery:
    """Validate grounded identifiers and return one parameterized SQL template.

    The API substitutes parameters into a reviewed literal template before passing
    it to SqlGateway, whose existing AST and domain checks remain authoritative.
    """
    family = plan.query_family
    if family is QueryFamily.PLAYER_SEASON_SUMMARY:
        metrics = _columns(plan.requested_metrics, PLAYER_METRICS)
        requirements = [("player", METRIC_CAPABILITIES[m]) for m in metrics if m in METRIC_CAPABILITIES] or [("player", "points")]
        quality_status, limitations = _disclosure(
            connection, plan.season_id, plan.competition_phase.value, requirements,
            aggregate_scope=True, matrix_path=matrix_path,
        )
        _require_row(connection, "SELECT person_id FROM player_identity WHERE person_id = ?", (plan.player_id,), "player")
        _require_scope(
            connection,
            "SELECT 1 FROM player_season_stat WHERE person_id = ? AND season_id = ? AND competition_phase = ? LIMIT 1",
            (plan.player_id, plan.season_id, plan.competition_phase.value),
        )
        _require_metric_inputs(connection, plan, metrics, subject='player')
        select = []
        for metric in metrics:
            if metric == "points_per_appearance":
                select.append("points_total * 1.0 / games_appeared AS points_per_appearance")
            else:
                select.append(metric)
        return CompiledQuery(
            f"SELECT {', '.join(select)} FROM player_season_stat WHERE person_id = ? AND season_id = ? AND competition_phase = ?",
            (plan.player_id, plan.season_id, plan.competition_phase.value),
            quality_status, limitations,
        )

    if family is QueryFamily.TEAM_SEASON_SUMMARY:
        metrics = _columns(plan.requested_metrics, TEAM_METRICS)
        requirements = [("team", METRIC_CAPABILITIES[m]) for m in metrics if m in METRIC_CAPABILITIES] or [("team", "points")]
        requirements.append(("team", "team_participation"))
        quality_status, limitations = _disclosure(
            connection, plan.season_id, plan.competition_phase.value, requirements,
            aggregate_scope=True, matrix_path=matrix_path,
        )
        _require_row(connection, "SELECT team_id FROM team_identity WHERE team_id = ?", (plan.team_id,), "team")
        _require_scope(
            connection,
            "SELECT 1 FROM team_season_stat WHERE team_id = ? AND season_id = ? AND competition_phase = ? LIMIT 1",
            (plan.team_id, plan.season_id, plan.competition_phase.value),
        )
        _require_metric_inputs(connection, plan, metrics, subject='team')
        return CompiledQuery(
            f"SELECT {', '.join(metrics)} FROM team_season_stat WHERE team_id = ? AND season_id = ? AND competition_phase = ?",
            (plan.team_id, plan.season_id, plan.competition_phase.value),
            quality_status, limitations,
        )

    if family is QueryFamily.SINGLE_GAME_SCORE:
        if set(plan.requested_metrics) != GAME_METRICS:
            raise PlanCompileError("unsupported_metric", "single game score requires home_score and away_score")
        _require_row(connection, "SELECT game_id FROM game WHERE game_id = ?", (plan.game_id,), "game")
        game_scope = connection.execute(
            "SELECT season_id, competition_phase FROM game WHERE game_id = ?", (plan.game_id,)
        ).fetchone()
        if game_scope and game_scope[0] and game_scope[1]:
            quality_status, limitations = _disclosure(connection,
                game_scope[0], game_scope[1], [("team", "team_participation")],
                aggregate_scope=False, matrix_path=matrix_path,
            )
        else:
            quality_status, limitations = "unverified", ("capability_not_profiled",)
        return CompiledQuery(
            "SELECT home_score, away_score FROM game WHERE game_id = ?",
            (plan.game_id,),
            quality_status, limitations,
        )

    if family is QueryFamily.PLAYER_VS_OPPONENT_AGGREGATION:
        metrics = _columns(plan.requested_metrics, PLAYER_METRICS)
        requirements = [("player", METRIC_CAPABILITIES[m]) for m in metrics if m in METRIC_CAPABILITIES] or [("player", "points")]
        requirements.append(("player", "team_opponent_filter"))
        quality_status, limitations = _disclosure(
            connection, plan.season_id, plan.competition_phase.value, requirements,
            aggregate_scope=True, matrix_path=matrix_path,
        )
        _require_row(connection, "SELECT person_id FROM player_identity WHERE person_id = ?", (plan.player_id,), "player")
        _require_row(connection, "SELECT team_id FROM team_identity WHERE team_id = ?", (plan.opponent_team_id,), "opponent team")
        _require_scope(
            connection,
            """SELECT 1 FROM player_game_stat p JOIN game g ON g.game_id = p.game_id
               WHERE p.person_id = ? AND p.opponent_team_id = ? AND g.season_id = ?
                 AND g.competition_phase = ? AND p.appearance_status IN ('played', 'played_inferred') LIMIT 1""",
            (plan.player_id, plan.opponent_team_id, plan.season_id, plan.competition_phase.value),
        )
        _require_metric_inputs(connection, plan, metrics, subject='player')
        select = []
        for metric in metrics:
            if metric == "points_total":
                select.append("SUM(p.points) AS points_total")
            elif metric == "games_appeared":
                select.append("COUNT(*) AS games_appeared")
            elif metric == "points_per_appearance":
                select.append("SUM(p.points) * 1.0 / COUNT(*) AS points_per_appearance")
            elif metric == "assists_total":
                select.append("SUM(p.assists) AS assists_total")
            elif metric == "rebounds_total":
                select.append("SUM(p.rebounds_total) AS rebounds_total")
        return CompiledQuery(
            "SELECT " + ", ".join(select) + " FROM player_game_stat p JOIN game g ON g.game_id = p.game_id "
            "WHERE p.person_id = ? AND p.opponent_team_id = ? AND g.season_id = ? "
            "AND g.competition_phase = ? AND p.appearance_status IN ('played', 'played_inferred')",
            (plan.player_id, plan.opponent_team_id, plan.season_id, plan.competition_phase.value),
            quality_status, limitations,
        )

    raise PlanCompileError("unsupported_query_family", f"unsupported query family: {family}")
