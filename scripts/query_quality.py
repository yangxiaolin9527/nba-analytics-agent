"""Conservative capability assessment shared by direct SQL and query plans."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from scripts.relational_baseline import CAPABILITY_MATRIX


@dataclass(frozen=True)
class QualityScope:
    season_id: str | None
    competition_phase: str | None
    subject: str
    metrics: tuple[str, ...]
    aggregate_scope: bool = True
    opponent_filter: bool = False
    team_participation: bool = False


@dataclass(frozen=True)
class QualityAssessment:
    status: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class SqlScopeResult:
    scope: QualityScope | None
    reason: str | None = None


_COLUMN_METRICS = {
    "points_total": "points",
    "points_per_appearance": "points",
    "games_appeared": "points",
    "games_played": "points",
    "home_score": "points",
    "away_score": "points",
    "assists_total": "assists",
    "rebounds_total": "rebounds",
    "efg_percentage": "efg_percentage",
    "field_goals_made": "efg_percentage",
    "three_pointers_made": "efg_percentage",
    "field_goals_attempted": "efg_percentage",
}


def assess_scope(
    scope: QualityScope,
    *,
    matrix_path: Path = CAPABILITY_MATRIX,
    source_snapshot: str,
) -> QualityAssessment:
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return QualityAssessment("unverified", ("capability_matrix_unavailable",))
    if matrix.get("source_snapshot") != source_snapshot:
        return QualityAssessment("unverified", ("capability_matrix_snapshot_mismatch",))
    if not scope.season_id or not scope.competition_phase:
        return QualityAssessment("unverified", ("quality_scope_unverified",))
    period = matrix.get("periods", {}).get(f"{scope.season_id}|{scope.competition_phase}")
    if not isinstance(period, dict):
        return QualityAssessment("unverified", ("capability_not_profiled",))

    target = period.get("subjects", {}).get(scope.subject, {})
    status_by_capability = {
        metric: target.get("metrics", {}).get(metric)
        for metric in scope.metrics
    }
    if scope.opponent_filter:
        status_by_capability["team_opponent_filter"] = target.get("team_opponent_filter")
    if scope.team_participation:
        status_by_capability["team_participation"] = target.get("participation")

    limitations: list[str] = []
    unsupported = [name for name, status in status_by_capability.items() if status == "unsupported"]
    if unsupported:
        return QualityAssessment("unsupported", tuple(f"unsupported:{name}" for name in dict.fromkeys(unsupported)))

    period_limitations = list(period.get("limitations", []))
    for limitation in period_limitations:
        if limitation in {"season_completeness_not_checked", "game_count_mismatch"}:
            if scope.aggregate_scope:
                limitations.append(limitation)
        elif limitation == "team_context_incomplete" and scope.opponent_filter:
            limitations.append(limitation)
        elif limitation == "historical_shooting_unverified" and "efg_percentage" in scope.metrics:
            limitations.append(limitation)

    if any(status is None for status in status_by_capability.values()):
        limitations.append("capability_not_profiled")
    if any(status == "conditional" for status in status_by_capability.values()):
        limitations.append("capability_evidence_conditional")
    unique = tuple(dict.fromkeys(limitations))
    if "capability_not_profiled" in unique:
        return QualityAssessment("unverified", unique)
    return QualityAssessment("conditional" if unique else "validated", unique)


def infer_sql_scope(sql: str, connection: sqlite3.Connection | None = None) -> SqlScopeResult:
    """Infer a scope only for simple, reviewable SQL over supported facts/marts."""
    try:
        query = parse_one(sql, read="sqlite")
    except ParseError:
        return SqlScopeResult(None, "sql_parse_failed")
    if not isinstance(query, exp.Select) or query.args.get("with"):
        return SqlScopeResult(None, "sql_shape_unverified")
    tables = {table.name for table in query.find_all(exp.Table)}
    if tables == {"game"}:
        subject = "team"
        aggregate_scope = False
    elif tables == {"player_season_stat"}:
        subject = "player"
        aggregate_scope = True
    elif tables == {"team_season_stat"}:
        subject = "team"
        aggregate_scope = True
    elif tables == {"player_game_stat", "game"}:
        subject = "player"
        aggregate_scope = True
    else:
        return SqlScopeResult(None, "sql_shape_unverified")

    values: dict[str, set[str]] = {}
    for predicate in query.find_all(exp.EQ):
        if isinstance(predicate.left, exp.Column) and isinstance(predicate.right, exp.Literal):
            values.setdefault(predicate.left.name, set()).add(str(predicate.right.this))
        if isinstance(predicate.right, exp.Column) and isinstance(predicate.left, exp.Literal):
            values.setdefault(predicate.right.name, set()).add(str(predicate.left.this))

    season_values = values.get("season_id", set())
    phase_values = values.get("competition_phase", set())
    season_id = next(iter(season_values)) if len(season_values) == 1 else None
    phase = next(iter(phase_values)) if len(phase_values) == 1 else None
    if tables == {"game"} and connection is not None and season_id is None and phase is None:
        game_ids = values.get("game_id", set())
        if len(game_ids) == 1:
            row = connection.execute(
                "SELECT season_id, competition_phase FROM game WHERE game_id = ?",
                (next(iter(game_ids)),),
            ).fetchone()
            if row is not None:
                season_id, phase = row
    projected = set()
    for projection in query.expressions:
        alias = projection.alias_or_name
        if alias in _COLUMN_METRICS:
            projected.add(alias)
        else:
            projected.update(column.name for column in projection.find_all(exp.Column))
    metrics = tuple(sorted({
        _COLUMN_METRICS[column]
        for column in projected
        if column in _COLUMN_METRICS
    }))
    opponent_filter = "opponent_team_id" in values
    if not metrics or season_id is None or phase is None:
        return SqlScopeResult(None, "sql_scope_unverified")
    required_id = "game_id" if tables == {"game"} else (
        "person_id" if subject == "player" else "team_id"
    )
    if required_id not in values:
        return SqlScopeResult(None, "sql_scope_unverified")
    return SqlScopeResult(QualityScope(
        season_id=season_id,
        competition_phase=phase,
        subject=subject,
        metrics=metrics,
        aggregate_scope=aggregate_scope,
        opponent_filter=opponent_filter,
        team_participation=subject == "team",
    ))


def read_database_snapshot(connection: sqlite3.Connection) -> str | None:
    rows = connection.execute(
        "SELECT source_name, source_version FROM dataset_snapshot LIMIT 2"
    ).fetchall()
    if len(rows) != 1:
        return None
    return f"{rows[0][0]}@{rows[0][1]}"
