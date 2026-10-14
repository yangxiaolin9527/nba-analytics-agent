"""Deterministic comparison of one entity across two season snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import sqlite3
from pathlib import Path
from typing import Literal

from scripts.query_quality import QualityScope, assess_scope, read_database_snapshot
from scripts.relational_baseline import CAPABILITY_MATRIX


class ComparisonError(ValueError):
    """Raised when source observations cannot be compared safely."""

    def __init__(self, message: str, reason_code: str = "comparison_not_computable") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class MetricDefinition:
    subject: Literal["player", "team"]
    metric: str
    source_column: str
    unit: str
    formula: str
    capability_metric: str


@dataclass(frozen=True)
class Observation:
    value: float | int | None
    sample_count: int | None
    quality_status: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonResult:
    status: Literal["complete", "partial"]
    baseline_value: float | int
    comparison_value: float | int
    baseline_sample_count: int | None
    comparison_sample_count: int | None
    absolute_delta: float | int
    relative_change_pct: float | None
    relative_change_status: Literal["computable", "not_computable"]
    quality_status: str
    limitations: tuple[str, ...]
    absolute_delta_formula: str
    relative_change_formula: str
    limitation: str | None = None


@dataclass(frozen=True)
class SourceStatistic:
    season_id: str
    competition_phase: str
    source_table: str
    source_sql: str
    observation: Observation


_METRICS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "player": {
        "points_total": ("points_total", "points", "comparison_value - baseline_value", "points"),
        "assists_total": ("assists_total", "assists", "comparison_value - baseline_value", "assists"),
        "rebounds_total": ("rebounds_total", "rebounds", "comparison_value - baseline_value", "rebounds"),
        "games_appeared": ("games_appeared", "games", "comparison_value - baseline_value", "points"),
        "points_per_appearance": (
            "points_per_appearance", "points per appearance", "comparison_value - baseline_value", "points",
        ),
    },
    "team": {
        "points_total": ("points_total", "points", "comparison_value - baseline_value", "points"),
        "assists_total": ("assists_total", "assists", "comparison_value - baseline_value", "assists"),
        "rebounds_total": ("rebounds_total", "rebounds", "comparison_value - baseline_value", "rebounds"),
        "games_played": ("games_played", "games", "comparison_value - baseline_value", "points"),
    },
}


def metric_definition(subject: str, metric: str) -> MetricDefinition:
    if subject not in _METRICS:
        raise ComparisonError(f"unsupported subject {subject}", "unsupported_subject")
    values = _METRICS[subject].get(metric)
    if values is None:
        raise ComparisonError(f"metric {metric} is not supported for subject {subject}", "unsupported_metric")
    source_column, unit, formula, capability_metric = values
    return MetricDefinition(subject, metric, source_column, unit, formula, capability_metric)


def _quality_status(observations: tuple[Observation, Observation]) -> tuple[str, tuple[str, ...]]:
    statuses = {item.quality_status for item in observations}
    if "unsupported" in statuses:
        raise ComparisonError("comparison contains unsupported source data", "unsupported_scope")
    limitations = tuple(dict.fromkeys(
        limitation for item in observations for limitation in item.limitations
    ))
    if "unverified" in statuses:
        return "unverified", limitations
    if "conditional" in statuses or limitations:
        return "conditional", limitations
    return "validated", limitations


def calculate_comparison(
    definition: MetricDefinition,
    baseline: Observation,
    comparison: Observation,
) -> ComparisonResult:
    if baseline.value is None:
        raise ComparisonError("missing baseline value", "comparison_inputs_missing")
    if comparison.value is None:
        raise ComparisonError("missing comparison value", "comparison_inputs_missing")
    if isinstance(baseline.value, bool) or isinstance(comparison.value, bool):
        raise ComparisonError("comparison values must be numeric", "comparison_inputs_invalid")
    if not all(isfinite(float(value)) for value in (baseline.value, comparison.value)):
        raise ComparisonError("comparison values must be finite", "comparison_inputs_invalid")
    quality_status, limitations = _quality_status((baseline, comparison))
    absolute_delta = comparison.value - baseline.value
    if baseline.value == 0:
        return ComparisonResult(
            status="partial",
            baseline_value=baseline.value,
            comparison_value=comparison.value,
            baseline_sample_count=baseline.sample_count,
            comparison_sample_count=comparison.sample_count,
            absolute_delta=absolute_delta,
            relative_change_pct=None,
            relative_change_status="not_computable",
            quality_status=quality_status,
            limitations=limitations,
            absolute_delta_formula=definition.formula,
            relative_change_formula="(comparison_value - baseline_value) / baseline_value * 100",
            limitation="zero_baseline",
        )
    return ComparisonResult(
        status="complete",
        baseline_value=baseline.value,
        comparison_value=comparison.value,
        baseline_sample_count=baseline.sample_count,
        comparison_sample_count=comparison.sample_count,
        absolute_delta=absolute_delta,
        relative_change_pct=absolute_delta / baseline.value * 100,
        relative_change_status="computable",
        quality_status=quality_status,
        limitations=limitations,
        absolute_delta_formula=definition.formula,
        relative_change_formula="(comparison_value - baseline_value) / baseline_value * 100",
    )


def load_source_statistic(
    connection: sqlite3.Connection,
    *,
    subject: str,
    entity_id: str,
    metric: str,
    season_id: str,
    competition_phase: str,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> SourceStatistic:
    """Load one metric from a reviewed season aggregate and assess its quality."""
    definition = metric_definition(subject, metric)
    table = "player_game_stat" if subject == "player" else "team_game_stat"
    entity_column = "person_id" if subject == "player" else "team_id"
    source_column = {
        "points_total": "points" if subject == "player" else "team_score",
        "assists_total": "assists",
        "rebounds_total": "rebounds_total",
        "games_appeared": "game_id",
        "games_played": "game_id",
    }.get(metric)
    if source_column is None:
        raise ComparisonError(f"metric {metric} has no source column", "unsupported_metric")
    sql = (
        f"SELECT s.game_id, s.{source_column}, g.game_date, g.season_status, g.phase_status "
        f"FROM {table} s JOIN game g ON g.game_id = s.game_id "
        + ("JOIN game_team_participation p ON p.game_id = s.game_id AND p.team_id = s.team_id " if subject == "team" else "")
        + f"WHERE s.{entity_column} = ? AND g.season_id = ? AND g.competition_phase = ? "
        + ("AND p.participation_quality = 'valid' " if subject == "team" else "")
        + ("AND s.appearance_status IN ('played', 'played_inferred') " if subject == "player" else "AND s.team_id <> '' ")
        + "ORDER BY g.game_date ASC, s.game_id ASC"
    )
    snapshot = read_database_snapshot(connection)
    if snapshot is None:
        raise ComparisonError("database snapshot is unavailable", "snapshot_unavailable")
    assessment = assess_scope(
        QualityScope(
            season_id=season_id,
            competition_phase=competition_phase,
            subject=subject,
            metrics=(definition.capability_metric,),
        ),
        matrix_path=matrix_path,
        source_snapshot=snapshot,
    )
    if assessment.status == "unsupported":
        raise ComparisonError(
            f"metric {metric} is unsupported for {season_id} {competition_phase}: "
            + ", ".join(assessment.limitations)
        )
    rows = connection.execute(sql, (entity_id, season_id, competition_phase)).fetchall()
    if not rows:
        raise ComparisonError(
            f"no {subject} observation for {entity_id} in {season_id} {competition_phase}",
            "no_observation",
        )
    if any(row[3] != "valid" or row[4] != "valid" for row in rows):
        raise ComparisonError("source game has invalid season or phase status", "invalid_scope")
    if len({row[0] for row in rows}) != len(rows):
        raise ComparisonError(f"duplicate {subject} game rows", f"duplicate_{subject}_game")
    if metric not in {"games_appeared", "games_played"} and any(row[1] is None for row in rows):
        raise ComparisonError("requested comparison input is missing", "comparison_inputs_missing")
    value = len(rows) if metric in {"games_appeared", "games_played"} else sum(row[1] for row in rows)
    sample_count = len(rows)
    if metric == "points_per_appearance":
        value = value / sample_count if sample_count else None
    return SourceStatistic(
        season_id=season_id,
        competition_phase=competition_phase,
        source_table=table,
        source_sql=sql,
        observation=Observation(
            value=value,
            sample_count=sample_count,
            quality_status=assessment.status,
            limitations=assessment.limitations,
        ),
    )
