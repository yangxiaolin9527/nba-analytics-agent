"""Deterministic player/team date and recent-game window queries."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from scripts.query_quality import QualityScope, assess_scope, read_database_snapshot
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.query_plan import METRIC_CAPABILITIES, PLAYER_METRICS, TEAM_METRICS


class WindowQueryError(ValueError):
    def __init__(
        self, message: str, reason_code: str = "window_not_computable",
        *, quality_status: str = "unverified", limitations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.quality_status = quality_status
        self.limitations = limitations


@dataclass(frozen=True)
class WindowResult:
    status: str
    values: dict[str, int | float | None]
    actual_sample_count: int
    available_sample_count: int
    selected_game_ids: tuple[str, ...]
    actual_date_from: str | None
    actual_date_to: str | None
    source_table: str
    source_sql: str
    source_parameters: tuple[str, ...]
    quality_status: str
    limitations: tuple[str, ...]


def query_window(
    connection: sqlite3.Connection,
    *,
    subject: str,
    entity_id: str,
    season_id: str,
    competition_phase: str,
    metrics: tuple[str, ...],
    opponent_team_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    last_n: int | None = None,
    matrix_path=CAPABILITY_MATRIX,
) -> WindowResult:
    supported = PLAYER_METRICS if subject == "player" else TEAM_METRICS if subject == "team" else set()
    if not supported:
        raise WindowQueryError(f"unsupported subject {subject}", "unsupported_subject")
    if not metrics or len(set(metrics)) != len(metrics) or any(metric not in supported for metric in metrics):
        raise WindowQueryError("one or more metrics are unsupported for subject", "unsupported_metric")
    if last_n is not None and not 1 <= last_n <= 100:
        raise WindowQueryError("last_n must be between 1 and 100", "invalid_last_n")
    if date_from and date_to and date_from > date_to:
        raise WindowQueryError("date_from must not be after date_to", "invalid_date_range")
    snapshot = read_database_snapshot(connection)
    if snapshot is None:
        raise WindowQueryError("database snapshot is unavailable", "snapshot_unavailable")
    assessment = assess_scope(QualityScope(
        season_id=season_id, competition_phase=competition_phase,
        subject=subject, metrics=tuple(dict.fromkeys(METRIC_CAPABILITIES[m] for m in metrics if m in METRIC_CAPABILITIES)),
        opponent_filter=opponent_team_id is not None,
        team_participation=subject == "team",
    ), matrix_path=matrix_path, source_snapshot=snapshot)
    if assessment.status == "unsupported":
        raise WindowQueryError(
            "window scope is unsupported: " + ", ".join(assessment.limitations),
            "unsupported_scope", quality_status="unsupported", limitations=assessment.limitations,
        )
    if subject == "player":
        source_table = "player_game_stat"
        where = ["p.person_id = ?", "g.season_id = ?", "g.competition_phase = ?", "p.appearance_status IN ('played', 'played_inferred')"]
        params: list[object] = [entity_id, season_id, competition_phase]
        sql = ("SELECT p.game_id, g.game_date, p.points, p.appearance_status, p.team_id, p.opponent_team_id, g.season_status, g.phase_status, "
               "(SELECT gp.participation_quality FROM game_team_participation gp WHERE gp.game_id = p.game_id AND gp.team_id = p.team_id), p.assists, p.rebounds_total "
               "FROM player_game_stat p JOIN game g ON g.game_id = p.game_id WHERE " + " AND ".join(where) +
               " ORDER BY g.game_date ASC, p.game_id ASC")
    else:
        source_table = "team_game_stat"
        where = ["t.team_id = ?", "g.season_id = ?", "g.competition_phase = ?", "t.team_id <> ''"]
        params = [entity_id, season_id, competition_phase]
        sql = ("SELECT t.game_id, g.game_date, t.team_score, t.opponent_team_id, p.participation_quality, g.season_status, g.phase_status, t.assists, t.rebounds_total "
               "FROM team_game_stat t JOIN game g ON g.game_id = t.game_id "
               "JOIN game_team_participation p ON p.game_id = t.game_id AND p.team_id = t.team_id "
               "WHERE " + " AND ".join(where) + " ORDER BY g.game_date ASC, t.game_id ASC")
    date_clauses = []
    if date_from:
        date_clauses.append("g.game_date >= ?")
        params.append(date_from)
    if date_to:
        date_clauses.append("g.game_date <= ?")
        params.append(date_to)
    if date_clauses:
        sql = sql.replace(" ORDER BY", " AND " + " AND ".join(date_clauses) + " ORDER BY")
    rows = connection.execute(sql, params).fetchall()
    if opponent_team_id is not None:
        if subject == "player" and any(not row[5] for row in rows):
            raise WindowQueryError("opponent context is incomplete", "opponent_context_incomplete")
        if subject == "team" and any(not row[3] for row in rows):
            raise WindowQueryError("opponent context is incomplete", "opponent_context_incomplete")
        alias = "p" if subject == "player" else "t"
        sql = sql.replace(" ORDER BY", f" AND {alias}.opponent_team_id = ? ORDER BY")
        params.append(opponent_team_id)
        rows = connection.execute(sql, params).fetchall()
    invalid_scope = (
        any(row[6] != "valid" or row[7] != "valid" for row in rows)
        if subject == "player"
        else any(row[5] != "valid" or row[6] != "valid" for row in rows)
    )
    if invalid_scope:
        raise WindowQueryError("source game has invalid season or phase status", "invalid_scope")
    if subject == "team" and any(row[4] != "valid" for row in rows):
        raise WindowQueryError("team participation is not valid", "invalid_participation")
    if opponent_team_id is not None:
        if subject == "player" and any(not row[4] or row[8] != "valid" for row in rows):
            raise WindowQueryError("team context is incomplete for opponent filter", "opponent_context_incomplete")
    if len({row[0] for row in rows}) != len(rows):
        raise WindowQueryError("duplicate entity game rows", f"duplicate_{subject}_game")
    if not rows:
        raise WindowQueryError("no observations in requested window", "no_observation")
    available_count = len(rows)
    selected = rows[-last_n:] if last_n is not None else rows
    totals = {}
    for metric, index in (('assists_total', -2), ('rebounds_total', -1)):
        if metric in metrics:
            if any(row[index] is None for row in selected):
                raise WindowQueryError(f'requested {metric} input is missing', 'window_inputs_missing')
            totals[metric] = sum(row[index] for row in selected)
    if subject == "player":
        needs_points = "points_total" in metrics or "points_per_appearance" in metrics
        if any(row[2] is None for row in selected) and needs_points:
            raise WindowQueryError("requested points input is missing", "window_inputs_missing")
        values: dict[str, int | float | None] = dict(totals)
        points = sum(row[2] for row in selected) if all(row[2] is not None for row in selected) else None
        appearances = len(selected)
        if "points_total" in metrics:
            values["points_total"] = points
        if "games_appeared" in metrics:
            values["games_appeared"] = appearances
        if "points_per_appearance" in metrics:
            values["points_per_appearance"] = points / appearances if points is not None else None
        limitations = list(assessment.limitations)
        if any(row[3] == "played_inferred" for row in selected):
            limitations.append("inferred_appearance_included")
    else:
        if any(row[2] is None for row in selected) and "points_total" in metrics:
            raise WindowQueryError("requested team score input is missing", "window_inputs_missing")
        values = dict(totals)
        if "points_total" in metrics:
            values["points_total"] = sum(row[2] for row in selected)
        if "games_played" in metrics:
            values["games_played"] = len(selected)
        limitations = list(assessment.limitations)
    return WindowResult(
        status="partial" if last_n is not None and available_count < last_n else "complete",
        values=values, actual_sample_count=len(selected),
        available_sample_count=available_count,
        selected_game_ids=tuple(row[0] for row in selected),
        actual_date_from=selected[0][1], actual_date_to=selected[-1][1],
        source_table=source_table, source_sql=sql,
        source_parameters=tuple(str(value) for value in params),
        quality_status=("unverified" if assessment.status == "unverified" else
                        "conditional" if assessment.status == "conditional" or limitations else "validated"),
        limitations=tuple(dict.fromkeys(limitations)),
    )
