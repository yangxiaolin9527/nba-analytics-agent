"""Deterministic effective field goal percentage calculation."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path
from typing import Literal

from scripts.query_quality import QualityScope, assess_scope, read_database_snapshot
from scripts.relational_baseline import CAPABILITY_MATRIX


EFG_FORMULA = "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100"


class EfgCalculationError(ValueError):
    """Raised when a source observation cannot be loaded safely."""

    def __init__(self, message: str, reason_code: str = "efg_not_computable") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class EfgInputs:
    field_goals_made: int | None
    three_pointers_made: int | None
    field_goals_attempted: int | None


@dataclass(frozen=True)
class EfgResult:
    status: Literal["complete", "not_computable"]
    efg_percentage: float | None
    inputs: EfgInputs
    formula: str
    unit: str
    rounding: str
    quality_status: str
    limitations: tuple[str, ...]
    calculation_limitation: str | None = None

    @property
    def reason_code(self) -> str | None:
        return self.calculation_limitation


@dataclass(frozen=True)
class EfgSourceStatistic:
    season_id: str
    competition_phase: str
    source_table: str
    source_sql: str
    inputs: EfgInputs
    sample_count: int | None
    quality_status: str
    limitations: tuple[str, ...]


def _result(
    inputs: EfgInputs,
    *,
    quality_status: str,
    limitations: tuple[str, ...],
    efg_percentage: float | None = None,
    calculation_limitation: str | None = None,
) -> EfgResult:
    return EfgResult(
        status="complete" if calculation_limitation is None else "not_computable",
        efg_percentage=efg_percentage,
        inputs=inputs,
        formula=EFG_FORMULA,
        unit="percent",
        rounding="full_precision",
        quality_status=quality_status,
        limitations=limitations,
        calculation_limitation=calculation_limitation,
    )


def calculate_efg(
    inputs: EfgInputs,
    *,
    quality_status: str = "validated",
    limitations: tuple[str, ...] = (),
) -> EfgResult:
    """Calculate eFG% from one compatible cumulative input range."""
    if quality_status == "unsupported":
        return _result(
            inputs,
            quality_status=quality_status,
            limitations=limitations,
            calculation_limitation="unsupported_data",
        )
    values = (
        inputs.field_goals_made,
        inputs.three_pointers_made,
        inputs.field_goals_attempted,
    )
    if any(value is None for value in values):
        return _result(
            inputs,
            quality_status=quality_status,
            limitations=limitations,
            calculation_limitation="efg_inputs_missing",
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return _result(
            inputs,
            quality_status=quality_status,
            limitations=limitations,
            calculation_limitation="efg_inputs_invalid",
        )
    fgm = inputs.field_goals_made
    three_pointers = inputs.three_pointers_made
    fga = inputs.field_goals_attempted
    assert fgm is not None and three_pointers is not None and fga is not None
    if fga == 0:
        return _result(
            inputs,
            quality_status=quality_status,
            limitations=limitations,
            calculation_limitation="efg_zero_denominator",
        )
    if three_pointers > fgm or fgm > fga:
        return _result(
            inputs,
            quality_status=quality_status,
            limitations=limitations,
            calculation_limitation="efg_inputs_inconsistent",
        )
    return _result(
        inputs,
        quality_status=quality_status,
        limitations=limitations,
        efg_percentage=(fgm + 0.5 * three_pointers) / fga * 100,
    )


def load_source_efg_statistic(
    connection: sqlite3.Connection,
    *,
    entity_id: str,
    season_id: str,
    competition_phase: str,
    matrix_path: Path = CAPABILITY_MATRIX,
) -> EfgSourceStatistic:
    """Load cumulative player shooting inputs and assess their capability."""
    snapshot = read_database_snapshot(connection)
    if snapshot is None:
        raise EfgCalculationError("database snapshot is unavailable", "snapshot_unavailable")
    assessment = assess_scope(
        QualityScope(
            season_id=season_id,
            competition_phase=competition_phase,
            subject="player",
            metrics=("efg_percentage",),
        ),
        matrix_path=matrix_path,
        source_snapshot=snapshot,
    )
    if assessment.status == "unsupported":
        raise EfgCalculationError(
            "eFG% is unsupported for the requested scope: "
            + ", ".join(assessment.limitations), "unsupported_scope"
        )
    source_sql = (
        "SELECT p.game_id, p.field_goals_made, p.three_pointers_made, "
        "p.field_goals_attempted, p.appearance_status, g.season_status, g.phase_status "
        "FROM player_game_stat p JOIN game g ON g.game_id = p.game_id "
        "WHERE p.person_id = ? AND g.season_id = ? AND g.competition_phase = ? "
        "AND p.appearance_status IN ('played', 'played_inferred') "
        "ORDER BY g.game_date ASC, p.game_id ASC"
    )
    rows = connection.execute(
        source_sql, (entity_id, season_id, competition_phase)
    ).fetchall()
    if not rows:
        raise EfgCalculationError(
            f"no player observation for {entity_id} in {season_id} {competition_phase}",
            "no_observation",
        )
    if any(row[5] != "valid" or row[6] != "valid" for row in rows):
        raise EfgCalculationError("source game has invalid season or phase status", "invalid_scope")
    duplicates = {row[0] for row in rows if sum(candidate[0] == row[0] for candidate in rows) > 1}
    if duplicates:
        raise EfgCalculationError("duplicate player game rows", "duplicate_player_game")
    values = [(row[1], row[2], row[3]) for row in rows]
    if any(any(value is None for value in triple) for triple in values):
        raise EfgCalculationError("one or more eFG inputs are missing", "efg_inputs_missing")
    if any(
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in triple)
        or triple[1] > triple[0] or triple[0] > triple[2]
        for triple in values
    ):
        raise EfgCalculationError("one or more eFG inputs are invalid", "efg_inputs_invalid")
    fgm = sum(row[0] for row in values)
    threes = sum(row[1] for row in values)
    fga = sum(row[2] for row in values)
    return EfgSourceStatistic(
        season_id=season_id,
        competition_phase=competition_phase,
        source_table="player_game_stat",
        source_sql=source_sql,
        inputs=EfgInputs(field_goals_made=fgm, three_pointers_made=threes, field_goals_attempted=fga),
        sample_count=len(rows),
        quality_status=assessment.status,
        limitations=assessment.limitations,
    )
