"""Strict, model-facing query plan contract for M1.8B."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator
from scripts.query_scope import require_canonical_season


class QueryFamily(str, Enum):
    PLAYER_SEASON_SUMMARY = "player_season_summary"
    TEAM_SEASON_SUMMARY = "team_season_summary"
    SINGLE_GAME_SCORE = "single_game_score"
    PLAYER_VS_OPPONENT_AGGREGATION = "player_vs_opponent_aggregation"


class CompetitionPhase(str, Enum):
    REGULAR_SEASON = "regular_season"
    PLAYOFFS = "playoffs"


PLAYER_METRICS = frozenset({"points_total", "games_appeared", "points_per_appearance", "assists_total", "rebounds_total"})
TEAM_METRICS = frozenset({"points_total", "games_played", "assists_total", "rebounds_total"})
METRIC_CAPABILITIES = {"points_total": "points", "points_per_appearance": "points",
                       "assists_total": "assists", "rebounds_total": "rebounds"}
GAME_METRICS = frozenset({"home_score", "away_score"})

_ID_RE = re.compile(r"^\d+$")


class QueryPlan(BaseModel):
    """Only the fields needed by the four supported deterministic templates."""

    # JSON shape is strict via extra=forbid and parse_query_plan; enum values
    # still need normal Pydantic conversion from their JSON strings.
    model_config = ConfigDict(extra="forbid")

    query_family: QueryFamily
    player_id: StrictStr | None = None
    team_id: StrictStr | None = None
    game_id: StrictStr | None = None
    opponent_team_id: StrictStr | None = None
    season_id: StrictStr | None = None
    competition_phase: CompetitionPhase | None = None
    requested_metrics: list[StrictStr] = Field(min_length=1)

    @field_validator("player_id", "team_id", "game_id", "opponent_team_id")
    @classmethod
    def require_numeric_id(cls, value: str | None) -> str | None:
        if value is not None and not _ID_RE.fullmatch(value):
            raise ValueError("entity IDs must contain digits only")
        return value

    @field_validator("season_id")
    @classmethod
    def require_season_format(cls, value: str | None) -> str | None:
        return require_canonical_season(value) if value is not None else None

    @field_validator("requested_metrics")
    @classmethod
    def require_known_metrics(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("requested_metrics must not contain duplicates")
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("requested_metrics must contain non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_family_fields(self) -> "QueryPlan":
        family = self.query_family
        if family is QueryFamily.PLAYER_SEASON_SUMMARY:
            self._require_fields("player_id", "season_id", "competition_phase")
            self._forbid_fields("team_id", "game_id", "opponent_team_id")
            self._require_metrics(PLAYER_METRICS)
        elif family is QueryFamily.TEAM_SEASON_SUMMARY:
            self._require_fields("team_id", "season_id", "competition_phase")
            self._forbid_fields("player_id", "game_id", "opponent_team_id")
            self._require_metrics(TEAM_METRICS)
        elif family is QueryFamily.SINGLE_GAME_SCORE:
            self._require_fields("game_id")
            self._forbid_fields("player_id", "team_id", "opponent_team_id", "season_id", "competition_phase")
            self._require_metrics(GAME_METRICS, exact=True)
        elif family is QueryFamily.PLAYER_VS_OPPONENT_AGGREGATION:
            self._require_fields("player_id", "opponent_team_id", "season_id", "competition_phase")
            self._forbid_fields("team_id", "game_id")
            self._require_metrics(PLAYER_METRICS)
        return self

    def _require_fields(self, *names: str) -> None:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

    def _forbid_fields(self, *names: str) -> None:
        present = [name for name in names if getattr(self, name) is not None]
        if present:
            raise ValueError(f"fields are not valid for query family: {', '.join(present)}")

    def _require_metrics(self, allowed: frozenset[str], *, exact: bool = False) -> None:
        unknown = set(self.requested_metrics) - allowed
        if unknown:
            raise ValueError(f"unsupported metrics: {', '.join(sorted(unknown))}")
        if exact and set(self.requested_metrics) != allowed:
            raise ValueError("single_game_score requires home_score and away_score")


def parse_query_plan(raw: str) -> QueryPlan:
    """Parse exactly one JSON object; no repair or Markdown stripping is allowed."""
    import json

    value: Any = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("query plan must be one JSON object")
    return QueryPlan.model_validate(value)
