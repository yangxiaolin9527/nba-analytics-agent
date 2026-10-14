"""One-call model plan contract for the structured /ask route."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolName = Literal[
    "season_summary", "single_game_score", "window_query", "season_compare", "player_efg",
]


class AskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["call", "clarify", "unsupported"]
    tool: ToolName | None = None
    arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "AskPlan":
        if self.action == "call" and (self.tool is None or self.arguments is None):
            raise ValueError("call requires one registered tool and arguments")
        if self.action != "call" and (self.tool is not None or self.arguments is not None):
            raise ValueError("non-call action must not include a tool")
        return self


class SeasonSummaryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: Literal["player", "team"]
    entity_id: str
    season_id: str
    competition_phase: Literal["regular_season", "playoffs"]
    metrics: list[str] = Field(min_length=1)


class SingleGameArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    game_id: str


def parse_ask_plan(raw: str) -> AskPlan:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("ask plan must be exactly one JSON object")
    return AskPlan.model_validate(value)


def build_ask_prompt(question: str, verified_values: dict[str, Any]) -> str:
    """Give the model only reviewed tool names and grounded values, never SQL."""
    tools = {
        "season_summary": "subject(player/team), entity_id, season_id, competition_phase, metrics(points_total, games_appeared, points_per_appearance, games_played, assists_total, rebounds_total); include all requested metrics",
        "single_game_score": "game_id; returns home/away scores and winner",
        "window_query": "subject, entity_id, season_id, competition_phase, metrics(points_total, games_appeared, points_per_appearance, assists_total, rebounds_total, games_played), optional opponent_team_id/date_from/date_to/last_n (1-100)",
        "season_compare": "subject, entity_id, metric, baseline_season, comparison_season, competition_phase",
        "player_efg": "entity_id, season_id, competition_phase",
    }
    return (
        "Return exactly one JSON object with action=call and one registered tool plus arguments, "
        "or action=clarify/unsupported with tool and arguments omitted. No prose, SQL, code, "
        "numeric answer, or extra fields. Use only verified entity IDs, seasons, phases, dates "
        "and the metric named by the question. Never default a missing season or phase. "
        "For recent games use window_query; for an opponent aggregate use window_query; "
        "for a previous-season comparison derive the immediately preceding season. "
        "A tool call must contain action, tool and arguments.\n"
        f"Registered tools: {json.dumps(tools, ensure_ascii=False, sort_keys=True)}\n"
        f"Verified values: {json.dumps(verified_values, ensure_ascii=False, sort_keys=True)}\n"
        f"Question: {question}"
    )
