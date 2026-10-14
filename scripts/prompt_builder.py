"""Build reviewable, versioned Text-to-SQL prompts."""

from __future__ import annotations

import json
from typing import Any

from scripts.query_context import QueryLinks
from scripts.query_lexicon import load_lexicon


def build_prompt(question: str, context: dict[str, Any], *, links: QueryLinks | None = None) -> str:
    lexicon = load_lexicon()
    domains = lexicon["value_domains"]
    value_context = ""
    if links is not None:
        value_context = (
            "Use verified entity IDs and canonical phase values below when they match the question. "
            "Do not invent IDs or infer unlisted aliases.\n"
            f"Verified values: {links.model_dump_json(ensure_ascii=False)}\n"
        )
    return (
        "Generate exactly one SQLite SELECT statement for the question. "
        "Use only the supplied schema and return SQL without Markdown or explanation.\n"
        "Use canonical database values exactly, never display labels as SQL literals. "
        "Statistical questions support regular_season and playoffs only; do not mix phases.\n"
        f"Canonical value domains: {json.dumps(domains, ensure_ascii=False, sort_keys=True)}\n"
        "For player appearance averages, include only played and played_inferred rows. "
        "Return stable column names: points_total, games_appeared, points_per_appearance, "
        "assists_total, rebounds_total, games_played, home_score, away_score. Do not rename these columns.\n"
        f"Schema: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
        f"{value_context}"
        f"Question: {question}"
    )


def build_plan_prompt(question: str, context: dict[str, Any], *, links: QueryLinks | None = None) -> str:
    """Build a strict JSON-only prompt for the bounded M1.8B plan contract."""
    verified = links.model_dump(mode="json", exclude_none=True) if links is not None else {}
    return (
        "Return exactly one JSON object and no Markdown, prose, or code fence. "
        "Do not invent IDs or values. Use only verified IDs and canonical values from the context. "
        "The supported query_family values are player_season_summary, team_season_summary, "
        "single_game_score, and player_vs_opponent_aggregation. "
        "requested_metrics must be an array of stable metric names. "
        "Use the canonical metric points_per_appearance for average or per-appearance points. "
        "For a single_game_score request, requested_metrics must contain home_score and away_score. "
        "For single_game_score, set player_id, team_id, opponent_team_id, season_id, and "
        "competition_phase to null. For season families, provide the requested season and phase. "
        "Do not add fields outside the contract. If required information is missing or ambiguous, "
        "still return the best structurally valid plan only when the verified context contains it; "
        "otherwise use the required fields with no invented value so validation fails explicitly.\n"
        "QueryPlan JSON fields: query_family, player_id, team_id, game_id, opponent_team_id, "
        "season_id, competition_phase, requested_metrics.\n"
        f"Verified values: {json.dumps(verified, ensure_ascii=False, sort_keys=True)}\n"
        f"Schema context: {json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
        f"Question: {question}"
    )
