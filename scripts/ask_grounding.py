"""Deterministic validation of one model-proposed NBA tool call."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from pydantic import ValidationError

from app.schemas import ComparisonRequest, EfgRequest, WindowQueryRequest
from scripts.ask_plan import AskPlan, SeasonSummaryArguments, SingleGameArguments
from scripts.query_context import QueryLinks, SEASON_PATTERN, _link_entities, find_game_candidates, normalize_question_seasons
from scripts.query_scope import (
    QueryScopeError, explicit_dates, normalize_question_dates, requested_date_bounds,
)
from scripts.query_plan import PLAYER_METRICS, TEAM_METRICS


class AskGroundingError(ValueError):
    def __init__(self, code: str, message: str, status: str = "plan_invalid") -> None:
        super().__init__(message)
        self.code = code
        self.status = status


_MODELS = {
    "season_summary": SeasonSummaryArguments,
    "single_game_score": SingleGameArguments,
    "window_query": WindowQueryRequest,
    "season_compare": ComparisonRequest,
    "player_efg": EfgRequest,
}
_METRIC_PATTERNS = {
    "points_total": r"总得分|累计.{0,4}得分|一共.{0,3}得分|得分多少|得了多少分|总分|total points",
    "points_per_appearance": r"场均得分|平均得分|每场.{0,3}得分|points per game|\bppg\b",
    "games_appeared": r"出场.{0,3}(次数|场数|多少场)|出场次数|出场场数",
    "games_played": r"比赛场数|打了多少场|打了几场|总场数",
    "assists_total": r"助攻|\bassists?\b",
    "rebounds_total": r"篮板|\brebounds?\b",
    "assists_per_appearance": r"(?:场均|平均|每场)(?:平均|每场|有|大约|大概|拿到|贡献|可以|能|多少|次|个|的|值|数|是|\s){0,12}助攻|助攻(?:的)?(?:场均|平均|每场)|(?:average|per.game).{0,3}assists?|assists?.{0,8}(?:per game|average)|\bapg\b",
    "rebounds_per_appearance": r"(?:场均|平均|每场)(?:平均|每场|有|大约|大概|拿到|贡献|可以|能|多少|次|个|的|值|数|是|\s){0,12}篮板|篮板(?:的)?(?:场均|平均|每场)|(?:average|per.game).{0,3}rebounds?|rebounds?.{0,8}(?:per game|average)|\brpg\b",
    "steals_total": r"抢断|\bsteals?\b",
    "blocks_total": r"盖帽|\bblocks?\b",
    "turnovers_total": r"失误|\bturnovers?\b",
    "three_pointers_made": r"三分(?:球)?命中|命中.{0,3}三分|three.pointers made",
    "three_point_percentage": r"三分命中率|three.point percentage",
    "free_throws_made": r"罚球|free throws?",
    "wins_total": r"胜场|赢了多少场|胜率",
    "efg_percentage": r"efg\s*%?|有效命中率",
}
_PLAYER_METRICS = PLAYER_METRICS
_TEAM_METRICS = TEAM_METRICS
_COMPARE_PLAYER_METRICS = _PLAYER_METRICS | {"assists_total", "rebounds_total"}
_COMPARE_TEAM_METRICS = _TEAM_METRICS | {"assists_total", "rebounds_total"}
_CHINESE_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9}


def ground_ask_question(connection: sqlite3.Connection, question: str) -> tuple[QueryLinks, str]:
    try:
        question = normalize_question_seasons(question)
    except ValueError as exc:
        raise AskGroundingError('invalid_season', '赛季年份必须连续，请使用如 2024-25 或 2024-2025。', 'clarification_required') from exc
    try:
        normalized = normalize_question_dates(question)
        explicit_dates(normalized)
    except QueryScopeError as exc:
        raise AskGroundingError(exc.code, str(exc), 'clarification_required') from exc
    links = _link_entities(connection, normalized)
    return links.model_copy(update={"games": find_game_candidates(connection, normalized)}), normalized


def _number(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    if token in {"百", "一百"}:
        return 100
    if "十" in token:
        left, right = token.split("十", 1)
        if left not in {"", *_CHINESE_DIGITS} or right not in {"", *_CHINESE_DIGITS}:
            return None
        return (_CHINESE_DIGITS.get(left, 1) * 10) + _CHINESE_DIGITS.get(right, 0)
    return _CHINESE_DIGITS.get(token)


def recent_n(question: str) -> int | None:
    match = re.search(r"最近\s*([0-9]{1,3}|[一二两三四五六七八九十百]+)\s*(?:场|次)", question)
    return _number(match.group(1)) if match else None


def previous_season(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{(year + 1) % 100:02d}"


def verified_prompt_values(links: QueryLinks, question: str) -> dict[str, Any]:
    values = links.model_dump(mode="json")
    values["dates"] = explicit_dates(question)
    values["recent_n"] = recent_n(question)
    if "上一季" in question and len(links.seasons) == 1:
        values["derived_previous_season"] = previous_season(links.seasons[0])
    return values


def preflight_rejection(question: str, links: QueryLinks) -> AskGroundingError | None:
    if links.ambiguous_entities:
        return AskGroundingError("ambiguous_entity", "entity name is ambiguous", "clarification_required")
    if re.search(r"(?<![A-Za-z])(?:select|insert|update|delete|drop|alter|exec)(?![A-Za-z])|执行代码|运行命令|任意代码", question, re.I):
        return AskGroundingError("arbitrary_execution_unsupported", "arbitrary SQL or code is not a registered tool", "unsupported")
    if re.search(r"TS%|Net Rating|真实命中率|净效率|置信区间|预测|概率|知识解释|规则解释", question, re.I):
        return AskGroundingError("unsupported_request", "requested capability is not registered", "unsupported")
    if re.search(r"前[一二三四五六七八九十\d]+名|排名|排行榜|各队分别|转会.{0,8}分段", question):
        return AskGroundingError("multi_row_query_out_of_scope", "this result shape is outside the one-tool route", "unsupported")
    return None


def clarification_gap(question: str, links: QueryLinks) -> tuple[str, str]:
    """Name a missing grounded slot when the model chooses to clarify."""
    single_game = bool(links.games and re.search(r"比分|那场|单场|谁赢|谁获胜", question))
    if not links.players and not links.teams and not single_game:
        return "missing_entity", "specify a player or team"
    if not single_game and not links.seasons:
        return "missing_season", "specify a historical season"
    if not single_game and not links.phases:
        return "missing_phase", "specify regular season or playoffs phase"
    if "最近" in question and recent_n(question) is None:
        return "missing_window_size", "specify the number of recent games"
    return "model_requested_clarification", "specify the missing scope or metric"


def _require_entity(entity_id: str, subject: str, links: QueryLinks) -> None:
    linked = links.players if subject == "player" else links.teams
    if not linked:
        raise AskGroundingError("missing_entity", "name the requested player or team", "clarification_required")
    if entity_id not in {item.id for item in linked}:
        raise AskGroundingError("unverified_entity", "entity ID is not grounded in the question")
    if subject == "player" and len({item.id for item in linked}) > 1:
        raise AskGroundingError("multi_entity_out_of_scope", "one player per call is supported", "unsupported")


def _require_season_phase(arguments: dict[str, Any], links: QueryLinks) -> None:
    if arguments["competition_phase"] not in {"regular_season", "playoffs"}:
        raise AskGroundingError("unsupported_phase", "only regular season and playoffs are registered", "unsupported")
    if not links.seasons:
        raise AskGroundingError("missing_season", "specify a historical season", "clarification_required")
    if not links.phases:
        raise AskGroundingError("missing_phase", "specify regular season or playoffs", "clarification_required")
    if len(links.phases) != 1:
        raise AskGroundingError("ambiguous_phase", "one competition phase is required", "clarification_required")
    if arguments["competition_phase"] != links.phases[0]:
        raise AskGroundingError("unverified_scope", "phase is not grounded in the question")
    if "season_id" in arguments and arguments["season_id"] not in links.seasons:
        raise AskGroundingError("unverified_scope", "season is not grounded in the question")


def requested_metrics(question: str) -> set[str]:
    return {name for name, pattern in _METRIC_PATTERNS.items() if re.search(pattern, question, re.I)}


def _require_metrics(metrics: list[str], subject: str, question: str, *, compare: bool = False) -> None:
    allowed = (_COMPARE_PLAYER_METRICS if subject == "player" else _COMPARE_TEAM_METRICS) if compare else (
        _PLAYER_METRICS if subject == "player" else _TEAM_METRICS)
    if not metrics or len(metrics) != len(set(metrics)):
        raise AskGroundingError("invalid_metric", "metrics must be non-empty and unique")
    requested = requested_metrics(question)
    if requested - allowed:
        raise AskGroundingError("unsupported_metric", "a requested metric is not registered for this tool", "unsupported")
    for metric in metrics:
        if metric not in allowed:
            raise AskGroundingError("unsupported_metric", f"metric {metric} is not registered", "unsupported")
        if not re.search(_METRIC_PATTERNS[metric], question, re.I):
            raise AskGroundingError("unverified_metric", f"metric {metric} was not requested")
    if requested != set(metrics):
        raise AskGroundingError("unverified_metric", "the tool plan omitted a requested metric")


def _require_window_filters(arguments: dict[str, Any], links: QueryLinks, question: str) -> None:
    requested_n = recent_n(question)
    if "最近" in question and requested_n is None:
        raise AskGroundingError("missing_window_size", "specify a recent-game count", "clarification_required")
    if arguments.get("last_n") != requested_n:
        raise AskGroundingError("unverified_filter", "last_n does not match the question")
    actual = (arguments.get("date_from"), arguments.get("date_to"))
    try:
        expected = requested_date_bounds(question)
    except QueryScopeError as exc:
        raise AskGroundingError(exc.code, str(exc), 'clarification_required') from exc
    if actual != expected:
        raise AskGroundingError("unverified_filter", "date bounds do not match the question")
    opponent = arguments.get("opponent_team_id")
    if opponent is not None:
        if opponent not in {team.id for team in links.teams} or not re.search(r"对|面对|\bvs\b", question, re.I):
            raise AskGroundingError("unverified_entity", "opponent is not grounded in the question")
        if arguments["subject"] == "team":
            ordered = sorted(links.teams, key=lambda item: question.find(item.alias))
            if len(ordered) < 2 or arguments["entity_id"] != ordered[0].id or opponent != ordered[1].id:
                raise AskGroundingError("unverified_entity", "team/opponent direction is not grounded")
    elif re.search(r"面对|对阵|对手|\b(?:vs|against)\b", question, re.I) or (re.search(r"对|面对|\bvs\b", question, re.I) and (
        (arguments["subject"] == "player" and links.teams) or
        (arguments["subject"] == "team" and len(links.teams) > 1)
    )):
        raise AskGroundingError("unverified_filter", "opponent filter was omitted")


def _needs_window_filter(subject: str, links: QueryLinks, question: str) -> bool:
    return bool(
        "最近" in question or explicit_dates(question)
        or re.search(r"面对|对阵|对手|\b(?:vs|against)\b", question, re.I)
        or (re.search(r"对|面对|\bvs\b", question, re.I)
            and (len(links.teams) > 1 or (subject == "player" and links.teams)))
    )


def _require_comparison_direction(arguments: dict[str, Any], links: QueryLinks, question: str) -> None:
    baseline, comparison = arguments["baseline_season"], arguments["comparison_season"]
    if len(links.seasons) == 1 and re.search(r"上一季|上赛季|相对上赛季|相对上一季", question):
        expected = (previous_season(links.seasons[0]), links.seasons[0])
    elif len(links.seasons) == 2:
        first, second = links.seasons
        if re.search(rf"{re.escape(first)}.*比.*{re.escape(second)}", question):
            expected = (second, first)
        elif re.search(rf"从\s*{re.escape(first)}.*到\s*{re.escape(second)}", question):
            expected = (first, second)
        else:
            raise AskGroundingError("ambiguous_comparison_direction", "state which season is the baseline", "clarification_required")
    else:
        raise AskGroundingError("missing_comparison_season", "specify both comparison seasons", "clarification_required")
    if (baseline, comparison) != expected:
        raise AskGroundingError("unverified_scope", "comparison direction is not grounded")


def validate_ask_call(plan: AskPlan, links: QueryLinks, question: str) -> dict[str, Any]:
    if plan.action != "call" or plan.tool is None or plan.arguments is None:
        raise AskGroundingError("invalid_plan", "a tool call is required")
    model = _MODELS[plan.tool]
    unknown = set(plan.arguments) - set(model.model_fields)
    if unknown:
        raise AskGroundingError("invalid_arguments", "tool arguments contain unknown fields")
    try:
        validated = model.model_validate(plan.arguments).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise AskGroundingError("invalid_arguments", "tool arguments do not match the registered schema") from exc
    if plan.tool == "single_game_score":
        if re.search(r"篮板|助攻|抢断|盖帽|失误|三分|罚球|投篮|命中率|efg|分钟|正负值|犯规|出场", question, re.I):
            raise AskGroundingError("game_metric_out_of_scope", "the game-score tool has no requested statistic", "unsupported")
        if links.players and (
            re.search(r"得分|多少分|几分|篮板|助攻|抢断|盖帽|数据|表现", question)
            or not re.search(r"比分|双方|谁赢|谁获胜|胜者|球队", question)
        ):
            raise AskGroundingError("player_game_stat_out_of_scope", "player single-game statistics need another tool", "unsupported")
        if not links.games:
            raise AskGroundingError("missing_game", "specify one identifiable game", "clarification_required")
        if len(links.games) > 1:
            raise AskGroundingError("ambiguous_game", "multiple games match", "clarification_required")
        if validated["game_id"] != links.games[0].game_id:
            raise AskGroundingError("unverified_game", "game ID is not grounded in the question")
        return validated
    subject = "player" if plan.tool == "player_efg" else validated["subject"]
    _require_entity(validated["entity_id"], subject, links)
    if subject == "team" and len({team.id for team in links.teams}) > 1 and not (
        plan.tool == "window_query" and validated.get("opponent_team_id")
    ):
        raise AskGroundingError("multi_entity_out_of_scope", "two-team comparison needs another tool", "unsupported")
    _require_season_phase(validated, links)
    if plan.tool != "season_compare" and len(set(links.seasons)) != 1:
        raise AskGroundingError("multi_season_out_of_scope", "this tool covers one season", "unsupported")
    if plan.tool != "season_compare" and re.search(
        r"上一季|上赛季|较上季|增幅|增长率|变化率|差值|相差|跨赛季", question,
    ):
        raise AskGroundingError("unverified_intent", "this tool cannot compare seasons")
    if plan.tool != "window_query" and _needs_window_filter(subject, links, question):
        raise AskGroundingError("unverified_filter", "this tool would ignore a requested filter")
    if plan.tool == "season_compare":
        _require_metrics([validated["metric"]], subject, question, compare=True)
        _require_comparison_direction(validated, links, question)
    elif plan.tool == "player_efg":
        if not re.search(r"efg\s*%?|有效命中率", question, re.I):
            raise AskGroundingError("unverified_metric", "eFG was not requested")
        if any(re.search(pattern, question, re.I) for name, pattern in _METRIC_PATTERNS.items()
               if name not in {"efg_percentage", "three_pointers_made"}):
            raise AskGroundingError("multi_metric_out_of_scope", "the question needs another metric tool", "unsupported")
    else:
        _require_metrics(validated["metrics"], subject, question)
        if plan.tool == "window_query":
            _require_window_filters(validated, links, question)
    return validated
