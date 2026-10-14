"""Query and tool grounding checks kept separate from HTTP route code."""

from __future__ import annotations

import re

from sqlglot import exp, parse

from scripts.query_context import QueryLinks
from scripts.query_plan import QueryPlan
from app.schemas import PlanGroundingError


_DIRECT_METRICS = {
    "points_total": (r"总得分|总分|累计.{0,4}得分|total points", "points_total"),
    "games_appeared": (r"出场.{0,3}(次数|场数)|出场次数|出场场数", "games_appeared"),
    "games_played": (r"比赛场数|打了多少场|打了几场|总场数", "games_played"),
    "points_per_appearance": (r"场均得分|平均得分|每场.{0,3}得分|points per game|\bppg\b", "points_per_appearance"),
    "assists_total": (r"助攻|\bassists?\b", "assists_total"),
    "rebounds_total": (r"篮板|\brebounds?\b", "rebounds_total"),
    "efg_percentage": (r"efg\s*%?|有效命中率", "efg_percentage"),
}


def verify_plan_grounding(plan: QueryPlan, links: QueryLinks, question: str | None = None) -> None:
    if question:
        from scripts.ask_grounding import requested_metrics
        from scripts.query_plan import PLAYER_METRICS, TEAM_METRICS, QueryFamily
        requested = requested_metrics(question)
        if plan.query_family is not QueryFamily.SINGLE_GAME_SCORE:
            supported = PLAYER_METRICS if plan.player_id else TEAM_METRICS
            if plan.player_id and 'games_played' in requested:
                requested = (requested - {'games_played'}) | {'games_appeared'}
            if requested - supported:
                raise PlanGroundingError('unsupported_metric', 'a requested metric is outside the plan contract')
            if requested - set(plan.requested_metrics):
                raise PlanGroundingError('unverified_metric', 'the plan omitted a requested metric')
    if plan.player_id and plan.player_id not in {item.id for item in links.players}:
        raise PlanGroundingError("unverified_entity", "player_id is not verified for this question")
    team_ids = {item.id for item in links.teams}
    for value in (plan.team_id, plan.opponent_team_id):
        if value and value not in team_ids:
            raise PlanGroundingError("unverified_entity", "team ID is not verified for this question")
    if plan.season_id and plan.season_id not in links.seasons:
        raise PlanGroundingError("unverified_scope", "season_id is not verified for this question")
    if plan.competition_phase and plan.competition_phase.value not in links.phases:
        raise PlanGroundingError("unverified_scope", "competition_phase is not verified for this question")
    if plan.game_id:
        if len(links.games) > 1:
            raise PlanGroundingError("ambiguous_game", "multiple games match this question")
        if plan.game_id not in {item.game_id for item in links.games}:
            raise PlanGroundingError("unverified_game", "game_id is not verified for this question")


def verify_direct_sql_grounding(sql: str, question: str, links: QueryLinks) -> None:
    """Reject direct SQL that cannot be shown to answer the grounded question.

    Direct mode remains a compatibility route, but a syntactically valid query
    is insufficient: its entity, scope, and projected metric must agree with
    the deterministic links extracted from the question.
    """
    try:
        statements = parse(sql, read="sqlite")
        if len(statements) != 1:
            raise PlanGroundingError("unverified_sql_shape", "SQL must contain exactly one statement")
        query = statements[0]
    except PlanGroundingError:
        raise
    except Exception as exc:
        raise PlanGroundingError("invalid_sql", "SQL could not be parsed") from exc
    if not isinstance(query, exp.Select):
        if isinstance(query, (exp.Union, exp.Intersect, exp.Except)):
            raise PlanGroundingError("unverified_sql_shape", "set operations are not groundable in direct mode")
        # Preserve the legacy gateway's rejection contract for non-SELECT SQL;
        # the read-only gateway remains authoritative for this safety boundary.
        return

    if query.args.get("with_") is not None or len(list(query.find_all(exp.Select))) > 1:
        raise PlanGroundingError("unverified_sql_shape", "nested SELECT and CTE shapes are not groundable in direct mode")

    values: dict[str, set[str]] = {}
    where = query.args.get("where")
    if (links.players or links.teams or links.seasons or links.phases) and where and (where.find(exp.Or) or where.find(exp.Not)):
        raise PlanGroundingError("unverified_sql_scope", "SQL scope cannot be verified through OR or NOT")
    for predicate in (where.find_all(exp.EQ) if where else []):
        if isinstance(predicate.left, exp.Column) and isinstance(predicate.right, exp.Literal):
            values.setdefault(predicate.left.name, set()).add(str(predicate.right.this))
        elif isinstance(predicate.right, exp.Column) and isinstance(predicate.left, exp.Literal):
            values.setdefault(predicate.right.name, set()).add(str(predicate.left.this))

    expected_entities = {item.id for item in links.players} | {item.id for item in links.teams}
    entity_columns = values.get("person_id", set()) | values.get("team_id", set())
    if expected_entities and (not entity_columns or not entity_columns <= expected_entities):
        raise PlanGroundingError("unverified_sql_entity", "SQL entity is not grounded in the question")
    if links.seasons and values.get("season_id") != set(links.seasons):
        raise PlanGroundingError("unverified_sql_scope", "SQL season is not grounded in the question")
    if links.phases and values.get("competition_phase") != set(links.phases):
        raise PlanGroundingError("unverified_sql_scope", "SQL competition phase is not grounded in the question")

    projected: set[str] = set()
    for projection in query.expressions:
        projected.update(column.name for column in projection.find_all(exp.Column))
    requested = {
        metric for pattern, metric in _DIRECT_METRICS.values()
        if re.search(pattern, question, re.IGNORECASE)
    }
    aliases = {
        "points_total": {"points_total", "points"},
        "games_appeared": {"games_appeared", "games"},
        "games_played": {"games_played", "games"},
        "points_per_appearance": {"points_per_appearance"},
        "assists_total": {"assists_total", "assists"},
        "rebounds_total": {"rebounds_total", "rebounds"},
        "efg_percentage": {"efg_percentage", "efg"},
    }
    if any(not projected & aliases.get(metric, {metric}) for metric in requested):
        raise PlanGroundingError("unverified_sql_metric", "SQL projection does not include the requested metric")
