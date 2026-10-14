"""Language-facing tool contracts; execution IDs are supplied by the server."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, RootModel

# Execution capacity, not a restriction on the meaning of comparison.
MAX_PLAN_STEPS = 4


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    continue_pending: bool = False


class EntityReference(Contract):
    mention: str | None = Field(default=None, max_length=120)
    canonical_name: str | None = Field(default=None, max_length=120)
    inherit: bool = False


class Scope(Contract):
    season_id: str | None = None
    competition_phase: Literal['regular_season', 'playoffs'] | None = None
    inherit_scope: list[Literal['season_id', 'competition_phase', 'metrics']] = Field(default_factory=list)
    scope_evidence: dict[str, str] = Field(default_factory=dict)


class SeasonSummary(Scope):
    subject: Literal['player', 'team']
    entity: EntityReference
    metrics: list[str] = Field(min_length=1, max_length=5)


class WindowQuery(SeasonSummary):
    opponent: EntityReference | None = None
    date_from: str | None = None
    date_to: str | None = None
    last_n: int | None = Field(default=None, ge=1, le=100)


class SeasonCompare(Scope):
    subject: Literal['player', 'team']
    entity: EntityReference
    metric: str
    baseline_season: str | None = None
    comparison_season: str | None = None


class PlayerEfg(Scope):
    entity: EntityReference
    knowledge_question: str | None = Field(default=None, min_length=1, max_length=1000,
        description='Verbatim contiguous knowledge subquestion from the current question. Do not rewrite, translate, or add topics; downstream retrieval handles rewriting.')


class ComparePlayerEfg(Scope):
    left: EntityReference
    right: EntityReference


class SingleGame(Contract):
    game_id: str | None = None
    game_date: str | None = None
    home_or_away_teams: list[EntityReference] = Field(default_factory=list, max_length=2)


class Knowledge(Contract):
    question: str = Field(min_length=1, max_length=1000)


class Clarify(Contract):
    missing: list[str] = Field(min_length=1, max_length=5)
    reason: str = Field(min_length=1, max_length=500)


class Unsupported(Contract):
    reason: str = Field(min_length=1, max_length=500)


class SummaryQueryCall(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    capability: Literal['season_summary']
    arguments: SeasonSummary


class WindowQueryCall(SummaryQueryCall):
    capability: Literal['window_query']
    arguments: WindowQuery


class EfgQueryCall(SummaryQueryCall):
    capability: Literal['player_efg']
    arguments: PlayerEfg


class KnowledgeQueryCall(SummaryQueryCall):
    capability: Literal['knowledge_glossary']
    arguments: Knowledge


class SummaryCall(SummaryQueryCall):
    subquestion: str | None = Field(default=None, min_length=1, max_length=1000,
        description='For heterogeneous numeric requests, quote the contiguous current-user subquestion assigned to this call. Include its entity mention when entities need different metrics. Do not rewrite; shared scope/filters remain server-validated.')


class WindowCall(SummaryCall):
    capability: Literal['window_query']
    arguments: WindowQuery


class EfgCall(SummaryCall):
    capability: Literal['player_efg']
    arguments: PlayerEfg


class KnowledgeCall(SummaryCall):
    capability: Literal['knowledge_glossary']
    arguments: Knowledge


class CollectWorkflow(Contract):
    mode: Literal['collect']
    calls: list[Annotated[SummaryCall | WindowCall | EfgCall | KnowledgeCall,
                          Field(discriminator='capability')]] = Field(min_length=2, max_length=MAX_PLAN_STEPS)


class CompareWorkflow(Contract):
    mode: Literal['compare']
    calls: list[Annotated[SummaryQueryCall | WindowQueryCall | EfgQueryCall | KnowledgeQueryCall,
                          Field(discriminator='capability')]] = Field(min_length=2, max_length=MAX_PLAN_STEPS,
        description='One numeric call per target with shared capability/metrics/scope/filters, in user target order; optional separate quoted knowledge call. Shared numeric queries do not require invented per-target subquestions.')


class ComposedWorkflow(RootModel[Annotated[CollectWorkflow | CompareWorkflow,
                                          Field(discriminator='mode')]]):
    pass


CONTRACTS = {
    'season_summary': (SeasonSummary, 'Season totals/averages: player points_total, games_appeared, points_per_appearance, assists_total, rebounds_total; team points_total, games_played, assists_total, rebounds_total. Return all requested metrics in one call. No assists/rebounds per-game averages.'),
    'window_query': (WindowQuery, 'Same metrics as season_summary, filtered by opponent, ISO dates or last_n appearances (1-100).'),
    'season_compare': (SeasonCompare, 'One player/team across two seasons. Player points_total, games_appeared, points_per_appearance, assists_total, rebounds_total; team points_total, games_played, assists_total, rebounds_total. Preserve baseline -> comparison direction.'),
    'single_game_score': (SingleGame, 'Final team scores and winner for exactly one identifiable game. No individual player game statistics.'),
    'player_efg': (PlayerEfg, 'Compute effective field goal percentage for one player, season and competition phase.'),
    'compare_player_efg': (ComparePlayerEfg, 'Existing bounded comparison: two players in the same season and phase, deterministic eFG difference and cited eFG explanation. Left minus right.'),
    'composed_workflow': (ComposedWorkflow, 'Compose existing atomic calls within execution capacity; this selects a plan, not a new tool. mode=compare: one numeric call for EACH of two or more targets, identical capability/metrics/scope/filters, user target order; server projects verified results into a comparison matrix. No numeric subquestion rewrites. Optional quoted knowledge call. mode=collect: heterogeneous requested results with verbatim subquestion evidence for entity-specific metrics. Never omit an entity/metric or invent tools.'),
    'knowledge_glossary': (Knowledge, 'Answer NBA rulebook/rules and statistics glossary questions from pinned documents. No player statistics, predictions, news or web search. For mixed eFG tasks ask only the definition/limitations subquestion.'),
    'clarify': (Clarify, 'Ask for genuinely missing/ambiguous information. Never guess a missing season or competition phase.'),
    'unsupported': (Unsupported, 'Reject requests outside registered capabilities; do not silently substitute another metric or omit requirements.'),
}


def tool_schemas(*, structured_only=False):
    names = set(CONTRACTS)
    if structured_only:
        names -= {'knowledge_glossary', 'compare_player_efg', 'composed_workflow'}
    return [{'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': model.model_json_schema()}}
            for name, (model, description) in CONTRACTS.items() if name in names]


def plan_selector_schema(*, structured_only=False):
    """One wire function selects one complete, existing capability contract."""
    choices, definitions = [], {}
    for tool in tool_schemas(structured_only=structured_only):
        fn = tool['function']
        schema = fn['parameters']
        definitions.update(schema.pop('$defs', {}))
        choices.append({'type': 'object', 'properties': {
            'capability': {'type': 'string', 'enum': [fn['name']], 'description': fn['description']},
            'arguments': schema}, 'required': ['capability', 'arguments'], 'additionalProperties': False})
    return {'type': 'function', 'function': {'name': 'select_plan',
        'description': 'Select exactly one complete bounded workflow for the full request. This selects a plan; it executes no tools.',
        'parameters': {'type': 'object', 'oneOf': choices, '$defs': definitions}}}
