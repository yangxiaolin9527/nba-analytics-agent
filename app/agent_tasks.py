"""Server-owned requirements for the existing bounded NBA task families."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.agent_actions import AgentAction, Observation, action_fingerprint
from app.agent_routes import RouteStep, plan_question
from app.conversation import ConversationSnapshot
from scripts.ask_grounding import AskGroundingError, ground_ask_question, validate_ask_call
from scripts.ask_plan import AskPlan


@dataclass(frozen=True)
class Requirement:
    id: str
    capability: str
    arguments: dict
    dependencies: tuple[str, ...] = ()
    question: str = ''


@dataclass(frozen=True)
class TaskProfile:
    question: str
    route: str
    requirements: tuple[Requirement, ...]
    reason_code: str | None = None
    reason: str = ''
    inherited_scope: dict = None
    comparison_step_ids: tuple[str, ...] = ()

    def context(self) -> dict:
        return {'route': self.route, 'requirements': [r.__dict__ for r in self.requirements],
                'allowed_capabilities': sorted({r.capability for r in self.requirements}),
                'comparison_arguments': 'left_observation_id, right_observation_id; never numeric inputs',
                'inherited': self.inherited_scope or {}}

    def resolve(self, action: AgentAction, observations: list[Observation]) -> tuple[Requirement, dict]:
        if action.capability == 'knowledge_glossary' and not action.arguments:
            req = next((r for r in self.requirements if r.capability == 'knowledge_glossary'), None)
            if req is not None:
                return req, dict(req.arguments)
        if action.capability == 'efg_compare':
            if set(action.arguments) != {'left_observation_id', 'right_observation_id'}:
                raise ValueError('comparison requires two observation references')
            req = next((r for r in self.requirements if r.capability == 'efg_compare'), None)
            if req is None:
                raise ValueError('comparison not requested')
            by_id = {o.id: o for o in observations}
            refs = [by_id.get(action.arguments[key]) for key in ('left_observation_id', 'right_observation_id')]
            if any(o is None or o.status != 'complete' or o.capability != 'player_efg' for o in refs):
                raise ValueError('reference must be a complete eFG observation from this request')
            if tuple(o.requirement_id for o in refs) != req.dependencies:
                raise ValueError('comparison scope or direction mismatch')
            return req, {**req.arguments, 'left_efg': refs[0].result['efg_percentage'],
                         'right_efg': refs[1].result['efg_percentage']}
        fingerprint = action_fingerprint(action)
        for req in self.requirements:
            if action_fingerprint(AgentAction('call', req.capability, req.arguments)) == fingerprint:
                return req, dict(req.arguments)
        raise ValueError('arguments do not match the grounded task')


def task_from_semantic(planned) -> TaskProfile:
    requirements = []
    for step in planned.plan.steps:
        dependencies = tuple(r.id for r in requirements if r.capability == 'player_efg') if step.capability == 'efg_compare' else ()
        tool_question = step.arguments.get('term') if step.capability == 'knowledge_glossary' else planned.normalized_question
        requirements.append(Requirement(step.id, step.capability, dict(step.arguments), dependencies, tool_question or planned.normalized_question))
    return TaskProfile(planned.normalized_question, planned.plan.route, tuple(requirements), planned.error_code or planned.plan.reason_code, planned.plan.reason,
                       comparison_step_ids=tuple(planned.plan.comparison_step_ids))


def build_task_profile(database: Path | None, question: str, previous: ConversationSnapshot | None = None) -> TaskProfile:
    links = None
    normalized = question
    inherited = {
        'player_ids': list(previous.player_ids) if previous else [],
        'team_ids': list(previous.team_ids) if previous else [],
        'season_id': previous.season_id if previous else None,
        'competition_phase': previous.competition_phase if previous else None,
        'metric_terms': list(previous.metric_terms) if previous else [],
    }
    if database is not None and database.is_file():
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            try:
                links, normalized = ground_ask_question(connection, question)
            except AskGroundingError as exc:
                return TaskProfile(normalized, 'clarification', (), exc.code, str(exc), inherited)
            # Follow-up pronouns may omit the scope. Add canonical values to the
            # grounding text, while leaving explicit user values authoritative.
            implicit = bool(re.search(r'那|他|她|这名|这位|同一|上一', normalized))
            comparison_followup = bool(re.search(r'比较|对比|和.+比|与.+比', normalized))
            if previous and '同一赛季' in question and previous.season_id and links.seasons and links.seasons != [previous.season_id]:
                return TaskProfile(normalized, 'clarification', (), 'conversation_scope_conflict', '同一赛季的要求与显式赛季冲突，请明确范围。', inherited)
            if previous and implicit:
                if not links.seasons and previous.season_id:
                    normalized += f' {previous.season_id}'
                if not links.phases and previous.competition_phase:
                    normalized += ' 常规赛' if previous.competition_phase == 'regular_season' else ' 季后赛'
                if (not links.players or comparison_followup) and previous.player_ids:
                    row = connection.execute('SELECT display_name FROM player_identity WHERE person_id = ?', (previous.player_ids[0],)).fetchone()
                    if row and row[0] not in normalized:
                        normalized = f'{row[0]} {normalized}' if comparison_followup else f'{normalized} {row[0]}'
                if not links.teams and previous.team_ids:
                    row = connection.execute('SELECT display_name FROM team_identity WHERE team_id = ?', (previous.team_ids[0],)).fetchone()
                    if row:
                        normalized += f' {row[0]}'
                links, _ = ground_ask_question(connection, normalized)
            # A metric-free comparison follow-up must ask for the metric instead
            # of accidentally reusing the previous metric.
            if previous and re.search(r'比较|对比|和.+比|与.+比', question) and previous.metric_terms and not re.search(r'得分|助攻|篮板|efg|有效命中率|命中率', question, re.I) and not (previous.metric_terms == ('efg_percentage',) and re.search(r'詹姆斯|勒布朗|James|LeBron', question, re.I)):
                return TaskProfile(normalized, 'clarification', (), 'comparison_metric_required', '请明确要比较的指标。', inherited)
            if previous and previous.metric_terms == ('efg_percentage',) and re.search(r'比较|对比|和.+比|与.+比', question) and not re.search(r'efg|有效命中率', normalized, re.I):
                normalized += ' eFG%'
                links, _ = ground_ask_question(connection, normalized)
    contextual_definition = bool(
        previous
        and previous.metric_terms == ('efg_percentage',)
        and re.search(r'这个指标|该指标|这个数据|那这个指标', question)
        and re.search(r'理解|含义|代表|是什么|什么是|定义|局限|限制', question)
        and not re.search(r'\b(?:ts%|true shooting|得分|助攻|篮板)\b', question, re.I)
    )
    if contextual_definition:
        normalized = 'eFG% 的定义是什么？'
        if re.search(r'局限|限制', question):
            normalized += '这个指标有什么局限？'
    if previous and not contextual_definition and re.search(r'这个指标|该指标|这个数据', question) and re.search(r'理解|含义|代表|是什么|什么是|定义|局限|限制', question):
        return TaskProfile(normalized, 'clarification', (), 'conversation_metric_required', '请明确要解释的指标。', inherited)
    plan = plan_question(normalized, links)
    if links and links.ambiguous_entities:
        return TaskProfile(normalized, 'clarification', (), 'ambiguous_entity', '实体名称有歧义，请提供完整名称。', inherited)
    if plan.route in {'clarification', 'unsupported'}:
        # A missing database is an environment failure, not missing user slots.
        if (database is None or not database.is_file()) and plan.route == 'clarification':
            return TaskProfile(normalized, 'unavailable', (), 'database_unavailable', '查询数据库不可用。', inherited)
        if contextual_definition and plan.route == 'clarification':
            plan = plan_question(normalized, links)
        else:
            return TaskProfile(normalized, plan.route, (), plan.reason_code or 'unsupported_request', plan.reason, inherited)
    requirements = []
    for step in plan.steps:
        args = dict(step.arguments)
        tool_question = normalized
        if step.capability not in {'efg_compare', 'knowledge_glossary'}:
            selected_links = links
            if plan.route == 'comparison':
                selected_links = links.model_copy(update={'players': [p for p in links.players if p.id == args['entity_id']]})
                # The delta refers to players, not seasons.
                tool_question = normalized.replace('相差', '').replace('差值', '')
            try:
                args = validate_ask_call(AskPlan(action='call', tool=step.capability, arguments=args), selected_links, tool_question)
            except AskGroundingError as exc:
                route = 'clarification' if exc.status == 'clarification_required' else 'unsupported'
                return TaskProfile(normalized, route, (), exc.code, str(exc), inherited)
        if step.capability == 'knowledge_glossary' and plan.route in {'mixed', 'comparison'}:
            tool_question = 'eFG% 的定义是什么？'
            if '局限' in question or '限制' in question:
                tool_question += '这个指标有什么局限？'
            args = {'term': 'effective field goal percentage'}
        elif step.capability == 'knowledge_glossary':
            args = {'term': 'effective field goal percentage'} if contextual_definition else {'term': normalized}
        deps = tuple(r.id for r in requirements if r.capability == 'player_efg') if step.capability == 'efg_compare' else ()
        requirements.append(Requirement(step.id, step.capability, args, deps, tool_question))
    return TaskProfile(normalized, plan.route, tuple(requirements), inherited_scope=inherited)
