"""Typed opt-in HTTP contract; the client cannot define task coverage."""
from __future__ import annotations

import time
from typing import Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.agent_tasks import build_task_profile, task_from_semantic
from app.conversation import ConversationStore
from app.agent_loop_response import loop_response
from scripts.tracing import set_attribute, set_text, span, trace_id


class LoopRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    routing_mode: Literal['model', 'rules'] | None = None

    @field_validator('question')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('question cannot be blank')
        return value.strip()


def register_loop_route(application, settings, loop):
    conversations: ConversationStore = getattr(application.state, 'agent_conversations', ConversationStore())
    application.state.agent_conversations = conversations

    def grounded_scope(task, observations):
        player_ids, team_ids, season_id, phase, metrics = [], [], None, None, []
        for requirement in task.requirements:
            observation = next((o for o in observations if o.requirement_id == requirement.id and o.status in {'complete', 'partial'}), None)
            if observation is None or requirement.capability == 'knowledge_glossary':
                continue
            args = requirement.arguments
            if args.get('subject') == 'team' and args.get('entity_id'):
                team_ids.append(str(args['entity_id']))
            elif args.get('entity_id'):
                player_ids.append(str(args['entity_id']))
            if args.get('season_id'):
                season_id = str(args['season_id'])
            elif args.get('comparison_season'):
                season_id = str(args['comparison_season'])
            if args.get('competition_phase'):
                phase = str(args['competition_phase'])
            if requirement.capability in {'player_efg', 'efg_compare'}:
                metrics.append('efg_percentage')
            metrics.extend(str(metric) for metric in args.get('metrics', []))
            if args.get('metric'):
                metrics.append(str(args['metric']))
        return {
            'player_ids': list(dict.fromkeys(player_ids)),
            'team_ids': list(dict.fromkeys(team_ids)),
            'season_id': season_id,
            'competition_phase': phase,
            'metric_terms': list(dict.fromkeys(metrics)),
        }

    @application.post('/agent/loop')
    async def loop_answer(request: LoopRequest):
        conversation_id = request.conversation_id or uuid4().hex
        if not conversations.try_acquire(conversation_id):
            raise HTTPException(409, detail={'code': 'conversation_busy', 'conversation_id': conversation_id})
        try:
            return await execute_request(request, conversation_id)
        finally:
            conversations.release(conversation_id)

    async def execute_request(request: LoopRequest, conversation_id: str):
        started = time.monotonic()
        request_id = uuid4().hex
        previous = conversations.get(conversation_id) if request.conversation_id else None
        with span(settings.tracer, 'nba.agent.loop.request') as handle:
            set_attribute(handle, 'nba.agent.request_id', request_id)
            context_status = 'continued' if previous else 'reset' if request.conversation_id else 'new'
            set_attribute(handle, 'nba.agent.conversation_context_status', context_status)
            set_attribute(handle, 'nba.agent.turn_index', (previous.turn_index + 1) if previous else 1)
            set_attribute(handle, 'nba.agent.inherited_scope_fields', [
                key for key in ('player_ids', 'team_ids', 'season_id', 'competition_phase', 'metric_terms')
                if previous is not None and getattr(previous, key, None)
            ])
            mode = request.routing_mode or settings.routing_mode
            planned = None
            if mode == 'model':
                from app.routing_service import plan_request
                planned = await plan_request(request, settings, previous, started+loop.deadline_seconds)
                task = task_from_semantic(planned)
            else:
                task = build_task_profile(settings.database, request.question, previous)
            base = {'request_id': request_id, 'trace_id': trace_id(), 'route': task.route,
                    'model': getattr(loop.decider, 'model_name', None), 'observations': [], 'actions': [],
                    'steps': [], 'evidence': [], 'warnings': [], 'decisions': 0, 'tool_calls': 0,
                    'decision_failures': [],
                    'conversation_id': conversation_id, 'context': task.context()}
            base.update(routing_mode=mode or 'rules', planning=planned.diagnostics if planned else {'planner_type': 'rules'})
            if not task.requirements:
                status = {'clarification': 'clarification_required', 'unsupported': 'unsupported'}.get(task.route, 'incomplete')
                if planned and planned.error_code and status != 'clarification_required':
                    from app.routing_service import failure_status
                    status = failure_status(planned.error_code)
                base.update(status=status, reason_code=task.reason_code, termination_reason=task.reason_code, answer=task.reason)
                base['error_stage'] = 'planning'
            else:
                execution_loop = loop
                if mode == 'model':
                    from app.agent_loop import AgentLoop
                    from app.agent_decider import NativeDecisionProvider
                    from scripts.model_adapter import OpenAICompatibleModel
                    selected_model = settings.model or OpenAICompatibleModel(config=settings.provider_config)
                    execution_loop = AgentLoop(loop.registry, decider=NativeDecisionProvider(selected_model, timeout=settings.ask_model_timeout_seconds),
                                               budget=loop.budget, deadline_seconds=loop.deadline_seconds, tracer=loop.tracer)
                result = await execution_loop.run(task.question, task=task, deadline=started+loop.deadline_seconds)
                converted = loop_response(result, task, request_id=request_id).model_dump(mode='json')
                # Retain the independent loop endpoint's historical step arguments.
                for step, observation in zip(converted['steps'], result.observations):
                    step['arguments'] = observation.arguments
                for key in ('status', 'termination_reason', 'reason_code', 'answer', 'observations',
                            'actions', 'decisions', 'tool_calls', 'steps', 'evidence', 'warnings',
                            'error_stage', 'decision_failures', 'execution_mode'):
                    base[key] = converted[key]
            observations = [type('ObservationView', (), item) for item in base.get('observations', [])]
            scope = grounded_scope(task, observations)
            usable = base['status'] in {'complete', 'partial'} and any(scope[key] for key in ('player_ids', 'team_ids', 'season_id', 'competition_phase', 'metric_terms'))
            if planned:
                from app.routing_service import save_scope
                snapshot = save_scope(conversations, conversation_id, previous, planned, base['status'])
            elif usable:
                snapshot = conversations.put(conversation_id=conversation_id, **scope)
            else:
                # Advance the turn while preserving the last grounded scope.
                snapshot = conversations.put(
                    conversation_id=conversation_id,
                    player_ids=previous.player_ids if previous else (),
                    team_ids=previous.team_ids if previous else (),
                    season_id=previous.season_id if previous else None,
                    competition_phase=previous.competition_phase if previous else None,
                    metric_terms=previous.metric_terms if previous else (),
                )
            base['turn_index'] = snapshot.turn_index
            base['context']['inherited'] = task.inherited_scope or {}
            base['timings'] = {'total_ms': round((time.monotonic()-started)*1000,3)}
            set_attribute(handle, 'nba.agent.status', base['status'])
            set_attribute(handle, 'nba.agent.reason_code', base.get('reason_code') or '')
            set_attribute(handle, 'nba.agent.turn_index', snapshot.turn_index)
            set_text(handle, 'output.value', base)
            set_attribute(handle, 'output.mime_type', 'application/json')
            return base
