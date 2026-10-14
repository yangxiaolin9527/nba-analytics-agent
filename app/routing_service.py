"""Shared model/rules planning, execution and conversational scope updates."""
from __future__ import annotations

import sqlite3
import logging
import time
from uuid import uuid4

from fastapi import HTTPException
from app.agent_routes import AgentResponse, StructuredToolAdapter, execute_agent_plan
from app.context import database_evidence, query_context
from app.schemas import AskResponse
from app.semantic_planner import SemanticPlanResult, plan_semantic
from scripts.model_adapter import OpenAICompatibleModel
from scripts.query_context import QueryLinks
from scripts.tracing import mark_error, mark_success, set_attribute, set_text, span, trace_id
from app.execution_events import emit_execution_event

logger = logging.getLogger(__name__)


def failure_status(code):
    runtime_or_contract = {'provider_failed', 'model_timeout', 'invalid_arguments', 'invalid_tool_calls',
                           'native_tools_unavailable', 'database_unavailable', 'model_unconfigured', 'unregistered_capability'}
    return 'incomplete' if code in runtime_or_contract else 'unsupported'


async def plan_request(request, settings, previous, deadline, *, structured_only=False):
    mode = request.routing_mode or settings.routing_mode or 'model'
    if mode == 'model':
        try:
            model = settings.model or OpenAICompatibleModel(config=settings.provider_config)
        except ValueError:
            from app.agent_routes import RoutePlan
            return SemanticPlanResult(RoutePlan(route='unsupported', reason='模型尚未配置。', reason_code='model_unconfigured'), request.question, QueryLinks(), {'planner_type': 'model'}, 'model_unconfigured')
        planned = await plan_semantic(request.question, settings.database, model, previous=previous,
                                   timeout=settings.ask_model_timeout_seconds, deadline=deadline,
                                   structured_only=structured_only, tracer=settings.tracer)
        if structured_only and planned.plan.steps and (planned.plan.route != 'structured' or len(planned.plan.steps) != 1):
            from app.agent_routes import RoutePlan
            planned.plan = RoutePlan(route='unsupported', reason='组合问题请使用统一 Agent 入口。', reason_code='unsupported_request')
            planned.pending = {}
        return planned
    from app.agent_tasks import build_task_profile
    from app.agent_routes import RoutePlan, RouteStep
    task = build_task_profile(settings.database, request.question, previous)
    if task.route not in {'structured', 'knowledge', 'mixed', 'comparison', 'clarification', 'unsupported'}:
        return SemanticPlanResult(RoutePlan(route='unsupported', reason=task.reason, reason_code=task.reason_code), task.question, QueryLinks(), {'planner_type': 'rules'}, task.reason_code)
    plan = RoutePlan(route=task.route, steps=[RouteStep(id=r.id, capability=r.capability, arguments=r.arguments) for r in task.requirements], reason=task.reason, reason_code=task.reason_code)
    if structured_only and (plan.route != 'structured' and plan.steps):
        plan = RoutePlan(route='unsupported', reason='结构化入口只支持单个统计工具。', reason_code='unsupported_request')
    return SemanticPlanResult(plan, task.question, QueryLinks(), {'planner_type': 'rules'})


def save_scope(store, conversation_id, previous, planned, status, *, response=None):
    preserved = {key: getattr(previous, key, ()) if key in {'player_ids', 'team_ids', 'metric_terms'} else getattr(previous, key, None)
                 for key in ('player_ids', 'team_ids', 'season_id', 'competition_phase', 'metric_terms')}
    accepted = planned.plan.steps and status in {'complete', 'partial', 'not_computable'}
    if accepted:
        players, teams, metrics = [], [], []
        seasons, phases = [], []
        for step in planned.plan.steps:
            if response is not None:
                if response.execution_mode == 'bounded_loop':
                    accepted_step = any(item.get('requirement_id') == step.id
                        and item.get('status') in {'complete', 'partial'} for item in response.observations)
                else:
                    accepted_step = any(item.id == step.id and item.status in {'complete', 'partial'}
                                        for item in response.steps)
                if not accepted_step:
                    continue
            args = step.arguments
            if step.capability == 'knowledge_glossary':
                continue
            if args.get('entity_id'):
                (teams if args.get('subject') == 'team' else players).append(args['entity_id'])
            seasons.extend([args['season_id']] if args.get('season_id') else [args['comparison_season']] if args.get('comparison_season') else [])
            phases.extend([args['competition_phase']] if args.get('competition_phase') else [])
            metrics.extend(args.get('metrics', []))
            if args.get('metric'):
                metrics.append(args['metric'])
            if step.capability == 'player_efg':
                metrics.append('efg_percentage')
        if players or teams:
            preserved.update(player_ids=list(dict.fromkeys(players)), team_ids=list(dict.fromkeys(teams)), metric_terms=list(dict.fromkeys(metrics)))
            if len(set(seasons)) == 1:
                preserved['season_id'] = seasons[0]
            if len(set(phases)) == 1:
                preserved['competition_phase'] = phases[0]
    return store.put(conversation_id=conversation_id, **preserved, pending=planned.pending)


async def run_routed_agent(request, settings, registry, conversation_id):
    started = time.monotonic()
    request_id = uuid4().hex
    previous = settings.conversations.get(conversation_id)
    deadline = started + settings.ask_total_timeout_seconds
    with span(settings.tracer, 'nba.agent.semantic_request') as handle:
        set_attribute(handle, 'nba.agent.request_id', request_id)
        set_attribute(handle, 'nba.agent.routing_mode', request.routing_mode or settings.routing_mode or 'model')
        set_text(handle, 'nba.agent.question', request.question)
        context_status = 'continued' if previous else 'reset' if request.conversation_id else 'new'
        set_attribute(handle, 'nba.agent.conversation_context_status', context_status)
        set_attribute(handle, 'nba.agent.turn_index', (previous.turn_index + 1) if previous else 1)
        inherited_fields = [key for key in ('player_ids', 'team_ids', 'season_id', 'competition_phase', 'metric_terms')
                            if previous is not None and getattr(previous, key, None)]
        set_attribute(handle, 'nba.agent.inherited_scope_fields', inherited_fields)
        emit_execution_event('planning', 'running', '模型规划')
        with span(settings.tracer, 'nba.agent.semantic_planner', 'CHAIN') as planner:
            set_attribute(planner, 'nba.agent.conversation_context_status', context_status)
            set_attribute(planner, 'nba.agent.inherited_scope_fields', inherited_fields)
            try:
                planned = await plan_request(request, settings, previous, deadline)
                set_text(planner, 'nba.agent.normalized_question', planned.normalized_question)
                set_attribute(planner, 'nba.agent.planner_type', planned.diagnostics.get('planner_type'))
                set_attribute(planner, 'nba.agent.route', planned.plan.route)
                set_attribute(planner, 'nba.agent.step_count', len(planned.plan.steps))
                set_attribute(planner, 'nba.agent.capabilities', [step.capability for step in planned.plan.steps])
                set_attribute(planner, 'nba.agent.planning_diagnostics', planned.diagnostics)
                set_attribute(planner, 'nba.agent.status', 'failed' if planned.error_code else 'complete')
                if planned.error_code:
                    set_attribute(planner, 'nba.agent.reason_code', planned.error_code)
                    mark_error(planner, RuntimeError(planned.error_code), reason_code=planned.error_code)
                else:
                    mark_success(planner)
            except Exception as exc:
                mark_error(planner, exc)
                raise
        emit_execution_event('planning', 'complete' if not planned.error_code else 'failed',
                             '模型规划完成' if not planned.error_code else '模型规划失败',
                             metadata={'route': planned.plan.route, 'reason_code': planned.error_code})
        if planned.error_code and planned.plan.route != 'clarification':
            status = failure_status(planned.error_code)
            response = AgentResponse(status=status, request_id=request_id, route=planned.plan.route,
                                     answer=planned.plan.reason, reason_code=planned.error_code, error_stage='planning')
            emit_execution_event('result', 'failed', 'Agent 规划失败', metadata={
                'status': status, 'reason_code': planned.error_code,
                'error_stage': 'planning', 'execution_mode': 'bounded_plan'})
        elif len(planned.plan.steps) > settings.agent_max_tool_calls:
            response = AgentResponse(status='incomplete', request_id=request_id, route=planned.plan.route,
                                     answer='工具调用预算不足。', reason_code='tool_budget_exceeded')
        else:
            eligible = (settings.agent_unified_loop_enabled
                        and (request.routing_mode or settings.routing_mode) == 'model'
                        and planned.plan.route in {'mixed', 'comparison', 'composed'})
            with span(settings.tracer, 'nba.agent.orchestrator') as orchestrator:
                mode = 'bounded_loop' if eligible else 'bounded_plan'
                set_attribute(orchestrator, 'nba.agent.execution_mode', mode)
                set_attribute(orchestrator, 'nba.agent.delegation_enabled', settings.agent_unified_loop_enabled)
                set_attribute(orchestrator, 'nba.agent.delegation_reason',
                              'complex_model_plan' if eligible else 'bounded_plan_policy')
                set_attribute(orchestrator, 'nba.agent.route', planned.plan.route)
                invalid_registry = registry.validate(planned.plan) if eligible else None
                if invalid_registry:
                    response = AgentResponse(status='unsupported', request_id=request_id,
                        route=planned.plan.route, answer='当前 Agent capability 未注册。',
                        reason_code=invalid_registry, error_stage='planning')
                    set_attribute(orchestrator, 'nba.agent.delegation_reason', invalid_registry)
                elif eligible:
                    from app.agent_loop import AgentLoop
                    from app.agent_actions import LoopBudget
                    from app.agent_decider import NativeDecisionProvider
                    from app.agent_tasks import task_from_semantic
                    from app.agent_loop_response import loop_response
                    decider = settings.agent_loop_decider
                    if decider is None:
                        model = settings.model or OpenAICompatibleModel(config=settings.provider_config)
                        decider = NativeDecisionProvider(model, timeout=settings.ask_model_timeout_seconds)
                    task = task_from_semantic(planned)
                    with span(settings.tracer, 'nba.agent.loop.request') as loop_span:
                        loop = AgentLoop(registry, decider=decider,
                            budget=LoopBudget(max_tool_calls=settings.agent_max_tool_calls),
                            deadline_seconds=settings.ask_total_timeout_seconds, tracer=settings.tracer)
                        result = await loop.run(task.question, task=task, deadline=deadline)
                        response = loop_response(result, task, request_id=request_id, tracer=settings.tracer)
                        emit_execution_event('result', 'complete' if result.status == 'complete' else 'failed',
                            'Agent 执行完成', metadata={'status': result.status,
                                'reason_code': response.reason_code, 'execution_mode': 'bounded_loop'})
                        set_text(loop_span, 'output.value', response.model_dump(mode='json'))
                        set_attribute(loop_span, 'nba.agent.status', response.status)
                        if response.status == 'incomplete':
                            mark_error(loop_span, RuntimeError(result.termination_reason), reason_code=result.termination_reason)
                        else:
                            mark_success(loop_span)
                else:
                    response = await execute_agent_plan(planned.normalized_question, planned.plan, registry,
                        request_id=request_id, deadline_seconds=max(0, deadline-time.monotonic()),
                        tracer=settings.tracer, request_span=orchestrator, skip_planner_span=True, validate_outputs=True)
                set_attribute(orchestrator, 'nba.agent.status', response.status)
                set_attribute(orchestrator, 'nba.agent.decisions', response.decisions)
                set_attribute(orchestrator, 'nba.agent.tool_calls', response.tool_calls or len(response.steps))
                set_attribute(orchestrator, 'nba.agent.termination_reason', response.termination_reason or '')
                if response.status == 'incomplete':
                    mark_error(orchestrator, RuntimeError(response.reason_code or 'incomplete'), reason_code=response.reason_code)
                else:
                    mark_success(orchestrator)
        snapshot = save_scope(settings.conversations, conversation_id, previous, planned, response.status,
                              response=response)
        if response.status in {'clarification_required', 'unsupported'} and response.reason_code:
            response = response.model_copy(update={'error_stage': 'planning'})
        logger.info('agent request_id=%s planner=%s route=%s status=%s reason=%s', request_id,
                    planned.diagnostics.get('planner_type'), planned.plan.route, response.status, response.reason_code)
        final_response = response.model_copy(update={'conversation_id': conversation_id, 'turn_index': snapshot.turn_index,
            'trace_id': trace_id(),
            'tool_calls': response.tool_calls if response.execution_mode == 'bounded_loop' else len(response.steps),
            'conversation_context_status': 'continued' if previous else 'reset' if request.conversation_id else 'new',
            'routing_mode': planned.diagnostics.get('planner_type'), 'planning': planned.diagnostics,
            'timings': {**response.timings, 'planning_ms': planned.diagnostics.get('elapsed_ms', 0), 'total_ms': round((time.monotonic()-started)*1000, 3)}})
        set_attribute(handle, 'nba.agent.status', final_response.status)
        set_attribute(handle, 'nba.agent.reason_code', final_response.reason_code or '')
        set_attribute(handle, 'nba.agent.error_stage', final_response.error_stage or '')
        set_attribute(handle, 'nba.agent.turn_index', snapshot.turn_index)
        set_text(handle, 'output.value', final_response.model_dump(mode='json'))
        set_attribute(handle, 'output.mime_type', 'application/json')
        if final_response.status == 'incomplete':
            mark_error(handle, RuntimeError(final_response.reason_code or final_response.status),
                       reason_code=final_response.reason_code or final_response.status)
        else:
            mark_success(handle)
        return final_response


async def run_routed_ask(request, settings, *, request_id=None):
    started = time.monotonic()
    conversation_id = request.conversation_id or uuid4().hex
    store = settings.conversations
    if not store.try_acquire(conversation_id):
        raise HTTPException(409, detail={'code': 'conversation_busy', 'conversation_id': conversation_id})
    try:
        previous = store.get(conversation_id)
        deadline = started + settings.ask_total_timeout_seconds
        planned = await plan_request(request, settings, previous, deadline, structured_only=True)
        tool_calls = 0
        error_stage = 'planning'
        tool_started = None
        if planned.plan.route == 'clarification':
            status, code, result = 'clarification_required', planned.plan.reason_code, None
        elif not planned.plan.steps or planned.error_code:
            code = planned.error_code or planned.plan.reason_code
            status = failure_status(code)
            result = None
        elif settings.ask_max_tool_calls < 1:
            status, code, result = 'incomplete', 'tool_budget_exceeded', None
        else:
            step = planned.plan.steps[0]
            error_stage = 'execution'
            adapter = StructuredToolAdapter(step.capability, settings)
            from app.agent_loop import validate_result
            try:
                import asyncio
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('request deadline reached')
                tool_calls = 1
                tool_started = time.monotonic()
                result = await asyncio.wait_for(adapter.execute(planned.normalized_question, step.arguments), remaining)
                status = validate_result(step.capability, result, step.arguments)
                status = 'incomplete' if status == 'failed' else status
                code = result.get('reason_code')
                error_stage = result.get('error_stage', error_stage)
            except TimeoutError:
                status, code, result = 'incomplete', 'tool_timeout', None
            except (ValueError, KeyError, TypeError):
                status, code, result = 'incomplete', 'invalid_adapter_result', None
                error_stage = 'validation'
        step = planned.plan.steps[0] if planned.plan.steps else None
        evidence = None
        if result is not None:
            with sqlite3.connect(settings.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                context, source = query_context()
                evidence = database_evidence(connection, context, source)
        snapshot = save_scope(store, conversation_id, previous, planned, status)
        return AskResponse(status=status, reason_code=code, reason=planned.plan.reason,
            error_stage=None if status in {'complete', 'partial'} else error_stage,
            tool=step.capability if step else None, arguments=step.arguments if step else None,
            tool_result=result, resolved_values=planned.links, evidence=evidence,
            tool_calls_used=tool_calls,
            tool_elapsed_ms=round((time.monotonic()-tool_started)*1000, 3) if tool_started is not None else 0,
            elapsed_ms=round((time.monotonic()-started)*1000, 3), model_elapsed_ms=planned.diagnostics.get('elapsed_ms', 0),
            conversation_id=conversation_id, turn_index=snapshot.turn_index,
            conversation_context_status='continued' if previous else 'reset' if request.conversation_id else 'new',
            routing_mode=planned.diagnostics['planner_type'], planning=planned.diagnostics)
    finally:
        store.release(conversation_id)
