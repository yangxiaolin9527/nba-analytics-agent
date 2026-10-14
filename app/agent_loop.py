"""Observation-driven loop with server-owned requirements and hard budgets."""
from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import math
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from app.agent_actions import AgentAction, ActionValidationError, ACTION_SCHEMA_HINT, CoverageLedger, LoopBudget, Observation, action_fingerprint
from app.agent_tasks import TaskProfile
from scripts.tracing import mark_error, mark_success, set_attribute, set_text, span
from app.execution_events import emit_execution_event

RETRYABLE = {'tool_timeout', 'generation_timeout', 'retrieval_timeout', 'verification_timeout', 'temporary_unavailable'}


@dataclass(frozen=True)
class LoopResult:
    status: str
    observations: tuple[Observation, ...]
    termination_reason: str
    decisions: int
    tool_calls: int
    actions: tuple[dict, ...] = ()
    decision_failures: tuple[dict[str, Any], ...] = ()


Decider = Callable[[dict[str, Any]], AgentAction | dict[str, Any] | Awaitable[AgentAction | dict[str, Any]]]


def validate_result(capability: str, result: dict, arguments: dict) -> str:
    """Reject successful-looking but unusable or malformed observations."""
    from app.agent_routes import RouteStep, _bundle
    from scripts.answer_renderer import render_tool_answer
    status = result.get('status')
    if status not in {'complete', 'partial', 'answered', 'failed', 'not_computable', 'unsupported', 'incomplete', 'abstained'}:
        raise ValueError('invalid adapter status')
    normalized = 'partial' if status == 'partial' else 'complete' if status in {'complete', 'answered'} else 'failed'
    bundle = _bundle(RouteStep(id='validation', capability=capability, arguments=arguments), normalized, result)
    if status not in {'complete', 'partial', 'answered'}:
        return 'not_computable' if status == 'not_computable' else 'failed'
    if capability == 'knowledge_glossary':
        if not isinstance(result.get('answer'), str) or not result['answer'].strip():
            raise ValueError('missing knowledge answer')
        if status != 'partial' and not any(c.get('citation_id') and c.get('excerpt') for c in result.get('claims', []) if isinstance(c, dict)):
            raise ValueError('missing citations')
    else:
        if capability == 'player_efg':
            values = [result.get('efg_percentage')]
            inputs = result.get('inputs') or {}
            values += [inputs.get(k) for k in ('field_goals_made','three_pointers_made','field_goals_attempted','sample_count')]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                fgm, threes, fga, count = values[1:]
                if fga <= 0 or count <= 0 or not math.isclose(values[0], (fgm + .5 * threes) / fga * 100, rel_tol=1e-9):
                    raise ValueError('invalid eFG calculation')
        elif capability == 'season_summary':
            values = [result.get(k) for k in arguments['metrics']]
        elif capability == 'window_query':
            values = [(result.get('values') or {}).get(k) for k in arguments['metrics']] + [result.get('actual_sample_count')]
        elif capability == 'season_compare':
            values = [result.get('absolute_delta'), (result.get('baseline') or {}).get('value'), (result.get('comparison') or {}).get('value')]
        elif capability == 'single_game_score':
            values = [result.get('home_score'), result.get('away_score')]
            if not result.get('winner_team_id'):
                raise ValueError('missing winner')
        elif capability == 'efg_compare':
            values = [result.get(k) for k in ('left_efg_percentage','right_efg_percentage','absolute_delta')]
            if values[0] != arguments['left_efg'] or values[1] != arguments['right_efg'] or values[2] != arguments['left_efg'] - arguments['right_efg']:
                raise ValueError('invalid comparison')
        else:
            raise ValueError('unknown capability')
        if not values or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in values):
            raise ValueError('missing finite numeric result')
        # Validate the same rendering boundary before the ledger credits coverage.
        render_tool_answer(capability, arguments, result, evidence=bundle.structured.model_dump(mode='json'))
    return 'partial' if status == 'partial' else 'complete'


class AgentLoop:
    def __init__(self, registry, *, decider: Decider | None = None,
                 budget: LoopBudget | None = None, deadline_seconds: float = 60.0, tracer=None):
        if deadline_seconds <= 0:
            raise ValueError('deadline must be positive')
        self.registry, self.decider = registry, decider
        self.budget = budget or LoopBudget()
        self.deadline_seconds, self.tracer = deadline_seconds, tracer

    async def run(self, question: str, *, task: TaskProfile, deadline: float | None = None) -> LoopResult:
        if not task.requirements:
            return LoopResult('incomplete', (), 'empty_task', 0, 0)
        if self.decider is None:
            return LoopResult('incomplete', (), 'decider_unconfigured', 0, 0)
        ledger = CoverageLedger(tuple(r.id for r in task.requirements))
        observations, actions = [], []
        decision_failures = []
        latest = {}
        decisions = calls = retries = corrections = decision_repairs = 0
        feedback = None
        deadline = deadline if deadline is not None else time.monotonic() + self.deadline_seconds

        def end(code, *, finished=False):
            usable = any(o.status in {'complete','partial'} for o in observations)
            if finished and ledger.complete:
                status = 'complete'
            elif usable:
                status = 'partial'
            elif any(o.status == 'not_computable' for o in observations):
                status = 'not_computable'
            else:
                status = 'incomplete'
            result = LoopResult(status, tuple(observations), code, decisions, calls, tuple(actions), tuple(decision_failures))
            with span(self.tracer, 'nba.agent.loop.termination') as termination:
                set_attribute(termination, 'nba.agent.status', status)
                set_attribute(termination, 'nba.agent.termination_reason', code)
                set_attribute(termination, 'nba.agent.decisions', decisions)
                set_attribute(termination, 'nba.agent.tool_calls', calls)
                set_attribute(termination, 'nba.agent.coverage_missing', ledger.missing)
                mark_error(termination, RuntimeError(code), reason_code=code) if status == 'incomplete' else mark_success(termination)
            emit_execution_event('termination', 'complete' if status in {'complete', 'partial'} else 'failed',
                                 f'循环终止：{code}', metadata={
                                     'termination_reason': code, 'execution_mode': 'bounded_loop',
                                     'decisions': decisions, 'tool_calls': calls, 'status': status,
                                 })
            return result

        def record_failure(exc, default_code):
            diagnostic = {
                'sequence': decisions,
                'code': exc.code if isinstance(exc, ActionValidationError) else default_code,
                'detail_code': exc.detail_code if isinstance(exc, ActionValidationError) else default_code,
                'elapsed_ms': round((time.monotonic() - started) * 1000, 3),
            }
            decision_failures.append(diagnostic)
            emit_execution_event('loop_decision', 'failed', f'决策 #{decisions} 失败',
                                 elapsed_ms=diagnostic['elapsed_ms'], metadata={
                                     'decision_number': decisions, 'reason_code': diagnostic['code'],
                                     'execution_mode': 'bounded_loop'})
            return diagnostic

        def requirements_settled() -> bool:
            observed = {o.requirement_id for o in observations}
            return all(requirement.id in observed for requirement in task.requirements)

        async def wait_call(fn, *args):
            # Synchronous injected implementations are also bounded without blocking the loop.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, '__call__', None)):
                output = await asyncio.wait_for(fn(*args), remaining)
            else:
                # Keep synchronous injected deciders bounded without using
                # asyncio's default executor.  Python 3.14 can wait forever
                # while asyncio.run() shuts down that executor after a short
                # lived request, even when its worker is already idle.
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nba-agent-loop')
                future = executor.submit(contextvars.copy_context().run, fn, *args)
                try:
                    while not future.done():
                        if time.monotonic() >= deadline:
                            raise TimeoutError()
                        await asyncio.sleep(0.01)
                    output = future.result()
                finally:
                    future.cancel()
                    executor.shutdown(wait=future.done(), cancel_futures=True)
            if inspect.isawaitable(output):
                output = await asyncio.wait_for(output, max(.000001, deadline-time.monotonic()))
            if time.monotonic() >= deadline:
                raise TimeoutError()
            return output

        while decisions < self.budget.max_decisions:
            if time.monotonic() >= deadline:
                return end('deadline_exceeded')
            context = {'question': question, 'task': task.context(),
                       'observations': [o.__dict__ for o in observations], 'coverage': {'missing': ledger.missing},
                       'feedback': feedback,
                       'budget': {'decisions': decisions, 'tool_calls': calls, 'max_decisions': self.budget.max_decisions,
                                  'max_tool_calls': self.budget.max_tool_calls, 'remaining_seconds': deadline-time.monotonic()}}
            context['budget']['remaining_retries'] = self.budget.max_retries - retries
            if len(json.dumps(context, ensure_ascii=False)) > 24000:
                return end('context_budget_exceeded')
            decisions += 1
            started = time.monotonic()
            emit_execution_event('loop_decision', 'running', f'决策 #{decisions}',
                                 metadata={'decision_number': decisions, 'execution_mode': 'bounded_loop'})
            handle = None
            try:
                with span(self.tracer, 'nba.agent.loop.decision') as handle:
                    set_attribute(handle, 'nba.agent.decision_number', decisions)
                    set_attribute(handle, 'nba.agent.model', getattr(self.decider, 'model_name', 'injected'))
                    set_attribute(handle, 'nba.agent.decision_protocol', getattr(self.decider, 'protocol', 'json-action-v1'))
                    set_text(handle, 'input.value', context)
                    set_attribute(handle, 'input.mime_type', 'application/json')
                    decision_error = None
                    try:
                        raw = await wait_call(self.decider, context)
                        action = raw if isinstance(raw, AgentAction) else AgentAction.from_mapping(raw)
                    except Exception as exc:
                        code = exc.code if isinstance(exc, ActionValidationError) else 'decision_timeout' if isinstance(exc, TimeoutError) else 'decision_failed'
                        detail = exc.detail_code if isinstance(exc, ActionValidationError) else code
                        set_attribute(handle, 'nba.agent.decision_detail_code', detail)
                        metadata = {key: value for key, value in getattr(exc, 'diagnostics', {}).items()
                                    if key in {'native_call_count', 'native_tool_names'}}
                        for key, value in metadata.items():
                            set_attribute(handle, 'nba.agent.' + key, value)
                        set_text(handle, 'output.value', {'code': code, 'detail_code': detail, **metadata})
                        set_attribute(handle, 'output.mime_type', 'application/json')
                        # Provider/injected exception messages may contain payloads;
                        # export fixed diagnostics, never their arbitrary text.
                        mark_error(handle, RuntimeError(code + ':' + detail), reason_code=code)
                        decision_error = exc
                    if decision_error is None:
                        set_attribute(handle, 'nba.agent.decision_number', decisions)
                        set_attribute(handle, 'nba.agent.action', action.action)
                        set_attribute(handle, 'nba.agent.capability', action.capability or '')
                        set_attribute(handle, 'nba.agent.coverage_missing', ledger.missing)
                        set_attribute(handle, 'nba.agent.remaining_seconds', context['budget']['remaining_seconds'])
                        set_text(handle, 'output.value', {
                            'action': action.action,
                            'capability': action.capability,
                            'arguments': action.arguments,
                            'reason': action.reason,
                        })
                        set_attribute(handle, 'output.mime_type', 'application/json')
                        mark_success(handle)
                if decision_error is not None:
                    # Re-raise after the span closes: automatic OpenTelemetry
                    # exception events must not export the original message.
                    raise decision_error
            except TimeoutError as exc:
                record_failure(exc, 'decision_timeout')
                return end('deadline_exceeded' if time.monotonic() >= deadline else 'decision_timeout')
            except (ValueError, TypeError, KeyError) as exc:
                diagnostic = record_failure(exc, 'invalid_decision')
                if ledger.complete:
                    return end('coverage_complete', finished=True)
                if requirements_settled() and any(o.status in {'complete', 'partial'} for o in observations):
                    return end('coverage_incomplete')
                if decision_repairs < self.budget.max_decision_repairs:
                    decision_repairs += 1
                    feedback = {
                        'reason_code': diagnostic['code'],
                        'detail_code': diagnostic['detail_code'],
                        'message': getattr(self.decider, 'repair_hint', ACTION_SCHEMA_HINT),
                    }
                    continue
                return end('invalid_decision')
            except Exception as exc:
                record_failure(exc, 'decision_failed')
                if ledger.complete:
                    return end('coverage_complete', finished=True)
                if requirements_settled() and any(o.status in {'complete', 'partial'} for o in observations):
                    return end('coverage_incomplete')
                return end('decision_failed')
            record = {'sequence': decisions, 'action': action.action, 'capability': action.capability,
                      'arguments': action.arguments, 'reason': action.reason, 'elapsed_ms': round((time.monotonic()-started)*1000,3)}
            actions.append(record)
            emit_execution_event('loop_decision', 'complete', f'决策 #{decisions}: {action.action}',
                                 elapsed_ms=record['elapsed_ms'],
                                 metadata={'decision_number': decisions, 'action': action.action,
                                           'capability': action.capability,
                                           'execution_mode': 'bounded_loop'})
            if action.action == 'finish':
                if ledger.complete:
                    return end('finish', finished=True)
                # Terminal tool outcomes need not trigger repeated finish decisions.
                settled = {o.requirement_id for o in observations}
                completed = {o.requirement_id for o in observations if o.status == 'complete'}
                # A terminal dependency settles downstream work as blocked,
                # without fabricating observations or crediting their coverage.
                while True:
                    blocked = {r.id for r in task.requirements
                               if any(dep in settled - completed for dep in r.dependencies)}
                    if blocked <= settled:
                        break
                    settled.update(blocked)
                if set(ledger.missing) <= settled:
                    return end('coverage_incomplete')
                feedback = {'reason_code': 'coverage_incomplete', 'missing': ledger.missing}
                record['reason_code'] = 'coverage_incomplete'
                continue
            if action.action in {'clarify','unsupported'}:
                return end('unjustified_' + action.action)
            try:
                requirement, arguments = task.resolve(action, observations)
            except (ValueError, TypeError, KeyError):
                corrections += 1
                record['reason_code'] = 'unverified_arguments'
                if corrections > 1:
                    return end('invalid_arguments')
                feedback = {'reason_code': 'unverified_arguments', 'message': 'Use grounded requirement arguments or valid observation references.'}
                continue
            fingerprint = action_fingerprint(AgentAction('call', action.capability, arguments))
            previous = latest.get(fingerprint)
            if previous is not None:
                if previous.status != 'failed' or previous.reason_code not in RETRYABLE or retries >= self.budget.max_retries:
                    return end('duplicate_tool_call')
                retries += 1
            if ledger.complete:
                return end('coverage_complete', finished=True)
            if calls >= self.budget.max_tool_calls:
                return end('tool_budget_exceeded')
            adapter = self.registry.get(action.capability)
            if adapter is None:
                return end('unregistered_capability')
            calls += 1
            started = time.monotonic()
            emit_execution_event('tool', 'running', f'调用 {requirement.capability}',
                                 metadata={'step_id': requirement.id, 'capability': requirement.capability,
                                           'decision_number': decisions,
                                           'execution_mode': 'bounded_loop'})
            code = error = None
            result = None
            handle = None
            try:
                with span(self.tracer, 'nba.agent.loop.tool', 'TOOL') as handle:
                    set_attribute(handle, 'nba.agent.requirement_id', requirement.id)
                    set_attribute(handle, 'nba.agent.capability', requirement.capability)
                    set_text(handle, 'input.arguments', arguments)
                    set_attribute(handle, 'input.mime_type', 'application/json')
                    result = await wait_call(adapter.execute, requirement.question or question, arguments)
                    if not isinstance(result, dict):
                        raise ValueError('invalid tool response')
                    status = validate_result(requirement.capability, result, arguments)
                    code = result.get('reason_code')
                    set_text(handle, 'output.value', result)
                    set_attribute(handle, 'output.mime_type', 'application/json')
                    set_attribute(handle, 'nba.agent.status', status)
                    set_attribute(handle, 'nba.agent.observation_id', f'o{len(observations)+1}')
                    set_attribute(handle, 'nba.agent.reason_code', code or '')
                    set_attribute(handle, 'nba.agent.elapsed_ms', round((time.monotonic()-started)*1000,3))
                    if status in {'complete', 'partial'}:
                        mark_success(handle)
                    else:
                        mark_error(handle, RuntimeError(code or status), reason_code=code or status)
            except TimeoutError:
                if handle is not None:
                    mark_error(handle, TimeoutError('tool timeout'), reason_code='deadline_exceeded')
                emit_execution_event('tool', 'failed', f'失败 {requirement.capability}', metadata={
                    'step_id': requirement.id, 'capability': requirement.capability,
                    'reason_code': 'deadline_exceeded', 'execution_mode': 'bounded_loop'})
                return end('deadline_exceeded')
            except (ValueError, KeyError, TypeError, AttributeError):
                status, code, error = 'failed', 'invalid_adapter_result', 'Tool result failed validation.'
                result = {'status': 'failed', 'reason_code': code, 'error_stage': 'validation'}
            except Exception as exc:
                if handle is not None:
                    mark_error(handle, exc, reason_code=getattr(exc, 'reason_code', 'adapter_failed'))
                status = 'failed'
                code = getattr(exc, 'reason_code', 'adapter_failed')
                error = 'Tool execution failed.'
                from app.agent_routes import AgentCapabilityError
                if isinstance(exc, AgentCapabilityError) and (exc.stage or exc.timings):
                    result = {'status': 'failed', 'reason_code': code,
                              'error_stage': exc.stage, 'timings': dict(exc.timings),
                              'stage_statuses': dict(exc.stage_statuses), 'model_calls': dict(exc.model_calls)}
            observation = Observation(f'o{len(observations)+1}', requirement.capability, status,
                                      result=result, error=error, requirement_id=requirement.id,
                                      arguments=arguments, reason_code=code,
                                      elapsed_ms=round((time.monotonic()-started)*1000,3))
            observations.append(observation)
            emit_execution_event('tool', 'complete' if status in {'complete', 'partial'} else 'failed',
                                 f'{"完成" if status in {"complete", "partial"} else "失败"} {requirement.capability}',
                                 elapsed_ms=observation.elapsed_ms,
                                 metadata={'step_id': requirement.id, 'capability': requirement.capability,
                                           'decision_number': decisions,
                                           'observation_id': observation.id, 'reason_code': code,
                                           'execution_mode': 'bounded_loop'})
            emit_execution_event('observation', 'complete' if status in {'complete', 'partial'} else 'failed',
                                 f'观察 {observation.id}', elapsed_ms=observation.elapsed_ms,
                                 metadata={'observation_id': observation.id,
                                           'capability': requirement.capability,
                                           'requirement_id': requirement.id,
                                           'reason_code': code,
                                           'execution_mode': 'bounded_loop'})
            latest[fingerprint] = observation
            before = ledger.missing
            ledger.observe(observation)
            with span(self.tracer, 'nba.agent.loop.observation') as observation_span:
                set_attribute(observation_span, 'nba.agent.observation_id', observation.id)
                set_attribute(observation_span, 'nba.agent.requirement_id', requirement.id)
                set_attribute(observation_span, 'nba.agent.capability', requirement.capability)
                set_attribute(observation_span, 'nba.agent.status', status)
                set_attribute(observation_span, 'nba.agent.coverage_before', before)
                set_attribute(observation_span, 'nba.agent.coverage_after', ledger.missing)
                mark_success(observation_span) if status in {'complete', 'partial'} else mark_error(
                    observation_span, RuntimeError(code or status), reason_code=code or status)
            feedback = {'reason_code': code, 'observation_id': observation.id} if status != 'complete' else None
            if ledger.complete:
                return end('coverage_complete', finished=True)
        return end('decision_budget_exceeded')
