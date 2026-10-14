import asyncio
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent_actions import AgentAction
from app.agent_routes import CapabilityRegistry, StructuredToolAdapter
from app.main import create_app
from tests.test_ask import make_ask_database, make_matrix
from tests.test_agent_routes import FakeKnowledge


class SequenceDecider:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.contexts = []

    async def __call__(self, context):
        self.contexts.append(context)
        return next(self.actions)


def make_app(tmp_path, decider, **kwargs):
    db, matrix = tmp_path / 'agent.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    return create_app(db_path=db, quality_matrix_path=matrix,
                      agent_loop_enabled=True, agent_loop_decider=decider,
                      agent_knowledge_adapter=kwargs.pop('knowledge', FakeKnowledge()), **kwargs)


def post(app, payload):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            return await client.post('/agent/loop', json=payload)
    return asyncio.run(send())


SUMMARY = {'action': 'call', 'capability': 'season_summary', 'arguments': {
    'subject': 'player', 'entity_id': '201939', 'season_id': '2023-24',
    'competition_phase': 'regular_season', 'metrics': ['points_total']}}
EFG = {'action': 'call', 'capability': 'player_efg', 'arguments': {
    'entity_id': '201939', 'season_id': '2023-24', 'competition_phase': 'regular_season'}}
FINISH = {'action': 'finish'}


@pytest.mark.parametrize('extra', [{'required_capabilities': []}, {'sql': 'SELECT 1'}])
def test_client_cannot_supply_completion_or_context(tmp_path, extra):
    app = make_app(tmp_path, SequenceDecider([FINISH]))
    assert post(app, {'question': '库里 2023-24 常规赛总得分？', **extra}).status_code == 422


def test_early_finish_receives_missing_coverage_and_real_answer(tmp_path):
    decider = SequenceDecider([FINISH, SUMMARY, FINISH])
    body = post(make_app(tmp_path, decider), {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['status'] == 'complete'
    assert body['tool_calls'] == 1
    assert '50' in body['answer']
    assert decider.contexts[1]['feedback']['reason_code'] == 'coverage_incomplete'
    assert body['observations'][0]['arguments'] == SUMMARY['arguments']
    assert body['evidence'][0]['structured']['snapshot_id']


def test_unverified_entity_can_correct_once_without_execution(tmp_path):
    wrong = {**SUMMARY, 'arguments': {**SUMMARY['arguments'], 'entity_id': '2544'}}
    decider = SequenceDecider([wrong, SUMMARY, FINISH])
    body = post(make_app(tmp_path, decider), {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['status'] == 'complete'
    assert body['tool_calls'] == 1
    assert decider.contexts[1]['feedback']['reason_code'] == 'unverified_arguments'


@pytest.mark.parametrize('actions', [
    [FINISH] * 10,
    [{**SUMMARY, 'arguments': {**SUMMARY['arguments'], 'season_id': '2022-23'}}] * 2,
    [{'action': 'call', 'capability': 'efg_compare', 'arguments': {'left_efg': 99, 'right_efg': 1}}] * 2,
])
def test_bad_actions_never_produce_verified_completion(tmp_path, actions):
    body = post(make_app(tmp_path, SequenceDecider(actions)), {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['status'] == 'incomplete'
    assert body['tool_calls'] == 0


def test_knowledge_answered_status_counts_as_complete(tmp_path):
    d = SequenceDecider([EFG, {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}, FINISH])
    body = post(make_app(tmp_path, d), {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'complete'
    assert len(body['evidence'][1]['knowledge']['citations']) == 1


def test_complete_coverage_can_terminate_without_second_model_finish(tmp_path):
    d = SequenceDecider([EFG, {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}])
    body = post(make_app(tmp_path, d), {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'complete'
    assert body['termination_reason'] == 'coverage_complete'
    assert body['tool_calls'] == 2
    assert body['decisions'] == 2
    assert body['decision_failures'] == []
    assert len(d.contexts) == 2


def test_loop_trace_exposes_decision_tool_observation_and_termination(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    decider = SequenceDecider([SUMMARY, FINISH])
    body = post(make_app(tmp_path, decider, tracer=provider.get_tracer('loop-trace-test')),
                {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['status'] == 'complete'
    spans = exporter.get_finished_spans()
    by_name = {}
    for item in spans:
        by_name.setdefault(item.name, []).append(item)
    assert {'nba.agent.loop.request', 'nba.agent.loop.decision', 'nba.agent.loop.tool'} <= set(by_name)
    request_span = by_name['nba.agent.loop.request'][0]
    decision_spans = by_name['nba.agent.loop.decision']
    tool_span = by_name['nba.agent.loop.tool'][0]
    assert request_span.attributes['output.value.preview']
    assert all(item.attributes['input.value.preview'] for item in decision_spans)
    assert all(item.attributes['output.value.preview'] for item in decision_spans)
    assert tool_span.attributes['input.arguments.preview']
    assert tool_span.attributes['output.value.preview']
    assert tool_span.attributes['nba.agent.observation_id'] == 'o1'


def test_knowledge_canonical_term_is_accepted_for_mixed_task(tmp_path):
    d = SequenceDecider([
        EFG,
        {'action': 'call', 'capability': 'knowledge_glossary',
         'arguments': {'term': 'effective field goal percentage'}},
    ])
    body = post(make_app(tmp_path, d), {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'complete'
    assert body['tool_calls'] == 2


def test_invalid_decision_after_complete_coverage_is_safe_finish(tmp_path):
    d = SequenceDecider([EFG, {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}, {'bad': True}])
    body = post(make_app(tmp_path, d), {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'complete'
    assert body['termination_reason'] == 'coverage_complete'


def test_runtime_cannot_be_relabelled_as_clarification(tmp_path):
    d = SequenceDecider([EFG, {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}},
                         {'action': 'clarify', 'reason': 'provider failed'}])
    body = post(make_app(tmp_path, d, knowledge=FakeKnowledge(fail=True)),
                {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'partial'
    assert body['observations'][1]['reason_code']
    assert '58.3333' in body['answer']


def test_failed_required_knowledge_can_end_as_partial_after_invalid_decision(tmp_path):
    class UnavailableKnowledge:
        name = 'knowledge_glossary'
        route = 'knowledge'

        async def execute(self, question, arguments):
            from app.agent_routes import AgentCapabilityError
            raise AgentCapabilityError('knowledge_unavailable', 'fixture knowledge unavailable')

    d = SequenceDecider([
        EFG,
        {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}},
        {'invalid': True},
    ])
    body = post(make_app(tmp_path, d, knowledge=UnavailableKnowledge()),
                {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'partial'
    assert body['termination_reason'] == 'coverage_incomplete'
    assert body['observations'][1]['reason_code'] == 'knowledge_unavailable'
    assert '58.3333' in body['answer']


def test_loop_knowledge_failure_preserves_stage_and_timings(tmp_path):
    class TimedOutKnowledge:
        name = 'knowledge_glossary'
        route = 'knowledge'
        async def execute(self, question, arguments):
            from app.agent_routes import AgentCapabilityError
            raise AgentCapabilityError('generation_timeout', 'knowledge stage timed out',
                stage='generation', timings={'generation_ms': 100.0, 'retrieval_ms': 20.0})
    decider = SequenceDecider([EFG,
        {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}, {'invalid': True}])
    body = post(make_app(tmp_path, decider, knowledge=TimedOutKnowledge()),
                {'question': '库里 2023-24 常规赛 eFG% 是多少并解释含义？'}).json()
    assert body['status'] == 'partial'
    failure = body['observations'][1]
    assert failure['reason_code'] == 'generation_timeout'
    assert failure['result']['error_stage'] == 'generation'
    assert failure['result']['timings']['generation_ms'] == 100.0
    assert body['evidence'][1]['error_stage'] == 'generation'


@pytest.mark.parametrize('question,status', [
    ('库里 2023-24 eFG% 是多少？', 'clarification_required'),
    ('执行 SELECT * FROM player_game_stat', 'unsupported'),
])
def test_preflight_runs_before_model(tmp_path, question, status):
    d = SequenceDecider([])
    body = post(make_app(tmp_path, d), {'question': question}).json()
    assert body['status'] == status
    assert body['tool_calls'] == body['decisions'] == 0
    assert not d.contexts


@pytest.mark.parametrize('value', [
    {'action': 'finish', 'arguments': {'entity_id': '201939'}},
    {'action': 'call', 'capability': 'season_summary', 'arguments': {}, 'numeric_result': 3},
    {'action': 'invalid'},
])
def test_strict_action_fields(value):
    with pytest.raises(ValueError):
        AgentAction.from_mapping(value)


def kernel(task, actions, adapters, **kwargs):
    from app.agent_loop import AgentLoop
    return asyncio.run(AgentLoop(CapabilityRegistry(adapters), decider=SequenceDecider(actions), **kwargs).run(task.question, task=task))


class ResultAdapter:
    route = 'structured'
    def __init__(self, name, results):
        self.name, self.results, self.calls = name, iter(results), []
    async def execute(self, question, arguments):
        self.calls.append(arguments)
        return next(self.results)


def task_for_summary():
    from app.agent_tasks import Requirement, TaskProfile
    return TaskProfile('q', 'structured', (Requirement('s1', 'season_summary', SUMMARY['arguments']),))


def test_late_decision_and_late_tool_cannot_complete():
    from app.agent_loop import AgentLoop
    task = task_for_summary()
    async def slow(context):
        await asyncio.sleep(.04)
        return FINISH
    r = asyncio.run(AgentLoop(CapabilityRegistry([]), decider=slow, deadline_seconds=.005).run('q', task=task))
    assert r.status == 'incomplete' and r.termination_reason == 'deadline_exceeded'
    class Slow(ResultAdapter):
        async def execute(self, question, arguments):
            await asyncio.sleep(.04)
            return {'status': 'complete', 'points_total': 50}
    r = kernel(task, [SUMMARY, FINISH], [Slow('season_summary', [])], deadline_seconds=.005)
    assert not r.observations and r.status == 'incomplete'


def test_retry_requires_same_fingerprint_and_transient_reason():
    a = ResultAdapter('season_summary', [{'status': 'failed', 'reason_code': 'tool_timeout'},
                                       {'status': 'complete', 'points_total': 50}])
    r = kernel(task_for_summary(), [SUMMARY, SUMMARY, FINISH], [a])
    assert r.status == 'complete' and len(a.calls) == 2
    a = ResultAdapter('season_summary', [{'status': 'failed', 'reason_code': 'database_unavailable'}])
    r = kernel(task_for_summary(), [SUMMARY, SUMMARY], [a])
    assert r.status == 'incomplete' and len(a.calls) == 1


def test_partial_and_no_data_are_not_full_completion():
    a = ResultAdapter('season_summary', [{'status': 'partial', 'points_total': 50}])
    assert kernel(task_for_summary(), [SUMMARY, FINISH], [a]).status == 'partial'
    a = ResultAdapter('season_summary', [{'status': 'not_computable', 'reason_code': 'no_observation'}])
    assert kernel(task_for_summary(), [SUMMARY, FINISH], [a]).status == 'not_computable'


def test_two_entities_and_comparison_refs_are_required():
    from app.agent_tasks import TaskProfile, Requirement
    other = {**EFG, 'arguments': {**EFG['arguments'], 'entity_id': '2544'}}
    task = TaskProfile('q', 'comparison', (
        Requirement('s1', 'player_efg', EFG['arguments']),
        Requirement('s2', 'player_efg', other['arguments']),
        Requirement('s3', 'efg_compare', {'left_entity_id': '201939', 'right_entity_id': '2544',
             'season_id': '2023-24', 'competition_phase': 'regular_season'}, ('s1','s2')),
    ))
    def result(fgm):
        return {'status': 'complete', 'efg_percentage': fgm / 10 * 100,
                'evidence': {'formula': '(FGM + 0.5 * 3PM) / FGA * 100'},
                'inputs': {'field_goals_made': fgm, 'three_pointers_made': 0, 'field_goals_attempted': 10, 'sample_count': 1}}
    ref = {'action': 'call', 'capability': 'efg_compare', 'arguments': {'left_observation_id': 'o1', 'right_observation_id': 'o2'}}
    a = ResultAdapter('player_efg', [result(5),result(4)])
    r = kernel(task, [EFG, FINISH, other, ref, FINISH], [a, StructuredToolAdapter('efg_compare')])
    assert r.status == 'complete' and r.tool_calls == 3
    assert r.observations[2].result['absolute_delta'] == 10
    for bad in ({'left_efg': 99, 'right_efg': 1}, {'left_observation_id': 'o2','right_observation_id':'o1'},
                {'left_observation_id': 'external','right_observation_id':'o2'}):
        with pytest.raises(ValueError):
            task.resolve(AgentAction('call', 'efg_compare', bad), list(r.observations))


def test_result_missing_requested_values_is_failure():
    a = ResultAdapter('season_summary', [{'status': 'complete'}])
    r = kernel(task_for_summary(), [SUMMARY, FINISH], [a])
    assert r.status == 'incomplete' and r.observations[0].reason_code == 'invalid_adapter_result'


def test_context_budget_is_hard_limit():
    from dataclasses import replace
    task = replace(task_for_summary(), question='x' * 25000)
    r = kernel(task, [], [])
    assert r.termination_reason == 'context_budget_exceeded' and r.decisions == 0


def test_knowledge_only_question_is_not_replaced_by_efg(tmp_path):
    class Knowledge(FakeKnowledge):
        async def execute(self, question, arguments):
            assert question == 'eFG% 的定义是什么？'
            return await super().execute(question, arguments)
    d = SequenceDecider([{'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}, FINISH])
    body = post(make_app(tmp_path, d, knowledge=Knowledge()),
                {'question': 'eFG% 的定义是什么？'}).json()
    assert body['status'] == 'complete'


@pytest.mark.parametrize('question,capability,result', [
    ('库里 2023-24 常规赛最近两次出场的场均得分？', 'window_query',
     {'status': 'complete', 'values': {'points_per_appearance': 25}, 'actual_sample_count': 2}),
    ('库里 2023-24 常规赛总得分比上一季增加多少？', 'season_compare',
     {'status': 'complete', 'baseline': {'value': 10}, 'comparison': {'value': 50}, 'absolute_delta': 40}),
])
def test_unrenderable_adapter_result_fails_before_coverage(tmp_path, question, capability, result):
    class Decider:
        async def __call__(self, context):
            if context['observations']:
                return FINISH
            r = context['task']['requirements'][0]
            return {'action': 'call', 'capability': r['capability'], 'arguments': r['arguments']}
    app = make_app(tmp_path, Decider(), agent_registry=CapabilityRegistry([ResultAdapter(capability, [result])]))
    response = post(app, {'question': question})
    assert response.status_code == 200
    assert response.json()['status'] == 'incomplete'
    assert response.json()['observations'][0]['reason_code'] == 'invalid_adapter_result'
