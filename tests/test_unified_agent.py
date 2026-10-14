"""Unified product transport exercises the existing bounded loop, not another planner."""
import json
import sqlite3

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.main import create_app
from scripts.provider_config import ProviderConfig
from scripts.evaluate_agent_loop_conversation import FixtureDecider
from tests.test_agent_routes import FakeKnowledge
from tests.test_ask import make_ask_database, make_matrix
from tests.test_semantic_routing import NativeFixture, request, summary


MIXED_QUESTION = '库里 2023-24 常规赛 eFG% 是多少，并解释 eFG%？'


def mixed_call():
    return ('player_efg', {'entity': {'mention': '库里'}, 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'knowledge_question': '解释 eFG%'})


def unified_app(tmp_path, calls, *, enabled=True, decider=None, **kwargs):
    db, matrix = tmp_path / 'agent.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    model = NativeFixture(calls)
    app = create_app(db_path=db, quality_matrix_path=matrix, model=model,
        provider_config=ProviderConfig.empty(), routing_mode='model',
        agent_unified_loop_enabled=enabled, agent_loop_decider=decider,
        agent_knowledge_adapter=kwargs.pop('knowledge', FakeKnowledge()), **kwargs)
    return app, model


@pytest.mark.parametrize('enabled,question,call,mode', [
    (False, MIXED_QUESTION, mixed_call(), 'bounded_plan'),
    (True, MIXED_QUESTION, mixed_call(), 'bounded_loop'),
    (True, '库里 2023-24 常规赛总得分', summary(), 'bounded_plan'),
    (True, '什么是 eFG%？', ('knowledge_glossary', {'question': '什么是 eFG%？'}), 'bounded_plan'),
])
def test_policy_preserves_simple_routes_and_flag_default(tmp_path, enabled, question, call, mode):
    app, model = unified_app(tmp_path, [call], enabled=enabled, decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': question}).json()
    assert body['status'] == 'complete', body
    assert body['execution_mode'] == mode
    assert len(model.requests) == 1
    assert body['evidence']


def test_rules_request_never_delegates(tmp_path):
    app, model = unified_app(tmp_path, [], decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION, 'routing_mode': 'rules'}).json()
    assert body['status'] == 'complete', body
    assert body['execution_mode'] == 'bounded_plan'
    assert model.requests == []


def test_native_decisions_reuse_requirements_and_share_request_trace(tmp_path):
    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app, model = unified_app(tmp_path, [mixed_call(),
        ('player_efg', {'requirement_id': 's1'}),
        ('knowledge_glossary', {'requirement_id': 's2'}), ('finish', {})],
        tracer=provider.get_tracer('unified-test'))
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    assert body['status'] == 'complete', body
    assert body['tool_calls'] == 2 and body['decisions'] == 2
    assert len(model.requests) == 3  # planner + two decisions; coverage owns finish
    assert [r.tools[0]['function']['name'] for r in model.requests] == [
        'select_plan', 'select_requirement', 'select_requirement']
    assert body['decision_failures'] == []
    spans = exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    root = by_name['nba.agent.semantic_request']
    orchestrator = by_name['nba.agent.orchestrator']
    loop = by_name['nba.agent.loop.request']
    assert orchestrator.parent.span_id == root.context.span_id
    assert loop.parent.span_id == orchestrator.context.span_id
    assert orchestrator.attributes['nba.agent.execution_mode'] == 'bounded_loop'
    assert body['trace_id'] == format(root.context.trace_id, '032x')
    assert all(s.context.trace_id == root.context.trace_id for s in spans)
    for s in spans:
        if s.name in {'nba.agent.loop.decision', 'nba.agent.loop.tool',
                      'nba.agent.loop.observation', 'nba.agent.loop.termination'}:
            assert s.parent.span_id == loop.context.span_id
    assert by_name['nba.agent.loop.termination'].attributes['nba.agent.termination_reason'] == 'coverage_complete'
    assert by_name['nba.agent.loop.observation'].attributes['nba.agent.coverage_after'] == '[]'
    assert root.attributes['output.value.preview']


def comparison_question():
    return '比较 库里 与 詹姆斯 2023-24 赛季常规赛的 eFG%，并解释 eFG 是什么。'


def invalid_comparison_call():
    return []


@pytest.mark.parametrize('enabled', [False, True])
def test_invalid_planner_calls_fail_closed_without_losing_request_components(tmp_path, enabled):
    app, model = unified_app(tmp_path, [invalid_comparison_call(), invalid_comparison_call()], enabled=enabled,
                             decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': comparison_question()}).json()
    assert body['status'] == 'incomplete', body
    assert body['reason_code'] == 'invalid_tool_calls'
    assert body['tool_calls'] == 0
    assert body['planning']['planner_type'] == 'model'
    assert 'fallback_reason' not in body['planning']
    assert body['planning']['attempts'] == 2
    assert len(model.requests) == 2
    assert body['error_stage'] == 'planning'


def test_planner_timeout_does_not_use_deterministic_fallback(tmp_path):
    app, model = unified_app(tmp_path, [TimeoutError('fixture')], enabled=True,
                             decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': comparison_question()}).json()
    assert body['status'] == 'incomplete'
    assert body['reason_code'] == 'model_timeout'
    assert body['execution_mode'] == 'bounded_plan'
    assert 'fallback_reason' not in body['planning']
    assert len(model.requests) == 1


def test_comparison_uses_observation_refs_and_deterministic_delta(tmp_path):
    question = '比较库里与詹姆斯 2023-24 常规赛的 eFG%，并解释 eFG%'
    call = ('compare_player_efg', {'left': {'mention': '库里'}, 'right': {'mention': '詹姆斯'},
        'season_id': '2023-24', 'competition_phase': 'regular_season'})
    app, model = unified_app(tmp_path, [call], decider=FixtureDecider())
    # LeBron has an identity but no rows in the standard fixture. Add two
    # synthetic rows so both real SQL/calculation paths can be validated.
    with sqlite3.connect(tmp_path / 'agent.sqlite') as connection:
        connection.execute('''INSERT INTO player_game_stat
            SELECT source_snapshot_id, source_row_id + 100, game_id, '2544',
                   team_id, opponent_team_id, appearance_status, minutes_quality,
                   raw_comment, points, assists, rebounds_total, 6, 1, 16, raw_record_json
            FROM player_game_stat WHERE person_id='201939' AND game_id!='22200001' ''')
    body = request(app, '/agent/answer', {'question': question}).json()
    assert body['execution_mode'] == 'bounded_loop'
    assert body['status'] == 'complete', body
    assert [o['capability'] for o in body['observations']] == [
        'player_efg', 'player_efg', 'efg_compare', 'knowledge_glossary']
    left, right, compare = [o['result'] for o in body['observations'][:3]]
    assert compare['absolute_delta'] == left['efg_percentage'] - right['efg_percentage']
    assert body['actions'][2]['arguments'] == {'left_observation_id': 'o1', 'right_observation_id': 'o2'}
    assert len(model.requests) == 1


def test_native_comparison_executes_all_requirements_and_binds_dependencies(tmp_path):
    call = ('compare_player_efg', {'left': {'mention': '库里'}, 'right': {'mention': '詹姆斯'},
        'season_id': '2023-24', 'competition_phase': 'regular_season'})
    app, model = unified_app(tmp_path, [call,
        ('select_requirement', {'requirement_id': 's2'}),
        ('select_requirement', {'requirement_id': 's4'}),
        ('select_requirement', {'requirement_id': 's1'}),
        ('select_requirement', {'requirement_id': 's3'})])
    with sqlite3.connect(tmp_path / 'agent.sqlite') as connection:
        connection.execute('''INSERT INTO player_game_stat
            SELECT source_snapshot_id, source_row_id + 100, game_id, '2544',
                   team_id, opponent_team_id, appearance_status, minutes_quality,
                   raw_comment, points, assists, rebounds_total, 6, 1, 16, raw_record_json
            FROM player_game_stat WHERE person_id='201939' AND game_id!='22200001' ''')
    body = request(app, '/agent/answer', {'question': comparison_question()}).json()
    assert body['status'] == 'complete', body
    assert body['tool_calls'] == body['decisions'] == 4
    assert body['decision_failures'] == []
    by_requirement = {o['requirement_id']: o for o in body['observations']}
    assert set(by_requirement) == {'s1', 's2', 's3', 's4'}
    assert by_requirement['s3']['result']['absolute_delta'] == (
        by_requirement['s1']['result']['efg_percentage'] - by_requirement['s2']['result']['efg_percentage'])
    assert body['actions'][-1]['arguments'] == {'left_observation_id': 'o3', 'right_observation_id': 'o1'}
    assert 'eFG' in body['answer']
    assert all(len(r.tools) == 1 for r in model.requests)


def test_mixed_verbatim_subquestion_keeps_server_grounded_metric_context(tmp_path):
    from app.agent_routes import KnowledgeGlossaryAdapter
    seen = []
    async def answerer(question):
        seen.append(question)
        return {'status': 'answered', 'answer': 'eFG% 是有效命中率。',
                'claims': [{'citation_id': 'fixture', 'excerpt': 'eFG% 是有效命中率。'}]}
    adapter = KnowledgeGlossaryAdapter(None, answerer=type('Answerer', (), {'aanswer': staticmethod(answerer)})())
    call = mixed_call()
    call[1]['knowledge_question'] = '解释这个指标的含义'
    app, _ = unified_app(tmp_path, [call], decider=FixtureDecider(), knowledge=adapter)
    body = request(app, '/agent/answer', {'question':
        '库里 2023-24 常规赛 eFG% 是多少，并解释这个指标的含义？'}).json()
    assert body['status'] == 'complete', body
    assert seen and '解释这个指标的含义' in seen[0]
    assert 'effective field goal percentage' in seen[0]
    assert body['planning']['tool_arguments']['knowledge_question'] == '解释这个指标的含义'


def test_native_mixed_quoted_knowledge_text_accepts_natural_language_punctuation(tmp_path):
    subquestion = 'explain eFG%; give its formula'
    call = mixed_call()
    call[1]['knowledge_question'] = subquestion
    app, _ = unified_app(tmp_path, [call, ('select_requirement', {'requirement_id': 's1'}),
        ('select_requirement', {'requirement_id': 's2'})])
    body = request(app, '/agent/answer', {'question':
        f'库里 2023-24 常规赛 eFG% 是多少，{subquestion}？'}).json()
    assert body['status'] == 'complete', body
    assert body['decision_failures'] == []
    assert body['observations'][-1]['arguments']['term'] == subquestion


@pytest.mark.parametrize('code', ['generation_timeout', 'verification_timeout', 'database_unavailable'])
def test_tool_failures_preserve_partial_answer_and_specific_stage(tmp_path, code):
    class FailingKnowledge(FakeKnowledge):
        async def execute(self, question, arguments):
            from app.agent_routes import AgentCapabilityError
            raise AgentCapabilityError(code, stage=code.removesuffix('_timeout'), timings={'failed_ms': 7.0})
    app, _ = unified_app(tmp_path, [mixed_call()], decider=FixtureDecider(), knowledge=FailingKnowledge())
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    assert body['status'] == 'partial', body
    assert '58.3333' in body['answer']
    assert body['observations'][-1]['reason_code'] == code
    assert body['evidence'][-1]['reason_code'] == code
    assert body['error_stage'] == code.removesuffix('_timeout')
    assert body['tool_calls'] == 2


def test_planning_failure_never_enters_loop(tmp_path):
    app, model = unified_app(tmp_path, [RuntimeError('provider fixture')], decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    assert body['status'] == 'incomplete'
    assert body['reason_code'] == 'provider_failed'
    assert body['execution_mode'] == 'bounded_plan'
    assert body['actions'] == [] and body['tool_calls'] == 0
    assert len(model.requests) == 1
    assert body['execution_events'][-1]['stage'] == 'result'
    assert body['execution_events'][-1]['metadata']['reason_code'] == 'provider_failed'


def test_loop_then_single_step_followup_preserves_grounded_scope(tmp_path):
    second = ('player_efg', {'entity': {'inherit': True},
        'inherit_scope': ['season_id', 'competition_phase']})
    app, _ = unified_app(tmp_path, [mixed_call(), second], decider=FixtureDecider())
    first = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    scope = app.state.agent_conversations.get(first['conversation_id'])
    assert scope.player_ids == ('201939',) and scope.season_id == '2023-24'
    assert scope.metric_terms == ('efg_percentage',)
    followup = request(app, '/agent/answer', {'question': '那他同一赛季的 eFG% 呢？',
        'conversation_id': first['conversation_id']}).json()
    assert followup['status'] == 'complete', followup
    assert followup['execution_mode'] == 'bounded_plan'
    assert followup['turn_index'] == 2
    assert followup['steps'][0]['result']['efg_percentage'] == first['steps'][0]['result']['efg_percentage']


def test_unified_sse_streams_loop_events_before_final_contract(tmp_path):
    app, _ = unified_app(tmp_path, [mixed_call()], decider=FixtureDecider())
    response = request(app, '/agent/answer/stream', {'question': MIXED_QUESTION})
    frames = [frame for frame in response.text.split('\n\n') if frame.strip()]
    pairs = [(frame.splitlines()[0].removeprefix('event: '),
              json.loads(frame.splitlines()[1].removeprefix('data: '))) for frame in frames]
    assert pairs[-2][0] == 'result' and pairs[-1][0] == 'done'
    final = pairs[-2][1]
    assert final['execution_mode'] == 'bounded_loop' and final['status'] == 'complete'
    assert any(p['stage'] == 'loop_decision' for event, p in pairs if event == 'stage')
    assert any(p['stage'] == 'observation' for event, p in pairs if event == 'stage')
    assert all(e['request_id'] == final['request_id'] for e in final['execution_events'])


def test_failed_numeric_requirement_does_not_replace_previous_scope(tmp_path):
    from app.agent_routes import AgentCapabilityError, CapabilityRegistry
    from tests.test_agent_loop_contract import SequenceDecider, EFG, FINISH

    class FailedEfg:
        name, route = 'player_efg', 'structured'
        async def execute(self, question, arguments):
            raise AgentCapabilityError('database_unavailable')

    app, _ = unified_app(tmp_path, [mixed_call()],
        decider=SequenceDecider([
            {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}}, EFG, FINISH]),
        agent_registry=CapabilityRegistry([FailedEfg(), FakeKnowledge()]))
    store = app.state.agent_conversations
    previous = store.put(conversation_id='scope-safe', player_ids=['2544'],
        season_id='2022-23', competition_phase='regular_season', metric_terms=['points_total'])
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION,
        'conversation_id': 'scope-safe'}).json()
    assert body['status'] == 'partial', body
    saved = store.get('scope-safe')
    assert saved.player_ids == previous.player_ids
    assert saved.season_id == previous.season_id
    assert saved.metric_terms == previous.metric_terms
    assert saved.turn_index == previous.turn_index + 1


def test_unregistered_plan_is_rejected_before_any_loop_tool(tmp_path):
    from app.agent_routes import CapabilityRegistry, StructuredToolAdapter
    registry = CapabilityRegistry([StructuredToolAdapter('player_efg')])
    app, _ = unified_app(tmp_path, [mixed_call()], decider=FixtureDecider(), agent_registry=registry)
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    assert body['status'] == 'unsupported', body
    assert body['reason_code'] == 'unregistered_capability'
    assert body['tool_calls'] == 0
    assert body['observations'] == []


def test_invalid_knowledge_result_exposes_validation_stage(tmp_path):
    class InvalidKnowledge(FakeKnowledge):
        async def execute(self, question, arguments):
            return {'status': 'partial', 'reason_code': 'missing_answer'}
    app, _ = unified_app(tmp_path, [mixed_call()], decider=FixtureDecider(), knowledge=InvalidKnowledge())
    body = request(app, '/agent/answer', {'question': MIXED_QUESTION}).json()
    assert body['status'] == 'partial', body
    assert body['error_stage'] == 'validation'
    assert body['observations'][-1]['reason_code'] == 'invalid_adapter_result'
    assert '58.3333' in body['answer']
