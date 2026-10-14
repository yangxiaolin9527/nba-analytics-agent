import asyncio
import json

import pytest
from urllib.request import Request

from scripts.model_adapter import ModelRequest, ModelResponse, OpenAICompatibleModel
from scripts.provider_config import ProviderConfig
from tests.test_ask import make_ask_database, make_matrix


class NativeFixture:
    model = 'native-fixture'

    def __init__(self, calls):
        self.calls = list(calls)
        self.requests = []

    def generate_tools(self, request):
        self.requests.append(request)
        item = self.calls.pop(0)
        if isinstance(item, Exception):
            raise item
        calls = item if isinstance(item, list) else [item]
        selector = request.tools[0]['function']['name'] if request.tools and len(request.tools) == 1 else None
        if selector == 'select_plan':
            from app.semantic_contracts import CONTRACTS
            calls = [('select_plan', {'capability': name, 'arguments': args}) if name in CONTRACTS else (name, args)
                     for name, args in calls]
        elif selector == 'select_requirement':
            calls = [('select_requirement', args) if 'requirement_id' in args else (name, args)
                     for name, args in calls]
        return ModelResponse('', self.model, [
            {'id': f'call_{i}', 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}
            for i, (name, args) in enumerate(calls)
        ], {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120})


def summary(season='2023-24', entity=None):
    return ('season_summary', {
        'subject': 'player', 'entity': entity or {'mention': '库里'},
        'season_id': season, 'competition_phase': 'regular_season',
        'metrics': ['points_total'],
    })


def test_native_protocol_preserves_null_content_calls_usage_and_timeout():
    captured = {}

    def transport(request: Request, timeout):
        captured.update(payload=json.loads(request.data), timeout=timeout)
        return json.dumps({'choices': [{'message': {'content': None, 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'knowledge_glossary',
             'arguments': '{"question":"offensive three seconds"}'}}
        ]}}], 'usage': {'total_tokens': 123}}).encode()

    model = OpenAICompatibleModel(base_url='https://example.test', api_key='fixture',
                                  model='fixture', transport=transport)
    response = model.generate_tools(ModelRequest(prompt='', messages=[{'role': 'user', 'content': 'test'}],
        tools=[{'type': 'function', 'function': {'name': 'knowledge_glossary'}}],
        tool_choice='required', timeout=3))
    assert response.text == ''
    assert response.tool_calls[0]['id'] == 'c1'
    assert response.usage == {'total_tokens': 123}
    assert captured['payload']['tool_choice'] == 'required'
    assert captured['timeout'] == 3


@pytest.fixture
def db(tmp_path):
    path = tmp_path / 'native.sqlite'
    make_ask_database(path)
    return path


def plan(db, question, calls, **kwargs):
    from app.semantic_planner import plan_semantic
    model = NativeFixture(calls)
    return asyncio.run(plan_semantic(question, db, model, **kwargs)), model


def test_paraphrase_does_not_need_metric_regex(db):
    result, model = plan(db, '库里 2023-2024 常规赛合计拿下几分？', [summary()])
    assert result.plan.route == 'structured'
    assert result.plan.steps[0].arguments['entity_id'] == '201939'
    assert result.plan.steps[0].arguments['metrics'] == ['points_total']
    assert result.diagnostics['usage']['total_tokens'] == 120
    assert model.requests[0].tools


def test_knowledge_routing_requires_no_database(db):
    result, _ = plan(None, '什么是进攻三秒违例', [('knowledge_glossary', {'question': '什么是进攻三秒违例'})])
    assert result.plan.route == 'knowledge'
    assert result.plan.steps[0].arguments['term'] == '什么是进攻三秒违例'


def test_explicit_season_cannot_be_replaced_by_model(db):
    result, model = plan(db, '库里 2020-2021 常规赛总得分', [summary(), summary()])
    assert result.plan.route == 'clarification'
    assert result.plan.reason_code == 'scope_conflict'
    assert len(model.requests) == 1


def test_known_alias_cannot_be_overridden_with_another_identity(db):
    result, _ = plan(db, '库里 2023-24 常规赛总得分', [summary(entity={'mention': '库里', 'canonical_name': 'LeBron James'})])
    assert result.plan.route == 'clarification'
    assert result.plan.reason_code == 'entity_conflict'


def test_missing_phase_is_clarification_not_provider_repair(db):
    call = summary()
    call[1]['competition_phase'] = None
    result, model = plan(db, '库里 2023-24 总得分', [call])
    assert result.plan.reason_code == 'missing_phase'
    assert len(model.requests) == 1


def test_unknown_tools_get_one_repair_without_execution(db):
    result, model = plan(db, '库里 2023-24 常规赛总得分', [('run_sql', {'sql': 'SELECT 1'}), summary()])
    assert result.plan.route == 'structured'
    assert len(model.requests) == 2
    assert result.diagnostics['repairs'] == 1


def test_unresolved_language_mention_repairs_missing_canonical_candidate(db):
    import sqlite3
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO player_identity VALUES ('201935', 'James Harden', 'fixture')")
    first = summary(entity={'mention': '哈登'})
    second = summary(entity={'mention': '哈登', 'canonical_name': 'James Harden'})
    result, model = plan(db, '哈登 2023-24 常规赛总得分', [first, second])
    assert result.plan.route == 'structured', result
    assert result.plan.steps[0].arguments['entity_id'] == '201935'
    assert len(model.requests) == 2
    assert result.diagnostics['repairs'] == 1
    assert 'entity_candidate_required' in model.requests[1].messages[-1]['content']


def test_provider_failure_never_falls_back_to_rules(db):
    result, model = plan(db, '库里 2023-24 常规赛总得分', [TimeoutError('fixture')])
    assert result.error_code == 'model_timeout'
    assert result.plan.steps == []
    assert len(model.requests) == 1


def native_app(tmp_path, calls, **kwargs):
    from app.main import create_app
    path = tmp_path / 'integration.sqlite'
    matrix = tmp_path / 'matrix.json'
    make_ask_database(path)
    make_matrix(matrix)
    return create_app(db_path=path, quality_matrix_path=matrix, model=NativeFixture(calls),
        provider_config=ProviderConfig.empty(), routing_mode='model', **kwargs)


def request(app, path, payload):
    from httpx import ASGITransport, AsyncClient
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            return await client.post(path, json=payload)
    return asyncio.run(send())


@pytest.mark.parametrize('path', ['/ask', '/answer', '/agent/answer'])
def test_native_entrypoints_return_sql_evidence_and_conversation(tmp_path, path):
    app = native_app(tmp_path, [summary()])
    body = request(app, path, {'question': '库里 2023-2024 常规赛合计拿下几分？'}).json()
    assert body['status'] == 'complete'
    assert body['conversation_id']
    assert body['turn_index'] == 1
    assert body['routing_mode'] == 'model'
    assert body.get('tool_result', body.get('steps', [{}])[0].get('result'))['points_total'] == 50


def test_native_stats_inherit_and_explicit_override_without_stale_values(tmp_path):
    second = summary('2022-23', {'inherit': True})
    second[1]['inherit_scope'] = ['competition_phase']
    app = native_app(tmp_path, [summary(), second])
    first = request(app, '/answer', {'question': '库里 2023-24 常规赛累计拿几分'}).json()
    body = request(app, '/answer', {'question': '那他 2022-2023 呢？', 'conversation_id': first['conversation_id']}).json()
    assert body['status'] == 'complete'
    assert body['tool_result']['points_total'] == 10
    assert body['arguments']['season_id'] == '2022-23'
    assert body['turn_index'] == 2


@pytest.mark.parametrize('path', ['/answer', '/agent/answer'])
def test_session_context_reports_new_continued_and_reset(tmp_path, path):
    app = native_app(tmp_path, [summary(), summary(), summary()])
    first = request(app, path, {'question': '库里 2023-24 常规赛总得分'}).json()
    assert first['conversation_context_status'] == 'new'
    second = request(app, path, {'question': '库里 2023-24 常规赛总得分', 'conversation_id': first['conversation_id']}).json()
    assert second['conversation_context_status'] == 'continued'
    app.state.agent_conversations._items.clear()
    third = request(app, path, {'question': '库里 2023-24 常规赛总得分', 'conversation_id': first['conversation_id']}).json()
    assert third['conversation_context_status'] == 'reset'
    assert third['turn_index'] == 1


def test_pending_clarification_can_be_completed_and_abandoned(tmp_path):
    missing = summary()
    missing[1]['competition_phase'] = None
    complete = summary()
    complete[1]['continue_pending'] = True
    app = native_app(tmp_path, [missing, complete, missing, summary('2022-23')])
    first = request(app, '/answer', {'question': '库里 2023-24 总得分'}).json()
    second = request(app, '/answer', {'question': '常规赛', 'conversation_id': first['conversation_id']}).json()
    assert second['status'] == 'complete'
    third = request(app, '/answer', {'question': '库里 2023-24 总得分', 'conversation_id': first['conversation_id']}).json()
    fourth = request(app, '/answer', {'question': '库里 2022-23 常规赛总得分', 'conversation_id': first['conversation_id']}).json()
    assert third['status'] == 'clarification_required'
    assert fourth['status'] == 'complete'
    assert fourth['arguments']['season_id'] == '2022-23'


def test_loop_native_decisions_reference_verified_requirements(tmp_path):
    app = native_app(tmp_path, [summary(), ('season_summary', {'requirement_id': 's1'}), ('finish', {})], agent_loop_enabled=True)
    body = request(app, '/agent/loop', {'question': '库里 2023-24 常规赛合计拿几分'}).json()
    assert body['status'] == 'complete'
    assert body['routing_mode'] == 'model'
    assert body['observations'][0]['result']['points_total'] == 50
    assert body['tool_calls'] == 1


def test_native_provider_error_is_visible_http_response(tmp_path):
    app = native_app(tmp_path, [RuntimeError('fixture provider')])
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛总得分'}).json()
    assert body['status'] == 'incomplete'
    assert body['reason_code'] == 'provider_failed'
    assert body['steps'] == []


def test_semantic_trace_contains_final_planning_failure_response_without_database(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from app.main import create_app

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(
        db_path=None,
        model=NativeFixture([summary()]),
        provider_config=ProviderConfig.empty(),
        routing_mode='model',
        tracer=provider.get_tracer('semantic-routing-test'),
    )
    body = request(app, '/agent/answer', {'question': '库里在 2015-16 赛季常规赛的 eFG%是多少'}).json()
    assert body['status'] == 'incomplete'
    assert body['reason_code'] == 'database_unavailable'
    spans = exporter.get_finished_spans()
    assert {
        'nba.agent.semantic_request', 'nba.agent.semantic_planner',
        'nba.agent.grounding', 'nba.agent.grounding.compile',
        'nba.agent.semantic_planner.llm',
    } <= {item.name for item in spans}
    root = next(item for item in spans if item.name == 'nba.agent.semantic_request')
    assert 'database_unavailable' in root.attributes['output.value.preview']
    assert root.attributes['output.mime_type'] == 'application/json'


def test_semantic_structured_trace_exposes_grounding_tool_sql_and_calculation(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    from app.main import create_app

    # Recreate the app with the exporter without changing the model/database fixtures.
    efg_call = ('player_efg', {'entity': {'mention': '库里'}, 'season_id': '2023-24',
                               'competition_phase': 'regular_season'})
    make_ask_database(tmp_path / 'integration.sqlite')
    make_matrix(tmp_path / 'matrix.json')
    app = create_app(db_path=tmp_path / 'integration.sqlite',
                     quality_matrix_path=tmp_path / 'matrix.json',
                     model=NativeFixture([efg_call]), provider_config=ProviderConfig.empty(),
                     routing_mode='model', tracer=provider.get_tracer('semantic-trace-test'))

    response = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛 eFG% 是多少？'})
    assert response.status_code == 200
    assert response.json()['status'] == 'complete'
    spans = exporter.get_finished_spans()
    by_name = {item.name: item for item in spans}
    assert {
        'nba.agent.semantic_request', 'nba.agent.semantic_planner',
        'nba.agent.semantic_planner.llm', 'nba.agent.grounding',
        'nba.agent.grounding.compile', 'nba.agent.step',
        'nba.agent.sql', 'nba.agent.deterministic_calculation',
    } <= set(by_name)

    root = by_name['nba.agent.semantic_request']
    planner = by_name['nba.agent.semantic_planner']
    step = by_name['nba.agent.step']
    sql = by_name['nba.agent.sql']
    calculation = by_name['nba.agent.deterministic_calculation']
    assert planner.parent.span_id == root.context.span_id
    assert by_name['nba.agent.semantic_planner.llm'].parent.span_id == planner.context.span_id
    assert by_name['nba.agent.grounding'].parent.span_id == planner.context.span_id
    assert by_name['nba.agent.grounding.compile'].parent.span_id == planner.context.span_id
    orchestrator = by_name['nba.agent.orchestrator']
    assert orchestrator.parent.span_id == root.context.span_id
    assert step.parent.span_id == orchestrator.context.span_id
    assert sql.parent.span_id == step.context.span_id
    assert calculation.parent.span_id == step.context.span_id
    assert step.attributes['nba.agent.capability'] == 'player_efg'
    assert step.attributes['input.arguments.preview']
    assert step.attributes['output.value.preview']
    assert sql.attributes['db.system'] == 'sqlite'
    assert 'SELECT' in sql.attributes['db.statement.preview']
    assert calculation.attributes['calculation.formula']
    assert calculation.attributes['output.status'] == 'complete'


def test_semantic_trace_records_conversation_scope_inheritance(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from app.main import create_app

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    path = tmp_path / 'integration.sqlite'
    matrix = tmp_path / 'matrix.json'
    make_ask_database(path)
    make_matrix(matrix)
    app = create_app(db_path=path, quality_matrix_path=matrix,
                     model=NativeFixture([
                         ('player_efg', {'entity': {'mention': '库里'}, 'season_id': '2023-24',
                                         'competition_phase': 'regular_season'}),
                         ('player_efg', {'entity': {'inherit': True}, 'season_id': None,
                                         'competition_phase': None,
                                         'inherit_scope': ['season_id', 'competition_phase']}),
                     ]),
                     provider_config=ProviderConfig.empty(), routing_mode='model',
                     tracer=provider.get_tracer('conversation-trace-test'))

    first = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛 eFG% 是多少？'}).json()
    request(app, '/agent/answer', {
        'question': '那他同一赛季的 eFG% 是多少？',
        'conversation_id': first['conversation_id'],
    })
    roots = [item for item in exporter.get_finished_spans() if item.name == 'nba.agent.semantic_request']
    assert len(roots) == 2
    assert roots[0].attributes['nba.agent.conversation_context_status'] == 'new'
    assert roots[1].attributes['nba.agent.conversation_context_status'] == 'continued'
    assert roots[1].attributes['nba.agent.turn_index'] == 2
    assert 'season_id' in roots[1].attributes['nba.agent.inherited_scope_fields']


def test_semantic_mixed_trace_keeps_structured_and_knowledge_tool_steps(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from app.main import create_app

    class KnowledgeFixture:
        name = 'knowledge_glossary'
        route = 'knowledge'

        async def execute(self, question, arguments):
            return {'status': 'answered', 'answer': 'eFG% 是有效命中率。',
                    'claims': [{'citation_id': 'fixture-efg', 'excerpt': 'eFG% 是有效命中率。'}],
                    'unresolved': []}

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    path = tmp_path / 'integration.sqlite'
    matrix = tmp_path / 'matrix.json'
    make_ask_database(path)
    make_matrix(matrix)
    app = create_app(
        db_path=path, quality_matrix_path=matrix,
        model=NativeFixture([('player_efg', {
            'entity': {'mention': '库里'}, 'season_id': '2023-24',
            'competition_phase': 'regular_season', 'knowledge_question': '解释 eFG%'
        })]), provider_config=ProviderConfig.empty(), routing_mode='model',
        tracer=provider.get_tracer('mixed-trace-test'),
        agent_knowledge_adapter=KnowledgeFixture(),
    )
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛的 eFG% 是多少，并解释 eFG%？'}).json()
    assert body['status'] == 'complete', body
    spans = exporter.get_finished_spans()
    steps = [item for item in spans if item.name == 'nba.agent.step']
    assert {item.attributes['nba.agent.capability'] for item in steps} == {'player_efg', 'knowledge_glossary'}


def test_unified_loop_delegates_mixed_model_plan_without_replanning(tmp_path):
    from app.main import create_app

    class KnowledgeFixture:
        name = 'knowledge_glossary'
        route = 'knowledge'

        async def execute(self, question, arguments):
            return {'status': 'answered', 'answer': 'eFG% 是有效命中率。',
                    'claims': [{'citation_id': 'fixture-efg', 'excerpt': 'eFG% 是有效命中率。'}],
                    'unresolved': []}

    actions = iter([
        {'action': 'call', 'capability': 'player_efg', 'arguments': {
            'entity_id': '201939', 'season_id': '2023-24', 'competition_phase': 'regular_season'}},
        {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}},
        {'action': 'finish'},
    ])
    decisions = []

    async def decider(context):
        decisions.append(context)
        return next(actions)

    model = NativeFixture([('player_efg', {
        'entity': {'mention': '库里'}, 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'knowledge_question': '解释 eFG%'
    })])
    make_ask_database(tmp_path / 'integration.sqlite')
    make_matrix(tmp_path / 'matrix.json')
    app = create_app(
        db_path=tmp_path / 'integration.sqlite', quality_matrix_path=tmp_path / 'matrix.json',
        model=model, provider_config=ProviderConfig.empty(), routing_mode='model',
        agent_loop_enabled=True, agent_unified_loop_enabled=True,
        agent_loop_decider=decider, agent_knowledge_adapter=KnowledgeFixture())
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛的 eFG% 是多少，并解释 eFG%？'}).json()
    assert body['status'] == 'complete', body
    assert body['execution_mode'] == 'bounded_loop'
    assert body['decisions'] == 2
    assert body['tool_calls'] == 2
    assert [item['capability'] for item in body['observations']] == ['player_efg', 'knowledge_glossary']
    assert len(model.requests) == 1
    assert len(decisions) == 2
    assert body['termination_reason'] == 'coverage_complete'
    assert [event['stage'] for event in body['execution_events'] if event['stage'] in
            {'loop_decision', 'tool', 'observation', 'termination'}] == [
        'loop_decision', 'loop_decision', 'tool', 'tool', 'observation',
        'loop_decision', 'loop_decision', 'tool', 'tool', 'observation',
        'termination']


def test_unified_loop_keeps_single_step_structured_plan(tmp_path):
    app = native_app(tmp_path, [summary()], agent_loop_enabled=True,
                     agent_unified_loop_enabled=True)
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛总得分'}).json()
    assert body['status'] == 'complete'
    assert body['execution_mode'] == 'bounded_plan'
    assert 'observations' not in body or body['observations'] == []


def test_unified_loop_trigger_requires_successful_mixed_or_comparison_plan(tmp_path):
    from app.main import create_app
    class KnowledgeFixture:
        name = 'knowledge_glossary'
        route = 'knowledge'
        async def execute(self, question, arguments):
            return {'status': 'answered', 'answer': 'eFG% 是有效命中率。',
                    'claims': [{'citation_id': 'fixture', 'excerpt': 'eFG%'}]}
    actions = iter([
        {'action': 'call', 'capability': 'player_efg', 'arguments': {
            'entity_id': '201939', 'season_id': '2023-24', 'competition_phase': 'regular_season'}},
        {'action': 'call', 'capability': 'knowledge_glossary', 'arguments': {}},
    ])
    class Decider:
        async def __call__(self, context):
            return next(actions)
    model = NativeFixture([('player_efg', {
        'entity': {'mention': '库里'}, 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'knowledge_question': '解释 eFG%'
    })])
    make_ask_database(tmp_path / 'integration.sqlite')
    make_matrix(tmp_path / 'matrix.json')
    app = create_app(db_path=tmp_path / 'integration.sqlite', quality_matrix_path=tmp_path / 'matrix.json',
        model=model, provider_config=ProviderConfig.empty(), routing_mode='model',
        agent_loop_enabled=True, agent_unified_loop_enabled=True,
        agent_loop_decider=Decider(), agent_knowledge_adapter=KnowledgeFixture())
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛 eFG% 是多少，并解释 eFG%？'}).json()
    assert body['execution_mode'] == 'bounded_loop'
    assert body['decisions'] == 2
    assert any(event['stage'] == 'loop_decision' for event in body['execution_events'])


def test_model_metrics_are_still_allowlisted(db):
    call = summary()
    call[1]['metrics'] = ['steals_total']
    result, model = plan(db, '库里 2023-24 常规赛抢断合计', [call])
    assert result.error_code == 'unsupported_metric'
    assert not result.plan.steps
    assert len(model.requests) == 1


def test_model_must_not_invent_a_missing_phase(db):
    result, _ = plan(db, '库里 2023-24 总得分', [summary()])
    assert result.plan.reason_code == 'missing_phase'
    assert not result.plan.steps


def test_ambiguous_surname_is_not_narrowed_by_canonical_guess(db):
    import sqlite3
    with sqlite3.connect(db) as connection:
        connection.executemany('INSERT INTO player_identity(person_id,display_name,identity_quality) VALUES (?,?,?)', [
            ('202695', 'Kawhi Leonard', 'fixture'), ('203086', 'Meyers Leonard', 'fixture')])
    result, _ = plan(db, 'Leonard 2023-24 常规赛总得分', [summary(entity={'mention': 'Leonard', 'canonical_name': 'Kawhi Leonard'})])
    assert result.plan.reason_code == 'ambiguous_entity'
    assert not result.plan.steps


def test_model_translated_name_is_database_checked_and_traced(db):
    result, _ = plan(db, '斯蒂芬·库里 2023-24 常规赛总得分', [summary(entity={'mention': '斯蒂芬·库里', 'canonical_name': 'Stephen Curry'})])
    assert result.plan.steps[0].arguments['entity_id'] == '201939'
    assert result.diagnostics['entity_resolution']


def test_invalid_schema_is_repaired_once_then_stops(db):
    call = summary()
    call[1]['sql'] = 'SELECT 1'
    result, model = plan(db, '库里 2023-24 常规赛总得分', [call, call])
    assert result.error_code == 'invalid_arguments'
    assert len(model.requests) == 2
    assert not result.plan.steps


def test_loop_rejects_fabricated_requirement_reference(tmp_path):
    app = native_app(tmp_path, [summary(), ('season_summary', {'requirement_id': 'evil'}), ('season_summary', {'requirement_id': 'evil'})], agent_loop_enabled=True)
    body = request(app, '/agent/loop', {'question': '库里 2023-24 常规赛合计拿几分'}).json()
    assert body['status'] == 'incomplete'
    assert body['tool_calls'] == 0
    assert body['decision_failures']


def test_mixed_planner_cannot_add_unrequested_knowledge_scope(db):
    question = '库里 2023-24 常规赛 eFG% 是多少，并解释 eFG%？'
    base = {'entity': {'mention': '库里'}, 'season_id': '2023-24', 'competition_phase': 'regular_season'}
    result, model = plan(db, question, [
        ('player_efg', {**base, 'knowledge_question': '解释 eFG% 的定义和局限性'}),
        ('player_efg', {**base, 'knowledge_question': '解释 eFG%'}),
    ])
    assert result.error_code is None
    assert result.diagnostics['repairs'] == 1
    assert result.plan.steps[1].arguments['term'] == '解释 eFG%'
    assert len(model.requests) == 2


def test_repeated_unquoted_knowledge_scope_fails_before_tools(db):
    call = ('player_efg', {'entity': {'mention': '库里'}, 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'knowledge_question': '指标的历史和局限性'})
    result, _ = plan(db, '库里 2023-24 常规赛 eFG% 是多少，并解释 eFG%？', [call, call])
    assert result.error_code == 'knowledge_scope_conflict'
    assert result.plan.steps == []


@pytest.mark.parametrize('call', [
    {'id': 'old', 'type': 'function', 'function': {'name': 'knowledge_glossary', 'arguments': '{"question":"q"}'}},
    {'id': '', 'type': 'function', 'function': {'name': 'select_plan', 'arguments': '{"capability":"knowledge_glossary","arguments":{"question":"q"}}'}},
])
def test_plan_selector_rejects_unadvertised_function_or_empty_call_id(call):
    from app.semantic_planner import decode_call, PlanError
    with pytest.raises(PlanError):
        decode_call(ModelResponse('', 'fixture', [call]))


def test_conversation_busy_is_rejected_by_new_stats_entrypoint(tmp_path):
    app = native_app(tmp_path, [summary()])
    store = app.state.agent_conversations
    store.try_acquire('occupied')
    response = request(app, '/answer', {'question': '库里 2023-24 常规赛总得分', 'conversation_id': 'occupied'})
    assert response.status_code == 409
    assert store.get('occupied') is None


def test_evaluation_budget_is_durable_and_counts_failures(tmp_path):
    from scripts.evaluate_semantic_routing import CallBudgetModel
    ledger = tmp_path/'ledger.json'
    model = CallBudgetModel(NativeFixture([RuntimeError('fixture')]), maximum=1, ledger=ledger)
    with pytest.raises(RuntimeError):
        model.generate_tools(ModelRequest('fixture'))
    assert model.used == 1
    reloaded = CallBudgetModel(NativeFixture([summary()]), maximum=1, ledger=ledger)
    with pytest.raises(RuntimeError, match='budget_exceeded'):
        reloaded.generate_tools(ModelRequest('fixture'))
    assert not reloaded.wrapped.requests


def test_evaluation_oracle_is_independent_of_production_tool(db):
    from scripts.evaluate_semantic_routing import sql_oracle, read_cases
    import sqlite3
    with sqlite3.connect(db) as connection:
        actual = sql_oracle(connection, {'tool': 'season_summary', 'entity': '201939', 'season': '2023-24', 'phase': 'regular_season', 'metrics': ['points_total']})
    assert actual == {'points_total': 50}
    assert len(read_cases()[1]) == 40


@pytest.mark.parametrize('count,proposed,code', [(101, 100, 'window_out_of_range'), (5, 4, 'filter_conflict')])
def test_explicit_window_number_cannot_be_clamped_or_changed(db, count, proposed, code):
    call = summary()
    call = ('window_query', {**call[1], 'last_n': proposed})
    result, _ = plan(db, f'库里 2023-24 常规赛最近{count}次出场得分', [call])
    assert result.plan.reason_code == code
    assert not result.plan.steps


@pytest.mark.parametrize('calls', [[summary()], [[summary(), summary()]]])
def test_scope_ambiguity_is_reported_before_tool_execution(db, calls):
    result, model = plan(db, '库里 2023-24 常规赛和季后赛得分', calls)
    assert result.plan.route == 'clarification'
    assert result.plan.reason_code == 'ambiguous_scope'
    assert len(model.requests) == 1


def test_evaluation_rejects_numeric_result_when_no_rows_expected():
    from scripts.evaluate_semantic_routing import score
    expected = {'tool': 'season_summary', 'entity': '201939', 'status': 'not_computable'}
    body = {'route': 'structured', 'status': 'not_computable', 'steps': [
        {'capability': 'season_summary', 'arguments': {'entity_id': '201939'},
         'result': {'points_total': 123}}]}
    assert not score(expected, body, [{'points_total': None}])['passed']


def test_evaluation_checks_comparison_delta():
    from scripts.evaluate_semantic_routing import score
    expected = {'tools': ['efg_compare'], 'route': 'comparison'}
    body = {'route': 'comparison', 'status': 'complete', 'steps': [
        {'capability': 'efg_compare', 'result': {'absolute_delta': 99}}]}
    assert not score(expected, body, [{'absolute_delta': 1.25}])['passed']


def test_native_decision_rejects_invalid_envelope():
    from app.agent_decider import NativeDecisionProvider
    from app.agent_actions import ActionValidationError

    class InvalidModel:
        def generate_tools(self, request):
            return ModelResponse('', 'fixture', [{'type': 'text', 'function': {
                'name': 'finish', 'arguments': '{}'}}])

    with pytest.raises(ActionValidationError):
        asyncio.run(NativeDecisionProvider(InvalidModel())({
            'task': {'requirements': [{'id': 's1', 'capability': 'season_summary', 'arguments': {}}]},
            'coverage': {'missing': ['s1']}}))


def test_evaluation_summary_reports_observed_usage_and_check_denominators():
    from scripts.evaluate_semantic_routing import summarize
    rows = [{'arm': 'model', 'passed': True, 'latency_ms': 100, 'runtime_failure': False,
             'checks': {'entity_0': True, 'numeric_0_points_total': True},
             'model_requests': [{'usage': {'total_tokens': 12}}, {'usage': None}]}]
    summary = summarize(rows)['model']
    assert summary['p95_latency_ms'] == 100
    assert summary['usage']['observed_total_tokens'] == 12
    assert summary['usage']['requests_with_usage'] == 1
    assert summary['check_accuracy']['entity'] == {'passed': 1, 'total': 1, 'accuracy': 1}


@pytest.mark.parametrize('endpoint', ['/ask', '/answer'])
def test_structured_entrypoint_does_not_drop_mixed_requirement(tmp_path, endpoint):
    call = ('player_efg', {'entity': {'mention': '库里'}, 'season_id': '2023-24',
            'competition_phase': 'regular_season', 'knowledge_question': '这个指标如何定义？'})
    app = native_app(tmp_path, [call])
    body = request(app, endpoint, {'question': '库里 2023-24 常规赛 eFG% 多少，这个指标如何定义？'}).json()
    assert body['status'] == 'unsupported'
    assert body['tool_calls_used'] == 0


def test_zero_tool_budget_still_does_not_execute_agent(tmp_path):
    app = native_app(tmp_path, [summary()], agent_max_tool_calls=0)
    body = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛总得分'}).json()
    assert body['status'] == 'incomplete'
    assert body['reason_code'] == 'tool_budget_exceeded'
    assert not body['steps']


def test_window_partial_is_preserved_in_answer(tmp_path):
    call = ('window_query', {**summary()[1], 'last_n': 5})
    app = native_app(tmp_path, [call])
    body = request(app, '/answer', {'question': '库里 2023-24 常规赛最近5次出场得分'}).json()
    assert body['status'] == 'partial'
    assert body['tool_result']['actual_sample_count'] == 2


def test_not_computable_scope_is_saved_for_followup(tmp_path):
    no_rows = summary('2020-21')
    app = native_app(tmp_path, [no_rows])
    body = request(app, '/answer', {'question': '库里 2020-21 常规赛总得分'}).json()
    assert body['status'] == 'not_computable'
    saved = app.state.agent_conversations.get(body['conversation_id'])
    assert saved.season_id == '2020-21'


def test_knowledge_failure_does_not_overwrite_previous_scope(tmp_path):
    from tests.test_agent_routes import FakeKnowledge
    app = native_app(tmp_path, [summary(), ('knowledge_glossary', {'question': '什么是进攻三秒'})], agent_knowledge_adapter=FakeKnowledge(fail=True))
    first = request(app, '/agent/answer', {'question': '库里 2023-24 常规赛总得分'}).json()
    second = request(app, '/agent/answer', {'question': '什么是进攻三秒', 'conversation_id': first['conversation_id']}).json()
    assert second['status'] == 'incomplete'
    assert app.state.agent_conversations.get(first['conversation_id']).season_id == '2023-24'
