import asyncio
import json
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent_decider import ModelDecisionProvider, NativeDecisionProvider
from scripts.evaluate_agent_loop_conversation import FixtureDecider
from scripts.model_adapter import ModelResponse
from tests.test_agent_loop_contract import make_app, post


@pytest.mark.parametrize('arguments, detail', [
    ({'term': 'explain eFG%; formula', 'sql': 'SELECT 1'}, 'forbidden_argument'),
    ({'term': {'sql': 'SELECT 1'}}, 'forbidden_argument'),
    ({'term': ['text;']}, 'statement_separator'),
    ({'context_metric': 'metric;'}, 'statement_separator'),
    ({'extra': {'term': 'text;'}}, 'statement_separator'),
])
def test_knowledge_text_exception_preserves_execution_argument_checks(arguments, detail):
    from app.agent_actions import AgentAction, ActionValidationError
    from app.agent_routes import RouteStep
    from pydantic import ValidationError
    with pytest.raises(ActionValidationError) as caught:
        AgentAction('call', 'knowledge_glossary', arguments)
    assert caught.value.detail_code == detail
    with pytest.raises(ValidationError):
        RouteStep(id='s1', capability='knowledge_glossary', arguments=arguments)


class ReplyModel:
    model = "diagnostic-fixture"

    def __init__(self, replies):
        self.replies = iter(replies)

    def generate(self, request):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(reply, self.model)


class NativeReplyModel:
    model = "native-diagnostic-fixture"

    def __init__(self, replies):
        self.replies = iter(replies)

    def generate_tools(self, request):
        reply = next(self.replies)
        name, arguments = reply
        return ModelResponse('', self.model, [{
            'id': 'decision-call', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)},
        }])


def test_new_request_does_not_inherit_failed_request_diagnostics(tmp_path):
    provider = ModelDecisionProvider(ReplyModel(['{"action":"teleport"}'] * 2))
    app = make_app(tmp_path, provider)
    failed = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    app.state.agent_loop.decider = FixtureDecider()
    succeeded = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    assert len(failed["decision_failures"]) == 2
    assert succeeded["status"] == "complete"
    assert succeeded["decision_failures"] == []


def test_concurrent_conversations_do_not_share_diagnostics(tmp_path):
    async def send():
        entered, release = asyncio.Event(), asyncio.Event()

        class ConcurrentDecider(FixtureDecider):
            async def __call__(self, context):
                if "总得分" in context["question"] and not context["observations"]:
                    if not context.get("feedback"):
                        return {"action": "teleport"}
                    entered.set()
                    await release.wait()
                return await super().__call__(context)

        app = make_app(tmp_path, ConcurrentDecider())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/agent/loop", json={
                "question": "库里 2023-24 常规赛总得分？", "conversation_id": "first"}))
            await asyncio.wait_for(entered.wait(), 2)
            try:
                clean = (await client.post("/agent/loop", json={
                    "question": "库里 2023-24 常规赛 eFG% 是多少？", "conversation_id": "second"})).json()
                assert clean["status"] == "complete"
                assert clean["decision_failures"] == []
            finally:
                release.set()
                repaired = (await pending).json()
            assert repaired["status"] == "complete"
            assert len(repaired["decision_failures"]) == 1
    asyncio.run(send())


def test_shared_provider_exceptions_remain_local_under_concurrency(tmp_path):
    async def send():
        entered, release = threading.Event(), threading.Event()

        class ConcurrentModel:
            model = "concurrent-fixture"

            def generate(self, request):
                if "总得分" in request.prompt:
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError()
                    return ModelResponse('{"action":"finish","reason":null}', self.model)
                return ModelResponse('{"action":"teleport"}', self.model)

        app = make_app(tmp_path, ModelDecisionProvider(ConcurrentModel()))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/agent/loop", json={
                "question": "库里 2023-24 常规赛总得分？", "conversation_id": "first"}))
            # Python 3.14 can leave the default executor alive when
            # asyncio.to_thread(threading.Event.wait, ...) is used here.
            # Polling keeps this synchronization fully on the event loop.
            deadline = asyncio.get_running_loop().time() + 2
            while not entered.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert entered.is_set()
            try:
                second = (await client.post("/agent/loop", json={
                    "question": "库里 2023-24 常规赛 eFG% 是多少？", "conversation_id": "second"})).json()
            finally:
                release.set()
                first = (await pending).json()
            assert [d["detail_code"] for d in first["decision_failures"]] == ["reason_type"] * 2
            assert [d["detail_code"] for d in second["decision_failures"]] == ["invalid_action"] * 2
    asyncio.run(send())


@pytest.mark.parametrize(("reply", "code", "detail"), [
    ('{"action":"call","capability":"season_summary","arguments":[]}', "invalid_action_schema", "arguments_type"),
    ('{"action":"finish","reason":"x","secret-field":"secret-value"}', "invalid_action_schema", "unknown_fields"),
    ('{"action":[]}', "invalid_action_schema", "invalid_action"),
    ('{"action":"call","capability":7}', "invalid_action_schema", "capability_type"),
    ('{"action":"finish","reason":null}', "invalid_action_schema", "reason_type"),
    ('{"action":"finish","reason":"' + 'x' * 501 + '"}', "invalid_action_schema", "reason_length"),
    ('{"action":"finish","capability":"season_summary"}', "invalid_action_schema", "non_call_capability"),
    ('{"action":"finish","arguments":{"x":1}}', "invalid_action_schema", "non_call_arguments"),
    ('{"action":"call","capability":""}', "invalid_action_schema", "capability_name"),
    ('{"action":"call","capability":"season_summary","arguments":{"sql":"secret-value"}}', "invalid_action_schema", "forbidden_argument"),
    ('{"action":"call","capability":"season_summary","arguments":{"term":"secret-value;"}}', "invalid_action_schema", "statement_separator"),
    ('{"action":"call","capability":"unknown"}', "unregistered_capability", "unregistered_capability"),
    ("secret-value", "invalid_json", "invalid_json"),
    ('[]', "invalid_json", "json_object_required"),
    ('', "invalid_json", "empty_response"),
])
def test_provider_exception_carries_safe_fixed_diagnostic(reply, code, detail):
    provider = ModelDecisionProvider(ReplyModel([reply]), allowed_capabilities={"season_summary"})
    with pytest.raises(ValueError) as caught:
        asyncio.run(provider({"budget": {}}))
    assert caught.value.code == code
    assert caught.value.detail_code == detail
    assert "secret-value" not in str(caught.value)
    assert not hasattr(provider, "last_failure")


def test_native_decision_provider_accepts_grounded_requirement_selection():
    provider = NativeDecisionProvider(NativeReplyModel([
        ('select_requirement', {'requirement_id': 's1'}),
    ]))
    action = asyncio.run(provider({
        'task': {'requirements': [{'id': 's1', 'capability': 'season_summary', 'arguments': {'entity_id': '201939'}}]},
        'observations': [], 'coverage': {'missing': ['s1']}, 'budget': {'remaining_seconds': 5},
    }))
    assert action.action == 'call'
    assert action.capability == 'season_summary'
    assert action.arguments == {'entity_id': '201939'}


def test_native_protocol_exposes_one_selector_and_dependency_ready_ids():
    from tests.test_semantic_routing import NativeFixture
    model = NativeFixture([('select_requirement', {'requirement_id': 'b'})])
    provider = NativeDecisionProvider(model)
    action = asyncio.run(provider({
        'task': {'requirements': [
            {'id': 'a', 'capability': 'player_efg', 'arguments': {}, 'dependencies': []},
            {'id': 'b', 'capability': 'knowledge_glossary', 'arguments': {'term': 'metric'}, 'dependencies': []},
            {'id': 'c', 'capability': 'efg_compare', 'arguments': {}, 'dependencies': ['a', 'd']},
            {'id': 'd', 'capability': 'player_efg', 'arguments': {}, 'dependencies': []},
        ]}, 'observations': [], 'coverage': {'missing': ['a', 'b', 'c', 'd']}, 'budget': {}}))
    tool = model.requests[0].tools[0]['function']
    assert len(model.requests[0].tools) == 1
    assert tool['name'] == 'select_requirement'
    assert tool['parameters']['properties']['requirement_id']['enum'] == ['a', 'b', 'd']
    assert action.capability == 'knowledge_glossary'
    assert action.arguments == {'term': 'metric'}


def test_native_repair_trace_records_failure_while_span_is_open(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from tests.test_semantic_routing import NativeFixture
    exporter, tracer_provider = InMemorySpanExporter(), TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = NativeFixture([[], ('select_requirement', {'requirement_id': 's1'})])
    body = post(make_app(tmp_path, NativeDecisionProvider(model),
        tracer=tracer_provider.get_tracer('native-repair')), {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['status'] == 'complete', body
    assert body['decision_failures'][0]['detail_code'] == 'native_call_count'
    assert 'select_requirement' in model.requests[1].prompt
    assert 'Legal shapes' not in model.requests[1].prompt
    failed = [s for s in exporter.get_finished_spans() if s.name == 'nba.agent.loop.decision'][0]
    assert failed.attributes['nba.agent.decision_detail_code'] == 'native_call_count'
    assert failed.attributes['nba.agent.native_call_count'] == 0
    assert failed.attributes['output.value.preview']


def test_decision_trace_exception_events_do_not_leak_provider_message(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exporter, provider = InMemorySpanExporter(), TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    async def failure(context):
        raise RuntimeError('secret-value-review')
    body = post(make_app(tmp_path, failure, tracer=provider.get_tracer('safe-decision')),
                {'question': '库里 2023-24 常规赛总得分？'}).json()
    assert body['reason_code'] == 'decision_failed'
    spans = exporter.get_finished_spans()
    exported = [dict(s.attributes) for s in spans]
    exported += [dict(e.attributes) for s in spans for e in s.events]
    exported += [s.status.description for s in spans]
    assert 'secret-value-review' not in json.dumps(exported)


@pytest.mark.parametrize(('reply', 'detail'), [
    (('select_requirement', {'requirement_id': 'blocked'}), 'native_requirement_not_ready'),
    (('select_requirement', {'requirement_id': 's1', 'efg': 99}), 'native_argument_keys'),
    (('execute_sql', {'requirement_id': 's1'}), 'native_tool_name'),
])
def test_native_selector_rejects_unready_extra_or_unregistered_choices(reply, detail):
    provider = NativeDecisionProvider(NativeReplyModel([reply]))
    context = {'task': {'requirements': [
        {'id': 's1', 'capability': 'season_summary', 'arguments': {'entity_id': 'grounded'}},
        {'id': 'blocked', 'capability': 'efg_compare', 'arguments': {}, 'dependencies': ['s1', 'missing']},
    ]}, 'observations': [], 'coverage': {'missing': ['s1', 'blocked']}}
    with pytest.raises(ValueError) as caught:
        asyncio.run(provider(context))
    assert caught.value.detail_code == detail
    assert caught.value.diagnostics['native_call_count'] == 1


def test_native_selector_only_retries_transient_failure_with_remaining_retry_budget():
    from tests.test_semantic_routing import NativeFixture
    context = {'task': {'requirements': [
        {'id': 's1', 'capability': 'season_summary', 'arguments': {'entity_id': 'grounded'}},
    ]}, 'observations': [{'id': 'o1', 'requirement_id': 's1', 'status': 'failed', 'reason_code': 'tool_timeout'}],
        'coverage': {'missing': ['s1']}, 'budget': {'remaining_retries': 1}}
    model = NativeFixture([('select_requirement', {'requirement_id': 's1'})])
    action = asyncio.run(NativeDecisionProvider(model)(context))
    assert action.action == 'call' and action.arguments == {'entity_id': 'grounded'}
    context['budget']['remaining_retries'] = 0
    assert asyncio.run(NativeDecisionProvider(model)(context)).action == 'finish'
    assert len(model.requests) == 1


def test_failed_dependency_terminates_without_selecting_blocked_comparison():
    from app.agent_loop import AgentLoop
    from app.agent_routes import CapabilityRegistry, StructuredToolAdapter
    from app.agent_tasks import TaskProfile, Requirement
    from tests.test_agent_loop_contract import ResultAdapter
    from tests.test_semantic_routing import NativeFixture
    scope = {'season_id': '2023-24', 'competition_phase': 'regular_season'}
    left, right = {**scope, 'entity_id': '201939'}, {**scope, 'entity_id': '2544'}
    task = TaskProfile('q', 'comparison', (
        Requirement('s1', 'player_efg', left), Requirement('s2', 'player_efg', right),
        Requirement('s3', 'efg_compare', {**scope, 'left_entity_id': '201939', 'right_entity_id': '2544'}, ('s1', 's2'))))
    model = NativeFixture([('select_requirement', {'requirement_id': 's1'}),
                           ('select_requirement', {'requirement_id': 's2'})])
    adapter = ResultAdapter('player_efg', [
        {'status': 'not_computable', 'reason_code': 'no_observation'},
        {'status': 'complete', 'efg_percentage': 50,
         'inputs': {'field_goals_made': 5, 'three_pointers_made': 0, 'field_goals_attempted': 10, 'sample_count': 1}},
    ])
    result = asyncio.run(AgentLoop(CapabilityRegistry([adapter, StructuredToolAdapter('efg_compare')]),
        decider=NativeDecisionProvider(model)).run('q', task=task))
    assert result.status == 'partial'
    assert result.termination_reason == 'coverage_incomplete'
    assert result.tool_calls == 2
    assert result.decisions == 3
    assert len(model.requests) == 2


@pytest.mark.parametrize(("error", "expected"), [(TimeoutError("secret-value"), "decision_timeout"),
                                                      (RuntimeError("secret-value"), "decision_failed")])
def test_loop_records_runtime_diagnostic_without_retry_or_exception_message(tmp_path, error, expected):
    model = ReplyModel([error])
    provider = ModelDecisionProvider(model)
    body = post(make_app(tmp_path, provider), {"question": "库里 2023-24 常规赛总得分？"}).json()
    assert body["termination_reason"] == expected
    assert body["decisions"] == 1
    assert body["decision_failures"][0]["code"] == expected
    assert "secret-value" not in json.dumps(body)


def test_cancelled_decision_is_not_converted_to_a_failure():
    async def send():
        async def cancelled(context):
            raise asyncio.CancelledError()

        from app.agent_loop import AgentLoop
        from app.agent_tasks import TaskProfile, Requirement
        loop = AgentLoop(None, decider=cancelled)
        with pytest.raises(asyncio.CancelledError):
            await loop.run("q", task=TaskProfile("q", "structured", (Requirement("s1", "season_summary", {}),)))
    asyncio.run(send())
