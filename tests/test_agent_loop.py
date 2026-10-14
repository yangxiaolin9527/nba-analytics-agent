import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app

from app.agent_actions import AgentAction, CoverageLedger, LoopBudget, Observation
from app.agent_loop import AgentLoop
from app.agent_tasks import Requirement, TaskProfile
from app.agent_decider import ModelDecisionProvider, build_decision_prompt
from scripts.model_adapter import ModelRequest, ModelResponse


class Adapter:
    route = "structured"

    def __init__(self, name, results):
        self.name = name
        self.results = list(results)
        self.calls = []

    async def execute(self, question, arguments):
        self.calls.append((question, arguments))
        result = self.results.pop(0) if self.results else {"status": "complete"}
        if isinstance(result, Exception):
            raise result
        return result


class Registry:
    def __init__(self, *adapters):
        self.adapters = {item.name: item for item in adapters}

    def get(self, name):
        return self.adapters.get(name)


def run(loop, decisions):
    queue = iter(decisions)

    async def decide(context):
        del context
        return next(queue)

    loop.decider = decide
    return asyncio.run(loop.run("q", task=TaskProfile("q", "structured", (Requirement("s1", "season_summary", {"entity_id": "201939", "metrics": ["points_total"]}),))))


def test_loop_requires_observed_tool_result_before_finish():
    adapter = Adapter("season_summary", [{"status": "complete", "points_total": 50}])
    result = run(
        AgentLoop(Registry(adapter)),
        [
            {"action": "finish", "reason": "done"},
            {"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}},
            {"action": "finish", "reason": "observed"},
        ],
    )
    assert result.status == "complete"
    assert result.observations[0].result["points_total"] == 50
    assert result.termination_reason == "coverage_complete"


def test_loop_rejects_duplicate_tool_calls():
    adapter = Adapter("season_summary", [{"status": "partial", "points_total": 50}])
    result = run(
        AgentLoop(Registry(adapter)),
        [
            {"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}},
            {"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}},
        ],
    )
    assert result.status == "partial"
    assert result.termination_reason == "duplicate_tool_call"
    assert len(adapter.calls) == 1


def test_loop_allows_observation_failure_to_change_next_action():
    adapter = Adapter("season_summary", [RuntimeError("temporary")])
    result = run(
        AgentLoop(Registry(adapter)),
        [
            {"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}},
            {"action": "clarify", "reason": "数据暂不可用"},
        ],
    )
    assert result.status == "incomplete"
    assert result.observations[0].status == "failed"
    assert result.termination_reason == "unjustified_clarify"


def test_loop_repairs_one_invalid_decision_without_retrying_provider_failures():
    adapter = Adapter("season_summary", [{"status": "complete", "points_total": 50}])
    result = run(
        AgentLoop(Registry(adapter)),
        [
            {"action": "call", "capability": "season_summary", "arguments": {}, "extra": "invalid"},
            {"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}},
            {"action": "finish", "reason": "observed"},
        ],
    )
    assert result.status == "complete"
    assert result.decisions == 2
    assert len(adapter.calls) == 1


def test_loop_result_keeps_decision_failures_local_to_one_run():
    class FlakyDecider:
        def __init__(self):
            self.calls = 0

        async def __call__(self, context):
            self.calls += 1
            if self.calls == 1:
                return {"action": "teleport"}
            if context["observations"]:
                return {"action": "finish"}
            return {"action": "call", "capability": "season_summary", "arguments": {
                "entity_id": "201939", "metrics": ["points_total"]}}

    decider = FlakyDecider()
    loop = AgentLoop(Registry(Adapter("season_summary", [{"status": "complete", "points_total": 50}] * 2)), decider=decider)
    task = TaskProfile("q", "structured", (Requirement("s1", "season_summary", {
        "entity_id": "201939", "metrics": ["points_total"]}),))
    first = asyncio.run(loop.run("q", task=task))
    second = asyncio.run(loop.run("q", task=task))
    assert first.status == second.status == "complete"
    assert len(first.decision_failures) == 1
    assert first.decision_failures[0]["sequence"] == 1
    assert first.decision_failures[0]["code"] == "invalid_action_schema"
    assert second.decision_failures == ()


def test_loop_enforces_tool_budget():
    adapter = Adapter("season_summary", [{"status": "complete"}])
    result = run(
        AgentLoop(Registry(adapter), budget=LoopBudget(max_tool_calls=0)),
        [{"action": "call", "capability": "season_summary", "arguments": {"entity_id": "201939", "metrics": ["points_total"]}}],
    )
    assert result.status == "incomplete"
    assert result.termination_reason == "tool_budget_exceeded"


def test_coverage_ledger_requires_all_required_capabilities():
    ledger = CoverageLedger(("season_summary", "player_efg"))
    ledger.observe(Observation(id="o1", capability="season_summary", status="complete", result={}, requirement_id="season_summary"))
    assert not ledger.complete
    ledger.observe(Observation(id="o2", capability="player_efg", status="complete", result={}, requirement_id="player_efg"))
    assert ledger.complete


def test_action_rejects_code_and_sql_arguments():
    with pytest.raises(ValueError):
        AgentAction(action="call", capability="season_summary", arguments={"sql": "SELECT 1"})


class DecisionModel:
    model = "fixture-decision-model"

    def __init__(self, text):
        self.text = text
        self.requests = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(self.text, self.model)


def test_decision_provider_requests_json_and_returns_validated_action():
    model = DecisionModel('{"action":"call","capability":"season_summary","arguments":{"entity_id":"201939"}}')
    provider = ModelDecisionProvider(model, allowed_capabilities={"season_summary"})
    action = asyncio.run(provider({"question": "q", "observations": [], "budget": {}}))
    assert action.action == "call"
    assert action.capability == "season_summary"
    assert model.requests[0].response_format == "json_object"
    assert "hidden chain" in model.requests[0].prompt


def test_decision_provider_rejects_unregistered_capability():
    model = DecisionModel('{"action":"call","capability":"knowledge_glossary","arguments":{}}')
    provider = ModelDecisionProvider(model, allowed_capabilities={"season_summary"})
    with pytest.raises(ValueError, match="capability is not allowed"):
        asyncio.run(provider({"question": "q", "observations": [], "budget": {}}))


def test_decision_provider_rejects_non_json_output():
    provider = ModelDecisionProvider(DecisionModel("not json"))
    with pytest.raises(ValueError, match="valid JSON"):
        asyncio.run(provider({"question": "q", "observations": [], "budget": {}}))


def test_decision_provider_accepts_fenced_json_with_surrounding_prose():
    model = DecisionModel(
        'Here is the next action:\n```JSON\n{"action":"finish","reason":"covered"}\n```'
    )
    action = asyncio.run(ModelDecisionProvider(model)({"question": "q", "observations": [], "budget": {}}))
    assert action.action == "finish"
    assert action.reason == "covered"


def test_decision_provider_rejects_json_array_even_when_embedded_in_prose():
    provider = ModelDecisionProvider(DecisionModel("The answer is [1, 2]."))
    with pytest.raises(ValueError, match="JSON"):
        asyncio.run(provider({"question": "q", "observations": [], "budget": {}}))


@pytest.mark.parametrize(("response", "error_code"), [
    ("not json", "invalid_json"),
    ('{"action":"teleport"}', "invalid_action_schema"),
    ('{"action":"call","capability":"unknown","arguments":{}}', "unregistered_capability"),
])
def test_decision_provider_classifies_safe_failures_without_raw_response(response, error_code):
    class RecordingErrors:
        def __init__(self):
            self.events = []
        def record(self, code, elapsed_ms):
            self.events.append({"code": code, "elapsed_ms": elapsed_ms})

    errors = RecordingErrors()
    provider = ModelDecisionProvider(DecisionModel(response), allowed_capabilities={"season_summary"},
                                     diagnostics=errors)
    with pytest.raises(ValueError):
        asyncio.run(provider({"question": "q", "observations": [], "budget": {}}))
    assert errors.events[0]["code"] == error_code
    assert isinstance(errors.events[0]["elapsed_ms"], float)
    assert response not in repr(errors.events)


def test_decision_prompt_contains_observations_without_chain_of_thought_request():
    prompt = build_decision_prompt({
        "question": "compare players",
        "observations": [{"id": "o1", "capability": "season_summary", "status": "complete", "result": {"points_total": 50}}],
        "budget": {"decisions": 2, "tool_calls": 1},
    })
    assert "points_total" in prompt
    assert "Do not provide chain-of-thought" in prompt


def test_decision_prompt_shows_exact_action_variants_and_grounded_arguments():
    prompt = build_decision_prompt({"question": "q"}, allowed_capabilities={"player_efg"})
    for action in ("call", "finish", "clarify", "unsupported"):
        assert '"action":"' + action + '"' in prompt
    assert "500" in prompt
    assert "observation IDs" in prompt
    assert "requirement" in prompt


def test_schema_correction_feedback_contains_fixed_detail_code():
    from tests.test_agent_loop_contract import SequenceDecider, SUMMARY, FINISH
    decider = SequenceDecider([{"action": "teleport"}, SUMMARY, FINISH])
    adapter = Adapter("season_summary", [{"status": "complete", "points_total": 50}])
    task = TaskProfile("q", "structured", (Requirement("s1", "season_summary", SUMMARY["arguments"]),))
    loop = AgentLoop(Registry(adapter), decider=decider)
    result = asyncio.run(loop.run("q", task=task))
    assert result.status == "complete"
    assert decider.contexts[1]["feedback"]["detail_code"] == "invalid_action"
    assert '"action":"finish"' in decider.contexts[1]["feedback"]["message"]


def test_opt_in_loop_route_is_isolated_from_rules_route(tmp_path):
    from tests.test_agent_loop_contract import make_app, post, SequenceDecider, SUMMARY, FINISH
    app = make_app(tmp_path, SequenceDecider([SUMMARY, FINISH]))
    body = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    assert body["status"] == "complete"
    assert body["observations"][0]["result"]["points_total"] == 50


def test_opt_in_loop_uses_application_model_as_decider(tmp_path):
    from tests.test_agent_loop_contract import make_app, post
    model = DecisionModel('{"action":"finish","reason":"no required work"}')
    app = make_app(tmp_path, None, model=model)
    body = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    assert body["status"] == "incomplete"
    assert body["tool_calls"] == 0
    assert body["model"] == "fixture-decision-model"
    assert model.requests[0].response_format == "json_object"
