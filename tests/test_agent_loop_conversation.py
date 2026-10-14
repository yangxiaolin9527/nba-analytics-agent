import asyncio
import pytest

from httpx import ASGITransport, AsyncClient

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


SUMMARY = {
    "action": "call",
    "capability": "season_summary",
    "arguments": {
        "subject": "player",
        "entity_id": "201939",
        "season_id": "2023-24",
        "competition_phase": "regular_season",
        "metrics": ["points_total"],
    },
}
EFG = {
    "action": "call",
    "capability": "player_efg",
    "arguments": {
        "entity_id": "201939",
        "season_id": "2023-24",
        "competition_phase": "regular_season",
    },
}
JAMES_EFG = {**EFG, "arguments": {**EFG["arguments"], "entity_id": "2544"}}
FINISH = {"action": "finish"}


def make_app(tmp_path, decider, *, knowledge=None):
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    return create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_loop_enabled=True,
        agent_loop_decider=decider,
        agent_knowledge_adapter=knowledge or FakeKnowledge(),
    )


def post(app, payload):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/agent/loop", json=payload)

    return asyncio.run(send())


def test_loop_returns_conversation_id_and_inherits_scope(tmp_path):
    decider = SequenceDecider([SUMMARY, FINISH, EFG, FINISH])
    app = make_app(tmp_path, decider)

    first = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    second = post(app, {
        "question": "那他的 eFG% 呢？",
        "conversation_id": first["conversation_id"],
    }).json()

    assert first["status"] == "complete"
    assert first["turn_index"] == 1
    assert second["status"] == "complete"
    assert second["conversation_id"] == first["conversation_id"]
    assert second["turn_index"] == 2
    assert second["observations"][0]["arguments"] == EFG["arguments"]
    assert second["context"]["inherited"]["player_ids"] == ["201939"]


def test_loop_rejects_invalid_explicit_season_before_decision(tmp_path):
    decider = SequenceDecider([SUMMARY, FINISH])
    app = make_app(tmp_path, decider)
    first = post(app, {'question': '库里 2023-24 常规赛总得分？'}).json()
    second = post(app, {'question': '那库里 2020-2022 常规赛总得分？',
                        'conversation_id': first['conversation_id']}).json()
    assert second['status'] == 'clarification_required'
    assert second['reason_code'] == 'invalid_season'
    assert second['observations'] == []
    assert len(decider.contexts) == 1  # no redundant finish decision after coverage


def test_loop_allows_explicit_metric_switch_without_stale_metric(tmp_path):
    decider = SequenceDecider([SUMMARY, FINISH, EFG, FINISH])
    app = make_app(tmp_path, decider)
    first = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    second = post(app, {
        "question": "库里 2023-24 常规赛 eFG% 是多少？",
        "conversation_id": first["conversation_id"],
    }).json()

    assert second["status"] == "complete"
    assert second["observations"][0]["capability"] == "player_efg"
    assert second["observations"][0]["arguments"]["season_id"] == "2023-24"


def test_loop_clarification_preserves_previous_scope(tmp_path):
    decider = SequenceDecider([SUMMARY, FINISH])
    app = make_app(tmp_path, decider)
    first = post(app, {"question": "库里 2023-24 常规赛总得分？"}).json()
    second = post(app, {
        "question": "那和詹姆斯比呢？",
        "conversation_id": first["conversation_id"],
    }).json()

    assert second["status"] == "clarification_required"
    assert second["turn_index"] == 2
    assert second["reason_code"] == "comparison_metric_required"
    store = app.state.agent_conversations
    snapshot = store.get(first["conversation_id"])
    assert snapshot is not None
    assert snapshot.season_id == "2023-24"
    assert snapshot.metric_terms == ("points_total",)


def test_loop_partial_keeps_grounded_scope_for_followup(tmp_path):
    class UnavailableKnowledge:
        name = "knowledge_glossary"
        route = "knowledge"

        async def execute(self, question, arguments):
            from app.agent_routes import AgentCapabilityError

            raise AgentCapabilityError("knowledge_unavailable", "fixture knowledge unavailable")

    decider = SequenceDecider([EFG, {"action": "call", "capability": "knowledge_glossary", "arguments": {}}])
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_loop_enabled=True,
        agent_loop_decider=decider,
        agent_knowledge_adapter=UnavailableKnowledge(),
    )
    first = post(app, {
        "question": "库里 2023-24 常规赛 eFG% 是多少并解释含义？",
    }).json()

    assert first["status"] == "partial"
    snapshot = app.state.agent_conversations.get(first["conversation_id"])
    assert snapshot is not None
    assert snapshot.player_ids == ("201939",)
    assert snapshot.season_id == "2023-24"
    assert snapshot.metric_terms == ("efg_percentage",)


def test_loop_efg_followup_can_compare_new_player(tmp_path):
    class ComparisonDecider:
        async def __call__(self, context):
            observed = {item.get("requirement_id") for item in context["observations"]}
            for requirement in context["task"]["requirements"]:
                if requirement["id"] not in observed:
                    arguments = dict(requirement["arguments"])
                    if requirement["capability"] == "efg_compare":
                        arguments = {"left_observation_id": "o1", "right_observation_id": "o2"}
                    return {"action": "call", "capability": requirement["capability"], "arguments": arguments}
            return FINISH
    decider = ComparisonDecider()
    app = make_app(tmp_path, decider)
    first = post(app, {"question": "库里 2023-24 常规赛 eFG% 是多少？"}).json()
    second = post(app, {"question": "那和詹姆斯比呢？", "conversation_id": first["conversation_id"]}).json()
    assert second["route"] == "comparison"
    assert second["status"] == "partial"
    assert [item["capability"] for item in second["observations"]] == ["player_efg", "player_efg"]


def test_loop_same_season_conflict_requires_clarification(tmp_path):
    decider = SequenceDecider([EFG, FINISH, EFG, FINISH])
    app = make_app(tmp_path, decider)
    first = post(app, {"question": "库里 2023-24 常规赛 eFG% 是多少？"}).json()
    second = post(app, {
        "question": "同一赛季的库里 2022-23 常规赛 eFG% 是多少？",
        "conversation_id": first["conversation_id"],
    }).json()
    assert second["status"] == "clarification_required"
    assert second["reason_code"] == "conversation_scope_conflict"
    assert second["observations"] == []


@pytest.mark.parametrize("question", ["这个指标怎么理解？", "那这个指标有什么局限？", "该指标是什么？"])
def test_loop_definition_followup_is_grounded_before_model_and_tool(tmp_path, question):
    from scripts.evaluate_agent_loop_conversation import FixtureDecider

    class GroundedKnowledge(FakeKnowledge):
        async def execute(self, tool_question, arguments):
            assert "eFG%" in tool_question
            assert arguments == {"term": "effective field goal percentage"}
            if "局限" in question:
                assert "局限" in tool_question
            return await super().execute(tool_question, arguments)

    app = make_app(tmp_path, FixtureDecider(), knowledge=GroundedKnowledge())
    first = post(app, {"question": "库里 2023-24 常规赛 eFG% 是多少？"}).json()
    before = app.state.agent_conversations.get(first["conversation_id"])
    second = post(app, {"question": question, "conversation_id": first["conversation_id"]}).json()
    assert second["status"] == "complete"
    assert second["route"] == "knowledge"
    assert [o["capability"] for o in second["observations"]] == ["knowledge_glossary"]
    assert second["context"]["requirements"][0]["arguments"] == {"term": "effective field goal percentage"}
    after = app.state.agent_conversations.get(first["conversation_id"])
    assert after.player_ids == before.player_ids
    assert after.metric_terms == before.metric_terms
    assert after.turn_index == before.turn_index + 1


@pytest.mark.parametrize("metrics", [(), ("points_total",), ("efg_percentage", "points_total")])
def test_loop_definition_reference_requires_unique_supported_metric(tmp_path, metrics):
    decider = SequenceDecider([])
    app = make_app(tmp_path, decider)
    app.state.agent_conversations.put(conversation_id="unclear", metric_terms=metrics)
    body = post(app, {"question": "这个指标怎么理解？", "conversation_id": "unclear"}).json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "conversation_metric_required"
    assert body["tool_calls"] == body["decisions"] == 0
    assert not decider.contexts


def test_explicit_definition_does_not_inherit_stale_efg(tmp_path):
    from app.agent_tasks import build_task_profile
    from app.conversation import ConversationSnapshot
    make_app(tmp_path, SequenceDecider([]))
    task = build_task_profile(tmp_path / "agent.sqlite", "TS% 是什么？",
                             ConversationSnapshot("explicit", 1, metric_terms=("efg_percentage",)))
    assert task.route == "knowledge"
    assert task.requirements[0].question == "TS% 是什么？"
    assert task.requirements[0].arguments == {"term": "TS% 是什么？"}


def test_definition_context_does_not_require_structured_database():
    from app.agent_tasks import build_task_profile
    from app.conversation import ConversationSnapshot
    task = build_task_profile(None, "这个指标怎么理解？",
                             ConversationSnapshot("knowledge", 1, metric_terms=("efg_percentage",)))
    assert task.route == "knowledge"
    assert task.requirements[0].arguments == {"term": "effective field goal percentage"}
    assert "eFG%" in task.requirements[0].question


def test_contextual_definition_keeps_argument_validation_strict(tmp_path):
    from app.agent_tasks import build_task_profile
    from app.conversation import ConversationSnapshot
    from app.agent_actions import AgentAction
    make_app(tmp_path, SequenceDecider([]))
    task = build_task_profile(tmp_path / "agent.sqlite", "这个指标怎么理解？",
                             ConversationSnapshot("strict", 1, metric_terms=("efg_percentage",)))
    with pytest.raises(ValueError, match="grounded task"):
        task.resolve(AgentAction("call", "knowledge_glossary", {"term": "true shooting percentage"}), [])
