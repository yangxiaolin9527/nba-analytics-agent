import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent_routes import (
    MIXED_FIXTURE_QUESTION,
    CapabilityRegistry,
    RoutePlan,
    RouteStep,
    execute_agent_plan,
    plan_question,
    StructuredToolAdapter,
)
from app.main import create_app
from scripts.model_adapter import FakeModel
from tests.test_ask import make_ask_database, make_matrix


TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


class FakeKnowledge:
    name = "knowledge_glossary"
    route = "knowledge"

    def __init__(self, *, fail=False, partial=False):
        self.fail = fail
        self.partial = partial
        self.calls = []

    async def execute(self, question, arguments):
        self.calls.append((question, arguments))
        if self.fail:
            raise RuntimeError("fixture knowledge unavailable")
        return {
            "status": "partial" if self.partial else "answered",
            "answer": "eFG% 反映投篮效率，也会受到三分出手结构影响。",
            "claims": [{
                "question": "这个指标代表什么、有什么局限？",
                "claim": "eFG% reflects shooting efficiency.",
                "text": "eFG% 反映投篮效率，也会受到三分出手结构影响。",
                "citation_id": "glossary-efg",
                "excerpt": "effective field goal percentage",
                "source_id": "nba-glossary",
                "source_version": "fixture-v1",
                "page": 4,
                "validation": "validated",
            }],
            "unresolved": [],
        }


def post(app, question):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/agent/answer", json={"question": question})
    return asyncio.run(send())


def post_with_conversation(app, question, conversation_id=None):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            payload = {"question": question}
            if conversation_id:
                payload["conversation_id"] = conversation_id
            return await client.post("/agent/answer", json=payload)
    return asyncio.run(send())


@pytest.mark.parametrize('question', [
    '什么是进攻三秒违例', '什么是防守三秒违例',
    '什么是有效命中率', '什么是 eFG%',
])
def test_prefix_definition_questions_route_to_knowledge(question):
    plan = plan_question(question)
    assert plan.route == 'knowledge'
    assert [step.capability for step in plan.steps] == ['knowledge_glossary']


def test_rule_definition_reaches_knowledge_adapter_with_original_question(tmp_path):
    from app.agent_routes import KnowledgeGlossaryAdapter

    class RecordingAnswerer:
        def __init__(self):
            self.questions = []

        async def aanswer(self, question):
            self.questions.append(question)
            return {
                'status': 'answered', 'answer': '规则回答测试记录',
                'claims': [{'text': '规则回答测试记录', 'citation_id': 'fixture-rule',
                            'excerpt': 'fixture rule excerpt', 'source_id': 'fixture-rulebook',
                            'source_version': 'fixture-v1', 'validation': 'validated'}],
                'unresolved': [],
            }

    answerer = RecordingAnswerer()
    app = create_app(agent_knowledge_adapter=KnowledgeGlossaryAdapter(None, answerer=answerer))
    body = post(app, '什么是进攻三秒违例').json()
    assert body['route'] == 'knowledge'
    assert body['status'] == 'complete'
    assert [step['capability'] for step in body['steps']] == ['knowledge_glossary']
    assert answerer.questions == ['什么是进攻三秒违例']
    assert body['evidence'][0]['knowledge']['citations'][0]['source_id'] == 'fixture-rulebook'


def test_prefix_efg_definition_after_numeric_turn_does_not_inherit_scope(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, '库里 2023-24 常规赛 eFG% 是多少？').json()
    second = post_with_conversation(app, '什么是 eFG%', first['conversation_id']).json()
    assert second['route'] == 'knowledge'
    assert second['status'] == 'complete'
    assert [step['capability'] for step in second['steps']] == ['knowledge_glossary']


def test_mixed_route_runs_ordered_steps_and_preserves_evidence(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    knowledge = FakeKnowledge()
    app = create_app(
        db_path=database, quality_matrix_path=matrix,
        agent_knowledge_adapter=knowledge,
    )

    response = post(app, MIXED_FIXTURE_QUESTION)
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["route"] == "mixed"
    assert body["task_state"]["status"] == "complete"
    assert body["task_state"]["deadline_seconds"] == 10.0
    assert [step["capability"] for step in body["steps"]] == ["player_efg", "knowledge_glossary"]
    assert body["steps"][0]["result"]["efg_percentage"] == pytest.approx(58.333333333333336)
    assert body["evidence"][0]["structured"]["source_sql"].startswith("SELECT")
    assert body["evidence"][0]["structured"]["formula"]
    assert body["evidence"][0]["structured"]["units"] == {"efg_percentage": "percent"}
    assert body["evidence"][0]["structured"]["precision"]["rounding"] == "full_precision"
    assert body["evidence"][1]["knowledge"]["citations"][0]["citation_id"] == "glossary-efg"
    assert len(knowledge.calls) == 1


def test_agent_deadline_returns_partial_after_structured_result():
    class FastStructured:
        name = "player_efg"
        route = "structured"

        async def execute(self, question, arguments):
            return {
                "status": "complete", "efg_percentage": 50.0,
                "inputs": {"field_goals_made": 1, "three_pointers_made": 0,
                            "field_goals_attempted": 2, "sample_count": 1},
                "evidence": {"formula": "(FGM + 0.5 * 3PM) / FGA * 100",
                              "units": {"efg_percentage": "percent"},
                              "precision": {"rounding": "full_precision"}},
            }

    class SlowKnowledge(FakeKnowledge):
        async def execute(self, question, arguments):
            await asyncio.sleep(0.05)
            return await super().execute(question, arguments)

    plan = RoutePlan(route="mixed", steps=[
        RouteStep(id="s1", capability="player_efg", arguments={
            "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
        }),
        RouteStep(id="s2", capability="knowledge_glossary"),
    ])
    response = asyncio.run(execute_agent_plan(
        MIXED_FIXTURE_QUESTION, plan,
        CapabilityRegistry([FastStructured(), SlowKnowledge()]),
        deadline_seconds=0.01,
    ))
    assert response.status == "partial"
    assert response.steps[0].status == "complete"
    assert response.steps[1].status == "failed"
    assert response.steps[1].result["reason_code"] == "agent_timeout"


def test_knowledge_failure_is_partial_and_keeps_structured_result(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(
        db_path=database, quality_matrix_path=matrix,
        agent_knowledge_adapter=FakeKnowledge(fail=True),
    )

    body = post(app, MIXED_FIXTURE_QUESTION).json()

    assert body["status"] == "partial"
    assert "58.3333" in body["answer"]
    assert body["steps"][0]["status"] == "complete"
    assert body["steps"][1]["status"] == "failed"
    assert body["reason_code"] == "knowledge_step_failed"
    assert body["steps"][1]["evidence"]["knowledge"]["unresolved"]
    assert body["steps"][1]["evidence"]["warnings"]
    tool_events = [event for event in body["execution_events"]
                   if event["stage"] == "tool" and event["metadata"].get("step_id") == "s2"]
    assert [event["status"] for event in tool_events] == ["running", "failed"]
    assert tool_events[-1]["metadata"]["reason_code"] == "adapter_failed"


def test_partial_knowledge_answer_is_retained_with_partial_status(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(
        db_path=database, quality_matrix_path=matrix,
        agent_knowledge_adapter=FakeKnowledge(partial=True),
    )

    body = post(app, MIXED_FIXTURE_QUESTION).json()

    assert body["status"] == "partial"
    assert "eFG% 反映投篮效率" in body["answer"]
    assert body["steps"][1]["status"] == "partial"


def test_knowledge_only_route_returns_complete_answer(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_knowledge_adapter=FakeKnowledge(),
    )

    body = post(app, "这个指标代表什么？").json()

    assert body["status"] == "complete"
    assert body["route"] == "knowledge"
    assert body["steps"][0]["status"] == "complete"


def test_knowledge_only_partial_stays_partial(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        agent_knowledge_adapter=FakeKnowledge(partial=True),
    )

    body = post(app, "这个指标代表什么？").json()

    assert body["status"] == "partial"
    assert body["steps"][0]["status"] == "partial"


def test_conversation_id_is_returned_and_context_can_supply_scope(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=FakeKnowledge())
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？")
    first_body = first.json()
    assert first_body["conversation_id"]
    assert first_body["turn_index"] == 1
    second = post_with_conversation(app, "那和詹姆斯比呢？", first_body["conversation_id"])
    second_body = second.json()
    assert second_body["conversation_id"] == first_body["conversation_id"]
    assert second_body["turn_index"] == 2
    assert second_body["route"] == "comparison"
    assert [step["arguments"].get("season_id") for step in second_body["task_state"]["route"]["steps"][:2]] == ["2023-24", "2023-24"]


def test_conversation_scope_conflict_requires_clarification(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=FakeKnowledge())
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    second = post_with_conversation(app, "同一赛季的库里 2022-23 常规赛 eFG% 是多少？", first["conversation_id"]).json()
    assert second["status"] == "clarification_required"
    assert second["reason_code"] == "conversation_scope_conflict"


def conversation_app(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    return create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=FakeKnowledge())


def test_explicit_current_scope_overrides_history(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    second = post_with_conversation(app, "库里 2022-23 常规赛 eFG% 是多少？", first["conversation_id"]).json()
    assert second["status"] == "complete"
    assert second["task_state"]["route"]["steps"][0]["arguments"]["season_id"] == "2022-23"


def test_full_year_followup_overrides_history_and_returns_new_result(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, '库里 2023-24 常规赛总得分？').json()
    second = post_with_conversation(app, '库里 2022-2023 常规赛总得分？', first['conversation_id']).json()
    assert second['status'] == 'complete'
    assert second['task_state']['route']['steps'][0]['arguments']['season_id'] == '2022-23'
    assert first['steps'][0]['result']['points_total'] == 50
    assert second['steps'][0]['result']['points_total'] == 10


@pytest.mark.parametrize('season', ['2020-2022', '2020-22', '2023-2023', '2023-202'])
def test_invalid_explicit_season_cannot_fall_back_to_history(tmp_path, season):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, '库里 2023-24 常规赛总得分？').json()
    second = post_with_conversation(app, f'库里 {season} 常规赛总得分？', first['conversation_id']).json()
    assert second['status'] == 'clarification_required'
    assert second['reason_code'] == 'invalid_season'
    assert second['steps'] == []


def test_explicit_new_player_replaces_history_for_single_query(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    second = post_with_conversation(app, "那利拉德的 eFG% 呢？", first["conversation_id"]).json()
    assert second["route"] == "structured"
    assert second["task_state"]["route"]["steps"][0]["arguments"]["entity_id"] == "203081"


def test_comparison_without_metric_does_not_invent_efg(tmp_path):
    app = conversation_app(tmp_path)
    body = post_with_conversation(app, "比较库里和詹姆斯 2023-24 常规赛？").json()
    assert body["steps"] == []
    assert body["status"] != "complete"


def test_expired_context_and_other_apps_do_not_inherit_scope(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    store = app.state.agent_conversations
    store._clock = lambda: float("inf")
    body = post_with_conversation(app, "那库里的 eFG% 呢？", first["conversation_id"]).json()
    assert body["status"] == "clarification_required"
    assert body["steps"] == []
    assert body["turn_index"] == 1
    other = create_app(db_path=tmp_path / "agent.sqlite", quality_matrix_path=tmp_path / "matrix.json")
    body = post_with_conversation(other, "那库里的 eFG% 呢？", first["conversation_id"]).json()
    assert body["status"] == "clarification_required"
    assert body["steps"] == []


def test_definition_question_does_not_reuse_numeric_scope(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    body = post_with_conversation(app, "eFG% 是什么？", first["conversation_id"]).json()
    assert body["route"] == "knowledge"


def test_comparison_execution_uses_results_and_preserves_plan():
    from scripts.query_context import EntityLink, QueryLinks
    links = QueryLinks(players=[EntityLink(alias="库里", id="201939", display_name="Curry"),
                                EntityLink(alias="詹姆斯", id="2544", display_name="James")],
                       seasons=["2023-24"], phases=["regular_season"])
    plan = plan_question("库里和詹姆斯 2023-24 常规赛 eFG% 比较，解释定义", links=links)
    before = plan.model_dump()
    class Efg:
        name = "player_efg"
        route = "structured"
        async def execute(self, question, arguments):
            return {"status": "complete", "efg_percentage": 60.25 if arguments["entity_id"] == "201939" else 50.0, "inputs": {"field_goals_made": 20, "three_pointers_made": 4, "field_goals_attempted": 40, "sample_count": 2}, "evidence": {"formula": "(FGM + 0.5 * 3PM) / FGA * 100"}}
    response = asyncio.run(execute_agent_plan("q", plan, CapabilityRegistry([
        Efg(), StructuredToolAdapter("efg_compare"), FakeKnowledge(fail=True),
    ])))
    assert response.steps[2].result["absolute_delta"] == 10.25
    assert response.steps[2].evidence.structured.formula == "left_efg_percentage - right_efg_percentage"
    assert response.status == "partial"
    assert "10.25" in response.answer
    assert plan.model_dump() == before


def test_second_structured_failure_retains_first_result_as_partial(tmp_path):
    app = conversation_app(tmp_path)
    # Lillard exists in identity data, but has no shooting samples here.
    body = post(app, "库里和利拉德 2023-24 常规赛 eFG% 比较").json()
    assert body["status"] == "partial"
    assert body["reason_code"] == "structured_step_incomplete"
    assert "58.3333" in body["answer"]
    assert len(body["steps"]) == 2


def test_explicit_entity_typo_does_not_silently_reuse_prior_player(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    body = post_with_conversation(app, "某个不存在的球员 2023-24 常规赛 eFG% 是多少？", first["conversation_id"]).json()
    assert body["status"] == "clarification_required"
    assert body["steps"] == []


@pytest.mark.parametrize(
    ("question", "metric", "value"),
    [
        ("Curry 在 2023-24 regular season 得了多少分？", "points_total", 50),
        ("Stephen Curry 2023-24 常规赛出场场数是多少？", "games_appeared", 2),
    ],
)
def test_english_curry_aliases_ground_to_season_summary(tmp_path, question, metric, value):
    body = post(conversation_app(tmp_path), question).json()
    assert body["status"] == "complete"
    assert body["steps"][0]["capability"] == "season_summary"
    assert body["steps"][0]["result"][metric] == value


def test_relative_previous_season_is_planned_as_season_compare(tmp_path):
    body = post(conversation_app(tmp_path), "库里 2023-24 常规赛总得分相对上赛季增加了多少？").json()
    assert body["status"] == "complete"
    assert body["steps"][0]["capability"] == "season_compare"
    assert body["steps"][0]["result"]["absolute_delta"] == 40


def test_date_window_wording_routes_total_points_to_window_query(tmp_path):
    body = post(conversation_app(tmp_path), "库里在 2023-24 常规赛从 2023-10-01 到 2023-10-03 一共得分多少？").json()
    assert body["status"] == "complete"
    assert body["steps"][0]["capability"] == "window_query"
    assert body["steps"][0]["result"]["values"]["points_total"] == 50


def test_chinese_date_grounding_finds_one_game(tmp_path):
    body = post(conversation_app(tmp_path), "2023 年 10 月 1 日湖人和勇士的比分以及赢家？").json()
    assert body["status"] == "complete"
    assert body["steps"][0]["capability"] == "single_game_score"
    assert body["steps"][0]["result"]["winner_team_id"] == "1610612747"


def test_team_without_phase_requests_phase_clarification(tmp_path):
    body = post(conversation_app(tmp_path), "湖人 2023-24 赛季总得分？").json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "missing_phase"
    assert body["steps"] == []


def test_metric_free_total_points_comparison_does_not_enter_efg_route(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛总得分？").json()
    body = post_with_conversation(app, "那和詹姆斯比呢？", first["conversation_id"]).json()
    assert body["status"] == "clarification_required"
    assert body["route"] == "clarification"
    assert body["steps"] == []


def test_metric_definition_followup_inherits_efg_context(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛有效命中率？").json()
    body = post_with_conversation(app, "这个指标怎么理解？", first["conversation_id"]).json()
    assert body["status"] == "complete"
    assert body["route"] == "knowledge"
    assert body["steps"][0]["capability"] == "knowledge_glossary"


def test_followup_after_metric_switch_does_not_reuse_stale_efg(tmp_path):
    app = conversation_app(tmp_path)
    first = post_with_conversation(app, "库里 2023-24 常规赛 eFG% 是多少？").json()
    conversation_id = first["conversation_id"]
    second = post_with_conversation(app, "库里 2023-24 常规赛总得分？", conversation_id).json()
    assert second["route"] == "structured"
    third = post_with_conversation(app, "那和詹姆斯比呢？", conversation_id).json()
    assert third["status"] in {"clarification_required", "unsupported"}
    assert third["route"] in {"clarification", "unsupported"}
    assert not any(step["capability"] == "player_efg" for step in third["steps"])


def test_clarification_and_unsupported_refuse_before_tools(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    knowledge = FakeKnowledge()
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=knowledge)

    clarification = post(app, "库里 2023-24 的 eFG% 是多少？").json()
    unsupported = post(app, "请执行 SELECT * FROM player_game_stat").json()

    assert clarification["status"] == "clarification_required"
    assert clarification["steps"] == []
    assert unsupported["status"] == "unsupported"
    assert unsupported["steps"] == []
    assert knowledge.calls == []


@pytest.mark.parametrize("question", [
    "请执行 SELECT*FROM player_game_stat",
    "请调用 eval (payload)",
    "请运行 Python 代码计算 eFG",
])
def test_sql_and_code_tokens_are_refused_before_tools(tmp_path, question):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    knowledge = FakeKnowledge()
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=knowledge)

    body = post(app, question).json()
    assert body["status"] == "unsupported"
    assert knowledge.calls == []


def test_phase_grounding_distinguishes_playoffs_and_ambiguity():
    from scripts.query_context import EntityLink, QueryLinks
    grounded = QueryLinks(
        players=[EntityLink(alias="库里", id="201939", display_name="Stephen Curry")],
        seasons=["2023-24"], phases=["playoffs"],
    )
    ambiguous_links = grounded.model_copy(update={"phases": ["regular_season", "playoffs"]})
    playoffs = plan_question("库里 2023-24 季后赛 eFG% 是多少？", links=grounded)
    ambiguous = plan_question("库里 2023-24 常规赛和季后赛 eFG% 是多少？", links=ambiguous_links)

    assert playoffs.steps[0].arguments["competition_phase"] == "playoffs"
    assert ambiguous.route == "clarification"
    assert ambiguous.reason_code == "ambiguous_phase"


def test_comparison_plan_has_bounded_ordered_steps():
    from scripts.query_context import EntityLink, QueryLinks
    links = QueryLinks(
        players=[
            EntityLink(alias="库里", id="201939", display_name="Stephen Curry"),
            EntityLink(alias="詹姆斯", id="2544", display_name="LeBron James"),
        ], seasons=["2023-24"], phases=["regular_season"],
    )
    plan = plan_question("比较库里和詹姆斯 2023-24 常规赛的 eFG%，并解释 eFG% 是什么？", links=links)
    assert plan.route == "comparison"
    assert [step.capability for step in plan.steps] == [
        "player_efg", "player_efg", "efg_compare", "knowledge_glossary",
    ]
    assert [step.arguments["entity_id"] for step in plan.steps[:2]] == ["201939", "2544"]


def test_efg_comparison_is_deterministic():
    adapter = StructuredToolAdapter("efg_compare")
    result = asyncio.run(adapter.execute("q", {"left_efg": 58.3333333333, "right_efg": 50.0}))
    assert result["status"] == "complete"
    assert result["absolute_delta"] == pytest.approx(8.3333333333)
    assert result["higher"] == "left"


@pytest.mark.parametrize("left,right", [(True, 50), (float("nan"), 50), (50, float("inf")), (None, 50)])
def test_comparison_rejects_invalid_numeric_inputs(left, right):
    result = asyncio.run(StructuredToolAdapter("efg_compare").execute("q", {"left_efg": left, "right_efg": right}))
    assert result["status"] == "not_computable"


def test_comparison_registry_rejects_wrong_domains():
    from scripts.query_context import EntityLink, QueryLinks
    links = QueryLinks(players=[EntityLink(alias="库里", id="201939", display_name="Curry"),
                                EntityLink(alias="詹姆斯", id="2544", display_name="James")],
                       seasons=["2023-24"], phases=["regular_season"])
    plan = plan_question("比较库里和詹姆斯 2023-24 常规赛 eFG%", links=links)
    class Wrong:
        name = "player_efg"
        route = "knowledge"
        async def execute(self, question, arguments):
            raise AssertionError("must reject before execution")
    result = asyncio.run(execute_agent_plan("q", plan, CapabilityRegistry([
        Wrong(), StructuredToolAdapter("efg_compare"), FakeKnowledge(),
    ])))
    assert result.status == "unsupported"
    assert result.steps == []


def test_comparison_difference_wording_runs_the_two_player_route(tmp_path):
    app = conversation_app(tmp_path)
    body = post(app, "库里和詹姆斯 2023-24 常规赛 eFG% 相差多少？").json()
    assert body["route"] == "comparison"
    assert body["steps"]


def test_comparison_preserves_requested_knowledge_limitations(tmp_path):
    import sqlite3
    from scripts.relational_baseline import insert_fixture_rows
    app = conversation_app(tmp_path)
    with sqlite3.connect(tmp_path / "agent.sqlite") as connection:
        insert_fixture_rows(connection, games=[], teams=[], players=[{
            "source_row_id": 90, "game_id": "22300001", "person_id": "2544",
            "appearance_status": "played", "minutes_quality": "positive",
            "field_goals_made": 10, "three_pointers_made": 0, "field_goals_attempted": 20,
        }])
    knowledge = FakeKnowledge()
    app = create_app(db_path=tmp_path / "agent.sqlite", quality_matrix_path=tmp_path / "matrix.json", agent_knowledge_adapter=knowledge)
    body = post(app, "库里和詹姆斯 2023-24 常规赛 eFG% 比较，它有什么局限？").json()
    assert body["status"] == "complete"
    assert "局限" in knowledge.calls[0][0]
    assert "常规赛" not in knowledge.calls[0][0]


def test_real_database_comparison_returns_exact_delta_and_sources(tmp_path):
    import sqlite3
    from scripts.relational_baseline import insert_fixture_rows
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    with sqlite3.connect(database) as connection:
        insert_fixture_rows(connection, games=[], teams=[], players=[{
            "source_row_id": 90, "game_id": "22300001", "person_id": "2544",
            "display_name": "LeBron James", "team_id": "1610612747", "opponent_team_id": "1610612744",
            "appearance_status": "played", "minutes_quality": "positive", "points": 20,
            "field_goals_made": 10, "three_pointers_made": 0, "field_goals_attempted": 20,
        }])
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=FakeKnowledge())
    body = post(app, "库里和詹姆斯 2023-24 常规赛 eFG% 比较，并解释定义").json()
    assert body["status"] == "complete"
    assert [step["capability"] for step in body["steps"]] == ["player_efg", "player_efg", "efg_compare", "knowledge_glossary"]
    # Grounding order follows the question, never lexicon alias lengths.
    assert body["steps"][2]["result"]["absolute_delta"] == pytest.approx(58.333333333333336 - 50)
    for item in body["evidence"][:2]:
        assert item["structured"]["source_sql"].startswith("SELECT")
        assert len(item["structured"]["source_parameters"]) == 3
    assert "Stephen Curry" in body["answer"]
    assert "LeBron James" in body["answer"]
    assert body["evidence"][2]["provenance"] == "deterministic calculation"


def test_comparison_with_partial_knowledge_keeps_numeric_answer(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    from scripts.query_context import EntityLink, QueryLinks
    links = QueryLinks(
        players=[
            EntityLink(alias="库里", id="201939", display_name="Stephen Curry"),
            EntityLink(alias="詹姆斯", id="2544", display_name="LeBron James"),
        ], seasons=["2023-24"], phases=["regular_season"],
    )
    plan = plan_question("比较库里和詹姆斯 2023-24 常规赛的 eFG%，并解释 eFG% 是什么？", links=links)
    class FakeEfg:
        name = "player_efg"
        route = "structured"

        async def execute(self, question, arguments):
            return {"status": "complete", "efg_percentage": 58.0 if arguments["entity_id"] == "201939" else 52.0, "inputs": {"field_goals_made": 20, "three_pointers_made": 4, "field_goals_attempted": 40, "sample_count": 2}, "evidence": {"formula": "(FGM + 0.5 * 3PM) / FGA * 100"}}

    registry = CapabilityRegistry([FakeEfg(), StructuredToolAdapter("efg_compare"), FakeKnowledge(partial=True)])
    result = asyncio.run(execute_agent_plan("q", plan, registry))
    assert result.status == "partial"
    assert result.steps[2].capability == "efg_compare"
    assert result.steps[2].status == "complete"
    assert "相差" in result.answer


def test_mixed_registry_rejects_swapped_adapter_routes_before_execution():
    class WrongStructured:
        name = "player_efg"
        route = "knowledge"

        async def execute(self, question, arguments):
            raise AssertionError("must not execute")

    class WrongKnowledge:
        name = "knowledge_glossary"
        route = "structured"

        async def execute(self, question, arguments):
            raise AssertionError("must not execute")

    plan = RoutePlan(
        route="mixed",
        steps=[RouteStep(id="s1", capability="player_efg"), RouteStep(id="s2", capability="knowledge_glossary")],
    )
    result = asyncio.run(execute_agent_plan("q", plan, CapabilityRegistry([WrongStructured(), WrongKnowledge()])))
    assert result.status == "unsupported"
    assert result.reason_code == "route_capability_mismatch"
    assert result.steps == []


def test_invalid_or_unregistered_plan_is_rejected_before_execution():
    called = []

    class Structured:
        name = "player_efg"
        route = "structured"

        async def execute(self, question, arguments):
            called.append(arguments)
            return {"status": "complete"}

    plan = RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="unknown_capability")])
    result = asyncio.run(execute_agent_plan("q", plan, CapabilityRegistry([Structured()])))

    assert result.status == "unsupported"
    assert result.reason_code == "unregistered_capability"
    assert result.steps == []
    assert called == []


def test_existing_answer_contract_stays_available(tmp_path):
    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    app = create_app(
        db_path=database, quality_matrix_path=matrix,
        model=FakeModel({None: '{"action":"call","tool":"player_efg","arguments":{"entity_id":"201939","season_id":"2023-24","competition_phase":"regular_season"}}'}),
    )
    response = post(app, MIXED_FIXTURE_QUESTION)
    assert response.status_code == 200

    async def send_answer():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/answer", json={"question": "计算库里 2023-24 常规赛的 eFG%？"})

    answer = asyncio.run(send_answer())
    assert answer.status_code == 200
    assert answer.json()["status"] == "complete"


def test_agent_trace_has_request_planner_and_step_spans_without_sensitive_text(tmp_path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    database = tmp_path / "agent.sqlite"
    make_ask_database(database)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(
        db_path=database,
        quality_matrix_path=matrix,
        tracer=provider.get_tracer("agent-test"),
        agent_knowledge_adapter=FakeKnowledge(),
    )

    response = post(app, MIXED_FIXTURE_QUESTION)
    spans = exporter.get_finished_spans()
    names = [span.name for span in spans]

    assert response.json()["trace_id"]
    assert names.count("nba.agent.request") == 1
    assert names.count("nba.agent.planner") == 1
    assert names.count("nba.agent.step") == 2
    root = next(span for span in spans if span.name == "nba.agent.request")
    assert all(span.context.trace_id == root.context.trace_id for span in spans)
    planner = next(span for span in spans if span.name == "nba.agent.planner")
    steps = [span for span in spans if span.name == "nba.agent.step"]
    assert planner.parent.span_id == root.context.span_id
    assert all(step.parent.span_id == root.context.span_id for step in steps)
    assert all(span.parent.span_id in {root.context.span_id, *(step.context.span_id for step in steps)}
               for span in spans if span.name not in {root.name, planner.name, *(step.name for step in steps)})
    assert root.attributes["nba.agent.status"] == "complete"
    assert {span.attributes.get("nba.agent.capability") for span in spans if span.name == "nba.agent.step"} == {
        "player_efg", "knowledge_glossary",
    }
    exported_text = " ".join(str(span.attributes) for span in spans)
    assert MIXED_FIXTURE_QUESTION not in exported_text
    assert "api_key" not in exported_text.lower()
