import asyncio
from pathlib import Path

from app.agent_routes import (
    CapabilityRegistry,
    RoutePlan,
    RouteStep,
    StructuredToolAdapter,
    plan_question,
)
from scripts.query_context import EntityLink, QueryLinks


def links(*, player=False, team=False, game=False, player_name="库里", player_id="201939", season="2023-24"):
    return QueryLinks(
        players=[EntityLink(alias=player_name, id=player_id, display_name="Stephen Curry")] if player else [],
        teams=[EntityLink(alias="湖人", id="1610612747", display_name="Los Angeles Lakers")] if team else [],
        seasons=[season],
        phases=["regular_season"],
        games=[{"game_id": "22300001", "game_date": "2023-10-01", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "1610612747", "away_team_id": "1610612744"}] if game else [],
    )


def test_rules_planner_routes_existing_structured_capabilities():
    assert plan_question("库里 2023-24 常规赛总得分？", links=links(player=True)).steps[0].capability == "season_summary"
    assert plan_question("库里 2023-24 常规赛最近两次出场的总得分？", links=links(player=True)).steps[0].capability == "window_query"
    assert plan_question("库里 2023-24 常规赛最近两次出场的总得分？", links=links(player=True)).steps[0].arguments["last_n"] == 2
    assert plan_question("库里 2023-24 常规赛比上一季总得分增加多少？", links=links(player=True)).steps[0].capability == "season_compare"
    assert plan_question("2023-10-01 湖人对勇士谁赢了？", links=links(team=True, game=True)).steps[0].capability == "single_game_score"
    summary = plan_question("湖人 2023-24 常规赛总得分和比赛场数？", links=links(team=True))
    assert summary.steps[0].arguments["metrics"] == ["points_total", "games_played"]


def test_rules_planner_preserves_mixed_boundary():
    plan = plan_question("库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么？", links=links(player=True))
    assert plan.route == "mixed"
    assert [step.capability for step in plan.steps] == ["player_efg", "knowledge_glossary"]


def test_efg_planner_uses_grounded_non_curry_player_and_season():
    plan = plan_question("戴维斯 2022-23 常规赛 eFG% 是多少？", links=links(player=True, player_name="戴维斯", player_id="2544", season="2022-23"))
    assert plan.route == "structured"
    assert plan.steps[0].arguments == {"entity_id": "2544", "season_id": "2022-23", "competition_phase": "regular_season"}


def test_efg_planner_requires_entity_and_season_without_grounding():
    plan = plan_question("某位球员的 eFG% 是多少？")
    assert plan.route == "clarification"
    assert plan.steps == []
    assert plan.reason_code == "missing_phase"


def test_agent_preflight_preserves_grounding_clarification_code(tmp_path):
    from app.main import create_app
    from httpx import ASGITransport, AsyncClient
    from tests.test_ask import make_ask_database, make_matrix
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    app = create_app(db_path=database, quality_matrix_path=matrix)
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/agent/answer", json={"question": "库里 2023-24 最近两次出场的总得分？"})
    body = asyncio.run(send()).json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "missing_phase"


def test_structured_registry_accepts_all_five_capabilities():
    adapters = [StructuredToolAdapter(name) for name in (
        "season_summary", "single_game_score", "window_query", "season_compare", "player_efg",
    )]
    registry = CapabilityRegistry(adapters)
    for adapter in adapters:
        plan = RoutePlan(route="structured", steps=[RouteStep(id="s1", capability=adapter.name)])
        assert registry.validate(plan) is None


def test_structured_adapter_rejects_missing_database():
    adapter = StructuredToolAdapter("season_summary", database=Path("/does/not/exist"))
    result = asyncio.run(adapter.execute("q", {}))
    assert result["status"] == "failed"
    assert result["reason_code"] == "database_unavailable"


def test_structured_adapter_adds_rendering_evidence_for_summary_and_game(tmp_path):
    from tests.test_ask import make_ask_database, make_matrix
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    summary = asyncio.run(StructuredToolAdapter("season_summary", database=database, quality_matrix_path=matrix).execute(
        "q", {"subject": "player", "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season", "metrics": ["points_total"]}
    ))
    game = asyncio.run(StructuredToolAdapter("single_game_score", database=database, quality_matrix_path=matrix).execute(
        "q", {"game_id": "22300001"}
    ))
    assert summary["evidence"]["entity"]["name"] == "Stephen Curry"
    assert game["evidence"]["home_team"]["name"] == "Los Angeles Lakers"


def test_structured_execution_uses_structured_evidence_for_non_efg_capabilities(tmp_path):
    from tests.test_ask import make_ask_database, make_matrix
    from app.agent_routes import execute_agent_plan
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    adapter = StructuredToolAdapter("season_summary", database=database, quality_matrix_path=matrix)
    plan = RoutePlan(route="structured", steps=[RouteStep(
        id="s1", capability="season_summary", arguments={
            "subject": "player", "entity_id": "201939", "season_id": "2023-24",
            "competition_phase": "regular_season", "metrics": ["points_total"],
        })])
    response = asyncio.run(execute_agent_plan("库里 2023-24 常规赛总得分？", plan, CapabilityRegistry([adapter])))
    assert response.status == "complete"
    assert response.evidence[0].structured is not None
    assert response.evidence[0].knowledge is None
