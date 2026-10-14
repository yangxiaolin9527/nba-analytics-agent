import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from app.agent_routes import plan_question
from app.main import create_app
from scripts.query_context import EntityLink, QueryLinks
from tests.test_ask import make_ask_database, make_matrix


def _links(*, player=False, team=False, phases=None, seasons=None, opponent=False):
    teams = []
    if team or opponent:
        teams.append(EntityLink(alias="湖人", id="1610612747", display_name="Los Angeles Lakers"))
    if opponent:
        teams.append(EntityLink(alias="勇士", id="1610612744", display_name="Golden State Warriors"))
    return QueryLinks(
        players=[EntityLink(alias="库里", id="201939", display_name="Stephen Curry")] if player else [],
        teams=teams,
        seasons=list(seasons or ["2023-24"]),
        phases=list(phases or ["regular_season"]),
    )


def _post(app, question):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/agent/answer", json={"question": question})
    return asyncio.run(send())


def test_m43_planner_preserves_average_metric_and_opponent_filter():
    plan = plan_question(
        "库里 2023-24 常规赛最近两次出场对湖人的场均得分？",
        links=_links(player=True, opponent=True),
    )
    step = plan.steps[0]
    assert step.capability == "window_query"
    assert step.arguments["metrics"] == ["points_per_appearance"]
    assert step.arguments["opponent_team_id"] == "1610612747"
    assert step.arguments["last_n"] == 2


def test_m43_planner_routes_team_comparison_with_grounded_team():
    plan = plan_question(
        "湖人 2023-24 常规赛总得分比上一季增加多少？",
        links=_links(team=True),
    )
    assert plan.route == "structured"
    assert plan.steps[0].capability == "season_compare"
    assert plan.steps[0].arguments["subject"] == "team"
    assert plan.steps[0].arguments["entity_id"] == "1610612747"


def test_m43_planner_routes_definition_only_efg_to_knowledge():
    plan = plan_question("eFG% 是什么？")
    assert plan.route == "knowledge"
    assert [step.capability for step in plan.steps] == ["knowledge_glossary"]


def test_m43_http_preserves_ambiguous_phase_clarification(tmp_path: Path):
    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    response = _post(create_app(db_path=database, quality_matrix_path=matrix), "库里 2023-24 常规赛和季后赛总得分？")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "clarification_required"
    assert body["route"] == "clarification"
    assert body["reason_code"] == "ambiguous_phase"
    assert body["steps"] == []


def test_m43_http_definition_only_efg_does_not_require_statistical_scope(tmp_path: Path):
    from tests.test_agent_routes import FakeKnowledge

    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    response = _post(
        create_app(db_path=database, quality_matrix_path=matrix, agent_knowledge_adapter=FakeKnowledge()),
        "eFG% 是什么？",
    )
    body = response.json()
    assert body["status"] == "complete"
    assert body["route"] == "knowledge"
    assert body["steps"][0]["capability"] == "knowledge_glossary"


def test_m43_http_maps_grounding_clarification_and_keeps_trace(tmp_path: Path):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    database = tmp_path / "agent.sqlite"
    matrix = tmp_path / "matrix.json"
    make_ask_database(database)
    make_matrix(matrix)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("m4.3-test")
    body = _post(
        create_app(db_path=database, quality_matrix_path=matrix, tracer=tracer),
        "库里 2023-24 最近两次出场的总得分？",
    ).json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "missing_phase"
    assert body["trace_id"]
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["nba.agent.planner", "nba.agent.request"]
    assert spans[0].parent.span_id == spans[1].context.span_id

