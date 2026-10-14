import asyncio
import json
import re
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.generate_demo_fixtures import generate_demo_catalog
from scripts.relational_baseline import insert_fixture_rows
from tests.test_ask import make_ask_database, make_matrix


def test_home_redirects_to_demo_without_changing_api_routes():
    app = create_app(knowledge_search=object())

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            home = await client.get("/")
            page = await client.get("/demo")
        return home, page

    home, page = asyncio.run(request())
    assert home.status_code == 307
    assert home.headers["location"] == "/demo"
    assert page.status_code == 200
    assert '<base href="/demo/static/">' in page.text
    local_page = Path("static/demo.html").read_text(encoding="utf-8")
    assert re.search(r'href="\./demo\.css(?:\?[^"\s]+)?"', local_page)
    assert re.search(r'src="\./demo\.js(?:\?[^"\s]+)?"', local_page)


def test_demo_supporting_assets_are_served_without_exposing_repository_files():
    app = create_app(knowledge_search=object())

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assets = [await client.get(f"/demo/static/{name}") for name in ("demo-data.js", "lucide.min.js", "demo-state.js")]
            private = await client.get("/demo/static/.env.local")
            source = await client.get("/demo/static/agent_routes.py")
        return assets, private, source

    assets, private, source = asyncio.run(request())
    assert all(asset.status_code == 200 for asset in assets)
    assert "NBA_DEMO_DATA" in assets[0].text
    assert private.status_code == 404
    assert source.status_code == 404


def test_demo_page_and_case_catalog_are_available(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            page = await client.get("/demo")
            cases = await client.get("/demo/cases")
            status = await client.get("/demo/status")
            stylesheet = await client.get("/demo/static/demo.css")
            script = await client.get("/demo/static/demo.js")
        return page, cases, status, stylesheet, script

    page, cases, status, stylesheet, script = asyncio.run(request())
    assert page.status_code == 200
    assert "NBA Analytics Agent" in page.text
    assert "统一 Agent" in page.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
    assert ".sidebar{position:fixed" in stylesheet.text
    assert ".app-content{margin-left:var(--sidebar)" in stylesheet.text
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "candidate.case_id === $('agent-case-select').value" in script.text
    assert "function failureDiagnostic" in script.text
    assert "generation_timeout" in script.text
    assert "candidate.code !== 'no_usable_result'" in script.text
    assert "function executionChainHtml" in script.text
    assert "function renderTranscript" in script.text
    assert "function removeHistory" in script.text
    assert "sessionHistory" in script.text
    assert re.search(r'src="\./demo-state\.js(?:\?[^"\s]+)?"', page.text)
    assert 'id="clear-history"' in page.text
    assert 'id="agent-transcript"' in page.text
    assert ".transcript-turn" in stylesheet.text
    assert status.status_code == 200
    assert status.json()["modes"]["replay"] is True
    assert status.json()["modes"]["structured"] is True
    assert cases.status_code == 200
    body = cases.json()
    assert body["version"] == "2.0.0"
    assert body["fixture_mode"] == "answer_flow_replay"
    assert body["provenance"]["generator"] == "scripts/generate_demo_fixtures.py"
    assert {case["tool"] for case in body["cases"]} == {
        "season_summary", "single_game_score", "window_query", "season_compare", "player_efg",
    }


def test_demo_fixture_returns_only_registered_case(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            valid = await client.post("/demo/fixture", json={"case_id": "M34-FIX-001"})
            invalid = await client.post("/demo/fixture", json={"case_id": "unknown"})
        return valid, invalid

    valid, invalid = asyncio.run(request())
    assert valid.status_code == 200
    assert valid.json()["mode"] == "fixture"
    assert valid.json()["response"]["status"] == "complete"
    assert "9679" in valid.json()["response"]["answer"]
    assert invalid.status_code == 404


def test_demo_fixture_does_not_change_answer_contract(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/demo/fixture", json={"case_id": "M34-FIX-001"})

    response = asyncio.run(request())
    assert set(response.json()) == {"mode", "case_id", "response"}


def test_demo_agent_catalog_and_replay_are_explicitly_provenanced(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            catalog = await client.get("/demo/agent/cases")
            replay = await client.post("/demo/agent/replay", json={"case_id": "M43-001"})
            unknown = await client.post("/demo/agent/replay", json={"case_id": "unknown"})
        return catalog, replay, unknown

    catalog, replay, unknown = asyncio.run(request())
    assert catalog.status_code == 200
    assert {item["source_type"] for item in catalog.json()["cases"]} >= {"fixture_contract"}
    assert replay.status_code == 200
    assert replay.json()["mode"] == "replay"
    assert replay.json()["provenance"]["source_type"] == "fixture_contract"
    assert replay.json()["response"]["status"] == "complete"
    assert unknown.status_code == 404


def test_demo_agent_replay_uses_case_id_for_duplicate_questions(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            catalog = await client.get("/demo/agent/cases")
            complete = await client.post("/demo/agent/replay", json={"case_id": "M43-011"})
            partial = await client.post("/demo/agent/replay", json={"case_id": "M43-019"})
        return catalog, complete, partial

    catalog, complete, partial = asyncio.run(request())
    duplicate_ids = [
        item["case_id"] for item in catalog.json()["cases"]
        if item["question"] == "库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么？"
    ]
    assert duplicate_ids == ["M43-011", "M43-019"]
    assert complete.json()["response"]["status"] == "complete"
    assert partial.json()["response"]["status"] == "partial"
    assert partial.json()["response"]["reason_code"] == "knowledge_step_partial"


def test_demo_status_does_not_claim_mixed_ready_without_model_and_embedding(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    index_path = tmp_path / "knowledge-index"
    index_path.mkdir()
    (index_path / "nodes.json").write_text("{}")
    (index_path / "dense.json").write_text("{}")
    from scripts.provider_config import ProviderConfig
    config = ProviderConfig(knowledge_index_path=str(index_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    app = create_app(db_path=db_path, quality_matrix_path=matrix_path, knowledge_search=object(), provider_config=config)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/demo/status")

    body = asyncio.run(request()).json()
    assert body["modes"]["structured"] is True
    assert body["modes"]["mixed"] is False
    assert body["resources"]["model"]["configured"] is False
    assert body["resources"]["embedding"]["configured"] is False


def test_demo_status_exposes_database_readiness_without_leaking_path(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    missing = tmp_path / "missing.sqlite"
    app = create_app(db_path=missing, quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/demo/status")

    response = asyncio.run(request())
    database = response.json()["resources"]["database"]
    assert database == {"configured": True, "exists": False, "readable": False, "available": False}
    assert str(missing) not in response.text


def test_health_and_readiness_endpoints_are_explicit(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    app = create_app(db_path=tmp_path / "missing.sqlite", quality_matrix_path=matrix_path)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/healthz"), await client.get("/readyz")

    health, ready = asyncio.run(request())
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"


def test_app_factory_keeps_replay_available_when_local_qdrant_is_locked(monkeypatch) -> None:
    import scripts.knowledge_search

    def locked_search(*args, **kwargs):
        raise RuntimeError("Storage folder /data/qdrant is already accessed by another instance of Qdrant client")

    monkeypatch.setattr(scripts.knowledge_search, "configured_search", locked_search)
    app = create_app()
    assert app is not None


def test_generate_demo_catalog_replays_registered_answer_flow(tmp_path: Path) -> None:
    db_path = tmp_path / "nba.sqlite"
    make_ask_database(db_path)
    matrix_path = tmp_path / "matrix.json"
    make_matrix(matrix_path)
    output_path = tmp_path / "demo-cases.json"
    import sqlite3
    with sqlite3.connect(db_path) as connection:
        insert_fixture_rows(
            connection,
            games=[{"game_id": "22300061", "game_date": "2023-10-24", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "1610612743", "away_team_id": "1610612747", "home_score": 119, "away_score": 107, "season_status": "valid", "phase_status": "valid"}],
            players=[],
            teams=[
                {"source_row_id": 99, "game_id": "22300061", "team_id": "1610612743", "team_name": "Denver Nuggets", "opponent_team_id": "1610612747", "team_score": 119, "opponent_score": 107, "home_away": "home", "win": 1},
                {"source_row_id": 100, "game_id": "22300061", "team_id": "1610612747", "team_name": "Los Angeles Lakers", "opponent_team_id": "1610612743", "team_score": 107, "opponent_score": 119, "home_away": "away", "win": 0},
            ],
        )

    document = asyncio.run(generate_demo_catalog(
        db_path=db_path, matrix_path=matrix_path, output_path=output_path,
    ))

    assert document["version"] == "2.0.0"
    assert document["fixture_mode"] == "answer_flow_replay"
    assert len(document["cases"]) == 5
    assert {case["tool"] for case in document["cases"]} == {
        "season_summary", "single_game_score", "window_query", "season_compare", "player_efg",
    }
    for case in document["cases"]:
        response = case["response"]
        assert response["status"] == "complete"
        assert response["evidence"]["snapshot_id"]
        assert response["query_metadata"]["catalog_version"]
        assert case["source_case_id"]
    assert output_path.is_file()
