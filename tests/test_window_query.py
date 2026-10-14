import asyncio
import json
import sqlite3
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from scripts.relational_baseline import insert_fixture_rows


TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


def make_window_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    insert_fixture_rows(
        connection,
        games=[
            {"game_id": "1", "game_date": "2023-10-01", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "a", "away_team_id": "b", "home_score": 100, "away_score": 90, "season_status": "valid", "phase_status": "valid"},
            {"game_id": "2", "game_date": "2023-10-03", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "a", "away_team_id": "c", "home_score": 110, "away_score": 95, "season_status": "valid", "phase_status": "valid"},
            {"game_id": "3", "game_date": "2023-10-05", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "a", "away_team_id": "d", "home_score": 120, "away_score": 100, "season_status": "valid", "phase_status": "valid"},
        ],
        players=[
            {"source_row_id": 1, "game_id": "1", "person_id": "p1", "team_id": "a", "opponent_team_id": "b", "appearance_status": "played", "minutes_quality": "positive", "points": 10},
            {"source_row_id": 2, "game_id": "2", "person_id": "p1", "team_id": "a", "opponent_team_id": "c", "appearance_status": "played_inferred", "minutes_quality": "zero", "points": 20},
            {"source_row_id": 3, "game_id": "3", "person_id": "p1", "team_id": "a", "opponent_team_id": "d", "appearance_status": "not_played", "minutes_quality": "zero", "points": 0},
        ],
        teams=[
            {"source_row_id": 1, "game_id": "1", "team_id": "a", "opponent_team_id": "b", "team_score": 100, "opponent_score": 90, "home_away": "home", "win": 1},
            {"source_row_id": 2, "game_id": "2", "team_id": "a", "opponent_team_id": "c", "team_score": 110, "opponent_score": 95, "home_away": "home", "win": 1},
            {"source_row_id": 3, "game_id": "3", "team_id": "a", "opponent_team_id": "d", "team_score": 120, "opponent_score": 100, "home_away": "home", "win": 1},
        ],
    )
    connection.close()


def post_window(app: object, payload: dict):
    async def send_request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/query/window", json=payload)
    return asyncio.run(send_request())


def test_window_query_selects_latest_n_and_discloses_inferred_appearance(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    response = post_window(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"subject": "player", "entity_id": "p1", "season_id": "2023-24",
         "competition_phase": "regular_season", "metrics": ["points_total", "games_appeared", "points_per_appearance"], "last_n": 2},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["values"] == {"points_total": 30, "games_appeared": 2, "points_per_appearance": 15.0}
    assert body["actual_sample_count"] == 2
    assert body["selected_game_ids"] == ["1", "2"]
    assert body["actual_date_from"] == "2023-10-01"
    assert body["actual_date_to"] == "2023-10-03"
    assert "inferred_appearance_included" in body["limitations"]


def test_window_query_returns_partial_without_backfill_when_n_exceeds_available(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    body = post_window(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"subject": "team", "entity_id": "a", "season_id": "2023-24",
         "competition_phase": "regular_season", "metrics": ["points_total", "games_played"], "last_n": 5},
    ).json()
    assert body["status"] == "partial"
    assert body["actual_sample_count"] == 3
    assert body["selected_game_ids"] == ["1", "2", "3"]
    assert body["values"] == {"points_total": 330, "games_played": 3}


def test_window_query_rejects_invalid_last_n(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    response = post_window(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"subject": "player", "entity_id": "p1", "season_id": "2023-24",
         "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 0},
    )
    assert response.status_code == 422


def test_window_query_applies_inclusive_dates_and_opponent_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    body = post_window(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"subject": "team", "entity_id": "a", "season_id": "2023-24",
         "competition_phase": "regular_season", "metrics": ["points_total"],
         "opponent_team_id": "c", "date_from": "2023-10-03", "date_to": "2023-10-03"},
    ).json()
    assert body["status"] == "complete"
    assert body["values"] == {"points_total": 110}
    assert body["selected_game_ids"] == ["2"]
    assert "t.opponent_team_id = ?" in body["source_sql"]


def test_player_opponent_window_uses_valid_participation_and_exact_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    body = post_window(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"subject": "player", "entity_id": "p1", "season_id": "2023-24",
         "competition_phase": "regular_season", "metrics": ["points_total"],
         "opponent_team_id": "c", "last_n": 2},
    ).json()
    assert body["status"] == "partial"
    assert body["values"] == {"points_total": 20}
    assert body["selected_game_ids"] == ["2"]
    assert "p.opponent_team_id = ?" in body["source_sql"]


def test_count_only_window_does_not_require_points_capability(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    matrix = json.loads(TEST_MATRIX.read_text())
    matrix["periods"]["2023-24|regular_season"]["subjects"]["player"]["metrics"]["points"] = "unsupported"
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    from app.main import create_app

    body = post_window(create_app(db_path=db_path, quality_matrix_path=matrix_path), {
        "subject": "player", "entity_id": "p1", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["games_appeared"],
    }).json()
    assert body["status"] == "complete"
    assert body["values"] == {"games_appeared": 2}
    assert body["quality_status"] == "conditional"  # one appearance is inferred
    assert "unsupported:points" not in body["limitations"]


def test_team_opponent_window_discloses_unprofiled_filter(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    body = post_window(create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX), {
        "subject": "team", "entity_id": "a", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["games_played"],
        "opponent_team_id": "c",
    }).json()
    assert body["values"] == {"games_played": 1}
    assert body["quality_status"] == "unverified"
    assert "capability_not_profiled" in body["limitations"]


def test_window_evidence_replays_filter_and_selection(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    request = {
        "subject": "team", "entity_id": "a", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
        "date_from": "2023-10-01", "date_to": "2023-10-03", "last_n": 1,
    }
    body = post_window(create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX), request).json()
    assert body["values"] == {"points_total": 110}
    assert body["available_sample_count"] == 2
    assert body["selected_game_ids"] == ["2"]
    assert body["source_parameters"] == ["a", "2023-24", "regular_season", "2023-10-01", "2023-10-03"]
    assert body["evidence"]["requested_scope"] == request
    assert body["evidence"]["selection_rule"] == "filter_then_date_game_order_then_last_n"


def test_window_no_observation_preserves_scope_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    from app.main import create_app

    body = post_window(create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX), {
        "subject": "player", "entity_id": "p1", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
        "date_from": "2023-11-01",
    }).json()
    assert body["status"] == "not_computable"
    assert body["reason_code"] == "no_observation"
    assert body["values"] == {}
    assert body["evidence"]["requested_scope"]["date_from"] == "2023-11-01"
    assert body["evidence"]["source_snapshot"].endswith("@515")


def test_unrelated_opponent_invalid_game_does_not_reject_requested_window(tmp_path: Path) -> None:
    db_path = tmp_path / "window.sqlite"
    make_window_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE game SET phase_status = 'conflict' WHERE game_id = '1'")
    from app.main import create_app

    body = post_window(create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX), {
        "subject": "team", "entity_id": "a", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
        "opponent_team_id": "c",
    }).json()
    assert body["status"] == "complete"
    assert body["selected_game_ids"] == ["2"]
    assert body["values"] == {"points_total": 110}
