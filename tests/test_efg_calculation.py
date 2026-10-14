import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from scripts.efg_calculation import EfgInputs, calculate_efg
from scripts.relational_baseline import create_schema, insert_fixture_rows


TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


def test_calculate_efg_uses_cumulative_inputs() -> None:
    result = calculate_efg(EfgInputs(field_goals_made=100, three_pointers_made=40, field_goals_attempted=200))

    assert result.status == "complete"
    assert result.efg_percentage == pytest.approx(60.0)
    assert result.formula == "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100"
    assert result.rounding == "full_precision"


@pytest.mark.parametrize(
    ("inputs", "limitation"),
    [
        (EfgInputs(field_goals_made=None, three_pointers_made=40, field_goals_attempted=200), "efg_inputs_missing"),
        (EfgInputs(field_goals_made=0, three_pointers_made=0, field_goals_attempted=0), "efg_zero_denominator"),
        (EfgInputs(field_goals_made=100, three_pointers_made=101, field_goals_attempted=200), "efg_inputs_inconsistent"),
        (EfgInputs(field_goals_made=201, three_pointers_made=40, field_goals_attempted=200), "efg_inputs_inconsistent"),
        (EfgInputs(field_goals_made=-1, three_pointers_made=0, field_goals_attempted=1), "efg_inputs_invalid"),
    ],
)
def test_calculate_efg_reports_non_computable_input(
    inputs: EfgInputs, limitation: str,
) -> None:
    result = calculate_efg(inputs)

    assert result.status == "not_computable"
    assert result.efg_percentage is None
    assert result.calculation_limitation == limitation


def test_calculate_efg_preserves_conditional_quality() -> None:
    result = calculate_efg(
        EfgInputs(field_goals_made=100, three_pointers_made=40, field_goals_attempted=200),
        quality_status="conditional",
        limitations=("historical_shooting_unverified",),
    )

    assert result.status == "complete"
    assert result.quality_status == "conditional"
    assert result.limitations == ("historical_shooting_unverified",)


def _make_efg_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    insert_fixture_rows(
        connection,
        games=[
            {
                "game_id": "22300061", "game_date": "2023-10-24",
                "season_id": "2023-24", "competition_phase": "regular_season",
                "home_team_id": "a", "away_team_id": "b",
                "home_score": 119, "away_score": 107,
                "season_status": "valid", "phase_status": "valid",
            },
            {
                "game_id": "22300062", "game_date": "2023-10-26",
                "season_id": "2023-24", "competition_phase": "regular_season",
                "home_team_id": "b", "away_team_id": "c",
                "home_score": 110, "away_score": 100,
                "season_status": "valid", "phase_status": "valid",
            },
        ],
        players=[
            {
                "source_row_id": 1, "game_id": "22300061", "person_id": "p1",
                "team_id": "a", "opponent_team_id": "b", "appearance_status": "played",
                "minutes_quality": "positive", "points": 30, "assists": 5,
                "rebounds_total": 6, "field_goals_made": 10,
                "three_pointers_made": 4, "field_goals_attempted": 20,
            },
            {
                "source_row_id": 2, "game_id": "22300062", "person_id": "p1",
                "team_id": "b", "opponent_team_id": "c", "appearance_status": "played",
                "minutes_quality": "positive", "points": 20, "assists": 4,
                "rebounds_total": 5, "field_goals_made": 8,
                "three_pointers_made": 2, "field_goals_attempted": 16,
            },
        ],
        teams=[],
    )
    connection.close()


def _post_efg(app: object, payload: dict):
    async def send_request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/calculate/efg", json=payload)

    return asyncio.run(send_request())


def test_efg_api_returns_inputs_result_and_lineage(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)

    from app.main import create_app

    response = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["efg_percentage"] == pytest.approx(58.333333333333336)
    assert body["inputs"] == {
        "field_goals_made": 18,
        "three_pointers_made": 6,
        "field_goals_attempted": 36,
        "sample_count": 2,
    }
    assert body["quality_status"] == "conditional"
    assert "capability_evidence_conditional" in body["limitations"]
    assert body["evidence"]["source_snapshot"] == "eoinamoore/historical-nba-data-and-player-box-scores@515"
    assert body["evidence"]["formula"] == "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100"


def test_efg_api_rejects_missing_player_observation(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    connection = sqlite3.connect(db_path)
    create_schema(connection)
    connection.close()

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "missing", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert "no player observation" in body["reason"]


def test_efg_api_rejects_missing_field_in_one_source_row(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE player_game_stat SET field_goals_attempted = NULL WHERE source_row_id = 2")
        connection.commit()

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert body["efg_percentage"] is None
    assert body["reason_code"] == "efg_inputs_missing"


def test_efg_api_rejects_invalid_single_game_counts_hidden_by_valid_totals(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE player_game_stat SET three_pointers_made = 11 WHERE source_row_id = 1")
        connection.commit()

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert body["efg_percentage"] is None
    assert body["reason_code"] == "efg_inputs_invalid"


def test_efg_api_rejects_duplicate_player_game_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO player_game_stat SELECT source_snapshot_id, 99, game_id, person_id, team_id, opponent_team_id, appearance_status, minutes_quality, raw_comment, points, assists, rebounds_total, field_goals_made, three_pointers_made, field_goals_attempted, raw_record_json FROM player_game_stat WHERE source_row_id = 1"
        )
        connection.commit()

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert body["efg_percentage"] is None
    assert body["reason_code"] == "duplicate_player_game"


def test_efg_api_reports_zero_denominator_without_numeric_result(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE player_game_stat SET field_goals_made = 0, three_pointers_made = 0, field_goals_attempted = 0"
        )
        connection.commit()

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=TEST_MATRIX),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert body["efg_percentage"] is None
    assert body["calculation_limitation"] == "efg_zero_denominator"
    assert body["inputs"]["field_goals_attempted"] == 0


def test_efg_api_rejects_unsupported_capability_before_calculation(tmp_path: Path) -> None:
    db_path = tmp_path / "efg.sqlite"
    _make_efg_database(db_path)
    matrix = json.loads(TEST_MATRIX.read_text(encoding="utf-8"))
    matrix["periods"]["2023-24|regular_season"]["subjects"]["player"]["metrics"]["efg_percentage"] = "unsupported"
    matrix_path = tmp_path / "unsupported-matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    from app.main import create_app

    body = _post_efg(
        create_app(db_path=db_path, quality_matrix_path=matrix_path),
        {"entity_id": "p1", "season_id": "2023-24", "competition_phase": "regular_season"},
    ).json()

    assert body["status"] == "not_computable"
    assert body["efg_percentage"] is None
    assert "unsupported" in body["reason"]
