import sqlite3

import pytest

from scripts.relational_baseline import (
    create_schema,
    insert_fixture_rows,
    query_game_score,
    query_player_season,
)


def fixture_db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
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
            {"source_row_id": 1, "game_id": "22300061", "person_id": "p1", "team_id": "a", "opponent_team_id": "b", "appearance_status": "played", "minutes_quality": "positive", "points": 30, "assists": 5, "rebounds_total": 6, "field_goals_made": 10, "three_pointers_made": 4, "field_goals_attempted": 20, "raw_comment": ""},
            {"source_row_id": 2, "game_id": "22300062", "person_id": "p1", "team_id": "b", "opponent_team_id": "c", "appearance_status": "played_inferred", "minutes_quality": "zero", "points": 20, "assists": 4, "rebounds_total": 5, "field_goals_made": 8, "three_pointers_made": 2, "field_goals_attempted": 16, "raw_comment": ""},
            {"source_row_id": 3, "game_id": "22300061", "person_id": "p2", "team_id": "b", "opponent_team_id": "a", "appearance_status": "played", "minutes_quality": "positive", "points": 25, "assists": 7, "rebounds_total": 4, "field_goals_made": 9, "three_pointers_made": 3, "field_goals_attempted": 18, "raw_comment": ""},
            {"source_row_id": 4, "game_id": "22300062", "person_id": "p3", "team_id": "", "opponent_team_id": "c", "appearance_status": "played", "minutes_quality": "positive", "points": 10, "assists": 2, "rebounds_total": 2, "field_goals_made": 4, "three_pointers_made": 1, "field_goals_attempted": 8, "raw_comment": ""},
        ],
        teams=[
            {"source_row_id": 1, "game_id": "22300061", "team_id": "a", "opponent_team_id": "b", "team_score": 119, "opponent_score": 107, "home_away": "home", "win": 1, "assists": 25, "rebounds_total": 40, "field_goals_made": 45, "three_pointers_made": 15, "field_goals_attempted": 90},
            {"source_row_id": 2, "game_id": "22300061", "team_id": "b", "opponent_team_id": "a", "team_score": 107, "opponent_score": 119, "home_away": "away", "win": 0, "assists": 20, "rebounds_total": 38, "field_goals_made": 40, "three_pointers_made": 12, "field_goals_attempted": 88},
            {"source_row_id": 3, "game_id": "22300062", "team_id": "b", "opponent_team_id": "c", "team_score": 110, "opponent_score": 100, "home_away": "home", "win": 1, "assists": 22, "rebounds_total": 39, "field_goals_made": 42, "three_pointers_made": 10, "field_goals_attempted": 85},
            {"source_row_id": 4, "game_id": "22300062", "team_id": "c", "opponent_team_id": "b", "team_score": 100, "opponent_score": 110, "home_away": "away", "win": 0, "assists": 18, "rebounds_total": 35, "field_goals_made": 38, "three_pointers_made": 8, "field_goals_attempted": 82},
        ],
    )
    return connection


def test_game_score_requires_two_complete_team_rows_and_returns_lineage() -> None:
    result = query_game_score(fixture_db(), "22300061")

    assert result["status"] == "ok"
    assert result["home_score"] == 119
    assert result["away_score"] == 107
    assert result["team_row_count"] == 2
    assert result["lineage"]["source_snapshot_id"] == "eoin-v515"


def test_player_season_aggregation_uses_appearance_status_and_recomputes_efg() -> None:
    result = query_player_season(fixture_db(), "p1", "2023-24", "regular_season")

    assert result["status"] == "ok"
    assert result["games_appeared"] == 2
    assert result["points_total"] == 50
    assert result["points_per_appearance"] == 25
    assert result["efg_percentage"] == pytest.approx(58.333333333333336)
    assert result["quality_status"] == "conditional"


def test_player_opponent_filter_rejects_missing_team_context() -> None:
    with pytest.raises(ValueError, match="team context"):
        query_player_season(
            fixture_db(), "p3", "2023-24", "regular_season", opponent_team_id="c"
        )


def test_zero_fga_rejects_efg_without_rejecting_points() -> None:
    connection = fixture_db()
    connection.execute(
        "UPDATE player_game_stat SET field_goals_attempted = 0, field_goals_made = 0, three_pointers_made = 0 WHERE person_id = 'p1'"
    )
    connection.commit()

    result = query_player_season(connection, "p1", "2023-24", "regular_season")

    assert result["points_total"] == 50
    assert result["efg_percentage"] is None
    assert "efg_zero_denominator" in result["limitations"]


def test_opponent_filter_rejects_partial_context_even_when_other_rows_match() -> None:
    connection = fixture_db()
    connection.execute("UPDATE player_game_stat SET opponent_team_id = '' WHERE source_row_id = 2")
    connection.commit()

    with pytest.raises(ValueError, match="team context"):
        query_player_season(connection, "p1", "2023-24", "regular_season", opponent_team_id="b")


def test_duplicate_player_game_rows_do_not_inflate_season_total() -> None:
    connection = fixture_db()
    connection.execute(
        "INSERT INTO player_game_stat SELECT source_snapshot_id, 5, game_id, person_id, team_id, opponent_team_id, appearance_status, minutes_quality, raw_comment, points, assists, rebounds_total, field_goals_made, three_pointers_made, field_goals_attempted, raw_record_json FROM player_game_stat WHERE source_row_id = 1"
    )
    connection.commit()

    with pytest.raises(ValueError, match="duplicate player game"):
        query_player_season(connection, "p1", "2023-24", "regular_season")


def test_score_rejects_conflicting_team_and_game_facts() -> None:
    connection = fixture_db()
    connection.execute("UPDATE game_team_participation SET team_score = 118 WHERE game_id = '22300061' AND home_away = 'home'")
    connection.commit()

    with pytest.raises(ValueError, match="score mismatch"):
        query_game_score(connection, "22300061")
