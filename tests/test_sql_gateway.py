import sqlite3

import pytest

from scripts.relational_baseline import create_schema, insert_fixture_rows
from scripts.sql_gateway import GatewayStatus, SqlGateway


def fixture_db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    insert_fixture_rows(
        connection,
        games=[{
            "game_id": "g1", "game_date": "2023-10-24", "season_id": "2023-24",
            "competition_phase": "regular_season", "home_team_id": "a", "away_team_id": "b",
            "home_score": 100, "away_score": 90, "season_status": "valid", "phase_status": "valid",
        }],
        players=[],
        teams=[],
    )
    connection.execute("CREATE TABLE private_notes (secret TEXT)")
    return connection


def test_gateway_executes_allowed_select_and_returns_rows() -> None:
    result = SqlGateway(fixture_db()).execute("SELECT game_id, home_score FROM game ORDER BY game_id")

    assert result.status is GatewayStatus.OK
    assert result.columns == ["game_id", "home_score"]
    assert result.rows == [("g1", 100)]


@pytest.mark.parametrize("sql", [
    "UPDATE game SET home_score = 1",
    "DROP TABLE game",
    "SELECT * FROM source_record",
    "SELECT missing FROM game",
    "SELECT game_id FROM game; SELECT game_id FROM game",
    "SELECT secret FROM private_notes",
    "PRAGMA table_info(game)",
    "ATTACH DATABASE ':memory:' AS other",
])
def test_gateway_rejects_unsafe_or_unknown_sql_before_execution(sql: str) -> None:
    result = SqlGateway(fixture_db()).execute(sql)

    assert result.status is GatewayStatus.REJECTED
    assert result.rows == []
    assert result.reason


def test_gateway_accepts_cte_and_enforces_row_limit() -> None:
    gateway = SqlGateway(fixture_db(), max_rows=0)
    result = gateway.execute("WITH games AS (SELECT game_id FROM game) SELECT game_id FROM games")

    assert result.status is GatewayStatus.TRUNCATED
    assert result.rows == []


def test_gateway_accepts_column_projected_by_subquery() -> None:
    result = SqlGateway(fixture_db()).execute(
        "SELECT home_points FROM (SELECT home_score AS home_points FROM game) AS scores"
    )

    assert result.status is GatewayStatus.OK
    assert result.rows == [(100,)]


def test_gateway_rejects_unknown_column_despite_unrelated_cte_alias() -> None:
    result = SqlGateway(fixture_db()).execute(
        "WITH unused AS (SELECT home_score AS missing FROM game) SELECT missing FROM game"
    )

    assert result.status is GatewayStatus.REJECTED
    assert result.rows == []


@pytest.mark.parametrize("sql", [
    "SELECT game_id FROM game WHERE competition_phase = 'Regular Season'",
    "SELECT game_id FROM game WHERE LOWER(competition_phase) = 'regular season'",
    "SELECT game_id FROM game WHERE competition_phase LIKE '%season%'",
    "SELECT points FROM player_game_stat WHERE appearance_status = 'Did Not Play'",
    "SELECT game_id FROM game_team_participation WHERE home_away = 'Home'",
])
def test_gateway_rejects_uncanonical_filter_value_before_execution(sql: str) -> None:
    result = SqlGateway(fixture_db()).execute(sql)

    assert result.status is GatewayStatus.REJECTED
    assert result.rows == []
    assert "domain" in result.reason.lower()


def test_gateway_accepts_canonical_filter_value() -> None:
    result = SqlGateway(fixture_db()).execute(
        "SELECT game_id FROM game WHERE competition_phase = 'regular_season'"
    )

    assert result.status is GatewayStatus.OK
    assert result.rows == [("g1",)]


@pytest.mark.parametrize("sql", [
    "SELECT game_id FROM (SELECT game_id, competition_phase AS phase FROM game) AS q WHERE phase = 'Regular Season'",
    "WITH q AS (SELECT game_id, competition_phase AS phase FROM game) SELECT game_id FROM q WHERE phase = 'Regular Season'",
])
def test_gateway_rejects_domain_filter_hidden_behind_alias(sql: str) -> None:
    result = SqlGateway(fixture_db()).execute(sql)

    assert result.status is GatewayStatus.REJECTED
    assert result.rows == []
    assert "domain" in result.reason.lower()
