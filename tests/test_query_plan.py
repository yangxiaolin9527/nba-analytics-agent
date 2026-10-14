import json
import sqlite3
from pathlib import Path

import pytest

from scripts.query_plan import QueryPlan, parse_query_plan
from scripts.query_plan_compiler import compile_query_plan
from scripts.relational_baseline import create_schema, insert_fixture_rows

TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


def plan(**overrides):
    value = {
        "query_family": "single_game_score", "player_id": None, "team_id": None,
        "game_id": "22300061", "opponent_team_id": None, "season_id": None,
        "competition_phase": None, "requested_metrics": ["home_score", "away_score"],
    }
    value.update(overrides)
    return value


def test_plan_requires_one_strict_json_object() -> None:
    parsed = parse_query_plan(json.dumps(plan()))
    assert isinstance(parsed, QueryPlan)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_query_plan("```json\n" + json.dumps(plan()) + "\n```")
    with pytest.raises(ValueError):
        parse_query_plan(json.dumps({**plan(), "unexpected": True}))
    with pytest.raises(ValueError):
        parse_query_plan(json.dumps({**plan(), "game_id": 22300061}))


@pytest.mark.parametrize(("family", "metrics"), [
    ("player_season_summary", ["points_total", "games_appeared", "points_per_appearance"]),
    ("team_season_summary", ["points_total", "games_played"]),
    ("player_vs_opponent_aggregation", ["points_total", "games_appeared", "points_per_appearance"]),
])
def test_query_families_require_their_scope(family, metrics) -> None:
    values = plan(query_family=family, game_id=None, requested_metrics=metrics,
                  player_id="201939" if family != "team_season_summary" else None,
                  team_id="1610612738" if family == "team_season_summary" else None,
                  opponent_team_id="1610612747" if family == "player_vs_opponent_aggregation" else None,
                  season_id="2023-24", competition_phase="regular_season")
    assert QueryPlan.model_validate(values).query_family.value == family


def test_compiler_uses_reviewed_sql_template() -> None:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    insert_fixture_rows(connection, games=[{
        "game_id": "22300061", "game_date": "2023-10-24", "season_id": "2023-24",
        "competition_phase": "regular_season", "home_team_id": "1610612747",
        "away_team_id": "1610612743", "home_score": 119, "away_score": 107,
        "season_status": "valid", "phase_status": "valid",
    }], players=[], teams=[])
    compiled = compile_query_plan(connection, QueryPlan.model_validate(plan()), matrix_path=TEST_MATRIX)
    assert compiled.render() == "SELECT home_score, away_score FROM game WHERE game_id = '22300061'"
    assert connection.execute(compiled.render()).fetchone() == (119, 107)
    assert compiled.quality_status == "validated"
    assert compiled.limitations == ()


def test_compiler_rejects_missing_entity_or_scope() -> None:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    with pytest.raises(ValueError, match="player"):
        compile_query_plan(connection, QueryPlan.model_validate({
            "query_family": "player_season_summary", "player_id": "999",
            "season_id": "2023-24", "competition_phase": "regular_season",
            "requested_metrics": ["points_total"],
        }))


def test_compiler_checks_injected_quality_matrix_before_entity_probes(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    matrix = json.loads(TEST_MATRIX.read_text(encoding="utf-8"))
    matrix["periods"]["2023-24|regular_season"]["subjects"]["player"]["metrics"]["points"] = "unsupported"
    matrix_path = tmp_path / "unsupported.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    player_plan = QueryPlan.model_validate({
        "query_family": "player_season_summary", "player_id": "99999999",
        "season_id": "2023-24", "competition_phase": "regular_season",
        "requested_metrics": ["points_total"],
    })

    with pytest.raises(ValueError, match="unsupported:points") as error:
        compile_query_plan(connection, player_plan, matrix_path=matrix_path)

    assert getattr(error.value, "code") == "unsupported_data"


def test_compiler_marks_single_game_with_unknown_period_unverified() -> None:
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    insert_fixture_rows(connection, games=[{
        "game_id": "22300061", "competition_phase": "unknown",
        "home_score": 119, "away_score": 107,
    }], players=[], teams=[])

    compiled = compile_query_plan(connection, QueryPlan.model_validate(plan()), matrix_path=TEST_MATRIX)

    assert compiled.quality_status == "unverified"
    assert compiled.limitations == ("capability_not_profiled",)
