import asyncio
import json
import math
import sqlite3
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import app.ask_routes as ask_routes
from app.main import create_app
from scripts.model_adapter import FakeModel, ModelRequest, ModelResponse
from scripts.relational_baseline import insert_fixture_rows


TEST_MATRIX = Path(__file__).parent / "fixtures/capability_matrix_v1.json"


def make_ask_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        games = [
            {"game_id": "22200001", "game_date": "2022-10-01", "season_id": "2022-23", "competition_phase": "regular_season", "home_team_id": "1610612744", "away_team_id": "1610612747", "home_score": 95, "away_score": 99, "season_status": "valid", "phase_status": "valid"},
            {"game_id": "22300001", "game_date": "2023-10-01", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "1610612747", "away_team_id": "1610612744", "home_score": 100, "away_score": 90, "season_status": "valid", "phase_status": "valid"},
            {"game_id": "22300002", "game_date": "2023-10-03", "season_id": "2023-24", "competition_phase": "regular_season", "home_team_id": "1610612744", "away_team_id": "1610612747", "home_score": 110, "away_score": 105, "season_status": "valid", "phase_status": "valid"},
        ]
        players = [
            {"source_row_id": 1, "game_id": "22200001", "person_id": "201939", "display_name": "Stephen Curry", "team_id": "1610612744", "opponent_team_id": "1610612747", "appearance_status": "played", "minutes_quality": "positive", "points": 10, "field_goals_made": 4, "three_pointers_made": 1, "field_goals_attempted": 10},
            {"source_row_id": 2, "game_id": "22300001", "person_id": "201939", "display_name": "Stephen Curry", "team_id": "1610612744", "opponent_team_id": "1610612747", "appearance_status": "played", "minutes_quality": "positive", "points": 20, "field_goals_made": 8, "three_pointers_made": 2, "field_goals_attempted": 16},
            {"source_row_id": 3, "game_id": "22300002", "person_id": "201939", "display_name": "Stephen Curry", "team_id": "1610612744", "opponent_team_id": "1610612747", "appearance_status": "played", "minutes_quality": "positive", "points": 30, "field_goals_made": 10, "three_pointers_made": 4, "field_goals_attempted": 20},
        ]
        teams = []
        for index, game in enumerate(games):
            home, away = game["home_team_id"], game["away_team_id"]
            for offset, (team_id, opponent_id, score, opposite_score, home_away) in enumerate((
                (home, away, game["home_score"], game["away_score"], "home"),
                (away, home, game["away_score"], game["home_score"], "away"),
            )):
                teams.append({"source_row_id": index * 2 + offset + 1, "game_id": game["game_id"],
                              "team_id": team_id, "team_name": "Los Angeles Lakers" if team_id == "1610612747" else "Golden State Warriors",
                              "opponent_team_id": opponent_id, "team_score": score, "opponent_score": opposite_score,
                              "home_away": home_away, "win": int(score > opposite_score)})
        insert_fixture_rows(connection, games=games, players=players, teams=teams)
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("203081", "Damian Lillard", "fixture"),
        )
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("2544", "LeBron James", "fixture"),
        )


def make_matrix(path: Path) -> None:
    matrix = json.loads(TEST_MATRIX.read_text())
    matrix["periods"]["2022-23|regular_season"] = matrix["periods"]["2023-24|regular_season"]
    path.write_text(json.dumps(matrix))


def post_ask(app, question: str):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/ask", json={"question": question})
    return asyncio.run(send())


def ask_app(tmp_path: Path, plan: dict, **kwargs):
    db = tmp_path / "ask.sqlite"
    make_ask_database(db)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    model = FakeModel({None: plan if isinstance(plan, str) else json.dumps(plan)})
    return create_app(db_path=db, quality_matrix_path=matrix, model=model, **kwargs)


def test_ask_routes_recent_games_to_bounded_window_tool(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_per_appearance"], "last_n": 2,
    }})
    response = post_ask(app, "库里在 2023-24 常规赛最近两次出场的场均得分？")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["tool"] == "window_query"
    assert body["tool_calls_used"] == 1
    assert body["tool_result"]["values"] == {"points_per_appearance": 25.0}
    assert body["tool_result"]["selected_game_ids"] == ["22300001", "22300002"]
    assert body["evidence"]["source_version"] == "515"


def test_ask_routes_previous_season_comparison_to_deterministic_tool(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_compare", "arguments": {
        "subject": "player", "entity_id": "201939", "metric": "points_total",
        "baseline_season": "2022-23", "comparison_season": "2023-24",
        "competition_phase": "regular_season",
    }})
    body = post_ask(app, "库里 2023-24 常规赛总得分比上一季增加多少，增幅多少？").json()
    assert body["status"] == "complete"
    assert body["tool_result"]["absolute_delta"] == 40
    assert body["tool_result"]["relative_change_pct"] == 400.0
    assert body["tool_result"]["baseline"]["season_id"] == "2022-23"


def test_ask_routes_efg_to_calculator_with_inputs_and_formula(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {
        "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
    }})
    body = post_ask(app, "计算库里 2023-24 常规赛的 eFG%？").json()
    assert body["status"] == "complete"
    assert body["tool_result"]["inputs"] == {
        "field_goals_made": 18, "three_pointers_made": 6,
        "field_goals_attempted": 36, "sample_count": 2,
    }
    assert math.isclose(body["tool_result"]["efg_percentage"], 58.333333333333336)
    assert "field_goals_made" in body["tool_result"]["evidence"]["formula"]


def test_ask_efg_question_may_name_formula_inputs(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {
        "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
    }})
    body = post_ask(app, "用库里 2023-24 常规赛累计命中数、三分命中数和出手数计算 eFG%。").json()
    assert body["status"] == "complete"
    assert body["tool"] == "player_efg"
    assert body["tool_result"]["inputs"]["three_pointers_made"] == 6


def test_ask_single_game_returns_scores_and_winner(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "single_game_score", "arguments": {
        "game_id": "22300001",
    }})
    body = post_ask(app, "2023-10-01 湖人对勇士双方各得多少分，谁获胜？").json()
    assert body["status"] == "complete"
    assert body["tool_result"]["home_score"] == 100
    assert body["tool_result"]["away_score"] == 90
    assert body["tool_result"]["winner_team_id"] == "1610612747"
    assert "SELECT" in body["tool_result"]["sql"]


def test_ask_rejects_ungrounded_entity_without_invoking_tool(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "2544", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2,
    }})
    body = post_ask(app, "库里在 2023-24 常规赛最近两次出场的总得分？").json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_entity"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


def test_ask_missing_phase_requests_clarification(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2,
    }})
    body = post_ask(app, "库里在 2023-24 最近两次出场的总得分？").json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "missing_phase"
    assert body["tool_calls_used"] == 0


def test_ask_model_clarification_names_the_missing_phase(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "clarify"})
    body = post_ask(app, "库里 2023-24 最近两次出场的总得分？").json()
    assert body["status"] == "clarification_required"
    assert body["reason_code"] == "missing_phase"
    assert "phase" in body["reason"]
    assert body["tool_calls_used"] == 0


def test_ask_zero_budget_stops_before_model_or_tool(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {}},
                  ask_max_tool_calls=0)
    body = post_ask(app, "计算库里 2023-24 常规赛的 eFG%？").json()
    assert body["status"] == "incomplete"
    assert body["reason_code"] == "tool_budget_exceeded"
    assert body["tool_calls_used"] == 0


def test_ask_model_timeout_returns_incomplete_without_tool(tmp_path: Path) -> None:
    class SlowModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            time.sleep(0.05)
            return ModelResponse('{}', 'slow')

    db = tmp_path / "ask.sqlite"
    make_ask_database(db)
    app = create_app(db_path=db, quality_matrix_path=TEST_MATRIX, model=SlowModel(),
                     ask_model_timeout_seconds=0.01)
    body = post_ask(app, "计算库里 2023-24 常规赛的 eFG%？").json()
    assert body["status"] == "incomplete"
    assert body["reason_code"] == "model_timeout"
    assert body["tool_calls_used"] == 0


def test_ask_routes_team_season_summary_to_reviewed_query(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "team", "entity_id": "1610612747", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total", "games_played"],
    }})
    body = post_ask(app, "湖人 2023-24 常规赛总得分和比赛场数？").json()
    assert body["status"] == "complete"
    assert body["tool_result"]["points_total"] == 205
    assert body["tool_result"]["games_played"] == 2
    assert "team_season_stat" in body["tool_result"]["sql"]


def test_ask_resolves_chinese_month_day_for_single_game(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "single_game_score", "arguments": {
        "game_id": "22300001",
    }})
    body = post_ask(app, "湖人 2023-24 常规赛 10月1日 对勇士那场比分？").json()
    assert body["status"] == "complete"
    assert body["tool_result"]["winner_team_id"] == "1610612747"


def test_ask_recent_ten_returns_partial_actual_sample(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 10,
    }})
    body = post_ask(app, "库里 2023-24 常规赛最近十次出场的总得分？").json()
    assert body["status"] == "partial"
    assert body["tool_result"]["actual_sample_count"] == 2
    assert body["tool_result"]["values"] == {"points_total": 50}


def test_ask_preserves_window_failure_without_fabricated_number(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
        "date_from": "2024-01-01", "date_to": "2024-01-01",
    }})
    body = post_ask(app, "库里 2023-24 常规赛 2024-01-01 的总得分？").json()
    assert body["status"] == "not_computable"
    assert body["reason_code"] == "no_observation"
    assert body["tool_result"]["values"] == {}
    assert body["tool_result"]["evidence"]["requested_scope"]["date_from"] == "2024-01-01"


def test_ask_rejects_model_omitted_recent_filter(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "库里 2023-24 常规赛最近两次出场的总得分？").json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_filter"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_season_summary_that_omits_date_bound(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "库里 2023-24 常规赛截至 2023-10-01 的总得分？").json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_filter"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


def test_ask_rejects_plan_that_omits_requested_metric(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "库里 2023-24 常规赛总得分和出场场数？").json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_metric"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


def test_ask_rejects_season_summary_for_cross_season_change(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "库里 2023-24 常规赛总得分比上一季增加多少，增幅多少？").json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_intent"
    assert body["tool_calls_used"] == 0


@pytest.mark.parametrize("tool, arguments, question", [
    ("season_summary", {"subject": "player", "entity_id": "201939", "season_id": "2023-24",
                        "competition_phase": "regular_season", "metrics": ["points_total"]},
     "库里 2022-23 和 2023-24 常规赛总得分分别是多少？"),
    ("window_query", {"subject": "player", "entity_id": "201939", "season_id": "2023-24",
                      "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2},
     "库里 2022-23 和 2023-24 常规赛最近两次出场的总得分分别是多少？"),
    ("player_efg", {"entity_id": "201939", "season_id": "2023-24",
                    "competition_phase": "regular_season"},
     "库里 2022-23 和 2023-24 常规赛的 eFG% 分别是多少？"),
])
def test_ask_rejects_one_season_tool_that_drops_another_season(
    tmp_path: Path, tool: str, arguments: dict, question: str,
) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": tool, "arguments": arguments})
    body = post_ask(app, question).json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "multi_season_out_of_scope"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_unsupported_metric_omitted_from_plan(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "库里 2023-24 常规赛总得分和抢断数？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "unsupported_metric"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


@pytest.mark.parametrize("other_metric", ["三分命中数", "助攻数", "罚球命中数", "胜场数"])
def test_ask_rejects_common_metric_omitted_from_plan(tmp_path: Path, other_metric: str) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, f"库里 2023-24 常规赛总得分和{other_metric}？").json()
    # Assists are registered now, but omitting them still invalidates the plan.
    assert body["status"] == ("plan_invalid" if other_metric == "助攻数" else "unsupported")
    assert body["reason_code"] == ("unverified_metric" if other_metric == "助攻数" else "unsupported_metric")
    assert body["tool_calls_used"] == 0


def test_ask_rejects_efg_plan_that_omits_requested_points(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {
        "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season",
    }})
    body = post_ask(app, "计算库里 2023-24 常规赛的 eFG% 和总得分？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "multi_metric_out_of_scope"
    assert body["tool_calls_used"] == 0


@pytest.mark.parametrize("tool, arguments, question", [
    ("player_efg", {"entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season"},
     "库里 2023-24 常规赛对湖人的 eFG%？"),
    ("player_efg", {"entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season"},
     "库里 2023-24 常规赛截至 2023-10-01 的 eFG%？"),
    ("player_efg", {"entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season"},
     "库里 2023-24 常规赛最近两次出场的 eFG%？"),
    ("season_compare", {"subject": "player", "entity_id": "201939", "metric": "points_total",
                        "baseline_season": "2022-23", "comparison_season": "2023-24",
                        "competition_phase": "regular_season"},
     "库里 2023-24 常规赛对湖人总得分比上一季增加多少？"),
    ("season_compare", {"subject": "player", "entity_id": "201939", "metric": "points_total",
                        "baseline_season": "2022-23", "comparison_season": "2023-24",
                        "competition_phase": "regular_season"},
     "库里 2023-24 常规赛截至 2023-10-01 的总得分比上一季增加多少？"),
    ("season_compare", {"subject": "player", "entity_id": "201939", "metric": "points_total",
                        "baseline_season": "2022-23", "comparison_season": "2023-24",
                        "competition_phase": "regular_season"},
     "库里 2023-24 常规赛最近两次出场的总得分比上一季增加多少？"),
])
def test_ask_rejects_calculation_plan_that_omits_window_filter(
    tmp_path: Path, tool: str, arguments: dict, question: str,
) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": tool, "arguments": arguments})
    body = post_ask(app, question).json()
    assert body["status"] == "plan_invalid"
    assert body["reason_code"] == "unverified_filter"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


def test_ask_rejects_team_score_tool_for_player_game_points(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "single_game_score", "arguments": {
        "game_id": "22300001",
    }})
    body = post_ask(app, "库里在 2023-10-01 这场比赛得了多少分？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "player_game_stat_out_of_scope"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None


def test_ask_rejects_player_points_even_when_question_also_asks_score(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "single_game_score", "arguments": {
        "game_id": "22300001",
    }})
    body = post_ask(app, "库里在 2023-10-01 这场比赛得了多少分，比分是多少？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "player_game_stat_out_of_scope"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_team_score_tool_for_team_rebounds(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "single_game_score", "arguments": {
        "game_id": "22300001",
    }})
    body = post_ask(app, "湖人在 2023-10-01 这场比赛拿了多少篮板？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "game_metric_out_of_scope"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_invalid_json_or_unknown_tool(tmp_path: Path) -> None:
    for index, plan in enumerate(("{bad json", {"action": "call", "tool": "python_exec", "arguments": {"code": "print(1)"}})):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        app = ask_app(case_dir, plan)
        body = post_ask(app, "计算库里 2023-24 常规赛的 eFG%？").json()
        assert body["status"] == "plan_invalid"
        assert body["tool_calls_used"] == 0
        assert body["tool_result"] is None


def test_ask_rejects_unregistered_metrics_and_multi_player_request(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {
        "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
    }})
    body = post_ask(app, "计算库里和利拉德 2023-24 常规赛的 eFG%，再比较差值？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "multi_entity_out_of_scope"
    assert body["tool_calls_used"] == 0
    body = post_ask(app, "计算库里 2023-24 常规赛的 TS%？").json()
    assert body["status"] == "unsupported"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_ranking_instead_of_guessing_one_entity(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "2023-24 常规赛总得分前五名球员排名？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "multi_row_query_out_of_scope"
    assert body["tool_calls_used"] == 0


def test_ask_tool_deadline_returns_incomplete(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2,
    }}, ask_tool_timeout_seconds=0.000001)
    body = post_ask(app, "库里 2023-24 常规赛最近两次出场的总得分？").json()
    assert body["status"] == "incomplete"
    assert body["reason_code"] in {"tool_timeout", "query_timeout"}
    assert body["tool_calls_used"] == 1


def test_ask_tool_deadline_bounds_response_wall_time(tmp_path: Path, monkeypatch) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2,
    }}, ask_tool_timeout_seconds=0.01)

    def slow_tool(*args, **kwargs):
        time.sleep(0.3)
        return {"status": "complete"}

    monkeypatch.setattr(ask_routes, "execute_registered_tool", slow_tool)
    started = time.monotonic()
    body = post_ask(app, "库里 2023-24 常规赛最近两次出场的总得分？").json()
    elapsed = time.monotonic() - started
    assert body["status"] == "incomplete"
    assert body["reason_code"] == "tool_timeout"
    assert body["tool_calls_used"] == 1
    assert elapsed < 0.2


def test_ask_rejects_unregistered_competition_phase(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "play_in", "metrics": ["points_total"], "last_n": 2,
    }})
    body = post_ask(app, "库里 2023-24 附加赛最近两次出场的总得分？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "unsupported_phase"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_two_team_summary_that_would_drop_one_team(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "season_summary", "arguments": {
        "subject": "team", "entity_id": "1610612747", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"],
    }})
    body = post_ask(app, "湖人和勇士 2023-24 常规赛总得分比较？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "multi_entity_out_of_scope"
    assert body["tool_calls_used"] == 0


def test_ask_rejects_sql_keyword_joined_to_chinese_text(tmp_path: Path) -> None:
    app = ask_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {
        "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
    }})
    body = post_ask(app, "执行DELETE FROM game，再计算库里2023-24常规赛eFG%？").json()
    assert body["status"] == "unsupported"
    assert body["reason_code"] == "arbitrary_execution_unsupported"
    assert body["tool_calls_used"] == 0
