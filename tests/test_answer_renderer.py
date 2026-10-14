from scripts.answer_renderer import render_tool_answer


def test_render_season_summary_preserves_tool_values_and_scope() -> None:
    rendered = render_tool_answer(
        "season_summary",
        {"subject": "player", "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season", "metrics": ["points_total"]},
        {
            "status": "complete", "points_total": 50, "games_appeared": 2,
            "sql": "SELECT points_total FROM player_season_stat", "source_table": "player_season_stat",
            "quality_status": "validated", "limitations": [], "source_snapshot": "eoin-v515",
        },
    )
    assert rendered.text == "player 201939在 2023-24 常规赛的总得分为 50。"
    assert rendered.values == {"points_total": 50}
    assert rendered.warnings == []


def test_render_window_partial_discloses_actual_sample_without_recomputing() -> None:
    rendered = render_tool_answer(
        "window_query",
        {"last_n": 10, "season_id": "2023-24", "competition_phase": "regular_season"},
        {
            "status": "partial", "values": {"points_total": 50},
            "actual_sample_count": 2, "available_sample_count": 2,
            "actual_date_from": "2023-10-01", "actual_date_to": "2023-10-03",
            "quality_status": "conditional", "limitations": ["fewer_than_requested_games"],
        },
    )
    assert "50" in rendered.text
    assert "2" in rendered.text
    assert "partial_result" in rendered.warnings
    assert rendered.values == {"points_total": 50}
    assert "结果只覆盖已取得的记录" in rendered.text
    assert "有条件可用" not in rendered.text
    assert "实际场次少于请求数量" in rendered.text


def test_render_quality_limitations_use_plain_language_and_match_query_scope() -> None:
    rendered = render_tool_answer(
        "season_summary",
        {"subject": "player", "entity_id": "201939", "season_id": "2023-24", "competition_phase": "playoffs", "metrics": ["points_total"]},
        {"status": "complete", "points_total": 12, "quality_status": "conditional",
         "limitations": ["season_completeness_not_checked", "capability_evidence_conditional"]},
    )
    assert "以上结果按当前数据库中的比赛记录计算" in rendered.text
    assert "尚未确认是否覆盖该赛季全部比赛" in rendered.text
    assert "数据能力证据" not in rendered.text
    assert "season_completeness_not_checked" not in rendered.text


def test_render_single_game_does_not_inherit_season_quality_warning() -> None:
    rendered = render_tool_answer(
        "single_game_score",
        {"game_id": "22200001"},
        {"status": "complete", "home_score": 100, "away_score": 90,
         "winner_team_id": "1", "quality_status": "conditional",
         "limitations": ["season_completeness_not_checked"]},
        evidence={"game_date": "2023-10-01", "home_team": {"id": "1", "name": "主队"},
                  "away_team": {"id": "2", "name": "客队"}},
    )
    assert "主队" in rendered.text
    assert "赛季全部比赛" not in rendered.text
    assert "有条件可用" not in rendered.text


def test_render_window_uses_query_range_for_completeness_caveat() -> None:
    rendered = render_tool_answer(
        "window_query",
        {"subject": "player", "entity_id": "201939", "season_id": "2023-24",
         "competition_phase": "regular_season", "last_n": 5, "metrics": ["points_total"]},
        {"status": "complete", "values": {"points_total": 50}, "actual_sample_count": 5,
         "actual_date_from": "2023-10-01", "actual_date_to": "2023-10-10",
         "quality_status": "conditional", "limitations": ["season_completeness_not_checked"]},
    )
    assert "当前查询范围内的比赛记录" in rendered.text
    assert "该赛季全部比赛" not in rendered.text


def test_render_efg_includes_formula_and_inputs() -> None:
    rendered = render_tool_answer(
        "player_efg",
        {"entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season"},
        {
            "status": "complete", "efg_percentage": 58.333333333333336,
            "inputs": {"field_goals_made": 18, "three_pointers_made": 6, "field_goals_attempted": 36, "sample_count": 2},
            "evidence": {"formula": "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100", "unit": "percent"},
            "quality_status": "validated", "limitations": [],
        },
    )
    assert "58.3333%" in rendered.text
    assert "18" in rendered.text and "6" in rendered.text and "36" in rendered.text
    assert rendered.evidence["formula"].startswith("(field_goals_made")
