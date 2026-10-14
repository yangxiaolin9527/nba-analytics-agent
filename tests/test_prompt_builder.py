from app.main import _query_context
from scripts.prompt_builder import build_plan_prompt, build_prompt
from scripts.query_context import EntityLink, QueryLinks


def test_prompt_includes_authoritative_values_and_output_contract() -> None:
    context, _ = _query_context()

    prompt = build_prompt("库里在 2023-24 常规赛的场均得分？", context)

    assert "regular_season" in prompt
    assert "Regular Season" in prompt
    assert "played_inferred" in prompt
    assert "points_per_appearance" in prompt
    assert "home_score" in prompt
    assert "Schema:" in prompt
    assert "Question: 库里在 2023-24 常规赛的场均得分？" in prompt


def test_prompt_includes_verified_entity_ids_only_for_value_retrieval() -> None:
    context, _ = _query_context()
    links = QueryLinks(players=[EntityLink(alias="库里", id="201939", display_name="Stephen Curry")])

    prompt = build_prompt("库里得分？", context, links=links)

    assert '"id":"201939"' in prompt
    assert "Stephen Curry" in prompt
    assert "Use verified entity IDs" in prompt


def test_plan_prompt_declares_canonical_metric_and_family_constraints() -> None:
    context, _ = _query_context()
    prompt = build_plan_prompt("库里在 2023-24 常规赛的场均得分？", context)

    assert "points_per_appearance" in prompt
    assert "canonical metric points_per_appearance" in prompt
    assert "single_game_score" in prompt
    assert "set player_id" in prompt
