import json
from pathlib import Path

from httpx import ASGITransport, AsyncClient
import asyncio

import pytest

from scripts.model_adapter import FakeModel
from tests.test_ask import make_ask_database, make_matrix


def post_answer(app, question: str):
    async def send():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/answer", json={"question": question})
    return asyncio.run(send())


def answer_app(tmp_path: Path, plan: dict, **kwargs):
    db = tmp_path / "answer.sqlite"
    make_ask_database(db)
    matrix = tmp_path / "matrix.json"
    make_matrix(matrix)
    model = FakeModel({None: json.dumps(plan, ensure_ascii=False)})
    from app.main import create_app
    return create_app(db_path=db, quality_matrix_path=matrix, model=model, **kwargs)


def test_answer_returns_deterministic_window_answer_and_evidence(tmp_path: Path) -> None:
    app = answer_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_per_appearance"], "last_n": 2,
    }})
    response = post_answer(app, "库里在 2023-24 常规赛最近两次出场的场均得分？")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "complete"
    assert body["tool"] == "window_query"
    assert body["tool_result"]["values"] == {"points_per_appearance": 25.0}
    assert "25" in body["answer"]
    assert body["query_metadata"]["snapshot_id"] == "eoin-v515"
    assert body["evidence"]["source_sql"]
    assert body["tool_calls_used"] == 1


def test_answer_accepts_full_year_season_with_grounded_model_plan(tmp_path):
    app = answer_app(tmp_path, {'action': 'call', 'tool': 'season_summary', 'arguments': {
        'subject': 'player', 'entity_id': '201939', 'season_id': '2022-23',
        'competition_phase': 'regular_season', 'metrics': ['points_total'],
    }})
    body = post_answer(app, '库里 2022-2023 常规赛总得分？').json()
    assert body['status'] == 'complete'
    assert body['arguments']['season_id'] == '2022-23'
    assert body['tool_result']['points_total'] == 10


@pytest.mark.parametrize('season', ['2020-2022', '2020-22', '2023-2023', '2023-202'])
def test_answer_rejects_invalid_explicit_season_before_model_execution(tmp_path, season):
    app = answer_app(tmp_path, {'action': 'call', 'tool': 'season_summary', 'arguments': {
        'subject': 'player', 'entity_id': '201939', 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'metrics': ['points_total'],
    }})
    body = post_answer(app, f'库里 {season} 常规赛总得分？').json()
    assert body['status'] == 'clarification_required'
    assert body['reason_code'] == 'invalid_season'
    assert body['tool_calls_used'] == 0


def test_answer_clarification_does_not_call_tool(tmp_path: Path) -> None:
    app = answer_app(tmp_path, {"action": "call", "tool": "window_query", "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 2,
    }})
    body = post_answer(app, "库里在 2023-24 最近两次出场的总得分？").json()
    assert body["status"] == "clarification_required"
    assert body["tool_calls_used"] == 0
    assert body["tool_result"] is None
    assert "阶段" in body["answer"]


def test_answer_invalid_plan_is_incomplete(tmp_path: Path) -> None:
    app = answer_app(tmp_path, {"action": "call", "tool": "not_registered", "arguments": {}})
    body = post_answer(app, "计算库里 2023-24 常规赛的 eFG%？").json()
    assert body["status"] == "incomplete"
    assert body["reason_code"] == "invalid_plan"
    assert body["tool_calls_used"] == 0


def test_answer_unsupported_request_does_not_call_tool(tmp_path: Path) -> None:
    app = answer_app(tmp_path, {"action": "call", "tool": "player_efg", "arguments": {}})
    body = post_answer(app, "计算库里 2023-24 常规赛的 TS%？").json()
    assert body["status"] == "unsupported"
    assert body["tool_calls_used"] == 0
    assert "支持" in body["answer"] or "不支持" in body["answer"]


def test_answer_uses_shared_executor_and_preserves_raw_result(tmp_path, monkeypatch):
    import app.ask_routes as routes
    original = routes.execute_registered_tool
    calls = []
    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(routes, 'execute_registered_tool', tracked)
    app = answer_app(tmp_path, {'action':'call', 'tool':'season_summary', 'arguments':{
        'subject':'player','entity_id':'201939','season_id':'2023-24',
        'competition_phase':'regular_season','metrics':['points_total']}})
    body = post_answer(app, '库里 2023-24 常规赛总得分？').json()
    assert len(calls) == 1
    assert body['tool_result'] == calls[0]
    assert body['evidence']['sample_count'] == 2
    assert body['evidence']['entity']['name'] in body['answer']
    assert body['evidence']['source_name'].startswith('eoinamoore/')


def test_answer_grounding_failure_is_incomplete_not_sixth_status(tmp_path):
    app = answer_app(tmp_path, {'action':'call','tool':'player_efg','arguments':{
        'entity_id':'2544','season_id':'2023-24','competition_phase':'regular_season'}})
    body = post_answer(app, '库里 2023-24 常规赛 eFG%？').json()
    assert (body['status'], body['reason_code']) == ('incomplete', 'invalid_plan')
    assert body['tool_calls_used'] == 0


def test_answer_database_unavailable_has_structured_failure(tmp_path):
    from app.main import create_app
    response = post_answer(create_app(db_path=tmp_path/'absent.sqlite'), '库里得分？')
    assert response.status_code == 200
    assert response.json()['status'] == 'incomplete'
    assert response.json()['reason_code'] == 'database_unavailable'


def test_answer_zero_baseline_never_renders_none_or_invented_percentage(tmp_path):
    import sqlite3
    app = answer_app(tmp_path, {'action':'call','tool':'season_compare','arguments':{
        'subject':'player','entity_id':'201939','metric':'points_total',
        'baseline_season':'2022-23','comparison_season':'2023-24','competition_phase':'regular_season'}})
    with sqlite3.connect(tmp_path/'answer.sqlite') as conn:
        conn.execute("UPDATE player_game_stat SET points=0 WHERE game_id='22200001'")
    body = post_answer(app, '库里 2023-24 常规赛总得分比上一季增加多少，增幅多少？').json()
    assert body['status'] == 'complete'
    assert body['tool_result']['status'] == 'partial'
    assert 'None' not in body['answer'] and '无法计算' in body['answer']
    assert 'zero_baseline' in body['limitations']
    assert body['evidence']['sources'][0]['source_parameters'] == ['201939','2022-23','regular_season']


def test_answer_partial_window_explains_requested_and_actual_samples(tmp_path):
    app = answer_app(tmp_path, {'action':'call','tool':'window_query','arguments':{
        'subject':'player','entity_id':'201939','season_id':'2023-24','competition_phase':'regular_season',
        'metrics':['points_total'],'last_n':10}})
    body = post_answer(app, '库里 2023-24 常规赛最近十次出场的总得分？').json()
    assert body['status'] == 'complete' and 'partial_result' in body['warnings']
    assert 'partial_result' in body['limitations']
    assert '10' in body['answer'] and '2' in body['answer'] and '不足' in body['answer']
    assert body['evidence']['requested_scope'] == body['arguments']
    assert body['evidence']['source_snapshot'].endswith('@515')


def test_answer_efg_evidence_has_query_bindings_formula_inputs_and_precision(tmp_path):
    app = answer_app(tmp_path, {'action':'call','tool':'player_efg','arguments':{
        'entity_id':'201939','season_id':'2023-24','competition_phase':'regular_season'}})
    body = post_answer(app, '库里 2023-24 常规赛 eFG%？').json()
    evidence = body['evidence']
    assert evidence['source_table'] == 'player_game_stat'
    assert evidence['source_sql'].count('?') == len(evidence['source_parameters']) == 3
    assert evidence['inputs'] == body['tool_result']['inputs']
    assert evidence['precision']['display_decimal_places'] == 4
    assert 'FGM' in body['answer'] and '公式' in body['answer']
    assert '2023-24' in body['answer'] and '常规赛' in body['answer']


def test_answer_not_computable_does_not_render_a_numeric_answer(tmp_path, monkeypatch):
    import app.ask_routes as routes

    def unavailable(*args, **kwargs):
        return {'status': 'not_computable', 'reason_code': 'no_observation',
                'reason': 'no matching observation', 'quality_status': 'unverified',
                'limitations': ['no_observation']}

    monkeypatch.setattr(routes, 'execute_registered_tool', unavailable)
    app = answer_app(tmp_path, {'action': 'call', 'tool': 'player_efg', 'arguments': {
        'entity_id': '201939', 'season_id': '2023-24', 'competition_phase': 'regular_season'}})
    body = post_answer(app, '库里 2023-24 常规赛 eFG%？').json()
    assert body['status'] == 'not_computable'
    assert body['reason_code'] == 'no_observation'
    assert body['tool_calls_used'] == 1
    assert body['tool_result']['status'] == 'not_computable'
    assert body['answer'] == '当前数据或质量条件不足，无法可靠计算该结果。'
