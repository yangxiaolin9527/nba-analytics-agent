"""Cross-layer regressions for historical seasons and complete query scope."""
import json
import sqlite3
from copy import deepcopy

import pytest

from app.main import create_app
from scripts.ask_grounding import normalize_question_dates
from scripts.query_scope import QueryScopeError, requested_date_bounds
from scripts.provider_config import ProviderConfig
from scripts.query_plan import QueryPlan
from tests.test_ask import make_ask_database, make_matrix
from tests.test_semantic_routing import NativeFixture, request, summary


@pytest.mark.parametrize('season', ['1946-47', '1997-98', '1999-00', '2023-24'])
def test_query_plan_accepts_consecutive_historical_seasons(season):
    plan = QueryPlan(query_family='player_season_summary', player_id='893',
                     season_id=season, competition_phase='playoffs', requested_metrics=['points_total'])
    assert plan.season_id == season


@pytest.mark.parametrize('season', ['1997-99', '1999-99', '2023-25'])
def test_query_plan_rejects_nonconsecutive_seasons(season):
    with pytest.raises(ValueError):
        QueryPlan(query_family='player_season_summary', player_id='893',
                  season_id=season, competition_phase='playoffs', requested_metrics=['points_total'])


@pytest.mark.parametrize('question,expected', [
    ('库里 2023-24 常规赛截至2023年12月31日总得分', '2023-12-31'),
    ('乔丹 1997-98 季后赛截至1998年6月1日总得分', '1998-06-01'),
    ('库里 2023-24 常规赛截至2025年1月1日总得分', '2025-01-01'),
])
def test_explicit_chinese_year_is_authoritative_and_normalization_is_idempotent(question, expected):
    normalized = normalize_question_dates(question)
    assert normalized.count(expected) == 1
    assert normalize_question_dates(normalized) == normalized
    assert '2024-01-01' not in normalized


@pytest.mark.parametrize('question,expected', [
    ('从1998-05-01到1998-06-01', ('1998-05-01', '1998-06-01')),
    ('截至1998-06-01', (None, '1998-06-01')),
    ('自1998-06-01以来', ('1998-06-01', None)),
    ('1998-06-01当天', ('1998-06-01', '1998-06-01')),
])
def test_date_bound_direction(question, expected):
    assert requested_date_bounds(question) == expected


@pytest.mark.parametrize('question', ['截至1998-02-30', '从1998-06-01到1998-05-01',
                                       '1998-05-01和1998-05-02和1998-05-03'])
def test_invalid_date_scope_is_rejected(question):
    with pytest.raises(QueryScopeError):
        requested_date_bounds(question)


def historical_app(tmp_path, mode, calls=()):
    db, matrix = tmp_path / 'scope.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE game SET season_id='1997-98', competition_phase='playoffs', game_date=CASE game_id WHEN '22300001' THEN '1998-05-30' ELSE '1998-06-03' END WHERE season_id='2023-24'")
        conn.execute("UPDATE player_identity SET display_name='Michael Jordan' WHERE person_id='201939'")
    data = json.loads(matrix.read_text())
    data['periods']['1997-98|playoffs'] = deepcopy(data['periods']['2023-24|playoffs'])
    matrix.write_text(json.dumps(data))
    return create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                      model=NativeFixture(calls), routing_mode=mode, agent_loop_enabled=True)


@pytest.mark.parametrize('endpoint', ['/ask', '/answer', '/agent/answer', '/agent/loop'])
@pytest.mark.parametrize('mode', ['model', 'rules'])
@pytest.mark.parametrize('cutoff,expected', [(False, 50), (True, 20)])
def test_historical_scope_executes_across_entrypoints(tmp_path, endpoint, mode, cutoff, expected):
    args = {**summary()[1], 'entity': {'mention': 'Michael Jordan'},
            'season_id': '1997-98', 'competition_phase': 'playoffs'}
    tool = 'window_query' if cutoff else 'season_summary'
    if cutoff:
        args['date_to'] = '1998-06-01'
    calls = [(tool, args), (tool, {'requirement_id': 's1'}), ('finish', {})]
    app = historical_app(tmp_path, mode, calls)
    if mode == 'rules' and endpoint == '/agent/loop':
        app.state.agent_loop.decider = lambda context: (
            {'action': 'finish'} if context['observations'] else
            {'action': 'call', 'capability': tool, 'arguments': context['task']['requirements'][0]['arguments']})
    question = 'Michael Jordan 1997-1998 季后赛' + ('截至1998-06-01的' if cutoff else '') + '总得分？'
    body = request(app, endpoint, {'question': question}).json()
    assert body['status'] == 'complete', body
    result = body.get('tool_result') or body['steps'][0]['result']
    assert result.get('points_total', result.get('values', {}).get('points_total')) == expected
    assert result['quality_status'] == 'conditional'
    assert 'season_completeness_not_checked' in result['limitations']


@pytest.mark.parametrize('endpoint', ['/ask', '/answer', '/agent/answer', '/agent/loop'])
@pytest.mark.parametrize('question,call', [
    ('库里 2023-24 常规赛截至2024-01-01总得分', summary()),
    ('库里 2023-24 常规赛截至2024-01-01总得分', ('window_query', {**summary()[1], 'date_from': '2024-01-01'})),
    ('库里 2023-24 常规赛总得分', ('window_query', {**summary()[1], 'date_to': '2024-01-01'})),
    ('库里 2023-24 常规赛面对湖人的总得分', summary()),
    ('库里 2023-24 常规赛面对犹他爵士的总得分', summary()),
    ('库里 2023-24 常规赛面对湖人的总得分', ('window_query', summary()[1])),
    ('库里 2023-24 常规赛总得分和总助攻', summary()),
    ('库里 2023-24 常规赛总得分', ('window_query', {**summary()[1], 'last_n': 1})),
    ('库里 2023-24 常规赛截至2024-01-01的efg%', ('player_efg', {k: v for k, v in summary()[1].items() if k not in {'subject', 'metrics'}})),
    ('库里 2023-24 常规赛总得分和出场次数', summary()),
    ('库里 2023-24 常规赛总得分', ('window_query', {**summary()[1], 'opponent': {'mention': '库里', 'canonical_name': 'Los Angeles Lakers'}})),
    ('库里2023-24常规赛截至2024-01-01总得分比上一季增加多少', ('season_compare', {
        'subject': 'player', 'entity': {'mention': '库里'}, 'baseline_season': '2022-23',
        'comparison_season': '2023-24', 'competition_phase': 'regular_season', 'metric': 'points_total'})),
    ('2023-10-01湖人对勇士比赛比分和总助攻是多少？', ('single_game_score', {
        'game_date': '2023-10-01', 'home_or_away_teams': [{'mention': '湖人'}, {'mention': '勇士'}]})),
])
def test_native_wrong_scope_never_executes(tmp_path, endpoint, question, call):
    db, matrix = tmp_path / 'scope.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    model = NativeFixture([call])
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                      model=model, routing_mode='model', agent_loop_enabled=True)
    body = request(app, endpoint, {'question': question}).json()
    assert body['status'] not in {'complete', 'partial'}, body
    assert body.get('tool_calls_used', body.get('tool_calls', len(body.get('steps', [])))) == 0
    assert len(model.requests) == 1
    assert not body.get('context', {}).get('requirements')


@pytest.mark.parametrize('endpoint', ['/ask', '/answer', '/agent/answer'])
def test_valid_native_opponent_and_date_window_executes(tmp_path, endpoint):
    db, matrix = tmp_path / 'valid.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    call = ('window_query', {**summary()[1], 'opponent': {'mention': '湖人'},
                             'date_from': '2023-10-01', 'date_to': '2023-10-02'})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                      model=NativeFixture([call]), routing_mode='model')
    body = request(app, endpoint, {'question': '库里2023-24常规赛从2023年10月1日到2023年10月2日面对湖人的总得分'}).json()
    assert body['status'] == 'complete', body
    result = body.get('tool_result') or body['steps'][0]['result']
    assert result['values']['points_total'] == 20


@pytest.mark.parametrize('question,metrics', [
    ('库里2023-24常规赛打了多少场？', ['games_appeared']),
    ('库里2023-24常规赛总得分和比赛场数？', ['points_total', 'games_appeared']),
])
def test_native_player_game_count_preserves_paraphrases(tmp_path, question, metrics):
    db, matrix = tmp_path / 'count.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    call = ('season_summary', {**summary()[1], 'metrics': metrics})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                      model=NativeFixture([call]), routing_mode='model')
    body = request(app, '/answer', {'question': question}).json()
    assert body['status'] == 'complete', body
    assert body['tool_result']['games_appeared'] == 2


def test_native_player_vs_comparison_is_not_team_filter(tmp_path):
    from tests.test_semantic_routing import plan
    db = tmp_path / 'comparison.sqlite'
    make_ask_database(db)
    call = ('compare_player_efg', {'left': {'mention': 'Stephen Curry'},
        'right': {'mention': 'LeBron James'}, 'season_id': '2023-24', 'competition_phase': 'regular_season'})
    result, _ = plan(db, 'Stephen Curry vs LeBron James 2023-24 regular season eFG% comparison', [call])
    assert result.plan.route == 'comparison', result.plan
    assert len(result.plan.steps) == 4


def test_historical_single_game_date_is_grounded(tmp_path):
    app = historical_app(tmp_path, 'rules')
    body = request(app, '/answer', {'question': '1998年5月30日湖人对勇士比分？'}).json()
    assert body['status'] == 'complete', body
    assert body['evidence']['game_date'] == '1998-05-30'


def test_legacy_ask_path_executes_historical_season(tmp_path):
    from scripts.model_adapter import FakeModel
    historical_app(tmp_path, None)
    # The legacy adapter consumes JSON text rather than native tool calls.
    model = FakeModel({None: json.dumps({'action': 'call', 'tool': 'season_summary',
        'arguments': {'subject': 'player', 'entity_id': '201939', 'season_id': '1997-98',
                      'competition_phase': 'playoffs', 'metrics': ['points_total']}})})
    app = create_app(db_path=tmp_path / 'scope.sqlite', quality_matrix_path=tmp_path / 'matrix.json',
                      provider_config=ProviderConfig.empty(), model=model)
    body = request(app, '/answer', {'question': 'Michael Jordan 1997-1998季后赛总得分'}).json()
    assert body['status'] == 'complete', body
    assert body['tool_result']['points_total'] == 50


@pytest.mark.parametrize('endpoint', ['/ask', '/answer', '/agent/answer'])
@pytest.mark.parametrize('exception_type', [ValueError, RuntimeError])
def test_structured_failure_has_safe_stage_diagnostics(tmp_path, monkeypatch, endpoint, caplog, exception_type):
    import app.agent_routes as routes
    db = tmp_path / 'failure.sqlite'
    make_ask_database(db)

    def invalid(*args, **kwargs):
        raise exception_type('secret fixture exception')

    monkeypatch.setattr(routes, 'execute_registered_tool', invalid)
    app = create_app(db_path=db, provider_config=ProviderConfig.empty(),
                      model=NativeFixture([summary()]), routing_mode='model')
    body = request(app, endpoint, {'question': '库里2023-24常规赛总得分'}).json()
    assert body['status'] == 'incomplete'
    if endpoint == '/agent/answer':
        assert body['steps'][0]['result']['reason_code'] == 'tool_failed'
        assert body['evidence'][0]['error_stage'] == 'execution'
        assert body['evidence'][0]['reason_code'] == 'tool_failed'
    else:
        assert body['reason_code'] == 'tool_failed'
    assert body['error_stage'] == 'execution'
    assert 'secret fixture exception' not in json.dumps(body)
    assert exception_type.__name__ in caplog.text
    assert 'secret fixture exception' not in caplog.text


def test_demo_nonempty_failure_answer_still_displays_diagnostic():
    import subprocess
    from pathlib import Path

    source = Path('static/demo.js').read_text()
    source = source[:source.index("['agent', 'stats', 'knowledge'].forEach((view) => $(view + '-submit')")]
    script = """
const vm = require('node:vm');
const state = require('./static/demo-state.js');
const nodes = new Map();
const node = () => ({ hidden: true, innerHTML: '', removeAttribute() {}, querySelectorAll() { return []; }, querySelector() { return { addEventListener() {} }; } });
const context = { ...state, window: {}, document: { getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); } }, Intl, setTimeout, clearTimeout };
vm.createContext(context);
vm.runInContext(SOURCE, context);
vm.runInContext(`renderResult('stats', {status:'incomplete', answer:'请求未能完成', reason_code:'tool_failed', error_stage:'execution', tool:'season_summary', evidence:{tool_status:'failed', quality_status:null}}, 'test')`, context);
const html = nodes.get('stats-result').innerHTML;
if (!html.includes('失败阶段：执行') || !html.includes('错误码：tool_failed')) throw Error(html);
if (!html.includes('未生成可验证结果') || html.includes('质量未标注')) throw Error(html);
""".replace('SOURCE', json.dumps(source))
    completed = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
