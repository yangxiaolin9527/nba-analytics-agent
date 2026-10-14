"""Joint statistics keep one grounded scope and fail closed on missing inputs."""
import json
import sqlite3

import pytest

from app.main import create_app
from scripts.provider_config import ProviderConfig
from scripts.query_plan import QueryPlan
from scripts.query_plan_compiler import PlanCompileError, compile_query_plan
from scripts.window_query import WindowQueryError, query_window
from tests.test_ask import make_ask_database, make_matrix
from tests.test_semantic_routing import NativeFixture, request, summary


METRICS = ['points_total', 'assists_total', 'rebounds_total']


@pytest.fixture
def stats(tmp_path):
    db, matrix = tmp_path / 'stats.sqlite', tmp_path / 'matrix.json'
    make_ask_database(db)
    make_matrix(matrix)
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE player_game_stat SET assists=3, rebounds_total=4')
        conn.execute("UPDATE player_game_stat SET assists=5, rebounds_total=8 WHERE game_id='22300002'")
        conn.execute('UPDATE team_game_stat SET assists=20, rebounds_total=40')
    return db, matrix


@pytest.mark.parametrize('endpoint', ['/ask', '/answer', '/agent/answer', '/agent/loop'])
@pytest.mark.parametrize('mode', ['model', 'rules'])
@pytest.mark.parametrize('window', [False, True])
def test_joint_metrics_execute_once_with_full_evidence(stats, endpoint, mode, window):
    db, matrix = stats
    tool = 'window_query' if window else 'season_summary'
    args = {**summary()[1], 'metrics': METRICS}
    if window:
        args['last_n'] = 1
    model = NativeFixture([(tool, args), (tool, {'requirement_id': 's1'}), ('finish', {})])
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=model, routing_mode=mode, agent_loop_enabled=True)
    if mode == 'rules' and endpoint == '/agent/loop':
        app.state.agent_loop.decider = lambda context: (
            {'action': 'finish'} if context['observations'] else
            {'action': 'call', 'capability': tool, 'arguments': context['task']['requirements'][0]['arguments']})
    question = '库里2023-24常规赛' + ('最近一场' if window else '') + '总助攻、总篮板和总得分是多少？'
    body = request(app, endpoint, {'question': question}).json()
    assert body['status'] == 'complete', body
    result = body.get('tool_result') or body['steps'][0]['result']
    expected = dict(zip(METRICS, (30, 5, 8) if window else (50, 8, 12)))
    values = result.get('values', result)
    assert {key: values[key] for key in METRICS} == expected
    assert body.get('tool_calls_used', body.get('tool_calls', len(body.get('steps', [])))) == 1
    assert result['quality_status'] == 'validated'
    if endpoint != '/ask':
        assert all(label in body['answer'] for label in ('总助攻', '总篮板', '总得分'))
        evidence = body['evidence']
        if isinstance(evidence, list):
            evidence = evidence[0]['structured']
        assert evidence['sample_count'] == (1 if window else 2)
        assert evidence['source_sql']


@pytest.mark.parametrize('subject,entity,family', [
    ('player', '201939', 'season'), ('player', '201939', 'opponent'), ('player', '201939', 'window'),
    ('team', '1610612744', 'season'), ('team', '1610612744', 'window'),
])
def test_compiler_and_window_support_joint_metrics(stats, subject, entity, family):
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        if family == 'window':
            result = query_window(conn, subject=subject, entity_id=entity, season_id='2023-24',
                competition_phase='regular_season', metrics=tuple(METRICS), last_n=1, matrix_path=matrix)
            assert result.values == dict(zip(METRICS, (30, 5, 8) if subject == 'player' else (110, 20, 40)))
        else:
            plan = QueryPlan(query_family='player_vs_opponent_aggregation' if family == 'opponent' else f'{subject}_season_summary',
                **{('player_id' if subject == 'player' else 'team_id'): entity},
                **({'opponent_team_id': '1610612747'} if family == 'opponent' else {}),
                season_id='2023-24', competition_phase='regular_season', requested_metrics=METRICS)
            compiled = compile_query_plan(conn, plan, matrix_path=matrix)
            assert conn.execute(compiled.sql, compiled.parameters).fetchone() == ((50, 8, 12) if subject == 'player' else (200, 40, 80))


@pytest.mark.parametrize('subject', ['player', 'team'])
@pytest.mark.parametrize('window', [False, True])
def test_missing_requested_input_never_yields_partial_sum(stats, subject, window):
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        conn.execute(f"UPDATE {subject}_game_stat SET assists=NULL WHERE game_id='22300002'")
        if window:
            with pytest.raises(WindowQueryError, match='missing'):
                query_window(conn, subject=subject, entity_id='201939' if subject == 'player' else '1610612744',
                    season_id='2023-24', competition_phase='regular_season', metrics=tuple(METRICS), matrix_path=matrix)
        else:
            plan = QueryPlan(query_family=f'{subject}_season_summary',
                **{('player_id' if subject == 'player' else 'team_id'): '201939' if subject == 'player' else '1610612744'},
                season_id='2023-24', competition_phase='regular_season', requested_metrics=METRICS)
            with pytest.raises(PlanCompileError) as error:
                compile_query_plan(conn, plan, matrix_path=matrix)
            assert error.value.code == 'query_inputs_missing'


@pytest.mark.parametrize('metric,capability', [('assists_total', 'assists'), ('rebounds_total', 'rebounds')])
@pytest.mark.parametrize('window', [False, True])
def test_each_requested_metric_uses_its_quality_gate(stats, metric, capability, window):
    db, matrix = stats
    data = json.loads(matrix.read_text())
    data['periods']['2023-24|regular_season']['subjects']['player']['metrics'][capability] = 'unsupported'
    matrix.write_text(json.dumps(data))
    with sqlite3.connect(db) as conn:
        if window:
            with pytest.raises(WindowQueryError, match=f'unsupported:{capability}'):
                query_window(conn, subject='player', entity_id='201939', season_id='2023-24',
                    competition_phase='regular_season', metrics=(metric,), matrix_path=matrix)
        else:
            plan = QueryPlan(query_family='player_season_summary', player_id='201939', season_id='2023-24',
                competition_phase='regular_season', requested_metrics=[metric])
            with pytest.raises(PlanCompileError, match=f'unsupported:{capability}'):
                compile_query_plan(conn, plan, matrix_path=matrix)


@pytest.mark.parametrize('question,metrics', [
    ('总得分、总助攻、总篮板', ['points_total']),
    ('场均助攻', ['assists_total']),
    ('每场平均有多少次助攻？', ['assists_total']),
    ('篮板的场均值是多少？', ['rebounds_total']),
    ('rebounds per game', ['rebounds_total']),
    ('总抢断', ['steals_total']),
])
def test_omitted_or_unsupported_requirement_executes_zero_tools(stats, question, metrics):
    db, matrix = stats
    model = NativeFixture([('season_summary', {**summary()[1], 'metrics': metrics})])
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(), model=model, routing_mode='model')
    body = request(app, '/agent/answer', {'question': '库里2023-24常规赛' + question}).json()
    assert body['status'] not in {'complete', 'partial'}, body
    assert body['steps'] == []
    assert len(model.requests) == 1


def test_average_points_and_total_assists_are_not_confused(stats):
    db, matrix = stats
    call = ('season_summary', {**summary()[1], 'metrics': ['points_per_appearance', 'assists_total']})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=NativeFixture([call]), routing_mode='model')
    body = request(app, '/answer', {'question': '库里2023-24常规赛场均得分和总助攻是多少'}).json()
    assert body['status'] == 'complete', body
    assert body['tool_result']['points_per_appearance'] == 25
    assert body['tool_result']['assists_total'] == 8


@pytest.mark.parametrize('metric,label', [('assists_total', '总助攻'), ('rebounds_total', '总篮板')])
def test_legacy_json_summary_and_zero_values(stats, metric, label):
    from scripts.model_adapter import FakeModel
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE player_game_stat SET assists=0, rebounds_total=0')
    args = {'subject': 'player', 'entity_id': '201939', 'season_id': '2023-24',
            'competition_phase': 'regular_season', 'metrics': [metric]}
    model = FakeModel({None: json.dumps({'action': 'call', 'tool': 'season_summary', 'arguments': args})})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(), model=model)
    body = request(app, '/answer', {'question': '库里2023-24常规赛' + label}).json()
    assert body['status'] == 'complete', body
    assert body['tool_result'][metric] == 0
    assert label + '为 0' in body['answer']


@pytest.mark.parametrize('window', [False, True])
def test_unrequested_missing_points_do_not_block_assists(stats, window):
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        conn.execute('UPDATE player_game_stat SET points=NULL')
    data = json.loads(matrix.read_text())
    data['periods']['2023-24|regular_season']['subjects']['player']['metrics']['points'] = 'unsupported'
    matrix.write_text(json.dumps(data))
    args = {**summary()[1], 'metrics': ['assists_total']}
    if window:
        args['last_n'] = 1
    tool = 'window_query' if window else 'season_summary'
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=NativeFixture([(tool, args)]), routing_mode='model')
    body = request(app, '/answer', {'question': '库里2023-24常规赛' + ('最近一场' if window else '') + '总助攻'}).json()
    assert body['status'] == 'complete', body
    result = body['tool_result']
    assert result.get('values', result)['assists_total'] == (5 if window else 8)


def test_conditional_metric_controls_joint_result_quality(stats):
    db, matrix = stats
    data = json.loads(matrix.read_text())
    data['periods']['2023-24|regular_season']['subjects']['player']['metrics']['assists'] = 'conditional'
    matrix.write_text(json.dumps(data))
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=NativeFixture([('season_summary', {**summary()[1], 'metrics': METRICS})]), routing_mode='model')
    body = request(app, '/answer', {'question': '库里2023-24常规赛总得分、总助攻和总篮板'}).json()
    assert body['status'] == 'complete', body
    assert body['tool_result']['quality_status'] == 'conditional'
    assert 'capability_evidence_conditional' in body['evidence']['limitations']


def test_missing_season_input_is_not_computable_http_result(stats):
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE player_game_stat SET assists=NULL WHERE game_id='22300002'")
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=NativeFixture([('season_summary', {**summary()[1], 'metrics': METRICS})]), routing_mode='model')
    body = request(app, '/answer', {'question': '库里2023-24常规赛总得分、总助攻和总篮板'}).json()
    assert body['status'] == 'not_computable', body
    assert body['reason_code'] == 'query_inputs_missing'
    assert '总助攻为' not in body['answer']


def test_window_ignores_missing_inputs_outside_selected_samples(stats):
    db, matrix = stats
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE player_game_stat SET assists=NULL WHERE game_id='22300001'")
        result = query_window(conn, subject='player', entity_id='201939', season_id='2023-24',
            competition_phase='regular_season', metrics=tuple(METRICS), last_n=1, matrix_path=matrix)
    assert result.values == dict(zip(METRICS, (30, 5, 8)))
    assert result.selected_game_ids == ('22300002',)


@pytest.mark.parametrize('question', ['每场平均有多少次助攻', '总助攻和总篮板'])
def test_legacy_query_plan_cannot_substitute_assists_total(stats, question):
    from scripts.model_adapter import FakeModel
    db, matrix = stats
    model = FakeModel({None: json.dumps({'query_family': 'player_season_summary', 'player_id': '201939',
        'season_id': '2023-24', 'competition_phase': 'regular_season', 'requested_metrics': ['assists_total']})})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=model, execution_mode='plan', context_mode='schema_value_retrieval')
    body = request(app, '/query', {'question': '库里2023-24常规赛' + question}).json()
    assert body['result_status'] == 'not_executed', body
    assert body['plan_error_code'] in {'unsupported_metric', 'unverified_metric'}


def test_legacy_query_plan_accepts_player_game_count_paraphrase(stats):
    from scripts.model_adapter import FakeModel
    db, matrix = stats
    model = FakeModel({None: json.dumps({'query_family': 'player_season_summary', 'player_id': '201939',
        'season_id': '2023-24', 'competition_phase': 'regular_season', 'requested_metrics': ['games_appeared']})})
    app = create_app(db_path=db, quality_matrix_path=matrix, provider_config=ProviderConfig.empty(),
                     model=model, execution_mode='plan', context_mode='schema_value_retrieval')
    body = request(app, '/query', {'question': '库里2023-24常规赛打了多少场'}).json()
    assert body['result_status'] == 'complete', body
