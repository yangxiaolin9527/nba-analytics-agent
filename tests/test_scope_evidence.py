"""R001: native Agent plans must not manufacture season/phase from a quote."""

import sqlite3

import pytest

from scripts.query_scope import normalize_question_seasons
from scripts.evaluate_agent_loop_conversation import FixtureDecider
from tests.test_semantic_routing import native_app, plan, request, summary
from tests.test_agent_composition import composed_call
from tests.test_unified_agent import unified_app


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('question,evidence,reason', [
    ('库里常规赛总得分？', {'season_id': '库里'}, 'missing_season'),
    ('库里 2023-24 总得分？', {'competition_phase': '库里'}, 'missing_phase'),
    ('库里常规赛总得分？', {'season_id': '二零二三至二零二四'}, 'missing_season'),
    ('库里大学二年级常规赛总得分？', {'season_id': '大学二年级'}, 'missing_season'),
])
def test_agent_rejects_unverified_scope_before_any_tool(tmp_path, enabled, question, evidence, reason):
    call = summary()
    call[1]['scope_evidence'] = evidence
    app, model = unified_app(tmp_path, [call], enabled=enabled, decider=FixtureDecider())
    body = request(app, '/agent/answer', {'question': question, 'routing_mode': 'model'}).json()
    assert body['status'] == 'clarification_required', body
    assert body['reason_code'] == reason
    assert body['tool_calls'] == 0
    assert not body['steps']
    assert len(model.requests) == 1


@pytest.mark.parametrize('question,key,quote', [
    ('库里常规赛总得分？', 'season_id', '库里'),
    ('库里2023-24总得分？', 'competition_phase', '库里'),
])
def test_invalid_scope_produces_no_executable_requirements(tmp_path, question, key, quote):
    from tests.test_ask import make_ask_database
    database = tmp_path / 'scope.sqlite'
    make_ask_database(database)
    call = summary()
    call[1]['scope_evidence'] = {key: quote}
    result, _ = plan(database, question, [call])
    assert result.plan.route == 'clarification'
    assert result.plan.steps == []


@pytest.mark.parametrize('key,quote,proposed,accepted', [
    ('season_id', '2023 至 2024', '2023-24', True),
    ('season_id', '二零二三至二零二四', '2023-24', True),
    ('season_id', '2023-24', '2022-23', False),
    ('season_id', '2022-23 和 2023-24', '2023-24', False),
    ('season_id', '二零二三至二零二五', '2023-24', False),
    ('season_id', '库里', '2023-24', False),
    ('competition_phase', 'Regular Season', 'regular_season', True),
    ('competition_phase', '季后赛', 'playoffs', True),
    ('competition_phase', '季后赛', 'regular_season', False),
    ('competition_phase', '常规赛与季后赛', 'regular_season', False),
    ('competition_phase', '库里', 'regular_season', False),
])
def test_scope_fallback_requires_evidence_for_the_value(key, quote, proposed, accepted):
    from app.semantic_contracts import SeasonSummary
    from app.semantic_planner import PlanError, _scope
    from scripts.query_context import QueryLinks
    args = summary()[1]
    args[key] = proposed
    args['scope_evidence'] = {key: quote}
    # Isolate the fallback from independently grounded slots.
    links = QueryLinks(seasons=[] if key == 'season_id' else ['2023-24'],
                       phases=['regular_season'] if key == 'season_id' else [])
    contract = SeasonSummary.model_validate(args)
    if accepted:
        assert _scope(contract, links, f'库里 {quote} 总得分？', None)[key] == proposed
    else:
        with pytest.raises(PlanError) as error:
            _scope(contract, links, f'库里 {quote} 总得分？', None)
        assert error.value.code == ('missing_season' if key == 'season_id' else 'missing_phase')


@pytest.mark.parametrize('question,expected', [
    ('二零二三至二零二四赛季', '2023-24赛季'),
    ('二〇二三—二〇二四赛季', '2023-24赛季'),
    ('一九九七年到一九九八年赛季', '1997-98赛季'),
    ('二零二二/二三', '2022-23'),
    ('2023 至 2024 赛季', '2023-24 赛季'),
    ('二〇二三-2024赛季', '2023-24赛季'),
    ('2023-二四赛季', '2023-24赛季'),
    ('库里 2023-24 截至2024-01-01', '库里 2023-24 截至2024-01-01'),
    ('1998年5月30日', '1998年5月30日'),
    ('大学二年级的赛季', '大学二年级的赛季'),
])
def test_shared_season_normalization_is_general_and_idempotent(question, expected):
    assert normalize_question_seasons(question) == expected
    assert normalize_question_seasons(expected) == expected


@pytest.mark.parametrize('question', [
    '二零二三至二零二五赛季', '一九九七至九九赛季', '二零二三-二零二三',
    '2023-202', '2020-2022',
])
def test_nonconsecutive_or_malformed_years_are_never_truncated(question):
    with pytest.raises(ValueError):
        normalize_question_seasons(question)


@pytest.mark.parametrize('path', ['/ask', '/answer', '/agent/answer', '/agent/loop'])
def test_native_endpoints_accept_chinese_season_and_preserve_oracle(tmp_path, path):
    calls = [summary()]
    if path == '/agent/loop':
        calls += [('season_summary', {'requirement_id': 's1'})]
    app = native_app(tmp_path, calls, agent_loop_enabled=True)
    body = request(app, path, {'question': '库里二〇二三至二〇二四赛季常规赛总得分？',
                               'routing_mode': 'model'}).json()
    assert body['status'] == 'complete', body
    if path in {'/ask', '/answer'}:
        result = body['tool_result']
    elif path == '/agent/answer':
        result = body['steps'][0]['result']
    else:
        result = body['observations'][0]['result']
    with sqlite3.connect(tmp_path / 'integration.sqlite') as connection:
        points = connection.execute(
            "SELECT p.points FROM player_game_stat p JOIN game g USING (game_id) "
            "WHERE p.person_id='201939' AND g.season_id='2023-24' "
            "AND g.competition_phase='regular_season' "
            "AND p.appearance_status IN ('played','played_inferred')").fetchall()
    assert result['points_total'] == sum(row[0] for row in points) == 50


@pytest.mark.parametrize('question,changes,reason', [
    ('库里二零二三至二零二四常规赛总得分？', {'season_id': '2022-23'}, 'scope_conflict'),
    ('库里二零二三至二零二四常规赛与季后赛总得分？', {}, 'ambiguous_scope'),
    ('库里二零二二至二零二三与二零二三至二零二四常规赛总得分？', {}, 'ambiguous_scope'),
    ('库里二零二三至二零二五常规赛总得分？', {}, 'invalid_season'),
])
def test_agent_cannot_cherry_pick_or_mistranslate_explicit_scope(tmp_path, question, changes, reason):
    call = summary()
    call[1].update(changes)
    call[1]['scope_evidence'] = {'season_id': '二零二三至二零二四'}
    app = native_app(tmp_path, [call])
    body = request(app, '/agent/answer', {'question': question}).json()
    assert body['status'] == 'clarification_required', body
    assert body['reason_code'] == reason
    assert body['tool_calls'] == 0


def test_explicit_chinese_season_overrides_previous_scope(tmp_path):
    second = summary('2022-23', {'inherit': True})
    second[1]['inherit_scope'] = ['season_id', 'competition_phase']
    app = native_app(tmp_path, [summary(), second])
    first = request(app, '/agent/answer', {'question': '库里2023-24常规赛总得分？'}).json()
    body = request(app, '/agent/answer', {'question': '那他二零二二至二零二三赛季呢？',
        'conversation_id': first['conversation_id']}).json()
    assert body['status'] == 'complete', body
    assert body['task_state']['route']['steps'][0]['arguments']['season_id'] == '2022-23'
    assert body['steps'][0]['result']['points_total'] == 10


@pytest.mark.parametrize('enabled', [False, True])
def test_composed_agent_rejects_invented_scope_before_plan_or_loop(tmp_path, enabled):
    call = composed_call()
    for item in call[1]['calls']:
        item['arguments']['scope_evidence'] = {'season_id': '库里'}
    app, model = unified_app(tmp_path, [call], enabled=enabled, decider=FixtureDecider())
    body = request(app, '/agent/answer', {
        'question': '比较库里与詹姆斯常规赛的总得分和总篮板。'}).json()
    assert body['status'] == 'clarification_required', body
    assert body['reason_code'] == 'missing_season'
    assert body['tool_calls'] == 0
    assert len(model.requests) == 1


@pytest.mark.parametrize('enabled', [False, True])
def test_composed_chinese_season_uses_same_atomic_queries(tmp_path, enabled):
    call = composed_call()
    app, _ = unified_app(tmp_path, [call,
        ('select_requirement', {'requirement_id': 's1'}),
        ('select_requirement', {'requirement_id': 's2'})], enabled=enabled)
    with sqlite3.connect(tmp_path / 'agent.sqlite') as connection:
        connection.execute('UPDATE player_game_stat SET rebounds_total=5')
        connection.execute('''INSERT INTO player_game_stat
            SELECT source_snapshot_id, source_row_id+100, game_id, '2544',
            team_id, opponent_team_id, appearance_status, minutes_quality,
            raw_comment, points+10, assists, rebounds_total+2, field_goals_made,
            three_pointers_made, field_goals_attempted, raw_record_json
            FROM player_game_stat WHERE person_id='201939' ''')
    body = request(app, '/agent/answer', {
        'question': '比较库里与詹姆斯二零二三至二零二四常规赛的总得分和总篮板。'}).json()
    assert body['status'] == 'complete', body
    assert body['execution_mode'] == ('bounded_loop' if enabled else 'bounded_plan')
    assert body['comparison']['deltas'] == {'points_total': -20, 'rebounds_total': -4}
