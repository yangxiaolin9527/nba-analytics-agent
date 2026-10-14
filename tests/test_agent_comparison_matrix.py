"""Cardinality-independent comparison contracts and verified result projection."""
import copy
import sqlite3

import pytest

from tests.test_agent_composition import composed_call
from tests.test_ask import make_ask_database
from tests.test_semantic_routing import plan, request
from tests.test_unified_agent import unified_app


TARGETS = [('库里', '201939', 'Stephen Curry'),
           ('詹姆斯', '2544', 'LeBron James'),
           ('欧文', '202681', 'Kyrie Irving'),
           ('哈登', '201935', 'James Harden')]
METRICS = ['points_total', 'rebounds_total', 'assists_total']


def matrix_call(count):
    args = composed_call()[1]['calls'][0]['arguments']
    args['metrics'] = METRICS.copy()
    calls = []
    for mention, _, canonical in TARGETS[:count]:
        item = copy.deepcopy(args)
        item['entity'] = {'mention': mention, 'canonical_name': canonical}
        calls.append({'capability': 'season_summary', 'arguments': item})
    return 'composed_workflow', {'mode': 'compare', 'calls': calls}


def seed_targets(database):
    with sqlite3.connect(database) as db:
        db.executemany('INSERT OR IGNORE INTO player_identity VALUES (?, ?, ?)',
                       [(key, name, 'fixture') for _, key, name in TARGETS])
        db.execute("UPDATE player_game_stat SET rebounds_total=5, assists=3 WHERE person_id='201939'")
        for i, (_, key, _) in enumerate(TARGETS[1:], 1):
            db.execute('''INSERT INTO player_game_stat
                SELECT source_snapshot_id, source_row_id + ?, game_id, ?,
                       team_id, opponent_team_id, appearance_status, minutes_quality,
                       raw_comment, points + ?, assists + ?, rebounds_total + ?,
                       6, 1, 16, raw_record_json
                FROM player_game_stat WHERE person_id='201939' AND game_id!='22200001' ''',
                       (i * 100, key, i * 10, i, i * 2))


def matrix_question(count):
    return f"比较{'、'.join(t[0] for t in TARGETS[:count])}在2023-24常规赛的总得分、总篮板和总助攻。"


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('enabled', [False, True])
def test_compare_each_target_reuses_same_tool_and_preserves_matrix_order(tmp_path, count, enabled):
    selectors = [('select_requirement', {'requirement_id': f's{i}'})
                 for i in range(count, 0, -1)]
    app, model = unified_app(tmp_path, [matrix_call(count), *selectors], enabled=enabled)
    seed_targets(tmp_path / 'agent.sqlite')
    body = request(app, '/agent/answer', {'question': matrix_question(count)}).json()
    assert body['status'] == 'complete', body
    assert body['tool_calls'] == count
    assert [s['capability'] for s in body['steps']] == ['season_summary'] * count
    comparison = body['comparison']
    assert comparison['step_ids'] == [f's{i}' for i in range(1, count + 1)]
    assert [r['entity_id'] for r in comparison['rows']] == [t[1] for t in TARGETS[:count]]
    for i, row in enumerate(comparison['rows']):
        assert row['values'] == {'points_total': 50 + i * 20,
                                 'rebounds_total': 10 + i * 4,
                                 'assists_total': 6 + i * 2}
        assert row['name'] in body['answer']
    if count > 2:
        assert comparison['kind'] == 'matrix'
        assert comparison['deltas'] == {} and comparison['formula'] is None
    else:
        assert comparison['deltas'] == {'points_total': -20, 'rebounds_total': -4, 'assists_total': -2}
    assert len(model.requests) == (count + 1 if enabled else 1)


@pytest.mark.parametrize('count', [3, 4])
def test_missing_or_partial_target_never_produces_complete_comparison(count):
    from app.agent_composition import summarize_comparison
    ids = [f's{i}' for i in range(count)]
    entries = {key: {'capability': 'season_summary', 'arguments': {
        'entity_id': key, 'subject': 'player', 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'metrics': ['points_total']},
        'status': 'complete', 'result': {'points_total': 10}} for key in ids}
    entries[ids[-1]]['status'] = 'partial'
    assert summarize_comparison(ids, entries) is None
    del entries[ids[-1]]
    assert summarize_comparison(ids, entries) is None


def test_comparison_contract_does_not_accept_entity_specific_rewritten_subquestions(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    for c, name in zip(call[1]['calls'], ('库里', '詹姆斯')):
        c['arguments']['metrics'] = ['points_total']
        c['subquestion'] = name + '总得分'
    result, _ = plan(db, '比较库里总得分和詹姆斯总得分，2023-24常规赛', [call, call])
    assert result.error_code == 'invalid_arguments', result


@pytest.mark.parametrize('mutation,reason', [
    ('omit', 'composition_entity_coverage'),
    ('reverse', 'composition_direction'),
    ('duplicate', 'composition_duplicate'),
])
def test_matrix_plan_checks_every_target_not_just_first_pair(tmp_path, mutation, reason):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    seed_targets(db)
    call = matrix_call(3)
    calls = call[1]['calls']
    if mutation == 'omit':
        del calls[-1]
    elif mutation == 'reverse':
        calls[1:] = reversed(calls[1:])
    else:
        calls[-1] = copy.deepcopy(calls[0])
    result, _ = plan(db, matrix_question(3), [call, call])
    assert result.error_code == reason, result


def test_comparison_cannot_expand_target_specific_metrics_to_every_entity(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    result, _ = plan(db, '比较库里总得分和詹姆斯总篮板，2023-24常规赛', [call, call])
    assert result.plan.route == 'clarification', result
    assert result.plan.reason_code == 'comparison_metric_binding'


@pytest.mark.parametrize('question', [
    '比较库里、詹姆斯和欧文 2023-24赛季常规赛的总得分、总篮板和总助攻。',
    '比较总得分、总篮板和总助攻：库里、詹姆斯和欧文，范围是2023-24赛季常规赛。',
    '比较库里、詹姆斯和欧文的总得分、总篮板和总助攻，2023-24常规赛。',
])
def test_unresolved_list_member_cannot_be_omitted_for_different_scope_positions(tmp_path, question):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    seed_targets(db)
    call = matrix_call(2)
    result, _ = plan(db, question, [call, call])
    assert result.error_code == 'composition_entity_coverage', result


def test_comparison_order_comes_from_target_list_not_background_mentions(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    seed_targets(db)
    question = '库里是之前提到的球员。比较詹姆斯和库里在2023-24常规赛的总得分、总篮板和总助攻。'
    wrong = matrix_call(2)
    result, _ = plan(db, question, [wrong, wrong])
    assert result.error_code == 'composition_direction', result
    correct = matrix_call(2)
    correct[1]['calls'].reverse()
    result, _ = plan(db, question, [correct])
    assert result.plan.route == 'composed', result
