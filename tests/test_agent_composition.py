import copy
import sqlite3

import pytest

from tests.test_semantic_routing import plan, summary, request
from tests.test_unified_agent import unified_app
from tests.test_ask import make_ask_database


QUESTION = '比较 库里 与 詹姆斯 2023-24 常规赛的总得分和总篮板。'


def composed_call():
    left = summary()[1]
    left['metrics'] = ['points_total', 'rebounds_total']
    right = copy.deepcopy(left)
    right['entity'] = {'mention': '詹姆斯'}
    return ('composed_workflow', {'mode': 'compare', 'calls': [
        {'capability': 'season_summary', 'arguments': left},
        {'capability': 'season_summary', 'arguments': right},
    ]})


@pytest.mark.parametrize('enabled', [False, True])
def test_reuses_registered_summary_twice_and_summarizes_verified_results(tmp_path, enabled):
    app, model = unified_app(tmp_path, [composed_call(),
        ('select_requirement', {'requirement_id': 's2'}),
        ('select_requirement', {'requirement_id': 's1'})], enabled=enabled)
    with sqlite3.connect(tmp_path / 'agent.sqlite') as connection:
        connection.execute("UPDATE player_game_stat SET rebounds_total=5, assists=3 WHERE person_id='201939'")
        connection.execute('''INSERT INTO player_game_stat
            SELECT source_snapshot_id, source_row_id + 100, game_id, '2544',
                   team_id, opponent_team_id, appearance_status, minutes_quality,
                   raw_comment, points + 10, assists, rebounds_total + 2, 6, 1, 16, raw_record_json
            FROM player_game_stat WHERE person_id='201939' AND game_id!='22200001' ''')
    body = request(app, '/agent/answer', {'question': QUESTION}).json()
    assert body['status'] == 'complete', body
    assert body['route'] == 'composed'
    assert body['tool_calls'] == 2
    assert [s['capability'] for s in body['steps']] == ['season_summary'] * 2
    comparison = body['comparison']
    assert comparison['step_ids'] == ['s1', 's2']
    assert comparison['rows'][0]['entity_id'] == '201939'
    assert comparison['rows'][1]['entity_id'] == '2544'
    assert comparison['deltas']['points_total'] == -20
    assert comparison['deltas']['rebounds_total'] == -4
    assert '总得分' in body['answer'] and '总篮板' in body['answer']
    assert body['execution_mode'] == ('bounded_loop' if enabled else 'bounded_plan')


@pytest.mark.parametrize('mutation,reason', [
    ('omit_entity', 'composition_entity_coverage'),
    ('omit_metric', 'metric_conflict'),
    ('extra_metric', 'composition_metric_coverage'),
    ('different_scope', 'scope_conflict'),
    ('duplicate', 'composition_duplicate'),
    ('unknown_tool', 'invalid_arguments'),
    ('reverse_entities', 'composition_direction'),
    ('extra_metrics_both', 'composition_metric_coverage'),
    ('duplicate_metric_order', 'composition_duplicate'),
])
def test_composition_fails_closed_without_discarding_requirements(tmp_path, mutation, reason):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    calls = call[1]['calls']
    if mutation == 'omit_entity':
        call[1]['mode'] = 'collect'
        calls[1] = {'capability': 'knowledge_glossary', 'arguments': {'question': '总篮板'}}
    elif mutation == 'omit_metric':
        calls[1]['arguments']['metrics'] = ['points_total']
    elif mutation == 'extra_metric':
        calls[1]['arguments']['metrics'].append('assists_total')
    elif mutation == 'different_scope':
        calls[1]['arguments']['season_id'] = '2020-21'
    elif mutation == 'duplicate':
        calls[1] = copy.deepcopy(calls[0])
    elif mutation == 'reverse_entities':
        calls.reverse()
    elif mutation == 'extra_metrics_both':
        for item in calls:
            item['arguments']['metrics'].append('assists_total')
    elif mutation == 'duplicate_metric_order':
        call[1]['mode'] = 'collect'
        calls[1] = copy.deepcopy(calls[0])
        calls[1]['arguments']['metrics'].reverse()
    else:
        calls[1]['capability'] = 'run_sql'
    result, _ = plan(db, QUESTION, [call, call])
    assert (result.error_code or result.plan.reason_code) == reason, result
    assert not result.plan.steps


def test_generic_summary_never_compares_partial_or_missing_results():
    from app.agent_composition import summarize_comparison
    assert summarize_comparison(('s1', 's2'), {}) is None
    entries = {key: {'capability': 'season_summary', 'arguments': {
        'entity_id': key, 'subject': 'player', 'season_id': '2023-24',
        'competition_phase': 'regular_season', 'metrics': ['points_total']},
        'status': 'complete', 'result': {'points_total': 10}} for key in ('s1', 's2')}
    entries['s2']['status'] = 'partial'
    assert summarize_comparison(('s1', 's2'), entries) is None


def test_comparison_uses_metric_identity_not_model_list_order(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['calls'][1]['arguments']['metrics'].reverse()
    result, _ = plan(db, QUESTION, [call])
    assert result.plan.route == 'composed', result


def test_ask_keeps_single_query_transport_boundary(tmp_path):
    app, model = unified_app(tmp_path, [composed_call(), composed_call()])
    body = request(app, '/ask', {'question': QUESTION}).json()
    assert body['status'] != 'complete'
    assert body['tool_calls_used'] == 0


def efg_composed(*, knowledge=False):
    calls = [{'capability': 'player_efg', 'arguments': {
        'entity': {'mention': name}, 'season_id': '2023-24', 'competition_phase': 'regular_season'}}
        for name in ('库里', '詹姆斯')]
    if knowledge:
        calls.append({'capability': 'knowledge_glossary', 'arguments': {'question': '解释这个指标'}})
    return ('composed_workflow', {'mode': 'compare', 'calls': calls})


def test_explicit_knowledge_requirement_cannot_be_dropped(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = efg_composed()
    result, _ = plan(db, '比较库里与詹姆斯 2023-24 常规赛的eFG%，并解释 eFG% 是什么', [call, call])
    assert result.error_code == 'composition_knowledge_coverage'


def test_composed_knowledge_pronoun_binds_single_verified_metric(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    result, _ = plan(db, '比较库里与詹姆斯 2023-24 常规赛的eFG%，并解释这个指标', [efg_composed(knowledge=True)])
    assert result.plan.steps[-1].arguments['term'] == '解释这个指标'
    assert 'effective field goal percentage' in result.plan.steps[-1].arguments['context_metric']


def test_casefolded_entity_mentions_preserve_comparison_direction(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['calls'][0]['arguments']['entity'] = {'mention': 'LeBron James'}
    call[1]['calls'][1]['arguments']['entity'] = {'mention': 'Stephen Curry'}
    result, _ = plan(db, 'Compare STEPHEN CURRY with LEBRON JAMES in 2023-24 regular season total points and total rebounds', [call, call])
    assert result.error_code == 'composition_direction'


def test_collect_can_combine_different_registered_numeric_tools(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['mode'] = 'collect'
    left, right = call[1]['calls']
    left['arguments']['metrics'] = ['points_total']
    left['subquestion'] = '库里总得分'
    right['capability'] = 'player_efg'
    del right['arguments']['metrics']
    del right['arguments']['subject']
    right['subquestion'] = '詹姆斯eFG%'
    result, _ = plan(db, '查询 2023-24 常规赛的库里总得分与詹姆斯eFG%。', [call])
    assert result.plan.route == 'composed', result
    assert [s.capability for s in result.plan.steps] == ['season_summary', 'player_efg']


def test_knowledge_metric_is_not_misclassified_as_a_requested_numeric_query(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['mode'] = 'collect'
    call[1]['calls'] = [call[1]['calls'][0],
        {'capability': 'knowledge_glossary', 'arguments': {'question': '解释 eFG% 的定义'}}]
    call[1]['calls'][0]['arguments']['metrics'] = ['points_total']
    result, _ = plan(db, '库里 2023-24 常规赛总得分是多少，并解释 eFG% 的定义', [call])
    assert result.plan.route == 'composed', result
    assert 'context_metric' not in result.plan.steps[1].arguments


def test_subquestion_cannot_authorize_unrequested_metric_for_one_entity(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['mode'] = 'collect'
    call[1]['calls'][0]['subquestion'] = '库里总得分'
    call[1]['calls'][1]['subquestion'] = '詹姆斯总篮板'
    result, _ = plan(db, '查询 2023-24 常规赛的库里总得分和詹姆斯总篮板', [call, call])
    assert result.error_code == 'composition_subquestion_metric'


def test_entity_only_quotes_cannot_swap_entity_metric_requirements(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['mode'] = 'collect'
    for item, metric, name in zip(call[1]['calls'], ['rebounds_total', 'points_total'], ['库里', '詹姆斯']):
        item['arguments']['metrics'] = [metric]
        item['subquestion'] = name
    result, _ = plan(db, '查询 2023-24 常规赛的库里总得分和詹姆斯总篮板', [call, call])
    assert result.error_code == 'composition_subquestion_metric'


@pytest.mark.parametrize('knowledge', ['eFG%', '解释 eFG% 的定义'])
def test_knowledge_quote_must_cover_the_full_requested_clause(tmp_path, knowledge):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = efg_composed(knowledge=True)
    call[1]['calls'][-1]['arguments']['question'] = knowledge
    result, _ = plan(db, '比较库里与詹姆斯 2023-24 常规赛的eFG%，并解释 eFG% 的定义和局限性', [call, call])
    assert result.error_code == 'composition_knowledge_coverage'


def test_knowledge_quote_cannot_erase_numeric_requirements_elsewhere(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    for c in call[1]['calls']:
        c['arguments']['metrics'] = ['assists_total']
    call[1]['calls'].append({'capability': 'knowledge_glossary', 'arguments': {'question': '总得分'}})
    result, _ = plan(db, '比较库里与詹姆斯 2023-24 常规赛总得分，并解释总得分', [call, call])
    assert result.error_code == 'composition_knowledge_coverage'


def test_discontinuous_knowledge_requirements_are_clarified_not_silently_dropped(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = efg_composed(knowledge=True)
    call[1]['calls'][-1]['arguments']['question'] = '解释 eFG% 的定义，'
    result, _ = plan(db, '解释 eFG% 的定义，并比较库里与詹姆斯 2023-24 常规赛的 eFG%，并解释 eFG% 的局限性。', [call])
    assert result.plan.reason_code == 'composition_knowledge_boundary', result
    assert not result.plan.steps


def test_ambiguous_knowledge_clause_cannot_swallow_numeric_metric_evidence(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    question = '比较库里与詹姆斯 2023-24 常规赛总得分及定义。'
    call = composed_call()
    for c in call[1]['calls']:
        c['arguments']['metrics'] = ['assists_total']
    call[1]['calls'].append({'capability': 'knowledge_glossary', 'arguments': {'question': question}})
    result, _ = plan(db, question, [call])
    assert result.plan.reason_code == 'composition_knowledge_boundary', result
    assert not result.plan.steps


def test_entity_specific_subquestion_cannot_authorize_another_entity_metric(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = composed_call()
    call[1]['mode'] = 'collect'
    call[1]['calls'][0]['subquestion'] = '库里总得分和詹姆斯总篮板'
    call[1]['calls'][1]['subquestion'] = '詹姆斯总篮板'
    call[1]['calls'][1]['arguments']['metrics'] = ['rebounds_total']
    result, _ = plan(db, '查询 2023-24 常规赛的库里总得分和詹姆斯总篮板', [call, call])
    assert result.error_code == 'composition_subquestion_entity', result


def test_separate_knowledge_clause_can_precede_numeric_requests(tmp_path):
    db = tmp_path / 'plan.sqlite'
    make_ask_database(db)
    call = efg_composed(knowledge=True)
    call[1]['calls'][-1]['arguments']['question'] = '解释 eFG% 的定义'
    result, _ = plan(db, '解释 eFG% 的定义，并比较库里与詹姆斯 2023-24 常规赛的 eFG%。', [call, call])
    assert result.plan.route == 'composed', result


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('knowledge_first', [False, True])
@pytest.mark.parametrize('failed_entity', ['201939', '2544', 'both'])
def test_independent_failed_numeric_work_never_hides_remaining_work_or_reports_complete(tmp_path, enabled, knowledge_first, failed_entity):
    from tests.test_agent_routes import FakeKnowledge
    call = composed_call()
    call[1]['calls'].append({'capability': 'knowledge_glossary', 'arguments': {'question': '解释总得分'}})
    if knowledge_first:
        call[1]['calls'].insert(0, call[1]['calls'].pop())
    numeric = [(f's{i+1}', c) for i, c in enumerate(call[1]['calls']) if c['capability'] != 'knowledge_glossary']
    knowledge_id = next(f's{i+1}' for i, c in enumerate(call[1]['calls']) if c['capability'] == 'knowledge_glossary')
    app, _ = unified_app(tmp_path, [call, ('select_requirement', {'requirement_id': knowledge_id}),
        ('select_requirement', {'requirement_id': numeric[0][0]}),
        ('select_requirement', {'requirement_id': numeric[1][0]})], enabled=enabled, knowledge=FakeKnowledge())
    registry = app.state.agent_registry
    original = registry.get('season_summary')
    class FailingSummary:
        name, route = 'season_summary', 'structured'
        async def execute(self, question, arguments):
            if failed_entity == 'both' or arguments['entity_id'] == failed_entity:
                return {'status': 'failed', 'reason_code': 'data_unavailable'}
            return await original.execute(question, arguments)
    registry._adapters['season_summary'] = FailingSummary()
    # Both numerical calls can finish validation; the chosen entity fails for
    # this test. The other independent tool must still run.
    with sqlite3.connect(tmp_path / 'agent.sqlite') as db:
        db.execute('UPDATE player_game_stat SET rebounds_total=5')
        db.execute('''INSERT INTO player_game_stat
            SELECT source_snapshot_id, source_row_id + 100, game_id, '2544',
                   team_id, opponent_team_id, appearance_status, minutes_quality,
                   raw_comment, points, assists, rebounds_total, 6, 1, 16, raw_record_json
            FROM player_game_stat WHERE person_id='201939' AND game_id!='22200001' ''')
    body = request(app, '/agent/answer', {'question': QUESTION + '并解释总得分。'}).json()
    assert body['status'] == 'partial', body
    assert len(body['steps']) == 3
    assert body['comparison'] is None
    if failed_entity != 'both':
        assert any(s['capability'] == 'season_summary' and s['status'] == 'complete' for s in body['steps'])
