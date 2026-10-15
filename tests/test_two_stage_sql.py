"""Safety/semantic counterexamples for the isolated two-stage SQL experiment."""
import json
import sqlite3
from copy import deepcopy

import pytest

from scripts.two_stage_sql.contracts import AnalysisRequest
from scripts.two_stage_sql.grounding import ground_request, GroundingFailure
from scripts.two_stage_sql.sql_validation import validate_sql, SqlContractError
from scripts.two_stage_sql.oracle import reference_result, compare_result


@pytest.fixture
def db():
    c=sqlite3.connect(':memory:')
    c.executescript('''
    CREATE TABLE player_identity(person_id TEXT,display_name TEXT);
    CREATE TABLE team_identity(team_id TEXT,display_name TEXT);
    INSERT INTO player_identity VALUES('201939','Stephen Curry'),('2544','LeBron James'),('1','Kawhi Leonard'),('2','Meyers Leonard');
    INSERT INTO team_identity VALUES('1610612747','Los Angeles Lakers');
    CREATE TABLE game(game_id TEXT,game_date TEXT,season_id TEXT,competition_phase TEXT,home_team_id TEXT,away_team_id TEXT);
    INSERT INTO game VALUES('g1','2024-01-01','2023-24','regular_season','w','l'),('g2','2024-01-02','2023-24','regular_season','w','l'),('g3','2024-01-03','2023-24','regular_season','w','l');
    CREATE TABLE player_game_stat(person_id TEXT,game_id TEXT,team_id TEXT,opponent_team_id TEXT,appearance_status TEXT,points INTEGER,assists INTEGER,rebounds_total INTEGER,field_goals_made INTEGER,three_pointers_made INTEGER,field_goals_attempted INTEGER);
    INSERT INTO player_game_stat VALUES('201939','g1','w','1610612747','played',30,5,4,10,5,20),('201939','g2','w','1610612747','played',31,6,5,11,6,21),('201939','g3','w','x','not_played',0,0,0,0,0,0),('2544','g1','l','x','played',40,8,9,12,4,22);
    CREATE VIEW player_season_stat AS SELECT p.person_id,g.season_id,g.competition_phase,COUNT(*) games_appeared,SUM(points) points_total,SUM(assists) assists_total,SUM(rebounds_total) rebounds_total FROM player_game_stat p JOIN game g ON g.game_id=p.game_id WHERE appearance_status IN ('played','played_inferred') GROUP BY p.person_id,g.season_id,g.competition_phase;
    ''')
    yield c
    c.close()


def proposal(**kw):
    return AnalysisRequest.model_validate({'operation':'summary','subject':'player','entity':{'mention':'库里'},'season_id':'2023-24','competition_phase':'regular_season','metrics':['points_total'],'evidence':{'season_id':'2023-24','competition_phase':'常规赛'},**kw})


def verified(db,**kw):
    return ground_request(db,'库里 2023-24 常规赛总得分',proposal(**kw))


def test_malformed_or_injected_fields_rejected():
    with pytest.raises(ValueError):proposal(sql='DROP TABLE game')
    with pytest.raises(ValueError):proposal(limit=21)


def test_unknown_translation_is_database_checked(db):
    p=proposal(entity={'mention':'斯蒂芬·库里','canonical_name':'Stephen Curry'})
    r=ground_request(db,'斯蒂芬·库里 2023-24 常规赛总得分',p)
    assert r.entity_id=='201939'
    with pytest.raises(GroundingFailure):ground_request(db,'库里 2023-24 常规赛总得分',proposal(entity={'mention':'库里','canonical_name':'LeBron James'}))


def test_ambiguous_entity_not_resolved_by_model_guess(db):
    with pytest.raises(GroundingFailure):
        ground_request(db,'Leonard 2023-24 常规赛总得分',proposal(entity={'mention':'Leonard','canonical_name':'Kawhi Leonard'}))


def test_unrelated_quote_cannot_back_season(db):
    with pytest.raises(GroundingFailure):
        ground_request(db,'库里常规赛总得分',proposal(evidence={'season_id':'库里','competition_phase':'常规赛'}))


def test_recognized_filter_or_metric_cannot_disappear(db):
    with pytest.raises(GroundingFailure):ground_request(db,'库里 2023-24 常规赛截至2024-01-01总得分',proposal())
    with pytest.raises(GroundingFailure):ground_request(db,'库里 2023-24 常规赛总得分和总助攻',proposal())


def test_valid_mart_and_raw_queries(db):
    r=verified(db)
    for sql in ["SELECT points_total FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24' AND competition_phase='regular_season'", "SELECT SUM(p.points) AS points_total FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred')"]:
        validate_sql(sql,r)


@pytest.mark.parametrize('sql',[
 "SELECT points_total FROM player_season_stat WHERE person_id='2544' AND season_id='2023-24' AND competition_phase='regular_season'",
 "SELECT points_total FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24'",
 "SELECT points_total FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24' AND competition_phase='regular_season' AND points_total>0",
 "SELECT points_total,1 AS extra FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24' AND competition_phase='regular_season'",
 "SELECT points_total FROM player_season_stat WHERE person_id='201939' OR season_id='2023-24'",
 "DELETE FROM player_identity",
 "SELECT points_total FROM player_season_stat; DROP TABLE game",
 "WITH x AS (SELECT * FROM player_season_stat) SELECT * FROM x",
 "SELECT SUM(p.points) AS points_total FROM player_game_stat p JOIN game g ON p.person_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred')",
])
def test_wrong_scope_shape_or_unsafe_sql_rejected(db,sql):
    with pytest.raises(SqlContractError):validate_sql(sql,verified(db))


def test_integer_division_is_not_an_approved_average(db):
    p=proposal(metrics=['points_per_appearance'])
    r=ground_request(db,'库里 2023-24 常规赛场均得分',p)
    base=" FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24' AND competition_phase='regular_season'"
    with pytest.raises(SqlContractError):validate_sql('SELECT points_total/games_appeared AS points_per_appearance'+base,r)
    validate_sql('SELECT 1.0*points_total/NULLIF(games_appeared,0) AS points_per_appearance'+base,r)


def test_threshold_boundary_and_ranking_order(db):
    p=proposal(operation='game_count',metrics=['qualifying_games'],predicates=[{'field':'points','operator':'ge','value':30,'quote':'得分至少30分'}])
    r=ground_request(db,'库里 2023-24 常规赛单场得分至少30分的场数',p)
    base="SELECT COUNT(*) AS qualifying_games FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred') AND p.points"
    validate_sql(base+'>=30',r)
    with pytest.raises(SqlContractError):validate_sql(base+'>30',r)
    assert reference_result(db,r)['rows']==[[2]]


def test_oracle_rejects_extra_rows_extra_fields_and_wrong_values(db):
    ref=reference_result(db,verified(db))
    assert ref['rows']==[[61]]
    assert compare_result(['points_total'],[[61]],ref)
    for cols,rows in [(['points_total'],[[61],[1]]),(['points_total','wrong'],[[61,0]]),(['points_total'],[[60]]),(['points_total','points_total'],[[61,61]])]:
        assert not compare_result(cols,rows,ref)


def test_null_inputs_are_not_silently_ignored(db):
    db.execute("UPDATE player_game_stat SET points=NULL WHERE game_id='g1' AND person_id='201939'")
    assert reference_result(db,verified(db))['status']=='not_computable'


def test_ranking_requires_tie_break_and_limit(db):
    p=proposal(operation='rank',entity=None,limit=2,evidence={'season_id':'2023-24','competition_phase':'常规赛','limit':'前2名','sort_direction':'最高'})
    r=ground_request(db,'2023-24 常规赛总得分最高的前2名',p)
    sql="SELECT person_id AS entity_id,points_total FROM player_season_stat WHERE season_id='2023-24' AND competition_phase='regular_season' ORDER BY points_total DESC,entity_id ASC LIMIT 2"
    validate_sql(sql,r)
    with pytest.raises(SqlContractError):validate_sql(sql.replace(',entity_id ASC',''),r)
    assert reference_result(db,r)['rows']==[['201939',61],['2544',40]]


def test_threshold_quote_cannot_change_inclusive_boundary(db):
    p=proposal(operation='game_count',metrics=['qualifying_games'],predicates=[{'field':'points','operator':'gt','value':30,'quote':'得分至少30分'}])
    with pytest.raises(GroundingFailure):ground_request(db,'库里 2023-24 常规赛单场得分至少30分的场数',p)


def test_latest_window_is_rows_then_aggregation(db):
    p=proposal(operation='window',last_n=1)
    r=ground_request(db,'库里 2023-24 常规赛最近1场总得分',p)
    sql="SELECT p.game_id,g.game_date,p.points FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred') ORDER BY g.game_date DESC,p.game_id DESC LIMIT 1"
    validate_sql(sql,r)
    assert reference_result(db,r)['rows']==[[31]]
    with pytest.raises(SqlContractError):validate_sql(sql.replace('LIMIT 1','LIMIT 2'),r)


def test_runtime_missing_input_gate_is_independent_of_oracle(db):
    from scripts.two_stage_sql.runtime import input_evidence
    assert input_evidence(db,verified(db))['status']=='available'
    db.execute("UPDATE player_game_stat SET points=NULL WHERE game_id='g1' AND person_id='201939'")
    assert input_evidence(db,verified(db))['status']=='not_computable'


def test_runtime_rejects_before_execution_and_checks_full_projection(db):
    from scripts.two_stage_sql.runtime import execute_sql
    r=verified(db)
    result=execute_sql(db,r,"SELECT * FROM player_identity")
    assert result['status']=='rejected' and result['stage']=='sql_contract'


def test_scorer_does_not_use_oracle_to_repair_candidate(db):
    from scripts.two_stage_sql.evaluation import score_candidate
    ref=reference_result(db,verified(db))
    wrong={'status':'complete','accepted':True,'columns':['points_total'],'rows':[[60]]}
    assert not score_candidate(wrong,ref)['passed']
    assert wrong['rows']==[[60]]


def test_hybrid_is_capability_policy_never_failure_fallback(db):
    from scripts.two_stage_sql.evaluation import hybrid_arm
    r=verified(db)
    assert hybrid_arm(r)=='B'
    from dataclasses import replace
    assert hybrid_arm(replace(r,operation='rank'))=='C'
    assert hybrid_arm(None) is None
    with pytest.raises(Exception):r.entity_id='2544'


def test_runtime_success_and_zero_denominator(db,tmp_path):
    from scripts.two_stage_sql.runtime import execute_sql,input_evidence
    db.executescript("CREATE TABLE dataset_snapshot(source_name TEXT,source_version TEXT); INSERT INTO dataset_snapshot VALUES('test','1');")
    matrix=tmp_path/'matrix.json'
    matrix.write_text(json.dumps({'source_snapshot':'test@1','periods':{'2023-24|regular_season':{'subjects':{'player':{'metrics':{'points':'validated'}}}}}}))
    r=verified(db)
    sql="SELECT points_total FROM player_season_stat WHERE person_id='201939' AND season_id='2023-24' AND competition_phase='regular_season'"
    result=execute_sql(db,r,sql,matrix_path=matrix)
    assert result['accepted'] and result['rows']==[[61]]
    from dataclasses import replace
    db.execute("PRAGMA query_only=OFF")
    db.execute('UPDATE player_game_stat SET field_goals_attempted=0')
    assert input_evidence(db,replace(r,operation='efg',metrics=('efg_percentage',)))['reason']=='zero_denominator'
    assert not execute_sql(db,replace(r,season_id='1900-01'),sql,matrix_path=matrix)['accepted']


def test_frozen_manifest_counts_and_gold_is_not_in_prompts():
    from scripts.two_stage_sql.cases import read_cases
    from scripts.two_stage_sql.prompts import semantic_prompt, sql_prompt
    manifest=read_cases()
    assert len(manifest['cases'])==36
    assert sum(c['split']=='development' for c in manifest['cases'])==12
    for case in manifest['cases']:
        assert 'gold' in case
    text=semantic_prompt('test question',{'players':[]})
    assert 'gold' not in text and 'oracle' not in text


def test_agent_adapter_scores_verified_arguments(db):
    from scripts.two_stage_sql.evaluation import adapt_agent, requirements_equal
    r=verified(db)
    body={'status':'complete','steps':[{'capability':'season_summary','arguments':{'subject':'player','entity_id':'201939','season_id':'2023-24','competition_phase':'regular_season','metrics':['points_total']},'result':{'status':'complete','points_total':61,'quality_status':'conditional'}}]}
    candidate,request=adapt_agent(body)
    assert candidate['rows']==[[61]] and requirements_equal(request,r)
    body['steps'][0]['arguments']['entity_id']='2544'
    assert not requirements_equal(adapt_agent(body)[1],r)


def test_raw_window_identity_aliases_allowed_but_renamed_fields_rejected(db):
    r=ground_request(db,'库里2023-24常规赛最近1场总得分',proposal(operation='window',last_n=1))
    sql="SELECT p.game_id AS game_id,g.game_date AS game_date,p.points AS points FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred') ORDER BY g.game_date DESC,p.game_id DESC LIMIT 1"
    validate_sql(sql,r)
    with pytest.raises(SqlContractError):validate_sql(sql.replace('p.points AS points','p.assists AS points'),r)


@pytest.mark.parametrize('identity',["201939 "," 201939"])
def test_sql_literal_whitespace_cannot_change_verified_identity(db,identity):
    r=ground_request(db,'库里2023-24常规赛出场场数',proposal(metrics=['games_appeared']))
    sql="SELECT COUNT(*) AS games_appeared FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='"+identity+"' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred')"
    with pytest.raises(SqlContractError):validate_sql(sql,r)


@pytest.mark.parametrize('joining',['且','或'])
def test_all_threshold_conjuncts_preserved_and_or_rejected(db,joining):
    p=proposal(operation='game_count',metrics=['qualifying_games'],predicates=[{'field':'points','operator':'ge','value':30,'quote':'得分至少30分'}])
    with pytest.raises(GroundingFailure):ground_request(db,'库里2023-24常规赛单场得分至少30分'+joining+'助攻至少6次的场数',p)


def test_saved_fidelity_is_recomputed_from_gold():
    from scripts.two_stage_sql.evaluation import saved_fidelity
    from dataclasses import asdict
    from scripts.two_stage_sql.cases import gold_request,read_cases
    case=read_cases()['cases'][0]
    data=asdict(gold_request(case));data['entity_id']='WRONG'
    assert saved_fidelity(data,case) is False


def test_research_gate_needs_all_frozen_records():
    from scripts.evaluate_two_stage_sql import summarize
    from scripts.two_stage_sql.cases import read_cases
    rows=[]
    for case in [c for c in read_cases()['cases'] if c['split']=='frozen' and c['group']=='expanded']:
        for repeat in range(2):
            rows.append({'arm':'C','case_id':case['id'],'split':'frozen','group':'expanded','repeat':repeat,'score':{'passed':True,'accepted_wrong':False,'requirements_match':True},'candidate':{'status':'complete'},'latency_ms':1,'model_requests':[]})
    assert summarize(rows)['research_gate']['met'] is False


def test_noninteger_threshold_is_explicit_boundary_not_integer_prefix(db):
    p=proposal(operation='game_count',metrics=['qualifying_games'],predicates=[{'field':'points','operator':'ge','value':30,'quote':'得分至少30.5分'}])
    with pytest.raises(GroundingFailure):ground_request(db,'库里2023-24常规赛单场得分至少30.5分的场数',p)
