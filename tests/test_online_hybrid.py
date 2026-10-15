"""Online policy, isolation and scoring regressions; provider is the only fake."""
import asyncio
import json
import multiprocessing
import sqlite3
import time
from pathlib import Path

import pytest

from scripts.model_adapter import ModelResponse


@pytest.fixture
def tiny_database(tmp_path):
    path = tmp_path / 'nba.sqlite'
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE player_identity(person_id TEXT,display_name TEXT);
        CREATE TABLE team_identity(team_id TEXT,display_name TEXT);
        INSERT INTO player_identity VALUES('201939','Stephen Curry');
        CREATE TABLE game(game_id TEXT,game_date TEXT,season_id TEXT,competition_phase TEXT);
        INSERT INTO game VALUES('g1','2024-01-01','2023-24','regular_season'),('g2','2024-01-02','2023-24','regular_season');
        CREATE TABLE player_game_stat(person_id TEXT,game_id TEXT,appearance_status TEXT,points INTEGER,assists INTEGER,rebounds_total INTEGER,field_goals_made INTEGER,three_pointers_made INTEGER,field_goals_attempted INTEGER);
        INSERT INTO player_game_stat VALUES('201939','g1','played',30,5,4,10,5,20),('201939','g2','played',31,6,5,11,6,21);
        CREATE VIEW player_season_stat AS SELECT p.person_id,g.season_id,g.competition_phase,COUNT(*) games_appeared,SUM(points) points_total,SUM(assists) assists_total,SUM(rebounds_total) rebounds_total FROM player_game_stat p JOIN game g ON g.game_id=p.game_id GROUP BY p.person_id,g.season_id,g.competition_phase;
        CREATE TABLE dataset_snapshot(snapshot_id TEXT,source_name TEXT,source_version TEXT);
        INSERT INTO dataset_snapshot VALUES('fixture','eoinamoore/historical-nba-data-and-player-box-scores','515');
        """)
    matrix = tmp_path / 'matrix.json'
    matrix.write_text(json.dumps({'source_snapshot':'eoinamoore/historical-nba-data-and-player-box-scores@515','periods':{'2023-24|regular_season':{'subjects':{'player':{'metrics':{'points':'validated','assists':'validated'}}}}}}))
    return path, matrix


class ProviderFixture:
    def __init__(self, operation='summary', missing=False, omit_condition=False):
        self.operation=operation; self.missing=missing; self.omit_condition=omit_condition
        self.sql_calls=0

    def generate_tools(self, request):
        proposal={'operation':'clarify' if self.missing else self.operation,'subject':'player','entity':{'mention':'库里'},'season_id':'2023-24','competition_phase':'regular_season','metrics':['qualifying_games'] if self.operation=='game_count' else ['points_total'],'evidence':{'season_id':'2023-24','competition_phase':'常规赛'}}
        if self.operation=='game_count':
            proposal['predicates']=[{'field':'points','operator':'ge','value':30,'quote':'得分至少30分'}]
            if not self.omit_condition:proposal['predicates'].append({'field':'assists','operator':'ge','value':6,'quote':'助攻至少6次'})
        return ModelResponse('', 'fixture', tool_calls=[{'id':'1','type':'function','function':{'name':'describe_analysis','arguments':json.dumps(proposal)}}])

    def generate(self, request):
        self.sql_calls+=1
        return ModelResponse("SELECT COUNT(*) AS qualifying_games FROM player_game_stat p JOIN game g ON p.game_id=g.game_id WHERE p.person_id='201939' AND g.season_id='2023-24' AND g.competition_phase='regular_season' AND p.appearance_status IN ('played','played_inferred') AND p.points>=30 AND p.assists>=6",'fixture')


def query(path, matrix, model, question):
    from scripts.two_stage_sql.online import create_hybrid_app
    from httpx import AsyncClient, ASGITransport
    async def send():
        async with AsyncClient(transport=ASGITransport(app=create_hybrid_app(path, model, matrix_path=matrix)),base_url='http://experiment') as client:
            response=await client.post('/experiment/structured/answer',json={'question':question})
            assert response.status_code==200
            return response.json()
    return asyncio.run(send())


def test_online_tool_branch_returns_real_result_without_sql(tiny_database):
    path,matrix=tiny_database; model=ProviderFixture()
    body=query(path,matrix,model,'库里2023-24常规赛总得分')
    assert body['selected_arm']=='B'
    assert body['candidate']['accepted'] and body['candidate']['rows']==[[61]]
    assert model.sql_calls==0
    assert '61' in body['answer'] and body['grounded_request']['entity_id']=='201939'


def test_online_sql_branch_preserves_both_thresholds(tiny_database):
    path,matrix=tiny_database; model=ProviderFixture('game_count')
    body=query(path,matrix,model,'库里2023-24常规赛得分至少30分且助攻至少6次的场数')
    assert body['selected_arm']=='C'
    assert body['candidate']['rows']==[[1]] and body['candidate']['accepted']
    assert len(body['grounded_request']['predicates'])==2 and model.sql_calls==1


def test_online_missing_input_stops_before_branch(tiny_database):
    path,matrix=tiny_database; model=ProviderFixture(missing=True)
    body=query(path,matrix,model,'库里常规赛总得分')
    assert body['selected_arm'] is None and not body['candidate']['accepted']
    assert body['candidate']['status']=='clarify' and model.sql_calls==0


def test_online_tool_failure_never_becomes_sql_fallback(tiny_database):
    path,matrix=tiny_database; model=ProviderFixture()
    matrix.write_text(json.dumps({'source_snapshot':'wrong@2','periods':{}}))
    body=query(path,matrix,model,'库里2023-24常规赛总得分')
    assert body['selected_arm']=='B' and not body['candidate']['accepted']
    assert model.sql_calls==0


def test_online_omitted_threshold_is_rejected_before_sql(tiny_database):
    path,matrix=tiny_database; model=ProviderFixture('game_count',omit_condition=True)
    body=query(path,matrix,model,'库里2023-24常规赛得分至少30分且助攻至少6次的场数')
    assert not body['candidate']['accepted'] and body['selected_arm'] is None
    assert body['candidate']['reason']=='threshold_coverage_unverified' and model.sql_calls==0


def _delayed_side_effect(connection, path):
    connection.send({'ready':True})
    time.sleep(1)
    Path(path).write_text('worker survived timeout')


def test_hard_deadline_terminates_worker_instead_of_leaving_background_work(tmp_path):
    from scripts.two_stage_sql.online_worker import receive_response, RequestDeadline
    context=multiprocessing.get_context('spawn'); parent,child=context.Pipe()
    path=tmp_path/'side-effect'
    process=context.Process(target=_delayed_side_effect,args=(child,str(path)))
    process.start(); child.close()
    try:
        assert parent.poll(10) and parent.recv()['ready']
        with pytest.raises(RequestDeadline):receive_response(parent,process,.02)
        assert not process.is_alive() and not path.exists()
    finally:
        if process.is_alive():process.terminate()
        process.join(2);parent.close()


def test_online_gate_cannot_pass_partial_or_duplicated_run():
    from scripts.evaluate_online_hybrid import summarize_online, read_manifest
    manifest=read_manifest(); rows=[]
    for case in manifest['cases']:
        for repeat in range(2):
            for arm in ['A','H']:
                rows.append({'case_id':case['id'],'group':case['group'],'repeat':repeat,'arm':arm,'candidate':{'status':'complete'},'latency_ms':100,'model_requests':[],'score':{'passed':True,'accepted_wrong':False,'requirements_match':True}})
    assert summarize_online(rows,manifest)['replacement_gate']['met']
    assert not summarize_online(rows[:-1],manifest)['replacement_gate']['met']
    assert not summarize_online(rows[:-1]+[rows[0]],manifest)['replacement_gate']['met']
    rows[-1]['score']['accepted_wrong']=True
    assert not summarize_online(rows,manifest)['replacement_gate']['met']


def _partial_frame_worker(connection):
    import os
    import pickle
    import struct
    payload=pickle.dumps({'complete':True})
    connection.send({'ready':True})
    os.write(connection.fileno(),struct.pack('!i',len(payload)))
    time.sleep(.4)
    os.write(connection.fileno(),payload)


def test_deadline_covers_complete_frame_not_just_readable_header():
    from scripts.two_stage_sql.online_worker import receive_response, RequestDeadline
    context=multiprocessing.get_context('spawn'); parent,child=context.Pipe()
    process=context.Process(target=_partial_frame_worker,args=(child,))
    process.start();child.close()
    try:
        assert parent.poll(10) and parent.recv()['ready']
        with pytest.raises(RequestDeadline):receive_response(parent,process,.05)
        assert not process.is_alive()
    finally:
        if process.is_alive():process.terminate()
        process.join(2);parent.close()


def _interrupted_fixture_worker(connection,arm,db_path,sink):
    connection.send({'ready':True})
    connection.recv()
    Path(sink).write_text(json.dumps({'sequence':1,'stage':'semantic','usage':{'prompt_tokens':17,'completion_tokens':3}})+'\n')
    time.sleep(1)


def test_timeout_recovers_completed_usage_from_durable_ledger(tmp_path,monkeypatch):
    import scripts.two_stage_sql.online_worker as worker
    monkeypatch.setattr(worker,'worker_main',_interrupted_fixture_worker)
    client=worker.WorkerClient('H',tmp_path/'unused.sqlite',tmp_path)
    try:
        result=client.ask('test',timeout_s=.05)
        assert result['error']=='RequestDeadline'
        assert len(result['model_requests'])==1
        assert result['model_requests'][0]['usage']['prompt_tokens']==17
    finally:client.close()


def test_record_audit_does_not_trust_mutually_consistent_wrong_expected_and_candidate(tiny_database):
    from scripts.evaluate_online_hybrid import verify_record
    path,matrix=tiny_database
    body=query(path,matrix,ProviderFixture(),'库里2023-24常规赛总得分')
    case={'gold':{'operation':'summary','subject':'player','entity_id':'201939','season_id':'2023-24','competition_phase':'regular_season','metrics':['points_total']}}
    candidate={**body['candidate'],'rows':[[-999]]}
    row={'arm':'H','response':body,'http_status':200,'candidate':candidate,'grounded_request':body['grounded_request'],'expected':{'status':'complete','columns':['points_total'],'rows':[[-999]],'sample_count':2,'reason':None},'selected_arm':'B','score':{'passed':True,'numeric_match':True,'outcome_match':True,'requirements_match':True,'accepted_wrong':False}}
    with sqlite3.connect(path) as connection:
        with pytest.raises(ValueError,match='oracle|response'):verify_record(row,case,connection)
