"""Candidate execution and data evidence; deliberately independent of gold labels."""
import sqlite3,time
from app.ask_tools import execute_registered_tool
from scripts.sql_gateway import SqlGateway
from scripts.query_quality import QualityScope,assess_scope,read_database_snapshot
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.two_stage_sql.sql_validation import validate_sql,SqlContractError


def tool_arguments(r):
    if r.operation not in {'summary','window','efg'}:return None
    args={'entity_id':r.entity_id,'season_id':r.season_id,'competition_phase':r.competition_phase}
    if r.operation!='efg':args.update(subject=r.subject,metrics=list(r.metrics))
    if r.operation=='window':
        args.update({k:v for k,v in [('opponent_team_id',r.opponent_id),('date_from',r.date_from),('date_to',r.date_to),('last_n',r.last_n)] if v is not None})
    return {'tool':{'summary':'season_summary','window':'window_query','efg':'player_efg'}[r.operation],'arguments':args}


def input_evidence(c,r):
    """Inspect required raw inputs, never compute the candidate's requested answer."""
    player=r.subject=='player';table,key=('player_game_stat','person_id') if player else ('team_game_stat','team_id')
    mapping={'points_total':'points' if player else 'team_score','points_per_appearance':'points','assists_total':'assists','rebounds_total':'rebounds_total'}
    columns={mapping[m] for m in r.metrics if m in mapping}|{f for f,_,_ in r.predicates}
    if r.operation=='efg':columns|={'field_goals_made','three_pointers_made','field_goals_attempted'}
    fields=sorted(columns)
    sql='SELECT '+(','.join('p.'+f for f in fields) if fields else 'p.game_id')+f' FROM {table} p JOIN game g ON p.game_id=g.game_id WHERE g.season_id=? AND g.competition_phase=?'
    params=[r.season_id,r.competition_phase]
    if player:sql+=" AND p.appearance_status IN ('played','played_inferred')"
    if r.entity_id:sql+=f' AND p.{key}=?';params.append(r.entity_id)
    if r.opponent_id:sql+=' AND p.opponent_team_id=?';params.append(r.opponent_id)
    if r.date_from:sql+=' AND g.game_date>=?';params.append(r.date_from)
    if r.date_to:sql+=' AND g.game_date<=?';params.append(r.date_to)
    if r.last_n:sql+=' ORDER BY g.game_date DESC,p.game_id DESC LIMIT ?';params.append(r.last_n)
    rows=c.execute(sql,params).fetchall()
    reason='data_unavailable' if not rows else 'query_inputs_missing' if fields and any(any(v is None for v in row) for row in rows) else None
    if not reason and r.operation=='efg' and not any(row[fields.index('field_goals_attempted')] for row in rows):reason='zero_denominator'
    return {'status':'not_computable' if reason else 'available','reason':reason,'sample_count':len(rows),'checked_input_columns':fields}


def execute_tool(c,r,matrix_path=CAPABILITY_MATRIX):
    selection=tool_arguments(r)
    if selection is None:return {'status':'unsupported','accepted':False,'reason':'outside_existing_tools','stage':'capability'}
    deadline=time.monotonic()+2;c.set_progress_handler(lambda:int(time.monotonic()>=deadline),1000)
    try:
        body=execute_registered_tool(c,**selection,matrix_path=matrix_path,max_seconds=2)
        status=body.get('status');values=body.get('values') or body
        return {'status':status,'accepted':status=='complete' and body.get('quality_status') in {'validated','conditional'},'columns':list(r.metrics),'rows':[[values.get(m) for m in r.metrics]] if status=='complete' else [],'quality_status':body.get('quality_status'),'limitations':body.get('limitations',[]),'reason':body.get('reason_code'),'result':body}
    except (ValueError,sqlite3.Error) as exc:return {'status':getattr(exc,'status','incomplete'),'accepted':False,'stage':'execution','reason':getattr(exc,'code',type(exc).__name__)}
    finally:c.set_progress_handler(None,0)


def execute_sql(c,r,sql,matrix_path=CAPABILITY_MATRIX):
    try:contract=validate_sql(sql,r)
    except (ValueError,TypeError) as exc:return {'status':'rejected','accepted':False,'stage':'sql_contract','reason':str(exc)}
    deadline=time.monotonic()+2;c.set_progress_handler(lambda:int(time.monotonic()>=deadline),1000)
    try:
        inputs=input_evidence(c,r)
        if inputs['status']!='available':return {'status':'not_computable','accepted':False,'stage':'input_evidence','reason':inputs['reason'],'input_evidence':inputs}
        capabilities=tuple(sorted({'points' if m in {'points_total','points_per_appearance','games_appeared','games_played','qualifying_games'} else 'assists' if m=='assists_total' else 'rebounds' if m=='rebounds_total' else 'efg_percentage' for m in r.metrics}|{('rebounds' if f=='rebounds_total' else f) for f,_,_ in r.predicates}))
        snapshot=read_database_snapshot(c)
        assessment=assess_scope(QualityScope(r.season_id,r.competition_phase,r.subject,capabilities,opponent_filter=bool(r.opponent_id),team_participation=r.subject=='team'),matrix_path=matrix_path,source_snapshot=snapshot or '')
        if assessment.status not in {'validated','conditional'}:return {'status':'rejected','accepted':False,'stage':'data_quality','reason':'quality_not_verified','quality_status':assessment.status,'limitations':list(assessment.limitations)}
        result=SqlGateway(c,max_seconds=max(.001,deadline-time.monotonic()),max_rows=100).execute(sql)
        if result.status.value!='ok':return {'status':result.status.value,'accepted':False,'stage':'gateway','reason':result.reason}
        columns=result.columns;rows=[list(x) for x in result.rows]
        if r.last_n:
            values=[]
            for metric in r.metrics:
                if metric in {'games_appeared','games_played'}:values.append(len(rows));continue
                field={'points_total':'points' if r.subject=='player' else 'team_score','points_per_appearance':'points','assists_total':'assists','rebounds_total':'rebounds_total'}[metric]
                cells=[row[columns.index(field)] for row in rows]
                if not cells or any(v is None for v in cells):return {'status':'not_computable','accepted':False,'stage':'result','reason':'empty_or_missing_window_inputs'}
                value=sum(cells);values.append(value/len(cells) if metric=='points_per_appearance' else value)
            columns=list(r.metrics);rows=[values]
        if not rows or any(any(v is None for v in row) for row in rows):return {'status':'not_computable','accepted':False,'stage':'result','reason':'empty_or_null'}
        return {'status':'complete','accepted':True,'columns':columns,'rows':rows,'sql':sql,'contract':contract,'input_evidence':inputs,'quality_status':assessment.status,'limitations':list(assessment.limitations),'source_snapshot':snapshot}
    except (sqlite3.Error,ValueError) as exc:return {'status':'incomplete','accepted':False,'stage':'execution','reason':type(exc).__name__}
    finally:c.set_progress_handler(None,0)
