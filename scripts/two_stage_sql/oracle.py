"""Independent raw-fact oracle and strict complete-payload scoring.

Never called by candidate generation or the runtime acceptance gate.
"""
import math
from collections import defaultdict

OPERATORS={'ge':lambda a,b:a>=b,'gt':lambda a,b:a>b,'le':lambda a,b:a<=b,'lt':lambda a,b:a<b,'eq':lambda a,b:a==b}

def reference_result(c,r):
    player=r.subject=='player'
    table,key=('player_game_stat','person_id') if player else ('team_game_stat','team_id')
    fields=['points' if player else 'team_score','assists','rebounds_total']+(['field_goals_made','three_pointers_made','field_goals_attempted'] if player else ['NULL','NULL','NULL'])
    sql=f"SELECT p.{key},p.game_id,g.game_date,"+','.join('p.'+x if x!='NULL' else x for x in fields)+f' FROM {table} p JOIN game g ON g.game_id=p.game_id WHERE g.season_id=? AND g.competition_phase=?'
    params=[r.season_id,r.competition_phase]
    if player:sql+=" AND p.appearance_status IN ('played','played_inferred')"
    if r.entity_id:sql+=f' AND p.{key}=?';params.append(r.entity_id)
    if r.opponent_id:sql+=' AND p.opponent_team_id=?';params.append(r.opponent_id)
    for col,value,op in [('g.game_date',r.date_from,'>='),('g.game_date',r.date_to,'<=')]:
        if value:sql+=f' AND {col}{op}?';params.append(value)
    sql+=' ORDER BY g.game_date DESC,p.game_id DESC'
    rows=c.execute(sql,params).fetchall()
    if r.last_n:rows=rows[:r.last_n]
    def result(status,columns=(),values=(),reason=None):return {'status':status,'columns':list(columns),'rows':list(values),'reason':reason,'sample_count':len(rows)}
    if not rows:return result('not_computable',reason='data_unavailable')
    raw_index={'points':3,'assists':4,'rebounds_total':5}
    if r.operation=='game_count':
        if any(row[raw_index[field]] is None for row in rows for field,_,_ in r.predicates):return result('not_computable',reason='query_inputs_missing')
        count=sum(all(OPERATORS[op](row[raw_index[field]],value) for field,op,value in r.predicates) for row in rows)
        return result('complete',['qualifying_games'],[[count]])
    def metric(data,name):
        if name in {'games_appeared','games_played'}:return len(data)
        if name=='efg_percentage':
            if any(any(x is None for x in row[6:9]) for row in data):return None
            fgm,three,fga=(sum(row[k] for row in data) for k in (6,7,8))
            return 100*(fgm+.5*three)/fga if fga else None
        index={'points_total':3,'points_per_appearance':3,'assists_total':4,'rebounds_total':5}[name]
        if any(row[index] is None for row in data):return None
        total=sum(row[index] for row in data)
        return total/len(data) if name=='points_per_appearance' else total
    if r.operation=='rank':
        groups=defaultdict(list)
        for row in rows:groups[str(row[0])].append(row)
        numbers=[(identity,metric(data,r.metrics[0])) for identity,data in groups.items()]
        if any(value is None for _,value in numbers):return result('not_computable',reason='query_inputs_missing')
        numbers.sort(key=lambda x:((-x[1] if r.sort_direction=='desc' else x[1]),x[0]))
        return result('complete',['entity_id',*r.metrics],[[i,n] for i,n in numbers[:r.limit]])
    numbers=[metric(rows,name) for name in r.metrics]
    if any(value is None for value in numbers):return result('not_computable',reason='query_inputs_missing_or_zero_denominator')
    return result('complete',r.metrics,[numbers])


def compare_result(columns,rows,expected):
    if expected['status']!='complete' or len(columns)!=len(set(columns)) or set(columns)!=set(expected['columns']) or len(rows)!=len(expected['rows']):return False
    for actual,wanted in zip(rows,expected['rows']):
        if len(actual)!=len(columns):return False
        values=dict(zip(columns,actual))
        for key,target in zip(expected['columns'],wanted):
            value=values[key]
            if isinstance(target,(int,float)) and not isinstance(target,bool):
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isclose(value,target,rel_tol=1e-8,abs_tol=1e-8):return False
            elif value!=target:return False
    return True
