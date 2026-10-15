"""Independent outcome scoring; never modifies a candidate."""
from scripts.two_stage_sql.oracle import compare_result

def score_candidate(candidate,expected,*,requirements_match=True):
    status=expected['status']
    numeric=compare_result(candidate.get('columns',[]),candidate.get('rows',[]),expected) if status=='complete' else None
    matched=(candidate.get('status')=='complete' and candidate.get('accepted') is True and numeric is True) if status=='complete' else candidate.get('status')==status and not candidate.get('accepted')
    return {'passed':bool(matched and requirements_match),'numeric_match':numeric,'outcome_match':bool(matched),'requirements_match':requirements_match,'accepted_wrong':bool(candidate.get('accepted') and (numeric is not True or not requirements_match))}


def hybrid_arm(request):
    if request is None:return None
    return 'C' if request.operation in {'rank','game_count'} else 'B'


def requirements_equal(actual,wanted):
    from dataclasses import asdict
    if actual is None or wanted is None:return actual is wanted
    left,right=asdict(actual),asdict(wanted)
    for key in ('metrics','predicates'):
        left[key]=sorted(left[key]);right[key]=sorted(right[key])
    return left==right


def adapt_agent(body):
    """Read executed arguments, never infer a successful plan from its answer text."""
    from scripts.two_stage_sql.contracts import GroundedRequest
    status={'clarification_required':'clarify','clarification':'clarify'}.get(body.get('status'),body.get('status','incomplete'))
    if body.get('route')=='clarification':status='clarify'
    steps=body.get('steps',[])
    candidate={'status':status,'accepted':False,'columns':[],'rows':[],'reason':body.get('reason_code')}
    if len(steps)!=1:return candidate,None
    step=steps[0]
    args=step.get('arguments') or {}
    route_steps=(body.get('task_state') or {}).get('route',{}).get('steps',[])
    if not args and route_steps:args=route_steps[0].get('arguments') or {}
    observations=body.get('observations') or []
    if not args and observations:args=observations[0].get('arguments') or {}
    operation={'season_summary':'summary','window_query':'window','player_efg':'efg'}.get(step.get('capability'))
    if operation is None:return candidate,None
    try:
        metrics=tuple(args.get('metrics') or (['efg_percentage'] if operation=='efg' else []))
        request=GroundedRequest(operation,args.get('subject','player'),args['entity_id'],args['season_id'],args['competition_phase'],metrics,args.get('opponent_team_id'),args.get('date_from'),args.get('date_to'),args.get('last_n'))
    except (KeyError,TypeError):return candidate,None
    result=step.get('result') or {}
    values=result.get('values') or result
    # Data payload only; SQL/provenance keys are not extra requested metrics.
    metric_names={'points_total','assists_total','rebounds_total','points_per_appearance','games_appeared','games_played','efg_percentage'}
    columns=[m for m in values if m in metric_names]
    candidate.update(status=status,accepted=status=='complete' and result.get('quality_status') in {'validated','conditional'},columns=columns,rows=[[values[m] for m in columns]] if status=='complete' else [],quality_status=result.get('quality_status'))
    return candidate,request


def saved_fidelity(data,case):
    from scripts.two_stage_sql.cases import gold_request
    from scripts.two_stage_sql.contracts import GroundedRequest
    if not data:return 'status' in case['gold']
    if 'status' in case['gold']:return False
    actual=GroundedRequest(**{**data,'metrics':tuple(data['metrics']),'predicates':tuple(tuple(x) for x in data.get('predicates',[]))})
    return requirements_equal(actual,gold_request(case))
