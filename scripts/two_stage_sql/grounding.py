"""Mechanical evidence checks; database identity does not prove translation meaning."""
import re
from collections import Counter
from app.entity_resolution import resolve_entity
from app.semantic_contracts import EntityReference
from scripts.ask_grounding import ground_ask_question, requested_metrics, recent_n
from scripts.query_scope import normalize_question_seasons, requested_date_bounds
from scripts.query_lexicon import load_lexicon
from scripts.two_stage_sql.contracts import GroundedRequest

class GroundingFailure(ValueError):
    def __init__(self,code):super().__init__(code);self.code=code

METRICS={'player':{'points_total','points_per_appearance','games_appeared','assists_total','rebounds_total'},'team':{'points_total','games_played','assists_total','rebounds_total'}}
OPS={'ge':r'至少|不低于|大于等于|≥|>=|at least','gt':r'超过|高于|大于|(?<![<>=])>(?!=)|more than','le':r'至多|不超过|小于等于|≤|<=|at most','lt':r'低于|少于|小于|(?<![<>=])<(?!=)|less than','eq':r'恰好|正好|等于|(?<![<>=])=(?!=)|exactly'}
FIELDS={'points':r'得分|分|points?','assists':r'助攻|assists?','rebounds_total':r'篮板|rebounds?'}

def fail(code):raise GroundingFailure(code)

def ground_request(connection,question,p):
    if p.operation in {'clarify','unsupported'}:fail(p.operation)
    try:links,normalized=ground_ask_question(connection,question)
    except ValueError as exc:raise GroundingFailure(getattr(exc,'code','invalid_scope')) from exc
    if len(links.seasons)!=1:fail('missing_or_ambiguous_season')
    if len(links.phases)!=1:fail('missing_or_ambiguous_phase')
    season=links.seasons[0];phase=links.phases[0]
    if p.season_id is None or normalize_question_seasons(p.season_id)!=season:fail('scope_conflict')
    if p.competition_phase!=phase:fail('scope_conflict')
    for key,value in [('season_id',season),('competition_phase',phase)]:
        quote=p.evidence.get(key,'')
        if not quote or quote not in question:fail('unverified_scope_evidence')
        if key=='season_id' and season not in normalize_question_seasons(quote):fail('unverified_scope_evidence')
        if key=='competition_phase' and not any(a.casefold() in quote.casefold() for a in load_lexicon()['value_domains']['competition_phase'][value]):fail('unverified_scope_evidence')
    identity=None
    if p.operation=='rank':
        if p.entity or links.players or links.teams:fail('ranking_entity_conflict')
        if p.limit is None:fail('missing_rank_limit')
        quote=p.evidence.get('limit','')
        if quote not in question or not quote or not re.search(rf'(?<!\d){p.limit}(?!\d)',quote):fail('unverified_limit')
        quote=p.evidence.get('sort_direction','')
        pattern=r'最高|最多|降序|从高到低|highest|most|descending' if p.sort_direction=='desc' else r'最低|最少|升序|从低到高|lowest|least|ascending'
        if not quote or quote not in question or not re.search(pattern,quote,re.I):fail('unverified_sort')
        if len(p.metrics)!=1 or p.metrics[0] not in {'points_total','assists_total','rebounds_total'}:fail('unsupported_rank_metric')
    else:
        if not p.entity:fail('missing_entity')
        try:identity=resolve_entity(connection,EntityReference(**p.entity.model_dump()),p.subject,question)[0].id
        except ValueError as exc:raise GroundingFailure(getattr(exc,'code','unverified_entity')) from exc
        explicit=links.players if p.subject=='player' else links.teams
        if explicit and {x.id for x in explicit}!={identity} and not p.opponent:fail('omitted_entity')
    bounds=requested_date_bounds(normalized)
    if (p.date_from,p.date_to)!=bounds:fail('date_filter_conflict')
    if p.last_n!=recent_n(normalized):fail('window_filter_conflict')
    if p.operation not in {'window','game_count'} and (any(bounds) or p.last_n):fail('operation_cannot_preserve_filter')
    if p.operation=='game_count' and (p.subject!='player' or p.last_n):fail('unsupported_count_shape')
    if p.operation=='efg' and (p.subject!='player' or p.metrics!=['efg_percentage']):fail('unsupported_efg')
    if p.operation=='game_count':
        if p.metrics!=['qualifying_games'] or not p.predicates:fail('invalid_count')
    elif p.operation!='efg':
        if not p.metrics or len(set(p.metrics))!=len(p.metrics) or set(p.metrics)-METRICS[p.subject]:fail('unsupported_metric')
        known=requested_metrics(question)
        if p.subject=='player' and 'games_played' in known:known=(known-{'games_played'})|{'games_appeared'}
        if known-set(p.metrics):fail('omitted_metric')
    if p.operation!='game_count' and p.predicates:fail('unexpected_predicate')
    # Do not silently treat a threshold question as an unfiltered summary.
    threshold=bool(re.search(r'至少|超过|不低于|至多|少于|恰好|正好|>=|<=|≥|≤|at least|more than',question,re.I))
    if threshold and p.operation!='game_count':fail('omitted_threshold')
    conditions=[]
    for pred in p.predicates:
        if pred.quote not in question or not re.search(FIELDS[pred.field],pred.quote,re.I):fail('predicate_evidence')
        if not re.search(rf'(?<!\d){pred.value}(?!\d)',pred.quote):fail('predicate_value')
        if not re.search(OPS[pred.operator],pred.quote,re.I):fail('predicate_operator')
        if pred.operator in {'gt','lt','eq'} and re.search(OPS['ge']+'|'+OPS['le'],pred.quote,re.I):fail('predicate_operator')
        conditions.append((pred.field,pred.operator,pred.value))
    if p.operation=='game_count':
        if re.search(r'或|\bor\b',question,re.I):fail('unsupported_disjunction')
        # Match the supported field/operator/number grammar independently of the
        # proposal. Unknown paraphrases still require semantic evaluation.
        obligations=[]
        for field,field_pattern in FIELDS.items():
            for op,operator_pattern in OPS.items():
                pattern=rf'(?:{field_pattern})\s*(?:{operator_pattern})\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
                for match in re.finditer(pattern,question,re.I):
                    if not match.group(1).isdigit():fail('unsupported_noninteger_threshold')
                    obligations.append((field,op,int(match.group(1))))
        if Counter(obligations)!=Counter(conditions):fail('threshold_coverage_unverified')
    opponent=None
    opponent_requested=bool(re.search(r'面对|对阵|对手|against',question,re.I))
    if bool(p.opponent)!=opponent_requested:fail('opponent_filter_conflict')
    if p.opponent:
        if p.operation!='window':fail('operation_cannot_preserve_opponent')
        try:opponent=resolve_entity(connection,EntityReference(**p.opponent.model_dump()),'team',question)[0].id
        except ValueError as exc:raise GroundingFailure(getattr(exc,'code','unverified_opponent')) from exc
    if p.operation not in {'rank'} and p.limit is not None:fail('unexpected_limit')
    return GroundedRequest(p.operation,p.subject,identity,season,phase,tuple(p.metrics),opponent,*bounds,p.last_n,p.sort_direction,p.limit,tuple(conditions))
