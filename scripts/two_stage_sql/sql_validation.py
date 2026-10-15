"""Finite AST acceptance contract, not a general SQL equivalence prover.

This inspects generated SQL; it never generates a candidate or executes an oracle.
"""
from collections import Counter
from sqlglot import exp,parse

class SqlContractError(ValueError):pass

def require(ok,reason):
    if not ok:raise SqlContractError(reason)

def bare(node):
    node=node.copy()
    def clean(x):
        if isinstance(x,exp.Paren):return x.this
        if isinstance(x,exp.Column):return exp.column(x.name.lower())
        return x
    # Two passes remove nested parentheses without depending on table aliases.
    return node.transform(clean).transform(clean).sql(dialect='sqlite').lower().replace(' ','').replace('"','')

def forms(texts):return {bare(parse('SELECT '+s,read='sqlite')[0].expressions[0]) for s in texts}

def metric_forms(metric,mart,subject):
    points='points' if subject=='player' else 'team_score'
    total='points_total' if mart else f'SUM({points})'
    count=('games_appeared' if subject=='player' else 'games_played') if mart else 'COUNT(*)'
    if metric=='points_total':return forms([total])
    if metric in {'assists_total','rebounds_total'}:
        col='assists' if metric=='assists_total' else 'rebounds_total'
        return forms([metric if mart else f'SUM({col})'])
    if metric in {'games_appeared','games_played','qualifying_games'}:return forms([count,'COUNT(*)'] if not mart else [count])
    if metric=='points_per_appearance':
        numerators=[f'1.0*{total}',f'{total}*1.0',f'CAST({total} AS REAL)',f'{total}*1.0']
        denominators=[count,f'NULLIF({count},0)']
        return forms([f'{n}/{d}' for n in numerators for d in denominators]+([f'AVG({points})'] if not mart else []))
    if metric=='efg_percentage':
        fgm='SUM(field_goals_made)';three='SUM(three_pointers_made)';fga='SUM(field_goals_attempted)'
        return forms([f'100.0*({fgm}+0.5*{three})/{d}' for d in [fga,f'NULLIF({fga},0)']]+[f'({fgm}+0.5*{three})/{d}*100.0' for d in [fga,f'NULLIF({fga},0)']])
    return set()

def literal_value(node):
    # Preserve SQL string identity exactly: spaces and case are meaningful data.
    return ('text' if node.is_string else 'number',node.this)


def predicates(node):
    if node is None:return []
    if isinstance(node,exp.And):return predicates(node.left)+predicates(node.right)
    if isinstance(node,exp.Paren):return predicates(node.this)
    if isinstance(node,exp.In):
        require(not node.args.get('query') and all(isinstance(x,exp.Literal) for x in node.expressions),'subquery_predicate')
        return [('in',bare(node.this),tuple(sorted(literal_value(x) for x in node.expressions if isinstance(x,exp.Literal))))]
    classes={exp.EQ:'eq',exp.GTE:'ge',exp.GT:'gt',exp.LTE:'le',exp.LT:'lt'}
    for cls,op in classes.items():
        if isinstance(node,cls):
            left,right=node.left,node.right
            if isinstance(left,exp.Literal) and isinstance(right,exp.Column):left,right=right,left;op={'ge':'le','gt':'lt','le':'ge','lt':'gt','eq':'eq'}[op]
            require(isinstance(left,exp.Column) and isinstance(right,exp.Literal),'nonliteral_predicate')
            return [(op,bare(left),literal_value(right))]
    raise SqlContractError('predicate_shape_unverified')

def validate_sql(sql,r):
    try:statements=parse(sql,read='sqlite')
    except Exception as exc:raise SqlContractError('invalid_sql') from exc
    require(len(statements)==1 and isinstance(statements[0],exp.Select),'one_select_required')
    q=statements[0]
    require(len(list(q.find_all(exp.Select)))==1 and not q.args.get('with_'),'nested_query_unverified')
    require(not any(q.args.get(k) for k in ['distinct','having','offset','qualify','windows']),'query_shape_unverified')
    tables=list(q.find_all(exp.Table));names=[t.name for t in tables]
    mart_name=r.subject+'_season_stat';fact=r.subject+'_game_stat'
    mart=names==[mart_name]
    require(mart or len(names)==2 and set(names)=={fact,'game'},'table_shape_unverified')
    joins=q.args.get('joins') or []
    if mart:require(not joins,'mart_join_unverified')
    else:
        require(len(joins)==1,'one_fact_game_join_required')
        join=joins[0]
        require(not join.args.get('side') and join.args.get('kind') in {None,'INNER'},'join_type_unverified')
        on=join.args.get('on');aliases={t.alias_or_name:t.name for t in tables}
        require(isinstance(on,exp.EQ) and all(isinstance(x,exp.Column) and x.name=='game_id' for x in [on.left,on.right]),'join_key_mismatch')
        require({aliases.get(on.left.table),aliases.get(on.right.table)}=={fact,'game'},'join_table_mismatch')
    # Reject columns bound to the wrong table even if the unqualified name exists.
    aliases={t.alias_or_name:t.name for t in tables}
    for column in q.find_all(exp.Column):
        if column.table:
            table=aliases.get(column.table)
            require(table in names,'unknown_alias')
            if column.name in {'season_id','competition_phase','game_date'} and not mart:require(table=='game','scope_column_table_mismatch')
            elif column.name not in {'game_id'} and not mart:require(table==fact,'metric_column_table_mismatch')
    expected=[('eq','season_id',('text',r.season_id)),('eq','competition_phase',('text',r.competition_phase))]
    if r.entity_id:expected.append(('eq','person_id' if r.subject=='player' else 'team_id',('text',r.entity_id)))
    if not mart and r.subject=='player':expected.append(('in','appearance_status',(('text','played'),('text','played_inferred'))))
    if r.opponent_id:expected.append(('eq','opponent_team_id',('text',r.opponent_id)))
    for name,value,op in [('game_date',r.date_from,'ge'),('game_date',r.date_to,'le')]:
        if value:expected.append((op,name,('text',value)))
    expected.extend((op,field,('number',str(value))) for field,op,value in r.predicates)
    actual=predicates(q.args['where'].this if q.args.get('where') else None)
    require(Counter(actual)==Counter(expected),'filter_mismatch')
    require(not mart or r.operation in {'summary','rank'},'mart_cannot_express_filters')
    if r.last_n:
        require(not mart,'window_requires_raw_facts')
        fields={'game_id','game_date'}
        mapping={'points_total':'points' if r.subject=='player' else 'team_score','points_per_appearance':'points','assists_total':'assists','rebounds_total':'rebounds_total'}
        fields.update(mapping[m] for m in r.metrics if m in mapping)
        require({x.alias_or_name for x in q.expressions}==fields and len(q.expressions)==len(fields),'window_projection_mismatch')
        require(all(isinstance(x,exp.Column) or isinstance(x,exp.Alias) and isinstance(x.this,exp.Column) and x.alias==x.this.name for x in q.expressions),'window_requires_raw_columns')
        require(not q.args.get('group'),'window_group_unverified')
        expected_order=[('game_date',True),('game_id',True)];limit=r.last_n
    else:
        expected_cols=set(r.metrics)|({'entity_id'} if r.operation=='rank' else set())
        require(len(q.expressions)==len(expected_cols) and {x.alias_or_name for x in q.expressions}==expected_cols,'projection_mismatch')
        for projection in q.expressions:
            name=projection.alias_or_name;expr=projection.this if isinstance(projection,exp.Alias) else projection
            if name=='entity_id':require(bare(expr)==('person_id' if r.subject=='player' else 'team_id'),'ranking_entity_mismatch')
            else:require(bare(expr) in metric_forms(name,mart,r.subject),'metric_expression_unverified:'+name)
        group=q.args.get('group')
        if r.operation=='rank' and not mart:require(group is not None and [bare(x) for x in group.expressions]==['person_id' if r.subject=='player' else 'team_id'],'group_mismatch')
        else:require(group is None,'unexpected_group')
        expected_order=[(r.metrics[0],r.sort_direction=='desc'),('entity_id',False)] if r.operation=='rank' else []
        limit=r.limit if r.operation=='rank' else None
    order=q.args.get('order');actual_order=[(bare(x.this),bool(x.args.get('desc'))) for x in order.expressions] if order else []
    # Physical identity and its approved output alias are equivalent in ORDER BY.
    actual_order=[('entity_id' if name in {'person_id','team_id'} else name,desc) for name,desc in actual_order]
    require(actual_order==expected_order,'order_mismatch')
    clause=q.args.get('limit')
    require((int(clause.expression.this) if clause and isinstance(clause.expression,exp.Literal) and clause.expression.is_int else None)==limit and bool(clause)==(limit is not None),'limit_mismatch')
    return {'status':'contract_checked','boundary':'finite_ast_shapes_not_general_semantic_proof','raw_window':bool(r.last_n)}
