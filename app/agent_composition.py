"""Compile bounded compositions and synthesize comparisons from verified outputs.

This module adds no execution tools. Every numeric input comes from an existing
registered adapter, and comparison references preserve the plan's entity order.
"""
from __future__ import annotations

import json
import re

from scripts.answer_renderer import _METRIC_LABELS, _PHASES, _fmt
from scripts.tracing import mark_success, set_attribute, set_text, span
from scripts.ask_grounding import requested_metrics, _METRIC_PATTERNS
from app.semantic_contracts import MAX_PLAN_STEPS


ATOMIC_CAPABILITIES = {'season_summary', 'window_query', 'player_efg', 'knowledge_glossary'}
_KNOWLEDGE_REQUEST = re.compile(
    r'解释|定义|含义|局限|限制|代表什么|意味着|什么是|指标是什么|'
    r'\b(?:explain|define|definition|meaning|limitations)\b', re.I)


def knowledge_clause_span(question):
    """Conservative clause boundary for explicit explanation instructions.

    It identifies a source span, not a rewritten question. An unsupported or
    ambiguous decomposition is rejected instead of accepting a shorter quote.
    Shared numeric scope remains outside this text projection.
    """
    marker = _KNOWLEDGE_REQUEST.search(question)
    if marker is None:
        return None
    boundaries = list(re.finditer(r'[，,；;。!?！？]|\b(?:and|also|then)\b|并(?:且)?|以及', question[:marker.start()], re.I))
    start = boundaries[-1].end() if boundaries else 0
    next_numeric = re.search(r'(?:[，,；;。]|\b(?:and|then)\b|并(?:且)?|以及)\s*(?:同时)?\s*(?:查询|比较|对比|计算|获取|查一下|\b(?:query|compare|calculate|get)\b)',
                             question[marker.end():], re.I)
    end = marker.end() + next_numeric.start() if next_numeric else len(question)
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end-1] in '。！？?!；;,， \t\n':
        end -= 1
    return start, end


def query_fingerprint(capability, arguments):
    canonical = dict(arguments)
    if 'metrics' in canonical:
        canonical['metrics'] = sorted(canonical['metrics'])
    return json.dumps([capability, canonical], sort_keys=True, ensure_ascii=False)


def validate_composed_steps(steps, comparison_ids):
    if not 2 <= len(steps) <= MAX_PLAN_STEPS or any(s.capability not in ATOMIC_CAPABILITIES for s in steps):
        raise ValueError('composition exceeds atomic execution capacity')
    if sum(s.capability == 'knowledge_glossary' for s in steps) > 1 or all(s.capability == 'knowledge_glossary' for s in steps):
        raise ValueError('composition requires numeric work and at most one knowledge subquestion')
    fingerprints = [query_fingerprint(s.capability, s.arguments) for s in steps]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('duplicate composition step')
    if not comparison_ids:
        return
    numeric = [s for s in steps if s.capability != 'knowledge_glossary']
    if len(numeric) < 2 or comparison_ids != [s.id for s in numeric]:
        raise ValueError('comparison must reference every numeric target in order')
    def signature(step):
        args = {k: v for k, v in step.arguments.items() if k != 'entity_id'}
        if 'metrics' in args:
            args['metrics'] = sorted(args['metrics'])
        return step.capability, args
    entity_ids = [s.arguments.get('entity_id') for s in numeric]
    if (any(signature(s) != signature(numeric[0]) for s in numeric)
            or not all(entity_ids) or len(set(entity_ids)) != len(entity_ids)):
        raise ValueError('comparison capability, scope, filters and metrics must match for distinct entities')


def explicit_target_group(question):
    """Witness only an explicit delimited target list before shared scope.

    This validates source coverage even for mentions absent from the lexicon;
    it never resolves identities or routes tools. Other wording still uses the
    existing grounded entity checks rather than guessing list membership.
    """
    marker = re.search(r'比较|对比|\bcompare\b', question, re.I)
    if marker is None:
        return []
    suffix = question[marker.end():]
    # A colon introduces a separate list after a metric heading. Inspect the
    # whole declared list, including unknown names, before any model proposal.
    for clause in re.split(r'[:：；;]', suffix):
        boundary = re.search(r'在|范围(?:是|为)?|\b(?:in|during|scope)\b|\d{4}-\d{2}', clause, re.I)
        # Metric evidence and scope evidence delimit a target-list expression.
        # Reuse the same metric lexicon as grounding; names before shared
        # suffix metrics must remain visible even when scope comes last.
        metric_starts = [m.start() for pattern in _METRIC_PATTERNS.values()
                         if (m := re.search(pattern, clause, re.I))]
        ends = metric_starts + ([boundary.start()] if boundary else [])
        phrase = clause[:min(ends)] if ends else clause
        phrase = phrase.strip(' 的，,。.!?！？')
        members = [s.strip() for s in re.split(r'[、,，]|与|和|\b(?:and|with|vs\.?)\b', phrase, flags=re.I) if s.strip()]
        if len(members) > 1:
            return members
    return []


def compile_composition(contract, connection, question, links, previous, diagnostics):
    from app.agent_routes import RoutePlan, RouteStep
    from app.semantic_planner import PlanError, compile_semantic

    steps, resolved_players, resolved_teams = [], {}, {}
    explicit_players = {p.id for p in links.players}
    explicit_teams = {t.id for t in links.teams}
    fingerprints = set()
    numeric_mentions = []
    quoted_bindings = []
    numeric_question = question
    knowledge_span = knowledge_clause_span(question)
    knowledge_calls = [c for c in contract.calls if c.capability == 'knowledge_glossary']
    if knowledge_span:
        start, end = knowledge_span
        if len(knowledge_calls) != 1 or question[start:end] not in knowledge_calls[0].arguments.question:
            raise PlanError('composition_knowledge_coverage', 'Quote the complete requested knowledge clause, including all its aspects.', repairable=True)
        # Remove exactly this grounded span, never every occurrence of a word.
        numeric_question = question[:start] + question[end:]
        if (_KNOWLEDGE_REQUEST.search(numeric_question)
                or (requested_metrics(question) and not requested_metrics(numeric_question))):
            raise PlanError('composition_knowledge_boundary',
                'Cannot prove one separate contiguous knowledge clause and numeric scope. Separate the numeric and knowledge questions explicitly.', clarify=True)
    for call in contract.calls:
        if call.capability == 'knowledge_glossary':
            if call.arguments.question not in question:
                raise PlanError('knowledge_scope_conflict', 'Quote the knowledge subquestion from the current question.', repairable=True)
    for call in contract.calls:
        if call.arguments.continue_pending:
            raise PlanError('composition_nested_pending', 'Only the enclosing plan may continue a pending question.', repairable=True)
        if call.capability == 'knowledge_glossary' and call.arguments.question not in question:
            raise PlanError('knowledge_scope_conflict', 'Quote the knowledge subquestion from the current question.', repairable=True)
        if call.capability != 'knowledge_glossary':
            numeric_mentions.append(call.arguments.entity.mention)
            subquestion = getattr(call, 'subquestion', None)
            if subquestion is not None:
                if subquestion not in question:
                    raise PlanError('composition_subquestion_conflict', 'Quote numeric subquestions verbatim.', repairable=True)
                mention = call.arguments.entity.mention
                if len(explicit_players | explicit_teams) > 1 and (not mention or mention.casefold() not in subquestion.casefold()):
                    raise PlanError('composition_subquestion_entity', 'An entity-specific subquestion must retain its entity mention.', repairable=True)
                submetrics = requested_metrics(subquestion)
                if getattr(call.arguments, 'subject', 'player') == 'player' and 'games_played' in submetrics:
                    submetrics = (submetrics - {'games_played'}) | {'games_appeared'}
                declared = {'efg_percentage'} if call.capability == 'player_efg' else set(call.arguments.metrics)
                if call.capability == 'player_efg':
                    submetrics.discard('three_pointers_made')
                if (requested_metrics(numeric_question) and not submetrics) or (submetrics and submetrics != declared):
                    raise PlanError('composition_subquestion_metric', 'The call must preserve only the metrics requested by its quoted subquestion.', repairable=True)
            if call.capability == 'player_efg' and call.arguments.knowledge_question:
                raise PlanError('composition_nested_knowledge', 'Use a separate quoted knowledge call in a composed plan.', repairable=True)
        local_links = links.model_copy(deep=True)
        compiled = compile_semantic(call.capability, call.arguments, connection, question,
                                    local_links, previous, diagnostics, composing=True,
                                    comparing=contract.mode == 'compare', metric_question=getattr(call, 'subquestion', None) or numeric_question)
        if call.capability != 'knowledge_glossary':
            quoted_bindings.append((compiled.steps[0].arguments['entity_id'], call.arguments.entity.mention, getattr(call, 'subquestion', None)))
        for step in compiled.steps:
            fingerprint = query_fingerprint(step.capability, step.arguments)
            if fingerprint in fingerprints:
                raise PlanError('composition_duplicate', 'Do not repeat identical requirements.', repairable=True)
            fingerprints.add(fingerprint)
            steps.append(RouteStep(id=f's{len(steps)+1}', capability=step.capability, arguments=step.arguments))
        resolved_players.update({p.id: p for p in local_links.players})
        resolved_teams.update({t.id: t for t in local_links.teams})
    for entity_id, mention, quote in quoted_bindings:
        if quote and any(other_id != entity_id and other_mention and other_mention.casefold() in quote.casefold()
                         for other_id, other_mention, _ in quoted_bindings):
            raise PlanError('composition_subquestion_entity',
                            'Quote a subquestion associated with this target only; multiple targets are ambiguous.', repairable=True)
    targets = {'player': set(), 'team': set()}
    opponents = set()
    for step in steps:
        if step.capability != 'knowledge_glossary':
            targets[step.arguments.get('subject', 'player')].add(step.arguments['entity_id'])
            if step.arguments.get('opponent_team_id'):
                opponents.add(step.arguments['opponent_team_id'])
    if (explicit_players - targets['player'] or explicit_teams - targets['team'] - opponents):
        raise PlanError('composition_entity_coverage', 'The plan must preserve every requested target entity.', repairable=True)
    numeric = [s for s in steps if s.capability != 'knowledge_glossary']
    metrics = {m for s in numeric for m in (['efg_percentage'] if s.capability == 'player_efg' else s.arguments['metrics'])}
    requested = requested_metrics(numeric_question)
    if targets['player'] and not targets['team'] and 'games_played' in requested:
        requested = (requested - {'games_played'}) | {'games_appeared'}
    if 'efg_percentage' in metrics:
        requested.discard('three_pointers_made')  # existing eFG formula-input policy
    if requested and metrics != requested:
        raise PlanError('composition_metric_coverage', 'Preserve requested metrics without adding unrequested ones.', repairable=True)
    knowledge = [s for s in steps if s.capability == 'knowledge_glossary']
    for step in knowledge:
        contextual = re.search(r'这个指标|该指标|\b(?:this|that|the) metric\b', step.arguments['term'], re.I)
        if contextual and len(metrics) == 1:
            metric = next(iter(metrics))
            step.arguments['context_metric'] = ('effective field goal percentage (eFG%)' if metric == 'efg_percentage' else _METRIC_LABELS[metric])
        elif contextual:
            raise PlanError('composition_metric_reference', 'Multiple metrics are present; specify which one to explain.', clarify=True)
    comparison_ids = [s.id for s in steps if s.capability != 'knowledge_glossary'] if contract.mode == 'compare' else []
    if contract.mode == 'compare':
        group = explicit_target_group(numeric_question)
        if len(group) > MAX_PLAN_STEPS:
            raise PlanError('composition_resource_limit', f'当前单次执行最多支持 {MAX_PLAN_STEPS} 个原子步骤，请拆分请求。', clarify=True)
        if group and {m.casefold() for m in group} != {m.casefold() for m in numeric_mentions if m}:
            raise PlanError('composition_entity_coverage', 'Preserve each member of the explicit target list.', repairable=True)
        if group and [m.casefold() for m in group] != [m.casefold() for m in numeric_mentions]:
            raise PlanError('composition_direction', 'Preserve the declared target list order, not background mention order.', repairable=True)
        if not group and all(m and m.casefold() in numeric_question.casefold() for m in numeric_mentions):
            positions = [numeric_question.casefold().index(m.casefold()) for m in numeric_mentions]
            if positions != sorted(positions):
                raise PlanError('composition_direction', 'Preserve the target entity order from the question.', repairable=True)
            # Shared suffix/prefix metrics need no fabricated per-target quote.
            # Explicit metric clauses between targets must each name the same
            # set; otherwise comparing all union metrics invents requirements.
            clauses = [numeric_question[p + len(m):end] for p, m, end in
                       zip(positions, numeric_mentions, positions[1:] + [len(numeric_question)])]
            assigned = [requested_metrics(c) for c in clauses]
            if any(assigned[:-1]) and (any(not a for a in assigned) or any(a != assigned[0] for a in assigned)):
                raise PlanError('comparison_metric_binding', '请明确各目标共同需要比较的指标；不同目标的独立指标请分别查询。', clarify=True)
    if contract.mode != 'compare' and re.search(r'比较|对比|\bcompare\b|\bcomparison\b', question, re.I) and len(targets['player'] | targets['team']) > 1:
        raise PlanError('comparison_required', 'Preserve the requested comparison, not only individual results.', repairable=True)
    try:
        validate_composed_steps(steps, comparison_ids)
    except ValueError as exc:
        raise PlanError('comparison_contract_conflict', str(exc), repairable=True) from exc
    links.players = list(resolved_players.values())
    links.teams = list(resolved_teams.values())
    return RoutePlan(route='composed', steps=steps, comparison_step_ids=comparison_ids)


def summarize_comparison(step_ids, entries, *, tracer=None):
    """Project all comparison targets only after complete validated coverage.

    entries are keyed by plan requirement ID, never by model execution order.
    Partial/failed/missing outputs remain visible through their ordinary answer
    and evidence; they do not produce a successful comparison.
    """
    if not step_ids or len(step_ids) < 2:
        return None
    selected = [entries.get(key) for key in step_ids]
    if any(e is None or e['status'] != 'complete' for e in selected):
        return None
    from app.agent_routes import RouteStep
    steps = [RouteStep(id=key, capability=e['capability'], arguments=e['arguments'])
             for key, e in zip(step_ids, selected)]
    validate_composed_steps(steps, list(step_ids))
    first = steps[0]
    args = first.arguments
    metrics = ['efg_percentage'] if first.capability == 'player_efg' else args['metrics']
    rows = []
    for step, entry in zip(steps, selected):
        result = entry['result']
        values = result['values'] if step.capability == 'window_query' else result
        ev = result.get('evidence') or {}
        rows.append({'step_id': step.id, 'entity_id': step.arguments['entity_id'],
                     'name': (ev.get('entity') or {}).get('name') or step.arguments['entity_id'],
                     'values': {m: values[m] for m in metrics}})
    # Keep the established two-target response; N-target comparisons are a
    # matrix, not an arbitrary first pair or O(N^2) collection of differences.
    pairwise = len(rows) == 2
    deltas = {m: rows[0]['values'][m] - rows[1]['values'][m] for m in metrics} if pairwise else {}
    units = {m: 'percentage_points' if m == 'efg_percentage' else
             ('points_per_appearance' if m == 'points_per_appearance' else 'count') for m in metrics}
    labels = {**_METRIC_LABELS, 'efg_percentage': '有效命中率 eFG%'}
    names = '、'.join(r['name'] for r in rows)
    lines = [f"{args['season_id']} {_PHASES[args['competition_phase']]}，{names}的比较" + ('（前者减后者）' if pairwise else '') + '：']
    for metric in metrics:
        if pairwise:
            lvalue, rvalue, delta = (rows[0]['values'][metric], rows[1]['values'][metric], deltas[metric])
            unit = ' 个百分点' if metric == 'efg_percentage' else ''
            lines.append(f"{labels[metric]}：{_fmt(lvalue)} 对 {_fmt(rvalue)}；有向差值 {_fmt(delta)}{unit}。")
        else:
            lines.append(f"{labels[metric]}：" + '；'.join(f"{r['name']} {_fmt(r['values'][metric])}" for r in rows) + '。')
    comparison = {'step_ids': list(step_ids), 'rows': rows, 'deltas': deltas, 'delta_units': units,
                  'scope': {k: v for k, v in args.items() if k not in {'entity_id', 'metrics'}},
                  'kind': 'pairwise' if pairwise else 'matrix', 'metrics': list(metrics),
                  'formula': 'left - right' if pairwise else None, 'text': '\n'.join(lines)}
    with span(tracer, 'nba.agent.composition', 'CHAIN') as handle:
        set_attribute(handle, 'nba.agent.comparison.target_count', len(rows))
        set_attribute(handle, 'nba.agent.comparison.metric_count', len(metrics))
        set_text(handle, 'input.value', {'step_ids': list(step_ids), 'rows': rows})
        set_text(handle, 'output.value', comparison)
        mark_success(handle)
    return comparison
