"""Real-provider composition smoke with an independent read-only SQL oracle.

Run outside the sandbox. Configuration comes exclusively from .env.local.
No knowledge calls are required by these statistical cases; an isolated copied
index keeps app startup independent of an active demo's Qdrant lock.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from httpx import ASGITransport, AsyncClient
from scripts.provider_config import load_provider_config
from scripts.tracing import configured_tracer, flush_tracer


CASES = [
    ('reported', '比较 库里 与 哈登 2015-2016赛季季后赛的总得分和总篮板。',
     'player', ['201939', '201935'], '2015-16', 'playoffs', ['points_total', 'rebounds_total']),
    ('other_metrics', '对比詹姆斯与库里 2023-24赛季常规赛的总得分和总助攻。',
     'player', ['2544', '201939'], '2023-24', 'regular_season', ['points_total', 'assists_total']),
    ('reverse_order', '比较哈登与库里 2016-17赛季常规赛的总篮板和总得分。',
     'player', ['201935', '201939'], '2016-17', 'regular_season', ['rebounds_total', 'points_total']),
    ('teams', '比较湖人与勇士 2023-24赛季常规赛的总得分和总篮板。',
     'team', ['1610612747', '1610612744'], '2023-24', 'regular_season', ['points_total', 'rebounds_total']),
    ('heterogeneous', '查询 2023-24 常规赛的库里总得分与詹姆斯eFG%。',
     'player', ['201939', '2544'], '2023-24', 'regular_season', ['points_total', 'efg_percentage']),
    ('entity_metrics', '查询 2023-24 常规赛的库里总得分与詹姆斯总篮板。',
     'player', ['201939', '2544'], '2023-24', 'regular_season', ['points_total', 'rebounds_total']),
]
COLLECT_CASES = {'heterogeneous', 'entity_metrics'}
COMPARISON_CASES = [
    ('reported_three', '比较 库里、詹姆斯和欧文在 2015-2016赛季季后赛的总得分、总篮板和总助攻。',
     'player', ['201939', '2544', '202681'], '2015-16', 'playoffs', ['points_total', 'rebounds_total', 'assists_total']),
    CASES[0],
    ('four_targets', '对比詹姆斯、欧文、库里和哈登在2023-24赛季常规赛的总得分和总助攻。',
     'player', ['2544', '202681', '201939', '201935'], '2023-24', 'regular_season', ['points_total', 'assists_total']),
    ('reversed_three', '比较欧文、詹姆斯和库里在2015-16季后赛的总助攻、总篮板和总得分。',
     'player', ['202681', '2544', '201939'], '2015-16', 'playoffs', ['assists_total', 'rebounds_total', 'points_total']),
    ('english_three', 'Compare Stephen Curry, LeBron James and Kyrie Irving in the 2015-16 playoffs on total points, rebounds and assists.',
     'player', ['201939', '2544', '202681'], '2015-16', 'playoffs', ['points_total', 'rebounds_total', 'assists_total']),
    ('metric_prefix', '比较总得分、总篮板和总助攻：库里、詹姆斯和欧文，范围是2015-16赛季季后赛。',
     'player', ['201939', '2544', '202681'], '2015-16', 'playoffs', ['points_total', 'rebounds_total', 'assists_total']),
]


def sql_oracle(database, subject, entity, season, phase, metrics):
    # Metric identifiers are from this fixed offline case catalog, never model output.
    columns = {'points_total': 'points' if subject == 'player' else 'team_score',
               'rebounds_total': 'rebounds_total', 'assists_total': 'assists'}
    expressions = {m: 'SUM(p.' + column + ')' for m, column in columns.items()}
    expressions['efg_percentage'] = '(SUM(p.field_goals_made) + 0.5 * SUM(p.three_pointers_made)) * 100.0 / NULLIF(SUM(p.field_goals_attempted), 0)'
    table, key = ('player_game_stat', 'person_id') if subject == 'player' else ('team_game_stat', 'team_id')
    query = (f"SELECT {', '.join(expressions[m] for m in metrics)}, COUNT(*) "
             f"FROM {table} p JOIN game g ON g.game_id=p.game_id "
             f"WHERE p.{key}=? AND g.season_id=? AND g.competition_phase=?")
    if subject == 'player':
        query += " AND p.appearance_status IN ('played', 'played_inferred')"
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        row = db.execute(query, (entity, season, phase)).fetchone()
    return dict(zip(metrics, row[:-1])) if row[-1] else {m: None for m in metrics}


def assess(body, subject, entities, season, phase, metrics, expected):
    def matches(actual, oracle):
        return (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (actual, oracle)) and math.isclose(actual, oracle, rel_tol=0, abs_tol=1e-9))
    attempts = [s for s in body.get('steps', []) if s.get('capability') == 'season_summary']
    numeric = [s for s in attempts if s.get('status') == 'complete']
    comp = body.get('comparison') or {}
    rows = comp.get('rows') or []
    checks = {'complete': body.get('status') == 'complete', 'route': body.get('route') == 'composed',
              'atomic_reuse': len(numeric) == len(entities) and body.get('tool_calls') == len(attempts)
                  and len(body.get('steps', [])) == len(attempts),
              'no_decision_failures': not body.get('decision_failures'),
              'direction': [r.get('entity_id') for r in rows] == entities,
              'metrics': len(rows) == len(entities) and all(set(r.get('values', {})) == set(metrics) for r in rows),
              'scope': all(comp.get('scope', {}).get(k) == v for k, v in {
                  'subject': subject, 'season_id': season, 'competition_phase': phase}.items()),
              'numeric': len(rows) == len(entities) and all(matches(r.get('values', {}).get(m), expected.get(r.get('entity_id'), {}).get(m))
                                            for r in rows for m in metrics),
              'evidence': len(numeric) == len(entities) and all(s.get('evidence', {}).get('structured', {}).get('source_sql') for s in numeric)}
    deltas = {}
    for m in metrics:
        left, right = expected[entities[0]][m], expected[entities[1]][m]
        deltas[m] = left - right if matches(left, left) and matches(right, right) else None
    checks['deltas'] = (set(comp.get('deltas', {})) == set(metrics) and all(matches(comp.get('deltas', {}).get(m), v) for m, v in deltas.items())
                        if len(entities) == 2 else bool(comp.get('kind') == 'matrix' and comp.get('deltas') == {} and comp.get('formula') is None))
    return checks


def assess_collect(body, entities, season, phase, metrics, expected):
    observations = body.get('observations') or []
    complete = [o for o in observations if o.get('status') == 'complete']
    checks = {'complete': body.get('status') == 'complete', 'route': body.get('route') == 'composed',
              'no_decision_failures': not body.get('decision_failures'),
              'coverage': len(complete) == 2 and {o.get('arguments', {}).get('entity_id') for o in complete} == set(entities),
              'no_comparison': body.get('comparison') is None}
    for entity, metric in zip(entities, metrics):
        selected = [o for o in complete if o.get('arguments', {}).get('entity_id') == entity]
        if len(selected) != 1:
            checks[entity] = False
            continue
        observation, oracle = selected[0], expected[entity][metric]
        args, result = observation.get('arguments') or {}, observation.get('result') or {}
        actual = result.get(metric)
        checks[entity] = bool(args.get('season_id') == season and args.get('competition_phase') == phase
            and observation.get('capability') == ('player_efg' if metric == 'efg_percentage' else 'season_summary')
            and (metric == 'efg_percentage' or args.get('metrics') == [metric])
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (actual, oracle))
            and math.isclose(actual, oracle, rel_tol=0, abs_tol=1e-9)
            and (result.get('evidence') or {}).get('source_sql'))
    return checks


async def evaluate(output, repeat, *, suite='composition', loop_enabled=True):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from scripts.knowledge_search import configured_search
    from app.main import create_app

    config = load_provider_config()
    tracer = configured_tracer(config)
    provider = trace.get_tracer_provider() if tracer else TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer or provider.get_tracer('agent-composition-evaluation')
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='nba-composition-index-') as tmp:
        snapshot = Path(tmp) / 'index'
        shutil.copytree(config.knowledge_index_path, snapshot)
        config = replace(config, knowledge_index_path=str(snapshot))
        search = configured_search(snapshot, provider_config=config)
        try:
            app = create_app(provider_config=config, knowledge_search=search, tracer=tracer,
                             routing_mode='model', agent_unified_loop_enabled=loop_enabled)
            async with AsyncClient(transport=ASGITransport(app=app), base_url='http://experiment') as client:
                for iteration in range(repeat):
                    cases = COMPARISON_CASES if suite == 'comparison' else CASES
                    for case_id, question, subject, entities, season, phase, metrics in cases:
                        started = time.monotonic()
                        response = await client.post('/agent/answer', json={'question': question})
                        body = response.json()
                        collecting = case_id in COLLECT_CASES
                        expected = {e: sql_oracle(config.database_path, subject, e, season, phase,
                                                 [metrics[i]] if collecting else metrics) for i, e in enumerate(entities)}
                        checks = (assess_collect(body, entities, season, phase, metrics, expected) if collecting
                                  else assess(body, subject, entities, season, phase, metrics, expected))
                        spans = [s for s in exporter.get_finished_spans() if format(s.context.trace_id, '032x') == body.get('trace_id')]
                        names = [s.name for s in spans]
                        checks['trace'] = (names.count('nba.agent.loop.tool') == body.get('tool_calls')
                            and names.count('nba.agent.sql') == body.get('tool_calls')
                            and names.count('nba.agent.composition') == (0 if collecting else 1)
                            and names.count('nba.agent.loop.observation') == len(body.get('observations', []))
                            and names.count('nba.agent.loop.decision') == body.get('decisions')
                            and 'nba.agent.loop.termination' in names)
                        if not loop_enabled:
                            checks['trace'] = (names.count('nba.agent.step') == body.get('tool_calls')
                                and names.count('nba.agent.sql') == body.get('tool_calls')
                                and names.count('nba.agent.composition') == (0 if collecting else 1)
                                and not any(n.startswith('nba.agent.loop.') for n in names))
                        by_name = {s.name: s for s in spans}
                        root, orchestrator, loop, synthesis = [by_name.get(n) for n in (
                            'nba.agent.semantic_request', 'nba.agent.orchestrator', 'nba.agent.loop.request', 'nba.agent.composition')]
                        checks['trace_parentage'] = bool(root and orchestrator and loop
                            and orchestrator.parent and orchestrator.parent.span_id == root.context.span_id
                            and loop.parent and loop.parent.span_id == orchestrator.context.span_id
                            and (collecting or (synthesis and synthesis.parent and synthesis.parent.span_id == loop.context.span_id))
                            and all(s.parent and s.parent.span_id == loop.context.span_id for s in spans if s.name == 'nba.agent.loop.tool'))
                        if not loop_enabled:
                            checks['trace_parentage'] = bool(root and orchestrator
                                and orchestrator.parent and orchestrator.parent.span_id == root.context.span_id
                                and (collecting or (synthesis and synthesis.parent and synthesis.parent.span_id == orchestrator.context.span_id))
                                and all(s.parent and s.parent.span_id == orchestrator.context.span_id for s in spans if s.name == 'nba.agent.step'))
                        row = {'case_id': case_id, 'iteration': iteration+1, 'question': question,
                               'elapsed_ms': round((time.monotonic()-started)*1000, 3),
                               'checks': checks, 'passed': all(checks.values()), 'sql_oracle': expected,
                               'span_names': names,
                               'planner_outputs': [s.attributes.get('output.value.preview') for s in spans
                                                   if s.name == 'nba.agent.semantic_planner.llm'],
                               'response': body}
                        rows.append(row)
                        output.write_text(json.dumps({'experiment': 'agent-composition', 'model': config.model_name,
                            'limitations': 'Development smoke; SQL checks pinned local snapshot, not season completeness or external source correctness.',
                            'runs': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(json.dumps({k: row[k] for k in ('case_id', 'iteration', 'elapsed_ms', 'passed', 'checks')}, ensure_ascii=False), flush=True)
        finally:
            if search and search.dense:
                search.dense.close()
            flush_tracer()
    return all(r['passed'] for r in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/agent-loop/composition-live.json'))
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--suite', choices=['composition', 'comparison'], default='composition')
    parser.add_argument('--loop-mode', choices=['enabled', 'disabled'], default='enabled')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    raise SystemExit(0 if asyncio.run(evaluate(args.output, args.repeat, suite=args.suite,
        loop_enabled=args.loop_mode == 'enabled')) else 1)
