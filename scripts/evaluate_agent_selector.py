"""Real-provider smoke: native selectors, SQL oracle, citations and complete trace.

Run outside the Codex sandbox. Uses .env.local exclusively and a temporary
copy of the local index so an active demo's Qdrant lock remains untouched.
This is a focused regression experiment, not a representative quality benchmark.
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
    ('reported', '比较 库里 与 哈登 2025-2026赛季常规赛的eFG%，并解释 eFG 是什么。',
     ['201939', '201935'], '2025-26', 'comparison'),
    ('other_entities', '对比詹姆斯与库里 2023-24 常规赛的 eFG%，并说明这个指标的定义。',
     ['2544', '201939'], '2023-24', 'comparison'),
    ('mixed', '哈登在 2016-17 常规赛的有效命中率是多少，并解释 eFG%？',
     ['201935'], '2016-17', 'mixed'),
    ('contextual_mixed', '库里 2023-24 常规赛 eFG% 是多少，并解释这个指标的含义？',
     ['201939'], '2023-24', 'mixed'),
]


def sql_oracle(database, entity_id, season):
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as db:
        fgm, threes, fga, count = db.execute('''
            SELECT SUM(p.field_goals_made), SUM(p.three_pointers_made),
                   SUM(p.field_goals_attempted), COUNT(*)
            FROM player_game_stat p JOIN game g ON g.game_id=p.game_id
            WHERE p.person_id=? AND g.season_id=? AND g.competition_phase='regular_season'
              AND p.appearance_status IN ('played', 'played_inferred')
        ''', (entity_id, season)).fetchone()
    return (fgm + .5 * threes) / fga * 100 if fga and count else None


def assess_response(body, entities, season, route, expected):
    observations = body.get('observations') or []
    numeric = [o for o in observations if o.get('capability') == 'player_efg']
    def result(observation):
        value = observation.get('result')
        return value if isinstance(value, dict) else {}
    def matches(actual, oracle):
        return (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (actual, oracle))
                and math.isclose(actual, oracle, rel_tol=0, abs_tol=1e-9))
    checks = {'complete': body.get('status') == 'complete', 'route': body.get('route') == route,
              'no_decision_failures': not body.get('decision_failures'),
              'entities': len(numeric) == len(entities) and
                  {o.get('arguments', {}).get('entity_id') for o in numeric} == set(entities),
              'numeric': len(numeric) == len(entities) and all(
                  o.get('status') == 'complete'
                  and o.get('arguments', {}).get('season_id') == season
                  and o.get('arguments', {}).get('competition_phase') == 'regular_season'
                  and matches(result(o).get('efg_percentage'), expected.get(o.get('arguments', {}).get('entity_id')))
                  for o in numeric),
              'knowledge_cited': any(o.get('capability') == 'knowledge_glossary' and o.get('status') == 'complete'
                  and any(c.get('citation_id') and c.get('excerpt') for c in result(o).get('claims', []) if isinstance(c, dict))
                  for o in observations),
              'verification': any(o.get('capability') == 'knowledge_glossary'
                  and result(o).get('stage_statuses', {}).get('verification') == 'complete' for o in observations)}
    if route == 'comparison':
        compares = [o for o in observations if o.get('capability') == 'efg_compare']
        left, right = (expected.get(e) for e in entities)
        delta = left - right if matches(left, left) and matches(right, right) else None
        checks['delta'] = len(compares) == 1 and matches(result(compares[0]).get('absolute_delta'), delta)
    return checks


async def evaluate(output: Path, repeat: int):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from scripts.knowledge_search import configured_search
    from app.main import create_app

    config = load_provider_config()
    tracer = configured_tracer(config)
    exporter = InMemorySpanExporter()
    if tracer is not None:
        trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.sdk.trace import TracerProvider
        local = TracerProvider()
        local.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = local.get_tracer('agent-selector-evaluation')
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='nba-selector-index-') as tmp:
        snapshot = Path(tmp) / 'index'
        shutil.copytree(config.knowledge_index_path, snapshot)
        config = replace(config, knowledge_index_path=str(snapshot))
        search = configured_search(snapshot, provider_config=config)
        if search is None or search.dense is None:
            raise RuntimeError('real dense + BM25 index required')
        try:
            app = create_app(provider_config=config, knowledge_search=search, tracer=tracer,
                             routing_mode='model', agent_unified_loop_enabled=True,
                             ask_total_timeout_seconds=60, ask_model_timeout_seconds=15)
            async with AsyncClient(transport=ASGITransport(app=app), base_url='http://experiment') as client:
                for iteration in range(repeat):
                    for case_id, question, entities, season, route in CASES:
                        started = time.monotonic()
                        try:
                            response = await client.post('/agent/answer', json={'question': question})
                            body = response.json()
                        except Exception as exc:
                            body = {'status': 'incomplete', 'reason_code': 'evaluation_transport_failed',
                                    'error_type': type(exc).__name__, 'observations': []}
                        expected = {e: sql_oracle(config.database_path, e, season) for e in entities}
                        checks = assess_response(body, entities, season, route, expected)
                        spans = [s for s in exporter.get_finished_spans()
                                 if format(s.context.trace_id, '032x') == body.get('trace_id')]
                        names = [s.name for s in spans]
                        checks['full_trace'] = {'nba.agent.loop.tool', 'nba.knowledge.generation',
                            'nba.knowledge.citation_validation', 'nba.knowledge.verification',
                            'nba.agent.loop.termination', 'nba.agent.semantic_planner.llm',
                            'nba.agent.loop.decision', 'nba.agent.loop.observation',
                            'nba.agent.semantic_request'} <= set(names)
                        checks['trace_counts'] = (
                            names.count('nba.agent.semantic_planner.llm') == body.get('planning', {}).get('attempts', 0)
                            and names.count('nba.agent.loop.decision') == body.get('decisions', 0)
                            and names.count('nba.agent.loop.tool') == body.get('tool_calls', 0)
                            and names.count('nba.agent.loop.observation') == len(body.get('observations') or []))
                        by_name = {s.name: s for s in spans}
                        root = by_name.get('nba.agent.semantic_request')
                        orchestrator = by_name.get('nba.agent.orchestrator')
                        loop = by_name.get('nba.agent.loop.request')
                        checks['trace_parentage'] = bool(root and orchestrator and loop
                            and orchestrator.parent and orchestrator.parent.span_id == root.context.span_id
                            and loop.parent and loop.parent.span_id == orchestrator.context.span_id
                            and all(s.parent and s.parent.span_id == loop.context.span_id for s in spans
                                    if s.name in {'nba.agent.loop.decision', 'nba.agent.loop.tool',
                                                  'nba.agent.loop.observation', 'nba.agent.loop.termination'}))
                        row = {'case_id': case_id, 'iteration': iteration + 1, 'question': question,
                               'elapsed_ms': round((time.monotonic()-started)*1000, 3),
                               'checks': checks, 'passed': all(checks.values()), 'sql_oracle': expected,
                               'span_names': names, 'response': body}
                        rows.append(row)
                        output.write_text(json.dumps({'experiment': 'native-selector-live', 'checks_version': 'selector-smoke-v2',
                            'protocol': 'plan-selector-v1/requirement-selector-v1', 'model': config.model_name,
                            'limitations': 'Small development smoke; numeric oracle checks the pinned local snapshot, not season completeness or externally verified data.',
                            'runs': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(json.dumps({k:row[k] for k in ('case_id', 'iteration', 'elapsed_ms', 'passed', 'checks')}, ensure_ascii=False), flush=True)
        finally:
            search.dense.close()
            flush_tracer()
    return all(row['passed'] for row in rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/agent-loop/native-selector-live.json'))
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    raise SystemExit(0 if asyncio.run(evaluate(args.output, args.repeat)) else 1)
