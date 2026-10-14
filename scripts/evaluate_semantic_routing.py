"""Paired routing evaluation with independent SQL oracles and a durable call cap."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'metadata/semantic_routing_v1.json'


def read_cases(path=CASES, split='all'):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    cases = manifest['cases']
    if len(cases) != 40 or len({case['id'] for case in cases}) != 40:
        raise ValueError('expected 40 unique frozen scenarios')
    if sum(case['split'] == 'development' for case in cases) != 12:
        raise ValueError('expected 12 development and 28 frozen scenarios')
    return manifest, [case for case in cases if split == 'all' or case['split'] == split]


class CallBudgetModel:
    """Count every attempted provider request, including failed RAG/Loop calls."""
    def __init__(self, model, *, maximum=200, ledger=None):
        self.wrapped, self.maximum, self.ledger = model, maximum, ledger
        self.model = getattr(model, 'model', 'unknown')
        self.base_url = getattr(model, 'base_url', '')
        self.lock = threading.Lock()
        self.records = []
        self.used = json.loads(ledger.read_text())['used'] if ledger and ledger.exists() else 0

    def _invoke(self, request, native):
        with self.lock:
            if self.maximum is not None and self.used >= self.maximum:
                raise RuntimeError('evaluation_model_call_budget_exceeded')
            self.used += 1
            if self.ledger:
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                self.ledger.write_text(json.dumps({'used': self.used, 'maximum': self.maximum}), encoding='utf-8')
            record = {'sequence': self.used, 'kind': 'native_tools' if native else 'text_model', 'usage': None, 'error': None}
            self.records.append(record)
        started = time.monotonic()
        try:
            response = (self.wrapped.generate_tools if native else self.wrapped.generate)(request)
            record['usage'] = response.usage
            return response
        except Exception as exc:
            record['error'] = type(exc).__name__
            raise
        finally:
            record['elapsed_ms'] = round((time.monotonic()-started)*1000, 3)

    def generate(self, request):
        return self._invoke(request, False)

    def generate_tools(self, request):
        return self._invoke(request, True)


def sql_oracle(connection, expected, tool=None, entity=None):
    """Read raw rows rather than invoking the production tool or compiler."""
    tool = tool or expected.get('tool')
    identity = entity or expected.get('entity')
    if tool not in {'season_summary', 'window_query', 'player_efg', 'season_compare'}:
        return {}
    subject = expected.get('subject', 'player')
    columns = ('p.points, p.assists, p.rebounds_total, p.field_goals_made, p.three_pointers_made, p.field_goals_attempted'
               if subject == 'player' else 'p.team_score, p.assists, p.rebounds_total, NULL, NULL, NULL')
    table, key = ('player_game_stat', 'person_id') if subject == 'player' else ('team_game_stat', 'team_id')

    def rows(season):
        clauses = [f'p.{key} = ?', 'g.season_id = ?', 'g.competition_phase = ?']
        params = [identity, season, expected['phase']]
        if subject == 'player':
            clauses.append("p.appearance_status IN ('played','played_inferred')")
        for field, sql in (('opponent_team_id', 'p.opponent_team_id = ?'), ('date_from', 'g.game_date >= ?'), ('date_to', 'g.game_date <= ?')):
            if expected.get(field):
                clauses.append(sql); params.append(expected[field])
        query = f'SELECT {columns} FROM {table} p JOIN game g ON g.game_id=p.game_id WHERE ' + ' AND '.join(clauses) + ' ORDER BY g.game_date DESC, p.game_id DESC'
        if expected.get('last_n'):
            query += ' LIMIT ?'; params.append(expected['last_n'])
        return connection.execute(query, params).fetchall()

    def value(data, metric):
        if not data:
            return None
        if metric in {'games_appeared', 'games_played'}:
            return len(data)
        index = {'points_total': 0, 'points_per_appearance': 0, 'assists_total': 1, 'rebounds_total': 2}[metric]
        values = [row[index] for row in data]
        if any(v is None for v in values):
            return None
        result = sum(values)
        return result/len(values) if metric == 'points_per_appearance' else result

    if tool == 'season_compare':
        metric = expected['metric']
        left, right = value(rows(expected['baseline_season']), metric), value(rows(expected['comparison_season']), metric)
        return {'absolute_delta': None if left is None or right is None else right-left}
    data = rows(expected['season'])
    if tool == 'player_efg':
        if not data or any(any(v is None for v in row[3:]) for row in data):
            return {'efg_percentage': None}
        fgm, three, fga = (sum(row[index] for row in data) for index in (3, 4, 5))
        return {'efg_percentage': (fgm+.5*three)/fga*100 if fga else None}
    return {metric: value(data, metric) for metric in expected['metrics']}


def score(expected, body, oracles):
    tools = expected.get('tools') or ([expected['tool']] if expected.get('tool') else [])
    route = expected.get('route', 'knowledge' if tools == ['knowledge_glossary'] else 'structured')
    status = expected.get('status', 'complete')
    checks = {'route': body.get('route') == route, 'status': body.get('status') == status}
    steps = body.get('steps', [])
    if tools:
        checks['tools'] = [step.get('capability') for step in steps] == tools
        for index, step in enumerate(steps):
            if index >= len(tools):
                break
            args = (body.get('task_state') or {}).get('route', {}).get('steps', [])
            arguments = args[index]['arguments'] if index < len(args) else step.get('arguments', {})
            # Loop records expose verified execution arguments in observations.
            if not arguments and index < len(body.get('observations', [])):
                arguments = body['observations'][index].get('arguments', {})
            if tools[index] in {'season_summary', 'window_query', 'player_efg', 'season_compare'}:
                entity = expected.get('entities', [expected.get('entity')])[min(index, len(expected.get('entities', [None]))-1)]
                checks[f'entity_{index}'] = arguments.get('entity_id') == entity
                for field, target in (('season_id', expected.get('season')), ('competition_phase', expected.get('phase')),
                                      ('baseline_season', expected.get('baseline_season')), ('comparison_season', expected.get('comparison_season')),
                                      ('opponent_team_id', expected.get('opponent_team_id')), ('last_n', expected.get('last_n')),
                                      ('date_from', expected.get('date_from')), ('date_to', expected.get('date_to'))):
                    if target is not None:
                        checks[f'{field}_{index}'] = arguments.get(field) == target
                if expected.get('metrics'):
                    checks[f'metrics_{index}'] = set(arguments.get('metrics', [])) == set(expected['metrics'])
                actual = step.get('result') or {}
                if tools[index] == 'window_query':
                    actual = actual.get('values') or {}
                for key, target in (oracles[index] if index < len(oracles) else {}).items():
                    number = actual.get(key)
                    checks[f'numeric_{index}_{key}'] = (target is None and number is None) if status == 'not_computable' else (
                        isinstance(number, (int, float)) and not isinstance(number, bool) and target is not None and math.isclose(number, target, rel_tol=1e-8, abs_tol=1e-8))
                if status == 'complete':
                    checks[f'evidence_{index}'] = bool((step.get('evidence') or {}).get('structured', {}).get('source_sql'))
            elif tools[index] == 'efg_compare':
                for key, target in (oracles[index] if index < len(oracles) else {}).items():
                    number = (step.get('result') or {}).get(key)
                    checks[f'numeric_{index}_{key}'] = (target is None and number is None) if status == 'not_computable' else (
                        isinstance(number, (int, float)) and not isinstance(number, bool) and target is not None
                        and math.isclose(number, target, rel_tol=1e-8, abs_tol=1e-8))
            elif tools[index] == 'knowledge_glossary' and status == 'complete':
                checks[f'citation_shape_{index}'] = bool((step.get('evidence') or {}).get('knowledge', {}).get('citations'))
    else:
        checks['no_execution'] = not steps
    return {'passed': all(checks.values()), 'checks': checks}


def summarize(records):
    summary = {}
    for arm in dict.fromkeys(row['arm'] for row in records):
        rows = [row for row in records if row['arm'] == arm]
        latencies = sorted(row['latency_ms'] for row in rows)

        def percentile(fraction):
            position = (len(latencies)-1)*fraction
            lower, upper = math.floor(position), math.ceil(position)
            return round(latencies[lower] + (latencies[upper]-latencies[lower])*(position-lower), 3)

        categories = {}
        for row in rows:
            for key, passed in row['checks'].items():
                category = ('scope' if key.startswith(('season_id_', 'competition_phase_', 'baseline_season_', 'comparison_season_'))
                            else 'filters' if key.startswith(('opponent_team_id_', 'last_n_', 'date_from_', 'date_to_'))
                            else key.split('_')[0])
                counts = categories.setdefault(category, {'passed': 0, 'total': 0})
                counts['passed'] += bool(passed)
                counts['total'] += 1
        for counts in categories.values():
            counts['accuracy'] = counts['passed']/counts['total']
        requests = [request for row in rows for request in row['model_requests']]
        token_totals = [(request.get('usage') or {}).get('total_tokens') for request in requests]
        observed = [value for value in token_totals if isinstance(value, int)]
        summary[arm] = {
            'turns': len(rows), 'passed': sum(row['passed'] for row in rows),
            'success_rate': sum(row['passed'] for row in rows)/len(rows),
            'model_requests': len(requests), 'mean_latency_ms': sum(latencies)/len(rows),
            'p50_latency_ms': percentile(.5), 'p95_latency_ms': percentile(.95),
            'runtime_failures': sum(row['runtime_failure'] for row in rows),
            'check_accuracy': categories,
            'usage': {'requests_with_usage': len(observed), 'requests_without_usage': len(requests)-len(observed),
                      'observed_total_tokens': sum(observed) if observed else None},
        }
    return summary


async def run(args):
    from app.main import app as existing, create_app
    from app.agent_routes import KnowledgeGlossaryAdapter
    from scripts.model_adapter import OpenAICompatibleModel
    from scripts.provider_config import load_provider_config
    config = load_provider_config()
    database = args.db or Path(config.database_path)
    manifest, cases = read_cases(args.cases, args.split)
    if args.case_ids:
        selected_ids = set(args.case_ids.split(','))
        cases = [case for case in cases if case['id'] in selected_ids]
        if {case['id'] for case in cases} != selected_ids:
            raise ValueError('unknown case IDs or IDs outside selected split')
    if args.endpoint in {'/answer', '/ask'}:
        cases = [case for case in cases if all(turn.get('tool') in {'season_summary', 'window_query', 'season_compare', 'player_efg', 'single_game_score'} for turn in case['turns'])]
    model = CallBudgetModel(OpenAICompatibleModel(config=config), maximum=args.max_model_calls,
                            ledger=ROOT/'reports/semantic-routing/model-call-budget.json')
    search = existing.state.agent_registry.get('knowledge_glossary').search
    knowledge = KnowledgeGlossaryAdapter(search, model=model, provider_config=config)
    application = create_app(db_path=database, model=model, provider_config=config, routing_mode='model',
        knowledge_search=search, agent_knowledge_adapter=knowledge, agent_loop_enabled=True,
        ask_total_timeout_seconds=60, ask_model_timeout_seconds=15)
    from httpx import AsyncClient, ASGITransport
    records = []
    before = model.used
    try:
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as connection:
            async with AsyncClient(transport=ASGITransport(app=application), base_url='http://evaluation') as client:
                for position, case in enumerate(cases):
                    arms = ['loop'] if args.loop else (['rules', 'model'] if position % 2 else ['model', 'rules'])
                    for arm in arms:
                        conversation = uuid4().hex
                        for turn in case['turns']:
                            if model.used >= model.maximum:
                                break
                            tools = turn.get('tools') or ([turn['tool']] if turn.get('tool') else [])
                            oracles = [sql_oracle(connection, turn, tool, (turn.get('entities') or [turn.get('entity')])[min(i, len(turn.get('entities') or [None])-1)]) for i, tool in enumerate(tools)]
                            begin = len(model.records)
                            started = time.monotonic()
                            response = await client.post('/agent/loop' if args.loop else args.endpoint, json={
                                'question': turn['q'], 'conversation_id': conversation, 'routing_mode': 'model' if args.loop else arm})
                            body = response.json()
                            view = body
                            if args.endpoint in {'/answer', '/ask'}:
                                from app.agent_routes import RouteStep, _bundle
                                tool = body.get('tool')
                                route = 'structured' if tool else 'clarification' if body.get('status') == 'clarification_required' else 'unsupported'
                                steps = []
                                if tool:
                                    step = RouteStep(id='s1', capability=tool, arguments=body['arguments'])
                                    evidence = _bundle(step, body['status'] if body['status'] in {'complete', 'partial', 'not_computable'} else 'failed', body.get('tool_result') or {})
                                    steps = [{'capability': tool, 'arguments': body['arguments'], 'result': body.get('tool_result'), 'evidence': evidence.model_dump(mode='json')}]
                                view = {**body, 'route': route, 'steps': steps}
                            if tools == ['player_efg', 'player_efg', 'efg_compare', 'knowledge_glossary']:
                                left, right = oracles[0]['efg_percentage'], oracles[1]['efg_percentage']
                                oracles[2] = {'absolute_delta': left-right if left is not None and right is not None else None}
                            scored = score(turn, view, oracles)
                            runtime_codes = {'model_timeout', 'provider_failed', 'model_unconfigured', 'agent_timeout', 'retrieval_timeout', 'generation_timeout', 'verification_timeout', 'no_usable_result', 'decision_failed', 'decision_timeout', 'deadline_exceeded'}
                            runtime = body.get('reason_code') in runtime_codes or any((step.get('result') or {}).get('reason_code') in runtime_codes for step in body.get('steps', []))
                            record = {'case_id': case['id'], 'split': case['split'], 'arm': arm, 'question': turn['q'],
                                      'latency_ms': round((time.monotonic()-started)*1000, 3), **scored,
                                      'runtime_failure': runtime, 'response': body, 'oracle': oracles,
                                      'model_requests': model.records[begin:], 'semantic_review': 'unreviewed'}
                            records.append(record)
                            print(f"{case['id']} {arm}: {'passed' if scored['passed'] else 'failed'} {body.get('status')} {body.get('reason_code')}", flush=True)
                    if model.used >= model.maximum:
                        break
    finally:
        report = {'version': '1.0.0', 'mode': 'real_native_loop' if args.loop else 'paired_real_routing',
            'provider': config.summary(), 'manifest_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                  'implementation_sha256': {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
                      'scripts/model_adapter.py', 'app/semantic_contracts.py', 'app/entity_resolution.py', 'app/semantic_planner.py',
                      'app/routing_service.py', 'app/agent_routes.py', 'app/agent_tasks.py', 'app/agent_decider.py', 'app/agent_loop.py',
                      'app/agent_loop_routes.py', 'scripts/evaluate_semantic_routing.py')},
                  'source_snapshot': manifest['source_snapshot'], 'split': args.split,
                  'endpoint': '/agent/loop' if args.loop else args.endpoint,
                  'case_ids': args.case_ids, 'evaluation_kind': 'selected_regression' if args.case_ids else 'frozen_scenario_comparison',
                  'model_calls_this_run': model.used-before, 'model_calls_total': model.used,
                  'budget_exhausted': model.used >= model.maximum, 'records': records, 'summary': summarize(records),
                  'limitations': [manifest['limitations'], 'Runtime failures are included in end-to-end success denominator.',
                                 'No statistical significance claim; scenarios and repeated turns are not independent samples.',
                                 'Cost is unknown unless provider pricing is explicitly pinned.'], 'cost': None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--split', choices=['all', 'development', 'frozen'], default='all')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--case-ids', help='comma-separated IDs for explicitly labeled regression runs')
    parser.add_argument('--endpoint', choices=['/agent/answer', '/answer', '/ask'], default='/agent/answer')
    parser.add_argument('--max-model-calls', type=int, default=200)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/semantic-routing/paired-v1.json')
    args = parser.parse_args()
    if not 1 <= args.max_model_calls <= 200:
        parser.error('max-model-calls must be between 1 and 200')
    report = asyncio.run(run(args))
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
