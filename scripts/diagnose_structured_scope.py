"""Dedicated correctness smoke; never replaces the frozen routing benchmark."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

from scripts.evaluate_semantic_routing import CallBudgetModel, sql_oracle
from scripts.model_adapter import ModelResponse, OpenAICompatibleModel
from scripts.provider_config import ProviderConfig, load_provider_config

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'reports/semantic-routing/model-call-budget.json'
CASES = [
    ('jordan', '乔丹1997-1998季后赛总得分？', 'Michael Jordan', '893', '1997-98', 'playoffs', {}, 680, 21, 'conditional'),
    ('curry', '库里 2023-24 常规赛总得分？', 'Stephen Curry', '201939', '2023-24', 'regular_season', {}, 1956, 74, 'validated'),
    ('jordan_cutoff', 'Michael Jordan 1997-98 季后赛截至1998-06-01总得分？', 'Michael Jordan', '893', '1997-98', 'playoffs', {'date_to': '1998-06-01'}, 479, 15, 'conditional'),
    ('curry_cutoff', '库里 2023-24 常规赛截至2024-01-01总得分？', 'Stephen Curry', '201939', '2023-24', 'regular_season', {'date_to': '2024-01-01'}, 819, 30, 'validated'),
    ('curry_lakers', '库里 2023-24 常规赛面对湖人的总得分？', 'Stephen Curry', '201939', '2023-24', 'regular_season', {'opponent_team_id': '1610612747'}, 132, 4, 'validated'),
]


class ReplayPlanModel:
    model = 'injected-plan-not-live-provider'

    def __init__(self, tool, arguments):
        self.tool, self.arguments = tool, arguments

    def generate_tools(self, request):
        return ModelResponse('', self.model, [{'id': 'scope_smoke', 'type': 'function',
            'function': {'name': self.tool, 'arguments': json.dumps(self.arguments, ensure_ascii=False)}}])


async def run(args):
    from app.main import app as existing, create_app
    from httpx import ASGITransport, AsyncClient
    config = load_provider_config()
    database = args.db or Path(config.database_path)
    search = existing.state.agent_registry.get('knowledge_glossary').search
    live = CallBudgetModel(OpenAICompatibleModel(config=config), ledger=LEDGER) if args.live else None
    before = live.used if live else json.loads(LEDGER.read_text())['used']
    if live and 200 - before < 8:
        raise ValueError('live smoke requires eight remaining calls to bound four requests plus repairs')
    records = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        for case_id, question, name, identity, season, phase, filters, points, count, quality in CASES[:2] if live else CASES:
            tool = 'window_query' if filters else 'season_summary'
            expected = {'entity': identity, 'season': season, 'phase': phase, 'metrics': ['points_total'], **filters}
            oracle = sql_oracle(connection, expected, tool)['points_total']
            table_count = connection.execute('SELECT games_appeared FROM player_season_stat WHERE person_id=? AND season_id=? AND competition_phase=?', (identity, season, phase)).fetchone()[0]
            arguments = {'subject': 'player', 'entity': {'mention': '乔丹' if case_id == 'jordan' else 'Michael Jordan' if case_id.startswith('jordan') else '库里', 'canonical_name': name},
                         'season_id': season, 'competition_phase': phase, 'metrics': ['points_total'],
                         **{k: v for k, v in filters.items() if k != 'opponent_team_id'}}
            if filters.get('opponent_team_id'):
                arguments['opponent'] = {'mention': '湖人'}
            for endpoint in ('/answer', '/agent/answer'):
                for mode in ('model',) if live else ('model', 'rules'):
                    model = live or ReplayPlanModel(tool, arguments)
                    app = create_app(db_path=database, provider_config=config if live else ProviderConfig.empty(),
                        model=model, routing_mode=mode, knowledge_search=search,
                        ask_total_timeout_seconds=60, ask_model_timeout_seconds=15)
                    # Rule baseline needs the database name; native routing exercises Chinese names.
                    actual_question = question.replace('乔丹', name) if mode == 'rules' else question
                    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://scope-smoke') as client:
                        body = (await client.post(endpoint, json={'question': actual_question})).json()
                    result = body.get('tool_result') or next((s['result'] for s in body.get('steps', []) if s.get('result')), {})
                    evidence = body.get('evidence') or {}
                    if isinstance(evidence, list):
                        evidence = evidence[0].get('structured') or {} if evidence else {}
                    actual = result.get('points_total', (result.get('values') or {}).get('points_total'))
                    actual_count = evidence.get('sample_count')
                    checks = {'complete': body.get('status') == 'complete', 'raw_sql_oracle': actual == oracle,
                              'snapshot_expectation': oracle == points, 'sample_count': actual_count == count,
                              'unfiltered_count': bool(filters) or actual_count == table_count,
                              'quality': result.get('quality_status') == quality,
                              'sql_evidence': bool(evidence.get('source_sql'))}
                    records.append({'case': case_id, 'endpoint': endpoint, 'routing_mode': mode,
                                    'question': actual_question, 'oracle_points': oracle, 'checks': checks,
                                    'passed': all(checks.values()), 'response': body})
    sources = ('scripts/query_scope.py', 'scripts/query_plan.py', 'scripts/ask_grounding.py',
               'app/semantic_planner.py', 'app/agent_routes.py', 'static/demo.js')
    report = {'kind': 'live_provider_smoke' if live else 'real_database_injected_plan_and_rules_smoke',
              'limitations': 'Dedicated known regression cases; not a new blind benchmark or independent historical completeness verification.',
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
              'model': live.model if live else ReplayPlanModel.model, 'calls_before': before,
              'calls_after': live.used if live else before, 'model_requests': live.records if live else [],
              'passed': sum(r['passed'] for r in records), 'total': len(records), 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('kind', 'passed', 'total', 'calls_before', 'calls_after')}))
    return 0 if report['passed'] == report['total'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--live', action='store_true', help='Four original-query requests, at most eight provider calls.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; preserve previous reports')
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    main()
