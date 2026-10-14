"""Dedicated joint-statistic regression smoke, independent of frozen benchmarks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.diagnose_structured_scope import LEDGER, ROOT, ReplayPlanModel
from scripts.evaluate_semantic_routing import CallBudgetModel, sql_oracle
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig, load_provider_config

METRICS = ['points_total', 'assists_total', 'rebounds_total']
CASES = [
    ('joint', '乔丹1996-1997赛季季后赛的总助攻、总篮板和总得分都是多少', METRICS, {}),
    ('assists', '乔丹1996-1997赛季季后赛的总助攻是多少', ['assists_total'], {}),
    ('rebounds', '乔丹1996-1997赛季季后赛的总篮板是多少', ['rebounds_total'], {}),
    ('window', '乔丹1996-1997赛季季后赛最近两场总助攻、总篮板和总得分是多少', METRICS, {'last_n': 2}),
]


async def run(args):
    from app.main import app as existing, create_app
    from httpx import ASGITransport, AsyncClient

    config = load_provider_config()
    database = args.db or Path(config.database_path)
    before = json.loads(LEDGER.read_text())['used']
    maximum = json.loads(LEDGER.read_text())['maximum']
    if args.live and maximum - before < 4:
        raise ValueError('two live requests require four reserved calls including possible plan repairs')
    live = CallBudgetModel(OpenAICompatibleModel(config=config), ledger=LEDGER) if args.live else None
    search = existing.state.agent_registry.get('knowledge_glossary').search
    records = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for case_id, question, metrics, filters in CASES[:1] if args.live else CASES:
            tool = 'window_query' if filters else 'season_summary'
            expected = {'entity': '893', 'season': '1996-97', 'phase': 'playoffs', 'metrics': metrics, **filters}
            oracle = sql_oracle(conn, expected, tool)
            args_for_tool = {'subject': 'player', 'entity': {'mention': '乔丹', 'canonical_name': 'Michael Jordan'},
                            'season_id': '1996-97', 'competition_phase': 'playoffs', 'metrics': metrics, **filters}
            sample_count = sql_oracle(conn, {**expected, 'metrics': ['games_appeared']}, tool)['games_appeared']
            for endpoint in ('/answer', '/agent/answer'):
                for mode in ('model',) if args.live else ('model', 'rules'):
                    model = live or ReplayPlanModel(tool, args_for_tool)
                    app = create_app(db_path=database, model=model, routing_mode=mode,
                        provider_config=config if args.live else ProviderConfig.empty(), knowledge_search=search,
                        ask_model_timeout_seconds=15, ask_total_timeout_seconds=60)
                    actual_question = question if mode == 'model' else question.replace('乔丹', 'Michael Jordan')
                    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://multi-metric-smoke') as client:
                        response = await client.post(endpoint, json={'question': actual_question})
                        body = response.json()
                    result = body.get('tool_result') or next((s['result'] for s in body.get('steps', []) if s.get('result')), {})
                    values = result.get('values', result)
                    evidence = body.get('evidence') or {}
                    if isinstance(evidence, list):
                        evidence = evidence[0].get('structured') or {} if evidence else {}
                    actual_args = body.get('arguments') or next(
                        (s['arguments'] for s in (body.get('task_state') or {}).get('route', {}).get('steps', []) if s['capability'] == tool), {})
                    checks = {'http_ok': response.status_code == 200, 'complete': body.get('status') == 'complete',
                        'raw_sql_oracle': all(values.get(m) == oracle[m] for m in metrics),
                        'one_tool': body.get('tool_calls_used', len(body.get('steps', []))) == 1,
                        'metrics_preserved': set(actual_args.get('metrics', [])) == set(metrics),
                        'sample_count': evidence.get('sample_count') == sample_count,
                        'conditional': result.get('quality_status') == 'conditional',
                        'limitations_preserved': 'season_completeness_not_checked' in (evidence.get('limitations') or []),
                        'sql_evidence': bool(evidence.get('source_sql'))}
                    records.append({'case': case_id, 'endpoint': endpoint, 'mode': mode, 'question': actual_question,
                        'oracle': oracle, 'checks': checks, 'passed': all(checks.values()), 'response': body})
    sources = ['scripts/query_plan.py', 'scripts/query_plan_compiler.py', 'scripts/window_query.py',
               'scripts/ask_grounding.py', 'app/semantic_planner.py', 'app/semantic_contracts.py',
               'app/agent_routes.py', 'app/main_grounding.py', 'app/query_routes.py',
               'scripts/answer_renderer.py', 'scripts/diagnose_multi_metric_summary.py']
    report = {'kind': 'live_provider_joint_statistics_smoke' if args.live else 'real_database_injected_plan_and_rules_smoke',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'limitations': 'Known regression cases; injected plans are not model accuracy; SQL snapshot consistency does not independently verify season completeness.',
        'model': live.model if live else ReplayPlanModel.model,
        'calls_before': before, 'calls_after': live.used if live else before,
        'model_requests': live.records if live else [],
        'passed': sum(r['passed'] for r in records), 'total': len(records), 'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('kind', 'passed', 'total', 'calls_before', 'calls_after')}))
    return 0 if report['passed'] == report['total'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; preserve previous reports')
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    main()
