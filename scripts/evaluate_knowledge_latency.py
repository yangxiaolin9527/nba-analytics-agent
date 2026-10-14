"""Small, paired runtime diagnosis. Frozen historical benchmarks stay untouched."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import threading
from time import perf_counter

from scripts.evaluate_semantic_routing import CallBudgetModel
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import DEFAULT_ENV_PATH, load_provider_config

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {'planner': 10, 'embedding': 12, 'retrieval': 24,
           'generation': 10, 'verification': 10, 'total': 30}


def available_run_budget(balance, run_limit):
    maximum = balance['maximum']
    return run_limit if maximum is None else min(run_limit, maximum - balance['used'] - 4)


class RunBudgetModel:
    """Durable cumulative cap plus a per-run cap; failed attempts also count."""
    def __init__(self, model, *, ledger, run_limit=36):
        maximum = json.loads(ledger.read_text())['maximum']
        self.budget = CallBudgetModel(model, maximum=maximum, ledger=ledger)
        self.model, self.base_url = model.model, model.base_url
        self.initial_used, self.run_limit = self.budget.used, run_limit
        self.lock, self.attempted = threading.Lock(), 0

    def generate(self, request):
        with self.lock:
            if self.attempted >= self.run_limit:
                raise RuntimeError('knowledge_latency_run_budget_exceeded')
            self.attempted += 1
        return self.budget.generate(request)


def read_cases():
    cases = []
    for name, identifier, scenario in [
        ('m24_revised_v1', 'M24-001', 'ordinary_rules'),
        ('m22rlc_v1', 'M22RLC-R09', 'conditional_rules'),
        ('m25_holdout_v1', 'M25-003', 'simple_glossary'),
        ('m24_revised_v1', 'M24-011', 'complex_multifact'),
    ]:
        path = ROOT / f'metadata/knowledge_answer_cases_{name}.json'
        case = next(row for row in json.loads(path.read_text())['cases'] if row['id'] == identifier)
        cases.append({**case, 'scenario': scenario, 'label_source': str(path.relative_to(ROOT))})
    cases.append({'id': 'LATENCY-THREE-SECONDS', 'scenario': 'observed_timeout',
                  'question': 'NBA的进攻三秒和防守三秒规则是什么？',
                  'evaluation_role': 'diagnostic_only; not a new scored benchmark'})
    cases.append({'id': 'LATENCY-MIXED', 'scenario': 'mixed_agent',
                  'question': '库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么？',
                  'evaluation_role': 'rules routing + real structured and knowledge tools'})
    return cases


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)] if values else None


def summarize(records):
    summaries = {}
    for arm in ('baseline', 'candidate'):
        rows = [r for r in records if r['arm'] == arm]
        if not rows:
            continue
        calls = [c for row in rows for c in row.get('model_requests', [])]
        tokens = [c['usage']['total_tokens'] for c in calls
                  if isinstance((c.get('usage') or {}).get('total_tokens'), int)]
        latencies = [r['latency_ms'] for r in rows]
        answered = [r for r in rows if r['result'].get('status') == 'answered']
        summaries[arm] = {'requests': len(rows), 'answered': len(answered),
            'runtime_failures': sum(bool(r['result'].get('error_code')) for r in rows),
            'timeout_count': sum(str(r['result'].get('error_code', '')).endswith('timeout') for r in rows),
            'p50_total_ms': median(latencies), 'p95_total_ms': percentile(latencies, .95),
            'p50_success_total_ms': median([r['latency_ms'] for r in answered]) if answered else None,
            'p50_generation_ms': median([r['result']['generation_ms'] for r in answered]) if answered else None,
            'model_requests': len(calls), 'requests_with_usage': len(tokens),
            'observed_total_tokens': sum(tokens) if tokens else None, 'retry_count': 0}
    return summaries


def promotion_gate(records):
    by_arm = {arm: {r['case_id']: r for r in records if r['arm'] == arm}
              for arm in ('baseline', 'candidate')}
    paired = set(by_arm['baseline']) & set(by_arm['candidate'])
    regressions, complete = [], []
    for identifier in paired:
        a, b = (by_arm[arm][identifier] for arm in ('baseline', 'candidate'))
        if a['result'].get('status') == 'answered':
            if b['result'].get('status') != 'answered':
                regressions.append(identifier)
            else:
                complete.append((a, b))
    generation_gain = None
    if complete:
        before = median([a['result']['generation_ms'] for a, _ in complete])
        after = median([b['result']['generation_ms'] for _, b in complete])
        generation_gain = (before - after) / before if before else None
    summaries = summarize(records)
    enough = len(complete) >= 2 and len(paired) == len(by_arm['baseline']) == len(by_arm['candidate'])
    p95_ok = enough and summaries['candidate']['p95_total_ms'] <= 1.05 * summaries['baseline']['p95_total_ms']
    compound = by_arm['candidate'].get('LATENCY-THREE-SECONDS', {}).get('result', {})
    compound_recovered = (by_arm['baseline'].get('LATENCY-THREE-SECONDS', {}).get('result', {}).get('error_code') == 'generation_timeout'
                          and compound.get('status') == 'answered')
    passed = bool(enough and not regressions and p95_ok and ((generation_gain or 0) >= .10 or compound_recovered))
    return {'latency_gate_passed': passed, 'promotion_ready': False,
            'generation_gain_on_joint_successes': generation_gain,
            'success_to_failure_regressions': sorted(regressions),
            'joint_success_pairs': len(complete), 'total_p95_gate_passed': bool(p95_ok),
            'quality_review': 'pending; citation checks and verifier verdicts are not human semantic grading'}


async def run(args):
    if args.output.exists():
        raise ValueError('refusing to overwrite an existing report')
    config = load_provider_config(args.config)
    config.require_model()
    config.require_embedding()
    ledger = ROOT / 'reports/semantic-routing/model-call-budget.json'
    balance = json.loads(ledger.read_text())
    cap = available_run_budget(balance, args.run_call_limit)
    if cap <= 0:
        raise ValueError('no call budget remains after reserving four requests')
    model = RunBudgetModel(OpenAICompatibleModel(config=config), ledger=ledger, run_limit=cap)
    # Import the app once: local Qdrant permits only one owner of this index.
    from app.main import app as default_app, create_app
    from app.agent_routes import KnowledgeGlossaryAdapter
    from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_observation import RequestObservation
    from scripts.knowledge_search import configured_search
    from httpx import ASGITransport, AsyncClient
    search = default_app.state.agent_registry.get('knowledge_glossary').search
    if args.config.resolve() != DEFAULT_ENV_PATH.resolve():
        if search is not None and search.dense is not None:
            search.dense.close()
        search = configured_search(provider_config=config)
    if search is None or search.dense is None:
        raise ValueError('configured knowledge dense index unavailable')
    cases = read_cases()
    if args.case_id:
        cases = [case for case in cases if case['id'] in args.case_id]
    records = []
    index = Path(config.knowledge_index_path)
    payload = {'schema_version': 1, 'experiment': 'knowledge-generation-thinking-paired-v2',
        'evaluation_role': 'small diagnostic, one run per arm; not a blind holdout',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'tracked_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for directory in ('app', 'scripts') for path in sorted((ROOT / directory).rglob('*.py'))},
        'provider_config': config.summary(), 'timeout_budgets_seconds': BUDGETS,
        'cases': cases, 'case_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'index_hashes': {name: hashlib.sha256((index / name).read_bytes()).hexdigest() for name in ('nodes.json', 'dense.json')},
        'controls': {'concurrency': 1, 'retries': 0, 'inter_case_delay_seconds': args.inter_case_delay,
                     'mixed_routing': 'rules; no outer model planner call'},
        'call_budget': {'initial_used': model.initial_used, 'run_limit': cap, 'reserve': 4},
        'records': records, 'complete': False}

    def checkpoint():
        payload['summary'] = summarize(records)
        payload['decision'] = promotion_gate(records)
        payload['call_budget']['final_used'] = model.budget.used
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

    try:
        for position, case in enumerate(cases):
            # Alternate order to limit systematic provider/caching order effects.
            arms = ('baseline', 'candidate') if position % 2 == 0 else ('candidate', 'baseline')
            for arm in arms:
                observation = RequestObservation()
                setup_started = perf_counter()
                guided = GuidedKnowledgeSearch(search, model, planner_timeout=10, embedding_timeout=12, observer=observation)
                answerer = KnowledgeAnswer(search, model, guided=guided, retrieval_timeout=24,
                    generation_timeout=10, verification_timeout=10, total_timeout=30,
                    generation_thinking='disabled' if arm == 'candidate' else 'enabled', observer=observation)
                application = None
                if case['scenario'] == 'mixed_agent':
                    application = create_app(db_path=Path(config.database_path) if config.database_path else None, knowledge_search=search,
                        provider_config=config, routing_mode='rules', ask_total_timeout_seconds=60,
                        agent_knowledge_adapter=KnowledgeGlossaryAdapter(search, answerer=answerer))
                setup_ms = (perf_counter() - setup_started) * 1000
                start_calls = len(model.budget.records)
                started = perf_counter()
                body = None
                try:
                    if application is not None:
                        async with AsyncClient(transport=ASGITransport(app=application), base_url='http://evaluation') as client:
                            response = await client.post('/agent/answer', json={'question': case['question'], 'routing_mode': 'rules'})
                        body = response.json()
                        result = next(step['result'] for step in body['steps'] if step['capability'] == 'knowledge_glossary')
                        if result.get('status') == 'failed':
                            result = {**result, **result.get('timings', {}), 'error_code': result['reason_code']}
                    else:
                        result = await answerer.aanswer(case['question'])
                except KnowledgeAnswerError as exc:
                    result = {'status': 'failed', 'error_code': exc.reason_code, 'error_stage': exc.stage,
                              **exc.timings, 'model_calls': getattr(exc, 'model_calls', {})}
                elapsed = round((perf_counter() - started) * 1000, 3)
                observation.close()
                records.append({'case_id': case['id'], 'scenario': case['scenario'], 'arm': arm,
                    'generation_thinking': 'disabled' if arm == 'candidate' else 'provider_default',
                    'latency_ms': elapsed, 'setup_ms': round(setup_ms, 3), 'result': result,
                    'agent_response': body, 'observation': observation.snapshot(),
                    'model_requests': model.budget.records[start_calls:], 'retry_count': 0})
                checkpoint()
                print(f"{case['id']} {arm} {result.get('status')} {result.get('error_code', '')} {elapsed:.0f}ms", flush=True)
                await asyncio.sleep(args.inter_case_delay)
        payload['complete'] = True
    finally:
        checkpoint()
        search.dense.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-call-limit', type=int, default=36)
    parser.add_argument('--inter-case-delay', type=float, default=1)
    parser.add_argument('--case-id', action='append', choices=[case['id'] for case in read_cases()],
                        help='Run only these fixed cases into a new report; never resume/overwrite archived rows')
    args = parser.parse_args()
    if args.run_call_limit <= 0 or args.inter_case_delay < 0:
        parser.error('call limit must be positive and delay non-negative')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
