"""M2.4: one frozen, serial evaluation of the real async answering path."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from scripts.evaluate_knowledge_answer import validate_cases
from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError
from scripts.knowledge_observation import RequestObservation
from scripts.model_adapter import ModelResponse

BUDGETS = {"model_request": 11.0, "planner": 10.0, "embedding": 12.0, "retrieval": 24.0,
           "generation": 10.0, "verification": 10.0, "total_request": 30.0}


def cases_hash(cases):
    return hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class CachedPlanner:
    model = "deepseek-flash"

    def __init__(self, plans):
        self.plans = plans

    def generate(self, request):
        question = request.prompt.rsplit("Question: ", 1)[-1].strip()
        try:
            question = json.loads(question)
        except json.JSONDecodeError:
            pass
        return ModelResponse(json.dumps(self.plans[question], ensure_ascii=False), self.model)


def load_plan_cache(path, cases):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cases_sha256") != cases_hash(cases):
        raise ValueError("planner cache belongs to different cases")
    plans = payload.get("plans")
    if not isinstance(plans, dict) or set(plans) != {case["question"] for case in cases}:
        raise ValueError("planner cache does not cover exactly the selected cases")
    return payload, plans


def execution_order(cases):
    ordered = list(cases)
    random.Random(24).shuffle(ordered)
    return ordered


def _save(path, report):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


async def evaluate_cases(make_answerer, cases, *, output: Path, manifest: dict,
                         planner_latencies: dict[str, float] | None = None,
                         inter_case_delay: float = 2.0) -> dict:
    validate_cases(cases)
    if not cases or inter_case_delay < 0:
        raise ValueError('nonempty cases and nonnegative cadence required')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents accidentally replacing a completed live run.
    with output.open('x', encoding='utf-8') as handle:
        handle.write('{}\n')
    ordered = execution_order(cases)
    report = {'schema_version': 1, 'evaluation_kind': 'production_e2e',
              'run_status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'case_sha256': cases_hash(cases), 'case_count': len(cases),
              'execution_order': [case['id'] for case in ordered],
              'concurrency': 1, 'inter_case_delay_seconds': inter_case_delay,
              'manifest': manifest, 'cases': []}
    _save(output, report)
    try:
        for index, case in enumerate(ordered):
            observation = RequestObservation()
            started = perf_counter()
            interrupted = False
            try:
                answerer = make_answerer(observation)
                result = await asyncio.wait_for(answerer.aanswer(case['question']), answerer.total_timeout)
            except KnowledgeAnswerError as exc:
                result = {'status': 'error', 'claims': [], 'unresolved': [case['question']],
                          'error_code': exc.reason_code, 'error_stage': exc.stage,
                          'error_type': type(exc).__name__, 'timings': exc.timings}
            except TimeoutError:
                result = {'status': 'error', 'claims': [], 'unresolved': [case['question']],
                          'error_code': 'total_timeout', 'error_stage': 'total', 'error_type': 'TimeoutError'}
            except asyncio.CancelledError:
                interrupted = True
                result = {'status': 'error', 'claims': [], 'unresolved': [case['question']],
                          'error_code': 'run_cancelled', 'error_stage': 'runner', 'error_type': 'CancelledError'}
            except Exception as exc:
                result = {'status': 'error', 'claims': [], 'unresolved': [case['question']],
                          'error_code': 'runner_error', 'error_stage': 'runner', 'error_type': type(exc).__name__}
            finally:
                observation.close()
            row = {'id': case['id'], 'question': case['question'], 'language': case['language'],
                   'expected_outcome': case.get('expected_outcome'), 'result': result,
                   'observation': observation.snapshot(),
                   'wall_ms': round((perf_counter() - started) * 1000, 3)}
            preflight_ms = float((planner_latencies or {}).get(case['question'], 0.0))
            row['planner_preflight_ms'] = preflight_ms
            row['estimated_live_endpoint_ms'] = round(row['wall_ms'] + preflight_ms, 3)
            report['cases'].append(row)
            _save(output, report)
            print(f"{case['id']} {result.get('status')} {result.get('error_code') or result.get('reason_code') or ''} {row['wall_ms']:.0f}ms", flush=True)
            if interrupted:
                raise asyncio.CancelledError()
            if index + 1 < len(ordered) and inter_case_delay:
                await asyncio.sleep(inter_case_delay)
        report['run_status'] = 'completed'
    finally:
        if report['run_status'] != 'completed':
            report['run_status'] = 'interrupted'
        elapsed = sorted(row['wall_ms'] for row in report['cases'])
        report['completed_count'] = len(elapsed)
        report['mean_wall_ms'] = round(mean(elapsed), 3) if elapsed else None
        report['p95_wall_ms'] = elapsed[max(0, (95 * len(elapsed) + 99) // 100 - 1)] if elapsed else None
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        _save(output, report)
    return report


def build_manifest(cases_file, index, model, embedding, *, smoke=False):
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/knowledge_guidance.py', 'scripts/knowledge_answer.py',
             'scripts/knowledge_observation.py', 'scripts/evaluate_knowledge_e2e.py',
             'scripts/knowledge_search.py', 'scripts/knowledge_vector.py', 'scripts/model_adapter.py',
             'metadata/knowledge_guided_policy_v1.json', 'uv.lock']
    return {'protocol': 'm24-v1', 'smoke': smoke,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'source_file_hashes': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
            'case_file_sha256': hashlib.sha256(cases_file.read_bytes()).hexdigest(),
            'index_nodes_sha256': hashlib.sha256((index / 'nodes.json').read_bytes()).hexdigest(),
            'index_manifest_hash': json.loads((index / 'nodes.json').read_text())['manifest_hash'],
            'planner_model': model.model, 'embedding_model': embedding.model,
            'chat_endpoint_sha256': hashlib.sha256(model.base_url.encode()).hexdigest(),
            'embedding_provider': embedding.metadata.provider,
            'embedding_dimension': embedding.metadata.dimension,
            'embedding_normalized': embedding.metadata.normalized,
            'embedding_endpoint_sha256': hashlib.sha256(getattr(embedding, 'base_url', embedding.metadata.provider).encode()).hexdigest(),
            'timeout_budgets_seconds': BUDGETS, 'top_k': 5, 'min_dense_score': .43,
            'max_pages': 5, 'max_chunks': 12, 'max_prompt_chars': 24000, 'order_seed': 24,
            'automatic_retries': 0, 'phoenix_enabled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('metadata/knowledge_answer_cases_m24_v1.json'))
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--embedding-provider', choices=('ali_bailian', 'local_sentence_transformers'))
    parser.add_argument('--embedding-model')
    parser.add_argument('--embedding-dimension', type=int, default=1024)
    parser.add_argument('--embedding-normalize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--embedding-device', default=None)
    parser.add_argument('--plan-cache', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; never overwrite a live run')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))['cases']
    if args.ids:
        if not args.smoke or len(args.ids) != len(set(args.ids)):
            parser.error('--ids requires --smoke and unique IDs')
        cases = [case for case in cases if case['id'] in args.ids]
        if len(cases) != len(args.ids):
            parser.error('unknown case ID')
    if not args.smoke:
        from collections import Counter
        counts = Counter(c.get('expected_outcome') for c in cases)
        if counts != {'answer': 20, 'refusal': 9, 'premise_denial': 1}:
            parser.error('formal evaluation requires 20 answer, 9 refusal and 1 premise denial')
    from scripts.knowledge_search import KnowledgeSearch, load_nodes
    from scripts.knowledge_vector import DenseIndex
    from scripts.embedding_adapters import load_embedding_adapter
    from scripts.provider_config import load_provider_config
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.model_adapter import OpenAICompatibleModel
    config = load_provider_config().with_embedding_overrides(
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        embedding_normalize=args.embedding_normalize,
        embedding_local_device=args.embedding_device,
    )
    embedding = load_embedding_adapter(config)
    model = OpenAICompatibleModel(timeout=BUDGETS['model_request'])
    if model.model != 'deepseek-flash':
        parser.error('chat model does not match the frozen M2.4 protocol')
    planner = model
    planner_latencies = {}
    plan_payload = None
    if args.plan_cache:
        plan_payload, plans = load_plan_cache(args.plan_cache, cases)
        planner = CachedPlanner(plans)
        planner_latencies = {key: float(value) for key, value in plan_payload.get("latency_ms", {}).items()}
    dense = DenseIndex.open(args.index, embedding)
    try:
        payload = json.loads((args.index / 'nodes.json').read_text())
        search = KnowledgeSearch(load_nodes(args.index), dense=dense, manifest_hash=payload['manifest_hash'])
        validate_cases(cases, search)
        def make_answerer(observer):
            guided = GuidedKnowledgeSearch(search, planner, planner_timeout=BUDGETS['planner'],
                                           embedding_timeout=BUDGETS['embedding'], min_dense_score=.43,
                                           observer=observer)
            return KnowledgeAnswer(search, model, guided=guided, observer=observer,
                retrieval_timeout=BUDGETS['retrieval'], generation_timeout=BUDGETS['generation'],
                verification_timeout=BUDGETS['verification'], total_timeout=BUDGETS['total_request'])
        # Validate calibrated policy before creating a formal output file.
        make_answerer(None)
        manifest = build_manifest(args.cases, args.index, model, embedding, smoke=args.smoke)
        if plan_payload:
            manifest['planner_cache_sha256'] = hashlib.sha256(args.plan_cache.read_bytes()).hexdigest()
            manifest['planner_cache_model'] = plan_payload.get('planner_model')
        asyncio.run(evaluate_cases(make_answerer, cases, output=args.output, manifest=manifest,
                                   planner_latencies=planner_latencies))
    finally:
        dense.close()


if __name__ == '__main__':
    main()
