"""Real request-time A versus B/C dispatch: preflight, run and audited replay."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time

from scripts.evaluate_two_stage_sql import dump, expected_for, fingerprint, record, request_from_dict
from scripts.two_stage_sql.cases import gold_request, read_cases
from scripts.two_stage_sql.evaluation import adapt_agent, saved_fidelity, score_candidate
from scripts.two_stage_sql.online_worker import WorkerClient

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'metadata/online_hybrid_v1.json'


def read_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    cases = manifest['cases']
    if len(cases)!=24 or len({case['id'] for case in cases})!=24:
        raise ValueError('expected 24 unique online scenarios')
    if Counter(case['group'] for case in cases)!=Counter(common=10,expanded=8,protection=6):
        raise ValueError('online group count mismatch')
    for case in cases:
        if case['split']!='online_holdout':
            raise ValueError('unexpected split')
        if 'status' not in case['gold']:
            gold_request(case)
    return manifest


def aggregate(rows):
    if not rows:
        return {}
    times = sorted(row['latency_ms'] for row in rows)
    calls = [call for row in rows for call in row.get('model_requests',[])]
    usage = [call.get('usage') or {} for call in calls]
    return {'turns':len(rows),'passed':sum(row['score']['passed'] for row in rows),'accepted_wrong':sum(row['score']['accepted_wrong'] for row in rows),'mean_latency_ms':round(statistics.mean(times),3),'p50_latency_ms':round(statistics.median(times),3),'p95_latency_ms':round(times[min(len(times)-1,int(.95*len(times)))],3),'completed_model_calls':len(calls),'calls_with_usage':sum(bool(u) for u in usage),'prompt_tokens':sum(u.get('prompt_tokens',0) for u in usage),'completion_tokens':sum(u.get('completion_tokens',0) for u in usage),'cache_hit_tokens':sum(u.get('prompt_cache_hit_tokens',0) for u in usage),'cache_miss_tokens':sum(u.get('prompt_cache_miss_tokens',0) for u in usage),'statuses':dict(Counter(row['candidate']['status'] for row in rows))}


def summarize_online(rows, manifest):
    result = {arm:{'all':aggregate([r for r in rows if r['arm']==arm]),**{group:aggregate([r for r in rows if r['arm']==arm and r['group']==group]) for group in ['common','expanded','protection']}} for arm in ['A','H']}
    required = {(case['id'],repeat,arm) for case in manifest['cases'] for repeat in range(2) for arm in ['A','H']}
    actual = [(row['case_id'],row['repeat'],row['arm']) for row in rows]
    complete = len(actual)==len(required) and set(actual)==required
    expanded_ids = {case['id'] for case in manifest['cases'] if case['group']=='expanded'}
    stable = sum(len(found:=[r for r in rows if r['arm']=='H' and r['case_id']==case_id])==2 and all(r['score']['passed'] for r in found) for case_id in expanded_ids)
    a,h = result['A']['common'],result['H']['common']
    checks = {'complete_unique_run':complete,'existing_queries_preserved':h.get('passed',0)==20 and h.get('passed',0)>=a.get('passed',0),'protection_passed':result['H']['protection'].get('passed',0)==12,'zero_accepted_wrong':not any(r['score']['accepted_wrong'] for r in rows if r['arm']=='H'),'expanded_at_least_7_of_8_stable':stable>=7,'common_p95_within_25_percent':bool(a and h and h['p95_latency_ms']<=1.25*a['p95_latency_ms']),'all_p95_within_5_seconds':bool(result['H']['all'] and result['H']['all']['p95_latency_ms']<=5000)}
    result['replacement_gate'] = {'checks':checks,'expanded_scenarios_both_correct':stable,'met':all(checks.values()),'meaning':'structured-subscope integration research only; no deployment or whole-Agent replacement'}
    return result


def source_hashes():
    paths = [path for folder in ['app','scripts','metadata'] for path in (ROOT/folder).rglob('*') if path.is_file() and path.suffix in {'.py','.json'}]
    paths += [ROOT/'tests/test_online_hybrid.py',ROOT/'tests/test_two_stage_sql.py',ROOT/'data/profile/eoin-v515-capability-matrix.json',ROOT/'docs/experiments/2026-10-15-online-hybrid-plan.md']
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def archive_sources(output, hashes):
    for name,digest in hashes.items():
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('source changed while freezing')
        target = output/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)


def ledger(output):
    calls = [json.loads(line) for path in sorted(output.glob('calls-*.jsonl')) for line in path.read_text().splitlines()]
    events = [json.loads(line) for path in sorted(output.glob('events-*.jsonl')) for line in path.read_text().splitlines()]
    return {'initiated_calls':sum(event['event']=='started' for event in events),'completed_calls':len(calls),'failed_calls':sum('error' in call for call in calls),'unfinished_calls':sum(event['event']=='started' for event in events)-len(calls)}


def run(args, config):
    manifest = read_manifest(args.cases)
    if args.phase=='regression':
        cases = [case for case in read_cases()['cases'] if case['split']=='development']
        repeats = 1
    else:
        cases = manifest['cases']
        repeats = 2
    args.output.mkdir(parents=True,exist_ok=False)
    workers = {arm:WorkerClient(arm,args.db,args.output) for arm in ['A','H']}
    rows = []
    with sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        hashes = source_hashes()
        provenance = {'version':'online-hybrid-v2-io-audit-hardening','phase':args.phase,'provider':config.summary(),'unified_loop_enabled':config.unified_agent_loop_enabled,'case_manifest':manifest if args.phase=='holdout' else {'cases':cases},'repeat':repeats,'source_hashes':hashes,'database':fingerprint(connection,args.db),'budgets':{'hard_request_seconds':60,'model_seconds':15,'db_seconds':2,'semantic_protocol_repairs':1,'sql_repairs':0},'timing':'parent request preparation/send(question) to full ASGI response receive; ready-worker startup and Oracle excluded and separately recorded','scope':'structured finite language; no RAG/multitarget/cross-season/conversation coverage','selection':'H chooses exactly one branch before execution; no gold/output-based fallback','transport':'process IPC + in-process ASGI; real external provider, not TCP/concurrency benchmark','oracle_policy':'compute only after both runtime candidates complete; no gold reaches workers','usage_policy':'completed calls recovered from durable per-worker logs; unfinished started calls have unknown token usage','gates_fixed_before_calls':{'common_correct':20,'protection_correct':12,'expanded_stable':7,'common_p95_ratio_max':1.25,'all_p95_ms_max':5000}}
        archive_sources(args.output,hashes)
        dump(args.output/'freeze.json',provenance)
        started = time.perf_counter()
        def checkpoint():
            dump(args.output/'report.json',{'provenance':provenance,'records':rows,'summary':summarize_online(rows,manifest),'call_ledger':ledger(args.output),'bootstraps':{arm:worker.bootstraps for arm,worker in workers.items()},'complete':len(rows)==len(cases)*repeats*2,'implementation_unchanged':source_hashes()==hashes,'wall_time_ms':round((time.perf_counter()-started)*1000,3)})
        try:
            for worker in workers.values():
                worker.start()
            for index,case in enumerate(cases):
                for repeat in range(repeats):
                    pair = {}
                    order = ['A','H'] if (index+repeat)%2==0 else ['H','A']
                    for arm in order:
                        pair[arm] = workers[arm].ask(case['question'])
                    oracle_start = time.perf_counter()
                    expected = expected_for(connection,case)
                    oracle_ms = (time.perf_counter()-oracle_start)*1000
                    for arm in order:
                        raw = pair[arm]; body = raw['response']
                        if raw.get('error') or raw['http_status']!=200:
                            candidate,request = {'status':'incomplete','accepted':False,'reason':raw.get('error') or f"HTTP_{raw['http_status']}"},None
                        elif arm=='A':
                            candidate,request = adapt_agent(body)
                        else:
                            candidate,request = body['candidate'],request_from_dict(body['grounded_request'])
                        row = record(arm,case,repeat,candidate,request,expected,raw['latency_ms'],raw['model_requests'],response=body,http_status=raw['http_status'],transport_error=raw.get('error'),service_latency_ms=raw.get('service_latency_ms'),worker_generation=raw['worker_generation'],oracle_ms=round(oracle_ms,3),selected_arm=body.get('selected_arm'))
                        rows.append(row)
                        with (args.output/'records.jsonl').open('a') as stream:
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        print(f"{case['id']} r{repeat+1} {arm} branch={row['selected_arm']} status={candidate['status']} pass={row['score']['passed']} {raw['latency_ms']:.0f}ms {candidate.get('reason') or ''}",flush=True)
                    checkpoint()
        finally:
            for worker in workers.values():
                worker.close()
            checkpoint()
    return json.loads((args.output/'report.json').read_text())['summary']


def verify_record(row, case, connection, *, expected=None):
    expected = expected_for(connection,case) if expected is None else expected
    if expected!=row['expected']:
        raise ValueError('independent oracle reference drift')
    body = row['response']
    if row.get('transport_error') or row['http_status']!=200:
        candidate,request = {'status':'incomplete','accepted':False,'reason':row.get('transport_error') or f"HTTP_{row['http_status']}"},None
    elif row['arm']=='A':
        candidate,request = adapt_agent(body)
    else:
        candidate,request = body['candidate'],request_from_dict(body['grounded_request'])
        from scripts.two_stage_sql.online import choose_branch
        if choose_branch(request)!=row['selected_arm'] or body['selected_arm']!=row['selected_arm']:
            raise ValueError('online routing replay drift')
    if candidate!=row['candidate'] or request!=request_from_dict(row['grounded_request']):
        raise ValueError('raw response and scored candidate differ')
    score = score_candidate(candidate,expected,requirements_match=saved_fidelity(row['grounded_request'],case))
    if row['score']!=score:
        raise ValueError('score replay drift')


def replay(path):
    report = json.loads(Path(path).read_text())
    output = Path(path).parent
    if not report['implementation_unchanged']:
        raise ValueError('implementation changed during provider run')
    if json.loads((output/'freeze.json').read_text())!=report['provenance']:
        raise ValueError('freeze provenance drift')
    stored_rows = [json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    if stored_rows!=report['records']:
        raise ValueError('raw record ledger drift')
    cases = {case['id']:case for case in report['provenance']['case_manifest']['cases']}
    source_manifest = json.loads((output/'source/metadata/online_hybrid_v1.json').read_text())
    manifest = source_manifest if report['provenance']['phase']=='regression' else report['provenance']['case_manifest']
    archived_cases = source_manifest['cases'] if report['provenance']['phase']=='holdout' else [case for case in json.loads((output/'source/metadata/two_stage_sql_v1.json').read_text())['cases'] if case['split']=='development']
    if list(cases.values())!=archived_cases:
        raise ValueError('case labels differ from archived manifest')
    db_path = Path(report['provenance']['database']['path'])
    with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True) as connection:
        expected_by_case = {case_id:expected_for(connection,case) for case_id,case in cases.items()}
        for row in report['records']:
            verify_record(row,cases[row['case_id']],connection,expected=expected_by_case[row['case_id']])
    if summarize_online(report['records'],manifest)!=report['summary']:
        raise ValueError('summary replay drift')
    for name,digest in report['provenance']['source_hashes'].items():
        if hashlib.sha256((output/'source'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('archived source hash mismatch')
    if ledger(output)!=report['call_ledger']:
        raise ValueError('actual calls ledger drift')
    stored_calls = {(path.name.split('-')[1],int(path.stem.split('-')[2]),call['sequence']):call for path in output.glob('calls-*.jsonl') for call in [json.loads(line) for line in path.read_text().splitlines()]}
    attributed_calls = {}
    for row in report['records']:
        for call in row['model_requests']:
            key = (row['arm'],row['worker_generation'],call['sequence'])
            if key in attributed_calls:
                raise ValueError('call attributed more than once')
            attributed_calls[key] = call
    if stored_calls!=attributed_calls:
        raise ValueError('per-request call/token attribution drift')
    expected_keys = {(case_id,repeat,arm) for case_id in cases for repeat in range(report['provenance']['repeat']) for arm in ['A','H']}
    actual_keys = [(row['case_id'],row['repeat'],row['arm']) for row in report['records']]
    if report['complete'] and (len(actual_keys)!=len(expected_keys) or set(actual_keys)!=expected_keys):
        raise ValueError('incomplete or duplicated run')
    return {'replay':'passed','records':len(report['records']),'complete':report['complete'],'call_ledger':report['call_ledger'],'replacement_gate':report['summary']['replacement_gate']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run','replay'])
    parser.add_argument('--db',type=Path)
    parser.add_argument('--cases',type=Path,default=MANIFEST)
    parser.add_argument('--phase',choices=['regression','holdout'],default='holdout')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input',type=Path)
    args = parser.parse_args()
    if args.command=='replay':
        if args.input is None:
            parser.error('--input required')
        result = replay(args.input)
    else:
        from scripts.provider_config import load_provider_config
        config = load_provider_config()
        args.db = (args.db or Path(config.database_path)).resolve()
        if args.command=='preflight':
            manifest = read_manifest(args.cases)
            with sqlite3.connect(args.db.as_uri()+'?mode=ro',uri=True) as connection:
                results = [{'id':case['id'],'expected':expected_for(connection,case)} for case in manifest['cases']]
            result = {'preflight':'passed','scenarios':len(results),'oracle_statuses':dict(Counter(row['expected']['status'] for row in results)),'cases':results}
        else:
            if args.output is None:
                parser.error('--output must be a new directory')
            result = run(args,config)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
