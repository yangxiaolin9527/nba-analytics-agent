"""Paired real-data smoke for the rules workflow and grounded model loop."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient
from opentelemetry.trace import NoOpTracerProvider
from app.main import create_app
from app.agent_routes import AgentCapabilityError, KnowledgeGlossaryAdapter
from scripts.evaluation_contract import report_provenance
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig, load_provider_config
from scripts.provider_health import ProviderHealthError, probe_embedding, probe_model, probe_model_json

CASES = ROOT / 'metadata/agent_loop_smoke_v1.json'
SNAPSHOT = 'eoinamoore/historical-nba-data-and-player-box-scores@515'
MATRIX = ROOT / 'data/profile/eoin-v515-capability-matrix.json'
RUNTIME = {'database_unavailable','knowledge_unavailable','knowledge_unconfigured','tool_timeout','tool_failed',
           'adapter_failed','decision_timeout','decision_failed','deadline_exceeded','agent_timeout',
           'generation_timeout','generation_error','verification_timeout','verification_error','retrieval_timeout',
           'retrieval_error','planner_timeout','planner_error','embedding_timeout','embedding_error'}


def load_cases(path: Path = CASES) -> list[dict]:
    document = json.loads(path.read_text())
    cases = document.get('cases')
    if document.get('source_snapshot') != SNAPSHOT or not isinstance(cases, list) or len(cases) != 8:
        raise ValueError('smoke requires eight pinned real-snapshot cases')
    ids = set()
    for c in cases:
        if not c.get('question') or c.get('case_id') in ids or not c.get('case_id'):
            raise ValueError('invalid smoke case')
        ids.add(c['case_id'])
        if {'plan','required_capabilities','timeout'} & c.keys():
            raise ValueError('smoke cases cannot contain runtime plans')
    return cases


def preflight(database: Path, matrix: Path, index: Path | None, *, require_model=True,
              open_index=False, provider_config: ProviderConfig | None = None,
              check_provider: bool = False) -> dict:
    provider_config = provider_config or ProviderConfig.empty()
    database = database.resolve()
    if not database.is_file():
        raise ValueError(f'database unavailable: {database}; build explicitly with uv run python -m scripts.relational_baseline --db {database}')
    if not matrix.is_file():
        raise ValueError(f'capability matrix unavailable: {matrix}')
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as c:
        rows = c.execute('SELECT snapshot_id, source_name, source_version FROM dataset_snapshot').fetchall()
        if rows != [('eoin-v515', SNAPSHOT.split('@')[0], '515')]:
            raise ValueError('database snapshot mismatch')
        files = c.execute('SELECT file_name,file_sha256,row_count FROM source_file ORDER BY file_name').fetchall()
    config = json.loads(matrix.read_text())
    if config.get('source_snapshot') != SNAPSHOT:
        raise ValueError('matrix snapshot mismatch')
    catalog = json.loads((ROOT / 'metadata/schema_catalog.json').read_text())
    if catalog['source_snapshot'] != SNAPSHOT:
        raise ValueError('catalog snapshot mismatch')
    if require_model and not provider_config.model_configured:
        raise ValueError('model configuration unavailable')
    health = None
    if check_provider:
        try:
            health = {'model': probe_model(provider_config),
                      'model_json': probe_model_json(provider_config),
                      'embedding': probe_embedding(provider_config)}
        except ProviderHealthError as exc:
            raise ValueError(f'provider preflight failed: {exc}') from exc
    search = None
    if index is not None:
        if not all((index / k).is_file() for k in ('dense.json','nodes.json')):
            raise ValueError('dense index unavailable')
        dense = json.loads((index / 'dense.json').read_text())
        if not provider_config.embedding_configured and open_index:
            raise ValueError('embedding configuration unavailable')
        if provider_config.embedding_model and dense['model'] != provider_config.embedding_model:
            raise ValueError('embedding model differs from index')
        if open_index:
            from scripts.knowledge_search import configured_search
            try:
                search = configured_search(index, provider_config=provider_config)
                if search is None or search.dense is None:
                    raise ValueError('dense index failed to open')
            except Exception as exc:
                raise ValueError(f'dense index unavailable or locked ({type(exc).__name__})') from exc
    return {'database': str(database), 'bytes': database.stat().st_size, 'snapshot': rows[0],
            'source_files': files, 'matrix': str(matrix.resolve()), 'index': str(index.resolve()) if index else None,
            'search': search, 'provider_health': health}


def build_oracle(database: Path, kind: str | None) -> list[dict]:
    if kind is None:
        return []
    scope = {'entity_id': '201939', 'season_id':'2023-24', 'competition_phase':'regular_season'}
    records = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as c:
        if kind == 'season_summary':
            sql = "SELECT SUM(p.points) FROM player_game_stat p JOIN game g ON g.game_id=p.game_id WHERE p.person_id=? AND g.season_id=? AND g.competition_phase=? AND p.appearance_status IN ('played','played_inferred')"
            params = list(scope.values())
            value = c.execute(sql, params).fetchone()[0]
            return [{'capability':kind,'arguments': {'subject':'player',**scope,'metrics':['points_total']},
                     'values':{'points_total':value},'sql':sql,'parameters':params}]
        if kind == 'window_query':
            sql = "SELECT p.points,g.game_date,p.game_id FROM player_game_stat p JOIN game g ON g.game_id=p.game_id WHERE p.person_id=? AND g.season_id=? AND g.competition_phase=? AND p.appearance_status IN ('played','played_inferred') ORDER BY g.game_date DESC,p.game_id DESC LIMIT 2"
            params = list(scope.values())
            rows = c.execute(sql, params).fetchall()
            return [{'capability':kind,'arguments': {'subject':'player',**scope,'metrics':['points_per_appearance'],'last_n':2},
                     'values':{'values':{'points_per_appearance':sum(r[0] for r in rows)/len(rows)},'actual_sample_count':len(rows)},'sql':sql,'parameters':params}]
        sql = "SELECT SUM(p.field_goals_made),SUM(p.three_pointers_made),SUM(p.field_goals_attempted),COUNT(*) FROM player_game_stat p JOIN game g ON g.game_id=p.game_id WHERE p.person_id=? AND g.season_id=? AND g.competition_phase=? AND p.appearance_status IN ('played','played_inferred')"
        for entity in (['201939','2544'] if kind == 'efg_compare' else ['201939']):
            args = {**scope,'entity_id':entity}
            params = [args[k] for k in ('entity_id','season_id','competition_phase')]
            fgm, threes, fga, n = c.execute(sql, params).fetchone()
            records.append({'capability':'player_efg','arguments':args,
                'values': {'efg_percentage':(fgm+.5*threes)/fga*100,'inputs':{'field_goals_made':fgm,'three_pointers_made':threes,'field_goals_attempted':fga,'sample_count':n}},
                'sql':sql,'parameters':params})
        if kind == 'efg_compare':
            left,right = [r['values']['efg_percentage'] for r in records]
            records.append({'capability':'efg_compare','arguments':{'left_entity_id':'201939','right_entity_id':'2544','season_id':'2023-24','competition_phase':'regular_season'},
                'values':{'left_efg_percentage':left,'right_efg_percentage':right,'absolute_delta':left-right}, 'formula':'left - right, percentage points'})
    return records


def same(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and same(actual[k],v) for k,v in expected.items())
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):
        return isinstance(actual,(float,int)) and not isinstance(actual,bool) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-9)
    return actual == expected


def step_arguments(body, step):
    if isinstance(step.get('arguments'),dict):
        return step['arguments']
    planned = ((body.get('task_state') or {}).get('route') or {}).get('steps',[])
    return next((s['arguments'] for s in planned if s.get('id') == step.get('id')), {})


def score_response(case, body, oracle):
    codes = {body.get('reason_code'),body.get('termination_reason')}
    recovered = set()
    latest = {}
    for s in body.get('steps', []):
        key = (s.get('capability'), json.dumps(step_arguments(body, s), sort_keys=True))
        latest[key] = s
    for s in body.get('steps',[]):
        code = (s.get('result') or {}).get('reason_code')
        key = (s.get('capability'), json.dumps(step_arguments(body, s), sort_keys=True))
        if latest[key].get('status') == 'complete' and s.get('status') != 'complete' and code in RUNTIME:
            recovered.add(code)
        else:
            codes.add(code)
    latest_obs = {o.get('requirement_id', o.get('id')): o for o in body.get('observations', [])}
    for o in latest_obs.values():
        codes.add(o.get('reason_code'))
    runtime = codes & RUNTIME
    if case.get('scenario') == 'injected_knowledge_unavailable':
        runtime -= {'knowledge_unavailable','knowledge_step_failed'}
    if runtime:
        return {'outcome':'provider_runtime','checks':{},'reason':sorted(runtime)[0]}
    if not isinstance(body.get('status'),str) or not isinstance(body.get('route'),str):
        return {'outcome':'evaluation_contract','checks':{},'reason':'malformed_response'}
    expected = case['expected']
    checks = {'status':body['status']==expected['status'], 'route':body['route']==expected['route']}
    if case.get('scenario') == 'injected_knowledge_unavailable':
        checks['injected_failure_observed'] = any(
            s.get('capability') == 'knowledge_glossary' and s.get('status') == 'failed'
            and (s.get('result') or {}).get('reason_code') == 'knowledge_unavailable'
            for s in body.get('steps', []))
        checks['valid_partial_termination'] = body.get('termination_reason') in {
            None, 'coverage_incomplete', 'finish', 'duplicate_tool_call', 'invalid_arguments',
            'unjustified_unsupported',
        }
    if expected.get('reason_code'):
        checks['reason_code'] = body.get('reason_code') == expected['reason_code']
    if expected.get('no_steps'):
        checks['no_steps'] = not body.get('steps') and not body.get('tool_calls')
    numeric, evidence = [], []
    matched = set()
    for o in oracle:
        candidates = [(i,s) for i,s in enumerate(body.get('steps',[])) if i not in matched and s.get('status') == 'complete' and s.get('capability')==o['capability'] and same(step_arguments(body,s),o['arguments'])]
        if not candidates:
            numeric.append(False); evidence.append(False); continue
        i,s = candidates[0]; matched.add(i)
        numeric.append(s.get('status')=='complete' and same(s.get('result'),o['values']))
        ev = (s.get('evidence') or {}).get('structured') or {}
        evidence.append(bool(ev.get('formula')) if o['capability']=='efg_compare' else bool(ev.get('source_sql') and ev.get('snapshot_id')))
    if oracle:
        checks['numeric_fidelity'] = all(numeric)
        checks['structured_evidence'] = all(evidence)
        checks['answer_present'] = bool(body.get('answer'))
    if expected.get('knowledge_required'):
        steps = [s for s in body.get('steps',[]) if s.get('capability')=='knowledge_glossary']
        checks['knowledge_status'] = bool(steps) and steps[-1].get('status')=='complete'
        citations = ((steps[-1].get('evidence') or {}).get('knowledge') or {}).get('citations',[]) if steps else []
        checks['citations'] = any(c.get('citation_id') and c.get('excerpt') for c in citations)
    passed = all(checks.values())
    return {'outcome':'passed' if passed else 'failed','checks':checks,'reason':None if passed else next(k for k,v in checks.items() if not v),
            'recovered_runtime_codes': sorted(recovered)}


def summarize_pairs(records):
    arms = {}
    for arm in ('rules','loop'):
        rows = [r for r in records if r['arm']==arm]
        natural = [r for r in rows if not r.get('scenario')]
        capability = [r for r in rows if r['outcome'] not in {'provider_runtime','evaluation_contract'}]
        arms[arm] = {'total':len(rows),'passed':sum(r['outcome']=='passed' for r in rows),
                     'full_request_denominator':len(rows), 'capability_denominator':len(capability),
                     'natural_total':len(natural),'natural_runtime_failures':sum(r['outcome']=='provider_runtime' for r in natural),
                     'contract_failures':sum(r['outcome']=='evaluation_contract' for r in rows),
                     'mean_latency_ms':round(sum(r['latency_ms'] for r in rows)/len(rows),3) if rows else None}
        arms[arm]['full_request_success_rate'] = arms[arm]['passed']/len(rows) if rows else None
        arms[arm]['capability_success_rate'] = sum(r['outcome']=='passed' for r in capability)/len(capability) if capability else None
    by_case = {}
    for r in records:
        by_case.setdefault(r['case_id'],{})[r['arm']] = r['outcome']=='passed'
    pairs = [p for p in by_case.values() if set(p)=={'rules','loop'}]
    return {'arms':arms,'paired':{'total':len(pairs),'both_passed':sum(p['rules'] and p['loop'] for p in pairs),
            'rules_only_passed':sum(p['rules'] and not p['loop'] for p in pairs),
            'loop_only_passed':sum(p['loop'] and not p['rules'] for p in pairs)}}


def summarize_repeats(reports: list[dict]) -> dict:
    """Aggregate repeated runs without treating them as independent cases."""
    summaries = [report['summary'] for report in reports]
    result = {'repeats': len(reports), 'runs': summaries}
    for arm in ('rules', 'loop'):
        rates = [item['arms'][arm]['full_request_success_rate'] for item in summaries]
        latencies = [item['arms'][arm]['mean_latency_ms'] for item in summaries]
        result[arm] = {
            'full_request_success_rates': rates,
            'full_request_success_rate_min': min(rates) if rates else None,
            'full_request_success_rate_max': max(rates) if rates else None,
            'mean_latency_ms': round(sum(latencies) / len(latencies), 3) if latencies else None,
        }
    result['paired'] = {
        'both_passed': [item['paired']['both_passed'] for item in summaries],
        'rules_only_passed': [item['paired']['rules_only_passed'] for item in summaries],
        'loop_only_passed': [item['paired']['loop_only_passed'] for item in summaries],
    }
    return result


class InjectedUnavailable:
    name, route = 'knowledge_glossary', 'knowledge'
    async def execute(self, question, arguments):
        raise AgentCapabilityError('knowledge_unavailable', 'Injected knowledge unavailable')


class ScenarioKnowledge:
    name, route = 'knowledge_glossary', 'knowledge'
    def __init__(self, real):
        self.real, self.injected = real, False
    async def execute(self, question, arguments):
        if self.injected:
            return await InjectedUnavailable().execute(question, arguments)
        return await self.real.execute(question, arguments)


class RecordingModel:
    """Count calls and hash prompts without storing prompts or credentials."""
    def __init__(self, model):
        self.inner, self.model = model, model.model
        self.records = []
    def generate(self, request):
        record = {'prompt_sha256':hashlib.sha256(request.prompt.encode()).hexdigest(), 'timeout':request.timeout,
                  'token_usage':None, 'cost':None}
        self.records.append(record)
        started = time.monotonic()
        try:
            return self.inner.generate(request)
        except Exception as exc:
            # Never store exception strings: provider bodies/URLs can contain secrets.
            chain, seen = [], set()
            while exc is not None and id(exc) not in seen and len(chain) < 5:
                seen.add(id(exc))
                entry = {'type': type(exc).__name__}
                if isinstance(exc, HTTPError):
                    entry['http_status'] = exc.code
                elif isinstance(exc, URLError):
                    entry['reason_type'] = type(exc.reason).__name__
                    entry['errno'] = getattr(exc.reason, 'errno', None)
                chain.append(entry)
                exc = exc.__cause__
            record['error_chain'] = chain
            raise
        finally:
            record['elapsed_ms'] = round((time.monotonic()-started)*1000,3)


async def run_paired(database: Path, matrix: Path, index: Path, *, cases_path=CASES,
                     output=None, model=None, search=None,
                     provider_config: ProviderConfig | None = None):
    cases = load_cases(cases_path)
    provider_config = provider_config or (load_provider_config() if model is None else ProviderConfig.empty())
    resources = preflight(database, matrix, index, require_model=model is None,
                          open_index=search is None, provider_config=provider_config,
                          check_provider=model is None)
    owned_search = resources.pop('search')
    search = search or owned_search
    records = []
    try:
        recorder = RecordingModel(model or OpenAICompatibleModel(timeout=15, config=provider_config))
        tracer = NoOpTracerProvider().get_tracer('loop-smoke')
        knowledge = ScenarioKnowledge(KnowledgeGlossaryAdapter(search,model=recorder,tracer=tracer))
        app = create_app(model=recorder, db_path=database, quality_matrix_path=matrix,
                         knowledge_search=search, provider_config=provider_config,
                         tracer=tracer,agent_knowledge_adapter=knowledge,agent_loop_enabled=True,ask_total_timeout_seconds=60)
        async with AsyncClient(transport=ASGITransport(app=app),base_url='http://smoke') as client:
            for position,case in enumerate(cases):
                oracle = build_oracle(database,case.get('oracle'))
                knowledge.injected = bool(case.get('scenario'))
                # Alternate arm order to disclose and reduce a fixed-order latency effect.
                for arm in (('rules','loop') if position%2==0 else ('loop','rules')):
                    begin = len(recorder.records)
                    started = time.monotonic()
                    try:
                        response = await asyncio.wait_for(client.post('/agent/answer' if arm=='rules' else '/agent/loop',json={'question':case['question']}),61)
                        body, status = response.json(), response.status_code
                        scored = score_response(case,body,oracle)
                    except Exception as exc:
                        body, status = {'status':'incomplete','reason_code':'request_failed'}, 599
                        scored = {'outcome':'provider_runtime','checks':{},'reason':type(exc).__name__}
                    records.append({'case_id':case['case_id'],'arm':arm,'scenario':case.get('scenario'),
                        'question':case['question'],'http_status':status,'latency_ms':round((time.monotonic()-started)*1000,3),
                        **scored, 'oracle':oracle,'response':body,'model_calls':len(recorder.records)-begin,
                        'model_requests':recorder.records[begin:], 'tool_calls':body.get('tool_calls',len(body.get('steps',[]))),
                        'decisions':body.get('decisions',0),'token_usage':None,'cost':None})
                    print(f"{case['case_id']} {arm}: {scored['outcome']} ({scored.get('reason')})",flush=True)
        provenance = report_provenance(root=ROOT,source_snapshot=SNAPSHOT,model_id=recorder.model,route='rules_vs_loop',
                    input_paths={'cases':cases_path,'matrix':matrix,'catalog':ROOT/'metadata/schema_catalog.json',
                        'loop':ROOT/'app/agent_loop.py','tasks':ROOT/'app/agent_tasks.py','prompt':ROOT/'app/agent_decider.py',
                        'nodes':index/'nodes.json','dense':index/'dense.json'})
        report = {'mode':'paired_real_snapshot_smoke','case_set_version':'1.0.0','resources':resources,
                  'provenance': provenance,
                  'final_provenance': {'clean_worktree': provenance.get('git_dirty') is False,
                                       'current_head': provenance.get('git_commit') if provenance.get('git_dirty') is False else None},
                  'provider_host':urlsplit(provider_config.model_base_url or '').hostname,
                  'budgets':{'requests_per_arm':8,'repeats':1,'total_seconds':60,'max_tools':8,'max_decisions':10},
                  'limitations':['Selected smoke, not blind holdout','One run per arm; no statistical significance claim',
                    'Injected knowledge failure reported separately','Rules comparison uses canonical definition question; loop normalizes glossary questions for mixed tasks',
                    'Citation shape is not semantic support; manual review required','Token usage/cost unavailable in existing model adapter'],
                  'summary':summarize_pairs(records),'records':records}
        if output:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
    finally:
        if owned_search is not None:
            owned_search.dense.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--capability-matrix',type=Path,default=MATRIX)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--cases',type=Path,default=CASES)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/agent-loop/paired-smoke-v1.json')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--provider-health', action='store_true',
                        help='probe model and embedding providers during preflight')
    args = parser.parse_args()
    provider_config = load_provider_config()
    database = args.db or Path(provider_config.database_path or '/tmp/nba-m1.sqlite')
    index = args.index or Path(provider_config.knowledge_index_path or str(ROOT/'data/indexes/knowledge-500-dense'))
    try:
        if args.preflight_only:
            resources = preflight(database, args.capability_matrix, index, open_index=True,
                                  provider_config=provider_config,
                                  check_provider=args.provider_health)
            search = resources.pop('search')
            try:
                print(json.dumps(resources,ensure_ascii=False,indent=2))
            finally:
                if search:
                    search.dense.close()
            return 0
        if args.repeats < 1:
            raise ValueError('--repeats must be positive')
        reports = []
        for repeat in range(1, args.repeats + 1):
            report = asyncio.run(run_paired(database, args.capability_matrix, index,
                                            cases_path=args.cases, output=None,
                                            provider_config=provider_config))
            reports.append(report)
            if args.repeats > 1:
                repeat_path = args.output.with_name(f'{args.output.stem}-r{repeat}{args.output.suffix}')
                repeat_path.parent.mkdir(parents=True, exist_ok=True)
                repeat_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        if args.repeats == 1:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2) + '\n')
            summary = reports[0]['summary']
        else:
            aggregate = {'mode': 'paired_real_snapshot_repeated_smoke',
                         'case_set_version': '1.0.0', 'runs': reports,
                         'summary': summarize_repeats(reports)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n')
            summary = aggregate['summary']
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return int(any(r['outcome']!='passed' for report in reports for r in report['records']))
    except (ValueError,OSError,sqlite3.Error) as exc:
        print(f'Preflight/evaluation failed: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
