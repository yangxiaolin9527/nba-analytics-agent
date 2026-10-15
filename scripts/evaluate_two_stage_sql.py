"""Isolated paired experiment: preflight, live run, offline replay. No product wiring."""
from __future__ import annotations
import argparse,asyncio,hashlib,json,sqlite3,time,statistics
from dataclasses import asdict
from pathlib import Path
from collections import Counter

from scripts.two_stage_sql.cases import MANIFEST,read_cases,gold_request
from scripts.two_stage_sql.contracts import GroundedRequest
from scripts.two_stage_sql.oracle import reference_result
from scripts.two_stage_sql.evaluation import score_candidate,requirements_equal,adapt_agent,hybrid_arm,saved_fidelity

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def request_from_dict(data):
    if not data:return None
    return GroundedRequest(**{**data,'metrics':tuple(data['metrics']),'predicates':tuple(tuple(x) for x in data.get('predicates',[]))})


def expected_for(c,case):
    return case['gold'] if 'status' in case['gold'] else reference_result(c,gold_request(case))


def implementation_hashes():
    files=list((ROOT/'scripts').rglob('*.py'))+list((ROOT/'app').rglob('*.py'))+list((ROOT/'metadata').glob('*.json'))+[Path(__file__),ROOT/'metadata/two_stage_sql_v1.json',ROOT/'tests/test_two_stage_sql.py',ROOT/'data/profile/eoin-v515-capability-matrix.json']
    files += [ROOT/p for p in ['app/semantic_planner.py','app/entity_resolution.py','app/agent_routes.py','app/agent_tasks.py','app/agent_decider.py','app/agent_loop.py','app/ask_tools.py','scripts/model_adapter.py','scripts/sql_gateway.py','scripts/query_quality.py','scripts/provider_config.py','scripts/query_scope.py','scripts/query_lexicon.py','metadata/schema_catalog.json','metadata/query_lexicon.json'] if (ROOT/p).exists()]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def fingerprint(c,db):
    from scripts.query_quality import read_database_snapshot
    return {'path':str(db.resolve()),'size_bytes':db.stat().st_size,'mtime_ns':db.stat().st_mtime_ns,'snapshot':read_database_snapshot(c),'source_files':c.execute('SELECT file_name,file_sha256,row_count FROM source_file ORDER BY file_name').fetchall()}


def summarize(records):
    def aggregate(rows):
        if not rows:return {}
        calls=[call for row in rows for call in row.get('model_requests',[])]
        usage=[call.get('usage') or {} for call in calls]
        times=sorted(row['latency_ms'] for row in rows)
        return {'turns':len(rows),'passed':sum(r['score']['passed'] for r in rows),'accepted_wrong':sum(r['score']['accepted_wrong'] for r in rows),'requirements_matched':sum(r['score']['requirements_match'] for r in rows),'mean_latency_ms':round(statistics.mean(times),3),'p50_latency_ms':round(statistics.median(times),3),'p95_latency_ms':round(times[min(len(times)-1,int(.95*len(times)))],3),'model_calls':len(calls),'calls_with_usage':sum(bool(x) for x in usage),'prompt_tokens':sum(x.get('prompt_tokens',0) for x in usage),'completion_tokens':sum(x.get('completion_tokens',0) for x in usage),'cache_hit_tokens':sum(x.get('prompt_cache_hit_tokens',0) for x in usage),'cache_miss_tokens':sum(x.get('prompt_cache_miss_tokens',0) for x in usage),'statuses':dict(Counter(r['candidate']['status'] for r in rows))}
    summary={arm:{'all':aggregate([r for r in records if r['arm']==arm]),**{group:aggregate([r for r in records if r['arm']==arm and r['group']==group]) for group in ['common','expanded','protection']}} for arm in ['A','B','C','hybrid']}
    expanded=[r for r in records if r['arm']=='C' and r['group']=='expanded']
    ids={r['case_id'] for r in expanded}
    both=sum(all(r['score']['passed'] for r in expanded if r['case_id']==i) and sum(r['case_id']==i for r in expanded)==2 for i in ids)
    required={(case['id'],repeat,arm) for case in read_cases()['cases'] if case['split']=='frozen' for repeat in range(2) for arm in ['A','B','C','hybrid']}
    actual=[(r['case_id'],r['repeat'],r['arm']) for r in records]
    complete=len(actual)==len(required) and set(actual)==required
    summary['research_gate']={'expanded_scenarios_both_repeats_correct':both,'expanded_scenarios':len(ids),'zero_accepted_wrong':not any(r['score']['accepted_wrong'] for r in records if r['arm']=='C'),'complete_frozen_run':complete,'met':complete and len(ids)==8 and both>=7 and not any(r['score']['accepted_wrong'] for r in records if r['arm']=='C'),'meaning':'eligibility for further study only; never deployment authorization'}
    return summary


def record(arm,case,repeat,candidate,request,expected,latency,calls,**extra):
    wanted=None if 'status' in case['gold'] else gold_request(case)
    fidelity=requirements_equal(request,wanted) if wanted else request is None
    return {'case_id':case['id'],'split':case['split'],'group':case['group'],'question':case['question'],'repeat':repeat,'arm':arm,'candidate':candidate,'grounded_request':asdict(request) if request else None,'expected':expected,'score':score_candidate(candidate,expected,requirements_match=fidelity),'latency_ms':round(latency,3),'model_requests':calls,**extra}


async def run(args,config):
    from scripts.two_stage_sql.modeling import Meter,parse_requirement,generate_sql
    from scripts.two_stage_sql.runtime import execute_tool,execute_sql
    from scripts.model_adapter import OpenAICompatibleModel
    # Only this process: importing app.main otherwise opens a local RAG index and tracer.
    # Structured benchmark never uses RAG; no production file/config is changed.
    import scripts.knowledge_search,scripts.tracing
    scripts.knowledge_search.configured_search=lambda **kw:None
    scripts.tracing.configured_tracer=lambda *a,**kw:None
    from app.main import create_app
    from httpx import AsyncClient,ASGITransport
    args.output.mkdir(parents=True,exist_ok=False)
    model=Meter(OpenAICompatibleModel(config=config),args.output/'calls.jsonl')
    application=create_app(db_path=args.db,model=model,provider_config=config,routing_mode='model',ask_total_timeout_seconds=60,ask_model_timeout_seconds=15,ask_tool_timeout_seconds=2)
    manifest=read_cases(args.cases);cases=[x for x in manifest['cases'] if x['split']==args.split]
    records=[]
    with sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True,check_same_thread=False) as c:
        provenance={'split':args.split,'repeat':2,'version':'offline-hardened-v3','provider':config.summary(),'implementation_sha256':implementation_hashes(),'database':fingerprint(c,args.db),'agent_request_timeout_s':60,'experimental_branch_budget_policy':'per-model 15s and DB 2s; no hard end-to-end wall-clock cancellation','model_timeout_s':15,'db_timeout_s':2,'semantic_stage_max_protocol_repairs':1,'sql_repairs':0,'scoring':'complete payload + executed requirement fidelity + outcome; contract acceptance is not proof','cost_policy':'shared parser counted once in actual calls; B/C/hybrid costs include their attributable parser; no price conversion','limitations':['Author-selected scenarios, two correlated repeats; no statistical significance claim.','Finite request language and AST whitelist; not unrestricted SQL.','Database existence does not prove semantic correctness of proposed translated names.','Numeric payload evaluation, not answer wording/human usefulness evaluation.','RAG/tracing initialized unavailable in this process; structured production contracts unchanged.']}
        dump(args.output/'freeze.json',provenance)
        async with AsyncClient(transport=ASGITransport(app=application),base_url='http://experiment') as client:
            try:
                for index,case in enumerate(cases):
                    for repeat in range(2):
                        expected=None;pair=[]
                        async def agent():
                            model.stage='agent';begin=len(model.records);start=time.perf_counter()
                            try:
                                response=await asyncio.wait_for(client.post('/agent/answer',json={'question':case['question'],'routing_mode':'model'}),timeout=60)
                                body=response.json();candidate,request=adapt_agent(body)
                            except Exception as exc:body={};candidate={'status':'incomplete','accepted':False,'reason':type(exc).__name__};request=None
                            return candidate,request,(time.perf_counter()-start)*1000,model.records[begin:],body
                        order=['A','shared'] if (index+repeat)%2==0 else ['shared','A']
                        for stage in order:
                            if stage=='A':
                                candidate,request,latency,calls,body=await agent()
                                # Gold is used only after runtime candidates have completed.
                                if expected is None:expected=expected_for(c,case)
                                row=record('A',case,repeat,candidate,request,expected,latency,calls,response=body)
                                records.append(row)
                                with (args.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                                print(f"{case['id']} r{repeat+1} A {candidate['status']} pass={row['score']['passed']}",flush=True)
                            else:
                                model.stage='semantic';begin=len(model.records);start=time.perf_counter()
                                try:request,semantic=await asyncio.to_thread(parse_requirement,c,case['question'],model)
                                except Exception as exc:request=None;semantic={'status':'incomplete','accepted':False,'stage':'semantic','reason':type(exc).__name__}
                                parser_ms=(time.perf_counter()-start)*1000;parser_calls=model.records[begin:]
                                branches=['B','C'] if (index+repeat)%2==0 else ['C','B']
                                for arm in branches:
                                    branch_start=time.perf_counter();before=len(model.records)
                                    if request is None:candidate=dict(semantic)
                                    else:
                                        try:
                                            if arm=='B':candidate=execute_tool(c,request)
                                            else:
                                                model.stage='sql';sql=await asyncio.to_thread(generate_sql,c,case['question'],request,model)
                                                candidate=execute_sql(c,request,sql);candidate['generated_sql']=sql
                                        except Exception as exc:candidate={'status':'incomplete','accepted':False,'reason':type(exc).__name__,'stage':'tool' if arm=='B' else 'sql'}
                                    latency=parser_ms+(time.perf_counter()-branch_start)*1000
                                    if expected is None:expected=expected_for(c,case)
                                    row=record(arm,case,repeat,candidate,request,expected,latency,parser_calls+model.records[before:],semantic=semantic,shared_parser_id=f"{case['id']}:{repeat}",parser_ms=round(parser_ms,3))
                                    pair.append(row);records.append(row)
                                    print(f"{case['id']} r{repeat+1} {arm} {candidate['status']} pass={row['score']['passed']} {candidate.get('reason','')}",flush=True)
                                    with (args.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        chosen=hybrid_arm(request_from_dict(pair[0]['grounded_request']))
                        selected=next((r for r in pair if r['arm']==chosen),pair[0])
                        hybrid={**selected,'arm':'hybrid','selected_arm':chosen,'policy':'capability only, never failure fallback'}
                        records.append(hybrid)
                        # Checkpoint complete records after every case/repetition.
                        dump(args.output/'report.json',{'provenance':provenance,'records':records,'summary':summarize(records),'actual_calls':len(model.records)})
            finally:
                dump(args.output/'report.json',{'provenance':provenance,'records':records,'summary':summarize(records),'actual_calls':len(model.records),'complete':len(records)==len(cases)*2*4,'implementation_unchanged':implementation_hashes()==provenance['implementation_sha256']})
    return summarize(records)


def preflight(args):
    manifest=read_cases(args.cases)
    with sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro',uri=True,check_same_thread=False) as c:
        rows=[{'id':case['id'],'group':case['group'],'expected':expected_for(c,case)} for case in manifest['cases']]
        if sum(r['expected']['status']=='complete' for r in rows)!=25:raise ValueError('unexpected computability; inspect preflight results')
        return {'preflight':'passed','scenarios':len(rows),'oracle_statuses':dict(Counter(r['expected']['status'] for r in rows)),'database':fingerprint(c,args.db)}


def replay(args):
    report=json.loads(args.input.read_text())
    rows=report['records'];output=[]
    cases={case['id']:case for case in read_cases(args.cases)['cases']}
    for row in rows:
        if row['arm']=='hybrid':continue
        score=score_candidate(row['candidate'],row['expected'],requirements_match=saved_fidelity(row['grounded_request'],cases[row['case_id']]))
        if score!=row['score']:raise ValueError('score replay drift')
        output.append(row)
    for key in dict.fromkeys((r['case_id'],r['repeat']) for r in output):
        pair=[r for r in output if (r['case_id'],r['repeat'])==key and r['arm'] in {'B','C'}]
        if len(pair)!=2:continue
        chosen=hybrid_arm(request_from_dict(pair[0]['grounded_request']))
        selected=next((r for r in pair if r['arm']==chosen),pair[0]);output.append({**selected,'arm':'hybrid','selected_arm':chosen,'policy':'capability only, never failure fallback'})
    result=summarize(output)
    historical=json.loads(json.dumps(result))
    if 'complete_frozen_run' not in report['summary']['research_gate']:historical['research_gate'].pop('complete_frozen_run')
    if historical!=report['summary']:raise ValueError('summary/policy replay drift')
    return {'replay':'passed','records':len(rows),'summary':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['preflight','run','replay'])
    parser.add_argument('--db',type=Path)
    parser.add_argument('--cases',type=Path,default=MANIFEST)
    parser.add_argument('--split',choices=['development','frozen'],default='development')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    if args.command=='replay':result=replay(args)
    else:
        from scripts.provider_config import load_provider_config
        config=load_provider_config();args.db=args.db or Path(config.database_path)
        if args.command=='preflight':result=preflight(args)
        else:
            if args.output is None:parser.error('--output must be a new run directory')
            result=asyncio.run(run(args,config))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
