"""Provider telemetry and the two generation stages (no evaluator dependency)."""
import hashlib,json,threading,time
from dataclasses import replace
from scripts.model_adapter import ModelRequest
from scripts.ask_grounding import ground_ask_question
from scripts.two_stage_sql.contracts import AnalysisRequest
from scripts.two_stage_sql.grounding import ground_request,GroundingFailure
from scripts.two_stage_sql.prompts import semantic_prompt,sql_prompt

class Meter:
    def __init__(self,wrapped,sink=None):
        self.wrapped=wrapped;self.records=[];self.lock=threading.Lock();self.sink=sink
        self.model=getattr(wrapped,'model','unknown');self.base_url=getattr(wrapped,'base_url','')
        self.stage='agent'
    def invoke(self,request,native):
        request=replace(request,timeout=min(request.timeout or 15,15),thinking={'type':'disabled'})
        record={'sequence':len(self.records)+1,'stage':self.stage,'kind':'native' if native else 'text','request':{k:v for k,v in request.__dict__.items() if v is not None},'usage':None}
        record['request_sha256']=hashlib.sha256(json.dumps(record['request'],sort_keys=True).encode()).hexdigest()
        self.records.append(record);started=time.perf_counter()
        try:
            response=(self.wrapped.generate_tools if native else self.wrapped.generate)(request)
            record.update(usage=response.usage,response={'text':response.text,'tool_calls':response.tool_calls,'finish_reason':response.finish_reason})
            return response
        except Exception as exc:
            record['error']=type(exc).__name__+': '+str(exc)[:180]
            raise
        finally:
            record['elapsed_ms']=round((time.perf_counter()-started)*1000,3)
            if self.sink:
                with self.lock:
                    with self.sink.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    def generate(self,request):return self.invoke(request,False)
    def generate_tools(self,request):return self.invoke(request,True)


def parse_requirement(c,question,model):
    try:
        links,_=ground_ask_question(c,question)
        trusted={k:v for k,v in links.model_dump(mode='json').items() if k!='games'}
    except ValueError:trusted={}
    prompt=semantic_prompt(question,trusted)
    tools=[{'type':'function','function':{'name':'describe_analysis','description':'Describe the requested analytical operation without SQL or IDs','parameters':AnalysisRequest.model_json_schema()}}]
    messages=[{'role':'user','content':prompt}]
    for attempt in range(2):
        response=model.generate_tools(ModelRequest(prompt=prompt,messages=messages,tools=tools,tool_choice={'type':'function','function':{'name':'describe_analysis'}},thinking={'type':'disabled'},timeout=15,max_tokens=1400))
        try:
            calls=response.tool_calls or []
            if len(calls)!=1 or calls[0].get('function',{}).get('name')!='describe_analysis':raise ValueError('one describe_analysis required')
            arguments=calls[0]['function']['arguments']
            proposal=AnalysisRequest.model_validate_json(arguments) if isinstance(arguments,str) else AnalysisRequest.model_validate(arguments)
        except (ValueError,TypeError,KeyError) as exc:
            if attempt: return None,{'status':'incomplete','accepted':False,'reason':'protocol_invalid','stage':'semantic_protocol'}
            messages.append({'role':'user','content':'Protocol invalid: return exactly one describe_analysis call obeying the JSON schema, with all original requirements preserved. No other change.'})
            continue
        try:
            request=ground_request(c,question,proposal)
            return request,{'proposal':proposal.model_dump(mode='json')}
        except GroundingFailure as exc:
            status='unsupported' if exc.code=='unsupported' else 'clarify' if exc.code in {'clarify','missing_or_ambiguous_season','missing_or_ambiguous_phase','missing_entity','unknown_entity','ambiguous_entity','entity_conflict'} else 'rejected'
            return None,{'status':status,'accepted':False,'stage':'semantic_grounding','reason':exc.code,'proposal':proposal.model_dump(mode='json')}


def physical_schema(c,subject):
    names=[subject+'_season_stat',subject+'_game_stat','game']
    return {name:[{'column':r[1],'type':r[2]} for r in c.execute('PRAGMA table_info('+name+')')] for name in names}


def generate_sql(c,question,request,model):
    response=model.generate(ModelRequest(prompt=sql_prompt(question,request,physical_schema(c,request.subject)),thinking={'type':'disabled'},timeout=15,max_tokens=1000))
    sql=response.text.strip()
    if sql.startswith('```') and sql.endswith('```'):
        sql='\n'.join(sql.splitlines()[1:-1]).strip()
    return sql
