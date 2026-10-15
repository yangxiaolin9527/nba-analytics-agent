"""Request-time capability dispatch. No gold labels, Oracle or product wiring."""
from dataclasses import asdict
from pathlib import Path
import sqlite3
import time

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.two_stage_sql.modeling import parse_requirement, generate_sql
from scripts.two_stage_sql.runtime import execute_tool, execute_sql


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=4000)


def choose_branch(request):
    if request is None:
        return None
    if request.operation in {'summary', 'window', 'efg'}:
        return 'B'
    if request.operation in {'rank', 'game_count'}:
        return 'C'
    raise ValueError('outside_experimental_capabilities')


LABELS = {'points_total':'总得分','points_per_appearance':'场均得分','assists_total':'总助攻','rebounds_total':'总篮板','games_appeared':'出场场数','games_played':'比赛场数','efg_percentage':'有效命中率（%）','qualifying_games':'符合条件的场数','entity_id':'实体 ID'}


def render_answer(candidate):
    if not candidate.get('accepted'):
        return {'clarify':'需要补充或确认查询条件。','unsupported':'当前实验查询能力不支持这个问题。','not_computable':'当前数据库记录不足以计算这个结果。'}.get(candidate.get('status'),'本次查询未能完成。')
    columns = candidate['columns']
    lines = ['；'.join(f'{LABELS.get(key,key)}：{value}' for key,value in zip(columns,row)) for row in candidate['rows']]
    return '\n'.join(lines)


def answer_question(connection, question, model, *, matrix_path=CAPABILITY_MATRIX):
    started = time.perf_counter()
    model.stage = 'semantic'
    request = None
    selected = None
    timings = {}
    semantic = {}
    try:
        request, semantic = parse_requirement(connection, question, model)
        timings['semantic_ms'] = (time.perf_counter()-started)*1000
        dispatch_start = time.perf_counter()
        selected = choose_branch(request)
        timings['dispatch_ms'] = (time.perf_counter()-dispatch_start)*1000
        execution_start = time.perf_counter()
        if selected is None:
            candidate = dict(semantic)
        elif selected == 'B':
            candidate = execute_tool(connection, request, matrix_path=matrix_path)
        else:
            model.stage = 'sql'
            sql_start = time.perf_counter()
            sql = generate_sql(connection, question, request, model)
            timings['sql_generation_ms'] = (time.perf_counter()-sql_start)*1000
            candidate = execute_sql(connection, request, sql, matrix_path=matrix_path)
            candidate['generated_sql'] = sql
        timings['branch_ms'] = (time.perf_counter()-execution_start)*1000
    except Exception as exc:
        candidate = {'status':'incomplete','accepted':False,'reason':type(exc).__name__,'stage':'semantic' if selected is None else selected}
    answer = render_answer(candidate)
    timings['runtime_ms'] = (time.perf_counter()-started)*1000
    return {'status':candidate['status'],'answer':answer,'candidate':candidate,'grounded_request':asdict(request) if request else None,'selected_arm':selected,'semantic':semantic,'timings_ms':{k:round(v,3) for k,v in timings.items()},'policy':'capability before execution; never failure fallback'}


def create_hybrid_app(db_path: Path, model, *, matrix_path=CAPABILITY_MATRIX):
    application = FastAPI(title='Isolated online structured-query experiment')

    @application.post('/experiment/structured/answer')
    def answer(body: Question):
        with sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=ro',uri=True) as connection:
            return answer_question(connection, body.question, model, matrix_path=matrix_path)

    return application
