"""Frozen human-authored requirements, accessible only to the evaluator."""
import json
from pathlib import Path
from scripts.two_stage_sql.contracts import GroundedRequest

MANIFEST=Path(__file__).resolve().parents[2]/'metadata/two_stage_sql_v1.json'

def read_cases(path=MANIFEST):
    manifest=json.loads(path.read_text())
    cases=manifest['cases']
    if len(cases)!=36 or len({c['id'] for c in cases})!=36:
        raise ValueError('expected 36 unique scenarios')
    for split,n in [('development',12),('frozen',24)]:
        if sum(c['split']==split for c in cases)!=n:raise ValueError('split count mismatch')
    for case in cases:
        if 'status' not in case['gold']:gold_request(case)
    return manifest

def gold_request(case):
    data=dict(case['gold'])
    data['metrics']=tuple(data['metrics'])
    data['predicates']=tuple(tuple(p) for p in data.get('predicates',[]))
    return GroundedRequest(**data)
