import asyncio
import json

from scripts.knowledge_answer import KnowledgeAnswer
from scripts.knowledge_guidance import GuidedKnowledgeSearch
from scripts.knowledge_search import KnowledgeSearch
from test_knowledge_guidance import FakeDense, make_node


def build(observer=None, malformed=False):
    from scripts.model_adapter import ModelResponse
    target = make_node('target', 'A target rule.', page=1)
    other = make_node('other', 'Another rule.', page=2)
    dense = FakeDense({'target': target, 'other': other},
                      {0: [('target', .8), ('other', .2)], 1: [('target', .9)]})
    search = KnowledgeSearch([target, other], dense=dense)

    class Model:
        model = 'deepseek-flash'
        def __init__(self):
            self.requests = []
            self.outputs = [json.dumps({'action': 'search', 'intent': 'rules', 'english_query': 'target rule'}),
                'bad json' if malformed else json.dumps({'parts': [{'question': 'What?', 'claim': 'A target rule.',
                    'citation_id': 'target', 'excerpt': 'A target rule.'}]}),
                json.dumps({'verdicts': [{'index': 0, 'supported': True}]})]
        def generate(self, request):
            self.requests.append(request)
            return ModelResponse(self.outputs.pop(0), self.model)
    model = Model()
    guided = GuidedKnowledgeSearch(search, model, min_dense_score=.43, observer=observer)
    return KnowledgeAnswer(search, model, guided=guided, observer=observer), model, dense


def stable(result):
    return {k: v for k, v in result.items() if not k.endswith('_ms')}


def test_recording_preserves_answer_ranking_prompt_and_call_count():
    from scripts.knowledge_observation import RequestObservation
    observation = RequestObservation()
    plain, plain_model, plain_dense = build()
    recorded, recorded_model, recorded_dense = build(observation)
    a = asyncio.run(plain.aanswer('What is the target rule?'))
    b = asyncio.run(recorded.aanswer('What is the target rule?'))
    assert stable(a) == stable(b)
    assert a['status'] == 'answered'
    assert a['claims'][0]['citation_id'] == 'target'
    assert [r.prompt for r in plain_model.requests] == [r.prompt for r in recorded_model.requests]
    assert len(recorded_model.requests) == 3
    assert plain_dense.embedding.inputs == recorded_dense.embedding.inputs
    events = observation.snapshot()['events']
    assert events['final_top_k'][0]['citation_id'] == 'target'
    assert [h['citation_id'] for h in events['dense_threshold']] == ['target']
    assert events['final_top_k'][0]['dense_score'] == .9
    assert events['bounded_evidence'][0]['citation_id'] == 'target'
    assert events['generation_response']['raw']
    assert events['verification_result']['verdicts'] == [{'index': 0, 'supported': True}]
    assert observation.snapshot()['valid']


def test_bad_generation_keeps_retrieval_and_evidence_trace():
    from scripts.knowledge_observation import RequestObservation
    observation = RequestObservation()
    answerer, model, _ = build(observation, malformed=True)
    result = asyncio.run(answerer.aanswer('What is the target rule?'))
    assert result['reason_code'] == 'invalid_generation'
    events = observation.snapshot()['events']
    assert events['bounded_evidence'][0]['citation_id'] == 'target'
    assert events['generation_response']['raw'] == 'bad json'
    assert 'verification_result' not in events
    assert len(model.requests) == 2


def test_observation_snapshots_are_isolated_and_closed_recorders_ignore_late_events():
    from scripts.knowledge_observation import RequestObservation
    recorder = RequestObservation()
    payload = [{'citation_id': 'before'}]
    recorder.record('stage', payload)
    payload[0]['citation_id'] = 'changed'
    first = recorder.snapshot()
    first['events']['stage'].clear()
    assert recorder.snapshot()['events']['stage'] == [{'citation_id': 'before'}]
    recorder.close()
    recorder.record('late', {})
    assert 'late' not in recorder.snapshot()['events']
    assert RequestObservation().snapshot()['events'] == {}


def test_observer_failure_cannot_change_answer_but_invalidates_trace():
    from scripts.knowledge_observation import RequestObservation
    class Broken(RequestObservation):
        def record(self, stage, payload):
            raise RuntimeError('recorder unavailable')
    recorder = Broken()
    answerer, _, _ = build(recorder)
    result = asyncio.run(answerer.aanswer('What is the target rule?'))
    assert result['status'] == 'answered'
    assert not recorder.snapshot()['valid']


def test_outer_timeout_preserves_completed_stages():
    from scripts.knowledge_observation import RequestObservation
    recorder = RequestObservation()
    answerer, _, _ = build(recorder)
    async def slow(prompt, timeout):
        await asyncio.sleep(1)
    answerer._generate = slow
    async def run():
        try:
            await asyncio.wait_for(answerer.aanswer('What is the target rule?'), .15)
        except TimeoutError:
            pass
        finally:
            recorder.close()
    asyncio.run(run())
    events = recorder.snapshot()['events']
    assert events['bounded_evidence']
    assert events['answer_error']['error_type'] == 'CancelledError'
    assert 'generation_response' not in events
