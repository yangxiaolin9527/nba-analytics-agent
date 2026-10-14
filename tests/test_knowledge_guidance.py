import asyncio
import json

from httpx import ASGITransport, AsyncClient
from llama_index.core.schema import NodeWithScore, TextNode
import pytest


def make_node(citation_id, text, *, page=None, anchor=None):
    return TextNode(id_=citation_id, text=text, metadata={
        "citation_id": citation_id, "id": "rules" if page else "glossary",
        "source_url": "https://example.com/source", "version": "test",
        "sha256": "fixture", "page": page, "anchor": anchor,
        "heading": None if page else anchor,
    })


class FakeEmbedding:
    model = "text-embedding-3-small"

    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        self.inputs.append(texts)
        return [[float(index)] for index, _ in enumerate(texts)]

    async def aembed(self, texts):
        return self.embed(texts)


class FakeDense:
    def __init__(self, nodes, scores):
        self.embedding = FakeEmbedding()
        self.nodes = nodes
        self.scores = scores
        self.last_embedding_ms = 0.0
        self.last_vector_query_ms = 0.0

    def retrieve_vector(self, vector, top_k):
        return [NodeWithScore(node=self.nodes[name], score=score)
                for name, score in self.scores[int(vector[0])]][:top_k]


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def generate(self, request):
        from scripts.model_adapter import ModelResponse
        assert request.response_format == "json_object"
        self.calls += 1
        return ModelResponse(json.dumps(self.response), self.model)


def test_embedding_provider_read_timeout_keeps_timeout_classification():
    import httpx
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch
    target = make_node('rules:target', 'A game starts with a jump ball.', page=25)
    dense = FakeDense({'target': target}, {0: [('target', .9)]})
    async def timed_out(texts):
        raise httpx.ReadTimeout('provider response stalled')
    dense.embedding.aembed = timed_out
    guided = GuidedKnowledgeSearch(KnowledgeSearch([target], dense=dense),
        FakeModel({'action': 'search', 'intent': 'rules', 'english_query': 'jump ball'}))
    result = asyncio.run(guided.asearch('How does the game start?'))
    assert result['reason_code'] == 'embedding_timeout'
    assert result['error_stage'] == 'embedding'


def test_embedding_http_budget_is_bounded_by_guided_stage(monkeypatch):
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch
    from scripts.knowledge_vector import APIEmbedding
    from scripts.provider_config import ProviderConfig
    budgets = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'index': i, 'embedding': [float(i)]} for i in range(2)]}
    class Client:
        def __init__(self, *, timeout):
            budgets.append(timeout)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr('scripts.knowledge_vector.httpx.AsyncClient', Client)
    target = make_node('rules:target', 'A game starts with a jump ball.', page=25)
    dense = FakeDense({'target': target}, {0: [('target', .9)], 1: [('target', .9)]})
    dense.embedding = APIEmbedding(ProviderConfig(embedding_base_url='https://example.test',
        embedding_api_key='fixture', embedding_model='text-embedding-3-small', embedding_timeout_seconds=30))
    guided = GuidedKnowledgeSearch(KnowledgeSearch([target], dense=dense),
        FakeModel({'action': 'search', 'intent': 'rules', 'english_query': 'jump ball'}), embedding_timeout=.2)
    result = asyncio.run(guided.asearch('How does the game start?'))
    assert result['retrieval_status'] == 'candidates_found'
    assert 0 < budgets[0] <= .2


def run_request(app, payload):
    async def request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/knowledge/search", json=payload)
    return asyncio.run(request())


def test_guidance_plan_rejects_inconsistent_action_and_dropped_numbers():
    from scripts.knowledge_guidance import parse_guidance_plan

    with pytest.raises(ValueError):
        parse_guidance_plan('{"action":"search","intent":"outside","english_query":"rule"}', "rule")
    with pytest.raises(ValueError, match="number"):
        parse_guidance_plan('{"action":"search","intent":"rules","english_query":"What is the rule?"}',
                            "What is the 2025-26 rule?")
    with pytest.raises(ValueError):
        parse_guidance_plan('{"action":"abstain","intent":"uncertain","english_query":"rule"}', "rule")


def test_guided_retrieval_uses_only_original_and_planner_queries():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    target = make_node("rules:target", "A dribbler shall not charge into a legal guarding position.", page=46)
    dense = FakeDense({"target": target}, {0: [("target", .9)], 1: [("target", .8)]})
    model = FakeModel({"action": "search", "intent": "rules",
                       "english_query": "When is a charging foul called?"})
    response = run_request(create_app(model=model, knowledge_search=KnowledgeSearch([target], dense=dense)),
                           {"question": "什么时候吹罚带球撞人犯规？", "mode": "guided"})

    assert response.status_code == 200
    assert dense.embedding.inputs == [["什么时候吹罚带球撞人犯规？", "When is a charging foul called?"]]
    assert response.json()["rewritten_query"] == "When is a charging foul called?"


def test_rrf_fuses_source_locations_before_deduplicating_chunks():
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch
    from llama_index.core.schema import NodeWithScore

    page_a = [make_node(f"rules:a{index}", f"common page A chunk {index}", page=1)
              for index in range(3)]
    target = make_node("rules:target", "target rule on page 2", page=2)
    nodes = page_a + [target]
    dense = FakeDense({node.id_: node for node in nodes}, {
        0: [("rules:a0", .98), ("rules:target", .99), ("rules:a1", .97), ("rules:a2", .96)],
        1: [("rules:target", .99), ("rules:a0", .98), ("rules:a1", .97), ("rules:a2", .96)],
    })
    guided = GuidedKnowledgeSearch(KnowledgeSearch(nodes, dense=dense),
                                   FakeModel({"action": "search", "intent": "rules",
                                              "english_query": "target rule"}),
                                   min_dense_score=0.0)

    result = guided._rank("target rule", "rules", [[0], [1]], top_k=1,
                          planning_ms=0, embedding_ms=0, started=0)

    assert result["hits"][0]["citation_id"] == "rules:target"


def test_rrf_location_representative_remains_dense_candidate_when_bm25_adds_same_page():
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    dense_node = make_node("rules:dense", "target", page=2)
    lexical_node = make_node("rules:lexical", "target context", page=2)
    dense = FakeDense({"dense": dense_node}, {0: [("dense", .9)]})
    guided = GuidedKnowledgeSearch(KnowledgeSearch([dense_node, lexical_node], dense=dense),
                                   FakeModel({"action": "search", "intent": "rules", "english_query": "target"}),
                                   min_dense_score=0.0)
    result = guided._rank("target", "rules", [[0]], top_k=1,
                          planning_ms=0, embedding_ms=0, started=0)
    assert result["hits"][0]["citation_id"] == "rules:dense"




def test_legacy_glossary_plan_defaults_to_single_scope_for_adaptive_policy():
    from scripts.knowledge_guidance import parse_guidance_plan

    plan = parse_guidance_plan(
        '{"action":"search","intent":"glossary","english_query":"effective field goal percentage"}',
        "What is eFG%?",
    )
    assert plan.question_scope == "single"


def test_guided_route_abstains_outside_corpus_when_probe_is_below_threshold():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("glossary:three", "Three point field goal percentage is a rate.", anchor="3ppct")
    dense = FakeDense({"three": node}, {0: [("three", 0.2)]})
    model = FakeModel({"action": "abstain", "intent": "outside", "english_query": None})
    app = create_app(model=model, knowledge_search=KnowledgeSearch([node], dense=dense))
    response = run_request(app, {"question": "2023-24 谁是三分命中率最高的球员？", "mode": "guided"})

    assert response.status_code == 200
    assert response.json()["retrieval_status"] == "abstained"
    assert response.json()["reason_code"] == "outside_corpus_scope"
    assert response.json()["hits"] == []
    assert model.calls == 1
    assert response.json()["scope_recovery"] is None
    assert dense.embedding.inputs == [["2023-24 谁是三分命中率最高的球员？"]]


def test_guided_route_recovers_in_domain_scope_from_thresholded_corpus_probe():
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    rule = make_node("rules:charging", "A player may not charge into a legal guarding position.", page=46)
    dense = FakeDense({"rule": rule}, {0: [("rule", 0.91)]})
    model = FakeModel({"action": "abstain", "intent": "outside", "english_query": None})
    guided = GuidedKnowledgeSearch(
        KnowledgeSearch([rule], dense=dense), model, min_dense_score=0.43,
    )

    result = asyncio.run(guided.asearch("什么时候吹罚带球撞人犯规？"))

    assert result["retrieval_status"] == "candidates_found"
    assert result["intent"] == "rules"
    assert result["scope_recovery"] == "corpus_probe"
    assert result["hits"][0]["citation_id"] == "rules:charging"
    assert dense.embedding.inputs == [["什么时候吹罚带球撞人犯规？"]]


def test_guided_route_fuses_queries_and_returns_original_evidence():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    target = make_node("rules:target", "A game starts with a jump ball at center circle.", page=25)
    distractor = make_node("rules:other", "A referee may call a timeout.", page=24)
    dense = FakeDense({"target": target, "other": distractor}, {
        0: [("other", 0.61), ("target", 0.45)],
        1: [("target", 0.72), ("other", 0.44)],
    })
    model = FakeModel({"action": "search", "intent": "rules",
                       "english_query": "How do NBA games and overtime periods start?"})
    app = create_app(model=model, knowledge_search=KnowledgeSearch([target, distractor], dense=dense))
    response = run_request(app, {"question": "NBA比赛和加时赛如何开始？", "mode": "guided"})

    assert response.status_code == 200
    body = response.json()
    assert body["retrieval_status"] == "candidates_found"
    assert body["hits"][0]["citation_id"] == "rules:target"
    assert body["hits"][0]["excerpt"] == target.text
    assert body["hits"][0]["dense_score"] == 0.72
    assert body["score_type"] == "rrf"
    assert dense.embedding.inputs == [["NBA比赛和加时赛如何开始？", "How do NBA games and overtime periods start?"]]
    assert model.calls == 1


def test_guided_route_abstains_when_dense_evidence_is_weak():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:weak", "Unrelated foul rule.", page=45)
    dense = FakeDense({"weak": node}, {0: [("weak", 0.2)]})
    model = FakeModel({"action": "search", "intent": "rules", "english_query": "What is the foul rule?"})
    app = create_app(model=model, knowledge_search=KnowledgeSearch([node], dense=dense))
    response = run_request(app, {"question": "What is the foul rule?", "mode": "guided"})

    assert response.status_code == 200
    assert response.json()["retrieval_status"] == "abstained"
    assert response.json()["reason_code"] == "low_relevance"
    assert response.json()["hits"] == []


def test_existing_bm25_response_stays_compatible_with_guided_mode():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("glossary:efg", "Effective field goal percentage weights three pointers.", anchor="efg")
    response = run_request(create_app(knowledge_search=KnowledgeSearch([node])),
                           {"question": "effective field goal percentage"})

    assert response.status_code == 200
    assert response.json()["mode"] == "bm25"
    assert response.json()["hits"][0]["citation_id"] == "glossary:efg"
    assert "retrieval_status" not in response.json()


def test_guided_zero_threshold_ignores_lexical_only_candidates():
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    target = make_node("rules:target", "A game starts with a jump ball.", page=25)
    lexical_only = make_node("rules:lexical", "A game can start after a timeout.", page=24)
    dense = FakeDense({"target": target}, {0: [("target", 0.4)]})
    model = FakeModel({"action": "search", "intent": "rules", "english_query": "How does a game start?"})
    guided = GuidedKnowledgeSearch(KnowledgeSearch([target, lexical_only], dense=dense), model,
                                   min_dense_score=0)

    result = guided.search("How does a game start?")

    assert [hit["citation_id"] for hit in result["hits"]] == ["rules:target"]


def test_guided_rule_intent_excludes_glossary_candidates():
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    rule = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    glossary = make_node("glossary:term", "A jump ball is a jump ball.", anchor="jump_ball")
    dense = FakeDense({"rule": rule, "glossary": glossary}, {
        0: [("glossary", 0.9), ("rule", 0.6)],
    })
    model = FakeModel({"action": "search", "intent": "rules", "english_query": "How does a game start?"})
    guided = GuidedKnowledgeSearch(KnowledgeSearch([rule, glossary], dense=dense), model)

    result = guided.search("How does a game start?")

    assert [hit["citation_id"] for hit in result["hits"]] == ["rules:rule"]


def test_guided_route_rejects_model_that_differs_from_calibrated_policy():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    dense = FakeDense({"rule": node}, {0: [("rule", 0.7)]})
    model = FakeModel({"action": "search", "intent": "rules", "english_query": "How does a game start?"})
    model.model = "different-model"
    app = create_app(model=model, knowledge_search=KnowledgeSearch([node], dense=dense))

    response = run_request(app, {"question": "How does a game start?", "mode": "guided"})

    assert response.status_code == 503
    assert model.calls == 0


def test_guided_route_reports_invalid_model_plan():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    dense = FakeDense({"rule": node}, {0: [("rule", 0.7)]})
    model = FakeModel({"action": "search", "intent": "outside", "english_query": "rule"})
    response = run_request(create_app(model=model, knowledge_search=KnowledgeSearch([node], dense=dense)),
                           {"question": "How does a game start?", "mode": "guided"})

    assert response.status_code == 502
    assert response.json()["detail"] == "guided retrieval failed"
    assert dense.embedding.inputs == []


def test_guided_route_reports_embedding_failure():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    dense = FakeDense({"rule": node}, {0: [("rule", 0.7)]})

    async def fail_embedding(texts):
        raise RuntimeError("provider unavailable")

    dense.embedding.aembed = fail_embedding
    model = FakeModel({"action": "search", "intent": "rules", "english_query": "How does a game start?"})
    response = run_request(create_app(model=model, knowledge_search=KnowledgeSearch([node], dense=dense)),
                           {"question": "How does a game start?", "mode": "guided"})

    assert response.status_code == 502
    assert response.json()["detail"] == "guided retrieval failed"


def test_guided_planner_timeout_is_configurable():
    import asyncio
    import json
    import time
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    dense = FakeDense({"rule": node}, {0: [("rule", 0.7)]})

    class SlowModel:
        model = "fixture"

        def generate(self, request):
            time.sleep(0.06)
            from scripts.model_adapter import ModelResponse
            return ModelResponse(json.dumps({"action": "abstain", "intent": "outside",
                                            "english_query": None}), self.model)

    guided = GuidedKnowledgeSearch(
        KnowledgeSearch([node], dense=dense), SlowModel(), min_dense_score=0.0,
        planner_timeout=0.2,
    )
    result = asyncio.run(guided.asearch("How does a game start?"))

    assert result["retrieval_status"] == "candidates_found"
    assert result["scope_recovery"] == "corpus_probe"
    assert result["planning_ms"] >= 50


def test_guided_provider_timeout_is_reported_as_planner_timeout():
    import asyncio
    from scripts.knowledge_guidance import GuidedKnowledgeSearch
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    dense = FakeDense({"rule": node}, {0: [("rule", 0.7)]})

    class TimedOutModel:
        model = "fixture"

        def generate(self, request):
            raise TimeoutError("provider deadline")

    guided = GuidedKnowledgeSearch(
        KnowledgeSearch([node], dense=dense), TimedOutModel(), min_dense_score=0.0,
        planner_timeout=0.2,
    )
    result = asyncio.run(guided.asearch("How does a game start?"))

    assert result["reason_code"] == "planner_timeout"
    assert result["error_stage"] == "planner"


def test_guided_route_requires_dense_index():
    from app.main import create_app
    from scripts.knowledge_search import KnowledgeSearch

    node = make_node("rules:rule", "The game starts with a jump ball.", page=25)
    response = run_request(create_app(knowledge_search=KnowledgeSearch([node])),
                           {"question": "How does a game start?", "mode": "guided"})

    assert response.status_code == 503
