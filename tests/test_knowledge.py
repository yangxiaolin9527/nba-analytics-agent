import asyncio
import hashlib
import json

from httpx import ASGITransport, AsyncClient
import pytest


def fixture_corpus(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    page = raw / "glossary.html"
    content = b"<html><main><h1>NBA Glossary</h1><h2>Effective Field Goal Percentage</h2><p>eFG% adjusts field goal percentage for the fact that a 3-point field goal is worth one more point than a 2-point field goal.</p><h2>Usage Percentage</h2><p>Usage percentage estimates possessions used by a player.</p></main></html>"
    page.write_bytes(content)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "sources": [{
        "id": "nba-glossary", "path": str(page.relative_to(tmp_path)),
        "source_url": "https://www.nba.com/stats/help/glossary", "version": "test",
        "retrieved_at": "2026-09-28", "license": "NBA official; local research use",
        "sha256": hashlib.sha256(content).hexdigest(), "format": "html",
    }]}))
    return manifest


def test_ingest_preserves_heading_and_stable_citation(tmp_path):
    from scripts.knowledge_corpus import load_corpus, split_corpus

    manifest = fixture_corpus(tmp_path)
    first = split_corpus(load_corpus(manifest), chunk_size=300, overlap=30)
    second = split_corpus(load_corpus(manifest), chunk_size=300, overlap=30)
    assert [n.id_ for n in first] == [n.id_ for n in second]
    assert any(n.metadata["heading"] == "Effective Field Goal Percentage" for n in first)
    assert all(n.metadata["source_url"].startswith("https://www.nba.com/") for n in first)
    assert all(n.metadata["citation_id"].startswith("nba-glossary:") for n in first)


def test_ingest_rejects_tampered_source(tmp_path):
    from scripts.knowledge_corpus import load_corpus

    manifest = fixture_corpus(tmp_path)
    (tmp_path / "raw/glossary.html").write_text("changed")
    with pytest.raises(ValueError, match="sha256"):
        load_corpus(manifest)


def test_glossary_definition_list_and_anchor_are_extracted(tmp_path):
    from scripts.knowledge_corpus import load_corpus

    manifest = fixture_corpus(tmp_path)
    html = ("<html><body><article id='efgpct'><h2><abbr title='Effective Field Goal Percentage'>eFG%</abbr></h2>"
            "<dl><dt>Name<dd>Effective Field Goal Percentage<dt>Definition<dd>Measures field goal "
            "percentage adjusting for made 3-point field goals.</dd></dt></dd></dt></dl></article></body></html>")
    source = tmp_path / "raw/glossary.html"
    source.write_text(html)
    payload = json.loads(manifest.read_text())
    payload["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))

    units = load_corpus(manifest)
    assert len(units) == 1
    assert units[0].metadata["heading"] == "eFG%"
    assert units[0].metadata["anchor"] == "efgpct"
    assert "Measures field goal percentage" in units[0].text
    assert units[0].text.count("Effective Field Goal Percentage") == 1


def test_glossary_extracts_all_articles_without_main_tag(tmp_path):
    from scripts.knowledge_corpus import load_corpus

    manifest = fixture_corpus(tmp_path)
    source = tmp_path / "raw/glossary.html"
    source.write_text("<html><body><article id='efg'><h2>eFG%</h2><p>Effective field goal percentage.</p></article>"
                      "<article id='ast'><h2>AST</h2><p>An assist is a pass leading to a basket.</p></article>"
                      "<h2>Jump To</h2><ul><li>eFG%</li><li>AST</li></ul></body></html>")
    payload = json.loads(manifest.read_text())
    payload["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload))

    units = load_corpus(manifest)
    assert [(unit.metadata["heading"], unit.metadata["anchor"]) for unit in units] == [
        ("eFG%", "efg"), ("AST", "ast")]


def test_search_api_returns_resolvable_original_excerpt(tmp_path):
    from app.main import create_app
    from scripts.knowledge_corpus import load_corpus, split_corpus
    from scripts.knowledge_search import KnowledgeSearch

    nodes = split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300, overlap=30)
    search = KnowledgeSearch(nodes)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app(knowledge_search=search)), base_url="http://test") as client:
            return await client.post("/knowledge/search", json={"question": "What is effective field goal percentage?", "mode": "bm25"})

    response = asyncio.run(request())
    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert "3-point field goal" in hit["excerpt"]
    assert hit["heading"] == "Effective Field Goal Percentage"
    assert hit["citation_id"] in {n.metadata["citation_id"] for n in nodes}
    assert "answer" not in response.json()
    assert response.json()["embedding_ms"] == 0


def test_search_api_reports_unavailable_index():
    from app.main import create_app

    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
            return await client.post("/knowledge/search", json={"question": "traveling rule"})

    response = asyncio.run(request())
    assert response.status_code == 503


def test_search_api_awaits_async_retrieval():
    from app.main import create_app

    class Search:
        def search(self, question, **kwargs):
            raise AssertionError("synchronous retrieval must not run in the async route")

        async def asearch(self, question, **kwargs):
            await asyncio.sleep(0)
            return {"mode": kwargs["mode"], "hits": []}

    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app(knowledge_search=Search())), base_url="http://test") as client:
            return await client.post("/knowledge/search", json={"question": "rule"})

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["mode"] == "bm25"


def test_dense_search_api_uses_async_embedding_path(tmp_path):
    from app.main import create_app
    from llama_index.core.schema import NodeWithScore
    from scripts.knowledge_corpus import load_corpus, split_corpus
    from scripts.knowledge_search import KnowledgeSearch

    nodes = split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300)

    class Dense:
        last_embedding_ms = 1.0
        last_vector_query_ms = 1.0

        def retrieve(self, question, top_k):
            raise AssertionError("synchronous embedding path called")

        async def aretrieve(self, question, top_k):
            await asyncio.sleep(0)
            return [NodeWithScore(node=nodes[0], score=0.8)]

    async def request():
        app = create_app(knowledge_search=KnowledgeSearch(nodes, dense=Dense()))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/knowledge/search", json={"question": "efg", "mode": "dense"})

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["hits"][0]["citation_id"] == nodes[0].id_


def test_dense_index_reopens_and_rejects_model_mismatch(tmp_path):
    from scripts.knowledge_corpus import load_corpus, split_corpus
    from scripts.knowledge_vector import DenseIndex

    class FixedEmbedding:
        model = "fixture-multilingual"

        def embed(self, texts):
            return [[1.0, 0.0] if "field goal" in text.lower() else [0.0, 1.0] for text in texts]

    nodes = split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300, overlap=30)
    directory = tmp_path / "index"
    DenseIndex.build(nodes, directory, FixedEmbedding()).close()
    reopened = DenseIndex.open(directory, FixedEmbedding())
    results = reopened.retrieve("field goal", 3)
    assert results[0].node.metadata["heading"] == "Effective Field Goal Percentage"
    assert reopened.last_embedding_ms >= 0
    reopened.close()
    other = FixedEmbedding()
    other.model = "changed-model"
    with pytest.raises(ValueError, match="model"):
        DenseIndex.open(directory, other)


def test_dense_index_document_and_query_adapter_kinds(tmp_path):
    from scripts.knowledge_vector import DenseIndex
    from llama_index.core.schema import TextNode
    class Fixed:
        model = 'fixture-kinds'
        def __init__(self): self.kinds = []
        def embed(self, texts, *, kind='query'):
            self.kinds.append(kind)
            return [[1.0, 0.0] for _ in texts]
    node = TextNode(id_='x', text='text', metadata={'citation_id':'x','id':'s','page':1,
        'anchor':None,'heading':None,'source_url':'u','version':'v','sha256':'h'})
    adapter = Fixed()
    index = DenseIndex.build([node], tmp_path / 'index', adapter)
    index.retrieve('query', 1)
    assert adapter.kinds == ['document', 'query']
    index.close()


def test_retrieval_metrics_count_citation_and_no_answer_errors():
    from scripts.evaluate_knowledge import evaluate_cases

    cases = [
        {"id": "Q1", "split": "development", "question": "efg", "answerable": True,
         "relevant": [{"source_id": "glossary", "heading": "eFG"}]},
        {"id": "Q2", "split": "development", "question": "unknown", "answerable": False, "relevant": []},
    ]

    class Search:
        def search(self, question, **kwargs):
            return {"elapsed_ms": 2, "hits": [{"citation_id": "glossary:1", "source_id": "glossary",
                    "heading": "eFG", "page": None, "score": 1.0}] if question == "efg" else []}

    report = evaluate_cases(Search(), cases, split="development", mode="bm25")
    assert report["recall_at_5"] == 1
    assert report["mrr_at_10"] == 1
    assert report["ndcg_at_10"] == 1
    assert report["no_answer_false_acceptance_rate"] == 0
    assert report["citation_location_accuracy"] == 1


def test_recall_counts_unique_relevant_locations():
    from scripts.evaluate_knowledge import evaluate_cases

    class Search:
        def search(self, question, **kwargs):
            return {"elapsed_ms": 1, "hits": [
                {"citation_id": "one", "source_id": "rules", "page": 1, "heading": None, "score": 1},
                {"citation_id": "two", "source_id": "rules", "page": 1, "heading": None, "score": 0.9},
            ]}

    cases = [{"id": "M1", "split": "development", "question": "q", "answerable": True,
              "relevant": [{"source_id": "rules", "page": 1}, {"source_id": "rules", "page": 2}]}]
    report = evaluate_cases(Search(), cases, split="development", mode="bm25")
    assert report["recall_at_5"] == 0.5
    assert report["ndcg_at_10"] < 1


def test_citation_integrity_has_per_hit_denominator_and_no_empty_success():
    from scripts.evaluate_knowledge import evaluate_cases

    cases = [{"id": "N1", "split": "development", "question": "unknown",
              "answerable": False, "relevant": []}]

    class EmptySearch:
        def search(self, question, **kwargs):
            return {"elapsed_ms": 1, "hits": []}

    report = evaluate_cases(EmptySearch(), cases, split="development", mode="bm25")
    assert report["citation_location_accuracy"] is None
    assert report["citation_checked_hit_count"] == 0


def test_persisted_lexical_index_is_loaded_by_default_app(tmp_path, monkeypatch):
    from app.main import create_app
    from scripts.knowledge_corpus import load_corpus, split_corpus
    from scripts.knowledge_search import save_nodes

    nodes = split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300, overlap=30)
    index = tmp_path / "index"
    save_nodes(nodes, index, manifest_hash="fixture", chunk_size=300, overlap=30)
    from scripts.provider_config import ProviderConfig
    config = ProviderConfig(knowledge_index_path=str(index))

    async def request():
        async with AsyncClient(transport=ASGITransport(app=create_app(provider_config=config)), base_url="http://test") as client:
            return await client.post("/knowledge/search", json={"question": "effective field goal", "mode": "bm25"})

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["hits"][0]["source_id"] == "nba-glossary"


def test_api_embedding_batches_and_restores_response_order(monkeypatch):
    from scripts.knowledge_vector import APIEmbedding

    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            values = calls[-1]
            return {"data": [{"index": index, "embedding": [float(value)]}
                             for index, value in reversed(list(enumerate(values)))]}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            calls.append(json["input"])
            return Response()

    from scripts.provider_config import ProviderConfig
    config = ProviderConfig(embedding_base_url="https://example.test/v1",
                            embedding_model="fixture", embedding_api_key="fixture-secret")
    monkeypatch.setattr("scripts.knowledge_vector.httpx.Client", Client)
    vectors = APIEmbedding(config=config).embed([str(i) for i in range(130)])
    assert [len(call) for call in calls] == [64, 64, 2]
    assert vectors == [[float(i)] for i in range(130)]


def test_api_embedding_supports_async_requests(monkeypatch):
    from scripts.knowledge_vector import APIEmbedding

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, headers, json):
            assert json["input"] == ["question"]
            return Response()

    from scripts.provider_config import ProviderConfig
    config = ProviderConfig(embedding_base_url="https://example.test/v1",
                            embedding_model="fixture", embedding_api_key="fixture-secret")
    monkeypatch.setattr("scripts.knowledge_vector.httpx.AsyncClient", Client)
    assert asyncio.run(APIEmbedding(config=config).aembed(["question"])) == [[1.0, 2.0]]


def test_knowledge_cases_have_unique_ids_and_resolvable_locations(tmp_path):
    from copy import deepcopy
    from scripts.evaluate_knowledge import validate_cases
    from scripts.knowledge_corpus import load_corpus, split_corpus
    from scripts.knowledge_search import KnowledgeSearch

    search = KnowledgeSearch(split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300))
    cases = [{"id": "K001", "split": "development", "answerable": True,
              "relevant": [{"source_id": "nba-glossary", "heading": "Effective Field Goal Percentage"}]}]
    validate_cases(search, cases)
    with pytest.raises(ValueError, match="unique"):
        validate_cases(search, cases * 2)
    wrong = deepcopy(cases)
    wrong[0]["relevant"][0]["heading"] = "Missing heading"
    with pytest.raises(ValueError, match="missing source location"):
        validate_cases(search, wrong)


def test_legacy_index_rejects_model_mismatch_before_opening_provider(tmp_path):
    import json
    from scripts.knowledge_search import configured_search, save_nodes
    from scripts.provider_config import ProviderConfig
    from scripts.knowledge_corpus import load_corpus, split_corpus
    index = tmp_path / 'index'
    nodes = split_corpus(load_corpus(fixture_corpus(tmp_path)), chunk_size=300, overlap=30)
    save_nodes(nodes, index, manifest_hash='fixture', chunk_size=300, overlap=30)
    (index / 'dense.json').write_text(json.dumps({'model': 'text-embedding-3-small', 'dimension': 1536}))
    config = ProviderConfig(embedding_base_url='https://example.invalid/v1', embedding_api_key='test',
                            embedding_model='qwen3.7-text-embedding', embedding_dimension=1024)
    with pytest.raises(ValueError, match='legacy embedding index'):
        configured_search(index, provider_config=config)


def test_direct_legacy_dense_open_rejects_wrong_model_before_qdrant(tmp_path):
    from scripts.knowledge_vector import DenseIndex
    (tmp_path / 'dense.json').write_text(json.dumps({
        'model': 'historical-model', 'dimension': 1024, 'collection': 'fixture'}))
    embedding = type('Embedding', (), {'model': 'different-model'})()
    with pytest.raises(ValueError, match='embedding model changed'):
        DenseIndex.open(tmp_path, embedding)
    assert not (tmp_path / 'qdrant').exists()
