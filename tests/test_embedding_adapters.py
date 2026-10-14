import asyncio
import json

import pytest


@pytest.mark.parametrize('dimension,normalized', [(1536, True), (1024, False)])
def test_local_loader_rejects_unsupported_config_before_loading_model(monkeypatch, dimension, normalized):
    import scripts.embedding_adapters as adapters
    from scripts.provider_config import ProviderConfig
    loaded = []
    monkeypatch.setattr(adapters, 'load_local_sentence_transformer', lambda *args, **kwargs: loaded.append(args))
    config = ProviderConfig(embedding_provider='local_sentence_transformers',
                            embedding_model='BAAI/bge-m3', embedding_dimension=dimension,
                            embedding_normalize=normalized)
    with pytest.raises(ValueError, match='local embedding requires'):
        adapters.load_embedding_adapter(config)
    assert loaded == []


def test_embedding_metadata_is_part_of_the_adapter_contract():
    from scripts.embedding_adapters import EmbeddingMetadata

    metadata = EmbeddingMetadata(provider="fixture", model="fixture-v1", dimension=3,
                                 normalized=True, query_instruction="query", document_instruction="")
    assert metadata.to_dict() == {
        "provider": "fixture", "model": "fixture-v1", "dimension": 3,
        "normalized": True, "query_instruction": "query", "document_instruction": "",
    }


def test_local_adapter_uses_query_prompt_only_for_queries(monkeypatch):
    from scripts.embedding_adapters import LocalSentenceTransformerAdapter

    calls = []
    class Model:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[1.0, 0.0], [0.0, 1.0]][:len(texts)]
    adapter = LocalSentenceTransformerAdapter.from_model(Model(), model_name="fixture", dimension=2,
                                                         query_prompt_name="query")
    assert adapter.embed(["a", "b"], kind="query") == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[-1][1]["prompt_name"] == "query"
    adapter.embed(["a"], kind="document")
    assert "prompt_name" not in calls[-1][1]


def test_api_adapter_sends_dimensions_and_preserves_response_order(monkeypatch):
    from scripts.embedding_adapters import APIEmbeddingAdapter

    requests = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                              {"index": 0, "embedding": [1.0, 0.0]}]}
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs):
            requests.append((url, kwargs)); return Response()
    monkeypatch.setattr("scripts.embedding_adapters.httpx.Client", Client)
    adapter = APIEmbeddingAdapter(base_url="https://example.test/v1", api_key="secret",
                                  model="fixture", dimension=2, timeout=3)
    assert adapter.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert requests[0][1]["json"]["dimensions"] == 2


def test_api_adapter_rejects_response_dimension_mismatch(monkeypatch):
    from scripts.embedding_adapters import APIEmbeddingAdapter

    class Response:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kwargs): return Response()
    monkeypatch.setattr("scripts.embedding_adapters.httpx.Client", Client)
    adapter = APIEmbeddingAdapter(base_url="https://example.test/v1", api_key="secret",
                                   model="fixture", dimension=2)
    with pytest.raises(ValueError, match="configured dimension"):
        adapter.embed(["a"])


def test_dense_index_rejects_embedding_metadata_mismatch(tmp_path):
    from llama_index.core.schema import TextNode
    from scripts.embedding_adapters import EmbeddingMetadata
    from scripts.knowledge_vector import DenseIndex

    class Fixed:
        model = "fixture"
        metadata = EmbeddingMetadata("fixture", "fixture", 2, True, "", "")
        def embed(self, texts, *, kind="document"):
            return [[1.0, 0.0] for _ in texts]
    node = TextNode(id_="x", text="a", metadata={"citation_id":"x", "id":"s", "page":1,
                       "anchor":None, "heading":None, "source_url":"u", "version":"v", "sha256":"h"})
    DenseIndex.build([node], tmp_path / "index", Fixed()).close()
    class Wrong(Fixed):
        metadata = EmbeddingMetadata("other", "other", 2, True, "", "")
    with pytest.raises(ValueError, match="embedding metadata"):
        DenseIndex.open(tmp_path / "index", Wrong())
