"""OpenAI-compatible embeddings with a persistent local Qdrant index."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

import httpx
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from scripts.provider_config import ProviderConfig, load_provider_config
from scripts.knowledge_runtime import current_runtime, embedding_deadline, stage_budget
from scripts.embedding_adapters import APIEmbeddingAdapter, EmbeddingMetadata


class APIEmbedding:
    def __init__(self, config: ProviderConfig | None = None, *, model: str | None = None,
                 dimension: int = 1536):
        selected = config or load_provider_config()
        self.base_url, self.key, configured_model = selected.require_embedding()
        self.model = model or configured_model
        self.base_url = self.base_url.rstrip("/")
        self.timeout = selected.embedding_timeout_seconds
        self.adapter = APIEmbeddingAdapter(base_url=self.base_url, api_key=self.key,
            model=self.model, dimension=dimension, normalized=False, timeout=self.timeout,
            batch_size=64, provider="openai_compatible_legacy")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dimension = None
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), 64):
                batch = texts[start:start + 64]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()["data"]
                if len(data) != len(batch) or sorted(item["index"] for item in data) != list(range(len(batch))):
                    raise ValueError("embedding response has missing or duplicate indices")
                for item in sorted(data, key=lambda item: item["index"]):
                    vector = list(item["embedding"])
                    if not vector or (dimension is not None and len(vector) != dimension):
                        raise ValueError("embedding response has inconsistent dimensions")
                    dimension = len(vector)
                    vectors.append(vector)
        return vectors

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        dimension = None
        budget = stage_budget(self.timeout, deadline=embedding_deadline.get())
        async with httpx.AsyncClient(timeout=budget) as client:
            for start in range(0, len(texts), 64):
                batch = texts[start:start + 64]
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={"model": self.model, "input": batch},
                )
                response.raise_for_status()
                body = response.json()
                data = body["data"]
                runtime = current_runtime.get()
                if runtime is not None:
                    runtime.model_calls["embedding"] = {
                        "model": self.model, "provider": urlsplit(self.base_url).hostname,
                        "usage": body.get("usage"), "input_count": len(batch),
                        "timeout_seconds": budget,
                        "input_chars": sum(map(len, batch)),
                        "output_vector_count": len(data),
                    }
                if len(data) != len(batch) or sorted(item["index"] for item in data) != list(range(len(batch))):
                    raise ValueError("embedding response has missing or duplicate indices")
                for item in sorted(data, key=lambda item: item["index"]):
                    vector = list(item["embedding"])
                    if not vector or (dimension is not None and len(vector) != dimension):
                        raise ValueError("embedding response has inconsistent dimensions")
                    dimension = len(vector)
                    vectors.append(vector)
        return vectors


class DenseIndex:
    COLLECTION = "nba_knowledge"

    def __init__(self, client: QdrantClient, store: QdrantVectorStore, embedding, dimension: int):
        self.client = client
        self.store = store
        self.embedding = embedding
        self.dimension = dimension
        self.last_embedding_ms = 0.0
        self.last_vector_query_ms = 0.0

    @classmethod
    def build(cls, nodes: list[TextNode], path: Path, embedding=None):
        embedding = embedding or APIEmbedding()
        path = Path(path)
        if (path / "dense.json").exists() or (path / "qdrant").exists():
            raise ValueError("dense index already exists; use a new index directory")
        if not nodes:
            raise ValueError("cannot build an empty dense index")
        try:
            vectors = embedding.embed([node.text for node in nodes], kind="document")
        except TypeError:
            # Legacy test/integration embeddings predate the Adapter kind.
            vectors = embedding.embed([node.text for node in nodes])
        dimension = len(vectors[0])
        if not dimension or len(vectors) != len(nodes) or any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding response has inconsistent dimensions")
        path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(path / "qdrant"))
        store = QdrantVectorStore(client=client, collection_name=cls.COLLECTION)
        indexed = []
        for node, vector in zip(nodes, vectors):
            copy = TextNode(
                id_=str(uuid5(NAMESPACE_URL, node.id_)), text=node.text,
                metadata=node.metadata.copy(), embedding=vector,
            )
            indexed.append(copy)
        store.add(indexed)
        metadata = getattr(embedding, "metadata", EmbeddingMetadata(
            "legacy", embedding.model, dimension, False, "", ""))
        if metadata.dimension != dimension:
            raise ValueError("embedding metadata dimension differs from generated vectors")
        (path / "dense.json").write_text(json.dumps({"model": embedding.model, "dimension": dimension,
                                                    "collection": cls.COLLECTION,
                                                    "embedding": metadata.to_dict()}, indent=2), encoding="utf-8")
        return cls(client, store, embedding, dimension)

    @classmethod
    def open(cls, path: Path, embedding=None):
        embedding = embedding or APIEmbedding()
        path = Path(path)
        config = json.loads((path / "dense.json").read_text(encoding="utf-8"))
        expected = config.get("embedding")
        actual = getattr(embedding, "metadata", None)
        if expected and actual is not None and expected != actual.to_dict():
            raise ValueError("embedding metadata changed; rebuild index")
        if config["model"] != embedding.model:
            raise ValueError("embedding model changed; rebuild index")
        client = QdrantClient(path=str(path / "qdrant"))
        collection = client.get_collection(config["collection"])
        vectors = collection.config.params.vectors
        actual_dimension = vectors.size if hasattr(vectors, "size") else next(iter(vectors.values())).size
        if actual_dimension != config["dimension"]:
            client.close()
            raise ValueError("embedding dimension changed; rebuild index")
        return cls(client, QdrantVectorStore(client=client, collection_name=config["collection"]), embedding, actual_dimension)

    def retrieve(self, question: str, top_k: int) -> list[NodeWithScore]:
        start = perf_counter()
        try:
            vector = self.embedding.embed([question], kind="query")[0]
        except TypeError:
            vector = self.embedding.embed([question])[0]
        self.last_embedding_ms = (perf_counter() - start) * 1000
        return self.retrieve_vector(vector, top_k)

    async def aretrieve(self, question: str, top_k: int) -> list[NodeWithScore]:
        start = perf_counter()
        try:
            vector = (await self.embedding.aembed([question], kind="query"))[0]
        except TypeError:
            vector = (await self.embedding.aembed([question]))[0]
        self.last_embedding_ms = (perf_counter() - start) * 1000
        return self.retrieve_vector(vector, top_k)

    def retrieve_vector(self, vector: list[float], top_k: int) -> list[NodeWithScore]:
        if len(vector) != self.dimension:
            raise ValueError("query embedding dimension differs from index")
        start = perf_counter()
        result = self.store.query(VectorStoreQuery(query_embedding=vector, similarity_top_k=top_k))
        self.last_vector_query_ms = (perf_counter() - start) * 1000
        return [NodeWithScore(node=node, score=score) for node, score in zip(result.nodes or [], result.similarities or [])]

    def close(self):
        self.client.close()
