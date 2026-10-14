"""Provider-independent dense embedding adapters used by index build and query paths."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx


EmbeddingKind = Literal["query", "document"]


def _normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    normalized = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("embedding response contains a zero vector")
        normalized.append([value / norm for value in vector])
    return normalized


@dataclass(frozen=True)
class EmbeddingMetadata:
    provider: str
    model: str
    dimension: int
    normalized: bool
    query_instruction: str
    document_instruction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmbeddingAdapter(Protocol):
    metadata: EmbeddingMetadata

    @property
    def model(self) -> str: ...

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]: ...

    async def aembed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]: ...


def _validate_vectors(vectors: list[list[float]], expected: int, dimension: int) -> list[list[float]]:
    if len(vectors) != expected:
        raise ValueError("embedding response has inconsistent dimensions")
    if not vectors and expected == 0:
        return []
    if not vectors or len(vectors[0]) != dimension:
        raise ValueError(f"embedding response dimension differs from configured dimension {dimension}")
    if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
        raise ValueError("embedding response contains non-finite values")
    if not vectors[0]:
        raise ValueError("embedding response has empty vectors")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding response has inconsistent dimensions")
    return [list(map(float, vector)) for vector in vectors]


class APIEmbeddingAdapter:
    def __init__(self, *, base_url: str, api_key: str, model: str, dimension: int,
                 normalized: bool = True, timeout: float = 30.0, batch_size: int = 20,
                 provider: str = "openai_compatible"):
        if dimension <= 0 or batch_size <= 0:
            raise ValueError("dimension and batch_size must be positive")
        self.base_url, self.api_key, self._model = base_url.rstrip("/"), api_key, model
        self.dimension, self.normalized, self.timeout, self.batch_size = dimension, normalized, timeout, batch_size
        self.metadata = EmbeddingMetadata(provider, model, dimension, normalized, "", "")

    @classmethod
    def from_config(cls, config):
        return cls(base_url=config.embedding_base_url, api_key=config.embedding_api_key,
            model=config.embedding_model, dimension=config.embedding_dimension,
            normalized=config.embedding_normalize, timeout=config.embedding_timeout_seconds,
            batch_size=20 if config.embedding_provider == "ali_bailian" else 64,
            provider=config.embedding_provider)

    @property
    def model(self):
        return self._model

    def _body(self, input_texts: list[str]) -> dict[str, Any]:
        return {"model": self._model, "input": input_texts, "dimensions": self.dimension,
                "encoding_format": "float"}

    @staticmethod
    def _vectors(body: dict[str, Any], count: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != count:
            raise ValueError("embedding response has missing vectors")
        ordered = sorted(data, key=lambda item: item.get("index", -1))
        if [item.get("index") for item in ordered] != list(range(count)):
            raise ValueError("embedding response has missing or duplicate indices")
        return [item["embedding"] for item in ordered]

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]:
        del kind
        result = []
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                response = client.post(f"{self.base_url}/embeddings",
                                       headers={"Authorization": f"Bearer {self.api_key}"},
                                       json=self._body(texts[start:start + self.batch_size]))
                response.raise_for_status()
                result.extend(self._vectors(response.json(), min(self.batch_size, len(texts) - start)))
        vectors = _validate_vectors(result, len(texts), self.dimension)
        if self.normalized:
            vectors = _normalize_vectors(vectors)
        return vectors

    async def aembed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]:
        del kind
        result = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                response = await client.post(f"{self.base_url}/embeddings",
                                             headers={"Authorization": f"Bearer {self.api_key}"},
                                             json=self._body(texts[start:start + self.batch_size]))
                response.raise_for_status()
                result.extend(self._vectors(response.json(), min(self.batch_size, len(texts) - start)))
        vectors = _validate_vectors(result, len(texts), self.dimension)
        if self.normalized:
            vectors = _normalize_vectors(vectors)
        return vectors


class LocalSentenceTransformerAdapter:
    def __init__(self, model, *, metadata: EmbeddingMetadata, batch_size: int = 8,
                 normalize: bool = True, device: str | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if metadata.normalized != normalize:
            raise ValueError("embedding metadata normalization differs from encoder configuration")
        self.model_instance, self.metadata = model, metadata
        self.batch_size, self.normalize, self.device = batch_size, normalize, device

    @classmethod
    def from_model(cls, model, *, model_name: str, dimension: int,
                   query_prompt_name: str | None = None, provider: str = "local_sentence_transformers",
                   batch_size: int = 8, device: str | None = None):
        return cls(model, metadata=EmbeddingMetadata(provider, model_name, dimension, True,
            query_prompt_name or "", ""), batch_size=batch_size, device=device)

    @property
    def model(self):
        return self.metadata.model

    def embed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]:
        kwargs = {"batch_size": self.batch_size, "normalize_embeddings": self.normalize,
                  "convert_to_numpy": True, "show_progress_bar": False}
        if self.device:
            kwargs["device"] = self.device
        if kind == "query" and self.metadata.query_instruction:
            kwargs["prompt_name"] = self.metadata.query_instruction
        vectors = self.model_instance.encode(texts, **kwargs)
        vectors = _validate_vectors(vectors.tolist() if hasattr(vectors, "tolist") else vectors,
                                    len(texts), self.metadata.dimension)
        return _normalize_vectors(vectors) if self.normalize else vectors

    async def aembed(self, texts: list[str], *, kind: EmbeddingKind = "query") -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self.embed, texts, kind=kind)


def load_local_sentence_transformer(model_name: str, *, device: str = "cuda",
                                    batch_size: int = 8, max_seq_length: int = 2048):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("local embedding requires sentence-transformers and torch") from exc
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    prompt_name = "query" if "Qwen3-Embedding" in model_name else None
    return LocalSentenceTransformerAdapter.from_model(model, model_name=model_name,
        dimension=1024, query_prompt_name=prompt_name, batch_size=batch_size, device=device)


def load_embedding_adapter(config):
    if config.embedding_provider in {"openai_compatible", "ali_bailian"}:
        return APIEmbeddingAdapter.from_config(config)
    if config.embedding_provider == "local_sentence_transformers":
        if config.embedding_dimension != 1024 or not config.embedding_normalize:
            raise ValueError("local embedding requires dimension 1024 and normalization enabled")
        return load_local_sentence_transformer(config.embedding_model,
            device=config.embedding_local_device, batch_size=config.embedding_local_batch_size,
            max_seq_length=config.embedding_local_max_length)
    raise ValueError(f"unsupported embedding provider: {config.embedding_provider}")
