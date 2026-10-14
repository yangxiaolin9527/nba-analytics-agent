"""Lexical, dense and reciprocal-rank fusion retrieval over pinned chunks."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from scripts.provider_config import ProviderConfig, load_provider_config


TERM_MAP = {
    "有效命中率": "effective field goal percentage eFG",
    "使用率": "usage percentage",
    "走步": "traveling",
    "犯规": "foul",
    "进攻犯规": "offensive foul",
    "防守三秒": "defensive three seconds",
    "24秒": "shot clock",
    "三分": "three point field goal",
}


def lexical_query(question: str) -> str:
    return " ".join([question, *(english for chinese, english in TERM_MAP.items() if chinese in question)])


class KnowledgeSearch:
    def __init__(self, nodes: list[TextNode], dense=None, *, manifest_hash: str | None = None):
        if not nodes:
            raise ValueError("knowledge index has no nodes")
        self.nodes = {node.id_: node for node in nodes}
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(10, len(nodes)))
        self.dense = dense
        self.manifest_hash = manifest_hash

    def search(self, question: str, *, mode: str = "bm25", top_k: int = 5,
               _vector=None) -> dict:
        if mode not in {"bm25", "dense", "rrf"}:
            raise ValueError("unsupported retrieval mode")
        if mode in {"dense", "rrf"} and self.dense is None:
            raise RuntimeError("dense index is unavailable")
        start = perf_counter()
        lexical = self.bm25.retrieve(lexical_query(question)) if mode in {"bm25", "rrf"} else []
        vector = (_vector if _vector is not None else self.dense.retrieve(question, max(top_k, 10))) if mode in {"dense", "rrf"} else []
        if mode == "rrf":
            scores: dict[str, float] = {}
            for ranked in (lexical, vector):
                for rank, item in enumerate(ranked, 1):
                    key = item.node.metadata["citation_id"]
                    scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            ranked_hits = [(self.nodes[key], score) for key, score in sorted(scores.items(), key=lambda pair: -pair[1])]
        else:
            ranked_hits = [(item.node, item.score or 0.0) for item in (lexical if mode == "bm25" else vector)]
        hits = []
        for node, score in ranked_hits[:top_k]:
            meta = node.metadata
            hits.append({
                "citation_id": meta["citation_id"], "excerpt": node.text,
                "source_id": meta["id"], "source_url": meta["source_url"],
                "source_version": meta["version"], "sha256": meta["sha256"],
                "page": meta["page"], "heading": meta["heading"], "anchor": meta.get("anchor"),
                "score": float(score),
            })
        return {"mode": mode, "hits": hits, "elapsed_ms": round((perf_counter() - start) * 1000, 3),
                "embedding_ms": round(self.dense.last_embedding_ms, 3) if vector else 0.0,
                "vector_query_ms": round(self.dense.last_vector_query_ms, 3) if vector else 0.0}

    async def asearch(self, question: str, *, mode: str = "bm25", top_k: int = 5) -> dict:
        if mode not in {"bm25", "dense", "rrf"}:
            raise ValueError("unsupported retrieval mode")
        if mode == "bm25":
            return self.search(question, mode=mode, top_k=top_k)
        if self.dense is None:
            raise RuntimeError("dense index is unavailable")
        start = perf_counter()
        vector = await self.dense.aretrieve(question, max(top_k, 10))
        result = self.search(question, mode=mode, top_k=top_k, _vector=vector)
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        return result


def save_nodes(nodes: list[TextNode], path: Path, *, manifest_hash: str, chunk_size: int, overlap: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "manifest_hash": manifest_hash, "chunk_size": chunk_size,
               "overlap": overlap, "nodes": [{"id": n.id_, "text": n.text, "metadata": n.metadata} for n in nodes]}
    (path / "nodes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_nodes(path: Path, *, manifest_hash: str | None = None) -> list[TextNode]:
    payload = json.loads((path / "nodes.json").read_text(encoding="utf-8"))
    if manifest_hash and payload["manifest_hash"] != manifest_hash:
        raise ValueError("corpus manifest changed; rebuild index")
    return [TextNode(id_=item["id"], text=item["text"], metadata=item["metadata"]) for item in payload["nodes"]]


def configured_search(path: Path | None = None, *, provider_config: ProviderConfig | None = None) -> KnowledgeSearch | None:
    config = provider_config or load_provider_config()
    configured = path or config.knowledge_index_path
    if not configured:
        return None
    index_path = Path(configured)
    if not (index_path / "nodes.json").exists():
        return None
    nodes = load_nodes(index_path)
    dense = None
    if (index_path / "dense.json").exists() and config.embedding_configured:
        from scripts.knowledge_vector import DenseIndex
        from scripts.knowledge_vector import APIEmbedding
        metadata = json.loads((index_path / "dense.json").read_text(encoding="utf-8"))
        if "embedding" not in metadata:
            if (config.embedding_model != metadata.get("model")
                    or config.embedding_dimension != metadata["dimension"]
                    or config.embedding_provider != "openai_compatible"
                    or config.embedding_normalize):
                raise ValueError("legacy embedding index differs from local configuration; restore original provider/model or select a matching index")
            dense = DenseIndex.open(index_path, APIEmbedding(config))
        else:
            from scripts.embedding_adapters import load_embedding_adapter
            adapter = load_embedding_adapter(config)
            dense = DenseIndex.open(index_path, adapter)
    manifest_hash = json.loads((index_path / "nodes.json").read_text(encoding="utf-8"))["manifest_hash"]
    return KnowledgeSearch(nodes, dense, manifest_hash=manifest_hash)
