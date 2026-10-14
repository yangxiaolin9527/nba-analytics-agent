"""Build a reproducible local BM25 corpus and optional Qdrant vector index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from scripts.knowledge_corpus import load_corpus, split_corpus
from scripts.knowledge_search import save_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, choices=[300, 500, 800], default=500)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--embedding-provider", choices=("openai_compatible", "ali_bailian", "local_sentence_transformers"), default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--embedding-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--embedding-max-length", type=int, default=None)
    args = parser.parse_args()
    overlap = args.overlap if args.overlap is not None else args.chunk_size // 10
    units = load_corpus(args.manifest)
    nodes = split_corpus(units, chunk_size=args.chunk_size, overlap=overlap)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if args.index.exists() and any(args.index.iterdir()):
        parser.error("index directory is not empty; choose a new path for a reproducible build")
    save_nodes(nodes, args.index, manifest_hash=manifest_hash, chunk_size=args.chunk_size, overlap=overlap)
    if args.dense:
        from scripts.knowledge_vector import DenseIndex
        from scripts.embedding_adapters import load_embedding_adapter
        from scripts.provider_config import load_provider_config
        config = load_provider_config()
        config = config.with_embedding_overrides(
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            embedding_normalize=args.embedding_normalize,
            embedding_local_device=args.embedding_device,
            embedding_local_batch_size=args.embedding_batch_size,
            embedding_local_max_length=args.embedding_max_length,
        )
        DenseIndex.build(nodes, args.index, load_embedding_adapter(config)).close()
    print(f"Indexed {len(nodes)} chunks from {len(units)} source locations at {args.index}")


if __name__ == "__main__":
    main()
