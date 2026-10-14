# Data preparation and provenance

## Structured data

Source: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores,
fixed version 515. Review the publisher's current terms before downloading.
The repository contains source identifiers, schema and derived demonstrations,
not the raw dataset or SQLite snapshot.

From the repository root, with uv dependencies installed:

```bash
bash scripts/download_eoin.sh --include-identity
mkdir -p data/profile runtime-data
uv run --no-sync python -m scripts.profile_eoin > data/profile/eoin-v515-profile.json
uv run --no-sync python -m scripts.validate_eoin > data/profile/eoin-v515-validation.json
uv run --no-sync python -m scripts.game_semantics > data/profile/eoin-v515-game-semantics.json
uv run --no-sync python -m scripts.player_quality > data/profile/eoin-v515-player-quality.json
uv run --no-sync python -m scripts.capability_matrix
uv run --no-sync python -m scripts.relational_baseline --db runtime-data/nba.sqlite
```

The download script verifies the three core CSVs against pinned sizes and hashes;
optional identity-file hashes are recorded when obtained. Confirm the generated
quality matrix at data/profile/eoin-v515-capability-matrix.json. The snapshot can
contain incomplete historical fields; capability/quality metadata matters even
when SQL is valid. These commands download/process substantial local resources
and are not executed by the default Docker replay.

## Knowledge corpus

Official sources: https://official.nba.com/rulebook/ and
https://www.nba.com/stats/help/glossary. Obtain permitted local copies yourself.
Copy metadata/knowledge_corpus_v1.example.json to a private manifest, fill the
edition/capture date, local source paths, terms and actual SHA-256 hashes.
Do not treat a public URL as redistribution permission.

```bash
cp metadata/knowledge_corpus_v1.example.json metadata/knowledge_corpus_v1.json
# Edit the manifest and .env.local before indexing.
uv run --no-sync python -m scripts.ingest_knowledge   --manifest metadata/knowledge_corpus_v1.json   --index runtime-data/knowledge --chunk-size 500 --overlap 50 --dense
```

Dense ingestion calls the explicitly configured embedding provider and incurs
its cost; rerunning into a nonempty index is rejected. The query configuration
must match index model, dimension, normalization and instructions. Indexes are
not interchangeable simply because vectors have the same dimension.

The selected historical online experiment uses qwen3.7-text-embedding with
1024 dimensions and normalization. Other supported API embeddings require their
own matching index. For local BGE/Qwen experimentation explicitly install
`uv sync --locked --extra local-embedding`; the default Docker image does not
include torch, GPU libraries or weights.
