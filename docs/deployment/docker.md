# Running the project

## Credential-free replay

```bash
docker compose up --build -d
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
# Open http://127.0.0.1:8000/demo
docker compose down
```

Requires Docker Engine/Desktop with Compose v2 and internet access on first
build. `NBA_PORT=8001 docker compose up --build -d` chooses another host port.
Compose binds localhost. No provider, database, index or GPU is needed. Select a
registered replay; entering a new live question requires real resources.
`/readyz` checks replay readiness, not live provider health.

## Why code changes do not reinstall everything

The Dockerfile installs locked default dependencies before copying source, using
a uv download cache and separate runtime stage. Model extras are opt-in.
The final runtime directly invokes the installed interpreter, so startup does
not call uv sync or download a model. No old project image or host .venv is used.
Code-only builds reuse the dependency layer; lock/dependency changes rebuild it.

On the release workstation, the default image measured 590,374,532 bytes
(about 590 MB); a CSS-only rebuild took 6.65 seconds with dependency installation
cached. These are local build observations, not a cold-build or hardware guarantee.
The release check executes all 16 packaged replays; CI runs the same script:

```bash
docker compose exec -T nba-agent python - < tests/smoke_replay.py
```

For development without copying code into a new image on every edit:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up --build
```

This mounts source directories read-only and enables Uvicorn reload. The packaged
release is tested without these mounts. Do not run both services on the same port.
Do not mount the repository root or the host virtual environment over /app.

## Real API mode: explicit resource mounts

Prepare data and index as described in [data preparation](../data-sources.md).
Supply the following paths:

```text
runtime-data/nba.sqlite
runtime-data/profile/eoin-v515-capability-matrix.json
runtime-data/knowledge/  (versioned nodes + Dense index)
.env.container.local   (private, ignored)
```

Copy data/profile's generated quality metadata to runtime-data/profile.
Copy .env.example to .env.container.local and edit it for the container:
`NBA_DB_PATH=/runtime/nba.sqlite`, `NBA_KNOWLEDGE_INDEX_PATH=/runtime/knowledge`.
Set model and embedding endpoint/key/model explicitly and ensure embedding
metadata matches the index. Keep Phoenix disabled unless an external collector
is configured. Enable the optional loop with NBA_AGENT_UNIFIED_LOOP_ENABLED=1
only when desired. Configuration is read from the mounted /app/.env.local;
shell/provider environment variables are intentionally not a fallback.

```bash
docker compose -f compose.yaml -f compose.live.yaml up --build -d
curl -fsS http://127.0.0.1:8000/demo/status
```

The override refuses to invent missing resource directories. UID 10001 needs
read access to config/SQLite/profile and write access to the Qdrant directory;
grant that access to the prepared copy rather than changing the original index.
Run one process per local Qdrant index; do not mount the same index into concurrent
workers. Stop the container before rebuilding that index. This release documents
the real-resource path but does not perform new paid provider experiments.

## Local Python mode

```bash
uv sync --locked
uv run --no-sync uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Local .env.local settings may activate real resources. For a credential-free
local run use a fresh clone without .env.local. Local model experiments require
`uv sync --locked --extra local-embedding`; no automatic fallback is selected.
