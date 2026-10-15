# Evaluation and experimental evidence

Separate deterministic contracts, retrieval, answer support, runtime outcomes and
cost. Offline tests and archived replay do not measure live model accuracy.

## Frozen embedding comparison

The existing comparison uses 30 paired questions: 20 ordinary answer cases,
9 negatives and one supported premise denial. Retrieval metrics below use the
21 source-grounded cases. The set includes previously used development questions,
so it is not an entirely unseen holdout.

| Metric | Online Qwen | Local BGE-M3 | Denominator |
| --- | ---: | ---: | --- |
| Location Recall@5 | 1.0000 | 1.0000 | 21 |
| MRR@5 | 0.9325 | 0.9206 | 21 |
| nDCG@5, location-deduplicated ideal DCG | 0.9491 | 0.9410 | 21 |
| Supporting-chunk Recall@5 | 1.0000 | 0.9524 | 21 |
| Estimated endpoint p50 | 5.77 s | 6.42 s | 30 |
| Estimated endpoint p95 | 19.06 s | 14.22 s | 30 |
| Runtime timeout | 5/30 | 7/30 | 30 |

Both arms use a fixed corpus, chunks, question set, downstream model/prompt and
budgets; their embedding model/provider/index differs. Preflight planning time
was cached and added back to real downstream wall time. Each question/arm ran
once, with the online arm preceding the local arm. These are estimated endpoint
timings, not a complete outer-Agent HTTP benchmark or production SLO.

BGE did not meet the predeclared replacement gate (at least 15% better p50,
no increased timeout, bounded p95 degradation, quality and deployment readiness).
The online option was retained. BGE had a better p95 in this sample; do not claim
an overall or significant win. The paired mean-latency 95% interval spans zero,
and numerical per-case semantic-quality labels are incomplete. Recall is not
answer accuracy; retrieval ranking does not establish answer regression.

Frozen evidence: [paired statistics](../reports/embedding-final/paired-statistics-frozen.json)
and [promotion decision](../reports/embedding-final/promotion-decision-frozen.json).
They retain their historical experiment identity. The source hashes describe
local original reports, which are not all distributed; this release does not
claim full regeneration of those online runs.

## Structured-query route selection

An isolated study compares the current model-planned Agent, shared-grounding
tools/SQL, and request-time capability dispatch. The independent Oracle reads
raw facts and calculates results in Python; it does not call the tested compiler
or provide gold requirements to the runtime. Scores require the complete scope,
columns, rows, ordering and outcome; gateway acceptance alone is not correctness.

On 24 new author-selected cases repeated twice, the actual Agent passed 32/48
and online hybrid 43/48. Common queries were 20/20 versus 19/20, expansions 0/16
(explicit unsupported) versus 12/16, and protection 12/12 for both. Only 6/8
expanded scenarios passed twice, below 7/8; common P95 was 1.641 versus 3.266 s,
above the allowed 1.25 ratio. The replacement gate failed; defaults are unchanged.

Timings include real provider, IPC/ASGI and full response; worker bootstrap and
post-request Oracle are excluded. No TCP, concurrent-load, RAG or conversation
claim follows. Repeats are correlated, and small samples do not establish
statistical significance. Historical scores predate later IO/audit hardening;
the hardened version was only regressed on old cases, not another new holdout.

The [full study](research/structured-query-selection.md) explains the older
shared-grounding comparison, version boundaries, scan-budget diagnosis and
replication commands. The [aggregate-only export](../reports/structured-query-selection/summary.json)
omits raw provider outputs and source archives; it cannot independently replay
historical calls. The published cases are now replication material, not fresh
holdout data. Running them again requires the user's own data/config and incurs
provider charges; no live calls run in CI.

## Running offline checks

```bash
uv sync --locked
uv run --no-sync pytest -q
node --test tests/frontend/demo-state.test.cjs
```

Tests cover grounding, parameterized tools, evidence/language contracts,
requirement coverage, bounded execution, memory/concurrency and presentation.
Fixtures use deterministic adapters or tiny temporary SQLite resources.
No live provider credentials are supplied in CI. Historical/live markers remain
excluded by default. Replays preserve their original fixture/live labels and
are not evidence that today's service recomputed the displayed statistics.
