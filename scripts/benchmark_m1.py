"""Measure deterministic M1 API latency under a fixed local workload."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from scripts.model_adapter import FakeModel, ModelRequest, ModelResponse
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.evaluation_contract import report_provenance

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "metadata/schema_catalog.json"


class FixedSQLModel:
    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            "SELECT home_score, away_score FROM game WHERE game_id = '22300061'",
            "benchmark-fixed-sql",
        )


ASK_PLAN = json.dumps({
    "action": "call",
    "tool": "window_query",
    "arguments": {
        "subject": "player", "entity_id": "201939", "season_id": "2023-24",
        "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 10,
    },
}, ensure_ascii=False)


def _load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


async def _measure(
    app,
    request_factory: Callable[[AsyncClient], Awaitable[Any]],
    *, iterations: int, concurrency: int,
) -> dict[str, Any]:
    durations: list[float] = []
    errors = 0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://benchmark") as client:
        for start in range(0, iterations, concurrency):
            batch_size = min(concurrency, iterations - start)

            async def one() -> tuple[float, bool]:
                started = time.perf_counter()
                response = await request_factory(client)
                elapsed = (time.perf_counter() - started) * 1000
                body = response.json()
                ok = response.status_code == 200 and body.get("status") in {
                    "ok", "complete", "partial",
                }
                return elapsed, ok

            results = await asyncio.gather(*(one() for _ in range(batch_size)))
            for elapsed, ok in results:
                durations.append(elapsed)
                errors += int(not ok)
    return {
        "iterations": iterations,
        "concurrency": concurrency,
        "errors": errors,
        "latency_ms": {
            "mean": round(statistics.mean(durations), 3) if durations else None,
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "p99": _percentile(durations, 0.99),
            "max": round(max(durations), 3) if durations else None,
        },
    }


async def run_benchmark(
    db_path: Path, *, matrix_path: Path = CAPABILITY_MATRIX,
    iterations: int = 20, concurrencies: tuple[int, ...] = (1, 4),
) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    if iterations < 20:
        raise ValueError("iterations must be at least 20 per workload")
    if any(value < 1 for value in concurrencies):
        raise ValueError("concurrency must be positive")
    catalog = _load_catalog()
    apps = {
        "query": create_app(model=FixedSQLModel(), db_path=db_path, quality_matrix_path=matrix_path),
        "window_query": create_app(db_path=db_path, quality_matrix_path=matrix_path),
        "season_compare": create_app(db_path=db_path, quality_matrix_path=matrix_path),
        "player_efg": create_app(db_path=db_path, quality_matrix_path=matrix_path),
        "ask": create_app(
            db_path=db_path, quality_matrix_path=matrix_path,
            model=FakeModel({None: ASK_PLAN}),
        ),
    }

    async def query(client: AsyncClient):
        return await client.post("/query", json={"question": "2023-24 常规赛湖人对掘金的比分？"})

    async def window(client: AsyncClient):
        return await client.post("/query/window", json={
            "subject": "player", "entity_id": "201939", "season_id": "2023-24",
            "competition_phase": "regular_season", "metrics": ["points_total"], "last_n": 10,
        })

    async def compare(client: AsyncClient):
        return await client.post("/calculate/compare", json={
            "subject": "player", "entity_id": "201939", "metric": "points_total",
            "baseline_season": "2022-23", "comparison_season": "2023-24",
            "competition_phase": "regular_season",
        })

    async def efg(client: AsyncClient):
        return await client.post("/calculate/efg", json={
            "entity_id": "201939", "season_id": "2023-24", "competition_phase": "regular_season",
        })

    async def ask(client: AsyncClient):
        return await client.post("/ask", json={"question": "库里在 2023-24 常规赛最近十次出场的总得分？"})

    factories = {"query": query, "window_query": window, "season_compare": compare,
                 "player_efg": efg, "ask": ask}
    results: dict[str, list[dict[str, Any]]] = {}
    for concurrency in concurrencies:
        for name, app in apps.items():
            measured = await _measure(
                app, factories[name], iterations=iterations, concurrency=concurrency,
            )
            results.setdefault(name, []).append(measured)
    return {
        "benchmark_version": "1.0.0",
        "source_snapshot": catalog["source_snapshot"],
        "catalog_version": catalog["catalog_version"],
        "workload": {
            "iterations_per_endpoint_and_concurrency": iterations,
            "concurrencies": list(concurrencies),
            "transport": "httpx ASGITransport",
            "mode": "fixed SQL/model plans and read-only deterministic tools",
            "endpoints": {"query": 0.40, "window_query": 0.20, "season_compare": 0.15,
                          "player_efg": 0.15, "ask": 0.10},
        },
        "results": results,
        "provenance": report_provenance(
            root=ROOT, source_snapshot=catalog["source_snapshot"], model_id="fixed-benchmark",
            route="m1_performance", input_paths={"catalog": CATALOG, "benchmark": Path(__file__)},
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--capability-matrix", type=Path, default=CAPABILITY_MATRIX)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    concurrencies = tuple(int(value) for value in args.concurrency.split(",") if value.strip())
    try:
        report = asyncio.run(run_benchmark(
            args.db, matrix_path=args.capability_matrix,
            iterations=args.iterations, concurrencies=concurrencies,
        ))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"benchmark failed: {exc}")
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(any(item["errors"] for rows in report["results"].values() for item in rows))


if __name__ == "__main__":
    raise SystemExit(main())
