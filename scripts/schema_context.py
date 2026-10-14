"""Compile a small, deterministic schema context for Text-to-SQL prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SchemaContextError(ValueError):
    """Raised when a requested schema context is not safe or available."""


def _load_catalog(path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaContextError(f"cannot load schema catalog: {exc}") from exc
    if not isinstance(catalog, dict) or not isinstance(catalog.get("tables"), dict):
        raise SchemaContextError("schema catalog has no tables")
    return catalog


def compile_context(
    catalog_path: Path,
    *,
    tables: list[str] | tuple[str, ...],
    metrics: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return stable prompt metadata for the explicitly selected query scope."""
    catalog = _load_catalog(catalog_path)
    selected_tables: dict[str, Any] = {}
    for name in sorted(set(tables)):
        entry = catalog["tables"].get(name)
        if entry is None:
            raise SchemaContextError(f"unknown table: {name}")
        if entry.get("layer") not in {"conformed", "mart"}:
            raise SchemaContextError(f"table is not queryable: {name}")
        columns = [
            column for column in entry.get("columns", [])
            if not column.endswith("_json") and not column.startswith("source_")
        ]
        selected_tables[name] = {
            key: entry[key]
            for key in ("layer", "grain", "primary_key", "columns", "source", "quality_filters")
            if key in entry
        }
        selected_tables[name]["columns"] = columns
    selected_metrics: dict[str, Any] = {}
    for name in sorted(set(metrics)):
        entry = catalog.get("metrics", {}).get(name)
        if entry is None:
            raise SchemaContextError(f"unknown metric: {name}")
        selected_metrics[name] = entry
    relations = {
        key: value
        for key, value in sorted(catalog.get("relations", {}).items())
        if key.split(".", 1)[0] in selected_tables
        and value.split(".", 1)[0] in selected_tables
    }
    return {
        "catalog_version": catalog.get("catalog_version"),
        "model_status": catalog.get("model_status"),
        "query_policy": catalog.get("query_policy", {}),
        "tables": selected_tables,
        "relations": relations,
        "metrics": selected_metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--metric", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(compile_context(args.catalog, tables=args.tables, metrics=args.metric), indent=2))
