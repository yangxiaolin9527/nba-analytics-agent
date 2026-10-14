"""Shared catalog, database evidence and runtime context helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from scripts.schema_context import compile_context
from scripts.sql_gateway import DEFAULT_CATALOG

from app.schemas import Evidence


def query_context() -> tuple[dict[str, Any], str]:
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    tables = [
        name for name, entry in catalog["tables"].items()
        if entry.get("layer") in {"conformed", "mart"}
    ]
    context = compile_context(
        DEFAULT_CATALOG,
        tables=tables,
        metrics=list(catalog.get("metrics", {})),
    )
    return context, str(catalog["source_snapshot"])


def database_evidence(
    connection: sqlite3.Connection,
    context: dict[str, Any],
    source_snapshot: str,
) -> Evidence:
    snapshots = connection.execute(
        "SELECT snapshot_id, source_name, source_version FROM dataset_snapshot LIMIT 2"
    ).fetchall()
    if len(snapshots) != 1:
        raise ValueError("expected one dataset snapshot")
    snapshot_id, source_name, source_version = snapshots[0]
    if f"{source_name}@{source_version}" != source_snapshot:
        raise ValueError("database snapshot does not match schema catalog")
    return Evidence(
        source_name=source_name,
        source_version=source_version,
        snapshot_id=snapshot_id,
        catalog_version=context["catalog_version"],
    )
