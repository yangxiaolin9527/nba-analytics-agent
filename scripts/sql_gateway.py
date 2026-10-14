"""Safety gateway and bounded read-only executor for generated SQLite SQL."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

from scripts.query_lexicon import load_lexicon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "metadata" / "schema_catalog.json"


class GatewayStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TRUNCATED = "truncated"
    ERROR = "error"


@dataclass(frozen=True)
class GatewayResult:
    status: GatewayStatus
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    reason: str | None = None
    elapsed_ms: float = 0.0


class SqlGateway:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        catalog_path: Path = DEFAULT_CATALOG,
        max_seconds: float = 5.0,
        max_rows: int = 10_000,
    ) -> None:
        self.connection = connection
        self.max_seconds = max_seconds
        self.max_rows = max_rows
        self._schema = self._introspect_schema(catalog_path)
        self._domains = load_lexicon()["value_domains"]

    def _introspect_schema(self, catalog_path: Path) -> dict[str, set[str]]:
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        database_schema: dict[str, set[str]] = {}
        for (name,) in rows:
            columns = self.connection.execute(f'PRAGMA table_info("{name}")').fetchall()
            database_schema[name] = {str(row[1]) for row in columns}
        try:
            import json

            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load schema catalog: {exc}") from exc
        allowed = {
            name for name, item in catalog.get("tables", {}).items()
            if item.get("layer") in {"conformed", "mart"}
        }
        schema = {name: database_schema[name] for name in sorted(allowed & database_schema.keys())}
        if not schema:
            raise ValueError("SQLite database has no catalog-approved query tables")
        return schema

    def _validate(self, sql: str) -> None:
        try:
            statements = parse(sql, read="sqlite")
        except ParseError as exc:
            raise ValueError(f"SQL parse error: {exc}") from exc
        if len(statements) != 1:
            raise ValueError("only one SQL statement is allowed")
        statement = statements[0]
        if not isinstance(statement, exp.Select):
            raise ValueError("only SELECT statements are allowed")
        cte_names = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
        tables: dict[str, str] = {}
        for table in statement.find_all(exp.Table):
            name = table.name
            if name in cte_names:
                continue
            if name not in self._schema:
                raise ValueError(f"table is not allowlisted: {name}")
            if name in {"source_record", "source_file", "dataset_snapshot"}:
                raise ValueError(f"lineage table is not queryable: {name}")
            tables[table.alias_or_name] = name
        if not tables:
            raise ValueError("query must reference an allowlisted table")
        schema = {table: {column: "TEXT" for column in columns}
                  for table, columns in self._schema.items()}
        try:
            qualify(statement.copy(), dialect="sqlite", schema=schema)
        except OptimizeError as exc:
            raise ValueError(f"column is not allowlisted: {exc}") from exc
        self._validate_domain_filters(statement)

    def _validate_domain_filters(self, statement: exp.Select) -> None:
        for nested in statement.find_all(exp.Select):
            if nested is statement:
                continue
            if any(column.name in self._domains for expression in nested.expressions
                   for column in expression.find_all(exp.Column)):
                raise ValueError("domain columns cannot be projected through nested queries")
        for column in statement.find_all(exp.Column):
            if column.name not in self._domains:
                continue
            if column.find_ancestor(exp.Where, exp.Having, exp.Join) is None:
                continue
            parent = column.parent
            while isinstance(parent, exp.Paren):
                parent = parent.parent
            if isinstance(parent, (exp.EQ, exp.NEQ)):
                other = parent.right if parent.left is column else parent.left
                values = [other]
            elif isinstance(parent, exp.In) and parent.this is column and not parent.args.get("query"):
                values = parent.expressions
            else:
                raise ValueError(f"domain filter must use a canonical literal: {column.name}")
            allowed = self._domains[column.name]
            if not values or any(not isinstance(value, exp.Literal) or not value.is_string
                                 or value.this not in allowed for value in values):
                raise ValueError(f"domain value is not canonical: {column.name}")

    def execute(self, sql: str) -> GatewayResult:
        started = time.monotonic()
        try:
            self._validate(sql)
        except ValueError as exc:
            return GatewayResult(GatewayStatus.REJECTED, reason=str(exc), elapsed_ms=_elapsed(started))
        deadline = started + self.max_seconds
        interrupted = False

        def progress() -> int:
            return int(time.monotonic() >= deadline)

        try:
            self.connection.execute("PRAGMA query_only = ON")
            self.connection.set_progress_handler(progress, 1_000)
            cursor = self.connection.execute(sql)
            rows = cursor.fetchmany(self.max_rows + 1)
            if len(rows) > self.max_rows:
                return GatewayResult(
                    GatewayStatus.TRUNCATED,
                    [description[0] for description in cursor.description or ()],
                    rows[:self.max_rows],
                    reason=f"result exceeds {self.max_rows} rows",
                    elapsed_ms=_elapsed(started),
                )
            return GatewayResult(
                GatewayStatus.OK,
                [description[0] for description in cursor.description or ()],
                rows,
                elapsed_ms=_elapsed(started),
            )
        except sqlite3.OperationalError as exc:
            message = str(exc)
            status = GatewayStatus.TIMEOUT if "interrupted" in message.lower() else GatewayStatus.ERROR
            return GatewayResult(status, reason=message, elapsed_ms=_elapsed(started))
        finally:
            self.connection.set_progress_handler(None, 0)


def _elapsed(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--sql", required=True)
    args = parser.parse_args()
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as connection:
        result = SqlGateway(connection).execute(args.sql)
    print({
        "status": result.status.value,
        "columns": result.columns,
        "rows": result.rows,
        "reason": result.reason,
        "elapsed_ms": result.elapsed_ms,
    })
