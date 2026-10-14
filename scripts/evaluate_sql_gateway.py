"""Run auditable SQL gateway cases against a SQLite connection."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from scripts.sql_gateway import GatewayStatus, SqlGateway


def evaluate_case(connection: sqlite3.Connection, case: dict[str, Any]) -> dict[str, Any]:
    result = SqlGateway(connection).execute(case["sql"])
    expected = case.get("expected")
    actual: dict[str, Any] = {}
    if result.status is GatewayStatus.OK and expected is not None:
        if result.rows:
            actual = dict(zip(result.columns, result.rows[0]))
        result_match = all(actual.get(key) == value for key, value in expected.items())
    elif expected is None:
        result_match = result.status is GatewayStatus.REJECTED
    else:
        result_match = False
    return {
        "case_id": case["case_id"],
        "status": "passed" if result_match else "failed",
        "gateway_status": result.status.value,
        "result_match": result_match,
        "columns": result.columns,
        "rows": result.rows,
        "reason": result.reason,
        "elapsed_ms": result.elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as connection:
        results = [evaluate_case(connection, case) for case in cases]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
