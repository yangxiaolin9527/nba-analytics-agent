import json
from pathlib import Path

import pytest

from scripts.schema_context import SchemaContextError, compile_context


CATALOG = Path(__file__).parents[1] / "metadata" / "schema_catalog.json"


def test_compile_context_selects_tables_metrics_and_policy_deterministically() -> None:
    context = compile_context(CATALOG, tables=["player_game_stat", "game"], metrics=["points_per_game"])

    assert context["catalog_version"] == "0.5.2"
    assert list(context["tables"]) == ["game", "player_game_stat"]
    assert context["metrics"]["points_per_game"]["formula"] == "points_total / games_appeared"
    assert context["query_policy"]["read_only"] is True
    assert "raw_record_json" not in context["tables"]["player_game_stat"]["columns"]


def test_compile_context_rejects_unknown_table_and_lineage_table() -> None:
    with pytest.raises(SchemaContextError, match="unknown table"):
        compile_context(CATALOG, tables=["does_not_exist"])

    with pytest.raises(SchemaContextError, match="not queryable"):
        compile_context(CATALOG, tables=["source_record"])


def test_compile_context_is_json_serializable() -> None:
    json.dumps(compile_context(CATALOG, tables=["game"]))
