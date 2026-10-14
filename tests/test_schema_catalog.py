import json
import sqlite3
from pathlib import Path

from scripts.relational_baseline import create_schema


CATALOG = Path(__file__).parents[1] / "metadata/schema_catalog.json"


def test_schema_catalog_has_safe_query_policy_and_required_grains() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    assert catalog["query_policy"]["read_only"] is True
    assert catalog["query_policy"]["unknown_availability"] == "reject_or_request_clarification"
    assert catalog["tables"]["player_game_stat"]["grain"] == "one source player record per game"
    assert catalog["tables"]["game_team_participation"]["primary_key"] == ["game_id", "team_id"]
    assert catalog["model_status"] == "m1-physical-sqlite-baseline"
    assert "raw_record_json" in catalog["tables"]["player_game_stat"]["columns"]


def test_efg_metric_declares_inputs_and_zero_denominator_boundary() -> None:
    metric = json.loads(CATALOG.read_text(encoding="utf-8"))["metrics"]["efg_percentage"]

    assert metric["inputs"] == ["field_goals_made", "three_pointers_made", "field_goals_attempted"]
    assert "field_goals_attempted > 0" in metric["availability"]
    assert metric["do_not_average_rows"] is True


def test_exact_oracles_pin_the_four_m1_reference_cases() -> None:
    oracle = json.loads((CATALOG.parent / "agent_evaluation_oracles_v1.json").read_text(encoding="utf-8"))

    assert oracle["source_snapshot"].endswith("@515")
    assert set(oracle["cases"]) >= {"AG-001", "AG-002", "AG-004", "AG-009"}
    assert oracle["cases"]["AG-001"]["points_total"] == 1956
    assert oracle["cases"]["AG-004"]["points_per_appearance"] == 33.0


def test_advertised_query_columns_exist_in_sqlite_schema() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    with sqlite3.connect(":memory:") as connection:
        create_schema(connection)
        for name, entry in catalog["tables"].items():
            if entry["layer"] not in {"conformed", "mart"}:
                continue
            actual = {row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')}
            assert set(entry["columns"]) <= actual, name
