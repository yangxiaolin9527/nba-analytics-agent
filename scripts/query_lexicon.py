"""Versioned database value domains and curated query aliases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from scripts.game_semantics import RULES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEXICON = ROOT / "metadata" / "query_lexicon_v1.json"
DEFAULT_CATALOG = ROOT / "metadata" / "schema_catalog.json"


def load_lexicon(path: Path = DEFAULT_LEXICON) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_lexicon(connection: sqlite3.Connection, lexicon: dict[str, Any]) -> None:
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    if lexicon["source_snapshot"] != catalog["source_snapshot"]:
        raise ValueError("query lexicon snapshot does not match schema catalog")
    phases = set(rules["phase_by_type"].values()) | set(rules["phase_by_id_prefix"].values())
    if set(lexicon["value_domains"]["competition_phase"]) != phases:
        raise ValueError("competition phase domain does not match game semantics")
    if not set(lexicon["supported_phases"]) <= phases:
        raise ValueError("supported phase is not in game semantics")
    for kind, table, key in (
        ("player", "player_identity", "person_id"),
        ("team", "team_identity", "team_id"),
    ):
        for identity_id in sorted(set(lexicon["entity_aliases"][kind].values())):
            row = connection.execute(f"SELECT 1 FROM {table} WHERE {key} = ?", (identity_id,)).fetchone()
            if row is None:
                raise ValueError(f"unknown {kind} identity: {identity_id}")
    for table, column, domain in (
        ("game", "competition_phase", "competition_phase"),
        ("player_game_stat", "appearance_status", "appearance_status"),
        ("game_team_participation", "home_away", "home_away"),
    ):
        observed = {row[0] for row in connection.execute(f"SELECT DISTINCT {column} FROM {table}")}
        if not observed <= set(lexicon["value_domains"][domain]):
            raise ValueError(f"query lexicon missing database values for {table}.{column}")
