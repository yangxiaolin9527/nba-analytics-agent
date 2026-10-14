import sqlite3
from pathlib import Path

import pytest

from scripts.query_lexicon import load_lexicon, validate_lexicon


def test_query_lexicon_contains_canonical_phase_and_appearance_values() -> None:
    lexicon = load_lexicon()

    assert lexicon["value_domains"]["competition_phase"]["regular_season"] == ["常规赛", "Regular Season"]
    assert lexicon["value_domains"]["appearance_status"]["played_inferred"] == []
    assert lexicon["entity_aliases"]["player"]["库里"] == "201939"


def test_query_lexicon_rejects_alias_not_present_in_database(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "fixture.sqlite")
    connection.execute("CREATE TABLE player_identity (person_id TEXT PRIMARY KEY, display_name TEXT)")
    connection.execute("CREATE TABLE team_identity (team_id TEXT PRIMARY KEY, display_name TEXT)")
    lexicon = load_lexicon()

    with pytest.raises(ValueError, match="unknown player identity"):
        validate_lexicon(connection, lexicon)

    connection.close()
