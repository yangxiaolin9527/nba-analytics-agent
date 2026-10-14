import sqlite3
from pathlib import Path

import pytest

from scripts.query_context import build_query_context, select_tables
from scripts.ask_grounding import ground_ask_question
from scripts.relational_baseline import create_schema, insert_fixture_rows


def fixture_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        create_schema(connection)
        insert_fixture_rows(connection, games=[], players=[], teams=[])
        connection.execute("INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES ('201939', 'Stephen Curry', 'valid')")
        connection.execute("INSERT INTO team_identity (team_id, display_name, identity_quality) VALUES ('1610612743', 'Denver Nuggets', 'valid')")


def test_schema_retrieval_selects_only_relevant_allowlisted_tables() -> None:
    assert select_tables("库里在 2023-24 常规赛对掘金的场均得分？") == (
        "game", "player_game_stat", "player_identity", "team_identity"
    )
    assert select_tables("2023-24 常规赛湖人总得分？") == (
        "team_season_stat", "team_identity"
    )


def test_value_retrieval_links_curated_entities_and_uses_physical_schema(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)

    result = build_query_context(
        database, "库里在 2023-24 常规赛对掘金的场均得分？", "schema_value_retrieval"
    )

    assert result.links.players[0].id == "201939"
    assert result.links.teams[0].id == "1610612743"
    assert result.links.phases == ["regular_season"]
    assert result.links.seasons == ["2023-24"]
    assert set(result.context["tables"]) == set(select_tables("库里在 2023-24 常规赛对掘金的场均得分？"))
    assert "Table 'player_game_stat'" in result.context["physical_schema"]["player_game_stat"]
    assert "sample rows" not in str(result.context).lower()


def test_value_retrieval_resolves_database_display_names_without_alias(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)

    result = build_query_context(
        database, "Stephen Curry 在 2023-24 常规赛得分？", "schema_value_retrieval"
    )

    assert [(item.id, item.display_name) for item in result.links.players] == [
        ("201939", "Stephen Curry")
    ]
    assert set(result.context["tables"]) == {"player_season_stat", "player_identity"}


def test_value_retrieval_marks_ambiguous_database_display_names(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("1", "Alex Smith", "valid"),
        )
        connection.execute(
            "INSERT INTO player_identity (person_id, display_name, identity_quality) VALUES (?, ?, ?)",
            ("2", "Alex Smith", "valid"),
        )
        connection.commit()

    result = build_query_context(
        database, "Alex Smith 在 2023-24 常规赛得分？", "schema_value_retrieval"
    )

    assert result.links.players == []
    assert result.links.ambiguous_entities == ["player:Alex Smith"]


def test_schema_retrieval_does_not_inject_entity_ids(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)

    result = build_query_context(database, "库里在 2023-24 常规赛得分？", "schema_retrieval")

    assert result.links.players == []
    assert "201939" not in str(result.context)


def test_game_date_is_not_misread_as_season(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)

    result = build_query_context(database, "2023-10-27 凯尔特人对热火比分？", "schema_value_retrieval")

    assert result.links.seasons == []


@pytest.mark.parametrize('spelling', ['2024-25', '2024-2025', '2024 / 2025', '2024–2025', '2024—25'])
def test_grounding_normalizes_equivalent_season_spellings(tmp_path, spelling):
    database = tmp_path / 'fixture.sqlite'
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        links, normalized = ground_ask_question(connection, f'库里 {spelling} 季后赛总得分？')
    assert links.seasons == ['2024-25']
    assert '2024-25' in normalized


def test_full_season_normalization_preserves_game_dates(tmp_path):
    database = tmp_path / 'fixture.sqlite'
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        links, normalized = ground_ask_question(connection, '库里 2024-2025 常规赛从 2024-10-27 到 2025-01-02 总得分？')
    assert links.seasons == ['2024-25']
    assert '2024-10-27' in normalized and '2025-01-02' in normalized


def test_historical_full_year_season_is_not_treated_as_missing(tmp_path):
    database = tmp_path / 'fixture.sqlite'
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        links, normalized = ground_ask_question(connection, '库里 1999-2000 常规赛总得分？')
    assert links.seasons == ['1999-00']
    assert '1999-00' in normalized


def test_leonard_aliases_are_verified_against_database_identity(tmp_path):
    database = tmp_path / 'fixture.sqlite'
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        # The alias is insufficient when the pinned database lacks that identity.
        assert ground_ask_question(connection, '伦纳德 2024-25 季后赛总得分？')[0].players == []
        connection.execute("INSERT INTO player_identity VALUES ('202695', 'Kawhi Leonard', 'valid')")
        connection.execute("INSERT INTO player_identity VALUES ('203086', 'Meyers Leonard', 'valid')")
        for name in ['伦纳德', '科怀·伦纳德', '莱昂纳德', 'Kawhi Leonard']:
            links, _ = ground_ask_question(connection, f'{name} 2024-25 季后赛总得分？')
            assert [player.id for player in links.players] == ['202695']


def test_curated_alias_does_not_hide_another_explicit_database_name(tmp_path):
    database = tmp_path / 'fixture.sqlite'
    fixture_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO player_identity VALUES ('202695', 'Kawhi Leonard', 'valid')")
        links, _ = ground_ask_question(connection, '比较库里和 Kawhi Leonard 2024-25 季后赛 eFG%')
    assert {player.id for player in links.players} == {'201939', '202695'}


def test_opponent_scope_does_not_inject_all_historical_games(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite"
    fixture_database(database)
    result = build_query_context(
        database, "库里在 2023-24 常规赛对掘金的场均得分？", "schema_value_retrieval"
    )

    assert result.links.games == []
    assert len(result.links.model_dump_json()) < 1000
