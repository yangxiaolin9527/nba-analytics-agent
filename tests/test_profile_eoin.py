import csv
from pathlib import Path

from scripts.profile_eoin import file_inventory, profile_identity_file


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_profiles_identity_ids_without_using_display_names(tmp_path: Path) -> None:
    path = tmp_path / "Players.csv"
    write_csv(
        path,
        ["personId", "firstName", "lastName"],
        [
            {"personId": "p1", "firstName": "A", "lastName": "One"},
            {"personId": "p1", "firstName": "A", "lastName": "One"},
            {"personId": "", "firstName": "B", "lastName": "Two"},
        ],
    )

    assert profile_identity_file(path, ("personId", "id")) == {
        "rows": 3,
        "id_column": "personId",
        "key_columns": ["personId"],
        "missing_id": 1,
        "duplicate_id": 1,
        "duplicate_key": 1,
    }


def test_file_inventory_marks_optional_sources_as_missing(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "Games.csv",
        ["gameId"],
        [{"gameId": "g1"}],
    )

    inventory = file_inventory(tmp_path)

    assert inventory["Games.csv"]["available"] is True
    assert inventory["Games.csv"]["role"] == "core_fact"
    assert inventory["Players.csv"]["available"] is False
    assert inventory["PlaybyPlay.parquet"]["role"] == "deferred_event_fact"


def test_team_history_allows_multiple_names_per_team_but_requires_history_key(tmp_path: Path) -> None:
    path = tmp_path / "TeamHistories.csv"
    write_csv(
        path,
        ["teamId", "seasonFounded", "seasonActiveTill"],
        [
            {"teamId": "t1", "seasonFounded": "1946", "seasonActiveTill": "1950"},
            {"teamId": "t1", "seasonFounded": "1951", "seasonActiveTill": "2100"},
        ],
    )

    result = profile_identity_file(
        path,
        ("teamId",),
        (("teamId", "seasonFounded", "seasonActiveTill"),),
    )

    assert result["duplicate_id"] == 1
    assert result["duplicate_key"] == 0
