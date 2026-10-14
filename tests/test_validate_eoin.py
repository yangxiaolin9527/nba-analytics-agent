import csv
from pathlib import Path

from scripts.validate_eoin import validate


def write_rows(path: Path, fieldnames: list[str], values: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def test_validates_game_coverage_and_source_type_labels(tmp_path: Path) -> None:
    write_rows(
        tmp_path / "Games.csv",
        ["gameId", "gameDate", "gameType", "hometeamId", "awayteamId", "homeScore", "awayScore"],
        [
            {"gameId": "g1", "gameDate": "2024-01-01", "gameType": "NBA Emirates Cup", "hometeamId": "a", "awayteamId": "b", "homeScore": "100", "awayScore": "90"},
            {"gameId": "g2", "gameDate": "2024-05-01", "gameType": "Playoffs", "hometeamId": "a", "awayteamId": "b", "homeScore": "", "awayScore": ""},
        ],
    )
    write_rows(
        tmp_path / "TeamStatistics.csv",
        ["gameId", "teamId", "gameType"],
        [
            {"gameId": "g1", "teamId": "a", "gameType": "Regular Season"},
            {"gameId": "g1", "teamId": "b", "gameType": "Regular Season"},
            {"gameId": "g2", "teamId": "a", "gameType": "Playoffs"},
            {"gameId": "g2", "teamId": "a", "gameType": "Playoffs"},
        ],
    )
    write_rows(
        tmp_path / "PlayerStatistics.csv",
        ["gameId", "playerteamId", "gameType", "comment", "numMinutes"],
        [
            {"gameId": "g1", "playerteamId": "a", "gameType": "NBA Emirates Cup", "comment": "", "numMinutes": "20"},
            {"gameId": "g1", "playerteamId": "", "gameType": "NBA Emirates Cup", "comment": "DNP", "numMinutes": ""},
            {"gameId": "orphan", "playerteamId": "a", "gameType": "Playoffs", "comment": "", "numMinutes": "10"},
        ],
    )

    result = validate(tmp_path)
    cup = result["periods"]["2024|NBA Emirates Cup"]
    playoffs = result["periods"]["2024|Playoffs"]
    assert cup["games"] == 1
    assert cup["team_rows_type_mismatch"] == 2
    assert cup["player_rows_missing_team"] == 1
    assert cup["games_without_two_matching_team_rows"] == 0
    assert playoffs["games_without_two_matching_team_rows"] == 1
    assert playoffs["games_without_player_rows"] == 1
    assert playoffs["games_missing_score"] == 1
    assert result["totals"]["player_rows_without_game"] == 1
