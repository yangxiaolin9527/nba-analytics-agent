import csv
from pathlib import Path

from scripts.player_quality import classify_player, parse_minutes, scan_player_coverage


def player(minutes: str = "", comment: str = "", **stats: str) -> dict[str, str]:
    return {"gameId": "22400001", "personId": "p1", "playerteamId": "a",
            "opponentteamId": "b", "numMinutes": minutes, "comment": comment,
            "points": "0", "assists": "0", "reboundsTotal": "0",
            "fieldGoalsMade": "0", "threePointersMade": "0", "fieldGoalsAttempted": "0",
            **stats}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_minutes_support_decimal_clock_and_invalid_values() -> None:
    assert parse_minutes("12.5")[0] == "positive"
    assert parse_minutes("12:30")[0] == "positive"
    assert parse_minutes("0")[0] == "zero"
    assert parse_minutes("")[0] == "missing"
    assert parse_minutes("22:60")[0] == "invalid"
    assert parse_minutes("-1")[0] == "invalid"


def test_appearance_evidence_priority() -> None:
    assert classify_player(player("20"))["appearance_status"] == "played"
    inferred = classify_player(player("0", points="5"))
    assert inferred["appearance_status"] == "played_inferred"
    assert inferred["minutes_status"] == "zero"
    assert classify_player(player("", points="2"))["appearance_status"] == "played_inferred"
    assert classify_player(player("", "DNP - Coach's Decision"))["appearance_status"] == "not_played"
    assert classify_player(player("", "DND - Injury/Illness"))["appearance_status"] == "not_played"
    assert classify_player(player("", "NWT - Personal"))["appearance_status"] == "not_played"
    assert classify_player(player("0"))["appearance_status"] == "unknown"
    assert classify_player(player("10", "DNP - Coach's Decision"))["appearance_status"] == "conflict"
    assert classify_player(player("", "DNP - Coach's Decision", points="1"))["appearance_status"] == "conflict"
    assert classify_player(player("", "other note"))["appearance_status"] == "unknown"


def test_metric_status_flags_missing_and_impossible_shooting() -> None:
    result = classify_player(player("10", fieldGoalsMade="4", fieldGoalsAttempted="3", threePointersMade="5", assists=""))
    assert result["metric_statuses"]["assists"] == "missing"
    assert result["metric_statuses"]["field_goals_made"] == "inconsistent"
    assert result["metric_statuses"]["three_pointers_made"] == "inconsistent"
    assert result["metric_statuses"]["field_goals_attempted"] == "inconsistent"
    assert result["metric_statuses"]["efg_inputs"] == "inconsistent"


def test_coverage_separates_inferred_and_unlinked_rows(tmp_path: Path) -> None:
    games = [{"gameId": "22400001", "gameDate": "2024-12-01", "gameType": "Regular Season",
              "hometeamId": "a", "awayteamId": "b"}]
    players = [player("0", points="5"), player("", "DNP - Coach's Decision", personId="p2"),
               player("20", personId="p3", assists=""),
               player("10", personId="p4", gameId="missing")]
    write_rows(tmp_path / "Games.csv", games)
    write_rows(tmp_path / "PlayerStatistics.csv", players)

    result = scan_player_coverage(tmp_path)
    group = result["by_season_phase"]["2024-25|regular_season"]
    assert result["rows"] == 4
    assert result["unlinked_game_rows"] == 1
    assert group["appearance_statuses"] == {"not_played": 1, "played": 1, "played_inferred": 1}
    assert group["metrics"]["points"]["row_valid"] == 2
    assert group["metrics"]["points"]["nonzero"] == 1
    assert group["metrics"]["assists"]["missing"] == 1
    assert group["metrics"]["efg_inputs"]["row_valid"] == 2
