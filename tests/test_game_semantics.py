import csv
from pathlib import Path

from scripts.game_semantics import classify_game, scan_games


def game(game_id: str, date: str, game_type: str, label: str = "", sublabel: str = "") -> dict[str, str]:
    return {"gameId": game_id, "gameDate": date, "gameType": game_type,
            "gameLabel": label, "gameSubLabel": sublabel}


def test_season_code_crosses_centuries_and_validates_date() -> None:
    for game_id, date, season in [
        ("24600001", "1947-03-01", "1946-47"),
        ("29900001", "2000-03-01", "1999-00"),
        ("20000001", "2000-11-01", "2000-01"),
        ("22500001", "2026-01-01", "2025-26"),
    ]:
        result = classify_game(game(game_id, date, "Regular Season"))
        assert result["season_id"] == season
        assert result["season_status"] == "valid"

    result = classify_game(game("22400001", "2027-01-01", "Regular Season"))
    assert result["season_id"] == "2024-25"
    assert result["season_status"] == "date_conflict"
    assert classify_game(game("bad", "2024-01-01", "Regular Season"))["season_status"] == "unknown_id"


def test_phase_mapping_and_cup_championship_exception() -> None:
    cases = [
        ("22400001", "Regular Season", "regular_season"),
        ("22301230", "NBA Emirates Cup", "regular_season"),
        ("42500001", "Playoffs", "playoffs"),
        ("52500001", "Play-in Tournament", "play_in"),
        ("12500001", "Preseason", "preseason"),
        ("32500001", "All-Star Game", "all_star"),
        ("62300001", "NBA Cup", "cup_championship"),
        ("62500001", "Emirates NBA Cup", "cup_championship"),
    ]
    for game_id, raw_type, expected in cases:
        assert classify_game(game(game_id, "2025-01-01", raw_type))["competition_phase"] == expected

    exception = classify_game(game("62400001", "2024-12-17", "Regular Season", "Emirates NBA Cup", "Championship"))
    assert exception["competition_phase"] == "cup_championship"
    assert exception["phase_status"] == "source_type_conflict"
    assert exception["raw_game_type"] == "Regular Season"

    unexpected = classify_game(game("62500002", "2025-12-17", "Regular Season"))
    assert unexpected["competition_phase"] == "unknown"
    assert unexpected["phase_status"] == "prefix_type_conflict"


def test_scan_reports_counts_and_problem_ids(tmp_path: Path) -> None:
    rows = [game("22301230", "2023-12-07", "NBA Emirates Cup"),
            game("62400001", "2024-12-17", "Regular Season", "Emirates NBA Cup", "Championship"),
            game("22400001", "2027-01-01", "Regular Season")]
    with (tmp_path / "Games.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    result = scan_games(tmp_path / "Games.csv")
    assert result["rows"] == 3
    assert result["phases"] == {"cup_championship": 1, "regular_season": 2}
    assert result["season_statuses"]["date_conflict"] == 1
    assert result["phase_statuses"]["source_type_conflict"] == 1
    assert {x["game_id"] for x in result["exceptions"]} == {"62400001", "22400001"}
