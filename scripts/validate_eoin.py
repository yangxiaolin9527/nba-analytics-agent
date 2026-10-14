"""Report game-level coverage and source inconsistencies for Eoin v515."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


RAW = Path(__file__).resolve().parents[1] / "data/raw/eoinamoore/v515"
CORE_METRICS = (
    "games",
    "games_missing_score",
    "team_rows",
    "games_without_two_matching_team_rows",
    "games_without_player_rows",
    "player_rows",
    "player_rows_missing_team",
    "team_rows_type_mismatch",
)


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def validate(raw=RAW):
    games = {}
    periods = defaultdict(Counter)
    team_ids_by_game = defaultdict(Counter)
    player_games = Counter()
    totals = Counter()

    for row in rows(raw / "Games.csv"):
        game_id = row["gameId"]
        period = (row["gameDate"][:4], row["gameType"])
        if game_id in games:
            totals["duplicate_game_ids"] += 1
            continue
        games[game_id] = (period, row["hometeamId"], row["awayteamId"])
        periods[period]["games"] += 1
        if not row["homeScore"] or not row["awayScore"]:
            periods[period]["games_missing_score"] += 1

    for row in rows(raw / "TeamStatistics.csv"):
        game_id = row["gameId"]
        game = games.get(game_id)
        if game is None:
            totals["team_rows_without_game"] += 1
            continue
        period, home, away = game
        periods[period]["team_rows"] += 1
        team_ids_by_game[game_id][row["teamId"]] += 1
        if row["teamId"] not in (home, away):
            periods[period]["team_rows_unknown_team"] += 1
        if row["gameType"] != period[1]:
            periods[period]["team_rows_type_mismatch"] += 1

    for row in rows(raw / "PlayerStatistics.csv"):
        game_id = row["gameId"]
        game = games.get(game_id)
        if game is None:
            totals["player_rows_without_game"] += 1
            continue
        period, home, away = game
        periods[period]["player_rows"] += 1
        player_games[game_id] += 1
        if not row["playerteamId"]:
            periods[period]["player_rows_missing_team"] += 1
        elif row["playerteamId"] not in (home, away):
            periods[period]["player_rows_unknown_team"] += 1
        if row["gameType"] != period[1]:
            periods[period]["player_rows_type_mismatch"] += 1
        if row["comment"] or not row["numMinutes"]:
            periods[period]["player_rows_without_confirmed_minutes"] += 1

    for game_id, (period, home, away) in games.items():
        teams = team_ids_by_game[game_id]
        if teams != Counter((home, away)):
            periods[period]["games_without_two_matching_team_rows"] += 1
        if player_games[game_id] == 0:
            periods[period]["games_without_player_rows"] += 1

    return {
        "dataset": "eoinamoore/historical-nba-data-and-player-box-scores",
        "version": 515,
        "scope": "calendar year of gameDate and Games.csv gameType; not an NBA season mapping",
        "totals": dict(totals),
        "periods": {
            f"{year}|{game_type}": {
                **{metric: counts[metric] for metric in CORE_METRICS},
                **dict(counts),
            }
            for (year, game_type), counts in sorted(periods.items())
        },
    }


if __name__ == "__main__":
    try:
        json.dump(validate(), sys.stdout, indent=2, ensure_ascii=False)
        print()
    except (OSError, KeyError, csv.Error) as exc:
        sys.exit(f"Validation failed: {exc}")
