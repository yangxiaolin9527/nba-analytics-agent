"""Profile the pinned Eoin Moore raw CSV files without changing them."""

import csv
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/eoinamoore/v515"
EXPECTED_FILES = {
    "Games.csv": "core_fact",
    "PlayerStatistics.csv": "core_fact",
    "TeamStatistics.csv": "core_fact",
    "Players.csv": "identity_dimension",
    "TeamHistories.csv": "identity_dimension",
    "PlayerStatisticsExtended.csv": "deferred_advanced_fact",
    "TeamStatisticsExtended.csv": "deferred_advanced_fact",
    "PlaybyPlay.parquet": "deferred_event_fact",
    "LeagueSchedule24_25.csv": "deferred_schedule",
    "LeagueSchedule25_26.csv": "deferred_schedule",
}


def records(name):
    with (RAW / name).open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(raw=RAW):
    inventory = {}
    for name, role in EXPECTED_FILES.items():
        path = raw / name
        entry = {"role": role, "available": path.exists()}
        if path.exists():
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = sha256(path)
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    entry["columns"] = next(csv.reader(handle), [])
        inventory[name] = entry
    return inventory


def profile_identity_file(
    path: Path,
    id_candidates: tuple[str, ...],
    key_candidates: tuple[tuple[str, ...], ...] = (),
) -> dict:
    rows = 0
    missing_id = 0
    duplicate_id = 0
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        id_column = next((name for name in id_candidates if name in (reader.fieldnames or [])), None)
        key_columns = next(
            (columns for columns in key_candidates if all(column in (reader.fieldnames or []) for column in columns)),
            (id_column,) if id_column else (),
        )
        if id_column is None:
            return {
                "rows": 0,
                "id_column": None,
                "key_columns": list(key_columns),
                "missing_id": None,
                "duplicate_id": None,
                "duplicate_key": None,
                "columns": reader.fieldnames or [],
            }
        seen_keys = set()
        duplicate_key = 0
        for row in reader:
            rows += 1
            value = row.get(id_column, "")
            if not value:
                missing_id += 1
            elif value in seen:
                duplicate_id += 1
            seen.add(value)
            key = tuple(row.get(column, "") for column in key_columns)
            if key in seen_keys:
                duplicate_key += 1
            seen_keys.add(key)
    return {
        "rows": rows,
        "id_column": id_column,
        "key_columns": list(key_columns),
        "missing_id": missing_id,
        "duplicate_id": duplicate_id,
        "duplicate_key": duplicate_key,
    }


def profile():
    games = {}
    game_types = Counter()
    games_by_year_type = Counter()
    duplicate_games = 0
    for row in records("Games.csv"):
        key = row["gameId"]
        if key in games:
            duplicate_games += 1
        games[key] = (row["hometeamId"], row["awayteamId"], row["gameType"])
        game_types[row["gameType"]] += 1
        games_by_year_type[(row["gameDate"][:4], row["gameType"])] += 1

    result = {
        "dataset": "eoinamoore/historical-nba-data-and-player-box-scores",
        "version": 515,
        "files": file_inventory(),
        "games": {
            "rows": sum(game_types.values()),
            "duplicate_game_ids": duplicate_games,
            "game_types": dict(game_types),
            "calendar_year_by_type": {
                f"{year}|{game_type}": count
                for (year, game_type), count in sorted(games_by_year_type.items())
            },
        },
    }

    for name, entity_id, team_id in (
        ("PlayerStatistics.csv", "personId", "playerteamId"),
        ("TeamStatistics.csv", "teamId", "teamId"),
    ):
        seen = set()
        counts = Counter()
        by_type = Counter()
        issues_by_year_type = Counter()
        comments = Counter()
        for row in records(name):
            counts["rows"] += 1
            game_id = row["gameId"]
            key = (game_id, row[entity_id])
            if key in seen:
                counts["duplicate_business_keys"] += 1
            seen.add(key)
            game = games.get(game_id)
            if game is None:
                counts["missing_game_reference"] += 1
                issues_by_year_type[(row["gameDate"][:4], row["gameType"], "missing_game_reference")] += 1
            else:
                if row[team_id] not in game[:2]:
                    counts["team_not_in_game"] += 1
                    issues_by_year_type[(row["gameDate"][:4], row["gameType"], "team_not_in_game")] += 1
                if row["gameType"] != game[2]:
                    counts["game_type_mismatch"] += 1
                    issues_by_year_type[(row["gameDate"][:4], row["gameType"], "game_type_mismatch")] += 1
            by_type[row["gameType"]] += 1
            if not row[team_id]:
                counts["empty_team_id"] += 1
            if name == "PlayerStatistics.csv":
                if row["comment"]:
                    comments[row["comment"]] += 1
                if not row["numMinutes"]:
                    counts["empty_minutes"] += 1
            for field in ("fieldGoalsMade", "threePointersMade", "fieldGoalsAttempted"):
                if not row[field]:
                    counts[f"missing_{field}"] += 1
            try:
                fgm = Decimal(row["fieldGoalsMade"])
                tpm = Decimal(row["threePointersMade"])
                fga = Decimal(row["fieldGoalsAttempted"])
                if any(value != value.to_integral_value() for value in (fgm, tpm, fga)):
                    counts["fractional_shooting_counts"] += 1
                if not 0 <= tpm <= fgm <= fga:
                    counts["invalid_shooting_counts"] += 1
                    issues_by_year_type[(row["gameDate"][:4], row["gameType"], "invalid_shooting_counts")] += 1
            except InvalidOperation:
                counts["nonnumeric_shooting_counts"] += 1
        result[name] = {
            **counts,
            "game_types": dict(by_type),
            "issues_by_calendar_year_and_type": {
                f"{year}|{game_type}|{issue}": count
                for (year, game_type, issue), count in sorted(issues_by_year_type.items())
            },
            "top_comments": comments.most_common(12),
        }
    for name, candidates in (
        ("Players.csv", ("personId", "playerId", "id")),
        ("TeamHistories.csv", ("teamId", "id")),
    ):
        path = RAW / name
        if path.exists():
            key_candidates = (
                (("teamId", "seasonFounded", "seasonActiveTill"),)
                if name == "TeamHistories.csv"
                else ()
            )
            result[name] = profile_identity_file(path, candidates, key_candidates)
    return result


if __name__ == "__main__":
    try:
        json.dump(profile(), sys.stdout, ensure_ascii=False, indent=2)
        print()
    except (OSError, KeyError, csv.Error) as exc:
        sys.exit(f"Profiling failed: {exc}")
