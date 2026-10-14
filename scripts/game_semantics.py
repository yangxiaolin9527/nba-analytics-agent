"""Classify Eoin v515 games without changing source records."""

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "metadata/game_semantics_v1.json"
GAMES = ROOT / "data/raw/eoinamoore/v515/Games.csv"


@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES.read_text(encoding="utf-8"))


def classify_game(row: dict[str, str]) -> dict[str, str | None]:
    config = rules()
    game_id = row.get("gameId", "")
    raw_type = row.get("gameType", "")
    season_id = None
    season_status = "unknown_id"
    parsed_date = None
    try:
        parsed_date = date.fromisoformat(row.get("gameDate", "")[:10])
    except ValueError:
        pass

    if len(game_id) == 8 and game_id.isascii() and game_id.isdigit():
        suffix = int(game_id[1:3])
        candidates = [year for year in (1900 + suffix, 2000 + suffix)
                      if config["season_start_min"] <= year <= config["season_start_max"]]
        if len(candidates) == 1:
            year = candidates[0]
            season_id = f"{year}-{(year + 1) % 100:02d}"
            season_status = "unknown_date" if parsed_date is None else (
                "valid" if parsed_date.year in (year, year + 1) else "date_conflict"
            )

    prefix_phase = config["phase_by_id_prefix"].get(game_id[:1])
    type_phase = config["phase_by_type"].get(raw_type)
    if game_id in config["cup_championship_ids"]:
        phase = "cup_championship"
        phase_status = "valid" if raw_type in config["cup_championship_types"] else "source_type_conflict"
    elif prefix_phase and type_phase == prefix_phase:
        phase, phase_status = prefix_phase, "valid"
    elif prefix_phase == "cup_championship" or (prefix_phase and type_phase and prefix_phase != type_phase):
        phase, phase_status = "unknown", "prefix_type_conflict"
    elif type_phase:
        phase, phase_status = type_phase, "unknown_id_prefix"
    else:
        phase, phase_status = "unknown", "unknown_type"

    return {
        "game_id": game_id,
        "season_id": season_id,
        "season_status": season_status,
        "competition_phase": phase,
        "phase_status": phase_status,
        "raw_game_type": raw_type,
        "raw_game_label": row.get("gameLabel", ""),
        "raw_game_sub_label": row.get("gameSubLabel", ""),
    }


def scan_games(path: Path = GAMES) -> dict:
    phases = Counter()
    season_statuses = Counter()
    phase_statuses = Counter()
    source_types = Counter()
    by_season_phase = defaultdict(Counter)
    exceptions = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result = classify_game(row)
            phases[result["competition_phase"]] += 1
            season_statuses[result["season_status"]] += 1
            phase_statuses[result["phase_status"]] += 1
            source_types[result["raw_game_type"]] += 1
            by_season_phase[result["season_id"] or "unknown"][result["competition_phase"]] += 1
            if result["season_status"] != "valid" or result["phase_status"] != "valid":
                exceptions.append(result)
    return {
        "source_snapshot": rules()["source_snapshot"],
        "rule_version": rules()["version"],
        "rows": sum(phases.values()),
        "phases": dict(sorted(phases.items())),
        "season_statuses": dict(sorted(season_statuses.items())),
        "phase_statuses": dict(sorted(phase_statuses.items())),
        "source_types": dict(sorted(source_types.items())),
        "by_season_phase": {key: dict(sorted(value.items())) for key, value in sorted(by_season_phase.items())},
        "exceptions": exceptions,
    }


if __name__ == "__main__":
    try:
        json.dump(scan_games(), sys.stdout, indent=2, ensure_ascii=False)
        print()
    except (OSError, KeyError, csv.Error) as exc:
        sys.exit(f"Game semantics scan failed: {exc}")
