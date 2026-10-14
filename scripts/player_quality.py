"""Classify player appearance evidence and report metric coverage in Eoin v515."""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

from scripts.game_semantics import ROOT, classify_game


RAW = ROOT / "data/raw/eoinamoore/v515"
RULES = ROOT / "metadata/player_quality_v1.json"


@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES.read_text(encoding="utf-8"))


def parse_minutes(value: str) -> tuple[str, Decimal | None]:
    value = value.strip()
    if not value:
        return "missing", None
    if re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]", value):
        minute, second = map(int, value.split(":"))
        amount = Decimal(minute) + Decimal(second) / Decimal(60)
    else:
        try:
            amount = Decimal(value)
        except InvalidOperation:
            return "invalid", None
    if not amount.is_finite() or amount < 0:
        return "invalid", None
    return ("positive" if amount > 0 else "zero"), amount


def parse_count(value: str) -> tuple[str, Decimal | None]:
    if not value.strip():
        return "missing", None
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return "invalid", None
    if not amount.is_finite() or amount < 0:
        return "invalid", None
    return "valid", amount


def classify_player(row: dict[str, str]) -> dict:
    config = rules()
    minutes_status, _ = parse_minutes(row.get("numMinutes", ""))
    comment = row.get("comment", "").strip()
    upper_comment = comment.upper().lstrip('"\' ')
    absence = any(upper_comment.startswith(prefix) and
                  (len(upper_comment) == len(prefix) or upper_comment[len(prefix)] in " -_(:")
                  for prefix in config["absence_prefixes"])
    comment_status = "absence" if absence else "other" if comment else "none"

    metrics = {}
    values = {}
    for name, column in config["metric_columns"].items():
        status, value = parse_count(row.get(column, ""))
        metrics[name], values[name] = status, value

    if values["field_goals_made"] is not None and values["field_goals_attempted"] is not None:
        if values["field_goals_made"] > values["field_goals_attempted"]:
            metrics["field_goals_made"] = metrics["field_goals_attempted"] = "inconsistent"
    if values["three_pointers_made"] is not None and values["field_goals_made"] is not None:
        if values["three_pointers_made"] > values["field_goals_made"]:
            metrics["three_pointers_made"] = metrics["field_goals_made"] = "inconsistent"
    efg_statuses = [metrics[name] for name in
                    ("field_goals_made", "three_pointers_made", "field_goals_attempted")]
    metrics["efg_inputs"] = next((status for status in ("inconsistent", "invalid", "missing")
                                  if status in efg_statuses), "valid")

    activity = any(value is not None and value > 0 for value in values.values())
    for column in config["additional_activity_columns"]:
        _, value = parse_count(row.get(column, ""))
        activity |= value is not None and value > 0

    if absence:
        appearance = "conflict" if minutes_status == "positive" or activity else "not_played"
    elif minutes_status == "positive":
        appearance = "played"
    elif activity and not comment:
        appearance = "played_inferred"
    elif activity:
        appearance = "conflict"
    else:
        appearance = "unknown"

    return {
        "appearance_status": appearance,
        "minutes_status": minutes_status,
        "comment_status": comment_status,
        "activity_signal": activity,
        "metric_statuses": metrics,
        "nonzero_metrics": [name for name, value in values.items()
                            if value is not None and value > 0 and metrics[name] == "valid"],
    }


def scan_player_coverage(raw: Path = RAW) -> dict:
    games = {}
    with (raw / "Games.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            game_id = row["gameId"]
            if game_id in games:
                raise ValueError(f"Duplicate game ID: {game_id}")
            games[game_id] = (classify_game(row), row["hometeamId"], row["awayteamId"])

    totals = Counter()
    groups = defaultdict(lambda: {"appearance_statuses": Counter(), "minutes_statuses": Counter(),
                                  "comment_statuses": Counter(), "team_context": Counter(),
                                  "metrics": defaultdict(Counter)})
    with (raw / "PlayerStatistics.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            totals["rows"] += 1
            result = classify_player(row)
            totals[f"appearance_{result['appearance_status']}"] += 1
            if result["minutes_status"] == "invalid":
                totals["invalid_minutes_rows"] += 1
            game = games.get(row["gameId"])
            if game is None:
                totals["unlinked_game_rows"] += 1
                continue
            mapped, home, away = game
            if mapped["season_status"] != "valid" or mapped["competition_phase"] == "unknown":
                totals["unresolved_game_semantics_rows"] += 1
                continue
            key = f"{mapped['season_id']}|{mapped['competition_phase']}"
            group = groups[key]
            group["appearance_statuses"][result["appearance_status"]] += 1
            group["minutes_statuses"][result["minutes_status"]] += 1
            group["comment_statuses"][result["comment_status"]] += 1
            team, opponent = row.get("playerteamId", ""), row.get("opponentteamId", "")
            if not team:
                group["team_context"]["missing_team"] += 1
            elif team not in (home, away):
                group["team_context"]["unknown_team"] += 1
            elif opponent != (away if team == home else home):
                group["team_context"]["opponent_missing_or_mismatch"] += 1
            else:
                group["team_context"]["valid"] += 1
            if result["appearance_status"] in ("played", "played_inferred"):
                for metric, status in result["metric_statuses"].items():
                    group["metrics"][metric]["row_valid" if status == "valid" else status] += 1
                for metric in result["nonzero_metrics"]:
                    group["metrics"][metric]["nonzero"] += 1

    return {
        "source_snapshot": rules()["source_snapshot"],
        "rule_version": rules()["version"],
        "rows": totals["rows"],
        "unlinked_game_rows": totals["unlinked_game_rows"],
        "unresolved_game_semantics_rows": totals["unresolved_game_semantics_rows"],
        "invalid_minutes_rows": totals["invalid_minutes_rows"],
        "appearance_statuses": {status: totals[f"appearance_{status}"] for status in
                                ("played", "played_inferred", "not_played", "unknown", "conflict")},
        "by_season_phase": {
            key: {
                "rows": sum(value["appearance_statuses"].values()),
                "appearance_statuses": dict(sorted(value["appearance_statuses"].items())),
                "minutes_statuses": dict(sorted(value["minutes_statuses"].items())),
                "comment_statuses": dict(sorted(value["comment_statuses"].items())),
                "team_context": dict(sorted(value["team_context"].items())),
                "metrics": {metric: dict(sorted(counts.items())) for metric, counts in sorted(value["metrics"].items())},
            }
            for key, value in sorted(groups.items())
        },
    }


if __name__ == "__main__":
    try:
        json.dump(scan_player_coverage(), sys.stdout, indent=2, ensure_ascii=False)
        print()
    except (OSError, KeyError, csv.Error, ValueError) as exc:
        sys.exit(f"Player quality scan failed: {exc}")
