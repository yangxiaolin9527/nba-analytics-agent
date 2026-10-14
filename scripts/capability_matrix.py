"""Build a conservative, reproducible capability report from M1 profiles."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.game_semantics import ROOT, classify_game
from scripts.player_quality import parse_count


PROFILE = ROOT / "data/profile"
REFERENCES = ROOT / "metadata/official_reference_games_v1.json"
GAMES = ROOT / "data/raw/eoinamoore/v515/Games.csv"
OUTPUT = PROFILE / "eoin-v515-capability-matrix.json"
METRIC_COLUMNS = {
    "points": "points",
    "rebounds": "rebounds_total",
    "assists": "assists",
    "efg_percentage": "efg_inputs",
}


def load_reference_catalog(path: Path = REFERENCES) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    seen_ids = set()
    seen_evidence = set()
    for sample in catalog["samples"]:
        required = {"game_id", "evidence_id", "verification_status", "source_url", "official_url"}
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"reference sample missing fields: {sorted(missing)}")
        game_id = sample["game_id"]
        evidence_id = sample["evidence_id"]
        if game_id in seen_ids or evidence_id in seen_evidence:
            raise ValueError(f"duplicate reference game or evidence ID: {game_id}, {evidence_id}")
        seen_ids.add(game_id)
        seen_evidence.add(evidence_id)
        if sample["verification_status"] not in ("independent_score_match", "pending_external_check"):
            raise ValueError(f"unknown reference status: {evidence_id}")
        if not sample["source_url"].startswith("https://") or not sample["official_url"].startswith("https://"):
            raise ValueError(f"reference URL missing: {evidence_id}")
    return catalog


def verify_reference_snapshot(catalog: dict, games_path: Path = GAMES) -> dict:
    by_id = {sample["game_id"]: sample for sample in catalog["samples"]}
    found = {}
    with games_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample = by_id.get(row["gameId"])
            if sample is None:
                continue
            if row["gameId"] in found:
                raise ValueError(f"duplicate source game ID: {row['gameId']}")
            mapped = classify_game(row)
            expected = sample["expected"]
            found[row["gameId"]] = {
                "evidence_id": sample["evidence_id"],
                "match": mapped["season_id"] == sample["season_id"]
                and mapped["competition_phase"] == sample["phase"]
                and row["homeScore"] == str(expected["home_score"])
                and row["awayScore"] == str(expected["away_score"]),
                "verification_status": sample["verification_status"],
            }
    for game_id in by_id:
        if game_id not in found:
            found[game_id] = {"evidence_id": by_id[game_id]["evidence_id"], "match": False,
                              "verification_status": "missing_source_game"}
    return dict(sorted(found.items()))


def verify_player_stat_samples(catalog: dict, player_path: Path) -> dict:
    samples = {(item["game_id"], item["person_id"]): item
               for item in catalog.get("player_stat_samples", [])}
    found = {}
    columns = {"points": "points", "rebounds_total": "reboundsTotal", "assists": "assists",
               "field_goals_made": "fieldGoalsMade", "three_pointers_made": "threePointersMade",
               "field_goals_attempted": "fieldGoalsAttempted"}
    with player_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["gameId"], row["personId"]
            if key not in samples:
                continue
            if key in found:
                raise ValueError(f"duplicate player reference row: {key}")
            sample = samples[key]
            found[key] = all(parse_count(row[columns[name]]) == ("valid", value)
                             for name, value in sample["expected"].items())
    return {item["evidence_id"]: found.get((item["game_id"], item["person_id"]), False)
            for item in catalog.get("player_stat_samples", [])}


def verify_team_stat_samples(catalog: dict, team_path: Path) -> dict:
    samples = {(item["game_id"], item["team_id"]): item
               for item in catalog.get("team_stat_samples", [])}
    found = {}
    columns = {"points": "teamScore", "rebounds_total": "reboundsTotal", "assists": "assists"}
    with team_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["gameId"], row["teamId"]
            if key not in samples:
                continue
            if key in found:
                raise ValueError(f"duplicate team reference row: {key}")
            found[key] = all(parse_count(row[columns[name]]) == ("valid", value)
                             for name, value in samples[key]["expected"].items())
    return {item["evidence_id"]: found.get((item["game_id"], item["team_id"]), False)
            for item in catalog.get("team_stat_samples", [])}


def scan_team_coverage(raw: Path) -> dict:
    games = {}
    groups = {}
    with (raw / "Games.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            mapped = classify_game(row)
            if mapped["season_status"] != "valid" or mapped["competition_phase"] == "unknown":
                continue
            key = f"{mapped['season_id']}|{mapped['competition_phase']}"
            games[row["gameId"]] = (key, row)
            group = groups.setdefault(key, {"games": 0, "games_with_two_matching_rows": 0,
                                             "metrics": {name: Counter() for name in METRIC_COLUMNS}})
            group["games"] += 1

    matched = {}
    with (raw / "TeamStatistics.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            game = games.get(row["gameId"])
            if not game:
                continue
            key, source = game
            home, away = source["hometeamId"], source["awayteamId"]
            team = row["teamId"]
            expected_opponent = away if team == home else home if team == away else None
            expected_score = source["homeScore"] if team == home else source["awayScore"]
            expected_opponent_score = source["awayScore"] if team == home else source["homeScore"]
            valid = (expected_opponent is not None and row["opponentTeamId"] == expected_opponent
                     and row["teamScore"] == expected_score
                     and row["opponentScore"] == expected_opponent_score)
            seen = matched.setdefault(row["gameId"], [])
            seen.append((team, valid))
            if not valid:
                continue
            counts = groups[key]["metrics"]
            counts["points"]["row_valid"] += 1
            for name, column in (("rebounds", "reboundsTotal"), ("assists", "assists")):
                status, value = parse_count(row[column])
                if status == "valid":
                    counts[name]["row_valid"] += 1
                    if value > 0:
                        counts[name]["nonzero"] += 1
            fgm = parse_count(row["fieldGoalsMade"])[1]
            three = parse_count(row["threePointersMade"])[1]
            fga = parse_count(row["fieldGoalsAttempted"])[1]
            if None not in (fgm, three, fga) and three <= fgm <= fga:
                counts["efg_percentage"]["row_valid"] += 1
    for game_id, rows in matched.items():
        key, source = games[game_id]
        if len(rows) == 2 and {team for team, valid in rows if valid} == {
                source["hometeamId"], source["awayteamId"]}:
            groups[key]["games_with_two_matching_rows"] += 1
    return {"by_season_phase": {key: {**group, "metrics": {name: dict(counts)
            for name, counts in group["metrics"].items()}} for key, group in sorted(groups.items())}}


def capability_status(eligible_rows: int, valid_rows: int, *, evidence: bool) -> str:
    if eligible_rows == 0 or valid_rows == 0:
        return "unsupported"
    if valid_rows == eligible_rows and evidence:
        return "validated"
    return "conditional"


def build_capability_matrix(semantics: dict, quality: dict, catalog: dict, team_quality: dict | None = None) -> dict:
    verified = Counter()
    tags = {}
    for sample in catalog["samples"]:
        if sample.get("verification_status") == "independent_score_match":
            key = f"{sample['season_id']}|{sample['phase']}"
            verified[key] += 1
            tags.setdefault(key, set()).update(sample.get("evidence_tags", []))
    team_reference_periods = set()
    sample_by_game = {sample["game_id"]: sample for sample in catalog["samples"]}
    for sample in catalog.get("team_stat_samples", []):
        game = sample_by_game.get(sample["game_id"])
        if game and sample.get("verification_status") == "independent_boxscore_match":
            team_reference_periods.add(f"{game['season_id']}|{game['phase']}")

    periods = {}
    for season, phases in sorted(semantics["by_season_phase"].items()):
        for phase, game_count in sorted(phases.items()):
            key = f"{season}|{phase}"
            group = quality["by_season_phase"].get(key, {})
            appearances = group.get("appearance_statuses", {})
            eligible = appearances.get("played", 0) + appearances.get("played_inferred", 0)
            context = group.get("team_context", {})
            known_expected = (catalog.get("regular_season_game_count_references", {})
                              .get(season, {}).get("expected_games") if phase == "regular_season" else None)
            completeness = "not_checked" if known_expected is None else (
                "matches_reference_count" if game_count == known_expected else "count_mismatch"
            )
            reference = verified[key] > 0
            player_reference = any(sample["game_id"] in {item["game_id"] for item in catalog["samples"]
                                    if item["season_id"] == season and item["phase"] == phase}
                                   and sample.get("verification_status") == "independent_score_and_boxscore_match"
                                   for sample in catalog.get("player_stat_samples", []))
            basic_evidence = reference and player_reference and completeness == "matches_reference_count"
            metrics = {}
            metric_details = {}
            for name, column in METRIC_COLUMNS.items():
                counts = group.get("metrics", {}).get(column, {})
                valid = counts.get("row_valid", 0)
                if name == "efg_percentage":
                    evidence = basic_evidence and {"shooting_inputs", "historical_shooting"} <= tags.get(key, set())
                else:
                    evidence = basic_evidence
                status = capability_status(eligible, valid, evidence=evidence)
                nonzero = counts.get("nonzero", 0)
                if valid and nonzero == 0 and name in {"rebounds", "assists"}:
                    status = "unsupported"
                metrics[name] = status
                metric_details[name] = {"eligible_rows": eligible, "valid_rows": valid,
                                        "invalid_or_missing_rows": max(eligible - valid, 0)}
            team_status = capability_status(group.get("rows", 0), context.get("valid", 0),
                                            evidence=basic_evidence)
            team_group = (team_quality or {}).get("by_season_phase", {}).get(key, {})
            expected_team_rows = game_count * 2
            team_evidence = reference and completeness == "matches_reference_count"
            team_metrics = {}
            for name in METRIC_COLUMNS:
                counts = team_group.get("metrics", {}).get(name, {})
                valid = counts.get("row_valid", 0)
                metric_evidence = team_evidence and (name == "points" or key in team_reference_periods)
                evidence = metric_evidence and (name != "efg_percentage" or
                            {"shooting_inputs", "historical_shooting"} <= tags.get(key, set()))
                team_metrics[name] = capability_status(expected_team_rows, valid, evidence=evidence)
                if name in {"rebounds", "assists"} and valid and counts.get("nonzero") == 0:
                    team_metrics[name] = "unsupported"
            participation = capability_status(game_count,
                team_group.get("games_with_two_matching_rows", 0), evidence=team_evidence)
            reasons = []
            if completeness == "count_mismatch":
                reasons.append("game_count_mismatch")
            elif completeness == "not_checked":
                reasons.append("season_completeness_not_checked")
            if not reference:
                reasons.append("no_independent_score_sample")
            if context.get("valid", 0) != group.get("rows", 0):
                reasons.append("team_context_incomplete")
            if "historical_shooting" not in tags.get(key, set()):
                reasons.append("historical_shooting_unverified")
            for name, column in (("rebounds", "rebounds_total"), ("assists", "assists")):
                counts = group.get("metrics", {}).get(column, {})
                if counts.get("row_valid", 0) and counts.get("nonzero", 0) == 0:
                    reasons.append(f"zero_only_{name}")
            periods[key] = {
                "retained": True,
                "games": game_count,
                "player_rows": group.get("rows", 0),
                "eligible_player_rows": eligible,
                "reference_score_samples": verified[key],
                "game_count_reference": known_expected,
                "game_count_status": completeness,
                "subjects": {
                    "player": {"metrics": metrics, "metric_details": metric_details,
                               "team_opponent_filter": team_status},
                    "team": {"metrics": team_metrics, "participation": participation,
                             "games_with_two_matching_rows": team_group.get("games_with_two_matching_rows", 0)},
                },
                "limitations": reasons,
            }

    return {
        "matrix_version": "1.0.0",
        "source_snapshot": semantics.get("source_snapshot"),
        "reference_catalog": "metadata/official_reference_games_v1.json",
        "policy": {
            "validated": "All eligible rows pass the metric gate and period-level reference evidence exists.",
            "conditional": "Queryable only with the recorded coverage or evidence limitation disclosed.",
            "unsupported": "Reject the affected metric or filter, not the entire season.",
            "scope": "Period-level screening; entity-specific checks remain mandatory before answering.",
        },
        "periods": periods,
    }


def main() -> None:
    semantics = json.loads((PROFILE / "eoin-v515-game-semantics.json").read_text(encoding="utf-8"))
    quality = json.loads((PROFILE / "eoin-v515-player-quality.json").read_text(encoding="utf-8"))
    catalog = load_reference_catalog()
    checks = verify_reference_snapshot(catalog)
    if not all(item["match"] for item in checks.values()):
        raise ValueError(f"reference catalog does not match source snapshot: {checks}")
    player_checks = verify_player_stat_samples(catalog, ROOT / "data/raw/eoinamoore/v515/PlayerStatistics.csv")
    if not all(player_checks.values()):
        raise ValueError(f"player reference catalog does not match source snapshot: {player_checks}")
    team_checks = verify_team_stat_samples(catalog, ROOT / "data/raw/eoinamoore/v515/TeamStatistics.csv")
    if not all(team_checks.values()):
        raise ValueError(f"team reference catalog does not match source snapshot: {team_checks}")
    team_quality = scan_team_coverage(ROOT / "data/raw/eoinamoore/v515")
    matrix = build_capability_matrix(semantics, quality, catalog, team_quality)
    matrix["reference_snapshot_checks"] = checks
    matrix["player_reference_snapshot_checks"] = player_checks
    matrix["team_reference_snapshot_checks"] = team_checks
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(matrix['periods'])} season-phase records to {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, csv.Error) as exc:
        sys.exit(f"Capability matrix failed: {exc}")
