import csv
import json
from pathlib import Path

import pytest

from scripts.capability_matrix import (build_capability_matrix, capability_status,
                                       load_reference_catalog, scan_team_coverage)


def test_evaluation_baseline_binds_every_taxonomy_case() -> None:
    baseline = json.loads(Path("metadata/agent_evaluation_baseline_v1.json").read_text(encoding="utf-8"))
    assert {case["taxonomy_id"] for case in baseline["cases"]} == {
        *(f"QT-{index:03d}" for index in range(1, 19))
    }
    assert len({case["case_id"] for case in baseline["cases"]}) == len(baseline["cases"])
    assert all(case["split"] in {"evaluation", "development"} for case in baseline["cases"])


def quality_group(*, rows=10, played=10, team_valid=10, metrics=None):
    return {
        "rows": rows,
        "appearance_statuses": {"played": played, "played_inferred": 0},
        "team_context": {"valid": team_valid},
        "metrics": metrics or {
            "points": {"row_valid": played},
            "rebounds_total": {"row_valid": played},
            "assists": {"row_valid": played},
            "efg_inputs": {"row_valid": played},
        },
    }


def test_reference_catalog_has_traceable_score_samples() -> None:
    catalog = load_reference_catalog()
    assert catalog["catalog_version"]
    assert len(catalog["samples"]) >= 10
    for sample in catalog["samples"]:
        assert sample["game_id"].isdigit()
        assert sample["season_id"] in {"2023-24", "2024-25"}
        assert sample["source_url"].startswith("https://")
        assert sample["evidence_id"]
        assert sample["expected"]["home_score"] >= 0
        assert sample["expected"]["away_score"] >= 0


def test_metric_gates_distinguish_validated_conditional_and_unsupported() -> None:
    assert capability_status(100, 100, evidence=True) == "validated"
    assert capability_status(100, 97, evidence=True) == "conditional"
    assert capability_status(0, 0, evidence=True) == "unsupported"
    assert capability_status(100, 100, evidence=False) == "conditional"


def test_all_zero_historical_metric_is_not_treated_as_valid_zero() -> None:
    semantics = {"by_season_phase": {"1946-47": {"regular_season": 18}}}
    quality = {"by_season_phase": {"1946-47|regular_season": quality_group(
        rows=100, played=100, team_valid=100,
        metrics={"points": {"row_valid": 100, "nonzero": 95},
                 "rebounds_total": {"row_valid": 100},
                 "assists": {"row_valid": 100},
                 "efg_inputs": {"row_valid": 0}})}}
    period = build_capability_matrix(semantics, quality, {"samples": []})["periods"]["1946-47|regular_season"]
    assert period["subjects"]["player"]["metrics"]["points"] == "conditional"
    assert period["subjects"]["player"]["metrics"]["rebounds"] == "unsupported"
    assert "zero_only_rebounds" in period["limitations"]
    team_quality = {"by_season_phase": {"1946-47|regular_season": {
        "games_with_two_matching_rows": 18,
        "metrics": {"rebounds": {"row_valid": 36, "nonzero": 0}, "assists": {"row_valid": 36, "nonzero": 0}}}}}
    team_period = build_capability_matrix(semantics, quality, {"samples": []}, team_quality)["periods"]["1946-47|regular_season"]
    assert team_period["subjects"]["team"]["metrics"]["rebounds"] == "unsupported"


def test_matrix_retains_full_history_and_discloses_metric_limits() -> None:
    semantics = {
        "by_season_phase": {
            "1946-47": {"regular_season": 18},
            "2023-24": {"regular_season": 1230},
        },
        "phase_statuses": {"valid": 1248},
    }
    quality = {
        "by_season_phase": {
            "1946-47|regular_season": quality_group(
                played=10,
                metrics={
                    "points": {"row_valid": 10},
                        "rebounds_total": {"row_valid": 10, "nonzero": 8},
                        "assists": {"row_valid": 8, "nonzero": 5},
                    "efg_inputs": {"row_valid": 2, "inconsistent": 8},
                },
            ),
            "2023-24|regular_season": quality_group(),
        }
    }
    catalog = {"regular_season_game_count_references": {"2023-24": {"expected_games": 1230}},
               "player_stat_samples": [{"game_id": "22300061",
                                        "verification_status": "independent_score_and_boxscore_match"}],
               "samples": [{"game_id": "22300061", "season_id": "2023-24", "phase": "regular_season",
                           "verification_status": "independent_score_match",
                           "evidence_tags": ["shooting_inputs", "historical_shooting"]}]}
    matrix = build_capability_matrix(semantics, quality, catalog)
    early = matrix["periods"]["1946-47|regular_season"]
    recent = matrix["periods"]["2023-24|regular_season"]
    assert early["retained"] is True
    assert early["subjects"]["player"]["metrics"]["points"] == "conditional"
    assert early["subjects"]["player"]["metrics"]["assists"] == "conditional"
    assert early["subjects"]["player"]["metrics"]["efg_percentage"] == "conditional"
    assert recent["subjects"]["player"]["metrics"]["efg_percentage"] == "validated"


def test_matrix_marks_team_filter_conditional_when_context_is_incomplete() -> None:
    semantics = {"by_season_phase": {"2024-25": {"regular_season": 1225}}, "phase_statuses": {"valid": 1}}
    quality = {"by_season_phase": {"2024-25|regular_season": quality_group(team_valid=9)}}
    matrix = build_capability_matrix(semantics, quality, {"samples": []})
    assert matrix["periods"]["2024-25|regular_season"]["subjects"]["player"]["team_opponent_filter"] == "conditional"


def test_team_coverage_requires_two_matching_participations(tmp_path: Path) -> None:
    games = [{"gameId": "22400001", "gameDate": "2024-10-22", "gameType": "Regular Season",
              "hometeamId": "a", "awayteamId": "b", "homeScore": "100", "awayScore": "90"},
             {"gameId": "22400002", "gameDate": "2024-10-23", "gameType": "Regular Season",
              "hometeamId": "a", "awayteamId": "c", "homeScore": "95", "awayScore": "80"}]
    teams = [{"gameId": "22400001", "teamId": "a", "opponentTeamId": "b", "teamScore": "100",
              "opponentScore": "90", "assists": "20", "reboundsTotal": "42", "fieldGoalsMade": "40",
              "threePointersMade": "10", "fieldGoalsAttempted": "80"},
             {"gameId": "22400001", "teamId": "b", "opponentTeamId": "a", "teamScore": "90",
              "opponentScore": "100", "assists": "18", "reboundsTotal": "40", "fieldGoalsMade": "35",
              "threePointersMade": "8", "fieldGoalsAttempted": "79"}]
    for name, rows in [("Games.csv", games), ("TeamStatistics.csv", teams)]:
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    result = scan_team_coverage(tmp_path)
    period = result["by_season_phase"]["2024-25|regular_season"]
    assert period["games"] == 2
    assert period["games_with_two_matching_rows"] == 1
    assert period["metrics"]["points"]["row_valid"] == 2


def test_team_assists_need_metric_reference_not_only_score() -> None:
    semantics = {"by_season_phase": {"2023-24": {"regular_season": 1230}}}
    quality = {"by_season_phase": {"2023-24|regular_season": quality_group()}}
    teams = {"by_season_phase": {"2023-24|regular_season": {
        "games_with_two_matching_rows": 1230,
        "metrics": {"points": {"row_valid": 2460}, "assists": {"row_valid": 2460},
                    "rebounds": {"row_valid": 2460}}}}}
    reference = {"regular_season_game_count_references": {"2023-24": {"expected_games": 1230}},
                 "samples": [{"game_id": "22300061", "season_id": "2023-24",
                              "phase": "regular_season", "verification_status": "independent_score_match"}]}
    period = build_capability_matrix(semantics, quality, reference, teams)["periods"]["2023-24|regular_season"]
    assert period["subjects"]["team"]["metrics"]["points"] == "validated"
    assert period["subjects"]["team"]["metrics"]["assists"] == "conditional"


def test_reference_catalog_rejects_duplicate_game_ids(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    sample = {"game_id": "1", "evidence_id": "REF-1", "source_url": "https://example.com",
              "official_url": "https://example.com", "verification_status": "pending_external_check"}
    path.write_text(json.dumps({"catalog_version": "x", "samples": [sample, sample]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_reference_catalog(path)
