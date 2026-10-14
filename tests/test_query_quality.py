from __future__ import annotations

import json
import sqlite3

from scripts.query_quality import QualityScope, assess_scope, infer_sql_scope


def test_assess_scope_preserves_period_limitations(tmp_path) -> None:
    matrix = {
        "source_snapshot": "test@snapshot",
        "periods": {
            "2023-24|playoffs": {
                "subjects": {
                    "player": {
                        "metrics": {"points": "conditional"},
                        "team_opponent_filter": "conditional",
                    }
                },
                "limitations": ["season_completeness_not_checked"],
            }
        },
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")

    assessment = assess_scope(
        QualityScope(
            season_id="2023-24",
            competition_phase="playoffs",
            subject="player",
            metrics=("points",),
        ),
        matrix_path=path,
        source_snapshot="test@snapshot",
    )

    assert assessment.status == "conditional"
    assert assessment.limitations == (
        "season_completeness_not_checked",
        "capability_evidence_conditional",
    )


def test_assess_scope_marks_unknown_period_unverified(tmp_path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"source_snapshot": "test@snapshot", "periods": {}}), encoding="utf-8")

    assessment = assess_scope(
        QualityScope(
            season_id="2023-24",
            competition_phase="regular_season",
            subject="player",
            metrics=("points",),
        ),
        matrix_path=path,
        source_snapshot="test@snapshot",
    )

    assert assessment.status == "unverified"
    assert assessment.limitations == ("capability_not_profiled",)


def test_assess_scope_rejects_unsupported_metric(tmp_path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(
        json.dumps({
            "source_snapshot": "test@snapshot",
            "periods": {
                "2023-24|regular_season": {
                    "subjects": {"player": {"metrics": {"points": "unsupported"}}},
                    "limitations": [],
                }
            },
        }),
        encoding="utf-8",
    )

    assessment = assess_scope(
        QualityScope(
            season_id="2023-24",
            competition_phase="regular_season",
            subject="player",
            metrics=("points",),
        ),
        matrix_path=path,
        source_snapshot="test@snapshot",
    )

    assert assessment.status == "unsupported"
    assert assessment.limitations == ("unsupported:points",)


def test_assess_scope_names_all_unsupported_filters(tmp_path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({
        "source_snapshot": "test@snapshot",
        "periods": {"2023-24|regular_season": {
            "subjects": {"team": {
                "metrics": {"points": "validated"},
                "team_opponent_filter": "unsupported",
                "participation": "unsupported",
            }},
            "limitations": [],
        }},
    }), encoding="utf-8")

    assessment = assess_scope(
        QualityScope(
            season_id="2023-24", competition_phase="regular_season", subject="team",
            metrics=("points",), opponent_filter=True, team_participation=True,
        ),
        matrix_path=path,
        source_snapshot="test@snapshot",
    )

    assert assessment.status == "unsupported"
    assert assessment.limitations == (
        "unsupported:team_opponent_filter",
        "unsupported:team_participation",
    )


def test_infer_sql_scope_reads_actual_season_phase_and_opponent_filter() -> None:
    inferred = infer_sql_scope(
        """SELECT COUNT(*) AS games_appeared, SUM(p.points) AS points_total,
                  SUM(p.points) * 1.0 / COUNT(*) AS points_per_appearance
           FROM player_game_stat AS p JOIN game AS g ON g.game_id = p.game_id
           WHERE p.person_id = '201939' AND p.opponent_team_id = '1610612747'
             AND g.season_id = '2023-24' AND g.competition_phase = 'regular_season'
             AND p.appearance_status IN ('played', 'played_inferred')"""
    )

    assert inferred.scope is not None
    assert inferred.scope.season_id == "2023-24"
    assert inferred.scope.competition_phase == "regular_season"
    assert inferred.scope.subject == "player"
    assert inferred.scope.metrics == ("points",)
    assert inferred.scope.opponent_filter is True


def test_infer_sql_scope_does_not_trust_missing_scope_filters() -> None:
    inferred = infer_sql_scope("SELECT points_total FROM player_season_stat")

    assert inferred.scope is None
    assert inferred.reason == "sql_scope_unverified"
