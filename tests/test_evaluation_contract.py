from __future__ import annotations

import pytest

from scripts.evaluation_contract import (
    score_plan_response,
    validate_case_document,
    validate_quality_expectations,
)


def case() -> dict:
    return {
        "case_id": "CASE-001",
        "split": "development",
        "expected": {"points_total": 344, "games_appeared": 12},
        "expected_quality": {
            "status": "conditional",
            "limitations": ["season_completeness_not_checked"],
        },
    }


def response(**overrides):
    body = {
        "status": "ok",
        "result_status": "complete",
        "columns": ["points_total", "games_appeared"],
        "rows": [[344, 12]],
        "evidence": {"source_name": "source"},
        "quality_status": "conditional",
        "limitations": ["season_completeness_not_checked"],
    }
    body.update(overrides)
    return body


def test_plan_score_rejects_extra_rows_even_when_first_row_matches() -> None:
    record = score_plan_response(
        response(rows=[[344, 12], [999, 999]]),
        case(),
        expected_evidence={"source_name": "source"},
    )

    assert record["result_match"] is False
    assert record["quality_match"] is True
    assert record["outcome"] == "failed"
    assert record["reason"] == "result_shape_mismatch"


def test_plan_score_requires_quality_disclosure() -> None:
    record = score_plan_response(
        response(quality_status="available", limitations=[]),
        case(),
        expected_evidence={"source_name": "source"},
    )

    assert record["result_match"] is True
    assert record["quality_match"] is False
    assert record["outcome"] == "failed"
    assert record["reason"] == "quality_mismatch"


def test_plan_score_rejects_duplicate_columns() -> None:
    record = score_plan_response(
        response(columns=["points_total", "points_total"]),
        case(),
        expected_evidence={"source_name": "source"},
    )

    assert record["outcome"] == "failed"
    assert record["reason"] == "duplicate_columns"


def test_plan_score_fails_wrong_value_and_evidence() -> None:
    wrong_value = score_plan_response(
        response(rows=[[345, 12]]), case(), expected_evidence={"source_name": "source"}
    )
    wrong_evidence = score_plan_response(
        response(), case(), expected_evidence={"source_name": "other"}
    )

    assert wrong_value["reason"] == "result_mismatch"
    assert wrong_evidence["reason"] == "evidence_mismatch"


def test_plan_score_marks_unknown_projection_unscorable() -> None:
    record = score_plan_response(
        response(columns=["total", "games"], rows=[[344, 12]]),
        case(), expected_evidence={"source_name": "source"},
    )

    assert record["result_match"] is None
    assert record["outcome"] == "unscorable"
    assert record["reason"] == "projection_mismatch"


def test_case_document_rejects_empty_duplicate_and_wrong_snapshot() -> None:
    valid_case = {
        "case_id": "CASE-001", "split": "development",
        "expected": {"points_total": 344},
    }
    with pytest.raises(ValueError, match="at least one case"):
        validate_case_document({"source_snapshot": "snapshot", "cases": []}, source_snapshot="snapshot")
    with pytest.raises(ValueError, match="duplicate case ID"):
        validate_case_document(
            {"source_snapshot": "snapshot", "cases": [valid_case, valid_case]},
            source_snapshot="snapshot",
        )
    with pytest.raises(ValueError, match="does not match"):
        validate_case_document(
            {"source_snapshot": "other", "cases": [valid_case]}, source_snapshot="snapshot"
        )


def test_quality_expectations_require_the_matching_snapshot() -> None:
    document = {"source_snapshot": "other", "expectations": {
        "CASE-001": {"status": "validated", "limitations": []},
    }}

    with pytest.raises(ValueError, match="does not match"):
        validate_quality_expectations(document, source_snapshot="snapshot")
