import pytest

from scripts.season_comparison import (
    ComparisonError,
    Observation,
    calculate_comparison,
    metric_definition,
)


def observation(value, *, samples=10, quality="validated", limitations=()):
    return Observation(
        value=value,
        sample_count=samples,
        quality_status=quality,
        limitations=tuple(limitations),
    )


def test_calculates_absolute_and_relative_change_from_source_values() -> None:
    result = calculate_comparison(
        metric_definition("player", "points_total"),
        observation(20),
        observation(25, samples=12),
    )

    assert result.status == "complete"
    assert result.baseline_value == 20
    assert result.comparison_value == 25
    assert result.absolute_delta == 5
    assert result.relative_change_pct == 25.0
    assert result.relative_change_status == "computable"
    assert result.absolute_delta_formula == "comparison_value - baseline_value"
    assert result.relative_change_formula == "(comparison_value - baseline_value) / baseline_value * 100"


def test_zero_baseline_keeps_delta_and_marks_relative_change_uncomputable() -> None:
    result = calculate_comparison(
        metric_definition("player", "points_total"),
        observation(0),
        observation(25),
    )

    assert result.status == "partial"
    assert result.absolute_delta == 25
    assert result.relative_change_pct is None
    assert result.relative_change_status == "not_computable"
    assert result.limitation == "zero_baseline"


def test_conditional_quality_is_propagated_to_comparison() -> None:
    result = calculate_comparison(
        metric_definition("team", "points_total"),
        observation(100, quality="conditional", limitations=("season_completeness_not_checked",)),
        observation(110),
    )

    assert result.quality_status == "conditional"
    assert result.limitations == ("season_completeness_not_checked",)


def test_missing_source_value_is_not_treated_as_zero() -> None:
    with pytest.raises(ComparisonError, match="missing baseline value"):
        calculate_comparison(
            metric_definition("player", "points_total"),
            observation(None),
            observation(25),
        )


def test_metric_definition_rejects_wrong_subject() -> None:
    with pytest.raises(ComparisonError, match="not supported for subject team"):
        metric_definition("team", "points_per_appearance")


def test_unsupported_quality_is_rejected_before_calculation() -> None:
    with pytest.raises(ComparisonError, match="unsupported source data"):
        calculate_comparison(
            metric_definition("player", "points_total"),
            observation(20, quality="unsupported"),
            observation(25),
        )


def test_unverified_quality_is_preserved() -> None:
    result = calculate_comparison(
        metric_definition("player", "points_total"),
        observation(20, quality="unverified", limitations=("capability_not_profiled",)),
        observation(25),
    )
    assert result.quality_status == "unverified"
    assert result.limitations == ("capability_not_profiled",)
