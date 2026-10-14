"""Shared exact-result, evidence, and quality scoring for query evaluations."""

from __future__ import annotations

import math
import hashlib
import subprocess
from typing import Any, Mapping
from pathlib import Path


def _equal(actual: Any, expected: Any) -> bool:
    if actual is None:
        return expected is None
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            return False
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def score_plan_response(
    body: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    expected_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a successful exact-oracle response without accepting extra rows."""
    columns = body.get("columns")
    rows = body.get("rows")
    expected = case.get("expected")
    quality_expected = case.get("expected_quality")
    shape_reason = None
    if not isinstance(expected, dict) or not expected:
        shape_reason = "missing_exact_expectation"
    elif not isinstance(columns, list) or len(columns) != len(set(columns)):
        shape_reason = "duplicate_columns"
    elif not isinstance(rows, list) or len(rows) != 1:
        shape_reason = "result_shape_mismatch"
    elif len(columns) != len(rows[0]):
        shape_reason = "result_shape_mismatch"
    elif set(columns) != set(expected):
        shape_reason = "projection_mismatch"

    if shape_reason is None:
        actual = dict(zip(columns, rows[0]))
        result_match = all(_equal(actual[key], value) for key, value in expected.items())
    elif shape_reason == "projection_mismatch":
        actual = {}
        result_match = None
    else:
        actual = {}
        result_match = False

    evidence_match = body.get("evidence") == dict(expected_evidence)
    quality_match: bool | None = None
    if isinstance(quality_expected, dict):
        quality_match = (
            body.get("quality_status") == quality_expected.get("status")
            and set(body.get("limitations", [])) == set(quality_expected.get("limitations", []))
        )
    if shape_reason == "missing_exact_expectation":
        outcome, reason = "unscorable", shape_reason
    elif shape_reason == "projection_mismatch":
        outcome, reason = "unscorable", shape_reason
    elif shape_reason:
        outcome, reason = "failed", shape_reason
    elif not result_match:
        outcome, reason = "failed", "result_mismatch"
    elif not evidence_match:
        outcome, reason = "failed", "evidence_mismatch"
    elif quality_match is False:
        outcome, reason = "failed", "quality_mismatch"
    elif body.get("status") != "ok" or body.get("result_status") != "complete":
        outcome, reason = "failed", "response_not_complete"
    else:
        outcome, reason = "passed", None

    return {
        "result_match": result_match,
        "evidence_match": evidence_match,
        "quality_match": quality_match,
        "outcome": outcome,
        "reason": reason,
        "actual": actual,
    }


def validate_case_document(
    document: Mapping[str, Any],
    *,
    source_snapshot: str,
    allow_missing_expected: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    if document.get("source_snapshot") != source_snapshot:
        raise ValueError("case set source snapshot does not match catalog")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("case set must contain at least one case; no scorable cases")
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"invalid or duplicate case ID: {case_id}")
        seen.add(case_id)
        if "split" in case and case["split"] not in {"development", "holdout"}:
            raise ValueError(f"invalid split for {case_id}")
        expected = case.get("expected")
        if (not isinstance(expected, dict) or not expected) and case_id not in allow_missing_expected:
            raise ValueError(f"missing exact expectation: {case_id}")
    return cases


def validate_quality_expectations(document: Mapping[str, Any], *, source_snapshot: str) -> dict[str, dict[str, Any]]:
    if document.get("source_snapshot") != source_snapshot:
        raise ValueError("quality expectations source snapshot does not match catalog")
    expectations = document.get("expectations")
    if not isinstance(expectations, dict) or not expectations:
        raise ValueError("quality expectations must be a non-empty mapping")
    for case_id, expected in expectations.items():
        if not isinstance(case_id, str) or not case_id or not isinstance(expected, dict):
            raise ValueError("invalid quality expectation entry")
        if expected.get("status") not in {"validated", "conditional", "unverified", "unsupported"}:
            raise ValueError(f"invalid quality status for {case_id}")
        if not isinstance(expected.get("limitations"), list):
            raise ValueError(f"invalid quality limitations for {case_id}")
    return expectations


def report_provenance(
    *,
    root: Path,
    source_snapshot: str,
    model_id: str,
    route: str,
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        for name, path in input_paths.items()
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "source_snapshot": source_snapshot,
        "model_id": model_id,
        "route": route,
        "git_commit": commit,
        "git_dirty": dirty,
        "input_sha256": hashes,
    }
