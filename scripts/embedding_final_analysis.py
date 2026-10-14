"""Auditable paired analysis for the final embedding selection experiment."""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from statistics import mean
from typing import Iterable


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def paired_bootstrap(ali: list[float], bge: list[float], *, seed: int = 24,
                     resamples: int = 10000) -> dict:
    if len(ali) != len(bge) or not ali:
        raise ValueError("paired samples must be non-empty and equal length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = [float(a) - float(b) for a, b in zip(ali, bge)]
    observed = _mean(differences)
    rng = random.Random(seed)
    samples = sorted(_mean(rng.choices(differences, k=len(differences))) for _ in range(resamples))
    low = samples[max(0, int(0.025 * resamples) - 1)]
    high = samples[min(resamples - 1, int(0.975 * resamples))]
    return {"difference": observed, "ci95": [low, high], "resamples": resamples, "seed": seed}


def mcnemar_exact(ali: list[bool], bge: list[bool]) -> dict:
    if len(ali) != len(bge) or not ali:
        raise ValueError("paired samples must be non-empty and equal length")
    ali_only = sum(bool(a) and not bool(b) for a, b in zip(ali, bge))
    bge_only = sum(bool(b) and not bool(a) for a, b in zip(ali, bge))
    discordant = ali_only + bge_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(ali_only, bge_only) + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {"ali_only": ali_only, "bge_only": bge_only,
            "discordant": discordant, "p_value": p_value}


def latency_summary(latencies_ms: list[float], timeouts: list[bool]) -> dict:
    if len(latencies_ms) != len(timeouts) or not latencies_ms:
        raise ValueError("latencies and timeout labels must be non-empty and equal length")
    values = sorted(float(value) for value in latencies_ms)

    def percentile(rank: float) -> float:
        if rank == .5 and len(values) % 2 == 0:
            middle = len(values) // 2
            return (values[middle - 1] + values[middle]) / 2
        # Nearest-rank avoids interpolation ambiguity for tail percentiles.
        return values[max(0, math.ceil(rank * len(values)) - 1)]

    return {"count": len(values), "mean_ms": _mean(values), "p50_ms": percentile(.5),
            "p95_ms": percentile(.95), "timeout_count": sum(bool(x) for x in timeouts),
            "timeout_rate": _mean(bool(x) for x in timeouts)}


def summarize_answer_reviews(rows: list[dict]) -> dict[str, dict]:
    required = {"case_id", "candidate", "answerable", "semantic_support", "claim_complete",
                "correct_abstention", "runtime_failure"}
    if not rows or any(not required <= set(row) for row in rows):
        raise ValueError("answer review row is missing required fields")
    output = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        answerable = [row for row in selected if row["answerable"]
                      or row.get("expected_outcome") == "premise_denial"]
        refusals = [row for row in selected if row not in answerable]
        citations = [row.get("citation_integrity_auto", row.get("citation_integrity")) for row in selected]
        labels = citations + [row[key] for row in answerable for key in ("semantic_support", "claim_complete")]
        labels += [row["correct_abstention"] for row in refusals]
        if any(type(label) is not bool for label in labels):
            raise ValueError("answer review requires explicit boolean labels; overall confirmation is not a numeric label")
        output[candidate] = {
            "case_count": len(selected),
            "answerable_count": len(answerable),
            "semantic_support_rate": _mean(row["semantic_support"] for row in answerable),
            "claim_completeness_rate": _mean(row["claim_complete"] for row in answerable),
            "citation_integrity_rate": _mean(citations),
            "correct_abstention_rate": _mean(row["correct_abstention"] for row in refusals) if refusals else None,
            "false_acceptance_rate": (_mean(row["status"] == "answered" for row in refusals)
                                      if refusals and all("status" in row for row in refusals) else None),
            "runtime_failure_rate": _mean(row["runtime_failure"] for row in selected),
        }
    return output


def promotion_decision(report: dict) -> dict:
    quality = report["quality"]
    latency = report["latency"]
    readiness = report["readiness"]
    failed = []
    if quality is None:
        failed.append("numeric_quality_labels")
    else:
        if quality["semantic_support_difference"] < -0.05:
            failed.append("semantic_support")
        if quality["completeness_difference"] < -0.05:
            failed.append("claim_completeness")
        for key, name in (("citation_integrity_difference", "citation_integrity"),
                          ("correct_abstention_difference", "correct_abstention")):
            if quality[key] < 0:
                failed.append(name)
        if quality["false_acceptance_difference"] is None:
            failed.append("false_acceptance_labels")
        elif quality["false_acceptance_difference"] > 0:
            failed.append("false_acceptance")
        if quality["runtime_failure_difference"] > 0:
            failed.append("runtime_failure")
    if latency["p50_relative_improvement"] < 0.15:
        failed.append("endpoint_p50")
    if latency["p95_relative_change"] > 0.10:
        failed.append("endpoint_p95")
    if latency["timeout_rate_difference"] > 0:
        failed.append("timeout_rate")
    if not readiness["passed"]:
        failed.append("readiness")
    return {"promote_bge": not failed, "failed_conditions": failed,
            "decision": "promote_bge" if not failed else "retain_historical_default"}


def stable_hash(payload: object) -> str:
    import json
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
