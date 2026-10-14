"""Adapt the historical M2.4 agent review for revised diagnostic scoring."""
from __future__ import annotations

import json
from pathlib import Path


def adapt(source: Path, target: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    cases = {}
    for case_id, review in payload.get("cases", {}).items():
        cases[case_id] = {
            "facts_present": [review.get("facts_complete") is True],
            "claim_support": review.get("claim_support", []),
            "review_origin": "historical_agent_review",
        }
    target.write_text(json.dumps({
        "reviewer": payload.get("reviewer", "unknown"),
        "reviewer_type": payload.get("reviewer_type", "agent"),
        "review_origin": "compatibility_adapter",
        "human_review_confirmed": False,
        "cases": cases,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    adapt(args.source, args.target)
