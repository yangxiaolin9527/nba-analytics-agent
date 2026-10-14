"""Build the reviewed M2.4 contract without mutating the frozen manifest."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "metadata/knowledge_answer_cases_m24_v1.json"
TARGET = ROOT / "metadata/knowledge_answer_cases_m24_revised_v1.json"


def build():
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["version"] = 1
    payload["split"] = "m24_revised_diagnostic"
    payload["label_policy"] = (
        "Revised question-scoped contract. expected_facts describe only the requested "
        "answer scope. acceptable_evidence lists equivalent source locations; exact "
        "supporting_citation_ids are diagnostic and are not required for semantic support."
    )
    for case in payload["cases"]:
        outcome = case["expected_outcome"]
        if outcome == "refusal":
            continue
        case["question_scope"] = {
            "M24-001": "player only; do not require coach, trainer, or bench-person scope",
            "M24-007": "ejection and suspension after the supplied no-contact punch premise",
        }.get(case["id"], "all facts required by the question")
        if case["id"] == "M24-001":
            case["expected_facts"] = [
                "A player may be ejected for one unsportsmanlike act and must be ejected for two."
            ]
            case["fact_evidence"][0]["fact"] = case["expected_facts"][0]
        if case["id"] == "M24-007":
            case["expected_facts"] = [
                "After replay confirmation, the player is ejected and suspended for at least one game."
            ]
            case["fact_evidence"] = [dict(case["fact_evidence"][0], fact=case["expected_facts"][0])]
        acceptable = []
        for evidence in case["fact_evidence"]:
            acceptable.append({
                "fact": evidence["fact"],
                "citation_ids": [evidence["citation_id"]],
                "locations": [{
                    "source_id": evidence["source_id"], "page": evidence["page"],
                    "anchor": evidence["anchor"],
                }],
                "quote": evidence["quote"],
            })
        if case["id"] == "M24-003":
            acceptable[0]["citation_ids"].append("nba-rulebook-2025-26:814d25f36d6fec0c")
            acceptable[0]["locations"].append({"source_id": "nba-rulebook-2025-26", "page": 57, "anchor": None})
        if case["id"] == "M24-007":
            acceptable[0]["citation_ids"].append("nba-rulebook-2025-26:585e27e0fb907939")
            acceptable[0]["locations"].append({"source_id": "nba-rulebook-2025-26", "page": 49, "anchor": None})
        case["acceptable_evidence"] = acceptable
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
