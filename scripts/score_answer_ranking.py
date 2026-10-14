"""Score a paired baseline/candidate answer-ranking report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_answer_ranking import score_pair_report
from scripts.knowledge_search import load_nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    result = score_pair_report(cases, report, reviews, nodes)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}; promotion_ready={result['promotion_ready']}")


if __name__ == "__main__":
    main()
