"""Create a deterministic, observation-only RAG failure attribution report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.knowledge_attribution import attribute_case
from scripts.knowledge_search import load_nodes
from scripts.score_revised_m24 import json_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--score", type=Path, help="optional bound revised score for semantic labels")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    score = json.loads(args.score.read_text(encoding="utf-8")) if args.score else None
    if score and (score.get("source_report_sha256") != json_hash(report)
                  or score.get("label_case_sha256") != json_hash(cases)):
        raise ValueError("score does not match report and labels")
    by_id = {case["id"]: case for case in cases}
    rows = report.get("cases", [])
    if len(rows) != len(cases) or {row.get("id") for row in rows} != set(by_id):
        raise ValueError("report cases do not match labels")
    for row in rows:
        if row.get("question") != by_id[row["id"]]["question"]:
            raise ValueError(f"{row['id']}: report question mismatch")
    nodes = {node.metadata["citation_id"]: node for node in load_nodes(args.index)}
    index_hash = hashlib.sha256((args.index / "nodes.json").read_bytes()).hexdigest()
    if report.get("manifest", {}).get("index_nodes_sha256") not in (None, index_hash):
        raise ValueError("index hash mismatch")
    attributed = []
    for row in rows:
        item = score.get("cases", {}).get(row["id"]) if score else None
        semantic = item.get("semantic_supported") if item else None
        unsupported = bool(item and item.get("classification") == "unsupported_claim")
        attributed.append({"id": row["id"], "question": row.get("question"),
                           **attribute_case(by_id[row["id"]], row, nodes,
                                            reviewed_semantic=semantic, unsupported=unsupported)})
    counts = {}
    for row in attributed:
        counts[row["primary"]] = counts.get(row["primary"], 0) + 1
    output = {
        "schema_version": 1, "kind": "rag_failure_attribution", "source_report": str(args.report),
        "source_report_sha256": json_hash(report), "label_case_sha256": json_hash(cases),
        "index_nodes_sha256": index_hash, "semantic_labels": str(args.score) if args.score else None,
        "observation_only": True, "counts": counts, "cases": attributed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}; cases={len(attributed)}")


if __name__ == "__main__":
    main()
