"""Controlled generation-only benchmark; retrieval is deliberately excluded."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from scripts.knowledge_answer import AnswerDraft, generation_prompt
from scripts.model_adapter import ModelRequest, OpenAICompatibleModel
from scripts.provider_config import load_provider_config

ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "metadata/generation_path_cases_v1.json"
NODE_PATH = ROOT / "data/indexes/knowledge-500-dense/nodes.json"

ARMS = {
    "baseline_thinking_nonstream": ("baseline", "enabled", False),
    "baseline_no_thinking_nonstream": ("baseline", "disabled", False),
    "baseline_thinking_stream": ("baseline", "enabled", True),
    "baseline_no_thinking_stream": ("baseline", "disabled", True),
    "concise_thinking_nonstream": ("concise", "enabled", False),
    "concise_no_thinking_nonstream": ("concise", "disabled", False),
}


def load_cases() -> list[dict]:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))["cases"]


def load_nodes() -> dict[str, dict]:
    return {item["metadata"]["citation_id"]: item for item in
            json.loads(NODE_PATH.read_text(encoding="utf-8"))["nodes"]}


def fixture_evidence(case: dict, nodes: dict[str, dict]) -> list[dict]:
    result = []
    for citation_id in case["evidence_citation_ids"]:
        node = nodes[citation_id]
        meta = node["metadata"]
        result.append({"citation_id": citation_id, "excerpt": node["text"],
                       "source_id": meta["id"], "source_url": meta["source_url"],
                       "source_version": meta["version"], "sha256": meta["sha256"],
                       "page": meta["page"], "anchor": meta.get("anchor")})
    return result


def assess(response, case: dict, evidence: list[dict]) -> dict:
    try:
        draft = AnswerDraft.model_validate(json.loads(response.text))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"json_valid": False, "claim_present": False, "citation_valid": False,
                "completeness": "unassessed", "error": type(exc).__name__}
    expected = set(case["expected_citation_ids"])
    cited = {part.citation_id for part in draft.parts if part.claim and part.citation_id}
    valid_ids = {hit["citation_id"] for hit in evidence}
    return {"json_valid": True, "claim_present": bool(cited),
            "citation_valid": cited <= valid_ids and bool(cited & expected),
            "completeness": "manual_review_required",
            "cited_ids": sorted(cited), "part_count": len(draft.parts)}


def run_case(model, case: dict, nodes: dict[str, dict], arm: str, repeat: int) -> list[dict]:
    strategy, thinking, stream = ARMS[arm]
    evidence = fixture_evidence(case, nodes)
    prompt = generation_prompt(case["question"], evidence, strategy=strategy)
    rows = []
    for iteration in range(repeat):
        request = ModelRequest(prompt=prompt, response_format="json_object",
                               thinking={"type": thinking}, stream=stream)
        started = perf_counter()
        try:
            response = model.generate_stream(request) if stream else model.generate(request)
            error = None
            quality = assess(response, case, evidence)
        except TimeoutError as exc:
            response = None
            error = {"code": "generation_timeout", "type": type(exc).__name__}
            quality = {"json_valid": False, "claim_present": False, "citation_valid": False,
                       "completeness": "unassessed"}
        except Exception as exc:
            response = None
            error = {"code": "provider_error", "type": type(exc).__name__}
            quality = {"json_valid": False, "claim_present": False, "citation_valid": False,
                       "completeness": "unassessed"}
        rows.append({"case_id": case["id"], "scenario": case["scenario"], "arm": arm,
                     "iteration": iteration, "prompt_chars": len(prompt),
                     "evidence_chars": sum(len(hit["excerpt"]) for hit in evidence),
                     "latency_ms": round((perf_counter() - started) * 1000, 3),
                     "ttft_ms": response.ttft_ms if response else None,
                     "finish_reason": response.finish_reason if response else None,
                     "usage": response.usage if response else None,
                     "reasoning_tokens": response.reasoning_tokens if response else None,
                     "quality": quality, "error": error})
    return rows


def summarize(rows: list[dict]) -> dict:
    output = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        latencies = [row["latency_ms"] for row in selected]
        ttfts = [row["ttft_ms"] for row in selected if row["ttft_ms"] is not None]
        output[arm] = {
            "requests": len(selected),
            "timeouts": sum(bool(row["error"] and row["error"]["code"] == "generation_timeout") for row in selected),
            "provider_errors": sum(bool(row["error"] and row["error"]["code"] == "provider_error") for row in selected),
            "json_valid": sum(row["quality"]["json_valid"] for row in selected),
            "citation_valid": sum(row["quality"]["citation_valid"] for row in selected),
            "p50_generation_ms": median(latencies),
            "p50_ttft_ms": median(ttfts) if ttfts else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or args.output.exists():
        raise ValueError("repeat must be positive and output must not already exist")
    config = load_provider_config(args.config) if args.config else load_provider_config()
    config.require_model()
    model = OpenAICompatibleModel(config=config)
    cases = load_cases()
    nodes = load_nodes()
    rows = [row for case in cases for arm in ARMS for row in run_case(model, case, nodes, arm, args.repeat)]
    report = {
        "schema_version": 1, "experiment": "generation-only-controlled-v1",
        "provider": config.summary(), "controls": {"retries": 0, "repeat": args.repeat,
        "retrieval": "excluded", "temperature": 0},
        "case_sha256": hashlib.sha256(CASE_PATH.read_bytes()).hexdigest(),
        "arms": ARMS, "unsupported_provider_comparison": True,
        "records": rows, "summary": summarize(rows), "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
