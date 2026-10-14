"""Generate one fixed planner cache for embedding candidate comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter
from pathlib import Path

from scripts.knowledge_guidance import build_guidance_prompt, parse_guidance_plan
from scripts.model_adapter import ModelRequest, OpenAICompatibleModel
from scripts.provider_config import load_provider_config


def cases_hash(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout", "validation", "blind_holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite plan cache")
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    cases = all_cases if args.split == "blind_holdout" else [case for case in all_cases if case["split"] == args.split]
    if args.reuse_cache:
        source = json.loads(args.reuse_cache.read_text(encoding="utf-8"))
        if set(source.get("plans", {})) != {case["question"] for case in cases}:
            raise ValueError("source planner cache does not cover the selected case questions")
        payload = dict(source)
        payload["cases_sha256"] = cases_hash(cases)
        payload["source_cache"] = str(args.reuse_cache)
        payload["split"] = args.split
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"reused {len(cases)} planner plans from {args.reuse_cache}")
        return
    config = load_provider_config()
    model = OpenAICompatibleModel(timeout=config.agent_decision_timeout_seconds)
    plans = {}
    latency_ms = {}
    for case in cases:
        started = perf_counter()
        response = model.generate(ModelRequest(prompt=build_guidance_prompt(case["question"]), response_format="json_object"))
        try:
            plan = parse_guidance_plan(response.text, case["question"])
        except ValueError:
            # Some compatible endpoints append a short reasoning tail despite
            # response_format=json_object. Keep only the first JSON object and
            # still run the repository's semantic/number validation.
            decoder = json.JSONDecoder()
            raw = response.text.lstrip()
            cursor = 0
            plan = None
            while cursor < len(raw):
                try:
                    obj, end = decoder.raw_decode(raw[cursor:])
                    cursor += end
                    if isinstance(obj, dict) and "action" in obj:
                        try:
                            plan = parse_guidance_plan(json.dumps(obj, ensure_ascii=False), case["question"])
                        except ValueError:
                            plan = None
                        if plan is not None:
                            break
                except (ValueError, TypeError):
                    cursor += 1
            if plan is None:
                raise ValueError(f"could not extract a valid guidance plan for {case['id']}")
        plans[case["question"]] = plan.model_dump()
        latency_ms[case["question"]] = round((perf_counter() - started) * 1000, 3)
    payload = {"schema_version": 1, "cases_sha256": cases_hash(cases), "planner_model": model.model,
               "split": args.split, "case_selection": "all_cases" if args.split == "blind_holdout" else "split",
               "plans": plans, "latency_ms": latency_ms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cached {len(plans)} planner plans using {model.model}")


if __name__ == "__main__":
    main()
