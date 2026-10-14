"""Paired per-turn evaluation of real-provider bounded loop conversations."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any
from uuid import uuid4
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from httpx import ASGITransport, AsyncClient, RequestError
from opentelemetry.trace import NoOpTracerProvider
from app.main import create_app
from app.agent_routes import KnowledgeGlossaryAdapter
from scripts.evaluate_agent_loop import (
    MATRIX, SNAPSHOT, RecordingModel, ScenarioKnowledge, preflight, same,
    score_response, step_arguments,
)
from scripts.evaluation_contract import report_provenance
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig, load_provider_config

CASES = ROOT / "metadata/agent_loop_conversation_live_v1.json"
SCOPE_KEYS = ("player_ids", "team_ids", "season_id", "competition_phase", "metric_terms")
DIAGNOSTICS_VERSION = "request-local-v1"


def load_cases(path: Path = CASES) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cases = document.get("cases")
    if document.get("case_set_version") != "1.0.0" or document.get("source_snapshot") != SNAPSHOT:
        raise ValueError("conversation manifest snapshot/version mismatch")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("conversation manifest requires twelve cases")
    if any(not isinstance(c, dict) or not isinstance(c.get("case_id"), str) or not c["case_id"] for c in cases):
        raise ValueError("invalid conversation ID")
    if len({c.get("case_id") for c in cases}) != 12:
        raise ValueError("conversation IDs must be unique")
    if sum(c.get("split") == "development" for c in cases) != 4 or sum(c.get("split") == "holdout" for c in cases) != 8:
        raise ValueError("conversation manifest requires 4 development / 8 holdout")
    for case in cases:
        if not isinstance(case.get("turns"), list) or not 2 <= len(case["turns"]) <= 3:
            raise ValueError("each conversation requires 2-3 turns")
        for turn in case["turns"]:
            expected = turn.get("expected")
            if not turn.get("question") or not isinstance(expected, dict) or not {"status", "route", "scope", "tools"} <= expected.keys():
                raise ValueError("each turn requires a question and complete expectation")
            if expected["status"] not in {"complete", "partial", "clarification_required", "unsupported", "not_computable", "incomplete"}:
                raise ValueError("invalid expected status")
            if expected["route"] not in {"structured", "knowledge", "mixed", "comparison", "clarification", "unsupported"}:
                raise ValueError("invalid expected route")
            if not isinstance(expected["scope"], dict) or not isinstance(expected["tools"], list):
                raise ValueError("invalid scope/tools expectation")
            if turn.get("scenario") not in {None, "injected_knowledge_unavailable"}:
                raise ValueError("unsupported injected scenario")
            for tool in expected["tools"]:
                if tool.get("capability") not in {"season_summary", "player_efg", "window_query", "efg_compare"}:
                    raise ValueError("unsupported numeric oracle")
    return cases


def select_cases(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in {"development", "holdout", "all"}:
        raise ValueError("unknown conversation split")
    return [case for case in cases if split == "all" or case["split"] == split]


def snapshot_scope(snapshot) -> dict[str, Any]:
    values = asdict(snapshot) if snapshot is not None else {}
    return {key: list(values.get(key, ())) if key in {"player_ids", "team_ids", "metric_terms"}
            else values.get(key) for key in SCOPE_KEYS}


def build_oracles(database: Path, tools: list[dict]) -> list[dict]:
    """Independent raw-table SQL; never derive expected values from app output."""
    records, shooting = [], {}
    base = (" FROM player_game_stat p JOIN game g ON g.game_id=p.game_id "
            "WHERE p.person_id=? AND g.season_id=? AND g.competition_phase=? "
            "AND p.appearance_status IN ('played','played_inferred')")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for tool in tools:
            kind, args = tool["capability"], tool["arguments"]
            if kind == "efg_compare":
                left, right = shooting[args["left_entity_id"]], shooting[args["right_entity_id"]]
                records.append({**tool, "values": {"left_efg_percentage": left, "right_efg_percentage": right,
                    "absolute_delta": left - right}, "formula": "left - right, percentage points"})
                continue
            params = [args[k] for k in ("entity_id", "season_id", "competition_phase")]
            if kind == "season_summary":
                sql = "SELECT SUM(p.points)" + base
                values = {"points_total": connection.execute(sql, params).fetchone()[0]}
            elif kind == "window_query":
                sql = "SELECT p.points,g.game_date,p.game_id" + base + " ORDER BY g.game_date DESC,p.game_id DESC LIMIT ?"
                params.append(args["last_n"])
                rows = connection.execute(sql, params).fetchall()
                if not rows:
                    raise ValueError("window oracle has no samples")
                values = {"values": {"points_per_appearance": sum(r[0] for r in rows) / len(rows)},
                          "actual_sample_count": len(rows)}
            else:
                sql = "SELECT SUM(p.field_goals_made),SUM(p.three_pointers_made),SUM(p.field_goals_attempted),COUNT(*)" + base
                fgm, threes, fga, n = connection.execute(sql, params).fetchone()
                if not fga or not n:
                    raise ValueError("eFG oracle has no valid samples")
                shooting[args["entity_id"]] = (fgm + .5 * threes) / fga * 100
                values = {"efg_percentage": shooting[args["entity_id"]], "inputs": {
                    "field_goals_made": fgm, "three_pointers_made": threes,
                    "field_goals_attempted": fga, "sample_count": n}}
            records.append({**tool, "values": values, "sql": sql, "parameters": params})
    return records


def score_turn(turn, body, oracle, scope, *, turn_index, conversation_id, prior_scope=None):
    score = score_response(turn, body, oracle)
    checks = dict(score["checks"])
    checks["turn_index"] = body.get("turn_index") == turn_index
    checks["conversation_id"] = body.get("conversation_id") == conversation_id
    checks["scope"] = same(scope, turn["expected"].get("scope", {}))
    if prior_scope is not None and "context" in body:
        checks["prior_context"] = same(body.get("context", {}).get("inherited", {}), prior_scope)
    wanted = turn["expected"].get("tools", [])
    numeric_steps = [s for s in body.get("steps", []) if s.get("capability") != "knowledge_glossary" and s.get("status") == "complete"]
    checks["grounded_tools"] = len(numeric_steps) == len(wanted) and all(
        any(s.get("capability") == tool["capability"] and same(step_arguments(body, s), tool["arguments"])
            for s in numeric_steps) for tool in wanted)
    checks["no_unrequested_knowledge"] = (
        turn["expected"].get("knowledge_required", False)
        or turn.get("scenario") == "injected_knowledge_unavailable"
        or not any(s.get("capability") == "knowledge_glossary" for s in body.get("steps", [])))
    if score["outcome"] in {"provider_runtime", "evaluation_contract"}:
        return {**score, "checks": checks}
    passed = all(checks.values())
    return {**score, "checks": checks, "outcome": "passed" if passed else "failed",
            "reason": None if passed else next(k for k, v in checks.items() if not v)}


def summarize(records):
    arms = {}
    for arm in ("rules", "loop"):
        selected = [r for r in records if r["arm"] == arm]
        turns = [t for r in selected for t in r["turns"]]
        passed = sum(all(t["outcome"] == "passed" for t in r["turns"]) for r in selected)
        scope = [t["checks"]["scope"] for t in turns if "scope" in t["checks"]]
        numeric = [t["checks"]["numeric_fidelity"] for t in turns if "numeric_fidelity" in t["checks"]]
        natural = [t for t in turns if not t.get("scenario")]
        latencies = sorted(t["latency_ms"] for t in turns)
        arms[arm] = {
            "conversations": len(selected), "conversations_passed": passed,
            "conversation_success_rate": passed / len(selected) if selected else None,
            "turns": len(turns), "turns_passed": sum(t["outcome"] == "passed" for t in turns),
            "turn_success_rate": sum(t["outcome"] == "passed" for t in turns) / len(turns) if turns else None,
            "scope_checks": len(scope), "scope_accuracy": sum(scope) / len(scope) if scope else None,
            "numeric_checks": len(numeric), "numeric_fidelity": sum(numeric) / len(numeric) if numeric else None,
            "natural_turns": len(natural),
            "provider_runtime_failures": sum(t["outcome"] == "provider_runtime" for t in natural),
            "evaluation_contract_failures": sum(t["outcome"] == "evaluation_contract" for t in turns),
            "injected_failure_turns": sum(bool(t.get("scenario")) for t in turns),
            "mean_turn_latency_ms": sum(t["latency_ms"] for t in turns) / len(turns) if turns else None,
            "p95_turn_latency_ms": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else None,
            "mean_conversation_latency_ms": sum(latencies) / len(selected) if selected else None,
            "model_calls": sum(t.get("model_calls", 0) for t in turns),
            "tool_calls": sum(t.get("tool_calls", 0) for t in turns),
            "model_error_attempts": sum(bool(req.get("error_chain")) for t in turns for req in t.get("model_requests", [])),
            "decision_failures": sum(len(t.get("decision_failures", [])) for t in turns),
            "decision_failure_categories": dict(Counter(
                failure.get("detail_code", failure["code"])
                for t in turns for failure in t.get("decision_failures", []))),
        }
    by_case = {}
    for record in records:
        by_case.setdefault(record["case_id"], {})[record["arm"]] = all(t["outcome"] == "passed" for t in record["turns"])
    pairs = [p for p in by_case.values() if set(p) == {"rules", "loop"}]
    return {"arms": arms, "paired": {
        "total": len(pairs), "both_passed": sum(p["rules"] and p["loop"] for p in pairs),
        "rules_only_passed": sum(p["rules"] and not p["loop"] for p in pairs),
        "loop_only_passed": sum(p["loop"] and not p["rules"] for p in pairs)}}


def summarize_repeats(reports):
    return {"repeats": len(reports), "unique_conversations": reports[0]["summary"]["paired"]["total"],
            "runs": [r["summary"] for r in reports],
            "limitations": ["Repeated observations are not independent cases; no significance claim."]}


async def run_paired(database, matrix, index, *, cases=None, cases_path=CASES, decider=None,
                     knowledge=None, provider_config=None, provider_health=False, repeat=1, split="all"):
    cases = select_cases(cases if cases is not None else load_cases(cases_path), split)
    config = provider_config or load_provider_config()
    live = decider is None
    resources = preflight(database, matrix, index, require_model=live, open_index=live,
                          provider_config=config, check_provider=live and provider_health)
    search = resources.pop("search")
    tracer = NoOpTracerProvider().get_tracer("loop-conversation")
    inputs = {"cases": cases_path, "matrix": matrix, "catalog": ROOT / "metadata/schema_catalog.json",
              "loop": ROOT / "app/agent_loop.py", "tasks": ROOT / "app/agent_tasks.py",
              "actions": ROOT / "app/agent_actions.py", "loop_route": ROOT / "app/agent_loop_routes.py",
              "prompt": ROOT / "app/agent_decider.py", "evaluator": Path(__file__)}
    if index is not None:
        inputs.update(nodes=index / "nodes.json", dense=index / "dense.json")
    records = []
    try:
        model = RecordingModel(OpenAICompatibleModel(config=config)) if live else None
        provenance = report_provenance(root=ROOT, source_snapshot=SNAPSHOT,
            model_id=model.model if model else "fixture-decider", route="paired_conversation", input_paths=inputs)
        adapter = ScenarioKnowledge(knowledge or KnowledgeGlossaryAdapter(search, model=model, tracer=tracer, provider_config=config))
        app = create_app(model=model, db_path=database, quality_matrix_path=matrix, knowledge_search=search,
            provider_config=config, tracer=tracer, agent_knowledge_adapter=adapter,
            agent_loop_enabled=True, agent_loop_decider=decider, ask_total_timeout_seconds=60)
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://conversation") as client:
            for position, case in enumerate(cases):
                oracles = [build_oracles(database, turn["expected"]["tools"]) for turn in case["turns"]]
                for arm in (("rules", "loop") if (position + repeat) % 2 else ("loop", "rules")):
                    conversation_id, turns = uuid4().hex, []
                    for number, (turn, oracle) in enumerate(zip(case["turns"], oracles), 1):
                        before = snapshot_scope(app.state.agent_conversations.get(conversation_id))
                        adapter.injected = bool(turn.get("scenario"))
                        begin = len(model.records) if model else 0
                        started = time.monotonic()
                        try:
                            response = await asyncio.wait_for(client.post(
                                "/agent/answer" if arm == "rules" else "/agent/loop",
                                json={"question": turn["question"], "conversation_id": conversation_id}), 65)
                            status = response.status_code
                            body = response.json() if status == 200 else {}
                            scope = snapshot_scope(app.state.agent_conversations.get(conversation_id))
                            if status != 200:
                                scored = {"outcome": "evaluation_contract", "reason": f"http_{status}", "checks": {}}
                            else:
                                scored = score_turn(turn, body, oracle, scope, turn_index=number,
                                                    conversation_id=conversation_id, prior_scope=before)
                        except (asyncio.TimeoutError, OSError, RequestError) as exc:
                            status, body = 599, {}
                            scope = snapshot_scope(app.state.agent_conversations.get(conversation_id))
                            scored = {"outcome": "provider_runtime", "reason": type(exc).__name__, "checks": {}}
                        except (ValueError, TypeError, KeyError) as exc:
                            status, body = 599, {}
                            scope = snapshot_scope(app.state.agent_conversations.get(conversation_id))
                            scored = {"outcome": "evaluation_contract", "reason": type(exc).__name__, "checks": {}}
                        if "scope" not in scored["checks"]:
                            scored["checks"]["scope"] = same(scope, turn["expected"]["scope"])
                        turns.append({"turn_index": number, "question": turn["question"], "scenario": turn.get("scenario"),
                            "http_status": status, "latency_ms": round((time.monotonic() - started) * 1000, 3),
                            **scored, "response": body, "prior_scope": before, "saved_scope": scope,
                            "oracle": oracle, "model_calls": len(model.records) - begin if model else 0,
                            "model_requests": model.records[begin:] if model else [],
                            "tool_calls": body.get("tool_calls", len(body.get("steps", []))),
                            "decision_failures": body.get("decision_failures", []),
                            "token_usage": None, "cost": None})
                    passed = all(t["outcome"] == "passed" for t in turns)
                    records.append({"case_id": case["case_id"], "split": case["split"], "arm": arm,
                                    "conversation_id": conversation_id, "outcome": "passed" if passed else "failed", "turns": turns})
                    print(f"{case['case_id']} {arm}: {'passed' if passed else 'failed'}", flush=True)
        final = report_provenance(root=ROOT, source_snapshot=SNAPSHOT,
            model_id=model.model if model else "fixture-decider", route="paired_conversation", input_paths=inputs)
        return {"mode": "paired_real_conversation" if live else "fixture_conversation_contract", "case_set_version": "1.0.0",
            "diagnostics_version": DIAGNOSTICS_VERSION, "split": split,
            "repeat": repeat, "provenance": provenance, "final_provenance": final,
            "resources": resources, "provider": config.summary(), "records": records,
            "summary": summarize(records), "splits": {split: summarize([r for r in records if r["split"] == split])
                for split in ("development", "holdout")},
            "limitations": ["Citation shape is not semantic support; manual evidence review is separate.",
                "Previously inspected manifests are regression evidence, not blind holdout.",
                "Missing token usage/cost is unknown; repeated runs are not independent samples."]}
    finally:
        if search is not None:
            search.dense.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--capability-matrix", type=Path, default=MATRIX)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--split", choices=("development", "holdout", "all"), default="all")
    parser.add_argument("--provider-health", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/agent-loop/conversation-live-v1.json")
    args = parser.parse_args()
    config = load_provider_config(args.config) if args.config else load_provider_config()
    database = args.db or Path(config.database_path or "/tmp/nba-m1.sqlite")
    index = args.index or Path(config.knowledge_index_path or str(ROOT / "data/indexes/knowledge-500-dense"))
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        load_cases(args.cases)
        if args.preflight_only:
            resources = preflight(database, args.capability_matrix, index, open_index=True,
                provider_config=config, check_provider=args.provider_health)
            search = resources.pop("search")
            try:
                print(json.dumps(resources, ensure_ascii=False, indent=2))
            finally:
                if search is not None:
                    search.dense.close()
            return 0
        reports = []
        for repeat in range(1, args.repeats + 1):
            report = asyncio.run(run_paired(database, args.capability_matrix, index, cases_path=args.cases,
                provider_config=config, provider_health=args.provider_health, repeat=repeat, split=args.split))
            reports.append(report)
        # Write after all repeats so new report files cannot dirty later run provenance.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for repeat, report in enumerate(reports, 1):
            path = args.output.with_name(f"{args.output.stem}-r{repeat}{args.output.suffix}")
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        aggregate = {"mode": "paired_real_conversation_repeated", "diagnostics_version": DIAGNOSTICS_VERSION,
            "split": args.split, "runs": [
            {"report": f"{args.output.stem}-r{i}{args.output.suffix}", "provenance": r["provenance"],
             "summary": r["summary"], "splits": r["splits"]} for i, r in enumerate(reports, 1)],
             "summary": summarize_repeats(reports)}
        args.output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(aggregate["summary"], ensure_ascii=False, indent=2))
        return int(any(r["outcome"] != "passed" for report in reports for r in report["records"]))
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Preflight/evaluation failed ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
