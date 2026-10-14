"""Run the deterministic M4.1 Agent contract and observability evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent_routes import (
    AgentCapabilityError,
    CapabilityRegistry,
    RoutePlan,
    RouteStep,
    execute_agent_plan,
    plan_question,
)
from scripts.query_context import EntityLink, QueryLinks, SEASON_PATTERN, _aliases
from scripts.query_lexicon import load_lexicon


CASES = ROOT / "metadata/m4_1_agent_contract_cases_v1.json"
ORACLE = {
    "efg_percentage": 57.3356401384083,
    "inputs": {
        "field_goals_made": 650,
        "three_pointers_made": 357,
        "field_goals_attempted": 1445,
        "sample_count": 74,
    },
}


class ContractStructuredAdapter:
    name = "player_efg"
    route = "structured"

    def __init__(self, scenario: str):
        self.scenario = scenario

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question, arguments
        if self.scenario == "database_unavailable":
            raise AgentCapabilityError("database_unavailable", "fixture database unavailable")
        return {
            "status": "complete",
            **ORACLE,
            "source_sql": "SELECT fixture_efg",
            "evidence": {
                "source_sql": "SELECT fixture_efg",
                "formula": "(FGM + 0.5 * 3PM) / FGA * 100",
                "units": {"efg_percentage": "percent"},
                "precision": {"rounding": "full_precision"},
                "sample_count": ORACLE["inputs"]["sample_count"],
            },
        }


class ContractKnowledgeAdapter:
    name = "knowledge_glossary"
    route = "knowledge"

    def __init__(self, scenario: str):
        self.scenario = scenario

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question, arguments
        if self.scenario == "mixed_knowledge_failed":
            raise AgentCapabilityError("knowledge_unavailable", "fixture knowledge unavailable")
        if self.scenario == "mixed_corpus_scope":
            raise AgentCapabilityError("outside_corpus_scope", "fixture corpus has no limitation evidence")
        if self.scenario == "knowledge_index_unavailable":
            raise AgentCapabilityError("knowledge_unavailable", "fixture knowledge index unavailable")
        if self.scenario == "provider_timeout":
            raise AgentCapabilityError("generation_timeout", "fixture provider timed out")
        if self.scenario == "mixed_knowledge_partial":
            return {
                "status": "partial",
                "answer": "eFG% 反映投篮效率。",
                "claims": [],
                "unresolved": ["limitations"],
            }
        return {
            "status": "answered",
            "answer": "eFG% 调整三分球价值后衡量投篮效率。",
            "claims": [{
                "citation_id": "fixture-efg",
                "excerpt": "Effective Field Goal Percentage",
                "source_id": "fixture-glossary",
                "source_version": "fixture-v1",
                "validation": "validated",
            }],
            "unresolved": [],
        }


def _plan_for_case(case: Mapping[str, Any]) -> RoutePlan:
    scenario = case["scenario"]
    if scenario == "unregistered_plan":
        return RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="unknown_capability")])
    # This adapter-only contract has no database. Supply explicit synthetic
    # links from the fixture question instead of planner fallback identities.
    question = case["question"]
    links = QueryLinks(
        players=[EntityLink(alias=alias, id=identity, display_name=alias) for alias, identity in _aliases(question, "player")],
        seasons=list(dict.fromkeys(SEASON_PATTERN.findall(question))),
        phases=[phase for phase, aliases in load_lexicon()["value_domains"]["competition_phase"].items()
                if any(alias.lower() in question.lower() for alias in aliases)],
    )
    return plan_question(question, links=links)


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return left == right


def _trace_names(exporter: InMemorySpanExporter) -> list[str]:
    spans = sorted(exporter.get_finished_spans(), key=lambda item: item.start_time or 0)
    return [item.name for item in spans]


def _score(response: Mapping[str, Any], case: Mapping[str, Any], *, trace_names: list[str]) -> dict[str, Any]:
    expected = case["expected"]
    checks: dict[str, bool] = {
        "planning": response.get("route") == expected.get("route"),
        "status": response.get("status") == expected.get("status"),
        "reason_code": response.get("reason_code") == expected.get("reason_code") if expected.get("reason_code") else True,
        "observability": bool(response.get("trace_id")) and trace_names.count("nba.agent.request") == 1 and trace_names.count("nba.agent.planner") == 1,
    }
    if expected.get("numeric"):
        step = next((item for item in response.get("steps", []) if item.get("capability") == "player_efg"), {})
        result = step.get("result") or {}
        checks["numeric_fidelity"] = _same(result.get("efg_percentage"), ORACLE["efg_percentage"]) and result.get("inputs") == ORACLE["inputs"]
        evidence = step.get("evidence", {}).get("structured", {})
        checks["structured_evidence"] = all(evidence.get(key) for key in ("source_sql", "formula", "units", "precision"))
    if expected.get("knowledge"):
        knowledge = next((item for item in response.get("steps", []) if item.get("capability") == "knowledge_glossary"), {})
        checks["knowledge_status"] = knowledge.get("status") == expected["knowledge"]
        if expected["knowledge"] == "complete":
            citations = knowledge.get("evidence", {}).get("knowledge", {}).get("citations", [])
            checks["citation_shape"] = bool(citations and citations[0].get("citation_id") and citations[0].get("excerpt"))
        if expected.get("knowledge_reason"):
            checks["knowledge_reason"] = (knowledge.get("result") or {}).get("reason_code") == expected["knowledge_reason"]
    if case["category"] in {"clarification", "unsupported"}:
        checks["no_executed_steps"] = not response.get("steps")
    passed = all(checks.values())
    scenario = case["scenario"]
    runtime = scenario in {"database_unavailable", "knowledge_index_unavailable", "provider_timeout"}
    return {
        "outcome": "passed" if passed else "failed",
        "checks": checks,
        "failure_class": "provider_runtime" if runtime else (None if passed else "capability"),
        "provider_runtime": runtime,
        "evaluation_contract": False,
        "reason": None if passed else next(name for name, value in checks.items() if not value),
    }


async def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("m4.1-contract")
    plan = _plan_for_case(case)
    scenario = case["scenario"]
    registry = CapabilityRegistry([
        ContractStructuredAdapter(scenario),
        ContractKnowledgeAdapter(scenario),
    ])
    response = await execute_agent_plan(case["question"], plan, registry, request_id=case["case_id"], tracer=tracer, deadline_seconds=10.0)
    payload = response.model_dump(mode="json")
    names = _trace_names(exporter)
    scored = _score(payload, case, trace_names=names)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "scenario": scenario,
        "question": case["question"],
        "outcome": scored["outcome"],
        "failure_class": scored["failure_class"],
        "provider_runtime": scored["provider_runtime"],
        "evaluation_contract": scored["evaluation_contract"],
        "reason": scored["reason"],
        "checks": scored["checks"],
        "trace_id": payload.get("trace_id"),
        "trace_available": bool(payload.get("trace_id")) and bool(names),
        "span_names": names,
        "step_timings_ms": {key: value for key, value in (payload.get("timings") or {}).items() if key != "total_ms"},
        "response": payload,
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capability = [row for row in records if not row["provider_runtime"] and not row["evaluation_contract"]]
    dimensions = sorted({key for row in capability for key in row["checks"]})
    dimension_passes = {key: sum(bool(row["checks"].get(key)) for row in capability) for key in dimensions}
    dimension_denominators = {key: sum(key in row["checks"] for row in capability) for key in dimensions}
    dimension_passes["observability"] = sum(bool(row["checks"].get("observability")) for row in records)
    dimension_denominators["observability"] = len(records)
    return {
        "total": len(records),
        "passed": sum(row["outcome"] == "passed" for row in records),
        "failed": sum(row["outcome"] == "failed" for row in records),
        "provider_runtime_failures": sum(row["provider_runtime"] for row in records),
        "evaluation_contract_failures": sum(row["evaluation_contract"] for row in records),
        "capability_denominator": len(capability),
        "capability_success_rate": round(sum(row["outcome"] == "passed" for row in capability) / len(capability), 4) if capability else None,
        "trace_available": sum(row["trace_available"] for row in records),
        "dimension_passes": dimension_passes,
        "dimension_denominators": dimension_denominators,
    }


def run_contract_evaluation(*, cases_path: Path = CASES, output_path: Path | None = None, case_ids: Sequence[str] | None = None) -> dict[str, Any]:
    document = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = document["cases"]
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case["case_id"] in wanted]
        if len(cases) != len(wanted):
            raise ValueError("unknown contract case ID")
    records = asyncio.run(_run_all(cases))
    report = {
        "case_set_version": document["case_set_version"],
        "mode": "deterministic_agent_contract_m4_1",
        "summary": _summary(records),
        "cases": records,
        "provenance": {"cases": str(cases_path), "route": "agent_answer"},
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


async def _run_all(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        results.append(await _run_case(case))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_contract_evaluation(cases_path=args.cases, output_path=args.output, case_ids=args.case_ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
