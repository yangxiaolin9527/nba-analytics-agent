"""Bounded mixed-capability orchestration for the experimental agent route.

This module deliberately keeps planning and execution small.  A route plan is
validated before any adapter is called, and an execution can contain no more
than four ordered registered capabilities.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from math import isfinite
import re
import sqlite3
import time
from collections.abc import Mapping
from contextlib import closing, nullcontext
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from opentelemetry.trace import use_span

from app.ask_tools import AskToolError, execute_registered_tool
from app.config import AppContext
from app.conversation import ConversationStore
from app.context import database_evidence, query_context
from scripts.answer_renderer import render_tool_answer
from scripts.knowledge_answer import KnowledgeAnswer, KnowledgeAnswerError
from scripts.knowledge_guidance import GuidedKnowledgeSearch
from scripts.model_adapter import OpenAICompatibleModel
from scripts.provider_config import ProviderConfig
from scripts.tracing import mark_error, mark_success, set_attribute, set_text, span, trace_id
from scripts.ask_grounding import AskGroundingError, ground_ask_question, validate_ask_call, recent_n
from scripts.ask_plan import AskPlan
from scripts.query_context import EntityLink
from app.execution_events import ExecutionEventSink, emit_execution_event, execution_event_context
from app.live_stream import execution_sse
from app.semantic_contracts import MAX_PLAN_STEPS


logger = logging.getLogger(__name__)

MIXED_FIXTURE_QUESTION = "库里 2023-24 常规赛 eFG% 是多少？这个指标代表什么、有什么局限？"

MAX_STEPS = MAX_PLAN_STEPS
_FORBIDDEN_ARGUMENT_KEYS = {
    "sql", "query", "statement", "python", "code", "script", "expression",
}
RouteName = Literal["structured", "knowledge", "mixed", "comparison", "composed", "clarification", "unsupported"]
StepStatus = Literal["pending", "running", "complete", "partial", "failed", "not_computable"]
TaskStatus = Literal["pending", "running", "complete", "partial", "failed", "incomplete"]


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    routing_mode: Literal['model', 'rules'] | None = None

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RouteStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    capability: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments")
    @classmethod
    def reject_untrusted_execution_inputs(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    if str(key).lower() in _FORBIDDEN_ARGUMENT_KEYS:
                        raise ValueError(f"argument {key!r} is not allowed")
                    visit(nested)
            elif isinstance(item, list):
                for nested in item:
                    visit(nested)
            elif isinstance(item, str) and ";" in item:
                raise ValueError("statement separators are not allowed")

        for key, nested in value.items():
            if info.data.get('capability') == 'knowledge_glossary' and key == 'term' and isinstance(nested, str):
                continue
            visit({key: nested})
        return value


class RoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: RouteName
    steps: list[RouteStep] = Field(default_factory=list, max_length=MAX_STEPS)
    reason: str = Field(default="", max_length=500)
    reason_code: str | None = Field(default=None, max_length=64)
    comparison_step_ids: list[str] = Field(default_factory=list, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def validate_shape(self) -> "RoutePlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("route step IDs must be unique")
        if self.route in {"clarification", "unsupported"} and self.steps:
            raise ValueError(f"{self.route} route cannot declare executable steps")
        if self.route == "mixed" and len(self.steps) != 2:
            raise ValueError("mixed route requires exactly two ordered steps")
        if self.route in {"structured", "knowledge"} and len(self.steps) != 1:
            raise ValueError(f"{self.route} route requires one step")
        if self.route == "mixed" and [step.capability for step in self.steps] != ["player_efg", "knowledge_glossary"]:
            raise ValueError("mixed route has a fixed ordered capability boundary")
        if self.route == "comparison" and [step.capability for step in self.steps] != ["player_efg", "player_efg", "efg_compare", "knowledge_glossary"]:
            raise ValueError("comparison route has a fixed ordered capability boundary")
        if self.route == 'composed':
            from app.agent_composition import validate_composed_steps
            validate_composed_steps(self.steps, self.comparison_step_ids)
        elif self.comparison_step_ids:
            raise ValueError('comparison references only belong to composed plans')
        return self


class StructuredEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_sql: str | None = None
    source_parameters: list[Any] = Field(default_factory=list)
    snapshot_id: str | None = None
    catalog_version: str | None = None
    entity: dict[str, Any] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    sample_count: int | dict[str, Any] | None = None
    formula: str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    precision: dict[str, Any] = Field(default_factory=dict)
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)


class KnowledgeCitation(BaseModel):
    model_config = ConfigDict(extra="allow")

    citation_id: str
    excerpt: str
    source_id: str | None = None
    source_version: str | None = None
    page: int | None = None
    anchor: str | None = None
    validation: str | bool | None = None


class KnowledgeEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    citations: list[KnowledgeCitation] = Field(default_factory=list)
    source_version: str | None = None
    validation: str | None = None
    unresolved: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="allow")

    capability: str
    step_id: str
    provenance: str | None = None
    execution_status: StepStatus
    warnings: list[str] = Field(default_factory=list)
    structured: StructuredEvidence | None = None
    knowledge: KnowledgeEvidence | None = None
    error_stage: str | None = None
    reason_code: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    capability: str
    status: StepStatus
    result: dict[str, Any] | None = None
    error: str | None = None
    evidence: EvidenceBundle | None = None


class TaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    route: RoutePlan
    current_step: str | None = None
    completed_steps: list[StepResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    deadline_ms: float | None = Field(default=None, ge=0)
    deadline_seconds: float | None = Field(default=None, gt=0)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "complete", "partial", "clarification_required", "unsupported",
        "not_computable", "incomplete",
    ]
    request_id: str
    trace_id: str | None = None
    route: str
    answer: str | None = None
    steps: list[StepResult] = Field(default_factory=list)
    evidence: list[EvidenceBundle] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reason_code: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    task_state: TaskState | None = None
    conversation_id: str | None = None
    conversation_context_status: Literal['new', 'continued', 'reset'] | None = None
    turn_index: int | None = None
    routing_mode: str | None = None
    planning: dict[str, Any] = Field(default_factory=dict)
    error_stage: str | None = None
    execution_events: list[dict[str, Any]] = Field(default_factory=list)
    execution_mode: Literal["bounded_plan", "bounded_loop"] = "bounded_plan"
    actions: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    decisions: int = 0
    tool_calls: int = 0
    termination_reason: str | None = None
    decision_failures: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None


class AgentCapabilityError(RuntimeError):
    """Adapter failure that is safe to expose as a bounded step error."""

    def __init__(self, reason_code: str, message: str | None = None, *, stage: str | None = None,
                 timings: dict[str, float] | None = None, stage_statuses: dict[str, str] | None = None,
                 model_calls: dict[str, Any] | None = None):
        super().__init__(message or reason_code)
        self.reason_code = reason_code
        self.stage = stage
        self.timings = timings or {}
        self.stage_statuses = stage_statuses or {}
        self.model_calls = model_calls or {}


class CapabilityAdapter(Protocol):
    name: str
    route: Literal["structured", "knowledge"]

    async def execute(self, question: str, arguments: dict[str, Any]) -> Any: ...


class CapabilityRegistry:
    """Small explicit registry; registration is the only way to execute a tool."""

    def __init__(self, adapters: list[CapabilityAdapter] | None = None) -> None:
        self._adapters: dict[str, CapabilityAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: CapabilityAdapter) -> None:
        name = getattr(adapter, "name", None)
        route = getattr(adapter, "route", None)
        if not isinstance(name, str) or not name or route not in {"structured", "knowledge"}:
            raise ValueError("adapter must declare a name and structured or knowledge route")
        if name in self._adapters:
            raise ValueError(f"capability already registered: {name}")
        self._adapters[name] = adapter

    def get(self, name: str) -> CapabilityAdapter | None:
        return self._adapters.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def validate(self, plan: RoutePlan) -> str | None:
        expected = {"structured": "structured", "knowledge": "knowledge", "mixed": None}.get(plan.route)
        allowed = {
            "structured": {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg"},
            "knowledge": {"knowledge_glossary"},
            "mixed": {"player_efg", "knowledge_glossary"},
            "comparison": {"player_efg", "efg_compare", "knowledge_glossary"},
            "composed": {"season_summary", "window_query", "player_efg", "knowledge_glossary"},
        }.get(plan.route)
        for step in plan.steps:
            adapter = self.get(step.capability)
            if adapter is None:
                return "unregistered_capability"
            if allowed is not None and step.capability not in allowed:
                return "unsupported_capability"
            if expected is not None and adapter.route != expected:
                return "route_capability_mismatch"
            if plan.route == 'composed' and adapter.route != ('knowledge' if step.capability == 'knowledge_glossary' else 'structured'):
                return 'route_capability_mismatch'
        if plan.route in {"mixed", "comparison"}:
            structured = self.get("player_efg")
            knowledge = self.get("knowledge_glossary")
            if structured is None or knowledge is None:
                return "unregistered_capability"
            if structured.route != "structured" or knowledge.route != "knowledge":
                return "route_capability_mismatch"
        if plan.route == "comparison":
            compare = self.get("efg_compare")
            if compare is None or compare.route != "structured":
                return "unregistered_capability" if compare is None else "route_capability_mismatch"
        return None


def _structured_evidence(result: dict[str, Any], step: RouteStep) -> StructuredEvidence:
    source = dict(result.get("evidence") or {})
    inputs = result.get("inputs") or {}
    for key in ("source_sql", "source_parameters", "sample_count", "quality_status", "limitations"):
        if key not in source and key in result:
            source[key] = result[key]
    if "sample_count" not in source and inputs.get("sample_count") is not None:
        source["sample_count"] = inputs["sample_count"]
    if "formula" not in source:
        source["formula"] = source.get("calculation_formula")
    if not source.get("units") and source.get("unit"):
        source["units"] = {"efg_percentage": source["unit"]}
    if not source.get("precision"):
        source["precision"] = {
            "calculation": "full_precision",
            "rounding": source.get("rounding", "full_precision"),
            "display_decimal_places": 4,
        }
    source.setdefault("source_parameters", [step.arguments.get(key) for key in ("entity_id", "season_id", "competition_phase") if key in step.arguments])
    source.setdefault("filters", {key: value for key, value in step.arguments.items() if key not in {"entity_id", "season_id", "competition_phase"}})
    source.setdefault("quality_status", result.get("quality_status"))
    source.setdefault("limitations", list(result.get("limitations") or []))
    source.setdefault("entity", {"id": step.arguments.get("entity_id"), "subject": "player"})
    source.setdefault("precision", {"calculation": "full_precision", "rounding": "full_precision", "display_decimal_places": 4})
    if step.capability == "season_compare":
        baseline = result.get("baseline") or {}
        comparison = result.get("comparison") or {}
        if not source.get("source_sql"):
            source["source_sql"] = baseline.get("source_sql") or comparison.get("source_sql")
        if source.get("sample_count") is None and (baseline or comparison):
            source["sample_count"] = baseline.get("sample_count")
        if source.get("formula") is None:
            source["formula"] = result.get("evidence", {}).get("absolute_delta_formula")
        source["units"] = {result.get("evidence", {}).get("metric", "metric"): result.get("evidence", {}).get("unit", "value")}
    if step.capability == "efg_compare":
        source["formula"] = result.get("formula")
        source.setdefault("units", {"efg_percentage": "percent"})
    return StructuredEvidence.model_validate(source)


def _knowledge_evidence(result: dict[str, Any]) -> KnowledgeEvidence:
    citations = []
    for claim in result.get("claims") or []:
        if claim.get("citation_id") and claim.get("excerpt"):
            citations.append(KnowledgeCitation(
                citation_id=claim["citation_id"], excerpt=claim["excerpt"],
                source_id=claim.get("source_id"), source_version=claim.get("source_version"),
                page=claim.get("page"), anchor=claim.get("anchor"),
                validation=claim.get("validation", claim.get("supported")),
            ))
    return KnowledgeEvidence(
        citations=citations,
        source_version=result.get("source_version"),
        validation="validated" if citations and result.get("status") in {"answered", "complete", "partial"} else None,
        unresolved=list(result.get("unresolved") or []),
    )


def _bundle(step: RouteStep, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> EvidenceBundle:
    result = result or {}
    structured_capabilities = {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"}
    domain = _structured_evidence(result, step) if step.capability in structured_capabilities else _knowledge_evidence(result)
    kwargs = {"capability": step.capability, "step_id": step.id, "execution_status": status}
    if step.capability in structured_capabilities:
        return EvidenceBundle(**kwargs, provenance="deterministic calculation" if step.capability == "efg_compare" else "nba structured database", structured=domain,
                              error_stage=result.get('error_stage'), reason_code=result.get('reason_code'),
                              timings=result.get('timings') or {},
                              warnings=list(result.get("limitations") or []))
    knowledge_warnings = list(result.get("warnings") or [])
    if error:
        knowledge_warnings.append(error)
    knowledge_warnings.extend(domain.unresolved)
    return EvidenceBundle(**kwargs, provenance="pinned knowledge corpus", knowledge=domain,
                          error_stage=result.get("error_stage"), reason_code=result.get("reason_code"),
                          timings=result.get("timings") or {key: value for key, value in result.items() if key.endswith("_ms") and isinstance(value, (int, float))},
                          warnings=list(dict.fromkeys(knowledge_warnings)))


class PlayerEfgAdapter:
    name = "player_efg"
    route = "structured"

    def __init__(self, settings: AppContext):
        self.settings = settings

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question
        database = self.settings.database
        if database is None or not database.is_file():
            raise AgentCapabilityError("database_unavailable", "query database is unavailable")
        required = {"entity_id", "season_id", "competition_phase"}
        if set(arguments) != required:
            raise AgentCapabilityError("invalid_arguments", "player_efg requires grounded entity, season and phase")

        def run() -> dict[str, Any]:
            with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                connection.execute("PRAGMA query_only = ON")
                deadline = time.monotonic() + self.settings.ask_tool_timeout_seconds
                connection.set_progress_handler(
                    lambda: int(time.monotonic() >= deadline), 1000,
                )
                try:
                    return execute_registered_tool(
                        connection, tool="player_efg", arguments=arguments,
                        matrix_path=self.settings.quality_matrix_path,
                        max_seconds=self.settings.ask_tool_timeout_seconds,
                    )
                except AskToolError as exc:
                    raise AgentCapabilityError(exc.code, str(exc)) from exc
                except sqlite3.OperationalError as exc:
                    if "interrupt" in str(exc).lower():
                        raise AgentCapabilityError("tool_timeout", "structured capability exceeded its timeout") from exc
                    raise AgentCapabilityError("tool_failed", "structured capability query failed") from exc

        # Use an explicit worker and poll its Future.  This matches the
        # existing /ask execution path and avoids SQLite wake-up differences
        # across supported Python runtimes.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nba-agent-efg")
        future = executor.submit(run)
        try:
            deadline = time.monotonic() + self.settings.ask_tool_timeout_seconds
            while not future.done():
                if time.monotonic() >= deadline:
                    raise AgentCapabilityError("tool_timeout", "structured capability exceeded its timeout")
                await asyncio.sleep(0.01)
            return future.result()
        finally:
            future.cancel()
            executor.shutdown(wait=future.done(), cancel_futures=True)


class StructuredToolAdapter:
    """Adapter over the existing bounded, read-only structured tool gateway."""

    route = "structured"

    def __init__(self, name: str, settings: AppContext | None = None, *, database=None,
                 quality_matrix_path=None, timeout_seconds: float = 2.0):
        if name not in {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"}:
            raise ValueError(f"unknown structured capability: {name}")
        self.name = name
        self.database = database if database is not None else (settings.database if settings else None)
        self.quality_matrix_path = quality_matrix_path if quality_matrix_path is not None else (
            settings.quality_matrix_path if settings else None
        )
        self.timeout_seconds = timeout_seconds if settings is None else settings.ask_tool_timeout_seconds
        self.tracer = getattr(settings, "tracer", None)

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del question
        if self.name == "efg_compare":
            left = arguments.get("left_efg")
            right = arguments.get("right_efg")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) for value in (left, right)):
                return {"status": "not_computable", "reason_code": "comparison_inputs_missing"}
            with span(self.tracer, "nba.agent.deterministic_calculation", "CHAIN") as calculation_span:
                set_attribute(calculation_span, "calculation.kind", "efg_comparison")
                set_attribute(calculation_span, "calculation.formula", "left_efg_percentage - right_efg_percentage")
                set_attribute(calculation_span, "calculation.input.left_efg", left)
                set_attribute(calculation_span, "calculation.input.right_efg", right)
                delta = left - right
                output = {
                    "status": "complete", "left_efg_percentage": left,
                    "right_efg_percentage": right, "absolute_delta": delta,
                    "higher": "left" if delta > 0 else "right" if delta < 0 else "tie",
                    "formula": "left_efg_percentage - right_efg_percentage",
                    "evidence": {"formula": "left_efg_percentage - right_efg_percentage",
                                 "units": {"efg_percentage": "percent", "absolute_delta": "percentage_points"},
                                 "precision": {"rounding": "full_precision"}},
                }
                set_attribute(calculation_span, "output.status", output["status"])
                set_text(calculation_span, "output.value", output)
                mark_success(calculation_span)
                return output
        if self.database is None or not self.database.is_file():
            return {"status": "failed", "reason_code": "database_unavailable", "error_stage": "execution"}

        def run_in_worker() -> dict[str, Any]:
            with closing(sqlite3.connect(f"{self.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                connection.execute("PRAGMA query_only = ON")
                deadline = time.monotonic() + self.timeout_seconds
                connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                try:
                    with span(self.tracer, "nba.agent.sql", "TOOL") as sql_span:
                        set_attribute(sql_span, "db.system", "sqlite")
                        set_attribute(sql_span, "db.operation", self.name)
                        set_text(sql_span, "input.arguments", arguments)
                        try:
                            result = execute_registered_tool(
                                connection, tool=self.name, arguments=arguments,
                                matrix_path=self.quality_matrix_path, max_seconds=self.timeout_seconds,
                            )
                            set_text(sql_span, "db.statement", result.get("sql") or result.get("source_sql") or "")
                            set_attribute(sql_span, "output.status", result.get("status", ""))
                            set_attribute(sql_span, "output.row_count", 1 if result.get("status") in {"complete", "partial"} else 0)
                            mark_success(sql_span)
                        except Exception as exc:
                            mark_error(sql_span, exc, reason_code=getattr(exc, "code", None))
                            raise
                    if result.get('status') not in {'complete', 'partial'}:
                        result.setdefault('error_stage', 'execution')
                    context, source_snapshot = query_context()
                    dataset = database_evidence(connection, context, source_snapshot)
                    evidence = dict(result.get("evidence") or {})
                    evidence.update(
                        source_snapshot=source_snapshot,
                        snapshot_id=dataset.snapshot_id,
                        catalog_version=dataset.catalog_version,
                        source_table=result.get("source_table"),
                        source_sql=result.get("sql") or result.get("source_sql"),
                        source_parameters=result.get("source_parameters", []),
                        quality_status=result.get("quality_status"),
                        limitations=list(result.get("limitations") or []),
                    )
                    if self.name in {"season_summary", "window_query", "player_efg", "season_compare"}:
                        entity_id = arguments.get("entity_id")
                        subject = arguments.get("subject", "player")
                        table, key = ("player_identity", "person_id") if subject == "player" else ("team_identity", "team_id")
                        row = connection.execute(f"SELECT display_name FROM {table} WHERE {key} = ?", (entity_id,)).fetchone()
                        evidence["entity"] = {"id": entity_id, "subject": subject, "name": row[0] if row else entity_id}
                        evidence["filters"] = {key: value for key, value in arguments.items() if key in {"date_from", "date_to", "last_n", "opponent_team_id"}}
                        if arguments.get("opponent_team_id"):
                            opponent = connection.execute("SELECT display_name FROM team_identity WHERE team_id = ?", (arguments["opponent_team_id"],)).fetchone()
                            evidence["opponent"] = {"id": arguments["opponent_team_id"], "name": opponent[0] if opponent else arguments["opponent_team_id"]}
                    if self.name == "season_summary":
                        count = "games_appeared" if arguments["subject"] == "player" else "games_played"
                        table = "player_season_stat" if arguments["subject"] == "player" else "team_season_stat"
                        row = connection.execute(f"SELECT {count} FROM {table} WHERE {'person_id' if arguments['subject']=='player' else 'team_id'} = ? AND season_id = ? AND competition_phase = ?", (arguments["entity_id"], arguments["season_id"], arguments["competition_phase"])).fetchone()
                        evidence["sample_count"] = row[0] if row else None
                    if self.name == "window_query":
                        evidence["sample_count"] = result.get("actual_sample_count")
                    if self.name == "player_efg":
                        evidence.update(
                            formula=(result.get("evidence") or {}).get("formula", "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted * 100"),
                            units={"efg_percentage": "percent"},
                            precision={"calculation": "full_precision", "rounding": "full_precision", "display_decimal_places": 4},
                            sample_count=(result.get("inputs") or {}).get("sample_count"),
                            source_parameters=[arguments[key] for key in ("entity_id", "season_id", "competition_phase")],
                        )
                    if self.name == "single_game_score":
                        row = connection.execute("SELECT game_date, home_team_id, away_team_id FROM game WHERE game_id = ?", (arguments["game_id"],)).fetchone()
                        if row:
                            evidence["game_date"] = row[0]
                            for role, team_id in (("home_team", row[1]), ("away_team", row[2])):
                                name = connection.execute("SELECT display_name FROM team_identity WHERE team_id = ?", (team_id,)).fetchone()
                                evidence[role] = {"id": team_id, "name": name[0] if name else team_id}
                            evidence["sample_count"] = 1
                    with span(self.tracer, "nba.agent.deterministic_calculation", "CHAIN") as calculation_span:
                        formula = (evidence.get("formula") or evidence.get("calculation_formula") or
                                   (result.get("evidence") or {}).get("formula"))
                        set_attribute(calculation_span, "calculation.kind", self.name)
                        set_attribute(calculation_span, "calculation.formula", formula or "")
                        set_text(calculation_span, "input.value", result.get("inputs") or arguments)
                        set_attribute(calculation_span, "output.status", result.get("status", ""))
                        set_text(calculation_span, "output.value", {
                            "status": result.get("status"),
                            "result_keys": sorted(result.keys()),
                            "efg_percentage": result.get("efg_percentage"),
                        })
                        mark_success(calculation_span)
                    result["evidence"] = evidence
                    return result
                except AskToolError as exc:
                    return {"status": exc.status, "reason": str(exc), "reason_code": exc.code, "error_stage": "execution"}
                except sqlite3.OperationalError as exc:
                    code = "tool_timeout" if "interrupt" in str(exc).lower() else "tool_failed"
                    logger.warning('structured capability=%s error_type=%s reason_code=%s', self.name, type(exc).__name__, code)
                    return {"status": "failed", "reason": "structured capability query failed", "reason_code": code, "error_stage": "execution"}

        # ThreadPoolExecutor does not propagate the active OpenTelemetry context.
        # Attach the caller context so SQL/calculation spans remain children of
        # the visible agent.step span in Phoenix.
        otel_context = None
        parent_context = None
        # Context propagation is needed only when this adapter is actually
        # traced.  Importing/attaching an OTel context for a no-op adapter
        # creates a cross-thread lifecycle dependency in short-lived tests.
        if self.tracer is not None:
            try:
                from opentelemetry import context as otel_context
                parent_context = otel_context.get_current()
            except Exception:
                pass

        def run() -> dict[str, Any]:
            token = otel_context.attach(parent_context) if otel_context is not None and parent_context is not None else None
            try:
                return run_in_worker()
            finally:
                if token is not None:
                    otel_context.detach(token)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"nba-agent-{self.name}")
        future = executor.submit(run)
        try:
            deadline = time.monotonic() + self.timeout_seconds
            while not future.done():
                if time.monotonic() >= deadline:
                    return {"status": "failed", "reason_code": "tool_timeout", "error_stage": "execution"}
                await asyncio.sleep(0.01)
            return future.result()
        except ValidationError as exc:
            fields = sorted({'.'.join(str(part) for part in item['loc']) for item in exc.errors()})
            logger.warning('structured capability=%s error_type=%s fields=%s', self.name, type(exc).__name__, fields)
            return {"status": "failed", "reason_code": "invalid_arguments", "error_stage": "validation"}
        except Exception as exc:
            logger.warning('structured capability=%s error_type=%s reason_code=tool_failed', self.name, type(exc).__name__)
            return {"status": "failed", "reason_code": "tool_failed", "error_stage": "execution"}
        finally:
            future.cancel()
            executor.shutdown(wait=future.done(), cancel_futures=True)


class KnowledgeGlossaryAdapter:
    name = "knowledge_glossary"
    route = "knowledge"

    def __init__(self, search, *, model=None, tracer=None, answerer=None,
                 provider_config: ProviderConfig | None = None):
        self.search = search
        self.model = model
        self.tracer = tracer
        self.answerer = answerer
        self.provider_config = provider_config or ProviderConfig.empty()

    async def execute(self, question: str, arguments: dict[str, Any]) -> dict[str, Any]:
        question = arguments.get('term') or question
        if arguments.get('context_metric'):
            # Grounded capability context resolves pronouns without allowing
            # the planner to rewrite or expand the quoted user subquestion.
            question = f"Verified metric: {arguments['context_metric']}\nUser subquestion: {question}"
        if self.answerer is not None:
            answerer = self.answerer
        else:
            if self.search is None:
                raise AgentCapabilityError("knowledge_unavailable", "knowledge index unavailable")
            if getattr(self.search, "dense", None) is None:
                raise AgentCapabilityError("knowledge_unavailable", "knowledge answering requires a dense index")
            try:
                model_timeout = self.provider_config.knowledge_model_timeout_seconds
                model = self.model if self.model is not None else OpenAICompatibleModel(
                    timeout=model_timeout, config=self.provider_config,
                )
                guidance_options = ({"min_dense_score": self.provider_config.knowledge_min_dense_score}
                                    if self.provider_config.knowledge_min_dense_score is not None else {})
                guided = GuidedKnowledgeSearch(
                    self.search,
                    model,
                    planner_timeout=self.provider_config.knowledge_planner_timeout_seconds,
                    embedding_timeout=self.provider_config.knowledge_embedding_timeout_seconds,
                    tracer=self.tracer, **guidance_options,
                )
                answerer = KnowledgeAnswer(
                    self.search,
                    model,
                    guided=guided,
                    retrieval_timeout=self.provider_config.knowledge_retrieval_timeout_seconds,
                    generation_timeout=self.provider_config.knowledge_generation_timeout_seconds,
                    verification_timeout=self.provider_config.knowledge_verification_timeout_seconds,
                    total_timeout=self.provider_config.knowledge_total_timeout_seconds,
                    tracer=self.tracer,
                    generation_thinking=self.provider_config.knowledge_generation_thinking,
                    verification_thinking=self.provider_config.knowledge_verification_thinking,
                    generation_prompt_strategy=self.provider_config.knowledge_generation_prompt_strategy,
                    generation_policy=self.provider_config.knowledge_generation_policy,
                    candidate_top_k=self.provider_config.knowledge_candidate_top_k,
                )
            except ValueError as exc:
                raise AgentCapabilityError("knowledge_unconfigured", "knowledge answering is unavailable") from exc
        try:
            result = answerer.aanswer(question)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                raise AgentCapabilityError("knowledge_invalid_response", "knowledge adapter returned an invalid result")
            return result
        except KnowledgeAnswerError as exc:
            raise AgentCapabilityError(exc.reason_code, str(exc), stage=exc.stage,
                                       timings=exc.timings, stage_statuses=getattr(exc, 'stage_statuses', {}),
                                       model_calls=getattr(exc, 'model_calls', {})) from exc


def plan_question(question: str, links=None) -> RoutePlan:
    """Rules-only bounded planner for the M4.0 fixture and nearby requests."""
    text = question.strip()
    lowered = text.lower()
    compact = re.sub(r"[\s`\"'();,]+", "", lowered)
    sql_or_code = (
        re.search(r"\b(?:select|drop|insert|update|delete|truncate|alter)\b", lowered)
        or re.search(r"(?:select|drop|insert|update|delete)(?:\*|from|into|table|where)", compact)
        or re.search(r"\b(?:eval|exec|compile|__import__)\s*\(", lowered)
        or re.search(r"\b(?:python|javascript|shell|bash)\b", lowered)
    )
    if sql_or_code:
        return RoutePlan(route="unsupported", reason="arbitrary query or code is outside the route boundary", reason_code="unsupported_request")
    has_efg = "efg" in lowered or "有效命中率" in text
    has_glossary = any(token in text for token in ("代表", "含义", "理解", "局限", "限制", "指标是什么", "是什么", "什么是"))
    has_efg_numeric_intent = any(token in text for token in ("多少", "是多少", "计算", "数据", "常规赛", "季后赛", "赛季"))
    if has_efg and has_glossary and not has_efg_numeric_intent and not any(
        token in text for token in ("库里", "curry", "球员", "2023", "201939")
    ):
        return RoutePlan(route="knowledge", steps=[RouteStep(id="s1", capability="knowledge_glossary")], reason="glossary explanation")
    if has_efg:
        missing = []
        if not (links and links.players):
            missing.append("entity")
        if not (links and links.seasons):
            missing.append("season")
        regular = any(token in lowered for token in ("常规赛", "regular season"))
        playoffs = any(token in lowered for token in ("季后赛", "playoffs", "postseason"))
        if links and len(links.phases) > 1:
            return RoutePlan(route="clarification", reason="请只指定一个赛事阶段：常规赛或季后赛", reason_code="ambiguous_phase")
        if not regular and not playoffs and links and len(links.phases) == 1:
            regular = links.phases[0] == "regular_season"
            playoffs = links.phases[0] == "playoffs"
        if not regular and not playoffs:
            missing.append("phase")
        if missing:
            return RoutePlan(route="clarification", reason="请补充问题所缺少的实体、赛季或赛事阶段", reason_code="missing_phase" if "phase" in missing else "missing_scope")
        if regular and playoffs:
            return RoutePlan(route="clarification", reason="请只指定一个赛事阶段：常规赛或季后赛", reason_code="ambiguous_phase")
        phase = "regular_season" if regular else "playoffs"
        if len(links.seasons) != 1 or len(links.players) > 2:
            return RoutePlan(route="clarification", reason="比较要求两名球员、一个共同赛季和赛事阶段", reason_code="ambiguous_comparison_scope")
        if len(links.players) == 2:
            ordered_players = sorted(links.players, key=lambda item: text.find(item.alias) if item.alias in text else -1)
            efg_steps = [
                RouteStep(id=f"s{index}", capability="player_efg", arguments={
                    "entity_id": player.id, "season_id": links.seasons[0],
                    "competition_phase": phase,
                })
                for index, player in enumerate(ordered_players, start=1)
            ]
            efg_steps.extend([
                RouteStep(id="s3", capability="efg_compare", arguments={
                    "left_entity_id": ordered_players[0].id,
                    "right_entity_id": ordered_players[1].id,
                    "season_id": links.seasons[0],
                    "competition_phase": phase,
                }),
                RouteStep(id="s4", capability="knowledge_glossary"),
            ])
            return RoutePlan(route="comparison", steps=efg_steps, reason="bounded two-player eFG comparison")
        player_id = links.players[0].id
        season_id = links.seasons[0]
        efg = RouteStep(id="s1", capability="player_efg", arguments={
            "entity_id": player_id, "season_id": season_id, "competition_phase": phase,
        })
        if has_glossary:
            return RoutePlan(route="mixed", steps=[efg, RouteStep(id="s2", capability="knowledge_glossary")], reason="numeric result plus metric explanation")
        return RoutePlan(route="structured", steps=[efg], reason="deterministic eFG result")
    if has_glossary:
        return RoutePlan(route="knowledge", steps=[RouteStep(id="s1", capability="knowledge_glossary")], reason="glossary explanation")
    if any(token in text for token in ("比分", "谁赢", "谁获胜", "胜者", "哪场")):
        game_id = links.games[0].game_id if links and len(links.games) == 1 else None
        if game_id:
            return RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="single_game_score", arguments={"game_id": game_id})], reason="deterministic game score")
        return RoutePlan(route="clarification", reason="请提供能唯一确定比赛的日期和双方球队", reason_code="missing_game")
    if any(token in text for token in ("比上一季", "上赛季", "相对上赛季", "相对上一季", "增幅", "增长率", "变化率", "差值", "相差")):
        if links and (links.players or links.teams) and links.seasons and links.phases:
            subject = "player" if links.players else "team"
            entity_id = (links.players or links.teams)[0].id
            current = links.seasons[-1]
            previous = f"{int(current[:4]) - 1}-{(int(current[:4]) - 1 + 1) % 100:02d}"
            return RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="season_compare", arguments={
                "subject": subject, "entity_id": entity_id, "metric": "points_total",
                "baseline_season": previous, "comparison_season": current,
                "competition_phase": links.phases[0],
            })], reason="deterministic season comparison")
        return RoutePlan(route="clarification", reason="请补充实体、赛季和赛事阶段", reason_code="missing_scope")
    from scripts.query_scope import explicit_dates, requested_date_bounds
    if "最近" in text or explicit_dates(text) or any(token in text for token in ("截至", "自", "以来", "对手", "面对")):
        if links and (links.players or links.teams) and links.seasons and links.phases:
            subject = "player" if links.players else "team"
            entity_id = (links.players or links.teams)[0].id
            from scripts.ask_grounding import requested_metrics
            metrics = sorted(requested_metrics(text)) or ["points_total"]
            if subject == 'player' and 'games_played' in metrics:
                metrics = ['games_appeared' if m == 'games_played' else m for m in metrics]
            opponent_team_id = None
            if subject == "player" and links.teams:
                opponent_team_id = links.teams[0].id
            from scripts.query_scope import QueryScopeError
            try:
                date_from, date_to = requested_date_bounds(text)
            except QueryScopeError as exc:
                return RoutePlan(route='clarification', reason=str(exc), reason_code=exc.code)
            date_args = {key: value for key, value in (('date_from', date_from), ('date_to', date_to)) if value is not None}
            return RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="window_query", arguments={
                "subject": subject, "entity_id": entity_id, "season_id": links.seasons[0],
                "competition_phase": links.phases[0], "metrics": metrics,
                "last_n": recent_n(text) if "最近" in text else None,
                **({"opponent_team_id": opponent_team_id} if opponent_team_id else {}),
                **date_args,
            })], reason="deterministic window query")
        missing_code = "missing_phase" if links and (links.players or links.teams) and links.seasons and not links.phases else "missing_scope"
        return RoutePlan(route="clarification", reason="请补充实体、赛季或赛事阶段", reason_code=missing_code)
    if any(token in text for token in ("总得分", "出场次数", "出场场数", "比赛场数", "场均得分", "总助攻", "总篮板", "得了多少分", "total points", "points")):
        if links and (links.players or links.teams) and links.seasons and links.phases:
            subject = "player" if links.players else "team"
            entity_id = (links.players or links.teams)[0].id
            metrics = []
            if "总得分" in text or re.search(r"得了多少分|total points|points", text, re.I):
                metrics.append("points_total")
            if "比赛场数" in text:
                metrics.append("games_played")
            if "出场次数" in text or "出场场数" in text:
                metrics.append("games_appeared")
            if "场均得分" in text:
                metrics.append("points_per_appearance")
            if "总助攻" in text:
                metrics.append("assists_total")
            if "总篮板" in text:
                metrics.append("rebounds_total")
            metrics = metrics or ["points_total"]
            return RoutePlan(route="structured", steps=[RouteStep(id="s1", capability="season_summary", arguments={
                "subject": subject, "entity_id": entity_id, "season_id": links.seasons[0],
                "competition_phase": links.phases[0], "metrics": metrics,
            })], reason="deterministic season summary")
        missing_code = "missing_phase" if links and (links.players or links.teams) and links.seasons and not links.phases else "missing_scope"
        return RoutePlan(route="clarification", reason="请补充实体、赛季或赛事阶段", reason_code=missing_code)
    return RoutePlan(route="unsupported", reason="question is outside the registered capability boundary")


async def execute_agent_plan(
    question: str,
    plan: RoutePlan,
    registry: CapabilityRegistry,
    *,
    request_id: str | None = None,
    deadline_seconds: float | None = None,
    tracer=None,
    request_span=None,
    planner_span=None,
    skip_planner_span: bool = False,
    validate_outputs: bool = False,
) -> AgentResponse:
    request_id = request_id or uuid4().hex
    started = time.monotonic()
    request_scope = use_span(request_span, end_on_exit=False) if request_span is not None else span(tracer, "nba.agent.request")
    with request_scope as request_handle:
        request_span = request_handle
        set_attribute(request_span, "nba.agent.request_id", request_id)
        set_attribute(request_span, "nba.agent.route", plan.route)
        set_attribute(request_span, "nba.agent.step_count", len(plan.steps))
        if not skip_planner_span:
            emit_execution_event("planning", "running", "规则规划")
        planner_scope = (
            nullcontext(None) if skip_planner_span else
            (nullcontext(planner_span) if planner_span is not None else span(tracer, "nba.agent.planner", "CHAIN"))
        )
        with planner_scope as planner_handle:
            planner_span = planner_handle
            set_attribute(planner_span, "nba.agent.request_id", request_id)
            set_attribute(planner_span, "nba.agent.route", plan.route)
            set_attribute(planner_span, "nba.agent.planner_type", "rules")
            invalid = registry.validate(plan)
            if invalid:
                set_attribute(planner_span, "nba.agent.reason_code", invalid)
                status = "unsupported"
                response = AgentResponse(status=status, request_id=request_id, trace_id=trace_id(), route=plan.route,
                                         reason_code=invalid, warnings=["plan rejected before execution"],
                                         timings={"total_ms": round((time.monotonic() - started) * 1000, 3)})
                set_attribute(request_span, "nba.agent.status", status)
                set_attribute(request_span, "nba.agent.reason_code", invalid)
                return response
        if not skip_planner_span:
            emit_execution_event("planning", "complete", "规则规划完成", metadata={"route": plan.route})
        state = TaskState(
            question=question, route=plan, status="running",
            deadline_seconds=deadline_seconds,
            deadline_ms=(deadline_seconds * 1000 if deadline_seconds is not None else None),
        )
        if plan.route in {"clarification", "unsupported"}:
            state.status = "pending"
            status = "clarification_required" if plan.route == "clarification" else "unsupported"
            code = plan.reason_code or ("clarification_required" if status == "clarification_required" else "unsupported_request")
            response = AgentResponse(status=status, request_id=request_id, trace_id=trace_id(), route=plan.route,
                                     answer=plan.reason, reason_code=code,
                                     timings={"total_ms": round((time.monotonic() - started) * 1000, 3)}, task_state=state)
            set_attribute(request_span, "nba.agent.status", status)
            set_attribute(request_span, "nba.agent.reason_code", code)
            return response

        steps: list[StepResult] = []
        evidence: list[EvidenceBundle] = []
        warnings: list[str] = []
        structured_result: dict[str, Any] | None = None
        knowledge_result: dict[str, Any] | None = None
        timings: dict[str, float] = {}
        for step in plan.steps:
            adapter = registry.get(step.capability)
            assert adapter is not None
            remaining = None
            if deadline_seconds is not None:
                remaining = deadline_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    failure_result = {"status": "failed", "reason_code": "agent_timeout", "unresolved": []}
                    bundle = _bundle(step, "failed", result=failure_result, error="agent deadline reached")
                    record = StepResult(id=step.id, capability=step.capability, status="failed",
                                        result=failure_result, error="agent deadline reached", evidence=bundle)
                    steps.append(record)
                    state.completed_steps.append(record)
                    state.errors.append(f"{step.capability}:agent_timeout")
                    evidence.append(bundle)
                    warnings.append(f"{step.capability}:agent_timeout")
                    break
            state.current_step = step.id
            emit_execution_event("tool", "running", f"调用 {step.capability}", metadata={
                "step_id": step.id, "capability": step.capability,
            })
            output = None
            execution_arguments = dict(step.arguments)
            if step.capability == "efg_compare":
                efg_results = [item.result for item in steps if item.capability == "player_efg" and item.result]
                if len(efg_results) != 2:
                    output = {"status": "not_computable", "reason_code": "comparison_inputs_missing"}
                else:
                    execution_arguments = {
                        **step.arguments,
                        "left_efg": efg_results[0].get("efg_percentage"),
                        "right_efg": efg_results[1].get("efg_percentage"),
                    }
            step_started = time.monotonic()
            with span(tracer, "nba.agent.step", "TOOL") as step_span:
                set_attribute(step_span, "nba.agent.request_id", request_id)
                set_attribute(step_span, "nba.agent.step_id", step.id)
                set_attribute(step_span, "nba.agent.capability", step.capability)
                set_attribute(step_span, "nba.agent.tool_invocation", True)
                set_text(step_span, "input.arguments", execution_arguments)
                set_attribute(step_span, "input.mime_type", "application/json")
                try:
                    if output is None:
                        tool_question = question
                        if plan.route == "comparison" and step.capability == "knowledge_glossary":
                            tool_question = "eFG% 的定义是什么？"
                            if any(token in question for token in ("局限", "限制")):
                                tool_question += "这个指标有什么局限？"
                        output = adapter.execute(tool_question, execution_arguments)
                    if inspect.isawaitable(output):
                        output = await asyncio.wait_for(output, timeout=remaining)
                    if not isinstance(output, dict):
                        raise AgentCapabilityError("invalid_adapter_result", "adapter returned an invalid result")
                    set_text(step_span, "output.value", output)
                    set_attribute(step_span, "output.mime_type", "application/json")
                    if validate_outputs:
                        from app.agent_loop import validate_result
                        validate_result(step.capability, output, execution_arguments)
                    raw_status = str(output.get("status", "complete"))
                    usable = (step.capability in {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"} and raw_status in {"complete", "partial"}) or (step.capability == "knowledge_glossary" and raw_status in {"answered", "complete", "partial"} and bool(output.get("answer")))
                    result_status = "complete" if usable and raw_status != "partial" else (
                        "partial" if raw_status == "partial" else
                        "not_computable" if raw_status == "not_computable" else "failed"
                    )
                    bundle = _bundle(step, result_status, output)
                    record = StepResult(id=step.id, capability=step.capability, status=result_status, result=output, evidence=bundle)
                    steps.append(record)
                    state.completed_steps.append(record)
                    evidence.append(bundle)
                    if step.capability in {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"}:
                        if usable:
                            structured_result = output
                    else:
                        knowledge_result = output if usable else None
                    if not usable:
                        warnings.append(f"{step.capability}:{output.get('reason_code') or 'no_usable_result'}")
                        if step.capability != "knowledge_glossary" and plan.route != 'composed':
                            break
                    emit_execution_event("tool", "complete" if usable else "failed", f"完成 {step.capability}" if usable else f"失败 {step.capability}", metadata={
                        "step_id": step.id, "capability": step.capability,
                        "reason_code": output.get("reason_code") if not usable else None,
                    })
                except asyncio.TimeoutError as exc:
                    code = "agent_timeout"
                    message = "agent deadline reached"
                    step_elapsed = round((time.monotonic() - step_started) * 1000, 3)
                    failure_result = {
                        "status": "failed", "reason_code": code,
                        "unresolved": [message] if step.capability == "knowledge_glossary" else [],
                    }
                    bundle = _bundle(step, "failed", result=failure_result, error=message)
                    record = StepResult(id=step.id, capability=step.capability, status="failed",
                                        result=failure_result, error=message, evidence=bundle)
                    steps.append(record)
                    state.completed_steps.append(record)
                    state.errors.append(f"{step.capability}:{code}")
                    evidence.append(bundle)
                    warnings.append(f"{step.capability}:{code}")
                    emit_execution_event("tool", "failed", f"失败 {step.capability}", elapsed_ms=step_elapsed, metadata={
                        "step_id": step.id, "capability": step.capability, "reason_code": code,
                        "error_stage": "execution",
                    })
                    set_attribute(step_span, "nba.agent.reason_code", code)
                    if plan.route != 'composed' and step.capability in {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"}:
                        break
                except Exception as exc:
                    code = exc.reason_code if isinstance(exc, AgentCapabilityError) else "adapter_failed"
                    message = str(exc) or code
                    step_elapsed = round((time.monotonic() - step_started) * 1000, 3)
                    failure_result = {
                        "status": "failed",
                        "reason_code": code,
                        "unresolved": [message] if step.capability == "knowledge_glossary" else [],
                    }
                    if isinstance(exc, AgentCapabilityError):
                        if exc.stage is not None:
                            failure_result["error_stage"] = exc.stage
                        if exc.timings:
                            failure_result["timings"] = exc.timings
                        if exc.stage_statuses:
                            failure_result['stage_statuses'] = exc.stage_statuses
                        if exc.model_calls:
                            failure_result['model_calls'] = exc.model_calls
                    bundle = _bundle(step, "failed", result=failure_result, error=message)
                    record = StepResult(id=step.id, capability=step.capability, status="failed", result=failure_result, error=message, evidence=bundle)
                    steps.append(record)
                    state.completed_steps.append(record)
                    state.errors.append(f"{step.capability}:{code}")
                    evidence.append(bundle)
                    warnings.append(f"{step.capability}:{code}")
                    emit_execution_event("tool", "failed", f"失败 {step.capability}", elapsed_ms=step_elapsed, metadata={
                        "step_id": step.id, "capability": step.capability, "reason_code": code,
                        "error_stage": failure_result.get("error_stage", "execution"),
                    })
                    set_attribute(step_span, "nba.agent.reason_code", code)
                    if plan.route != 'composed' and step.capability in {"season_summary", "single_game_score", "window_query", "season_compare", "player_efg", "efg_compare"}:
                        break
                finally:
                    step_elapsed = round((time.monotonic() - step_started) * 1000, 3)
                    timings[f"{step.id}_ms"] = step_elapsed
                    set_attribute(step_span, "nba.agent.status", steps[-1].status if steps else "failed")
                    set_attribute(step_span, "nba.agent.elapsed_ms", step_elapsed)

        answer_parts: list[str] = []
        for record in steps:
            if record.capability == "knowledge_glossary" or record.status not in {"complete", "partial"}:
                continue
            planned_step = next(item for item in plan.steps if item.id == record.id)
            ev = record.evidence.structured.model_dump(mode="json") if record.evidence and record.evidence.structured else None
            rendered = render_tool_answer(record.capability, planned_step.arguments, record.result, evidence=ev)
            answer_parts.append(rendered.text)
            warnings.extend(rendered.warnings)
        if knowledge_result is not None and knowledge_result.get("answer"):
            answer_parts.append(str(knowledge_result["answer"]))
        from app.agent_composition import summarize_comparison
        comparison = summarize_comparison(plan.comparison_step_ids, {
            record.id: {'capability': record.capability,
                        'arguments': next(item.arguments for item in plan.steps if item.id == record.id),
                        'status': record.status, 'result': record.result}
            for record in steps}, tracer=tracer)
        if comparison:
            answer_parts.append(comparison['text'])
        knowledge_step = next((step for step in steps if step.capability == "knowledge_glossary"), None)
        knowledge_code = ((knowledge_step.result or {}).get("reason_code") if knowledge_step else None)
        knowledge_code = knowledge_code if knowledge_code not in {None, "no_usable_result"} else None
        if plan.route == 'composed':
            if len(steps) == len(plan.steps) and all(s.status == 'complete' for s in steps):
                status, reason_code = 'complete', None
            elif any(s.status in {'complete', 'partial'} for s in steps):
                status, reason_code = 'partial', 'coverage_incomplete'
            elif any(s.status == 'not_computable' for s in steps):
                status, reason_code = 'not_computable', 'coverage_incomplete'
            else:
                status, reason_code = 'incomplete', 'coverage_incomplete'
        elif structured_result is not None and any(step.capability != "knowledge_glossary" and step.status not in {"complete", "partial"} for step in steps):
            status, reason_code = "partial", "structured_step_incomplete"
        elif structured_result is not None and any(step.capability != "knowledge_glossary" and step.status == "partial" for step in steps):
            status, reason_code = "partial", "structured_step_partial"
        elif structured_result is not None and any(step.capability == "knowledge_glossary" and step.status == "failed" for step in steps):
            status, reason_code = "partial", "knowledge_step_failed"
        elif structured_result is not None and any(step.capability == "knowledge_glossary" and step.status == "partial" for step in steps):
            status, reason_code = "partial", knowledge_code or "knowledge_step_partial"
        elif structured_result is not None and plan.route in {"mixed", "comparison"} and knowledge_result is None:
            status, reason_code = "partial", "knowledge_step_incomplete"
        elif structured_result is not None and len(steps) != len(plan.steps):
            status, reason_code = 'partial', 'coverage_incomplete'
        elif structured_result is not None:
            status, reason_code = "complete", None
        elif knowledge_result is not None:
            if knowledge_step is not None and knowledge_step.status == "partial":
                status, reason_code = "partial", knowledge_code or "knowledge_step_partial"
            else:
                status, reason_code = "complete", None
        elif any(step.status == "not_computable" for step in steps):
            failed = next(step for step in steps if step.status == "not_computable")
            status, reason_code = "not_computable", (failed.result or {}).get("reason_code", "not_computable")
        elif any(step.status == "failed" for step in steps):
            failed = next(step for step in steps if step.status == "failed")
            status, reason_code = "incomplete", (
                (failed.result or {}).get("reason_code") or "no_usable_result"
                if failed.capability == "knowledge_glossary" else "no_usable_result"
            )
        else:
            status, reason_code = "not_computable", "no_usable_result"
        total = round((time.monotonic() - started) * 1000, 3)
        timing_map = dict(timings)
        for record in steps:
            for key, value in ((record.result or {}).get("timings") or {}).items():
                timing_map[f"{record.id}_{key}"] = value
        timing_map["total_ms"] = total
        state.current_step = None
        state.status = status
        response = AgentResponse(status=status, request_id=request_id, trace_id=trace_id(), route=plan.route,
                                 answer="\n".join(answer_parts) if answer_parts else None,
                                 steps=steps, evidence=evidence, warnings=list(dict.fromkeys(warnings)),
                                 reason_code=reason_code, timings=timing_map, task_state=state,
                                 tool_calls=len(steps), comparison=comparison,
                                 error_stage=next(((record.result or {}).get("error_stage") for record in steps if record.status == "failed"), None))
        emit_execution_event("result", "complete" if status == "complete" else "failed", "Agent 执行完成", metadata={
            "status": status, "reason_code": reason_code,
        })
        set_attribute(request_span, "nba.agent.status", status)
        if reason_code:
            set_attribute(request_span, "nba.agent.reason_code", reason_code)
        set_attribute(request_span, "nba.agent.elapsed_ms", total)
        set_attribute(request_span, "nba.agent.evidence_count", len(evidence))
        set_attribute(request_span, "nba.agent.citation_count", sum(len(bundle.knowledge.citations) for bundle in evidence if bundle.knowledge))
        return response


def register_agent_routes(
    application: FastAPI,
    settings: AppContext,
    knowledge_search=None,
    *,
    registry: CapabilityRegistry | None = None,
    structured_adapter: CapabilityAdapter | None = None,
    knowledge_adapter: CapabilityAdapter | None = None,
    provider_config: ProviderConfig | None = None,
) -> None:
    selected_registry = registry or CapabilityRegistry([
        structured_adapter or StructuredToolAdapter("season_summary", settings),
        StructuredToolAdapter("single_game_score", settings),
        StructuredToolAdapter("window_query", settings),
        StructuredToolAdapter("season_compare", settings),
        structured_adapter or StructuredToolAdapter("player_efg", settings),
        StructuredToolAdapter("efg_compare", settings),
        knowledge_adapter or KnowledgeGlossaryAdapter(
            knowledge_search, model=settings.model, tracer=settings.tracer,
            provider_config=provider_config or settings.provider_config,
        ),
    ])
    application.state.agent_registry = selected_registry

    conversations = settings.conversations or ConversationStore()
    application.state.agent_conversations = conversations

    def current_metric_terms(response: AgentResponse) -> tuple[str, ...]:
        terms: list[str] = []
        route = response.task_state.route if response.task_state is not None else None
        for step in route.steps if route is not None else []:
            if step.capability in {"player_efg", "efg_compare"}:
                terms.append("efg_percentage")
            elif step.capability in {"season_summary", "window_query"}:
                terms.extend(str(metric) for metric in step.arguments.get("metrics", []))
            elif step.capability == "season_compare":
                metric = step.arguments.get("metric")
                if metric:
                    terms.append(str(metric))
        return tuple(dict.fromkeys(terms))

    def finish(response, conversation_id, previous, links=None):
        # Persist only grounded, unambiguous values from accepted numeric plans.
        numeric = response.route in {"structured", "mixed", "comparison"} and response.status in {"complete", "partial", "not_computable"}
        metrics = current_metric_terms(response) if numeric else ()
        snapshot = conversations.put(
            conversation_id=conversation_id,
            player_ids=[item.id for item in links.players] if numeric and links else (previous.player_ids if previous else ()),
            team_ids=[item.id for item in links.teams] if numeric and links else (previous.team_ids if previous else ()),
            season_id=links.seasons[0] if numeric and links and len(links.seasons) == 1 else (previous.season_id if previous else None),
            competition_phase=links.phases[0] if numeric and links and len(links.phases) == 1 else (previous.competition_phase if previous else None),
            metric_terms=metrics or (previous.metric_terms if previous else ()),
        )
        return response.model_copy(update={"conversation_id": conversation_id, "turn_index": snapshot.turn_index})

    @application.post("/agent/answer", response_model=AgentResponse)
    async def agent_answer(request: AgentRequest) -> AgentResponse:
        conversation_id = request.conversation_id or uuid4().hex
        if not conversations.try_acquire(conversation_id):
            raise HTTPException(409, detail={"code": "conversation_busy", "conversation_id": conversation_id})
        try:
            sink = ExecutionEventSink(request_id=uuid4().hex)
            with execution_event_context(sink):
                response = await execute_request(request, conversation_id)
            sink.rebind(request_id=response.request_id, trace_id=response.trace_id)
            return response.model_copy(update={"execution_events": sink.snapshot()})
        finally:
            conversations.release(conversation_id)

    @application.post("/agent/answer/stream")
    async def agent_answer_stream(request: AgentRequest):
        async def execute_stream(sink):
            conversation_id = request.conversation_id or uuid4().hex
            if not conversations.try_acquire(conversation_id):
                raise HTTPException(409, detail={"code": "conversation_busy", "conversation_id": conversation_id})
            try:
                response = await execute_request(request, conversation_id)
                sink.rebind(request_id=response.request_id, trace_id=response.trace_id)
                return response.model_copy(update={"execution_events": sink.snapshot()})
            finally:
                conversations.release(conversation_id)

        return execution_sse(execute_stream, request_id=uuid4().hex)

    async def execute_request(request: AgentRequest, conversation_id: str) -> AgentResponse:
        if request.routing_mode or settings.routing_mode:
            from app.routing_service import run_routed_agent
            return await run_routed_agent(request, settings, selected_registry, conversation_id)
        request_id = uuid4().hex
        previous = conversations.get(conversation_id) if request.conversation_id else None
        started = time.monotonic()
        question = request.question
        links = None
        with span(settings.tracer, "nba.agent.request") as request_span:
            set_attribute(request_span, "nba.agent.request_id", request_id)
            try:
                with span(settings.tracer, "nba.agent.planner", "CHAIN") as planner_span:
                    if settings.database is not None and settings.database.is_file():
                        with closing(sqlite3.connect(f"{settings.database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
                            links, question = ground_ask_question(connection, question)
                            if links.ambiguous_entities:
                                raise AskGroundingError("ambiguous_entity", "实体名称有歧义，请提供完整名称", "clarification_required")
                            if previous:
                                conflict = (("同一赛季" in question and links.seasons and previous.season_id and links.seasons != [previous.season_id])
                                            or ("同一阶段" in question and links.phases and previous.competition_phase and links.phases != [previous.competition_phase]))
                                if conflict:
                                    raise AskGroundingError("conversation_scope_conflict", "同一范围的要求与显式赛季或阶段冲突，请明确范围", "clarification_required")
                                # A definition-only turn should not become a statistics request.
                                definition_only = bool(
                                    re.search(r"(?:efg|有效命中率|这个指标)", question, re.I)
                                    and re.search(r"是什么|什么是|定义|含义|代表|理解|局限|限制", question)
                                    and not links.players and not links.seasons and not links.phases
                                )
                                if (
                                    previous.metric_terms == ("efg_percentage",)
                                    and re.search(r"这个指标|该指标|这个数据", question)
                                    and re.search(r"理解|含义|代表|是什么|什么是|定义|局限|限制", question)
                                    and not links.players and not links.seasons and not links.phases
                                ):
                                    question += " eFG%"
                                    definition_only = True
                                if not definition_only:
                                    compare_followup = bool(re.search(r"比较|对比|和.+比|与.+比", question))
                                    if compare_followup and previous.metric_terms and previous.metric_terms != ("efg_percentage",) and not re.search(r"efg|有效命中率", question, re.I):
                                        raise AskGroundingError("comparison_metric_required", "请明确要比较的指标", "clarification_required")
                                    inherited_players = []
                                    implicit_entity = bool(re.search(r"那|他|这名|这位|同一|上一", question)) and not re.search(r"不存在|某个|某位|另一名|另外", question)
                                    if (not links.players and implicit_entity) or compare_followup:
                                        for player_id in previous.player_ids:
                                            row = connection.execute("SELECT display_name FROM player_identity WHERE person_id = ?", (player_id,)).fetchone()
                                            if row and player_id not in {item.id for item in links.players}:
                                                inherited_players.append(EntityLink(alias=row[0], id=player_id, display_name=row[0]))
                                    links = links.model_copy(update={
                                        "players": inherited_players + links.players,
                                        "seasons": links.seasons or ([previous.season_id] if previous.season_id else []),
                                        "phases": links.phases or ([previous.competition_phase] if previous.competition_phase else []),
                                    })
                                    if compare_followup and previous.metric_terms and not re.search(r"efg|有效命中率|得分|助攻|篮板|命中率", question, re.I):
                                        question += " " + " ".join(previous.metric_terms)
                    plan = plan_question(question, links=links)
                    set_attribute(planner_span, "nba.agent.route", plan.route)
                    set_attribute(planner_span, "nba.agent.planner_type", "rules")
                    if plan.route in {"structured", "mixed", "comparison"} and links is not None:
                        for step in plan.steps:
                            if step.capability in {"efg_compare", "knowledge_glossary"}:
                                continue
                            validation_links = links
                            if plan.route == "comparison":
                                validation_links = links.model_copy(update={"players": [item for item in links.players if item.id == step.arguments["entity_id"]]})
                            # These words describe the orchestration's player delta,
                            # not a cross-season intent of the individual eFG call.
                            validation_question = re.sub(r"相差|差值", "", question) if plan.route == "comparison" else question
                            step.arguments = validate_ask_call(AskPlan(action="call", tool=step.capability, arguments=step.arguments), validation_links, validation_question)
                    response = await execute_agent_plan(
                        question, plan, selected_registry, request_id=request_id,
                        deadline_seconds=settings.ask_total_timeout_seconds, tracer=settings.tracer,
                        request_span=request_span, skip_planner_span=True,
                    )
                response = response.model_copy(update={'routing_mode': 'rules', 'planning': {'planner_type': 'rules'}})
                return finish(response, conversation_id, previous, links)
            except AskGroundingError as exc:
                status = exc.status if exc.status in {"clarification_required", "unsupported"} else "incomplete"
                response = AgentResponse(
                    status=status, request_id=request_id, trace_id=trace_id(),
                    route="clarification" if status == "clarification_required" else "unsupported",
                    answer=str(exc), reason_code=exc.code,
                    timings={"total_ms": round((time.monotonic() - started) * 1000, 3)},
                )
                set_attribute(request_span, "nba.agent.status", status)
                set_attribute(request_span, "nba.agent.reason_code", exc.code)
                return finish(response, conversation_id, previous)
            except ValueError:
                logger.warning("agent preflight request_id=%s invalid plan", request_id)
                response = AgentResponse(status="incomplete", request_id=request_id, trace_id=trace_id(), route="unsupported", reason_code="invalid_plan",
                                         timings={"total_ms": round((time.monotonic() - started) * 1000, 3)})
                set_attribute(request_span, "nba.agent.status", "incomplete")
                set_attribute(request_span, "nba.agent.reason_code", "invalid_plan")
                return finish(response, conversation_id, previous)
