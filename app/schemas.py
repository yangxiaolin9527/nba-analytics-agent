"""HTTP request and response contracts for the application layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from scripts.efg_calculation import EFG_FORMULA
from scripts.query_context import ContextMode, QueryLinks
from scripts.query_scope import require_canonical_season


class PlanGroundingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def require_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    routing_mode: Literal['model', 'rules'] | None = None

    @field_validator("question")
    @classmethod
    def require_ask_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question must not be blank")
        return question


class AnswerRequest(AskRequest):
    """Request contract for deterministic user-facing structured answers."""


class Evidence(BaseModel):
    source_name: str
    source_version: str
    snapshot_id: str
    catalog_version: str


class AskResponse(BaseModel):
    status: str
    error_stage: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    resolved_values: QueryLinks | None = None
    evidence: Evidence | None = None
    tool_calls_used: int = 0
    elapsed_ms: float = 0.0
    model_elapsed_ms: float = 0.0
    tool_elapsed_ms: float = 0.0
    conversation_id: str | None = None
    conversation_context_status: Literal['new', 'continued', 'reset'] | None = None
    turn_index: int | None = None
    routing_mode: str | None = None
    planning: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    status: Literal["complete", "partial", "clarification_required", "unsupported", "not_computable", "incomplete"]
    request_id: str = ""
    error_stage: str | None = None
    answer: str
    reason_code: str | None = None
    reason: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    query_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
    tool_calls_used: int = 0
    timings: dict[str, float] = Field(default_factory=dict)
    conversation_id: str | None = None
    conversation_context_status: Literal['new', 'continued', 'reset'] | None = None
    turn_index: int | None = None
    routing_mode: str | None = None
    planning: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    status: str
    result_status: str
    result_reason: str | None
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    reason: str | None
    elapsed_ms: float
    model_elapsed_ms: float
    gateway_elapsed_ms: float
    context_elapsed_ms: float
    context_mode: str
    prompt_chars: int
    resolved_values: QueryLinks
    evidence: Evidence
    execution_mode: str = "direct"
    route: str = "direct_schema_value"
    plan: dict[str, Any] | None = None
    plan_error_code: str | None = None
    plan_error: str | None = None
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
    plan_elapsed_ms: float = 0.0
    compile_elapsed_ms: float = 0.0


class ComparisonRequest(BaseModel):
    subject: Literal["player", "team"]
    entity_id: str
    metric: str
    baseline_season: str
    comparison_season: str
    competition_phase: str

    @field_validator("entity_id", "metric", "competition_phase")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("baseline_season", "comparison_season")
    @classmethod
    def require_season_format(cls, value: str) -> str:
        value = value.strip()
        return require_canonical_season(value)


class ComparisonSource(BaseModel):
    season_id: str
    competition_phase: str
    value: int | float | None
    sample_count: int | None
    source_table: str
    source_sql: str


class ComparisonEvidence(BaseModel):
    source_snapshot: str
    snapshot_id: str
    catalog_version: str
    entity_id: str
    subject: str
    metric: str
    unit: str
    absolute_delta_formula: str
    relative_change_formula: str


class ComparisonResponse(BaseModel):
    status: str
    reason: str | None = None
    reason_code: str | None = None
    baseline: ComparisonSource | None = None
    comparison: ComparisonSource | None = None
    absolute_delta: int | float | None = None
    relative_change_pct: float | None = None
    relative_change_status: str | None = None
    calculation_limitation: str | None = None
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence: ComparisonEvidence | None = None


class EfgRequest(BaseModel):
    entity_id: str
    season_id: str
    competition_phase: str

    @field_validator("entity_id", "competition_phase")
    @classmethod
    def require_non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("season_id")
    @classmethod
    def require_season_format(cls, value: str) -> str:
        value = value.strip()
        return require_canonical_season(value)


class EfgInputsResponse(BaseModel):
    field_goals_made: int | None
    three_pointers_made: int | None
    field_goals_attempted: int | None
    sample_count: int | None


class EfgEvidence(BaseModel):
    source_snapshot: str
    snapshot_id: str
    catalog_version: str
    entity_id: str
    season_id: str
    competition_phase: str
    metric: str = "efg_percentage"
    unit: str = "percent"
    formula: str = EFG_FORMULA
    rounding: str = "full_precision"


class EfgResponse(BaseModel):
    status: str
    reason: str | None = None
    reason_code: str | None = None
    efg_percentage: float | None = None
    inputs: EfgInputsResponse | None = None
    source_table: str | None = None
    source_sql: str | None = None
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
    calculation_limitation: str | None = None
    evidence: EfgEvidence | None = None


class WindowQueryRequest(BaseModel):
    subject: Literal["player", "team"]
    entity_id: str
    season_id: str
    competition_phase: str
    metrics: list[str] = Field(min_length=1)
    opponent_team_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    last_n: int | None = Field(default=None, ge=1, le=100)

    @field_validator("entity_id", "competition_phase")
    @classmethod
    def require_window_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("season_id")
    @classmethod
    def require_window_season(cls, value: str) -> str:
        return require_canonical_season(value.strip())

    @field_validator("metrics")
    @classmethod
    def require_unique_metrics(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(not item.strip() for item in value):
            raise ValueError("metrics must be unique and non-blank")
        return [item.strip() for item in value]

    @field_validator("date_from", "date_to")
    @classmethod
    def require_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from datetime import date

        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("dates must use YYYY-MM-DD format") from exc
        if parsed.isoformat() != value:
            raise ValueError("dates must use YYYY-MM-DD format")
        return value

    @model_validator(mode="after")
    def require_ordered_dates(self) -> "WindowQueryRequest":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class WindowEvidence(BaseModel):
    source_snapshot: str
    snapshot_id: str
    catalog_version: str
    subject: str
    entity_id: str
    season_id: str
    competition_phase: str
    requested_scope: dict[str, Any]
    selection_rule: str = "filter_then_date_game_order_then_last_n"


class WindowQueryResponse(BaseModel):
    status: str
    reason: str | None = None
    reason_code: str | None = None
    values: dict[str, int | float | None] = Field(default_factory=dict)
    actual_sample_count: int = 0
    available_sample_count: int = 0
    selected_game_ids: list[str] = Field(default_factory=list)
    actual_date_from: str | None = None
    actual_date_to: str | None = None
    source_table: str | None = None
    source_sql: str | None = None
    source_parameters: list[str] = Field(default_factory=list)
    quality_status: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence: WindowEvidence | None = None
