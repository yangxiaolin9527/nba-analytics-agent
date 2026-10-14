"""Application runtime configuration assembled at startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.model_adapter import TextToSQLModel
from scripts.provider_config import ProviderConfig
from scripts.query_context import ContextMode, MODES
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.tracing import configured_tracer
from app.conversation import ConversationStore


@dataclass(frozen=True)
class AppContext:
    model: TextToSQLModel | None
    database: Path | None
    context_mode: ContextMode
    execution_mode: str
    route: str
    tracer: Any
    quality_matrix_path: Path
    ask_max_tool_calls: int
    ask_model_timeout_seconds: float
    ask_tool_timeout_seconds: float
    ask_total_timeout_seconds: float
    provider_config: ProviderConfig
    routing_mode: str | None = None
    conversations: ConversationStore | None = None
    agent_max_tool_calls: int = 4
    agent_unified_loop_enabled: bool = False
    agent_loop_decider: Any = None


def build_context(
    *,
    model: TextToSQLModel | None = None,
    db_path: Path | None = None,
    context_mode: ContextMode | None = None,
    execution_mode: str | None = None,
    route: str | None = None,
    tracer: Any = None,
    quality_matrix_path: Path = CAPABILITY_MATRIX,
    ask_max_tool_calls: int = 1,
    ask_model_timeout_seconds: float = 7.0,
    ask_tool_timeout_seconds: float = 2.0,
    ask_total_timeout_seconds: float = 10.0,
    provider_config: ProviderConfig | None = None,
    routing_mode: str | None = None,
    agent_max_tool_calls: int = 4,
    agent_unified_loop_enabled: bool | None = None,
    agent_loop_decider: Any = None,
) -> AppContext:
    selected_execution_mode = execution_mode or "direct"
    if routing_mode not in {None, 'model', 'rules'}:
        raise ValueError('routing_mode must be model or rules')
    if not 0 <= agent_max_tool_calls <= 4:
        raise ValueError('agent_max_tool_calls must be between 0 and 4')
    selected_context_mode = context_mode or "schema_value_retrieval"
    if selected_context_mode not in MODES:
        raise ValueError(f"unknown context mode: {selected_context_mode}")
    if selected_execution_mode not in {"plan", "direct"}:
        raise ValueError(f"unknown execution mode: {selected_execution_mode}")
    if ask_max_tool_calls < 0 or min(ask_model_timeout_seconds, ask_tool_timeout_seconds, ask_total_timeout_seconds) <= 0:
        raise ValueError("ask budgets must be non-negative and timeouts positive")
    selected_provider = provider_config or ProviderConfig.empty()
    configured_path = selected_provider.database_path
    database = db_path if db_path is not None else Path(configured_path) if configured_path else None
    selected_route = route or f"{selected_execution_mode}_{selected_context_mode.removesuffix('_retrieval')}"
    return AppContext(
        model=model,
        database=database,
        context_mode=selected_context_mode,
        execution_mode=selected_execution_mode,
        route=selected_route,
        tracer=tracer if tracer is not None else configured_tracer(selected_provider),
        quality_matrix_path=quality_matrix_path,
        ask_max_tool_calls=ask_max_tool_calls,
        ask_model_timeout_seconds=ask_model_timeout_seconds,
        ask_tool_timeout_seconds=ask_tool_timeout_seconds,
        ask_total_timeout_seconds=ask_total_timeout_seconds,
        provider_config=selected_provider,
        routing_mode=routing_mode,
        conversations=ConversationStore(),
        agent_max_tool_calls=agent_max_tool_calls,
        agent_unified_loop_enabled=(selected_provider.unified_agent_loop_enabled
                                    if agent_unified_loop_enabled is None
                                    else agent_unified_loop_enabled),
        agent_loop_decider=agent_loop_decider,
    )
