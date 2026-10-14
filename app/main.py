"""Application factory and ASGI entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from scripts.model_adapter import TextToSQLModel
from scripts.query_context import ContextMode
from scripts.relational_baseline import CAPABILITY_MATRIX
from scripts.provider_config import ProviderConfig, load_provider_config

from app.calculation_routes import register_calculation_routes
from app.ask_routes import register_ask_routes
from app.answer_routes import register_answer_routes
from app.config import build_context
from app.context import query_context
from app.query_routes import register_query_routes
from app.window_routes import register_window_routes
from app.schemas import PlanGroundingError
from app.main_grounding import verify_plan_grounding
from app.knowledge_routes import register_knowledge_routes
from app.demo_routes import register_demo_routes
from app.agent_routes import CapabilityRegistry, CapabilityAdapter, register_agent_routes
from app.agent_loop import AgentLoop
from app.agent_decider import ModelDecisionProvider, NativeDecisionProvider


logger = logging.getLogger(__name__)


def create_app(
    *,
    model: TextToSQLModel | None = None,
    db_path=None,
    context_mode: ContextMode | None = None,
    execution_mode: str | None = None,
    route: str | None = None,
    tracer=None,
    quality_matrix_path=CAPABILITY_MATRIX,
    ask_max_tool_calls: int = 1,
    ask_model_timeout_seconds: float = 7.0,
    ask_tool_timeout_seconds: float = 2.0,
    ask_total_timeout_seconds: float = 10.0,
    knowledge_search=None,
    agent_registry: CapabilityRegistry | None = None,
    agent_structured_adapter: CapabilityAdapter | None = None,
    agent_knowledge_adapter: CapabilityAdapter | None = None,
    agent_loop_enabled: bool = False,
    agent_loop_decider=None,
    provider_config: ProviderConfig | None = None,
    routing_mode: str | None = None,
    agent_max_tool_calls: int = 4,
    agent_unified_loop_enabled: bool | None = None,
) -> FastAPI:
    settings = build_context(
        model=model,
        db_path=db_path,
        context_mode=context_mode,
        execution_mode=execution_mode,
        route=route,
        tracer=tracer,
        quality_matrix_path=quality_matrix_path,
        ask_max_tool_calls=ask_max_tool_calls,
        ask_model_timeout_seconds=ask_model_timeout_seconds,
        ask_tool_timeout_seconds=ask_tool_timeout_seconds,
        ask_total_timeout_seconds=ask_total_timeout_seconds,
        provider_config=provider_config,
        routing_mode=routing_mode,
        agent_max_tool_calls=agent_max_tool_calls,
        agent_unified_loop_enabled=agent_unified_loop_enabled,
        agent_loop_decider=agent_loop_decider,
    )
    application = FastAPI(title="NBA Analytics Agent", version="0.1.0")
    register_calculation_routes(application, settings)
    register_window_routes(application, settings)
    register_query_routes(application, settings)
    register_ask_routes(application, settings)
    register_answer_routes(application, settings)
    register_demo_routes(application, settings)
    # A supplied knowledge adapter owns the knowledge capability for this app
    # instance. Avoid opening the process-local Qdrant index just to register
    # unrelated knowledge routes; repeated local clients can leak a semaphore
    # and block asyncio shutdown in tests and short-lived workers.
    if knowledge_search is None and agent_knowledge_adapter is None:
        from scripts.knowledge_search import configured_search
        try:
            knowledge_search = configured_search(provider_config=settings.provider_config)
        except RuntimeError as exc:
            # Local Qdrant indexes are single-process resources. Keep the
            # replay/demo surface alive when a live worker already owns it;
            # /demo/status and knowledge routes expose the missing capability.
            if "already accessed by another instance of Qdrant client" not in str(exc):
                raise
            logger.warning("knowledge index unavailable because its local Qdrant lock is held")
            knowledge_search = None
    register_knowledge_routes(application, knowledge_search, model=settings.model,
                              tracer=settings.tracer, provider_config=settings.provider_config)
    register_agent_routes(
        application,
        settings,
        knowledge_search,
        registry=agent_registry,
        structured_adapter=agent_structured_adapter,
        knowledge_adapter=agent_knowledge_adapter,
        provider_config=settings.provider_config,
    )
    if agent_loop_enabled:
        selected_registry = agent_registry or application.state.agent_registry
        decider = agent_loop_decider
        if decider is None and settings.routing_mode == 'model':
            from scripts.model_adapter import OpenAICompatibleModel
            if settings.model is not None or settings.provider_config.model_configured:
                decider = NativeDecisionProvider(settings.model or OpenAICompatibleModel(config=settings.provider_config), timeout=settings.ask_model_timeout_seconds)
        if decider is None and settings.model is not None:
            names = selected_registry.names() if hasattr(selected_registry, "names") else tuple(getattr(selected_registry, "adapters", {}))
            decider = ModelDecisionProvider(
                settings.model,
                allowed_capabilities=set(names),
                timeout=settings.provider_config.agent_decision_timeout_seconds,
            )
        loop = AgentLoop(selected_registry, decider=decider, tracer=settings.tracer)
        application.state.agent_loop = loop

        from app.agent_loop_routes import register_loop_route
        register_loop_route(application, settings, loop)
    return application


app = create_app(provider_config=load_provider_config(), routing_mode='model',
                 ask_total_timeout_seconds=60.0, ask_model_timeout_seconds=15.0,
                 agent_loop_enabled=True)

# Compatibility for existing tests and scripts importing this helper.
_query_context = query_context
_verify_plan_grounding = verify_plan_grounding
