"""Optional local Phoenix tracing for reviewable query execution."""

from __future__ import annotations

import json
import hashlib
import logging
from contextlib import contextmanager
from functools import lru_cache
from contextvars import ContextVar
from typing import Any, Iterator

from scripts.provider_config import ProviderConfig, load_provider_config


logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = "pto-v1"
_diagnostic_capture: ContextVar[bool] = ContextVar("phoenix_diagnostic_capture", default=False)


class _NoopSpan:
    def set_attribute(self, key: str, value: object) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_input(self, value: object) -> None:
        pass

    def set_output(self, value: object) -> None:
        pass

    def set_attributes(self, attributes: dict[str, object]) -> None:
        pass


def diagnostic_capture_enabled() -> bool:
    return _diagnostic_capture.get()


@contextmanager
def diagnostic_capture(enabled: bool = True) -> Iterator[None]:
    token = _diagnostic_capture.set(enabled)
    try:
        yield
    finally:
        _diagnostic_capture.reset(token)


def text_metadata(value: object, *, max_length: int = 8000) -> dict[str, object]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result: dict[str, object] = {
        "chars": len(text),
        "sha256": digest,
        "truncated": len(text) > max_length,
    }
    if diagnostic_capture_enabled():
        result["value"] = text
    else:
        result["preview"] = text[:max_length]
    return result


def set_text(span_handle, key: str, value: object, *, max_length: int = 8000) -> None:
    metadata = text_metadata(value, max_length=max_length)
    for suffix, item in metadata.items():
        set_attribute(span_handle, f"{key}.{suffix}", item, max_length=max_length)
    if diagnostic_capture_enabled():
        set_attribute(span_handle, key, metadata["value"], max_length=max_length * 8)


def mark_success(span_handle) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode
        span_handle.set_status(Status(StatusCode.OK))
    except Exception:
        pass


def mark_error(span_handle, exc: BaseException, *, reason_code: str | None = None) -> None:
    set_attribute(span_handle, "error.type", type(exc).__name__)
    if reason_code:
        set_attribute(span_handle, "nba.reason_code", reason_code)
    try:
        from opentelemetry.trace import Status, StatusCode
        span_handle.record_exception(exc)
        span_handle.set_status(Status(StatusCode.ERROR))
    except Exception:
        pass


@lru_cache(maxsize=1)
def _phoenix_tracer(endpoint: str):
    from opentelemetry import trace
    from phoenix.otel import BatchSpanProcessor, PROJECT_NAME, Resource, TracerProvider

    provider = TracerProvider(
        resource=Resource.create({PROJECT_NAME: "nba-analytics-agent"}),
        verbose=False,
    )
    provider.add_span_processor(
        BatchSpanProcessor(endpoint=endpoint, protocol="http/protobuf")
    )
    trace.set_tracer_provider(provider)
    return provider.get_tracer("nba-analytics-agent")


def configured_tracer(config: ProviderConfig | None = None):
    config = config or load_provider_config()
    if not config.phoenix_enabled:
        return None
    try:
        return _phoenix_tracer(config.phoenix_collector_endpoint)
    except Exception as exc:
        logger.warning("Phoenix tracing disabled after setup error: %s", type(exc).__name__)
        return None


def trace_id() -> str | None:
    """Return the active trace ID without making tracing a runtime dependency."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x")
    except Exception:
        pass
    return None


def flush_tracer() -> bool:
    """Flush the configured provider so short-lived evaluations export their spans."""
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        return bool(force_flush(timeout_millis=5000)) if force_flush else False
    except Exception as exc:
        logger.warning("Phoenix trace flush failed: %s", type(exc).__name__)
        return False


def set_attribute(span_handle, key: str, value: object, *, max_length: int = 8000) -> None:
    """Set a bounded attribute while tolerating exporters with strict value types."""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    elif not isinstance(value, (str, bool, int, float)):
        value = str(value)
    if isinstance(value, str):
        value = value[:max_length]
    try:
        span_handle.set_attribute(key, value)
    except Exception:
        pass


@contextmanager
def span(tracer, name: str, kind: str = "CHAIN") -> Iterator:
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span(name) as current:
        current.set_attribute("openinference.span.kind", kind)
        current.set_attribute("nba.trace.schema_version", TRACE_SCHEMA_VERSION)
        yield current
