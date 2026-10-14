"""Request-local execution events shared by JSON and SSE transports."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


EventCallback = Callable[[dict[str, Any]], None]

current_execution_sink: ContextVar["ExecutionEventSink | None"] = ContextVar(
    "current_execution_sink", default=None,
)

_PRIVATE_KEY_PARTS = ("prompt", "raw", "api_key", "secret", "password", "token", "excerpt", "source_text")
_MAX_STRING_LENGTH = 240


def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return value[:_MAX_STRING_LENGTH]
        return value
    if isinstance(value, dict):
        return {
            str(key): _safe_value(nested)
            for key, nested in value.items()
            if not any(part in str(key).lower() for part in _PRIVATE_KEY_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:20]]
    return str(value)[:_MAX_STRING_LENGTH]


class ExecutionEventSink:
    """Synchronous, in-memory event ledger with an optional transport callback."""

    def __init__(self, *, request_id: str, trace_id: str | None = None,
                 callback: EventCallback | None = None) -> None:
        self.request_id = request_id
        self.trace_id = trace_id
        self.callback = callback
        self._events: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def emit(self, stage: str, status: str, message: str, *,
              elapsed_ms: float | None = None,
              metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "seq": len(self._events) + 1,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "stage": stage,
            "status": status,
            "message": message,
            "elapsed_ms": elapsed_ms,
            "metadata": _safe_value(metadata or {}),
        }
        self._events.append(event)
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception as exc:  # transport failures must not alter execution
                self.errors.append(type(exc).__name__)
        return event

    def snapshot(self) -> list[dict[str, Any]]:
        return [dict(event, metadata=dict(event["metadata"])) for event in self._events]

    def rebind(self, *, request_id: str | None = None, trace_id: str | None = None) -> None:
        if request_id is not None:
            self.request_id = request_id
        if trace_id is not None:
            self.trace_id = trace_id
        for event in self._events:
            event["request_id"] = self.request_id
            event["trace_id"] = self.trace_id


@contextmanager
def execution_event_context(sink: ExecutionEventSink) -> Iterator[ExecutionEventSink]:
    token = current_execution_sink.set(sink)
    try:
        yield sink
    finally:
        current_execution_sink.reset(token)


def emit_execution_event(stage: str, status: str, message: str, *,
                         elapsed_ms: float | None = None,
                         metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    sink = current_execution_sink.get()
    if sink is None:
        return None
    return sink.emit(stage, status, message, elapsed_ms=elapsed_ms, metadata=metadata)
