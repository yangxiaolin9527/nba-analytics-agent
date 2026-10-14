"""POST SSE transport for request-local execution events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from app.execution_events import ExecutionEventSink, execution_event_context


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _frame(event: str, data: Any) -> bytes:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(_jsonable(data), ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def execution_sse(execute: Callable[[ExecutionEventSink], Awaitable[Any]], *, request_id: str) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    def callback(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("stage", event))

    async def runner() -> None:
        sink = ExecutionEventSink(request_id=request_id, callback=callback)
        try:
            with execution_event_context(sink):
                result = await execute(sink)
            if hasattr(result, "request_id"):
                sink.rebind(request_id=result.request_id, trace_id=getattr(result, "trace_id", None))
                result = result.model_copy(update={"execution_events": sink.snapshot()})
            elif isinstance(result, dict) and result.get("request_id"):
                sink.rebind(request_id=result["request_id"], trace_id=result.get("trace_id"))
                result["execution_events"] = sink.snapshot()
            # Flush callbacks scheduled from provider worker threads before the
            # terminal result enters the same FIFO queue.
            await asyncio.sleep(0)
            await queue.put(("result", _jsonable(result)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("result", {"status": "failed",
                                         "reason_code": getattr(exc, "reason_code", "stream_execution_error"),
                                         "error_stage": getattr(exc, "stage", "transport"),
                                         "timings": getattr(exc, "timings", {}),
                                         "stage_statuses": getattr(exc, "stage_statuses", {})}))
        finally:
            await queue.put(("done", {"request_id": request_id}))

    task = asyncio.create_task(runner())

    async def body() -> AsyncIterator[bytes]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield _frame(event, data)
                if event == "done":
                    break
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(body(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
