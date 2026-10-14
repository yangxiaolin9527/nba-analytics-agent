import asyncio

import pytest


def test_event_sink_assigns_order_and_keeps_safe_metadata():
    from app.execution_events import ExecutionEventSink

    sink = ExecutionEventSink(request_id="req-1", trace_id="trace-1")
    first = sink.emit("generation", "running", "答案生成", metadata={"step_id": "s1"})
    second = sink.emit("generation", "complete", "答案生成完成", elapsed_ms=12.5)

    assert first["seq"] == 1
    assert second["seq"] == 2
    assert second["request_id"] == "req-1"
    assert sink.snapshot() == [first, second]


def test_event_context_is_request_local():
    from app.execution_events import (
        ExecutionEventSink,
        current_execution_sink,
        execution_event_context,
    )

    outer = ExecutionEventSink(request_id="outer")
    inner = ExecutionEventSink(request_id="inner")

    async def run():
        with execution_event_context(outer):
            current_execution_sink.get().emit("planning", "running", "规划")
            with execution_event_context(inner):
                current_execution_sink.get().emit("retrieval", "running", "检索")
            current_execution_sink.get().emit("result", "complete", "完成")

    asyncio.run(run())
    assert [event["request_id"] for event in outer.snapshot()] == ["outer", "outer"]
    assert [event["request_id"] for event in inner.snapshot()] == ["inner"]


def test_event_sink_callback_failure_does_not_break_execution():
    from app.execution_events import ExecutionEventSink

    seen = []

    def callback(event):
        seen.append(event["seq"])
        raise RuntimeError("transport closed")

    sink = ExecutionEventSink(request_id="req-1", callback=callback)
    event = sink.emit("result", "complete", "完成")

    assert event["seq"] == 1
    assert sink.snapshot() == [event]
    assert sink.errors == ["RuntimeError"]
    assert seen == [1]


def test_event_sink_can_rebind_to_authoritative_execution_identity():
    from app.execution_events import ExecutionEventSink

    sink = ExecutionEventSink(request_id="transport-id", trace_id="transport-trace")
    sink.emit("result", "complete", "完成")
    sink.rebind(request_id="execution-id", trace_id="execution-trace")

    event = sink.snapshot()[0]
    assert event["request_id"] == "execution-id"
    assert event["trace_id"] == "execution-trace"


def test_knowledge_runtime_emits_authoritative_stage_transitions():
    from app.execution_events import ExecutionEventSink, execution_event_context
    from scripts.knowledge_runtime import KnowledgeRuntime

    sink = ExecutionEventSink(request_id="req-1")
    runtime = KnowledgeRuntime(deadline=999999.0)
    with execution_event_context(sink):
        runtime.begin("generation")
        runtime.finish("generation")

    assert [(event["stage"], event["status"]) for event in sink.snapshot()] == [
        ("generation", "running"), ("generation", "complete"),
    ]
