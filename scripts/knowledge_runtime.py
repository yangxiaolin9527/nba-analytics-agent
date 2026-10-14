"""Request-local knowledge deadlines and call metadata; no persistent state."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from urllib.parse import urlsplit

from app.execution_events import emit_execution_event


@dataclass
class KnowledgeRuntime:
    deadline: float
    started: float = field(default_factory=perf_counter)
    timings: dict = field(default_factory=dict)
    model_calls: dict = field(default_factory=dict)
    stage_statuses: dict = field(default_factory=lambda: {
        'retrieval': 'skipped', 'generation': 'skipped',
        'citation_validation': 'skipped', 'verification': 'skipped',
    })
    stage: str | None = None
    stage_started: float | None = None

    def begin(self, stage):
        self.stage, self.stage_started = stage, perf_counter()
        self.stage_statuses[stage] = 'running'
        emit_execution_event(stage, 'running', _stage_message(stage))

    def finish(self, stage, status='complete'):
        self.stage_statuses[stage] = status
        elapsed_ms = None
        if self.stage == stage and self.stage_started is not None:
            elapsed_ms = round((perf_counter() - self.stage_started) * 1000, 3)
        emit_execution_event(stage, status, _stage_message(stage, status), elapsed_ms=elapsed_ms)

    def fail_active(self):
        if self.stage and self.stage_statuses.get(self.stage) == 'running':
            self.stage_statuses[self.stage] = 'failed'
            elapsed_ms = None
            if self.stage_started is not None:
                elapsed_ms = round((perf_counter() - self.stage_started) * 1000, 3)
            emit_execution_event(self.stage, 'failed', _stage_message(self.stage, 'failed'), elapsed_ms=elapsed_ms)

    def emit_skipped_stages(self):
        for stage, status in self.stage_statuses.items():
            if status == 'skipped':
                emit_execution_event(stage, 'skipped', _stage_message(stage, 'skipped'))

    def snapshot_timings(self):
        result = dict(self.timings)
        if self.stage and self.stage_started is not None:
            result[f"{self.stage}_ms"] = round((perf_counter() - self.stage_started) * 1000, 3)
        result["total_ms"] = round((perf_counter() - self.started) * 1000, 3)
        return result

current_runtime: ContextVar[KnowledgeRuntime | None] = ContextVar("knowledge_runtime", default=None)
embedding_deadline: ContextVar[float | None] = ContextVar("embedding_deadline", default=None)


def _stage_message(stage: str, status: str | None = None) -> str:
    labels = {
        'planning': '查询规划', 'retrieval': '知识检索', 'evidence': '整理相关证据',
        'generation': '答案生成', 'citation_validation': '引用校验',
        'verification': '语义验证', 'result': '完成',
    }
    label = labels.get(stage, stage)
    if status == 'complete':
        return f'{label}完成'
    if status == 'failed':
        return f'{label}失败'
    if status == 'skipped':
        return f'{label}未执行'
    return label


def stage_budget(timeout: float, *, deadline: float | None = None) -> float:
    runtime = current_runtime.get()
    deadlines = [value for value in (runtime.deadline if runtime else None, deadline) if value is not None]
    if not deadlines:
        return timeout
    remaining = min(deadlines) - perf_counter()
    if remaining <= 0:
        raise TimeoutError("knowledge deadline reached")
    return min(timeout, remaining)


def total_expired() -> bool:
    runtime = current_runtime.get()
    return runtime is not None and perf_counter() >= runtime.deadline


def record_model_call(stage, model, response=None, *, timeout, prompt_chars, status='complete'):
    usage = getattr(response, "usage", None)
    runtime = current_runtime.get()
    metadata = {"model": getattr(response, "model", getattr(model, "model", None)),
                "provider": urlsplit(getattr(model, "base_url", "")).hostname,
                "usage": usage, "timeout_seconds": timeout,
                "input_chars": prompt_chars, "output_chars": len(response.text) if response else None,
                "status": status,
                "streamed": bool(getattr(response, "streamed", False)) if response else False,
                "ttft_ms": getattr(response, "ttft_ms", None) if response else None,
                "finish_reason": getattr(response, "finish_reason", None) if response else None,
                "reasoning_tokens": getattr(response, "reasoning_tokens", None) if response else None}
    if runtime is not None:
        runtime.model_calls[stage] = metadata
    return metadata
