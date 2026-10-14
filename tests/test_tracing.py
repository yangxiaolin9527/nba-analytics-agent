from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_configured_tracer_registers_once_and_exports_spans() -> None:
    script = r'''
import phoenix.otel as phoenix_otel
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

exported_spans = []
processor_options = []

class RecordingExporter(SpanExporter):
    def export(self, spans):
        exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

def processor_factory(*, endpoint=None, protocol=None):
    processor_options.append((endpoint, protocol))
    return SimpleSpanProcessor(RecordingExporter())

phoenix_otel.BatchSpanProcessor = processor_factory

from scripts.tracing import configured_tracer

from scripts.provider_config import ProviderConfig
config = ProviderConfig(phoenix_enabled=True)
tracer = configured_tracer(config)
assert tracer is not None, "tracing should initialize"
provider = trace.get_tracer_provider()
assert provider.resource.attributes[phoenix_otel.PROJECT_NAME] == "nba-analytics-agent"

with tracer.start_as_current_span("regression-smoke"):
    pass

assert configured_tracer(config) is not None, "cached tracer should remain available"
assert trace.get_tracer_provider() is provider, "repeated access must not replace the provider"
assert processor_options == [
    ("http://127.0.0.1:6006/v1/traces", "http/protobuf")
], processor_options
assert [span.name for span in exported_spans] == ["regression-smoke"]
'''
    env = {
        **os.environ,
        "NBA_PHOENIX_ENABLED": "1",
        "PHOENIX_COLLECTOR_ENDPOINT": "http://127.0.0.1:6006/v1/traces",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Overriding of current TracerProvider" not in completed.stderr


def test_trace_helpers_bound_text_and_capture_only_when_enabled():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from scripts.tracing import diagnostic_capture, set_text, span, text_metadata

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    value = "x" * 9000
    assert text_metadata(value)["truncated"] is True
    with span(tracer, "bounded", "LLM") as handle:
        set_text(handle, "input.value", value, max_length=100)
        handle.set_attribute("input.mime_type", "text/plain")
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["nba.trace.schema_version"] == "pto-v1"
    assert attributes["input.value.chars"] == 9000
    assert "input.value" not in attributes
    assert len(attributes["input.value.preview"]) == 100

    exporter.clear()
    with diagnostic_capture():
        with span(tracer, "captured", "LLM") as handle:
            set_text(handle, "input.value", "secret prompt")
            handle.set_attribute("input.mime_type", "text/plain")
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["input.value"] == "secret prompt"
    assert attributes["input.mime_type"] == "text/plain"
