import json
from urllib.request import Request

import pytest

from scripts.model_adapter import FakeModel, ModelRequest, OpenAICompatibleModel


def test_fake_model_is_deterministic_for_known_case() -> None:
    model = FakeModel({"AG-002": "SELECT home_score, away_score FROM game WHERE game_id = '22300061'"})
    request = ModelRequest(prompt="question", case_id="AG-002")

    first = model.generate(request)
    second = model.generate(request)

    assert first.text == second.text
    assert first.model == "fake"


def test_openai_compatible_payload_and_response_parsing_without_network() -> None:
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "SELECT 1"}}]}).encode()

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1",
        api_key="secret",
        model="demo",
        transport=transport,
    )
    response = model.generate(ModelRequest(prompt="make sql"))

    assert response.text == "SELECT 1"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "demo"


def test_openai_compatible_model_requests_json_object_when_plan_mode() -> None:
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", transport=transport
    )
    model.generate(ModelRequest(prompt="make plan", response_format="json_object"))
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_model_streams_sse_and_records_ttft() -> None:
    captured: dict[str, object] = {}

    def stream_transport(request: Request, timeout: float):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return [
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{\\"ok\\":true}"},"finish_reason":"stop"}],"usage":{"total_tokens":9,"completion_tokens_details":{"reasoning_tokens":2}}}\n\n',
            b"data: [DONE]\n\n",
        ]

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo",
        stream_transport=stream_transport,
    )
    response = model.generate_stream(ModelRequest(prompt="answer", response_format="json_object"))

    assert response.text == '{"ok":true}'
    assert response.streamed is True
    assert response.ttft_ms is not None
    assert response.finish_reason == "stop"
    assert response.reasoning_tokens == 2
    assert captured["body"]["stream"] is True
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_model_request_forwards_max_tokens_only_when_explicit() -> None:
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", transport=transport,
    )
    model.generate(ModelRequest(prompt="answer", max_tokens=120))
    assert captured["body"]["max_tokens"] == 120


def test_openai_compatible_model_forwards_explicit_deepseek_thinking_setting() -> None:
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    model = OpenAICompatibleModel(
        base_url="https://api.deepseek.com/v1", api_key="secret", model="deepseek-flash",
        transport=transport,
    )
    model.generate(ModelRequest(prompt="answer", thinking={"type": "disabled"}))

    assert captured["body"]["thinking"] == {"type": "disabled"}


def test_openai_compatible_model_does_not_send_thinking_to_other_endpoints() -> None:
    captured: dict[str, object] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", transport=transport,
    )
    model.generate(ModelRequest(prompt="answer", thinking={"type": "disabled"}))

    assert "thinking" not in captured["body"]


def test_openai_compatible_timeout_ignores_shell_environment(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "91")
    model = OpenAICompatibleModel(base_url="https://example.test/v1", api_key="secret", model="demo", transport=lambda *_: b'{"choices":[{"message":{"content":"SELECT 1"}}]}')

    from scripts.provider_config import load_provider_config
    assert model.timeout == load_provider_config().model_timeout_seconds


def test_explicit_timeout_overrides_environment_for_bounded_ask(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "91")
    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", timeout=7.0,
        transport=lambda *_: b'{"choices":[{"message":{"content":"{}"}}]}',
    )
    assert model.timeout == 7.0


def test_model_request_timeout_overrides_shared_model_timeout() -> None:
    captured: dict[str, float] = {}

    def transport(request: Request, timeout: float) -> bytes:
        captured["timeout"] = timeout
        return b'{"choices":[{"message":{"content":"{}"}}]}'

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", timeout=30.0, transport=transport,
    )
    model.generate(ModelRequest(prompt="stage", timeout=4.0))

    assert captured["timeout"] == 4.0


def test_openai_compatible_timeout_error_remains_retryable() -> None:
    def transport(request: Request, timeout: float) -> bytes:
        raise TimeoutError("provider deadline")

    model = OpenAICompatibleModel(
        base_url="https://example.test/v1", api_key="secret", model="demo", timeout=30.0, transport=transport,
    )

    with pytest.raises(TimeoutError):
        model.generate(ModelRequest(prompt="stage", timeout=4.0))


@pytest.mark.parametrize(("error", "expected"), [
    ("http-401", "authentication_failed"),
    ("http-429", "rate_limited"),
    ("http-503", "provider_server_error"),
])
def test_model_adapter_classifies_provider_http_failures(error, expected) -> None:
    from urllib.error import HTTPError

    status = int(error.split("-")[1])
    def transport(request: Request, timeout: float) -> bytes:
        raise HTTPError(request.full_url, status, "fixture", {}, None)
    model = OpenAICompatibleModel(base_url="https://example.test/v1", api_key="secret",
                                  model="demo", transport=transport)
    with pytest.raises(RuntimeError, match=expected):
        model.generate(ModelRequest(prompt="stage"))
