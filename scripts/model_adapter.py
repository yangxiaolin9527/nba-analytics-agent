"""Offline and OpenAI-compatible model adapters used by the M1.5 gateway."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from scripts.provider_config import ProviderConfig, load_provider_config


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    case_id: str | None = None
    response_format: str | None = None
    timeout: float | None = None
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    stream: bool = False
    max_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    streamed: bool = False
    ttft_ms: float | None = None
    finish_reason: str | None = None
    reasoning_tokens: int | None = None


class TextToSQLModel(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...


class NativeToolModel(Protocol):
    def generate_tools(self, request: ModelRequest) -> ModelResponse: ...


class FakeModel:
    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = dict(responses)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.case_id not in self._responses:
            raise ValueError(f"no fake response for case: {request.case_id}")
        return ModelResponse(self._responses[request.case_id], "fake")

    def generate_tools(self, request: ModelRequest) -> ModelResponse:
        return self.generate(request)


Transport = Callable[[Request, float], bytes]
StreamTransport = Callable[[Request, float], Any]


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        config: ProviderConfig | None = None,
    ) -> None:
        selected = config or load_provider_config()
        self.base_url = (base_url or selected.model_base_url or "").rstrip("/")
        self.api_key = api_key if api_key is not None else selected.model_api_key
        self.model = model or selected.model_name
        self.timeout = timeout if timeout is not None else selected.model_timeout_seconds
        self._transport = transport or self._request
        self._stream_transport = stream_transport or self._stream_request
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL are required in .env.local")

    def _request(self, request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read()

    def _stream_request(self, request: Request, timeout: float) -> Any:
        return urlopen(request, timeout=timeout)

    @staticmethod
    def _payload(request: ModelRequest, model: str) -> dict[str, Any]:
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if request.stream:
            payload["stream"] = True
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _add_common_options(self, payload: dict[str, Any], request: ModelRequest) -> None:
        if request.tools:
            payload["tools"] = request.tools
            if urlsplit(self.base_url).hostname == 'api.deepseek.com':
                payload['thinking'] = {'type': 'disabled'}
        if request.thinking is not None and urlsplit(self.base_url).hostname == 'api.deepseek.com':
            payload['thinking'] = request.thinking
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.stream:
            return self.generate_stream(request)
        payload = self._payload(request, self.model)
        self._add_common_options(payload, request)
        http_request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = self._transport(http_request, request.timeout if request.timeout is not None else self.timeout)
            body = json.loads(raw)
            message = body["choices"][0]["message"]
            text = message.get("content")
            tool_calls = message.get("tool_calls")
            usage = body.get("usage")
        except TimeoutError:
            raise
        except HTTPError as exc:
            if exc.code in {401, 403}:
                code = "authentication_failed"
            elif exc.code == 429:
                code = "rate_limited"
            elif exc.code >= 500:
                code = "provider_server_error"
            else:
                code = "provider_http_error"
            raise RuntimeError(f"model request failed: {code}") from exc
        except (URLError, OSError) as exc:
            raise RuntimeError("model request failed: provider_connection_error") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("model request failed: invalid_provider_response") from exc
        if (not isinstance(text, str) or not text.strip()) and not tool_calls:
            raise RuntimeError("model response contained no text")
        finish_reason = body.get("choices", [{}])[0].get("finish_reason")
        return ModelResponse(text.strip() if isinstance(text, str) else "", str(self.model),
                             tool_calls=tool_calls if isinstance(tool_calls, list) else None,
                             usage=usage if isinstance(usage, dict) else None,
                             finish_reason=finish_reason if isinstance(finish_reason, str) else None,
                             reasoning_tokens=_reasoning_tokens(usage))

    def generate_stream(self, request: ModelRequest) -> ModelResponse:
        """Consume an OpenAI-compatible SSE response for benchmark telemetry.

        Production knowledge generation remains non-streaming because citation
        validation requires one complete JSON document.
        """
        payload_request = ModelRequest(**{**request.__dict__, "stream": True})
        payload = self._payload(payload_request, self.model)
        self._add_common_options(payload, payload_request)
        http_request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        first_token_at = None
        content: list[str] = []
        reasoning: list[str] = []
        finish_reason = None
        usage = None
        source = None
        try:
            source = self._stream_transport(http_request, request.timeout if request.timeout is not None else self.timeout)
            context = source if hasattr(source, "__enter__") else _StreamContext(source)
            with context as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = chunk.get("usage") or usage
                    choice = (chunk.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    reasoning_piece = delta.get("reasoning_content")
                    if isinstance(piece, str) and piece:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        content.append(piece)
                    if isinstance(reasoning_piece, str) and reasoning_piece:
                        reasoning.append(reasoning_piece)
        except TimeoutError:
            raise
        except (HTTPError, URLError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"model stream request failed: {type(exc).__name__}") from exc
        text = "".join(content).strip()
        if not text:
            raise RuntimeError("model stream response contained no text")
        return ModelResponse(
            text, str(self.model), usage=usage if isinstance(usage, dict) else None,
            streamed=True,
            ttft_ms=round((first_token_at - started) * 1000, 3) if first_token_at else None,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            reasoning_tokens=_reasoning_tokens(usage),
        )

    def generate_tools(self, request: ModelRequest) -> ModelResponse:
        return self.generate(request)


class _StreamContext:
    def __init__(self, iterable: Iterable[Any]):
        self.iterable = iterable

    def __enter__(self):
        return self.iterable

    def __exit__(self, *args):
        close = getattr(self.iterable, "close", None)
        if close is not None:
            close()


def _reasoning_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) else None
