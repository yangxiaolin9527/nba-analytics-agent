"""Sanitized health probes for the configured OpenAI-compatible providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.provider_config import ProviderConfig


@dataclass(frozen=True)
class ProviderHealthError(ValueError):
    stage: str
    code: str
    http_status: int | None = None

    def __str__(self) -> str:
        suffix = f" HTTP {self.http_status}" if self.http_status is not None else ""
        return f"{self.stage}: {self.code}{suffix}"


def _classify_status(status: int) -> str:
    if status == 401:
        return "authentication_failed"
    if status == 403:
        return "permission_denied"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "provider_error"
    return "http_error"


def _get_json(url: str, key: str, timeout: float) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        raise ProviderHealthError("model", _classify_status(exc.code), exc.code) from exc
    except TimeoutError as exc:
        raise ProviderHealthError("model", "timeout") from exc
    except URLError as exc:
        raise ProviderHealthError("model", "network_error") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise ProviderHealthError("model", "invalid_response") from exc


def probe_model(config: ProviderConfig, *, timeout: float | None = None) -> dict[str, object]:
    base_url, key, model = config.require_model()
    started = monotonic()
    body = _get_json(f"{base_url.rstrip('/')}/models", key, timeout or config.model_timeout_seconds)
    models = body.get("data") if isinstance(body, dict) else None
    ids = {item.get("id") for item in models or [] if isinstance(item, dict)}
    if model not in ids:
        raise ProviderHealthError("model", "model_unavailable")
    return {"stage": "model", "status": "ok", "model": model,
            "elapsed_ms": round((monotonic() - started) * 1000, 3)}


def probe_model_json(config: ProviderConfig, *, timeout: float | None = None) -> dict[str, object]:
    """Verify the JSON response format used by the loop decision provider."""
    base_url, key, model = config.require_model()
    started = monotonic()
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": 'Return a JSON object and exactly {"action":"finish"}.'}],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout or config.model_timeout_seconds) as response:
            body = json.loads(response.read())
    except HTTPError as exc:
        code = _classify_status(exc.code)
        if exc.code == 400:
            code = "json_incompatible"
        raise ProviderHealthError("model_json", code, exc.code) from exc
    except TimeoutError as exc:
        raise ProviderHealthError("model_json", "timeout") from exc
    except URLError as exc:
        raise ProviderHealthError("model_json", "network_error") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise ProviderHealthError("model_json", "invalid_response") from exc
    try:
        content = body["choices"][0]["message"]["content"]
        payload = json.loads(content)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProviderHealthError("model_json", "invalid_response") from exc
    if not isinstance(payload, dict) or payload.get("action") != "finish":
        raise ProviderHealthError("model_json", "invalid_response")
    return {"stage": "model_json", "status": "ok", "model": model,
            "action": "finish", "elapsed_ms": round((monotonic() - started) * 1000, 3)}


def probe_embedding(config: ProviderConfig, *, timeout: float | None = None) -> dict[str, object]:
    base_url, key, model = config.require_embedding()
    started = monotonic()
    request = Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps({"model": model, "input": ["health check"]}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout or config.embedding_timeout_seconds) as response:
            body = json.loads(response.read())
    except HTTPError as exc:
        raise ProviderHealthError("embedding", _classify_status(exc.code), exc.code) from exc
    except TimeoutError as exc:
        raise ProviderHealthError("embedding", "timeout") from exc
    except URLError as exc:
        raise ProviderHealthError("embedding", "network_error") from exc
    except (OSError, ValueError, TypeError) as exc:
        raise ProviderHealthError("embedding", "invalid_response") from exc
    data = body.get("data") if isinstance(body, dict) else None
    vector = data[0].get("embedding") if data and isinstance(data[0], dict) else None
    if not isinstance(vector, list) or not vector:
        raise ProviderHealthError("embedding", "invalid_response")
    return {"stage": "embedding", "status": "ok", "model": model,
            "dimension": len(vector), "elapsed_ms": round((monotonic() - started) * 1000, 3)}
