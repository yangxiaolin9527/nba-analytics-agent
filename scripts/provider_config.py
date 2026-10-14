"""Repository-local provider configuration with no shell credential fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT / ".env.local"


def _value(values: dict[str, str | None], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _timeout(values: dict[str, str | None], key: str, default: float) -> float:
    raw = _value(values, key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return value


def _bounded_int(values: dict[str, str | None], key: str, default: int,
                 *, minimum: int, maximum: int) -> int:
    raw = _value(values, key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class ProviderConfig:
    """Provider values parsed from one explicit local env file."""

    source: Path | None = None
    model_base_url: str | None = None
    model_api_key: str | None = None
    model_name: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    model_timeout_seconds: float = 30.0
    agent_decision_timeout_seconds: float = 20.0
    embedding_timeout_seconds: float = 30.0
    database_path: str | None = None
    knowledge_index_path: str | None = None
    knowledge_generation_thinking: str = "enabled"
    knowledge_verification_thinking: str = "disabled"
    knowledge_generation_prompt_strategy: str = 'baseline'
    knowledge_generation_policy: str = 'fixed'
    knowledge_model_timeout_seconds: float = 11.0
    knowledge_planner_timeout_seconds: float = 10.0
    knowledge_embedding_timeout_seconds: float = 12.0
    knowledge_retrieval_timeout_seconds: float = 24.0
    knowledge_generation_timeout_seconds: float = 10.0
    knowledge_verification_timeout_seconds: float = 10.0
    knowledge_total_timeout_seconds: float = 30.0
    knowledge_candidate_top_k: int = 10
    knowledge_min_dense_score: float | None = None
    unified_agent_loop_enabled: bool = False
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    embedding_provider: str = "openai_compatible"
    embedding_dimension: int = 1536
    embedding_normalize: bool = False
    embedding_local_device: str = "cuda"
    embedding_local_batch_size: int = 8
    embedding_local_max_length: int = 2048

    @classmethod
    def empty(cls) -> "ProviderConfig":
        return cls()

    @classmethod
    def from_file(cls, path: Path = DEFAULT_ENV_PATH) -> "ProviderConfig":
        path = Path(path).resolve()
        # interpolate=False is required: default interpolation can read inherited
        # environment variables, which would violate the repository policy.
        values = dict(dotenv_values(path, interpolate=False)) if path.is_file() else {}
        thinking = _value(values, "KNOWLEDGE_GENERATION_THINKING") or "enabled"
        if thinking not in {"enabled", "disabled"}:
            raise ValueError("KNOWLEDGE_GENERATION_THINKING must be enabled or disabled")
        verification_thinking = _value(values, "KNOWLEDGE_VERIFICATION_THINKING") or "disabled"
        if verification_thinking not in {"enabled", "disabled"}:
            raise ValueError("KNOWLEDGE_VERIFICATION_THINKING must be enabled or disabled")
        prompt_strategy = _value(values, 'KNOWLEDGE_GENERATION_PROMPT_STRATEGY') or 'baseline'
        if prompt_strategy not in {'baseline', 'concise'}:
            raise ValueError('KNOWLEDGE_GENERATION_PROMPT_STRATEGY must be baseline or concise')
        generation_policy = _value(values, 'KNOWLEDGE_GENERATION_POLICY') or 'fixed'
        if generation_policy not in {'fixed', 'adaptive'}:
            raise ValueError('KNOWLEDGE_GENERATION_POLICY must be fixed or adaptive')
        provider = _value(values, "EMBEDDING_PROVIDER") or "openai_compatible"
        try:
            dimension = int(_value(values, "EMBEDDING_DIMENSION") or "1536")
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer") from exc
        normalize = (_value(values, "EMBEDDING_NORMALIZE") or "false").lower()
        if normalize not in {"true", "false"} or dimension <= 0:
            raise ValueError("EMBEDDING_NORMALIZE must be true or false and dimension positive")
        threshold_raw = _value(values, "KNOWLEDGE_MIN_DENSE_SCORE")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("KNOWLEDGE_MIN_DENSE_SCORE must be between zero and one")
        phoenix = _value(values, "NBA_PHOENIX_ENABLED") or "0"
        if phoenix not in {"0", "1"}:
            raise ValueError("NBA_PHOENIX_ENABLED must be 0 or 1")
        unified_loop = _value(values, "NBA_AGENT_UNIFIED_LOOP_ENABLED") or "0"
        if unified_loop not in {"0", "1"}:
            raise ValueError("NBA_AGENT_UNIFIED_LOOP_ENABLED must be 0 or 1")
        return cls(
            source=path,
            model_base_url=_value(values, "OPENAI_BASE_URL"),
            model_api_key=_value(values, "OPENAI_API_KEY"),
            model_name=_value(values, "OPENAI_MODEL"),
            embedding_base_url=_value(values, "EMBEDDING_BASE_URL"),
            embedding_api_key=_value(values, "EMBEDDING_API_KEY"),
            embedding_model=_value(values, "EMBEDDING_MODEL"),
            model_timeout_seconds=_timeout(values, "MODEL_TIMEOUT_SECONDS", 30.0),
            agent_decision_timeout_seconds=_timeout(values, "AGENT_DECISION_TIMEOUT_SECONDS", 20.0),
            embedding_timeout_seconds=_timeout(values, "EMBEDDING_TIMEOUT_SECONDS", 30.0),
            database_path=_value(values, "NBA_DB_PATH"),
            knowledge_index_path=_value(values, "NBA_KNOWLEDGE_INDEX_PATH"),
            knowledge_generation_thinking=thinking,
            knowledge_verification_thinking=verification_thinking,
            knowledge_generation_prompt_strategy=prompt_strategy,
            knowledge_generation_policy=generation_policy,
            knowledge_model_timeout_seconds=_timeout(values, "KNOWLEDGE_MODEL_TIMEOUT_SECONDS", 11.0),
            knowledge_planner_timeout_seconds=_timeout(values, "KNOWLEDGE_PLANNER_TIMEOUT_SECONDS", 10.0),
            knowledge_embedding_timeout_seconds=_timeout(values, "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", 12.0),
            knowledge_retrieval_timeout_seconds=_timeout(values, "KNOWLEDGE_RETRIEVAL_TIMEOUT_SECONDS", 24.0),
            knowledge_generation_timeout_seconds=_timeout(values, "KNOWLEDGE_GENERATION_TIMEOUT_SECONDS", 10.0),
            knowledge_verification_timeout_seconds=_timeout(values, "KNOWLEDGE_VERIFICATION_TIMEOUT_SECONDS", 10.0),
            knowledge_total_timeout_seconds=_timeout(values, "KNOWLEDGE_TOTAL_TIMEOUT_SECONDS", 30.0),
            knowledge_candidate_top_k=_bounded_int(values, "KNOWLEDGE_CANDIDATE_TOP_K", 10, minimum=1, maximum=50),
            knowledge_min_dense_score=threshold,
            unified_agent_loop_enabled=unified_loop == "1",
            phoenix_enabled=phoenix == "1",
            phoenix_collector_endpoint=_value(values, "PHOENIX_COLLECTOR_ENDPOINT") or "http://127.0.0.1:6006/v1/traces",
            embedding_provider=provider,
            embedding_dimension=dimension,
            embedding_normalize=normalize == "true",
            embedding_local_device=_value(values, "EMBEDDING_LOCAL_DEVICE") or "cuda",
            embedding_local_batch_size=int(_value(values, "EMBEDDING_LOCAL_BATCH_SIZE") or "8"),
            embedding_local_max_length=int(_value(values, "EMBEDDING_LOCAL_MAX_LENGTH") or "2048"),
        )

    def require_model(self) -> tuple[str, str, str]:
        if not all((self.model_base_url, self.model_api_key, self.model_name)):
            raise ValueError("model provider configuration is unavailable")
        return self.model_base_url, self.model_api_key, self.model_name

    def require_embedding(self) -> tuple[str, str, str]:
        if not all((self.embedding_base_url, self.embedding_api_key, self.embedding_model)):
            raise ValueError("embedding provider configuration is unavailable")
        return self.embedding_base_url, self.embedding_api_key, self.embedding_model

    @property
    def model_configured(self) -> bool:
        return all((self.model_base_url, self.model_api_key, self.model_name))

    @property
    def embedding_configured(self) -> bool:
        if self.embedding_provider == "local_sentence_transformers":
            return bool(self.embedding_model)
        return all((self.embedding_base_url, self.embedding_api_key, self.embedding_model))

    def summary(self) -> dict[str, object]:
        """Return safe diagnostics without secrets or file contents."""
        from urllib.parse import urlsplit

        return {
            "source": str(self.source) if self.source else None,
            "model_host": urlsplit(self.model_base_url or "").hostname,
            "model": self.model_name,
            "model_configured": self.model_configured,
            "embedding_host": urlsplit(self.embedding_base_url or "").hostname,
            "embedding_model": self.embedding_model,
            "embedding_configured": self.embedding_configured,
            "knowledge_generation_thinking": self.knowledge_generation_thinking,
            "knowledge_verification_thinking": self.knowledge_verification_thinking,
            "knowledge_generation_prompt_strategy": self.knowledge_generation_prompt_strategy,
            "knowledge_generation_policy": self.knowledge_generation_policy,
            "knowledge_candidate_top_k": self.knowledge_candidate_top_k,
            "embedding_provider": self.embedding_provider,
            "embedding_dimension": self.embedding_dimension,
            "embedding_normalize": self.embedding_normalize,
        }

    def with_embedding_overrides(self, **overrides: object) -> "ProviderConfig":
        values = self.__dict__.copy()
        values.update({key: value for key, value in overrides.items() if value is not None})
        return type(self)(**values)


def load_provider_config(path: Path = DEFAULT_ENV_PATH) -> ProviderConfig:
    return ProviderConfig.from_file(path)
