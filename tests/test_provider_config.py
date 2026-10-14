import json
from urllib.error import HTTPError

import pytest

from scripts.provider_config import ProviderConfig
from scripts.provider_health import ProviderHealthError, probe_model, probe_model_json
from scripts.model_adapter import ModelRequest, OpenAICompatibleModel


def test_knowledge_generation_thinking_can_be_configured_in_local_file(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_GENERATION_THINKING=disabled\n')
    assert ProviderConfig.from_file(env).knowledge_generation_thinking == 'disabled'


def test_knowledge_verification_thinking_defaults_disabled_and_is_validated(tmp_path):
    env = tmp_path / '.env.local'
    assert ProviderConfig.from_file(env).knowledge_verification_thinking == 'disabled'
    env.write_text('KNOWLEDGE_VERIFICATION_THINKING=enabled\n')
    assert ProviderConfig.from_file(env).knowledge_verification_thinking == 'enabled'
    env.write_text('KNOWLEDGE_VERIFICATION_THINKING=invalid\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_VERIFICATION_THINKING'):
        ProviderConfig.from_file(env)


def test_generation_prompt_strategy_is_explicit_and_validated(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_GENERATION_PROMPT_STRATEGY=concise\n')
    assert ProviderConfig.from_file(env).knowledge_generation_prompt_strategy == 'concise'
    env.write_text('KNOWLEDGE_GENERATION_PROMPT_STRATEGY=invalid\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_GENERATION_PROMPT_STRATEGY'):
        ProviderConfig.from_file(env)


def test_generation_policy_is_explicit_and_validated(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_GENERATION_POLICY=adaptive\n')
    assert ProviderConfig.from_file(env).knowledge_generation_policy == 'adaptive'
    env.write_text('KNOWLEDGE_GENERATION_POLICY=unknown\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_GENERATION_POLICY'):
        ProviderConfig.from_file(env)


def test_knowledge_candidate_pool_is_explicit_and_bounded(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_CANDIDATE_TOP_K=12\n')
    assert ProviderConfig.from_file(env).knowledge_candidate_top_k == 12
    env.write_text('KNOWLEDGE_CANDIDATE_TOP_K=51\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_CANDIDATE_TOP_K'):
        ProviderConfig.from_file(env)


def test_unified_agent_loop_is_disabled_by_default_and_parses_flag(tmp_path):
    assert ProviderConfig.from_file(tmp_path / '.env.local').unified_agent_loop_enabled is False
    env = tmp_path / '.env.local'
    env.write_text('NBA_AGENT_UNIFIED_LOOP_ENABLED=1\n')
    assert ProviderConfig.from_file(env).unified_agent_loop_enabled is True
    env.write_text('NBA_AGENT_UNIFIED_LOOP_ENABLED=2\n')
    with pytest.raises(ValueError, match='NBA_AGENT_UNIFIED_LOOP_ENABLED'):
        ProviderConfig.from_file(env)




def test_knowledge_generation_thinking_rejects_invalid_local_value(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_GENERATION_THINKING=invalid\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_GENERATION_THINKING'):
        ProviderConfig.from_file(env)


def test_embedding_provider_and_dimension_are_explicit(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('EMBEDDING_PROVIDER=ali_bailian\nEMBEDDING_DIMENSION=1024\nEMBEDDING_NORMALIZE=true\n')
    config = ProviderConfig.from_file(env)
    assert config.embedding_provider == 'ali_bailian'
    assert config.embedding_dimension == 1024
    assert config.embedding_normalize is True


def test_embedding_dimension_rejects_invalid_value(tmp_path):
    env = tmp_path / '.env.local'
    env.write_text('EMBEDDING_DIMENSION=bad\n')
    with pytest.raises(ValueError, match='EMBEDDING_DIMENSION'):
        ProviderConfig.from_file(env)


def test_provider_config_reads_explicit_file_without_shell_fallback(tmp_path, monkeypatch):
    env = tmp_path / ".env.local"
    env.write_text(
        "OPENAI_BASE_URL=https://local.example/v1\n"
        "OPENAI_API_KEY=file-secret\n"
        "OPENAI_MODEL=local-model\n"
        "EMBEDDING_BASE_URL=https://embed.example/v1\n"
        "EMBEDDING_API_KEY=embed-secret\n"
        "EMBEDDING_MODEL=embed-model\n"
        "NBA_DB_PATH=/tmp/local.sqlite\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-shell-secret")
    config = ProviderConfig.from_file(env)

    assert config.require_model() == ("https://local.example/v1", "file-secret", "local-model")
    assert config.database_path == "/tmp/local.sqlite"
    assert config.agent_decision_timeout_seconds == 20.0
    assert "file-secret" not in json.dumps(config.summary())
    assert config.summary()["model_host"] == "local.example"


def test_provider_config_does_not_interpolate_inherited_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WRONG_KEY", "shell-secret")
    env = tmp_path / ".env.local"
    env.write_text(
        "OPENAI_BASE_URL=https://local.example/v1\n"
        "OPENAI_API_KEY=${WRONG_KEY}\n"
        "OPENAI_MODEL=local-model\n",
        encoding="utf-8",
    )

    config = ProviderConfig.from_file(env)
    assert config.model_api_key == "${WRONG_KEY}"


def test_provider_config_rejects_missing_or_blank_model_values(tmp_path):
    missing = ProviderConfig.from_file(tmp_path / "missing.env")
    with pytest.raises(ValueError, match="model provider"):
        missing.require_model()
    blank = tmp_path / ".env.local"
    blank.write_text("OPENAI_BASE_URL=\nOPENAI_API_KEY=\nOPENAI_MODEL=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model provider"):
        ProviderConfig.from_file(blank).require_model()


def test_model_health_classifies_401_without_exposing_response(monkeypatch):
    config = ProviderConfig(
        model_base_url="https://api.example/v1",
        model_api_key="secret",
        model_name="demo",
    )

    def fail(*args, **kwargs):
        raise HTTPError("https://api.example/v1/models", 401, "secret body", {}, None)

    monkeypatch.setattr("scripts.provider_health.urlopen", fail)
    with pytest.raises(ProviderHealthError) as error:
        probe_model(config)
    assert error.value.code == "authentication_failed"
    assert error.value.http_status == 401
    assert "secret" not in str(error.value)


def test_model_adapter_uses_config_key_when_shell_key_differs(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-shell-secret")
    config = ProviderConfig(
        model_base_url="https://local.example/v1",
        model_api_key="file-secret",
        model_name="demo",
    )

    def transport(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        return b'{"choices":[{"message":{"content":"ok"}}]}'

    model = OpenAICompatibleModel(config=config, transport=transport)
    model.generate(ModelRequest("hello"))
    assert captured["authorization"] == "Bearer file-secret"


def test_model_json_health_probe_requires_parseable_action(monkeypatch):
    config = ProviderConfig(
        model_base_url="https://local.example/v1",
        model_api_key="file-secret",
        model_name="demo",
    )

    class Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": '{"action":"finish"}'}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("scripts.provider_health.urlopen", lambda *args, **kwargs: Response())
    result = probe_model_json(config)
    assert result["status"] == "ok"
    assert result["action"] == "finish"


def test_runtime_budgets_and_tracing_read_only_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv('KNOWLEDGE_GENERATION_TIMEOUT_SECONDS', '999')
    monkeypatch.setenv('NBA_PHOENIX_ENABLED', '1')
    env = tmp_path / '.env.local'
    env.write_text('KNOWLEDGE_GENERATION_TIMEOUT_SECONDS=7\nKNOWLEDGE_TOTAL_TIMEOUT_SECONDS=25\n'
                   'KNOWLEDGE_MIN_DENSE_SCORE=0.43\nNBA_PHOENIX_ENABLED=0\n')
    config = ProviderConfig.from_file(env)
    assert config.knowledge_generation_timeout_seconds == 7
    assert config.knowledge_total_timeout_seconds == 25
    assert config.knowledge_min_dense_score == .43
    assert config.phoenix_enabled is False
    assert ProviderConfig.from_file(tmp_path / 'missing').knowledge_generation_timeout_seconds == 10


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '0'])
def test_nonfinite_or_nonpositive_timeout_is_rejected(tmp_path, value):
    env = tmp_path / '.env.local'
    env.write_text(f'KNOWLEDGE_GENERATION_TIMEOUT_SECONDS={value}\n')
    with pytest.raises(ValueError, match='KNOWLEDGE_GENERATION_TIMEOUT_SECONDS'):
        ProviderConfig.from_file(env)


def test_local_embedding_does_not_require_remote_credentials():
    assert ProviderConfig(embedding_provider='local_sentence_transformers',
                          embedding_model='BAAI/bge-m3').embedding_configured


def test_database_and_index_ignore_shell_configuration(monkeypatch):
    from app.config import build_context
    from scripts.knowledge_search import configured_search
    monkeypatch.setenv('NBA_DB_PATH', '/tmp/wrong.sqlite')
    monkeypatch.setenv('NBA_KNOWLEDGE_INDEX_PATH', '/tmp/wrong-index')
    monkeypatch.setenv('NBA_PHOENIX_ENABLED', '1')
    config = ProviderConfig.empty()
    assert build_context(provider_config=config).database is None
    assert build_context(provider_config=config).tracer is None
    assert configured_search(provider_config=config) is None
