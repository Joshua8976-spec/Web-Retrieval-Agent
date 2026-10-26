import pytest
from src.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.agent_max_pages == 10
    assert settings.agent_timeout_seconds == 120
    assert settings.agent_max_retries == 3
    assert settings.bm25_top_links == 25
    assert settings.llm_temperature == 0.0


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key-12345")
    monkeypatch.setenv("AGENT_MAX_PAGES", "15")

    settings = Settings()
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_model == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert settings.llm_api_key == "secret-test-key-12345"
    assert settings.agent_max_pages == 15

    # Verify secret masking
    safe_data = settings.safe_repr()
    assert "secret-test-key-12345" not in safe_data["llm_api_key"]
    assert safe_data["llm_api_key"].startswith("sec")
    assert "***" in safe_data["llm_api_key"]
