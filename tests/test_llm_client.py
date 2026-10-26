import json
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from openai import OpenAIError
from src.llm.client import LLMClient


def test_llm_client_initialization():
    client = LLMClient(
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.2,
        timeout=15.0,
    )
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "test-key"
    assert client.model == "test-model"
    assert client.temperature == 0.2
    assert client.timeout == 15.0


def test_check_connection_success():
    client = LLMClient(api_key="mock-key")
    mock_choice = MagicMock()
    mock_choice.message.content = "PONG"
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(client.client.chat.completions, "create", return_value=mock_response):
        result = client.check_connection()
        assert result["status"] == "connected"
        assert result["response"] == "PONG"
        assert "latency_ms" in result


def test_check_connection_failure():
    client = LLMClient(api_key="mock-key")

    with patch.object(client.client.chat.completions, "create", side_effect=OpenAIError("Connection refused")):
        result = client.check_connection()
        assert result["status"] == "error"
        assert "Connection refused" in result["error"]


@pytest.mark.asyncio
async def test_async_check_connection_success():
    client = LLMClient(api_key="mock-key")
    mock_choice = MagicMock()
    mock_choice.message.content = "PONG"
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(client.async_client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await client.async_check_connection()
        assert result["status"] == "connected"
        assert result["response"] == "PONG"


def test_chat_completion_json_valid():
    client = LLMClient(api_key="mock-key")
    mock_choice = MagicMock()
    mock_choice.message.content = '{"action": "answer", "answer": "Antigravity"}'
    mock_response = MagicMock(choices=[mock_choice])

    with patch.object(client.client.chat.completions, "create", return_value=mock_response):
        result = client.chat_completion_json([{"role": "user", "content": "What is it?"}])
        assert result == {"action": "answer", "answer": "Antigravity"}


def test_chat_completion_json_retry_on_invalid_syntax():
    client = LLMClient(api_key="mock-key")
    bad_choice = MagicMock()
    bad_choice.message.content = "Not JSON text"
    bad_response = MagicMock(choices=[bad_choice])

    good_choice = MagicMock()
    good_choice.message.content = '{"action": "give_up", "reason": "none"}'
    good_response = MagicMock(choices=[good_choice])

    with patch.object(client.client.chat.completions, "create", side_effect=[bad_response, good_response]) as mock_create:
        result = client.chat_completion_json([{"role": "user", "content": "Query"}], retry_on_invalid_json=True)
        assert result == {"action": "give_up", "reason": "none"}
        assert mock_create.call_count == 2
