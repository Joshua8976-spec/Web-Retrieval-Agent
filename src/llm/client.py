import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI, AsyncOpenAI, OpenAIError

from src.config import settings

logger = logging.getLogger(__name__)


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM output, handling <thought> tags, markdown fences, and prose."""
    cleaned = raw_text.strip()
    # 1. Strip thought blocks (common in reasoning models)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL).strip()

    # 2. Extract from markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # 3. Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Outermost JSON object search
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start : brace_end + 1]
        return json.loads(candidate)

    # Re-raise standard JSON decode error if no JSON structure found
    return json.loads(cleaned)


class LLMClient:
    """OpenAI-compatible LLM client supporting Gemini, local vLLM on AMD GPUs, or hosted endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url or settings.llm_base_url
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.timeout = timeout or settings.llm_timeout_seconds

        # Instantiate OpenAI sync and async clients
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def check_connection(self) -> Dict[str, Any]:
        """Test connectivity to the LLM endpoint and return latency and model info.
        
        Never logs API keys or sensitive auth headers.
        """
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a test agent. Respond only with 'PONG'."},
                    {"role": "user", "content": "PING"},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            content = (response.choices[0].message.content or "").strip()
            return {
                "status": "connected",
                "model": self.model,
                "base_url": self.base_url or "default (OpenAI)",
                "latency_ms": round(duration_ms, 2),
                "response": content,
            }
        except OpenAIError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("LLM connection check failed: %s", str(exc))
            return {
                "status": "error",
                "model": self.model,
                "base_url": self.base_url or "default (OpenAI)",
                "latency_ms": round(duration_ms, 2),
                "error": str(exc),
            }

    async def async_check_connection(self) -> Dict[str, Any]:
        """Asynchronously test connectivity to the LLM endpoint."""
        start_time = time.perf_counter()
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a test agent. Respond only with 'PONG'."},
                    {"role": "user", "content": "PING"},
                ],
                max_tokens=10,
                temperature=0.0,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            content = (response.choices[0].message.content or "").strip()
            return {
                "status": "connected",
                "model": self.model,
                "base_url": self.base_url or "default (OpenAI)",
                "latency_ms": round(duration_ms, 2),
                "response": content,
            }
        except OpenAIError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Async LLM connection check failed: %s", str(exc))
            return {
                "status": "error",
                "model": self.model,
                "base_url": self.base_url or "default (OpenAI)",
                "latency_ms": round(duration_ms, 2),
                "error": str(exc),
            }

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        retry_on_invalid_json: bool = True,
    ) -> Dict[str, Any]:
        """Request a chat completion with JSON mode, parsing and retrying once on JSON syntax error."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content or "{}"
            return extract_json_from_text(raw_text)
        except (json.JSONDecodeError, Exception) as json_err:
            if retry_on_invalid_json:
                logger.warning("Invalid JSON from LLM: %s. Retrying once with error feedback.", json_err)
                retry_messages = list(messages) + [
                    {"role": "assistant", "content": raw_text if "raw_text" in locals() else ""},
                    {
                        "role": "user",
                        "content": f"The response was not valid JSON: {json_err}. Please output strictly valid JSON only without commentary.",
                    },
                ]
                retry_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                retry_text = retry_response.choices[0].message.content or "{}"
                return extract_json_from_text(retry_text)
            raise

    async def async_chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        retry_on_invalid_json: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronously request a chat completion with JSON mode."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content or "{}"
            return extract_json_from_text(raw_text)
        except (json.JSONDecodeError, Exception) as json_err:
            if retry_on_invalid_json:
                logger.warning("Invalid JSON from LLM: %s. Retrying once with error feedback.", json_err)
                retry_messages = list(messages) + [
                    {"role": "assistant", "content": raw_text if "raw_text" in locals() else ""},
                    {
                        "role": "user",
                        "content": f"The response was not valid JSON: {json_err}. Please output strictly valid JSON only without commentary.",
                    },
                ]
                retry_response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=retry_messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                retry_text = retry_response.choices[0].message.content or "{}"
                return extract_json_from_text(retry_text)
            raise


# Default module-level client
default_llm_client = LLMClient()


if __name__ == "__main__":
    import sys
    print("Testing LLM endpoint connectivity...")
    status = default_llm_client.check_connection()
    print(json.dumps(status, indent=2))
    if status["status"] == "error":
        print("\nNote: Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL in .env or environment variables.")
        sys.exit(1)
    print("\nConnectivity test succeeded!")
    sys.exit(0)
