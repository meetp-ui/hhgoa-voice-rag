from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import ValidationError

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.llm.base import LLMProvider, StructuredAnswer

_logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_TRANSPORT_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0
_MAX_MALFORMED_RETRIES = 1

_RESPONSE_SCHEMA_INSTRUCTIONS = """\
Respond with a single JSON object and nothing else, matching exactly this shape:
{"answer": string, "grounded": boolean, "cited_chunk_ids": [string, ...], "confidence": number between 0 and 1}

- "answer": your answer to the user's question, grounded ONLY in the provided context.
- "grounded": true only if the context actually supports the answer; false if the \
context does not contain enough information to answer.
- "cited_chunk_ids": the chunk_id values (from the context) that support the answer. \
Empty list if not grounded.
- "confidence": your own confidence in the answer, 0 to 1.
"""


class OllamaCloudLLMProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        *,
        system_prompt: str,
        query: str,
        context: str,
    ) -> StructuredAnswer:
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{_RESPONSE_SCHEMA_INSTRUCTIONS}"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        for malformed_attempt in range(_MAX_MALFORMED_RETRIES + 1):
            raw_content = await self._call_with_transport_retry(messages)
            try:
                parsed = json.loads(raw_content)
                return StructuredAnswer(**parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                _logger.warning(
                    "ollama_malformed_response",
                    extra={"attempt": malformed_attempt + 1, "error": str(exc)},
                )
                if malformed_attempt < _MAX_MALFORMED_RETRIES:
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append(
                        {
                            "role": "user",
                            "content": "That response was not valid JSON matching the "
                            "required schema. Respond again with ONLY the JSON object.",
                        }
                    )
                    continue
                raise ExternalServiceError(
                    f"Ollama Cloud returned a malformed structured response after "
                    f"{_MAX_MALFORMED_RETRIES + 1} attempts: {exc}",
                    service="ollama_cloud",
                ) from exc

        # Unreachable -- loop above always returns or raises.
        raise ExternalServiceError("Unreachable", service="ollama_cloud")

    async def _call_with_transport_retry(self, messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for attempt in range(_MAX_TRANSPORT_RETRIES):
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                    },
                )
                response.raise_for_status()
                body = response.json()
                content: str = body["choices"][0]["message"]["content"]
                return content
            except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError) as exc:
                last_error = exc
                _logger.warning(
                    "ollama_transport_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < _MAX_TRANSPORT_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

        raise ExternalServiceError(
            f"Ollama Cloud request failed after {_MAX_TRANSPORT_RETRIES} attempts: {last_error}",
            service="ollama_cloud",
        )
