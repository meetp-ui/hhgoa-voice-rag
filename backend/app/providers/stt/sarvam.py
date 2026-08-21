from __future__ import annotations

import asyncio

import httpx

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.stt.base import STTProvider, TranscriptionResult

_logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


class SarvamSTTProvider(STTProvider):
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
            headers={"api-subscription-key": api_key},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        content_type: str = "application/octet-stream",
        filename: str = "audio",
    ) -> TranscriptionResult:
        data = {"model": self._model, "mode": "transcribe"}
        if language_hint:
            data["language_code"] = language_hint

        base_content_type = content_type.split(";", 1)[0].strip()

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/speech-to-text",
                    data=data,
                    files={"file": (filename, audio, base_content_type)},
                )
                if response.status_code >= 400:
                    _logger.warning(
                        "sarvam_stt_error_response",
                        extra={
                            "attempt": attempt + 1,
                            "status_code": response.status_code,
                            "body": response.text[:2000],
                        },
                    )
                response.raise_for_status()
                payload = response.json()
                return TranscriptionResult(
                    transcript=payload["transcript"],
                    language=payload.get("language_code") or language_hint or "unknown",
                    confidence=payload.get("language_probability"),
                )
            except (httpx.HTTPError, httpx.TimeoutException, KeyError) as exc:
                last_error = exc
                _logger.warning(
                    "sarvam_stt_request_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

        raise ExternalServiceError(
            f"Sarvam STT request failed after {_MAX_RETRIES} attempts: {last_error}",
            service="sarvam_stt",
        )
