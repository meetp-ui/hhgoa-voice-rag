"""Transcribes audio via an injected STTProvider, never a concrete provider
directly (see app/providers/stt/base.py).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.core.telemetry import timed_async
from app.providers.stt.base import STTProvider


class TranscribeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    transcript: str
    language: str


class TranscribeNode:
    def __init__(self, *, stt_provider: STTProvider) -> None:
        self._stt_provider = stt_provider

    @timed_async("transcribe")
    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        content_type: str = "application/octet-stream",
        filename: str = "audio",
    ) -> TranscribeResult:
        result = await self._stt_provider.transcribe(
            audio, language_hint=language_hint, content_type=content_type, filename=filename
        )
        return TranscribeResult(transcript=result.transcript, language=result.language)
