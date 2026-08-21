"""Abstract STT provider contract. Nodes depend on this interface, never a
concrete provider, so the provider can be swapped without touching pipeline
code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    transcript: str
    language: str
    confidence: float | None = None


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        content_type: str = "application/octet-stream",
        filename: str = "audio",
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes to text.

        Pass the real `content_type`/`filename` when known -- some STT
        vendors infer the codec from these, and a generic default caused a
        real rejection once. Implementations must retry with backoff and
        raise `ExternalServiceError`, never a bare exception.
        """
        raise NotImplementedError
