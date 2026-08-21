from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class StructuredAnswer(BaseModel):
    """The single structured-output contract every LLM provider must return."""

    model_config = ConfigDict(frozen=True)

    answer: str
    grounded: bool
    cited_chunk_ids: list[str]
    confidence: float


class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        system_prompt: str,
        query: str,
        context: str,
    ) -> StructuredAnswer:
        """Generate a grounded answer in a single call, returning the structured
        {answer, grounded, cited_chunk_ids, confidence} contract.

        Implementations must wrap the underlying call with a timeout and bounded
        retry with exponential backoff, and raise `app.core.exceptions.ExternalServiceError`
        (never a bare/unexpected exception) on unrecoverable failure.
        """
        raise NotImplementedError
