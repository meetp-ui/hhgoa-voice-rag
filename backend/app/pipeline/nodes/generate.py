"""Assembles context from retrieved chunks, calls the injected LLMProvider.
Groundedness rejection is the caller's job (app/guardrails/groundedness.py),
not this node's.
"""

from __future__ import annotations

from app.core.telemetry import timed_async
from app.pipeline.state import RetrievedChunk
from app.providers.llm.base import LLMProvider, StructuredAnswer

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. Each context passage is labeled with its chunk_id. "
    "If the context does not contain enough information to answer the "
    "question, say so plainly and set grounded to false. Never invent "
    "information that isn't in the context."
)


def assemble_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[chunk_id: {chunk.chunk_id}]\n{chunk.text}" for chunk in chunks)


class GenerateNode:
    def __init__(self, *, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    @timed_async("generate")
    async def generate(self, *, query: str, chunks: list[RetrievedChunk]) -> StructuredAnswer:
        context = assemble_context(chunks)
        return await self._llm_provider.generate(
            system_prompt=SYSTEM_PROMPT, query=query, context=context
        )
