"""Typed pipeline state threaded through every node: transcribe, guardrail_input,
retrieve, rerank, assemble_context, generate, guardrail_output, respond.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """A single chunk as returned by retrieval/rerank, with the metadata every
    chunk carries per the chunking module contract (doc_id, position, language,
    source_query_id)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    passage_position: int
    language: str
    source_query_id: str | None = None
    score: float


class GuardrailRejectionRecord(BaseModel):
    """One structured guardrail rejection, logged with a reason code (not free text)."""

    model_config = ConfigDict(frozen=True)

    stage: str
    reason_code: str
    message: str
    score: float | None = None
    """top1_score for guardrail_confidence_floor, LLM confidence for
    guardrail_output. None for guardrail_input, which fires before any
    score exists."""


class PipelineState(BaseModel):
    """State threaded through pipeline nodes. Each node returns a new
    PipelineState rather than mutating shared state."""

    request_id: str

    # transcribe
    audio_input: bytes | None = None
    transcript: str | None = None
    query_language: str | None = None

    # retrieve / rerank
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: list[RetrievedChunk] = Field(default_factory=list)

    # assemble_context / generate
    context: str | None = None
    answer: str | None = None
    grounded: bool | None = None
    confidence: float | None = None
    citations: list[str] = Field(default_factory=list)

    # guardrails (populated by any node that rejects)
    guardrail_rejections: list[GuardrailRejectionRecord] = Field(default_factory=list)

    # error / observability
    error: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
