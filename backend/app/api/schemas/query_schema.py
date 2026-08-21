from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class GuardrailInfo(BaseModel):
    stage: str
    reason_code: str
    message: str
    score: float | None = None


class SourceInfo(BaseModel):
    chunk_id: str
    text: str


class TimingsInfo(BaseModel):
    transcribe_ms: float | None = None
    retrieve_ms: float | None = None
    generate_ms: float | None = None
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["ok", "rejected", "error"]
    transcript: str | None = None
    query_language: str | None = None
    answer: str | None = None
    confidence: float | None = None
    sources: list[SourceInfo] = []
    reranked: bool | None = None
    guardrail: GuardrailInfo | None = None
    error: str | None = None
    timings: TimingsInfo = TimingsInfo()
