"""Wires the full pipeline in sequence: transcribe, guardrail_input,
retrieve (rerank conditional inside RetrieveNode), guardrail_confidence_floor,
generate, guardrail_output, respond. No separate graph-execution library --
this function's control flow is the graph. Guardrail rejections and
provider errors are both caught here and turned into a typed result;
nothing raises out of `PipelineGraph.run`.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ExternalServiceError, GuardrailRejection
from app.core.logging import get_logger
from app.core.telemetry import bind_request_id
from app.guardrails.confidence_floor import ConfidenceFloorGuardrail
from app.guardrails.groundedness import GroundednessGuardrail
from app.guardrails.input_filter import InputFilterGuardrail
from app.ingestion.embedding import EmbeddingClient
from app.pipeline.nodes.generate import GenerateNode
from app.pipeline.nodes.retrieve import RetrieveNode
from app.pipeline.nodes.transcribe import TranscribeNode
from app.pipeline.state import GuardrailRejectionRecord

_logger = get_logger(__name__)

_TOP_LEVEL_STAGE_KEYS = ("transcribe", "retrieve", "generate")


def _finalize_timings(timings: dict[str, float]) -> dict[str, float]:
    """Adds a "total" summing whichever top-level stages ran. Retrieval's
    sub-stage breakdown is merged into the same dict but excluded from the
    sum -- it would double-count against the "retrieve" entry."""
    timings["total"] = sum(timings.get(k, 0.0) for k in _TOP_LEVEL_STAGE_KEYS)
    return timings


class SourceChunk(BaseModel):
    """A retrieved chunk actually cited in the answer, carried through with
    its text so the API response can show supporting evidence without the
    caller having to re-fetch it from Qdrant."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    status: str  # "ok" | "rejected" | "error"
    transcript: str | None = None
    query_language: str | None = None
    answer: str | None = None
    confidence: float | None = None
    sources: list[SourceChunk] = Field(default_factory=list)
    reranked: bool | None = None
    guardrail: GuardrailRejectionRecord | None = None
    error: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    """Top-level stage durations in ms (transcribe/retrieve/generate/total).
    Retrieval's internal sub-stage breakdown isn't merged in here."""


class PipelineGraph:
    def __init__(
        self,
        *,
        transcribe_node: TranscribeNode,
        input_filter_guardrail: InputFilterGuardrail,
        embedding_client: EmbeddingClient,
        retrieve_node: RetrieveNode,
        confidence_floor_guardrail: ConfidenceFloorGuardrail,
        generate_node: GenerateNode,
        groundedness_guardrail: GroundednessGuardrail,
    ) -> None:
        self._transcribe_node = transcribe_node
        self._input_filter_guardrail = input_filter_guardrail
        self._embedding_client = embedding_client
        self._retrieve_node = retrieve_node
        self._confidence_floor_guardrail = confidence_floor_guardrail
        self._generate_node = generate_node
        self._groundedness_guardrail = groundedness_guardrail

    async def run(
        self,
        *,
        audio: bytes | None = None,
        audio_content_type: str = "application/octet-stream",
        audio_filename: str = "audio",
        text: str | None = None,
        language_hint: str | None = None,
        request_id: str | None = None,
        use_reranker: bool | None = None,
    ) -> PipelineResult:
        """Exactly one of `audio` or `text` must be given. `text` bypasses
        transcription, for testing without a live mic."""
        request_id = bind_request_id(request_id)
        timings: dict[str, float] = {}

        try:
            if text is not None:
                transcript, query_language = text, language_hint or "unknown"
            else:
                assert audio is not None, "run() requires either audio or text"
                t0 = time.perf_counter()
                transcribed = await self._transcribe_node.transcribe(
                    audio,
                    language_hint=language_hint,
                    content_type=audio_content_type,
                    filename=audio_filename,
                )
                timings["transcribe"] = (time.perf_counter() - t0) * 1000
                transcript, query_language = transcribed.transcript, transcribed.language

            try:
                self._input_filter_guardrail.check_transcript(transcript)
            except GuardrailRejection as rejection:
                return self._rejected(
                    request_id, "guardrail_input", rejection, transcript, query_language, timings
                )

            query_embedding = (await self._embedding_client.embed([transcript]))[0]

            try:
                self._input_filter_guardrail.check_off_topic(query_embedding)
            except GuardrailRejection as rejection:
                return self._rejected(
                    request_id, "guardrail_input", rejection, transcript, query_language, timings
                )

            t0 = time.perf_counter()
            retrieval = await self._retrieve_node.retrieve(
                transcript, query_embedding=query_embedding, use_reranker=use_reranker
            )
            timings["retrieve"] = (time.perf_counter() - t0) * 1000
            timings.update(retrieval.stage_timings_ms)

            try:
                self._confidence_floor_guardrail.check(retrieval.top1_score)
            except GuardrailRejection as rejection:
                return self._rejected(
                    request_id,
                    "guardrail_confidence_floor",
                    rejection,
                    transcript,
                    query_language,
                    timings,
                    reranked=retrieval.reranked,
                    score=retrieval.top1_score,
                )

            t0 = time.perf_counter()
            structured_answer = await self._generate_node.generate(
                query=transcript, chunks=retrieval.chunks
            )
            timings["generate"] = (time.perf_counter() - t0) * 1000

            try:
                self._groundedness_guardrail.check(structured_answer)
            except GuardrailRejection as rejection:
                return self._rejected(
                    request_id,
                    "guardrail_output",
                    rejection,
                    transcript,
                    query_language,
                    timings,
                    reranked=retrieval.reranked,
                    score=structured_answer.confidence,
                )

            text_by_chunk_id = {chunk.chunk_id: chunk.text for chunk in retrieval.chunks}
            sources = [
                SourceChunk(chunk_id=cid, text=text_by_chunk_id[cid])
                for cid in structured_answer.cited_chunk_ids
                if cid in text_by_chunk_id
            ]

            return PipelineResult(
                request_id=request_id,
                status="ok",
                transcript=transcript,
                query_language=query_language,
                answer=structured_answer.answer,
                confidence=structured_answer.confidence,
                sources=sources,
                reranked=retrieval.reranked,
                timings=_finalize_timings(timings),
            )

        except ExternalServiceError as exc:
            _logger.warning(
                "pipeline_external_service_error",
                extra={"request_id": request_id, "service": exc.service, "error": exc.message},
            )
            return PipelineResult(
                request_id=request_id,
                status="error",
                error=exc.message,
                timings=_finalize_timings(timings),
            )

    @staticmethod
    def _rejected(
        request_id: str,
        stage: str,
        rejection: GuardrailRejection,
        transcript: str | None,
        query_language: str | None,
        timings: dict[str, float],
        *,
        reranked: bool | None = None,
        score: float | None = None,
    ) -> PipelineResult:
        # Otherwise reason_code only ever exists in the HTTP response body,
        # never in logs -- this makes guardrail activity queryable.
        _logger.info(
            "guardrail_rejected",
            extra={
                "request_id": request_id,
                "stage": stage,
                "reason_code": rejection.reason_code,
            },
        )
        return PipelineResult(
            request_id=request_id,
            status="rejected",
            transcript=transcript,
            query_language=query_language,
            reranked=reranked,
            guardrail=GuardrailRejectionRecord(
                stage=stage,
                reason_code=rejection.reason_code,
                message=rejection.message,
                score=score,
            ),
            timings=_finalize_timings(timings),
        )
