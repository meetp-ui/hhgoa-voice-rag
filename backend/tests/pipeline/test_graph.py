from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from app.guardrails.confidence_floor import ConfidenceFloorGuardrail
from app.guardrails.groundedness import GroundednessGuardrail
from app.guardrails.input_filter import InputFilterGuardrail
from app.ingestion.sparse_encoding import SparseVector
from app.pipeline.graph import PipelineGraph
from app.pipeline.nodes.generate import GenerateNode
from app.pipeline.nodes.retrieve import RetrieveNode
from app.pipeline.nodes.transcribe import TranscribeNode
from app.pipeline.state import RetrievedChunk
from app.providers.llm.base import LLMProvider, StructuredAnswer
from app.providers.stt.base import STTProvider, TranscriptionResult
from app.providers.vector_db.base import VectorDBProvider, VectorPoint

CENTROID = [1.0, 0.0]


class FakeEmbeddingClient:
    def __init__(self, *, off_topic: bool = False) -> None:
        self._off_topic = off_topic

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [([0.0, 1.0] if self._off_topic else [1.0, 0.0]) for _ in texts]


class FakeSTTProvider(STTProvider):
    def __init__(self, transcript: str = "दिल्ली भारत की राजधानी है क्या") -> None:
        self._transcript = transcript

    async def transcribe(
        self,
        audio: bytes,
        *,
        language_hint: str | None = None,
        content_type: str = "application/octet-stream",
        filename: str = "audio",
    ) -> TranscriptionResult:
        return TranscriptionResult(transcript=self._transcript, language="hi", confidence=0.95)


class FakeVectorDBProvider(VectorDBProvider):
    def __init__(self, *, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks if chunks is not None else [
            RetrievedChunk(
                chunk_id="c1",
                doc_id="d1",
                parent_id="d1",
                text="दिल्ली भारत की राजधानी है।",
                passage_position=0,
                language="hi",
                score=0.9,
            )
        ]

    async def hybrid_search(
        self,
        *,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]

    async def get_chunks_by_parent_id(self, parent_id: str) -> list[RetrievedChunk]:
        return [c for c in self._chunks if c.parent_id == parent_id]

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True

    async def compute_corpus_centroid(self) -> list[float]:
        return CENTROID


class FakeLLMProvider(LLMProvider):
    def __init__(self, *, grounded: bool = True) -> None:
        self._grounded = grounded
        self.call_count = 0

    async def generate(
        self, *, system_prompt: str, query: str, context: str
    ) -> StructuredAnswer:
        self.call_count += 1
        return StructuredAnswer(
            answer="दिल्ली भारत की राजधानी है।" if self._grounded else "मुझे नहीं पता",
            grounded=self._grounded,
            cited_chunk_ids=["c1"] if self._grounded else [],
            confidence=0.9 if self._grounded else 0.2,
        )


def _sparse_encode_fn(texts: list[str]) -> list[SparseVector]:
    return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


class FakeRerankerClient:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.called = False

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.called = True
        return self.scores[: len(texts)]


def _build_graph(
    *,
    embedding_client: FakeEmbeddingClient,
    vector_db_provider: FakeVectorDBProvider,
    llm_provider: LLMProvider,
    off_topic_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    reranker_client: FakeRerankerClient | None = None,
) -> PipelineGraph:
    return PipelineGraph(
        transcribe_node=TranscribeNode(stt_provider=FakeSTTProvider()),
        input_filter_guardrail=InputFilterGuardrail(
            corpus_centroid=CENTROID,
            distance_threshold=off_topic_threshold,
            min_query_length_chars=3,
        ),
        embedding_client=embedding_client,  # type: ignore[arg-type]
        retrieve_node=RetrieveNode(
            embedding_client=embedding_client,  # type: ignore[arg-type]
            vector_db_provider=vector_db_provider,
            reranker_client=reranker_client,  # type: ignore[arg-type]
            sparse_encode_fn=_sparse_encode_fn,
        ),
        confidence_floor_guardrail=ConfidenceFloorGuardrail(threshold=confidence_threshold),
        generate_node=GenerateNode(llm_provider=llm_provider),
        groundedness_guardrail=GroundednessGuardrail(),
    )


@pytest.mark.asyncio
async def test_happy_path_end_to_end_with_mocked_providers() -> None:
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(),
        llm_provider=FakeLLMProvider(grounded=True),
    )

    result = await graph.run(text="दिल्ली की राजधानी क्या है")

    assert result.status == "ok"
    assert result.answer == "दिल्ली भारत की राजधानी है।"
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "c1"
    assert result.sources[0].text == "दिल्ली भारत की राजधानी है।"
    assert result.guardrail is None
    assert result.timings["total"] > 0


@pytest.mark.asyncio
async def test_off_topic_query_is_rejected_before_retrieval() -> None:
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(off_topic=True),
        vector_db_provider=FakeVectorDBProvider(),
        llm_provider=FakeLLMProvider(grounded=True),
    )

    result = await graph.run(text="कुछ भी असंबंधित सवाल")

    assert result.status == "rejected"
    assert result.guardrail is not None
    assert result.guardrail.stage == "guardrail_input"
    assert result.guardrail.reason_code == "off_topic"
    assert result.answer is None


@pytest.mark.asyncio
async def test_empty_text_is_rejected_by_input_filter() -> None:
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(),
        llm_provider=FakeLLMProvider(grounded=True),
    )

    result = await graph.run(text="ab")

    assert result.status == "rejected"
    assert result.guardrail is not None
    assert result.guardrail.reason_code == "query_too_short"


@pytest.mark.asyncio
async def test_low_confidence_retrieval_short_circuits_before_llm_call() -> None:
    llm_provider = FakeLLMProvider(grounded=True)

    low_score_chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        parent_id="d1",
        text="असंबंधित पाठ",
        passage_position=0,
        language="hi",
        score=0.3,
    )
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(chunks=[low_score_chunk]),
        llm_provider=llm_provider,
    )

    result = await graph.run(text="दिल्ली की राजधानी क्या है")

    assert result.status == "rejected"
    assert result.guardrail is not None
    assert result.guardrail.stage == "guardrail_confidence_floor"
    assert result.guardrail.reason_code == "low_confidence"
    assert result.guardrail.score == 0.3
    assert llm_provider.call_count == 0


@pytest.mark.asyncio
async def test_ungrounded_llm_response_is_caught_by_output_guardrail() -> None:
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(),
        llm_provider=FakeLLMProvider(grounded=False),
    )

    result = await graph.run(text="दिल्ली की राजधानी क्या है")

    assert result.status == "rejected"
    assert result.guardrail is not None
    assert result.guardrail.stage == "guardrail_output"
    assert result.guardrail.reason_code == "not_grounded"
    assert result.guardrail.score == 0.2


@pytest.mark.asyncio
async def test_use_reranker_override_threads_through_the_full_graph() -> None:
    reranker = FakeRerankerClient(scores=[0.95])
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(),  # default chunk has score=0.9, not ambiguous
        llm_provider=FakeLLMProvider(grounded=True),
        reranker_client=reranker,
    )

    result = await graph.run(text="दिल्ली की राजधानी क्या है", use_reranker=True)

    assert reranker.called is True
    assert result.reranked is True


@pytest.mark.asyncio
async def test_transcribe_path_used_when_audio_given_instead_of_text() -> None:
    graph = _build_graph(
        embedding_client=FakeEmbeddingClient(),
        vector_db_provider=FakeVectorDBProvider(),
        llm_provider=FakeLLMProvider(grounded=True),
    )

    result = await graph.run(audio=b"fake-audio-bytes")

    assert result.status == "ok"
    assert result.transcript == "दिल्ली भारत की राजधानी है क्या"
    assert result.query_language == "hi"
