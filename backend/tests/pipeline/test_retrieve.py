from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from app.ingestion.sparse_encoding import SparseVector
from app.pipeline.nodes.retrieve import RetrieveNode
from app.pipeline.state import RetrievedChunk
from app.providers.vector_db.base import VectorDBProvider, VectorPoint


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "1_0",
    parent_id: str | None = None,
    text: str = "पाठ",
    passage_position: int = 0,
    score: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        parent_id=parent_id or doc_id,
        text=text,
        passage_position=passage_position,
        language="hi",
        source_query_id="1",
        score=score,
    )


class FakeEmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorDBProvider(VectorDBProvider):
    def __init__(
        self,
        search_results: list[RetrievedChunk],
        siblings_by_parent: dict[str, list[RetrievedChunk]] | None = None,
    ) -> None:
        self.search_results = search_results
        self.siblings_by_parent = siblings_by_parent or {}
        self.last_ef_search: int | None = None

    async def hybrid_search(
        self,
        *,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievedChunk]:
        self.last_ef_search = ef_search
        return self.search_results[:top_k]

    async def get_chunks_by_parent_id(self, parent_id: str) -> list[RetrievedChunk]:
        return self.siblings_by_parent.get(parent_id, [])

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True

    async def compute_corpus_centroid(self) -> list[float]:
        return [1.0, 0.0]


class FakeRerankerClient:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.called = False
        self.last_texts: list[str] | None = None

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.called = True
        self.last_texts = texts
        return self.scores[: len(texts)]


def _sparse_encode_fn(texts: list[str]) -> list[SparseVector]:
    return [SparseVector(indices=[1, 2], values=[0.5, 0.3]) for _ in texts]


@pytest.mark.asyncio
async def test_confident_top1_skips_rerank() -> None:
    results = [_chunk("a", score=0.9), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)
    reranker = FakeRerankerClient(scores=[0.1, 0.9])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=reranker,  # type: ignore[arg-type]
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert reranker.called is False
    assert result.reranked is False
    assert result.top1_score == 0.9
    assert [c.chunk_id for c in result.chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_ambiguous_top1_triggers_rerank_and_resorts() -> None:
    # tied top1/top2 at 0.5, ambiguous per AMBIGUOUS_SCORE_THRESHOLD
    results = [_chunk("a", score=0.5), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)
    # reranker flips the order: "b" should now score higher than "a"
    reranker = FakeRerankerClient(scores=[0.2, 0.95])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=reranker,  # type: ignore[arg-type]
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert reranker.called is True
    assert result.reranked is True
    assert [c.chunk_id for c in result.chunks] == ["b", "a"]
    assert result.top1_score == 0.95


@pytest.mark.asyncio
async def test_no_reranker_configured_never_reranks_even_if_ambiguous() -> None:
    results = [_chunk("a", score=0.5), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert result.reranked is False
    assert [c.chunk_id for c in result.chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_use_reranker_true_forces_rerank_despite_confident_top1() -> None:
    results = [_chunk("a", score=0.9), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)
    reranker = FakeRerankerClient(scores=[0.1, 0.95])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=reranker,  # type: ignore[arg-type]
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न", use_reranker=True)

    assert reranker.called is True
    assert result.reranked is True
    assert [c.chunk_id for c in result.chunks] == ["b", "a"]


@pytest.mark.asyncio
async def test_use_reranker_false_skips_rerank_despite_ambiguous_top1() -> None:
    results = [_chunk("a", score=0.5), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)
    reranker = FakeRerankerClient(scores=[0.2, 0.95])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=reranker,  # type: ignore[arg-type]
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न", use_reranker=False)

    assert reranker.called is False
    assert result.reranked is False
    assert [c.chunk_id for c in result.chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_use_reranker_true_is_a_noop_without_a_configured_client() -> None:
    results = [_chunk("a", score=0.9), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न", use_reranker=True)

    assert result.reranked is False


@pytest.mark.asyncio
async def test_default_rerank_enabled_false_disables_the_ambiguous_band_heuristic() -> None:
    # use_reranker=None (server default) with default_rerank_enabled=False
    # should behave exactly like no reranker_client being configured at all,
    # even though a real client is present and the score is ambiguous.
    results = [_chunk("a", score=0.5), _chunk("b", doc_id="2_0", score=0.5)]
    provider = FakeVectorDBProvider(results)
    reranker = FakeRerankerClient(scores=[0.2, 0.95])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=reranker,  # type: ignore[arg-type]
        sparse_encode_fn=_sparse_encode_fn,
        default_rerank_enabled=False,
    )

    result = await node.retrieve("प्रश्न")

    assert reranker.called is False
    assert result.reranked is False


@pytest.mark.asyncio
async def test_self_parented_chunk_is_not_expanded() -> None:
    chunk = _chunk("a", doc_id="1_0", parent_id="1_0", text="मूल पाठ", score=0.9)
    provider = FakeVectorDBProvider(
        [chunk], siblings_by_parent={"1_0": [chunk]}  # only itself, no expansion
    )

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert result.chunks[0].text == "मूल पाठ"


@pytest.mark.asyncio
async def test_split_passage_chunk_is_expanded_to_reassembled_siblings() -> None:
    chunk = _chunk("a_1", doc_id="1_0", parent_id="1_0", text="दूसरा भाग", passage_position=1, score=0.9)
    sibling = _chunk("a_0", doc_id="1_0", parent_id="1_0", text="पहला भाग", passage_position=0, score=0.0)
    provider = FakeVectorDBProvider(
        [chunk], siblings_by_parent={"1_0": [chunk, sibling]}
    )

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert result.chunks[0].text == "पहला भाग दूसरा भाग"


@pytest.mark.asyncio
async def test_empty_search_results_returns_empty_result_without_crashing() -> None:
    provider = FakeVectorDBProvider([])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    result = await node.retrieve("प्रश्न")

    assert result.chunks == []
    assert result.top1_score is None
    assert result.reranked is False


@pytest.mark.asyncio
async def test_ef_search_is_passed_through_to_the_provider() -> None:
    provider = FakeVectorDBProvider([_chunk("a", score=0.9)])

    node = RetrieveNode(
        embedding_client=FakeEmbeddingClient(),  # type: ignore[arg-type]
        vector_db_provider=provider,
        reranker_client=None,
        sparse_encode_fn=_sparse_encode_fn,
    )

    await node.retrieve("प्रश्न", ef_search=128)

    assert provider.last_ef_search == 128
