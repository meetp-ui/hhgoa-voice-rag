"""Query goes through hybrid search, conditional rerank, and parent-child
expansion, attaching a top1_score confidence signal (rejection is the
confidence-floor guardrail's job, not this module's).

AMBIGUOUS_SCORE_THRESHOLD = 0.5 is picked from a real RRF score distribution
against the live index, not guessed -- it's roughly where top-1 and top-2
are typically tied, i.e. genuinely undifferentiated.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from app.core.telemetry import timed_async
from app.ingestion.embedding import EmbeddingClient
from app.ingestion.sparse_encoding import SparseVector
from app.ingestion.sparse_encoding import encode as _default_sparse_encode
from app.pipeline.reranker import RerankerClient
from app.pipeline.state import RetrievedChunk
from app.providers.vector_db.base import VectorDBProvider

SparseEncodeFn = Callable[[list[str]], list[SparseVector]]

AMBIGUOUS_SCORE_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
DEFAULT_RERANK_CANDIDATES = 5

# From a real ef sweep (16/64/128/256): latency is flat at this collection
# size, but recall isn't -- ef=16 only matches the ef=256 baseline 68% of
# the time, 64/128 plateau at 72%. 128 sits at that plateau.
DEFAULT_EF_SEARCH = 128


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: list[RetrievedChunk]
    scores: list[float] = Field(
        description="Per-chunk confidence scores, same order as `chunks`, for a "
        "later confidence-floor guardrail to threshold on. Not a rejection decision."
    )
    top1_score: float | None
    reranked: bool
    stage_timings_ms: dict[str, float] = Field(
        description="Per-stage wall-clock breakdown, not just a wrapped total."
    )


async def _expand_to_parent(provider: VectorDBProvider, chunk: RetrievedChunk) -> RetrievedChunk:
    siblings = await provider.get_chunks_by_parent_id(chunk.parent_id)
    if len(siblings) <= 1:
        return chunk
    ordered = sorted(siblings, key=lambda c: c.passage_position)
    expanded_text = " ".join(c.text for c in ordered)
    return chunk.model_copy(update={"text": expanded_text, "passage_position": 0})


class RetrieveNode:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_db_provider: VectorDBProvider,
        reranker_client: RerankerClient | None,
        sparse_encode_fn: SparseEncodeFn = _default_sparse_encode,
        ambiguous_score_threshold: float = AMBIGUOUS_SCORE_THRESHOLD,
        rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATES,
        default_rerank_enabled: bool = True,
    ) -> None:
        self._embedding_client = embedding_client
        self._provider = vector_db_provider
        self._reranker_client = reranker_client
        self._sparse_encode_fn = sparse_encode_fn
        self._ambiguous_score_threshold = ambiguous_score_threshold
        self._rerank_candidate_count = rerank_candidate_count
        self._default_rerank_enabled = default_rerank_enabled

    @timed_async("retrieve")
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        ef_search: int | None = DEFAULT_EF_SEARCH,
        query_embedding: list[float] | None = None,
        use_reranker: bool | None = None,
    ) -> RetrievalResult:
        stage_timings_ms: dict[str, float] = {}

        if query_embedding is not None:
            # Caller (off-topic guardrail) already embedded the query -- reuse it.
            dense_vector = query_embedding
            stage_timings_ms["embed"] = 0.0
        else:
            t0 = time.perf_counter()
            dense_vectors = await self._embedding_client.embed([query])
            stage_timings_ms["embed"] = (time.perf_counter() - t0) * 1000
            dense_vector = dense_vectors[0]

        t0 = time.perf_counter()
        sparse_vectors = self._sparse_encode_fn([query])
        stage_timings_ms["sparse_encode"] = (time.perf_counter() - t0) * 1000

        sparse_vector = dict(zip(sparse_vectors[0].indices, sparse_vectors[0].values, strict=True))

        t0 = time.perf_counter()
        results = await self._provider.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            ef_search=ef_search,
        )
        stage_timings_ms["hybrid_search"] = (time.perf_counter() - t0) * 1000

        if not results:
            return RetrievalResult(
                chunks=[], scores=[], top1_score=None, reranked=False, stage_timings_ms=stage_timings_ms
            )

        top1_score = results[0].score
        reranked = False

        # Explicit True/False overrides the ambiguous-band heuristic entirely;
        # None falls back to it (the server default).
        if use_reranker is False:
            should_rerank = False
        elif use_reranker is True:
            should_rerank = self._reranker_client is not None
        else:
            should_rerank = (
                self._reranker_client is not None
                and self._default_rerank_enabled
                and top1_score <= self._ambiguous_score_threshold
            )

        if should_rerank and self._reranker_client is not None:
            t0 = time.perf_counter()
            candidates = results[: self._rerank_candidate_count]
            tail = results[self._rerank_candidate_count :]
            rerank_scores = await self._reranker_client.rerank(
                query, [c.text for c in candidates]
            )
            reranked_candidates = sorted(
                (c.model_copy(update={"score": score}) for c, score in zip(candidates, rerank_scores, strict=True)),
                key=lambda c: c.score,
                reverse=True,
            )
            results = [*reranked_candidates, *tail]
            reranked = True
            stage_timings_ms["rerank"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        expanded = await asyncio.gather(
            *(_expand_to_parent(self._provider, chunk) for chunk in results)
        )
        stage_timings_ms["parent_expand"] = (time.perf_counter() - t0) * 1000

        return RetrievalResult(
            chunks=expanded,
            scores=[c.score for c in expanded],
            top1_score=expanded[0].score if reranked else top1_score,
            reranked=reranked,
            stage_timings_ms=stage_timings_ms,
        )
