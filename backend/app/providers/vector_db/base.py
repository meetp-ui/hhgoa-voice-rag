from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.pipeline.state import RetrievedChunk


class VectorPoint(BaseModel):
    """A single point to upsert: a chunk's dense + sparse vectors plus payload."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    dense_vector: list[float]
    sparse_vector: Mapping[int, float]
    payload: Mapping[str, Any]


class VectorDBProvider(ABC):
    @abstractmethod
    async def hybrid_search(
        self,
        *,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + sparse search fused via RRF, returns top_k chunks.

        `ef_search` trades recall for latency; `None` uses the collection
        default. Implementations must retry with backoff and raise
        `ExternalServiceError`, never a bare exception.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_chunks_by_parent_id(self, parent_id: str) -> list[RetrievedChunk]:
        """Gets all chunks with this parent_id. Payload filter, not a
        search -- ignore the score field. Most passages have just one
        chunk (self-parented, so this is a no-op). Split passages need
        this to reassemble the original text from siblings."""
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        """Upsert chunk vectors + payload into the store."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Used by GET /health. Must not raise -- return False on failure."""
        raise NotImplementedError

    @abstractmethod
    async def compute_corpus_centroid(self) -> list[float]:
        """Mean of every indexed chunk's dense vector, via a full collection
        scroll. Compute once at startup and reuse -- don't call per query."""
        raise NotImplementedError
