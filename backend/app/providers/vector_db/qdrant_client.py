"""Concrete Qdrant implementation of VectorDBProvider. Named vectors:
"dense" (bge-m3, 1024-dim, no Matryoshka truncation) and "sparse" (BM25).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

from qdrant_client import AsyncQdrantClient, models

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.pipeline.state import RetrievedChunk
from app.providers.vector_db.base import VectorDBProvider, VectorPoint

_logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


def point_id_for_chunk(chunk_id: str) -> str:
    """Qdrant point IDs must be an unsigned int or UUID, not an arbitrary
    string. Deterministic (uuid5) so re-running ingestion upserts the same
    point instead of duplicating it. The original chunk_id is kept in the
    payload for reference."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_id}"))


class QdrantVectorDBProvider(VectorDBProvider):
    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._collection_name = collection_name
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout) # type: ignore

    async def ensure_collection(self, *, dense_dim: int, recreate: bool = False) -> None:
        exists = await self._client.collection_exists(self._collection_name)
        if exists and not recreate:
            return
        if exists and recreate:
            await self._client.delete_collection(self._collection_name)

        await self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        last_error: Exception | None = None
        qdrant_points = [
            models.PointStruct(
                id=point_id_for_chunk(point.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: point.dense_vector,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(point.sparse_vector.keys()),
                        values=list(point.sparse_vector.values()),
                    ),
                },
                payload={**point.payload, "chunk_id": point.chunk_id},
            )
            for point in points
        ]

        for attempt in range(_MAX_RETRIES):
            try:
                await self._client.upsert(
                    collection_name=self._collection_name, points=qdrant_points
                )
                return
            except Exception as exc:  # noqa: BLE001 - qdrant_client raises assorted transport errors
                last_error = exc
                _logger.warning(
                    "qdrant_upsert_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

        raise ExternalServiceError(
            f"Qdrant upsert failed after {_MAX_RETRIES} attempts: {last_error}",
            service="qdrant",
        )

    async def hybrid_search(
        self,
        *,
        dense_vector: Sequence[float],
        sparse_vector: Mapping[int, float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = models.Filter(**filters) if filters else None
        dense_search_params = (
            models.SearchParams(hnsw_ef=ef_search) if ef_search is not None else None
        )
        try:
            result = await self._client.query_points(
                collection_name=self._collection_name,
                prefetch=[
                    models.Prefetch(
                        query=list(dense_vector),
                        using=DENSE_VECTOR_NAME,
                        limit=top_k,
                        filter=query_filter,
                        params=dense_search_params,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=top_k,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise ExternalServiceError(
                f"Qdrant hybrid search failed: {exc}", service="qdrant"
            ) from exc

        return [
            RetrievedChunk(
                chunk_id=str(point.payload["chunk_id"]),
                doc_id=str(point.payload["doc_id"]),
                parent_id=str(point.payload["parent_id"]),
                text=str(point.payload["text"]),
                passage_position=int(point.payload["passage_position"]),
                language=str(point.payload["language"]),
                source_query_id=point.payload.get("source_query_id"),
                score=point.score,
            )
            for point in result.points
            if point.payload is not None
        ]

    async def get_chunks_by_parent_id(self, parent_id: str) -> list[RetrievedChunk]:
        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="parent_id", match=models.MatchValue(value=parent_id))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=64,  # generous -- real sibling counts top out around a dozen
            )
        except Exception as exc:
            raise ExternalServiceError(
                f"Qdrant parent-id lookup failed: {exc}", service="qdrant"
            ) from exc

        return [
            RetrievedChunk(
                chunk_id=str(point.payload["chunk_id"]),
                doc_id=str(point.payload["doc_id"]),
                parent_id=str(point.payload["parent_id"]),
                text=str(point.payload["text"]),
                passage_position=int(point.payload["passage_position"]),
                language=str(point.payload["language"]),
                source_query_id=point.payload.get("source_query_id"),
                score=0.0,  # not a similarity result
            )
            for point in points
            if point.payload is not None
        ]

    async def health_check(self) -> bool:
        try:
            await self._client.collection_exists(self._collection_name)
            return True
        except Exception:  # noqa: BLE001 - health check must never raise
            return False

    async def compute_corpus_centroid(self) -> list[float]:
        sum_vector: list[float] | None = None
        count = 0
        offset = None
        try:
            while True:
                points, offset = await self._client.scroll(
                    collection_name=self._collection_name,
                    with_vectors=True,
                    with_payload=False,
                    limit=256,
                    offset=offset,
                )
                for point in points:
                    vector = point.vector
                    dense = vector[DENSE_VECTOR_NAME] if isinstance(vector, dict) else vector
                    # Narrows the client's stub type: "dense" is always a
                    # plain float list in this schema, never sparse/multi-vector.
                    if not isinstance(dense, list) or not all(
                        isinstance(component, float) for component in dense
                    ):
                        raise ExternalServiceError(
                            f"Expected a dense float vector for point {point.id}, "
                            f"got {type(dense).__name__}",
                            service="qdrant",
                        )
                    dense_floats = cast(list[float], dense)
                    if sum_vector is None:
                        sum_vector = list(dense_floats)
                    else:
                        sum_vector = [
                            a + b for a, b in zip(sum_vector, dense_floats, strict=True)
                        ]
                    count += 1
                if offset is None:
                    break
        except Exception as exc:
            raise ExternalServiceError(
                f"Qdrant corpus centroid computation failed: {exc}", service="qdrant"
            ) from exc

        if sum_vector is None or count == 0:
            raise ExternalServiceError(
                "Cannot compute corpus centroid: collection has no indexed points",
                service="qdrant",
            )
        return [component / count for component in sum_vector]
