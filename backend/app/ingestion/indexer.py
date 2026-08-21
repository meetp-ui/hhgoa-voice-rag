"""Collection setup and building indexable points from chunks + vectors.

Uses bge-m3's full 1024-dim output, no Matryoshka truncation -- a
correctness-first baseline. Changing this later means a full re-index.
"""

from __future__ import annotations

from app.ingestion.embedding import EMBEDDING_DIM
from app.ingestion.sparse_encoding import SparseVector
from app.ingestion.types import Chunk
from app.providers.vector_db.base import VectorPoint
from app.providers.vector_db.qdrant_client import QdrantVectorDBProvider

INDEX_BATCH_SIZE = 64


def build_vector_point(
    chunk: Chunk, dense_vector: list[float], sparse_vector: SparseVector
) -> VectorPoint:
    return VectorPoint(
        chunk_id=chunk.chunk_id,
        dense_vector=dense_vector,
        sparse_vector=dict(zip(sparse_vector.indices, sparse_vector.values, strict=True)),
        payload={
            "doc_id": chunk.doc_id,
            "parent_id": chunk.parent_id,
            "text": chunk.text,
            "passage_position": chunk.passage_position,
            "language": chunk.language,
            "source_query_id": chunk.source_query_id,
            "strategy": chunk.strategy,
        },
    )


async def index_chunks(
    provider: QdrantVectorDBProvider,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
) -> None:
    """Upserts `chunks` in batches. Caller is responsible for computing
    dense_vectors/sparse_vectors in the same order as `chunks`."""
    points = [
        build_vector_point(chunk, dense_vec, sparse_vec)
        for chunk, dense_vec, sparse_vec in zip(
            chunks, dense_vectors, sparse_vectors, strict=True
        )
    ]
    for i in range(0, len(points), INDEX_BATCH_SIZE):
        await provider.upsert(points[i : i + INDEX_BATCH_SIZE])


async def ensure_collection(provider: QdrantVectorDBProvider, *, recreate: bool = False) -> None:
    await provider.ensure_collection(dense_dim=EMBEDDING_DIM, recreate=recreate)
