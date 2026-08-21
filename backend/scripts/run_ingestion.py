#!/usr/bin/env python3
"""End-to-end ingestion: load MS MARCO-XI (Hindi), chunk via
strategy_router, embed (bge-m3), BM25 sparse encode, index into Qdrant.

Qdrant point IDs are deterministic (uuid5 of chunk_id), so re-running
upserts the same points instead of duplicating them. Each batch is
committed before the next, so a crash only loses the in-flight batch.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Literal

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.chunking.strategy_router import route_chunk
from app.ingestion.embedding import EmbeddingClient
from app.ingestion.indexer import ensure_collection, index_chunks
from app.ingestion.loaders.msmarco_xi import load_hindi_passages
from app.ingestion.sparse_encoding import encode as sparse_encode
from app.ingestion.types import Chunk
from app.providers.vector_db.qdrant_client import QdrantVectorDBProvider

_logger = get_logger(__name__)

BATCH_SIZE = 32


async def run(
    split: Literal["train", "validation"], *, limit: int | None, recreate: bool
) -> None:
    settings = get_settings()
    embedding_client = EmbeddingClient(settings.embedding_service_url)
    provider = QdrantVectorDBProvider(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    await ensure_collection(provider, recreate=recreate)

    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return await embedding_client.embed(texts)

    passages_processed = 0
    chunks_indexed = 0
    pending_chunks: list[Chunk] = []
    start = time.monotonic()

    async def flush(chunks: list[Chunk]) -> None:
        nonlocal chunks_indexed
        if not chunks:
            return
        texts = [c.text for c in chunks]
        dense_vectors = await embedding_client.embed(texts)
        sparse_vectors = sparse_encode(texts)
        await index_chunks(provider, chunks, dense_vectors, sparse_vectors)
        chunks_indexed += len(chunks)
        _logger.info(
            "ingestion_progress",
            extra={
                "passages_processed": passages_processed,
                "chunks_indexed": chunks_indexed,
                "elapsed_s": round(time.monotonic() - start, 1),
            },
        )

    for passage in load_hindi_passages(split=split):
        chunks = await route_chunk(passage, embed_fn=embed_fn)
        pending_chunks.extend(chunks)
        passages_processed += 1

        if len(pending_chunks) >= BATCH_SIZE:
            await flush(pending_chunks)
            pending_chunks = []

        if limit is not None and passages_processed >= limit:
            break

    await flush(pending_chunks)
    _logger.info(
        "ingestion_complete",
        extra={
            "passages_processed": passages_processed,
            "chunks_indexed": chunks_indexed,
            "elapsed_s": round(time.monotonic() - start, 1),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--limit", type=int, default=None, help="Max passages to process.")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and recreate the Qdrant collection first."
    )
    args = parser.parse_args()

    configure_logging()
    try:
        asyncio.run(run(args.split, limit=args.limit, recreate=args.recreate))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
