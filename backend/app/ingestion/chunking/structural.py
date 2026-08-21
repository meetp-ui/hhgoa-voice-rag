"""MS MARCO-XI passages arrive pre-segmented, so "structural" chunking is
just: treat the passage as one chunk. Correct for this dataset, not a stub.
"""

from __future__ import annotations

from app.ingestion.types import Chunk, Passage


def chunk(passage: Passage) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{passage.doc_id}_structural_0",
            doc_id=passage.doc_id,
            parent_id=passage.doc_id,
            text=passage.text,
            passage_position=0,
            language=passage.language,
            source_query_id=str(passage.query_id),
            strategy="structural",
        )
    ]
