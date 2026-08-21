"""Combines the three chunking strategies into the production ingestion
path: structural first if it fits the token budget, else semantic split,
then fixed_overlap for any semantic group still over budget.
"""

from __future__ import annotations

from app.ingestion.chunking import fixed_overlap, semantic, structural
from app.ingestion.chunking.semantic import EmbedFn
from app.ingestion.tokenization import count_tokens
from app.ingestion.types import Chunk, Passage

DEFAULT_MAX_TOKENS = 256


async def route_chunk(
    passage: Passage,
    *,
    embed_fn: EmbedFn,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    if count_tokens(passage.text) <= max_tokens:
        return structural.chunk(passage)

    semantic_chunks = await semantic.chunk(passage, embed_fn=embed_fn)

    final_pieces: list[tuple[str, str]] = []
    for semantic_chunk in semantic_chunks:
        if count_tokens(semantic_chunk.text) <= max_tokens:
            final_pieces.append((semantic_chunk.text, "semantic"))
        else:
            windows = fixed_overlap.split_text(semantic_chunk.text, chunk_size=max_tokens)
            final_pieces.extend((window, "semantic+fixed_overlap") for window in windows)

    return [
        Chunk(
            chunk_id=f"{passage.doc_id}_routed_{i}",
            doc_id=passage.doc_id,
            parent_id=passage.doc_id,
            text=text,
            passage_position=i,
            language=passage.language,
            source_query_id=str(passage.query_id),
            strategy=strategy,
        )
        for i, (text, strategy) in enumerate(final_pieces)
    ]
