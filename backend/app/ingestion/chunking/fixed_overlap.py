"""Fixed-size chunking with overlap -- the fallback for passages with no
clean semantic break. 256 tokens, 17.5% overlap, tokenized with bge-m3's
own tokenizer so the budget matches what the model actually sees.
"""

from __future__ import annotations

from app.ingestion.tokenization import encode_with_offsets
from app.ingestion.types import Chunk, Passage

DEFAULT_CHUNK_SIZE = 256
DEFAULT_OVERLAP_RATIO = 0.175


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[str]:
    """Splits `text` into overlapping windows of up to `chunk_size` tokens,
    sliced from the original text via token offsets (not decoded)."""
    _, offsets = encode_with_offsets(text)
    if not offsets:
        return []

    stride = max(1, int(chunk_size * (1 - overlap_ratio)))
    windows: list[str] = []
    start = 0
    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        windows.append(text[char_start:char_end])
        if end == len(offsets):
            break
        start += stride
    return windows


def chunk(
    passage: Passage,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[Chunk]:
    windows = split_text(passage.text, chunk_size=chunk_size, overlap_ratio=overlap_ratio)
    return [
        Chunk(
            chunk_id=f"{passage.doc_id}_fixed_{i}",
            doc_id=passage.doc_id,
            parent_id=passage.doc_id,
            text=window_text,
            passage_position=i,
            language=passage.language,
            source_query_id=str(passage.query_id),
            strategy="fixed_overlap",
        )
        for i, window_text in enumerate(windows)
    ]
