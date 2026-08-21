"""Splits a passage into sentences, embeds each, and cuts wherever
consecutive-sentence cosine distance exceeds a threshold (topic
discontinuity). `embed_fn` is injected so this is unit-testable without a
live service. Doesn't enforce token limits -- that's strategy_router.py's job.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from app.ingestion.types import Chunk, Passage

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।!?.])\s+")

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

DEFAULT_DISTANCE_THRESHOLD = 0.35


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - dot / (norm_a * norm_b))


async def chunk(
    passage: Passage,
    *,
    embed_fn: EmbedFn,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[Chunk]:
    sentences = _split_sentences(passage.text)
    if len(sentences) <= 1:
        return [
            Chunk(
                chunk_id=f"{passage.doc_id}_semantic_0",
                doc_id=passage.doc_id,
                parent_id=passage.doc_id,
                text=passage.text,
                passage_position=0,
                language=passage.language,
                source_query_id=str(passage.query_id),
                strategy="semantic",
            )
        ]

    embeddings = await embed_fn(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        distance = _cosine_distance(embeddings[i - 1], embeddings[i])
        if distance > distance_threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(
            chunk_id=f"{passage.doc_id}_semantic_{i}",
            doc_id=passage.doc_id,
            parent_id=passage.doc_id,
            text=" ".join(group),
            passage_position=i,
            language=passage.language,
            source_query_id=str(passage.query_id),
            strategy="semantic",
        )
        for i, group in enumerate(groups)
    ]
