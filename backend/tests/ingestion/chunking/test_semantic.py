from __future__ import annotations

import pytest

from app.ingestion.chunking import semantic
from app.ingestion.types import Passage


def _passage(text: str) -> Passage:
    return Passage(
        doc_id="9_0",
        query_id=9,
        passage_index=0,
        text=text,
        is_selected=True,
        language="hi",
        query_text="प्रश्न",
    )


def _fake_embed_fn_from(vectors: dict[str, list[float]]):
    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return [vectors[t] for t in texts]

    return embed_fn


@pytest.mark.asyncio
async def test_single_sentence_passage_returns_one_chunk_without_embedding() -> None:
    passage = _passage("केवल एक वाक्य है।")

    async def embed_fn_should_not_be_called(texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_fn should not be called for a single-sentence passage")

    chunks = await semantic.chunk(passage, embed_fn=embed_fn_should_not_be_called)

    assert len(chunks) == 1
    assert chunks[0].text == passage.text
    assert chunks[0].strategy == "semantic"


@pytest.mark.asyncio
async def test_discontinuous_sentences_are_split_into_separate_groups() -> None:
    s1, s2 = "पहला विषय वाक्य एक है।", "पहला विषय वाक्य दो है।"
    s3 = "बिल्कुल अलग दूसरा विषय है।"
    passage = _passage(f"{s1} {s2} {s3}")

    embed_fn = _fake_embed_fn_from(
        {
            s1: [1.0, 0.0],
            s2: [0.99, 0.01],  # close to s1, same group
            s3: [0.0, 1.0],  # orthogonal, new group
        }
    )

    chunks = await semantic.chunk(passage, embed_fn=embed_fn, distance_threshold=0.35)

    assert len(chunks) == 2
    assert chunks[0].text == f"{s1} {s2}"
    assert chunks[1].text == s3
    assert [c.passage_position for c in chunks] == [0, 1]


@pytest.mark.asyncio
async def test_continuous_sentences_stay_in_one_group() -> None:
    s1, s2, s3 = "वाक्य एक।", "वाक्य दो।", "वाक्य तीन।"
    passage = _passage(f"{s1} {s2} {s3}")

    embed_fn = _fake_embed_fn_from({s1: [1.0, 0.0], s2: [0.98, 0.02], s3: [0.97, 0.03]})

    chunks = await semantic.chunk(passage, embed_fn=embed_fn, distance_threshold=0.35)

    assert len(chunks) == 1
    assert chunks[0].text == f"{s1} {s2} {s3}"
