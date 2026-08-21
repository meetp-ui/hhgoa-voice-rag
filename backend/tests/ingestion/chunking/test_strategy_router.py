from __future__ import annotations

import pytest

from app.ingestion.chunking.strategy_router import route_chunk
from app.ingestion.tokenization import count_tokens
from app.ingestion.types import Passage


def _passage(text: str) -> Passage:
    return Passage(
        doc_id="3_0",
        query_id=3,
        passage_index=0,
        text=text,
        is_selected=True,
        language="hi",
        query_text="प्रश्न",
    )


def _repeat_to_min_tokens(unit: str, min_tokens: int) -> str:
    text = unit
    while count_tokens(text) < min_tokens:
        text += unit
    return text


@pytest.mark.asyncio
async def test_short_passage_uses_structural_only_and_never_calls_embed_fn() -> None:
    passage = _passage("यह एक छोटा पैसेज है।")
    assert count_tokens(passage.text) <= 256

    async def embed_fn_should_not_be_called(texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed_fn must not be called for a short (in-budget) passage")

    chunks = await route_chunk(passage, embed_fn=embed_fn_should_not_be_called)

    assert len(chunks) == 1
    assert chunks[0].strategy == "structural"
    assert chunks[0].text == passage.text


@pytest.mark.asyncio
async def test_long_passage_with_in_budget_semantic_groups_stays_semantic() -> None:
    s1 = _repeat_to_min_tokens("पहला खंड शब्द ", 150) + "।"
    s2 = _repeat_to_min_tokens("दूसरा खंड शब्द ", 150) + "।"
    passage = _passage(f"{s1} {s2}")
    assert count_tokens(passage.text) > 256  # router must not take the structural shortcut
    assert count_tokens(s1) <= 256
    assert count_tokens(s2) <= 256

    async def embed_fn(texts: list[str]) -> list[list[float]]:
        # orthogonal vectors, two distinct semantic groups, both within budget
        return [[1.0, 0.0] if t == s1 else [0.0, 1.0] for t in texts]

    chunks = await route_chunk(passage, embed_fn=embed_fn)

    assert len(chunks) == 2
    assert {c.text for c in chunks} == {s1, s2}
    assert all(c.strategy == "semantic" for c in chunks)


@pytest.mark.asyncio
async def test_oversized_semantic_group_falls_back_to_fixed_overlap() -> None:
    short_sentence = "छोटा वाक्य है।"
    long_sentence = _repeat_to_min_tokens("बहुत लंबा खंड शब्द दोहराव ", 400)
    passage = _passage(f"{short_sentence} {long_sentence}")

    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if t == short_sentence else [0.0, 1.0] for t in texts]

    chunks = await route_chunk(passage, embed_fn=embed_fn)

    # short_sentence's group stays one pure-semantic chunk; long_sentence's
    # group (>256 tokens) gets split by fixed_overlap into multiple bounded
    # pieces, labeled distinctly from the untouched semantic chunk.
    assert len(chunks) > 2
    assert chunks[0].text == short_sentence
    assert chunks[0].strategy == "semantic"
    for chunk in chunks[1:]:
        assert count_tokens(chunk.text) <= 256
        assert chunk.strategy == "semantic+fixed_overlap"
