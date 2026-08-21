from __future__ import annotations

from app.ingestion.chunking import structural
from app.ingestion.types import Passage


def _passage(text: str = "यह एक परीक्षण पैसेज है।") -> Passage:
    return Passage(
        doc_id="42_0",
        query_id=42,
        passage_index=0,
        text=text,
        is_selected=True,
        language="hi",
        query_text="परीक्षण प्रश्न",
    )


def test_structural_returns_single_chunk_matching_passage_text() -> None:
    passage = _passage()

    chunks = structural.chunk(passage)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.text == passage.text
    assert chunk.doc_id == passage.doc_id
    assert chunk.parent_id == passage.doc_id
    assert chunk.passage_position == 0
    assert chunk.source_query_id == str(passage.query_id)
    assert chunk.strategy == "structural"


def test_structural_chunk_id_is_deterministic() -> None:
    passage = _passage()

    first = structural.chunk(passage)[0].chunk_id
    second = structural.chunk(passage)[0].chunk_id

    assert first == second
