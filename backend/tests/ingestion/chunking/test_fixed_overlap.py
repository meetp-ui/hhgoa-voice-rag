from __future__ import annotations

from app.ingestion.chunking import fixed_overlap
from app.ingestion.tokenization import count_tokens
from app.ingestion.types import Passage

_LONG_TEXT = "यह एक लंबा पैराग्राफ है जिसमें कई वाक्य होंगे। " * 60


def _passage(text: str) -> Passage:
    return Passage(
        doc_id="7_1",
        query_id=7,
        passage_index=1,
        text=text,
        is_selected=False,
        language="hi",
        query_text="प्रश्न",
    )


def test_short_text_produces_single_window() -> None:
    windows = fixed_overlap.split_text("छोटा वाक्य।", chunk_size=256)

    assert windows == ["छोटा वाक्य।"]


def test_long_text_is_split_into_multiple_bounded_windows() -> None:
    windows = fixed_overlap.split_text(_LONG_TEXT, chunk_size=64, overlap_ratio=0.175)

    assert len(windows) > 1
    for window in windows:
        assert count_tokens(window) <= 64


def test_windows_overlap() -> None:
    windows = fixed_overlap.split_text(_LONG_TEXT, chunk_size=64, overlap_ratio=0.175)

    # consecutive windows should share a non-empty trailing/leading substring
    # of meaningful length given ~17.5% overlap over a 64-token window.
    assert len(windows) >= 2
    tail_of_first = windows[0][-20:]
    assert any(tail_of_first[-5:] in windows[i] for i in range(1, len(windows)))


def test_empty_text_produces_no_windows() -> None:
    assert fixed_overlap.split_text("", chunk_size=256) == []


def test_chunk_returns_typed_chunks_with_positions() -> None:
    passage = _passage(_LONG_TEXT)

    chunks = fixed_overlap.chunk(passage, chunk_size=64, overlap_ratio=0.175)

    assert len(chunks) > 1
    assert [c.passage_position for c in chunks] == list(range(len(chunks)))
    assert all(c.strategy == "fixed_overlap" for c in chunks)
    assert all(c.doc_id == passage.doc_id for c in chunks)
