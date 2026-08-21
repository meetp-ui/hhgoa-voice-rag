"""Typed contracts shared across the ingestion pipeline (loader, chunking,
embedding, indexing). No raw dicts cross module boundaries here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Passage(BaseModel):
    """One MS MARCO candidate passage. `doc_id` is synthesized as
    f"{query_id}_{passage_index}" since this dataset has no document IDs.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    query_id: int
    passage_index: int
    text: str
    is_selected: bool
    language: str
    query_text: str


class Chunk(BaseModel):
    """One indexable unit. `parent_id` is the passage's `doc_id` -- every
    chunk belongs to exactly one passage, so parent-child expansion is
    just a doc_id lookup.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    passage_position: int
    language: str
    source_query_id: str
    strategy: str
