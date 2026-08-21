"""BM25 sparse vector encoding via fastembed. Qdrant stores and searches
sparse vectors natively but does not compute BM25 from raw text itself 
this module is that missing piece, feeding indexer.py alongside the dense
bge-m3 embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding

_MODEL_NAME = "Qdrant/bm25"


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@lru_cache(maxsize=1)
def _get_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=_MODEL_NAME)


def encode(texts: list[str]) -> list[SparseVector]:
    embeddings = _get_model().embed(texts)
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in embeddings
    ]
