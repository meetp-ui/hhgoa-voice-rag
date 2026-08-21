from __future__ import annotations

import math
import unicodedata

from app.core.exceptions import GuardrailRejection

_NON_LINGUISTIC_RATIO_THRESHOLD = 0.5


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def _looks_non_linguistic(text: str) -> bool:
    """Cheap pre-embedding gibberish check, not a language-detection model.

    Uses Unicode category L*/M* (letter + combining mark), not `str.isalpha()`.
    isalpha() doesn't count Devanagari's combining vowel signs/virama, so it
    misclassified real Hindi queries as noise -- caught by testing.
    """
    stripped = text.strip()
    if not stripped:
        return True
    letter_like_count = sum(
        1 for ch in stripped if unicodedata.category(ch)[0] in ("L", "M")
    )
    non_space_count = sum(1 for ch in stripped if not ch.isspace())
    if non_space_count == 0:
        return True
    return (letter_like_count / non_space_count) < _NON_LINGUISTIC_RATIO_THRESHOLD


class InputFilterGuardrail:
    def __init__(
        self,
        *,
        corpus_centroid: list[float],
        distance_threshold: float,
        min_query_length_chars: int,
    ) -> None:
        self._centroid = corpus_centroid
        self._distance_threshold = distance_threshold
        self._min_query_length_chars = min_query_length_chars

    def check_transcript(self, transcript: str) -> None:
        """Cheap checks that don't require an embedding call. Raises
        GuardrailRejection on empty, too-short, or non-linguistic input."""
        stripped = transcript.strip()
        if len(stripped) < self._min_query_length_chars:
            raise GuardrailRejection(
                f"Query is too short ({len(stripped)} chars, minimum "
                f"{self._min_query_length_chars}) to be a meaningful question.",
                reason_code="query_too_short",
            )
        if _looks_non_linguistic(stripped):
            raise GuardrailRejection(
                "Query does not appear to contain meaningful text.",
                reason_code="non_linguistic_input",
            )

    def check_off_topic(self, query_embedding: list[float]) -> None:
        """Raises GuardrailRejection if the query embedding is too far from the
        corpus centroid to plausibly be answerable from this corpus."""
        distance = _cosine_distance(query_embedding, self._centroid)
        if distance > self._distance_threshold:
            raise GuardrailRejection(
                f"Query appears unrelated to the indexed corpus (distance="
                f"{distance:.3f}, threshold={self._distance_threshold}).",
                reason_code="off_topic",
            )
