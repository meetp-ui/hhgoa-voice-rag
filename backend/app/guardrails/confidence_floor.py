"""Short-circuits before the LLM call when retrieval's top1_score is too
low. Threshold defaults to 0.5, same as retrieve.py's AMBIGUOUS_SCORE_THRESHOLD
-- works whether top1_score is an RRF score or (post-rerank) a relevance
score, since 0.5 means "no clear winner" on both scales.
"""

from __future__ import annotations

from app.core.exceptions import GuardrailRejection


class ConfidenceFloorGuardrail:
    def __init__(self, *, threshold: float) -> None:
        self._threshold = threshold

    def check(self, top1_score: float | None) -> None:
        """Raises GuardrailRejection if retrieval found nothing, or found
        nothing confidently enough to ground an answer."""
        if top1_score is None:
            raise GuardrailRejection(
                "No relevant context was found for this query.",
                reason_code="no_retrieval_results",
            )
        if top1_score <= self._threshold:
            raise GuardrailRejection(
                f"Retrieved context is not confident enough to ground an answer "
                f"(top1_score={top1_score:.3f}, threshold={self._threshold}).",
                reason_code="low_confidence",
            )
