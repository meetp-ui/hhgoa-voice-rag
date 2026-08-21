from __future__ import annotations

import pytest

from app.core.exceptions import GuardrailRejection
from app.guardrails.input_filter import InputFilterGuardrail

CENTROID = [1.0, 0.0, 0.0]


def _guardrail(*, threshold: float = 0.5, min_len: int = 3) -> InputFilterGuardrail:
    return InputFilterGuardrail(
        corpus_centroid=CENTROID, distance_threshold=threshold, min_query_length_chars=min_len
    )


def test_empty_transcript_is_rejected() -> None:
    guardrail = _guardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check_transcript("   ")
    assert exc_info.value.reason_code == "query_too_short"


def test_extremely_short_transcript_is_rejected() -> None:
    guardrail = _guardrail(min_len=5)
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check_transcript("hi")
    assert exc_info.value.reason_code == "query_too_short"


def test_non_linguistic_noise_is_rejected() -> None:
    guardrail = _guardrail()
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check_transcript("!@#$%^&*()12345")
    assert exc_info.value.reason_code == "non_linguistic_input"


def test_normal_query_passes_transcript_checks() -> None:
    guardrail = _guardrail()
    guardrail.check_transcript("दिल्ली भारत की राजधानी है क्या")  # should not raise


def test_query_close_to_centroid_is_not_off_topic() -> None:
    guardrail = _guardrail(threshold=0.5)
    close_embedding = [0.99, 0.01, 0.0]
    guardrail.check_off_topic(close_embedding)  # should not raise


def test_query_far_from_centroid_is_rejected_as_off_topic() -> None:
    guardrail = _guardrail(threshold=0.5)
    far_embedding = [0.0, 1.0, 0.0]  # orthogonal to centroid, cosine distance 1.0
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check_off_topic(far_embedding)
    assert exc_info.value.reason_code == "off_topic"
