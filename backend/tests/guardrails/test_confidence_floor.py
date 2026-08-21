from __future__ import annotations

import pytest

from app.core.exceptions import GuardrailRejection
from app.guardrails.confidence_floor import ConfidenceFloorGuardrail


def test_none_top1_score_is_rejected() -> None:
    guardrail = ConfidenceFloorGuardrail(threshold=0.5)
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check(None)
    assert exc_info.value.reason_code == "no_retrieval_results"


def test_score_at_threshold_is_rejected() -> None:
    guardrail = ConfidenceFloorGuardrail(threshold=0.5)
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check(0.5)
    assert exc_info.value.reason_code == "low_confidence"


def test_score_below_threshold_is_rejected() -> None:
    guardrail = ConfidenceFloorGuardrail(threshold=0.5)
    with pytest.raises(GuardrailRejection):
        guardrail.check(0.2)


def test_score_above_threshold_passes() -> None:
    guardrail = ConfidenceFloorGuardrail(threshold=0.5)
    guardrail.check(0.9)  # should not raise
