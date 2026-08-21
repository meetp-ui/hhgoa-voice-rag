from __future__ import annotations

import pytest

from app.core.exceptions import GuardrailRejection
from app.guardrails.groundedness import GroundednessGuardrail
from app.providers.llm.base import StructuredAnswer


def test_grounded_answer_passes() -> None:
    guardrail = GroundednessGuardrail()
    answer = StructuredAnswer(
        answer="जवाब", grounded=True, cited_chunk_ids=["a"], confidence=0.9
    )
    guardrail.check(answer)  # should not raise


def test_ungrounded_answer_is_rejected() -> None:
    guardrail = GroundednessGuardrail()
    answer = StructuredAnswer(
        answer="मुझे नहीं पता", grounded=False, cited_chunk_ids=[], confidence=0.1
    )
    with pytest.raises(GuardrailRejection) as exc_info:
        guardrail.check(answer)
    assert exc_info.value.reason_code == "not_grounded"
