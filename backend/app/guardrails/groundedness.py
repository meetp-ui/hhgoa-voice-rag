from __future__ import annotations

from app.core.exceptions import GuardrailRejection
from app.providers.llm.base import StructuredAnswer


class GroundednessGuardrail:
    def check(self, answer: StructuredAnswer) -> None:
        if not answer.grounded:
            raise GuardrailRejection(
                "The retrieved context does not contain enough information to "
                "answer this question.",
                reason_code="not_grounded",
            )
