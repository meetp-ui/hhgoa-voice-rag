"""Domain exception hierarchy."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain exceptions raised anywhere in the app."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.message = message
        self.reason_code = reason_code


class ConfigurationError(DomainError):
    """Raised at startup when required configuration is missing or invalid."""

    def __init__(self, message: str, *, reason_code: str = "configuration_error") -> None:
        super().__init__(message, reason_code=reason_code)


class ExternalServiceError(DomainError):
    """Raised when a call to an external service (Sarvam, Qdrant, Ollama Cloud, ...) fails
    after exhausting retries, or times out. Carries the offending service's name so callers
    can attribute the failure without parsing the message string.
    """

    def __init__(
        self, message: str, *, service: str, reason_code: str = "external_service_error"
    ) -> None:
        super().__init__(message, reason_code=reason_code)
        self.service = service


class ValidationError(DomainError):
    """Raised when input fails validation beyond what Pydantic/flask-pydantic enforces
    at the API boundary (e.g. cross-field or business-rule validation)."""

    def __init__(self, message: str, *, reason_code: str = "validation_error") -> None:
        super().__init__(message, reason_code=reason_code)


class GuardrailRejection(DomainError):
    """Raised when a guardrail (input filter, confidence floor, groundedness check)
    rejects a request. reason_code is the structured code logged for later analysis,
    e.g. 'off_topic', 'low_confidence', 'not_grounded', 'pii_detected'."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message, reason_code=reason_code)
