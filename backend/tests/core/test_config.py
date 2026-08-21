from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError

_REQUIRED_VARS = ["FLASK_SECRET_KEY", "QDRANT_URL", "SARVAM_API_KEY", "OLLAMA_API_KEY"]


def test_settings_loads_with_all_required_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASK_SECRET_KEY", "secret")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("SARVAM_API_KEY", "sarvam-key")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.ollama_cloud_base_url == "https://ollama.com"
    assert settings.ollama_model == "gpt-oss:120b-cloud"
    assert settings.qdrant_api_key is None


def test_settings_missing_required_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(PydanticValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_wraps_validation_error_in_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in _REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()

    assert "missing required setting" in str(exc_info.value)
