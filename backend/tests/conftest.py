"""Sets baseline env vars before any `app.*` module imports, so `create_app()`
succeeds during test collection. QDRANT_URL points at an unreachable address
on purpose -- health tests assert graceful degradation, not a crash.
"""

from __future__ import annotations

import os

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")
os.environ.setdefault("QDRANT_URL", "http://localhost:1")
os.environ.setdefault("SARVAM_API_KEY", "test-sarvam-key")
os.environ.setdefault("OLLAMA_API_KEY", "test-ollama-key")

import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
