"""Client for the `embedding` TEI service (bge-m3, dense vectors only).

The httpx.AsyncClient is created once and reused across calls, not
recreated per `embed()` call -- a fresh client per call was a real bug
found during latency profiling (a new TCP connection every query).
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Self

import httpx

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

_logger = get_logger(__name__)

EMBEDDING_DIM = 1024

MAX_BATCH_SIZE = 8

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


class EmbeddingClient:
    def __init__(self, base_url: str, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds `texts`, batching to respect the service's max batch size.
        Order of results matches order of input."""
        results: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            results.extend(await self._embed_batch(batch))
        return results

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post("/embed", json={"inputs": batch})
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                _logger.warning(
                    "embedding_request_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

        raise ExternalServiceError(
            f"Embedding service request failed after {_MAX_RETRIES} attempts: {last_error}",
            service="embedding",
        )
