"""Client for the `reranker` TEI service (bge-reranker-v2-m3). Only called
conditionally from retrieve.py, gated on the ambiguous-band threshold.

The httpx.AsyncClient is created once and reused across calls, not
recreated per `rerank()` call -- a fresh client per call was a real bug
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

_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.0


class RerankerClient:
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

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Returns a relevance score per text, in the same order as `texts`
        (the TEI /rerank response is index-tagged and not necessarily
        returned in input order, so this re-sorts by index before returning)."""
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.post(
                    "/rerank", json={"query": query, "texts": texts}
                )
                response.raise_for_status()
                ranked = response.json()
                scores_by_index = {item["index"]: item["score"] for item in ranked}
                return [scores_by_index[i] for i in range(len(texts))]
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                _logger.warning(
                    "rerank_request_failed",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))

        raise ExternalServiceError(
            f"Reranker service request failed after {_MAX_RETRIES} attempts: {last_error}",
            service="reranker",
        )
