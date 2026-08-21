from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any, TypeVar

T = TypeVar("T")


class BackgroundEventLoop:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_forever, name="pipeline-event-loop", daemon=True
        )
        self._thread.start()

    def _run_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Schedules `coro` on the background loop and blocks the calling
        thread until it completes, returning its result (or re-raising its
        exception)."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
