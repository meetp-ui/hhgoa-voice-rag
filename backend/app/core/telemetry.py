from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, TypeVar

from app.core.logging import get_logger

_logger = get_logger("app.telemetry")

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

P = ParamSpec("P")
R = TypeVar("R")


def bind_request_id(request_id: str | None = None) -> str:
    """Set the request id for the current execution context, generating one if omitted.
    Returns the id that was bound."""
    resolved = request_id or str(uuid.uuid4())
    _request_id_var.set(resolved)
    return resolved


def current_request_id() -> str | None:
    return _request_id_var.get()


@contextmanager
def timed_stage(stage: str, *, request_id: str | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _logger.info(
            "stage_timing",
            extra={
                "stage": stage,
                "duration_ms": round(duration_ms, 3),
                "request_id": request_id or current_request_id(),
            },
        )


def timed(stage: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for sync functions. See `timed_async` for coroutine functions."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_stage(stage):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed_async(
    stage: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator for async functions/coroutine-returning callables."""

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timed_stage(stage):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
