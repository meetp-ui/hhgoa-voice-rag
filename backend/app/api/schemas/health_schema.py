from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ProviderStatus(BaseModel):
    configured: bool
    reachable: bool | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, ProviderStatus]
