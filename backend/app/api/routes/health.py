from __future__ import annotations

from flask import Blueprint, Response, jsonify

from app.api.schemas.health_schema import HealthResponse, ProviderStatus
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

health_bp = Blueprint("health", __name__)
_logger = get_logger(__name__)

_QDRANT_HEALTH_TIMEOUT_SECONDS = 3


async def _check_qdrant_reachable(settings: Settings) -> bool:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=_QDRANT_HEALTH_TIMEOUT_SECONDS,
    )
    try:
        await client.get_collections()
        return True
    except (UnexpectedResponse, ResponseHandlingException, OSError, TimeoutError) as exc:
        _logger.warning("qdrant_health_check_failed", extra={"error": str(exc)})
        return False
    finally:
        await client.close()


@health_bp.get("/health")
async def health() -> tuple[Response, int]:
    settings = get_settings()

    qdrant_reachable = await _check_qdrant_reachable(settings)

    response = HealthResponse(
        status="ok" if qdrant_reachable else "degraded",
        providers={
            "qdrant": ProviderStatus(configured=True, reachable=qdrant_reachable),
            "sarvam": ProviderStatus(configured=bool(settings.sarvam_api_key)),
            "ollama_cloud": ProviderStatus(configured=bool(settings.ollama_api_key)),
        },
    )
    status_code = 200 if qdrant_reachable else 503
    return jsonify(response.model_dump()), status_code
