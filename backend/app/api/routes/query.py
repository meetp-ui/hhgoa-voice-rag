from __future__ import annotations

from flask import Blueprint, Response, current_app, jsonify, request

from app.api.schemas.query_schema import GuardrailInfo, QueryResponse, SourceInfo, TimingsInfo
from app.core.background_loop import BackgroundEventLoop
from app.core.exceptions import ExternalServiceError, ValidationError
from app.core.logging import get_logger
from app.pipeline.factory import build_pipeline_graph
from app.pipeline.graph import PipelineGraph, PipelineResult

query_bp = Blueprint("query", __name__)
_logger = get_logger(__name__)


def _parse_bool_field(raw: str | None) -> bool | None:
    """Multipart form fields arrive as strings ("true"/"false"), unlike a
    JSON body where `use_reranker` is already a real bool or absent."""
    if raw is None:
        return None
    return raw.strip().lower() in ("true", "1", "yes")


def _get_pipeline_graph(loop: BackgroundEventLoop) -> PipelineGraph:
    """Builds and caches the pipeline graph on first use, on the background
    loop so its httpx clients bind there. Not built at import time --
    needs a live Qdrant, and import must succeed even if Qdrant is down."""
    graph = current_app.extensions.get("pipeline_graph")
    if isinstance(graph, PipelineGraph):
        return graph

    graph = loop.run(build_pipeline_graph(current_app.extensions["settings"]))
    current_app.extensions["pipeline_graph"] = graph
    return graph


def _to_response(result: PipelineResult) -> QueryResponse:
    return QueryResponse(
        request_id=result.request_id,
        status=result.status,  # type: ignore[arg-type]
        transcript=result.transcript,
        query_language=result.query_language,
        answer=result.answer,
        confidence=result.confidence,
        sources=[SourceInfo(chunk_id=s.chunk_id, text=s.text) for s in result.sources],
        reranked=result.reranked,
        guardrail=(
            GuardrailInfo(
                stage=result.guardrail.stage,
                reason_code=result.guardrail.reason_code,
                message=result.guardrail.message,
                score=result.guardrail.score,
            )
            if result.guardrail is not None
            else None
        ),
        error=result.error,
        timings=TimingsInfo(
            transcribe_ms=result.timings.get("transcribe"),
            retrieve_ms=result.timings.get("retrieve"),
            generate_ms=result.timings.get("generate"),
            total_ms=result.timings.get("total", 0.0),
        ),
    )


_STATUS_CODES = {"ok": 200, "rejected": 422, "error": 502}


@query_bp.post("/query")
def query() -> tuple[Response, int]:
    loop: BackgroundEventLoop = current_app.extensions["background_loop"]

    try:
        graph = _get_pipeline_graph(loop)
    except ExternalServiceError as exc:
        _logger.warning("pipeline_graph_build_failed", extra={"error": exc.message})
        return jsonify({"error": exc.message, "reason_code": "pipeline_unavailable"}), 502

    audio: bytes | None = None
    audio_content_type = "application/octet-stream"
    audio_filename = "audio"
    text: str | None = None
    language_hint: str | None = None
    use_reranker: bool | None = None

    if request.content_type and "multipart/form-data" in request.content_type:
        if "audio" not in request.files:
            raise ValidationError(
                "multipart request must include an 'audio' file field",
                reason_code="missing_audio",
            )
        audio_file = request.files["audio"]
        audio = audio_file.read()
        audio_content_type = audio_file.content_type or audio_content_type
        audio_filename = audio_file.filename or audio_filename
        language_hint = request.form.get("language_hint")
        use_reranker = _parse_bool_field(request.form.get("use_reranker"))
    else:
        body = request.get_json(silent=True) or {}
        text = body.get("text")
        language_hint = body.get("language_hint")
        use_reranker = body.get("use_reranker")
        if not text:
            raise ValidationError(
                "JSON request must include a non-empty 'text' field",
                reason_code="missing_text",
            )

    result = loop.run(
        graph.run(
            audio=audio,
            audio_content_type=audio_content_type,
            audio_filename=audio_filename,
            text=text,
            language_hint=language_hint,
            use_reranker=use_reranker,
        )
    )
    response = _to_response(result)
    return jsonify(response.model_dump()), _STATUS_CODES[result.status]


@query_bp.errorhandler(ValidationError)
def _handle_validation_error(exc: ValidationError) -> tuple[Response, int]:
    return jsonify({"error": exc.message, "reason_code": exc.reason_code}), 400
