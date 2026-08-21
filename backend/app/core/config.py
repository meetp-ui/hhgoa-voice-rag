from __future__ import annotations

from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.exceptions import ConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Flask
    flask_secret_key: str = Field(description="Flask session/signing secret. Required.")
    flask_env: str = Field(default="production", description="'development' or 'production'.")
    log_level: str = Field(default="INFO", description="Root logger level.")
    cors_allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated origins allowed to call the API cross-origin.",
    )

    # Qdrant (vector DB)
    qdrant_url: str = Field(description="Qdrant endpoint, e.g. http://qdrant:6333 in compose.")
    qdrant_api_key: str | None = Field(
        default=None, description="Qdrant API key. Unset for unauthenticated local/dev Qdrant."
    )
    qdrant_collection: str = Field(
        default="rag_passages", description="Qdrant collection name for indexed passages."
    )

    # Embedding / rerank (TEI services, compose-colocated)
    embedding_service_url: str = Field(
        default="http://embedding:80",
        description="TEI service URL serving bge-m3 dense embeddings.",
    )
    reranker_service_url: str = Field(
        default="http://reranker:80",
        description="TEI service URL serving bge-reranker-v2-m3.",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Max candidates sent to the reranker when it fires.",
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Reranker default when a request doesn't specify use_reranker. "
        "Off by default: P50 152.5ms without rerank vs 1872.9ms with it. "
        "See reports/retrieval_latency.md.",
    )

    # Ingestion (local dataset cache)
    dataset_cache_dir: str = Field(
        default="/app/.cache/datasets",
        description="Local dataset cache. Persisted via a compose volume so ingestion "
        "doesn't re-download.",
    )

    # Sarvam AI (STT)
    sarvam_api_key: str = Field(description="Sarvam AI API key. Required.")
    sarvam_stt_base_url: str = Field(
        default="https://api.sarvam.ai", description="Sarvam AI API base URL."
    )
    sarvam_stt_model: str = Field(
        default="saaras:v3",
        description="Sarvam speech-to-text model. saaras supports transcribe+translate "
        "across Indic languages; confirmed against Sarvam's current API reference.",
    )

    # Ollama Cloud (LLM)
    ollama_api_key: str = Field(description="Ollama Cloud API key. Required.")
    ollama_cloud_base_url: str = Field(
        default="https://ollama.com", description="Ollama Cloud API base URL."
    )
    ollama_model: str = Field(
        default="gpt-oss:120b-cloud", description="Ollama Cloud model identifier."
    )

    # Guardrails
    off_topic_distance_threshold: float = Field(
        default=0.5,
        description="Cosine distance from the corpus centroid above which a query is "
        "rejected as off-topic. See app/guardrails/input_filter.py for the threshold "
        "investigation.",
    )
    min_query_length_chars: int = Field(
        default=3,
        description="Queries shorter than this (after stripping whitespace) are "
        "rejected as garbage/empty input before any embedding call.",
    )
    confidence_floor_threshold: float = Field(
        default=0.5,
        description="Retrieval top1_score at or below which the pipeline short-circuits "
        "before calling the LLM. Same threshold as the reranker's ambiguous-band check.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]  # values sourced from env/.env
    except ValidationError as exc:
        missing = ", ".join(
            str(error["loc"][0]) for error in exc.errors() if error["type"] == "missing"
        )
        detail = f"missing required setting(s): {missing}" if missing else str(exc)
        raise ConfigurationError(
            f"Invalid application configuration: {detail}",
            reason_code="missing_or_invalid_config",
        ) from exc
