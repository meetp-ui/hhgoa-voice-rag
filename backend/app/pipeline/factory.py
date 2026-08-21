"""Builds a fully-wired PipelineGraph from Settings. Called once per worker
process at startup, not per request -- the corpus centroid computation is
a full collection scroll.
"""

from __future__ import annotations

from app.core.config import Settings
from app.guardrails.confidence_floor import ConfidenceFloorGuardrail
from app.guardrails.groundedness import GroundednessGuardrail
from app.guardrails.input_filter import InputFilterGuardrail
from app.ingestion.embedding import EmbeddingClient
from app.pipeline.graph import PipelineGraph
from app.pipeline.nodes.generate import GenerateNode
from app.pipeline.nodes.retrieve import RetrieveNode
from app.pipeline.nodes.transcribe import TranscribeNode
from app.pipeline.reranker import RerankerClient
from app.providers.llm.ollama_cloud import OllamaCloudLLMProvider
from app.providers.stt.sarvam import SarvamSTTProvider
from app.providers.vector_db.qdrant_client import QdrantVectorDBProvider


async def build_pipeline_graph(settings: Settings) -> PipelineGraph:
    embedding_client = EmbeddingClient(settings.embedding_service_url)
    vector_db_provider = QdrantVectorDBProvider(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )
    # Constructed unconditionally (cheap: just an httpx client, no network
    # call yet) so a per-request use_reranker=true override can still fire
    # it even when RERANK_ENABLED is off by default. settings.rerank_enabled
    # now only decides the *default* behavior when a request doesn't specify
    # (see RetrieveNode.retrieve's use_reranker tri-state).
    reranker_client = RerankerClient(settings.reranker_service_url)
    stt_provider = SarvamSTTProvider(
        api_key=settings.sarvam_api_key,
        base_url=settings.sarvam_stt_base_url,
        model=settings.sarvam_stt_model,
    )
    llm_provider = OllamaCloudLLMProvider(
        api_key=settings.ollama_api_key,
        base_url=settings.ollama_cloud_base_url,
        model=settings.ollama_model,
    )

    corpus_centroid = await vector_db_provider.compute_corpus_centroid()

    return PipelineGraph(
        transcribe_node=TranscribeNode(stt_provider=stt_provider),
        input_filter_guardrail=InputFilterGuardrail(
            corpus_centroid=corpus_centroid,
            distance_threshold=settings.off_topic_distance_threshold,
            min_query_length_chars=settings.min_query_length_chars,
        ),
        embedding_client=embedding_client,
        retrieve_node=RetrieveNode(
            embedding_client=embedding_client,
            vector_db_provider=vector_db_provider,
            reranker_client=reranker_client,
            default_rerank_enabled=settings.rerank_enabled,
        ),
        confidence_floor_guardrail=ConfidenceFloorGuardrail(
            threshold=settings.confidence_floor_threshold
        ),
        generate_node=GenerateNode(llm_provider=llm_provider),
        groundedness_guardrail=GroundednessGuardrail(),
    )
