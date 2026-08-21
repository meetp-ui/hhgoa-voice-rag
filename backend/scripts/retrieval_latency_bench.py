#!/usr/bin/env python3
"""Benchmarks retrieval latency (embed, search, rerank, parent expand)
against the live Compose stack, over held-out MS MARCO-XI validation
queries. Warmup queries are excluded so cold connection setup doesn't
skew the numbers. Writes P50/P70/P100 and a per-stage breakdown to
reports/retrieval_latency.md.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.embedding import EmbeddingClient
from app.ingestion.loaders.msmarco_xi import load_hindi_passages
from app.pipeline.nodes.retrieve import RetrievalResult, RetrieveNode
from app.pipeline.reranker import RerankerClient
from app.providers.vector_db.qdrant_client import QdrantVectorDBProvider

_logger = get_logger(__name__)

WARMUP_QUERIES = 3
LATENCY_TARGET_MS = 200


def _load_queries(n: int) -> list[str]:
    seen: dict[int, str] = {}
    for passage in load_hindi_passages(split="validation"):
        seen[passage.query_id] = passage.query_text
        if len(seen) >= n:
            break
    return list(seen.values())


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return sorted_values[idx]


async def run(*, num_queries: int, report_path: str) -> None:
    settings = get_settings()
    node = RetrieveNode(
        embedding_client=EmbeddingClient(settings.embedding_service_url),
        vector_db_provider=QdrantVectorDBProvider(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
        ),
        reranker_client=RerankerClient(settings.reranker_service_url),
        rerank_candidate_count=settings.rerank_top_k,
    )

    queries = _load_queries(num_queries + WARMUP_QUERIES)
    warmup_queries, bench_queries = queries[:WARMUP_QUERIES], queries[WARMUP_QUERIES:]

    _logger.info("warmup_start", extra={"count": len(warmup_queries)})
    for query in warmup_queries:
        await node.retrieve(query)

    _logger.info("benchmark_start", extra={"count": len(bench_queries)})
    total_latencies_ms: list[float] = []
    stage_totals_ms: dict[str, list[float]] = {}
    rerank_fired_count = 0

    for i, query in enumerate(bench_queries):
        t0 = time.perf_counter()
        result: RetrievalResult = await node.retrieve(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_latencies_ms.append(elapsed_ms)
        for stage, ms in result.stage_timings_ms.items():
            stage_totals_ms.setdefault(stage, []).append(ms)
        if result.reranked:
            rerank_fired_count += 1

        _logger.info(
            "benchmark_query_done",
            extra={
                "index": i,
                "elapsed_ms": round(elapsed_ms, 2),
                "reranked": result.reranked,
                "top1_score": result.top1_score,
            },
        )

    total_latencies_ms.sort()
    n = len(total_latencies_ms)
    p50 = _percentile(total_latencies_ms, 0.50)
    p70 = _percentile(total_latencies_ms, 0.70)
    p100 = total_latencies_ms[-1]

    stage_avgs_ms = {
        stage: sum(values) / len(values) for stage, values in stage_totals_ms.items()
    }

    _write_report(
        report_path,
        n=n,
        p50=p50,
        p70=p70,
        p100=p100,
        stage_avgs_ms=stage_avgs_ms,
        rerank_fired_count=rerank_fired_count,
    )

    _logger.info(
        "benchmark_complete",
        extra={"n": n, "p50_ms": round(p50, 2), "p70_ms": round(p70, 2), "p100_ms": round(p100, 2)},
    )


def _write_report(
    path: str,
    *,
    n: int,
    p50: float,
    p70: float,
    p100: float,
    stage_avgs_ms: dict[str, float],
    rerank_fired_count: int,
) -> None:
    lines = [
        "# Retrieval-Only Latency Benchmark",
        "",
        "Retrieval-only: embed query -> hybrid search -> optional rerank -> "
        "parent-child expansion. Does not include STT or LLM generation "
        "(separate SLOs, see README's latency-scoping decision).",
        "",
        f"Sample: {n} held-out MS MARCO-XI (Hindi, validation split) queries, "
        f"{WARMUP_QUERIES} warmup queries excluded. Reranker fired on "
        f"{rerank_fired_count}/{n} queries ({100 * rerank_fired_count / n:.0f}%).",
        "",
        "## Latency (ms)",
        "",
        "| Percentile | Latency (ms) |",
        "|---|---|",
        f"| P50 | {p50:.1f} |",
        f"| P70 | {p70:.1f} |",
        f"| P100 (max) | {p100:.1f} |",
        "",
        f"200ms target applies to this retrieval-only path (see README's "
        f"latency scoping). P100 here is {p100:.1f}ms.",
        "",
        "## Where the time goes (average per stage, ms)",
        "",
        "| Stage | Avg (ms) |",
        "|---|---|",
    ]
    for stage in ("embed", "sparse_encode", "hybrid_search", "rerank", "parent_expand"):
        if stage in stage_avgs_ms:
            note = " (only over queries where it fired)" if stage == "rerank" else ""
            lines.append(f"| {stage}{note} | {stage_avgs_ms[stage]:.1f} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-queries", type=int, default=40)
    parser.add_argument("--report-path", default="reports/retrieval_latency.md")
    args = parser.parse_args()

    configure_logging()
    asyncio.run(run(num_queries=args.num_queries, report_path=args.report_path))


if __name__ == "__main__":
    main()