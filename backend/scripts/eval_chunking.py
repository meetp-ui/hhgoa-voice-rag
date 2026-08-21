#!/usr/bin/env python3
"""Evaluates structural, semantic, and fixed_overlap chunking independently
(not through strategy_router) via in-memory recall@k over a sample of
MS MARCO-XI validation queries. Requires the embedding service, not Qdrant.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.chunking import fixed_overlap, semantic, structural
from app.ingestion.embedding import EmbeddingClient
from app.ingestion.loaders.msmarco_xi import load_hindi_passages
from app.ingestion.types import Chunk, Passage

_logger = get_logger(__name__)

PASSAGE_CAP = 750
DEFAULT_K = 5


@dataclass
class QueryGroup:
    query_id: int
    passages: list[Passage] = field(default_factory=list)

    @property
    def relevant_doc_ids(self) -> set[str]:
        return {p.doc_id for p in self.passages if p.is_selected}


def _sample_query_groups(split: str, passage_cap: int) -> list[QueryGroup]:
    groups: dict[int, QueryGroup] = {}
    total_passages = 0
    for passage in load_hindi_passages(split=split):
        group = groups.setdefault(passage.query_id, QueryGroup(query_id=passage.query_id))
        group.passages.append(passage)
        total_passages += 1
        if total_passages >= passage_cap:
            break
    # A query with no relevance judgment can't contribute to recall@k.
    return [g for g in groups.values() if g.relevant_doc_ids]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _recall_at_k_for_strategy(
    strategy_name: str,
    groups: list[QueryGroup],
    embedding_client: EmbeddingClient,
    k: int,
) -> float:
    all_passages = [p for g in groups for p in g.passages]

    chunks: list[Chunk] = []
    if strategy_name == "structural":
        for passage in all_passages:
            chunks.extend(structural.chunk(passage))
    elif strategy_name == "fixed_overlap":
        for passage in all_passages:
            chunks.extend(fixed_overlap.chunk(passage))
    elif strategy_name == "semantic":

        async def embed_fn(texts: list[str]) -> list[list[float]]:
            return await embedding_client.embed(texts)

        for passage in all_passages:
            chunks.extend(await semantic.chunk(passage, embed_fn=embed_fn))
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    chunk_texts = [c.text for c in chunks]
    chunk_vectors = await embedding_client.embed(chunk_texts)

    query_texts = [g.passages[0].query_text for g in groups]
    query_vectors = await embedding_client.embed(query_texts)

    hits = 0
    for group, query_vector in zip(groups, query_vectors, strict=True):
        scored = [
            (chunk.doc_id, _cosine_similarity(query_vector, chunk_vector))
            for chunk, chunk_vector in zip(chunks, chunk_vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_k_doc_ids = {doc_id for doc_id, _ in scored[:k]}
        if top_k_doc_ids & group.relevant_doc_ids:
            hits += 1

    return hits / len(groups) if groups else 0.0


async def run(*, split: str, passage_cap: int, k: int, report_path: str) -> None:
    settings = get_settings()
    embedding_client = EmbeddingClient(settings.embedding_service_url)

    groups = _sample_query_groups(split, passage_cap)
    total_passages = sum(len(g.passages) for g in groups)
    _logger.info(
        "eval_chunking_sample",
        extra={"query_groups": len(groups), "total_passages": total_passages},
    )

    results: dict[str, float] = {}
    for strategy_name in ("structural", "semantic", "fixed_overlap"):
        start = time.monotonic()
        recall = await _recall_at_k_for_strategy(strategy_name, groups, embedding_client, k)
        results[strategy_name] = recall
        _logger.info(
            "eval_chunking_strategy_done",
            extra={
                "strategy": strategy_name,
                "recall_at_k": recall,
                "elapsed_s": round(time.monotonic() - start, 1),
            },
        )

    _write_report(
        report_path,
        split=split,
        k=k,
        query_groups=len(groups),
        total_passages=total_passages,
        results=results,
    )


def _write_report(
    path: str,
    *,
    split: str,
    k: int,
    query_groups: int,
    total_passages: int,
    results: dict[str, float],
) -> None:
    lines = [
        "# Chunking Strategy Evaluation",
        "",
        f"Sample: {query_groups} query groups, {total_passages} passages, "
        f"`{split}` split (MS MARCO-XI, Hindi).",
        "",
        f"| Strategy | Recall@{k} |",
        "|---|---|",
    ]
    for strategy_name, recall in results.items():
        lines.append(f"| {strategy_name} | {recall:.3f} |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--passage-cap", type=int, default=PASSAGE_CAP)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--report-path", default="reports/chunking_eval.md")
    args = parser.parse_args()

    configure_logging()
    asyncio.run(
        run(split=args.split, passage_cap=args.passage_cap, k=args.k, report_path=args.report_path)
    )


if __name__ == "__main__":
    main()
