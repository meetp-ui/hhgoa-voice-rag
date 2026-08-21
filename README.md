# Voice RAG

Voice-first Hindi Q&A. Speak a question, get a grounded answer with sources,
or a clear refusal when the system isn't confident. Built on MS MARCO-XI as
a proof of concept — the same pattern applies to any Hindi knowledge base.

Pipeline: voice → STT → guardrail → hybrid retrieval → guardrail →
generation → guardrail → answer.

## Setup

Requires Docker, Node/npm, and API keys for Sarvam and Ollama Cloud.

```bash
git clone https://github.com/gmpaliwal07/voice-rag.git
cd voice-rag

# backend/.env — copy from .env.example, fill in:
# SARVAM_API_KEY, OLLAMA_API_KEY, QDRANT_URL, FLASK_SECRET_KEY

docker compose up -d          # qdrant + embedding + backend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

Index the dataset before first use:

```bash
cd backend
uv run python scripts/run_ingestion.py --limit 2000
```

Run tests:

```bash
cd backend
uv run pytest -v
uv run ruff check app/ tests/
uv run mypy app/
```

Optional reranker (off by default — see [why](#reranker)):

```bash
sed -i 's/RERANK_ENABLED=false/RERANK_ENABLED=true/' backend/.env
docker compose --profile rerank up -d --build backend
```

## Stack, and why

| Layer      | Choice                              | Why                                                                                                                   |
| ---------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| STT        | Sarvam AI                           | Built for Indic languages, not a Western-first tool with STT bolted on                                                |
| LLM        | Ollama Cloud,`gpt-oss:120b-cloud` | Reliable structured JSON output — needed for the`grounded`/`confidence`/`citations` fields the guardrails read |
| Vector DB  | Qdrant                              | Native hybrid search (dense + sparse fusion), quantization, self-hostable                                             |
| Embeddings | `BAAI/bge-m3` via TEI             | Strong multilingual model, real Devanagari support                                                                    |
| Reranker   | `bge-reranker-v2-m3` via TEI      | Opt-in only — see latency section below                                                                              |
| Backend    | Flask + ASGI (`uvicorn` workers)  | See "gevent bug" below — this wasn't the first choice, it's the fix                                                  |
| Frontend   | Next.js + TypeScript                | Minimal single-page voice UI                                                                                          |
| Containers | Docker Compose                      | One network, colocated for latency                                                                                    |

## Guardrails

Three checks, three different failure points. The point isn't "add
guardrails" — it's catching hallucination at every stage it can enter.

| Stage                          | Checks                                                                     | Threshold             |
| ------------------------------ | -------------------------------------------------------------------------- | --------------------- |
| `guardrail_input`            | Empty/garbage transcript, off-topic (cosine distance from corpus centroid) | distance > 0.5        |
| `guardrail_confidence_floor` | Retrieval too weak to trust — LLM never gets called                       | `top1_score` ≤ 0.5 |
| `guardrail_output`           | LLM self-reports its own answer isn't grounded in context                  | `grounded: false`   |

The 0.5 thresholds came from measuring real score distributions against the
live index, not guessed. In-domain distances ranged 0.33–0.52, off-topic
ranged 0.48–0.55 — they overlap, because this corpus is broad general QnA,
not a narrow topic. That's stated plainly, not hidden.

Every rejection logs a structured line (`request_id`, `stage`,
`reason_code`) — the reason code used to only exist in the HTTP response,
invisible to log monitoring. Fixed.

## Retrieval

Hybrid dense (bge-m3) + sparse (BM25) search, fused via Qdrant's RRF. Dense
vectors are int8-quantized. HNSW `ef_search=128` — chosen from a real sweep
(16/64/128/256): recall plateaus at 128, no gain going higher, so higher
just costs latency for nothing.

<a name="reranker"></a>
**Reranker is opt-in, off by default.** On CPU-only hardware it's ~20x
slower than everything else in the pipeline combined — see
[retrieval_latency.md](backend/reports/retrieval_latency.md) for the real
numbers. Toggle it per-request via `use_reranker: true/false` in the
`/query` body, or set `RERANK_ENABLED=true` server-wide.

Chunking: three strategies (structural, semantic, fixed-overlap), routed
by passage length. Full comparison and why semantic underperforms on this
corpus: [chunking_eval.md](backend/reports/chunking_eval.md).

## Latency

200ms target applies to retrieval (chunking + vector search) — the part
this project controls. STT and LLM generation are separate, vendor-bound
stages, reported honestly, not folded into the same number.

| Path                   | P50      | P70      | P100     |
| ---------------------- | -------- | -------- | -------- |
| Retrieval, no rerank   | 152.5ms  | 162.8ms  | 309.9ms  |
| Retrieval, with rerank | 1872.9ms | 2106.0ms | 2953.6ms |

Full breakdown, methodology, and every optimization attempt:
[retrieval_latency.md](backend/reports/retrieval_latency.md).

## Response shape

```json
{
  "status": "ok | rejected | error",
  "transcript": "...",
  "answer": "...",
  "confidence": 0.98,
  "sources": [{"chunk_id": "...", "text": "..."}],
  "reranked": false,
  "guardrail": {"stage": "...", "reason_code": "...", "message": "...", "score": 0.5},
  "timings": {"transcribe_ms": null, "retrieve_ms": 32.5, "generate_ms": 1101.3, "total_ms": 1133.8}
}
```

## Structure

```
backend/
  app/
    core/        settings, logging, exceptions, telemetry
    api/         /health, /query routes + schemas
    pipeline/    graph.py (orchestration), nodes/ (transcribe, retrieve, generate)
    guardrails/  input_filter, confidence_floor, groundedness
    providers/   Sarvam, Ollama Cloud, Qdrant — swappable behind interfaces
    ingestion/   dataset loader, chunking, embedding, indexer
  tests/         47 tests, no live-service dependency
  reports/       retrieval_latency.md, chunking_eval.md
frontend/
  src/app/       single-page voice UI
  src/api/       typed fetch client
```

## Dataset

MS MARCO-XI (Hindi). It's QnA format — one query, 10 candidate passages,
relevance labels — not a document corpus. Indexed subset: 2,144 chunks
(bounded from the full 778K for local dev; documented, not hidden).
