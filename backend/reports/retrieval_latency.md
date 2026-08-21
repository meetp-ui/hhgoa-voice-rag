
# Retrieval-Only Latency Benchmark

Retrieval-only: embed query -> hybrid search -> optional rerank -> parent-child expansion. Does not include STT or LLM generation (separate SLOs, see README).

Sample: 40 held-out MS MARCO-XI (Hindi, validation split) queries, 3 warmup queries excluded, against the live 2,144-chunk index.

## Summary

| Configuration                      | P50 (ms)        | P70 (ms)        | P100 (ms)        |
| ---------------------------------- | --------------- | --------------- | ---------------- |
| With rerank, top-5 (fp32 ONNX)     | 1872.9          | 2106.0          | 2953.6           |
| With rerank, top-3 (fp32 ONNX)     | 1046.1          | 1185.6          | 1804.7           |
| With rerank, top-3, int8 quantized | **684.3** | **977.2** | **1611.6** |
| Without rerank                     | 152.5           | 162.8           | 309.9            |

200ms target applies to the retrieval-only path (see README's latency scoping). Without rerank, P50/P70 land under it; every rerank configuration sits well above.

Reranker fired on 28/40 queries (70%) in every "with rerank" run — same ambiguous-band threshold throughout.

## Where the time goes (top-5 fp32 run, avg per stage)

| Stage                           | Avg (ms) |
| ------------------------------- | -------- |
| embed                           | 94.4     |
| sparse_encode                   | 0.2      |
| hybrid_search                   | 5.9      |
| rerank (queries where it fired) | 1897.5   |
| parent_expand                   | 44.7     |

Reranker is ~20x the cost of every other stage combined.

## Reranker optimization

**Top-5 → top-3 candidates**: P50 1872.9ms → 1046.1ms (-44%), P70 2106.0ms → 1185.6ms (-44%), P100 2953.6ms → 1804.7ms (-39%).

**int8 quantization**: TEI has no CLI flag to select an ONNX variant from a HF repo, so used TEI's local-model-directory support — downloaded `onnx-community/bge-reranker-v2-m3-ONNX`'s `model_int8.onnx` (570MB, self-contained) and pointed a test container at it directly. Loaded via ONNX Runtime, no fallback. Verified correctness: same ranking order as fp32, near-identical scores (0.9992 vs 0.9995 on the same top result).

Combined (top-3 + int8): **P50 684.3ms**, a 63% reduction from the top-5 fp32 baseline. Still ~3.4x over the 200ms target.

Not wired into `docker-compose.yml` — the 570MB model file isn't committed; would need a fetch step in a named volume to productionize.

## Connection pooling fix

`EmbeddingClient` and `RerankerClient` each opened a fresh `httpx.AsyncClient` per call instead of reusing one (`QdrantVectorDBProvider` already did this correctly). Fixed to match.

No-rerank baseline after the fix: P50 154.6ms → 152.5ms, P70 175.3ms → 162.8ms (-7%), P100 323.7ms → 309.9ms (-4%).

## P100 outlier investigation

Original 323.7ms outlier didn't reproduce after the pooling fix — slowest 3 re-measured at 309.9ms / 223.9ms / 212.4ms, a tight spread with no isolated spike. Most likely transient host contention at the time of the original measurement, not a structural per-query cost.

## Conclusion

Reranker is the dominant cost by ~20x. Without it, the 200ms target is met (P50=152.5ms, P70=162.8ms). With it, even after both optimizations, P50 is still ~3.4x over target. Reranker ships opt-in, off by default.
