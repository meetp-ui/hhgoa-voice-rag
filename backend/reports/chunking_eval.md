
# Chunking Strategy Evaluation

Sample: 38 query groups, 380 passages, `validation` split (MS MARCO-XI, Hindi).

| Strategy      | Recall@5 |
| ------------- | -------- |
| structural    | 0.842    |
| semantic      | 0.789    |
| fixed_overlap | 0.842    |

## Root cause: why semantic scores lower, and why structural/fixed_overlap tie

**1. `structural` and `fixed_overlap` are effectively identical on this corpus.** MS MARCO-XI passages are short: token-length distribution over the sample is min=12, avg=100.2, p50=89, p90=138, p99=213, max=2449. Only 2 of 380 passages (0.5%) exceed the 256-token fixed_overlap threshold, and only those same 2 produce more than one window. The two strategies produce nearly identical chunk sets here, so the matching 0.842 recall@5 is expected, not a coincidence — this eval currently compares semantic against structural/fixed-overlap as a pair, not three independently distinct strategies. A corpus with longer documents would be needed to exercise fixed_overlap's splitting and structural's document-boundary logic differently.

**2. `semantic` over-fragments already-coherent short passages.** 85.8% of passages (326/380) split into more than one semantic chunk, averaging 3.21 chunks per passage (up to 14 for one). Average chunk length drops from 100.2 tokens (the whole passage) to 31.3 tokens per chunk (median 25, some as small as 1 token).

The 0.35 cosine-distance threshold — tuned without a corpus-specific baseline — is splitting on ordinary sentence-to-sentence variance within a single coherent passage, not genuine topic shifts. These are short, single-topic answer snippets with no real internal discontinuities to detect. The result is mostly single-sentence fragments carrying less signal than the full passage would — the more likely explanation for the recall drop than "more candidate chunks should only help."

**Production implication.** `strategy_router.py` gates semantic/fixed_overlap behind a 256-token structural threshold, so this fallback path fires on the same 0.5% of passages — the production ingestion path is dominated by `structural` regardless of this result. This matters if a longer-document corpus gets ingested later and semantic chunking starts firing on a meaningful fraction of content: the 0.35 threshold should be retuned against that corpus's own baseline first, not carried over unchanged from this result.
