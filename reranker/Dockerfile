# TEI serving bge-reranker-v2-m3 (optional, off by default).
# Persist /data via a Railway volume so the model isn't re-downloaded on redeploy.
FROM ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.3

CMD [\
  "--model-id", "onnx-community/bge-reranker-v2-m3-ONNX", \
  "--max-batch-tokens", "4096", \
  "--max-client-batch-size", "8", \
  "--max-concurrent-requests", "32", \
  "--port", "80" \
]

EXPOSE 80
