# TEI serving bge-m3 dense embeddings (same image/args as docker-compose.yml).
# Persist /data via a Railway volume so the model isn't re-downloaded on redeploy.
FROM ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.3

CMD [\
  "--model-id", "BAAI/bge-m3", \
  "--pooling", "cls", \
  "--max-batch-tokens", "4096", \
  "--max-client-batch-size", "8", \
  "--max-concurrent-requests", "32", \
  "--port", "80" \
]

EXPOSE 80
