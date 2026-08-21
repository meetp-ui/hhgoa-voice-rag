# syntax=docker/dockerfile:1

# builder
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# runtime 
FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY --from=builder /usr/local /usr/local
WORKDIR /app
COPY app ./app

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface /app/.cache/datasets && chown -R app:app /app

USER app

# Railway injects PORT; fall back to 8000 for local docker runs.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT is expanded at runtime.
CMD gunicorn --worker-class uvicorn.workers.UvicornWorker --workers 2 --bind 0.0.0.0:${PORT} app.main:asgi_app
