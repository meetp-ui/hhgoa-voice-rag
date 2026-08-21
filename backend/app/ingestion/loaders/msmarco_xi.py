"""Loader for ai4bharat/MSMARCO-XI, Hindi split. Not a document corpus --
MS MARCO QnA format, one row per query with 10 candidate passages each.

Each Parquet file is one huge row group, so streaming/remote reads OOM.
Downloads to local disk first, then reads with DuckDB locally instead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import duckdb
import httpx

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.ingestion.types import Passage

_logger = get_logger(__name__)

_REPO_ID = "ai4bharat/MSMARCO-XI"
_REMOTE_PATHS = {
    "train": "train/hintrain.parquet",
    "validation": "validation/hinval.parquet",
}
_LANGUAGE = "hi"

_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB  plain byte streaming, low memory
_DUCKDB_MEMORY_LIMIT = "1500MB"
_DUCKDB_BATCH_SIZE = 256


def _local_path(split: Literal["train", "validation"]) -> Path:
    cache_dir = Path(get_settings().dataset_cache_dir)
    return cache_dir / f"msmarco_xi_hin_{split}.parquet"


def _ensure_local_file(split: Literal["train", "validation"]) -> Path:
    """Downloads the split's Parquet file to the local cache if not already
    present. Atomic: downloads to a .part file and renames on success, so a
    prior interrupted download can't be mistaken for a complete one."""
    final_path = _local_path(split)
    if final_path.exists():
        return final_path

    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_suffix(".parquet.part")
    url = f"https://huggingface.co/datasets/{_REPO_ID}/resolve/main/{_REMOTE_PATHS[split]}"

    _logger.info("dataset_download_start", extra={"split": split, "url": url})
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            downloaded = 0
            with open(part_path, "wb") as f:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
    except httpx.HTTPError as exc:
        if part_path.exists():
            part_path.unlink()
        raise ExternalServiceError(
            f"Failed to download {split} split from Hugging Face: {exc}",
            service="huggingface_hub",
        ) from exc

    os.replace(part_path, final_path)
    _logger.info("dataset_download_complete", extra={"split": split, "bytes": downloaded})
    return final_path


def load_hindi_passages(
    split: Literal["train", "validation"] = "train",
) -> Iterator[Passage]:
    """Yields one Passage per (query, candidate passage) pair, streaming
    from the locally-cached Parquet file via DuckDB."""
    local_path = _ensure_local_file(split)

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    result = connection.execute(
        f"""
        SELECT
            query_id,
            query,
            passages.Translated_passages AS translated_passages,
            passages.is_selected AS is_selected
        FROM read_parquet('{local_path}')
        """
    )

    reader = result.fetch_record_batch(rows_per_batch=_DUCKDB_BATCH_SIZE)
    for batch in reader:
        for row in batch.to_pylist():
            query_id = row["query_id"]
            query_text = row["query"]
            translated_passages = row["translated_passages"]
            is_selected = row["is_selected"]

            for passage_index, (text, selected) in enumerate(
                zip(translated_passages, is_selected, strict=True)
            ):
                yield Passage(
                    doc_id=f"{query_id}_{passage_index}",
                    query_id=query_id,
                    passage_index=passage_index,
                    text=text,
                    is_selected=bool(selected),
                    language=_LANGUAGE,
                    query_text=query_text,
                )
