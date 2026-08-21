"""Token counting via bge-m3's actual tokenizer, not a word-count heuristic
-- chunk size budgets need to match what the model actually sees.
"""

from __future__ import annotations

from functools import lru_cache

from tokenizers import Tokenizer

_TOKENIZER_MODEL_ID = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(_TOKENIZER_MODEL_ID)


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def encode_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Returns (token_ids, char_offsets) for `text`, so windows over token ids
    can be sliced back to substrings of the *original* text rather than
    decoded (decoding loses exact whitespace/formatting)."""
    encoding = get_tokenizer().encode(text, add_special_tokens=False)
    return encoding.ids, encoding.offsets
