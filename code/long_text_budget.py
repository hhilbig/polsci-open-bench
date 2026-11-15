"""Shared length cap for the long-text tasks.

Hive runs use max_model_len 16,384 with 256 output tokens and a 128-token
margin, so the prompt may use at most 16,000 tokens, and the runner refuses
(never truncates) longer prompts. Documents are therefore capped at 10,000
tokens, counted as the larger of OpenAI's o200k_base and cl100k_base
encodings. The cl100k count stands in for the less efficient tokenizers of
older open models: for Georgian and Armenian speeches it is up to 7 times the
o200k count, so an o200k-only cap would let through prompts that older models
cannot read. The remaining headroom (about 5,500 tokens after the task
instructions) covers tokenizers somewhat less efficient than cl100k.

Long-text tasks drop documents above the cap instead of truncating them, so
every model reads each document whole, as the human coders did.
"""
import tiktoken

MAX_DOC_TOKENS = 10_000
_ENCODINGS = [tiktoken.get_encoding("o200k_base"), tiktoken.get_encoding("cl100k_base")]


def doc_tokens(text: str) -> int:
    return max(len(enc.encode(text, disallowed_special=())) for enc in _ENCODINGS)
