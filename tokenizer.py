"""Shared tokenization used by the BM25 index, hashing embedder and reranker.

Splits out identifiers, then further splits snake_case / camelCase into
subtokens so a query for "charge invoice" can match `charge_invoice`.
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "how",
        "what",
        "why",
        "when",
        "where",
        "does",
        "do",
        "did",
        "we",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "they",
        "there",
        "here",
        "as",
        "at",
        "by",
        "from",
        "not",
        "no",
        "if",
        "then",
        "else",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "my",
        "our",
        "your",
        "their",
        "about",
        "into",
        "over",
        "under",
        "again",
        "more",
        "most",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "don",
        "now",
    ]
)


def split_identifier(token: str) -> list[str]:
    parts = token.split("_")
    out: list[str] = []
    for part in parts:
        out.extend(p for p in _CAMEL_RE.split(part) if p)
    return out


def tokenize(text: str) -> list[str]:
    """Lowercased tokens: full identifiers plus their subtokens."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text or ""):
        lower = match.lower()
        tokens.append(lower)
        if "_" in match or any(c.isupper() for c in match):
            parts = [p.lower() for p in split_identifier(match) if len(p) > 1]
            tokens.extend(p for p in parts if p != lower)
    return tokens


def keywords(text: str) -> list[str]:
    """Content-bearing tokens (stopwords dropped) - used for overlap scoring."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def keyword_counts(text: str) -> Counter:
    return Counter(keywords(text))
