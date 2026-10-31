"""Hybrid retrieval: vector (semantic) + BM25 (lexical), fused.

Fusion model per candidate:
    score = w_vec * normalized_cosine
          + w_bm25 * normalized_bm25
          + 0.15 * agreement bonus (retrieved by both sources)
          + recency_boost * recency (newer last_modified scores higher)

Identifier-heavy queries (function names, error codes, stack traces) tilt the
weights toward BM25, since exact token evidence dominates there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..config import Settings
from .bm25_index import BM25Index
from .embedder import BaseEmbedder
from .metadata_db import MetadataDB
from .tokenizer import STOPWORDS, split_identifier
from .vector_store import NumpyVectorStore

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
PATH_RE = re.compile(r"[\w\-./\\]+\.(?:py|js|jsx|ts|tsx|java|go|rb|rs|c|cpp|h|cs|php|md)\b")
STACK_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|File \"|line \d+|\bat .+\(.+:\d+\)|Error: \w+"
)
QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "does",
    "did",
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "can",
    "best",
    "way",
    "ways",
    "need",
    "want",
    "use",
    "using",
    "used",
    "any",
    "all",
    "some",
    "possible",
    "recommended",
    "find",
    "there",
    "get",
}


@dataclass
class QueryFeatures:
    raw: str
    identifiers: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    has_stack_trace: bool = False
    normalized: str = ""


def analyze_query(query: str) -> QueryFeatures:
    feats = QueryFeatures(raw=query)
    seen: set[str] = set()
    extra: list[str] = []
    for m in IDENT_RE.findall(query):
        low = m.lower()
        if low in STOPWORDS or low in QUESTION_WORDS or low in seen:
            continue
        seen.add(low)
        feats.identifiers.append(m)
        extra.extend(p for p in split_identifier(m) if p.lower() not in seen and len(p) > 1)
        seen.update(p.lower() for p in split_identifier(m))
    feats.paths = list(dict.fromkeys(PATH_RE.findall(query)))
    feats.has_stack_trace = bool(STACK_TRACE_RE.search(query))
    feats.normalized = " ".join([query, *feats.identifiers, *extra, *feats.paths])
    return feats


def _recency_score(last_modified: str | None, now: datetime | None = None) -> float:
    if not last_modified:
        return 0.0
    try:
        dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return max(0.0, 1.0 - days / 365.0)


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class Retriever:
    def __init__(
        self,
        settings: Settings,
        db: MetadataDB,
        embedder: BaseEmbedder,
        vectors: NumpyVectorStore,
        bm25: BM25Index,
    ):
        self.settings = settings
        self.db = db
        self.embedder = embedder
        self.vectors = vectors
        self.bm25 = bm25

    def retrieve(self, query: str, top_n: int | None = None) -> list[dict]:
        settings = self.settings
        top_n = top_n or settings.candidate_pool
        feats = analyze_query(query)

        vec_hits = self.vectors.search(self.embedder.embed_query(query), settings.vector_top_n)
        lex_hits = self.bm25.search(feats.normalized, settings.bm25_top_m)

        vec_scores = {cid: s for (cid, s) in vec_hits}
        lex_scores = {cid: s for (cid, s) in lex_hits}
        all_ids = list(dict.fromkeys([*vec_scores.keys(), *lex_scores.keys()]))
        if not all_ids:
            return []

        rows = {r["chunk_id"]: r for r in self.db.get_chunks(all_ids)}
        vec_norm = _minmax(vec_scores)
        lex_norm = _minmax(lex_scores)

        wv, wb = settings.weight_vector, settings.weight_bm25
        if feats.identifiers or feats.has_stack_trace:
            wb = max(wb, 0.65)
            wv = 1.0 - wb
        else:
            total = (wv + wb) or 1.0
            wv, wb = wv / total, wb / total

        now = datetime.now(UTC)
        candidates: list[dict] = []
        for cid in all_ids:
            row = rows.get(cid)
            if row is None:
                continue  # vector/BM25 entry without DB metadata (stale index)
            vs = vec_norm.get(cid, 0.0)
            ls = lex_norm.get(cid, 0.0)
            agreement = 0.15 if (cid in vec_scores and cid in lex_scores) else 0.0
            recency = _recency_score(row.get("last_modified"), now)
            score = wv * vs + wb * ls + agreement + settings.recency_boost * recency
            candidates.append(
                {
                    "chunk_id": cid,
                    "file_path": row["file_path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "language": row["language"],
                    "symbol": row["symbol"],
                    "kind": row["kind"],
                    "commit_id": row["commit_id"],
                    "author": row["author"],
                    "last_modified": row["last_modified"],
                    "text": row["text"],
                    "summary": row["summary"],
                    "score": score,
                    "scores": {
                        "vector": round(vec_scores.get(cid, 0.0), 4),
                        "bm25": round(lex_scores.get(cid, 0.0), 4),
                        "vector_norm": round(vs, 4),
                        "bm25_norm": round(ls, 4),
                        "agreement": agreement,
                        "recency": round(recency, 4),
                    },
                }
            )
        candidates.sort(key=lambda c: -c["score"])
        return candidates[:top_n]
