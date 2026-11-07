"""Pure-Python BM25 (Okapi) index with JSON persistence.

Replaces Elasticsearch for the local/dev profile while keeping the same
operations (add / remove / search / save / load) so a cluster-backed
implementation can drop in behind the same interface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .tokenizer import tokenize


class BM25Index:
    kind = "bm25"

    def __init__(self, path: Path | None = None, k1: float = 1.5, b: float = 0.75):
        self.path = path
        self.k1 = k1
        self.b = b
        # chunk_id -> {"len": doc_len, "tfs": {term: tf}}
        self._docs: dict[str, dict] = {}
        if path is not None and path.exists():
            self.load()

    # -- persistence --

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._docs), encoding="utf-8")

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._docs = json.loads(self.path.read_text(encoding="utf-8"))

    # -- mutation --

    def add(self, docs: dict[str, str]) -> None:
        for chunk_id, text in docs.items():
            tokens = tokenize(text)
            tfs: dict[str, int] = {}
            for t in tokens:
                tfs[t] = tfs.get(t, 0) + 1
            self._docs[chunk_id] = {"len": len(tokens), "tfs": tfs}

    def remove(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._docs.pop(cid, None)

    # -- search --

    def _avgdl(self) -> float:
        if not self._docs:
            return 1.0
        return sum(d["len"] for d in self._docs.values()) / len(self._docs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if not self._docs:
            return []
        q_terms = [t for t in tokenize(query) if t]
        if not q_terms:
            return []
        avgdl = self._avgdl()
        n_docs = len(self._docs)
        scores: dict[str, float] = {}
        for term in set(q_terms):
            df = sum(1 for d in self._docs.values() if term in d["tfs"])
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for cid, doc in self._docs.items():
                tf = doc["tfs"].get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc["len"] / avgdl)
                scores[cid] = scores.get(cid, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return ranked

    def count(self) -> int:
        return len(self._docs)
