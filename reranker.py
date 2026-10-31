"""Candidate reranking.

- HeuristicReranker: blends the fusion score with lexical-overlap features and
  exact identifier / symbol matches. Always available.
- CrossEncoderReranker: optional cross-encoder (sentence-transformers); enable
  with RERANKER=cross-encoder once the package is installed.
"""

from __future__ import annotations

import math

from .retriever import IDENT_RE
from .tokenizer import keyword_counts, split_identifier


class HeuristicReranker:
    name = "heuristic"

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        q_counts = keyword_counts(query)
        q_total = max(1, sum(q_counts.values()))
        raw_idents = [m for m in IDENT_RE.findall(query) if len(m) >= 4]
        for cand in candidates:
            text = cand.get("text", "")[:2000]
            c_counts = keyword_counts(text)
            overlap = sum(min(cnt, c_counts.get(t, 0)) for t, cnt in q_counts.items()) / q_total
            lowered = text.lower()
            ident_boost = 1.0 if any(i.lower() in lowered for i in raw_idents) else 0.0
            symbol_bonus = 0.0
            symbol = cand.get("symbol")
            if symbol:
                sym_tokens = {p.lower() for p in split_identifier(symbol)} | {symbol.lower()}
                if sym_tokens & set(q_counts):
                    symbol_bonus = 0.5
            base = min(cand["score"], 1.0)
            cand["score"] = 0.60 * base + 0.25 * overlap + 0.10 * ident_boost + 0.05 * symbol_bonus
            cand["scores"]["overlap"] = round(overlap, 4)
            cand["scores"]["ident_boost"] = ident_boost
            cand["scores"]["symbol_bonus"] = symbol_bonus
        candidates.sort(key=lambda c: -c["score"])
        return candidates[:top_k]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c.get("text", "")[:2000]] for c in candidates]
        scores = self._model.predict(pairs)
        for cand, ce in zip(candidates, scores, strict=False):
            ce_norm = self._sigmoid(float(ce))
            cand["scores"]["cross_encoder"] = round(ce_norm, 4)
            cand["score"] = 0.7 * ce_norm + 0.3 * min(cand["score"], 1.0)
        candidates.sort(key=lambda c: -c["score"])
        return candidates[:top_k]


def get_reranker(backend: str, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    backend = (backend or "heuristic").lower()
    if backend == "cross-encoder":
        try:
            return CrossEncoderReranker(model_name)
        except Exception:
            return HeuristicReranker()
    return HeuristicReranker()
