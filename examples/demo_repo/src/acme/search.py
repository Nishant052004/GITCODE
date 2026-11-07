"""Product search: a small in-memory inverted index with tf scoring."""
import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    """Maps each term to the documents containing it, scored by tf-idf."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_lengths: dict[str, int] = {}

    def add_document(self, doc_id: str, text: str) -> None:
        """Index one document; re-adding the same id replaces its terms."""
        self.remove_document(doc_id)
        tokens = _tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for term, tf in Counter(tokens).items():
            self.postings[term][doc_id] = tf

    def remove_document(self, doc_id: str) -> None:
        self.doc_lengths.pop(doc_id, None)
        for postings in self.postings.values():
            postings.pop(doc_id, None)

    def build_index(self, documents: dict[str, str]) -> None:
        """Bulk-index a mapping of doc_id -> text (products, descriptions)."""
        self.postings.clear()
        self.doc_lengths.clear()
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, {}))
        if df == 0:
            return 0.0
        n_docs = len(self.doc_lengths)
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def query(self, q: str, limit: int = 10) -> list[tuple[str, float]]:
        """Score documents by summed tf-idf over the query terms."""
        scores: dict[str, float] = {}
        for term in set(_tokenize(q)):
            idf = self._idf(term)
            for doc_id, tf in self.postings.get(term, {}).items():
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:limit]
