"""Embedding backends with graceful degradation.

Resolution order (EMBEDDING_BACKEND=auto):
  1. OpenAI-compatible API   - when OPENAI_API_KEY is set
  2. sentence-transformers   - when the package + model are available
  3. hashing fallback        - deterministic bag-of-subtokens projection
                               (no network, no dependencies, always works)

The hashing embedder is lexical rather than truly semantic; hybrid retrieval
(vector + BM25 + reranker) compensates, and any of the two neural backends
restores full semantic matching.
"""

from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from collections import Counter

from .tokenizer import tokenize


class BaseEmbedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class HashingEmbedder(BaseEmbedder):
    """Feature-hashing projection of tokens+subtokens. Deterministic and fast."""

    name = "hashing"

    def __init__(self, dim: int = 512):
        if dim < 64:
            raise ValueError("hashing embedding dim must be >= 64")
        self.dim = dim

    def _vectorize(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for tok, tf in counts.items():
            weight = 1.0 + math.log(tf)
            h1 = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h2 = hashlib.blake2b(b"#" + tok.encode("utf-8"), digest_size=8).digest()
            i1 = int.from_bytes(h1, "big") % self.dim
            i2 = int.from_bytes(h2, "big") % self.dim
            vec[i1] += weight
            vec[i2] += weight * 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]


OPENAI_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dim = OPENAI_MODEL_DIMS.get(model, 1536)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import httpx

        out: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                delay = 0.5
                last_err: Exception | None = None
                for attempt in range(self.max_retries):
                    try:
                        resp = client.post(
                            f"{self.base_url}/embeddings",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json={"model": self.model, "input": batch},
                        )
                        resp.raise_for_status()
                        data = resp.json()["data"]
                        out.extend(item["embedding"] for item in data)
                        if self.dim == 0:
                            self.dim = len(out[-1])
                        last_err = None
                        break
                    except (httpx.HTTPError, KeyError, ValueError) as exc:
                        last_err = exc
                        if attempt < self.max_retries - 1:
                            time.sleep(delay)
                            delay *= 2
                if last_err is not None:
                    raise RuntimeError(f"embedding request failed after retries: {last_err}") from last_err
        return out


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def get_embedder(
    backend: str,
    dim: int = 512,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
    openai_base_url: str = "https://api.openai.com/v1",
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> BaseEmbedder:
    backend = (backend or "auto").lower()
    if backend == "openai":
        if not openai_api_key:
            raise ValueError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY")
        return OpenAIEmbedder(openai_api_key, openai_model, openai_base_url)
    if backend == "hashing":
        return HashingEmbedder(dim)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(st_model)
    # auto
    if openai_api_key:
        return OpenAIEmbedder(openai_api_key, openai_model, openai_base_url)
    try:
        return SentenceTransformerEmbedder(st_model)
    except Exception:
        return HashingEmbedder(dim)
