"""Vector stores.

- NumpyVectorStore: cosine similarity over an in-memory matrix, persisted to a
  single .npz file. Zero external services - the default for local dev.
- QdrantVectorStore: same interface over qdrant-client in local path mode
  (no server needed) or a URL. Enable with VECTOR_BACKEND=qdrant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NumpyVectorStore:
    kind = "numpy"

    def __init__(self, dim: int, path: Path | None = None):
        self.dim = dim
        self.path = path
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        if path is not None and path.exists():
            self._load(path)

    # -- persistence --

    def _load(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        self._ids = [str(i) for i in json.loads(str(data["ids"]))]
        self._matrix = data["matrix"].astype(np.float32)
        self._index = {cid: i for i, cid in enumerate(self._ids)}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            ids=np.array(json.dumps(self._ids)),
            matrix=self._matrix,
        )

    # -- mutation --

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        if not ids:
            return
        self.remove(ids)
        mat = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        if self._matrix.shape[0] == 0 and self._matrix.shape[1] != mat.shape[1]:
            self.dim = mat.shape[1]
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._matrix = np.vstack([self._matrix, mat])
        start = len(self._ids)
        self._ids.extend(ids)
        for i, cid in enumerate(self._ids[start:]):
            self._index[cid] = start + i

    def remove(self, ids: list[str]) -> None:
        if not ids or not self._ids:
            return
        drop = {self._index[cid] for cid in ids if cid in self._index}
        if not drop:
            return
        keep = [i for i in range(len(self._ids)) if i not in drop]
        self._matrix = self._matrix[keep]
        self._ids = [self._ids[i] for i in keep]
        self._index = {cid: i for i, cid in enumerate(self._ids)}

    # -- search --

    def search(self, vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        if not self._ids or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        if q.shape[0] != self._matrix.shape[1]:
            return []
        norm = float(np.linalg.norm(q)) or 1.0
        sims = self._matrix @ (q / norm)
        k = min(k, len(self._ids))
        top = np.argsort(-sims)[:k]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in top]

    def count(self) -> int:
        return len(self._ids)


class QdrantVectorStore:
    kind = "qdrant"

    def __init__(self, dim: int, data_dir: Path, url: str | None = None):
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(url) if url else QdrantClient(path=str(data_dir / "qdrant"))
        self.dim = dim
        self._collection = "chunks"
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def save(self) -> None:  # qdrant persists incrementally
        return None

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        if not ids:
            return
        self._client.upsert(
            self._collection,
            points=[
                self._models.PointStruct(id=_uuid(cid), vector=v, payload={"chunk_id": cid})
                for cid, v in zip(ids, vectors, strict=False)
            ],
        )

    def remove(self, ids: list[str]) -> None:
        if not ids:
            return
        self._client.delete(
            self._collection,
            points_selector=self._models.PointIdsList(points=[_uuid(cid) for cid in ids]),
        )

    def search(self, vector: list[float], k: int = 10) -> list[tuple[str, float]]:
        hits = self._client.query_points(
            self._collection,
            query=vector,
            limit=k,
            with_payload=False,
            with_vectors=False,
        ).points
        return [(h.payload.get("chunk_id", ""), float(h.score)) for h in hits]

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)


def _uuid(chunk_id: str) -> str:
    """Deterministic UUID5 from a chunk id (qdrant requires uuid/int point ids)."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"chunk:{chunk_id}"))


def open_vector_store(backend: str, dim: int, data_dir: Path, qdrant_url: str | None = None):
    if backend == "qdrant":
        return QdrantVectorStore(dim, data_dir, qdrant_url)
    return NumpyVectorStore(dim, data_dir / "vectors.npz")
