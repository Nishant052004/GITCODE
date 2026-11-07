"""Unit tests for embedders, BM25 and the vector store."""

import math

import pytest

from backend.app.services.bm25_index import BM25Index
from backend.app.services.embedder import HashingEmbedder, get_embedder
from backend.app.services.vector_store import NumpyVectorStore

# ---------------------------------------------------------------- embedder


def test_hashing_embedder_deterministic():
    emb = HashingEmbedder(dim=256)
    v1 = emb.embed_query("def verify_password(password, stored)")
    v2 = emb.embed_query("def verify_password(password, stored)")
    v3 = emb.embed_query("charge invoice through gateway")
    assert v1 == v2
    assert v1 != v3


def test_hashing_embedder_normalized():
    emb = HashingEmbedder(dim=128)
    vec = emb.embed_query("some tokens here")
    assert len(vec) == 128
    assert math.isclose(sum(x * x for x in vec), 1.0, rel_tol=1e-6)


def test_hashing_embedder_empty_text_is_safe():
    emb = HashingEmbedder(dim=128)
    assert emb.embed_query("") == [0.0] * 128


def test_get_embedder_hashing_explicit():
    emb = get_embedder("hashing", dim=256)
    assert emb.name == "hashing" and emb.dim == 256


def test_get_embedder_openai_requires_key():
    with pytest.raises(ValueError):
        get_embedder("openai", openai_api_key=None)


# ------------------------------------------------------------------- bm25


def test_bm25_ranks_exact_terms_first(tmp_path):
    idx = BM25Index(tmp_path / "bm25.json")
    idx.add(
        {
            "auth": "verify password with pbkdf2 salted hash",
            "pay": "charge invoice through payment gateway",
        }
    )
    hits = idx.search("verify password", k=2)
    assert hits[0][0] == "auth"


def test_bm25_matches_identifier_subtokens(tmp_path):
    idx = BM25Index(tmp_path / "bm25.json")
    idx.add(
        {
            "auth": "def verify_password(password, stored): compare digests",
            "pay": "def charge_invoice(invoice, gateway): idempotency key",
        }
    )
    hits = idx.search("verify password", k=1)
    assert hits[0][0] == "auth"


def test_bm25_remove_and_count(tmp_path):
    idx = BM25Index(tmp_path / "bm25.json")
    idx.add({"a": "alpha beta", "b": "gamma delta"})
    assert idx.count() == 2
    idx.remove(["a"])
    assert idx.count() == 1
    assert all(cid != "a" for cid, _ in idx.search("alpha", k=5))


def test_bm25_persists(tmp_path):
    path = tmp_path / "bm25.json"
    idx = BM25Index(path)
    idx.add({"a": "alpha beta gamma"})
    idx.save()
    reloaded = BM25Index(path)
    assert reloaded.count() == 1
    assert reloaded.search("alpha", k=1)[0][0] == "a"


# ------------------------------------------------------------ vector store


def test_numpy_store_cosine_ranking(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path / "v.npz")
    store.add(["x", "y", "z"], [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
    hits = store.search([1, 0.05, 0], k=3)
    assert hits[0][0] in ("x", "z")
    assert hits[-1][0] == "y"


def test_numpy_store_upsert_and_remove(tmp_path):
    store = NumpyVectorStore(dim=2, path=tmp_path / "v.npz")
    store.add(["a"], [[1, 0]])
    store.add(["a"], [[0, 1]])  # upsert replaces the old vector
    assert store.count() == 1
    store.remove(["a"])
    assert store.count() == 0
    assert store.search([1, 0], k=5) == []


def test_numpy_store_persists(tmp_path):
    path = tmp_path / "v.npz"
    store = NumpyVectorStore(dim=2, path=path)
    store.add(["a"], [[1, 0]])
    store.save()
    reloaded = NumpyVectorStore(dim=2, path=path)
    assert reloaded.count() == 1
    assert reloaded.search([1, 0], k=1)[0][0] == "a"
