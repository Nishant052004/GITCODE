"""Unit tests for query analysis, fusion retrieval and reranking."""

from backend.app.services.reranker import HeuristicReranker
from backend.app.services.retriever import _minmax, _recency_score, analyze_query

# ---------------------------------------------------------- query analysis


def test_analyze_query_extracts_identifiers_and_subtokens():
    feats = analyze_query("How does verify_password work?")
    assert "verify_password" in feats.identifiers
    assert "password" in feats.normalized and "verify" in feats.normalized


def test_analyze_query_detects_stack_trace_and_paths():
    feats = analyze_query('Traceback: File "src/acme/auth.py", line 42 - UserNotFoundError')
    assert feats.has_stack_trace
    assert any(p.endswith("auth.py") for p in feats.paths)


def test_analyze_query_ignores_stopwords():
    feats = analyze_query("What is the best way to do this?")
    assert feats.identifiers == []


# ------------------------------------------------------------------ fusion


def test_minmax_bounds():
    out = _minmax({"a": 1.0, "b": 3.0, "c": 2.0})
    assert out["a"] == 0.0 and out["b"] == 1.0
    assert out["c"] == 0.5


def test_recency_scores():
    assert _recency_score(None) == 0.0
    assert _recency_score("not-a-date") == 0.0
    fresh = _recency_score("2026-10-01T00:00:00+00:00")  # ~10 days before 2026-10-11
    nine_months = _recency_score("2026-01-01T00:00:00+00:00")
    ancient = _recency_score("2019-01-01T00:00:00+00:00")
    assert fresh > nine_months > 0.0
    assert ancient == 0.0  # clamped at zero past one year


def test_retriever_returns_metadata_rich_candidates(ingested_pipeline):
    cands = ingested_pipeline.retriever.retrieve("verify password pbkdf2", top_n=10)
    assert cands, "expected hits from the ingested demo repo"
    top = cands[0]
    for key in (
        "chunk_id",
        "file_path",
        "start_line",
        "end_line",
        "score",
        "scores",
        "text",
    ):
        assert key in top
    assert "vector" in top["scores"] and "bm25" in top["scores"]


def test_retriever_union_of_both_sources(ingested_pipeline):
    cands = ingested_pipeline.retriever.retrieve("invoice charge gateway idempotency", top_n=20)
    assert any("payments" in c["file_path"] for c in cands)


# ---------------------------------------------------------------- reranker


def test_heuristic_reranker_prefers_exact_identifier():
    reranker = HeuristicReranker()
    candidates = [
        {
            "chunk_id": "generic",
            "text": "the gateway handles payment flows",
            "symbol": None,
            "score": 0.5,
            "scores": {},
        },
        {
            "chunk_id": "exact",
            "text": "def verify_password(password, stored): digest compare",
            "symbol": "verify_password",
            "score": 0.5,
            "scores": {},
        },
    ]
    ranked = reranker.rerank("verify_password", candidates, top_k=2)
    assert ranked[0]["chunk_id"] == "exact"
    assert ranked[0]["scores"]["ident_boost"] == 1.0


def test_heuristic_reranker_respects_top_k():
    reranker = HeuristicReranker()
    candidates = [
        {
            "chunk_id": str(i),
            "text": f"doc {i}",
            "symbol": None,
            "score": 0.1 * i,
            "scores": {},
        }
        for i in range(5)
    ]
    ranked = reranker.rerank("doc", candidates, top_k=3)
    assert len(ranked) == 3
