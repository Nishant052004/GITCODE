"""Unit tests for the answerer (extractive + LLM guards)."""

from backend.app.services.answerer import (
    INSUFFICIENT_EVIDENCE,
    ExtractiveAnswerer,
    LLMAnswerer,
)


def fake_candidates():
    return [
        {
            "chunk_id": "c1",
            "file_path": "src/acme/auth.py",
            "start_line": 30,
            "end_line": 40,
            "language": "python",
            "symbol": "verify_password",
            "kind": "function",
            "commit_id": "abcd12345678",
            "author": "ada",
            "last_modified": "2026-01-01T00:00:00+00:00",
            "text": "def verify_password(password, stored):\n    return True",
            "summary": "Constant-time password verification.",
            "score": 0.9,
            "scores": {},
        },
        {
            "chunk_id": "c2",
            "file_path": "src/acme/auth.py",
            "start_line": 50,
            "end_line": 60,
            "language": "python",
            "symbol": "login",
            "kind": "function",
            "commit_id": "abcd12345678",
            "author": "ada",
            "last_modified": "2026-01-01T00:00:00+00:00",
            "text": "def login(email, password, users):\n    ...",
            "summary": "Login flow.",
            "score": 0.7,
            "scores": {},
        },
    ]


def test_extractive_answer_cites_every_candidate():
    answer, citations = ExtractiveAnswerer().answer("how to verify?", fake_candidates(), top_k=2)
    assert "[1]" in answer and "[2]" in answer
    assert [c["index"] for c in citations] == [1, 2]
    assert citations[0]["file_path"] == "src/acme/auth.py"


def test_extractive_answer_empty_candidates_is_insufficient():
    answer, citations = ExtractiveAnswerer().answer("anything", [], top_k=3)
    assert answer == INSUFFICIENT_EVIDENCE
    assert citations == []


def test_llm_answer_keeps_valid_citations(monkeypatch):
    answerer = LLMAnswerer("sk-test")
    monkeypatch.setattr(
        answerer,
        "_chat",
        lambda q, ctx: "Use verify_password [1] and also login [3].",
    )
    _answer, citations = answerer.answer("how to verify?", fake_candidates(), top_k=2)
    # [3] does not exist in a two-snippet context: it must be dropped.
    assert [c["index"] for c in citations] == [1]


def test_llm_insufficient_evidence_marker(monkeypatch):
    answerer = LLMAnswerer("sk-test")
    monkeypatch.setattr(answerer, "_chat", lambda q, ctx: "INSUFFICIENT_EVIDENCE")
    answer, citations = answerer.answer("how to verify?", fake_candidates(), top_k=2)
    assert answer == INSUFFICIENT_EVIDENCE and citations == []


def test_llm_uncited_answer_falls_back_to_extractive(monkeypatch):
    answerer = LLMAnswerer("sk-test")
    monkeypatch.setattr(answerer, "_chat", lambda q, ctx: "The password check uses pbkdf2.")
    answer, citations = answerer.answer("how to verify?", fake_candidates(), top_k=2)
    # Hallucination guard: no [n] citations in the LLM reply -> deterministic
    # extractive answer with citations instead.
    assert "[1]" in answer and citations


def test_llm_error_falls_back_to_extractive(monkeypatch):
    answerer = LLMAnswerer("sk-test")

    def boom(q, ctx):
        raise RuntimeError("api down")

    monkeypatch.setattr(answerer, "_chat", boom)
    answer, citations = answerer.answer("how to verify?", fake_candidates(), top_k=2)
    assert citations and "[1]" in answer
