"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------- Ingestion ----------


class IngestRequest(BaseModel):
    repo: str = Field(..., description="GitHub URL or local path to a git repo")
    branch: str | None = Field(None, description="Optional branch to ingest")
    full: bool = Field(True, description="True = full re-ingest; False = skip unchanged files")
    sync: bool = Field(False, description="Run inline instead of as a background job (tests/CLI)")


class ReindexRequest(BaseModel):
    file_path: str | None = Field(
        None,
        description="Re-chunk a single file; omit to incrementally reindex all changed files",
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    stats: dict[str, Any] | None = None


# ---------- Query ----------


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)


class Citation(BaseModel):
    index: int
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None = None
    commit_id: str | None = None


class CandidateOut(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    symbol: str | None = None
    kind: str
    commit_id: str | None = None
    author: str | None = None
    last_modified: str | None = None
    score: float
    scores: dict[str, float] = {}
    snippet: str = ""


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = []
    candidates: list[CandidateOut] = []
    timings: dict[str, float] = {}
    answerer: str = "extractive"


# ---------- Status ----------


class ChunkOut(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    symbol: str | None = None
    kind: str
    commit_id: str | None = None
    author: str | None = None
    last_modified: str | None = None
    pr_id: str | None = None
    checksum: str | None = None
    summary: str | None = None
    text: str


class StatusResponse(BaseModel):
    ok: bool
    repos: list[dict[str, Any]] = []
    files: int = 0
    chunks: int = 0
    vectors: int = 0
    bm25_docs: int = 0
    stale_files: list[str] = []
    embedder: str = ""
    vector_backend: str = ""
    reranker: str = ""
    answerer: str = ""
