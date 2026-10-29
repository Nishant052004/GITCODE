"""GET /status, GET /files/{...}, GET /chunks/{chunk_id} - health & metadata."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.schemas import ChunkOut, StatusResponse

router = APIRouter(tags=["status"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
def status(request: Request):
    return StatusResponse(**request.app.state.pipeline.status())


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str, request: Request):
    chunk = request.app.state.pipeline.db.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"unknown chunk {chunk_id}")
    return ChunkOut(**chunk)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = request.app.state.pipeline.db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job


@router.get("/files")
def list_files(request: Request, repo_id: int | None = None):
    pipeline = request.app.state.pipeline
    repos = pipeline.db.list_repos()
    if not repos:
        return {"files": [], "repo_id": None}
    target_repo_id = repo_id or repos[0]["id"]
    return {"files": pipeline.db.list_files(target_repo_id), "repo_id": target_repo_id}


@router.get("/suggestions")
def get_suggestions():
    return {
        "suggestions": [
            "How do we verify a user's password?",
            "Where and how are invoices charged?",
            "How is user search or index filtering implemented?",
            "Show the database storage implementation and interface",
            "Why does login fail for unknown or inactive users?",
            "How is structured logging configured across modules?",
        ]
    }

