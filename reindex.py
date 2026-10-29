"""POST /reindex - refresh one file or incrementally reindex everything stale."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..models.schemas import ReindexRequest

router = APIRouter(tags=["reindex"])


@router.post("/reindex")
def reindex(payload: ReindexRequest, request: Request):
    pipeline = request.app.state.pipeline
    stats = pipeline.reindex(file_path=payload.file_path)
    return {"status": "done", "stats": stats}
