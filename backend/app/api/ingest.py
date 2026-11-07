"""POST /ingest - full or incremental ingestion, sync or background job."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.schemas import IngestRequest, JobResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=JobResponse)
def start_ingest(payload: IngestRequest, request: Request):
    pipeline = request.app.state.pipeline
    if payload.sync:
        try:
            job_id, stats = pipeline.jobs.run_inline(
                "ingest",
                pipeline.ingest,
                payload.repo,
                branch=payload.branch,
                full=payload.full,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JobResponse(job_id=job_id, status="done", stats=stats)
    job_id = pipeline.jobs.submit(
        "ingest",
        pipeline.ingest,
        payload.repo,
        branch=payload.branch,
        full=payload.full,
    )
    return JobResponse(job_id=job_id, status="running")


@router.get("/ingest/{job_id}", response_model=JobResponse)
def ingest_status(job_id: str, request: Request):
    job = request.app.state.pipeline.db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return JobResponse(job_id=job["id"], status=job["status"], detail=job["detail"], stats=job["stats"])
