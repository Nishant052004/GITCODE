"""POST /query - hybrid retrieval + rerank + cited answer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..models.schemas import CandidateOut, Citation, QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request):
    pipeline = request.app.state.pipeline
    try:
        result = pipeline.query(payload.query, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}") from exc
    candidates = [
        CandidateOut(
            chunk_id=c["chunk_id"],
            file_path=c["file_path"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            language=c["language"],
            symbol=c.get("symbol"),
            kind=c.get("kind", "block"),
            commit_id=c.get("commit_id"),
            author=c.get("author"),
            last_modified=c.get("last_modified"),
            score=c["score"],
            scores=c.get("scores", {}),
            snippet=(c.get("text") or "")[:240],
        )
        for c in result["candidates"]
    ]
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        candidates=candidates,
        timings=result["timings"],
        answerer=result["answerer"],
    )
