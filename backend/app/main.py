"""FastAPI application entrypoint: `uvicorn backend.app.main:app --reload`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import ingest, query, reindex, status
from .config import Settings
from .services.pipeline import IngestionPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built lazily so importing the app stays cheap (and tests can inject).
        if not hasattr(app.state, "pipeline"):
            app.state.settings = resolved
            app.state.pipeline = IngestionPipeline(resolved)
        yield

    app = FastAPI(
        title="Codebase Intelligence & Developer Copilot",
        version="0.1.0",
        description="Ingest a repo, chunk it semantically, index it, and ask cited questions.",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    @app.middleware("http")
    async def ensure_pipeline(request, call_next):
        if not hasattr(request.app.state, "pipeline"):
            st = getattr(request.app.state, "settings", None) or resolved
            request.app.state.settings = st
            request.app.state.pipeline = IngestionPipeline(st)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(reindex.router)
    app.include_router(status.router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
