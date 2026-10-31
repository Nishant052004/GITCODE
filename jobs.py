"""Background job runner backed by the jobs table.

MVP uses daemon threads; the interface (submit -> job_id, GET /jobs/{id})
matches a Celery/Redis deployment one-to-one.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from .metadata_db import MetadataDB


class JobManager:
    def __init__(self, db: MetadataDB):
        self._db = db
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = self._db.create_job(kind)

        def run() -> None:
            try:
                stats = fn(*args, **kwargs)
                self._db.finish_job(job_id, "done", stats=stats or {})
            except Exception as exc:  # job boundary
                self._db.finish_job(
                    job_id,
                    "failed",
                    detail=f"{type(exc).__name__}: {exc}",
                    stats={"traceback": traceback.format_exc(limit=5)},
                )

        thread = threading.Thread(target=run, daemon=True, name=f"job-{job_id}")
        self._threads[job_id] = thread
        thread.start()
        return job_id

    def run_inline(self, kind: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[str, Any]:
        """Run synchronously - used by tests and the CLI."""
        job_id = self._db.create_job(kind)
        try:
            result = fn(*args, **kwargs)
            self._db.finish_job(job_id, "done", stats=result if isinstance(result, dict) else None)
            return job_id, result
        except Exception as exc:
            self._db.finish_job(job_id, "failed", detail=f"{type(exc).__name__}: {exc}")
            raise
