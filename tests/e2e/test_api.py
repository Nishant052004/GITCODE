"""E2E tests over the HTTP API (TestClient), including the background job path."""

import time

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


def make_client(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        repo_workdir=tmp_path / "repos",
        embedding_backend="hashing",
        embedding_dim=256,
        vector_backend="numpy",
        reranker="heuristic",
    )
    app = create_app(settings)
    return TestClient(app)


def test_full_api_flow(demo_git_repo, tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/ingest", json={"repo": str(demo_git_repo), "sync": True})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "done"
        assert body["stats"]["files_indexed"] > 0

        r = client.get("/status")
        assert r.status_code == 200
        status = r.json()
        assert status["ok"] and status["chunks"] > 0
        assert status["stale_files"] == []

        r = client.post("/query", json={"query": "How do we verify a user's password?", "top_k": 3})
        assert r.status_code == 200
        result = r.json()
        assert result["answer"]
        assert result["citations"]
        assert any(c["file_path"] == "src/acme/auth.py" for c in result["citations"])
        assert result["timings"]["total_ms"] > 0

        chunk_id = result["citations"][0]["chunk_id"]
        r = client.get(f"/chunks/{chunk_id}")
        assert r.status_code == 200
        assert r.json()["text"]

        r = client.post("/reindex", json={})
        assert r.status_code == 200
        assert r.json()["status"] == "done"

        r = client.get("/files")
        assert r.status_code == 200
        files_data = r.json()
        assert "files" in files_data
        assert len(files_data["files"]) > 0

        r = client.get("/suggestions")
        assert r.status_code == 200
        assert len(r.json()["suggestions"]) > 0

        r = client.get("/health")
        assert r.json() == {"status": "ok"}


def test_background_ingest_job_lifecycle(demo_git_repo, tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/ingest", json={"repo": str(demo_git_repo)})
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            job = client.get(f"/ingest/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert job["status"] == "done", job
        assert job["stats"]["files_indexed"] > 0


def test_query_with_no_index_returns_insufficient(tmp_path):
    with make_client(tmp_path) as client:
        r = client.post("/query", json={"query": "anything at all", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert "couldn't find enough" in body["answer"]
        assert body["citations"] == []


def test_unknown_chunk_and_job_404(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/chunks/deadbeef").status_code == 404
        assert client.get("/jobs/deadbeef").status_code == 404
