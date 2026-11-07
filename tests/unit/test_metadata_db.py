"""Unit tests for the SQLite metadata store."""

from backend.app.services.chunker import make_chunk
from backend.app.services.metadata_db import MetadataDB


def sample_chunks(commit: str) -> list[dict]:
    c1 = make_chunk("src/a.py", 1, 10, "python", "foo", "function", "def foo():\n    return 1")
    c1["chunk_id"] = f"id-{commit}-1"
    c2 = make_chunk("src/a.py", 12, 14, "python", "bar", "function", "def bar():\n    return 2")
    c2["chunk_id"] = f"id-{commit}-2"
    return [c1, c2]


def test_upsert_repo_is_idempotent(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    id1 = db.upsert_repo("https://github.com/x/y", "/tmp/y")
    id2 = db.upsert_repo("https://github.com/x/y", "/tmp/y")
    assert id1 == id2
    assert len(db.list_repos()) == 1


def test_replace_file_chunks_reports_removed_added(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    repo_id = db.upsert_repo("u", "p")
    removed, added = db.replace_file_chunks(repo_id, "src/a.py", sample_chunks("c1"))
    assert removed == [] and sorted(added) == ["id-c1-1", "id-c1-2"]

    # Same content, new commit: chunk ids embed the commit id, so every chunk
    # is replaced (old ids removed from stores, new ids embedded).
    removed, added = db.replace_file_chunks(repo_id, "src/a.py", sample_chunks("c2"))
    assert sorted(removed) == ["id-c1-1", "id-c1-2"]
    assert sorted(added) == ["id-c2-1", "id-c2-2"]

    # Content actually changed: old ids removed, new ids added.
    changed = sample_chunks("c3")
    changed[0]["text"] = "def foo():\n    return 42"
    changed[0]["checksum"] = "different"
    removed, added = db.replace_file_chunks(repo_id, "src/a.py", changed)
    assert "id-c1-1" in removed or "id-c2-1" in removed
    assert len(added) >= 1


def test_get_chunks_roundtrip(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    repo_id = db.upsert_repo("u", "p")
    db.replace_file_chunks(repo_id, "src/a.py", sample_chunks("c1"))
    rows = db.get_chunks(["id-c1-1", "id-c1-2", "missing"])
    assert len(rows) == 2
    assert rows[0]["file_path"] == "src/a.py"
    assert db.get_chunk("id-c1-1")["symbol"] == "foo"


def test_file_checksums_and_counts(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    repo_id = db.upsert_repo("u", "p")
    db.replace_file_chunks(repo_id, "src/a.py", sample_chunks("c1"))
    db.upsert_file(repo_id, "src/a.py", "python", "c1", "alice", "2026-01-01T00:00:00+00:00", "cs1")
    assert db.file_checksums(repo_id) == {"src/a.py": "cs1"}
    assert db.file_heads(repo_id) == {"src/a.py": "c1"}
    counts = db.counts(repo_id)
    assert counts == {"files": 1, "chunks": 2}


def test_delete_file(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    repo_id = db.upsert_repo("u", "p")
    db.replace_file_chunks(repo_id, "src/a.py", sample_chunks("c1"))
    removed = db.delete_file(repo_id, "src/a.py")
    assert sorted(removed) == ["id-c1-1", "id-c1-2"]
    assert db.counts(repo_id) == {"files": 0, "chunks": 0}


def test_job_lifecycle(tmp_path):
    db = MetadataDB(tmp_path / "meta.db")
    job_id = db.create_job("ingest")
    assert db.get_job(job_id)["status"] == "running"
    db.finish_job(job_id, "done", stats={"files": 3})
    job = db.get_job(job_id)
    assert job["status"] == "done" and job["stats"] == {"files": 3}
    assert db.get_job("nope") is None
