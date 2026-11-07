"""Integration tests: full ingest -> query flow, staleness, incremental reindex."""

import subprocess


def test_ingest_indexes_demo_repo(ingested_pipeline):
    status = ingested_pipeline.status()
    assert status["files"] > 0
    assert status["chunks"] > status["files"], "files should be split into multiple chunks"
    assert status["vectors"] == status["chunks"]
    assert status["bm25_docs"] == status["chunks"]
    assert status["stale_files"] == []


def test_known_query_hits_expected_file(ingested_pipeline):
    result = ingested_pipeline.query("How do we verify a user's password?", top_k=3)
    top_files = [c["file_path"] for c in result["candidates"]]
    assert "src/acme/auth.py" in top_files
    assert result["answer"], "extractive answerer must produce output"
    assert result["citations"], "answers must carry citations"
    assert result["timings"]["total_ms"] > 0


def test_stack_trace_query_biases_lexical(ingested_pipeline):
    result = ingested_pipeline.query(
        'Traceback: File "src/acme/auth.py", line 52 in get_user - UserNotFoundError',
        top_k=3,
    )
    top = result["candidates"][0]
    assert top["file_path"].startswith("src/acme/")


def test_incremental_ingest_skips_unchanged_files(ingested_pipeline, demo_git_repo):
    stats = ingested_pipeline.ingest(str(demo_git_repo), full=False)
    assert stats["files_skipped"] == stats["files_scanned"]
    assert stats["files_indexed"] == 0


def test_staleness_detection_and_incremental_reindex(ingested_pipeline, demo_git_repo):
    # Simulate upstream development: edit a file and commit it.
    target = demo_git_repo / "src" / "acme" / "auth.py"
    original = target.read_text(encoding="utf-8")
    target.write_text(
        original
        + '\n\ndef audit_login(email: str, token: str) -> str:\n    """Append a login audit event."""\n    return f"audit:{email}:{token[:8]}"\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(demo_git_repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(demo_git_repo),
            "-c",
            "user.name=Dev",
            "-c",
            "user.email=dev@example.com",
            "commit",
            "-m",
            "add audit_login",
        ],
        check=True,
        capture_output=True,
    )

    stale = ingested_pipeline.staleness()
    assert "src/acme/auth.py" in stale["stale_files"]

    stats = ingested_pipeline.reindex()
    assert stats["files_reindexed"] == 1
    assert stats["chunks_added"] >= 1

    assert ingested_pipeline.staleness()["stale_files"] == []

    # The new function is retrievable and cites the *new* commit.
    result = ingested_pipeline.query("audit login event", top_k=3)
    new_chunks = [c for c in result["candidates"] if c["symbol"] == "audit_login"]
    assert new_chunks, "newly committed function must be indexed"


def test_reindex_removed_file_cleans_up(ingested_pipeline, demo_git_repo):
    repo_id = ingested_pipeline.db.list_repos()[0]["id"]

    subprocess.run(
        ["git", "-C", str(demo_git_repo), "rm", "-q", "src/acme/search.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(demo_git_repo),
            "-c",
            "user.name=Dev",
            "-c",
            "user.email=dev@example.com",
            "commit",
            "-m",
            "remove search",
        ],
        check=True,
        capture_output=True,
    )
    stats = ingested_pipeline.reindex()
    assert stats["files_removed"] == 1
    assert "src/acme/search.py" not in {f["path"] for f in ingested_pipeline.db.list_files(repo_id)}
    assert ingested_pipeline.db.chunk_ids_for_file(repo_id, "src/acme/search.py") == []


def test_chunk_ids_change_with_new_commit(ingested_pipeline, demo_git_repo):
    repo_id = ingested_pipeline.db.list_repos()[0]["id"]
    before = set(ingested_pipeline.db.chunk_ids_for_file(repo_id, "src/acme/payments.py"))
    target = demo_git_repo / "src" / "acme" / "payments.py"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n\n# touched\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(demo_git_repo), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(demo_git_repo),
            "-c",
            "user.name=Dev",
            "-c",
            "user.email=dev@example.com",
            "commit",
            "-m",
            "touch payments",
        ],
        check=True,
        capture_output=True,
    )
    ingested_pipeline.reindex()
    after = set(ingested_pipeline.db.chunk_ids_for_file(repo_id, "src/acme/payments.py"))
    assert before and after and not (before & after), "new commit => new chunk ids (citation versioning)"
