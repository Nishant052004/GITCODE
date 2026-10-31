"""The end-to-end pipeline: ingest -> chunk -> embed -> index -> retrieve ->
rerank -> answer, plus staleness detection and incremental reindexing.

One `IngestionPipeline` instance owns every component and the SQLite metadata
DB; the API, CLI and eval scripts all share it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from .answerer import get_answerer
from .bm25_index import BM25Index
from .chunker import chunk_file, chunk_id_for, detect_language, sha256_hex
from .embedder import get_embedder
from .git_ingest import (
    GitHubClient,
    current_branch,
    ensure_repo,
    head_commit,
    last_commit_for_file,
    list_files,
    read_worktree_file,
    recent_commits,
)
from .jobs import JobManager
from .metadata_db import MetadataDB
from .reranker import get_reranker
from .retriever import Retriever
from .vector_store import open_vector_store


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class IngestionPipeline:
    def __init__(self, settings: Settings, db: MetadataDB | None = None):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.repo_workdir.mkdir(parents=True, exist_ok=True)
        self.db = db or MetadataDB(settings.data_dir / "metadata.db")
        self.embedder = get_embedder(
            settings.embedding_backend,
            dim=settings.embedding_dim,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_embed_model,
            openai_base_url=settings.openai_base_url,
            st_model=settings.st_model,
        )
        self.vectors = open_vector_store(settings.vector_backend, self.embedder.dim, settings.data_dir)
        self.bm25 = BM25Index(settings.data_dir / "bm25.json")
        self.retriever = Retriever(settings, self.db, self.embedder, self.vectors, self.bm25)
        self.reranker = get_reranker(settings.reranker, settings.rerank_model)
        self.answerer = get_answerer(
            settings.openai_api_key,
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            max_chunks=settings.context_max_chunks,
            max_chars=settings.context_max_chars,
        )
        self.github = GitHubClient(settings.github_token)
        self.jobs = JobManager(self.db)

    # ------------------------------------------------------------- ingestion

    def ingest(self, source: str, branch: str | None = None, full: bool = True) -> dict[str, Any]:
        repo_path, url = ensure_repo(source, self.settings.repo_workdir, branch)
        repo_id = self.db.upsert_repo(url, str(repo_path), branch or current_branch(repo_path))
        self.db.save_commits(repo_id, recent_commits(repo_path))

        prs, issues = self.github.fetch_repo_data(url)
        if prs:
            self.db.upsert_prs(repo_id, prs)
        if issues:
            self.db.upsert_issues(repo_id, issues)

        files = list_files(repo_path, self.settings.file_extensions, self.settings.exclude_dirs)
        stored_checksums = {} if full else self.db.file_checksums(repo_id)
        stats = {
            "repo": url,
            "files_scanned": len(files),
            "files_indexed": 0,
            "files_skipped": 0,
            "chunks_added": 0,
            "chunks_removed": 0,
        }
        for rel in files:
            text = read_worktree_file(repo_path, rel)
            if text is None:
                continue
            fchecksum = sha256_hex(text)
            if not full and stored_checksums.get(rel) == fchecksum:
                stats["files_skipped"] += 1
                continue
            self._index_file(repo_id, repo_path, rel, text, fchecksum, stats)

        self.db.set_repo_head(repo_id, head_commit(repo_path))
        self.vectors.save()
        self.bm25.save()
        counts = self.db.counts(repo_id)
        stats.update({"files_total": counts["files"], "chunks_total": counts["chunks"]})
        return stats

    def _index_file(
        self,
        repo_id: int,
        repo_path: Path,
        rel: str,
        text: str,
        fchecksum: str,
        stats: dict[str, Any],
    ) -> None:
        meta = last_commit_for_file(repo_path, rel)
        if meta:
            commit_id, author, last_modified = (
                meta["hash"],
                meta["author"],
                meta["date"],
            )
        else:
            # No git history for this file (untracked / non-repo dir): chunks get
            # a stable "WORKTREE" identity, and the file head is stored as None
            # so staleness comparison (None vs None) stays consistent.
            commit_id, author, last_modified = "WORKTREE", None, None
        file_head = meta["hash"] if meta else None
        if not last_modified:
            try:
                mtime = (repo_path / rel).stat().st_mtime
                last_modified = datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds")
            except OSError:
                last_modified = _now_iso()

        raw_chunks = chunk_file(text, rel, self.settings.max_chunk_lines, self.settings.overlap_lines)
        chunks = []
        for c in raw_chunks:
            c["chunk_id"] = chunk_id_for(rel, c["start_line"], c["end_line"], commit_id)
            c["commit_id"] = commit_id
            c["author"] = author
            c["last_modified"] = last_modified
            chunks.append(c)

        removed_ids, added_ids = self.db.replace_file_chunks(repo_id, rel, chunks)
        self.vectors.remove(removed_ids)
        self.bm25.remove(removed_ids)

        fresh = [c for c in chunks if c["chunk_id"] in set(added_ids)]
        if fresh:
            vectors = self.embedder.embed_texts([c["text"] for c in fresh])
            self.vectors.add([c["chunk_id"] for c in fresh], vectors)
            self.bm25.add({c["chunk_id"]: c["text"] for c in fresh})

        self.db.upsert_file(
            repo_id,
            rel,
            detect_language(rel),
            file_head,
            author,
            last_modified,
            fchecksum,
        )
        stats["files_indexed"] = stats.get("files_indexed", 0) + 1
        stats["chunks_added"] = stats.get("chunks_added", 0) + len(added_ids)
        stats["chunks_removed"] = stats.get("chunks_removed", 0) + len(removed_ids)

    # ------------------------------------------------- staleness / reindexing

    def staleness(self, repo_id: int | None = None) -> dict[str, Any]:
        repos = [self.db.get_repo(repo_id)] if repo_id else self.db.list_repos()
        report: list[dict[str, Any]] = []
        stale_total: list[str] = []
        for repo in repos:
            if not repo:
                continue
            repo_path = Path(repo["path"])
            stored_heads = self.db.file_heads(repo["id"])
            stale: list[str] = []
            for rel, stored_head in stored_heads.items():
                current = last_commit_for_file(repo_path, rel)
                current_head = current["hash"] if current else None
                if current_head != stored_head or not (repo_path / rel).exists():
                    stale.append(rel)
            missing = [rel for rel in stored_heads if not (repo_path / rel).exists()]
            head_now = head_commit(repo_path)
            head_stale = head_now != repo.get("head_commit")
            report.append(
                {
                    "repo": repo["url"],
                    "head_commit": head_now,
                    "head_stale": head_stale,
                    "stale_files": stale,
                    "deleted_files": missing,
                }
            )
            stale_total.extend(stale)
        return {"repos": report, "stale_files": stale_total}

    def reindex(self, repo_id: int | None = None, file_path: str | None = None) -> dict[str, Any]:
        repos = [self.db.get_repo(repo_id)] if repo_id else self.db.list_repos()
        stats: dict[str, Any] = {
            "files_reindexed": 0,
            "files_removed": 0,
            "chunks_added": 0,
            "chunks_removed": 0,
        }
        for repo in repos:
            if not repo:
                continue
            repo_path = Path(repo["path"])
            stored_heads = self.db.file_heads(repo["id"])
            if file_path:
                targets = [file_path]
            else:
                targets = [
                    rel
                    for rel, stored in stored_heads.items()
                    if (last_commit_for_file(repo_path, rel) or {}).get("hash") != stored
                ]
                indexed = set(stored_heads)
                for rel in list_files(repo_path, self.settings.file_extensions, self.settings.exclude_dirs):
                    if rel not in indexed:
                        targets.append(rel)
            for rel in list(targets):
                text = read_worktree_file(repo_path, rel)
                if text is None:
                    removed = self.db.delete_file(repo["id"], rel)
                    self.vectors.remove(removed)
                    self.bm25.remove(removed)
                    stats["files_removed"] += 1
                    stats["chunks_removed"] += len(removed)
                    continue
                self._index_file(repo["id"], repo_path, rel, text, sha256_hex(text), stats)
                stats["files_reindexed"] += 1
            self.db.set_repo_head(repo["id"], head_commit(repo_path))
        self.vectors.save()
        self.bm25.save()
        return stats

    # ----------------------------------------------------------------- query

    def query(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        top_k = top_k or self.settings.top_k
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(query)
        t1 = time.perf_counter()
        ranked = self.reranker.rerank(query, candidates, top_k)
        t2 = time.perf_counter()
        answer, citations = self.answerer.answer(query, ranked, top_k)
        t3 = time.perf_counter()
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "candidates": ranked,
            "timings": {
                "retrieve_ms": round((t1 - t0) * 1000, 1),
                "rerank_ms": round((t2 - t1) * 1000, 1),
                "answer_ms": round((t3 - t2) * 1000, 1),
                "total_ms": round((t3 - t0) * 1000, 1),
            },
            "answerer": getattr(self.answerer, "name", "unknown"),
        }

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        repos = self.db.list_repos()
        counts = self.db.counts()
        staleness = self.staleness()
        return {
            "ok": True,
            "repos": repos,
            "files": counts["files"],
            "chunks": counts["chunks"],
            "vectors": self.vectors.count(),
            "bm25_docs": self.bm25.count(),
            "stale_files": staleness["stale_files"],
            "embedder": self.embedder.name,
            "vector_backend": getattr(self.vectors, "kind", "unknown"),
            "reranker": getattr(self.reranker, "name", "unknown"),
            "answerer": getattr(self.answerer, "name", "unknown"),
        }
