"""SQLite-backed metadata store: repos, files, chunks, commits, PRs, issues, jobs.

This is the source of truth for chunk -> commit -> author -> PR mapping and the
staleness ledger (stored per-file HEAD vs current HEAD).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE,
  path TEXT,
  branch TEXT,
  head_commit TEXT,
  last_ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS files(
  repo_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  language TEXT,
  head_commit TEXT,
  author TEXT,
  last_modified TEXT,
  checksum TEXT,
  indexed_at TEXT,
  PRIMARY KEY (repo_id, path)
);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY,
  repo_id INTEGER NOT NULL,
  file_path TEXT NOT NULL,
  start_line INTEGER,
  end_line INTEGER,
  language TEXT,
  symbol TEXT,
  kind TEXT,
  commit_id TEXT,
  author TEXT,
  last_modified TEXT,
  pr_id TEXT,
  checksum TEXT,
  summary TEXT,
  text TEXT,
  indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo_file ON chunks(repo_id, file_path);
CREATE TABLE IF NOT EXISTS commits(
  repo_id INTEGER NOT NULL,
  hash TEXT NOT NULL,
  author TEXT,
  message TEXT,
  committed_at TEXT,
  PRIMARY KEY (repo_id, hash)
);
CREATE TABLE IF NOT EXISTS pull_requests(
  repo_id INTEGER NOT NULL,
  number INTEGER NOT NULL,
  title TEXT,
  author TEXT,
  state TEXT,
  merged_at TEXT,
  url TEXT,
  PRIMARY KEY (repo_id, number)
);
CREATE TABLE IF NOT EXISTS issues(
  repo_id INTEGER NOT NULL,
  number INTEGER NOT NULL,
  title TEXT,
  author TEXT,
  state TEXT,
  created_at TEXT,
  url TEXT,
  PRIMARY KEY (repo_id, number)
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT,
  status TEXT,
  detail TEXT,
  stats TEXT,
  created_at TEXT,
  updated_at TEXT
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MetadataDB:
    """Thin DAO over SQLite. Thread-safe via a single lock (good enough for MVP)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- repos ----------

    def upsert_repo(self, url: str, path: str, branch: str | None = None) -> int:
        with self._lock:
            row = self._conn.execute("SELECT id FROM repos WHERE url = ?", (url,)).fetchone()
            now = utcnow()
            if row:
                self._conn.execute(
                    "UPDATE repos SET path = ?, branch = COALESCE(?, branch), last_ingested_at = ? WHERE id = ?",
                    (path, branch, now, row["id"]),
                )
                self._conn.commit()
                return int(row["id"])
            cur = self._conn.execute(
                "INSERT INTO repos(url, path, branch, last_ingested_at) VALUES (?, ?, ?, ?)",
                (url, path, branch, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_repo_head(self, repo_id: int, head_commit: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE repos SET head_commit = ?, last_ingested_at = ? WHERE id = ?",
                (head_commit, utcnow(), repo_id),
            )
            self._conn.commit()

    def get_repo(self, repo_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
            return dict(row) if row else None

    def list_repos(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute("SELECT * FROM repos ORDER BY id")]

    # ---------- files / chunks ----------

    def upsert_file(
        self,
        repo_id: int,
        path: str,
        language: str,
        head_commit: str | None,
        author: str | None,
        last_modified: str | None,
        checksum: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO files(repo_id, path, language, head_commit, author, last_modified, checksum, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, path) DO UPDATE SET
                  language=excluded.language, head_commit=excluded.head_commit,
                  author=excluded.author, last_modified=excluded.last_modified,
                  checksum=excluded.checksum, indexed_at=excluded.indexed_at
                """,
                (
                    repo_id,
                    path,
                    language,
                    head_commit,
                    author,
                    last_modified,
                    checksum,
                    utcnow(),
                ),
            )
            self._conn.commit()

    def file_checksums(self, repo_id: int) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, checksum FROM files WHERE repo_id = ?", (repo_id,)
            ).fetchall()
            return {r["path"]: r["checksum"] for r in rows}

    def file_heads(self, repo_id: int) -> dict[str, str | None]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, head_commit FROM files WHERE repo_id = ?", (repo_id,)
            ).fetchall()
            return {r["path"]: r["head_commit"] for r in rows}

    def list_files(self, repo_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, language, checksum, head_commit FROM files WHERE repo_id = ? ORDER BY path",
                (repo_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_file(self, repo_id: int, path: str) -> list[str]:
        """Drop a file row and all its chunk rows; returns removed chunk ids."""
        with self._lock:
            old = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE repo_id = ? AND file_path = ?",
                (repo_id, path),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM chunks WHERE repo_id = ? AND file_path = ?",
                (repo_id, path),
            )
            self._conn.execute("DELETE FROM files WHERE repo_id = ? AND path = ?", (repo_id, path))
            self._conn.commit()
            return [r["chunk_id"] for r in old]

    def replace_file_chunks(
        self, repo_id: int, file_path: str, chunks: list[dict]
    ) -> tuple[list[str], list[str]]:
        """Atomically swap in a new chunk set for a file.

        Returns (removed_ids, added_ids) so callers can sync vector/BM25 stores.
        """
        now = utcnow()
        with self._lock:
            old_rows = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE repo_id = ? AND file_path = ?",
                (repo_id, file_path),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM chunks WHERE repo_id = ? AND file_path = ?",
                (repo_id, file_path),
            )
            new_ids = []
            for c in chunks:
                cid = c["chunk_id"]
                new_ids.append(cid)
                self._conn.execute(
                    """
                    INSERT INTO chunks(chunk_id, repo_id, file_path, start_line, end_line, language,
                                       symbol, kind, commit_id, author, last_modified, pr_id,
                                       checksum, summary, text, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        repo_id,
                        file_path,
                        c["start_line"],
                        c["end_line"],
                        c["language"],
                        c.get("symbol"),
                        c.get("kind", "block"),
                        c.get("commit_id"),
                        c.get("author"),
                        c.get("last_modified"),
                        c.get("pr_id"),
                        c["checksum"],
                        c.get("summary"),
                        c["text"],
                        now,
                    ),
                )
            self._conn.commit()
        # Chunk ids embed the commit id, so identity is (path, lines, commit).
        # Any change there means new vectors for new ids and removal of old ones.
        new_id_set = set(new_ids)
        old_id_set = {r["chunk_id"] for r in old_rows}
        removed = [cid for cid in old_id_set if cid not in new_id_set]
        added = [cid for cid in new_ids if cid not in old_id_set]
        return removed, added

    def chunk_ids_for_file(self, repo_id: int, file_path: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE repo_id = ? AND file_path = ?",
                (repo_id, file_path),
            ).fetchall()
            return [r["chunk_id"] for r in rows]

    def get_chunk(self, chunk_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            return dict(row) if row else None

    def get_chunks(self, chunk_ids: Iterable[str]) -> list[dict]:
        ids = list(chunk_ids)
        if not ids:
            return []
        out: dict[str, dict] = {}
        with self._lock:
            for i in range(0, len(ids), 500):
                batch = ids[i : i + 500]
                q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})"
                for row in self._conn.execute(q, batch):
                    out[row["chunk_id"]] = dict(row)
        return [out[cid] for cid in ids if cid in out]

    # ---------- commits / PRs / issues ----------

    def save_commits(self, repo_id: int, commits: list[dict]) -> None:
        with self._lock:
            for c in commits:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO commits(repo_id, hash, author, message, committed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        repo_id,
                        c["hash"],
                        c.get("author"),
                        c.get("message"),
                        c.get("committed_at"),
                    ),
                )
            self._conn.commit()

    def upsert_prs(self, repo_id: int, prs: list[dict]) -> None:
        with self._lock:
            for p in prs:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO pull_requests(repo_id, number, title, author, state, merged_at, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        repo_id,
                        p["number"],
                        p.get("title"),
                        p.get("author"),
                        p.get("state"),
                        p.get("merged_at"),
                        p.get("url"),
                    ),
                )
            self._conn.commit()

    def upsert_issues(self, repo_id: int, issues: list[dict]) -> None:
        with self._lock:
            for it in issues:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO issues(repo_id, number, title, author, state, created_at, url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        repo_id,
                        it["number"],
                        it.get("title"),
                        it.get("author"),
                        it.get("state"),
                        it.get("created_at"),
                        it.get("url"),
                    ),
                )
            self._conn.commit()

    # ---------- jobs ----------

    def create_job(self, kind: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id, kind, status, detail, created_at, updated_at) VALUES (?, ?, 'running', NULL, ?, ?)",
                (job_id, kind, utcnow(), utcnow()),
            )
            self._conn.commit()
        return job_id

    def finish_job(
        self,
        job_id: str,
        status: str,
        detail: str | None = None,
        stats: dict | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, detail = ?, stats = ?, updated_at = ? WHERE id = ?",
                (
                    status,
                    detail,
                    json.dumps(stats) if stats is not None else None,
                    utcnow(),
                    job_id,
                ),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            job["stats"] = json.loads(job["stats"]) if job["stats"] else None
            return job

    # ---------- counts ----------

    def counts(self, repo_id: int | None = None) -> dict[str, int]:
        q_files = "SELECT COUNT(*) c FROM files" + (" WHERE repo_id = ?" if repo_id else "")
        q_chunks = "SELECT COUNT(*) c FROM chunks" + (" WHERE repo_id = ?" if repo_id else "")
        params = (repo_id,) if repo_id else ()
        with self._lock:
            files = self._conn.execute(q_files, params).fetchone()["c"]
            chunks = self._conn.execute(q_chunks, params).fetchone()["c"]
        return {"files": files, "chunks": chunks}
