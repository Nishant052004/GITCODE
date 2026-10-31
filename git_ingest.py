"""Git + GitHub ingestion helpers.

Everything works against a plain local clone via the `git` CLI, so no API token
is required for the core pipeline. The GitHub REST client (PRs/issues) is
optional enrichment gated on GITHUB_TOKEN.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Field separator for --format output that can't appear in commit metadata.
US = "\x1f"


class GitError(RuntimeError):
    pass


def run_git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on git install
        raise GitError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    return proc.stdout


def is_git_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        run_git(path, "rev-parse", "--is-inside-work-tree")
        return True
    except Exception:
        return False


def slugify(source: str) -> str:
    """Stable directory name for a cloned repo."""
    cleaned = re.sub(r"^https?://|^git@|\.git$", "", source.strip().rstrip("/"))
    cleaned = re.sub(r"[:/\\]+", "__", cleaned)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)[:120] or "repo"


def ensure_repo(source: str, workdir: Path, branch: str | None = None) -> tuple[Path, str]:
    """Resolve `source` to a local checkout, cloning/pulling when it's a URL.

    Returns (local_path, canonical_url).
    """
    source = source.strip()
    local = Path(source).expanduser()
    if local.exists():
        return local.resolve(), str(local.resolve())
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / slugify(source)
    url = source if re.match(r"^(https?://|git@|ssh://)", source) else f"https://github.com/{source}.git"
    if dest.exists() and is_git_repo(dest):
        run_git(dest, "fetch", "origin", "--prune")
        target = branch or "HEAD"
        run_git(dest, "reset", "--hard", f"origin/{target}" if branch else "FETCH_HEAD")
    else:
        args = ["clone"]
        if branch:
            args += ["--branch", branch]
        args += [url, str(dest)]
        subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8")
    return dest.resolve(), url


def head_commit(repo: Path) -> str:
    if not is_git_repo(repo):
        return "WORKTREE"
    try:
        return run_git(repo, "rev-parse", "HEAD").strip()
    except GitError:
        return "WORKTREE"


def current_branch(repo: Path) -> str | None:
    if not is_git_repo(repo):
        return None
    try:
        return run_git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip() or None
    except GitError:
        return None


def list_files(repo: Path, extensions: tuple[str, ...], exclude_dirs: tuple[str, ...]) -> list[str]:
    """Tracked files from git (respects .gitignore), filtered by extension."""
    if is_git_repo(repo):
        # -c cached + -o others (untracked), excluding gitignored: also covers
        # ingesting a subdirectory of a larger repo whose files aren't committed yet.
        out = run_git(repo, "ls-files", "-co", "--exclude-standard")
        paths = [p.replace("\\", "/") for p in out.splitlines() if p.strip()]
    else:  # plain directory fallback
        paths = [str(p.relative_to(repo)).replace("\\", "/") for p in repo.rglob("*") if p.is_file()]
    result = []
    for p in paths:
        if any(part in exclude_dirs for part in p.split("/")):
            continue
        if p.lower().endswith(tuple(extensions)):
            result.append(p)
    return sorted(result)


def read_worktree_file(repo: Path, rel_path: str) -> str | None:
    try:
        return (repo / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


_COMMIT_FIELDS = "%H%x1f%an%x1f%aI%x1f%s"


def last_commit_for_file(repo: Path, rel_path: str) -> dict | None:
    """Author/date of the most recent commit that touched the file."""
    if not is_git_repo(repo):
        return None
    try:
        out = run_git(repo, "log", "-1", f"--format={_COMMIT_FIELDS}", "--", rel_path)
    except GitError:
        return None
    line = out.strip().splitlines()
    if not line or not line[0].strip():
        return None
    h, author, date, msg = (*line[0].split(US), "", "", "", "")[:4]
    return {"hash": h, "author": author, "date": date, "message": msg}


def recent_commits(repo: Path, limit: int = 50) -> list[dict]:
    if not is_git_repo(repo):
        return []
    try:
        out = run_git(repo, "log", f"-n{limit}", f"--format={_COMMIT_FIELDS}")
    except GitError:
        return []
    commits = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        h, author, date, msg = (*line.split(US), "", "", "", "")[:4]
        commits.append({"hash": h, "author": author, "committed_at": date, "message": msg})
    return commits


def diff_paths(repo: Path, from_commit: str | None, to_commit: str = "HEAD") -> list[str]:
    """Files changed between two commits (used for incremental reindex)."""
    if not is_git_repo(repo) or not from_commit or from_commit in ("", "WORKTREE"):
        return []
    try:
        out = run_git(repo, "diff", "--name-status", f"{from_commit}..{to_commit}")
    except GitError:
        return []
    changed = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        path = path.replace("\\", "/")
        if status.strip().upper().startswith("D"):
            changed.append(path)
        else:
            changed.append(path)
    return changed


class GitHubClient:
    """Optional REST enrichment: PRs and issues for a github.com repo."""

    def __init__(self, token: str | None):
        self.token = token

    def available(self) -> bool:
        return bool(self.token)

    def fetch_repo_data(self, url: str, limit: int = 30) -> tuple[list[dict], list[dict]]:
        if not self.available() or "github.com" not in url:
            return [], []
        import httpx

        m = re.search(r"github\.com[/:]([^/]+)/([^/#?]+)", url)
        if not m:
            return [], []
        owner, name = m.group(1), m.group(2).removesuffix(".git")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
        }
        prs, issues = [], []
        with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=20) as client:
            try:
                r = client.get(
                    f"/repos/{owner}/{name}/pulls",
                    params={"state": "all", "per_page": limit},
                )
                if r.status_code == 200:
                    for p in r.json():
                        prs.append(
                            {
                                "number": p["number"],
                                "title": p.get("title"),
                                "author": (p.get("user") or {}).get("login"),
                                "state": p.get("state"),
                                "merged_at": p.get("merged_at"),
                                "url": p.get("html_url"),
                            }
                        )
                r = client.get(
                    f"/repos/{owner}/{name}/issues",
                    params={"state": "all", "per_page": limit},
                )
                if r.status_code == 200:
                    for it in r.json():
                        if "pull_request" in it:
                            continue
                        issues.append(
                            {
                                "number": it["number"],
                                "title": it.get("title"),
                                "author": (it.get("user") or {}).get("login"),
                                "state": it.get("state"),
                                "created_at": it.get("created_at"),
                                "url": it.get("html_url"),
                            }
                        )
            except httpx.HTTPError:
                return [], []
        return prs, issues
