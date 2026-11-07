"""`z` - developer CLI for the Codebase Intelligence Copilot.

Run with `python -m cli.main <command>` (works from the repo root) or
`python cli/main.py <command>`.

Commands:
    z ingest <repo> [--branch b] [--incremental]
    z query "question" [--k 5] [--json]
    z open src/foo.py:42 [--context 8]
    z status
    z reindex [file_path]
    z serve [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python cli/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from backend.app.config import Settings
from backend.app.services.git_ingest import read_worktree_file
from backend.app.services.pipeline import IngestionPipeline

app = typer.Typer(name="z", help="Codebase Intelligence & Developer Copilot", no_args_is_help=True)

_state: dict[str, IngestionPipeline] = {}


def get_pipeline() -> IngestionPipeline:
    if "pipeline" not in _state:
        _state["pipeline"] = IngestionPipeline(Settings.from_env())
    return _state["pipeline"]


@app.command()
def ingest(
    repo: str = typer.Argument(..., help="GitHub URL (owner/name or full URL) or local path"),
    branch: str | None = typer.Option(None, "--branch", "-b"),
    incremental: bool = typer.Option(
        False, "--incremental", "-i", help="Skip files whose checksum is unchanged"
    ),
) -> None:
    """Ingest a repository: clone/pull, chunk, embed, index."""
    pipeline = get_pipeline()
    mode = "incremental" if incremental else "full"
    typer.echo(f"Ingesting {repo} ({mode})...")
    stats = pipeline.ingest(repo, branch=branch, full=not incremental)
    typer.echo(
        f"Done. files scanned: {stats['files_scanned']}, indexed: {stats['files_indexed']}, "
        f"skipped: {stats['files_skipped']}; chunks +{stats['chunks_added']} / -{stats['chunks_removed']} "
        f"(total {stats.get('chunks_total', '?')})"
    )


@app.command()
def query(
    q: str = typer.Argument(..., help="Natural-language question about the codebase"),
    k: int = typer.Option(5, "--k", "-k", min=1, max=20),
    as_json: bool = typer.Option(False, "--json", help="Print the full response as JSON"),
) -> None:
    """Ask the codebase a question (answer includes [n] citations)."""
    result = get_pipeline().query(q, top_k=k)
    if as_json:
        slim = {
            "query": result["query"],
            "answer": result["answer"],
            "citations": result["citations"],
            "timings": result["timings"],
            "answerer": result["answerer"],
        }
        typer.echo(json.dumps(slim, indent=2))
        return
    typer.echo(result["answer"])
    typer.echo("")
    for c in result["citations"]:
        sym = f" ({c['symbol']})" if c.get("symbol") else ""
        typer.echo(f"  [{c['index']}] {c['file_path']}:{c['start_line']}-{c['end_line']}{sym}")
    t = result["timings"]
    typer.echo(
        f"\n({result['answerer']}: retrieve {t['retrieve_ms']}ms, rerank {t['rerank_ms']}ms, answer {t['answer_ms']}ms)"
    )


@app.command("open")
def open_(
    target: str = typer.Argument(..., help="file path with optional :line, e.g. src/foo.py:42"),
    context: int = typer.Option(8, "--context", "-c"),
) -> None:
    """Show a file location and the indexed chunk that covers it."""
    path, _, line_s = target.partition(":")
    line = int(line_s) if line_s.isdigit() else None
    pipeline = get_pipeline()
    repo = next(iter(pipeline.db.list_repos()), None)
    if repo is None:
        typer.echo("No repo ingested yet. Run: z ingest <repo>", err=True)
        raise typer.Exit(1)
    repo_path = Path(repo["path"])
    text = read_worktree_file(repo_path, path)
    if text is None:
        typer.echo(f"File not found in ingested repo: {path}", err=True)
        raise typer.Exit(1)
    lines = text.splitlines()
    start = max(1, (line or 1) - context)
    end = min(len(lines), (line or context) + context)
    typer.echo(f"--- {path}:{start}-{end} ---")
    for n in range(start, end + 1):
        marker = ">" if line and n == line else " "
        typer.echo(f"{marker} {n:>4}  {lines[n - 1]}")
    if line:
        for chunk in pipeline.db._conn.execute(
            "SELECT * FROM chunks WHERE file_path = ? AND start_line <= ? AND end_line >= ?",
            (path, line, line),
        ).fetchall():
            typer.echo(
                f"\nIndexed chunk {chunk['chunk_id']} ({chunk['kind']}"
                f"{', ' + chunk['symbol'] if chunk['symbol'] else ''}) "
                f"lines {chunk['start_line']}-{chunk['end_line']}, commit {(chunk['commit_id'] or '')[:8]}"
            )
            break


@app.command()
def status() -> None:
    """Index health: counts, backends, staleness."""
    pipeline = get_pipeline()
    s = pipeline.status()
    typer.echo(f"repos: {len(s['repos'])}, files: {s['files']}, chunks: {s['chunks']}")
    typer.echo(f"vectors: {s['vectors']} ({s['vector_backend']}), bm25 docs: {s['bm25_docs']}")
    typer.echo(f"embedder: {s['embedder']}, reranker: {s['reranker']}, answerer: {s['answerer']}")
    for r in s["repos"]:
        typer.echo(f"  repo: {r['url']} head={str(r.get('head_commit'))[:8]}")
    if s["stale_files"]:
        typer.echo(f"stale files ({len(s['stale_files'])}): {', '.join(s['stale_files'][:10])}")
        typer.echo("run `z reindex` to refresh them")
    else:
        typer.echo("no stale files")


@app.command()
def reindex(
    file_path: str | None = typer.Argument(None, help="Optional single file to re-chunk"),
) -> None:
    """Incrementally reindex changed files (or one specific file)."""
    stats = get_pipeline().reindex(file_path=file_path)
    typer.echo(
        f"reindexed: {stats['files_reindexed']}, removed: {stats['files_removed']}, "
        f"chunks +{stats['chunks_added']} / -{stats['chunks_removed']}"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the FastAPI backend + web UI."""
    import uvicorn

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
