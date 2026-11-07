# Codebase Intelligence & Developer Copilot

An end-to-end system that ingests a GitHub repository (code, docs, commits, optionally PRs and
issues), chunks the code semantically (functions / classes / modules / docblocks), embeds the
chunks into a vector index alongside a BM25 lexical index, and answers developer questions with
**citations down to file, line range, commit and author** — via a FastAPI backend, a web chat UI,
and a `z` CLI.

```
 ingest ──► chunk (tree-sitter / fallback) ──► embed ──► vector store ─┐
 (git + optional                                      (numpy / Qdrant) ├─► hybrid retrieval ─► reranker ─► cited answer
  GitHub REST)                                        BM25 lexical ────┘   (RRF fusion)     (heuristic /   (LLM or
                                                                                            cross-encoder)  extractive)
        ▲                                                                                              │
        └──────────── staleness detection + incremental reindex (per-file commit heads) ◄──────────────┘
```

## Design principles

- **Runs with zero API keys and zero external services.** Every heavy component has a local
  fallback: hashing embeddings when no embedding model is available, a pure-Python BM25 index
  instead of Elasticsearch, a numpy vector store instead of a vector DB, a heuristic reranker,
  and a deterministic extractive answerer when no LLM key is configured.
- **Every component is swappable.** Backends are selected by env vars (`EMBEDDING_BACKEND`,
  `VECTOR_BACKEND`, `RERANKER`); each implementation sits behind a small interface.
- **Citations are load-bearing.** Chunk ids are `sha256(path:start:end:commit)`, so answers
  reference an exact commit of an exact line range. The LLM answerer drops citations that don't
  map to a provided snippet and falls back to the extractive answerer if the model produces an
  uncited answer (hallucination guard).

## Quickstart (local, no keys)

Requires Python 3.10+ and git.

```bash
pip install -r requirements.txt -r requirements-dev.txt

# 1. Ingest the bundled demo repo (or any local path / GitHub URL)
python -m cli.main ingest examples/demo_repo

# 2. Ask a question
python -m cli.main query "How do we verify a user's password?" --k 3

# 3. Or run the API + web UI
python -m cli.main serve            # http://127.0.0.1:8000
```

The web UI is a chat interface at the server root: answers render citation chips that open the
exact chunk (file, lines, commit, author) in a viewer, plus the full reranked candidate list and
per-stage latencies.

Docker: `docker compose up --build` (add `--profile qdrant` to also start a Qdrant server).

## Environment configuration

Copy `.env.example` to `.env`. Highlights (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` / `REPO_WORKDIR` | `./data` / `./data/repos` | Indexes, SQLite DB, cloned repos |
| `EMBEDDING_BACKEND` | `auto` | `openai` \| `sentence-transformers` \| `hashing` |
| `VECTOR_BACKEND` | `numpy` | `numpy` (local .npz) \| `qdrant` (client; server or local-path mode) |
| `OPENAI_API_KEY` | – | Enables LLM answers + OpenAI embeddings when `EMBEDDING_BACKEND=auto` |
| `OPENAI_BASE_URL` / `OPENAI_CHAT_MODEL` | OpenAI defaults | Point at any OpenAI-compatible API |
| `RERANKER` | `heuristic` | `cross-encoder` after `pip install sentence-transformers` |
| `GITHUB_TOKEN` | – | Enriches ingestion with PRs/issues via the GitHub REST API |
| `TOP_K`, `VECTOR_TOP_N`, `BM25_TOP_M`, `WEIGHT_VECTOR`, `WEIGHT_BM25`, `RECENCY_BOOST` | see `.env.example` | Retrieval tuning |

Embedding resolution under `auto`: OpenAI key → OpenAI; `sentence-transformers` importable →
MiniLM; otherwise → deterministic hashing embedder (lexical, no network). Retrieval quality with
hashing is good on exact identifier questions and improves markedly with a neural backend.

## API

Interactive docs at `/docs`. Core endpoints:

```bash
# Ingest (background job) / poll status
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
     -d '{"repo": "https://github.com/owner/repo", "branch": "main"}'
curl localhost:8000/ingest/<job_id>

# Synchronous ingest (tests/CI)
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" \
     -d '{"repo": "examples/demo_repo", "sync": true}'

# Ask
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"query": "why does login fail for unknown users?", "top_k": 5}'
# -> {"answer", "citations":[{index, chunk_id, file_path, start_line, end_line, commit_id}],
#     "candidates":[... per-candidate vector/bm25/fusion scores ...], "timings":{...}}

# Staleness / maintenance
curl -X POST localhost:8000/reindex -H "Content-Type: application/json" -d '{}'
curl -X POST localhost:8000/reindex -H "Content-Type: application/json" -d '{"file_path": "src/foo.py"}'
curl localhost:8000/status
curl localhost:8000/chunks/<chunk_id>
```

## CLI (`z`)

```bash
python -m cli.main ingest <repo-url-or-path> [--branch b] [--incremental]
python -m cli.main query "question" [--k 5] [--json]
python -m cli.main open src/acme/auth.py:42 [--context 8]   # show source + the indexed chunk covering it
python -m cli.main status                                   # counts, backends, stale files
python -m cli.main reindex [file_path]
python -m cli.main serve [--port 8000]
```

## How ingestion works

1. Local paths are used as-is; GitHub URLs are cloned (or fetched+reset on re-ingest) into
   `REPO_WORKDIR`.
2. Files are listed via `git ls-files -co --exclude-standard` (respects `.gitignore`, includes
   untracked files), filtered by extension (`.py .js .ts ... .md`) and exclude dirs.
3. **Chunking** — tree-sitter grammars for Python/JavaScript/TypeScript: functions (incl.
   decorated), classes plus each method (`Class.method` symbols), module docstrings, top-level
   remainder. Long units are split into overlapping windows (`::partN`). Other languages get a
   boundary-aware fallback (`def|func|fn|class|struct...`); markdown splits on headings.
4. **Metadata** — per file, `git log -1` gives commit/author/date; per chunk we store
   `chunk_id, file_path, start/end line, language, symbol, kind, commit_id, author,
   last_modified, checksum, summary, text` in SQLite.
5. **Indexing** — new chunks are embedded and added to the vector store; all chunks are tokenized
   (identifiers split into snake/camel subtokens) into BM25. Removed chunks are pruned from both.
6. Commits are recorded; PRs/issues are fetched only when `GITHUB_TOKEN` is set and the URL is
   github.com.

### Staleness & incremental reindexing

Each file's stored HEAD commit is compared with the current `git log -1` head. `z status`
reports stale files; `z reindex` re-chunks and re-embeds **only changed files** (deleted files
are pruned, new files are picked up). Because chunk ids embed the commit id, answers cite the
exact commit a chunk came from — a new commit produces new chunk ids (citation versioning).

## Evaluation

The bundled dataset (`eval/queries.json` + `eval/ground_truth.json`) targets
`examples/demo_repo`; ground truth is expressed as file+symbol so it survives line shifts.

```bash
python eval/run_retrieval_eval.py --repo examples/demo_repo    # Recall@K, Precision@K, MRR, latency
python eval/run_answer_eval.py   --repo examples/demo_repo     # citation accuracy, latency percentiles,
                                                               # + answer_log.jsonl for human grading
```

Reference numbers with the zero-dependency profile (hashing embeddings + heuristic reranker):
Recall@1 ≈ 0.75, Recall@5 = 1.0, MRR ≈ 0.82, end-to-end latency ≈ 10 ms. Machine-checkable
citation accuracy equals Precision@K of the same rank list (the extractive answerer cites every
candidate it presents); `answers fully grounded` is the strict all-citations-relevant rate.
Neural embeddings (`EMBEDDING_BACKEND=sentence-transformers` or an OpenAI key) raise semantic
match substantially.

## Tests

```bash
pytest -q        # 52 tests: unit (chunker, embedders, BM25, vector store, DB, retrieval,
                 # reranker, answerer) + integration (full pipeline, staleness, reindex)
                 # + e2e (HTTP API incl. background jobs)
ruff check backend cli eval tests && ruff format --check backend cli eval tests
```

## Debugging tips

| Symptom | What to check |
|---|---|
| Ingestion finds 0 files | `git -C <repo> ls-files -co --exclude-standard` output; extension/exclude settings |
| Chunks look wrong | `z open file.py:line` shows the chunk covering a line; unit-test your file against `chunk_file()` directly; tree-sitter failures silently fall back to the line chunker |
| Retrieval misses | Look at `candidates[].scores` in `/query` — `vector` vs `bm25` vs `agreement` tells you which source failed; identifier-heavy queries deliberately tilt toward BM25 |
| Embedding errors | Log the failing batch; OpenAI path retries 3x with backoff — check `OPENAI_API_KEY`/`OPENAI_BASE_URL` |
| Answers say "insufficient evidence" | Index is empty or query is out of domain — `z status`, re-ingest |
| Stale answers after a pull | `z status` lists stale files, `z reindex` fixes them; verify stored vs current heads |
| Reranker slow | Cross-encoder batching is per-query; keep `candidate_pool` moderate or stay on the heuristic reranker |
| Index corruption | Delete `data/` and re-ingest — everything is derived state except clones |

Useful plumbing:

```bash
sqlite3 data/metadata.db "SELECT file_path, symbol, commit_id FROM chunks WHERE file_path='src/acme/auth.py'"
cat data/bm25.json | python -m json.tool | head    # BM25 doc store
python -c "from backend.app.services.chunker import chunk_file; print(len(chunk_file(open('src/acme/auth.py').read(), 'src/acme/auth.py')))"
```

## Project layout

```
backend/app/
  config.py                 # env-driven settings
  main.py                   # FastAPI app factory + static UI
  api/{ingest,query,reindex,status}.py
  models/schemas.py         # pydantic request/response models
  services/
    git_ingest.py           # clone/pull, file listing, git metadata, optional GitHub REST
    chunker.py              # tree-sitter semantic chunking + fallbacks
    tokenizer.py            # shared subtoken-aware tokenizer
    embedder.py             # openai | sentence-transformers | hashing
    vector_store.py         # numpy (default) | qdrant
    bm25_index.py           # pure-python BM25 with persistence
    retriever.py            # query analysis + hybrid RRF fusion + recency
    reranker.py             # heuristic | cross-encoder
    answerer.py             # LLM w/ citation guard | extractive
    metadata_db.py          # SQLite: repos/files/chunks/commits/PRs/issues/jobs
    pipeline.py             # orchestration: ingest, staleness, reindex, query, status
    jobs.py                 # background job runner (thread-based; Celery-shaped API)
  static/                   # dependency-free chat UI (HTML/JS/CSS)
cli/main.py                 # `z` CLI (typer)
examples/demo_repo/         # sample repo used by quickstart + eval
tests/{unit,integration,e2e}
eval/{queries.json,ground_truth.json,run_retrieval_eval.py,run_answer_eval.py}
```

## Production upgrade path

- **Vector DB at scale** → `VECTOR_BACKEND=qdrant` (client included; point `QdrantVectorStore` at
  a server URL) — or implement the same add/remove/search interface for Pinecone/Weaviate.
- **Lexical at scale** → swap `BM25Index` for Elasticsearch/OpenSearch behind the same interface
  (`add/remove/search/save`).
- **Async jobs** → `jobs.py` already models submit/poll with a jobs table; move execution to
  Celery/Redis workers without touching the API contract.
- **Webhooks** → point a GitHub push webhook at `POST /reindex` (it diffs stored vs current heads
  and touches only what changed).
- **Monitoring** → per-stage timings are already returned by `/query`; export them to
  Prometheus/Grafana and alert on p95 latency + stale-file count.
- **Frontend** → the static chat UI is dependency-free on purpose; a React/TypeScript client can
  consume the same `/query` + `/chunks/{id}` endpoints.
