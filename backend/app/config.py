"""Central configuration loaded from environment variables (with .env support)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXTENSIONS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rb",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".php",
    ".swift",
    ".kt",
    ".md",
    ".rst",
)
DEFAULT_EXCLUDES = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    ".idea",
    ".vscode",
)


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't require python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class Settings:
    """All tunables in one place; every value can be overridden via env."""

    # Storage
    data_dir: Path = field(default_factory=lambda: Path("./data").resolve())
    repo_workdir: Path = field(default_factory=lambda: Path("./data/repos").resolve())

    # Embeddings / vector store
    embedding_backend: str = "auto"  # auto | openai | sentence-transformers | hashing
    embedding_dim: int = 512  # hashing fallback dimension
    vector_backend: str = "numpy"  # numpy | qdrant
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # OpenAI-compatible LLM (optional)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Reranking
    reranker: str = "heuristic"  # heuristic | cross-encoder | auto
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ingestion / chunking
    file_extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDES
    max_chunk_lines: int = 120
    overlap_lines: int = 12

    # Retrieval
    vector_top_n: int = 20
    bm25_top_m: int = 20
    candidate_pool: int = 50
    rrf_k: int = 60
    weight_vector: float = 0.5
    weight_bm25: float = 0.5
    recency_boost: float = 0.05
    top_k: int = 5

    # Answering
    context_max_chunks: int = 6
    context_max_chars: int = 1800

    # GitHub enrichment (optional)
    github_token: str | None = None

    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        if env_file is None:
            env_file = Path(os.getenv("ENV_FILE", ".env"))
        _load_dotenv(env_file)
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "./data")).resolve(),
            repo_workdir=Path(os.getenv("REPO_WORKDIR", "./data/repos")).resolve(),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "auto").strip().lower(),
            embedding_dim=_int("EMBEDDING_DIM", 512),
            vector_backend=os.getenv("VECTOR_BACKEND", "numpy").strip().lower(),
            st_model=os.getenv("ST_MODEL", cls.st_model),
            openai_api_key=(os.getenv("OPENAI_API_KEY") or None),
            openai_base_url=os.getenv("OPENAI_BASE_URL", cls.openai_base_url),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", cls.openai_chat_model),
            openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", cls.openai_embed_model),
            reranker=os.getenv("RERANKER", "heuristic").strip().lower(),
            rerank_model=os.getenv("RERANK_MODEL", cls.rerank_model),
            max_chunk_lines=_int("MAX_CHUNK_LINES", 120),
            overlap_lines=_int("OVERLAP_LINES", 12),
            vector_top_n=_int("VECTOR_TOP_N", 20),
            bm25_top_m=_int("BM25_TOP_M", 20),
            candidate_pool=_int("CANDIDATE_POOL", 50),
            rrf_k=_int("RRF_K", 60),
            weight_vector=_float("WEIGHT_VECTOR", 0.5),
            weight_bm25=_float("WEIGHT_BM25", 0.5),
            recency_boost=_float("RECENCY_BOOST", 0.05),
            top_k=_int("TOP_K", 5),
            context_max_chunks=_int("CONTEXT_MAX_CHUNKS", 6),
            context_max_chars=_int("CONTEXT_MAX_CHARS", 1800),
            github_token=os.getenv("GITHUB_TOKEN") or None,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
