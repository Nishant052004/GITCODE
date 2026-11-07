"""Shared fixtures: a temporary git clone of the demo repo and a test pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings
from backend.app.services.pipeline import IngestionPipeline

DEMO_REPO_SRC = REPO_ROOT / "examples" / "demo_repo"


@pytest.fixture(scope="session")
def demo_git_repo(tmp_path_factory) -> Path:
    """Copy examples/demo_repo into a tmp dir and commit it as a git repo."""
    dst = tmp_path_factory.mktemp("demo") / "demo_repo"
    shutil.copytree(DEMO_REPO_SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Eval Bot",
        "GIT_AUTHOR_EMAIL": "eval@example.com",
        "GIT_COMMITTER_NAME": "Eval Bot",
        "GIT_COMMITTER_EMAIL": "eval@example.com",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
    }

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(dst), *args], check=True, capture_output=True, env=env)

    git("init", "-b", "main")
    git("add", ".")
    git("commit", "-m", "initial demo repo")
    return dst


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        repo_workdir=tmp_path / "repos",
        embedding_backend="hashing",
        embedding_dim=256,
        vector_backend="numpy",
        reranker="heuristic",
    )


@pytest.fixture()
def pipeline(settings: Settings) -> IngestionPipeline:
    return IngestionPipeline(settings)


@pytest.fixture()
def ingested_pipeline(pipeline: IngestionPipeline, demo_git_repo: Path) -> IngestionPipeline:
    pipeline.ingest(str(demo_git_repo))
    return pipeline
