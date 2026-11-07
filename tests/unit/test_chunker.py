"""Unit tests for semantic chunking (tree-sitter path + fallbacks)."""

from backend.app.services.chunker import chunk_file, chunk_id_for, detect_language

PY_SRC = '''"""Wallet module docstring."""
import os


class Wallet:
    """Holds coins and spends them."""

    def __init__(self, coins):
        self.coins = coins

    def spend(self, n):
        if n > self.coins:
            raise ValueError("insufficient")
        self.coins -= n
        return self.coins


def add(a, b):
    """Add two numbers."""
    return a + b
'''

JS_SRC = """const DEFAULT_RETRIES = 3;

export class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  async request(path) {
    const resp = await fetch(this.baseUrl + path);
    return resp.json();
  }
}

export function backoffMillis(attempt) {
  return 250 * Math.pow(2, attempt);
}
"""

GO_SRC = """package main

import "fmt"

// add returns the sum of two ints.
func add(a int, b int) int {
    return a + b
}

func main() {
    fmt.Println(add(1, 2))
}
"""

MD_SRC = """# Guide

Intro text.

## Setup

Run the installer.

## Usage

Call the API.
"""


def by_symbol(chunks):
    return {c["symbol"]: c for c in chunks if c["symbol"]}


def test_python_semantic_chunks():
    chunks = chunk_file(PY_SRC, "src/wallet.py")
    syms = by_symbol(chunks)
    assert "Wallet" in syms and syms["Wallet"]["kind"] == "class"
    assert "Wallet.spend" in syms and syms["Wallet.spend"]["kind"] == "function"
    assert "add" in syms
    module_doc = [c for c in chunks if c["kind"] == "module-docstring"]
    assert module_doc and "Wallet module docstring" in module_doc[0]["text"]
    # line numbers are 1-based and the spend chunk contains its body
    spend = syms["Wallet.spend"]
    assert spend["start_line"] < spend["end_line"]
    assert "insufficient" in spend["text"]
    # docstring survives inside the class chunk
    assert "Holds coins" in syms["Wallet"]["text"]


def test_chunk_ids_are_stable_and_unique():
    chunks = chunk_file(PY_SRC, "src/wallet.py")
    ids = [chunk_id_for(c["file_path"], c["start_line"], c["end_line"], "abc123") for c in chunks]
    assert len(ids) == len(set(ids))
    assert chunk_id_for("src/wallet.py", 1, 2, "abc123") == chunk_id_for("src/wallet.py", 1, 2, "abc123")
    assert chunk_id_for("src/wallet.py", 1, 2, "abc123") != chunk_id_for("src/wallet.py", 1, 3, "abc123")


def test_long_function_is_windowed():
    body = "\n".join(f"    x{i} = {i}" for i in range(200))
    src = f"def big():\n{body}\n    return x0\n"
    chunks = chunk_file(src, "big.py", max_chunk_lines=60, overlap_lines=10)
    parts = [c for c in chunks if c["symbol"] and c["symbol"].startswith("big::part")]
    assert len(parts) >= 3
    assert all(c["end_line"] - c["start_line"] + 1 <= 60 for c in parts)
    # overlap: consecutive parts share lines
    assert parts[1]["start_line"] < parts[0]["end_line"]


def test_javascript_semantic_chunks():
    chunks = chunk_file(JS_SRC, "frontend/client.js")
    syms = by_symbol(chunks)
    assert "ApiClient" in syms and syms["ApiClient"]["kind"] == "class"
    assert "ApiClient.request" in syms
    assert "backoffMillis" in syms
    assert syms["backoffMillis"]["text"].lstrip().startswith("export")


def test_fallback_chunker_unknown_language():
    chunks = chunk_file(GO_SRC, "cmd/main.go")
    assert chunks, "fallback must produce chunks for ungrammared languages"
    syms = by_symbol(chunks)
    assert "add" in syms
    assert all(c["language"] == "go" for c in chunks)


def test_markdown_chunked_by_heading():
    chunks = chunk_file(MD_SRC, "docs/guide.md")
    assert len(chunks) >= 3
    assert any("Setup" in c["text"] for c in chunks)


def test_detect_language():
    assert detect_language("a/b/c.py") == "python"
    assert detect_language("x.tsx") == "typescript"
    assert detect_language("noext") == "text"


def test_empty_source_yields_no_chunks():
    assert chunk_file("   \n", "empty.py") == []
