"""Semantic code chunking.

Strategy
--------
- Python / JavaScript / TypeScript: tree-sitter grammars when importable.
  Chunks are functions (incl. decorated ones), classes plus their individual
  methods, module docstrings, and any top-level statement remainder.
- Markdown: split on headings.
- Everything else (or when grammars are unavailable / parsing fails):
  boundary-aware sliding windows with overlap.

Every chunk carries file path, 1-based start/end lines, language, symbol and a
kind. The pipeline attaches commit metadata (author/date/commit id) and derives
chunk ids from path+lines+commit.
"""

from __future__ import annotations

import hashlib
import re

EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".md": "markdown",
    ".rst": "markdown",
}

_GRAMMARS: dict[str, object] = {}


def _load_grammars() -> dict[str, object]:
    """Import tree-sitter grammars defensively; the chunker works without them."""
    if _GRAMMARS:
        return _GRAMMARS
    try:
        import tree_sitter_python as tsp
        from tree_sitter import Language

        _GRAMMARS["python"] = Language(tsp.language())
    except Exception:
        pass
    try:
        import tree_sitter_javascript as tsj
        from tree_sitter import Language

        _GRAMMARS["javascript"] = Language(tsj.language())
    except Exception:
        pass
    try:
        import tree_sitter_typescript as tst
        from tree_sitter import Language

        _GRAMMARS["typescript"] = Language(tst.language_typescript())
    except Exception:
        pass
    return _GRAMMARS


def _make_parser(lang):
    from tree_sitter import Parser

    try:
        return Parser(lang)  # tree-sitter >= 0.22
    except TypeError:  # pragma: no cover - old bindings
        parser = Parser()
        try:
            parser.language = lang
        except Exception:
            parser.set_language(lang)
        return parser


def detect_language(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return EXT_TO_LANGUAGE.get(f".{ext}", "text")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_chunk(
    file_path: str,
    start_line: int,
    end_line: int,
    language: str,
    symbol: str | None,
    kind: str,
    text: str,
) -> dict:
    return {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": language,
        "symbol": symbol,
        "kind": kind,
        "text": text,
        "summary": summarize(text),
        "checksum": sha256_hex(text),
    }


def summarize(text: str) -> str:
    """Cheap deterministic summary: docstring/comment/first meaningful line."""
    if not text.strip():
        return ""
    doc = re.search(r'"""(.*?)"""', text, re.DOTALL) or re.search(r"'''(.*?)'''", text, re.DOTALL)
    if doc:
        first = doc.group(1).strip().splitlines()
        if first:
            return first[0].strip()[:160]
    comment = re.search(r"^\s*(?:#|//)\s*(.+)$", text, re.MULTILINE)
    if comment:
        return comment.group(1).strip()[:160]
    for line in text.strip().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "//")):
            return line[:160]
    return ""


def _window_ranges(start: int, end: int, max_lines: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) windows of at most max_lines lines (0-based)."""
    step = max(1, max_lines - overlap)
    ranges = []
    s = start
    while s < end:
        e = min(s + max_lines, end)
        ranges.append((s, e))
        if e == end:
            break
        s += step
    return ranges


# ---------------------------------------------------------------- python (AST)


def _chunk_python(src: str, file_path: str, max_lines: int, overlap: int) -> list[dict]:
    _load_grammars()
    parser = _make_parser(_GRAMMARS["python"])
    tree = parser.parse(src.encode("utf-8"))
    lines = src.splitlines()
    chunks: list[dict] = []
    covered: list[tuple[int, int]] = []

    def line_of(byte: int) -> int:
        return src.count("\n", 0, byte)

    def sl(node) -> int:
        return line_of(node.start_byte) + 1

    def el(node) -> int:
        return line_of(node.end_byte) + 1

    def add(start: int, end: int, symbol: str | None, kind: str) -> None:
        text = "\n".join(lines[start - 1 : end])
        if not text.strip():
            return
        total = end - start + 1
        if total <= max_lines:
            chunks.append(make_chunk(file_path, start, end, "python", symbol, kind, text))
            return
        n = 1
        for ws, we in _window_ranges(start - 1, end, max_lines, overlap):
            wtext = "\n".join(lines[ws:we])
            sym = f"{symbol}::part{n}" if symbol else None
            chunks.append(make_chunk(file_path, ws + 1, we, "python", sym, kind, wtext))
            n += 1

    def emit_function(outer, node, prefix: str | None = None) -> None:
        name_node = node.child_by_field_name("name")
        base = src[name_node.start_byte : name_node.end_byte] if name_node else None
        symbol = f"{prefix}.{base}" if prefix and base else base
        start, end = sl(outer), el(node)
        add(start, end, symbol, "function")
        covered.append((start, end))

    def emit_class(node) -> None:
        name_node = node.child_by_field_name("name")
        symbol = src[name_node.start_byte : name_node.end_byte] if name_node else "<class>"
        start, end = sl(node), el(node)
        body = node.child_by_field_name("body")
        doc_end = None
        if body is not None:
            for child in body.named_children:
                if (
                    child.type == "expression_statement"
                    and child.named_children
                    and child.named_children[0].type == "string"
                ):
                    doc_end = el(child)
                    break
        total = end - start + 1
        if total <= max_lines:
            add(start, end, symbol, "class")
            covered.append((start, end))
        else:
            header_end = doc_end or min(start + 14, end)
            add(start, header_end, symbol, "class")
            covered.append((start, header_end))
        if body is not None:
            for child in body.named_children:
                if child.type == "function_definition":
                    emit_function(child, child, prefix=symbol)
                elif child.type == "decorated_definition":
                    inner = child.named_children[-1] if child.named_children else None
                    if inner is not None and inner.type == "function_definition":
                        emit_function(child, inner, prefix=symbol)

    root = tree.root_node
    top = list(root.named_children)
    # Module docstring
    if (
        top
        and top[0].type == "expression_statement"
        and top[0].named_children
        and top[0].named_children[0].type == "string"
    ):
        add(sl(top[0]), el(top[0]), "<module>", "module-docstring")
        covered.append((sl(top[0]), el(top[0])))
        top = top[1:]

    for node in top:
        if node.type == "decorated_definition":
            inner = node.named_children[-1] if node.named_children else None
            if inner is not None and inner.type == "function_definition":
                emit_function(node, inner)
            elif inner is not None and inner.type == "class_definition":
                emit_class(node)
            else:
                add(sl(node), el(node), None, "module")
                covered.append((sl(node), el(node)))
        elif node.type == "function_definition":
            emit_function(node, node)
        elif node.type == "class_definition":
            emit_class(node)
        else:
            add(sl(node), el(node), None, "module")
            covered.append((sl(node), el(node)))

    # Remainder lines not covered by any chunk (comments, blank separators).
    covered_set: set[int] = set()
    for s, e in covered:
        covered_set.update(range(s, e + 1))
    run: list[int] = []
    for i in range(1, len(lines) + 1):
        if i not in covered_set and lines[i - 1].strip():
            run.append(i)
        elif run:
            add(run[0], run[-1], "<module>", "module")
            run = []
    if run:
        add(run[0], run[-1], "<module>", "module")
    return chunks


# ---------------------------------------------------- javascript / typescript


def _chunk_js(src: str, file_path: str, language: str, max_lines: int, overlap: int) -> list[dict]:
    _load_grammars()
    parser = _make_parser(_GRAMMARS[language])
    tree = parser.parse(src.encode("utf-8"))
    lines = src.splitlines()
    chunks: list[dict] = []
    covered: list[tuple[int, int]] = []

    def line_of(byte: int) -> int:
        return src.count("\n", 0, byte)

    def sl(node) -> int:
        return line_of(node.start_byte) + 1

    def el(node) -> int:
        return line_of(node.end_byte) + 1

    def add(start: int, end: int, symbol: str | None, kind: str) -> None:
        text = "\n".join(lines[start - 1 : end])
        if not text.strip():
            return
        total = end - start + 1
        if total <= max_lines:
            chunks.append(make_chunk(file_path, start, end, language, symbol, kind, text))
            return
        n = 1
        for ws, we in _window_ranges(start - 1, end, max_lines, overlap):
            wtext = "\n".join(lines[ws:we])
            sym = f"{symbol}::part{n}" if symbol else None
            chunks.append(make_chunk(file_path, ws + 1, we, language, sym, kind, wtext))
            n += 1

    def name_of(node) -> str | None:
        name_node = node.child_by_field_name("name")
        return src[name_node.start_byte : name_node.end_byte] if name_node else None

    def emit_value_symbol(node) -> str | None:
        """`const foo = () => {}` / `export const foo = function(){}`."""
        for declarator in node.named_children:
            if declarator.type not in ("variable_declarator",):
                continue
            value = declarator.child_by_field_name("value")
            if value is not None and value.type in (
                "arrow_function",
                "function_expression",
                "function",
            ):
                return name_of(declarator)
        return None

    def emit_class(node) -> None:
        symbol = name_of(node) or "<class>"
        start, end = sl(node), el(node)
        if end - start + 1 <= max_lines:
            add(start, end, symbol, "class")
            covered.append((start, end))
        else:
            add(start, min(start + 14, end), symbol, "class")
            covered.append((start, min(start + 14, end)))
        body = node.child_by_field_name("body")
        if body is not None:
            for child in body.named_children:
                if child.type == "method_definition":
                    add(
                        sl(child),
                        el(child),
                        f"{symbol}.{name_of(child) or ''}",
                        "method",
                    )
                    covered.append((sl(child), el(child)))

    def handle(node) -> None:
        inner = node
        if node.type == "export_statement":
            named = node.named_children
            inner = next(
                (
                    c
                    for c in named
                    if c.type
                    in (
                        "function_declaration",
                        "class_declaration",
                        "lexical_declaration",
                        "variable_declaration",
                    )
                ),
                None,
            )
            if inner is None:
                add(sl(node), el(node), None, "module")
                covered.append((sl(node), el(node)))
                return
        if inner.type in ("function_declaration", "generator_function_declaration"):
            add(sl(node), el(inner), name_of(inner), "function")
            covered.append((sl(node), el(inner)))
        elif inner.type == "class_declaration":
            emit_class(inner)
            covered.append((sl(node), el(inner)))
        elif inner.type in ("lexical_declaration", "variable_declaration"):
            sym = emit_value_symbol(inner)
            if sym is not None:
                add(sl(node), el(inner), sym, "function")
                covered.append((sl(node), el(inner)))
            else:
                add(sl(node), el(inner), None, "module")
                covered.append((sl(node), el(inner)))
        else:
            add(sl(node), el(inner), None, "module")
            covered.append((sl(node), el(inner)))

    for node in tree.root_node.named_children:
        handle(node)

    covered_set: set[int] = set()
    for s, e in covered:
        covered_set.update(range(s, e + 1))
    run: list[int] = []
    for i in range(1, len(lines) + 1):
        if i not in covered_set and lines[i - 1].strip():
            run.append(i)
        elif run:
            add(run[0], run[-1], "<module>", "module")
            run = []
    if run:
        add(run[0], run[-1], "<module>", "module")
    return chunks


# ------------------------------------------------------------------- fallback


_PY_BOUNDARY = re.compile(r"^\s*(async\s+def|def|class)\b")
_JS_BOUNDARY = re.compile(
    r"^\s*(export\s+)?(async\s+)?(function|class)\b|^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?(\(|function)"
)
_MD_BOUNDARY = re.compile(r"^#{1,3}\s")
_GENERIC_BOUNDARY = re.compile(
    r"^\s*(export\s+)?(async\s+)?(function|class)\b|^\s*(def|func|fn|class|struct)\b|"
    r"^\s*(public|private|protected)\s+(static\s+)?[\w<>\[\]]+\s+\w+\s*\("
)
_SYM_PY = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)|^\s*class\s+(\w+)")
_SYM_JS = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+))"
)
_SYM_GENERIC = re.compile(r"^\s*(?:func|fn|function|def|class|struct)\s+(\w+)")


def _fallback_chunks(src: str, file_path: str, language: str, max_lines: int, overlap: int) -> list[dict]:
    lines = src.splitlines()
    if language == "markdown":
        boundary = _MD_BOUNDARY
    elif language == "python":
        boundary = _PY_BOUNDARY
    else:
        boundary = _GENERIC_BOUNDARY
    sym_re = (
        _SYM_PY
        if language == "python"
        else (_SYM_JS if language in ("javascript", "typescript") else _SYM_GENERIC)
    )

    starts = [i for i, ln in enumerate(lines) if boundary.match(ln)]
    blocks: list[tuple[int, int, bool]] = []  # (start, end, starts_on_boundary)
    if not starts:
        blocks.extend((s, e, False) for s, e in _window_ranges(0, len(lines), max_lines, overlap))
    else:
        if starts[0] > 0:
            blocks.extend((s, e, False) for s, e in _window_ranges(0, starts[0], max_lines, overlap))
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(lines)
            if e - s > max_lines:
                blocks.extend((ws, we, True) for ws, we in _window_ranges(s, e, max_lines, overlap))
            else:
                blocks.append((s, e, True))

    chunks = []
    for s, e, is_boundary in blocks:
        text = "\n".join(lines[s:e])
        if not text.strip():
            continue
        symbol = None
        kind = "block"
        if is_boundary:
            m = sym_re.match(lines[s]) if s < len(lines) else None
            if m:
                symbol = next((g for g in m.groups() if g), None)
                if language == "python" and lines[s].lstrip().startswith("class"):
                    kind = "class"
                elif symbol:
                    kind = "function"
        chunks.append(make_chunk(file_path, s + 1, e, language, symbol, kind, text))
    return chunks


def chunk_file(
    src: str,
    file_path: str,
    max_chunk_lines: int = 120,
    overlap_lines: int = 12,
) -> list[dict]:
    """Chunk one file into semantic units with metadata."""
    language = detect_language(file_path)
    if not src.strip():
        return []
    if language == "python" and "python" in _load_grammars():
        try:
            return _chunk_python(src, file_path, max_chunk_lines, overlap_lines)
        except Exception:
            pass
    if language in ("javascript", "typescript") and language in _load_grammars():
        try:
            return _chunk_js(
                src,
                file_path,
                "javascript" if language == "javascript" else "typescript",
                max_chunk_lines,
                overlap_lines,
            )
        except Exception:
            pass
    return _fallback_chunks(src, file_path, language, max_chunk_lines, overlap_lines)


def chunk_id_for(file_path: str, start_line: int, end_line: int, commit_id: str) -> str:
    """Stable chunk identity: sha256(path + lines + commit), per the spec schema."""
    raw = f"{file_path}:{start_line}:{end_line}:{commit_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]
