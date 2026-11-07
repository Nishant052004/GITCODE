"""Answer generation with enforced citations.

- LLMAnswerer: OpenAI-compatible chat completion. The prompt only allows
  answers grounded in the numbered context snippets and requires [n]
  citations; a post-processor drops citations that don't map to a provided
  snippet, and answers without any valid citation fall back to the extractive
  answerer (hallucination guard).
- ExtractiveAnswerer: deterministic, no-LLM fallback that always cites the
  retrieved chunks it presents.
"""

from __future__ import annotations

import re
import time

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough relevant code in the indexed repository to answer "
    "this question. Try ingesting the repository, waiting for reindexing to "
    "finish, or rephrasing the query."
)

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


class ExtractiveAnswerer:
    name = "extractive"

    def __init__(self, max_chunks: int = 6, snippet_lines: int = 15):
        self.max_chunks = max_chunks
        self.snippet_lines = snippet_lines

    def answer(self, query: str, candidates: list[dict], top_k: int) -> tuple[str, list[dict]]:
        picks = candidates[: max(1, min(top_k, self.max_chunks))]
        if not picks:
            return INSUFFICIENT_EVIDENCE, []
        lines = [f"Found {len(picks)} relevant location(s) in the codebase:", ""]
        citations: list[dict] = []
        for i, cand in enumerate(picks, 1):
            symbol = f" **{cand['symbol']}**" if cand.get("symbol") else ""
            commit = (cand.get("commit_id") or "worktree")[:8]
            author = cand.get("author") or "unknown"
            lines.append(
                f"{i}.{symbol} `{cand['file_path']}:{cand['start_line']}-{cand['end_line']}` [{i}] "
                f"— last touched by **{author}** (commit `{commit}`)"
            )
            if cand.get("summary"):
                lines.append(f"   > {cand['summary']}")
            citations.append(
                {
                    "index": i,
                    "chunk_id": cand["chunk_id"],
                    "file_path": cand["file_path"],
                    "start_line": cand["start_line"],
                    "end_line": cand["end_line"],
                    "symbol": cand.get("symbol"),
                    "commit_id": cand.get("commit_id"),
                }
            )
        lines.append("")
        top = picks[0]
        snippet = "\n".join(top["text"].splitlines()[: self.snippet_lines])
        lines += [
            f"Top match — `{top['file_path']}:{top['start_line']}` [1]:",
            f"```{top['language']}",
            snippet,
            "```",
        ]
        return "\n".join(lines), citations


class LLMAnswerer:
    name = "openai"

    SYSTEM_PROMPT = (
        "You are a codebase assistant. You answer developer questions strictly "
        "using the numbered context snippets provided in the user message.\n"
        "Rules:\n"
        "1. Cite every claim with the snippet number in brackets, e.g. [2].\n"
        "2. Never invent files, functions, APIs or behaviour not present in the "
        "snippets.\n"
        "3. If the snippets are insufficient, reply with exactly: "
        "INSUFFICIENT_EVIDENCE"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        max_chunks: int = 6,
        max_chars: int = 1800,
        max_context_chars: int = 12000,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_chunks = max_chunks
        self.max_chars = max_chars
        self.max_context_chars = max_context_chars
        self._fallback = ExtractiveAnswerer(max_chunks=max_chunks)

    def _build_context(self, candidates: list[dict]) -> str:
        blocks, budget = [], self.max_context_chars
        for i, cand in enumerate(candidates[: self.max_chunks], 1):
            header = (
                f"[{i}] {cand['file_path']}:{cand['start_line']}-{cand['end_line']} "
                f"(symbol={cand.get('symbol')}, commit={(cand.get('commit_id') or 'worktree')[:8]})"
            )
            body = cand.get("text", "")[: self.max_chars]
            block = f"{header}\n{body}"
            if budget - len(block) < 0 and blocks:
                break
            blocks.append(block)
            budget -= len(block)
        return "\n\n".join(blocks)

    def _chat(self, query: str, context: str) -> str:
        import httpx

        last_err: Exception | None = None
        delay = 0.5
        for attempt in range(3):
            try:
                with httpx.Client(timeout=90) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": f"Question: {query}\n\nContext snippets:\n\n{context}",
                                },
                            ],
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"] or ""
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def answer(self, query: str, candidates: list[dict], top_k: int) -> tuple[str, list[dict]]:
        picks = candidates[: max(1, min(top_k, self.max_chunks))]
        if not picks:
            return INSUFFICIENT_EVIDENCE, []
        try:
            content = self._chat(query, self._build_context(picks))
        except RuntimeError:
            return self._fallback.answer(query, candidates, top_k)

        if "INSUFFICIENT_EVIDENCE" in content:
            return INSUFFICIENT_EVIDENCE, []

        valid = set(range(1, len(picks) + 1))
        used = {int(n) for n in _CITATION_RE.findall(content) if int(n) in valid}
        if not used:
            # Hallucination guard: an uncited answer is not acceptable.
            return self._fallback.answer(query, candidates, top_k)

        citations = []
        for i in sorted(used):
            cand = picks[i - 1]
            citations.append(
                {
                    "index": i,
                    "chunk_id": cand["chunk_id"],
                    "file_path": cand["file_path"],
                    "start_line": cand["start_line"],
                    "end_line": cand["end_line"],
                    "symbol": cand.get("symbol"),
                    "commit_id": cand.get("commit_id"),
                }
            )
        return content.strip(), citations


def get_answerer(
    openai_api_key: str | None,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    max_chunks: int = 6,
    max_chars: int = 1800,
):
    if openai_api_key:
        return LLMAnswerer(openai_api_key, model, base_url, max_chunks, max_chars)
    return ExtractiveAnswerer(max_chunks=max_chunks)
