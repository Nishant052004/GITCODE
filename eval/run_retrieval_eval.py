"""Retrieval evaluation: Recall@K, Precision@K and MRR.

Run from the repo root:
    python eval/run_retrieval_eval.py --repo examples/demo_repo

A "relevant" hit at rank r is a chunk whose file matches an expected file and
whose symbol (or text) contains the expected symbol. Results are printed as a
table and written to eval/results/retrieval_eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import Settings
from backend.app.services.pipeline import IngestionPipeline


def candidate_matches(candidate: dict, exp: dict) -> bool:
    """One ground-truth item matched by one candidate chunk."""
    if candidate["file_path"].replace("\\", "/") != exp["file"]:
        return False
    if not exp.get("symbol"):
        return True
    symbol = exp["symbol"].lower()
    if symbol in (candidate.get("symbol") or "").lower():
        return True
    return symbol in (candidate.get("text") or "").lower()


def is_relevant(candidate: dict, expected: list[dict]) -> bool:
    """Candidate matches ANY ground-truth item (used for precision and answers)."""
    return any(candidate_matches(candidate, exp) for exp in expected)


def matched_expected(ranked: list[dict], expected: list[dict], k: int | None = None) -> set[int]:
    """Indices of ground-truth items covered by the top-k candidates."""
    top = ranked[:k] if k else ranked
    covered: set[int] = set()
    for i, exp in enumerate(expected):
        if any(candidate_matches(cand, exp) for cand in top):
            covered.add(i)
    return covered


def first_relevant_rank(ranked: list[dict], expected: list[dict]) -> int:
    for i, cand in enumerate(ranked, 1):
        if is_relevant(cand, expected):
            return i
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_ROOT / "examples" / "demo_repo"))
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.json"))
    parser.add_argument("--ground-truth", default=str(REPO_ROOT / "eval" / "ground_truth.json"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", default=str(REPO_ROOT / "eval" / "results" / "retrieval_eval.json"))
    args = parser.parse_args()

    spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    queries = spec["queries"] if isinstance(spec, dict) else spec
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))

    # Deterministic eval defaults: hashing embeddings unless the env overrides.
    import os

    settings = Settings(
        data_dir=REPO_ROOT / "data" / "eval",
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "hashing"),
        reranker=os.getenv("RERANKER", "heuristic"),
    )
    pipeline = IngestionPipeline(settings)
    stats = pipeline.ingest(args.repo)
    print(f"ingested {stats['files_scanned']} files -> {stats['chunks_total']} chunks\n")

    ks = sorted(set(args.k))
    rows = []
    recall_sums = {k: 0.0 for k in ks}
    precision_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    retrieve_ms = []

    for q in queries:
        qid = q["id"]
        expected = ground_truth.get(qid, q.get("relevant", []))
        t0 = time.perf_counter()
        candidates = pipeline.retriever.retrieve(q["query"])
        ranked = pipeline.reranker.rerank(q["query"], candidates, top_k=max(ks))
        retrieve_ms.append((time.perf_counter() - t0) * 1000)

        hits = [1 if is_relevant(c, expected) else 0 for c in ranked]
        rank = first_relevant_rank(ranked, expected)
        rr_sum += 1.0 / rank if rank else 0.0
        per_k = {}
        for k in ks:
            # Recall = fraction of ground-truth items covered by top-k;
            # Precision = fraction of top-k that match any ground-truth item.
            covered = matched_expected(ranked, expected, k)
            recall = len(covered) / len(expected) if expected else 0.0
            precision = sum(hits[:k]) / k
            recall_sums[k] += recall
            precision_sums[k] += precision
            per_k[k] = {"recall": round(recall, 3), "precision": round(precision, 3)}
        rows.append(
            {
                "id": qid,
                "query": q["query"],
                "first_relevant_rank": rank or None,
                "metrics": per_k,
            }
        )
        top = f"top{max(ks)}: {ranked[0]['file_path']}:{ranked[0]['start_line']}" if ranked else "no hits"
        print(f"{qid}  rank={rank or '-'}  {top}")

    n = len(queries)
    summary = {
        "queries": n,
        "recall_at_k": {str(k): round(recall_sums[k] / n, 3) for k in ks},
        "precision_at_k": {str(k): round(precision_sums[k] / n, 3) for k in ks},
        "mrr": round(rr_sum / n, 3),
        "mean_retrieval_ms": round(sum(retrieve_ms) / n, 1),
    }
    print("\n=== Retrieval evaluation ===")
    print(f"queries: {n}")
    for k in ks:
        print(f"Recall@{k}:    {summary['recall_at_k'][str(k)]}")
        print(f"Precision@{k}: {summary['precision_at_k'][str(k)]}")
    print(f"MRR:          {summary['mrr']}")
    print(f"mean retrieval+rerank latency: {summary['mean_retrieval_ms']} ms")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_query": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
