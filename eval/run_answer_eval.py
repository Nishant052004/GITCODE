"""Answer evaluation: citation accuracy, latency percentiles, human-label log.

Run from the repo root:
    python eval/run_answer_eval.py --repo examples/demo_repo

For every query the script runs the full query pipeline, checks that each
citation points at a relevant file (machine-checkable part of citation
accuracy), aggregates latency percentiles per stage, and writes an
answer_log.jsonl with blank `correctness` / `citation_supported` fields for
human grading.

Results go to eval/results/answer_eval.json and answer_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_retrieval_eval import is_relevant

from backend.app.config import Settings
from backend.app.services.pipeline import IngestionPipeline


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(REPO_ROOT / "examples" / "demo_repo"))
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.json"))
    parser.add_argument("--ground-truth", default=str(REPO_ROOT / "eval" / "ground_truth.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "eval" / "results"))
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
    pipeline.ingest(args.repo)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "answer_log.jsonl"

    citation_valid_counts = []
    all_citation_counts = []
    grounded_flags = []
    latencies: dict[str, list[float]] = {
        "retrieve": [],
        "rerank": [],
        "answer": [],
        "total": [],
    }
    rows = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        for q in queries:
            expected = ground_truth.get(q["id"], q.get("relevant", []))
            result = pipeline.query(q["query"], top_k=args.top_k)
            citations = result["citations"]
            candidate_by_id = {c["chunk_id"]: c for c in result["candidates"]}

            valid = 0
            for cit in citations:
                cand = candidate_by_id.get(cit["chunk_id"])
                if cand and is_relevant(cand, expected):
                    valid += 1
            if citations:
                citation_valid_counts.append(valid / len(citations))
                all_citation_counts.append(len(citations))
                grounded_flags.append(valid == len(citations))

            timings = result["timings"]
            latencies["retrieve"].append(timings["retrieve_ms"])
            latencies["rerank"].append(timings["rerank_ms"])
            latencies["answer"].append(timings["answer_ms"])
            latencies["total"].append(timings["total_ms"])
            rows += 1

            log_file.write(
                json.dumps(
                    {
                        "query_id": q["id"],
                        "query": q["query"],
                        "answer": result["answer"],
                        "citations": citations,
                        "answerer": result["answerer"],
                        "timings": timings,
                        "human_grades": {
                            "correctness_1to5": None,
                            "citation_supported": None,
                        },
                    }
                )
                + "\n"
            )

    summary = {
        "queries": rows,
        "citation_accuracy": round(statistics.mean(citation_valid_counts), 3)
        if citation_valid_counts
        else None,
        "answers_fully_grounded": round(statistics.mean(grounded_flags), 3) if grounded_flags else 0.0,
        "mean_citations_per_answer": round(statistics.mean(all_citation_counts), 2)
        if all_citation_counts
        else 0.0,
        "latency_ms": {
            stage: {
                "mean": round(statistics.mean(vals), 1),
                "p50": round(percentile(vals, 50), 1),
                "p95": round(percentile(vals, 95), 1),
            }
            for stage, vals in latencies.items()
        },
    }
    print("=== Answer evaluation ===")
    print(f"queries: {rows}")
    print(f"citation accuracy (machine-checkable): {summary['citation_accuracy']}")
    print(f"answers fully grounded: {summary['answers_fully_grounded']}")
    for stage, agg in summary["latency_ms"].items():
        print(f"latency {stage:>8}: mean {agg['mean']} ms | p50 {agg['p50']} ms | p95 {agg['p95']} ms")
    print(f"\nhuman-grading log: {log_path} (fill correctness_1to5 / citation_supported)")

    (out_dir / "answer_eval.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'answer_eval.json'}")


if __name__ == "__main__":
    main()
