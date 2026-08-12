"""
Milestone 8 — Evaluation

A lightweight evaluation harness for the RAG pipeline. Not a replacement
for a framework like RAGAS, but enough to catch regressions in retrieval
and answer quality as you tune chunk size, top_k, or prompts.

Usage:
    python scripts/evaluate.py --dataset eval/questions.json --report eval/report.csv

Dataset format — a JSON list of objects:
    [
      {
        "question": "...",
        "expected_source": "policy.pdf",        # optional
        "expected_keywords": ["keyword1", "keyword2"]  # optional
      }
    ]

For each question the script:
  1. Runs the full RAG pipeline (retrieval + LLM answer).
  2. Retrieval hit: does `expected_source` appear among the citations?
  3. Answer coverage: do all `expected_keywords` appear in the answer text?
  4. Writes a per-question CSV report and prints an aggregate summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import rag_pipeline  # noqa: E402


def evaluate(dataset_path: Path, report_path: Path) -> None:
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not items:
        print("Dataset is empty — nothing to evaluate.")
        return

    rows = []
    retrieval_hits = 0
    retrieval_checked = 0
    keyword_hits = 0
    keyword_checked = 0
    total_latency = 0.0

    for item in items:
        question = item["question"]
        start = time.perf_counter()
        result = rag_pipeline.ask(question)
        latency = time.perf_counter() - start
        total_latency += latency

        retrieved_sources = {c.source for c in result.citations}
        expected_source = item.get("expected_source") or ""
        retrieval_hit = ""
        if expected_source:
            retrieval_checked += 1
            retrieval_hit = expected_source in retrieved_sources
            retrieval_hits += int(retrieval_hit)

        expected_keywords = item.get("expected_keywords", [])
        answer_lower = result.answer.lower()
        found_keywords = [k for k in expected_keywords if k.lower() in answer_lower]
        keyword_hit = ""
        if expected_keywords:
            keyword_checked += 1
            keyword_hit = len(found_keywords) == len(expected_keywords)
            keyword_hits += int(keyword_hit)

        rows.append(
            {
                "question": question,
                "answer": result.answer,
                "retrieved_sources": ", ".join(sorted(retrieved_sources)),
                "expected_source": expected_source,
                "retrieval_hit": retrieval_hit,
                "expected_keywords": ", ".join(expected_keywords),
                "found_keywords": ", ".join(found_keywords),
                "keyword_hit": keyword_hit,
                "latency_sec": round(latency, 2),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(items)
    print(f"Evaluated {n} question(s)")
    print(f"Average latency: {total_latency / n:.2f}s")
    if retrieval_checked:
        print(f"Retrieval hit rate: {retrieval_hits}/{retrieval_checked}")
    if keyword_checked:
        print(f"Keyword coverage rate: {keyword_hits}/{keyword_checked}")
    print(f"Full report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a Q&A dataset")
    parser.add_argument("--dataset", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--report", type=Path, default=Path("eval/report.csv"))
    args = parser.parse_args()
    evaluate(args.dataset, args.report)
