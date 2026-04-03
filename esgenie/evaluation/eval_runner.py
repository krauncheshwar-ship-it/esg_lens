"""
eval_runner.py
Runs a set of test queries against a PDF and compares answers to ground truth.
Outputs per-question scores and an aggregate report.
"""

import csv
import json
import os
from pathlib import Path

from agents.orchestrator import ask

_GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.csv"


def load_ground_truth(csv_path: str | Path = _GROUND_TRUTH_PATH) -> list[dict]:
    """
    Load ground truth Q&A pairs from a CSV file.

    Expected columns: query, expected_answer, theme, pdf_file
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _keyword_overlap(predicted: str, expected: str) -> float:
    """Simple keyword-overlap score (0.0 – 1.0) as a proxy for correctness."""
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    return len(pred_words & exp_words) / len(exp_words)


def evaluate(
    pdf_path: str,
    csv_path: str | Path = _GROUND_TRUTH_PATH,
    top_k: int = 10,
) -> dict:
    """
    Run evaluation for a single PDF against the ground truth CSV.

    Args:
        pdf_path: Path to the ESG PDF to evaluate.
        csv_path: Path to the ground truth CSV.
        top_k:    Number of chunks retrieved per query.

    Returns:
        {
            "results": list of per-question dicts,
            "aggregate": {
                "num_questions": int,
                "avg_keyword_overlap": float,
                "avg_confidence": float,
            }
        }
    """
    gt_rows = load_ground_truth(csv_path)
    # Filter rows matching this PDF (or run all if pdf_file column is empty)
    pdf_name = Path(pdf_path).name
    rows = [
        r for r in gt_rows
        if not r.get("pdf_file") or r["pdf_file"] == pdf_name
    ]

    per_question = []
    total_overlap = 0.0
    confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    total_conf = 0.0

    for row in rows:
        query = row["query"]
        expected = row["expected_answer"]

        result = ask(pdf_path, query, top_k=top_k)
        predicted = result.get("answer", "")
        confidence = result.get("confidence", "low")
        overlap = _keyword_overlap(predicted, expected)

        total_overlap += overlap
        total_conf += confidence_map.get(confidence, 0.0)

        per_question.append(
            {
                "query": query,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "confidence": confidence,
                "keyword_overlap": round(overlap, 3),
                "theme": row.get("theme", ""),
                "facts": result.get("facts", []),
                "limitations": result.get("limitations", ""),
            }
        )

    n = len(rows) or 1
    aggregate = {
        "num_questions": len(rows),
        "avg_keyword_overlap": round(total_overlap / n, 3),
        "avg_confidence": round(total_conf / n, 3),
    }

    return {"results": per_question, "aggregate": aggregate}


def print_report(eval_output: dict) -> None:
    """Pretty-print evaluation results to stdout."""
    agg = eval_output["aggregate"]
    print(f"\n=== ESGenie Evaluation Report ===")
    print(f"Questions evaluated : {agg['num_questions']}")
    print(f"Avg keyword overlap : {agg['avg_keyword_overlap']:.1%}")
    print(f"Avg confidence score: {agg['avg_confidence']:.2f} / 1.00")
    print("\n--- Per-Question Results ---")
    for r in eval_output["results"]:
        print(f"\nQ: {r['query']}")
        print(f"  Expected : {r['expected_answer'][:120]}")
        print(f"  Predicted: {r['predicted_answer'][:120]}")
        print(f"  Overlap  : {r['keyword_overlap']:.1%}  |  Confidence: {r['confidence']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_runner.py <path_to_pdf> [path_to_csv]")
        sys.exit(1)

    pdf = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else _GROUND_TRUTH_PATH
    output = evaluate(pdf, csv_path=csv_file)
    print_report(output)
