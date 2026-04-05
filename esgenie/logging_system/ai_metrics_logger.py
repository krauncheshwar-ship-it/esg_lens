"""
ai_metrics_logger.py
Writes retrieval and extraction quality metrics to logging_system/logs/ai_metrics.jsonl.
Precision@5 and MRR stubs here; real computation lives in evaluation/eval_runner.py.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

_LOGS_DIR = Path(__file__).parent / "logs"


def log_retrieval(
    run_id: str,
    query: str,
    theme: str,
    chunks_retrieved: int,
    top_chunk_page: int,
    rrf_score: float,
    semantic_rank: int,
    bm25_rank: int,
    value_found: bool,
    hallucination_flag: bool,
    confidence: str,
    latency_retrieval_ms: int,
    latency_extraction_ms: int,
) -> None:
    """
    Append one JSON entry to ai_metrics.jsonl.

    Args:
        run_id:                 Run identifier for correlation.
        query:                  The retrieval query string.
        theme:                  Classified ESG theme.
        chunks_retrieved:       Number of chunks returned by hybrid search.
        top_chunk_page:         Page number of the top-ranked chunk.
        rrf_score:              RRF score of the top chunk.
        semantic_rank:          Rank of top chunk in semantic results (1-based).
        bm25_rank:              Rank of top chunk in BM25 results (1-based).
        value_found:            Whether extraction returned a non-null value.
        hallucination_flag:     True if source_page != top chunk page.
        confidence:             Extraction confidence label.
        latency_retrieval_ms:   Retrieval wall-clock time in ms.
        latency_extraction_ms:  LLM extraction wall-clock time in ms.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Precision@5 stub — real computation requires ground truth in eval_runner.py
    precision_at_5 = 1.0

    # MRR stub — 1/rank if top chunk is in top 5, else 0
    best_rank = min(
        r for r in (semantic_rank, bm25_rank) if r is not None and r > 0
    ) if any(r is not None and r > 0 for r in (semantic_rank, bm25_rank)) else 0
    mrr = round(1.0 / best_rank, 4) if 0 < best_rank <= 5 else 0.0

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "query": query,
        "theme_classified": theme,
        "chunks_retrieved": chunks_retrieved,
        "top_chunk_page": top_chunk_page,
        "top_chunk_score_rrf": round(rrf_score, 4) if rrf_score else 0.0,
        "semantic_rank": semantic_rank,
        "bm25_rank": bm25_rank,
        "precision_at_5": precision_at_5,
        "mrr": mrr,
        "extraction_confidence": confidence,
        "value_found": value_found,
        "hallucination_flag": hallucination_flag,
        "latency_retrieval_ms": latency_retrieval_ms,
        "latency_extraction_ms": latency_extraction_ms,
    }

    with open(_LOGS_DIR / "ai_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_heuristic_eval(
    run_id: str,
    company: str,
    year: int,
    heuristic_result: dict,
) -> None:
    """
    Append a Track 2 heuristic eval summary entry to ai_metrics.jsonl.
    Stored with event="heuristic_eval" so Watchtower can load historical data.

    Args:
        run_id:            Run identifier for correlation.
        company:           Company name.
        year:              Reporting year.
        heuristic_result:  Output of run_eval_heuristic().
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cov = heuristic_result.get("coverage", {})
    tok = heuristic_result.get("token_efficiency", {})
    src = heuristic_result.get("source_type_dist", {})
    cfd = heuristic_result.get("confidence_dist", {})

    entry = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "event":                "heuristic_eval",
        "run_id":               run_id,
        "company":              company,
        "year":                 year,
        # Coverage
        "coverage_pct":         cov.get("overall_pct", 0),
        "kpis_found":           cov.get("found", 0),
        "kpis_attempted":       cov.get("attempted", 0),
        # Confidence
        "conf_high":            cfd.get("high", 0),
        "conf_medium":          cfd.get("medium", 0),
        "conf_low":             cfd.get("low", 0),
        "conf_not_found":       cfd.get("not_found", 0),
        # Hallucination
        "hallucination_rate":   heuristic_result.get("hallucination_rate", 0),
        "hallucination_count":  heuristic_result.get("hallucination_count", 0),
        # BM25 agreement
        "bm25_agreement":       heuristic_result.get("bm25_agreement", 0),
        # Token efficiency
        "pages_before":         tok.get("pages_before", 0),
        "pages_after":          tok.get("pages_after", 0),
        "token_reduction_pct":  tok.get("token_reduction_pct", 0),
        "total_tokens":         tok.get("total_tokens", 0),
        "total_cost":           tok.get("total_cost", 0),
        "cost_per_kpi":         tok.get("cost_per_kpi", 0),
        # Retrieval quality
        "hint_match_rate":      heuristic_result.get("hint_match_rate", 0),
        # Source type
        "table_pct":            src.get("table_pct", 0),
        "prose_pct":            src.get("prose_pct", 0),
    }

    with open(_LOGS_DIR / "ai_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
