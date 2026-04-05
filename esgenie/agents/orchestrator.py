"""
orchestrator.py
Mode 1 single-query coordinator. Wires router -> retrieval -> extraction.
Mode 2 (full profile) is handled by profile_graph.py (LangGraph).
"""

import time
from datetime import datetime

from agents.query_router import route_query
from retrieval.hybrid_retriever import hybrid_search
from agents.extraction_agent import extract
from utils.token_tracker import get_session_summary

# Logger imports are optional — modules created later in build sequence
try:
    from logging_system.system_logger import log_event as _log_event
except ImportError:
    _log_event = None

try:
    from logging_system.ai_metrics_logger import log_metrics as _log_metrics
except ImportError:
    _log_metrics = None


def run_single_query(
    query: str,
    faiss_index,
    chunks: list[dict],
    bm25_index,
    company: str,
    year: int,
    kpi_id: str = "",
    value_type: str = "numeric",
    unit_expected: str = "",
    bm25_disabled: bool = False,
) -> dict:
    """
    Execute a single ESG extraction query end-to-end (Mode 1).

    Steps: route -> hybrid_search -> extract -> log

    Args:
        query:         User question.
        faiss_index:   Built FAISS index.
        chunks:        Chunk metadata aligned with index.
        bm25_index:    BM25Okapi index.
        company:       Company name for prompt context.
        year:          Reporting year.
        kpi_id:        KPI identifier (e.g. "C001").
        value_type:    "numeric" | "boolean" | "subjective".
        unit_expected: Expected unit hint.
        bm25_disabled: Pass-through to hybrid_search for demo comparison.

    Returns:
        Combined result dict with extraction, retrieval metadata, and timings.
    """
    run_id = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    t_start = time.time()

    # 1. Log start
    if _log_event:
        _log_event(run_id=run_id, event="query_start", module="agents/orchestrator.py",
                   company=company, status="running", error=None,
                   extra={"query": query, "kpi_id": kpi_id})

    # 2. Route query → theme
    route = route_query(query)
    theme = route["theme"]

    # 3. Hybrid retrieval — theme-filtered, top 5 for LLM context
    t_retrieval = time.time()
    retrieval_results = hybrid_search(
        query=query,
        faiss_index=faiss_index,
        chunks=chunks,
        bm25_index=bm25_index,
        k=5,
        theme_filter=theme,
        bm25_disabled=bm25_disabled,
    )
    retrieval_ms = (time.time() - t_retrieval) * 1000

    top_chunks = [r["chunk"] for r in retrieval_results]
    top_rrf_score = retrieval_results[0]["rrf_score"] if retrieval_results else 0.0
    top_page = top_chunks[0]["page_number"] if top_chunks else None

    # 4. LLM extraction — ONE call on pre-filtered context
    t_extraction = time.time()
    extraction = extract(
        query=query,
        chunks=top_chunks,
        kpi_id=kpi_id,
        value_type=value_type,
        unit_expected=unit_expected,
        company=company,
        year=year,
    )
    extraction_ms = (time.time() - t_extraction) * 1000

    # 5. Hallucination flag already set in extraction_agent; surface here too
    hallucination_flag = extraction.get("hallucination_flag", 0)

    # 6. Log AI metrics
    if _log_metrics:
        _log_metrics(
            run_id=run_id,
            query=query,
            theme_classified=theme,
            chunks_retrieved=len(top_chunks),
            top_chunk_page=top_page,
            top_chunk_score_rrf=top_rrf_score,
            extraction_confidence=extraction.get("confidence", "not_found"),
            value_found=extraction.get("value") is not None,
            hallucination_flag=bool(hallucination_flag),
            latency_retrieval_ms=round(retrieval_ms),
            latency_extraction_ms=round(extraction_ms),
        )

    total_ms = (time.time() - t_start) * 1000
    session = get_session_summary()

    # 7. Log completion
    if _log_event:
        _log_event(run_id=run_id, event="query_complete", module="agents/orchestrator.py",
                   company=company, status="success", error=None,
                   extra={"latency_ms": round(total_ms), "cost_usd": session["total_cost_usd"]})

    return {
        **extraction,
        "run_id": run_id,
        "theme": theme,
        "route_confidence": route["confidence"],
        "matched_keywords": route["matched_keywords"],
        "retrieval_results": retrieval_results,
        "top_chunk_page": top_page,
        "top_rrf_score": top_rrf_score,
        "hallucination_flag": hallucination_flag,
        "latency_retrieval_ms": round(retrieval_ms),
        "latency_extraction_ms": round(extraction_ms),
        "total_latency_ms": round(total_ms),
        "total_cost_usd": session["total_cost_usd"],
    }
