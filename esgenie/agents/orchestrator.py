"""
orchestrator.py
Central controller that wires together ingestion, processing, retrieval,
and extraction agents.  Exposes two public modes:

  Mode 1 — ad-hoc Q&A:    orchestrator.ask(pdf_path, query)
  Mode 2 — full ESG sweep: orchestrator.sweep(pdf_path)
"""

import os
from pathlib import Path

from ingestion.pdf_parser import extract_pages
from processing.thematic_bucketer import score_pages
from processing.chunker import chunk_pages
from retrieval.embedder import build_faiss_index, load_faiss_index
from retrieval.hybrid_retriever import HybridRetriever
from agents.query_router import route_query
from agents.extraction_agent import extract, batch_extract

# Default sweep questions covering all three ESG pillars
_DEFAULT_SWEEP_QUERIES = [
    # Environment
    "What are the company's total Scope 1, Scope 2, and Scope 3 greenhouse gas emissions?",
    "What renewable energy targets or commitments has the company made?",
    "How does the company manage water consumption and withdrawal?",
    "What waste reduction or recycling programs does the company operate?",
    # Social
    "What is the gender diversity breakdown in the company's workforce and leadership?",
    "What is the company's total recordable injury rate (TRIR) or safety performance?",
    "What employee training and development programs does the company offer?",
    "How does the company address human rights in its supply chain?",
    # Governance
    "What is the composition and independence of the board of directors?",
    "How is executive compensation structured and linked to ESG performance?",
    "What anti-corruption policies and whistleblower mechanisms are in place?",
    "What cybersecurity governance frameworks does the company follow?",
]


def _get_index_name(pdf_path: str) -> str:
    return Path(pdf_path).stem.replace(" ", "_")


def _build_retriever(pdf_path: str) -> HybridRetriever:
    """
    Ingest and index a PDF, using the FAISS cache when available.

    Returns a ready-to-query HybridRetriever.
    """
    index_name = _get_index_name(pdf_path)
    cached = load_faiss_index(index_name)

    if cached is not None:
        faiss_index, chunks = cached
    else:
        pages = extract_pages(pdf_path)
        pages = score_pages(pages)
        chunks = chunk_pages(pages)
        faiss_index, chunks = build_faiss_index(chunks, index_name)

    return HybridRetriever(faiss_index, chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(pdf_path: str, query: str, top_k: int = 10) -> dict:
    """
    Mode 1 — Answer a single ad-hoc question about an ESG report.

    Args:
        pdf_path: Path to the ESG PDF.
        query:    User question.
        top_k:    Number of context chunks to retrieve.

    Returns:
        Extraction result dict (see extraction_agent.extract).
        Also includes "route" key with theme classification.
    """
    retriever = _build_retriever(pdf_path)
    route = route_query(query)
    chunks = retriever.retrieve(query, top_k=top_k)
    result = extract(query, chunks)
    result["route"] = route
    return result


def sweep(
    pdf_path: str,
    queries: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Mode 2 — Run a predefined set of ESG questions against a report.

    Args:
        pdf_path: Path to the ESG PDF.
        queries:  Custom question list (defaults to _DEFAULT_SWEEP_QUERIES).
        top_k:    Chunks per query.

    Returns:
        List of extraction result dicts, one per query.
    """
    if queries is None:
        queries = _DEFAULT_SWEEP_QUERIES

    retriever = _build_retriever(pdf_path)
    results = batch_extract(queries, retriever, top_k=top_k)
    return results
