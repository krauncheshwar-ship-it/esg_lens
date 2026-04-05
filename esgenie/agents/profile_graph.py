"""
profile_graph.py
LangGraph Mode 2 orchestration — full ESG profile generation.
LLM called EXACTLY ONCE (extract_all node). All other nodes are pure Python.

Architecture (video line 3):
  validate -> load_template -> run_ingestion -> run_bucketing -> run_chunking
  -> build_indexes -> [retrieve_climate | retrieve_social | retrieve_governance
                       | retrieve_water | retrieve_biodiversity]
  -> extract_all -> score_esg -> persist_database -> export_results -> log_completion
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional

import operator
import pandas as pd
from typing import Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from ingestion.pdf_parser import extract_pages
from processing.thematic_bucketer import ThematicBucketer
from processing.chunker import chunk_bucketed_pages
from retrieval.embedder import build_index, load_index
from retrieval.bm25_index import build_bm25, load_bm25
from retrieval.hybrid_retriever import hybrid_search
from agents.extraction_agent import extract
from scoring.esg_scorer import score_company
from database.db_manager import (
    init_db, get_or_create_company, save_extraction,
    save_extraction_run,
)
from export.excel_exporter import export_to_excel
from utils.token_tracker import get_session_summary, reset_session
from logging_system.system_logger import log_event
from logging_system.ai_metrics_logger import log_retrieval

_KPI_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "kpi_template_v8_FINAL.csv"
_AUDIT_DB = Path(__file__).parent.parent / "database" / "langgraph_audit.db"
_RETRIEVAL_TOP_K = 5


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ESGState(TypedDict):
    company_name: str
    reporting_year: int
    pdf_path: str
    run_id: str
    cache_prefix: str           # key for disk-cached FAISS + BM25 — serializable
    page_corpus: dict
    thematic_page_map: dict
    token_reduction_log: dict
    chunks: list
    kpi_by_theme: dict          # {theme: [{kpi_id, canonical, value_type, unit}]}
    retrieved_chunks: Annotated[dict, operator.or_]  # parallel writes merged by |
    extractions: list

    scores: dict
    total_cost: float
    errors: list
    status: str


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _safe(state: dict, fn, node_name: str) -> dict:
    """Wrap a node function; append to errors and mark status on exception."""
    try:
        return fn(state)
    except Exception as exc:
        errs = list(state.get("errors", []))
        errs.append(f"{node_name}: {exc}")
        return {"errors": errs, "status": "error"}


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def validate_input(state: ESGState) -> dict:
    errors = []
    for field in ("company_name", "pdf_path", "reporting_year"):
        if not state.get(field):
            errors.append(f"Missing required field: {field}")
    if not Path(state.get("pdf_path", "")).exists():
        errors.append(f"PDF not found: {state.get('pdf_path')}")
    if errors:
        return {"errors": errors, "status": "error"}

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reset_session()
    log_event("pipeline_start", "agents/profile_graph.py",
              run_id=run_id, company=state["company_name"], status="running")
    return {"run_id": run_id, "errors": [], "status": "running"}


def load_template(state: ESGState) -> dict:
    def _fn(state):
        df = pd.read_csv(_KPI_TEMPLATE_PATH)
        kpi_by_theme: dict[str, list[dict]] = {}
        for _, row in df.iterrows():
            theme = row["theme"]
            kpi_by_theme.setdefault(theme, []).append({
                "kpi_id":     row["kpi_id"],
                "canonical":  row["kpi_canonical_name"],
                "value_type": row["value_type"],
                "unit":       row.get("unit", ""),
            })
        return {"kpi_by_theme": kpi_by_theme}
    return _safe(state, _fn, "load_template")


def run_ingestion(state: ESGState) -> dict:
    def _fn(state):
        pages = extract_pages(state["pdf_path"])
        log_event("pdf_ingestion_complete", "ingestion/pdf_parser.py",
                  run_id=state["run_id"], company=state["company_name"],
                  status="success", extra={"total_pages": len(pages)})
        return {"page_corpus": pages}
    return _safe(state, _fn, "run_ingestion")


def run_bucketing(state: ESGState) -> dict:
    def _fn(state):
        bucketer = ThematicBucketer()
        page_map, reduction_log = bucketer.bucket(state["page_corpus"])
        log_event("bucketing_complete", "processing/thematic_bucketer.py",
                  run_id=state["run_id"], company=state["company_name"],
                  status="success",
                  extra={"reduction_pct": reduction_log["reduction_pct"]})
        return {"thematic_page_map": page_map, "token_reduction_log": reduction_log}
    return _safe(state, _fn, "run_bucketing")


def run_chunking(state: ESGState) -> dict:
    def _fn(state):
        chunks = chunk_bucketed_pages(
            state["page_corpus"],
            state["thematic_page_map"],
            company=state["company_name"],
            report_year=state["reporting_year"],
        )
        return {"chunks": chunks}
    return _safe(state, _fn, "run_chunking")


def build_indexes(state: ESGState) -> dict:
    def _fn(state):
        prefix = f"{state['company_name'].lower()}_{state['reporting_year']}"
        chunks = state["chunks"]
        build_index(chunks, cache_prefix=prefix)
        build_bm25(chunks, cache_prefix=prefix)
        # Store only the string prefix — FAISS/BM25 objects are not msgpack-serializable
        return {"cache_prefix": prefix}
    return _safe(state, _fn, "build_indexes")


def _retrieve_theme(state: ESGState, theme: str) -> dict:
    # Load indexes from disk — they are not stored in state (not serializable)
    prefix = state["cache_prefix"]
    faiss_idx, stored_chunks = load_index(prefix)
    bm25_idx, _ = load_bm25(prefix)

    results = hybrid_search(
        query=f"{theme} ESG metrics data",
        faiss_index=faiss_idx,
        chunks=stored_chunks,
        bm25_index=bm25_idx,
        k=_RETRIEVAL_TOP_K,
        theme_filter=theme,
    )
    top_chunks = [r["chunk"] for r in results]
    return {"retrieved_chunks": {theme: top_chunks}}


def retrieve_climate(state: ESGState) -> dict:
    return _safe(state, lambda s: _retrieve_theme(s, "climate"), "retrieve_climate")

def retrieve_social(state: ESGState) -> dict:
    return _safe(state, lambda s: _retrieve_theme(s, "employee"), "retrieve_social")

def retrieve_governance(state: ESGState) -> dict:
    return _safe(state, lambda s: _retrieve_theme(s, "governance"), "retrieve_governance")

def retrieve_water(state: ESGState) -> dict:
    return _safe(state, lambda s: _retrieve_theme(s, "water"), "retrieve_water")

def retrieve_biodiversity(state: ESGState) -> dict:
    return _safe(state, lambda s: _retrieve_theme(s, "biodiversity"), "retrieve_biodiversity")


def extract_all(state: ESGState) -> dict:
    """
    THE SINGLE LLM CALL PHASE.
    Loop through every KPI in the template; use theme-specific retrieved chunks.
    """
    def _fn(state):
        extractions: list[dict] = []
        retrieved = state.get("retrieved_chunks", {})
        kpi_by_theme = state.get("kpi_by_theme", {})

        for theme, kpis in kpi_by_theme.items():
            theme_chunks = retrieved.get(theme, [])
            if not theme_chunks:
                # Fall back to any available chunks for this theme
                theme_chunks = [c for c in state["chunks"] if c.get("theme") == theme][:5]

            for kpi in kpis:
                query = f"What is the {kpi['canonical']}?"
                result = extract(
                    query=query,
                    chunks=theme_chunks,
                    kpi_id=kpi["kpi_id"],
                    value_type=kpi["value_type"],
                    unit_expected=kpi["unit"],
                    company=state["company_name"],
                    year=state["reporting_year"],
                )
                result["kpi_theme"] = theme
                result["kpi_canonical_name"] = kpi["canonical"]
                result["reporting_year"] = state["reporting_year"]
                extractions.append(result)

        summary = get_session_summary()
        log_event("extraction_complete", "agents/profile_graph.py",
                  run_id=state["run_id"], company=state["company_name"],
                  status="success",
                  extra={"kpi_count": len(extractions),
                         "total_cost_usd": summary["total_cost_usd"]})
        return {"extractions": extractions, "total_cost": summary["total_cost_usd"]}
    return _safe(state, _fn, "extract_all")


def score_esg(state: ESGState) -> dict:
    def _fn(state):
        scores = score_company(
            state["extractions"],
            state["company_name"],
        )
        return {"scores": scores}
    return _safe(state, _fn, "score_esg")


def persist_database(state: ESGState) -> dict:
    def _fn(state):
        conn = init_db()
        company_id = get_or_create_company(conn, state["company_name"])
        reduction_log = state.get("token_reduction_log", {})

        save_extraction_run(
            conn,
            run_id=state["run_id"],
            company_name=state["company_name"],
            pdf_filename=Path(state["pdf_path"]).name,
            total_pages=reduction_log.get("total_pages", 0),
            pages_after_bucketing=reduction_log.get("estimated_tokens_after_bucketing", 0),
            token_reduction_pct=reduction_log.get("reduction_pct", 0.0),
            total_tokens=get_session_summary()["total_prompt_tokens"],
            total_cost=state.get("total_cost", 0.0),
            duration_seconds=0.0,
            status=state.get("status", "success"),
        )

        for ext in state.get("extractions", []):
            save_extraction(conn, company_id, state["run_id"], ext)

        conn.close()
        return {}
    return _safe(state, _fn, "persist_database")


def export_results(state: ESGState) -> dict:
    # Excel bytes are generated outside the graph in run_profile() to avoid
    # msgpack serialization errors with raw bytes in the checkpointer.
    return {}


def log_completion(state: ESGState) -> dict:
    def _fn(state):
        summary = get_session_summary()
        log_event("pipeline_complete", "agents/profile_graph.py",
                  run_id=state["run_id"], company=state["company_name"],
                  status=state.get("status", "success"),
                  extra={
                      "total_cost_usd": summary["total_cost_usd"],
                      "kpi_count": len(state.get("extractions", [])),
                      "overall_score": state.get("scores", {}).get("overall_score"),
                  })
        return {"status": state.get("status", "success")}
    return _safe(state, _fn, "log_completion")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(ESGState)

    # Register all nodes
    for name, fn in [
        ("validate_input",     validate_input),
        ("load_template",      load_template),
        ("run_ingestion",      run_ingestion),
        ("run_bucketing",      run_bucketing),
        ("run_chunking",       run_chunking),
        ("build_indexes",      build_indexes),
        ("retrieve_climate",   retrieve_climate),
        ("retrieve_social",    retrieve_social),
        ("retrieve_governance",retrieve_governance),
        ("retrieve_water",     retrieve_water),
        ("retrieve_biodiversity", retrieve_biodiversity),
        ("extract_all",        extract_all),
        ("score_esg",          score_esg),
        ("persist_database",   persist_database),
        ("export_results",     export_results),
        ("log_completion",     log_completion),
    ]:
        g.add_node(name, fn)

    # Linear pipeline up to index build
    g.set_entry_point("validate_input")
    g.add_edge("validate_input",  "load_template")
    g.add_edge("load_template",   "run_ingestion")
    g.add_edge("run_ingestion",   "run_bucketing")
    g.add_edge("run_bucketing",   "run_chunking")
    g.add_edge("run_chunking",    "build_indexes")

    # Fan-out: parallel retrieval
    for retriever in ["retrieve_climate", "retrieve_social",
                      "retrieve_governance", "retrieve_water",
                      "retrieve_biodiversity"]:
        g.add_edge("build_indexes", retriever)
        g.add_edge(retriever, "extract_all")

    # Fan-in continues linearly
    g.add_edge("extract_all",      "score_esg")
    g.add_edge("score_esg",        "persist_database")
    g.add_edge("persist_database", "export_results")
    g.add_edge("export_results",   "log_completion")
    g.add_edge("log_completion",   END)

    return g


# Build once at module load
_AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
_checkpointer_conn = sqlite3.connect(str(_AUDIT_DB), check_same_thread=False)
_checkpointer = SqliteSaver(_checkpointer_conn)
_graph = _build_graph()
app = _graph.compile(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_profile(initial_state: dict) -> dict:
    """
    Run a full ESG profile generation (Mode 2).

    Args:
        initial_state: Must contain company_name, reporting_year, pdf_path.
                       All other fields are populated by the graph.

    Returns:
        Final ESGState dict with extractions, scores, excel_bytes, etc.
    """
    defaults = {
        "run_id": "",
        "cache_prefix": "",
        "page_corpus": {},
        "thematic_page_map": {},
        "token_reduction_log": {},
        "chunks": [],
        "kpi_by_theme": {},
        "retrieved_chunks": {},
        "extractions": [],
        "scores": {},
        "total_cost": 0.0,
        "errors": [],
        "status": "pending",
    }
    state = {**defaults, **initial_state}
    thread = {"configurable": {"thread_id": state.get("company_name", "default")}}

    final = None
    for step in app.stream(state, config=thread):
        node_name = list(step.keys())[0]
        node_out  = step[node_name]
        if node_out is not None:
            print(f"  [graph] {node_name} done | errors={node_out.get('errors', [])}")
            final = node_out

    # Reconstruct full final state from checkpointer
    snapshot = app.get_state(thread)
    result = dict(snapshot.values) if snapshot else (final or state)

    # Generate Excel outside the graph — bytes are not msgpack-serializable
    try:
        result["excel_bytes"] = export_to_excel(
            result.get("extractions", []),
            company=result.get("company_name", ""),
            year=result.get("reporting_year", 2024),
            scores=result.get("scores"),
        )
    except Exception as exc:
        result["excel_bytes"] = b""
        result.setdefault("errors", []).append(f"excel_export: {exc}")

    return result
