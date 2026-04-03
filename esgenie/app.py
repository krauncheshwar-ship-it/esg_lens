"""
app.py
ESGenie — Streamlit UI
Mode 1: Ad-hoc Q&A against a single ESG PDF.
Mode 2: Full ESG sweep with Excel export.
"""

import os
import sys
from pathlib import Path

# Make sure all sub-packages are importable when running from the esgenie/ dir
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import streamlit as st

from agents.orchestrator import ask, sweep
from export.excel_exporter import export_to_excel
from utils.token_tracker import get_session_summary, reset_session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ESGenie",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — PDF selection
# ---------------------------------------------------------------------------
REPORTS_DIR = Path(__file__).parent / "sample_reports"
REPORTS_DIR.mkdir(exist_ok=True)

st.sidebar.title("🌿 ESGenie")
st.sidebar.caption("AI-powered ESG report analyst")
st.sidebar.divider()

pdf_files = sorted(REPORTS_DIR.glob("*.pdf"))
pdf_names = [p.name for p in pdf_files]

if not pdf_names:
    st.sidebar.warning("No PDFs found in `sample_reports/`. Add a PDF to get started.")
    st.stop()

selected_name = st.sidebar.selectbox("Select ESG Report", pdf_names)
selected_pdf = str(REPORTS_DIR / selected_name)

st.sidebar.divider()
mode = st.sidebar.radio(
    "Mode",
    ["Mode 1 — Ad-hoc Q&A", "Mode 2 — Full ESG Sweep"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Session Token Usage")
summary = get_session_summary()
st.sidebar.metric("Total Calls", summary["call_count"])
st.sidebar.metric("Total Cost (USD)", f"${summary['total_cost_usd']:.4f}")
st.sidebar.metric("Prompt Tokens", f"{summary['total_prompt_tokens']:,}")
st.sidebar.metric("Completion Tokens", f"{summary['total_completion_tokens']:,}")
st.sidebar.metric("Embedding Tokens", f"{summary['total_embedding_tokens']:,}")

if st.sidebar.button("Reset Session Counters"):
    reset_session()
    st.rerun()

# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.title("🌿 ESGenie")
st.caption(f"Analysing: **{selected_name}**")
st.divider()

# ---------------------------------------------------------------------------
# Mode 1 — Ad-hoc Q&A
# ---------------------------------------------------------------------------
if mode == "Mode 1 — Ad-hoc Q&A":
    st.subheader("Ask a question about this ESG report")

    example_queries = [
        "What are the company's total Scope 1 and Scope 2 emissions?",
        "What renewable energy commitments has the company made?",
        "What is the gender diversity breakdown in the workforce?",
        "What is the company's total recordable injury rate (TRIR)?",
        "How is executive compensation linked to ESG performance?",
    ]

    with st.expander("Example questions"):
        for q in example_queries:
            st.markdown(f"- {q}")

    query = st.text_area(
        "Your question",
        placeholder="e.g. What are the company's Scope 1 greenhouse gas emissions?",
        height=80,
    )

    col1, col2 = st.columns([1, 5])
    run_btn = col1.button("Ask", type="primary", use_container_width=True)

    if run_btn and query.strip():
        with st.spinner("Retrieving and extracting..."):
            try:
                result = ask(selected_pdf, query.strip())
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # Theme route
        route = result.get("route", {})
        themes = ", ".join(route.get("themes", []))
        rationale = route.get("rationale", "")
        st.info(f"**Theme(s) detected:** {themes}  |  {rationale}")

        # Answer
        confidence = result.get("confidence", "unknown")
        conf_colour = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        st.markdown(f"### Answer  {conf_colour} _{confidence} confidence_")
        st.markdown(result.get("answer", "_No answer returned._"))

        limitations = result.get("limitations", "")
        if limitations:
            st.caption(f"**Limitations:** {limitations}")

        # Facts & citations
        facts = result.get("facts", [])
        if facts:
            st.markdown("### Source Citations")
            for i, fact in enumerate(facts, start=1):
                with st.expander(f"[{i}] {fact.get('source','?')}  —  page {fact.get('page','?')}"):
                    st.markdown(f"**Statement:** {fact.get('statement','')}")
                    st.markdown(f"**Quote:** _{fact.get('quote','')}_")

        # Token usage delta
        new_summary = get_session_summary()
        st.caption(
            f"Session cost so far: **${new_summary['total_cost_usd']:.4f}** | "
            f"{new_summary['total_prompt_tokens']:,} prompt tokens | "
            f"{new_summary['total_completion_tokens']:,} completion tokens"
        )

    elif run_btn:
        st.warning("Please enter a question.")

# ---------------------------------------------------------------------------
# Mode 2 — Full ESG Sweep
# ---------------------------------------------------------------------------
elif mode == "Mode 2 — Full ESG Sweep":
    st.subheader("Full ESG Sweep")
    st.markdown(
        "Runs a predefined set of **12 ESG questions** across Environment, Social, "
        "and Governance pillars and exports results to a colour-coded Excel workbook."
    )

    sweep_queries = st.text_area(
        "Custom questions (one per line, leave blank to use defaults)",
        height=150,
        placeholder="Leave blank to use the 12 default ESG questions...",
    )

    run_sweep = st.button("Run Full ESG Sweep", type="primary")

    if run_sweep:
        custom_queries = (
            [q.strip() for q in sweep_queries.strip().splitlines() if q.strip()]
            if sweep_queries.strip()
            else None
        )

        progress = st.progress(0, text="Starting sweep...")
        status_box = st.empty()

        with st.spinner("Running ESG sweep — this may take a few minutes..."):
            try:
                results = sweep(selected_pdf, queries=custom_queries)
            except Exception as e:
                st.error(f"Sweep failed: {e}")
                st.stop()

        progress.progress(100, text="Sweep complete.")
        st.success(f"Extracted answers to {len(results)} questions.")

        # Results table
        st.divider()
        st.subheader("Results")
        for result in results:
            confidence = result.get("confidence", "unknown")
            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
            with st.expander(f"{conf_icon} {result.get('query','')}"):
                st.markdown(f"**Answer:** {result.get('answer','')}")
                limitations = result.get("limitations", "")
                if limitations:
                    st.caption(f"**Limitations:** {limitations}")
                facts = result.get("facts", [])
                if facts:
                    st.markdown("**Citations:**")
                    for fact in facts:
                        st.markdown(
                            f"- _{fact.get('source','?')}_, p.{fact.get('page','?')}: "
                            f"{fact.get('statement','')}"
                        )

        # Excel download
        st.divider()
        st.subheader("Download Results")
        excel_bytes = export_to_excel(results, report_name=selected_name)
        st.download_button(
            label="Download Excel Workbook",
            data=excel_bytes,
            file_name=f"esgenie_{Path(selected_name).stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        # Token usage
        new_summary = get_session_summary()
        st.caption(
            f"Session cost so far: **${new_summary['total_cost_usd']:.4f}** | "
            f"{new_summary['total_prompt_tokens']:,} prompt tokens | "
            f"{new_summary['total_completion_tokens']:,} completion tokens"
        )
