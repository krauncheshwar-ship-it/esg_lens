"""
app.py
ESGenie — Streamlit UI. Three tabs: Analyst Workbench / Configuration / Watchtower.
"""


import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")

from ingestion.pdf_parser import extract_pages
from processing.thematic_bucketer import ThematicBucketer
from processing.chunker import chunk_bucketed_pages
from retrieval.embedder import build_index, load_index
from retrieval.bm25_index import build_bm25, load_bm25
from agents.orchestrator import run_single_query
from agents.profile_graph import run_profile
from agents.extraction_agent import SYSTEM_PROMPT
from database.db_manager import init_db, get_all_runs
from export.excel_exporter import export_to_excel
from scoring.esg_scorer import score_company, CONTROVERSY_DEDUCTIONS
from evaluation.eval_runner import run_eval_heuristic, build_eval_csv, has_ground_truth
from logging_system.ai_metrics_logger import log_heuristic_eval
from utils.token_tracker import get_session_summary, reset_session


st.markdown("""
<style>
/* Reduce metric value font */
[data-testid="stMetricValue"] {
    font-size: 18px !important; 
}
/* Reduce metric label font */
[data-testid="stMetricLabel"] {
    font-size: 13px !important; 
}
/* Optional: Reduce spacing */
[data-testid="stMetric"] {
    padding: 8px 0px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Auto-detection helpers — no LLM calls
# ---------------------------------------------------------------------------

_COMPANY_PATTERNS = [
    r"(NIKE,?\s+Inc\.?)",
    r"(Nike,?\s+Inc\.?)",
    r"Annual\s+Report\s+[—\-–]\s+([A-Z][A-Za-z0-9&\s,\.]+?)(?:\s+\d{4}|\n)",
    r"Sustainability\s+Report\s+([A-Z][A-Za-z0-9&\s,\.]+?)(?:\s+\d{4}|\n)",
    r"([A-Z][A-Z0-9&\s]{2,30})\s+(?:ESG|Sustainability|Annual)\s+Report",
]

_YEAR_PATTERNS = [
    (r"FY(\d{2})\b",           "fy2"),    # FY24 → 2024
    (r"FY(\d{4})\b",           "fy4"),    # FY2024 → 2024
    (r"fiscal\s+year\s+(\d{4})", "full"), # fiscal year 2024
    (r"year\s+ended.*?(\d{4})", "full"),  # year ended ... 2024
    (r"\b(202[0-9])\b",        "full"),   # standalone 2020-2029
]


def _detect_company_year(page_corpus: dict) -> tuple[str, int]:
    """
    Scan first 5 pages to auto-detect company name and data year.
    Returns (company_name, reporting_year). No LLM calls.
    """
    pages_text = [page_corpus.get(i, "") for i in range(1, 6)]
    combined_early = "\n".join(pages_text[:3])  # first 3 pages for company
    combined_all   = "\n".join(pages_text)       # first 5 pages for year

    # --- Company name ---
    company = ""
    for pat in _COMPANY_PATTERNS:
        m = re.search(pat, combined_early)
        if m:
            company = m.group(1).strip().rstrip(",.")
            break
    if not company:
        # Fallback: first ALL-CAPS word sequence on page 1
        caps = re.findall(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b", page_corpus.get(1, ""))
        if caps:
            company = caps[0].title()

    # --- Report year ---
    year_hits: list[int] = []
    for pat, kind in _YEAR_PATTERNS:
        for m in re.finditer(pat, combined_all, re.IGNORECASE):
            raw = int(m.group(1))
            year = (2000 + raw) if (kind == "fy2" and raw < 100) else raw
            if 2018 <= year <= 2030:
                year_hits.append(year)

    # Most frequent year across first 5 pages
    if year_hits:
        detected_year = Counter(year_hits).most_common(1)[0][0]
    else:
        detected_year = 2024  # safe default

    # Nike-style: "FY24 Sustainability Data" means data year IS 2024
    # General rule: if report uses "FY{YY}" format, treat detected year as data year directly
    # Only subtract 1 if no FY-pattern found and year looks like a publication year
    has_fy_pattern = any(
        re.search(r"FY\d{2,4}", combined_all, re.IGNORECASE) for _ in [1]
    )
    if not has_fy_pattern and re.search(r"published\s+in\s+\d{4}|annual\s+report\s+\d{4}",
                                         combined_all, re.IGNORECASE):
        detected_year = detected_year - 1

    return company or "Unknown", detected_year


_BASE = Path(__file__).parent
_THEMES_YAML = _BASE / "config" / "esg_themes_v3_FINAL.yaml"
_KPI_TEMPLATE = _BASE / "config" / "kpi_template_v8_FINAL.csv"
_LOGS_DIR = _BASE / "logging_system" / "logs"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ESGenie", page_icon="🌿", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_STATE_DEFAULTS = {
    "faiss_index":        None,
    "chunks":             [],
    "bm25_index":         None,
    "page_corpus":        {},
    "thematic_page_map":  {},
    "token_reduction_log": {},
    "extraction_results": [],
    "scores":             {},
    "run_history":        [],
    "session_cost":       0.0,
    "pdf_name":           "",
    "excel_bytes":        b"",
    "company_name":       "",
    "reporting_year":     2024,
    "custom_weights":     None,   # dict or None — set by Tab 2 weight sliders
    "heuristic_eval":     {},     # Track 2 metrics, populated after Mode 2
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 🌿 ESGenie")
st.caption("ESG Intelligence from Unstructured Reports")

tab1, tab2, tab3 = st.tabs(["📊 Analyst Workbench", "⚙️ Configuration", "🔭 Watchtower"])

# ===========================================================================
# TAB 1 — ANALYST WORKBENCH
# ===========================================================================
with tab1:
    left, right = st.columns([4, 6])

    # ---- Left column -------------------------------------------------------
    with left:
        uploaded = st.file_uploader("Upload ESG Report PDF", type=["pdf"])

        # ---- Step 1: Ingest new PDF ----------------------------------------
        if uploaded and uploaded.name != st.session_state["pdf_name"]:
            with st.spinner("Ingesting and indexing PDF..."):
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                pages = extract_pages(tmp_path)

                # Auto-detect company and year — no LLM
                det_company, det_year = _detect_company_year(pages)

                bucketer = ThematicBucketer()
                page_map, reduction_log = bucketer.bucket(pages)
                chunks = chunk_bucketed_pages(
                    pages, page_map,
                    company=det_company,
                    report_year=det_year,
                )
                prefix = uploaded.name.replace(".pdf", "").lower().replace(" ", "_")
                build_index(chunks, cache_prefix=prefix)
                faiss_idx, _ = load_index(prefix)
                bm25_idx = build_bm25(chunks, cache_prefix=prefix)

                st.session_state.update({
                    "faiss_index":        faiss_idx,
                    "chunks":             chunks,
                    "bm25_index":         bm25_idx,
                    "page_corpus":        pages,
                    "thematic_page_map":  page_map,
                    "token_reduction_log": reduction_log,
                    "pdf_name":           uploaded.name,
                    "company_name":       det_company,
                    "reporting_year":     det_year,
                    "extraction_results": [],
                    "scores":             {},
                    "excel_bytes":        b"",
                })

            total_pages = reduction_log.get("total_pages", 0)
            avg_pages   = reduction_log.get("avg_pages_per_theme", 0)
            pct         = reduction_log.get("reduction_pct", 0)
            st.success(
                f"✅ {total_pages} pages bucketed to avg {avg_pages} pages/theme "
                f"— {pct}% token reduction before any LLM call"
            )

        # ---- Step 2: Company / year — only shown after PDF is loaded -------
        if st.session_state["pdf_name"]:
            st.success(
                f"Detected: **{st.session_state['company_name']}** | "
                f"Report year: **{st.session_state['reporting_year']}**"
            )
            company = st.text_input(
                "Company Name",
                value=st.session_state["company_name"],
                key="company_input",
            )
            year_options = list(range(2025, 2018, -1))
            default_idx  = year_options.index(st.session_state["reporting_year"]) \
                           if st.session_state["reporting_year"] in year_options else 0
            year = st.selectbox("Report Year", year_options, index=default_idx,
                                key="year_input")

            # Keep session state in sync with any manual override
            st.session_state["company_name"]   = company
            st.session_state["reporting_year"] = year
        else:
            # Placeholders so downstream code always has values
            company = st.session_state["company_name"] or "Nike"
            year    = st.session_state["reporting_year"] or 2024

        # ---- Step 3: Query controls (always visible, buttons disabled until PDF) ----
        st.divider()
        st.markdown("**Mode 1: Single Query**")
        query_input   = st.text_input("Ask about a specific metric",
                                       placeholder="What are Scope 1 GHG emissions?")
        bm25_disabled = st.checkbox("Disable BM25 (demo comparison mode)", value=False)

        if st.button("Search", type="primary", disabled=st.session_state["faiss_index"] is None):
            if not query_input.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Retrieving and extracting..."):
                    result = run_single_query(
                        query=query_input,
                        faiss_index=st.session_state["faiss_index"],
                        chunks=st.session_state["chunks"],
                        bm25_index=st.session_state["bm25_index"],
                        company=company,
                        year=year,
                        bm25_disabled=bm25_disabled,
                    )
                st.session_state["extraction_results"] = [result]
                st.session_state["session_cost"] = get_session_summary()["total_cost_usd"]

                conf = result.get("confidence", "not_found")
                icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                st.markdown(f"**Result** {icon} _{conf}_")
                st.markdown(
                    f"| KPI | Value | Unit | Page | Theme |\n"
                    f"|-----|-------|------|------|-------|\n"
                    f"| {result.get('kpi_id','–')} | **{result.get('value','null')}** "
                    f"| {result.get('unit','–')} | {result.get('source_page','–')} "
                    f"| {result.get('theme','–')} |"
                )
                if result.get("direct_quote"):
                    st.caption(f"Quote: _{result['direct_quote']}_")
                if result.get("hallucination_flag"):
                    st.warning("⚠️ Hallucination flag: cited page differs from top retrieved chunk.")

        st.divider()
        st.markdown("**Mode 2: Full ESG Profile**")

        if st.button("Generate Full Profile + Excel",
                     disabled=st.session_state["faiss_index"] is None):
            if not st.session_state["pdf_name"]:
                st.warning("Please upload a PDF first.")
            else:
                progress_bar = st.progress(0, text="Starting LangGraph pipeline...")
                with st.spinner("Running Mode 2 — this calls LLM for each KPI..."):
                    # Re-use the temp file already written; save a fresh copy for profile_graph
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
                        uploaded.seek(0)
                        tmp2.write(uploaded.read())
                        tmp2_path = tmp2.name

                    result = run_profile({
                        "company_name":   company,
                        "reporting_year": year,
                        "pdf_path":       tmp2_path,
                    })
                progress_bar.progress(100, text="Profile complete.")

                extractions_out = result.get("extractions", [])
                st.session_state["extraction_results"] = extractions_out
                st.session_state["scores"]             = result.get("scores", {})
                st.session_state["session_cost"]       = result.get("total_cost", 0.0)

                if result.get("excel_bytes"):
                    st.session_state["excel_bytes"] = result["excel_bytes"]

                # ── Track 2 heuristic eval ───────────────────────────────
                tok_red_log    = result.get("token_reduction_log", {})
                _session_sum   = get_session_summary()
                _h_eval = run_eval_heuristic(
                    extractions_out,
                    total_pages    = tok_red_log.get("total_pages", 0),
                    pages_bucketed = int(
                        tok_red_log.get("avg_pages_per_theme", 0) *
                        len(result.get("thematic_page_map", {}) or {"a":1,"b":2,"c":3,"d":4,"e":5})
                    ),
                    total_tokens   = _session_sum["total_prompt_tokens"] + _session_sum["total_completion_tokens"],
                    total_cost     = result.get("total_cost", 0.0),
                    run_id         = result.get("run_id", ""),
                )
                st.session_state["heuristic_eval"] = _h_eval

                # Log for Watchtower historical view
                try:
                    log_heuristic_eval(
                        run_id   = result.get("run_id", ""),
                        company  = company,
                        year     = year,
                        heuristic_result = _h_eval,
                    )
                except Exception:
                    pass

                st.success(
                    f"Profile generated: {len(extractions_out)} KPIs extracted. "
                    f"Overall: **{result.get('scores', {}).get('overall_rating', '–')}**"
                )

        if st.session_state["excel_bytes"]:
            st.download_button(
                label="⬇️ Download Excel Report",
                data=st.session_state["excel_bytes"],
                file_name=f"esgenie_{company}_{year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

        if st.session_state.get("heuristic_eval", {}).get("per_kpi"):
            _eval_csv = build_eval_csv(st.session_state["heuristic_eval"]["per_kpi"])
            st.download_button(
                label="⬇️ Download Eval Report (CSV)",
                data=_eval_csv,
                file_name=f"esgenie_eval_{company}_{year}.csv",
                mime="text/csv",
            )

    # ---- Right column — ESG Snapshot Dashboard ----------------------------
    with right:
        scores      = st.session_state.get("scores", {})
        extractions = st.session_state.get("extraction_results", [])
        cname       = st.session_state.get("company_name", "") or company

        # Helper: look up a KPI value from current extractions
        def _kpi(kpi_id: str, field: str = "value") -> str:
            for e in extractions:
                if e.get("kpi_id") == kpi_id:
                    return str(e.get(field) or "–")
            return "–"

        def _kpi_num(kpi_id: str) -> float | None:
            for e in extractions:
                if e.get("kpi_id") == kpi_id:
                    v = e.get("value_numeric") or e.get("value")
                    try:
                        return float(str(v).replace(",", ""))
                    except (TypeError, ValueError):
                        return None
            return None

        def get_extraction_value(exts: list, kpi_id: str) -> str | None:
            """Return raw value string for a kpi_id, or None if not found."""
            for e in exts:
                if e.get("kpi_id") == kpi_id:
                    v = e.get("value")
                    if v not in (None, "", "null", "Not Found"):
                        return str(v)
            return None

        def format_metric_value(value, unit: str) -> tuple[str, str]:
            """
            Format a raw value for display in a narrow metric card.
            Returns (display_value, unit_str).
            """
            if value is None or value == "" or value == "Not Found":
                return "—", ""
            try:
                num = float(str(value).replace(",", "").strip())
            except (ValueError, TypeError):
                return str(value).strip()[:8], unit
            if num >= 1_000_000:
                return f"{num / 1_000_000:.2f}M", unit
            elif num >= 1_000:
                return f"{num / 1_000:.1f}K", unit
            else:
                return f"{num:g}", unit

        # Re-score live whenever custom weights differ from stored scores
        if extractions and st.session_state.get("custom_weights"):
            rescored = score_company(
                extractions,
                cname,
                theme_weights=st.session_state["custom_weights"],
            )
            st.session_state["scores"] = rescored
            scores = rescored

        if not scores and not extractions:
            st.info("Upload a PDF and run Mode 1 or Mode 2 to see the ESG dashboard.")
        else:
            # ── SECTION 1: Header bar ────────────────────────────────────────
            st.markdown(
                f"""<div style="background:#1F4E2C;padding:14px 18px;border-radius:6px;margin-bottom:12px;">
                <span style="color:white;font-size:1.3rem;font-weight:700;">{cname}</span>
                <span style="color:#a8d5b5;font-size:1rem;margin-left:12px;">| ESG SNAPSHOT</span><br>
                <span style="color:#c8e6c9;font-size:0.78rem;">
                Method note: scores computed from PDF disclosure quality and extracted data.
                Not official third-party ratings.</span></div>""",
                unsafe_allow_html=True,
            )

            # ── SECTION 2: Two-column layout ─────────────────────────────────
            dash_left, dash_right = st.columns([4, 6])

            with dash_left:
                # A) Overall score block
                with st.container(border=True):
                    overall = scores.get("overall_score", 0) if scores else 0
                    rating  = scores.get("overall_rating", "–") if scores else "–"
                    interp  = scores.get("overall_interpretation", "") if scores else ""
                    st.metric(
                        label="Overall ESG Score",
                        value=f"{overall:.1f} / 100",
                        delta=f"{rating} — {interp}",
                    )
                    st.caption("First extraction — no prior year comparison available")

                # B) Sector-adjusted table
                with st.container(border=True):
                    st.markdown("**Sector-Adjusted Performance**")
                    theme_scores = scores.get("theme_scores", {}) if scores else {}
                    # Load sector weights as benchmark proxy
                    try:
                        import yaml as _yaml
                        _mat = _yaml.safe_load(open(_BASE / "config" / "sector_materiality.yaml"))
                        _weights = _mat["sectors"].get("apparel_consumer", {})
                    except Exception:
                        _weights = {"climate": 0.35, "employee": 0.25,
                                    "governance": 0.30, "water": 0.10}

                    _theme_label = {"climate": "Environment", "employee": "Social",
                                    "governance": "Governance", "water": "Water"}
                    sa_rows = []
                    for th, label in _theme_label.items():
                        sc  = theme_scores.get(th, {}).get("score", 0) if theme_scores else 0
                        wt  = _weights.get(th, 0.25)
                        benchmark = round(wt * 85)  # sector leader proxy
                        sa_rows.append({"Dimension": label,
                                        "Company": round(sc),
                                        "Sector Avg": benchmark,
                                        "1Y Delta": "N/A", "3Y Delta": "N/A"})
                    # Overall row
                    sa_rows.append({"Dimension": "Overall",
                                    "Company": round(overall),
                                    "Sector Avg": round(sum(_weights.get(t,0)*85
                                                            for t in _theme_label)),
                                    "1Y Delta": "N/A", "3Y Delta": "N/A"})
                    st.dataframe(pd.DataFrame(sa_rows), hide_index=True,
                                 use_container_width=True)

                # C) Reporting quality
                with st.container(border=True):
                    st.markdown("**Reporting Quality**")
                    try:
                        kpi_df_rq = pd.read_csv(_KPI_TEMPLATE)
                        for th, label in _theme_label.items():
                            expected = len(kpi_df_rq[
                                (kpi_df_rq["theme"] == th) &
                                (~kpi_df_rq["kpi_id"].str.startswith("CONT"))
                            ])
                            found = sum(
                                1 for e in extractions
                                if e.get("kpi_theme") == th
                                and e.get("value") not in (None, "", "null", "Not Found")
                            )
                            pct_rq = (found / expected * 100) if expected else 0
                            filled = int(pct_rq / 20)  # dots out of 5
                            dots = "●" * filled + "○" * (5 - filled)
                            st.markdown(f"`{label:<14}` {dots}  **{pct_rq:.0f}%**")
                    except Exception:
                        st.caption("Reporting quality unavailable.")

                # D) Auto-generated ESG summary (rule-based, no LLM)
                with st.container(border=True):
                    st.markdown("**ESG Summary**")
                    if theme_scores:
                        best_th  = max(theme_scores, key=lambda t: theme_scores[t].get("score", 0))
                        worst_th = min(theme_scores, key=lambda t: theme_scores[t].get("completeness", 1))
                        best_sc  = theme_scores[best_th].get("score", 0)
                        worst_comp = theme_scores[worst_th].get("completeness", 0)
                        st.markdown(
                            f"**{cname}** demonstrates strongest performance in "
                            f"**{best_th.capitalize()}** (score: {best_sc:.0f}/100)."
                        )
                        st.markdown(
                            f"**{worst_th.capitalize()}** disclosure remains limited, "
                            f"with only **{worst_comp:.0%}** of expected metrics reported."
                        )
                        # Notable highlight — renewable energy or scope 2
                        scope2 = _kpi_num("C002")
                        re_pct = _kpi_num("C009") or _kpi_num("C010")
                        if re_pct and re_pct >= 90:
                            st.markdown(
                                f"Renewable electricity coverage of **{re_pct:.0f}%** "
                                "signals deep decarbonization commitment."
                            )
                        elif scope2 and scope2 < 15000:
                            st.markdown(
                                f"Scope 2 market-based emissions of **{scope2:,.0f} tCO2e** "
                                "reflect significant renewable energy adoption."
                            )
                    else:
                        st.caption("Run Mode 2 for full ESG summary.")

            with dash_right:
                # A) Climate & Energy — 2×2 grid
                with st.container(border=True):
                    st.markdown("**Climate & Energy**")

                    scope1_v,   scope1_u   = format_metric_value(get_extraction_value(extractions, "C001"), "tCO2e")
                    scope3_v,   scope3_u   = format_metric_value(get_extraction_value(extractions, "C003"), "tCO2e")
                    renew_v,    renew_u    = format_metric_value(get_extraction_value(extractions, "C015"), "%")
                    scope2_v,   scope2_u   = format_metric_value(get_extraction_value(extractions, "C002"), "tCO2e")

                    _cr1, _cr2 = st.columns(2)
                    _cr3, _cr4 = st.columns(2)
                    with _cr1:
                        st.metric("Scope 1 Emissions",   f"{scope1_v} {scope1_u}".strip())
                    with _cr2:
                        st.metric("Scope 3 Total",        f"{scope3_v} {scope3_u}".strip())
                    with _cr3:
                        st.metric("Renewable Energy",     f"{renew_v} {renew_u}".strip())
                    with _cr4:
                        st.metric("Scope 2 Market-Based", f"{scope2_v} {scope2_u}".strip())

                # B) Water & Waste — 2×2 grid
                with st.container(border=True):
                    st.markdown("**Water & Waste**")

                    intensity_v, intensity_u = format_metric_value(get_extraction_value(extractions, "W011"), "L/kg")
                    zdhc_v,      zdhc_u      = format_metric_value(get_extraction_value(extractions, "W003"), "%")
                    waste_v,     waste_u     = format_metric_value(get_extraction_value(extractions, "E001"), "MT")
                    restored_v,  restored_u  = format_metric_value(get_extraction_value(extractions, "W014"), "L")

                    _wr1, _wr2 = st.columns(2)
                    _wr3, _wr4 = st.columns(2)
                    with _wr1:
                        st.metric("Freshwater Intensity", f"{intensity_v} {intensity_u}".strip())
                    with _wr2:
                        st.metric("ZDHC Compliance",      f"{zdhc_v} {zdhc_u}".strip())
                    with _wr3:
                        st.metric("Hazardous Waste",      f"{waste_v} {waste_u}".strip())
                    with _wr4:
                        st.metric("Water Restored",       f"{restored_v} {restored_u}".strip())

                # C) Key Material Issues table
                with st.container(border=True):
                    st.markdown("**Key Material Issues (SASB-aligned)**")
                    _issue_theme_map = [
                        ("Climate / GHG",       "climate"),
                        ("Water Management",    "water"),
                        ("Labour Practices",    "employee"),
                        ("Supply Chain",        "employee"),
                        ("Governance / Ethics", "governance"),
                    ]
                    issue_rows = []
                    for issue, th in _issue_theme_map:
                        sc = theme_scores.get(th, {}).get("score", 0) if theme_scores else 0
                        perf = "Strong" if sc > 70 else "Moderate" if sc >= 45 else "Weak"
                        wt   = _weights.get(th, 0.25)
                        issue_rows.append({
                            "Issue":          issue,
                            "Performance":    perf,
                            "Company Score":  round(sc),
                            "Sector Proxy":   round(wt * 85),
                            "Trend":          "–",
                        })
                    st.dataframe(pd.DataFrame(issue_rows), hide_index=True,
                                 use_container_width=True)

            # ── SECTION 3: Governance summary (full width) ────────────────────
            with st.container(border=True):
                st.markdown("**Governance Summary**")
                gov_rows = [
                    {"Indicator": "Board independence",
                     "Assessment": _kpi("G002") if _kpi("G002") != "–" else "Not disclosed",
                     "Sector Comparison": "~75% (S&P 500 avg)"},
                    {"Indicator": "Board diversity (% women)",
                     "Assessment": _kpi("G003") if _kpi("G003") != "–" else "Not disclosed",
                     "Sector Comparison": "~33% (S&P 500 avg)"},
                    {"Indicator": "Chair / CEO separation",
                     "Assessment": _kpi("G005") if _kpi("G005") != "–" else "Not disclosed",
                     "Sector Comparison": "~45% separated"},
                    {"Indicator": "ESG-linked executive pay",
                     "Assessment": _kpi("G007") if _kpi("G007") != "–" else "Not disclosed",
                     "Sector Comparison": "~64% of S&P 100"},
                    {"Indicator": "ESG assurance",
                     "Assessment": _kpi("G015") if _kpi("G015") != "–" else "Not disclosed",
                     "Sector Comparison": "Growing practice"},
                ]
                st.dataframe(pd.DataFrame(gov_rows), hide_index=True,
                             use_container_width=True)

            # ── SECTION 4: Controversies ──────────────────────────────────────
            with st.container(border=True):
                st.markdown("**Controversies & Flags**")
                cont1 = _kpi("CONT001", "value_subjective") or _kpi("CONT001")
                cont2 = _kpi("CONT002", "value_subjective") or _kpi("CONT002")
                cont_ded = scores.get("controversy_total_deduction", 0) if scores else 0

                if cont1 == "–" and cont2 == "–":
                    st.success("No controversies identified in current extraction run.")
                else:
                    cont_rows = [
                        {"Category": "Supply chain environment (CONT001)",
                         "Status": cont1, "Deduction": f"{CONTROVERSY_DEDUCTIONS.get(cont1.upper(), 0)} pts"},
                        {"Category": "Product / marketing claims (CONT002)",
                         "Status": cont2, "Deduction": f"{CONTROVERSY_DEDUCTIONS.get(cont2.upper(), 0)} pts"},
                    ]
                    st.dataframe(pd.DataFrame(cont_rows), hide_index=True,
                                 use_container_width=True)
                    if cont_ded < 0:
                        st.warning(f"Total controversy deduction: **{cont_ded} pts** applied to overall score.")

            # ── Full KPI table (collapsed by default) ─────────────────────────
            if extractions:
                with st.expander(f"Full KPI Extraction Table ({len(extractions)} KPIs)"):
                    rows = []
                    for e in extractions:
                        rows.append({
                            "KPI ID":    e.get("kpi_id", ""),
                            "Metric":    e.get("kpi_canonical_name") or e.get("metric", ""),
                            "Value":     e.get("value") or "Not Found",
                            "Unit":      e.get("unit", ""),
                            "Page":      e.get("source_page", ""),
                            "Conf":      e.get("confidence", "not_found"),
                            "Flag":      "⚠️" if e.get("hallucination_flag") else "",
                        })
                    df_full = pd.DataFrame(rows)

                    def _rc(conf):
                        return {"high": "background-color:#E2EFDA",
                                "medium": "background-color:#FFFF99",
                                "low": "background-color:#FFE0E0",
                                "not_found": "background-color:#F2F2F2"}.get(conf, "")

                    st.dataframe(
                        df_full.style.apply(lambda c: [_rc(v) for v in df_full["Conf"]], axis=0),
                        use_container_width=True, hide_index=True,
                    )

        cost = st.session_state.get("session_cost", 0.0)
        st.caption(f"Session cost: **${cost:.4f}**")

# ===========================================================================
# TAB 2 — CONFIGURATION
# ===========================================================================
with tab2:
    st.markdown("### Agent Prompts")
    st.markdown("Editing changes the in-session prompt only. Reload to reset.")
    prompt_val = st.text_area(
        "Extraction Agent System Prompt",
        value=SYSTEM_PROMPT,
        height=250,
    )
    if st.button("Save Prompt (session only)"):
        import agents.extraction_agent as _ea
        _ea.SYSTEM_PROMPT = prompt_val
        st.success("Prompt updated for this session.")

    st.divider()
    st.markdown("### ESG Theme Keywords (esg_themes_v3_FINAL.yaml)")
    try:
        with open(_THEMES_YAML, "r", encoding="utf-8") as f:
            yaml_text = f.read()
        yaml_edit = st.text_area("YAML Config", value=yaml_text, height=300)
        if st.button("Validate YAML"):
            try:
                yaml.safe_load(yaml_edit)
                st.success("Valid YAML.")
            except yaml.YAMLError as e:
                st.error(f"YAML error: {e}")
    except FileNotFoundError:
        st.error("esg_themes_v3_FINAL.yaml not found.")

    st.divider()
    st.markdown("### KPI Template (kpi_template_v8_FINAL.csv)")
    try:
        kpi_df = pd.read_csv(_KPI_TEMPLATE)
        edited_df = st.data_editor(kpi_df, use_container_width=True, num_rows="dynamic")
        if st.button("Save KPI Template"):
            edited_df.to_csv(_KPI_TEMPLATE, index=False)
            st.success("KPI template saved.")
    except FileNotFoundError:
        st.error("kpi_template_v8_FINAL.csv not found.")

    st.warning("Changes take effect on next run. Reload to reset defaults.")

    st.divider()
    st.markdown("### Scoring Model")

    # ── SUB-SECTION 1: Rating Scale Reference ────────────────────────────────
    st.markdown("#### Rating Scale Reference")
    _rating_rows = [
        {"Rating": "AAA", "Score Range": "85 – 100", "Label": "Leader",
         "Colour": "#1a6b2f"},
        {"Rating": "AA",  "Score Range": "75 – 84",  "Label": "Leader",
         "Colour": "#2e8b57"},
        {"Rating": "A",   "Score Range": "60 – 74",  "Label": "Above Average",
         "Colour": "#52b788"},
        {"Rating": "BBB", "Score Range": "45 – 59",  "Label": "Average",
         "Colour": "#b5c400"},
        {"Rating": "BB",  "Score Range": "30 – 44",  "Label": "Below Average",
         "Colour": "#f4a261"},
        {"Rating": "B",   "Score Range": "15 – 29",  "Label": "Laggard",
         "Colour": "#e07b39"},
        {"Rating": "CCC", "Score Range": "0 – 14",   "Label": "Severe Laggard",
         "Colour": "#c0392b"},
    ]
    _rating_table_html = """
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="padding:6px 10px;text-align:left;border:1px solid #ddd;">Rating</th>
          <th style="padding:6px 10px;text-align:left;border:1px solid #ddd;">Score Range</th>
          <th style="padding:6px 10px;text-align:left;border:1px solid #ddd;">Label</th>
        </tr>
      </thead>
      <tbody>
    """
    for r in _rating_rows:
        _rating_table_html += (
            f'<tr>'
            f'<td style="padding:6px 10px;border:1px solid #ddd;'
            f'background:{r["Colour"]};color:white;font-weight:700;">{r["Rating"]}</td>'
            f'<td style="padding:6px 10px;border:1px solid #ddd;">{r["Score Range"]}</td>'
            f'<td style="padding:6px 10px;border:1px solid #ddd;">{r["Label"]}</td>'
            f'</tr>\n'
        )
    _rating_table_html += "</tbody></table>"
    st.markdown(_rating_table_html, unsafe_allow_html=True)

    # ── SUB-SECTION 2: Theme Weight Sliders ──────────────────────────────────
    st.markdown("#### Theme Weight Sliders")
    st.caption(
        "Adjust weights to reflect your sector/portfolio materiality view. "
        "Weights must sum to 100%. Changes apply to the dashboard instantly "
        "when extractions are loaded."
    )

    # Defaults: apparel_consumer from sector_materiality.yaml
    _default_weights = {"climate": 35, "employee": 25, "governance": 30, "water": 10}

    # Initialise current slider values from session state or defaults
    if "weight_sliders" not in st.session_state:
        st.session_state["weight_sliders"] = dict(_default_weights)

    sw = st.session_state["weight_sliders"]

    _slider_cols = st.columns(4)
    _theme_labels = {
        "climate":    "Climate / Environment",
        "employee":   "Social / Labour",
        "governance": "Governance",
        "water":      "Water & Biodiversity",
    }
    new_sw = {}
    for i, (th, label) in enumerate(_theme_labels.items()):
        with _slider_cols[i]:
            new_sw[th] = st.slider(
                label, min_value=0, max_value=100, step=5,
                value=sw.get(th, _default_weights[th]),
                key=f"wslider_{th}",
            )

    _total_pct = sum(new_sw.values())
    if _total_pct == 100:
        st.success(f"Total: **{_total_pct}%** ✓ — weights are valid.")
        # Normalise to fractions and store
        _normalised = {k: v / 100 for k, v in new_sw.items()}
        st.session_state["weight_sliders"]  = new_sw
        st.session_state["custom_weights"]  = _normalised
    else:
        _diff = 100 - _total_pct
        _sign = "+" if _diff > 0 else ""
        st.warning(
            f"Total: **{_total_pct}%** — must equal 100%. "
            f"Adjust any slider by {_sign}{_diff}%."
        )
        # Keep last valid custom_weights; don't apply invalid allocation
        st.session_state["weight_sliders"] = new_sw

    _reset_col, _ = st.columns([1, 3])
    with _reset_col:
        if st.button("Reset to Sector Defaults (Apparel)"):
            st.session_state["weight_sliders"] = dict(_default_weights)
            st.session_state["custom_weights"] = None
            st.rerun()

    # ── SUB-SECTION 3: How Scores Are Calculated ──────────────────────────────
    st.markdown("#### How Scores Are Calculated")
    with st.expander("Scoring Methodology — click to expand", expanded=False):
        st.markdown("""
**Formula Overview**

```
Base Score  = (40% × Exposure_Relevance × 100)
            + (60% × Management_Effectiveness × 100)

Overall ESG = Σ (Theme_Score × Theme_Weight)
            + Controversy_Adjustment
```

**Exposure Relevance** — derived from the sector materiality weights
(adjustable via sliders above). Higher-weight themes contribute more to
the overall score.

**Management Effectiveness**
```
Completeness     = found_KPIs / expected_KPIs  (for theme)
Avg_Confidence   = mean of per-KPI confidence:
                   high=1.0 | medium=0.6 | low=0.3 | not_found=0.0
Management       = (Completeness × 0.5) + (Avg_Confidence × 0.5)
```

**Controversy Adjustment** (capped at –20 pts combined)
| Severity | Deduction |
|----------|-----------|
| NONE     | 0 pts     |
| MILD     | –3 pts    |
| MODERATE | –8 pts    |
| SEVERE   | –15 pts   |

Controversy KPIs: CONT001 (supply-chain env), CONT002 (product/marketing claims).

**Final score** is clamped to [0, 100] before rating assignment.

**Data source**: GPT-4o-mini extraction with hallucination flagging.
One LLM call per full profile, on BM25-bucketed context (~97% token reduction).

**This is an internal model** — not an official Sustainalytics/MSCI rating.
""")

    _METHODOLOGY_MD = """# ESGenie — Scoring Methodology

## Formula

    Base Score  = (40% × Exposure_Relevance × 100)
                + (60% × Management_Effectiveness × 100)

    Overall ESG = Σ (Theme_Score × Theme_Weight) + Controversy_Adjustment

## Exposure Relevance
Derived from sector materiality weights. Default (Apparel/Consumer):
- Climate/Environment: 35%
- Governance: 30%
- Social/Labour: 25%
- Water & Biodiversity: 10%

## Management Effectiveness
    Completeness   = found_KPIs / expected_KPIs
    Avg_Confidence = mean(high=1.0 | medium=0.6 | low=0.3 | not_found=0.0)
    Management     = (Completeness × 0.5) + (Avg_Confidence × 0.5)

## Controversy Adjustment
| Severity | Deduction |
|----------|-----------|
| NONE     | 0 pts     |
| MILD     | -3 pts    |
| MODERATE | -8 pts    |
| SEVERE   | -15 pts   |

## Rating Scale
| Rating | Range  | Label         |
|--------|--------|---------------|
| AAA    | 85-100 | Leader        |
| AA     | 75-84  | Leader        |
| A      | 60-74  | Above Average |
| BBB    | 45-59  | Average       |
| BB     | 30-44  | Below Average |
| B      | 15-29  | Laggard       |
| CCC    | 0-14   | Severe Laggard|

## Data Quality
Extraction: GPT-4o-mini with structured JSON output.
Retrieval: FAISS semantic + BM25 keyword hybrid (RRF merge).
Bucketing: BM25 page-level filtering → ~97% token reduction.
Hallucination flag: source_page ≠ top retrieved chunk page.

---
*ESGenie internal model. Not an official third-party ESG rating.*
"""
    st.download_button(
        label="⬇️ Download Methodology (Markdown)",
        data=_METHODOLOGY_MD,
        file_name="esgenie_scoring_methodology.md",
        mime="text/markdown",
    )

# ===========================================================================
# TAB 3 — WATCHTOWER
# ===========================================================================
with tab3:
    st.markdown("### 🔭 Watchtower")

    # ---- Section A: System Health -----------------------------------------
    st.markdown("#### System Health")
    try:
        conn = init_db()
        runs = get_all_runs(conn)
        conn.close()

        total_runs   = len(runs)
        success_runs = sum(1 for r in runs if r.get("status") == "success")
        success_rate = success_runs / total_runs if total_runs else 0
        avg_duration = sum(r.get("duration_seconds", 0) for r in runs) / total_runs if total_runs else 0

        h1, h2, h3 = st.columns(3)
        h1.metric("Total Runs",    total_runs)
        h2.metric("Success Rate",  f"{success_rate:.0%}")
        h3.metric("Avg Duration",  f"{avg_duration:.1f}s")

        if runs:
            st.markdown("**Last 10 Runs**")
            runs_df = pd.DataFrame(runs[:10])[
                ["run_id", "company_name", "status", "total_cost_usd",
                 "token_reduction_pct", "created_at"]
            ]
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"No run data yet. ({e})")

    st.divider()

    # ---- Section B: AI Performance ----------------------------------------
    st.markdown("#### AI Performance")

    _h = st.session_state.get("heuristic_eval", {})
    _cur_company = st.session_state.get("company_name", "")
    _cur_year    = st.session_state.get("reporting_year", 2024)

    # ── Track 2: Heuristic Eval (always shown when extractions exist) ────────
    if _h:
        st.markdown(
            "**Track 2 — Heuristic Evaluation** "
            "<span style='background:#e8f5e9;color:#1b5e20;"
            "padding:2px 8px;border-radius:4px;font-size:0.8rem;'>"
            "Works for any company</span>",
            unsafe_allow_html=True,
        )

        _cov = _h.get("coverage", {})
        _tok = _h.get("token_efficiency", {})
        _src = _h.get("source_type_dist", {})

        # Row 1: key metrics
        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric(
            "Extraction Coverage",
            f"{_cov.get('overall_pct', 0):.1f}%",
            help=f"{_cov.get('found', 0)} of {_cov.get('attempted', 0)} KPIs found",
        )
        hm2.metric(
            "Hallucination Rate",
            f"{_h.get('hallucination_rate', 0):.1f}%",
            delta=("⚠️ >10%" if _h.get("hallucination_rate", 0) > 10 else None),
            delta_color="inverse",
            help="Source page cited by LLM does not match top retrieved chunk",
        )
        hm3.metric(
            "Hint Match Rate",
            f"{_h.get('hint_match_rate', 0):.1f}%",
            help="Source page within ±5 pages of KPI template hint (directional signal)",
        )
        hm4.metric(
            "BM25 Agreement",
            f"{_h.get('bm25_agreement', 0):.1f}%",
            help="Both semantic and BM25 retrievers ranked chunk in top 3",
        )

        # Row 2: confidence distribution (donut) + source type
        conf_col, src_col = st.columns([3, 2])
        with conf_col:
            st.markdown("**Confidence Distribution**")
            _cfd = _h.get("confidence_dist", {})
            try:
                import plotly.graph_objects as go
                _conf_labels = ["High", "Medium", "Low", "Not Found"]
                _conf_values = [
                    _cfd.get("high", 0), _cfd.get("medium", 0),
                    _cfd.get("low", 0),  _cfd.get("not_found", 0),
                ]
                _conf_colors = ["#2e8b57", "#b5c400", "#f4a261", "#cccccc"]
                fig_donut = go.Figure(go.Pie(
                    labels=_conf_labels, values=_conf_values,
                    hole=0.55, marker_colors=_conf_colors,
                    textinfo="label+percent",
                ))
                fig_donut.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            except ImportError:
                st.caption(
                    f"High: {_cfd.get('high',0)} | Medium: {_cfd.get('medium',0)} "
                    f"| Low: {_cfd.get('low',0)} | Not Found: {_cfd.get('not_found',0)}"
                )

        with src_col:
            st.markdown("**Source Type**")
            st.caption("Table extractions are generally higher confidence.")
            st.metric("Table", f"{_src.get('table_pct', 0):.1f}%")
            st.metric("Prose", f"{_src.get('prose_pct', 0):.1f}%")
            if _src.get("unknown_pct", 0):
                st.metric("Unknown", f"{_src.get('unknown_pct', 0):.1f}%")

        # Row 3: Token Efficiency
        with st.container(border=True):
            st.markdown("**Token Efficiency**")
            st.caption(
                "BM25 bucketing reduces tokens before any LLM call — "
                "the core architectural differentiator."
            )
            te1, te2, te3, te4 = st.columns(4)
            te1.metric("Pages (raw)",     _tok.get("pages_before", "–"))
            te2.metric("Pages (bucketed)",_tok.get("pages_after", "–"))
            te3.metric(
                "Token Reduction",
                f"{_tok.get('token_reduction_pct', 0):.1f}%",
            )
            te4.metric(
                "Cost per KPI",
                f"${_tok.get('cost_per_kpi', 0):.5f}",
            )

        # Row 4: Per-theme coverage table
        _by_theme = _cov.get("by_theme", {})
        if _by_theme:
            with st.expander("Per-Theme Coverage", expanded=False):
                _theme_cov_rows = [
                    {
                        "Theme":       th,
                        "Found":       d.get("found", 0),
                        "Expected":    d.get("attempted", 0),
                        "Coverage %":  d.get("coverage_pct", 0),
                    }
                    for th, d in sorted(_by_theme.items())
                ]
                st.dataframe(
                    pd.DataFrame(_theme_cov_rows),
                    hide_index=True,
                    use_container_width=True,
                )

        # Eval CSV download
        if _h.get("per_kpi"):
            _eval_csv_wt = build_eval_csv(_h["per_kpi"])
            st.download_button(
                label="⬇️ Download Eval Report (CSV)",
                data=_eval_csv_wt,
                file_name=f"esgenie_eval_{_cur_company}_{_cur_year}.csv",
                mime="text/csv",
                key="eval_csv_watchtower",
            )

        # ── Track 1: Ground Truth Eval (Nike or matching GT data) ───────────
        if has_ground_truth(_cur_company, _cur_year):
            st.divider()
            st.markdown(
                "**Track 1 — Ground Truth Evaluation** "
                "<span style='background:#e3f2fd;color:#0d47a1;"
                "padding:2px 8px;border-radius:4px;font-size:0.8rem;'>"
                "Ground Truth Validated</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"ground_truth.csv has rows for {_cur_company} {_cur_year}. "
                "Running Ground Truth Eval requires re-running LLM queries — "
                "use this for benchmarking, not every run."
            )
            if st.button("Run Ground Truth Eval (LLM calls)"):
                if st.session_state.get("faiss_index") is None:
                    st.warning("Upload a PDF first to build indexes.")
                else:
                    with st.spinner("Running Track 1 eval — one LLM call per GT row..."):
                        from evaluation.eval_runner import run_eval, print_eval_report
                        gt_result = run_eval(
                            faiss_index=st.session_state["faiss_index"],
                            chunks=st.session_state["chunks"],
                            bm25_index=st.session_state["bm25_index"],
                            company=_cur_company,
                            year=_cur_year,
                        )
                    if gt_result:
                        g1, g2, g3, g4 = st.columns(4)
                        g1.metric("Retrieval Acc (P@5)",
                                  f"{gt_result['retrieval_accuracy']:.1%}")
                        g2.metric("Answer Accuracy",
                                  f"{gt_result['answer_accuracy']:.1%}")
                        g3.metric("Null Rate",
                                  f"{gt_result['null_rate']:.1%}")
                        g4.metric("Avg Cost/Query",
                                  f"${gt_result['avg_cost_per_query']:.5f}")
                        with st.expander("Per-Query GT Results"):
                            st.dataframe(
                                pd.DataFrame(gt_result.get("per_query", [])),
                                hide_index=True,
                                use_container_width=True,
                            )

    else:
        st.info("Run Mode 2 (Full ESG Profile) to see AI performance metrics.")

    # ── Historical log (retrieval-level entries, not heuristic summaries) ────
    ai_log_path = _LOGS_DIR / "ai_metrics.jsonl"
    if ai_log_path.exists():
        try:
            _all_ai = [
                json.loads(l)
                for l in ai_log_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            # Show retrieval-level entries only (exclude heuristic_eval summaries)
            _retrieval_entries = [
                e for e in _all_ai
                if e.get("event") != "heuristic_eval" and "query" in e
            ]
            if _retrieval_entries:
                with st.expander(
                    f"Retrieval Log — last {min(20, len(_retrieval_entries))} queries",
                    expanded=False,
                ):
                    _log_df = pd.DataFrame(_retrieval_entries).tail(20)
                    _log_cols = [
                        c for c in
                        ["query", "theme_classified", "precision_at_5",
                         "mrr", "extraction_confidence", "hallucination_flag"]
                        if c in _log_df.columns
                    ]
                    st.dataframe(
                        _log_df[_log_cols],
                        use_container_width=True, hide_index=True,
                    )

            # Historical heuristic eval summaries
            _heuristic_hist = [e for e in _all_ai if e.get("event") == "heuristic_eval"]
            if _heuristic_hist:
                with st.expander(
                    f"Historical Profile Evals ({len(_heuristic_hist)} runs)",
                    expanded=False,
                ):
                    _hist_df = pd.DataFrame(_heuristic_hist)[[
                        "timestamp", "company", "year",
                        "coverage_pct", "hallucination_rate",
                        "token_reduction_pct", "cost_per_kpi",
                    ]].sort_values("timestamp", ascending=False)
                    st.dataframe(_hist_df, hide_index=True, use_container_width=True)
        except Exception:
            pass

    st.divider()

    # ---- Section C: Cost Dashboard ----------------------------------------
    st.markdown("#### Cost Dashboard")

    tok_log_path = _LOGS_DIR / "token_costs.jsonl"
    cost_source = tok_log_path if tok_log_path.exists() else None

    session = get_session_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Session Tokens",      f"{session['total_prompt_tokens'] + session['total_completion_tokens']:,}")
    c2.metric("Session Cost (USD)",  f"${session['total_cost_usd']:.4f}")
    c3.metric("Embedding Tokens",    f"{session['total_embedding_tokens']:,}")

    if cost_source:
        try:
            cost_entries = [json.loads(l) for l in cost_source.read_text().splitlines() if l.strip()]
            if cost_entries:
                cost_df = pd.DataFrame(cost_entries)
                if "model" in cost_df.columns and "cost_usd" in cost_df.columns:
                    model_costs = cost_df.groupby("model")["cost_usd"].sum().reset_index()
                    st.markdown("**Cost by Model**")
                    st.bar_chart(model_costs.set_index("model")["cost_usd"])

                total_cost = cost_df["cost_usd"].sum() if "cost_usd" in cost_df.columns else 0
                profiles   = max(total_runs, 1)
                st.metric("Total Cost (all time)", f"${total_cost:.4f}")
                st.metric("Avg Cost per Profile",  f"${total_cost/profiles:.4f}")
        except Exception:
            pass
    else:
        st.info("No cost log yet. Token costs will appear here after first LLM call.")
