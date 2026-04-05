"""
eval_runner.py
Two-track evaluation system:

  Track 1 — Ground Truth Eval (run_eval):
    Compares extracted values against known expected values in ground_truth.csv.
    Only available when the CSV has matching company+year rows (e.g. Nike 2024).
    Computes: Retrieval Accuracy, Answer Accuracy, Precision@5, MRR, Null Rate.

  Track 2 — Heuristic Eval (run_eval_heuristic):
    Works for ANY company. Computes metrics directly from extraction results:
      A) Extraction Coverage     B) Confidence Distribution
      C) Hallucination Rate      D) BM25 vs Semantic Agreement
      E) Token Efficiency        F) Retrieval Quality (page hint match)
      G) Source Type Distribution
    Returns per-KPI rows for CSV export via build_eval_csv().
"""

import csv
import io
import json
import time
from pathlib import Path

import pandas as pd

from agents.orchestrator import run_single_query
from utils.token_tracker import get_session_summary, reset_session

_GT_PATH           = Path(__file__).parent.parent / "config" / "ground_truth.csv"
_KPI_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "kpi_template_v8_FINAL.csv"
_AI_METRICS_PATH   = Path(__file__).parent.parent / "logging_system" / "logs" / "ai_metrics.jsonl"

# Values that mean "not found"
_NULL_VALUES = {None, "", "null", "Not Found"}


# ===========================================================================
# Track 1 helpers — unchanged from original
# ===========================================================================

def _numeric_score(predicted: str | None, expected: str | float) -> float:
    """
    Compare a predicted numeric value against expected.
    Returns 1.0 (exact ≤2%), 0.8 (close ≤5%), 0.0 otherwise.
    """
    if predicted is None:
        return 0.0
    try:
        pred = float(str(predicted).replace(",", "").strip())
        exp  = float(str(expected).replace(",", "").strip())
        if exp == 0:
            return 1.0 if pred == 0 else 0.0
        pct_diff = abs(pred - exp) / abs(exp)
        if pct_diff <= 0.02:
            return 1.0
        if pct_diff <= 0.05:
            return 0.8
        return 0.0
    except (ValueError, TypeError):
        return 0.0


def _text_score(predicted: str | None, expected: str) -> float:
    """Exact case-insensitive match for boolean/subjective values."""
    if predicted is None:
        return 0.0
    return 1.0 if str(predicted).strip().lower() == str(expected).strip().lower() else 0.0


def _page_in_top5(retrieval_results: list[dict], expected_page: int | None) -> bool:
    """True if expected source page appears in any of the top-5 retrieved chunks."""
    if expected_page is None:
        return False
    for r in retrieval_results[:5]:
        chunk = r.get("chunk", r)
        page  = chunk.get("page_number") or chunk.get("page_num")
        if page == int(expected_page):
            return True
    return False


# ===========================================================================
# Track 1 — Ground Truth Eval
# ===========================================================================

def run_eval(
    faiss_index,
    chunks: list[dict],
    bm25_index,
    company: str = "Nike",
    year: int = 2024,
) -> dict:
    """
    Run Track 1 evaluation against ground truth for a given company/year.

    Args:
        faiss_index:  Built FAISS index.
        chunks:       Chunk metadata list.
        bm25_index:   BM25Okapi index.
        company:      Company name to filter ground truth rows.
        year:         Reporting year to filter ground truth rows.

    Returns:
        Dict with per-query results and aggregate metrics, or {} if no GT rows.
    """
    gt_df = pd.read_csv(_GT_PATH)
    rows = gt_df[
        (gt_df["company"] == company) &
        (gt_df["report_year"] == year)
    ].to_dict("records")

    if not rows:
        print(f"[eval] No ground truth rows found for {company} {year}")
        return {}

    reset_session()
    per_query: list[dict] = []
    retrieval_hits  = 0
    answer_scores: list[float] = []
    latencies: list[float]     = []
    null_count = 0

    for row in rows:
        kpi_id    = row["kpi_id"]
        canonical = row["kpi_canonical_name"]
        expected  = row["expected_value"]
        exp_page  = row.get("source_page")
        vtype     = row.get("value_type", "numeric")
        unit      = row.get("expected_unit", "")
        query     = f"What is the {canonical}?"

        t0 = time.time()
        result = run_single_query(
            query=query,
            faiss_index=faiss_index,
            chunks=chunks,
            bm25_index=bm25_index,
            company=company,
            year=year,
            kpi_id=kpi_id,
            value_type=vtype,
            unit_expected=str(unit),
        )
        latency_ms = (time.time() - t0) * 1000

        predicted         = result.get("value")
        retrieval_results = result.get("retrieval_results", [])

        if predicted in (None, "", "null", "Not Found"):
            score = 0.0
            null_count += 1
        elif vtype == "numeric":
            score = _numeric_score(predicted, expected)
        else:
            score = _text_score(predicted, expected)

        precision_hit = _page_in_top5(retrieval_results, exp_page)
        if precision_hit:
            retrieval_hits += 1

        answer_scores.append(score)
        latencies.append(latency_ms)

        per_query.append({
            "kpi_id":          kpi_id,
            "canonical":       canonical,
            "expected":        expected,
            "predicted":       predicted,
            "answer_score":    score,
            "precision_hit":   precision_hit,
            "source_page_exp": exp_page,
            "source_page_got": result.get("source_page"),
            "confidence":      result.get("confidence"),
            "hallucination":   result.get("hallucination_flag", 0),
            "latency_ms":      round(latency_ms),
        })

    n       = len(rows)
    session = get_session_summary()
    avg_cost = session["total_cost_usd"] / n if n else 0

    return {
        "track":              "ground_truth",
        "company":            company,
        "year":               year,
        "test_queries":       n,
        "retrieval_accuracy": retrieval_hits / n,
        "answer_accuracy":    sum(answer_scores) / n,
        "avg_latency_ms":     sum(latencies) / n,
        "avg_cost_per_query": avg_cost,
        "null_rate":          null_count / n,
        "total_cost_usd":     session["total_cost_usd"],
        "per_query":          per_query,
    }


def print_eval_report(eval_results: dict) -> None:
    """
    Print a clean evaluation summary to stdout.

    Args:
        eval_results: Output of run_eval().
    """
    if not eval_results:
        print("No evaluation results to display.")
        return

    print()
    print("=== ESGENIE EVALUATION REPORT ===")
    print(f"Company:                         {eval_results['company']} {eval_results['year']}")
    print(f"Test queries:                    {eval_results['test_queries']}")
    print(f"Retrieval Accuracy (Precision@5):{eval_results['retrieval_accuracy']:.1%}")
    print(f"Answer Accuracy:                 {eval_results['answer_accuracy']:.1%}")
    print(f"Avg Latency:                     {eval_results['avg_latency_ms']:.0f}ms")
    print(f"Avg Cost per Query:              ${eval_results['avg_cost_per_query']:.5f}")
    print(f"Null Rate:                       {eval_results['null_rate']:.1%}")
    print(f"Total Cost:                      ${eval_results['total_cost_usd']:.4f}")
    print()

    print(f"{'KPI ID':<10} {'Expected':<12} {'Got':<12} {'Score':<7} {'Page?':<6} {'Conf'}")
    print("-" * 65)
    for r in eval_results.get("per_query", []):
        page_ok       = "OK" if r["precision_hit"] else "MISS"
        predicted_str = str(r["predicted"] or "null")[:11]
        expected_str  = str(r["expected"]  or "")[:11]
        print(
            f"{r['kpi_id']:<10} {expected_str:<12} {predicted_str:<12} "
            f"{r['answer_score']:<7.2f} {page_ok:<6} {r['confidence'] or 'n/a'}"
        )


# ===========================================================================
# Track 2 — Heuristic Eval (any company, no ground truth required)
# ===========================================================================

def _load_kpi_template() -> pd.DataFrame:
    return pd.read_csv(_KPI_TEMPLATE_PATH)


def run_eval_heuristic(
    extractions: list[dict],
    total_pages: int = 0,
    pages_bucketed: int = 0,
    total_tokens: int = 0,
    total_cost: float = 0.0,
    run_id: str = "",
) -> dict:
    """
    Compute Track 2 heuristic evaluation metrics from extraction results.
    No ground truth required — works for any company.

    Metrics computed:
      A) Extraction Coverage     — % of KPIs found vs attempted
      B) Confidence Distribution — high/medium/low/not_found counts
      C) Hallucination Rate      — % with source_page mismatch flag
      D) BM25 Agreement          — semantic+BM25 top-3 overlap (from log)
      E) Token Efficiency        — pages, reduction%, cost per KPI
      F) Hint Match Rate         — source_page within ±5 of template hint
      G) Source Type Distribution — table% vs prose%

    Args:
        extractions:    List of extraction dicts from extraction_agent.extract().
        total_pages:    Pages in source PDF before bucketing.
        pages_bucketed: Total pages used after BM25 bucketing (all themes).
        total_tokens:   Total LLM prompt+completion tokens this session.
        total_cost:     Total LLM cost USD this session.
        run_id:         Run identifier for BM25 agreement log lookup.

    Returns:
        Dict with all metric groups and a per_kpi list for CSV export.
    """
    kpi_df     = _load_kpi_template()
    hint_map:     dict[str, int | None] = {}
    template_map: dict[str, dict]       = {}

    for _, row in kpi_df.iterrows():
        kid  = str(row["kpi_id"])
        hint = row.get("source_page_hint")
        try:
            hint_map[kid] = int(hint) if pd.notna(hint) else None
        except (ValueError, TypeError):
            hint_map[kid] = None
        template_map[kid] = {
            "kpi_canonical_name": row.get("kpi_canonical_name", ""),
            "theme":              row.get("theme", ""),
        }

    total_attempted = len(extractions)

    # ── A) Extraction Coverage ───────────────────────────────────────────────
    def _is_found(e: dict) -> bool:
        v = e.get("value")
        return v not in _NULL_VALUES and v is not None

    found_count = sum(1 for e in extractions if _is_found(e))
    coverage_pct = (found_count / total_attempted * 100) if total_attempted else 0.0

    by_theme: dict[str, dict] = {}
    for e in extractions:
        th = e.get("kpi_theme") or e.get("theme", "unknown")
        by_theme.setdefault(th, {"found": 0, "attempted": 0})
        by_theme[th]["attempted"] += 1
        if _is_found(e):
            by_theme[th]["found"] += 1
    for th, d in by_theme.items():
        d["coverage_pct"] = round(d["found"] / d["attempted"] * 100, 1) if d["attempted"] else 0.0

    # ── B) Confidence Distribution ───────────────────────────────────────────
    conf_dist: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "not_found": 0}
    for e in extractions:
        c = str(e.get("confidence") or "not_found").lower().strip()
        conf_dist[c] = conf_dist.get(c, 0) + 1

    # ── C) Hallucination Flag Rate ───────────────────────────────────────────
    halluc_count = sum(1 for e in extractions if e.get("hallucination_flag"))
    halluc_rate  = (halluc_count / total_attempted * 100) if total_attempted else 0.0

    # ── D) BM25 vs Semantic Agreement ────────────────────────────────────────
    # Source: ai_metrics.jsonl entries for this run (or recent session proxy)
    bm25_agreement = 0.0
    bm25_checked   = 0
    if _AI_METRICS_PATH.exists():
        try:
            raw_lines = _AI_METRICS_PATH.read_text(encoding="utf-8").splitlines()
            all_entries = [json.loads(l) for l in raw_lines if l.strip()]
            # Filter to this run if run_id provided, else use last 200 entries as session proxy
            if run_id:
                entries = [e for e in all_entries if e.get("run_id") == run_id]
            else:
                entries = all_entries[-200:]
            agreement_hits = 0
            for entry in entries:
                sr = entry.get("semantic_rank")
                br = entry.get("bm25_rank")
                if sr is not None and br is not None and sr > 0 and br > 0:
                    bm25_checked += 1
                    if sr <= 3 and br <= 3:
                        agreement_hits += 1
            if bm25_checked:
                bm25_agreement = agreement_hits / bm25_checked * 100
        except Exception:
            pass

    # ── E) Token Efficiency ──────────────────────────────────────────────────
    if total_pages and pages_bucketed:
        token_reduction_pct = round((1 - pages_bucketed / total_pages) * 100, 1)
    else:
        token_reduction_pct = 0.0

    cost_per_kpi = (total_cost / found_count) if found_count else 0.0

    # ── F) Retrieval Quality — page hint match ───────────────────────────────
    hints_checked = 0
    hint_matches  = 0
    for e in extractions:
        kid  = str(e.get("kpi_id") or "")
        hint = hint_map.get(kid)
        sp   = e.get("source_page")
        if hint is None or sp is None:
            continue
        hints_checked += 1
        try:
            if abs(int(sp) - int(hint)) <= 5:
                hint_matches += 1
        except (TypeError, ValueError):
            pass
    hint_match_rate = (hint_matches / hints_checked * 100) if hints_checked else 0.0

    # ── G) Source Type Distribution ──────────────────────────────────────────
    table_count = sum(1 for e in extractions if e.get("source_type") == "table")
    prose_count = sum(1 for e in extractions if e.get("source_type") == "prose")
    unk_count   = total_attempted - table_count - prose_count
    table_pct   = round(table_count / total_attempted * 100, 1) if total_attempted else 0.0
    prose_pct   = round(prose_count / total_attempted * 100, 1) if total_attempted else 0.0
    unk_pct     = round(unk_count   / total_attempted * 100, 1) if total_attempted else 0.0

    # ── Per-KPI rows (for CSV export) ────────────────────────────────────────
    per_kpi_rows: list[dict] = []
    for e in extractions:
        kid  = str(e.get("kpi_id") or "")
        hint = hint_map.get(kid)
        sp   = e.get("source_page")
        hint_match = False
        if hint is not None and sp is not None:
            try:
                hint_match = abs(int(sp) - int(hint)) <= 5
            except (TypeError, ValueError):
                pass
        val   = e.get("value")
        found = val not in _NULL_VALUES and val is not None
        per_kpi_rows.append({
            "kpi_id":             kid,
            "kpi_canonical_name": (
                template_map.get(kid, {}).get("kpi_canonical_name")
                or e.get("kpi_canonical_name", "")
                or e.get("metric", "")
            ),
            "theme":              e.get("kpi_theme") or e.get("theme", ""),
            "value_extracted":    "" if val is None else str(val),
            "confidence":         e.get("confidence") or "not_found",
            "source_page":        "" if sp is None else str(sp),
            "source_type":        e.get("source_type") or "",
            "hallucination_flag": int(bool(e.get("hallucination_flag"))),
            "hint_match":         int(hint_match),
            "precision_at_1":     int(found),
        })

    return {
        "track":   "heuristic",
        "coverage": {
            "overall_pct": round(coverage_pct, 1),
            "found":       found_count,
            "attempted":   total_attempted,
            "by_theme":    by_theme,
        },
        "confidence_dist": conf_dist,
        "hallucination_rate":  round(halluc_rate, 1),
        "hallucination_count": halluc_count,
        "bm25_agreement":      round(bm25_agreement, 1),
        "bm25_checked":        bm25_checked,
        "token_efficiency": {
            "pages_before":        total_pages,
            "pages_after":         pages_bucketed,
            "token_reduction_pct": token_reduction_pct,
            "total_tokens":        total_tokens,
            "total_cost":          round(total_cost, 6),
            "cost_per_kpi":        round(cost_per_kpi, 6),
            "cost_per_profile":    round(total_cost, 6),
        },
        "hint_match_rate": round(hint_match_rate, 1),
        "hints_checked":   hints_checked,
        "source_type_dist": {
            "table_pct":   table_pct,
            "prose_pct":   prose_pct,
            "unknown_pct": unk_pct,
        },
        "per_kpi": per_kpi_rows,
    }


def build_eval_csv(per_kpi_rows: list[dict]) -> str:
    """
    Build a per-KPI evaluation CSV string from run_eval_heuristic per_kpi output.

    Args:
        per_kpi_rows: List of per-KPI dicts from run_eval_heuristic()["per_kpi"].

    Returns:
        CSV-formatted string, UTF-8.
    """
    fieldnames = [
        "kpi_id", "kpi_canonical_name", "theme", "value_extracted",
        "confidence", "source_page", "source_type",
        "hallucination_flag", "hint_match", "precision_at_1",
    ]
    if not per_kpi_rows:
        return ",".join(fieldnames) + "\n"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(per_kpi_rows)
    return output.getvalue()


def has_ground_truth(company: str, year: int) -> bool:
    """
    Check whether ground_truth.csv has rows for the given company and year.

    Args:
        company: Company name string.
        year:    Reporting year integer.

    Returns:
        True if at least one matching row exists.
    """
    try:
        gt_df = pd.read_csv(_GT_PATH)
        return not gt_df[(gt_df["company"] == company) & (gt_df["report_year"] == year)].empty
    except Exception:
        return False
