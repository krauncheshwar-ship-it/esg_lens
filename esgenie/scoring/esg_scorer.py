"""
esg_scorer.py
Pure Python ESG scoring — ZERO LLM calls. Auditable rules-based system.
Implements: exposure_relevance, management_effectiveness, controversy deductions.
"""

import yaml
import pandas as pd
from pathlib import Path

_BASE = Path(__file__).parent.parent
_MATERIALITY_PATH = _BASE / "config" / "sector_materiality.yaml"
_KPI_TEMPLATE_PATH = _BASE / "config" / "kpi_template_v8_FINAL.csv"

CONTROVERSY_DEDUCTIONS: dict[str, int] = {
    "NONE": 0,
    "MILD": -3,
    "MODERATE": -8,
    "SEVERE": -15,
}

_CONF_MAP: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "not_found": 0.0,
}

# Lazy-loaded globals
_MATERIALITY: dict | None = None
_KPI_DF: pd.DataFrame | None = None


def _load_materiality() -> dict:
    global _MATERIALITY
    if _MATERIALITY is None:
        with open(_MATERIALITY_PATH, "r", encoding="utf-8") as f:
            _MATERIALITY = yaml.safe_load(f)["sectors"]
    return _MATERIALITY


def _load_kpi_df() -> pd.DataFrame:
    global _KPI_DF
    if _KPI_DF is None:
        _KPI_DF = pd.read_csv(_KPI_TEMPLATE_PATH)
    return _KPI_DF


def get_rating(score: float) -> tuple[str, str]:
    """
    Map a numeric score to an ESG rating band.

    Args:
        score: 0-100 ESG score.

    Returns:
        (rating, interpretation) e.g. ("AA", "Leader")
    """
    if score >= 85:
        return "AAA", "Leader"
    elif score >= 75:
        return "AA", "Leader"
    elif score >= 60:
        return "A", "Above Average"
    elif score >= 45:
        return "BBB", "Average"
    elif score >= 30:
        return "BB", "Below Average"
    elif score >= 15:
        return "B", "Laggard"
    else:
        return "CCC", "Severe Laggard"


def score_company(
    extractions: list[dict],
    company: str,
    sector: str = "apparel_consumer",
    theme_weights: dict | None = None,
) -> dict:
    """
    Score a company across all ESG themes using pure Python rules.

    Args:
        extractions:   List of extraction dicts from extraction_agent.extract().
                       Each must have: kpi_id, kpi_theme, confidence, and optionally
                       value / value_subjective (for controversy KPIs).
        company:       Company name (for logging).
        sector:        Sector key from sector_materiality.yaml.
        theme_weights: Optional override dict {theme: weight} that sums to 1.0.
                       When provided, replaces sector_materiality.yaml weights.

    Returns:
        Scoring result dict with overall_score, overall_rating, theme_scores,
        controversy_total_deduction, and sector.
    """
    materiality = _load_materiality()
    kpi_df = _load_kpi_df()
    if theme_weights:
        sector_weights: dict[str, float] = theme_weights
    else:
        sector_weights = materiality.get(sector, materiality["apparel_consumer"])

    # Index extractions by kpi_id for fast lookup
    ext_by_id: dict[str, dict] = {e["kpi_id"]: e for e in extractions}

    # Resolve controversy deductions from CONT001 / CONT002
    def _get_cont_deduction(kpi_id: str) -> int:
        ext = ext_by_id.get(kpi_id, {})
        severity = (
            ext.get("value_subjective")
            or ext.get("value")
            or "NONE"
        )
        severity = str(severity).strip().upper()
        return CONTROVERSY_DEDUCTIONS.get(severity, 0)

    cont_deduction_total = _get_cont_deduction("CONT001") + _get_cont_deduction("CONT002")

    theme_scores: dict[str, dict] = {}

    for theme in sector_weights:
        # Expected KPIs for this theme (exclude controversy KPIs)
        theme_kpis = kpi_df[
            (kpi_df["theme"] == theme) &
            (~kpi_df["kpi_id"].str.startswith("CONT"))
        ]["kpi_id"].tolist()

        expected_count = len(theme_kpis)
        if expected_count == 0:
            continue

        # Count found KPIs and accumulate confidence scores
        found_count = 0
        conf_scores: list[float] = []

        for kpi_id in theme_kpis:
            ext = ext_by_id.get(kpi_id)
            if ext is not None and ext.get("value") not in (None, "", "null"):
                found_count += 1
                conf = ext.get("confidence", "not_found")
            else:
                conf = "not_found"
            conf_scores.append(_CONF_MAP.get(str(conf).lower(), 0.0))

        completeness = found_count / expected_count
        avg_confidence = sum(conf_scores) / len(conf_scores)
        management_effectiveness = (completeness * 0.5) + (avg_confidence * 0.5)

        exposure_relevance = sector_weights[theme]
        base_score = (0.40 * exposure_relevance * 100) + (0.60 * management_effectiveness * 100)

        # Per-theme controversy (applied at overall level per formula, zero here)
        theme_controversy = 0
        final_score = max(0.0, min(100.0, base_score - theme_controversy))
        rating, interpretation = get_rating(final_score)

        theme_scores[theme] = {
            "score": round(final_score, 2),
            "rating": rating,
            "interpretation": interpretation,
            "completeness": round(completeness, 4),
            "avg_confidence": round(avg_confidence, 4),
            "management_effectiveness": round(management_effectiveness, 4),
            "exposure_relevance": exposure_relevance,
            "controversy_deduction": theme_controversy,
        }

    # Overall score = weighted sum of theme scores
    overall_base = sum(
        theme_scores[t]["score"] * sector_weights[t]
        for t in theme_scores
        if t in sector_weights
    )
    overall_score = max(0.0, min(100.0, overall_base + cont_deduction_total))
    overall_rating, overall_interpretation = get_rating(overall_score)

    print(
        f"[ESG SCORER] {company} | sector: {sector} | "
        f"overall: {overall_score:.1f} ({overall_rating})"
    )

    return {
        "overall_score": round(overall_score, 2),
        "overall_rating": overall_rating,
        "overall_interpretation": overall_interpretation,
        "theme_scores": theme_scores,
        "controversy_total_deduction": cont_deduction_total,
        "sector": sector,
    }
