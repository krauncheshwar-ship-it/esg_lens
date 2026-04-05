"""
query_router.py
Deterministic ESG theme classifier — NO LLM calls.
Routes queries by keyword matching against YAML page_bucketing_keywords.

Rationale (video line 2): "The router is rules-based, not LLM-based, because
routing by theme is a deterministic operation — using an LLM here would add
cost and latency for zero benefit."
"""

import yaml
from pathlib import Path

_YAML_PATH = Path(__file__).parent.parent / "config" / "esg_themes_v3_FINAL.yaml"

# Priority order for tie-breaking
_THEME_PRIORITY = ["governance", "climate", "employee", "water", "biodiversity", "waste"]

# Lazy-loaded keyword map
_KEYWORD_MAP: dict[str, str] | None = None


def _load_keyword_map() -> dict[str, str]:
    """Build flat {keyword_lower: theme} map from YAML page_bucketing_keywords."""
    global _KEYWORD_MAP
    if _KEYWORD_MAP is not None:
        return _KEYWORD_MAP

    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mapping: dict[str, str] = {}
    for theme, meta in data["themes"].items():
        for kw in meta.get("page_bucketing_keywords", []):
            mapping[str(kw).lower()] = theme

    _KEYWORD_MAP = mapping
    return _KEYWORD_MAP


def _detect_query_type(query_lower: str) -> str:
    """Classify query intent from surface-level keywords."""
    if any(w in query_lower for w in ("full", "all", "profile", "company")):
        return "full_profile"
    if any(w in query_lower for w in ("does", "is there", "do they", "have")):
        return "boolean"
    if any(w in query_lower for w in ("progress", "target", "achievement", "plan")):
        return "subjective"
    return "single_metric"


def route_query(query: str) -> dict:
    """
    Route a query to the most relevant ESG theme via deterministic keyword matching.

    Args:
        query: Raw user question.

    Returns:
        {
            "theme":            str,       # winning theme
            "confidence":       str,       # "high" | "medium" | "low"
            "matched_keywords": list[str], # keywords that matched
            "query_type":       str,       # "single_metric" | "full_profile" | "boolean" | "subjective"
        }
    """
    keyword_map = _load_keyword_map()
    query_lower = query.lower()

    # Count matches per theme and collect matched keywords
    theme_hits: dict[str, list[str]] = {}
    for kw, theme in keyword_map.items():
        if kw in query_lower:
            theme_hits.setdefault(theme, []).append(kw)

    if not theme_hits:
        print(f"Routed to theme 'climate' (confidence: low, matched: 0 keywords)")
        return {
            "theme": "climate",
            "confidence": "low",
            "matched_keywords": [],
            "query_type": _detect_query_type(query_lower),
        }

    # Sort themes by hit count, then by priority order for ties
    best_theme = max(
        theme_hits,
        key=lambda t: (len(theme_hits[t]), -_THEME_PRIORITY.index(t) if t in _THEME_PRIORITY else -99),
    )

    matched = theme_hits[best_theme]
    n = len(matched)
    confidence = "high" if n >= 3 else "medium" if n >= 1 else "low"

    print(f"Routed to theme '{best_theme}' (confidence: {confidence}, matched: {n} keywords)")

    return {
        "theme": best_theme,
        "confidence": confidence,
        "matched_keywords": matched,
        "query_type": _detect_query_type(query_lower),
    }
