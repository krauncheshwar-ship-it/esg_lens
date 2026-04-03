"""
thematic_bucketer.py
Scores each PDF page against ESG themes using BM25.
Returns a mapping of page_num -> {theme: score}.
"""

import yaml
from pathlib import Path
from rank_bm25 import BM25Okapi


_THEMES_PATH = Path(__file__).parent.parent / "config" / "esg_themes.yaml"


def load_themes(themes_path: str | Path = _THEMES_PATH) -> dict[str, list[str]]:
    """Load theme keyword lists from YAML config."""
    with open(themes_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {theme: meta["keywords"] for theme, meta in data["themes"].items()}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def score_pages(
    pages: list[dict],
    themes_path: str | Path = _THEMES_PATH,
) -> list[dict]:
    """
    Score each page against every ESG theme using BM25.

    Args:
        pages: Output from pdf_parser.extract_pages().
        themes_path: Path to esg_themes.yaml.

    Returns:
        Same pages list with an added "theme_scores" key per page:
            {"environment": float, "social": float, "governance": float}
        and a "top_theme" key for the highest-scoring theme.
    """
    themes = load_themes(themes_path)
    corpus = [_tokenize(p["text"]) for p in pages]

    if not corpus:
        return pages

    bm25 = BM25Okapi(corpus)

    for page in pages:
        scores = {}
        for theme, keywords in themes.items():
            query = _tokenize(" ".join(keywords))
            page_scores = bm25.get_scores(query)
            page_idx = page["page_num"] - 1
            scores[theme] = float(page_scores[page_idx])
        page["theme_scores"] = scores
        page["top_theme"] = max(scores, key=scores.get)

    return pages


def bucket_pages_by_theme(
    scored_pages: list[dict],
    threshold: float = 0.0,
) -> dict[str, list[dict]]:
    """
    Group pages by their top_theme.

    Args:
        scored_pages: Output of score_pages().
        threshold: Minimum BM25 score to include a page in a bucket.

    Returns:
        Dict mapping theme name -> list of page dicts.
    """
    buckets: dict[str, list[dict]] = {}
    for page in scored_pages:
        theme = page.get("top_theme", "unknown")
        score = page.get("theme_scores", {}).get(theme, 0.0)
        if score >= threshold:
            buckets.setdefault(theme, []).append(page)
    return buckets
