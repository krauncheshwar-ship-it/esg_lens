"""
bm25_index.py
BM25Okapi index builder over chunk texts with disk caching.
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

_CACHE_DIR = Path(__file__).parent.parent / "faiss_cache"


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer."""
    return text.lower().split()


def build_bm25(chunks: list[dict], cache_prefix: str = "default") -> BM25Okapi:
    """
    Build a BM25Okapi index from chunk texts and cache to disk.

    Args:
        chunks:       List of chunk dicts with at least a "text" key.
        cache_prefix: Filename prefix for faiss_cache/{prefix}_bm25.pkl.

    Returns:
        The built BM25Okapi index.
    """
    _CACHE_DIR.mkdir(exist_ok=True)

    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    cache_path = _CACHE_DIR / f"{cache_prefix}_bm25.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Built BM25 index on {len(chunks)} chunks")
    return bm25


def load_bm25(cache_prefix: str = "default") -> tuple[BM25Okapi, list[dict]]:
    """
    Load a cached BM25 index and its associated chunks from disk.

    Args:
        cache_prefix: Prefix used when build_bm25() was called.

    Returns:
        (bm25_index, chunks)

    Raises:
        FileNotFoundError: If cache file is missing — run build_bm25() first.
    """
    cache_path = _CACHE_DIR / f"{cache_prefix}_bm25.pkl"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached BM25 index found for prefix '{cache_prefix}' in {_CACHE_DIR}. "
            "Run build_bm25() first."
        )

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    return data["bm25"], data["chunks"]


def search_bm25(
    query: str,
    bm25_index: BM25Okapi,
    chunks: list[dict],
    k: int = 10,
    theme_filter: str | None = None,
) -> list[dict]:
    """
    Search the BM25 index and return top-k results.

    Args:
        query:        Search string.
        bm25_index:   BM25Okapi index from build_bm25() / load_bm25().
        chunks:       Chunk metadata list aligned with the index.
        k:            Number of results to return.
        theme_filter: If provided, only return chunks matching this theme.

    Returns:
        List of dicts: {"chunk": chunk_dict, "score": float, "rank": int}
    """
    tokens = _tokenize(query)
    scores = bm25_index.get_scores(tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results: list[dict] = []
    for idx, score in ranked:
        chunk = chunks[idx]
        if theme_filter and chunk.get("theme") != theme_filter:
            continue
        results.append({"chunk": chunk, "score": float(score), "rank": len(results) + 1})
        if len(results) >= k:
            break

    return results
