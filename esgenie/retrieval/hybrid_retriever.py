"""
hybrid_retriever.py
Reciprocal Rank Fusion (RRF) merge of FAISS dense and BM25 sparse results.
Achieves better recall than either retriever alone with no LLM cost.
"""

from retrieval.embedder import search as semantic_search
from retrieval.bm25_index import search_bm25


def hybrid_search(
    query: str,
    faiss_index,
    chunks: list[dict],
    bm25_index,
    k: int = 5,
    theme_filter: str | None = None,
    rrf_k: int = 60,
    bm25_disabled: bool = False,
) -> list[dict]:
    """
    Retrieve top-k chunks via RRF merge of semantic and BM25 results.

    Args:
        query:         Search string.
        faiss_index:   FAISS index from embedder.build_index().
        chunks:        Chunk metadata list aligned with the FAISS index.
        bm25_index:    BM25Okapi index from bm25_index.build_bm25().
        k:             Number of results to return after merge.
        theme_filter:  If provided, restrict results to this theme.
        rrf_k:         RRF constant (default 60, standard value).
        bm25_disabled: If True, skip BM25 (demo comparison mode).

    Returns:
        List of dicts: {
            "chunk":         chunk_dict,
            "rrf_score":     float,
            "semantic_rank": int | None,
            "bm25_rank":     int | None,
        }
    """
    # --- Fetch candidates from each retriever ---
    semantic_results = semantic_search(
        query, faiss_index, chunks, k=10, theme_filter=None
    )

    bm25_results = [] if bm25_disabled else search_bm25(
        query, bm25_index, chunks, k=10, theme_filter=None
    )

    # --- Build lookup: chunk_id → rank (1-based) ---
    sem_rank_map: dict[str, int] = {
        r["chunk"]["chunk_id"]: r["rank"] for r in semantic_results
    }
    bm25_rank_map: dict[str, int] = {
        r["chunk"]["chunk_id"]: r["rank"] for r in bm25_results
    }

    # Union of all candidate chunk_ids
    all_ids: set[str] = set(sem_rank_map) | set(bm25_rank_map)

    # Build chunk lookup from both result sets
    chunk_lookup: dict[str, dict] = {}
    for r in semantic_results:
        chunk_lookup[r["chunk"]["chunk_id"]] = r["chunk"]
    for r in bm25_results:
        chunk_lookup[r["chunk"]["chunk_id"]] = r["chunk"]

    # --- RRF scoring ---
    rrf_entries: list[dict] = []
    for cid in all_ids:
        sem_rank = sem_rank_map.get(cid)
        bm25_rank = bm25_rank_map.get(cid)

        rrf_score = 0.0
        if sem_rank is not None:
            rrf_score += 1.0 / (sem_rank + rrf_k)
        if bm25_rank is not None:
            rrf_score += 1.0 / (bm25_rank + rrf_k)

        rrf_entries.append(
            {
                "chunk": chunk_lookup[cid],
                "rrf_score": rrf_score,
                "semantic_rank": sem_rank,
                "bm25_rank": bm25_rank,
            }
        )

    # Sort by RRF score descending
    rrf_entries.sort(key=lambda x: x["rrf_score"], reverse=True)

    # --- Apply theme filter after merge ---
    if theme_filter:
        rrf_entries = [
            e for e in rrf_entries if e["chunk"].get("theme") == theme_filter
        ]

    top = rrf_entries[:k]

    # --- Print log ---
    sem_count = len(semantic_results)
    bm25_count = len(bm25_results)
    print(f"Hybrid retrieval: {sem_count} semantic + {bm25_count} BM25 -> {len(top)} merged")
    if top:
        best = top[0]
        print(
            f"Top chunk: page {best['chunk']['page_number']}, "
            f"RRF score {best['rrf_score']:.3f}"
        )

    return top
