"""
hybrid_retriever.py
Reciprocal Rank Fusion (RRF) merge of FAISS (dense) and BM25 (sparse) results.
"""

from retrieval.embedder import search as faiss_search
from retrieval.bm25_index import BM25Index


_RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """RRF score for a result at 1-based `rank`."""
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        dense_results:  Ranked list from FAISS (chunk dicts with "faiss_score").
        sparse_results: Ranked list from BM25  (chunk dicts with "bm25_score").
        dense_weight:   Multiplier applied to RRF scores from the dense list.
        sparse_weight:  Multiplier applied to RRF scores from the sparse list.

    Returns:
        Merged list of chunk dicts sorted by combined RRF score (descending),
        with an added "rrf_score" key.
    """
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + dense_weight * _rrf_score(rank)
        meta[cid] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + sparse_weight * _rrf_score(rank)
        meta.setdefault(cid, chunk)

    merged = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        chunk = dict(meta[cid])
        chunk["rrf_score"] = rrf_score
        merged.append(chunk)

    return merged


class HybridRetriever:
    """
    Combines FAISS dense retrieval and BM25 sparse retrieval via RRF.
    """

    def __init__(
        self,
        faiss_index,
        chunks: list[dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        """
        Args:
            faiss_index:   Built faiss.IndexFlatIP from embedder.build_faiss_index.
            chunks:        Aligned chunk metadata list.
            dense_weight:  RRF weight for the dense retriever.
            sparse_weight: RRF weight for the BM25 retriever.
        """
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.bm25_index = BM25Index(chunks)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fetch_k: int = 30,
    ) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Args:
            query:  User query string.
            top_k:  Number of results to return after RRF merge.
            fetch_k: Number of candidates to fetch from each retriever before merge.

        Returns:
            List of up to top_k chunk dicts sorted by rrf_score descending.
        """
        dense = faiss_search(self.faiss_index, self.chunks, query, top_k=fetch_k)
        sparse = self.bm25_index.search(query, top_k=fetch_k)

        merged = reciprocal_rank_fusion(
            dense,
            sparse,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
        )
        return merged[:top_k]
