"""
bm25_index.py
rank_bm25 index builder over chunk texts.
"""

import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Index:
    """
    Thin wrapper around BM25Okapi that keeps chunk metadata aligned
    with the corpus and provides a simple search interface.
    """

    def __init__(self, chunks: list[dict]):
        """
        Build the BM25 index from a list of chunk dicts.

        Args:
            chunks: List of dicts, each with at least a "text" key.
        """
        self.chunks = chunks
        corpus = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return the top_k chunks ranked by BM25 score.

        Returns:
            List of chunk dicts with an added "bm25_score" key.
        """
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair each chunk with its score, sort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(score)
            results.append(chunk)

        return results
