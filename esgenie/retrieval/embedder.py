"""
embedder.py
OpenAI text-embedding-3-small embeddings + FAISS IndexFlatIP build/load/search.
Caches index and chunk metadata to faiss_cache/ for reuse across runs.
"""

import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from utils.token_tracker import track_embedding

load_dotenv()

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536
_CACHE_DIR = Path(__file__).parent.parent / "faiss_cache"


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed texts in batches, track cost, return (N, 1536) float32 array."""
    client = _client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=_EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        track_embedding(
            model=_EMBED_MODEL,
            tokens_used=response.usage.total_tokens,
            num_texts=len(batch),
        )

    return np.array(all_embeddings, dtype=np.float32)


def build_index(chunks: list[dict], cache_prefix: str = "default") -> None:
    """
    Embed all chunk texts, build a FAISS IndexFlatIP, and cache to disk.

    Args:
        chunks:       List of chunk dicts with at least a "text" key.
        cache_prefix: Filename prefix for faiss_cache/{prefix}_index.faiss
                      and faiss_cache/{prefix}_chunks.pkl.
    """
    _CACHE_DIR.mkdir(exist_ok=True)

    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = _embed_texts(texts)

    # L2-normalize → inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(embeddings)

    faiss.write_index(index, str(_CACHE_DIR / f"{cache_prefix}_index.faiss"))
    with open(_CACHE_DIR / f"{cache_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    elapsed = time.time() - t0
    # Cost is already accumulated in token_tracker; read last embedding entry
    from utils.token_tracker import get_session_summary
    summary = get_session_summary()
    print(
        f"Embedded {len(chunks)} chunks. "
        f"Cost: ${summary['total_cost_usd']:.4f}. "
        f"Time: {elapsed:.1f}s"
    )


def load_index(cache_prefix: str = "default") -> tuple[faiss.Index, list[dict]]:
    """
    Load FAISS index and chunk metadata from faiss_cache/.

    Args:
        cache_prefix: Prefix used when build_index() was called.

    Returns:
        (faiss_index, chunks)

    Raises:
        FileNotFoundError: If cache files are missing — run build_index() first.
    """
    index_path = _CACHE_DIR / f"{cache_prefix}_index.faiss"
    chunks_path = _CACHE_DIR / f"{cache_prefix}_chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No cached index found for prefix '{cache_prefix}' in {_CACHE_DIR}. "
            "Run build_index() first."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(
    query: str,
    index: faiss.Index,
    chunks: list[dict],
    k: int = 10,
    theme_filter: str | None = None,
) -> list[dict]:
    """
    Embed a query and return the top-k most similar chunks.

    Args:
        query:        Search string.
        index:        FAISS index from build_index() / load_index().
        chunks:       Chunk metadata list aligned with the index.
        k:            Number of results to return.
        theme_filter: If provided, only return chunks matching this theme.

    Returns:
        List of dicts: {"chunk": chunk_dict, "score": float, "rank": int}
    """
    q_vec = _embed_texts([query])
    faiss.normalize_L2(q_vec)

    # Fetch k*2 candidates to allow for theme filtering headroom
    scores, indices = index.search(q_vec, k * 2)

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if theme_filter and chunk.get("theme") != theme_filter:
            continue
        results.append({"chunk": chunk, "score": float(score), "rank": len(results) + 1})
        if len(results) >= k:
            break

    return results
