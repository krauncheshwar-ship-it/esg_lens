"""
embedder.py
OpenAI text-embedding-3-small embeddings + FAISS index build/load/save.
"""

import os
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from utils.token_tracker import track_embedding

_CACHE_DIR = Path(__file__).parent.parent / "faiss_cache"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of strings using OpenAI text-embedding-3-small.

    Returns:
        numpy array of shape (len(texts), _EMBED_DIM), dtype float32.
    """
    client = _get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=_EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # Track token usage
        track_embedding(
            model=_EMBED_MODEL,
            tokens_used=response.usage.total_tokens,
            num_texts=len(batch),
        )

    return np.array(all_embeddings, dtype=np.float32)


def build_faiss_index(
    chunks: list[dict],
    index_name: str,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build a FAISS inner-product index from chunk texts.
    Normalizes embeddings so inner product == cosine similarity.

    Args:
        chunks:     List of chunk dicts with at least a "text" key.
        index_name: Identifier used for cache filenames.

    Returns:
        (faiss_index, chunks)  — chunks list is returned for metadata alignment.
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(embeddings)

    _save_index(index, chunks, index_name)
    return index, chunks


def load_faiss_index(index_name: str) -> tuple[faiss.IndexFlatIP, list[dict]] | None:
    """
    Load a previously saved FAISS index and its associated chunk metadata.

    Returns None if no cached index exists for index_name.
    """
    index_path = _CACHE_DIR / f"{index_name}.faiss"
    meta_path = _CACHE_DIR / f"{index_name}_chunks.pkl"

    if not index_path.exists() or not meta_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def _save_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    index_name: str,
) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(_CACHE_DIR / f"{index_name}.faiss"))
    with open(_CACHE_DIR / f"{index_name}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def search(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Embed a query and return the top_k most similar chunks.

    Returns:
        List of chunk dicts with an added "faiss_score" key (cosine similarity).
    """
    q_vec = embed_texts([query])
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["faiss_score"] = float(score)
        results.append(chunk)

    return results
