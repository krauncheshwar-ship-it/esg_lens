"""
chunker.py
Paragraph-aware text chunking with rich metadata.
Each chunk carries: text, source, page_num, chunk_id, theme_scores, top_theme.
"""

import re
import hashlib


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; collapse internal whitespace within each paragraph."""
    raw = re.split(r"\n\s*\n", text)
    paragraphs = []
    for para in raw:
        cleaned = " ".join(para.split())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _chunk_id(source: str, page_num: int, idx: int) -> str:
    key = f"{source}::{page_num}::{idx}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def chunk_pages(
    pages: list[dict],
    max_chars: int = 1500,
    overlap_chars: int = 150,
) -> list[dict]:
    """
    Convert a list of page dicts into smaller, overlapping chunks.

    Strategy:
      1. Split each page into paragraphs.
      2. Greedily accumulate paragraphs until max_chars is reached.
      3. When a single paragraph exceeds max_chars, split it by sentence.
      4. Carry overlap_chars from the previous chunk into the next.

    Args:
        pages:         Output from pdf_parser (optionally scored by thematic_bucketer).
        max_chars:     Target maximum characters per chunk.
        overlap_chars: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts:
            {
                "chunk_id":     str,
                "source":       str,
                "page_num":     int,
                "text":         str,
                "char_count":   int,
                "theme_scores": dict | None,
                "top_theme":    str | None,
            }
    """
    chunks = []

    for page in pages:
        source = page.get("source", "unknown")
        page_num = page.get("page_num", 0)
        theme_scores = page.get("theme_scores")
        top_theme = page.get("top_theme")

        paragraphs = _split_paragraphs(page["text"])
        buffer = ""
        chunk_idx = 0

        for para in paragraphs:
            # If the paragraph alone exceeds max_chars, split by sentence
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    if len(buffer) + len(sent) + 1 <= max_chars:
                        buffer = (buffer + " " + sent).strip()
                    else:
                        if buffer:
                            chunks.append(
                                _make_chunk(
                                    buffer, source, page_num, chunk_idx,
                                    theme_scores, top_theme
                                )
                            )
                            chunk_idx += 1
                            buffer = buffer[-overlap_chars:] + " " + sent
                            buffer = buffer.strip()
                        else:
                            buffer = sent
            else:
                if len(buffer) + len(para) + 1 <= max_chars:
                    buffer = (buffer + " " + para).strip()
                else:
                    if buffer:
                        chunks.append(
                            _make_chunk(
                                buffer, source, page_num, chunk_idx,
                                theme_scores, top_theme
                            )
                        )
                        chunk_idx += 1
                        buffer = buffer[-overlap_chars:] + " " + para
                        buffer = buffer.strip()
                    else:
                        buffer = para

        # Flush remaining buffer
        if buffer:
            chunks.append(
                _make_chunk(
                    buffer, source, page_num, chunk_idx,
                    theme_scores, top_theme
                )
            )

    return chunks


def _make_chunk(
    text: str,
    source: str,
    page_num: int,
    chunk_idx: int,
    theme_scores: dict | None,
    top_theme: str | None,
) -> dict:
    return {
        "chunk_id": _chunk_id(source, page_num, chunk_idx),
        "source": source,
        "page_num": page_num,
        "text": text,
        "char_count": len(text),
        "theme_scores": theme_scores,
        "top_theme": top_theme,
    }
