"""
chunker.py
Paragraph-aware chunking of BM25-bucketed PDF pages.
Each chunk carries theme, page, company, and year metadata for retrieval.
"""

import re


def chunk_bucketed_pages(
    page_corpus: dict[int, str],
    thematic_page_map: dict[str, list[int]],
    company: str = "",
    report_year: int = 2024,
    min_chunk_chars: int = 80,
) -> list[dict]:
    """
    Split bucketed pages into paragraph chunks with rich metadata.

    Args:
        page_corpus:       {page_number (1-indexed): page_text}
        thematic_page_map: {theme: [page_numbers]} from ThematicBucketer.bucket()
        company:           Company name tag added to every chunk.
        report_year:       Reporting year tag added to every chunk.
        min_chunk_chars:   Minimum character length; shorter paragraphs are dropped.

    Returns:
        List of chunk dicts:
            {
                "chunk_id":   str,   # "{theme}_p{page}_c{idx}"
                "text":       str,
                "page_number": int,
                "theme":      str,
                "company":    str,
                "report_year": int,
                "char_count": int,
            }
    """
    chunks: list[dict] = []
    unique_pages: set[int] = set()

    for theme, page_numbers in thematic_page_map.items():
        for page_num in page_numbers:
            unique_pages.add(page_num)
            page_text = page_corpus.get(page_num, "")
            if not page_text:
                continue

            paragraphs = page_text.split("\n\n")
            chunk_idx = 0

            for para in paragraphs:
                # Collapse internal whitespace
                cleaned = " ".join(para.split())
                if len(cleaned) < min_chunk_chars:
                    continue

                chunks.append(
                    {
                        "chunk_id": f"{theme}_p{page_num}_c{chunk_idx}",
                        "text": cleaned,
                        "page_number": page_num,
                        "theme": theme,
                        "company": company,
                        "report_year": report_year,
                        "char_count": len(cleaned),
                    }
                )
                chunk_idx += 1

    print(
        f"Created {len(chunks)} chunks from {len(unique_pages)} unique pages "
        f"across {len(thematic_page_map)} themes"
    )
    return chunks
