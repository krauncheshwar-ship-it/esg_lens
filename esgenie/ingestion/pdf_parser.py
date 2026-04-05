"""
pdf_parser.py
Extracts page-level text from PDF files using PyMuPDF (fitz).
"""

import re
from pathlib import Path

import fitz  # PyMuPDF


def extract_pages(pdf_path: str | Path) -> dict[int, str]:
    """
    Extract text from every page of a PDF, returning a 1-indexed dict.

    Args:
        pdf_path: Path to the PDF file (str or Path).

    Returns:
        Dict mapping page number (1-indexed) to cleaned page text.

    Raises:
        ValueError: If the file does not exist or is not a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    pages: dict[int, str] = {}

    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            # Collapse 3+ newlines → double newline, strip edges
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages[i + 1] = text  # 1-indexed

    print(f"Extracted {len(pages)} pages from {path.name}")
    return pages


def get_page_count(pdf_path: str | Path) -> int:
    """
    Return the total number of pages in a PDF without extracting text.

    Args:
        pdf_path: Path to the PDF file (str or Path).

    Returns:
        Integer page count.

    Raises:
        ValueError: If the file does not exist or is not a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    with fitz.open(str(path)) as doc:
        return doc.page_count


def get_page_range(pdf_path: str | Path, start: int, end: int) -> dict[int, str]:
    """
    Extract text from a subset of pages (1-indexed, inclusive).

    Args:
        pdf_path: Path to the PDF file (str or Path).
        start:    First page to extract (1-indexed).
        end:      Last page to extract (1-indexed, inclusive).

    Returns:
        Dict mapping page number (1-indexed) to cleaned page text.

    Raises:
        ValueError: If the file does not exist, is not a PDF, or range is invalid.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise ValueError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    pages: dict[int, str] = {}

    with fitz.open(str(path)) as doc:
        total = doc.page_count
        if start < 1 or end > total or start > end:
            raise ValueError(
                f"Invalid range [{start}, {end}] for {total}-page document."
            )
        for i in range(start - 1, end):  # fitz is 0-indexed
            text = doc[i].get_text()
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            pages[i + 1] = text

    return pages
