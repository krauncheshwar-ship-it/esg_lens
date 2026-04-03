"""
pdf_parser.py
Page-by-page PDF extraction using PyMuPDF (fitz).
Returns a list of dicts: {page_num, text, char_count}.
"""

import fitz  # PyMuPDF


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Open a PDF and extract text from every page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of page dicts:
            {
                "page_num": int,       # 1-based
                "text": str,           # raw extracted text
                "char_count": int,
                "source": str          # basename of the PDF
            }
    """
    import os

    source = os.path.basename(pdf_path)
    pages = []

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text("text")  # plain text, preserves whitespace
        pages.append(
            {
                "page_num": i + 1,
                "text": text,
                "char_count": len(text),
                "source": source,
            }
        )
    doc.close()

    return pages


def extract_pages_with_blocks(pdf_path: str) -> list[dict]:
    """
    Richer extraction that also returns bounding-box blocks per page.
    Useful for table detection and layout-aware chunking.
    """
    import os

    source = os.path.basename(pdf_path)
    pages = []

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,block_no,block_type)
        text = "\n".join(b[4] for b in blocks if b[6] == 0)  # type 0 = text block
        pages.append(
            {
                "page_num": i + 1,
                "text": text,
                "blocks": blocks,
                "char_count": len(text),
                "source": source,
            }
        )
    doc.close()

    return pages
