# ingestion.py
# OUR IMPLEMENTATION -- dispatches by file extension to format-specific
# readers. Every reader returns plain text, so nothing downstream needs
# to know which format a file came from.

from pathlib import Path

import pymupdf  # PyMuPDF -- https://pymupdf.readthedocs.io


def ingest_text(path: Path) -> str:
    """Read a plain text or Markdown file as-is."""
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_pdf(path: Path) -> str:
    """Extract plain text from a searchable PDF, page by page.
    Source: https://pymupdf.readthedocs.io (Text Extraction section) --
    page.get_text() extracts a page's text in original reading order.
    Note: this only works on searchable PDFs (real text), not scanned
    image-only PDFs -- those need OCR instead, which is a different
    problem (see assignment's "why OCR differs from text extraction").
    """
    doc = pymupdf.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_document(path: str) -> str:
    """Read any supported file and return its text content.

    Currently supported: .txt, .md, .pdf
    (Word and images are added in the next steps.)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if suffix in (".txt", ".md"):
        return ingest_text(file_path)
    if suffix == ".pdf":
        return ingest_pdf(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")