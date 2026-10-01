from pathlib import Path

import pymupdf  # PyMuPDF -- https://pymupdf.readthedocs.io
from docx import Document  # python-docx -- https://python-docx.readthedocs.io


def ingest_text(path: Path) -> str:
    """Read a plain text or Markdown file as-is."""
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_pdf(path: Path) -> str:
    """Extract plain text from a searchable PDF, page by page.
    Source: https://pymupdf.readthedocs.io (Text Extraction section) --
    page.get_text() extracts a page's text in original reading order.
    """
    doc = pymupdf.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_docx(path: Path) -> str:
    """Extract plain text from a Word document, paragraph by paragraph.
    Source: https://python-docx.readthedocs.io (Quickstart) --
    Document(path) opens an existing .docx file; each item in
    document.paragraphs has a .text property.
    """
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def ingest_document(path: str) -> str:
    """Read any supported file and return its text content.

    Currently supported: .txt, .md, .pdf, .docx
    (images are added in the next step.)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if suffix in (".txt", ".md"):
        return ingest_text(file_path)
    if suffix == ".pdf":
        return ingest_pdf(file_path)
    if suffix == ".docx":
        return ingest_docx(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")