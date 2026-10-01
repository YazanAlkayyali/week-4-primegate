import base64
from pathlib import Path

import pymupdf  # PyMuPDF -- https://pymupdf.readthedocs.io
from docx import Document  # python-docx -- https://python-docx.readthedocs.io
from groq import Groq  # https://console.groq.com/docs/vision

from src.week_4_primegate.ai_config import API_KEY as GROQ_API_KEY


def ingest_text(path: Path) -> str:
    """Read a plain text or Markdown file as-is."""
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_pdf(path: Path) -> str:
    """Extract plain text from a searchable PDF, page by page.
    Source: https://pymupdf.readthedocs.io (Text Extraction section).
    """
    doc = pymupdf.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def ingest_docx(path: Path) -> str:
    """Extract plain text from a Word document, paragraph by paragraph.
    Source: https://python-docx.readthedocs.io (Quickstart).
    """
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def ingest_image(path: Path) -> str:
    """Describe/read an image using a Groq vision model (this is OCR /
    visual understanding, fundamentally different from the text-extraction
    readers above -- there's no embedded text layer to pull out, so a
    vision model looks at the image and reports what it sees, including
    any text in it.
    Source: https://console.groq.com/docs/vision
    ("How to Pass Locally Saved Images as Input") -- base64-encode the
    file, send it as a data: URL in the image_url field.
    """
    with open(path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail, and transcribe "
                                "any readable text exactly as it appears.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                    },
                ],
            }
        ],
        model="qwen/qwen3.8-27b",
    )
    return completion.choices[0].message.content


def ingest_document(path: str) -> str:
    """Read any supported file and return its text content.

    Supported: .txt, .md, .pdf, .docx, .png, .jpg, .jpeg
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
    if suffix in (".png", ".jpg", ".jpeg"):
        return ingest_image(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")