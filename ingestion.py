    # From https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text
    # Chatgpt explination of the the very hard to read document

from pathlib import Path


def ingest_text(path: Path) -> str:
    """Read a plain text or Markdown file as-is."""
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_document(path: str) -> str:
    """Read any supported file and return its text content.

    Currently supported: .txt, .md
    (PDF, Word, and images are added in the next steps.)
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if suffix in (".txt", ".md"):
        return ingest_text(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")