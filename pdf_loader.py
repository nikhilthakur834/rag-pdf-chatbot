"""Utilities for reading PDF files and splitting their text into chunks."""

from pypdf import PdfReader


def load_pdf(file):
    """Read a PDF (file path or file-like object) and return (page_number, text) tuples."""
    reader = PdfReader(file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        # Skip pages that contain no extractable text
        if text and text.strip():
            pages.append((page_number, text))

    return pages


def split_into_chunks(pages, chunk_size=500, chunk_overlap=50, source=""):
    """Split page text into overlapping chunks and keep the source file and page number as metadata."""
    chunks = []
    # Each new chunk starts this many characters after the previous one
    step = chunk_size - chunk_overlap

    for page_number, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size]
            # Skip empty or whitespace-only pieces
            if piece.strip():
                chunks.append({"source": source, "page": page_number, "text": piece})

    return chunks