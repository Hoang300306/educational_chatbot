"""
Milestone 1 — Document Processing
PDF / DOCX  ->  Text  ->  Chunks
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document as DocxDocument
from pypdf import PdfReader

from .logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


@dataclass
class Chunk:
    """A single chunk of text ready to be embedded, plus the metadata
    needed to cite it back to the original document later on."""

    id: str
    text: str
    source: str            # original filename
    page: Optional[int]    # page number for PDFs, None for DOCX
    chunk_index: int       # position of this chunk within the document


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_pages_from_pdf(path: Path) -> List[Tuple[int, str]]:
    """Return [(page_number, raw_text), ...] for every page, 1-indexed."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def extract_text_from_docx(path: Path) -> str:
    """Extract paragraph and table text from a .docx file, in reading order."""
    doc = DocxDocument(str(path))
    parts: List[str] = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip(" |"):
                parts.append(row_text)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Cleaning + chunking
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Collapse repeated whitespace produced by PDF/DOCX extraction."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks, preferring to break on paragraph
    boundaries and falling back to sentence boundaries for long paragraphs.

    This is a small, dependency-free splitter written for this project —
    good enough for a "simple" RAG tool, and easy to read/modify. Swap in
    langchain's RecursiveCharacterTextSplitter later if you need something
    more sophisticated.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush(buffer: str) -> None:
        buffer = buffer.strip()
        if buffer:
            chunks.append(buffer)

    def start_new_chunk(previous: str, next_piece: str, sep: str) -> str:
        """Seed the next chunk with the tail of the previous one (overlap)."""
        overlap = previous[-chunk_overlap:] if chunk_overlap else ""
        return f"{overlap}{sep}{next_piece}".strip()

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph itself is too big — fall back to sentence splitting.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current = f"{current} {sent}".strip()
                else:
                    flush(current)
                    current = start_new_chunk(current, sent, " ")
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            flush(current)
            current = start_new_chunk(current, para, "\n\n")

    flush(current)
    return chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def process_document(path: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Chunk]:
    """
    Full Milestone 1 pipeline: load a PDF or DOCX file, extract its text,
    and split it into overlapping chunks carrying citation metadata.
    """
    suffix = path.suffix.lower()
    chunks: List[Chunk] = []

    if suffix == ".pdf":
        pages = extract_pages_from_pdf(path)
        idx = 0
        for page_num, raw_text in pages:
            for piece in chunk_text(raw_text, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(id=str(uuid.uuid4()), text=piece, source=path.name, page=page_num, chunk_index=idx)
                )
                idx += 1
    elif suffix == ".docx":
        raw_text = extract_text_from_docx(path)
        for idx, piece in enumerate(chunk_text(raw_text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(id=str(uuid.uuid4()), text=piece, source=path.name, page=None, chunk_index=idx)
            )
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Only .pdf and .docx are supported.")

    logger.info(f"Processed '{path.name}' -> {len(chunks)} chunks")
    return chunks
