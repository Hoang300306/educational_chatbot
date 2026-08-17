"""
Milestone 6 — RAG hoan chinh (complete RAG)
PDF/DOCX -> Processing -> Vector DB -> Retrieval -> LLM -> Answer + Citation

This module wires Milestones 1-5 together into two operations a caller
actually needs: `ingest()` a document, and `ask()` a question about
everything ingested so far.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .cache import answer_cache
from .config import settings
from .document_processor import process_document
from .embeddings import embed_texts, embedding_dim
from .llm import generate_answer
from .logger import get_logger
from .retrieval import RetrievedChunk, retrieve
from .vector_store import FaissVectorStore

logger = get_logger(__name__)


@dataclass
class Citation:
    index: int
    source: str
    page: Optional[int]
    score: float
    excerpt: str


@dataclass
class RAGAnswer:
    answer: str
    citations: List[Citation]


class RAGPipeline:
    def __init__(self) -> None:
        self.store_dir = Path(settings.vector_store_dir)
        self._store: Optional[FaissVectorStore] = None

    @property
    def store(self) -> FaissVectorStore:
        """
        Lazily load (or create) the vector store. Loading an *existing*
        store never needs the embedding model — the dimension is read
        straight from the saved metadata. A brand-new store does need the
        model, since we have to know its output dimension up front.
        """
        if self._store is None:
            if FaissVectorStore.exists(self.store_dir):
                self._store = FaissVectorStore.load(self.store_dir)
            else:
                logger.info("No existing vector store found — creating a new one")
                self._store = FaissVectorStore(dim=embedding_dim())
        return self._store

    # ---- Milestones 1 + 2 + 3 ----
    def ingest(self, file_path: Path) -> int:
        """Process a document and add its chunks to the vector store. Returns chunk count."""
        chunks = process_document(
            file_path, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        if not chunks:
            logger.warning(f"No text extracted from {file_path.name}")
            return 0

        vectors = embed_texts([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.store.save(self.store_dir)
        return len(chunks)

    # ---- Milestones 4 + 5 + 6 ----
    def ask(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        """Retrieve relevant chunks, ask the LLM, and return an answer with citations."""
        cached = answer_cache.get(question)
        if cached is not None:
            logger.info("Answer cache hit")
            return cached

        retrieved: List[RetrievedChunk] = retrieve(question, self.store, top_k=top_k or settings.top_k)
        answer_text = generate_answer(question, retrieved)

        citations = [
            Citation(
                index=i + 1,
                source=rc.chunk.source,
                page=rc.chunk.page,
                score=round(rc.score, 4),
                excerpt=rc.chunk.text[:280],
            )
            for i, rc in enumerate(retrieved)
        ]
        result = RAGAnswer(answer=answer_text, citations=citations)
        answer_cache.set(question, result)
        return result

    def list_documents(self) -> List[str]:
        return self.store.list_sources()

    def remove_document(self, source: str) -> int:
        """Remove a document's chunks from the vector store, delete its file
        from disk, and clear the answer cache (a cached answer may have been
        grounded in the chunks we just removed). Returns chunks removed."""
        removed = self.store.remove_source(source)
        if removed:
            self.store.save(self.store_dir)
            answer_cache.clear()

            uploaded_file = Path(settings.upload_dir) / source
            if uploaded_file.exists() and uploaded_file.is_file():
                uploaded_file.unlink()
        return removed

    @property
    def num_chunks(self) -> int:
        return self.store.num_chunks


# Singleton used by the FastAPI app (Milestone 7)
rag_pipeline = RAGPipeline()