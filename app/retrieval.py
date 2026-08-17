"""
Milestone 4 — Retrieval
Question  ->  Embedding  ->  Vector Search  ->  Top-K chunks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .document_processor import Chunk
from .embeddings import embed_query
from .logger import get_logger
from .vector_store import FaissVectorStore

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


def retrieve(question: str, store: FaissVectorStore, top_k: int = 4) -> List[RetrievedChunk]:
    """Embed the question and return its top-K most similar chunks."""
    query_vector = embed_query(question)
    raw_results = store.search(query_vector, top_k=top_k)
    results = [RetrievedChunk(chunk=c, score=s) for c, s in raw_results]
    logger.info(f"Retrieved {len(results)} chunks for question: '{question[:60]}'")
    return results
