"""
Milestone 3 — Vector Database
Chunks  ->  Embedding  ->  FAISS

A thin wrapper around a FAISS IndexFlatIP index plus a parallel Python
list of Chunk metadata, so a similarity search result can be mapped
straight back to the original text, source filename and page number.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .document_processor import Chunk
from .logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.pkl"


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vectors
        self.metadata: List[Chunk] = []

    # ---- writes ----
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors must have the same length")
        if vectors.shape[0] == 0:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.metadata.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to vector store (total={len(self.metadata)})")

    # ---- reads ----
    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query_vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def list_sources(self) -> List[str]:
        return sorted({c.source for c in self.metadata})

    def remove_source(self, source: str) -> int:
        """
        Remove every chunk belonging to `source`. FAISS's flat index has no
        native per-vector delete, so this reconstructs the surviving vectors
        and rebuilds the index from scratch — perfectly fine at the scale a
        "simple" tool like this runs at. Returns the number of chunks removed.
        """
        keep_mask = np.array([c.source != source for c in self.metadata], dtype=bool)
        removed = int((~keep_mask).sum())
        if removed == 0:
            return 0

        all_vectors = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else np.zeros((0, self.dim), dtype="float32")
        kept_chunks = [c for c, keep in zip(self.metadata, keep_mask) if keep]
        kept_vectors = all_vectors[keep_mask]

        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        if kept_chunks:
            self.add(kept_chunks, kept_vectors)

        logger.info(f"Removed {removed} chunks for source '{source}' ({len(self.metadata)} remaining)")
        return removed

    @property
    def num_chunks(self) -> int:
        return len(self.metadata)

    # ---- persistence ----
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        with open(directory / _META_FILE, "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)
        logger.info(f"Vector store saved to {directory}")

    @staticmethod
    def exists(directory: Path) -> bool:
        """Check for a saved store on disk WITHOUT loading the embedding model."""
        directory = Path(directory)
        return (directory / _INDEX_FILE).exists() and (directory / _META_FILE).exists()

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        directory = Path(directory)
        if not cls.exists(directory):
            raise FileNotFoundError(f"No vector store found at {directory}")
        with open(directory / _META_FILE, "rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"])
        store.index = faiss.read_index(str(directory / _INDEX_FILE))
        store.metadata = payload["metadata"]
        logger.info(f"Vector store loaded from {directory} ({len(store.metadata)} chunks)")
        return store