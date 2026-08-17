"""
Milestone 2 — Embedding
Text  ->  Embedding model  ->  Vector

Uses sentence-transformers so the whole pipeline runs locally, with no
API key and no per-call cost for embeddings (only the LLM step in
Milestone 5 needs an API key).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it for the lifetime of the
    process. The model is downloaded from Hugging Face on first use and
    cached locally afterwards, so only the very first call needs internet
    access.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model_name}")
    return SentenceTransformer(settings.embedding_model_name)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts -> float32 array of shape (N, dim), L2-normalized."""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype="float32")

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    )
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string -> float32 vector of shape (dim,)."""
    return embed_texts([text])[0]


def embedding_dim() -> int:
    """Dimensionality of the configured embedding model (e.g. 384 for MiniLM-L6)."""
    return get_embedding_model().get_sentence_embedding_dimension()
