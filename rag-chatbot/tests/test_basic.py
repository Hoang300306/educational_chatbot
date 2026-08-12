"""
Unit tests for the parts of the pipeline that don't need network access
(no embedding-model download, no LLM API calls). Run with: pytest
"""
from pathlib import Path

import numpy as np

from app.document_processor import Chunk, chunk_text
from app.vector_store import FaissVectorStore


def test_chunk_text_basic():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_respects_size_roughly():
    long_text = " ".join(f"sentence number {i}." for i in range(200))
    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # allow slack for the overlap seed + not splitting mid-sentence
    assert all(len(c) <= 150 for c in chunks)


def test_vector_store_add_and_search():
    dim = 8
    store = FaissVectorStore(dim=dim)
    chunks = [
        Chunk(id="1", text="hello world", source="a.pdf", page=1, chunk_index=0),
        Chunk(id="2", text="goodbye world", source="a.pdf", page=1, chunk_index=1),
    ]
    rng = np.random.RandomState(0)
    vectors = rng.rand(2, dim).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)  # mimic normalize_embeddings=True
    store.add(chunks, vectors)

    assert store.num_chunks == 2
    results = store.search(vectors[0], top_k=1)
    assert len(results) == 1
    assert results[0][0].id == "1"
    assert results[0][1] > 0.99  # searching with its own vector -> ~perfect match


def test_vector_store_save_and_load(tmp_path: Path):
    dim = 4
    store = FaissVectorStore(dim=dim)
    chunks = [Chunk(id="1", text="test", source="a.pdf", page=None, chunk_index=0)]
    vectors = np.random.RandomState(0).rand(1, dim).astype("float32")
    store.add(chunks, vectors)
    store.save(tmp_path)

    assert FaissVectorStore.exists(tmp_path)
    loaded = FaissVectorStore.load(tmp_path)
    assert loaded.num_chunks == 1
    assert loaded.metadata[0].source == "a.pdf"
    assert loaded.list_sources() == ["a.pdf"]


def test_vector_store_exists_false_for_empty_dir(tmp_path: Path):
    assert FaissVectorStore.exists(tmp_path) is False
