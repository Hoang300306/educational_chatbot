"""
API-level tests using FastAPI's TestClient. These deliberately stick to
paths that fail validation *before* touching the embedding model, so the
test suite runs fully offline (no Hugging Face / LLM network calls).
"""
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_health_is_fast_and_public():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_requires_api_key():
    resp = client.post("/api/ask", json={"question": "hello"})
    assert resp.status_code == 401


def test_ask_rejects_empty_question():
    resp = client.post(
        "/api/ask",
        headers={"X-API-Key": settings.app_api_key},
        json={"question": ""},
    )
    assert resp.status_code == 422  # fails Pydantic min_length validation


def test_upload_requires_api_key():
    resp = client.post("/api/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert resp.status_code == 401


def test_upload_rejects_unsupported_extension():
    resp = client.post(
        "/api/upload",
        headers={"X-API-Key": settings.app_api_key},
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400
    assert "pdf" in resp.json()["detail"].lower()


def test_delete_document_requires_api_key():
    resp = client.request("DELETE", "/api/documents?source=whatever.pdf")
    assert resp.status_code == 401


def test_delete_document_requires_source_param():
    resp = client.request("DELETE", "/api/documents", headers={"X-API-Key": settings.app_api_key})
    assert resp.status_code == 422  # missing required query param


def test_upload_rejects_empty_file():
    resp = client.post(
        "/api/upload",
        headers={"X-API-Key": settings.app_api_key},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400