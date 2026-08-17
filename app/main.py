"""
Milestone 7 — Web application
Frontend -> FastAPI -> RAG pipeline

Also carries the Milestone 8 security basics that belong at the HTTP
boundary: API-key auth, per-key rate limiting, upload validation, and
filename sanitization.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .logger import get_logger
from .rag_pipeline import rag_pipeline

logger = get_logger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

app = FastAPI(title="Simple RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Milestone 8 — Security: shared-secret API key + in-memory rate limiting
#
# This is intentionally simple (no Redis) since it's meant for a single
# instance. If you deploy multiple workers/replicas behind a load balancer,
# swap `_request_log` for a Redis-backed limiter (e.g. slowapi + redis) so
# all workers share the same counters.
# ---------------------------------------------------------------------------
_request_log: Dict[str, Deque[float]] = defaultdict(deque)


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    if settings.app_api_key and x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def rate_limit(x_api_key: str = Header(default="")) -> None:
    now = time.time()
    window_seconds = 60
    bucket = _request_log[x_api_key or "anonymous"]
    while bucket and now - bucket[0] > window_seconds:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests — please slow down")
    bucket.append(now)


def sanitize_filename(filename: str) -> str:
    """Strip any path components and prefix a random id to avoid collisions/overwrites."""
    name = Path(filename).name
    return f"{uuid.uuid4().hex[:8]}_{name}"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    index: int
    source: str
    page: Optional[int]
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationOut]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class DocumentsResponse(BaseModel):
    documents: List[str]
    total_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Lightweight liveness check — deliberately does not touch the vector
    store or embedding model, so it stays fast even before first use."""
    return {"status": "ok"}


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=400, detail=f"File too large (max {settings.max_upload_mb} MB)")
    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    safe_name = sanitize_filename(file.filename or "document")
    dest = UPLOAD_DIR / safe_name
    with open(dest, "wb") as f:
        f.write(contents)

    try:
        num_chunks = rag_pipeline.ingest(dest)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 instead of leaking a trace
        logger.exception("Failed to ingest document")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}") from exc

    logger.info(f"Ingested '{file.filename}' -> {num_chunks} chunks")
    return UploadResponse(filename=file.filename or safe_name, chunks_added=num_chunks)


@app.post(
    "/api/ask",
    response_model=AskResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
)
def ask_question(payload: AskRequest):
    if rag_pipeline.num_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents have been uploaded yet")

    try:
        result = rag_pipeline.ask(payload.question, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        citations=[CitationOut(**asdict(c)) for c in result.citations],
    )


@app.get("/api/documents", response_model=DocumentsResponse, dependencies=[Depends(verify_api_key)])
def list_documents():
    return DocumentsResponse(documents=rag_pipeline.list_documents(), total_chunks=rag_pipeline.num_chunks)


@app.delete("/api/documents", dependencies=[Depends(verify_api_key), Depends(rate_limit)])
def delete_document(source: str = Query(..., min_length=1)):
    """Remove a document (by its exact name as returned from /api/documents)
    from the vector store and delete its uploaded file. Uses a query param
    rather than a path param so filenames with spaces/parentheses/Unicode
    (very common with real uploads) never run into URL-path edge cases."""
    removed = rag_pipeline.remove_document(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{source}' not found")
    logger.info(f"Deleted '{source}' -> {removed} chunks removed")
    return {"source": source, "chunks_removed": removed}


# Serve the frontend last, so it acts as a catch-all without shadowing the API routes above.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")