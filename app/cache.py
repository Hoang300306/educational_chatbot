"""
Milestone 8 — Caching

Caches full RAG answers keyed by the (normalized) question text, so
asking the same question twice doesn't re-run retrieval or pay for a
second LLM call. Backed by diskcache, so the cache also survives
process restarts.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from diskcache import Cache

from .config import settings

_cache = Cache(str(Path(settings.cache_dir) / "answers"))


def _key(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class AnswerCache:
    def get(self, question: str) -> Optional[Any]:
        return _cache.get(_key(question))

    def set(self, question: str, value: Any) -> None:
        _cache.set(_key(question), value, expire=settings.cache_ttl_seconds)

    def clear(self) -> None:
        _cache.clear()

    def __len__(self) -> int:
        return len(_cache)


answer_cache = AnswerCache()
