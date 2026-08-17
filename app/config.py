"""
Application configuration.

All tunables live here and are loaded from environment variables / a .env
file (see .env.example). Nothing else in the codebase should call
os.environ directly — import `settings` from this module instead, so
there is a single source of truth for configuration.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- LLM (Milestone 5) ----
    llm_provider: str = "anthropic"  # "openai", "anthropic", or "gemini"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    # ---- Embeddings (Milestone 2) ----
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Chunking (Milestone 1) ----
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ---- Retrieval (Milestone 4) ----
    top_k: int = 4

    # ---- Storage (Milestone 3) ----
    vector_store_dir: str = "./data/vector_store"
    upload_dir: str = "./data/uploads"

    # ---- Security (Milestone 8) ----
    app_api_key: str = "change-me"
    rate_limit_per_minute: int = 20
    max_upload_mb: int = 20

    # ---- Caching (Milestone 8) ----
    cache_dir: str = "./data/cache"
    cache_ttl_seconds: int = 86400

    # ---- Logging (Milestone 8) ----
    log_level: str = "INFO"
    log_dir: str = "./logs"


settings = Settings()