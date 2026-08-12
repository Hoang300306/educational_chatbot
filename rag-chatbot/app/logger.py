"""
Milestone 8 — Logging.

A tiny wrapper around the stdlib `logging` module so every part of the
app logs consistently to both the console and a rotating file, without
each module having to repeat the setup boilerplate.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with console + file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoids duplicate handlers on reimport)

    logger.setLevel(settings.log_level)
    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If the filesystem is read-only (e.g. some serverless platforms),
        # fall back to console-only logging instead of crashing the app.
        logger.warning("Could not create log directory; file logging disabled")

    logger.propagate = False
    return logger
