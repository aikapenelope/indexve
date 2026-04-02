"""Structured logging configuration for ManualIQ.

Configures JSON-formatted logging for production (Docker/Phoenix)
and human-readable logging for development.

Usage in main.py lifespan:
    from manualiq.logging_config import setup_logging
    setup_logging()
"""

from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """Configure logging based on environment.

    Production (LOG_FORMAT=json): JSON lines for Docker log drivers
    and Phoenix ingestion.
    Development (default): Human-readable with colors.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "text")

    if log_format == "json":
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Reduce noise from chatty libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
