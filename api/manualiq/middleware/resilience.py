"""Resilience utilities: retry, caching, async guardrails, and PII detection.

Addresses issues from KNOWN_ISSUES.md:
- 2.5: Voyage rate limits -> Exponential backoff retry + Redis embedding cache
- 2.6: NeMo latencia -> Run input guardrails in parallel with retrieval
- 4.3: PII en output -> PII detection rail for output filtering

All external service calls (Voyage, Redis, NeMo) are abstracted behind
callables/protocols so the module stays testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.5: Voyage API retry with exponential backoff + embedding cache
# ---------------------------------------------------------------------------

# Default retry configuration for Voyage API calls.
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 60.0
EMBEDDING_CACHE_TTL_SECONDS = 172_800  # 48 hours as per ARCHITECTURE.md


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry."""

    max_retries: int = MAX_RETRIES
    base_delay: float = BASE_DELAY_SECONDS
    max_delay: float = MAX_DELAY_SECONDS
    retryable_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )


class EmbeddingError(Exception):
    """Raised when embedding generation fails after all retries."""

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


async def call_with_retry(
    fn: Callable[[], Coroutine[Any, Any, object]],
    config: RetryConfig | None = None,
) -> object:
    """Call an async function with exponential backoff retry.

    Designed for Voyage API calls that may hit rate limits (429) or
    experience transient failures.

    Args:
        fn: Async callable to execute.
        config: Retry configuration. Uses defaults if None.

    Returns:
        The result of the successful call.

    Raises:
        EmbeddingError: If all retries are exhausted.
    """
    if config is None:
        config = RetryConfig()

    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            last_error = exc
            error_msg = str(exc)

            # Check if this is a retryable error.
            is_retryable = False
            for code in config.retryable_status_codes:
                if str(code) in error_msg:
                    is_retryable = True
                    break

            if not is_retryable or attempt >= config.max_retries:
                break

            # Exponential backoff with jitter.
            delay = min(
                config.base_delay * (2**attempt),
                config.max_delay,
            )
            logger.warning(
                "Retry %d/%d after %.1fs (error: %s)",
                attempt + 1,
                config.max_retries,
                delay,
                error_msg[:100],
            )
            await asyncio.sleep(delay)

    raise EmbeddingError(f"All {config.max_retries} retries exhausted: {last_error}")


def _embedding_cache_key(text: str, model: str) -> str:
    """Generate a Redis cache key for an embedding.

    Uses SHA-256 of the text + model to create a deterministic key.
    If the text hasn't changed (same hash), the embedding is reused,
    saving Voyage API credits during re-indexing.
    """
    content = f"{model}:{text}"
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"emb_cache:{h}"


class EmbeddingCache:
    """Redis-backed cache for Voyage embeddings.

    Caches embeddings with a 48-hour TTL as specified in ARCHITECTURE.md.
    Expected hit rate: ~60%, saving 40-60% in API costs.
    """

    def __init__(self, redis_client: object) -> None:
        """Initialize with an async Redis client.

        Args:
            redis_client: A redis.asyncio.Redis instance. Typed as object
                to avoid import dependency.
        """
        self._redis = redis_client

    async def get(self, text: str, model: str = "voyage-4") -> list[float] | None:
        """Retrieve a cached embedding if available.

        Args:
            text: The text that was embedded.
            model: The embedding model name.

        Returns:
            The cached embedding vector, or None if not cached.
        """
        import json

        key = _embedding_cache_key(text, model)
        cached = await self._redis.get(key)  # type: ignore[union-attr]
        if cached is not None:
            logger.debug("Embedding cache HIT for key %s", key[:20])
            return json.loads(cached)  # type: ignore[no-any-return]
        return None

    async def set(
        self,
        text: str,
        embedding: list[float],
        model: str = "voyage-4",
        ttl: int = EMBEDDING_CACHE_TTL_SECONDS,
    ) -> None:
        """Cache an embedding with TTL.

        Args:
            text: The text that was embedded.
            embedding: The embedding vector.
            model: The embedding model name.
            ttl: Cache TTL in seconds (default 48h).
        """
        import json

        key = _embedding_cache_key(text, model)
        await self._redis.set(  # type: ignore[union-attr]
            key,
            json.dumps(embedding),
            ex=ttl,
        )
        logger.debug("Embedding cached: key=%s, ttl=%ds", key[:20], ttl)


# ---------------------------------------------------------------------------
# 2.6: Async parallel execution of guardrails + retrieval
# ---------------------------------------------------------------------------


@dataclass
class GuardrailResult:
    """Result of an input guardrail check."""

    passed: bool
    reason: str | None = None
    blocked_category: str | None = None


@dataclass
class ParallelQueryResult:
    """Result of parallel guardrail + retrieval execution."""

    guardrail: GuardrailResult
    retrieval_results: object  # The raw retrieval results
    guardrail_time_ms: float
    retrieval_time_ms: float


async def run_parallel_guardrail_and_retrieval(
    query: str,
    tenant_id: str,
    *,
    guardrail_fn: Callable[[str, str], Coroutine[Any, Any, GuardrailResult]],
    retrieval_fn: Callable[[str, str], Coroutine[Any, Any, object]],
) -> ParallelQueryResult:
    """Run input guardrails in parallel with vector retrieval.

    Instead of running guardrails -> retrieval sequentially (adding
    50-200ms), we run both concurrently. If the guardrail fails, we
    discard the retrieval results. This saves latency on the happy path.

    Args:
        query: The user's question.
        tenant_id: The requesting tenant's ID.
        guardrail_fn: Async function that checks input guardrails.
        retrieval_fn: Async function that performs vector search.

    Returns:
        ParallelQueryResult with both results and timing.
    """
    guardrail_start = time.monotonic()
    retrieval_start = time.monotonic()

    # Run both concurrently.
    guardrail_task = asyncio.create_task(guardrail_fn(query, tenant_id))
    retrieval_task = asyncio.create_task(retrieval_fn(query, tenant_id))

    guardrail_result, retrieval_result = await asyncio.gather(
        guardrail_task, retrieval_task
    )

    guardrail_time = (time.monotonic() - guardrail_start) * 1000
    retrieval_time = (time.monotonic() - retrieval_start) * 1000

    logger.info(
        "Parallel execution: guardrail=%.1fms, retrieval=%.1fms",
        guardrail_time,
        retrieval_time,
    )

    return ParallelQueryResult(
        guardrail=guardrail_result,
        retrieval_results=retrieval_result if guardrail_result.passed else None,
        guardrail_time_ms=guardrail_time,
        retrieval_time_ms=retrieval_time,
    )


# ---------------------------------------------------------------------------
# 4.3: PII detection in output
# ---------------------------------------------------------------------------

# Patterns for common PII types in Spanish/English technical context.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Email addresses.
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    # Phone numbers (various formats).
    (
        "phone",
        re.compile(
            r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
        ),
    ),
    # Venezuelan cedula (V-12345678 or similar).
    ("cedula", re.compile(r"\b[VEJGvejg]-?\d{6,10}\b")),
    # Social security / ID numbers (generic).
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card numbers (basic pattern).
    ("credit_card", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")),
]

# Words that indicate personal information context.
_PII_CONTEXT_WORDS = [
    "nombre completo",
    "cedula",
    "telefono",
    "direccion",
    "correo personal",
    "numero de empleado",
    "salario",
    "full name",
    "social security",
    "home address",
    "personal email",
    "employee id",
]


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan."""

    has_pii: bool
    detections: list[dict[str, str]]
    sanitized_text: str | None = None


def detect_pii(text: str) -> PIIDetectionResult:
    """Detect PII patterns in text.

    Scans for common PII patterns (emails, phones, IDs) that should
    not appear in ManualIQ responses. Technical manuals shouldn't
    contain PII, but OCR errors or metadata leakage can introduce it.

    Args:
        text: The text to scan (typically the LLM response).

    Returns:
        PIIDetectionResult with detection details.
    """
    detections: list[dict[str, str]] = []

    for pii_type, pattern in _PII_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            detections.append({"type": pii_type, "value": match})

    # Check for PII context words.
    lower_text = text.lower()
    for word in _PII_CONTEXT_WORDS:
        if word in lower_text:
            detections.append({"type": "context_word", "value": word})

    return PIIDetectionResult(
        has_pii=len(detections) > 0,
        detections=detections,
    )


def sanitize_pii(text: str) -> PIIDetectionResult:
    """Detect and redact PII from text.

    Replaces detected PII with redaction markers. This is the output
    rail version that modifies the response before sending to the user.

    Args:
        text: The text to sanitize.

    Returns:
        PIIDetectionResult with sanitized text.
    """
    detection = detect_pii(text)

    if not detection.has_pii:
        detection.sanitized_text = text
        return detection

    sanitized = text
    for pii_type, pattern in _PII_PATTERNS:
        sanitized = pattern.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)

    logger.warning(
        "PII detected and redacted: %d instances",
        len(detection.detections),
    )

    detection.sanitized_text = sanitized
    return detection
