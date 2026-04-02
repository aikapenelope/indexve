"""LLM response cache backed by Redis.

Caches LLM-generated answers with a 24h TTL as specified in
ARCHITECTURE.md Section 3.9 (Tier 2 cache). The cache key is
a hash of (query + tenant_id) so identical questions from the
same tenant get served from cache without calling Claude.

Expected hit rate: ~60% for manufacturing queries (technicians
ask the same questions repeatedly). Saves 40-60% in API costs.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# 24 hours TTL for LLM response cache.
LLM_CACHE_TTL_SECONDS = 86_400


def _cache_key(query: str, tenant_id: str) -> str:
    """Generate a deterministic cache key from query + tenant."""
    content = f"{tenant_id}:{query.strip().lower()}"
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"llm_cache:{h}"


class LLMResponseCache:
    """Redis-backed cache for LLM responses.

    Stores the full QueryResponse-equivalent dict so that cached
    responses include confidence, score, and chunk metadata.
    """

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    async def get(self, query: str, tenant_id: str) -> dict[str, object] | None:
        """Retrieve a cached response if available."""
        key = _cache_key(query, tenant_id)
        cached = await self._redis.get(key)  # type: ignore[union-attr]
        if cached is not None:
            logger.debug("LLM cache HIT: %s", key[:20])
            return json.loads(cached)  # type: ignore[no-any-return]
        return None

    async def set(
        self,
        query: str,
        tenant_id: str,
        response: dict[str, object],
        ttl: int = LLM_CACHE_TTL_SECONDS,
    ) -> None:
        """Cache an LLM response."""
        key = _cache_key(query, tenant_id)
        await self._redis.set(  # type: ignore[union-attr]
            key,
            json.dumps(response),
            ex=ttl,
        )
        logger.debug("LLM response cached: %s (ttl=%ds)", key[:20], ttl)
