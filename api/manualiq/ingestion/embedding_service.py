"""Embedding service: Voyage-4 API + Redis cache + Qdrant upsert.

Fills the placeholder in flows/operations.py (Step 3: "Embed + upsert
would happen here") and provides the embedding layer for the query
engine.

Two modes:
- Indexing: embed chunks with voyage-4 (full model), cache in Redis,
  upsert to Qdrant with metadata.
- Query: embed query with voyage-4-lite (cheaper, shared embedding
  space), check Redis cache first.

Uses the EmbeddingCache from middleware/resilience.py for Redis caching
and call_with_retry for Voyage API resilience (issue 2.5).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import httpx
from qdrant_client import AsyncQdrantClient, models

from manualiq.ingestion.chunker import Chunk
from manualiq.middleware.resilience import (
    EmbeddingCache,
    RetryConfig,
    call_with_retry,
)

logger = logging.getLogger(__name__)

# Voyage-4 API endpoint.
_VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"

# Models: full for indexing, lite for queries (shared embedding space).
# Voyage 4 series: all embeddings are compatible with each other.
VOYAGE_INDEX_MODEL = "voyage-4"
VOYAGE_QUERY_MODEL = "voyage-4-lite"

# Ollama fallback model (ARCHITECTURE.md: Qwen3-Embedding-0.6B, ~639MB).
# Used when Voyage API is down. Runs locally on CPU via Ollama.
OLLAMA_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
_OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Batch size for Voyage API (max 128 texts per request).
_VOYAGE_BATCH_SIZE = 64


@dataclass
class EmbeddingResult:
    """Result of embedding a single text."""

    text: str
    vector: list[float]
    model: str
    cached: bool = False


class EmbeddingService:
    """Voyage-4 embedding service with Redis cache and Qdrant upsert.

    Handles both indexing (bulk embed + upsert) and query (single embed)
    workflows with caching to minimize API costs.
    """

    def __init__(
        self,
        voyage_api_key: str,
        qdrant_client: AsyncQdrantClient,
        cache: EmbeddingCache | None = None,
        retry_config: RetryConfig | None = None,
        ollama_url: str = _OLLAMA_DEFAULT_URL,
    ) -> None:
        self._api_key = voyage_api_key
        self._qdrant = qdrant_client
        self._cache = cache
        self._retry_config = retry_config or RetryConfig()
        self._ollama_url = ollama_url
        self._http = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        """Clean up HTTP client."""
        await self._http.aclose()

    async def _call_ollama_embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Call Ollama local embedding API (Qwen3-Embedding-0.6B fallback).

        Used when Voyage API is completely down after all retries.
        Lower quality (MTEB ~63 vs Voyage 66.8) but works offline.

        Note: Ollama /api/embed supports batch input natively.
        """
        response = await self._http.post(
            f"{self._ollama_url}/api/embed",
            json={
                "model": OLLAMA_EMBEDDING_MODEL,
                "input": texts,
            },
            timeout=120.0,  # Local model can be slow on CPU.
        )
        response.raise_for_status()
        data = response.json()
        embeddings: list[list[float]] = data["embeddings"]
        return embeddings

    async def _call_voyage_api(
        self,
        texts: list[str],
        model: str,
    ) -> list[list[float]]:
        """Call Voyage API with retry for a batch of texts."""

        async def _do_call() -> object:
            response = await self._http.post(
                _VOYAGE_API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "input": texts,
                    "model": model,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]

        result = await call_with_retry(_do_call, self._retry_config)
        return result  # type: ignore[return-value]

    async def _embed_with_fallback(
        self,
        texts: list[str],
        model: str,
    ) -> tuple[list[list[float]], str]:
        """Embed texts with Voyage API, falling back to Ollama on failure.

        Returns (vectors, model_used) tuple.
        """
        try:
            vectors = await self._call_voyage_api(texts, model)
            return vectors, model
        except Exception as voyage_exc:
            logger.warning(
                "Voyage API failed for %d texts, falling back to Ollama Qwen3: %s",
                len(texts),
                str(voyage_exc)[:100],
            )
            try:
                vectors = await self._call_ollama_embed(texts)
                return vectors, OLLAMA_EMBEDDING_MODEL
            except Exception as ollama_exc:
                logger.error(
                    "Both Voyage and Ollama failed. Voyage: %s, Ollama: %s",
                    str(voyage_exc)[:100],
                    str(ollama_exc)[:100],
                )
                raise

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query using voyage-4-lite (cheaper).

        Checks Redis cache first. On miss, calls Voyage API with
        Ollama Qwen3 fallback if Voyage is down.
        """
        model = VOYAGE_QUERY_MODEL

        # Check cache.
        if self._cache:
            cached_vector = await self._cache.get(text, model)
            if cached_vector is not None:
                return EmbeddingResult(
                    text=text, vector=cached_vector, model=model, cached=True
                )

        # Cache miss: call API with fallback.
        vectors, model_used = await self._embed_with_fallback([text], model)
        vector = vectors[0]

        # Cache the result.
        if self._cache:
            await self._cache.set(text, vector, model_used)

        return EmbeddingResult(text=text, vector=vector, model=model_used)

    async def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddingResult]:
        """Embed multiple chunks using voyage-4 (full model).

        Checks Redis cache per chunk (by hash). Only calls Voyage API
        for cache misses, in batches of _VOYAGE_BATCH_SIZE.
        """
        model = VOYAGE_INDEX_MODEL
        results: list[EmbeddingResult | None] = [None] * len(chunks)
        uncached_indices: list[int] = []

        # Phase 1: Check cache for each chunk.
        for i, chunk in enumerate(chunks):
            if self._cache:
                cached_vector = await self._cache.get(chunk.text, model)
                if cached_vector is not None:
                    results[i] = EmbeddingResult(
                        text=chunk.text,
                        vector=cached_vector,
                        model=model,
                        cached=True,
                    )
                    continue
            uncached_indices.append(i)

        cached_count = len(chunks) - len(uncached_indices)
        if cached_count > 0:
            logger.info(
                "Embedding cache: %d/%d hits, %d misses",
                cached_count,
                len(chunks),
                len(uncached_indices),
            )

        # Phase 2: Batch-embed cache misses.
        for batch_start in range(0, len(uncached_indices), _VOYAGE_BATCH_SIZE):
            batch_indices = uncached_indices[
                batch_start : batch_start + _VOYAGE_BATCH_SIZE
            ]
            batch_texts = [chunks[i].text for i in batch_indices]

            vectors, model_used = await self._embed_with_fallback(batch_texts, model)

            for j, idx in enumerate(batch_indices):
                vector = vectors[j]
                results[idx] = EmbeddingResult(
                    text=chunks[idx].text, vector=vector, model=model_used
                )
                # Cache the new embedding.
                if self._cache:
                    await self._cache.set(chunks[idx].text, vector, model_used)

        # All slots should be filled now.
        return [r for r in results if r is not None]

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[EmbeddingResult],
        collection_name: str = "manualiq",
    ) -> int:
        """Upsert embedded chunks to Qdrant with full metadata.

        Uses the shard_key=tenant_id for native tenant isolation.

        Args:
            chunks: The chunks with metadata.
            embeddings: The embedding results (same order as chunks).
            collection_name: Qdrant collection name.

        Returns:
            Number of points upserted.
        """
        points: list[models.PointStruct] = []

        for chunk, emb in zip(chunks, embeddings):
            meta = chunk.metadata
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, meta.hash_sha256))

            payload = {
                "doc_id": meta.doc_id,
                "tenant_id": meta.tenant_id,
                "equipment": meta.equipment,
                "manufacturer": meta.manufacturer,
                "doc_language": meta.doc_language,
                "section_path": meta.section_path,
                "procedure_type": meta.procedure_type,
                "safety_level": meta.safety_level,
                "part_numbers": meta.part_numbers,
                "related_chunks": meta.related_chunks,
                "page_ref": meta.page_ref,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "hash_sha256": meta.hash_sha256,
                "indexed_at": meta.indexed_at,
                "text": chunk.text,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=emb.vector,
                    payload=payload,
                )
            )

        # Upsert in batches of 100.
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._qdrant.upsert(
                collection_name=collection_name,
                points=batch,
                shard_key_selector=chunks[0].metadata.tenant_id if chunks else None,
            )

        logger.info(
            "Upserted %d points to Qdrant collection '%s'",
            len(points),
            collection_name,
        )
        return len(points)

    async def index_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str = "manualiq",
    ) -> dict[str, int]:
        """Full indexing pipeline: embed + upsert.

        This is the method that fills the placeholder in
        flows/operations.py reindex_document().

        Args:
            chunks: Chunks with metadata from the chunker.
            collection_name: Qdrant collection name.

        Returns:
            Dict with counts: embedded, cached, upserted.
        """
        if not chunks:
            return {"embedded": 0, "cached": 0, "upserted": 0}

        embeddings = await self.embed_chunks(chunks)
        cached = sum(1 for e in embeddings if e.cached)
        upserted = await self.upsert_chunks(chunks, embeddings, collection_name)

        return {
            "embedded": len(embeddings),
            "cached": cached,
            "upserted": upserted,
        }
