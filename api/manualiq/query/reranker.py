"""Reranker: Cohere Rerank v3.5 (primary) + Qwen3-Reranker-0.6B via Ollama (fallback).

Addresses issue 1.4 (over-retrieval) from KNOWN_ISSUES.md.
Improves retrieval precision 15-30% over vector search alone
(ARCHITECTURE.md Section 3.4).

Pipeline position: after Qdrant retrieval (top-20), before LLM (top-5).

Fallback chain:
1. Cohere Rerank v3.5 API (best quality, ~$0.05/1M tokens)
2. Qwen3-Reranker-0.6B via Ollama (local, free, lower quality)
3. Original Qdrant vector scores (no reranking, worst quality)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cohere
import httpx

from manualiq.query.engine import RetrievedChunk

logger = logging.getLogger(__name__)

# Default rerank model.
RERANK_MODEL = "rerank-v3.5"

# Ollama fallback reranker (Qwen3-Reranker-0.6B, ~639MB).
OLLAMA_RERANKER_MODEL = "sam860/qwen3-reranker:0.6b-Q8_0"
_OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Prompt template for Qwen3 reranker via Ollama chat API.
# The model scores relevance as "true"/"false" per document.
_RERANK_PROMPT_TEMPLATE = (
    "Given the query: '{query}'\n\n"
    "Is the following document relevant? Answer only 'true' or 'false'.\n\n"
    "Document: {document}"
)


@dataclass
class RerankResult:
    """Result of reranking a set of chunks."""

    chunks: list[RetrievedChunk]
    original_count: int
    reranked_count: int
    reranker_used: str = "cohere"


class Reranker:
    """Cohere Rerank with Qwen3 Ollama fallback.

    Reranks retrieved chunks by semantic relevance to the query,
    replacing the raw vector similarity scores from Qdrant with
    cross-encoder scores.
    """

    def __init__(
        self,
        api_key: str,
        model: str = RERANK_MODEL,
        top_n: int = 5,
        ollama_url: str = _OLLAMA_DEFAULT_URL,
    ) -> None:
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model
        self._top_n = top_n
        self._ollama_url = ollama_url
        self._http = httpx.AsyncClient(timeout=30.0)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank chunks with Cohere, falling back to Ollama Qwen3.

        Fallback chain:
        1. Cohere Rerank v3.5 API
        2. Qwen3-Reranker-0.6B via Ollama (local)
        3. Original vector scores (no reranking)
        """
        if not chunks:
            return RerankResult(chunks=[], original_count=0, reranked_count=0)

        n = top_n or self._top_n

        # Try Cohere first.
        try:
            return await self._rerank_cohere(query, chunks, n)
        except Exception as cohere_exc:
            logger.warning(
                "Cohere rerank failed, trying Ollama Qwen3: %s",
                str(cohere_exc)[:100],
            )

        # Try Ollama Qwen3 reranker.
        try:
            return await self._rerank_ollama(query, chunks, n)
        except Exception as ollama_exc:
            logger.warning(
                "Ollama rerank also failed, using vector scores: %s",
                str(ollama_exc)[:100],
            )

        # Final fallback: original ordering by vector score.
        return RerankResult(
            chunks=chunks[:n],
            original_count=len(chunks),
            reranked_count=min(n, len(chunks)),
            reranker_used="none",
        )

    async def _rerank_cohere(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> RerankResult:
        """Rerank using Cohere Rerank v3.5 API."""
        documents = [c.text for c in chunks]

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(chunks)),
        )

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            idx = result.index
            original = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    text=original.text,
                    score=result.relevance_score,
                    doc_id=original.doc_id,
                    tenant_id=original.tenant_id,
                    section_path=original.section_path,
                    page_ref=original.page_ref,
                    safety_level=original.safety_level,
                    hash_sha256=original.hash_sha256,
                    doc_language=original.doc_language,
                    equipment=original.equipment,
                    part_numbers=original.part_numbers,
                )
            )

        logger.info(
            "Cohere reranked %d -> %d chunks (best: %.3f)",
            len(chunks),
            len(reranked),
            reranked[0].score if reranked else 0.0,
        )

        return RerankResult(
            chunks=reranked,
            original_count=len(chunks),
            reranked_count=len(reranked),
            reranker_used="cohere",
        )

    async def _rerank_ollama(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> RerankResult:
        """Rerank using Qwen3-Reranker-0.6B via Ollama.

        Scores each chunk individually by asking the model if the
        document is relevant to the query. Slower than Cohere (one
        call per chunk) but works offline.
        """
        scored: list[tuple[float, int]] = []

        for i, chunk in enumerate(chunks):
            prompt = _RERANK_PROMPT_TEMPLATE.format(
                query=query, document=chunk.text[:2000]
            )

            try:
                response = await self._http.post(
                    f"{self._ollama_url}/api/chat",
                    json={
                        "model": OLLAMA_RERANKER_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=15.0,
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("message", {}).get("content", "").strip().lower()

                # Score: 1.0 for "true", 0.0 for "false", 0.5 for unclear.
                if "true" in answer:
                    score = 1.0
                elif "false" in answer:
                    score = 0.0
                else:
                    score = 0.5

                scored.append((score, i))
            except Exception:
                # If a single chunk fails, give it the original score.
                scored.append((chunk.score, i))

        # Sort by score descending, take top-N.
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = [
            RetrievedChunk(
                text=chunks[idx].text,
                score=scored_val,
                doc_id=chunks[idx].doc_id,
                tenant_id=chunks[idx].tenant_id,
                section_path=chunks[idx].section_path,
                page_ref=chunks[idx].page_ref,
                safety_level=chunks[idx].safety_level,
                hash_sha256=chunks[idx].hash_sha256,
                doc_language=chunks[idx].doc_language,
                equipment=chunks[idx].equipment,
                part_numbers=chunks[idx].part_numbers,
            )
            for scored_val, idx in scored[:top_n]
        ]

        logger.info(
            "Ollama Qwen3 reranked %d -> %d chunks",
            len(chunks),
            len(reranked),
        )

        return RerankResult(
            chunks=reranked,
            original_count=len(chunks),
            reranked_count=len(reranked),
            reranker_used="ollama-qwen3",
        )
