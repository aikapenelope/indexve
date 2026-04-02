"""Cohere Rerank v3.5 integration for ManualIQ.

Addresses issue 1.4 (over-retrieval) from KNOWN_ISSUES.md.
Improves retrieval precision 15-30% over vector search alone
(ARCHITECTURE.md Section 3.4).

Pipeline position: after Qdrant retrieval (top-20), before LLM (top-5).
For safety-critical chunks (DANGER/WARNING/CAUTION), reranking is
always applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cohere

from manualiq.query.engine import RetrievedChunk

logger = logging.getLogger(__name__)

# Default rerank model.
RERANK_MODEL = "rerank-v3.5"


@dataclass
class RerankResult:
    """Result of reranking a set of chunks."""

    chunks: list[RetrievedChunk]
    original_count: int
    reranked_count: int


class Reranker:
    """Cohere Rerank wrapper with async support.

    Reranks retrieved chunks by semantic relevance to the query,
    replacing the raw vector similarity scores from Qdrant with
    Cohere's cross-encoder scores.
    """

    def __init__(
        self,
        api_key: str,
        model: str = RERANK_MODEL,
        top_n: int = 5,
    ) -> None:
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model
        self._top_n = top_n

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank chunks using Cohere Rerank v3.5.

        Args:
            query: The user's question.
            chunks: Retrieved chunks from Qdrant (typically top-20).
            top_n: How many to return. Defaults to self._top_n (5).

        Returns:
            RerankResult with reranked chunks and updated scores.
        """
        if not chunks:
            return RerankResult(chunks=[], original_count=0, reranked_count=0)

        n = top_n or self._top_n
        documents = [c.text for c in chunks]

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(n, len(chunks)),
            )

            reranked: list[RetrievedChunk] = []
            for result in response.results:
                idx = result.index
                original = chunks[idx]
                # Replace the vector similarity score with the rerank score.
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
                "Reranked %d -> %d chunks (best score: %.3f)",
                len(chunks),
                len(reranked),
                reranked[0].score if reranked else 0.0,
            )

            return RerankResult(
                chunks=reranked,
                original_count=len(chunks),
                reranked_count=len(reranked),
            )

        except Exception as exc:
            # If Cohere fails, fall back to original ordering (top-N by vector score).
            logger.warning(
                "Cohere rerank failed, falling back to vector scores: %s", exc
            )
            return RerankResult(
                chunks=chunks[:n],
                original_count=len(chunks),
                reranked_count=min(n, len(chunks)),
            )
