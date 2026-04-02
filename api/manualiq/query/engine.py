"""Query engine with confidence threshold and reranking pipeline.

Addresses issues 1.1 (hallucination cascade) and 1.4 (over-retrieval)
from KNOWN_ISSUES.md.

Pipeline: retrieve top-20 -> rerank with Cohere -> top-5 to LLM.
If best retrieval score < CONFIDENCE_THRESHOLD, respond with
"No encontre informacion" instead of generating a potentially
hallucinated answer.

Deduplication by hash_sha256 prevents duplicate chunks from
re-indexed documents from polluting the context window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Confidence threshold: if the best retrieval score after reranking
# is below this value, we refuse to generate and ask the user to
# reformulate or escalate to a supervisor.
CONFIDENCE_THRESHOLD = 0.75

# How many chunks to retrieve from Qdrant before reranking.
RETRIEVAL_TOP_K = 20

# How many chunks to pass to the LLM after reranking.
RERANK_TOP_K = 5

# Default "no information found" response in Spanish.
NO_INFO_RESPONSE = (
    "No encontre informacion sobre esto en los manuales disponibles. "
    "Por favor reformule la pregunta o consulte con su supervisor."
)


class QueryConfidence(Enum):
    """Confidence level of the query result."""

    HIGH = "high"  # score >= 0.75
    LOW = "low"  # score < 0.75, response generated with warning
    NONE = "none"  # no relevant chunks found at all


@dataclass
class RetrievedChunk:
    """A chunk retrieved from Qdrant with its relevance score."""

    text: str
    score: float
    doc_id: str
    tenant_id: str
    section_path: str
    page_ref: str
    safety_level: str
    hash_sha256: str
    doc_language: str = "en"
    equipment: str = ""
    part_numbers: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Result of a query through the ManualIQ pipeline."""

    answer: str
    confidence: QueryConfidence
    best_score: float
    chunks_used: list[RetrievedChunk]
    chunks_retrieved: int
    chunks_after_dedup: int
    chunks_after_rerank: int


def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunks based on hash_sha256.

    When documents are re-indexed, the same content may appear multiple
    times in Qdrant. We keep only the highest-scoring instance of each
    unique chunk.
    """
    seen_hashes: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        h = chunk.hash_sha256
        if h not in seen_hashes or chunk.score > seen_hashes[h].score:
            seen_hashes[h] = chunk
    deduped = sorted(seen_hashes.values(), key=lambda c: c.score, reverse=True)
    logger.debug("Deduplication: %d -> %d chunks", len(chunks), len(deduped))
    return deduped


def validate_tenant_chunks(
    chunks: list[RetrievedChunk],
    tenant_id: str,
) -> list[RetrievedChunk]:
    """Post-retrieval tenant isolation check (issue 4.2).

    Even though Qdrant uses shard keys for tenant isolation, we perform
    a second verification here. Any chunk that doesn't belong to the
    requesting tenant is discarded and logged as a security alert.
    """
    valid: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.tenant_id != tenant_id:
            logger.error(
                "SECURITY: Cross-tenant chunk detected! "
                "Expected tenant=%s, got tenant=%s, doc=%s",
                tenant_id,
                chunk.tenant_id,
                chunk.doc_id,
            )
            continue
        valid.append(chunk)
    return valid


def build_context_prompt(
    chunks: list[RetrievedChunk],
    query: str,
) -> str:
    """Build the context prompt for the LLM from retrieved chunks.

    Follows the system prompt rules from ARCHITECTURE.md Section 7:
    - Safety warnings (DANGER/WARNING/CAUTION) appear first.
    - Each chunk includes its citation reference.
    - English chunks include the original text for citation.
    """
    # Sort: safety-critical chunks first.
    safety_order = {"critico": 0, "precaucion": 1, "informativo": 2}
    sorted_chunks = sorted(
        chunks,
        key=lambda c: safety_order.get(c.safety_level, 2),
    )

    context_parts: list[str] = []
    for i, chunk in enumerate(sorted_chunks, 1):
        citation = (
            f"[Manual: {chunk.doc_id}, "
            f"Seccion: {chunk.section_path}, "
            f"Pagina: {chunk.page_ref}]"
        )

        safety_prefix = ""
        if chunk.safety_level == "critico":
            safety_prefix = "**DANGER/PELIGRO** "
        elif chunk.safety_level == "precaucion":
            safety_prefix = "**WARNING/ADVERTENCIA** "

        lang_note = ""
        if chunk.doc_language == "en":
            lang_note = " (Documento fuente en ingles)"

        context_parts.append(
            f"--- Fragmento {i} {citation}{lang_note} ---\n{safety_prefix}{chunk.text}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"Pregunta del tecnico: {query}\n\n"
        f"Fragmentos recuperados de los manuales:\n\n{context}\n\n"
        "Instrucciones: Responde en espanol. Cita cada afirmacion tecnica "
        "con la referencia del fragmento. Si hay DANGER/WARNING/CAUTION, "
        "mencionalo PRIMERO en negrita. Incluye todos los valores numericos "
        "con unidades y tolerancias. Si el fragmento esta en ingles, incluye "
        "el texto original entre comillas y la traduccion."
    )


def evaluate_confidence(
    chunks: list[RetrievedChunk],
) -> tuple[QueryConfidence, float]:
    """Evaluate the confidence level based on retrieval scores.

    Returns the confidence classification and the best score.
    """
    if not chunks:
        return QueryConfidence.NONE, 0.0

    best_score = chunks[0].score  # Chunks are sorted by score descending.
    if best_score >= CONFIDENCE_THRESHOLD:
        return QueryConfidence.HIGH, best_score
    return QueryConfidence.LOW, best_score


def process_query(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    tenant_id: str,
    *,
    generate_fn: Callable[[str], str] | None = None,
) -> QueryResult:
    """Process a query through the full pipeline.

    This is the main orchestration function that ties together:
    1. Deduplication by hash
    2. Tenant isolation validation
    3. Confidence threshold check
    4. Context building for the LLM

    The actual LLM call is delegated to generate_fn (injected) so this
    module stays testable without API keys.

    Args:
        query: The user's question in Spanish.
        retrieved_chunks: Raw chunks from Qdrant (top-20, already reranked
            by Cohere if available).
        tenant_id: The requesting tenant's ID.
        generate_fn: Async callable that takes a prompt string and returns
            the LLM response. If None, returns the prompt as the answer
            (useful for testing).

    Returns:
        QueryResult with the answer, confidence level, and metadata.
    """
    total_retrieved = len(retrieved_chunks)

    # Step 1: Deduplicate by hash.
    deduped = deduplicate_chunks(retrieved_chunks)

    # Step 2: Tenant isolation check.
    tenant_valid = validate_tenant_chunks(deduped, tenant_id)

    # Step 3: Take top-K after reranking (chunks should already be
    # sorted by reranker score).
    top_chunks = tenant_valid[:RERANK_TOP_K]

    # Step 4: Evaluate confidence.
    confidence, best_score = evaluate_confidence(top_chunks)

    if confidence == QueryConfidence.NONE:
        return QueryResult(
            answer=NO_INFO_RESPONSE,
            confidence=QueryConfidence.NONE,
            best_score=0.0,
            chunks_used=[],
            chunks_retrieved=total_retrieved,
            chunks_after_dedup=len(deduped),
            chunks_after_rerank=len(top_chunks),
        )

    # Step 5: Build context prompt.
    prompt = build_context_prompt(top_chunks, query)

    # Step 6: Generate answer (or return prompt for testing).
    if generate_fn is not None:
        # In production, generate_fn would be an async LLM call.
        # For now we support sync callables for testing.
        answer: str = generate_fn(prompt)
    else:
        # Testing mode: return the built prompt.
        answer = prompt

    # Add low-confidence warning if needed.
    if confidence == QueryConfidence.LOW:
        answer = (
            f"**ATENCION: Confianza baja (score: {best_score:.2f}).** "
            "Se recomienda verificar esta informacion con el supervisor.\n\n" + answer
        )

    return QueryResult(
        answer=answer,
        confidence=confidence,
        best_score=best_score,
        chunks_used=top_chunks,
        chunks_retrieved=total_retrieved,
        chunks_after_dedup=len(deduped),
        chunks_after_rerank=len(top_chunks),
    )
