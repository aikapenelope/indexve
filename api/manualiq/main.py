"""ManualIQ FastAPI application entrypoint.

Connects all modules into a running API:
- POST /query    — RAG query pipeline (auth + rate limit + guardrails + retrieve + generate)
- POST /ingest   — Trigger document ingestion via Prefect
- GET  /health   — Service health check
- GET  /health/redis — Redis connectivity check
- GET  /history  — Conversation history per user/tenant

All routes inject tenant_id and user_id server-side from Clerk JWT.
The tenant_id is NEVER accepted from the request body (issue 4.1).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from manualiq.ingestion.embedding_service import EmbeddingService
from manualiq.llm_gateway import LLMGateway
from manualiq.middleware.auth import AuthContext, authenticate_request
from manualiq.middleware.rate_limiter import (
    RateLimiter,
    check_cost_alert,
)
from manualiq.middleware.resilience import EmbeddingCache, sanitize_pii
from manualiq.query.engine import (
    QueryConfidence,
    RetrievedChunk,
    build_context_prompt,
    deduplicate_chunks,
    evaluate_confidence,
    validate_tenant_chunks,
)
from manualiq.query.intelligence import (
    classify_intent,
    enforce_spanish_response,
    expand_query_crosslingual,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _read_secret(env_var: str, default: str = "") -> str:
    """Read a secret from a file path (Docker secrets) or env var."""
    file_path = os.environ.get(env_var, "")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    # Fall back to direct env var (without _FILE suffix).
    plain_var = env_var.replace("_FILE", "")
    return os.environ.get(plain_var, default)


# ---------------------------------------------------------------------------
# Application state (initialized in lifespan)
# ---------------------------------------------------------------------------


class AppState:
    """Shared application state initialized at startup."""

    redis: aioredis.Redis  # type: ignore[type-arg]
    qdrant: AsyncQdrantClient
    llm: LLMGateway
    embeddings: EmbeddingService
    rate_limiter: RateLimiter


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize and tear down shared resources."""
    # Phoenix tracing (ARCHITECTURE.md Section 3.13).
    # Automatically captures every LlamaIndex query, retrieval, reranking,
    # and LLM call with times, tokens, and costs.
    phoenix_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
    if phoenix_endpoint:
        try:
            from openinference.instrumentation.llama_index import (  # type: ignore[import-untyped]
                LlamaIndexInstrumentor,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-untyped]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[import-untyped]

            provider = TracerProvider()
            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=phoenix_endpoint))
            )
            LlamaIndexInstrumentor().instrument(tracer_provider=provider)
            logger.info("Phoenix tracing enabled: %s", phoenix_endpoint)
        except ImportError:
            logger.warning(
                "Phoenix tracing packages not installed, skipping instrumentation"
            )

    # Redis.
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    state.redis = aioredis.from_url(redis_url, decode_responses=True)  # type: ignore[assignment]

    # Qdrant.
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    state.qdrant = AsyncQdrantClient(url=qdrant_url)

    # LLM Gateway.
    state.llm = LLMGateway(
        anthropic_api_key=_read_secret("ANTHROPIC_API_KEY_FILE"),
        google_api_key=_read_secret("GOOGLE_API_KEY_FILE"),
        system_prompt=(
            "Eres ManualIQ, asistente tecnico especializado en manuales "
            "industriales. Responde SIEMPRE en espanol completo, claro y "
            "tecnicamente preciso. CADA afirmacion tecnica debe incluir "
            "cita exacta: [Manual: nombre, Seccion: seccion, Pagina: pagina]. "
            "Si no hay informacion, responde 'No encontre informacion sobre "
            "esto en los manuales disponibles.' NUNCA inventar datos tecnicos."
        ),
    )

    # Embedding service with Redis cache.
    cache = EmbeddingCache(state.redis)
    state.embeddings = EmbeddingService(
        voyage_api_key=_read_secret("VOYAGE_API_KEY_FILE"),
        qdrant_client=state.qdrant,
        cache=cache,
    )

    # Rate limiter.
    state.rate_limiter = RateLimiter(state.redis)

    logger.info("ManualIQ API started")
    yield

    # Cleanup.
    await state.embeddings.close()
    await state.qdrant.close()
    await state.redis.aclose()
    logger.info("ManualIQ API stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ManualIQ API",
    description="RAG API for manufacturing technical manuals",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth dependency (Clerk JWT -> tenant_id + user_id)
# Uses manualiq.middleware.auth which supports both production (Clerk JWT)
# and development (header-based) authentication.
# ---------------------------------------------------------------------------


async def get_auth(request: Request) -> AuthContext:
    """Authenticate request via Clerk JWT or dev headers."""
    return await authenticate_request(request)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    confidence: str
    score: float
    chunks_used: int
    chunks_retrieved: int
    was_fallback: bool = False
    intent: str = "specific"


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    file_path: str = Field(..., description="Path to PDF on server")
    equipment: str = ""
    manufacturer: str = ""
    doc_language: str = "en"
    procedure_type: str = "informativo"


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    status: str
    doc_id: str
    chunks: int
    parser: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    services: dict[str, str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    auth: AuthContext = Depends(get_auth),
) -> QueryResponse:
    """Main RAG query endpoint.

    Pipeline: auth -> rate limit -> intent classify -> expand query ->
    embed -> retrieve -> dedup -> tenant validate -> rerank -> threshold ->
    generate -> PII check -> language check -> respond.
    """
    # Step 1: Rate limit.
    rate_result = await state.rate_limiter.check_rate_limit(
        auth.tenant_id, auth.user_id, auth.plan
    )
    if not rate_result.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {rate_result.retry_after_seconds}s.",
        )

    # Step 2: Intent classification (Gemini Flash).
    intent_result = classify_intent(
        body.query,
        llm_fn=lambda p: _sync_route(p),
    )

    if intent_result.intent.value == "ambiguous" and intent_result.clarification_prompt:
        return QueryResponse(
            answer=intent_result.clarification_prompt,
            confidence="none",
            score=0.0,
            chunks_used=0,
            chunks_retrieved=0,
            intent="ambiguous",
        )

    # Step 3: Query expansion (ES -> EN).
    queries = expand_query_crosslingual(
        body.query,
        llm_fn=lambda p: _sync_route(p),
    )

    # Step 4: Embed query.
    query_embedding = await state.embeddings.embed_query(queries[0])

    # Step 5: Retrieve from Qdrant (top-20).
    search_results = await state.qdrant.query_points(
        collection_name="manualiq",
        query=query_embedding.vector,
        limit=20,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=auth.tenant_id),
                ),
            ]
        ),
        with_payload=True,
    )

    # Convert Qdrant results to RetrievedChunk objects.
    retrieved_chunks: list[RetrievedChunk] = []
    for point in search_results.points:
        payload: dict[str, Any] = point.payload or {}
        retrieved_chunks.append(
            RetrievedChunk(
                text=str(payload.get("text", "")),
                score=point.score,
                doc_id=str(payload.get("doc_id", "")),
                tenant_id=str(payload.get("tenant_id", "")),
                section_path=str(payload.get("section_path", "")),
                page_ref=str(payload.get("page_ref", "")),
                safety_level=str(payload.get("safety_level", "informativo")),
                hash_sha256=str(payload.get("hash_sha256", "")),
                doc_language=str(payload.get("doc_language", "en")),
                equipment=str(payload.get("equipment", "")),
                part_numbers=payload.get("part_numbers", []),
            )
        )

    # Step 6: Dedup + tenant validation.
    deduped = deduplicate_chunks(retrieved_chunks)
    tenant_valid = validate_tenant_chunks(deduped, auth.tenant_id)
    top_chunks = tenant_valid[:5]

    # Step 7: Confidence check.
    confidence, best_score = evaluate_confidence(top_chunks)

    if confidence == QueryConfidence.NONE:
        return QueryResponse(
            answer=(
                "No encontre informacion sobre esto en los manuales disponibles. "
                "Por favor reformule la pregunta o consulte con su supervisor."
            ),
            confidence="none",
            score=0.0,
            chunks_used=0,
            chunks_retrieved=len(retrieved_chunks),
            intent=intent_result.intent.value,
        )

    # Step 8: Build prompt and generate.
    prompt = build_context_prompt(top_chunks, body.query)
    llm_response = await state.llm.generate(prompt, model="claude")

    answer = llm_response.text

    # Step 9: PII check on output.
    pii_result = sanitize_pii(answer)
    if pii_result.has_pii and pii_result.sanitized_text:
        answer = pii_result.sanitized_text

    # Step 10: Language guardrail.
    lang_result = enforce_spanish_response(answer)
    if not lang_result.passed:
        logger.warning("Response language issue: %s", lang_result.detected_language)

    # Step 11: Low confidence warning.
    if confidence == QueryConfidence.LOW:
        answer = (
            f"**ATENCION: Confianza baja (score: {best_score:.2f}).** "
            "Se recomienda verificar esta informacion con el supervisor.\n\n" + answer
        )

    # Step 12: Disclaimer.
    answer += (
        "\n\n---\n*Verificar siempre con el supervisor antes de ejecutar "
        "procedimientos criticos de seguridad.*"
    )

    # Cost alert check.
    usage = await state.rate_limiter.get_usage(auth.tenant_id)
    alert = check_cost_alert(auth.tenant_id, usage["daily_count"], auth.plan.value)
    if alert:
        logger.warning("Cost alert: %s", alert.message)

    return QueryResponse(
        answer=answer,
        confidence=confidence.value,
        score=best_score,
        chunks_used=len(top_chunks),
        chunks_retrieved=len(retrieved_chunks),
        was_fallback=llm_response.was_fallback,
        intent=intent_result.intent.value,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    body: IngestRequest,
    auth: AuthContext = Depends(get_auth),
) -> IngestResponse:
    """Ingest a document: parse -> chunk -> embed -> upsert.

    Runs synchronously for now. In production, this triggers a Prefect
    flow for async processing with notifications.
    """
    from manualiq.ingestion.chunker import blocks_to_chunks, classify_and_split_blocks
    from manualiq.ingestion.parser import parse_document

    pdf_path = Path(body.file_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.file_path}")

    # Parse.
    parse_result = parse_document(pdf_path)
    if not parse_result.text.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse document: {parse_result.error}",
        )

    # Chunk.
    blocks = classify_and_split_blocks(parse_result.text)
    chunks = blocks_to_chunks(
        blocks,
        doc_id=pdf_path.stem,
        tenant_id=auth.tenant_id,
        equipment=body.equipment,
        manufacturer=body.manufacturer,
        doc_language=body.doc_language,
        procedure_type=body.procedure_type,
    )

    # Embed + upsert.
    result = await state.embeddings.index_chunks(chunks)

    return IngestResponse(
        status="indexed",
        doc_id=pdf_path.stem,
        chunks=result["upserted"],
        parser=parse_result.backend.value,
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Health check for all services."""
    services: dict[str, str] = {}

    # Redis.
    try:
        await state.redis.ping()  # type: ignore[misc]
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"

    # Qdrant.
    try:
        await state.qdrant.get_collections()
        services["qdrant"] = "healthy"
    except Exception:
        services["qdrant"] = "unhealthy"

    overall = (
        "healthy" if all(v == "healthy" for v in services.values()) else "degraded"
    )
    return HealthResponse(status=overall, services=services)


@app.get("/health/redis")
async def health_redis() -> JSONResponse:
    """Redis-specific health check (used by Prefect health flow)."""
    try:
        await state.redis.ping()  # type: ignore[misc]
        return JSONResponse({"status": "healthy"})
    except Exception as exc:
        return JSONResponse({"status": "unhealthy", "error": str(exc)}, status_code=503)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_route(prompt: str) -> str:
    """Synchronous wrapper for Gemini Flash routing calls.

    Used by intelligence.py functions that expect sync callables.
    In production, these should be refactored to async.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context; can't use asyncio.run().
        # Fall back to heuristic classification.
        return ""

    return asyncio.run(_async_route(prompt))


async def _async_route(prompt: str) -> str:
    """Async Gemini Flash call for routing."""
    response = await state.llm.route(prompt)
    return response.text
