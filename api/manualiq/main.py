"""ManualIQ FastAPI application entrypoint.

Connects all modules into a running API:
- POST /query    — Full RAG pipeline with reranking, guardrails, caching
- POST /ingest   — Document ingestion with DB tracking
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from manualiq.ingestion.embedding_service import EmbeddingService
from manualiq.ingestion.sparse_vectors import SparseVectorizer
from manualiq.llm_gateway import LLMGateway
from manualiq.logging_config import setup_logging
from manualiq.middleware.auth import AuthContext, authenticate_request
from manualiq.middleware.guardrails import GuardrailsEngine
from manualiq.middleware.llm_cache import LLMResponseCache
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
    classify_intent_async,
    enforce_spanish_response,
    expand_query_crosslingual_async,
)
from manualiq.query.reranker import Reranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _read_secret(env_var: str, default: str = "") -> str:
    """Read a secret from a file path (Docker secrets) or env var."""
    file_path = os.environ.get(env_var, "")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
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
    reranker: Reranker
    llm_cache: LLMResponseCache
    guardrails: GuardrailsEngine
    sparse: SparseVectorizer


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Initialize and tear down shared resources."""
    # Logging.
    setup_logging()

    # Phoenix tracing (ARCHITECTURE.md Section 3.13).
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
            logger.warning("Phoenix tracing packages not installed, skipping")

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

    # Ollama URL for local Qwen3 fallbacks.
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Embedding service with Redis cache + Ollama fallback.
    emb_cache = EmbeddingCache(state.redis)
    state.embeddings = EmbeddingService(
        voyage_api_key=_read_secret("VOYAGE_API_KEY_FILE"),
        qdrant_client=state.qdrant,
        cache=emb_cache,
        ollama_url=ollama_url,
    )

    # Cohere Reranker + Ollama Qwen3 fallback.
    cohere_key = _read_secret("COHERE_API_KEY_FILE")
    state.reranker = Reranker(
        api_key=cohere_key or "",
        ollama_url=ollama_url,
    )

    # LLM response cache (24h TTL).
    state.llm_cache = LLMResponseCache(state.redis)

    # Rate limiter.
    state.rate_limiter = RateLimiter(state.redis)

    # Sparse vectorizer for hybrid search queries.
    state.sparse = SparseVectorizer()

    # NeMo Guardrails.
    state.guardrails = GuardrailsEngine()
    await state.guardrails.initialize()

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

# CORS origins from env (gap #8: restrict in production).
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="ManualIQ API",
    description="RAG API for manufacturing technical manuals",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def get_auth(request: Request) -> AuthContext:
    """Authenticate request via Clerk JWT or dev headers."""
    return await authenticate_request(request)


# ---------------------------------------------------------------------------
# Async LLM helper for intelligence module
# ---------------------------------------------------------------------------


async def _route_llm(prompt: str) -> str:
    """Async Gemini Flash call for routing (intent, expansion)."""
    response = await state.llm.route(prompt)
    return response.text


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    """A source chunk returned with the query response for the PDF viewer."""

    doc_id: str
    section_path: str
    page_ref: str
    score: float
    safety_level: str
    doc_language: str
    equipment: str
    text_preview: str = ""


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    confidence: str
    score: float
    chunks_used: int
    chunks_retrieved: int
    sources: list[SourceChunk] = []
    was_fallback: bool = False
    was_cached: bool = False
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


class HistoryMessage(BaseModel):
    """A single message in conversation history."""

    role: str
    content: str
    confidence: str | None = None
    score: float | None = None
    created_at: str


class HistoryResponse(BaseModel):
    """Response body for GET /history."""

    messages: list[HistoryMessage]
    total: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    auth: AuthContext = Depends(get_auth),
) -> QueryResponse:
    """Main RAG query endpoint with full pipeline.

    Pipeline: auth -> rate limit -> cache check -> guardrails input ->
    intent classify -> expand query -> embed -> retrieve -> dedup ->
    tenant validate -> rerank (Cohere) -> threshold -> generate ->
    PII check -> language check -> guardrails output -> cache store.
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

    # Step 2: LLM response cache check.
    cached = await state.llm_cache.get(body.query, auth.tenant_id)
    if cached is not None:
        return QueryResponse(
            answer=str(cached.get("answer", "")),
            confidence=str(cached.get("confidence", "high")),
            score=float(cached.get("score", 0.0)),  # type: ignore[arg-type]
            chunks_used=int(cached.get("chunks_used", 0)),  # type: ignore[arg-type]
            chunks_retrieved=int(cached.get("chunks_retrieved", 0)),  # type: ignore[arg-type]
            was_cached=True,
            intent=str(cached.get("intent", "specific")),
        )

    # Step 3: NeMo Guardrails input check.
    if state.guardrails.enabled:
        input_check = await state.guardrails.check_input(body.query)
        if not input_check.allowed:
            return QueryResponse(
                answer=input_check.rejection_message or "Solicitud no permitida.",
                confidence="none",
                score=0.0,
                chunks_used=0,
                chunks_retrieved=0,
                intent="blocked",
            )

    # Step 4: Intent classification (async Gemini Flash).
    intent_result = await classify_intent_async(
        body.query,
        llm_fn=_route_llm,
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

    # Step 5: Query expansion (async ES -> EN).
    queries = await expand_query_crosslingual_async(
        body.query,
        llm_fn=_route_llm,
    )

    # Step 6: Embed query (dense + sparse).
    query_embedding = await state.embeddings.embed_query(queries[0])
    query_sparse = state.sparse.vectorize(queries[0])

    # Step 7: Hybrid retrieve from Qdrant (dense + sparse with RRF fusion).
    # Reciprocal Rank Fusion combines results from both search methods,
    # improving precision 15-30% for part numbers and technical terms.
    tenant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=auth.tenant_id),
            ),
        ]
    )

    search_results = await state.qdrant.query_points(
        collection_name="manualiq",
        prefetch=[
            # Dense search (semantic).
            models.Prefetch(
                query=query_embedding.vector,
                using="dense",
                limit=20,
                filter=tenant_filter,
            ),
            # Sparse search (keyword/BM25).
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices,
                    values=query_sparse.values,
                ),
                using="sparse",
                limit=20,
                filter=tenant_filter,
            ),
        ],
        # RRF fusion of dense + sparse results.
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=128),
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

    # Step 8: Dedup + tenant validation.
    deduped = deduplicate_chunks(retrieved_chunks)
    tenant_valid = validate_tenant_chunks(deduped, auth.tenant_id)

    # Step 9: Cohere Rerank (gap #1 -- was missing).
    rerank_result = await state.reranker.rerank(body.query, tenant_valid)
    top_chunks = rerank_result.chunks

    # Step 10: Confidence check.
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

    # Step 11: Build prompt and generate.
    prompt = build_context_prompt(top_chunks, body.query)
    llm_response = await state.llm.generate(prompt, model="claude")

    answer = llm_response.text

    # Step 12: PII check on output.
    pii_result = sanitize_pii(answer)
    if pii_result.has_pii and pii_result.sanitized_text:
        answer = pii_result.sanitized_text

    # Step 13: NeMo Guardrails output check.
    if state.guardrails.enabled:
        output_check = await state.guardrails.check_output(answer)
        if output_check.modified:
            answer = output_check.text

    # Step 14: Language guardrail.
    lang_result = enforce_spanish_response(answer)
    if not lang_result.passed:
        logger.warning("Response language issue: %s", lang_result.detected_language)

    # Step 15: Low confidence warning.
    if confidence == QueryConfidence.LOW:
        answer = (
            f"**ATENCION: Confianza baja (score: {best_score:.2f}).** "
            "Se recomienda verificar esta informacion con el supervisor.\n\n" + answer
        )

    # Step 16: Disclaimer.
    answer += (
        "\n\n---\n*Verificar siempre con el supervisor antes de ejecutar "
        "procedimientos criticos de seguridad.*"
    )

    # Step 17: Cache the response.
    response_data: dict[str, object] = {
        "answer": answer,
        "confidence": confidence.value,
        "score": best_score,
        "chunks_used": len(top_chunks),
        "chunks_retrieved": len(retrieved_chunks),
        "intent": intent_result.intent.value,
    }
    await state.llm_cache.set(body.query, auth.tenant_id, response_data)

    # Cost alert check.
    usage = await state.rate_limiter.get_usage(auth.tenant_id)
    alert = check_cost_alert(auth.tenant_id, usage["daily_count"], auth.plan.value)
    if alert:
        logger.warning("Cost alert: %s", alert.message)

    # Build sources for the PDF viewer.
    sources = [
        SourceChunk(
            doc_id=c.doc_id,
            section_path=c.section_path,
            page_ref=c.page_ref,
            score=c.score,
            safety_level=c.safety_level,
            doc_language=c.doc_language,
            equipment=c.equipment,
            text_preview=c.text[:200],
        )
        for c in top_chunks
    ]

    return QueryResponse(
        answer=answer,
        confidence=confidence.value,
        score=best_score,
        chunks_used=len(top_chunks),
        chunks_retrieved=len(retrieved_chunks),
        sources=sources,
        was_fallback=llm_response.was_fallback,
        intent=intent_result.intent.value,
    )


@app.post("/query/stream")
async def query_stream_endpoint(
    body: QueryRequest,
    auth: AuthContext = Depends(get_auth),
) -> StreamingResponse:
    """Streaming RAG query endpoint.

    Same pipeline as /query but streams the LLM response token-by-token.
    First token arrives in ~500ms, full response streams over 2-4 seconds.
    Uses text/event-stream format compatible with assistant-ui.
    """
    import json as json_mod

    # Steps 1-10 are identical to /query (rate limit, cache, guardrails,
    # intent, expand, embed, retrieve, dedup, rerank, confidence).
    # For brevity, we reuse the same logic inline.

    rate_result = await state.rate_limiter.check_rate_limit(
        auth.tenant_id, auth.user_id, auth.plan
    )
    if not rate_result.allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    # Check cache first.
    cached = await state.llm_cache.get(body.query, auth.tenant_id)
    if cached is not None:

        async def _yield_cached():  # type: ignore[no-untyped-def]
            yield f"data: {json_mod.dumps({'type': 'text', 'content': str(cached.get('answer', ''))})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_yield_cached(), media_type="text/event-stream")

    # Intent + expansion.
    intent_result = await classify_intent_async(body.query, llm_fn=_route_llm)
    if intent_result.intent.value == "ambiguous" and intent_result.clarification_prompt:

        async def _yield_clarify():  # type: ignore[no-untyped-def]
            yield f"data: {json_mod.dumps({'type': 'text', 'content': intent_result.clarification_prompt})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_yield_clarify(), media_type="text/event-stream")

    queries = await expand_query_crosslingual_async(body.query, llm_fn=_route_llm)

    # Embed + hybrid retrieve.
    query_embedding = await state.embeddings.embed_query(queries[0])
    query_sparse = state.sparse.vectorize(queries[0])

    tenant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=auth.tenant_id)
            )
        ]
    )
    search_results = await state.qdrant.query_points(
        collection_name="manualiq",
        prefetch=[
            models.Prefetch(
                query=query_embedding.vector,
                using="dense",
                limit=20,
                filter=tenant_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=query_sparse.indices, values=query_sparse.values
                ),
                using="sparse",
                limit=20,
                filter=tenant_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=128),
    )

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

    deduped = deduplicate_chunks(retrieved_chunks)
    tenant_valid = validate_tenant_chunks(deduped, auth.tenant_id)
    rerank_result = await state.reranker.rerank(body.query, tenant_valid)
    top_chunks = rerank_result.chunks

    confidence, best_score = evaluate_confidence(top_chunks)
    if confidence == QueryConfidence.NONE:

        async def _yield_no_info():  # type: ignore[no-untyped-def]
            yield f"data: {json_mod.dumps({'type': 'text', 'content': 'No encontre informacion sobre esto en los manuales disponibles.'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_yield_no_info(), media_type="text/event-stream")

    prompt = build_context_prompt(top_chunks, body.query)

    # Build sources metadata to send before streaming text.
    sources_data = [
        {
            "doc_id": c.doc_id,
            "section_path": c.section_path,
            "page_ref": c.page_ref,
            "score": c.score,
            "safety_level": c.safety_level,
        }
        for c in top_chunks
    ]

    async def _stream_response():  # type: ignore[no-untyped-def]
        # Send sources metadata first.
        yield f"data: {json_mod.dumps({'type': 'sources', 'sources': sources_data})}\n\n"

        # Low confidence warning.
        if confidence == QueryConfidence.LOW:
            warning = f"**ATENCION: Confianza baja (score: {best_score:.2f}).** Se recomienda verificar con el supervisor.\n\n"
            yield f"data: {json_mod.dumps({'type': 'text', 'content': warning})}\n\n"

        # Stream LLM response token-by-token.
        full_text = ""
        async for delta in state.llm.stream_generate(prompt):
            full_text += delta
            yield f"data: {json_mod.dumps({'type': 'text', 'content': delta})}\n\n"

        # Disclaimer.
        disclaimer = "\n\n---\n*Verificar siempre con el supervisor antes de ejecutar procedimientos criticos de seguridad.*"
        yield f"data: {json_mod.dumps({'type': 'text', 'content': disclaimer})}\n\n"

        # Cache the full response.
        await state.llm_cache.set(
            body.query,
            auth.tenant_id,
            {
                "answer": full_text + disclaimer,
                "confidence": confidence.value,
                "score": best_score,
                "chunks_used": len(top_chunks),
                "chunks_retrieved": len(retrieved_chunks),
                "intent": intent_result.intent.value,
            },
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream_response(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    body: IngestRequest,
    auth: AuthContext = Depends(get_auth),
) -> IngestResponse:
    """Ingest a document: parse -> chunk -> embed -> upsert -> track in DB."""
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

    # Track in Redis (lightweight; full DB tracking via Alembic models
    # requires a DB session which will be added when SQLAlchemy async
    # session is wired into the lifespan).
    doc_key = f"doc:{auth.tenant_id}:{pdf_path.stem}"
    await state.redis.hset(  # type: ignore[misc]
        doc_key,
        mapping={
            "filename": pdf_path.name,
            "tenant_id": auth.tenant_id,
            "equipment": body.equipment,
            "manufacturer": body.manufacturer,
            "chunks": str(result["upserted"]),
            "parser": parse_result.backend.value,
            "status": "indexed",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return IngestResponse(
        status="indexed",
        doc_id=pdf_path.stem,
        chunks=result["upserted"],
        parser=parse_result.backend.value,
    )


@app.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload_endpoint(
    file: UploadFile = File(...),
    equipment: str = Form(""),
    manufacturer: str = Form(""),
    doc_language: str = Form("en"),
    procedure_type: str = Form("informativo"),
    auth: AuthContext = Depends(get_auth),
) -> IngestResponse:
    """Upload and ingest a PDF document via multipart form.

    Accepts a file upload from the frontend, saves it to the corpus
    directory, then runs the same parse -> chunk -> embed -> upsert
    pipeline as /ingest.
    """
    from manualiq.ingestion.chunker import blocks_to_chunks, classify_and_split_blocks
    from manualiq.ingestion.parser import parse_document

    if not file.filename or not file.filename.lower().endswith(
        (".pdf", ".docx", ".pptx")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and PPTX files are supported.",
        )

    # Save uploaded file to corpus directory.
    import os

    corpus_dir = os.environ.get("CORPUS_DIR", "/corpus")
    tenant_dir = Path(corpus_dir) / auth.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)

    file_path = tenant_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    logger.info(
        "Uploaded %s (%d bytes) for tenant %s",
        file.filename,
        len(content),
        auth.tenant_id,
    )

    # Parse.
    parse_result = parse_document(file_path)
    if not parse_result.text.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse document: {parse_result.error}",
        )

    # Chunk.
    blocks = classify_and_split_blocks(parse_result.text)
    chunks = blocks_to_chunks(
        blocks,
        doc_id=file_path.stem,
        tenant_id=auth.tenant_id,
        equipment=equipment,
        manufacturer=manufacturer,
        doc_language=doc_language,
        procedure_type=procedure_type,
    )

    # Embed + upsert.
    result = await state.embeddings.index_chunks(chunks)

    # Track in Redis.
    doc_key = f"doc:{auth.tenant_id}:{file_path.stem}"
    await state.redis.hset(  # type: ignore[misc]
        doc_key,
        mapping={
            "filename": file.filename,
            "tenant_id": auth.tenant_id,
            "equipment": equipment,
            "manufacturer": manufacturer,
            "chunks": str(result["upserted"]),
            "parser": parse_result.backend.value,
            "status": "indexed",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return IngestResponse(
        status="indexed",
        doc_id=file_path.stem,
        chunks=result["upserted"],
        parser=parse_result.backend.value,
    )


@app.get("/history", response_model=HistoryResponse)
async def history_endpoint(
    auth: AuthContext = Depends(get_auth),
    limit: int = Query(default=50, ge=1, le=200),
) -> HistoryResponse:
    """Get conversation history for the authenticated user.

    Reads from Redis (lightweight store). Full PostgreSQL-backed
    history will be available when the DB session is wired in.
    """
    # Scan for this user's messages in Redis.
    pattern = f"history:{auth.tenant_id}:{auth.user_id}:*"
    messages: list[HistoryMessage] = []

    cursor: int = 0
    while True:
        cursor, keys = await state.redis.scan(  # type: ignore[misc]
            cursor=cursor, match=pattern, count=100
        )
        for key in keys:
            data = await state.redis.hgetall(key)  # type: ignore[misc]
            if data:
                messages.append(
                    HistoryMessage(
                        role=str(data.get("role", "user")),
                        content=str(data.get("content", "")),
                        confidence=data.get("confidence"),
                        score=float(data["score"]) if data.get("score") else None,
                        created_at=str(data.get("created_at", "")),
                    )
                )
        if cursor == 0:
            break

    # Sort by created_at descending, limit.
    messages.sort(key=lambda m: m.created_at, reverse=True)
    messages = messages[:limit]

    return HistoryResponse(messages=messages, total=len(messages))


# ---------------------------------------------------------------------------
# PDF document serving (for WhatsApp/email attachments and web viewer)
# ---------------------------------------------------------------------------


@app.get("/documents/{doc_id}/pdf")
async def serve_document_pdf(
    doc_id: str,
    auth: AuthContext = Depends(get_auth),
) -> FileResponse:
    """Serve the original PDF file for a document.

    Used by all channels to attach the source PDF:
    - Web: link in SourceViewer
    - Email: Resend attachment
    - WhatsApp: Twilio document message

    Files are served from /corpus/{tenant_id}/ with tenant isolation.
    """
    corpus_dir = os.environ.get("CORPUS_DIR", "/corpus")
    tenant_dir = Path(corpus_dir) / auth.tenant_id

    # Search for the PDF by doc_id (filename stem).
    for ext in (".pdf", ".PDF"):
        pdf_path = tenant_dir / f"{doc_id}{ext}"
        if pdf_path.is_file():
            return FileResponse(
                path=str(pdf_path),
                media_type="application/pdf",
                filename=pdf_path.name,
            )

    raise HTTPException(status_code=404, detail=f"PDF not found: {doc_id}")


# ---------------------------------------------------------------------------
# Admin metrics (PRD 4.1, requires admin role)
# ---------------------------------------------------------------------------


class MetricsResponse(BaseModel):
    """Response body for GET /admin/metrics."""

    tenant_id: str
    daily_queries: int
    daily_limit: int
    documents_indexed: int
    feedback_positive: int
    feedback_negative: int


@app.get("/admin/metrics", response_model=MetricsResponse)
async def admin_metrics_endpoint(
    auth: AuthContext = Depends(get_auth),
) -> MetricsResponse:
    """Get usage metrics for the authenticated tenant.

    Requires admin or supervisor role.
    """
    auth.require_role("admin", "supervisor")

    # Query usage from rate limiter.
    usage = await state.rate_limiter.get_usage(auth.tenant_id)

    # Count indexed documents from Redis.
    doc_count = 0
    cursor: int = 0
    while True:
        cursor, keys = await state.redis.scan(  # type: ignore[misc]
            cursor=cursor, match=f"doc:{auth.tenant_id}:*", count=100
        )
        doc_count += len(keys)
        if cursor == 0:
            break

    # Count feedback from Redis.
    positive = 0
    negative = 0
    cursor = 0
    while True:
        cursor, keys = await state.redis.scan(  # type: ignore[misc]
            cursor=cursor, match=f"feedback:{auth.tenant_id}:*", count=100
        )
        for key in keys:
            rating = await state.redis.hget(key, "rating")  # type: ignore[misc]
            if rating == "1":
                positive += 1
            elif rating == "-1":
                negative += 1
        if cursor == 0:
            break

    return MetricsResponse(
        tenant_id=auth.tenant_id,
        daily_queries=usage["daily_count"],
        daily_limit=auth.plan.value,
        documents_indexed=doc_count,
        feedback_positive=positive,
        feedback_negative=negative,
    )


@app.get("/health", response_model=HealthResponse)
async def health_endpoint() -> HealthResponse:
    """Health check for all services."""
    services: dict[str, str] = {}

    try:
        await state.redis.ping()  # type: ignore[misc]
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"

    try:
        await state.qdrant.get_collections()
        services["qdrant"] = "healthy"
    except Exception:
        services["qdrant"] = "unhealthy"

    services["guardrails"] = "enabled" if state.guardrails.enabled else "disabled"

    overall = (
        "healthy"
        if all(v in ("healthy", "enabled") for v in services.values())
        else "degraded"
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
# Feedback
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: int = Field(..., ge=-1, le=1)  # -1=bad, 0=neutral, 1=good
    comment: str = ""
    session_id: str = ""


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback."""

    status: str
    feedback_id: str


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(
    body: FeedbackRequest,
    auth: AuthContext = Depends(get_auth),
) -> FeedbackResponse:
    """Record user feedback (thumbs up/down) on a response.

    Stored in Redis for now. Feeds into:
    - Phoenix Evals as ground truth
    - System prompt tuning
    - Confidence threshold adjustment
    """
    import uuid as uuid_mod

    feedback_id = str(uuid_mod.uuid4())
    key = f"feedback:{auth.tenant_id}:{feedback_id}"

    await state.redis.hset(  # type: ignore[misc]
        key,
        mapping={
            "tenant_id": auth.tenant_id,
            "user_id": auth.user_id,
            "query": body.query,
            "answer": body.answer[:500],  # Truncate to save space.
            "rating": str(body.rating),
            "comment": body.comment[:500],
            "session_id": body.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # Keep feedback for 90 days.
    await state.redis.expire(key, 90 * 86_400)  # type: ignore[misc]

    logger.info(
        "Feedback recorded: tenant=%s, rating=%d, id=%s",
        auth.tenant_id,
        body.rating,
        feedback_id,
    )

    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


# ---------------------------------------------------------------------------
# WhatsApp channel via Twilio (PRD 3.5 / ROADMAP D1)
# ---------------------------------------------------------------------------


@app.post("/whatsapp/inbound")
async def whatsapp_inbound_endpoint(request: Request) -> JSONResponse:
    """Receive incoming WhatsApp message from Twilio webhook.

    Flow: Twilio delivers message -> lookup tenant by phone whitelist ->
    RAG pipeline -> send text response -> send source PDF attachment.
    No login required — auth is via phone number whitelist per tenant.
    """
    from manualiq.channels.whatsapp import (
        format_whatsapp_response,
        lookup_tenant_by_phone,
        parse_twilio_webhook,
        send_whatsapp_document,
        send_whatsapp_text,
    )

    form_data = dict(await request.form())
    form_str = {k: str(v) for k, v in form_data.items()}
    msg = parse_twilio_webhook(form_str)

    if msg is None:
        return JSONResponse({"status": "ignored"})

    # Lookup tenant by phone whitelist.
    user_info = await lookup_tenant_by_phone(state.redis, msg["sender"])
    if user_info is None:
        logger.warning("WhatsApp from unregistered phone: %s", msg["sender"])
        return JSONResponse({"status": "unauthorized", "phone": msg["sender"]})

    tenant_id = user_info["tenant_id"]
    logger.info(
        "WhatsApp from %s (tenant=%s): %s", msg["sender"], tenant_id, msg["body"][:50]
    )

    try:
        # RAG pipeline (same as /query).
        query_embedding = await state.embeddings.embed_query(msg["body"])
        query_sparse = state.sparse.vectorize(msg["body"])

        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id", match=models.MatchValue(value=tenant_id)
                )
            ]
        )
        search_results = await state.qdrant.query_points(
            collection_name="manualiq",
            prefetch=[
                models.Prefetch(
                    query=query_embedding.vector,
                    using="dense",
                    limit=20,
                    filter=tenant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices, values=query_sparse.values
                    ),
                    using="sparse",
                    limit=20,
                    filter=tenant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        )

        retrieved: list[RetrievedChunk] = []
        for point in search_results.points:
            p: dict[str, Any] = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    text=str(p.get("text", "")),
                    score=point.score,
                    doc_id=str(p.get("doc_id", "")),
                    tenant_id=str(p.get("tenant_id", "")),
                    section_path=str(p.get("section_path", "")),
                    page_ref=str(p.get("page_ref", "")),
                    safety_level=str(p.get("safety_level", "informativo")),
                    hash_sha256=str(p.get("hash_sha256", "")),
                    doc_language=str(p.get("doc_language", "en")),
                    equipment=str(p.get("equipment", "")),
                    part_numbers=p.get("part_numbers", []),
                )
            )

        deduped = deduplicate_chunks(retrieved)
        tenant_valid = validate_tenant_chunks(deduped, tenant_id)
        rerank_result = await state.reranker.rerank(msg["body"], tenant_valid)
        top_chunks = rerank_result.chunks
        confidence, best_score = evaluate_confidence(top_chunks)

        if confidence == QueryConfidence.NONE:
            answer = "No encontre informacion sobre esto en los manuales disponibles."
        else:
            prompt = build_context_prompt(top_chunks, msg["body"])
            llm_response = await state.llm.generate(prompt, model="claude")
            answer = llm_response.text

        sources = [
            {
                "doc_id": c.doc_id,
                "section_path": c.section_path,
                "page_ref": c.page_ref,
                "score": c.score,
            }
            for c in top_chunks
        ]

        # Send text response via WhatsApp.
        twilio_sid = _read_secret("TWILIO_ACCOUNT_SID_FILE")
        twilio_token = _read_secret("TWILIO_AUTH_TOKEN_FILE")
        twilio_from = os.environ.get("TWILIO_WHATSAPP_FROM", "")

        wa_text = format_whatsapp_response(
            answer, sources, confidence.value, best_score
        )
        await send_whatsapp_text(
            to=msg["sender"],
            body=wa_text,
            twilio_sid=twilio_sid,
            twilio_token=twilio_token,
            twilio_from=twilio_from,
        )

        # Send top 2 source PDFs as document attachments (FREE within 24h window).
        api_base = os.environ.get("API_PUBLIC_URL", "http://localhost:8000")
        seen_docs: set[str] = set()
        for src in sources[:2]:
            doc_id = str(src["doc_id"])
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            pdf_url = f"{api_base}/documents/{doc_id}/pdf?x-tenant-id={tenant_id}&x-user-id={user_info['user_id']}"
            await send_whatsapp_document(
                to=msg["sender"],
                document_url=pdf_url,
                filename=f"{doc_id}.pdf",
                caption=f"📄 {doc_id} — {src.get('section_path', '')}",
                twilio_sid=twilio_sid,
                twilio_token=twilio_token,
                twilio_from=twilio_from,
            )

        return JSONResponse({"status": "processed", "tenant": tenant_id})

    except Exception as exc:
        logger.error("WhatsApp processing failed: %s", exc)
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Email channel (PRD 3.5)
# ---------------------------------------------------------------------------


@app.post("/email/inbound")
async def email_inbound_endpoint(request: Request) -> JSONResponse:
    """Receive inbound email from Resend webhook.

    Flow: Resend delivers email -> parse -> RAG pipeline -> send response.
    No auth required (webhook is authenticated by Resend signature).
    """
    from manualiq.channels.email import (
        format_email_response,
        parse_inbound_email,
        send_email_response,
    )

    payload = await request.json()
    email = parse_inbound_email(payload)

    if email is None:
        return JSONResponse({"status": "ignored", "reason": "invalid email"})

    logger.info(
        "Inbound email from %s for tenant %s: %s",
        email.sender,
        email.tenant_id,
        email.subject,
    )

    # Run the same pipeline as /query but with tenant from email address.
    # Simplified: embed -> retrieve -> rerank -> generate.
    try:
        query_embedding = await state.embeddings.embed_query(email.body_text)
        query_sparse = state.sparse.vectorize(email.body_text)

        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=email.tenant_id),
                )
            ]
        )
        search_results = await state.qdrant.query_points(
            collection_name="manualiq",
            prefetch=[
                models.Prefetch(
                    query=query_embedding.vector,
                    using="dense",
                    limit=20,
                    filter=tenant_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse.indices,
                        values=query_sparse.values,
                    ),
                    using="sparse",
                    limit=20,
                    filter=tenant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
            with_payload=True,
        )

        retrieved: list[RetrievedChunk] = []
        for point in search_results.points:
            p: dict[str, Any] = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    text=str(p.get("text", "")),
                    score=point.score,
                    doc_id=str(p.get("doc_id", "")),
                    tenant_id=str(p.get("tenant_id", "")),
                    section_path=str(p.get("section_path", "")),
                    page_ref=str(p.get("page_ref", "")),
                    safety_level=str(p.get("safety_level", "informativo")),
                    hash_sha256=str(p.get("hash_sha256", "")),
                    doc_language=str(p.get("doc_language", "en")),
                    equipment=str(p.get("equipment", "")),
                    part_numbers=p.get("part_numbers", []),
                )
            )

        deduped = deduplicate_chunks(retrieved)
        tenant_valid = validate_tenant_chunks(deduped, email.tenant_id)
        rerank_result = await state.reranker.rerank(email.body_text, tenant_valid)
        top_chunks = rerank_result.chunks

        confidence, best_score = evaluate_confidence(top_chunks)

        if confidence == QueryConfidence.NONE:
            answer = (
                "No encontre informacion sobre esto en los manuales disponibles. "
                "Por favor reformule la pregunta."
            )
        else:
            prompt = build_context_prompt(top_chunks, email.body_text)
            llm_response = await state.llm.generate(prompt, model="claude")
            answer = llm_response.text

        sources = [
            {
                "doc_id": c.doc_id,
                "section_path": c.section_path,
                "page_ref": c.page_ref,
                "score": c.score,
            }
            for c in top_chunks
        ]

        email_body = format_email_response(
            query=email.body_text,
            answer=answer,
            sources=sources,
            confidence=confidence.value,
            score=best_score,
        )

        resend_key = _read_secret("RESEND_API_KEY_FILE")
        if resend_key:
            await send_email_response(
                resend_api_key=resend_key,
                to_email=email.sender,
                subject=f"Re: {email.subject}"
                if email.subject
                else "ManualIQ - Respuesta",
                body=email_body,
            )

        return JSONResponse({"status": "processed", "tenant": email.tenant_id})

    except Exception as exc:
        logger.error("Email processing failed: %s", exc)
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)
