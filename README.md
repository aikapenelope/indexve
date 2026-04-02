# ManualIQ

Sistema RAG (Retrieval-Augmented Generation) multi-tenant SaaS para manufactura industrial. Permite a tecnicos consultar manuales de equipos en ingles y recibir respuestas precisas en espanol con citas exactas.

**Primer vertical**: Manufactura. **Futuros**: Financiero, Petroleos.

## Estado del proyecto

El proyecto esta en fase de **implementacion de backend**. La documentacion de arquitectura y el analisis de riesgos estan completos. Los modulos core del pipeline RAG estan implementados como librerias Python independientes, listas para integrarse en la aplicacion FastAPI.

### Que existe hoy

```
api/
  Dockerfile                          # Imagen Docker con pre-descarga de modelos Docling
  requirements.txt                    # Dependencias Python pinned
  manualiq/
    ingestion/
      chunker.py                      # Chunker semantico custom para manufactura
      parser.py                       # Docling wrapper con timeout + fallback LlamaParse
      qdrant_init.py                  # Inicializacion de coleccion Qdrant con indexes
    query/
      engine.py                       # Pipeline retrieve -> dedup -> rerank -> threshold
      intelligence.py                 # Sub-questions, intent classifier, multilingual
    middleware/
      rate_limiter.py                 # Rate limiting por tenant/usuario con Redis
      resilience.py                   # Retry, embedding cache, guardrails async, PII
    flows/
      operations.py                   # Prefect v3: re-indexacion, backups, health checks
docs/
  ARCHITECTURE.md                     # Stack tecnico completo y decisiones
  PRD.md                              # Requisitos del producto
  KNOWN_ISSUES.md                     # 29 problemas documentados + roadmap de mitigaciones
docker-compose.yml                    # Configuracion hardened (memory limits, secrets, healthchecks)
```

## Stack tecnico

| Capa | Herramienta | Rol |
|------|-------------|-----|
| PDF Parsing | Docling + LlamaParse (fallback) | Extraccion de texto/tablas de manuales |
| Embeddings | Voyage-4 API | Vectorizacion de chunks (1024 dims) |
| Vector DB | Qdrant OSS 1.13+ | Almacenamiento y busqueda de vectores |
| Re-ranker | Cohere Rerank v3.5 | Mejora precision de retrieval 15-30% |
| Framework RAG | LlamaIndex Workflows 0.14+ | Orquestacion del pipeline RAG |
| LLM principal | Claude Sonnet 4.6 | Generacion de respuestas |
| LLM routing | Gemini 3 Flash | Clasificacion de intencion, expansion de queries |
| Cache | Redis 7 | Embeddings (48h), respuestas LLM (24h) |
| Base de datos | PostgreSQL 16 | Datos de aplicacion (users, sessions, messages) |
| Auth | Clerk | Multi-tenancy nativa |
| Orquestacion | Prefect OSS v3 | Ingestion, re-indexacion, backups, health checks |
| Observabilidad | Arize Phoenix OSS | Tracing + evaluaciones RAG |
| Frontend | Next.js + assistant-ui | Chat UI con streaming (PWA-ready) |
| Guardrails | NeMo Guardrails (NVIDIA) | Prompt injection, PII, tenant isolation |
| Email | Resend | Notificaciones a tecnicos |

## Infraestructura

Un solo VPS Hetzner CX43 (8 vCPU, 16GB RAM, 160GB NVMe, ~$12/mes). Independiente de la infra existente (platform-infra). Costo total estimado: $53-164/mes.

## Desarrollo local

```bash
# 1. Crear directorio de secrets
mkdir -p secrets
echo "your_pg_password" > secrets/pg_password
echo "your_voyage_key" > secrets/voyage_key
echo "your_anthropic_key" > secrets/anthropic_key
echo "your_cohere_key" > secrets/cohere_key
echo "your_google_key" > secrets/google_key
echo "your_clerk_key" > secrets/clerk_key
echo "your_resend_key" > secrets/resend_key

# 2. Levantar servicios
docker compose up -d

# 3. Inicializar coleccion Qdrant (una sola vez)
docker compose exec api python -c "
import asyncio
from manualiq.ingestion.qdrant_init import initialize_collection
asyncio.run(initialize_collection())
"
```

## Documentacion

- [ARCHITECTURE.md](ARCHITECTURE.md) — Stack tecnico, decisiones, distribucion de RAM, costos
- [docs/PRD.md](docs/PRD.md) — Requisitos del producto, usuarios, funcionalidades
- [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) — 29 problemas documentados con mitigaciones + roadmap
- [docs/DIAGRAMS.md](docs/DIAGRAMS.md) — Diagramas de arquitectura, flujos de query/ingestion, experiencia de usuario

## Licencia

Apache License 2.0. Ver [LICENSE](LICENSE).
