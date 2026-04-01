# ManualIQ — Arquitectura RAG Enterprise para Manufactura

> Documento de planificacion tecnica. Marzo 2026.
> Sistema RAG multi-tenant para vender como SaaS institucional.
> Primer vertical: manufactura. Futuros: financiero, petroleos.

---

## 1. PROBLEMA QUE RESUELVE

Empresas manufactureras (Venezuela, LATAM) operan con equipos industriales importados cuyos manuales estan en ingles. Sus tecnicos tienen dominio limitado del ingles tecnico.

- **Antes**: Tecnico busca en PDF manualmente → no encuentra o malinterpreta → consulta supervisor → 20-60 minutos.
- **Con ManualIQ**: Tecnico pregunta en espanol → respuesta en espanol con cita exacta de pagina → menos de 5 segundos.

### Lo que el sistema NO hace
- No procesa datos personales de empleados ni clientes finales
- No toma decisiones autonomas de mantenimiento (asiste, no decide)
- No reemplaza al supervisor para procedimientos criticos de seguridad
- No accede a sistemas ERP/SCADA (solo lee documentos del corpus)

---

## 2. CORPUS DOCUMENTAL

| Tipo | Volumen Est. | Idioma | Reto Principal |
|------|-------------|--------|----------------|
| Manuales equipos | 500+ | EN | Tablas, torque specs, part numbers |
| Programas de MP | 1,000+ | ES/EN | Procedimientos secuenciales ordenados |
| SOPs | 500+ | ES | Referencias cruzadas internas |
| OTs historicas | 5,000+ | ES | Semi-estructurado, sin schema fijo |
| Normas COVENIN/ISO | 100+ | ES | Lenguaje regulatorio, jerarquia normativa |

### Reglas de Chunking
- Unidad minima = 1 paso de procedimiento completo. NUNCA dividir un paso a la mitad.
- Prefixar chunk con ruta de seccion: `[Manual > Cap > Seccion] + contenido`.
- Tablas de specs = chunk completo independiente. NUNCA dividir una tabla.
- Part numbers y codigos siempre en el mismo chunk que su descripcion.
- Referencias cruzadas ("ver paso 4.2.3") → resolver en indexacion, incluir chunk referenciado como `related_chunks` en metadata.
- Tamano objetivo: 400-600 tokens. Maximo: 800 tokens (tablas pueden exceder).
- Overlap: 80 tokens SOLO para texto corrido. Cero overlap para tablas/procedimientos.

### Metadata por Chunk
```json
{
  "doc_id":          "caterpillar_c7_service_manual_2019",
  "tenant_id":       "empresa_manufacturera_001",
  "equipment":       "Motor Caterpillar C7",
  "manufacturer":    "Caterpillar",
  "doc_language":    "en",
  "section_path":    "4.3 > Cylinder Head Installation",
  "procedure_type":  "mantenimiento_preventivo | reparacion | diagnostico | operacion",
  "safety_level":    "critico | precaucion | informativo",
  "part_numbers":    ["AB-4521-CX"],
  "related_chunks":  ["caterpillar_c7_manual_chunk_042"],
  "page_ref":        "pp. 4-18 to 4-20",
  "chunk_index":     15,
  "total_chunks":    87,
  "hash_sha256":     "a3f8b2c...",
  "indexed_at":      "2026-03-15T10:00:00Z"
}
```

---

## 3. STACK TECNICO DEFINITIVO

### 3.1 PDF Parsing: Docling (primario) + LlamaParse (fallback)

| Herramienta | Precision Tablas | Velocidad | Costo |
|-------------|-----------------|-----------|-------|
| **Docling** (IBM OSS) | 97.9% | Media | $0 |
| LlamaParse (fallback) | ~85% | Rapida (6s/doc) | $0 (LiteParse local, marzo 2026) |

- Docling como parser primario: mejor precision en datos numericos y tablas complejas.
- LlamaParse LiteParse como fallback para documentos escaneados con baja resolucion.
- Enfoque ensemble: correr ambos y elegir mejor output via scoring para documentos criticos.

### 3.2 Embeddings: Voyage-4 API (recomendado) o Qwen3-Embedding-0.6B (self-hosted)

#### Respuesta: GPU vs CPU para Qwen3-Embedding

**Qwen3-Embedding-0.6B funciona en CPU**, pero con limitaciones:
- Modelo Q8_0 via Ollama: **639MB en disco**, ~1.5GB RAM en ejecucion.
- En CPU (8 vCPU AMD EPYC): ~50-100ms por embedding individual, aceptable para queries en tiempo real.
- Para ingestion masiva (re-indexar 7000+ documentos): CPU sera lento (horas). GPU lo haria en minutos.
- **No necesita GPU para servir queries**, pero la ingestion inicial sera lenta.

#### Comparativa de opciones de embeddings

| Modelo | MTEB Score | Multilingue ES/EN | Costo | Self-hosted | Contexto |
|--------|-----------|-------------------|-------|-------------|----------|
| **Voyage-4** (API) | ~66.8 | Si, 100+ idiomas | $0.06/1M tokens | No | 32K tokens |
| **Voyage-4-lite** (API) | ~63 | Si | $0.02/1M tokens | No | 32K tokens |
| Qwen3-Embedding-0.6B | ~63 | Si, 100+ idiomas | $0 (self-hosted) | Si, CPU | 32K tokens |
| OpenAI text-embedding-3-large | 64.6 | Si | $0.13/1M tokens | No | 8K tokens |
| OpenAI text-embedding-3-small | ~62 | Si | $0.02/1M tokens | No | 8K tokens |
| Cohere embed-v4 | 65.2 | Si, 100+ idiomas | $0.10/1M tokens | No | 128K tokens |
| BGE-M3 (original propuesto) | 63.0 | Si, 100+ idiomas | $0 (self-hosted) | Si, CPU | 8K tokens |

#### Recomendacion final: Voyage-4 API

**Razon**: Para un VPS de 16GB sin GPU, usar embeddings via API es la decision correcta:
- **Voyage-4** ($0.06/1M tokens) con 200M tokens gratis al registrarse.
- Mejor calidad que OpenAI para retrieval (14% superior en RTEB benchmark).
- 32K tokens de contexto (vs 8K de OpenAI) = menos chunking agresivo.
- Shared embedding space: puedes usar voyage-4-lite para queries ($0.02/1M) y voyage-4 para indexacion.
- **Costo estimado**: Con ~100K chunks de 500 tokens = 50M tokens de indexacion = ~$3. Queries mensuales ~10M tokens = ~$0.60/mes.

**Alternativa self-hosted** (si quieres $0 en embeddings): Qwen3-Embedding-0.6B via Ollama en CPU. Funciona, pero consume ~1.5GB RAM permanente y la ingestion masiva sera lenta.

**OpenAI embeddings**: Funcional pero inferior a Voyage en retrieval y mas caro (text-embedding-3-large a $0.13/1M). El contexto de 8K tokens fuerza chunking mas agresivo. No recomendado como primera opcion.

### 3.3 Vector DB: Qdrant OSS

Sin cambios. Qdrant 1.13+ con:
- Tiered multitenancy (1 shard key por empresa cliente)
- Hybrid search (dense + sparse) nativo
- Quantizacion int8 para reducir memoria 4x
- **Estimacion de memoria**: ~500K vectores de 1024 dims con int8 quantization ≈ 500MB-1GB RAM

### 3.4 Re-ranker: Cohere Rerank v3.5 API

Con embeddings via API (Voyage), tiene sentido usar reranker via API tambien:
- **Cohere Rerank v3.5**: $0.05/1M tokens, 200M tokens gratis.
- Para safety-critical (DANGER/WARNING/CAUTION): siempre re-rankear.
- Alternativa self-hosted: Qwen3-Reranker-0.6B via Ollama (~639MB adicional en RAM).

### 3.5 Framework RAG: LlamaIndex Workflows

Sin cambios. LlamaIndex 0.14+ con:
- Sub-Question Engine nativo (critico para specs en chunks separados)
- Integracion directa con Qdrant + Voyage embeddings
- Workflows event-driven (@step decorators)
- **Es el framework mas liviano para RAG document-heavy** comparado con LangChain/LangGraph.

LlamaIndex Workflows es significativamente mas liviano que LangGraph:
- LlamaIndex core: ~50MB instalado
- Sin dependencias pesadas de grafos
- Event-driven, no requiere servidor separado
- Se ejecuta dentro de tu FastAPI app directamente

### 3.6 LLM Principal: Claude Sonnet 4.6 API

Sin cambios. $3/$15 por 1M tokens. 1M context window.

### 3.7 LLM Routing: Gemini 3 Flash API

Para expansion de queries y clasificacion de intencion. Mas rapido (134 tokens/seg) y mas barato que GPT-4o-mini.

### 3.8 LLM Gateway: LiteLLM — ESTADO DE SEGURIDAD

#### Incidente de seguridad (24 marzo 2026)

**Que paso**: El paquete PyPI de LiteLLM fue comprometido en un ataque de supply chain. Las versiones `1.82.7` y `1.82.8` contenian malware que robaba credenciales (API keys, SSH keys, cloud credentials, Kubernetes tokens).

**Estado actual (30 marzo 2026)**:
- Las versiones maliciosas fueron removidas de PyPI en ~40 minutos.
- LiteLLM lanzo **v1.83.0** con un nuevo pipeline CI/CD v2 con gates de seguridad reforzados.
- El codigo fuente en GitHub **NO fue comprometido** — solo las versiones publicadas en PyPI.
- Los usuarios del Docker image oficial (`ghcr.io/berriai/litellm`) **NO fueron afectados**.

**Veredicto**: LiteLLM v1.83.0+ es seguro de usar, especialmente via Docker image oficial. Sin embargo, el incidente revela un riesgo inherente de supply chain.

**Alternativas si prefieres evitar LiteLLM**:
| Alternativa | Tipo | Costo | Ventaja |
|-------------|------|-------|---------|
| **Portkey** | SaaS + OSS | Free tier disponible | 1600+ modelos, observabilidad integrada, guardrails |
| **Helicone** | SaaS + OSS | Free tier generoso | Mas simple, buen caching y cost tracking |
| **Codigo directo** | DIY | $0 | Con solo 2 providers (Claude + Gemini), un wrapper simple en Python basta |

**Recomendacion**: Para ManualIQ con solo 2 LLM providers (Claude + Gemini Flash), **no necesitas un gateway complejo**. Un wrapper simple en Python con retry y failover es suficiente y elimina la dependencia. Si escalas a 5+ providers, evalua Portkey.

### 3.9 Cache: Redis 7

Sin cambios. Tier 1: embeddings 48h TTL. Tier 2: LLM responses 24h TTL.

### 3.10 App Database: PostgreSQL 16

Solo para datos de aplicacion: sessions, messages, users, usage_logs. NUNCA vectores.

### 3.11 Auth: Clerk

**Confirmado: usamos Clerk.**
- Multi-tenancy nativa (organizaciones)
- Developer UX 4/4
- Ya lo usas en Aurora (consistencia de stack)
- $0-25/mes para MVP
- Cuando necesites SSO/LDAP enterprise, migras a Keycloak

### 3.12 Orquestacion: Prefect OSS

**Confirmado: usamos Prefect.**

#### Es necesario Prefect para los 3 RAGs?

**Si, es necesario y vale la pena.** Prefect centraliza:
1. **Pipelines de ingestion**: Cuando un cliente sube nuevos manuales, Prefect orquesta: parsing → chunking → embedding → indexacion en Qdrant.
2. **Re-indexacion programada**: Cron jobs para re-procesar documentos actualizados.
3. **Evaluacion automatica**: Correr RAGAS evaluations periodicamente.
4. **Monitoreo de pipelines**: Si un pipeline falla (PDF corrupto, API de embeddings caida), Prefect hace retry automatico y alerta.

**Para los 3 RAGs** (manufactura, financiero, petroleos): Prefect es el **centro de determinismo**. Cada RAG vertical tiene sus propios flows pero comparten la misma infraestructura de orquestacion. Sin Prefect, tendrias cron jobs fragiles sin visibilidad.

**Recursos de Prefect self-hosted**:
- Prefect server (single instance con SQLite): ~200-400MB RAM
- Con PostgreSQL backend (recomendado para produccion): ~300-500MB RAM
- Worker process: ~100-200MB RAM
- **Total Prefect**: ~500-700MB RAM

**Temporal vs Prefect**: Temporal requiere un cluster (server + history service + matching service + frontend) que consume 2-4GB RAM minimo. Prefect single-server con SQLite es 5-10x mas liviano. Decision correcta para tu escala.

### 3.13 Observabilidad: Helicone Cloud (recomendado) o Langfuse Cloud

#### Langfuse vs RAGAS — No son lo mismo

| Aspecto | Langfuse | RAGAS |
|---------|----------|-------|
| **Que es** | Plataforma de observabilidad LLM | Framework de evaluacion RAG |
| **Que hace** | Tracing de requests, costos, latencia, prompt management | Mide faithfulness, relevance, citation accuracy |
| **Cuando se usa** | En tiempo real, cada request | Periodicamente (batch evaluation) |
| **Analogia** | Datadog para LLMs | Unit tests para RAG |

**Se complementan, no compiten.** Langfuse observa tu sistema en produccion. RAGAS evalua la calidad de las respuestas en batch. Puedes correr RAGAS evaluations dentro de Langfuse.

#### Recomendacion: Helicone Cloud (free tier)

Dado que ManualIQ va en un VPS separado (no conectado a tu infra existente), self-hosted Langfuse consumiria RAM valiosa. Mejor usar un servicio cloud con free tier generoso:

| Servicio | Free Tier | Self-hosted | Mejor para |
|----------|-----------|-------------|------------|
| **Helicone** | Generoso, ilimitado requests basicos | Si (OSS) | Setup rapido, caching, cost tracking |
| **Langfuse Cloud** | 50K traces/mes, 2 usuarios | Si (OSS) | Tracing profundo, prompt management |
| **Braintrust** | 1M trace spans/mes | Solo Enterprise | CI/CD deployment blocking |
| **PostHog** | 100K eventos/mes | Si (OSS) | Si ya usas PostHog para analytics |

**Recomendacion**: **Helicone Cloud** free tier para empezar. Es el mas simple de integrar (proxy OpenAI-compatible), tiene caching integrado (ahorra costos de API), y cost tracking automatico. Si necesitas tracing mas profundo, migra a Langfuse Cloud.

**RAGAS** se ejecuta como pipeline en Prefect (batch evaluation semanal), no necesita servidor propio. Consume RAM solo durante la evaluacion (~200MB temporal).

### 3.14 Evaluacion: RAGAS via Prefect

- Corre como flow de Prefect, no como servicio permanente.
- Evalua: faithfulness, answer relevancy, context precision, citation accuracy.
- Set de 50-100 preguntas de regresion antes de cada deploy.
- ~200MB RAM temporal durante evaluacion.

### 3.15 Frontend: CopilotKit + Next.js

Sin cambios. MIT license, AG-UI protocol, streaming nativo con LlamaIndex.

### 3.16 Email: Resend

Sin cambios. $0-20/mes, 50K emails/mes.

---

## 4. INFRAESTRUCTURA — UN SOLO VPS HETZNER CX43

### Specs del CX43
- **8 vCPU** (AMD EPYC Rome, shared)
- **16 GB RAM**
- **160 GB NVMe SSD**
- **20 TB trafico/mes**
- **~$12/mes** (Helsinki, hel1)

### Este VPS es INDEPENDIENTE de tu infra existente

No se conecta con el App Plane (10.0.1.30) ni el Data Plane (10.0.1.20). ManualIQ es un producto separado con su propia infraestructura.

### Distribucion de RAM estimada

| Servicio | RAM Estimada | Notas |
|----------|-------------|-------|
| **Qdrant OSS** | 1.5-2.5 GB | ~500K vectores 1024d con int8 quantization |
| **PostgreSQL 16** | 0.5-1 GB | App data: users, sessions, messages |
| **Redis 7** | 0.3-0.5 GB | Cache embeddings + LLM responses |
| **FastAPI app** (ManualIQ) | 0.5-1 GB | LlamaIndex + API + workers |
| **Prefect server** | 0.3-0.5 GB | Single instance con SQLite |
| **Prefect worker** | 0.1-0.2 GB | Ejecuta flows de ingestion |
| **Next.js frontend** | 0.3-0.5 GB | CopilotKit UI |
| **Docling** (on-demand) | 0.5-1 GB | Solo durante ingestion de PDFs |
| **Ollama + Qwen3-Embedding** | 1.5-2 GB | SOLO si usas self-hosted embeddings |
| **Sistema operativo** | 0.5-1 GB | Ubuntu 24.04 + Docker overhead |
| **TOTAL (con Voyage API)** | **~4.5-8 GB** | Cabe holgadamente en 16GB |
| **TOTAL (con Qwen3 self-hosted)** | **~6-10 GB** | Cabe pero mas ajustado |

### Veredicto: SI CABE en un CX43

**Con Voyage API para embeddings**: Usas ~4.5-8 GB de 16 GB disponibles. Tienes 8+ GB de margen para picos de carga, ingestion de PDFs, y crecimiento.

**Con Qwen3-Embedding self-hosted**: Usas ~6-10 GB. Funciona pero con menos margen. La ingestion masiva de PDFs (Docling + embeddings simultaneos) podria causar presion de memoria.

**Recomendacion**: Usa **Voyage API** para embeddings. Liberas ~2GB de RAM, eliminas la complejidad de Ollama, y obtienes mejor calidad de retrieval. El costo es despreciable (~$3-5/mes).

### Docker Compose simplificado

```yaml
services:
  # --- Data Layer ---
  qdrant:
    image: qdrant/qdrant:v1.13.0
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
    deploy:
      resources:
        limits: { memory: 3G }

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: manualiq
      POSTGRES_USER: manualiq
      POSTGRES_PASSWORD_FILE: /run/secrets/pg_password
    volumes: ["pg_data:/var/lib/postgresql/data"]
    deploy:
      resources:
        limits: { memory: 1G }

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes: ["redis_data:/data"]

  # --- Application Layer ---
  api:
    build: ./api
    ports: ["8000:8000"]
    environment:
      - VOYAGE_API_KEY_FILE=/run/secrets/voyage_key
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_key
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://manualiq:${PG_PASS}@postgres:5432/manualiq
    depends_on: [qdrant, postgres, redis]
    deploy:
      resources:
        limits: { memory: 1.5G }

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    deploy:
      resources:
        limits: { memory: 512M }

  # --- Orchestration ---
  prefect-server:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    ports: ["4200:4200"]
    deploy:
      resources:
        limits: { memory: 512M }

  prefect-worker:
    image: prefecthq/prefect:3-latest
    command: prefect worker start --pool default
    depends_on: [prefect-server]
    deploy:
      resources:
        limits: { memory: 256M }

volumes:
  qdrant_data:
  pg_data:
  redis_data:
```

### Almacenamiento (160 GB NVMe)

| Uso | Estimado |
|-----|----------|
| Sistema operativo + Docker | ~10 GB |
| Qdrant data (vectores + indices) | ~5-15 GB |
| PostgreSQL data | ~2-5 GB |
| Redis (in-memory, no disco) | ~0 GB |
| PDFs originales (corpus) | ~20-50 GB |
| Prefect metadata | ~1-2 GB |
| Next.js build + node_modules | ~2-3 GB |
| **Total estimado** | **~40-85 GB de 160 GB** |

Margen suficiente. Si el corpus crece mucho, agrega un Hetzner Volume ($0.052/GB/mes).

---

## 5. COSTOS MENSUALES ESTIMADOS

| Concepto | Costo/mes |
|----------|-----------|
| Hetzner CX43 | ~$12 |
| Claude Sonnet 4.6 API | ~$30-80 (depende de uso) |
| Gemini 3 Flash API | ~$5-15 |
| Voyage-4 embeddings | ~$3-5 |
| Cohere Rerank | ~$2-5 |
| Clerk auth | $0-25 |
| Resend email | $0-20 |
| Helicone observabilidad | $0 (free tier) |
| Dominio + DNS | ~$1-2 |
| **TOTAL** | **~$53-164/mes** |

---

## 6. SYSTEM PROMPT DE MANUALIQ

```
Eres ManualIQ, asistente tecnico especializado en manuales industriales.

IMPORTANTE SOBRE EL CORPUS:
El 80% de los manuales esta en ESPANOL y el 20% en ingles. Tratas IGUAL
ambos idiomas en la recuperacion y la respuesta.

REGLAS ABSOLUTAS:
1. Responde SIEMPRE en espanol completo, claro y tecnicamente preciso.
2. INCLUYE TODA la informacion relevante recuperada, sin importar el idioma
   del documento fuente. No omitas informacion valiosa solo porque el chunk
   este en ingles.
3. CADA afirmacion tecnica debe incluir cita exacta:
   [Manual: {nombre}, Seccion: {seccion}, Pagina: {pagina}]
4. Para chunks en ingles: incluye el fragmento original entre comillas
   Y la traduccion al espanol en el cuerpo de la respuesta.
   Para chunks en espanol: cita directamente, no necesita traduccion.
5. Si multiples documentos responden la pregunta (ej: un SOP en ES y
   un manual tecnico en EN), INCLUYE AMBOS. La respuesta debe ser
   completa, no limitarse a la primera fuente encontrada.
6. Si hay valores numericos: incluir SIEMPRE unidades y tolerancias.
   Debe incluir el PDF del documento donde esta la informacion.
7. Si hay part numbers: incluirlos SIEMPRE completos.
8. Si hay DANGER/WARNING/CAUTION: mostrar PRIMERO, en negrita.
9. Si no hay informacion: "No encontre informacion sobre esto en los
   manuales disponibles." NUNCA inventar datos tecnicos.
10. Para procedimientos: pasos EN ORDEN, TODOS, sin omitir.
11. Confianza baja (score < 0.75): avisar al tecnico y al supervisor.
12. Finalizar SIEMPRE con disclaimer de verificacion con supervisor.
```

---

## 7. ARQUITECTURA MULTI-RAG (FUTURO)

Cuando escales a financiero y petroleos, la arquitectura se extiende asi:

```
                    ┌─────────────────┐
                    │   Clerk Auth    │
                    │  (multi-tenant) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   API Gateway   │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───┐  ┌──────▼─────┐  ┌────▼────────┐
     │ ManualIQ   │  │ FinanceIQ  │  │  PetroIQ    │
     │ (manufact) │  │ (financ)   │  │  (petroleo) │
     └────────┬───┘  └──────┬─────┘  └────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │     Qdrant      │
                    │ (1 collection   │
                    │  per vertical,  │
                    │  shard per      │
                    │  tenant)        │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Prefect      │
                    │ (shared flows   │
                    │  per vertical)  │
                    └─────────────────┘
```

Cada vertical comparte:
- Infraestructura (Qdrant, PostgreSQL, Redis, Prefect)
- Auth (Clerk con organizaciones)
- Observabilidad (Helicone)

Cada vertical tiene propio:
- System prompt especializado
- Reglas de chunking adaptadas al dominio
- Coleccion Qdrant separada
- Flows de Prefect especificos

Cuando 3 RAGs no quepan en un CX43, escala verticalmente a CX53 (16 vCPU, 32GB, ~$23/mes) o agrega un segundo VPS.

---

## 8. DECISIONES CLAVE RESUMIDAS

| Pregunta | Decision | Razon |
|----------|----------|-------|
| Embeddings GPU o CPU? | **API (Voyage-4)** | Sin GPU en CX43, API es mas eficiente y mejor calidad |
| Voyage vs OpenAI vs Cohere? | **Voyage-4** | 14% mejor retrieval que OpenAI, 32K contexto, $0.06/1M tokens, 200M gratis |
| LiteLLM es seguro? | **Si (v1.83.0+)** pero no lo necesitas | Solo 2 providers, un wrapper simple basta |
| Clerk confirmado? | **Si** | Multi-tenancy nativa, ya lo usas en Aurora |
| Prefect necesario para 3 RAGs? | **Si** | Centro de determinismo, retry, visibilidad, ~500MB RAM |
| Temporal? | **No** | 2-4GB RAM minimo, overkill para tu escala |
| Langfuse vs RAGAS? | **Ambos, diferentes roles** | Langfuse=observabilidad, RAGAS=evaluacion |
| Observabilidad cloud? | **Helicone Cloud** (free tier) | No consume RAM local, simple, caching integrado |
| Cabe en CX43 16GB? | **Si** | ~4.5-8GB con Voyage API, margen amplio |
| Conecta con infra existente? | **No** | VPS independiente, producto separado |
