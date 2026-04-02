# ManualIQ — Diagramas de Arquitectura

> Diagramas Mermaid que describen como funciona el sistema, como se conectan
> los componentes, y como es la experiencia del usuario desde el landing
> hasta el uso diario.

---

## 1. Arquitectura general — Como se conectan los componentes

```mermaid
graph TB
    subgraph USUARIOS["Usuarios"]
        TEC["Tecnico de planta<br/>(PC o celular)"]
        SUP["Supervisor de<br/>mantenimiento"]
        ADM["Admin de empresa<br/>(tenant admin)"]
        EMAIL_IN["Tecnico via email<br/>(sin acceso a PC)"]
    end

    subgraph FRONTEND["Frontend — Next.js + assistant-ui :3000"]
        LANDING["Landing page<br/>manualiq.com"]
        LOGIN["Login / Registro<br/>(Clerk)"]
        CHAT["Chat UI<br/>(streaming)"]
        VIEWER["Visor de PDF fuente<br/>+ scores"]
        HISTORY["Historial de<br/>conversaciones"]
        UPLOAD["Panel de ingestion<br/>(admin sube PDFs)"]
        DASH["Dashboard metricas<br/>(post-MVP)"]
    end

    subgraph API["API — FastAPI :8000"]
        AUTH["Clerk middleware<br/>tenant_id + user_id"]
        GUARD_IN["NeMo Guardrails<br/>INPUT rail"]
        RATE["Rate limiter<br/>(Redis counters)"]
        INTENT["Intent classifier<br/>(Gemini Flash)"]
        QUERY_ENG["Query engine<br/>(retrieve-rerank-threshold)"]
        GUARD_OUT["NeMo Guardrails<br/>OUTPUT rail + PII"]
        LANG["Language guardrail<br/>(enforce Spanish)"]
        LLM_GW["LLM Gateway<br/>(Claude + Gemini)"]
        INGEST_API["Ingestion endpoint<br/>(trigger Prefect flow)"]
        HEALTH["/health endpoint"]
    end

    subgraph DATA["Data Layer"]
        QDRANT[("Qdrant<br/>vectores 1024d<br/>int8 quantized<br/>shard key = tenant_id")]
        PG[("PostgreSQL 16<br/>users, sessions,<br/>messages, usage_logs")]
        REDIS[("Redis 7<br/>embedding cache 48h<br/>LLM cache 24h<br/>rate limit counters")]
    end

    subgraph ORCHESTRATION["Orchestration — Prefect v3"]
        PREFECT_S["Prefect Server<br/>(SQLite)"]
        PREFECT_W["Prefect Worker"]
        FLOW_ING["Flow: ingestion<br/>parse-chunk-embed-upsert"]
        FLOW_REIDX["Flow: re-indexacion<br/>(SHA-256 change detection)"]
        FLOW_BACKUP["Flow: Qdrant backup<br/>(snapshot diario)"]
        FLOW_HEALTH["Flow: health check<br/>(cada 5 min)"]
    end

    subgraph OBSERVABILITY["Observability"]
        PHOENIX["Arize Phoenix<br/>tracing + evals<br/>:6006 UI / :4317 OTLP"]
    end

    subgraph EXTERNAL["APIs Externas"]
        VOYAGE["Voyage-4 API<br/>(embeddings)"]
        CLAUDE["Claude Sonnet 4.6<br/>(generacion)"]
        GEMINI["Gemini 3 Flash<br/>(routing/expansion)"]
        COHERE["Cohere Rerank v3.5"]
        CLERK_API["Clerk API<br/>(auth/tenants)"]
        RESEND["Resend<br/>(email alerts)"]
    end

    TEC --> LANDING
    SUP --> LANDING
    ADM --> LANDING
    EMAIL_IN -.->|"consulta@empresa<br/>.manualiq.com"| API

    LANDING --> LOGIN
    LOGIN --> CHAT
    LOGIN --> UPLOAD
    CHAT --> VIEWER
    CHAT --> HISTORY

    CHAT -->|"POST /query"| AUTH
    UPLOAD -->|"POST /ingest"| AUTH
    HISTORY -->|"GET /history"| AUTH

    AUTH --> RATE
    RATE --> GUARD_IN
    GUARD_IN -->|"paralelo con retrieval<br/>(asyncio)"| INTENT
    INTENT --> QUERY_ENG
    QUERY_ENG --> LLM_GW
    LLM_GW --> GUARD_OUT
    GUARD_OUT --> LANG
    LANG -->|"respuesta final"| CHAT

    AUTH --> INGEST_API
    INGEST_API -->|"trigger flow"| PREFECT_S

    QUERY_ENG -->|"retrieve top-20"| QDRANT
    QUERY_ENG -->|"rerank"| COHERE
    QUERY_ENG -->|"embedding cache"| REDIS
    QUERY_ENG -->|"embed query"| VOYAGE

    LLM_GW --> CLAUDE
    LLM_GW --> GEMINI
    INTENT --> GEMINI

    AUTH --> CLERK_API
    RATE --> REDIS
    API --> PG

    PREFECT_S --> PREFECT_W
    PREFECT_W --> FLOW_ING
    PREFECT_W --> FLOW_REIDX
    PREFECT_W --> FLOW_BACKUP
    PREFECT_W --> FLOW_HEALTH

    FLOW_ING -->|"Docling/LlamaParse<br/>chunker - Voyage"| QDRANT
    FLOW_ING --> VOYAGE
    FLOW_REIDX --> QDRANT
    FLOW_BACKUP -->|"snapshot"| QDRANT
    FLOW_HEALTH --> RESEND

    API -->|"OpenTelemetry traces"| PHOENIX
```

---

## 2. Flujo de una query — Desde que el tecnico pregunta hasta que recibe respuesta

```mermaid
sequenceDiagram
    participant T as Tecnico
    participant FE as Frontend<br/>(Next.js)
    participant CK as Clerk
    participant API as FastAPI
    participant NI as NeMo Input<br/>Guardrail
    participant GF as Gemini Flash
    participant QD as Qdrant
    participant RD as Redis
    participant CO as Cohere<br/>Rerank
    participant CL as Claude<br/>Sonnet 4.6
    participant NO as NeMo Output<br/>Guardrail
    participant PH as Phoenix

    T->>FE: "Cual es el torque de la culata del C7?"
    FE->>CK: Verificar JWT
    CK-->>FE: tenant_id + user_id
    FE->>API: POST /query {query, tenant_id, user_id}

    API->>RD: Check rate limit (tenant + user)
    RD-->>API: OK (45/50 queries hoy)

    par Guardrail INPUT (paralelo)
        API->>NI: Check prompt injection + scope
        NI-->>API: PASS
    and Retrieval (paralelo)
        API->>GF: Clasificar intencion
        GF-->>API: intent=specific, confidence=0.9
        API->>GF: Expandir query ES a EN
        GF-->>API: "What is the cylinder head torque for C7?"
        API->>RD: Check embedding cache (query hash)
        RD-->>API: MISS
        API->>QD: Vector search top-20<br/>(shard_key=tenant_id,<br/>queries ES + EN)
        QD-->>API: 20 chunks con scores
    end

    API->>API: Dedup por hash_sha256
    API->>API: Validar tenant_id de cada chunk
    API->>CO: Rerank 20 chunks
    CO-->>API: Top-5 reranked (best=0.92)

    API->>API: Confidence check: 0.92 >= 0.75
    API->>API: Build context prompt<br/>(safety chunks primero)

    API->>CL: Generar respuesta con contexto
    CL-->>API: Respuesta en espanol con citas

    API->>NO: Check output: PII, citas validas
    NO-->>API: PASS (sin PII, citas OK)
    API->>API: Language guardrail: idioma=es
    API->>API: Append disclaimer supervisor

    API->>PH: Trace completo (latencias, tokens, costos)
    API->>RD: Cache respuesta (24h TTL)

    API-->>FE: {answer, chunks, confidence=HIGH, score=0.92}
    FE-->>T: Respuesta streaming + visor PDF + disclaimer

    Note over T,PH: Latencia total objetivo: menor a 5 segundos (P95)
```

---

## 3. Flujo de ingestion — Desde que el admin sube un PDF hasta que esta listo

```mermaid
sequenceDiagram
    participant AD as Admin Tenant
    participant FE as Frontend
    participant API as FastAPI
    participant PF as Prefect Server
    participant PW as Prefect Worker
    participant DC as Docling
    participant LP as LlamaParse
    participant CH as Chunker
    participant VY as Voyage-4
    participant RD as Redis
    participant QD as Qdrant
    participant RS as Resend

    AD->>FE: Sube PDF (manual Caterpillar C7)
    FE->>API: POST /ingest {file, tenant_id, metadata}
    API->>API: Guardar PDF en /corpus/{tenant_id}/
    API->>PF: Trigger flow "ingestion"
    PF->>PW: Dispatch task

    rect rgb(255, 243, 224)
        Note over PW,LP: PARSING (subprocess con timeout 120s)
        PW->>DC: Parse PDF (subprocess)
        alt Docling OK (menor a 120s)
            DC-->>PW: Markdown + tablas
        else Docling timeout/crash
            PW->>LP: Fallback LlamaParse
            LP-->>PW: Markdown (menor precision tablas)
        end
    end

    rect rgb(232, 245, 233)
        Note over PW,CH: CHUNKING SEMANTICO
        PW->>CH: classify_and_split_blocks(text)
        Note right of CH: Reglas:<br/>- Nunca dividir pasos<br/>- Nunca dividir tablas<br/>- Part numbers juntos<br/>- Overlap 80 tokens solo texto<br/>- Target 400-600 tokens
        CH-->>PW: DocumentBlocks
        PW->>CH: blocks_to_chunks(blocks, metadata)
        Note right of CH: Agrega:<br/>- section_path prefix<br/>- safety_level detection<br/>- SHA-256 hash<br/>- part_numbers extraction
        CH-->>PW: Chunks con metadata completa
    end

    rect rgb(227, 242, 253)
        Note over PW,QD: EMBEDDING + INDEXACION
        loop Para cada chunk
            PW->>RD: Check embedding cache (hash)
            alt Cache HIT
                RD-->>PW: Embedding cacheado
            else Cache MISS
                PW->>VY: Embed chunk (voyage-4)
                VY-->>PW: Vector 1024d
                PW->>RD: Cache embedding (48h TTL)
            end
        end
        PW->>QD: Upsert batch<br/>(shard_key=tenant_id,<br/>payload indexes activos)
        QD-->>PW: OK
    end

    PW-->>PF: Flow completado
    PF->>RS: Notificar admin
    RS-->>AD: "Su manual fue procesado: 87 chunks indexados"
```

---

## 4. Experiencia del usuario — Desde el landing hasta el uso diario

```mermaid
journey
    title Experiencia del tecnico con ManualIQ
    section Descubrimiento
      Visita manualiq.com: 3: Tecnico
      Ve landing con demo: 4: Tecnico
      Su empresa contrata el plan: 5: Admin
    section Onboarding
      Admin crea org en Clerk: 4: Admin
      Admin sube manuales PDF: 4: Admin
      Sistema procesa y notifica: 5: Sistema
      Admin invita tecnicos: 4: Admin
      Tecnico recibe email de invitacion: 4: Tecnico
    section Uso diario (PC)
      Login con Clerk: 5: Tecnico
      Escribe pregunta en espanol: 5: Tecnico
      Recibe respuesta con citas: 5: Tecnico
      Ve PDF fuente en visor: 4: Tecnico
      Exporta conversacion a PDF: 3: Tecnico
    section Uso diario (email)
      Envia pregunta por email: 5: Tecnico
      Recibe respuesta por email: 4: Tecnico
    section Supervision
      Recibe alerta de baja confianza: 4: Supervisor
      Revisa respuesta y valida: 4: Supervisor
      Ve dashboard de metricas: 3: Admin
    section Operaciones (automatico)
      Re-indexacion semanal: 5: Sistema
      Backup Qdrant diario: 5: Sistema
      Health check cada 5 min: 5: Sistema
      Alerta si servicio cae: 4: Sistema
```

---

## 5. Distribucion en el VPS — Que corre donde

```mermaid
graph LR
    subgraph VPS["Hetzner CX43 — 8 vCPU, 16GB RAM, 160GB NVMe"]
        subgraph DOCKER["Docker Compose"]
            direction TB

            subgraph APP_LAYER["Application Layer ~2GB"]
                API_C["api<br/>FastAPI + LlamaIndex<br/>+ NeMo Guardrails<br/>1.5GB limit"]
                FE_C["frontend<br/>Next.js + assistant-ui<br/>512MB limit"]
            end

            subgraph DATA_LAYER["Data Layer ~4.8GB"]
                QD_C["qdrant<br/>Vectores int8<br/>3GB limit"]
                PG_C["postgres<br/>App data<br/>1GB limit"]
                RD_C["redis<br/>Cache + counters<br/>768MB limit"]
            end

            subgraph OPS_LAYER["Operations Layer ~1.5GB"]
                PH_C["phoenix<br/>Tracing + evals<br/>512MB limit"]
                PS_C["prefect-server<br/>SQLite<br/>512MB limit"]
                PW_C["prefect-worker<br/>Docling + ingestion<br/>2GB limit"]
            end
        end

        OS["Ubuntu 24.04 + Docker ~1GB"]
        CORPUS["PDFs corpus 20-50GB"]
        BACKUPS["Qdrant snapshots"]
    end

    subgraph EXTERNAL["Servicios Externos"]
        V["Voyage-4"]
        CL["Claude"]
        GE["Gemini"]
        CO["Cohere"]
        CLK["Clerk"]
        RES["Resend"]
    end

    API_C --> QD_C
    API_C --> PG_C
    API_C --> RD_C
    API_C --> PH_C
    PW_C --> QD_C
    PW_C --> CORPUS

    API_C -.-> V
    API_C -.-> CL
    API_C -.-> GE
    API_C -.-> CO
    API_C -.-> CLK
    PS_C --> PW_C
    PW_C -.-> V
    PW_C -.-> RES
```

---

## 6. Flujos por canal de acceso

| Canal | Flujo |
|-------|-------|
| **Web (PC)** | Landing - Login Clerk - Chat UI - API - Pipeline RAG - Respuesta streaming + visor PDF |
| **Web (celular)** | Mismo flujo, responsive. Next.js adapta el layout. Instalable como PWA. |
| **Email** | Tecnico envia email a `consulta@empresa.manualiq.com` - API recibe (webhook Resend) - Pipeline RAG - Respuesta por email con citas + disclaimer. Latencia menor a 2 min. |
| **Admin** | Login - Panel de ingestion - Sube PDF - Prefect flow procesa - Notificacion cuando termina. Dashboard de metricas (post-MVP). |
| **Supervisor** | Recibe alertas automaticas cuando score menor a 0.75. Puede revisar la respuesta y el contexto en el historial. |
| **Operaciones** | Todo automatico via Prefect: re-indexacion semanal, backup diario, health check cada 5 min, alertas por email si algo cae. |

El canal de email es el unico que no pasa por el frontend — va directo al API via webhook de Resend. Todos los demas canales pasan por el frontend (Next.js) que se comunica con el API (FastAPI) que orquesta todo el pipeline RAG.
