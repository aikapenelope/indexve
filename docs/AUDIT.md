# ManualIQ — Auditoria de Componentes y Mejoras Pendientes

> Revision del codigo actual contra la documentacion oficial de cada
> componente. Identifica configuraciones faltantes, mejoras de rendimiento,
> y features adicionales que mejorarian el sistema.
>
> Fecha: Abril 2026

---

## 1. Voyage-4 Embeddings — Gaps de configuracion

### Problema: Falta `input_type` en las llamadas a la API

La documentacion de Voyage dice:

> "For retrieval/search purposes, we recommend specifying whether your
> inputs are intended as queries or documents by setting `input_type`
> to `query` or `document`, respectively."

Nuestro `embedding_service.py` no envia `input_type`. Esto reduce la
calidad del retrieval porque Voyage internamente prepende instrucciones
diferentes para queries vs documentos.

**Archivo**: `api/manualiq/ingestion/embedding_service.py`
**Lineas**: `_call_voyage_api()` — el JSON body solo envia `input` y `model`.

**Fix**: Agregar `input_type: "document"` para indexacion y `input_type: "query"`
para queries. Esto mejora retrieval quality sin costo adicional.

```python
# Indexacion
json={"input": texts, "model": model, "input_type": "document"}

# Query
json={"input": texts, "model": model, "input_type": "query"}
```

### Problema: No se usa `truncation` parameter

Voyage trunca automaticamente a 32K tokens, pero si un chunk excede el
limite, se trunca silenciosamente sin aviso. Deberiamos enviar
`truncation: true` explicitamente y loguear cuando ocurre.

---

## 2. Qdrant — Falta Hybrid Search (dense + sparse)

### Problema: Solo usamos dense vectors, no hybrid search

ARCHITECTURE.md dice "Hybrid search (dense + sparse) nativo" como feature
de Qdrant, pero la implementacion solo usa dense vectors (Voyage embeddings).

Segun la documentacion de Qdrant y los benchmarks de 2026, hybrid search
(dense + sparse/BM25) mejora precision 15-30% adicional sobre dense-only,
especialmente para:
- Part numbers exactos (AB-4521-CX) — sparse los encuentra por match exacto
- Terminologia tecnica especifica — BM25 captura keywords que embeddings pueden perder
- Queries cortas — sparse search es mas robusto con pocas palabras

**Que falta**:
1. Configurar la coleccion con named vectors: `"dense"` + `"sparse"`
2. Generar sparse vectors (BM25/IDF) durante ingestion con FastEmbed o similar
3. Usar Qdrant Query API con fusion (RRF o DBSF) para combinar resultados

**Archivo afectado**: `qdrant_init.py` (crear coleccion con sparse config),
`embedding_service.py` (generar sparse vectors), `main.py` (query con fusion).

**Impacto**: Alto. Para manuales tecnicos con part numbers y terminologia
especifica, hybrid search es critico.

### Problema: HNSW no esta tuneado

La coleccion se crea con defaults de HNSW. Para nuestro caso (500K vectores,
1024 dims, int8):
- `m=16` (default) es adecuado
- `ef_construct=100` (default) podria subir a 200 para mejor recall
- `ef` en busqueda deberia ser al menos 128 (no lo configuramos)

**Fix**: Agregar `hnsw_config` en `qdrant_init.py` y `search_params` en
las queries de `main.py`.

---

## 3. Cohere Rerank — Configuracion incompleta

### Problema: No enviamos `rank_fields` para datos semi-estructurados

Cohere Rerank v3.5 soporta `rank_fields` para indicar que campos del
documento usar para el ranking. Nuestros chunks tienen metadata rica
(section_path, equipment, safety_level) que podria mejorar el reranking
si se incluye como contexto estructurado.

**Fix**: Enviar documentos como dicts con campos en vez de strings planos:
```python
documents=[
    {"text": chunk.text, "section": chunk.section_path, "equipment": chunk.equipment}
]
```

### Problema: Falta `max_tokens_per_doc` para controlar costos

Si un chunk es muy largo, Cohere lo divide internamente y cobra por cada
sub-chunk. Deberiamos limitar a 4096 tokens (el maximo del modelo) para
evitar costos inesperados.

---

## 4. NeMo Guardrails — Configuracion con problemas

### Problema: config.yml usa `gpt-4o-mini` en vez de nuestros modelos

El archivo `guardrails/config/config.yml` configura `engine: openai` y
`model: gpt-4o-mini` para las evaluaciones de guardrails. Pero ManualIQ
usa Claude + Gemini, no OpenAI. Esto significa que NeMo haria llamadas
a un tercer proveedor LLM que no tenemos configurado.

**Fix**: Cambiar a Gemini Flash (nuestro modelo de routing) o usar las
acciones custom que ya tenemos (que no necesitan LLM para PII/injection).

### Problema: Falta `prompts.yml`

La estructura recomendada por NeMo incluye un `prompts.yml` que define
los prompts usados para self-check. Sin este archivo, NeMo usa defaults
en ingles que no estan optimizados para nuestro caso bilingue ES/EN.

### Problema: IORails (v0.21) no esta aprovechado

NeMo v0.21 introdujo `IORails`, un motor optimizado que ejecuta rails
de input/output en paralelo. Nuestro `guardrails.py` usa `generate_async()`
que es el metodo legacy. `check_async()` es el nuevo metodo recomendado
para validacion standalone sin flujo conversacional completo.

---

## 5. Prefect v3 — Mejoras de produccion

### Problema: Flows sin `retries` ni `retry_delay_seconds`

Los flows en `operations.py` no tienen configuracion de retries. Si
Qdrant esta temporalmente caido durante un backup, el flow falla sin
reintentar.

**Fix**:
```python
@flow(name="backup-qdrant", retries=3, retry_delay_seconds=60)
```

### Problema: Tasks sin `cache_result_in_memory`

El task `reindex_document` llama a Docling + Voyage para cada documento.
Si el flow falla a mitad de camino y se reintenta, re-procesa documentos
que ya estaban listos. Prefect v3 soporta `persist_result=True` para
evitar esto.

### Problema: Falta concurrency limit en deployments

`deployments.py` no configura `concurrency_limit`. Si se disparan
multiples re-indexaciones simultaneas, pueden saturar Voyage API y
Qdrant.

**Fix**: Agregar `concurrency_limit=1` para reindex y backup.

### Problema: Worker no tiene `OLLAMA_URL`

El `prefect-worker` en docker-compose no tiene `OLLAMA_URL` en su
environment. Si el worker necesita embeddings durante re-indexacion
y Voyage falla, no puede usar el fallback Ollama.

---

## 6. LlamaIndex Workflows — No esta integrado

### Problema: No usamos LlamaIndex Workflows en el pipeline

ARCHITECTURE.md dice "LlamaIndex Workflows 0.14+" como framework RAG,
pero el pipeline en `main.py` esta construido manualmente (llamadas
directas a Qdrant, Cohere, Claude). No usamos:
- `QueryEngine` de LlamaIndex
- `SubQuestionQueryEngine` para evidencia dispersa
- `ResponseSynthesizer` para generacion
- `@step` decorators de Workflows

Esto significa que Phoenix tracing (que instrumenta LlamaIndex
automaticamente) no captura los pasos del pipeline — solo veria las
llamadas HTTP raw.

**Recomendacion**: Esto es un refactor grande. Para MVP, el pipeline
manual funciona. Para v2, migrar a LlamaIndex Workflows daria:
- Tracing automatico en Phoenix (cada step como span)
- Sub-question engine nativo
- Streaming event-driven
- Integracion directa con Qdrant VectorStoreIndex

---

## 7. Chunker — Mejoras posibles

### Problema: No resuelve `related_chunks` en indexacion

ARCHITECTURE.md dice: "Referencias cruzadas ('ver paso 4.2.3') resolver
en indexacion, incluir chunk referenciado como `related_chunks` en metadata."

El chunker detecta `related_refs` en `DocumentBlock` pero nunca los
resuelve. El campo `related_chunks` en la metadata siempre esta vacio.

**Fix**: Agregar un paso post-chunking que busque patrones de referencia
cruzada ("ver paso X.Y.Z", "see section X", "referirse a") y los
resuelva contra los otros chunks del mismo documento.

### Problema: No detecta headings para section_path automatico

El chunker recibe `section_path` como parametro, pero no lo extrae
automaticamente del texto parseado. Docling genera markdown con headings
(`## Cap 4 > Seccion 3`) que podrian parsearse para generar el
section_path automaticamente.

---

## 8. Embedding Service — Mejoras de eficiencia

### Problema: No usa Voyage Batch API para ingestion masiva

Voyage ofrece una Batch API con 33% de descuento y ventana de 12h para
procesamiento masivo. Para re-indexar 7000 documentos, esto ahorraria
~$1-2 vs la API sincrona.

**Archivo**: `embedding_service.py`
**Fix**: Agregar metodo `embed_chunks_batch()` que use la Batch API
para ingestion masiva (>100 chunks).

### Problema: Ollama fallback no verifica que el modelo este cargado

`_call_ollama_embed()` llama a Ollama directamente. Si el modelo no
esta descargado, Ollama lo descarga on-demand (~639MB), lo cual puede
tomar minutos y causar timeout. Deberia verificar primero con
`GET /api/tags` y loguear un warning si el modelo no esta disponible.

---

## 9. LLM Gateway — Mejoras

### Problema: No trackea costos por request

`LLMResponse` tiene `input_tokens` y `output_tokens` pero no calcula
el costo. Para el dashboard de metricas y las alertas de costo, necesitamos:

```python
@dataclass
class LLMResponse:
    ...
    estimated_cost_usd: float = 0.0
```

Con precios: Claude $3/$15 por 1M tokens, Gemini Flash ~$0.15/$0.60.

### Problema: No hay streaming

El gateway genera la respuesta completa y la retorna. Para latencia
percibida <5s, deberiamos hacer streaming token-by-token al frontend.
Esto requiere `AsyncAnthropic.messages.stream()` y FastAPI
`StreamingResponse`.

---

## 10. Frontend — Mejoras

### Problema: No hay streaming real

El adapter (`manualiq-adapter.ts`) hace `fetch` y espera la respuesta
completa. assistant-ui soporta streaming nativo via `ReadableStream`.
El backend deberia exponer un endpoint `/query/stream` que use
`StreamingResponse` de FastAPI.

### Problema: No muestra chunks/fuentes al usuario

El `QueryResponse` incluye `chunks_used` (count) pero no los chunks
reales con sus citas. El frontend no puede mostrar el visor de PDF
fuente porque no recibe `doc_id`, `section_path`, `page_ref`, ni
`score` de cada chunk.

**Fix**: Agregar `sources: list[SourceChunk]` al response con la
metadata de cada chunk usado.

---

## 11. Mejoras adicionales para un RAG de produccion

### 11.1 Evaluation framework (Phoenix Evals)

Lo mas critico que falta. Sin evaluacion sistematica, no sabemos si
los cambios mejoran o empeoran el sistema. Necesita:
- Set de 50-100 preguntas de regresion con respuestas esperadas
- Metricas: faithfulness, answer relevancy, citation accuracy
- Ejecucion semanal automatica via Prefect
- Dashboard en Phoenix para comparar versiones

### 11.2 Query routing por complejidad

No todas las queries necesitan el pipeline completo. Queries simples
("cual es el part number del filtro?") podrian saltar el reranking
y usar solo top-3 chunks. Queries complejas ("compara todos los
torques del C7 vs C9") necesitan sub-question decomposition.

El intent classifier ya detecta el tipo, pero no ajusta el pipeline.

### 11.3 Conversation memory

El sistema es stateless — cada query es independiente. Para preguntas
de seguimiento ("y el del otro lado?"), necesita memoria de
conversacion. LlamaIndex tiene `ChatMemoryBuffer` que mantiene
contexto entre turnos.

### 11.4 Document versioning

Cuando un manual se actualiza, el sistema re-indexa pero no mantiene
la version anterior. Si un tecnico pregunta sobre un procedimiento
que cambio, deberia poder ver ambas versiones con fecha.

### 11.5 Feedback loop

El endpoint POST /feedback existe en el PRD pero no esta implementado.
Los thumbs up/down de los tecnicos son la senal mas valiosa para
mejorar el sistema. Deberian alimentar:
- Phoenix Evals (como ground truth)
- Fine-tuning del system prompt
- Ajuste del confidence threshold

### 11.6 Multimodal embeddings para diagramas

Voyage ofrece `voyage-multimodal-3.5` que puede vectorizar imagenes
de diagramas tecnicos, tablas escaneadas, y figuras. Los manuales
industriales tienen muchos diagramas que el chunker de texto ignora.

---

## Resumen de prioridades

### Alta prioridad (mejora directa de calidad)

| # | Mejora | Esfuerzo | Impacto |
|---|--------|----------|---------|
| 1 | Agregar `input_type` a Voyage API calls | 30 min | Alto |
| 2 | Hybrid search (dense + sparse) en Qdrant | 1-2 dias | Alto |
| 3 | Retries + result persistence en Prefect flows | 1 hora | Medio |
| 4 | Arreglar NeMo config (modelo, prompts.yml) | 2 horas | Medio |
| 5 | Agregar sources/chunks al QueryResponse | 2 horas | Alto |
| 6 | OLLAMA_URL en prefect-worker | 5 min | Bajo |

### Media prioridad (mejora de experiencia)

| # | Mejora | Esfuerzo | Impacto |
|---|--------|----------|---------|
| 7 | Streaming response (backend + frontend) | 1 dia | Alto |
| 8 | Resolver related_chunks en chunker | 4 horas | Medio |
| 9 | Evaluation framework con Phoenix Evals | 2-3 dias | Alto |
| 10 | Conversation memory entre turnos | 4 horas | Medio |
| 11 | Cost tracking en LLM gateway | 1 hora | Bajo |

### Baja prioridad (optimizaciones futuras)

| # | Mejora | Esfuerzo | Impacto |
|---|--------|----------|---------|
| 12 | Voyage Batch API para ingestion masiva | 4 horas | Bajo |
| 13 | HNSW tuning (ef_construct, ef) | 1 hora | Bajo |
| 14 | Cohere rank_fields para datos estructurados | 2 horas | Bajo |
| 15 | Query routing por complejidad | 4 horas | Medio |
| 16 | Document versioning | 1 dia | Bajo |
| 17 | Multimodal embeddings (diagramas) | 2-3 dias | Medio |
| 18 | LlamaIndex Workflows migration | 3-5 dias | Medio |
