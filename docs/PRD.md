# ManualIQ — Product Requirements Document

> Version 1.0 — Abril 2026

---

## 1. Vision del Producto

ManualIQ es un sistema RAG (Retrieval-Augmented Generation) SaaS multi-tenant que permite a tecnicos industriales consultar manuales de equipos en ingles y recibir respuestas precisas en espanol, con citas exactas de pagina, seccion y documento fuente.

El primer vertical es **manufactura**. Los siguientes seran **financiero** y **petroleos**.

---

## 2. Usuarios

### 2.1 Tecnico de planta (usuario final)

- Dominio limitado del ingles tecnico.
- Accede desde PC en planta o desde celular via email.
- Necesita respuestas rapidas (<5 segundos) con citas verificables.
- No tolera datos inventados: prefiere "no se" a una respuesta incorrecta.

### 2.2 Supervisor de mantenimiento

- Recibe alertas cuando el sistema tiene baja confianza en una respuesta.
- Valida respuestas criticas (procedimientos de seguridad).
- Gestiona que manuales estan disponibles para su equipo.

### 2.3 Administrador de empresa (tenant admin)

- Sube y gestiona el corpus documental de su empresa.
- Configura usuarios y permisos dentro de su organizacion.
- Ve metricas de uso: queries por dia, documentos mas consultados, temas frecuentes.

### 2.4 Administrador de plataforma (nosotros)

- Gestiona tenants, planes, facturacion.
- Monitorea salud del sistema, costos de API, calidad de respuestas.
- Ejecuta evaluaciones de calidad periodicas.

---

## 3. Funcionalidades Core (MVP)

### 3.1 Chat con manuales

**Que hace**: El tecnico escribe una pregunta en espanol. El sistema busca en los manuales indexados, recupera los fragmentos relevantes, y genera una respuesta en espanol con citas exactas.

**Requisitos**:
- Respuesta en <5 segundos (P95).
- Cada afirmacion tecnica incluye: `[Manual: nombre, Seccion: X, Pagina: Y]`.
- Si el chunk fuente esta en ingles, incluir fragmento original + traduccion.
- Si hay DANGER/WARNING/CAUTION, mostrar primero en negrita.
- Si no hay informacion, responder "No encontre informacion sobre esto en los manuales disponibles." NUNCA inventar.
- Confianza baja (score <0.75): avisar al tecnico y notificar al supervisor.
- Disclaimer de verificacion con supervisor al final de cada respuesta.

### 3.2 Visor de documentos fuente

**Que hace**: Junto a cada respuesta, mostrar los documentos fuente con score de relevancia y link para ver el PDF original.

**Requisitos**:
- Mostrar top 3-5 chunks recuperados con su score.
- Link directo al PDF en la pagina exacta citada.
- Indicador visual de idioma del documento fuente (ES/EN).

### 3.3 Historial de conversaciones

**Que hace**: Cada usuario tiene historial persistente de sus consultas y respuestas.

**Requisitos**:
- Historial por usuario, aislado por tenant.
- Busqueda en historial propio.
- Exportar conversacion a PDF (para adjuntar a ordenes de trabajo).

### 3.4 Ingestion de documentos

**Que hace**: El admin de empresa sube manuales (PDF, DOCX) y el sistema los procesa automaticamente.

**Requisitos**:
- Formatos soportados: PDF (prioridad), DOCX, PPTX.
- Parsing automatico: extraccion de texto, tablas, imagenes con texto.
- Chunking inteligente: respetar limites de procedimientos, tablas completas, part numbers.
- Metadata automatica: detectar idioma, tipo de documento, fabricante si es posible.
- Metadata manual: el admin puede agregar/editar equipment, manufacturer, procedure_type.
- Notificacion cuando el procesamiento termina (o falla).
- Re-procesamiento manual de documentos individuales.

### 3.5 Respuesta por email

**Que hace**: Un tecnico en planta sin acceso a PC envia su pregunta por email y recibe la respuesta por email.

**Requisitos**:
- Direccion de email dedicada por tenant (ej: consulta@empresa.manualiq.com).
- Respuesta en <2 minutos.
- Formato de email: respuesta + citas + links a documentos + disclaimer.
- Rate limit por email sender para prevenir abuso.

### 3.6 Multi-tenancy

**Que hace**: Cada empresa cliente tiene su propio espacio aislado con sus documentos, usuarios, y configuracion.

**Requisitos**:
- Aislamiento total de datos: un tenant NUNCA ve documentos de otro.
- Aislamiento a nivel de vector DB (shard keys en Qdrant) + aplicacion (Clerk orgs).
- Cada tenant tiene su propia coleccion de documentos.
- Configuracion por tenant: idioma preferido, equipos, fabricantes.

---

## 4. Funcionalidades Secundarias (Post-MVP)

### 4.1 Dashboard de metricas

- Queries por dia/semana/mes.
- Documentos mas consultados.
- Temas sin respuesta (queries con score bajo).
- Costos de API por tenant.
- Tiempo promedio de respuesta.

### 4.2 Feedback del usuario

- Pulgar arriba/abajo en cada respuesta.
- Comentario opcional: "la respuesta fue incorrecta porque..."
- Feedback alimenta evaluaciones de calidad.

### 4.3 Alertas proactivas

- Notificar al supervisor cuando un tecnico consulta sobre un procedimiento critico de seguridad.
- Notificar al admin cuando un documento no tiene respuestas utiles (posible problema de indexacion).

### 4.4 API publica

- REST API para integrar ManualIQ en sistemas existentes (CMMS, ERP).
- Autenticacion via API keys por tenant.
- Rate limiting por plan.

### 4.5 Soporte multilingue ampliado

- Portugues (Brasil) como tercer idioma.
- Queries en ingles con respuestas en ingles (para supervisores bilingues).

---

## 5. Requisitos No Funcionales

### 5.1 Performance

| Metrica | Objetivo | Critico |
|---------|----------|---------|
| Latencia de respuesta (P50) | <3 segundos | <5 segundos |
| Latencia de respuesta (P95) | <5 segundos | <10 segundos |
| Ingestion de documento | <5 minutos por PDF | <15 minutos |
| Disponibilidad | 99.5% mensual | 99% mensual |

### 5.2 Seguridad

- Aislamiento de datos por tenant (vector DB + aplicacion).
- Guardrails de input: prompt injection, PII, queries fuera de scope.
- Guardrails de output: verificacion de citas, deteccion de alucinaciones.
- Rate limiting por tenant y por usuario.
- Secrets en Docker secrets, nunca en codigo.
- HTTPS obligatorio en produccion.
- Acceso al VPS solo via Tailscale (sin SSH publico).

### 5.3 Calidad de respuestas

| Metrica | Objetivo MVP | Objetivo 6 meses |
|---------|-------------|-------------------|
| Faithfulness (respuesta fiel a los chunks) | >85% | >92% |
| Answer relevancy (responde la pregunta) | >80% | >90% |
| Citation accuracy (citas correctas) | >90% | >95% |
| Hallucination rate | <10% | <5% |

Medido con Phoenix Evals sobre set de regresion de 100 preguntas.

### 5.4 Escalabilidad

- MVP: 1-5 tenants, ~50 usuarios concurrentes, ~7000 documentos totales.
- 6 meses: 10-20 tenants, ~200 usuarios concurrentes, ~20000 documentos.
- 12 meses: 50+ tenants, 3 verticales (manufactura, financiero, petroleos).

### 5.5 Observabilidad

- Tracing end-to-end de cada query (Phoenix).
- Costos por request y por tenant.
- Evaluaciones automaticas semanales.
- Alertas cuando un servicio cae.

---

## 6. Fuera de Scope (NO hacer)

- No procesar datos personales de empleados ni clientes finales.
- No tomar decisiones autonomas de mantenimiento.
- No reemplazar al supervisor para procedimientos criticos.
- No acceder a sistemas ERP/SCADA.
- No generar ordenes de trabajo automaticamente.
- No hacer traduccion general (solo en contexto de manuales tecnicos).
- No soportar audio/voz en MVP (futuro).

---

## 7. Modelo de Negocio

### Planes propuestos

| Plan | Documentos | Usuarios | Queries/mes | Precio/mes |
|------|-----------|----------|-------------|------------|
| Starter | 100 | 5 | 1,000 | $49 |
| Professional | 500 | 20 | 5,000 | $149 |
| Enterprise | Ilimitado | Ilimitado | Ilimitado | Custom |

### Metricas de exito

- **Adopcion**: >70% de tecnicos usan el sistema al menos 1x/semana despues de 30 dias.
- **Calidad**: <5% de respuestas marcadas como incorrectas por usuarios.
- **Retencion**: >90% de tenants renuevan despues de 3 meses.
- **Tiempo ahorrado**: Reduccion de 20-60 min a <1 min por consulta tecnica.
