# ManualIQ — Roadmap V2: Features Adicionales

> Catalogo completo de features que se pueden agregar al sistema,
> organizados por categoria y priorizados por impacto vs esfuerzo.
>
> Fecha: Abril 2026

---

## A. Features del PRD pendientes

| # | Feature | PRD | Esfuerzo | Impacto | Estado |
|---|---------|-----|----------|---------|--------|
| A1 | Respuesta por email (Resend webhook inbound) | 3.5 | 1 dia | Alto | Sprint 4 |
| A2 | Visor de PDF fuente (componente React) | 3.2 | 2 dias | Alto | Pendiente |
| A3 | Exportar conversacion a PDF | 3.3 | 1 dia | Bajo | Pendiente |
| A4 | File upload multipart en /ingest | 3.4 | 4 horas | Alto | Sprint 4 |
| A5 | Dashboard de metricas (admin) | 4.1 | 3 dias | Medio | Pendiente |
| A6 | Alertas proactivas al supervisor | 4.3 | 4 horas | Alto | Sprint 4 |
| A7 | API publica con API keys por tenant | 4.4 | 1 dia | Medio | Pendiente |
| A8 | Portugues (Brasil) como tercer idioma | 4.5 | 2 dias | Bajo | Pendiente |

## B. RAG avanzado

| # | Feature | Esfuerzo | Impacto | Estado |
|---|---------|----------|---------|--------|
| B1 | Agentic RAG (agente decide retrieval en runtime) | 3-5 dias | Alto | Pendiente |
| B2 | Graph RAG (knowledge graph de entidades) | 5+ dias | Alto | Pendiente |
| B3 | Adaptive RAG (profundidad segun complejidad) | 4 horas | Medio | Pendiente |
| B4 | Contextual chunk embeddings (voyage-context-3) | 1 dia | Medio | Pendiente |
| B5 | Late interaction ColBERT (multivectors Qdrant) | 2-3 dias | Medio | Pendiente |

## C. Features para manufactura

| # | Feature | Esfuerzo | Impacto | Estado |
|---|---------|----------|---------|--------|
| C1 | Multimodal: diagramas y figuras (voyage-multimodal-3.5) | 2-3 dias | Alto | Pendiente |
| C2 | OCR mejorado para escaneos (LlamaParse VLMs) | 1 dia | Medio | Pendiente |
| C3 | Deteccion de procedimientos obsoletos | 1 dia | Medio | Pendiente |
| C4 | Alertas de seguridad proactivas (DANGER -> supervisor) | 4 horas | Alto | Sprint 4 |
| C5 | Integracion con CMMS (SAP PM, Maximo) | 3-5 dias | Medio | Pendiente |

## D. Canales de acceso

| # | Feature | Esfuerzo | Impacto | Estado |
|---|---------|----------|---------|--------|
| D1 | WhatsApp Business (Twilio/360dialog) | 2-3 dias | Alto | Pendiente |
| D2 | Voice input (Whisper/Groq transcripcion) | 2 dias | Medio | Pendiente |
| D3 | Telegram bot | 1 dia | Bajo | Pendiente |
| D4 | Slack/Teams integration | 1-2 dias | Bajo | Pendiente |

## E. Operaciones y gobernanza

| # | Feature | Esfuerzo | Impacto | Estado |
|---|---------|----------|---------|--------|
| E1 | CI/CD pipeline (GitHub Actions) | 4 horas | Alto | Sprint 4 |
| E2 | A/B testing de prompts (Phoenix Evals) | 2 dias | Medio | Pendiente |
| E3 | Audit log inmutable | 1 dia | Medio | Pendiente |
| E4 | Role-based access (Clerk roles) | 4 horas | Medio | Pendiente |
| E5 | Data retention policies (GDPR/LOPD) | 4 horas | Bajo | Pendiente |
| E6 | Disaster recovery runbook | 2 horas | Bajo | Pendiente |

## F. Monetizacion y crecimiento

| # | Feature | Esfuerzo | Impacto | Estado |
|---|---------|----------|---------|--------|
| F1 | Billing/Stripe integration | 2-3 dias | Alto | Pendiente |
| F2 | Usage-based billing (queries adicionales) | 1 dia | Medio | Pendiente |
| F3 | Onboarding wizard (nuevo tenant) | 2 dias | Medio | Pendiente |
| F4 | Landing page publica | 2-3 dias | Medio | Pendiente |
| F5 | Multi-vertical (FinanceIQ, PetroIQ) | 5+ dias | Alto | Pendiente |

---

## Orden de implementacion recomendado

### Sprint 4: Fundamentos de produccion (este sprint)
- E1: CI/CD pipeline
- A1: Respuesta por email
- C4/A6: Alertas de seguridad al supervisor
- A4: File upload real

### Sprint 5: Experiencia completa
- A2: Visor de PDF fuente
- A5: Dashboard de metricas
- B3: Adaptive RAG
- E4: Role-based access

### Sprint 6: Canales y monetizacion
- D1: WhatsApp Business
- F1: Billing/Stripe
- F3: Onboarding wizard
- F4: Landing page

### Sprint 7: RAG avanzado
- C1: Multimodal diagramas
- B1: Agentic RAG
- E2: A/B testing de prompts
- C2: OCR mejorado

### Sprint 8+: Expansion
- B2: Graph RAG
- F5: Multi-vertical
- D2: Voice input
- C5: Integracion CMMS
