"""Query intelligence: sub-questions, intent classification, and multilingual support.

Addresses issues from KNOWN_ISSUES.md:
- 1.2: Evidencia dispersa -> Sub-question decomposition
- 1.6: Queries ambiguas -> Intent classification with Gemini Flash
- 3.1: LLM responde en ingles -> Language detection guardrail + retry
- 3.2: Cross-lingual retrieval -> Query expansion ES->EN
- 3.3: Terminologia tecnica -> Configurable technical glossary per tenant

All LLM calls are injected as callables so the module stays testable
without API keys.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.6: Intent classification
# ---------------------------------------------------------------------------


class QueryIntent(Enum):
    """Classified intent of a user query."""

    SPECIFIC = "specific"  # Clear, answerable question
    AMBIGUOUS = "ambiguous"  # Needs clarification
    LISTING = "listing"  # "List all X" type queries
    PROCEDURE = "procedure"  # Step-by-step procedure request
    COMPARISON = "comparison"  # Compare two specs/procedures
    OUT_OF_SCOPE = "out_of_scope"  # Not related to technical manuals


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float
    clarification_prompt: str | None = None
    expanded_query: str | None = None


# Prompt template for Gemini Flash intent classification.
_INTENT_CLASSIFICATION_PROMPT = """Clasifica la siguiente pregunta de un tecnico industrial.

Pregunta: "{query}"

Responde SOLO con un JSON valido (sin markdown, sin explicacion):
{{
  "intent": "specific|ambiguous|listing|procedure|comparison|out_of_scope",
  "confidence": 0.0-1.0,
  "clarification": "pregunta de clarificacion si es ambigua, null si no",
  "expanded": "version mas especifica de la query si es posible, null si no"
}}

Reglas:
- "ambiguous": la pregunta podria referirse a multiples equipos, procedimientos o contextos.
- "out_of_scope": la pregunta no tiene relacion con manuales tecnicos o equipos industriales.
- "listing": pide una lista completa (todas las specs, todos los pasos, etc).
- "procedure": pide un procedimiento paso a paso.
- "comparison": compara dos o mas elementos.
- "specific": pregunta clara y directa sobre un tema tecnico.
"""


def classify_intent(
    query: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> IntentResult:
    """Classify the intent of a user query using Gemini Flash.

    If no LLM function is provided, falls back to heuristic classification.

    Args:
        query: The user's question in Spanish.
        llm_fn: Callable that takes a prompt and returns LLM response.
            In production, this calls Gemini Flash.

    Returns:
        IntentResult with the classified intent.
    """
    if llm_fn is not None:
        return _classify_with_llm(query, llm_fn)
    return _classify_heuristic(query)


def _classify_with_llm(
    query: str,
    llm_fn: Callable[[str], str],
) -> IntentResult:
    """Classify intent using Gemini Flash."""
    import json

    prompt = _INTENT_CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = llm_fn(prompt)
        # Strip markdown code fences if present.
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
        data = json.loads(cleaned)

        intent_str = data.get("intent", "specific")
        try:
            intent = QueryIntent(intent_str)
        except ValueError:
            intent = QueryIntent.SPECIFIC

        return IntentResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            clarification_prompt=data.get("clarification"),
            expanded_query=data.get("expanded"),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Intent classification LLM parse error: %s", exc)
        return _classify_heuristic(query)


def _classify_heuristic(query: str) -> IntentResult:
    """Fallback heuristic intent classification without LLM."""
    lower = query.lower().strip()

    # Check for listing patterns.
    if re.search(r"\b(lista|enumera|todos los|todas las|cuales son)\b", lower):
        return IntentResult(intent=QueryIntent.LISTING, confidence=0.7)

    # Check for procedure patterns.
    if re.search(r"\b(como se|procedimiento|pasos para|paso a paso)\b", lower):
        return IntentResult(intent=QueryIntent.PROCEDURE, confidence=0.7)

    # Check for comparison patterns.
    if re.search(r"\b(diferencia entre|comparar|versus|vs\.?)\b", lower):
        return IntentResult(intent=QueryIntent.COMPARISON, confidence=0.7)

    # Check for ambiguity: very short queries or missing context.
    words = lower.split()
    if len(words) <= 3:
        return IntentResult(
            intent=QueryIntent.AMBIGUOUS,
            confidence=0.6,
            clarification_prompt=(
                "Su pregunta es muy breve. Podria especificar el equipo "
                "o el tipo de procedimiento al que se refiere?"
            ),
        )

    return IntentResult(intent=QueryIntent.SPECIFIC, confidence=0.5)


# ---------------------------------------------------------------------------
# 1.2: Sub-question decomposition
# ---------------------------------------------------------------------------


@dataclass
class SubQuestion:
    """A decomposed sub-question for distributed evidence retrieval."""

    question: str
    focus: str  # What aspect this sub-question targets


_SUB_QUESTION_PROMPT = """Descompone la siguiente pregunta tecnica en sub-preguntas independientes
que se puedan buscar por separado en una base de datos de manuales tecnicos.

Pregunta original: "{query}"

Responde SOLO con un JSON valido (sin markdown):
{{
  "sub_questions": [
    {{"question": "sub-pregunta 1", "focus": "aspecto que busca"}},
    {{"question": "sub-pregunta 2", "focus": "aspecto que busca"}}
  ]
}}

Reglas:
- Maximo 4 sub-preguntas.
- Cada sub-pregunta debe ser autocontenida (entendible sin la original).
- Si la pregunta original es simple y directa, devuelve solo 1 sub-pregunta
  que sea la misma pregunta.
- Incluye sub-preguntas sobre seguridad si el tema lo amerita.
"""


def decompose_query(
    query: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> list[SubQuestion]:
    """Decompose a complex query into sub-questions for distributed retrieval.

    Uses Gemini Flash to break down queries like "List all torque specs
    for the C7 engine" into targeted sub-questions that can each retrieve
    relevant chunks independently.

    Args:
        query: The user's original question.
        llm_fn: Callable for Gemini Flash. If None, returns the original
            query as a single sub-question.

    Returns:
        List of SubQuestion objects.
    """
    if llm_fn is None:
        return [SubQuestion(question=query, focus="general")]

    import json

    prompt = _SUB_QUESTION_PROMPT.format(query=query)

    try:
        response = llm_fn(prompt)
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
        data = json.loads(cleaned)

        sub_questions: list[SubQuestion] = []
        for sq in data.get("sub_questions", []):
            sub_questions.append(
                SubQuestion(
                    question=sq.get("question", query),
                    focus=sq.get("focus", "general"),
                )
            )

        if not sub_questions:
            return [SubQuestion(question=query, focus="general")]

        return sub_questions[:4]  # Hard cap at 4.

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Sub-question decomposition parse error: %s", exc)
        return [SubQuestion(question=query, focus="general")]


# ---------------------------------------------------------------------------
# 3.1: Language detection guardrail
# ---------------------------------------------------------------------------


def detect_response_language(text: str) -> str:
    """Detect the primary language of a response using heuristics.

    Returns 'es' for Spanish, 'en' for English, or 'mixed'.
    This is a lightweight check to avoid adding a heavy NLP dependency.
    """
    # Common Spanish words that rarely appear in English technical text.
    es_markers = [
        " el ",
        " la ",
        " los ",
        " las ",
        " del ",
        " en ",
        " que ",
        " por ",
        " con ",
        " para ",
        " una ",
        " este ",
        " esta ",
        " son ",
        " tiene ",
        " debe ",
        " puede ",
        " como ",
    ]
    # Common English words that rarely appear in Spanish technical text.
    en_markers = [
        " the ",
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " with ",
        " from ",
        " this ",
        " that ",
        " should ",
        " must ",
        " can ",
        " will ",
    ]

    lower = f" {text.lower()} "
    es_count = sum(1 for m in es_markers if m in lower)
    en_count = sum(1 for m in en_markers if m in lower)

    total = es_count + en_count
    if total == 0:
        return "es"  # Default to Spanish (expected language).

    es_ratio = es_count / total
    if es_ratio > 0.6:
        return "es"
    if es_ratio < 0.4:
        return "en"
    return "mixed"


@dataclass
class LanguageGuardrailResult:
    """Result of the language guardrail check."""

    passed: bool
    detected_language: str
    retried: bool = False
    original_response: str | None = None


def enforce_spanish_response(
    response: str,
    *,
    generate_fn: Callable[[str], str] | None = None,
    original_prompt: str = "",
    max_retries: int = 1,
) -> LanguageGuardrailResult:
    """Guardrail: ensure the LLM response is in Spanish.

    If the response is detected as English or mixed, retry with a
    reinforced prompt. This addresses the documented issue where LLMs
    "forget" to respond in Spanish when chunks are in English.

    Args:
        response: The LLM's response to check.
        generate_fn: LLM callable for retry. If None, just reports.
        original_prompt: The original prompt for retry context.
        max_retries: Maximum retry attempts.

    Returns:
        LanguageGuardrailResult with the check outcome.
    """
    lang = detect_response_language(response)

    if lang == "es":
        return LanguageGuardrailResult(passed=True, detected_language=lang)

    logger.warning("Response language detected as '%s', expected 'es'", lang)

    if generate_fn is None or not original_prompt or max_retries < 1:
        return LanguageGuardrailResult(
            passed=False,
            detected_language=lang,
            original_response=response,
        )

    # Retry with reinforced Spanish instruction.
    reinforced_prompt = (
        "IMPORTANTE: Tu respuesta anterior fue detectada en ingles o mixta. "
        "Debes responder COMPLETAMENTE en espanol. Traduce toda la informacion "
        "tecnica al espanol. Solo incluye texto en ingles cuando cites "
        "directamente un fragmento del manual original.\n\n" + original_prompt
    )

    retried_response = generate_fn(reinforced_prompt)
    retried_lang = detect_response_language(retried_response)

    return LanguageGuardrailResult(
        passed=retried_lang == "es",
        detected_language=retried_lang,
        retried=True,
        original_response=response,
    )


# ---------------------------------------------------------------------------
# 3.2: Cross-lingual query expansion
# ---------------------------------------------------------------------------


_QUERY_EXPANSION_PROMPT = """Traduce la siguiente pregunta tecnica del espanol al ingles tecnico.
Mantiene la terminologia tecnica precisa. Responde SOLO con la traduccion, sin explicacion.

Pregunta en espanol: "{query}"

Traduccion al ingles:"""


def expand_query_crosslingual(
    query: str,
    *,
    llm_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Expand a Spanish query with an English translation for cross-lingual retrieval.

    When the user asks in Spanish but documents are in English, searching
    with both the original Spanish query and an English translation
    improves retrieval quality (Voyage-4 supports cross-lingual but
    explicit expansion helps).

    Args:
        query: The user's question in Spanish.
        llm_fn: Callable for Gemini Flash translation.

    Returns:
        List of queries to search with: [original_es, translated_en].
        If translation fails, returns just the original.
    """
    queries = [query]

    if llm_fn is None:
        return queries

    try:
        prompt = _QUERY_EXPANSION_PROMPT.format(query=query)
        en_query = llm_fn(prompt).strip()
        if en_query and en_query.lower() != query.lower():
            queries.append(en_query)
            logger.debug("Query expanded: ES='%s' -> EN='%s'", query, en_query)
    except Exception as exc:
        logger.warning("Cross-lingual expansion failed: %s", exc)

    return queries


# ---------------------------------------------------------------------------
# 3.3: Technical glossary
# ---------------------------------------------------------------------------


@dataclass
class TechnicalGlossary:
    """Configurable technical glossary per tenant.

    Maps English technical terms to their Spanish equivalents (and
    regional synonyms). Injected into the LLM prompt to ensure
    consistent terminology.
    """

    # Default manufacturing glossary.
    entries: dict[str, list[str]] = field(default_factory=dict)

    def add_term(self, english: str, spanish_variants: list[str]) -> None:
        """Add a term with its Spanish translations/synonyms."""
        self.entries[english.lower()] = [s.lower() for s in spanish_variants]

    def get_prompt_injection(self) -> str:
        """Generate a glossary section to inject into the LLM prompt.

        Returns a formatted string that can be appended to the system
        prompt to ensure consistent terminology.
        """
        if not self.entries:
            return ""

        lines = ["Glosario tecnico (usa estos terminos en la respuesta):"]
        for en_term, es_variants in sorted(self.entries.items()):
            variants_str = " / ".join(es_variants)
            lines.append(f"- {en_term} = {variants_str}")

        return "\n".join(lines)

    def expand_query_with_synonyms(self, query: str) -> list[str]:
        """Expand a query with known synonyms from the glossary.

        If the query contains a known term (in either language), add
        variants to improve retrieval.

        Returns:
            List of query variants including the original.
        """
        variants = [query]
        lower_query = query.lower()

        for en_term, es_terms in self.entries.items():
            all_terms = [en_term] + es_terms
            for term in all_terms:
                if term in lower_query:
                    # Add variants with the other synonyms.
                    for other in all_terms:
                        if other != term:
                            variant = lower_query.replace(term, other)
                            if variant != lower_query and variant not in variants:
                                variants.append(variant)
                    break  # Only expand the first matching term.

        return variants


def create_default_manufacturing_glossary() -> TechnicalGlossary:
    """Create the default glossary for the manufacturing vertical.

    Based on common terminology issues documented in KNOWN_ISSUES.md
    section 3.3.
    """
    glossary = TechnicalGlossary()

    # Common manufacturing terms with regional variants.
    glossary.add_term("torque wrench", ["llave dinamometrica", "torquimetro"])
    glossary.add_term("gasket", ["junta", "empaque", "empaquetadura"])
    glossary.add_term("bearing", ["rodamiento", "cojinete", "balero"])
    glossary.add_term("bolt", ["perno", "tornillo", "bulón"])
    glossary.add_term("nut", ["tuerca"])
    glossary.add_term("washer", ["arandela"])
    glossary.add_term("seal", ["sello", "retén", "obturador"])
    glossary.add_term("valve", ["válvula"])
    glossary.add_term("pump", ["bomba"])
    glossary.add_term("filter", ["filtro"])
    glossary.add_term("hose", ["manguera"])
    glossary.add_term("clamp", ["abrazadera", "grapa"])
    glossary.add_term("fitting", ["conexión", "racor", "acople"])
    glossary.add_term("o-ring", ["anillo o", "junta tórica"])
    glossary.add_term("crankshaft", ["cigüeñal"])
    glossary.add_term("camshaft", ["árbol de levas"])
    glossary.add_term("piston", ["pistón", "émbolo"])
    glossary.add_term("cylinder head", ["culata", "cabeza de cilindro"])
    glossary.add_term("connecting rod", ["biela"])
    glossary.add_term("flywheel", ["volante de inercia", "volante del motor"])
    glossary.add_term("coolant", ["refrigerante", "anticongelante"])
    glossary.add_term("lubricant", ["lubricante", "aceite"])
    glossary.add_term("throttle", ["acelerador", "mariposa"])
    glossary.add_term("injector", ["inyector"])
    glossary.add_term("turbocharger", ["turbocompresor", "turbo"])
    glossary.add_term("alternator", ["alternador"])
    glossary.add_term("starter motor", ["motor de arranque"])
    glossary.add_term("radiator", ["radiador"])
    glossary.add_term("thermostat", ["termostato"])
    glossary.add_term("exhaust manifold", ["múltiple de escape", "colector de escape"])
    glossary.add_term(
        "intake manifold", ["múltiple de admisión", "colector de admisión"]
    )

    return glossary
