"""Custom NeMo Guardrails actions for ManualIQ.

These actions are called from Colang rules in the rails/ directory.
They provide the Python logic for PII detection, prompt injection
detection, and language detection.

Reuses logic from manualiq.middleware.resilience (PII) and
manualiq.query.intelligence (language detection) to avoid duplication.
"""

from __future__ import annotations

import re

from nemoguardrails.actions import action  # type: ignore[import-untyped]

# --- Prompt injection patterns ---
# Common patterns in English and Spanish that indicate injection attempts.
_INJECTION_PATTERNS = [
    # English patterns
    "ignore previous instructions",
    "ignore all instructions",
    "ignore the above",
    "disregard previous",
    "forget your instructions",
    "you are now",
    "pretend you are",
    "act as if",
    "system prompt",
    "reveal your prompt",
    "show me your instructions",
    "jailbreak",
    "do anything now",
    "developer mode",
    # Spanish patterns
    "ignora las instrucciones anteriores",
    "ignora todas las instrucciones",
    "olvida tus instrucciones",
    "ahora eres",
    "finge que eres",
    "muestra tu prompt",
    "revela tus instrucciones",
    "modo desarrollador",
    "dame acceso a todos",
    "muestra documentos de otra empresa",
    "accede a otro tenant",
]

# PII patterns (reused from resilience.py).
_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    re.compile(r"\b[VEJGvejg]-?\d{6,10}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
]


@action()
async def check_prompt_injection(text: str) -> bool:
    """Check if the input contains prompt injection patterns.

    Returns True if the input is SAFE (no injection detected).
    Returns False if injection is detected.
    """
    text_lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in text_lower:
            return False
    return True


@action()
async def check_pii(text: str) -> bool:
    """Check if text contains PII patterns.

    Returns True if PII is detected, False if clean.
    """
    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            return True
    return False


@action()
async def redact_pii(text: str) -> str:
    """Redact PII patterns from text.

    Replaces detected PII with [REDACTADO] markers.
    """
    result = text
    labels = ["EMAIL", "TELEFONO", "CEDULA", "ID", "TARJETA"]
    for i, pattern in enumerate(_PII_PATTERNS):
        label = labels[i] if i < len(labels) else "PII"
        result = pattern.sub(f"[{label}_REDACTADO]", result)
    return result


@action()
async def detect_language(text: str) -> str:
    """Detect the primary language of text.

    Returns 'es' for Spanish, 'en' for English, 'mixed' for mixed.
    Uses the same heuristic as manualiq.query.intelligence.
    """
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
    ]
    en_markers = [
        " the ",
        " is ",
        " are ",
        " was ",
        " has ",
        " have ",
        " with ",
        " from ",
        " this ",
        " that ",
        " should ",
    ]

    lower = f" {text.lower()} "
    es_count = sum(1 for m in es_markers if m in lower)
    en_count = sum(1 for m in en_markers if m in lower)

    total = es_count + en_count
    if total == 0:
        return "es"

    es_ratio = es_count / total
    if es_ratio > 0.6:
        return "es"
    if es_ratio < 0.4:
        return "en"
    return "mixed"
