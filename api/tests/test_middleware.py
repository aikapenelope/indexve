"""Tests for rate limiter, PII detection, and language detection."""

from manualiq.middleware.rate_limiter import TenantPlan, check_cost_alert
from manualiq.middleware.resilience import detect_pii, sanitize_pii
from manualiq.query.intelligence import (
    QueryIntent,
    _classify_heuristic,
    detect_response_language,
)


class TestCheckCostAlert:
    def test_no_alert_below_threshold(self) -> None:
        result = check_cost_alert("t1", daily_count=10, daily_limit=50)
        assert result is None

    def test_alert_near_limit(self) -> None:
        result = check_cost_alert("t1", daily_count=48, daily_limit=50)
        assert result is not None
        assert result.tenant_id == "t1"
        assert result.percentage >= 0.9

    def test_zero_limit(self) -> None:
        result = check_cost_alert("t1", daily_count=10, daily_limit=0)
        assert result is None

    def test_plan_values(self) -> None:
        assert TenantPlan.FREE.value == 50
        assert TenantPlan.PRO.value == 500
        assert TenantPlan.ENTERPRISE.value == 10_000


class TestPIIDetection:
    def test_detects_email(self) -> None:
        result = detect_pii("Contact me at john@example.com for details")
        assert result.has_pii is True
        assert any(d["type"] == "email" for d in result.detections)

    def test_detects_cedula(self) -> None:
        result = detect_pii("Mi cedula es V-12345678")
        assert result.has_pii is True
        assert any(d["type"] == "cedula" for d in result.detections)

    def test_clean_text(self) -> None:
        result = detect_pii("El torque de la culata es 50 Nm")
        assert result.has_pii is False

    def test_detects_ssn(self) -> None:
        result = detect_pii("SSN: 123-45-6789")
        assert result.has_pii is True


class TestSanitizePII:
    def test_redacts_email(self) -> None:
        result = sanitize_pii("Send to john@example.com please")
        assert result.has_pii is True
        assert result.sanitized_text is not None
        assert "john@example.com" not in result.sanitized_text
        assert "EMAIL_REDACTED" in result.sanitized_text

    def test_clean_text_unchanged(self) -> None:
        text = "El motor opera a 2400 RPM."
        result = sanitize_pii(text)
        assert result.sanitized_text == text


class TestLanguageDetection:
    def test_spanish(self) -> None:
        text = "El motor debe ser revisado por el tecnico para verificar que las condiciones son correctas."
        assert detect_response_language(text) == "es"

    def test_english(self) -> None:
        text = "The engine should be inspected by the technician to verify that conditions are correct."
        assert detect_response_language(text) == "en"

    def test_default_to_spanish(self) -> None:
        # Very short text with no markers defaults to Spanish.
        assert detect_response_language("OK") == "es"


class TestHeuristicIntentClassification:
    def test_listing(self) -> None:
        result = _classify_heuristic("Lista todos los torques del motor C7")
        assert result.intent == QueryIntent.LISTING

    def test_procedure(self) -> None:
        result = _classify_heuristic("Como se cambia el filtro de aceite?")
        assert result.intent == QueryIntent.PROCEDURE

    def test_comparison(self) -> None:
        result = _classify_heuristic("Diferencia entre el C7 y el C9")
        assert result.intent == QueryIntent.COMPARISON

    def test_ambiguous_short(self) -> None:
        result = _classify_heuristic("cambio?")
        assert result.intent == QueryIntent.AMBIGUOUS

    def test_specific(self) -> None:
        result = _classify_heuristic(
            "Cual es el torque de apriete de la culata del motor Caterpillar C7?"
        )
        assert result.intent == QueryIntent.SPECIFIC
