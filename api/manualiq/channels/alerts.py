"""Safety alert notifications for ManualIQ.

When a technician queries about a critical safety procedure
(DANGER/WARNING/CAUTION), automatically notify the supervisor
via email so they can verify the technician follows the procedure
correctly.

Also notifies the admin when a document has consistently low
retrieval scores (possible indexing problem).

Reference: docs/PRD.md Section 4.3, docs/ROADMAP_V2.md C4/A6.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def notify_supervisor_safety(
    resend_api_key: str,
    supervisor_email: str,
    *,
    tenant_id: str,
    user_id: str,
    query: str,
    safety_level: str,
    confidence: str,
    score: float,
    answer_preview: str = "",
) -> bool:
    """Send a safety alert to the supervisor when a critical query is detected.

    Triggered when:
    - Any chunk in the response has safety_level == "critico" (DANGER/PELIGRO)
    - Confidence is LOW on a safety-related query

    Args:
        resend_api_key: Resend API key.
        supervisor_email: Supervisor's email address.
        tenant_id: Tenant identifier.
        user_id: User who made the query.
        query: The technician's question.
        safety_level: Highest safety level in the response chunks.
        confidence: Confidence level of the response.
        score: Best retrieval score.
        answer_preview: First 300 chars of the answer.

    Returns:
        True if the alert was sent successfully.
    """
    if not resend_api_key or not supervisor_email:
        logger.debug("Safety alert skipped: no Resend key or supervisor email")
        return False

    try:
        import resend  # type: ignore[import-untyped]

        resend.api_key = resend_api_key

        subject = f"[ManualIQ] Alerta de seguridad — {safety_level.upper()}"
        if confidence == "low":
            subject += " (confianza baja)"

        body = (
            f"ALERTA DE SEGURIDAD — ManualIQ\n"
            f"{'=' * 50}\n\n"
            f"Un tecnico ha consultado sobre un procedimiento critico.\n\n"
            f"Tenant: {tenant_id}\n"
            f"Usuario: {user_id}\n"
            f"Nivel de seguridad: {safety_level.upper()}\n"
            f"Confianza: {confidence} (score: {score:.2f})\n\n"
            f"Pregunta del tecnico:\n"
            f"  {query}\n\n"
        )

        if answer_preview:
            body += f"Respuesta generada (preview):\n  {answer_preview[:300]}...\n\n"

        if confidence == "low":
            body += (
                "⚠ La confianza de la respuesta es BAJA. "
                "Se recomienda verificar personalmente con el tecnico "
                "antes de que ejecute el procedimiento.\n\n"
            )

        body += (
            f"{'=' * 50}\n"
            "Acceda al dashboard de ManualIQ para ver el detalle completo.\n"
            "Este mensaje fue generado automaticamente."
        )

        resend.Emails.send(
            {
                "from": "ManualIQ Alertas <alertas@manualiq.com>",
                "to": [supervisor_email],
                "subject": subject,
                "text": body,
            }
        )

        logger.info(
            "Safety alert sent to %s for tenant %s (level=%s, confidence=%s)",
            supervisor_email,
            tenant_id,
            safety_level,
            confidence,
        )
        return True

    except Exception as exc:
        logger.error("Failed to send safety alert: %s", exc)
        return False


def should_alert_supervisor(
    chunks_safety_levels: list[str],
    confidence: str,
) -> bool:
    """Determine if a supervisor alert should be triggered.

    Triggers when:
    1. Any chunk has safety_level "critico" (DANGER/PELIGRO)
    2. Any chunk has safety_level "precaucion" AND confidence is "low"

    Args:
        chunks_safety_levels: Safety levels of the top chunks used.
        confidence: Confidence level of the response.

    Returns:
        True if an alert should be sent.
    """
    if "critico" in chunks_safety_levels:
        return True
    if "precaucion" in chunks_safety_levels and confidence == "low":
        return True
    return False
