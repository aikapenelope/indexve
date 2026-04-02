"""Email channel: receive questions via email, respond via email.

Implements PRD Section 3.5: A technician without PC access sends a
question by email to consulta@empresa.manualiq.com and receives the
answer with citations and disclaimer by email in <2 minutes.

Flow:
1. Resend inbound webhook delivers the email to POST /email/inbound
2. We extract sender, tenant (from recipient address), and query text
3. Run the same RAG pipeline as /query
4. Send the response via Resend with citations formatted for email

Reference: docs/PRD.md Section 3.5, docs/ROADMAP_V2.md A1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InboundEmail:
    """Parsed inbound email from Resend webhook."""

    sender: str
    recipient: str
    subject: str
    body_text: str
    tenant_id: str  # Extracted from recipient address.


def parse_inbound_email(payload: dict[str, object]) -> InboundEmail | None:
    """Parse a Resend inbound webhook payload.

    Resend sends a POST with the email data. We extract the sender,
    recipient, subject, and plain text body.

    The tenant_id is derived from the recipient address:
    consulta@{tenant_slug}.manualiq.com -> tenant_slug

    Args:
        payload: The webhook JSON payload from Resend.

    Returns:
        InboundEmail if parsing succeeds, None if the email is invalid.
    """
    try:
        sender = str(payload.get("from", ""))
        recipient = str(payload.get("to", ""))
        subject = str(payload.get("subject", ""))
        body_text = str(payload.get("text", ""))

        if not sender or not body_text.strip():
            logger.warning("Inbound email missing sender or body")
            return None

        # Extract tenant from recipient: consulta@acme.manualiq.com -> acme
        tenant_match = re.match(r"consulta@([a-zA-Z0-9_-]+)\.manualiq\.com", recipient)
        if not tenant_match:
            # Fallback: try to extract from any @tenant.manualiq.com pattern.
            tenant_match = re.match(r"[^@]+@([a-zA-Z0-9_-]+)\.manualiq\.com", recipient)

        tenant_id = tenant_match.group(1) if tenant_match else "default"

        return InboundEmail(
            sender=sender,
            recipient=recipient,
            subject=subject,
            body_text=body_text.strip(),
            tenant_id=tenant_id,
        )
    except Exception as exc:
        logger.error("Failed to parse inbound email: %s", exc)
        return None


def format_email_response(
    query: str,
    answer: str,
    sources: list[dict[str, object]],
    confidence: str,
    score: float,
) -> str:
    """Format the RAG response as an email body.

    Includes:
    - The original question
    - The answer with citations
    - Source references with scores
    - Low confidence warning if applicable
    - Disclaimer

    Args:
        query: The original question from the technician.
        answer: The RAG-generated answer.
        sources: List of source chunk dicts with doc_id, section_path, etc.
        confidence: Confidence level (high/low/none).
        score: Best retrieval score.

    Returns:
        Formatted plain text email body.
    """
    lines: list[str] = []

    lines.append("ManualIQ — Respuesta a su consulta tecnica")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Pregunta: {query}")
    lines.append("")

    if confidence == "low":
        lines.append(
            f"⚠ ATENCION: Confianza baja (score: {score:.2f}). "
            "Se recomienda verificar con el supervisor."
        )
        lines.append("")

    lines.append("Respuesta:")
    lines.append("-" * 30)
    lines.append(answer)
    lines.append("")

    if sources:
        lines.append("Fuentes consultadas:")
        lines.append("-" * 30)
        for i, src in enumerate(sources, 1):
            lines.append(
                f"  {i}. [{src.get('doc_id', '')}] "
                f"Seccion: {src.get('section_path', '')} | "
                f"Pagina: {src.get('page_ref', '')} | "
                f"Score: {float(str(src.get('score', 0))):.2f}"
            )
        lines.append("")

    lines.append("-" * 50)
    lines.append(
        "IMPORTANTE: Verificar siempre con el supervisor antes de "
        "ejecutar procedimientos criticos de seguridad."
    )
    lines.append("")
    lines.append("Este mensaje fue generado automaticamente por ManualIQ.")
    lines.append("No responda a este correo.")

    return "\n".join(lines)


async def send_email_response(
    resend_api_key: str,
    to_email: str,
    subject: str,
    body: str,
    from_email: str = "ManualIQ <respuesta@manualiq.com>",
) -> bool:
    """Send the response email via Resend API.

    Args:
        resend_api_key: Resend API key.
        to_email: Recipient email (the technician).
        subject: Email subject (Re: original subject).
        body: Formatted email body.
        from_email: Sender address.

    Returns:
        True if sent successfully.
    """
    try:
        import resend  # type: ignore[import-untyped]

        resend.api_key = resend_api_key

        resend.Emails.send(
            {
                "from": from_email,
                "to": [to_email],
                "subject": subject,
                "text": body,
            }
        )
        logger.info("Email response sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Failed to send email response to %s: %s", to_email, exc)
        return False
