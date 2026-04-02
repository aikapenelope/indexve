"""WhatsApp channel via Twilio for ManualIQ.

Technicians send questions via WhatsApp, receive answers with citations
and the source PDF attached. No login required — authentication is via
phone number whitelist per tenant.

Flow:
1. Twilio webhook delivers incoming WhatsApp message to POST /whatsapp/inbound
2. We look up the tenant by sender phone number (whitelist in Redis)
3. Run the full RAG pipeline (hybrid search + rerank + Claude)
4. Send response as WhatsApp text message
5. Send the top source PDF as a document attachment (same conversation, no extra cost)

Pricing: Within the 24h customer service window (technician initiates),
all messages including document attachments are FREE. The PDF attachment
counts as part of the same message, not an extra charge.

Reference: docs/PRD.md Section 3.5, docs/ROADMAP_V2.md D1.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Twilio WhatsApp sandbox prefix.
_WHATSAPP_PREFIX = "whatsapp:"


def parse_twilio_webhook(form_data: dict[str, str]) -> dict[str, str] | None:
    """Parse an incoming Twilio WhatsApp webhook.

    Twilio sends form-encoded data with fields like Body, From, To, etc.

    Args:
        form_data: The form fields from the Twilio webhook POST.

    Returns:
        Dict with sender, body, and message_sid, or None if invalid.
    """
    body = form_data.get("Body", "").strip()
    sender = form_data.get("From", "")
    to = form_data.get("To", "")
    message_sid = form_data.get("MessageSid", "")

    if not body or not sender:
        logger.warning("WhatsApp webhook missing Body or From")
        return None

    # Strip the whatsapp: prefix from phone numbers.
    sender_phone = sender.replace(_WHATSAPP_PREFIX, "")
    to_phone = to.replace(_WHATSAPP_PREFIX, "")

    return {
        "sender": sender_phone,
        "sender_raw": sender,
        "to": to_phone,
        "body": body,
        "message_sid": message_sid,
    }


async def lookup_tenant_by_phone(
    redis_client: object,
    phone: str,
) -> dict[str, str] | None:
    """Look up tenant and user info by phone number from Redis whitelist.

    Each tenant maintains a whitelist of authorized phone numbers:
    Key: whatsapp:phone:{phone_number}
    Value: hash with tenant_id, user_id, user_name

    Admins register phone numbers via the admin API or directly in Redis.

    Args:
        redis_client: Async Redis client.
        phone: Phone number (without whatsapp: prefix).

    Returns:
        Dict with tenant_id and user_id, or None if not whitelisted.
    """
    key = f"whatsapp:phone:{phone}"
    data = await redis_client.hgetall(key)  # type: ignore[union-attr]
    if not data:
        return None
    return {
        "tenant_id": str(data.get("tenant_id", "")),
        "user_id": str(data.get("user_id", phone)),
        "user_name": str(data.get("user_name", "")),
    }


async def send_whatsapp_text(
    to: str,
    body: str,
    twilio_sid: str = "",
    twilio_token: str = "",
    twilio_from: str = "",
) -> str | None:
    """Send a WhatsApp text message via Twilio.

    Args:
        to: Recipient phone (with whatsapp: prefix).
        body: Message text (max 4096 chars for WhatsApp).
        twilio_sid: Twilio Account SID.
        twilio_token: Twilio Auth Token.
        twilio_from: Twilio WhatsApp sender number.

    Returns:
        Message SID if sent, None on failure.
    """
    try:
        from twilio.rest import Client  # type: ignore[import-untyped]

        client = Client(twilio_sid, twilio_token)
        # Truncate to WhatsApp limit.
        truncated = body[:4096]

        message = client.messages.create(
            body=truncated,
            from_=f"{_WHATSAPP_PREFIX}{twilio_from}",
            to=f"{_WHATSAPP_PREFIX}{to}" if not to.startswith(_WHATSAPP_PREFIX) else to,
        )
        logger.info("WhatsApp text sent to %s: %s", to, message.sid)
        return message.sid
    except Exception as exc:
        logger.error("Failed to send WhatsApp text to %s: %s", to, exc)
        return None


async def send_whatsapp_document(
    to: str,
    document_url: str,
    filename: str,
    caption: str = "",
    twilio_sid: str = "",
    twilio_token: str = "",
    twilio_from: str = "",
) -> str | None:
    """Send a PDF document via WhatsApp using Twilio.

    The document is sent as a media message. Within the 24h customer
    service window, this is FREE (no extra charge for media).

    Args:
        to: Recipient phone number.
        document_url: Public URL to the PDF file.
        filename: Display filename for the document.
        caption: Optional caption text under the document.
        twilio_sid: Twilio Account SID.
        twilio_token: Twilio Auth Token.
        twilio_from: Twilio WhatsApp sender number.

    Returns:
        Message SID if sent, None on failure.
    """
    try:
        from twilio.rest import Client  # type: ignore[import-untyped]

        client = Client(twilio_sid, twilio_token)

        message = client.messages.create(
            media_url=[document_url],
            body=caption[:1024] if caption else filename,
            from_=f"{_WHATSAPP_PREFIX}{twilio_from}",
            to=f"{_WHATSAPP_PREFIX}{to}" if not to.startswith(_WHATSAPP_PREFIX) else to,
        )
        logger.info("WhatsApp document sent to %s: %s (%s)", to, message.sid, filename)
        return message.sid
    except Exception as exc:
        logger.error("Failed to send WhatsApp document to %s: %s", to, exc)
        return None


def format_whatsapp_response(
    answer: str,
    sources: list[dict[str, object]],
    confidence: str,
    score: float,
) -> str:
    """Format the RAG response for WhatsApp.

    WhatsApp supports basic formatting: *bold*, _italic_, ~strikethrough~.
    Max message length is 4096 characters.

    Args:
        answer: The RAG-generated answer.
        sources: Source chunk metadata.
        confidence: Confidence level.
        score: Best retrieval score.

    Returns:
        Formatted WhatsApp message text.
    """
    parts: list[str] = []

    if confidence == "low":
        parts.append(
            f"⚠️ *ATENCION: Confianza baja (score: {score:.2f})*\n"
            "Se recomienda verificar con el supervisor.\n"
        )

    parts.append(answer)

    if sources:
        parts.append("\n📄 *Fuentes:*")
        for i, src in enumerate(sources[:3], 1):
            parts.append(
                f"  {i}. {src.get('doc_id', '')} — "
                f"Sec: {src.get('section_path', '')} — "
                f"Pag: {src.get('page_ref', '')}"
            )

    parts.append(
        "\n---\n_Verificar siempre con el supervisor antes de "
        "ejecutar procedimientos criticos de seguridad._"
    )

    return "\n".join(parts)
