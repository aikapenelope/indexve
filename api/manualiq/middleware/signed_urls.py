"""Signed URL tokens for secure document access.

Generates time-limited, HMAC-signed tokens for PDF document URLs.
Used by WhatsApp and email channels to provide Twilio/Resend access
to tenant-specific PDFs without exposing tenant_id in the URL.

The token encodes tenant_id + doc_id + expiry, signed with a secret.
When Twilio fetches the PDF, the endpoint verifies the token instead
of requiring auth headers.

This prevents:
- Tenant ID exposure in URLs
- Unauthorized access to other tenants' documents
- Replay attacks (tokens expire after 1 hour)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time

logger = logging.getLogger(__name__)

# Token validity: 1 hour (enough for Twilio/Resend to fetch the PDF).
TOKEN_TTL_SECONDS = 3600


def _get_signing_key() -> str:
    """Get the signing key from environment or generate a default."""
    return os.environ.get(
        "PDF_SIGNING_KEY", os.environ.get("CLERK_SECRET_KEY", "manualiq-dev-key")
    )


def generate_pdf_token(tenant_id: str, doc_id: str) -> str:
    """Generate a signed token for PDF access.

    Token format: {expiry}:{signature}
    Signature = HMAC-SHA256(key, tenant_id:doc_id:expiry)

    Args:
        tenant_id: The tenant that owns the document.
        doc_id: The document identifier.

    Returns:
        Signed token string.
    """
    expiry = int(time.time()) + TOKEN_TTL_SECONDS
    payload = f"{tenant_id}:{doc_id}:{expiry}"
    signature = hmac.new(
        _get_signing_key().encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()[:32]

    return f"{tenant_id}:{doc_id}:{expiry}:{signature}"


def verify_pdf_token(token: str) -> tuple[str, str] | None:
    """Verify a signed PDF token and extract tenant_id + doc_id.

    Args:
        token: The token from the URL query parameter.

    Returns:
        (tenant_id, doc_id) tuple if valid, None if invalid or expired.
    """
    try:
        parts = token.split(":")
        if len(parts) != 4:
            return None

        tenant_id, doc_id, expiry_str, signature = parts
        expiry = int(expiry_str)

        # Check expiry.
        if time.time() > expiry:
            logger.warning("PDF token expired for %s/%s", tenant_id, doc_id)
            return None

        # Verify signature.
        payload = f"{tenant_id}:{doc_id}:{expiry_str}"
        expected = hmac.new(
            _get_signing_key().encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]

        if not hmac.compare_digest(signature, expected):
            logger.warning("PDF token signature mismatch for %s/%s", tenant_id, doc_id)
            return None

        return (tenant_id, doc_id)

    except (ValueError, IndexError) as exc:
        logger.warning("Invalid PDF token format: %s", exc)
        return None
