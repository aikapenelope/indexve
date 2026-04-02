"""Clerk JWT authentication middleware for FastAPI.

Verifies Clerk JWTs and extracts tenant_id + user_id server-side.
The tenant_id is NEVER accepted from the request body (issue 4.1).

In production, this verifies the JWT signature against Clerk's JWKS
endpoint. In development mode (CLERK_SECRET_KEY not set), it falls
back to reading from headers for local testing.

Reference: ARCHITECTURE.md Section 3.11, KNOWN_ISSUES.md issue 4.1.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from fastapi import HTTPException, Request

from manualiq.middleware.rate_limiter import TenantPlan

logger = logging.getLogger(__name__)

# Clerk JWKS endpoint for JWT verification.
_CLERK_JWKS_URL = "https://api.clerk.com/v1/jwks"


@dataclass
class AuthContext:
    """Authenticated request context extracted from Clerk JWT."""

    tenant_id: str
    user_id: str
    email: str = ""
    role: str = "tecnico"  # tecnico, supervisor, admin
    plan: TenantPlan = TenantPlan.FREE

    def require_role(self, *allowed_roles: str) -> None:
        """Raise 403 if the user's role is not in the allowed list.

        Usage in endpoints:
            auth.require_role("admin")
            auth.require_role("admin", "supervisor")
        """
        if self.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Requires role: {', '.join(allowed_roles)}. Current: {self.role}",
            )


def _is_dev_mode() -> bool:
    """Check if running in development mode (no Clerk secret key)."""
    return not os.environ.get("CLERK_SECRET_KEY", "")


async def _verify_clerk_jwt(token: str) -> dict[str, object]:
    """Verify a Clerk JWT and return the decoded claims.

    Uses Clerk's JWKS endpoint to get the public keys for verification.
    Requires PyJWT with cryptography extras.
    """
    import jwt  # type: ignore[import-untyped]
    from jwt import PyJWKClient  # type: ignore[import-untyped]

    clerk_secret = os.environ.get("CLERK_SECRET_KEY", "")
    if not clerk_secret:
        raise HTTPException(status_code=500, detail="CLERK_SECRET_KEY not configured")

    # Fetch JWKS from Clerk.
    jwks_client = PyJWKClient(_CLERK_JWKS_URL)
    signing_key = jwks_client.get_signing_key_from_jwt(token)

    # Decode and verify.
    decoded: dict[str, object] = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        options={"verify_aud": False},
    )
    return decoded


async def authenticate_request(request: Request) -> AuthContext:
    """Extract and verify authentication from the request.

    Production mode: Verifies Clerk JWT from Authorization header,
    extracts org_id (tenant_id) and sub (user_id) from claims.

    Development mode: Reads from x-tenant-id and x-user-id headers
    for local testing without Clerk.

    The tenant_id is ALWAYS server-side. Never from request body.
    """
    if _is_dev_mode():
        return _dev_mode_auth(request)

    # Production: verify Clerk JWT.
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected: Bearer <token>",
        )

    token = auth_header[7:]  # Strip "Bearer ".

    try:
        claims = await _verify_clerk_jwt(token)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    # Extract tenant and user from Clerk claims.
    user_id = str(claims.get("sub", ""))
    org_id = str(claims.get("org_id", ""))
    email = str(claims.get("email", ""))
    org_role = str(claims.get("org_role", "member"))

    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing user ID (sub)")

    if not org_id:
        raise HTTPException(
            status_code=403,
            detail="No organization selected. Please select an organization in Clerk.",
        )

    # Map Clerk org role to ManualIQ role.
    role = "admin" if org_role == "admin" else "tecnico"

    # Map plan from org metadata (would come from Clerk org public metadata).
    org_plan = str(claims.get("org_plan", "free")).upper()
    try:
        plan = TenantPlan[org_plan]
    except KeyError:
        plan = TenantPlan.FREE

    return AuthContext(
        tenant_id=org_id,
        user_id=user_id,
        email=email,
        role=role,
        plan=plan,
    )


def _dev_mode_auth(request: Request) -> AuthContext:
    """Development mode: read auth from headers."""
    tenant_id = request.headers.get("x-tenant-id", "")
    user_id = request.headers.get("x-user-id", "")

    if not tenant_id or not user_id:
        raise HTTPException(
            status_code=401,
            detail=(
                "Development mode: provide x-tenant-id and x-user-id headers. "
                "Set CLERK_SECRET_KEY for production JWT verification."
            ),
        )

    plan_str = request.headers.get("x-tenant-plan", "free").upper()
    try:
        plan = TenantPlan[plan_str]
    except KeyError:
        plan = TenantPlan.FREE

    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        plan=plan,
    )
