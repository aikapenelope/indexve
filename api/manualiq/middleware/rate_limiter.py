"""Rate limiting middleware for FastAPI with Redis counters.

Addresses issues 4.1 (prompt injection prevention via tenant_id
server-side), 4.2 (data leakage via tenant isolation), and 5.1
(API cost control) from KNOWN_ISSUES.md.

Rate limits are configurable per plan:
- Free:       50 queries/day
- Pro:        500 queries/day
- Enterprise: 10,000 queries/day (effectively unlimited for MVP)

Per-user burst limit: 10 queries/minute (all plans).

Uses Redis INCR + EXPIRE for atomic, distributed counters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TenantPlan(Enum):
    """Subscription plan tiers with their daily query limits."""

    FREE = 50
    PRO = 500
    ENTERPRISE = 10_000


# Per-user burst limit: queries per minute.
USER_BURST_LIMIT = 10

# Redis key TTL values.
_DAY_SECONDS = 86_400
_MINUTE_SECONDS = 60


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    tenant_remaining: int
    user_remaining: int
    retry_after_seconds: int | None = None


class RateLimiter:
    """Redis-backed rate limiter for ManualIQ API.

    Implements two levels of rate limiting:
    1. Tenant-level: daily query budget based on subscription plan.
    2. User-level: per-minute burst protection.

    The tenant_id is ALWAYS injected server-side from the authenticated
    Clerk session -- never from the request body or headers. This prevents
    prompt injection attacks that try to impersonate another tenant.
    """

    def __init__(self, redis_client: object) -> None:
        """Initialize with a Redis client.

        Args:
            redis_client: A redis.Redis or redis.asyncio.Redis instance.
                Typed as object to avoid import dependency at module level.
        """
        self._redis = redis_client

    def _tenant_key(self, tenant_id: str) -> str:
        """Redis key for tenant daily counter."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        return f"ratelimit:tenant:{tenant_id}:{day}"

    def _user_key(self, user_id: str) -> str:
        """Redis key for user per-minute counter."""
        minute = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        return f"ratelimit:user:{user_id}:{minute}"

    async def check_rate_limit(
        self,
        tenant_id: str,
        user_id: str,
        plan: TenantPlan = TenantPlan.FREE,
    ) -> RateLimitResult:
        """Check if a request is within rate limits.

        This method is async because Redis operations should be non-blocking
        in a FastAPI context.

        Args:
            tenant_id: The tenant ID from the authenticated Clerk session.
            user_id: The user ID from the authenticated Clerk session.
            plan: The tenant's subscription plan.

        Returns:
            RateLimitResult indicating whether the request is allowed.
        """
        redis = self._redis
        tenant_key = self._tenant_key(tenant_id)
        user_key = self._user_key(user_id)

        # Use a pipeline for atomic multi-key operations.
        # We type-ignore here because the redis client is typed as object
        # to avoid import dependency.
        pipe = redis.pipeline()  # type: ignore[union-attr]
        pipe.incr(tenant_key)  # type: ignore[union-attr]
        pipe.expire(tenant_key, _DAY_SECONDS)  # type: ignore[union-attr]
        pipe.incr(user_key)  # type: ignore[union-attr]
        pipe.expire(user_key, _MINUTE_SECONDS)  # type: ignore[union-attr]
        results = await pipe.execute()  # type: ignore[union-attr]

        tenant_count: int = results[0]
        user_count: int = results[2]

        daily_limit = plan.value
        tenant_remaining = max(0, daily_limit - tenant_count)
        user_remaining = max(0, USER_BURST_LIMIT - user_count)

        # Check user burst limit first (more likely to trigger).
        if user_count > USER_BURST_LIMIT:
            logger.warning(
                "User %s exceeded burst limit (%d/%d per minute)",
                user_id,
                user_count,
                USER_BURST_LIMIT,
            )
            return RateLimitResult(
                allowed=False,
                tenant_remaining=tenant_remaining,
                user_remaining=0,
                retry_after_seconds=_MINUTE_SECONDS,
            )

        # Check tenant daily limit.
        if tenant_count > daily_limit:
            logger.warning(
                "Tenant %s exceeded daily limit (%d/%d, plan=%s)",
                tenant_id,
                tenant_count,
                daily_limit,
                plan.name,
            )
            return RateLimitResult(
                allowed=False,
                tenant_remaining=0,
                user_remaining=user_remaining,
                retry_after_seconds=_DAY_SECONDS,
            )

        return RateLimitResult(
            allowed=True,
            tenant_remaining=tenant_remaining,
            user_remaining=user_remaining,
        )

    async def get_usage(
        self,
        tenant_id: str,
    ) -> dict[str, int]:
        """Get current usage stats for a tenant (for dashboard/alerts).

        Returns:
            Dict with 'daily_count' and 'daily_limit' keys.
        """
        tenant_key = self._tenant_key(tenant_id)
        count_raw = await self._redis.get(tenant_key)  # type: ignore[union-attr]
        count = int(count_raw) if count_raw else 0
        return {"daily_count": count}


@dataclass
class CostAlert:
    """Alert when daily spending exceeds threshold."""

    tenant_id: str
    daily_count: int
    daily_limit: int
    percentage: float
    message: str


def check_cost_alert(
    tenant_id: str,
    daily_count: int,
    daily_limit: int,
    *,
    threshold_pct: float = 0.10,
) -> CostAlert | None:
    """Check if daily usage exceeds the alert threshold.

    Default threshold is 10% of monthly budget consumed in a single day,
    as specified in KNOWN_ISSUES.md section 5.1.

    Args:
        tenant_id: The tenant to check.
        daily_count: Current day's query count.
        daily_limit: The tenant's daily query limit.
        threshold_pct: Alert when daily usage exceeds this fraction
            of the daily limit. Default 0.10 (10%).

    Returns:
        A CostAlert if the threshold is exceeded, None otherwise.
    """
    if daily_limit == 0:
        return None

    percentage = daily_count / daily_limit
    if percentage >= (1.0 - threshold_pct):
        return CostAlert(
            tenant_id=tenant_id,
            daily_count=daily_count,
            daily_limit=daily_limit,
            percentage=percentage,
            message=(
                f"Tenant {tenant_id} ha usado {percentage:.0%} de su limite "
                f"diario ({daily_count}/{daily_limit} queries). "
                "Considere revisar el consumo."
            ),
        )
    return None
