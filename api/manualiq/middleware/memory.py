"""Conversation memory backed by Redis.

Maintains chat history per session so follow-up questions like
"y el del otro lado?" have context from previous turns.

Each session is identified by (tenant_id, user_id, session_id).
Messages are stored as Redis lists with a configurable max length
to prevent unbounded growth.

Reference: AUDIT.md Section 11.3.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Max messages per session (user + assistant combined).
MAX_SESSION_MESSAGES = 50

# Session TTL: 24 hours of inactivity.
SESSION_TTL_SECONDS = 86_400


@dataclass
class ChatMessage:
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float


class ConversationMemory:
    """Redis-backed conversation memory per session.

    Stores the last MAX_SESSION_MESSAGES messages for context injection
    into the LLM prompt. Enables follow-up questions without repeating
    the full context.
    """

    def __init__(self, redis_client: object) -> None:
        self._redis = redis_client

    def _key(self, tenant_id: str, user_id: str, session_id: str) -> str:
        """Redis key for a conversation session."""
        return f"memory:{tenant_id}:{user_id}:{session_id}"

    async def add_message(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a message to the conversation history."""
        key = self._key(tenant_id, user_id, session_id)
        msg = json.dumps({"role": role, "content": content, "timestamp": time.time()})
        await self._redis.rpush(key, msg)  # type: ignore[union-attr]
        # Trim to max length.
        await self._redis.ltrim(key, -MAX_SESSION_MESSAGES, -1)  # type: ignore[union-attr]
        # Reset TTL on activity.
        await self._redis.expire(key, SESSION_TTL_SECONDS)  # type: ignore[union-attr]

    async def get_history(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        last_n: int = 10,
    ) -> list[ChatMessage]:
        """Get the last N messages from a session.

        Args:
            tenant_id: Tenant ID.
            user_id: User ID.
            session_id: Session identifier.
            last_n: Number of recent messages to return.

        Returns:
            List of ChatMessage objects, oldest first.
        """
        key = self._key(tenant_id, user_id, session_id)
        raw_messages = await self._redis.lrange(key, -last_n, -1)  # type: ignore[union-attr]

        messages: list[ChatMessage] = []
        for raw in raw_messages:
            try:
                data = json.loads(raw)
                messages.append(
                    ChatMessage(
                        role=data["role"],
                        content=data["content"],
                        timestamp=data.get("timestamp", 0.0),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue

        return messages

    def format_for_prompt(self, messages: list[ChatMessage]) -> str:
        """Format conversation history for injection into the LLM prompt.

        Returns a string that can be prepended to the context prompt
        to give the LLM awareness of previous turns.
        """
        if not messages:
            return ""

        lines = ["Historial de conversacion reciente:"]
        for msg in messages:
            prefix = "Tecnico" if msg.role == "user" else "ManualIQ"
            # Truncate long messages to save tokens.
            content = msg.content[:500]
            lines.append(f"- {prefix}: {content}")
        lines.append("")  # Blank line before current query.

        return "\n".join(lines)

    async def clear_session(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """Clear all messages in a session."""
        key = self._key(tenant_id, user_id, session_id)
        await self._redis.delete(key)  # type: ignore[union-attr]
