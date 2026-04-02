"""LLM Gateway: Claude (primary) + Gemini Flash (routing/fallback).

Simple wrapper with retry and failover for 2 providers, as specified
in ARCHITECTURE.md Section 3.8. No LiteLLM (supply chain risk).

Usage:
    gateway = LLMGateway(anthropic_api_key="...", google_api_key="...")
    response = await gateway.generate("prompt", model="claude")
    routing = await gateway.route("classify this query", model="gemini")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import anthropic
import google.genai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class LLMModel(Enum):
    """Available LLM models."""

    CLAUDE = "claude"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Response from the LLM gateway."""

    text: str
    model: LLMModel
    input_tokens: int
    output_tokens: int
    was_fallback: bool = False


# Retry configuration: 3 attempts with exponential backoff.
_RETRY_KWARGS = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=10),
    "retry": retry_if_exception_type((Exception,)),
    "reraise": True,
}


class LLMGateway:
    """Wrapper for Claude + Gemini with retry and failover.

    Claude Sonnet 4.6 is the primary model for generation (better
    factual correctness). Gemini 3 Flash is used for routing tasks
    (intent classification, query expansion) and as fallback.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,
        claude_model: str = "claude-sonnet-4-20250514",
        gemini_model: str = "gemini-2.0-flash",
        system_prompt: str = "",
    ) -> None:
        self._anthropic = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._google = genai.Client(api_key=google_api_key)
        self._claude_model = claude_model
        self._gemini_model = gemini_model
        self._system_prompt = system_prompt

    @retry(**_RETRY_KWARGS)
    async def _call_claude(
        self,
        prompt: str,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call Claude Sonnet 4.6 API."""
        kwargs: dict[str, object] = {
            "model": self._claude_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt

        response = await self._anthropic.messages.create(**kwargs)  # type: ignore[arg-type]
        text = response.content[0].text  # type: ignore[union-attr]
        return LLMResponse(
            text=text,
            model=LLMModel.CLAUDE,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    @retry(**_RETRY_KWARGS)
    async def _call_gemini(
        self,
        prompt: str,
    ) -> LLMResponse:
        """Call Gemini 3 Flash API."""
        full_prompt = prompt
        if self._system_prompt:
            full_prompt = f"{self._system_prompt}\n\n{prompt}"

        response = await self._google.aio.models.generate_content(
            model=self._gemini_model,
            contents=full_prompt,
        )
        text = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return LLMResponse(
            text=text,
            model=LLMModel.GEMINI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def generate(
        self,
        prompt: str,
        *,
        model: str = "claude",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response with automatic failover.

        Tries the requested model first. If it fails after retries,
        falls back to the other provider.

        Args:
            prompt: The prompt to send.
            model: "claude" or "gemini". Default "claude".
            max_tokens: Max tokens for Claude (ignored for Gemini).

        Returns:
            LLMResponse with the generated text and usage metadata.
        """
        try:
            if model == "claude":
                return await self._call_claude(prompt, max_tokens=max_tokens)
            return await self._call_gemini(prompt)
        except Exception as exc:
            logger.warning(
                "Primary LLM (%s) failed after retries: %s. Falling back.",
                model,
                str(exc)[:100],
            )
            try:
                if model == "claude":
                    response = await self._call_gemini(prompt)
                else:
                    response = await self._call_claude(prompt, max_tokens=max_tokens)
                response.was_fallback = True
                return response
            except Exception as fallback_exc:
                logger.error(
                    "Both LLMs failed. Primary: %s, Fallback: %s",
                    str(exc)[:100],
                    str(fallback_exc)[:100],
                )
                raise

    async def route(
        self,
        prompt: str,
    ) -> LLMResponse:
        """Call Gemini Flash for routing tasks (cheap, fast).

        Used for intent classification, query expansion, and other
        lightweight LLM tasks that don't need Claude's quality.
        """
        return await self.generate(prompt, model="gemini")
