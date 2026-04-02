"""NeMo Guardrails integration for ManualIQ.

Loads the Colang config from guardrails/config/ and provides
async functions to run input and output rails from FastAPI.

This module bridges the NeMo Guardrails engine with our pipeline
so that the Colang rules (input.co, output.co, dialog.co) and
custom actions (custom_actions.py) are actually executed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the guardrails config directory (relative to api/).
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "guardrails" / "config"


class GuardrailsEngine:
    """Wrapper around NeMo Guardrails LLMRails.

    Initialized once at startup, then called for each request
    to run input and output rails.
    """

    def __init__(self) -> None:
        self._rails: object | None = None
        self._enabled = False

    async def initialize(self) -> None:
        """Load NeMo Guardrails config and initialize the engine.

        Skips initialization if the config directory doesn't exist
        or if nemoguardrails is not installed (graceful degradation).
        """
        if not _CONFIG_DIR.is_dir():
            logger.warning(
                "Guardrails config dir not found: %s. Rails disabled.",
                _CONFIG_DIR,
            )
            return

        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-untyped]

            config = RailsConfig.from_path(str(_CONFIG_DIR))
            self._rails = LLMRails(config)
            self._enabled = True
            logger.info("NeMo Guardrails initialized from %s", _CONFIG_DIR)
        except ImportError:
            logger.warning("nemoguardrails not installed. Rails disabled.")
        except Exception as exc:
            logger.error("Failed to initialize NeMo Guardrails: %s", exc)

    @property
    def enabled(self) -> bool:
        """Whether guardrails are active."""
        return self._enabled

    async def check_input(self, user_message: str) -> InputCheckResult:
        """Run input rails on a user message.

        Returns whether the message is allowed and an optional
        rejection response if blocked.
        """
        if not self._enabled or self._rails is None:
            return InputCheckResult(allowed=True)

        try:
            from nemoguardrails import LLMRails  # type: ignore[import-untyped]

            rails: LLMRails = self._rails  # type: ignore[assignment]
            response = await rails.generate_async(
                messages=[{"role": "user", "content": user_message}]
            )

            bot_message: str = (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )  # type: ignore[union-attr]

            # If the rails blocked the input, the response will be a
            # refusal message (defined in our Colang rules).
            blocked_phrases = [
                "no puedo procesar",
                "informacion personal",
                "reformule su pregunta",
            ]
            is_blocked = any(
                phrase in bot_message.lower() for phrase in blocked_phrases
            )

            if is_blocked:
                logger.info("Input rail blocked message: %s", user_message[:50])
                return InputCheckResult(
                    allowed=False,
                    rejection_message=bot_message,
                )

            return InputCheckResult(allowed=True)

        except Exception as exc:
            # Guardrail failure should not block the pipeline.
            logger.warning("Input rail error (allowing through): %s", exc)
            return InputCheckResult(allowed=True)

    async def check_output(self, bot_response: str) -> OutputCheckResult:
        """Run output rails on a bot response.

        Checks for PII, inappropriate content, and language issues.
        Returns the (possibly sanitized) response.
        """
        if not self._enabled or self._rails is None:
            return OutputCheckResult(text=bot_response, modified=False)

        try:
            from nemoguardrails import LLMRails  # type: ignore[import-untyped]

            rails: LLMRails = self._rails  # type: ignore[assignment]

            # Run output rails by providing the bot message for checking.
            response = await rails.generate_async(
                messages=[
                    {"role": "user", "content": "check output"},
                    {"role": "assistant", "content": bot_response},
                ]
            )

            checked_text: str = (
                response.get("content", bot_response)
                if isinstance(response, dict)
                else str(response)
            )  # type: ignore[union-attr]
            modified = checked_text != bot_response

            if modified:
                logger.info("Output rail modified response")

            return OutputCheckResult(text=checked_text, modified=modified)

        except Exception as exc:
            # Guardrail failure should not block the response.
            logger.warning("Output rail error (passing through): %s", exc)
            return OutputCheckResult(text=bot_response, modified=False)


class InputCheckResult:
    """Result of running input rails."""

    def __init__(
        self,
        allowed: bool = True,
        rejection_message: str | None = None,
    ) -> None:
        self.allowed = allowed
        self.rejection_message = rejection_message


class OutputCheckResult:
    """Result of running output rails."""

    def __init__(
        self,
        text: str,
        modified: bool = False,
    ) -> None:
        self.text = text
        self.modified = modified
