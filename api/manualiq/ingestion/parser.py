"""Document parser with Docling (primary) and LlamaParse (fallback).

Addresses issues 2.1 (Docling hanging) and 2.2 (Docling memory crash)
from KNOWN_ISSUES.md.

Key design decisions:
- Docling runs in a subprocess with a hard 120s timeout (the built-in
  document_timeout is unreliable per GitHub issues #2109, #2381).
- If Docling fails or times out, we fall back to LlamaParse LiteParse.
- Docling models (~1.1GB) must be pre-downloaded in the Docker build.
- This module is meant to run inside a Prefect worker, NEVER in the
  FastAPI process.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Hard timeout for Docling subprocess (seconds).
DOCLING_TIMEOUT_SECONDS = 120


class ParserBackend(Enum):
    """Which parser produced the output."""

    DOCLING = "docling"
    LLAMAPARSE = "llamaparse"
    FAILED = "failed"


@dataclass
class ParseResult:
    """Result of parsing a single document."""

    text: str
    backend: ParserBackend
    pages: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Docling subprocess wrapper
# ---------------------------------------------------------------------------

# This script is executed in a child process so that a hanging Docling
# call can be killed cleanly without affecting the parent.
_DOCLING_SUBPROCESS_SCRIPT = """
import json
import sys
from pathlib import Path

def run_docling(pdf_path: str) -> dict:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    text = doc.export_to_markdown()
    pages = len(doc.pages) if hasattr(doc, "pages") else 0
    return {"text": text, "pages": pages}

if __name__ == "__main__":
    pdf_path = sys.argv[1]
    try:
        output = run_docling(pdf_path)
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
"""


def _parse_with_docling(pdf_path: Path) -> ParseResult:
    """Parse a PDF using Docling in a subprocess with hard timeout.

    The subprocess approach ensures that if Docling hangs (a known issue
    with complex PDFs), we can kill it cleanly after DOCLING_TIMEOUT_SECONDS.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(_DOCLING_SUBPROCESS_SCRIPT)
        script_path = script_file.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=DOCLING_TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Docling exited with non-zero code"
            logger.warning("Docling failed for %s: %s", pdf_path.name, error_msg)
            return ParseResult(
                text="",
                backend=ParserBackend.FAILED,
                pages=0,
                error=error_msg,
            )

        output = json.loads(result.stdout.strip())
        if "error" in output:
            logger.warning("Docling error for %s: %s", pdf_path.name, output["error"])
            return ParseResult(
                text="",
                backend=ParserBackend.FAILED,
                pages=0,
                error=output["error"],
            )

        return ParseResult(
            text=output["text"],
            backend=ParserBackend.DOCLING,
            pages=output.get("pages", 0),
        )

    except subprocess.TimeoutExpired:
        logger.warning(
            "Docling timed out after %ds for %s",
            DOCLING_TIMEOUT_SECONDS,
            pdf_path.name,
        )
        return ParseResult(
            text="",
            backend=ParserBackend.FAILED,
            pages=0,
            error=f"Timeout after {DOCLING_TIMEOUT_SECONDS}s",
        )
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse Docling output for %s: %s", pdf_path.name, exc)
        return ParseResult(
            text="",
            backend=ParserBackend.FAILED,
            pages=0,
            error=str(exc),
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LlamaParse fallback
# ---------------------------------------------------------------------------


def _parse_with_llamaparse(pdf_path: Path, api_key: str) -> ParseResult:
    """Parse a PDF using LlamaParse LiteParse as fallback.

    LlamaParse is faster (~6s/doc) and handles scanned documents better,
    but has lower table precision (~85% vs Docling's 97.9%).
    """
    try:
        from llama_parse import LlamaParse  # type: ignore[import-untyped]

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            parsing_instruction=(
                "This is a technical manufacturing manual. "
                "Preserve all tables, part numbers, torque specifications, "
                "and procedural steps exactly as they appear."
            ),
        )
        documents = parser.load_data(str(pdf_path))
        text = "\n\n".join(doc.text for doc in documents)
        return ParseResult(
            text=text,
            backend=ParserBackend.LLAMAPARSE,
            pages=len(documents),
        )
    except Exception as exc:
        logger.error("LlamaParse also failed for %s: %s", pdf_path.name, exc)
        return ParseResult(
            text="",
            backend=ParserBackend.FAILED,
            pages=0,
            error=f"Both parsers failed. LlamaParse error: {exc}",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_document(
    pdf_path: Path,
    *,
    llamaparse_api_key: str = "",
) -> ParseResult:
    """Parse a PDF document using Docling with LlamaParse fallback.

    Tries Docling first (better table precision). If Docling fails or
    times out, falls back to LlamaParse LiteParse.

    Args:
        pdf_path: Path to the PDF file.
        llamaparse_api_key: API key for LlamaParse (required for fallback).

    Returns:
        ParseResult with the extracted text and metadata about which
        backend was used.
    """
    if not pdf_path.exists():
        return ParseResult(
            text="",
            backend=ParserBackend.FAILED,
            pages=0,
            error=f"File not found: {pdf_path}",
        )

    logger.info(
        "Parsing %s with Docling (timeout=%ds)...",
        pdf_path.name,
        DOCLING_TIMEOUT_SECONDS,
    )
    result = _parse_with_docling(pdf_path)

    if result.backend == ParserBackend.DOCLING and result.text.strip():
        logger.info("Docling succeeded for %s (%d pages)", pdf_path.name, result.pages)
        return result

    # Docling failed or returned empty text -- try LlamaParse.
    logger.info(
        "Falling back to LlamaParse for %s (Docling error: %s)",
        pdf_path.name,
        result.error,
    )

    if not llamaparse_api_key:
        logger.error("No LlamaParse API key provided, cannot fall back")
        return ParseResult(
            text="",
            backend=ParserBackend.FAILED,
            pages=0,
            error="Docling failed and no LlamaParse API key available",
        )

    return _parse_with_llamaparse(pdf_path, llamaparse_api_key)
