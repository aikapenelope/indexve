"""Custom semantic chunker for manufacturing manuals.

Implements the chunking rules from ARCHITECTURE.md Section 2:
- NEVER split a procedure step in half.
- NEVER split a table.
- ALWAYS keep part numbers with their descriptions.
- Prefix each chunk with its section path: [Manual > Cap > Section].
- Target size: 400-600 tokens. Max: 800 tokens (tables may exceed).
- Overlap: 80 tokens ONLY for running text. Zero overlap for tables/procedures.

Addresses issue 1.3 (context fragmentation) from KNOWN_ISSUES.md.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence


class BlockType(Enum):
    """Classification of a parsed document block."""

    TEXT = "text"
    TABLE = "table"
    PROCEDURE = "procedure"
    HEADING = "heading"


@dataclass
class DocumentBlock:
    """A structural block extracted from a parsed document.

    Blocks are the intermediate representation between raw parsed output
    (from Docling/LlamaParse) and final chunks ready for embedding.
    """

    content: str
    block_type: BlockType
    page_ref: str = ""
    section_path: str = ""
    part_numbers: list[str] = field(default_factory=list)
    related_refs: list[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """Metadata attached to every chunk stored in Qdrant.

    Matches the schema defined in ARCHITECTURE.md Section 2.
    """

    doc_id: str
    tenant_id: str
    equipment: str
    manufacturer: str
    doc_language: str
    section_path: str
    procedure_type: str
    safety_level: str
    part_numbers: list[str]
    related_chunks: list[str]
    page_ref: str
    chunk_index: int
    total_chunks: int
    hash_sha256: str
    indexed_at: str


@dataclass
class Chunk:
    """A chunk ready for embedding and storage in Qdrant."""

    text: str
    metadata: ChunkMetadata


# ---------------------------------------------------------------------------
# Regex patterns for detecting structural elements
# ---------------------------------------------------------------------------

# Matches numbered procedure steps like "1.", "1.1", "Step 1:", "Paso 1:"
_STEP_PATTERN = re.compile(
    r"^(?:step|paso|etapa)?\s*\d+(?:\.\d+)*[.:)\-]\s",
    re.IGNORECASE | re.MULTILINE,
)

# Matches safety keywords that must stay with their context
_SAFETY_PATTERN = re.compile(
    r"\b(DANGER|WARNING|CAUTION|PELIGRO|ADVERTENCIA|PRECAUCION)\b",
    re.IGNORECASE,
)

# Matches part number patterns like "AB-4521-CX", "123-456-789"
_PART_NUMBER_PATTERN = re.compile(r"\b[A-Z]{1,4}[-\s]?\d{2,6}[-\s]?[A-Z0-9]{1,4}\b")

# Approximate tokens-per-character ratio for English/Spanish technical text.
_CHARS_PER_TOKEN = 4.0

# Chunk size targets (in tokens).
TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 600
HARD_MAX_TOKENS = 800
OVERLAP_TOKENS = 80


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length.

    Uses a conservative 4 chars/token ratio suitable for technical
    English/Spanish text. This avoids a tokenizer dependency at chunking
    time; the embedding model's actual tokenizer is used later.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _compute_hash(text: str) -> str:
    """SHA-256 hash of chunk text for deduplication and staleness detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_part_numbers(text: str) -> list[str]:
    """Extract part number patterns from text."""
    return list(set(_PART_NUMBER_PATTERN.findall(text)))


def _detect_safety_level(text: str) -> str:
    """Detect the highest safety level mentioned in the text."""
    upper = text.upper()
    if "DANGER" in upper or "PELIGRO" in upper:
        return "critico"
    if "WARNING" in upper or "ADVERTENCIA" in upper:
        return "precaucion"
    if "CAUTION" in upper or "PRECAUCION" in upper:
        return "precaucion"
    return "informativo"


def _is_procedure_block(text: str) -> bool:
    """Determine if a text block contains procedural steps."""
    matches = _STEP_PATTERN.findall(text)
    return len(matches) >= 2


def _split_running_text_with_overlap(
    text: str,
    section_path: str,
    page_ref: str,
) -> list[DocumentBlock]:
    """Split long running text into chunks with 80-token overlap.

    Splits at sentence boundaries to preserve readability. Only applies
    to running text -- tables and procedures are never split.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    blocks: list[DocumentBlock] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _estimate_tokens(sentence)

        if current_tokens + sentence_tokens > TARGET_MAX_TOKENS and current_sentences:
            block_text = " ".join(current_sentences)
            blocks.append(
                DocumentBlock(
                    content=block_text,
                    block_type=BlockType.TEXT,
                    page_ref=page_ref,
                    section_path=section_path,
                    part_numbers=_extract_part_numbers(block_text),
                )
            )

            # Calculate overlap: take trailing sentences up to OVERLAP_TOKENS.
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tokens = _estimate_tokens(s)
                if overlap_tokens + s_tokens > OVERLAP_TOKENS:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Emit remaining text.
    if current_sentences:
        block_text = " ".join(current_sentences)
        blocks.append(
            DocumentBlock(
                content=block_text,
                block_type=BlockType.TEXT,
                page_ref=page_ref,
                section_path=section_path,
                part_numbers=_extract_part_numbers(block_text),
            )
        )

    return blocks


def classify_and_split_blocks(
    raw_text: str,
    section_path: str = "",
    page_ref: str = "",
) -> list[DocumentBlock]:
    """Classify raw parsed text into structural blocks and split as needed.

    This is the main entry point for converting parser output into blocks
    that respect the semantic chunking rules.

    Args:
        raw_text: The full text output from Docling or LlamaParse for a
            section of a document.
        section_path: The hierarchical path like "Manual > Cap 4 > Section 3".
        page_ref: Page reference string like "pp. 4-18 to 4-20".

    Returns:
        A list of DocumentBlocks ready to be converted into Chunks.
    """
    blocks: list[DocumentBlock] = []

    # Split on double newlines to get paragraphs/sections.
    paragraphs = re.split(r"\n{2,}", raw_text.strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Detect tables: lines with pipe characters or tab-separated values.
        lines = para.split("\n")
        is_table = (
            sum(1 for line in lines if "|" in line or "\t" in line) > len(lines) * 0.5
        )

        if is_table:
            # Tables are NEVER split -- they are atomic chunks.
            blocks.append(
                DocumentBlock(
                    content=para,
                    block_type=BlockType.TABLE,
                    page_ref=page_ref,
                    section_path=section_path,
                    part_numbers=_extract_part_numbers(para),
                )
            )
        elif _is_procedure_block(para):
            # Procedures are NEVER split -- each procedure is an atomic chunk.
            # If a procedure exceeds HARD_MAX, we still keep it whole (safety).
            blocks.append(
                DocumentBlock(
                    content=para,
                    block_type=BlockType.PROCEDURE,
                    page_ref=page_ref,
                    section_path=section_path,
                    part_numbers=_extract_part_numbers(para),
                )
            )
        elif _estimate_tokens(para) > TARGET_MAX_TOKENS:
            # Long running text: split with overlap.
            blocks.extend(
                _split_running_text_with_overlap(para, section_path, page_ref)
            )
        else:
            # Short text block: keep as-is.
            blocks.append(
                DocumentBlock(
                    content=para,
                    block_type=BlockType.TEXT,
                    page_ref=page_ref,
                    section_path=section_path,
                    part_numbers=_extract_part_numbers(para),
                )
            )

    return blocks


def _merge_small_blocks(
    blocks: list[DocumentBlock],
) -> list[DocumentBlock]:
    """Merge consecutive small text blocks that are below TARGET_MIN_TOKENS.

    Only merges TEXT blocks. Tables and procedures are never merged.
    """
    if not blocks:
        return blocks

    merged: list[DocumentBlock] = []
    buffer: DocumentBlock | None = None

    for block in blocks:
        if block.block_type != BlockType.TEXT:
            # Flush buffer before non-text block.
            if buffer is not None:
                merged.append(buffer)
                buffer = None
            merged.append(block)
            continue

        if buffer is None:
            buffer = block
            continue

        combined_tokens = _estimate_tokens(buffer.content + " " + block.content)
        if combined_tokens <= TARGET_MAX_TOKENS:
            # Merge into buffer.
            buffer = DocumentBlock(
                content=buffer.content + "\n\n" + block.content,
                block_type=BlockType.TEXT,
                page_ref=buffer.page_ref or block.page_ref,
                section_path=buffer.section_path or block.section_path,
                part_numbers=list(set(buffer.part_numbers + block.part_numbers)),
                related_refs=list(set(buffer.related_refs + block.related_refs)),
            )
        else:
            merged.append(buffer)
            buffer = block

    if buffer is not None:
        merged.append(buffer)

    return merged


def blocks_to_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    doc_id: str,
    tenant_id: str,
    equipment: str = "",
    manufacturer: str = "",
    doc_language: str = "en",
    procedure_type: str = "informativo",
) -> list[Chunk]:
    """Convert DocumentBlocks into Chunks with full metadata.

    Applies the section path prefix rule and computes all metadata fields
    required by the Qdrant schema.

    Args:
        blocks: The blocks produced by classify_and_split_blocks.
        doc_id: Unique document identifier.
        tenant_id: The tenant that owns this document.
        equipment: Equipment name/model.
        manufacturer: Equipment manufacturer.
        doc_language: ISO language code of the source document.
        procedure_type: One of mantenimiento_preventivo, reparacion,
            diagnostico, operacion, informativo.

    Returns:
        A list of Chunks ready for embedding and Qdrant upsert.
    """
    # Merge small consecutive text blocks first.
    merged_blocks = _merge_small_blocks(list(blocks))
    total = len(merged_blocks)
    now = datetime.now(timezone.utc).isoformat()
    chunks: list[Chunk] = []

    for idx, block in enumerate(merged_blocks):
        # Prefix with section path as specified in ARCHITECTURE.md.
        if block.section_path:
            prefixed_text = f"[{block.section_path}] {block.content}"
        else:
            prefixed_text = block.content

        safety = _detect_safety_level(block.content)
        part_nums = block.part_numbers or _extract_part_numbers(block.content)

        chunk = Chunk(
            text=prefixed_text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                tenant_id=tenant_id,
                equipment=equipment,
                manufacturer=manufacturer,
                doc_language=doc_language,
                section_path=block.section_path,
                procedure_type=procedure_type,
                safety_level=safety,
                part_numbers=part_nums,
                related_chunks=block.related_refs,
                page_ref=block.page_ref,
                chunk_index=idx,
                total_chunks=total,
                hash_sha256=_compute_hash(prefixed_text),
                indexed_at=now,
            ),
        )
        chunks.append(chunk)

    return chunks
