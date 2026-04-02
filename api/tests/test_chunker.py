"""Tests for the semantic chunker module."""

from manualiq.ingestion.chunker import (
    BlockType,
    _detect_safety_level,
    _estimate_tokens,
    _extract_part_numbers,
    _is_procedure_block,
    blocks_to_chunks,
    classify_and_split_blocks,
)


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 1

    def test_short_text(self) -> None:
        # 20 chars / 4 = 5 tokens
        assert _estimate_tokens("This is a test text.") == 5

    def test_long_text(self) -> None:
        text = "a" * 400  # 400 chars / 4 = 100 tokens
        assert _estimate_tokens(text) == 100


class TestDetectSafetyLevel:
    def test_danger(self) -> None:
        assert _detect_safety_level("DANGER: High voltage") == "critico"

    def test_peligro(self) -> None:
        assert _detect_safety_level("PELIGRO: Alta tension") == "critico"

    def test_warning(self) -> None:
        assert _detect_safety_level("WARNING: Hot surface") == "precaucion"

    def test_caution(self) -> None:
        assert _detect_safety_level("CAUTION: Wear gloves") == "precaucion"

    def test_informativo(self) -> None:
        assert _detect_safety_level("Torque to 50 Nm") == "informativo"


class TestExtractPartNumbers:
    def test_finds_part_numbers(self) -> None:
        text = "Replace part AB-4521-CX and check CD-1234-EF"
        parts = _extract_part_numbers(text)
        assert len(parts) >= 1

    def test_no_part_numbers(self) -> None:
        text = "This is a regular sentence without part numbers."
        parts = _extract_part_numbers(text)
        assert len(parts) == 0


class TestIsProcedureBlock:
    def test_numbered_steps(self) -> None:
        text = "1. Remove the cover\n2. Disconnect the cable\n3. Replace the gasket"
        assert _is_procedure_block(text) is True

    def test_not_procedure(self) -> None:
        text = "The engine operates at 2400 RPM under normal conditions."
        assert _is_procedure_block(text) is False


class TestClassifyAndSplitBlocks:
    def test_table_detection(self) -> None:
        text = "| Part | Torque |\n|------|--------|\n| Bolt A | 50 Nm |\n| Bolt B | 75 Nm |"
        blocks = classify_and_split_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.TABLE

    def test_procedure_detection(self) -> None:
        text = "1. Remove the cover\n2. Disconnect the cable\n3. Replace the gasket"
        blocks = classify_and_split_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.PROCEDURE

    def test_text_block(self) -> None:
        text = "The engine operates at 2400 RPM under normal conditions."
        blocks = classify_and_split_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].block_type == BlockType.TEXT

    def test_empty_input(self) -> None:
        blocks = classify_and_split_blocks("")
        assert len(blocks) == 0

    def test_section_path_preserved(self) -> None:
        text = "Some content here."
        blocks = classify_and_split_blocks(text, section_path="Manual > Cap 4")
        assert blocks[0].section_path == "Manual > Cap 4"


class TestBlocksToChunks:
    def test_basic_conversion(self) -> None:
        blocks = classify_and_split_blocks("Simple text content for testing.")
        chunks = blocks_to_chunks(
            blocks,
            doc_id="test_doc",
            tenant_id="tenant_001",
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata.doc_id == "test_doc"
        assert chunks[0].metadata.tenant_id == "tenant_001"
        assert chunks[0].metadata.hash_sha256 != ""

    def test_section_path_prefix(self) -> None:
        blocks = classify_and_split_blocks(
            "Content here.", section_path="Manual > Cap 1"
        )
        chunks = blocks_to_chunks(blocks, doc_id="doc", tenant_id="t1")
        assert chunks[0].text.startswith("[Manual > Cap 1]")

    def test_safety_level_detection(self) -> None:
        blocks = classify_and_split_blocks("DANGER: Do not touch the rotor.")
        chunks = blocks_to_chunks(blocks, doc_id="doc", tenant_id="t1")
        assert chunks[0].metadata.safety_level == "critico"
