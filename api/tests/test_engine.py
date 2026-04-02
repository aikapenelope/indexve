"""Tests for the query engine module."""

from manualiq.query.engine import (
    NO_INFO_RESPONSE,
    QueryConfidence,
    RetrievedChunk,
    build_context_prompt,
    deduplicate_chunks,
    evaluate_confidence,
    process_query,
    validate_tenant_chunks,
)


def _make_chunk(
    score: float = 0.9,
    tenant_id: str = "t1",
    doc_id: str = "doc1",
    hash_sha256: str = "abc123",
    safety_level: str = "informativo",
    text: str = "Test chunk content.",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        doc_id=doc_id,
        tenant_id=tenant_id,
        section_path="Cap 1 > Sec 2",
        page_ref="p. 15",
        safety_level=safety_level,
        hash_sha256=hash_sha256,
    )


class TestDeduplicateChunks:
    def test_removes_duplicates(self) -> None:
        chunks = [
            _make_chunk(score=0.9, hash_sha256="same"),
            _make_chunk(score=0.8, hash_sha256="same"),
            _make_chunk(score=0.7, hash_sha256="different"),
        ]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2
        # Keeps the higher-scoring duplicate.
        assert result[0].score == 0.9

    def test_no_duplicates(self) -> None:
        chunks = [
            _make_chunk(hash_sha256="a"),
            _make_chunk(hash_sha256="b"),
        ]
        result = deduplicate_chunks(chunks)
        assert len(result) == 2


class TestValidateTenantChunks:
    def test_filters_wrong_tenant(self) -> None:
        chunks = [
            _make_chunk(tenant_id="t1"),
            _make_chunk(tenant_id="t2"),
            _make_chunk(tenant_id="t1"),
        ]
        result = validate_tenant_chunks(chunks, "t1")
        assert len(result) == 2
        assert all(c.tenant_id == "t1" for c in result)

    def test_all_valid(self) -> None:
        chunks = [_make_chunk(tenant_id="t1"), _make_chunk(tenant_id="t1")]
        result = validate_tenant_chunks(chunks, "t1")
        assert len(result) == 2


class TestEvaluateConfidence:
    def test_high_confidence(self) -> None:
        chunks = [_make_chunk(score=0.92)]
        conf, score = evaluate_confidence(chunks)
        assert conf == QueryConfidence.HIGH
        assert score == 0.92

    def test_low_confidence(self) -> None:
        chunks = [_make_chunk(score=0.5)]
        conf, score = evaluate_confidence(chunks)
        assert conf == QueryConfidence.LOW
        assert score == 0.5

    def test_no_chunks(self) -> None:
        conf, score = evaluate_confidence([])
        assert conf == QueryConfidence.NONE
        assert score == 0.0


class TestBuildContextPrompt:
    def test_includes_query(self) -> None:
        chunks = [_make_chunk()]
        prompt = build_context_prompt(chunks, "Cual es el torque?")
        assert "Cual es el torque?" in prompt

    def test_safety_first(self) -> None:
        chunks = [
            _make_chunk(safety_level="informativo", text="Normal info"),
            _make_chunk(safety_level="critico", text="DANGER info"),
        ]
        prompt = build_context_prompt(chunks, "test")
        danger_pos = prompt.find("DANGER")
        normal_pos = prompt.find("Normal")
        assert danger_pos < normal_pos


class TestProcessQuery:
    def test_no_chunks_returns_no_info(self) -> None:
        result = process_query("test query", [], "t1")
        assert result.confidence == QueryConfidence.NONE
        assert result.answer == NO_INFO_RESPONSE

    def test_high_confidence_generates(self) -> None:
        chunks = [_make_chunk(score=0.92, hash_sha256=f"h{i}") for i in range(5)]
        result = process_query(
            "test query",
            chunks,
            "t1",
            generate_fn=lambda p: "Generated answer",
        )
        assert result.confidence == QueryConfidence.HIGH
        assert result.answer == "Generated answer"

    def test_low_confidence_adds_warning(self) -> None:
        chunks = [_make_chunk(score=0.5, hash_sha256=f"h{i}") for i in range(3)]
        result = process_query(
            "test query",
            chunks,
            "t1",
            generate_fn=lambda p: "Some answer",
        )
        assert result.confidence == QueryConfidence.LOW
        assert "ATENCION" in result.answer

    def test_tenant_isolation(self) -> None:
        chunks = [
            _make_chunk(tenant_id="t1", hash_sha256="h1"),
            _make_chunk(tenant_id="t2", hash_sha256="h2"),
        ]
        result = process_query("test", chunks, "t1")
        # Only t1 chunks should be used.
        assert all(c.tenant_id == "t1" for c in result.chunks_used)
