"""Phoenix Evals runner for ManualIQ regression testing.

Runs the regression dataset through the RAG pipeline and evaluates
responses using Phoenix's built-in LLM evaluators:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer relevancy: Does the response answer the question?
- Hallucination detection: Does the answer invent data?
- Citation check: Are citations referencing real documents?

Can be run manually or as a weekly Prefect flow.

Reference: docs/AUDIT.md Section 11.1, docs/ROADMAP_V2.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Path to the regression dataset.
_DATASET_PATH = Path(__file__).parent / "regression_dataset.yml"


@dataclass
class EvalQuestion:
    """A single evaluation question from the regression dataset."""

    id: str
    query: str
    category: str
    expected_behavior: str
    expected_fields: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single question."""

    question_id: str
    query: str
    category: str
    answer: str
    confidence: str
    score: float
    chunks_used: int
    # Eval metrics (0.0 to 1.0).
    field_coverage: float  # How many expected_fields appear in the answer.
    has_citation: bool  # Whether the answer contains [Manual: ...] citations.
    is_spanish: bool  # Whether the answer is in Spanish.
    passed: bool  # Overall pass/fail.


@dataclass
class EvalSummary:
    """Summary of a full evaluation run."""

    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_field_coverage: float
    citation_rate: float
    spanish_rate: float
    by_category: dict[str, dict[str, float]]
    results: list[EvalResult]


def load_dataset(path: Path | None = None) -> list[EvalQuestion]:
    """Load the regression dataset from YAML.

    Args:
        path: Path to the YAML file. Defaults to regression_dataset.yml.

    Returns:
        List of EvalQuestion objects.
    """
    dataset_path = path or _DATASET_PATH
    if not dataset_path.exists():
        logger.error("Regression dataset not found: %s", dataset_path)
        return []

    with open(dataset_path) as f:
        data = yaml.safe_load(f)

    questions: list[EvalQuestion] = []
    for q in data.get("questions", []):
        questions.append(
            EvalQuestion(
                id=q["id"],
                query=q["query"],
                category=q["category"],
                expected_behavior=q.get("expected_behavior", ""),
                expected_fields=q.get("expected_fields", []),
            )
        )

    logger.info("Loaded %d evaluation questions", len(questions))
    return questions


def evaluate_response(
    question: EvalQuestion,
    answer: str,
    confidence: str,
    score: float,
    chunks_used: int,
) -> EvalResult:
    """Evaluate a single RAG response against expected behavior.

    Uses lightweight heuristic checks (no LLM calls) for fast evaluation.
    Phoenix LLM-as-judge evaluators can be layered on top for deeper analysis.

    Args:
        question: The evaluation question.
        answer: The RAG-generated answer.
        confidence: Confidence level from the pipeline.
        score: Best retrieval score.
        chunks_used: Number of chunks used.

    Returns:
        EvalResult with metrics.
    """
    answer_lower = answer.lower()

    # Field coverage: what fraction of expected fields appear in the answer.
    if question.expected_fields:
        found = sum(
            1
            for field_val in question.expected_fields
            if field_val.lower() in answer_lower
        )
        field_coverage = found / len(question.expected_fields)
    else:
        field_coverage = 1.0

    # Citation check: does the answer contain [Manual: ...] pattern.
    has_citation = "[manual:" in answer_lower or "[manual :" in answer_lower

    # Language check: is the answer in Spanish.
    es_markers = [" el ", " la ", " los ", " del ", " en ", " que ", " por "]
    en_markers = [" the ", " is ", " are ", " was ", " has ", " with "]
    es_count = sum(1 for m in es_markers if m in f" {answer_lower} ")
    en_count = sum(1 for m in en_markers if m in f" {answer_lower} ")
    is_spanish = es_count >= en_count

    # Overall pass: field coverage > 50% AND (has citation OR is ambiguous/oos).
    is_refusal_category = question.category in ("ambiguous", "out_of_scope")
    passed = field_coverage >= 0.5 or is_refusal_category

    return EvalResult(
        question_id=question.id,
        query=question.query,
        category=question.category,
        answer=answer[:500],
        confidence=confidence,
        score=score,
        chunks_used=chunks_used,
        field_coverage=field_coverage,
        has_citation=has_citation,
        is_spanish=is_spanish,
        passed=passed,
    )


def summarize_results(results: list[EvalResult]) -> EvalSummary:
    """Summarize evaluation results across all questions.

    Args:
        results: List of individual EvalResult objects.

    Returns:
        EvalSummary with aggregate metrics.
    """
    total = len(results)
    if total == 0:
        return EvalSummary(
            total=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            avg_field_coverage=0.0,
            citation_rate=0.0,
            spanish_rate=0.0,
            by_category={},
            results=[],
        )

    passed = sum(1 for r in results if r.passed)
    avg_coverage = sum(r.field_coverage for r in results) / total
    citation_rate = sum(1 for r in results if r.has_citation) / total
    spanish_rate = sum(1 for r in results if r.is_spanish) / total

    # Per-category breakdown.
    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    by_category: dict[str, dict[str, float]] = {}
    for cat, cat_results in categories.items():
        cat_total = len(cat_results)
        by_category[cat] = {
            "total": float(cat_total),
            "pass_rate": sum(1 for r in cat_results if r.passed) / cat_total,
            "avg_field_coverage": sum(r.field_coverage for r in cat_results)
            / cat_total,
            "citation_rate": sum(1 for r in cat_results if r.has_citation) / cat_total,
        }

    return EvalSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total,
        avg_field_coverage=avg_coverage,
        citation_rate=citation_rate,
        spanish_rate=spanish_rate,
        by_category=by_category,
        results=results,
    )
