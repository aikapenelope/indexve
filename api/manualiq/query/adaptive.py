"""Adaptive RAG: adjust pipeline depth based on query complexity.

Instead of running the full pipeline for every query, we adapt:
- SIMPLE queries (specific, high confidence): top-10 retrieve, skip rerank, top-3 to LLM
- STANDARD queries (specific/procedure): full pipeline (top-20, rerank, top-5)
- COMPLEX queries (listing/comparison): sub-question decomposition + top-20 + rerank + top-7

The intent classifier already detects the type. This module maps
intents to pipeline configurations.

Reference: docs/AUDIT.md Section 11.2, docs/ROADMAP_V2.md B3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from manualiq.query.intelligence import QueryIntent

logger = logging.getLogger(__name__)


class PipelineDepth(Enum):
    """Pipeline depth levels."""

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class PipelineConfig:
    """Configuration for a specific pipeline depth."""

    depth: PipelineDepth
    retrieve_limit: int
    rerank_enabled: bool
    rerank_top_n: int
    use_sub_questions: bool
    use_query_expansion: bool


# Pipeline configurations per depth level.
PIPELINE_CONFIGS: dict[PipelineDepth, PipelineConfig] = {
    PipelineDepth.SIMPLE: PipelineConfig(
        depth=PipelineDepth.SIMPLE,
        retrieve_limit=10,
        rerank_enabled=False,
        rerank_top_n=3,
        use_sub_questions=False,
        use_query_expansion=False,
    ),
    PipelineDepth.STANDARD: PipelineConfig(
        depth=PipelineDepth.STANDARD,
        retrieve_limit=20,
        rerank_enabled=True,
        rerank_top_n=5,
        use_sub_questions=False,
        use_query_expansion=True,
    ),
    PipelineDepth.COMPLEX: PipelineConfig(
        depth=PipelineDepth.COMPLEX,
        retrieve_limit=20,
        rerank_enabled=True,
        rerank_top_n=7,
        use_sub_questions=True,
        use_query_expansion=True,
    ),
}


def select_pipeline(
    intent: QueryIntent,
    intent_confidence: float,
) -> PipelineConfig:
    """Select the appropriate pipeline configuration based on intent.

    Args:
        intent: The classified query intent.
        intent_confidence: Confidence of the intent classification.

    Returns:
        PipelineConfig for the selected depth level.
    """
    # High-confidence specific queries use the simple pipeline.
    if intent == QueryIntent.SPECIFIC and intent_confidence >= 0.8:
        depth = PipelineDepth.SIMPLE
    # Listing and comparison queries need the complex pipeline.
    elif intent in (QueryIntent.LISTING, QueryIntent.COMPARISON):
        depth = PipelineDepth.COMPLEX
    # Procedure queries use standard (need full reranking for step ordering).
    elif intent == QueryIntent.PROCEDURE:
        depth = PipelineDepth.STANDARD
    # Everything else uses standard.
    else:
        depth = PipelineDepth.STANDARD

    config = PIPELINE_CONFIGS[depth]
    logger.info(
        "Adaptive RAG: intent=%s (%.2f) -> depth=%s (retrieve=%d, rerank=%s, sub_q=%s)",
        intent.value,
        intent_confidence,
        depth.value,
        config.retrieve_limit,
        config.rerank_enabled,
        config.use_sub_questions,
    )
    return config
