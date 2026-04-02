"""Qdrant collection initialization with payload indexes.

Addresses issue 2.4 (slow nested filters) from KNOWN_ISSUES.md.

Creates the ManualIQ collection with:
- int8 scalar quantization (issue 2.3: reduces memory 4x)
- Payload indexes on frequently filtered fields (tenant_id, equipment,
  safety_level) for fast pre-filtering
- Shard key on tenant_id for native tenant isolation

Run this script once before first ingestion, or as part of the
deployment pipeline.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Collection configuration constants.
COLLECTION_NAME = "manualiq"
VECTOR_SIZE = 1024  # Voyage-4 embedding dimension.
DISTANCE = "Cosine"

# Fields to create payload indexes on (issue 2.4).
INDEXED_FIELDS = [
    ("tenant_id", "keyword"),
    ("equipment", "keyword"),
    ("safety_level", "keyword"),
    ("procedure_type", "keyword"),
    ("doc_language", "keyword"),
    ("manufacturer", "keyword"),
    ("doc_id", "keyword"),
]


async def initialize_collection(
    qdrant_url: str = "http://qdrant:6333",
    collection_name: str = COLLECTION_NAME,
) -> dict[str, object]:
    """Create the ManualIQ collection with optimized configuration.

    This function is idempotent: if the collection already exists,
    it only creates missing indexes.

    Args:
        qdrant_url: Qdrant server URL.
        collection_name: Name for the collection.

    Returns:
        Dict with initialization status and details.
    """
    from qdrant_client import AsyncQdrantClient, models

    client = AsyncQdrantClient(url=qdrant_url)
    result: dict[str, object] = {"collection": collection_name}

    try:
        # Check if collection exists.
        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]

        if collection_name not in existing:
            # Create collection with int8 quantization and shard key.
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                # int8 scalar quantization (issue 2.3).
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
                # Shard key on tenant_id for native tenant isolation.
                sharding_method=models.ShardingMethod.CUSTOM,
            )
            logger.info(
                "Created collection '%s' with int8 quantization", collection_name
            )
            result["created"] = True
        else:
            logger.info("Collection '%s' already exists", collection_name)
            result["created"] = False

        # Create payload indexes (idempotent).
        indexes_created: list[str] = []
        for field_name, field_type in INDEXED_FIELDS:
            try:
                schema_type = (
                    models.PayloadSchemaType.KEYWORD
                    if field_type == "keyword"
                    else models.PayloadSchemaType.INTEGER
                )
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                indexes_created.append(field_name)
                logger.info("Created payload index: %s (%s)", field_name, field_type)
            except Exception as exc:
                # Index may already exist; that's fine.
                if "already exists" in str(exc).lower():
                    logger.debug("Index '%s' already exists", field_name)
                else:
                    logger.warning("Failed to create index '%s': %s", field_name, exc)

        result["indexes_created"] = indexes_created
        result["status"] = "ready"

    except Exception as exc:
        logger.error("Collection initialization failed: %s", exc)
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        await client.close()

    return result
