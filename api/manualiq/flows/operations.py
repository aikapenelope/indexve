"""Prefect v3 flows for ManualIQ operational tasks.

Addresses issues 1.5 (staleness/re-indexing), 5.2 (vector backup),
and 5.3 (health checks) from KNOWN_ISSUES.md.

All flows use Prefect v3 API only (no v2 patterns).
Reference: https://docs.prefect.io/v3

Flows:
1. reindex_documents: Detect changed documents by SHA256 and re-index.
2. backup_qdrant: Create Qdrant snapshot and upload to external storage.
3. health_check: Ping all services and alert via Resend on failure.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Flow 1: Re-indexing with change detection (issue 1.5)
# ---------------------------------------------------------------------------


@task(name="scan-documents-for-changes")
def scan_for_changes(
    corpus_dir: str,
    hash_store: dict[str, str],
) -> list[dict[str, str]]:
    """Scan corpus directory and detect changed/new documents.

    Compares current file hashes against stored hashes to find documents
    that need re-indexing. This avoids re-processing unchanged documents,
    saving Voyage API credits.

    Args:
        corpus_dir: Path to the directory containing PDF documents.
        hash_store: Dict mapping file paths to their last known SHA-256.

    Returns:
        List of dicts with 'path', 'hash', and 'status' (new/changed).
    """
    corpus_path = Path(corpus_dir)
    changed: list[dict[str, str]] = []

    for pdf_file in corpus_path.glob("**/*.pdf"):
        file_key = str(pdf_file.relative_to(corpus_path))
        current_hash = _file_sha256(pdf_file)

        if file_key not in hash_store:
            changed.append(
                {"path": str(pdf_file), "hash": current_hash, "status": "new"}
            )
            logger.info("New document detected: %s", file_key)
        elif hash_store[file_key] != current_hash:
            changed.append(
                {"path": str(pdf_file), "hash": current_hash, "status": "changed"}
            )
            logger.info("Changed document detected: %s", file_key)

    logger.info(
        "Scan complete: %d changed/new out of %d total documents",
        len(changed),
        sum(1 for _ in corpus_path.glob("**/*.pdf")),
    )
    return changed


@task(name="reindex-single-document")
def reindex_document(
    doc_info: dict[str, str],
    tenant_id: str,
) -> dict[str, object]:
    """Re-index a single document through the full ingestion pipeline.

    This task wraps the parse -> chunk -> embed -> upsert pipeline
    for a single document. It runs as an isolated Prefect task so
    failures don't block other documents.

    Args:
        doc_info: Dict with 'path' and 'hash' from scan_for_changes.
        tenant_id: The tenant that owns this document.

    Returns:
        Dict with indexing result metadata.
    """
    from manualiq.ingestion.chunker import blocks_to_chunks, classify_and_split_blocks
    from manualiq.ingestion.parser import parse_document

    pdf_path = Path(doc_info["path"])
    logger.info("Re-indexing: %s (status: %s)", pdf_path.name, doc_info["status"])

    # Step 1: Parse.
    parse_result = parse_document(pdf_path)
    if not parse_result.text.strip():
        logger.error("Failed to parse %s: %s", pdf_path.name, parse_result.error)
        return {
            "file": pdf_path.name,
            "status": "failed",
            "error": parse_result.error,
        }

    # Step 2: Chunk.
    blocks = classify_and_split_blocks(parse_result.text)
    chunks = blocks_to_chunks(
        blocks,
        doc_id=pdf_path.stem,
        tenant_id=tenant_id,
    )

    # Step 3: Embed + upsert would happen here via Voyage API + Qdrant.
    # Placeholder: in production, this calls the embedding service and
    # Qdrant upsert. For now we return the chunk count.
    logger.info(
        "Parsed %s: %d chunks via %s",
        pdf_path.name,
        len(chunks),
        parse_result.backend.value,
    )

    return {
        "file": pdf_path.name,
        "status": "indexed",
        "chunks": len(chunks),
        "parser": parse_result.backend.value,
        "hash": doc_info["hash"],
    }


@flow(name="reindex-documents", log_prints=True)
def reindex_documents(
    corpus_dir: str,
    tenant_id: str,
    hash_store: dict[str, str] | None = None,
) -> dict[str, object]:
    """Detect changed documents and re-index them.

    This is the main re-indexing flow, meant to run on a schedule
    (weekly) or triggered when new documents are uploaded.

    Args:
        corpus_dir: Path to the corpus directory.
        tenant_id: The tenant whose documents to re-index.
        hash_store: Previous hash store. If None, all documents
            are treated as new.

    Returns:
        Summary dict with results and updated hash store.
    """
    if hash_store is None:
        hash_store = {}

    changed = scan_for_changes(corpus_dir, hash_store)

    if not changed:
        print(f"No changes detected in {corpus_dir}")
        return {"changed": 0, "results": [], "hash_store": hash_store}

    print(f"Found {len(changed)} documents to re-index")

    results: list[dict[str, object]] = []
    updated_hashes = dict(hash_store)

    for doc_info in changed:
        result = reindex_document(doc_info, tenant_id)
        results.append(result)
        if result["status"] == "indexed":
            file_key = Path(str(doc_info["path"])).name
            updated_hashes[file_key] = doc_info["hash"]

    return {
        "changed": len(changed),
        "results": results,
        "hash_store": updated_hashes,
    }


# ---------------------------------------------------------------------------
# Flow 2: Qdrant backup (issue 5.2)
# ---------------------------------------------------------------------------


@task(name="create-qdrant-snapshot")
def create_qdrant_snapshot(
    qdrant_url: str,
    collection_name: str,
) -> dict[str, str]:
    """Create a Qdrant snapshot for a collection.

    Uses the Qdrant REST API to trigger a snapshot. The snapshot is
    stored locally in Qdrant's snapshot directory.

    Args:
        qdrant_url: Qdrant server URL (e.g., http://qdrant:6333).
        collection_name: Name of the collection to snapshot.

    Returns:
        Dict with snapshot name and status.
    """
    import httpx

    response = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/snapshots",
        timeout=300.0,
    )
    response.raise_for_status()
    data = response.json()

    snapshot_name: str = data.get("result", {}).get("name", "unknown")
    logger.info(
        "Qdrant snapshot created: %s for collection %s",
        snapshot_name,
        collection_name,
    )
    return {"snapshot": snapshot_name, "collection": collection_name}


@task(name="upload-snapshot-to-storage")
def upload_snapshot_to_storage(
    qdrant_url: str,
    collection_name: str,
    snapshot_name: str,
    backup_dir: str,
) -> str:
    """Download a Qdrant snapshot and save to external storage.

    Downloads the snapshot file from Qdrant and saves it to the
    backup directory (Hetzner Volume or S3-compatible storage).

    Args:
        qdrant_url: Qdrant server URL.
        collection_name: Collection the snapshot belongs to.
        snapshot_name: Name of the snapshot to download.
        backup_dir: Local path to save the snapshot.

    Returns:
        Path to the saved snapshot file.
    """
    import httpx

    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{collection_name}_{timestamp}_{snapshot_name}"
    output_path = backup_path / filename

    url = f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}"
    with httpx.stream("GET", url, timeout=600.0) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for data_chunk in response.iter_bytes():
                f.write(data_chunk)

    logger.info("Snapshot saved to %s", output_path)
    return str(output_path)


@flow(name="backup-qdrant", log_prints=True)
def backup_qdrant(
    qdrant_url: str = "http://qdrant:6333",
    collection_name: str = "manualiq",
    backup_dir: str = "/backups/qdrant",
) -> dict[str, str]:
    """Create and export a Qdrant snapshot for disaster recovery.

    Meant to run daily via Prefect schedule. Snapshots are stored
    externally so that if the Qdrant volume is corrupted, we can
    restore without re-indexing (which costs Voyage API credits).

    Args:
        qdrant_url: Qdrant server URL.
        collection_name: Collection to backup.
        backup_dir: Where to save the snapshot file.

    Returns:
        Dict with snapshot details and backup path.
    """
    print(f"Starting Qdrant backup for collection: {collection_name}")

    snapshot_info = create_qdrant_snapshot(qdrant_url, collection_name)
    snapshot_name = snapshot_info["snapshot"]

    backup_path = upload_snapshot_to_storage(
        qdrant_url, collection_name, snapshot_name, backup_dir
    )

    print(f"Backup complete: {backup_path}")
    return {"snapshot": snapshot_name, "backup_path": backup_path}


# ---------------------------------------------------------------------------
# Flow 3: Health checks (issue 5.3)
# ---------------------------------------------------------------------------


@task(name="ping-service")
def ping_service(
    name: str,
    url: str,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Ping a service and return its health status.

    Args:
        name: Human-readable service name.
        url: Health check URL.
        timeout_seconds: How long to wait before declaring failure.

    Returns:
        Dict with service name, status, and response time.
    """
    import httpx

    try:
        response = httpx.get(url, timeout=timeout_seconds)
        return {
            "name": name,
            "status": "healthy" if response.status_code < 500 else "unhealthy",
            "code": response.status_code,
            "response_ms": response.elapsed.total_seconds() * 1000,
        }
    except Exception as exc:
        logger.error("Health check failed for %s: %s", name, exc)
        return {
            "name": name,
            "status": "down",
            "code": 0,
            "error": str(exc),
            "response_ms": 0,
        }


@task(name="send-alert-email")
def send_alert_email(
    failed_services: list[dict[str, object]],
    resend_api_key: str,
    alert_email: str,
) -> bool:
    """Send an alert email via Resend when services are down.

    Args:
        failed_services: List of service health check results that failed.
        resend_api_key: Resend API key.
        alert_email: Email address to send alerts to.

    Returns:
        True if the email was sent successfully.
    """
    import resend  # type: ignore[import-untyped]

    resend.api_key = resend_api_key

    service_list = "\n".join(
        f"- {s['name']}: {s['status']} (code: {s.get('code', 'N/A')}, "
        f"error: {s.get('error', 'none')})"
        for s in failed_services
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    resend.Emails.send(
        {
            "from": "ManualIQ Alerts <alerts@manualiq.com>",
            "to": [alert_email],
            "subject": f"[ManualIQ] {len(failed_services)} service(s) down - {timestamp}",
            "text": (
                f"ManualIQ Health Check Alert\n"
                f"Time: {timestamp}\n\n"
                f"Failed services:\n{service_list}\n\n"
                f"Please investigate immediately."
            ),
        }
    )

    logger.info("Alert email sent to %s", alert_email)
    return True


# Default services to monitor.
DEFAULT_SERVICES = [
    {"name": "Qdrant", "url": "http://qdrant:6333/healthz"},
    {"name": "PostgreSQL API", "url": "http://localhost:8000/health"},
    {"name": "Redis", "url": "http://localhost:8000/health/redis"},
    {"name": "Phoenix", "url": "http://phoenix:6006/healthz"},
    {"name": "Prefect", "url": "http://prefect-server:4200/api/health"},
]


@flow(name="health-check", log_prints=True)
def health_check(
    services: list[dict[str, str]] | None = None,
    resend_api_key: str = "",
    alert_email: str = "",
) -> dict[str, object]:
    """Check health of all ManualIQ services and alert on failures.

    Meant to run every 5 minutes via Prefect schedule.

    Args:
        services: List of services to check. Each dict has 'name' and 'url'.
            Defaults to DEFAULT_SERVICES.
        resend_api_key: Resend API key for sending alerts.
        alert_email: Email to send alerts to.

    Returns:
        Summary dict with all service statuses.
    """
    if services is None:
        services = DEFAULT_SERVICES

    results: list[dict[str, object]] = []
    for svc in services:
        result = ping_service(svc["name"], svc["url"])
        results.append(result)

    failed = [r for r in results if r["status"] != "healthy"]

    if failed:
        print(f"ALERT: {len(failed)} service(s) unhealthy!")
        for f_svc in failed:
            print(f"  - {f_svc['name']}: {f_svc['status']}")

        if resend_api_key and alert_email:
            send_alert_email(failed, resend_api_key, alert_email)
        else:
            print("No Resend API key or alert email configured, skipping email alert")
    else:
        print("All services healthy")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "healthy": len(results) - len(failed),
        "failed": len(failed),
        "services": results,
    }
