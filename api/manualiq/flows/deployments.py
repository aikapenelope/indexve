"""Prefect v3 deployment schedules for ManualIQ operational flows.

Registers the three operational flows with their production schedules:
- reindex_documents: Weekly (Sunday 2:00 AM UTC)
- backup_qdrant: Daily (3:00 AM UTC)
- health_check: Every 5 minutes

Run this script once to create the deployments in Prefect:
    python -m manualiq.flows.deployments

Reference: ARCHITECTURE.md Section 3.12, KNOWN_ISSUES.md issues 1.5, 5.2, 5.3.
"""

from __future__ import annotations

import os

from manualiq.flows.operations import backup_qdrant, health_check, reindex_documents


def deploy_all() -> None:
    """Register all flow deployments with Prefect schedules."""
    # Common config from environment.
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    resend_api_key = os.environ.get("RESEND_API_KEY", "")
    alert_email = os.environ.get("ALERT_EMAIL", "")
    corpus_dir = os.environ.get("CORPUS_DIR", "/corpus")
    default_tenant = os.environ.get("DEFAULT_TENANT_ID", "default")

    # 1. Re-indexation: weekly Sunday 2:00 AM UTC.
    # Detects changed documents by SHA-256 and re-indexes only what changed.
    # Saves Voyage API credits by skipping unchanged documents.
    reindex_documents.serve(
        name="reindex-weekly",
        cron="0 2 * * 0",
        parameters={
            "corpus_dir": corpus_dir,
            "tenant_id": default_tenant,
        },
    )

    # 2. Qdrant backup: daily 3:00 AM UTC.
    # Creates a snapshot and uploads to external storage.
    # Critical for disaster recovery without re-indexing.
    backup_qdrant.serve(
        name="backup-daily",
        cron="0 3 * * *",
        parameters={
            "qdrant_url": qdrant_url,
            "collection_name": "manualiq",
            "backup_dir": "/backups/qdrant",
        },
    )

    # 3. Health check: every 5 minutes.
    # Pings all services and alerts via Resend if any are down.
    health_check.serve(
        name="health-check-5min",
        cron="*/5 * * * *",
        parameters={
            "resend_api_key": resend_api_key,
            "alert_email": alert_email,
        },
    )


if __name__ == "__main__":
    deploy_all()
