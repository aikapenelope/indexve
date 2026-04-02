"""Initial schema: tenants, users, sessions, messages, usage_logs, documents.

Revision ID: 001_initial
Revises: None
Create Date: 2026-04-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Tenants.
    op.create_table(
        "tenants",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("plan", sa.String(20), nullable=False, server_default="free"),
        sa.Column("clerk_org_id", sa.String(64), nullable=False, unique=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Users.
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("clerk_user_id", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="tecnico"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Sessions.
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.String(64),
            sa.ForeignKey("users.id"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("title", sa.String(255)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Messages.
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.Integer(),
            sa.ForeignKey("sessions.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(10)),
        sa.Column("score", sa.Float()),
        sa.Column("chunks_used", sa.Integer()),
        sa.Column("was_fallback", sa.Boolean(), server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Usage logs.
    op.create_table(
        "usage_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "user_id",
            sa.String(64),
            sa.ForeignKey("users.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(20)),
        sa.Column("confidence", sa.String(10)),
        sa.Column("score", sa.Float()),
        sa.Column("input_tokens", sa.Integer(), server_default=sa.text("0")),
        sa.Column("output_tokens", sa.Integer(), server_default=sa.text("0")),
        sa.Column("latency_ms", sa.Integer(), server_default=sa.text("0")),
        sa.Column("model_used", sa.String(50)),
        sa.Column("was_fallback", sa.Boolean(), server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    # Documents.
    op.create_table(
        "documents",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "tenant_id",
            sa.String(64),
            sa.ForeignKey("tenants.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("equipment", sa.String(255)),
        sa.Column("manufacturer", sa.String(255)),
        sa.Column("doc_language", sa.String(5), server_default="en"),
        sa.Column("procedure_type", sa.String(30), server_default="informativo"),
        sa.Column("total_chunks", sa.Integer(), server_default=sa.text("0")),
        sa.Column("parser_used", sa.String(20)),
        sa.Column("hash_sha256", sa.String(64)),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("error_message", sa.Text()),
        sa.Column("indexed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("documents")
    op.drop_table("usage_logs")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("users")
    op.drop_table("tenants")
