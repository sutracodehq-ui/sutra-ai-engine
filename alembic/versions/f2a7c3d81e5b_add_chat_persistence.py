"""Add chat persistence tables (chat_sessions + chat_messages).

Revision ID: f2a7c3d81e5b
Revises: e1f3a7b92d4c
Create Date: 2026-06-03
"""

from alembic import op
import sqlalchemy as sa

revision = "f2a7c3d81e5b"
down_revision = "e1f3a7b92d4c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── chat_sessions ──────────────────────────────────────────
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("tenant_id", sa.BigInteger(), sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("visitor_fingerprint", sa.String(255), nullable=True, index=True),
        sa.Column("channel", sa.String(20), server_default="websocket", nullable=False),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("status", sa.String(20), server_default="active", nullable=False, index=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("message_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # ── chat_messages ──────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id", sa.String(100),
            sa.ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("tenant_id", sa.BigInteger(), sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("actions", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
