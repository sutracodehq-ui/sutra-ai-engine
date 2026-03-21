"""Initial migration — create all tables.

Revision ID: 001_initial
Revises: -
Create Date: 2026-03-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Tenants ────────────────────────────────────────
    op.create_table(
        "tenants",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("live_key_hash", sa.String(255), nullable=False),
        sa.Column("live_key_prefix", sa.String(30), nullable=False),
        sa.Column("test_key_hash", sa.String(255), nullable=False),
        sa.Column("test_key_prefix", sa.String(30), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("config", sa.JSON, default=dict),
        sa.Column("rate_limits", sa.JSON, nullable=True),
        sa.Column("contact_email", sa.String(255), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── AI Conversations ───────────────────────────────
    op.create_table(
        "ai_conversations",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("agent_type", sa.String(50), nullable=False),
        sa.Column("external_user_id", sa.String(255), nullable=True),
        sa.Column("metadata_", sa.JSON, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── AI Tasks ───────────────────────────────────────
    op.create_table(
        "ai_tasks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("conversation_id", sa.BigInteger, sa.ForeignKey("ai_conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("agent_type", sa.String(50), nullable=False, index=True),
        sa.Column("status", sa.String(20), nullable=False, default="pending", index=True),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("driver_used", sa.String(50), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, default=0),
        sa.Column("completion_tokens", sa.Integer, default=0),
        sa.Column("total_tokens", sa.Integer, default=0),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("external_user_id", sa.String(255), nullable=True),
        sa.Column("metadata_", sa.JSON, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── Agent Feedback ─────────────────────────────────
    op.create_table(
        "agent_feedback",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("task_id", sa.BigInteger, sa.ForeignKey("ai_tasks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action", sa.String(20), nullable=False),  # accepted, edited, rejected
        sa.Column("original_content", sa.Text, nullable=True),
        sa.Column("edited_content", sa.Text, nullable=True),
        sa.Column("quality_score", sa.Float, nullable=True),
        sa.Column("feedback_text", sa.Text, nullable=True),
        sa.Column("metadata_", sa.JSON, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── Voice Profiles ─────────────────────────────────
    op.create_table(
        "voice_profiles",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tone", sa.String(100), nullable=True),
        sa.Column("style", sa.String(100), nullable=True),
        sa.Column("instructions", sa.Text, nullable=True),
        sa.Column("example_content", sa.Text, nullable=True),
        sa.Column("is_default", sa.Boolean, default=False),
        sa.Column("metadata_", sa.JSON, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── Agent Training Data ────────────────────────────
    op.create_table(
        "agent_training_data",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("agent_type", sa.String(50), nullable=False, index=True),
        sa.Column("input_prompt", sa.Text, nullable=False),
        sa.Column("output_content", sa.Text, nullable=False),
        sa.Column("quality_score", sa.Float, nullable=True),
        sa.Column("source", sa.String(50), nullable=True),  # feedback, manual, synthetic
        sa.Column("metadata_", sa.JSON, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ─── Token Usage Logs ───────────────────────────────
    op.create_table(
        "token_usage_logs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.BigInteger, sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("task_id", sa.BigInteger, sa.ForeignKey("ai_tasks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("driver", sa.String(50), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt_tokens", sa.Integer, default=0),
        sa.Column("completion_tokens", sa.Integer, default=0),
        sa.Column("total_tokens", sa.Integer, default=0),
        sa.Column("cost_usd", sa.Float, default=0.0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("token_usage_logs")
    op.drop_table("agent_training_data")
    op.drop_table("voice_profiles")
    op.drop_table("agent_feedback")
    op.drop_table("ai_tasks")
    op.drop_table("ai_conversations")
    op.drop_table("tenants")
