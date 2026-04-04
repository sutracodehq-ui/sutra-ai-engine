"""add_api_keys_table

Revision ID: c4e8a2f71b03
Revises: b3f1c22d9a01
Create Date: 2026-04-04 09:12:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c4e8a2f71b03'
down_revision: Union[str, None] = 'b3f1c22d9a01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create the api_keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('tenant_id', sa.BigInteger(), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False),
        sa.Column('key_prefix', sa.String(30), nullable=False),
        sa.Column('environment', sa.String(10), nullable=False, server_default='live'),
        sa.Column('label', sa.String(100), nullable=True),
        sa.Column('scopes', sa.JSON(), nullable=True, server_default='["*"]'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
    op.create_index('ix_api_keys_tenant_id', 'api_keys', ['tenant_id'])

    # 2. Migrate existing keys from tenants table into api_keys
    op.execute("""
        INSERT INTO api_keys (tenant_id, key_hash, key_prefix, environment, label, scopes, is_active, created_at, updated_at)
        SELECT id, live_key_hash, live_key_prefix, 'live', 'Default (migrated)', '["*"]'::json, true, created_at, NOW()
        FROM tenants
        WHERE live_key_hash IS NOT NULL AND live_key_hash != ''
    """)
    op.execute("""
        INSERT INTO api_keys (tenant_id, key_hash, key_prefix, environment, label, scopes, is_active, created_at, updated_at)
        SELECT id, test_key_hash, test_key_prefix, 'test', 'Default (migrated)', '["*"]'::json, true, created_at, NOW()
        FROM tenants
        WHERE test_key_hash IS NOT NULL AND test_key_hash != ''
    """)


def downgrade() -> None:
    op.drop_index('ix_api_keys_tenant_id', table_name='api_keys')
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_table('api_keys')
