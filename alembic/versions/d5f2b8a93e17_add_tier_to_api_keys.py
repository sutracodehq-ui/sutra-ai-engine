"""add_tier_to_api_keys

Revision ID: d5f2b8a93e17
Revises: c4e8a2f71b03
Create Date: 2026-04-04 09:35:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd5f2b8a93e17'
down_revision: Union[str, None] = 'c4e8a2f71b03'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add tier column with default 'standard'
    op.add_column('api_keys', sa.Column('tier', sa.String(20), nullable=False, server_default='standard'))
    op.create_index('ix_api_keys_tier', 'api_keys', ['tier'])


def downgrade() -> None:
    op.drop_index('ix_api_keys_tier', table_name='api_keys')
    op.drop_column('api_keys', 'tier')
