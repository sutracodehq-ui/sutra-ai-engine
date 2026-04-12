"""add_missing_agent_optimization_columns

Revision ID: b3f1c22d9a01
Revises: a7a6eb90c105
Create Date: 2026-03-22 15:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b3f1c22d9a01'
down_revision: Union[str, None] = 'a7a6eb90c105'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns that were in the model but missing from the original migration
    op.add_column('agent_optimizations', sa.Column('status', sa.String(length=20), nullable=False, server_default='candidate'))
    op.add_column('agent_optimizations', sa.Column('trial_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('agent_optimizations', sa.Column('total_score', sa.Float(), nullable=False, server_default='0'))
    op.add_column('agent_optimizations', sa.Column('win_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('agent_optimizations', sa.Column('loss_count', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    op.drop_column('agent_optimizations', 'loss_count')
    op.drop_column('agent_optimizations', 'win_count')
    op.drop_column('agent_optimizations', 'total_score')
    op.drop_column('agent_optimizations', 'trial_count')
    op.drop_column('agent_optimizations', 'status')
