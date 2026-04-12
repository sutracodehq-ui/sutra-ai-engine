"""add_identity_org_id

Revision ID: a7a6eb90c105
Revises: 2da4764a7206
Create Date: 2026-03-21 10:37:01.803686
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a7a6eb90c105'
down_revision: Union[str, None] = '2da4764a7206'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tenants', sa.Column('identity_org_id', sa.String(), nullable=True))
    op.create_index('ix_tenants_identity_org_id', 'tenants', ['identity_org_id'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_tenants_identity_org_id', table_name='tenants')
    op.drop_column('tenants', 'identity_org_id')
