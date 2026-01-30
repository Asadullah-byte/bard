"""add_onboarding_responses_table

Revision ID: 9d8633ce5d41
Revises: d573044e2507
Create Date: 2026-01-13 12:59:41.174373

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d8633ce5d41'
down_revision: Union[str, Sequence[str], None] = 'd573044e2507'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('onboarding_responses',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False, ForeignKeyConstraint=['users.id'],ondelete='CASCADE',onupdate='CASCADE'),
        sa.Column('responses', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('onboarding_responses')

