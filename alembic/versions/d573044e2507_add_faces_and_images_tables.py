from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'd573044e2507'
down_revision: Union[str, Sequence[str], None] = '42e8c2b5bd35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create face_embeddings table
    op.create_table('face_embeddings',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('embedding', Vector(128), nullable=True),
    sa.Column('face_owner', sa.UUID(), nullable=False),
    sa.Column('image_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['face_owner'], ['users.id'], ondelete='CASCADE', onupdate='CASCADE'),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE', onupdate='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )

    # Create images table
    op.create_table('images',
    sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('file_owner', sa.UUID(), nullable=False),
    sa.Column('file_stored', sa.Text(), nullable=False),
    sa.Column('metadata', sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['file_owner'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('images')
    op.drop_table('face_embeddings')
    op.execute('DROP EXTENSION IF EXISTS vector')
