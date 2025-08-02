"""Enable pgvector extension

Revision ID: a47c9cb5af36
Revises: 019_add_custom_commands_system
Create Date: 2025-08-02 01:49:55.944237

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a47c9cb5af36'
down_revision = '019_add_custom_commands_system'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Enable pgvector extension for vector similarity search
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')


def downgrade() -> None:
    """Downgrade database schema."""
    # Note: We don't drop the vector extension in downgrade 
    # as it may be used by other applications
    pass
