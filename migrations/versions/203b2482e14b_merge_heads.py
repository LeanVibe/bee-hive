"""merge_heads

Revision ID: 203b2482e14b
Revises: 021_add_advanced_github_integration, d36c23fd2bf9
Create Date: 2025-08-08 01:21:11.477210

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '203b2482e14b'
down_revision = ('021_add_advanced_github_integration', 'd36c23fd2bf9')
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    pass


def downgrade() -> None:
    """Downgrade database schema."""
    pass
