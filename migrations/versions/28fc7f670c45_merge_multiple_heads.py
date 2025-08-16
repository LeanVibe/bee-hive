"""merge multiple heads

Revision ID: 28fc7f670c45
Revises: 023_add_websocket_event_history, 203b2482e14b
Create Date: 2025-08-17 02:18:56.688182

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '28fc7f670c45'
down_revision = ('023_add_websocket_event_history', '203b2482e14b')
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    pass


def downgrade() -> None:
    """Downgrade database schema."""
    pass
