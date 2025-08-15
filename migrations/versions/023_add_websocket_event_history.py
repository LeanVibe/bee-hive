"""Add WebSocket event history table

Revision ID: 023_add_websocket_event_history
Revises: 022_add_project_index_system
Create Date: 2025-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '023_add_websocket_event_history'
down_revision = '022_add_project_index_system'
branch_labels = None
depends_on = None


def upgrade():
    """Add WebSocket event history table for project index events."""
    # Create the event history table
    op.create_table(
        'project_index_event_history',
        sa.Column('id', postgresql.UUID(), nullable=False),
        sa.Column('project_id', postgresql.UUID(), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('event_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('correlation_id', sa.String(length=36), nullable=False),
        sa.Column('persistence_level', sa.String(length=20), nullable=False),
        sa.Column('checksum', sa.String(length=64), nullable=False),
        sa.Column('client_delivered', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('replay_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for better query performance
    op.create_index('ix_project_index_event_history_project_id', 'project_index_event_history', ['project_id'])
    op.create_index('ix_project_index_event_history_event_type', 'project_index_event_history', ['event_type'])
    op.create_index('ix_project_index_event_history_correlation_id', 'project_index_event_history', ['correlation_id'])
    op.create_index('ix_project_index_event_history_persistence_level', 'project_index_event_history', ['persistence_level'])
    op.create_index('ix_project_index_event_history_created_at', 'project_index_event_history', ['created_at'])
    op.create_index('ix_project_index_event_history_expires_at', 'project_index_event_history', ['expires_at'])
    
    # Create compound indexes for common query patterns
    op.create_index(
        'ix_project_index_event_history_project_created', 
        'project_index_event_history', 
        ['project_id', 'created_at']
    )
    op.create_index(
        'ix_project_index_event_history_project_type', 
        'project_index_event_history', 
        ['project_id', 'event_type']
    )
    op.create_index(
        'ix_project_index_event_history_expires_cleanup', 
        'project_index_event_history', 
        ['expires_at', 'persistence_level']
    )


def downgrade():
    """Remove WebSocket event history table."""
    # Drop indexes first
    op.drop_index('ix_project_index_event_history_expires_cleanup', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_project_type', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_project_created', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_expires_at', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_created_at', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_persistence_level', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_correlation_id', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_event_type', 'project_index_event_history')
    op.drop_index('ix_project_index_event_history_project_id', 'project_index_event_history')
    
    # Drop the table
    op.drop_table('project_index_event_history')