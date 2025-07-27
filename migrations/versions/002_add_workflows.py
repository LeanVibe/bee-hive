"""Add workflows table for multi-agent workflow coordination

Revision ID: 002
Revises: 001
Create Date: 2024-07-26 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add workflows table and related functionality."""
    
    # Create workflows table with string fields (simpler, no enum complications)
    op.create_table(
        'workflows',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='CREATED', index=True),
        sa.Column('priority', sa.String(10), nullable=False, server_default='MEDIUM', index=True),
        sa.Column('definition', postgresql.JSON, nullable=False, server_default='{}'),
        sa.Column('task_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True, server_default='{}'),
        sa.Column('dependencies', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('context', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('variables', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('total_tasks', sa.Integer, nullable=False, server_default='0'),
        sa.Column('completed_tasks', sa.Integer, nullable=False, server_default='0'),
        sa.Column('failed_tasks', sa.Integer, nullable=False, server_default='0'),
        sa.Column('result', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('estimated_duration', sa.Integer, nullable=True),
        sa.Column('actual_duration', sa.Integer, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create performance indexes for workflows
    op.create_index('idx_workflows_status', 'workflows', ['status'])
    op.create_index('idx_workflows_priority', 'workflows', ['priority'])
    op.create_index('idx_workflows_status_priority', 'workflows', ['status', 'priority'])
    op.create_index('idx_workflows_created_at', 'workflows', ['created_at'])
    op.create_index('idx_workflows_started_at', 'workflows', ['started_at'])
    op.create_index('idx_workflows_due_date', 'workflows', ['due_date'])


def downgrade() -> None:
    """Remove workflows table and related functionality."""
    
    # Drop indexes
    op.drop_index('idx_workflows_due_date')
    op.drop_index('idx_workflows_started_at')
    op.drop_index('idx_workflows_created_at')
    op.drop_index('idx_workflows_status_priority')
    op.drop_index('idx_workflows_priority')
    op.drop_index('idx_workflows_status')
    
    # Drop workflows table
    op.drop_table('workflows')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS workflowpriority')
    op.execute('DROP TYPE IF EXISTS workflowstatus')