"""Add messaging and communication tables

Revision ID: 003_add_messaging
Revises: 002_add_workflows
Create Date: 2025-07-26 17:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add messaging and communication tables."""
    
    # Create message_audit table (using string fields for simplicity)
    op.create_table('message_audit',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('message_id', sa.String(length=255), nullable=False),
        sa.Column('stream_name', sa.String(length=255), nullable=False),
        sa.Column('from_agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('to_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('message_type', sa.String(length=20), nullable=False),
        sa.Column('priority', sa.String(length=10), nullable=False, server_default='normal'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('correlation_id', sa.String(length=255), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.String(), nullable=False, server_default='0'),
        sa.Column('delivery_latency_ms', sa.String(), nullable=True),
        sa.Column('processing_latency_ms', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['from_agent_id'], ['agents.id'], ),
        sa.ForeignKeyConstraint(['to_agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('idx_message_audit_message_id', 'message_audit', ['message_id'])
    op.create_index('idx_message_audit_stream_name', 'message_audit', ['stream_name'])
    op.create_index('idx_message_audit_from_agent_id', 'message_audit', ['from_agent_id'])
    op.create_index('idx_message_audit_to_agent_id', 'message_audit', ['to_agent_id'])
    op.create_index('idx_message_audit_message_type', 'message_audit', ['message_type'])
    op.create_index('idx_message_audit_status', 'message_audit', ['status'])
    op.create_index('idx_message_audit_correlation_id', 'message_audit', ['correlation_id'])
    op.create_index('idx_message_audit_sent_at', 'message_audit', ['sent_at'])
    
    # Create composite indexes for common query patterns
    op.create_index('idx_message_audit_agent_type_time', 'message_audit', 
                   ['from_agent_id', 'message_type', 'sent_at'])
    op.create_index('idx_message_audit_stream_status_time', 'message_audit', 
                   ['stream_name', 'status', 'sent_at'])


def downgrade() -> None:
    """Remove messaging and communication tables."""
    
    # Drop indexes
    op.drop_index('idx_message_audit_stream_status_time', table_name='message_audit')
    op.drop_index('idx_message_audit_agent_type_time', table_name='message_audit')
    op.drop_index('idx_message_audit_sent_at', table_name='message_audit')
    op.drop_index('idx_message_audit_correlation_id', table_name='message_audit')
    op.drop_index('idx_message_audit_status', table_name='message_audit')
    op.drop_index('idx_message_audit_message_type', table_name='message_audit')
    op.drop_index('idx_message_audit_to_agent_id', table_name='message_audit')
    op.drop_index('idx_message_audit_from_agent_id', table_name='message_audit')
    op.drop_index('idx_message_audit_stream_name', table_name='message_audit')
    op.drop_index('idx_message_audit_message_id', table_name='message_audit')
    
    # Drop table
    op.drop_table('message_audit')
    
    # Drop enums
    sa.Enum(name='messagestatus').drop(op.get_bind())
    sa.Enum(name='messagepriority').drop(op.get_bind())
    sa.Enum(name='messagetype').drop(op.get_bind())