"""Add observability and monitoring tables

Revision ID: 004_add_observability
Revises: 003_add_messaging
Create Date: 2025-07-26 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add observability and monitoring tables."""
    
    # Create agent_events table for observability (using string for event_type)
    op.create_table('agent_events',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(20), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance as specified in PRD
    op.create_index('idx_events_session', 'agent_events', ['session_id'])
    op.create_index('idx_events_type_time', 'agent_events', ['event_type', 'created_at'])
    op.create_index('idx_events_agent_id', 'agent_events', ['agent_id'])
    op.create_index('idx_events_created_at', 'agent_events', ['created_at'])
    
    # Create composite indexes for common query patterns
    op.create_index('idx_events_session_type_time', 'agent_events', 
                   ['session_id', 'event_type', 'created_at'])
    op.create_index('idx_events_agent_type_time', 'agent_events', 
                   ['agent_id', 'event_type', 'created_at'])
    
    # Create chat_transcripts table for optional S3/MinIO storage
    op.create_table('chat_transcripts',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('s3_key', sa.String(length=500), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for chat_transcripts
    op.create_index('idx_transcripts_session', 'chat_transcripts', ['session_id'])
    op.create_index('idx_transcripts_agent', 'chat_transcripts', ['agent_id'])
    op.create_index('idx_transcripts_created_at', 'chat_transcripts', ['created_at'])


def downgrade() -> None:
    """Remove observability and monitoring tables."""
    
    # Drop indexes
    op.drop_index('idx_transcripts_created_at', table_name='chat_transcripts')
    op.drop_index('idx_transcripts_agent', table_name='chat_transcripts')
    op.drop_index('idx_transcripts_session', table_name='chat_transcripts')
    
    op.drop_index('idx_events_agent_type_time', table_name='agent_events')
    op.drop_index('idx_events_session_type_time', table_name='agent_events')
    op.drop_index('idx_events_created_at', table_name='agent_events')
    op.drop_index('idx_events_agent_id', table_name='agent_events')
    op.drop_index('idx_events_type_time', table_name='agent_events')
    op.drop_index('idx_events_session', table_name='agent_events')
    
    # Drop tables
    op.drop_table('chat_transcripts')
    op.drop_table('agent_events')
    
    # Drop enum
    sa.Enum(name='eventtype').drop(op.get_bind())