"""Initial database schema for LeanVibe Agent Hive 2.0

Revision ID: 001
Revises: 
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Enable required PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create agents table
    op.create_table(
        'agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('type', sa.String(20), nullable=False, server_default='CLAUDE'),
        sa.Column('role', sa.String(100), nullable=True, index=True),
        sa.Column('capabilities', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('system_prompt', sa.Text, nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='INACTIVE', index=True),
        sa.Column('config', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('tmux_session', sa.String(255), nullable=True),
        sa.Column('total_tasks_completed', sa.String, nullable=True, server_default='0'),
        sa.Column('total_tasks_failed', sa.String, nullable=True, server_default='0'),
        sa.Column('average_response_time', sa.String, nullable=True, server_default='0.0'),
        sa.Column('context_window_usage', sa.String, nullable=True, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_active', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create sessions table
    op.create_table(
        'sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('session_type', sa.String(30), nullable=False, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='INACTIVE', index=True),
        sa.Column('participant_agents', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True, server_default='{}'),
        sa.Column('lead_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('state', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('shared_context', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('objectives', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('tmux_session_id', sa.String(255), nullable=True, unique=True),
        sa.Column('config', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('auto_consolidate', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('max_duration_hours', sa.String, nullable=True, server_default='24'),
        sa.Column('total_tasks', sa.String, nullable=True, server_default='0'),
        sa.Column('completed_tasks', sa.String, nullable=True, server_default='0'),
        sa.Column('failed_tasks', sa.String, nullable=True, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_activity', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create tasks table
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('title', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('task_type', sa.String(30), nullable=True, index=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='PENDING', index=True),
        sa.Column('priority', sa.String(10), nullable=False, server_default='MEDIUM', index=True),
        sa.Column('assigned_agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('created_by_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('dependencies', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True, server_default='{}'),
        sa.Column('blocking_tasks', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True, server_default='{}'),
        sa.Column('context', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('required_capabilities', postgresql.ARRAY(sa.String), nullable=True, server_default='{}'),
        sa.Column('estimated_effort', sa.Integer, nullable=True),
        sa.Column('actual_effort', sa.Integer, nullable=True),
        sa.Column('result', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('retry_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer, nullable=False, server_default='3'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['assigned_agent_id'], ['agents.id']),
        sa.ForeignKeyConstraint(['created_by_agent_id'], ['agents.id']),
    )
    
    # Create contexts table with vector embedding support
    op.create_table(
        'contexts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('title', sa.String(255), nullable=False, index=True),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('context_type', sa.String(30), nullable=False, index=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('parent_context_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('related_context_ids', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('importance_score', sa.Float, nullable=False, server_default='0.5'),
        sa.Column('access_count', sa.String, nullable=False, server_default='0'),
        sa.Column('relevance_decay', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('is_consolidated', sa.String, nullable=False, server_default='false'),
        sa.Column('consolidation_summary', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('accessed_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('consolidated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id']),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id']),
        sa.ForeignKeyConstraint(['parent_context_id'], ['contexts.id']),
    )
    
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('from_agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('to_agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('message_type', sa.String(20), nullable=False, index=True),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('context_refs', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id']),
        sa.ForeignKeyConstraint(['from_agent_id'], ['agents.id']),
        sa.ForeignKeyConstraint(['to_agent_id'], ['agents.id']),
    )
    
    # Create system_checkpoints table
    op.create_table(
        'system_checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('checkpoint_type', sa.String(20), nullable=False, index=True),
        sa.Column('state', postgresql.JSON, nullable=False),
        sa.Column('git_commit_hash', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Create sleep_wake_cycles table
    op.create_table(
        'sleep_wake_cycles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('cycle_type', sa.String(30), nullable=False, index=True),
        sa.Column('sleep_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('wake_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consolidation_summary', sa.Text, nullable=True),
        sa.Column('context_changes', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id']),
    )
    
    # Create performance_metrics table
    op.create_table(
        'performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('metric_name', sa.String(255), nullable=False, index=True),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id']),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id']),
    )
    
    # Create indexes for performance optimization
    op.create_index('idx_contexts_embedding_cosine', 'contexts', ['embedding'], postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('idx_conversations_embedding_cosine', 'conversations', ['embedding'], postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('idx_tasks_status_priority', 'tasks', ['status', 'priority'])
    op.create_index('idx_agents_status_role', 'agents', ['status', 'role'])
    op.create_index('idx_sessions_status_type', 'sessions', ['status', 'session_type'])
    op.create_index('idx_contexts_importance_score', 'contexts', ['importance_score'])
    op.create_index('idx_performance_metrics_timestamp', 'performance_metrics', ['timestamp'])


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop indexes
    op.drop_index('idx_performance_metrics_timestamp')
    op.drop_index('idx_contexts_importance_score')
    op.drop_index('idx_sessions_status_type')
    op.drop_index('idx_agents_status_role')
    op.drop_index('idx_tasks_status_priority')
    op.drop_index('idx_conversations_embedding_cosine')
    op.drop_index('idx_contexts_embedding_cosine')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('performance_metrics')
    op.drop_table('sleep_wake_cycles')
    op.drop_table('system_checkpoints')
    op.drop_table('conversations')
    op.drop_table('contexts')
    op.drop_table('tasks')
    op.drop_table('sessions')
    op.drop_table('agents')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS cycletype')
    op.execute('DROP TYPE IF EXISTS checkpointtype')
    op.execute('DROP TYPE IF EXISTS messagetype')
    op.execute('DROP TYPE IF EXISTS contexttype')
    op.execute('DROP TYPE IF EXISTS taskpriority')
    op.execute('DROP TYPE IF EXISTS taskstatus')
    op.execute('DROP TYPE IF EXISTS tasktype')
    op.execute('DROP TYPE IF EXISTS sessionstatus')
    op.execute('DROP TYPE IF EXISTS sessiontype')
    op.execute('DROP TYPE IF EXISTS agentstatus')
    op.execute('DROP TYPE IF EXISTS agenttype')