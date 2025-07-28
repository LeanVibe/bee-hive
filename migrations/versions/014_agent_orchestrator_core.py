"""Agent Orchestrator Core Enhanced Schema

Revision ID: 014_agent_orchestrator_core
Revises: 013_add_prompt_optimization_system
Create Date: 2025-01-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '014'
down_revision = '013'
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade to Agent Orchestrator Core enhanced schema."""
    
    # Add enhanced orchestrator fields to agents table
    op.add_column('agents', sa.Column('orchestrator_metadata', postgresql.JSONB(), nullable=True, server_default='{}'))
    op.add_column('agents', sa.Column('lifecycle_state', sa.String(20), nullable=False, server_default='inactive'))
    op.add_column('agents', sa.Column('agent_version', sa.String(50), nullable=True))
    op.add_column('agents', sa.Column('spawn_time', sa.DateTime(timezone=True), nullable=True))
    op.add_column('agents', sa.Column('termination_time', sa.DateTime(timezone=True), nullable=True))
    op.add_column('agents', sa.Column('restart_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('agents', sa.Column('health_score', sa.Float(), nullable=False, server_default='1.0'))
    op.add_column('agents', sa.Column('resource_usage', postgresql.JSONB(), nullable=True, server_default='{}'))
    
    # Add enhanced orchestrator fields to tasks table
    op.add_column('tasks', sa.Column('orchestrator_metadata', postgresql.JSONB(), nullable=True, server_default='{}'))
    op.add_column('tasks', sa.Column('assignment_strategy', sa.String(50), nullable=True))
    op.add_column('tasks', sa.Column('assignment_confidence', sa.Float(), nullable=True))
    op.add_column('tasks', sa.Column('execution_metadata', postgresql.JSONB(), nullable=True, server_default='{}'))
    op.add_column('tasks', sa.Column('retry_strategy', postgresql.JSONB(), nullable=True, server_default='{}'))
    op.add_column('tasks', sa.Column('timeout_seconds', sa.Integer(), nullable=True))
    op.add_column('tasks', sa.Column('queued_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('execution_start_time', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('execution_end_time', sa.DateTime(timezone=True), nullable=True))
    
    # Create agent_orchestrator_sessions table for session management
    op.create_table('agent_orchestrator_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('session_type', sa.String(50), nullable=False),  # 'task_execution', 'maintenance', 'idle'
        sa.Column('session_data', postgresql.JSONB(), nullable=True, server_default='{}'),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create task_queue table for enhanced task queuing
    op.create_table('task_queue',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('queue_name', sa.String(100), nullable=False, server_default='default'),
        sa.Column('priority_score', sa.Float(), nullable=False, server_default='5.0'),
        sa.Column('scheduling_metadata', postgresql.JSONB(), nullable=True, server_default='{}'),
        sa.Column('queue_position', sa.Integer(), nullable=True),
        sa.Column('estimated_wait_time', sa.Integer(), nullable=True),  # in seconds
        sa.Column('queued_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('dequeued_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('queue_status', sa.String(20), nullable=False, server_default='queued'),  # 'queued', 'assigned', 'expired'
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create orchestrator_health_checks table for health monitoring
    op.create_table('orchestrator_health_checks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('check_type', sa.String(50), nullable=False),  # 'heartbeat', 'resource', 'capability', 'performance'
        sa.Column('check_result', sa.String(20), nullable=False),  # 'healthy', 'warning', 'critical', 'failed'
        sa.Column('check_data', postgresql.JSONB(), nullable=True, server_default='{}'),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('checked_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create orchestrator_metrics table for performance tracking
    op.create_table('orchestrator_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('metric_type', sa.String(50), nullable=False),  # 'agent_spawn_time', 'task_assignment_time', 'throughput'
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(20), nullable=True),  # 'ms', 'seconds', 'count', 'percent'
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id', ondelete='CASCADE'), nullable=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True, server_default='{}'),
        sa.Column('measured_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    
    # Create comprehensive indexes for optimal performance
    
    # Agent-related indexes
    op.create_index('idx_agents_lifecycle_state', 'agents', ['lifecycle_state'])
    op.create_index('idx_agents_health_score', 'agents', ['health_score'])
    op.create_index('idx_agents_spawn_time', 'agents', ['spawn_time'])
    op.create_index('idx_agents_last_heartbeat_status', 'agents', ['last_heartbeat', 'status'])
    
    # Task-related indexes
    op.create_index('idx_tasks_assignment_confidence', 'tasks', ['assignment_confidence'])
    op.create_index('idx_tasks_queued_at', 'tasks', ['queued_at'])
    op.create_index('idx_tasks_execution_times', 'tasks', ['execution_start_time', 'execution_end_time'])
    op.create_index('idx_tasks_timeout', 'tasks', ['timeout_seconds'])
    
    # Agent orchestrator sessions indexes
    op.create_index('idx_orchestrator_sessions_agent_id', 'agent_orchestrator_sessions', ['agent_id'])
    op.create_index('idx_orchestrator_sessions_type_status', 'agent_orchestrator_sessions', ['session_type', 'status'])
    op.create_index('idx_orchestrator_sessions_started_at', 'agent_orchestrator_sessions', ['started_at'])
    
    # Task queue indexes
    op.create_index('idx_task_queue_task_id', 'task_queue', ['task_id'])
    op.create_index('idx_task_queue_priority_score', 'task_queue', ['priority_score'])
    op.create_index('idx_task_queue_name_status', 'task_queue', ['queue_name', 'queue_status'])
    op.create_index('idx_task_queue_position', 'task_queue', ['queue_position'])
    op.create_index('idx_task_queue_queued_at', 'task_queue', ['queued_at'])
    
    # Health checks indexes
    op.create_index('idx_health_checks_agent_id', 'orchestrator_health_checks', ['agent_id'])
    op.create_index('idx_health_checks_type_result', 'orchestrator_health_checks', ['check_type', 'check_result'])
    op.create_index('idx_health_checks_checked_at', 'orchestrator_health_checks', ['checked_at'])
    
    # Metrics indexes
    op.create_index('idx_orchestrator_metrics_type_name', 'orchestrator_metrics', ['metric_type', 'metric_name'])
    op.create_index('idx_orchestrator_metrics_agent_id', 'orchestrator_metrics', ['agent_id'])
    op.create_index('idx_orchestrator_metrics_task_id', 'orchestrator_metrics', ['task_id'])
    op.create_index('idx_orchestrator_metrics_measured_at', 'orchestrator_metrics', ['measured_at'])
    
    # Add constraints for data integrity
    
    # Lifecycle state constraint
    op.create_check_constraint(
        'chk_agents_lifecycle_state',
        'agents',
        "lifecycle_state IN ('inactive', 'initializing', 'active', 'busy', 'sleeping', 'error', 'shutting_down', 'terminated')"
    )
    
    # Health score constraint
    op.create_check_constraint(
        'chk_agents_health_score',
        'agents',
        'health_score >= 0.0 AND health_score <= 1.0'
    )
    
    # Queue status constraint
    op.create_check_constraint(
        'chk_task_queue_status',
        'task_queue',
        "queue_status IN ('queued', 'assigned', 'expired', 'cancelled')"
    )
    
    # Priority score constraint
    op.create_check_constraint(
        'chk_task_queue_priority_score',
        'task_queue',
        'priority_score >= 0.0 AND priority_score <= 10.0'
    )
    
    # Health check result constraint
    op.create_check_constraint(
        'chk_health_checks_result',
        'orchestrator_health_checks',
        "check_result IN ('healthy', 'warning', 'critical', 'failed')"
    )
    
    # Session status constraint
    op.create_check_constraint(
        'chk_orchestrator_sessions_status',
        'agent_orchestrator_sessions',
        "status IN ('active', 'completed', 'failed', 'cancelled')"
    )


def downgrade():
    """Downgrade from Agent Orchestrator Core enhanced schema."""
    
    # Drop constraints
    op.drop_constraint('chk_orchestrator_sessions_status', 'agent_orchestrator_sessions')
    op.drop_constraint('chk_health_checks_result', 'orchestrator_health_checks')
    op.drop_constraint('chk_task_queue_priority_score', 'task_queue')
    op.drop_constraint('chk_task_queue_status', 'task_queue')
    op.drop_constraint('chk_agents_health_score', 'agents')
    op.drop_constraint('chk_agents_lifecycle_state', 'agents')
    
    # Drop indexes
    op.drop_index('idx_orchestrator_metrics_measured_at')
    op.drop_index('idx_orchestrator_metrics_task_id')
    op.drop_index('idx_orchestrator_metrics_agent_id')
    op.drop_index('idx_orchestrator_metrics_type_name')
    op.drop_index('idx_health_checks_checked_at')
    op.drop_index('idx_health_checks_type_result')
    op.drop_index('idx_health_checks_agent_id')
    op.drop_index('idx_task_queue_queued_at')
    op.drop_index('idx_task_queue_position')
    op.drop_index('idx_task_queue_name_status')
    op.drop_index('idx_task_queue_priority_score')
    op.drop_index('idx_task_queue_task_id')
    op.drop_index('idx_orchestrator_sessions_started_at')
    op.drop_index('idx_orchestrator_sessions_type_status')
    op.drop_index('idx_orchestrator_sessions_agent_id')
    op.drop_index('idx_tasks_timeout')
    op.drop_index('idx_tasks_execution_times')
    op.drop_index('idx_tasks_queued_at')
    op.drop_index('idx_tasks_assignment_confidence')
    op.drop_index('idx_agents_last_heartbeat_status')
    op.drop_index('idx_agents_spawn_time')
    op.drop_index('idx_agents_health_score')
    op.drop_index('idx_agents_lifecycle_state')
    
    # Drop tables
    op.drop_table('orchestrator_metrics')
    op.drop_table('orchestrator_health_checks')
    op.drop_table('task_queue')
    op.drop_table('agent_orchestrator_sessions')
    
    # Remove columns from tasks table
    op.drop_column('tasks', 'execution_end_time')
    op.drop_column('tasks', 'execution_start_time')
    op.drop_column('tasks', 'queued_at')
    op.drop_column('tasks', 'timeout_seconds')
    op.drop_column('tasks', 'retry_strategy')
    op.drop_column('tasks', 'execution_metadata')
    op.drop_column('tasks', 'assignment_confidence')
    op.drop_column('tasks', 'assignment_strategy')
    op.drop_column('tasks', 'orchestrator_metadata')
    
    # Remove columns from agents table
    op.drop_column('agents', 'resource_usage')
    op.drop_column('agents', 'health_score')
    op.drop_column('agents', 'restart_count')
    op.drop_column('agents', 'termination_time')
    op.drop_column('agents', 'spawn_time')
    op.drop_column('agents', 'agent_version')
    op.drop_column('agents', 'lifecycle_state')
    op.drop_column('agents', 'orchestrator_metadata')