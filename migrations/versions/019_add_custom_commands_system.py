"""Add Custom Commands System - Phase 6.1

Revision ID: 019_add_custom_commands_system
Revises: 018_vs7_2_automated_scheduler
Create Date: 2025-01-26 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '019_add_custom_commands_system'
down_revision = '018_vs7_2_automated_scheduler'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add custom commands system tables and indexes."""
    
    # Command Registry Table
    op.create_table(
        'command_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(100), nullable=False, index=True),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('definition', postgresql.JSONB, nullable=False),
        sa.Column('author_id', sa.String(100), nullable=True),
        sa.Column('signature', sa.String(128), nullable=False),
        sa.Column('enabled', sa.Boolean, default=True, nullable=False),
        sa.Column('execution_count', sa.Integer, default=0, nullable=False),
        sa.Column('success_rate', sa.Float, default=0.0, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Constraints
        sa.UniqueConstraint('name', 'version', name='uq_command_name_version'),
        sa.CheckConstraint('success_rate >= 0.0 AND success_rate <= 100.0', name='ck_success_rate_range'),
        sa.CheckConstraint('execution_count >= 0', name='ck_execution_count_positive')
    )
    
    # Command Executions Table
    op.create_table(
        'command_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('command_name', sa.String(100), nullable=False),
        sa.Column('command_version', sa.String(20), nullable=False),
        sa.Column('requester_id', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('parameters', postgresql.JSONB, nullable=False, default={}),
        sa.Column('context', postgresql.JSONB, nullable=False, default={}),
        sa.Column('priority', sa.String(20), nullable=False, default='medium'),
        sa.Column('start_time', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('end_time', sa.DateTime, nullable=True),
        sa.Column('total_execution_time_seconds', sa.Float, nullable=True),
        sa.Column('total_steps', sa.Integer, nullable=False, default=0),
        sa.Column('completed_steps', sa.Integer, nullable=False, default=0),
        sa.Column('failed_steps', sa.Integer, nullable=False, default=0),
        sa.Column('final_outputs', postgresql.JSONB, nullable=False, default={}),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('failure_point', sa.String(100), nullable=True),
        sa.Column('resource_usage', postgresql.JSONB, nullable=False, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Foreign key to command registry
        sa.ForeignKeyConstraint(
            ['command_name', 'command_version'],
            ['command_registry.name', 'command_registry.version'],
            name='fk_execution_command',
            ondelete='CASCADE'
        ),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused')",
            name='ck_execution_status'
        ),
        sa.CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'critical')",
            name='ck_execution_priority'
        ),
        sa.CheckConstraint('completed_steps >= 0', name='ck_completed_steps_positive'),
        sa.CheckConstraint('failed_steps >= 0', name='ck_failed_steps_positive'),
        sa.CheckConstraint('total_steps >= 0', name='ck_total_steps_positive')
    )
    
    # Step Executions Table
    op.create_table(
        'step_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step_id', sa.String(100), nullable=False),
        sa.Column('step_name', sa.String(200), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('start_time', sa.DateTime, nullable=True),
        sa.Column('end_time', sa.DateTime, nullable=True),
        sa.Column('execution_time_seconds', sa.Float, nullable=True),
        sa.Column('outputs', postgresql.JSONB, nullable=False, default={}),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('retry_count', sa.Integer, nullable=False, default=0),
        sa.Column('resource_usage', postgresql.JSONB, nullable=False, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Foreign keys
        sa.ForeignKeyConstraint(
            ['execution_id'],
            ['command_executions.id'],
            name='fk_step_execution',
            ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['agent_id'],
            ['agents.id'],
            name='fk_step_agent',
            ondelete='SET NULL'
        ),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name='ck_step_status'
        ),
        sa.CheckConstraint('retry_count >= 0', name='ck_retry_count_positive'),
        sa.UniqueConstraint('execution_id', 'step_id', name='uq_execution_step')
    )
    
    # Task Assignments Table
    op.create_table(
        'task_assignments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step_id', sa.String(100), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assignment_score', sa.Float, nullable=False, default=0.0),
        sa.Column('assignment_reason', sa.String(100), nullable=False),
        sa.Column('estimated_completion_time', sa.DateTime, nullable=True),
        sa.Column('backup_agents', postgresql.ARRAY(postgresql.UUID), nullable=False, default=[]),
        sa.Column('assigned_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='assigned'),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Foreign keys
        sa.ForeignKeyConstraint(
            ['execution_id'],
            ['command_executions.id'],
            name='fk_assignment_execution',
            ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['agent_id'],
            ['agents.id'],
            name='fk_assignment_agent',
            ondelete='CASCADE'
        ),
        
        # Constraints
        sa.CheckConstraint(
            "status IN ('assigned', 'accepted', 'rejected', 'in_progress', 'completed', 'failed')",
            name='ck_assignment_status'
        ),
        sa.CheckConstraint('assignment_score >= 0.0 AND assignment_score <= 1.0', name='ck_assignment_score_range'),
        sa.UniqueConstraint('execution_id', 'step_id', name='uq_execution_step_assignment')
    )
    
    # Command Security Policies Table
    op.create_table(
        'command_security_policies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('command_name', sa.String(100), nullable=False),
        sa.Column('command_version', sa.String(20), nullable=False),
        sa.Column('policy_definition', postgresql.JSONB, nullable=False),
        sa.Column('approved_by', sa.String(100), nullable=True),
        sa.Column('approval_date', sa.DateTime, nullable=True),
        sa.Column('risk_level', sa.String(20), nullable=False, default='medium'),
        sa.Column('audit_requirements', postgresql.JSONB, nullable=False, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Foreign key to command registry
        sa.ForeignKeyConstraint(
            ['command_name', 'command_version'],
            ['command_registry.name', 'command_registry.version'],
            name='fk_security_policy_command',
            ondelete='CASCADE'
        ),
        
        # Constraints
        sa.CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name='ck_risk_level'
        ),
        sa.UniqueConstraint('command_name', 'command_version', name='uq_security_policy_command')
    )
    
    # Command Performance Metrics Table
    op.create_table(
        'command_performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('command_name', sa.String(100), nullable=False),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_unit', sa.String(20), nullable=False),
        sa.Column('timestamp', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('step_id', sa.String(100), nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=False, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        
        # Foreign keys
        sa.ForeignKeyConstraint(
            ['execution_id'],
            ['command_executions.id'],
            name='fk_metrics_execution',
            ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['agent_id'],
            ['agents.id'],
            name='fk_metrics_agent',
            ondelete='SET NULL'
        )
    )
    
    # Create indexes for performance
    
    # Command Registry indexes
    op.create_index('idx_command_registry_name', 'command_registry', ['name'])
    op.create_index('idx_command_registry_created_at', 'command_registry', ['created_at'])
    op.create_index('idx_command_registry_author', 'command_registry', ['author_id'])
    op.create_index('idx_command_registry_enabled', 'command_registry', ['enabled'])
    
    # JSONB indexes for command definitions
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_command_registry_category 
        ON command_registry USING GIN ((definition->'category'))
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_command_registry_tags 
        ON command_registry USING GIN ((definition->'tags'))
    """)
    
    # Command Executions indexes
    op.create_index('idx_command_executions_command', 'command_executions', ['command_name', 'command_version'])
    op.create_index('idx_command_executions_status', 'command_executions', ['status'])
    op.create_index('idx_command_executions_requester', 'command_executions', ['requester_id'])
    op.create_index('idx_command_executions_start_time', 'command_executions', ['start_time'])
    op.create_index('idx_command_executions_priority', 'command_executions', ['priority'])
    
    # Step Executions indexes
    op.create_index('idx_step_executions_execution_id', 'step_executions', ['execution_id'])
    op.create_index('idx_step_executions_agent_id', 'step_executions', ['agent_id'])
    op.create_index('idx_step_executions_status', 'step_executions', ['status'])
    op.create_index('idx_step_executions_start_time', 'step_executions', ['start_time'])
    
    # Task Assignments indexes
    op.create_index('idx_task_assignments_execution_id', 'task_assignments', ['execution_id'])
    op.create_index('idx_task_assignments_agent_id', 'task_assignments', ['agent_id'])
    op.create_index('idx_task_assignments_status', 'task_assignments', ['status'])
    op.create_index('idx_task_assignments_assigned_at', 'task_assignments', ['assigned_at'])
    
    # Performance Metrics indexes
    op.create_index('idx_command_metrics_command_name', 'command_performance_metrics', ['command_name'])
    op.create_index('idx_command_metrics_execution_id', 'command_performance_metrics', ['execution_id'])
    op.create_index('idx_command_metrics_type', 'command_performance_metrics', ['metric_type'])
    op.create_index('idx_command_metrics_timestamp', 'command_performance_metrics', ['timestamp'])
    op.create_index('idx_command_metrics_agent_id', 'command_performance_metrics', ['agent_id'])
    
    # Security Policies indexes
    op.create_index('idx_security_policies_command', 'command_security_policies', ['command_name', 'command_version'])
    op.create_index('idx_security_policies_risk_level', 'command_security_policies', ['risk_level'])
    op.create_index('idx_security_policies_approved_by', 'command_security_policies', ['approved_by'])
    
    # Add triggers for updated_at timestamps
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Apply triggers to relevant tables
    for table in ['command_registry', 'command_executions', 'step_executions', 
                  'task_assignments', 'command_security_policies']:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at 
            BEFORE UPDATE ON {table}
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();
        """)
    
    # Create materialized view for command statistics
    op.execute("""
        CREATE MATERIALIZED VIEW command_statistics AS
        SELECT 
            cr.name,
            cr.version,
            cr.execution_count,
            cr.success_rate,
            COUNT(ce.id) as total_executions_recorded,
            AVG(ce.total_execution_time_seconds) as avg_execution_time,
            COUNT(CASE WHEN ce.status = 'completed' THEN 1 END) as successful_executions,
            COUNT(CASE WHEN ce.status = 'failed' THEN 1 END) as failed_executions,
            MAX(ce.start_time) as last_execution,
            cr.created_at,
            cr.updated_at
        FROM command_registry cr
        LEFT JOIN command_executions ce ON cr.name = ce.command_name AND cr.version = ce.command_version
        GROUP BY cr.name, cr.version, cr.execution_count, cr.success_rate, cr.created_at, cr.updated_at;
    """)
    
    # Create index on materialized view
    op.create_index('idx_command_statistics_name', 'command_statistics', ['name'])
    op.create_index('idx_command_statistics_success_rate', 'command_statistics', ['success_rate'])
    op.create_index('idx_command_statistics_last_execution', 'command_statistics', ['last_execution'])


def downgrade() -> None:
    """Remove custom commands system tables and indexes."""
    
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS command_statistics")
    
    # Drop triggers
    for table in ['command_registry', 'command_executions', 'step_executions', 
                  'task_assignments', 'command_security_policies']:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table}_updated_at ON {table}")
    
    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop tables in reverse dependency order
    op.drop_table('command_performance_metrics')
    op.drop_table('command_security_policies')
    op.drop_table('task_assignments')
    op.drop_table('step_executions')
    op.drop_table('command_executions')
    op.drop_table('command_registry')