"""Enhanced workflow engine with DAG task dependencies for Vertical Slice 3.2

Revision ID: 015
Revises: 014
Create Date: 2024-07-28 15:00:00.000000

Adds task dependencies table and enhanced workflow capabilities for multi-step
workflow engine with DAG (Directed Acyclic Graph) task dependencies.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '015'
down_revision = '014'
branch_labels = None
depends_on = None


def add_column_if_not_exists(table_name: str, column: sa.Column):
    """Add column only if it doesn't already exist."""
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    
    if column.name not in columns:
        op.add_column(table_name, column)
        print(f"✅ Added column '{column.name}' to table '{table_name}'")
    else:
        print(f"✅ Column '{column.name}' already exists in table '{table_name}'")


def upgrade() -> None:
    """Add enhanced workflow engine capabilities with DAG task dependencies."""
    
    # Create task_dependencies table for explicit dependency tracking
    op.create_table(
        'task_dependencies',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('depends_on_task_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('dependency_type', sa.String(50), nullable=False, server_default='completion'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Add foreign key constraints
    op.create_foreign_key(
        'fk_task_dependencies_workflow',
        'task_dependencies',
        'workflows',
        ['workflow_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_task_dependencies_task',
        'task_dependencies',
        'tasks',
        ['task_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    op.create_foreign_key(
        'fk_task_dependencies_depends_on_task',
        'task_dependencies',
        'tasks',
        ['depends_on_task_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Create performance indexes for task dependencies
    op.create_index('idx_task_dependencies_workflow_id', 'task_dependencies', ['workflow_id'])
    op.create_index('idx_task_dependencies_task_id', 'task_dependencies', ['task_id'])
    op.create_index('idx_task_dependencies_depends_on_task_id', 'task_dependencies', ['depends_on_task_id'])
    op.create_index('idx_task_dependencies_type', 'task_dependencies', ['dependency_type'])
    op.create_index('idx_task_dependencies_workflow_task', 'task_dependencies', ['workflow_id', 'task_id'])
    
    # Add unique constraint to prevent duplicate dependencies
    op.create_unique_constraint(
        'uq_task_dependencies_unique',
        'task_dependencies',
        ['workflow_id', 'task_id', 'depends_on_task_id']
    )
    
    # Extend workflows table with enhanced DAG capabilities
    add_column_if_not_exists('workflows', sa.Column('execution_mode', sa.String(20), nullable=False, server_default='mixed'))
    add_column_if_not_exists('workflows', sa.Column('max_parallel_tasks', sa.Integer, nullable=True))
    add_column_if_not_exists('workflows', sa.Column('fail_fast', sa.Boolean, nullable=False, server_default='true'))
    add_column_if_not_exists('workflows', sa.Column('retry_failed_tasks', sa.Boolean, nullable=False, server_default='false'))
    add_column_if_not_exists('workflows', sa.Column('execution_plan', postgresql.JSON, nullable=True))
    add_column_if_not_exists('workflows', sa.Column('current_batch', sa.Integer, nullable=False, server_default='0'))
    add_column_if_not_exists('workflows', sa.Column('batch_progress', postgresql.JSON, nullable=True, server_default='{}'))
    add_column_if_not_exists('workflows', sa.Column('critical_path_duration', sa.Integer, nullable=True))
    add_column_if_not_exists('workflows', sa.Column('bottleneck_tasks', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True))
    
    # Add indexes for new workflow columns
    op.create_index('idx_workflows_execution_mode', 'workflows', ['execution_mode'])
    op.create_index('idx_workflows_current_batch', 'workflows', ['current_batch'])
    op.create_index('idx_workflows_fail_fast', 'workflows', ['fail_fast'])
    
    # Create workflow_execution_logs table for detailed execution tracking
    op.create_table(
        'workflow_execution_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('event_type', sa.String(50), nullable=False, index=True),
        sa.Column('event_data', postgresql.JSON, nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('batch_number', sa.Integer, nullable=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('agent_id', sa.String(255), nullable=True, index=True),
        sa.Column('duration_ms', sa.Integer, nullable=True),
        sa.Column('success', sa.Boolean, nullable=True),
    )
    
    # Add foreign key for workflow execution logs
    op.create_foreign_key(
        'fk_workflow_execution_logs_workflow',
        'workflow_execution_logs',
        'workflows',
        ['workflow_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for workflow execution logs
    op.create_index('idx_workflow_execution_logs_workflow_execution', 'workflow_execution_logs', ['workflow_id', 'execution_id'])
    op.create_index('idx_workflow_execution_logs_event_type', 'workflow_execution_logs', ['event_type'])
    op.create_index('idx_workflow_execution_logs_timestamp', 'workflow_execution_logs', ['timestamp'])
    op.create_index('idx_workflow_execution_logs_batch_number', 'workflow_execution_logs', ['batch_number'])
    op.create_index('idx_workflow_execution_logs_success', 'workflow_execution_logs', ['success'])
    
    # Create workflow_state_snapshots table for state recovery
    op.create_table(
        'workflow_state_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workflow_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('snapshot_type', sa.String(30), nullable=False, server_default='checkpoint'),
        sa.Column('batch_number', sa.Integer, nullable=False),
        sa.Column('state_data', postgresql.JSON, nullable=False),
        sa.Column('task_states', postgresql.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('can_resume_from', sa.Boolean, nullable=False, server_default='true'),
    )
    
    # Add foreign key for workflow state snapshots
    op.create_foreign_key(
        'fk_workflow_state_snapshots_workflow',
        'workflow_state_snapshots',
        'workflows',
        ['workflow_id'],
        ['id'],
        ondelete='CASCADE'
    )
    
    # Create indexes for workflow state snapshots
    op.create_index('idx_workflow_state_snapshots_workflow_execution', 'workflow_state_snapshots', ['workflow_id', 'execution_id'])
    op.create_index('idx_workflow_state_snapshots_batch_number', 'workflow_state_snapshots', ['batch_number'])
    op.create_index('idx_workflow_state_snapshots_type', 'workflow_state_snapshots', ['snapshot_type'])
    op.create_index('idx_workflow_state_snapshots_can_resume', 'workflow_state_snapshots', ['can_resume_from'])
    
    # Extend tasks table with DAG execution metadata
    add_column_if_not_exists('tasks', sa.Column('execution_batch', sa.Integer, nullable=True))
    add_column_if_not_exists('tasks', sa.Column('parallel_group', sa.String(100), nullable=True))
    add_column_if_not_exists('tasks', sa.Column('critical_path', sa.Boolean, nullable=False, server_default='false'))
    add_column_if_not_exists('tasks', sa.Column('dependency_depth', sa.Integer, nullable=False, server_default='0'))
    add_column_if_not_exists('tasks', sa.Column('estimated_start_time', sa.DateTime(timezone=True), nullable=True))
    add_column_if_not_exists('tasks', sa.Column('blocking_tasks', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True))
    
    # Add indexes for new task columns
    op.create_index('idx_tasks_execution_batch', 'tasks', ['execution_batch'])
    op.create_index('idx_tasks_parallel_group', 'tasks', ['parallel_group'])
    op.create_index('idx_tasks_critical_path', 'tasks', ['critical_path'])
    op.create_index('idx_tasks_dependency_depth', 'tasks', ['dependency_depth'])


def downgrade() -> None:
    """Remove enhanced workflow engine capabilities."""
    
    # Drop new task columns
    op.drop_index('idx_tasks_dependency_depth')
    op.drop_index('idx_tasks_critical_path')
    op.drop_index('idx_tasks_parallel_group')
    op.drop_index('idx_tasks_execution_batch')
    
    op.drop_column('tasks', 'blocking_tasks')
    op.drop_column('tasks', 'estimated_start_time')
    op.drop_column('tasks', 'dependency_depth')
    op.drop_column('tasks', 'critical_path')
    op.drop_column('tasks', 'parallel_group')
    op.drop_column('tasks', 'execution_batch')
    
    # Drop workflow state snapshots table
    op.drop_index('idx_workflow_state_snapshots_can_resume')
    op.drop_index('idx_workflow_state_snapshots_type')
    op.drop_index('idx_workflow_state_snapshots_batch_number')
    op.drop_index('idx_workflow_state_snapshots_workflow_execution')
    op.drop_constraint('fk_workflow_state_snapshots_workflow', 'workflow_state_snapshots', type_='foreignkey')
    op.drop_table('workflow_state_snapshots')
    
    # Drop workflow execution logs table
    op.drop_index('idx_workflow_execution_logs_success')
    op.drop_index('idx_workflow_execution_logs_batch_number')
    op.drop_index('idx_workflow_execution_logs_timestamp')
    op.drop_index('idx_workflow_execution_logs_event_type')
    op.drop_index('idx_workflow_execution_logs_workflow_execution')
    op.drop_constraint('fk_workflow_execution_logs_workflow', 'workflow_execution_logs', type_='foreignkey')
    op.drop_table('workflow_execution_logs')
    
    # Drop new workflow columns
    op.drop_index('idx_workflows_fail_fast')
    op.drop_index('idx_workflows_current_batch')
    op.drop_index('idx_workflows_execution_mode')
    
    op.drop_column('workflows', 'bottleneck_tasks')
    op.drop_column('workflows', 'critical_path_duration')
    op.drop_column('workflows', 'batch_progress')
    op.drop_column('workflows', 'current_batch')
    op.drop_column('workflows', 'execution_plan')
    op.drop_column('workflows', 'retry_failed_tasks')
    op.drop_column('workflows', 'fail_fast')
    op.drop_column('workflows', 'max_parallel_tasks')
    op.drop_column('workflows', 'execution_mode')
    
    # Drop task dependencies table
    op.drop_constraint('uq_task_dependencies_unique', 'task_dependencies', type_='unique')
    op.drop_index('idx_task_dependencies_workflow_task')
    op.drop_index('idx_task_dependencies_type')
    op.drop_index('idx_task_dependencies_depends_on_task_id')
    op.drop_index('idx_task_dependencies_task_id')
    op.drop_index('idx_task_dependencies_workflow_id')
    
    op.drop_constraint('fk_task_dependencies_depends_on_task', 'task_dependencies', type_='foreignkey')
    op.drop_constraint('fk_task_dependencies_task', 'task_dependencies', type_='foreignkey')
    op.drop_constraint('fk_task_dependencies_workflow', 'task_dependencies', type_='foreignkey')
    
    op.drop_table('task_dependencies')