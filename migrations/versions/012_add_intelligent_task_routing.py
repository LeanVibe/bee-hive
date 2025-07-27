"""Add intelligent task routing and performance tracking

Revision ID: 012_add_intelligent_task_routing
Revises: 011_vector_search_optimization
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '012_add_intelligent_task_routing'
down_revision = '011_vector_search_optimization'
branch_labels = None
depends_on = None


def upgrade():
    """Add intelligent task routing and performance tracking tables."""
    
    # Create agent_performance_history table
    op.create_table(
        'agent_performance_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id'), nullable=False, index=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id'), nullable=True, index=True),
        
        # Performance metrics
        sa.Column('task_type', sa.String(100), nullable=True, index=True),
        sa.Column('success', sa.Boolean, nullable=False, default=False),
        sa.Column('completion_time_minutes', sa.Float, nullable=True),
        sa.Column('estimated_time_minutes', sa.Float, nullable=True),
        sa.Column('time_variance_ratio', sa.Float, nullable=True),
        
        # Quality metrics
        sa.Column('retry_count', sa.Integer, nullable=False, default=0),
        sa.Column('error_rate', sa.Float, nullable=False, default=0.0),
        sa.Column('confidence_score', sa.Float, nullable=True),
        
        # Context and metadata
        sa.Column('context_window_usage', sa.Float, nullable=True),
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('cpu_usage_percent', sa.Float, nullable=True),
        
        # Task characteristics
        sa.Column('priority_level', sa.Integer, nullable=True),
        sa.Column('complexity_score', sa.Float, nullable=True),
        sa.Column('required_capabilities', postgresql.JSONB, nullable=True, default=sa.text('\'[]\'::jsonb')),
        
        # Timestamps
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )
    
    # Create task_routing_decisions table
    op.create_table(
        'task_routing_decisions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id'), nullable=False, index=True),
        sa.Column('selected_agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id'), nullable=False, index=True),
        
        # Routing context
        sa.Column('routing_strategy', sa.String(100), nullable=True),
        sa.Column('candidate_agents', postgresql.JSONB, nullable=True, default=sa.text('\'[]\'::jsonb')),
        sa.Column('selection_criteria', postgresql.JSONB, nullable=True, default=sa.text('\'{}\'::jsonb')),
        
        # Scoring information
        sa.Column('agent_scores', postgresql.JSONB, nullable=True, default=sa.text('\'{}\'::jsonb')),
        sa.Column('final_score', sa.Float, nullable=True),
        sa.Column('confidence_level', sa.Float, nullable=True),
        
        # Performance tracking
        sa.Column('routing_time_ms', sa.Float, nullable=True),
        sa.Column('decision_factors', postgresql.JSONB, nullable=True, default=sa.text('\'{}\'::jsonb')),
        
        # Outcome tracking
        sa.Column('task_completed', sa.Boolean, nullable=True),
        sa.Column('task_success', sa.Boolean, nullable=True),
        sa.Column('actual_completion_time', sa.Float, nullable=True),
        sa.Column('outcome_score', sa.Float, nullable=True),
        
        # Timestamps
        sa.Column('decided_at', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column('outcome_recorded_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create agent_capability_scores table
    op.create_table(
        'agent_capability_scores',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id'), nullable=False, index=True),
        
        # Capability identification
        sa.Column('capability_name', sa.String(255), nullable=False, index=True),
        sa.Column('task_type', sa.String(100), nullable=True, index=True),
        
        # Scoring metrics
        sa.Column('base_score', sa.Float, nullable=False, default=0.5),
        sa.Column('experience_factor', sa.Float, nullable=False, default=0.0),
        sa.Column('recent_performance', sa.Float, nullable=False, default=0.5),
        sa.Column('confidence_level', sa.Float, nullable=False, default=0.5),
        
        # Statistical data
        sa.Column('total_tasks', sa.Integer, nullable=False, default=0),
        sa.Column('successful_tasks', sa.Integer, nullable=False, default=0),
        sa.Column('average_completion_time', sa.Float, nullable=True),
        
        # Trending information
        sa.Column('trend_direction', sa.String(20), nullable=True),
        sa.Column('trend_strength', sa.Float, nullable=True),
        
        # Timestamps
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create workload_snapshots table
    op.create_table(
        'workload_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('agents.id'), nullable=False, index=True),
        
        # Workload metrics
        sa.Column('active_tasks', sa.Integer, nullable=False, default=0),
        sa.Column('pending_tasks', sa.Integer, nullable=False, default=0),
        sa.Column('context_usage_percent', sa.Float, nullable=False, default=0.0),
        
        # Resource utilization
        sa.Column('memory_usage_mb', sa.Float, nullable=True),
        sa.Column('cpu_usage_percent', sa.Float, nullable=True),
        sa.Column('estimated_capacity', sa.Float, nullable=False, default=1.0),
        sa.Column('utilization_ratio', sa.Float, nullable=False, default=0.0),
        
        # Task distribution
        sa.Column('priority_distribution', postgresql.JSONB, nullable=True, default=sa.text('\'{}\'::jsonb')),
        sa.Column('task_type_distribution', postgresql.JSONB, nullable=True, default=sa.text('\'{}\'::jsonb')),
        
        # Performance indicators
        sa.Column('average_response_time_ms', sa.Float, nullable=True),
        sa.Column('throughput_tasks_per_hour', sa.Float, nullable=True),
        sa.Column('error_rate_percent', sa.Float, nullable=False, default=0.0),
        
        # Timestamps
        sa.Column('snapshot_time', sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )
    
    # Create indexes for optimal query performance
    
    # Performance history indexes
    op.create_index(
        'idx_agent_performance_agent_task_type', 
        'agent_performance_history', 
        ['agent_id', 'task_type']
    )
    op.create_index(
        'idx_agent_performance_recorded_at_desc',
        'agent_performance_history',
        [sa.text('recorded_at DESC')]
    )
    op.create_index(
        'idx_agent_performance_success_time',
        'agent_performance_history',
        ['success', 'recorded_at']
    )
    
    # Routing decisions indexes
    op.create_index(
        'idx_routing_decisions_agent_strategy',
        'task_routing_decisions',
        ['selected_agent_id', 'routing_strategy']
    )
    op.create_index(
        'idx_routing_decisions_decided_at_desc',
        'task_routing_decisions',
        [sa.text('decided_at DESC')]
    )
    op.create_index(
        'idx_routing_decisions_outcome',
        'task_routing_decisions',
        ['task_success', 'confidence_level']
    )
    
    # Capability scores indexes
    op.create_index(
        'idx_capability_scores_agent_capability',
        'agent_capability_scores',
        ['agent_id', 'capability_name']
    )
    op.create_index(
        'idx_capability_scores_task_type_score',
        'agent_capability_scores',
        ['task_type', sa.text('base_score DESC')]
    )
    op.create_index(
        'idx_capability_scores_updated_desc',
        'agent_capability_scores',
        [sa.text('last_updated DESC')]
    )
    
    # Workload snapshots indexes
    op.create_index(
        'idx_workload_snapshots_agent_time',
        'workload_snapshots',
        ['agent_id', sa.text('snapshot_time DESC')]
    )
    op.create_index(
        'idx_workload_snapshots_utilization',
        'workload_snapshots',
        ['utilization_ratio', 'snapshot_time']
    )
    op.create_index(
        'idx_workload_snapshots_time_desc',
        'workload_snapshots',
        [sa.text('snapshot_time DESC')]
    )
    
    # Add unique constraints for logical uniqueness
    op.create_unique_constraint(
        'uq_capability_scores_agent_capability_type',
        'agent_capability_scores',
        ['agent_id', 'capability_name', 'task_type']
    )


def downgrade():
    """Remove intelligent task routing and performance tracking tables."""
    
    # Drop tables in reverse order
    op.drop_table('workload_snapshots')
    op.drop_table('agent_capability_scores')
    op.drop_table('task_routing_decisions')
    op.drop_table('agent_performance_history')