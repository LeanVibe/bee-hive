"""Add Self-Modification Engine Tables

Revision ID: 010
Revises: 009
Create Date: 2025-01-27 15:00:00.000000

Adds comprehensive self-modification engine tables for secure code evolution:
- modification_sessions: Track self-modification analysis sessions
- code_modifications: Individual file modifications with safety scoring
- modification_metrics: Performance metrics before/after modifications
- sandbox_executions: Results from isolated testing environments
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema with Self-Modification Engine tables."""
    
    # Create custom enums for self-modification engine
    op.execute("""
        CREATE TYPE modification_safety AS ENUM (
            'conservative', 'moderate', 'aggressive'
        );
    """)
    
    op.execute("""
        CREATE TYPE modification_status AS ENUM (
            'analyzing', 'suggestions_ready', 'applying', 'applied', 
            'failed', 'rolled_back', 'archived'
        );
    """)
    
    op.execute("""
        CREATE TYPE modification_type AS ENUM (
            'bug_fix', 'performance', 'feature_add', 'refactor', 
            'security_fix', 'style_improvement', 'dependency_update'
        );
    """)
    
    op.execute("""
        CREATE TYPE sandbox_execution_type AS ENUM (
            'unit_test', 'integration_test', 'security_scan', 
            'performance_benchmark', 'linting', 'type_check'
        );
    """)
    
    # Self-modification sessions - tracks analysis and modification attempts
    op.create_table(
        'modification_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('repository_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),  # Optional GitHub repo link
        sa.Column('codebase_path', sa.String(500), nullable=False, index=True),
        sa.Column('modification_goals', postgresql.JSON, nullable=False, server_default='[]'),  # ["improve_performance", "fix_bugs", "add_features"]
        sa.Column('safety_level', sa.Enum('conservative', 'moderate', 'aggressive', name='modification_safety'), 
                  nullable=False, server_default='conservative', index=True),
        sa.Column('status', sa.Enum('analyzing', 'suggestions_ready', 'applying', 'applied', 'failed', 'rolled_back', 'archived', name='modification_status'), 
                  nullable=False, server_default='analyzing', index=True),
        sa.Column('analysis_prompt', sa.Text, nullable=True),  # LLM prompt used for analysis
        sa.Column('analysis_context', postgresql.JSON, nullable=True, server_default='{}'),  # Context about codebase patterns
        sa.Column('total_suggestions', sa.Integer, nullable=False, server_default='0'),
        sa.Column('applied_modifications', sa.Integer, nullable=False, server_default='0'),
        sa.Column('success_rate', sa.Decimal(5,2), nullable=True),  # Percentage success rate
        sa.Column('performance_improvement', sa.Decimal(5,2), nullable=True),  # Overall performance gain/loss
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('session_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['repository_id'], ['github_repositories.id'], ondelete='SET NULL'),
    )
    
    # Individual code modifications with detailed tracking
    op.create_table(
        'code_modifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('file_path', sa.String(500), nullable=False, index=True),
        sa.Column('modification_type', sa.Enum('bug_fix', 'performance', 'feature_add', 'refactor', 'security_fix', 'style_improvement', 'dependency_update', name='modification_type'), 
                  nullable=False, index=True),
        sa.Column('original_content', sa.Text, nullable=True),  # Original file content
        sa.Column('modified_content', sa.Text, nullable=True),  # Modified file content
        sa.Column('content_diff', sa.Text, nullable=True),  # Unified diff for the change
        sa.Column('modification_reason', sa.Text, nullable=False),  # Explanation of why this change was made
        sa.Column('llm_reasoning', sa.Text, nullable=True),  # Detailed LLM reasoning
        sa.Column('safety_score', sa.Decimal(3,2), nullable=False),  # 0.0 to 1.0 safety assessment
        sa.Column('complexity_score', sa.Decimal(3,2), nullable=True),  # 0.0 to 1.0 complexity assessment
        sa.Column('performance_impact', sa.Decimal(5,2), nullable=True),  # Expected percentage change
        sa.Column('lines_added', sa.Integer, nullable=True),
        sa.Column('lines_removed', sa.Integer, nullable=True),
        sa.Column('functions_modified', sa.Integer, nullable=True),
        sa.Column('dependencies_changed', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('test_files_affected', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('approval_required', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('human_approved', sa.Boolean, nullable=True),
        sa.Column('approved_by', sa.String(255), nullable=True),  # Human approver identifier
        sa.Column('approval_token', sa.String(500), nullable=True),  # JWT token for approval
        sa.Column('git_commit_hash', sa.String(40), nullable=True, index=True),  # Associated git commit
        sa.Column('git_branch', sa.String(255), nullable=True),
        sa.Column('rollback_commit_hash', sa.String(40), nullable=True),  # Rollback restore point
        sa.Column('modification_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('applied_at', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('rollback_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['session_id'], ['modification_sessions.id'], ondelete='CASCADE'),
    )
    
    # Performance metrics before/after modifications
    op.create_table(
        'modification_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('modification_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('metric_name', sa.String(100), nullable=False, index=True),  # 'execution_time', 'memory_usage', 'error_rate', 'throughput'
        sa.Column('metric_category', sa.String(50), nullable=False, index=True),  # 'performance', 'quality', 'security'
        sa.Column('baseline_value', sa.Decimal(15,6), nullable=True),  # Value before modification
        sa.Column('modified_value', sa.Decimal(15,6), nullable=True),  # Value after modification
        sa.Column('improvement_percentage', sa.Decimal(8,4), nullable=True),  # Calculated improvement
        sa.Column('measurement_unit', sa.String(50), nullable=True),  # 'ms', 'MB', 'percent', 'count'
        sa.Column('measurement_context', sa.String(200), nullable=True),  # Test case or scenario
        sa.Column('measurement_tool', sa.String(100), nullable=True),  # Tool used for measurement
        sa.Column('confidence_score', sa.Decimal(3,2), nullable=True),  # 0.0 to 1.0 confidence in measurement
        sa.Column('statistical_significance', sa.Boolean, nullable=True),  # Whether improvement is statistically significant
        sa.Column('sample_size', sa.Integer, nullable=True),  # Number of measurements taken
        sa.Column('standard_deviation', sa.Decimal(10,6), nullable=True),
        sa.Column('measurement_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('measured_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['modification_id'], ['code_modifications.id'], ondelete='CASCADE'),
    )
    
    # Sandbox execution results for isolated testing
    op.create_table(
        'sandbox_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('modification_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('execution_type', sa.Enum('unit_test', 'integration_test', 'security_scan', 'performance_benchmark', 'linting', 'type_check', name='sandbox_execution_type'), 
                  nullable=False, index=True),
        sa.Column('container_id', sa.String(100), nullable=True),  # Docker container identifier
        sa.Column('image_name', sa.String(200), nullable=True),  # Docker image used
        sa.Column('command', sa.Text, nullable=False),  # Command executed in sandbox
        sa.Column('working_directory', sa.String(500), nullable=True),
        sa.Column('environment_variables', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('resource_limits', postgresql.JSON, nullable=True, server_default='{}'),  # CPU, memory, disk limits
        sa.Column('network_isolation', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('stdout', sa.Text, nullable=True),  # Standard output
        sa.Column('stderr', sa.Text, nullable=True),  # Standard error
        sa.Column('exit_code', sa.Integer, nullable=True, index=True),
        sa.Column('execution_time_ms', sa.Integer, nullable=True),  # Execution time in milliseconds
        sa.Column('memory_usage_mb', sa.Integer, nullable=True),  # Peak memory usage
        sa.Column('cpu_usage_percent', sa.Decimal(5,2), nullable=True),  # CPU utilization
        sa.Column('disk_usage_mb', sa.Integer, nullable=True),  # Disk space used
        sa.Column('network_attempts', sa.Integer, nullable=False, server_default='0'),  # Network access attempts (should be 0)
        sa.Column('security_violations', postgresql.JSON, nullable=True, server_default='[]'),  # Security issues detected
        sa.Column('file_system_changes', postgresql.JSON, nullable=True, server_default='[]'),  # Files created/modified/deleted
        sa.Column('test_results', postgresql.JSON, nullable=True, server_default='{}'),  # Structured test results
        sa.Column('performance_metrics', postgresql.JSON, nullable=True, server_default='{}'),  # Performance measurements
        sa.Column('sandbox_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['modification_id'], ['code_modifications.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['modification_sessions.id'], ondelete='CASCADE'),
    )
    
    # Learning and feedback tracking for context-aware improvements
    op.create_table(
        'modification_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('modification_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('feedback_source', sa.String(100), nullable=False, index=True),  # 'human', 'automated', 'metrics'
        sa.Column('feedback_type', sa.String(50), nullable=False, index=True),  # 'rating', 'comment', 'correction', 'approval'
        sa.Column('rating', sa.Integer, nullable=True),  # 1-5 scale rating
        sa.Column('feedback_text', sa.Text, nullable=True),  # Human comments or automated feedback
        sa.Column('patterns_identified', postgresql.JSON, nullable=True, server_default='[]'),  # Code patterns learned
        sa.Column('anti_patterns_identified', postgresql.JSON, nullable=True, server_default='[]'),  # Patterns to avoid
        sa.Column('improvement_suggestions', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('user_preferences', postgresql.JSON, nullable=True, server_default='{}'),  # Learned user preferences
        sa.Column('project_conventions', postgresql.JSON, nullable=True, server_default='{}'),  # Project-specific conventions
        sa.Column('impact_score', sa.Decimal(3,2), nullable=True),  # 0.0 to 1.0 impact assessment
        sa.Column('applied_to_learning', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('feedback_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['modification_id'], ['code_modifications.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['modification_sessions.id'], ondelete='CASCADE'),
    )
    
    # Create indexes for performance optimization
    op.create_index('idx_modification_sessions_agent_status', 'modification_sessions', ['agent_id', 'status'])
    op.create_index('idx_modification_sessions_safety_level', 'modification_sessions', ['safety_level', 'started_at'])
    op.create_index('idx_code_modifications_session_type', 'code_modifications', ['session_id', 'modification_type'])
    op.create_index('idx_code_modifications_safety_score', 'code_modifications', ['safety_score', 'applied_at'])
    op.create_index('idx_code_modifications_git_commit', 'code_modifications', ['git_commit_hash'])
    op.create_index('idx_modification_metrics_category_name', 'modification_metrics', ['metric_category', 'metric_name'])
    op.create_index('idx_modification_metrics_improvement', 'modification_metrics', ['improvement_percentage', 'measured_at'])
    op.create_index('idx_sandbox_executions_type_exit_code', 'sandbox_executions', ['execution_type', 'exit_code'])
    op.create_index('idx_sandbox_executions_security_violations', 'sandbox_executions', ['modification_id'], 
                    postgresql_where=sa.text("jsonb_array_length(security_violations) > 0"))
    op.create_index('idx_sandbox_executions_performance', 'sandbox_executions', ['execution_time_ms', 'memory_usage_mb'])
    op.create_index('idx_modification_feedback_source_type', 'modification_feedback', ['feedback_source', 'feedback_type'])
    op.create_index('idx_modification_feedback_rating', 'modification_feedback', ['rating', 'created_at'])
    
    # Create partial indexes for common queries
    op.create_index('idx_active_sessions', 'modification_sessions', ['agent_id', 'started_at'], 
                    postgresql_where=sa.text("status IN ('analyzing', 'suggestions_ready', 'applying')"))
    op.create_index('idx_applied_modifications', 'code_modifications', ['applied_at', 'performance_impact'], 
                    postgresql_where=sa.text("applied_at IS NOT NULL"))
    op.create_index('idx_failed_executions', 'sandbox_executions', ['modification_id', 'executed_at'], 
                    postgresql_where=sa.text("exit_code != 0 OR jsonb_array_length(security_violations) > 0"))


def downgrade() -> None:
    """Downgrade database schema - remove Self-Modification Engine tables."""
    
    # Drop partial indexes
    op.drop_index('idx_failed_executions')
    op.drop_index('idx_applied_modifications')
    op.drop_index('idx_active_sessions')
    
    # Drop regular indexes
    op.drop_index('idx_modification_feedback_rating')
    op.drop_index('idx_modification_feedback_source_type')
    op.drop_index('idx_sandbox_executions_performance')
    op.drop_index('idx_sandbox_executions_security_violations')
    op.drop_index('idx_sandbox_executions_type_exit_code')
    op.drop_index('idx_modification_metrics_improvement')
    op.drop_index('idx_modification_metrics_category_name')
    op.drop_index('idx_code_modifications_git_commit')
    op.drop_index('idx_code_modifications_safety_score')
    op.drop_index('idx_code_modifications_session_type')
    op.drop_index('idx_modification_sessions_safety_level')
    op.drop_index('idx_modification_sessions_agent_status')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('modification_feedback')
    op.drop_table('sandbox_executions')
    op.drop_table('modification_metrics')
    op.drop_table('code_modifications')
    op.drop_table('modification_sessions')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS sandbox_execution_type')
    op.execute('DROP TYPE IF EXISTS modification_type')
    op.execute('DROP TYPE IF EXISTS modification_status')
    op.execute('DROP TYPE IF EXISTS modification_safety')