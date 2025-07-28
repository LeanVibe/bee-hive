"""Add Prompt Optimization System tables

Revision ID: 013
Revises: 012
Create Date: 2024-01-28 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '013'
down_revision = '012'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create custom enums for prompt optimization
    op.execute("""
        CREATE TYPE prompt_status AS ENUM (
            'draft', 'active', 'archived', 'deprecated'
        );
    """)
    
    op.execute("""
        CREATE TYPE experiment_status AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled'
        );
    """)
    
    op.execute("""
        CREATE TYPE optimization_method AS ENUM (
            'meta_prompting', 'evolutionary', 'gradient_based', 'few_shot', 'manual'
        );
    """)
    
    # Create prompt_templates table
    op.create_table(
        'prompt_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('task_type', sa.String(100), nullable=True, index=True),
        sa.Column('domain', sa.String(100), nullable=True, index=True),
        sa.Column('template_content', sa.Text, nullable=False),
        sa.Column('template_variables', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('version', sa.Integer, nullable=False, server_default='1'),
        sa.Column('status', sa.Enum('draft', 'active', 'archived', 'deprecated', name='prompt_status'), nullable=False, server_default='draft'),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create optimization_experiments table
    op.create_table(
        'optimization_experiments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('experiment_name', sa.String(255), nullable=False, index=True),
        sa.Column('base_prompt_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('optimization_method', sa.Enum('meta_prompting', 'evolutionary', 'gradient_based', 'few_shot', 'manual', name='optimization_method'), nullable=False),
        sa.Column('target_metrics', postgresql.JSON, nullable=False, server_default='{}'),
        sa.Column('experiment_config', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled', name='experiment_status'), nullable=False, server_default='pending'),
        sa.Column('progress_percentage', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('current_iteration', sa.Integer, nullable=False, server_default='0'),
        sa.Column('max_iterations', sa.Integer, nullable=False, server_default='50'),
        sa.Column('best_score', sa.Float, nullable=True),
        sa.Column('baseline_score', sa.Float, nullable=True),
        sa.Column('improvement_percentage', sa.Float, nullable=True),
        sa.Column('convergence_threshold', sa.Float, nullable=False, server_default='0.01'),
        sa.Column('early_stopping', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['base_prompt_id'], ['prompt_templates.id'], ondelete='CASCADE'),
    )
    
    # Create prompt_variants table
    op.create_table(
        'prompt_variants',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('parent_prompt_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('variant_content', sa.Text, nullable=False),
        sa.Column('generation_method', sa.String(100), nullable=True),
        sa.Column('generation_reasoning', sa.Text, nullable=True),
        sa.Column('confidence_score', sa.Float, nullable=True),
        sa.Column('iteration', sa.Integer, nullable=False, server_default='0'),
        sa.Column('generation_time_seconds', sa.Float, nullable=True),
        sa.Column('token_count', sa.Integer, nullable=True),
        sa.Column('complexity_score', sa.Float, nullable=True),
        sa.Column('readability_score', sa.Float, nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('parameters', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('ancestry', postgresql.JSON, nullable=True, server_default='[]'),  # For tracking evolutionary history
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['experiment_id'], ['optimization_experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_prompt_id'], ['prompt_templates.id'], ondelete='CASCADE'),
    )
    
    # Create prompt_evaluations table
    op.create_table(
        'prompt_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('prompt_variant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('test_case_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('metric_name', sa.String(100), nullable=False, index=True),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('raw_output', sa.Text, nullable=True),
        sa.Column('expected_output', sa.Text, nullable=True),
        sa.Column('evaluation_context', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('evaluation_method', sa.String(100), nullable=True),
        sa.Column('evaluation_time_seconds', sa.Float, nullable=True),
        sa.Column('token_usage', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('cost_estimate', sa.Float, nullable=True),
        sa.Column('error_details', sa.Text, nullable=True),
        sa.Column('evaluated_by', sa.String(255), nullable=True),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['prompt_variant_id'], ['prompt_variants.id'], ondelete='CASCADE'),
    )
    
    # Create ab_test_results table
    op.create_table(
        'ab_test_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('test_name', sa.String(255), nullable=False, index=True),
        sa.Column('prompt_a_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prompt_b_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sample_size', sa.Integer, nullable=False),
        sa.Column('significance_level', sa.Float, nullable=False, server_default='0.05'),
        sa.Column('p_value', sa.Float, nullable=True),
        sa.Column('effect_size', sa.Float, nullable=True),
        sa.Column('confidence_interval_lower', sa.Float, nullable=True),
        sa.Column('confidence_interval_upper', sa.Float, nullable=True),
        sa.Column('winner_variant_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('test_power', sa.Float, nullable=True),
        sa.Column('mean_a', sa.Float, nullable=True),
        sa.Column('mean_b', sa.Float, nullable=True),
        sa.Column('std_a', sa.Float, nullable=True),
        sa.Column('std_b', sa.Float, nullable=True),
        sa.Column('test_statistic', sa.Float, nullable=True),
        sa.Column('degrees_of_freedom', sa.Integer, nullable=True),
        sa.Column('statistical_notes', sa.Text, nullable=True),
        sa.Column('test_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['experiment_id'], ['optimization_experiments.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['prompt_a_id'], ['prompt_variants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['prompt_b_id'], ['prompt_variants.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['winner_variant_id'], ['prompt_variants.id'], ondelete='SET NULL'),
    )
    
    # Create prompt_feedback table
    op.create_table(
        'prompt_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('prompt_variant_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('user_id', sa.String(255), nullable=True, index=True),
        sa.Column('session_id', sa.String(255), nullable=True, index=True),
        sa.Column('rating', sa.Integer, nullable=False),
        sa.Column('feedback_text', sa.Text, nullable=True),
        sa.Column('feedback_categories', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('context_data', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('response_quality_score', sa.Float, nullable=True),
        sa.Column('relevance_score', sa.Float, nullable=True),
        sa.Column('clarity_score', sa.Float, nullable=True),
        sa.Column('usefulness_score', sa.Float, nullable=True),
        sa.Column('sentiment_score', sa.Float, nullable=True),
        sa.Column('feedback_weight', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('is_validated', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('validation_notes', sa.Text, nullable=True),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['prompt_variant_id'], ['prompt_variants.id'], ondelete='CASCADE'),
    )
    
    # Create test_cases table for systematic evaluation
    op.create_table(
        'prompt_test_cases',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('domain', sa.String(100), nullable=True, index=True),
        sa.Column('task_type', sa.String(100), nullable=True, index=True),
        sa.Column('input_data', postgresql.JSON, nullable=False),
        sa.Column('expected_output', sa.Text, nullable=True),
        sa.Column('evaluation_criteria', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('difficulty_level', sa.String(50), nullable=True),
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    
    # Create optimization_metrics table for tracking system performance
    op.create_table(
        'optimization_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('experiment_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('metric_name', sa.String(100), nullable=False, index=True),
        sa.Column('metric_value', sa.Float, nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=True), # 'system', 'experiment', 'global'
        sa.Column('aggregation_period', sa.String(50), nullable=True), # 'hour', 'day', 'week'
        sa.Column('tags', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('additional_data', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['optimization_experiments.id'], ondelete='CASCADE'),
    )
    
    # Create performance-optimized indexes
    op.create_index('idx_prompt_templates_domain_task', 'prompt_templates', ['domain', 'task_type'])
    op.create_index('idx_prompt_templates_status_version', 'prompt_templates', ['status', 'version'])
    op.create_index('idx_optimization_experiments_status_method', 'optimization_experiments', ['status', 'optimization_method'])
    op.create_index('idx_optimization_experiments_created_at', 'optimization_experiments', ['created_at'])
    op.create_index('idx_prompt_variants_experiment_iteration', 'prompt_variants', ['experiment_id', 'iteration'])
    op.create_index('idx_prompt_variants_confidence_score', 'prompt_variants', ['confidence_score'])
    op.create_index('idx_prompt_variants_embedding_cosine', 'prompt_variants', ['embedding'], postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_index('idx_prompt_evaluations_metric_value', 'prompt_evaluations', ['metric_name', 'metric_value'])
    op.create_index('idx_prompt_evaluations_evaluated_at', 'prompt_evaluations', ['evaluated_at'])
    op.create_index('idx_ab_test_results_p_value', 'ab_test_results', ['p_value'])
    op.create_index('idx_ab_test_results_effect_size', 'ab_test_results', ['effect_size'])
    op.create_index('idx_prompt_feedback_rating', 'prompt_feedback', ['rating'])
    op.create_index('idx_prompt_feedback_submitted_at', 'prompt_feedback', ['submitted_at'])
    op.create_index('idx_prompt_test_cases_domain_task', 'prompt_test_cases', ['domain', 'task_type'])
    op.create_index('idx_optimization_metrics_recorded_at', 'optimization_metrics', ['recorded_at'])
    op.create_index('idx_optimization_metrics_name_type', 'optimization_metrics', ['metric_name', 'metric_type'])


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop indexes
    op.drop_index('idx_optimization_metrics_name_type')
    op.drop_index('idx_optimization_metrics_recorded_at')
    op.drop_index('idx_prompt_test_cases_domain_task')
    op.drop_index('idx_prompt_feedback_submitted_at')
    op.drop_index('idx_prompt_feedback_rating')
    op.drop_index('idx_ab_test_results_effect_size')
    op.drop_index('idx_ab_test_results_p_value')
    op.drop_index('idx_prompt_evaluations_evaluated_at')
    op.drop_index('idx_prompt_evaluations_metric_value')
    op.drop_index('idx_prompt_variants_embedding_cosine')
    op.drop_index('idx_prompt_variants_confidence_score')
    op.drop_index('idx_prompt_variants_experiment_iteration')
    op.drop_index('idx_optimization_experiments_created_at')
    op.drop_index('idx_optimization_experiments_status_method')
    op.drop_index('idx_prompt_templates_status_version')
    op.drop_index('idx_prompt_templates_domain_task')
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('optimization_metrics')
    op.drop_table('prompt_test_cases')
    op.drop_table('prompt_feedback')
    op.drop_table('ab_test_results')
    op.drop_table('prompt_evaluations')
    op.drop_table('prompt_variants')
    op.drop_table('optimization_experiments')
    op.drop_table('prompt_templates')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS optimization_method')
    op.execute('DROP TYPE IF EXISTS experiment_status')
    op.execute('DROP TYPE IF EXISTS prompt_status')