"""VS 7.2 Automated Scheduler Database Schema

Revision ID: 018_vs7_2_automated_scheduler
Revises: 017_context_compression_sharing
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

Creates database schema for VS 7.2 Automated Scheduler system components:
- Smart Scheduler configuration and state
- Automation Engine task management and history
- Feature Flag definitions and rollout tracking
- Load Prediction model storage and accuracy metrics
- Performance monitoring and alert management
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text
import uuid

# revision identifiers
revision = '018_vs7_2_automated_scheduler'
down_revision = '017_context_compression_sharing'
branch_labels = None
depends_on = None


def upgrade():
    """Create VS 7.2 Automated Scheduler database schema."""
    
    # Create enum types for VS 7.2 components
    automation_tier_enum = postgresql.ENUM(
        'IMMEDIATE', 'SCHEDULED', 'PREDICTIVE',
        name='automation_tier_enum'
    )
    automation_tier_enum.create(op.get_bind())
    
    scheduling_decision_enum = postgresql.ENUM(
        'CONSOLIDATE_AGENT', 'WAKE_AGENT', 'MAINTAIN_STATUS', 'DEFER_DECISION',
        name='scheduling_decision_enum'
    )
    scheduling_decision_enum.create(op.get_bind())
    
    safety_level_enum = postgresql.ENUM(
        'SAFE', 'CAUTIOUS', 'RESTRICTED', 'EMERGENCY_STOP',
        name='safety_level_enum'
    )
    safety_level_enum.create(op.get_bind())
    
    execution_mode_enum = postgresql.ENUM(
        'SHADOW', 'LIVE', 'VALIDATION',
        name='execution_mode_enum'
    )
    execution_mode_enum.create(op.get_bind())
    
    automation_status_enum = postgresql.ENUM(
        'IDLE', 'RUNNING', 'PAUSED', 'EMERGENCY_STOP',
        name='automation_status_enum'
    )
    automation_status_enum.create(op.get_bind())
    
    task_type_enum = postgresql.ENUM(
        'CONSOLIDATION', 'WAKE', 'HEALTH_CHECK', 'ROLLBACK',
        name='task_type_enum'
    )
    task_type_enum.create(op.get_bind())
    
    task_priority_enum = postgresql.ENUM(
        'EMERGENCY', 'HIGH', 'NORMAL', 'LOW',
        name='task_priority_enum'
    )
    task_priority_enum.create(op.get_bind())
    
    rollout_stage_enum = postgresql.ENUM(
        'DISABLED', 'CANARY_1PCT', 'CANARY_10PCT', 'PARTIAL_25PCT', 
        'PARTIAL_50PCT', 'FULL_100PCT', 'ROLLBACK',
        name='rollout_stage_enum'
    )
    rollout_stage_enum.create(op.get_bind())
    
    feature_type_enum = postgresql.ENUM(
        'AUTOMATION', 'SCHEDULING', 'PREDICTION', 'COORDINATION', 'SAFETY',
        name='feature_type_enum'
    )
    feature_type_enum.create(op.get_bind())
    
    rollback_trigger_enum = postgresql.ENUM(
        'ERROR_RATE', 'LATENCY', 'THROUGHPUT', 'MANUAL', 'CIRCUIT_BREAKER', 'HEALTH_CHECK',
        name='rollback_trigger_enum'
    )
    rollback_trigger_enum.create(op.get_bind())
    
    model_type_enum = postgresql.ENUM(
        'SIMPLE_MOVING_AVERAGE', 'EXPONENTIAL_SMOOTHING', 'LINEAR_REGRESSION', 
        'ARIMA', 'SEASONAL_DECOMPOSITION', 'ENSEMBLE',
        name='model_type_enum'
    )
    model_type_enum.create(op.get_bind())
    
    seasonal_pattern_enum = postgresql.ENUM(
        'HOURLY', 'DAILY', 'WEEKLY', 'NONE',
        name='seasonal_pattern_enum'
    )
    seasonal_pattern_enum.create(op.get_bind())
    
    alert_severity_enum = postgresql.ENUM(
        'INFO', 'WARNING', 'CRITICAL', 'EMERGENCY',
        name='alert_severity_enum'
    )
    alert_severity_enum.create(op.get_bind())
    
    alert_type_enum = postgresql.ENUM(
        'PERFORMANCE_DEGRADATION', 'EFFICIENCY_TARGET_MISS', 'OVERHEAD_THRESHOLD_EXCEEDED',
        'SAFETY_VIOLATION', 'SYSTEM_ERROR', 'FEATURE_ROLLBACK', 
        'AUTOMATION_FAILURE', 'PREDICTION_ACCURACY_DROP',
        name='alert_type_enum'
    )
    alert_type_enum.create(op.get_bind())
    
    # Smart Scheduler Configuration and State
    op.create_table(
        'smart_scheduler_config',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('enabled', sa.Boolean, nullable=False, default=False),
        sa.Column('shadow_mode', sa.Boolean, nullable=False, default=True),
        sa.Column('safety_level', safety_level_enum, nullable=False, default='SAFE'),
        sa.Column('prediction_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('max_simultaneous_consolidations_pct', sa.Float, nullable=False, default=30.0),
        sa.Column('min_agents_awake', sa.Integer, nullable=False, default=2),
        sa.Column('consolidation_cooldown_minutes', sa.Integer, nullable=False, default=10),
        sa.Column('hysteresis_threshold', sa.Float, nullable=False, default=0.15),
        sa.Column('decision_time_target_ms', sa.Integer, nullable=False, default=100),
        sa.Column('system_overhead_target_pct', sa.Float, nullable=False, default=1.0),
        sa.Column('configuration_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Scheduling Decisions History
    op.create_table(
        'scheduling_decisions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('decision', scheduling_decision_enum, nullable=False),
        sa.Column('automation_tier', automation_tier_enum, nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('reasoning', sa.Text, nullable=False),
        sa.Column('estimated_benefit', sa.Float, nullable=False),
        sa.Column('safety_checks_passed', sa.Boolean, nullable=False),
        sa.Column('decision_time_ms', sa.Float, nullable=False),
        sa.Column('executed', sa.Boolean, nullable=False, default=False),
        sa.Column('execution_success', sa.Boolean, nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('decision_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE')
    )
    
    # Automation Engine Configuration and State
    op.create_table(
        'automation_engine_config',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('execution_mode', execution_mode_enum, nullable=False, default='SHADOW'),
        sa.Column('status', automation_status_enum, nullable=False, default='IDLE'),
        sa.Column('enabled', sa.Boolean, nullable=False, default=False),
        sa.Column('max_concurrent_tasks', sa.Integer, nullable=False, default=5),
        sa.Column('max_consolidations_per_minute', sa.Integer, nullable=False, default=10),
        sa.Column('emergency_stop_error_threshold', sa.Float, nullable=False, default=0.2),
        sa.Column('rollback_latency_threshold_ms', sa.Float, nullable=False, default=5000),
        sa.Column('coordination_lock_timeout_seconds', sa.Integer, nullable=False, default=300),
        sa.Column('task_distribution_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('leader_election_enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('configuration_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Automation Tasks
    op.create_table(
        'automation_tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('task_type', task_type_enum, nullable=False),
        sa.Column('priority', task_priority_enum, nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('attempts', sa.Integer, nullable=False, default=0),
        sa.Column('max_attempts', sa.Integer, nullable=False, default=3),
        sa.Column('timeout_seconds', sa.Integer, nullable=False, default=300),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_time_ms', sa.Float, nullable=True),
        sa.Column('success', sa.Boolean, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('rollback_required', sa.Boolean, nullable=False, default=False),
        sa.Column('task_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE')
    )
    
    # Feature Flags
    op.create_table(
        'feature_flags',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('feature_type', feature_type_enum, nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('rollout_stage', rollout_stage_enum, nullable=False, default='DISABLED'),
        sa.Column('target_percentage', sa.Float, nullable=False, default=0.0),
        sa.Column('enabled', sa.Boolean, nullable=False, default=True),
        sa.Column('validation_period_hours', sa.Integer, nullable=False, default=24),
        sa.Column('min_sample_size', sa.Integer, nullable=False, default=100),
        sa.Column('error_rate_threshold', sa.Float, nullable=False, default=0.05),
        sa.Column('latency_threshold_ms', sa.Float, nullable=False, default=2000),
        sa.Column('throughput_threshold_pct', sa.Float, nullable=False, default=0.9),
        sa.Column('feature_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Feature Flag Rollout History
    op.create_table(
        'feature_rollout_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('feature_flag_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_stage', rollout_stage_enum, nullable=False),
        sa.Column('to_stage', rollout_stage_enum, nullable=False),
        sa.Column('from_percentage', sa.Float, nullable=False),
        sa.Column('to_percentage', sa.Float, nullable=False),
        sa.Column('trigger_type', sa.String(50), nullable=False),
        sa.Column('trigger_reason', sa.Text, nullable=False),
        sa.Column('rollback_trigger', rollback_trigger_enum, nullable=True),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('automated', sa.Boolean, nullable=False, default=False),
        sa.Column('rollout_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['feature_flag_id'], ['feature_flags.id'], ondelete='CASCADE')
    )
    
    # Feature Flag Performance Metrics
    op.create_table(
        'feature_performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('feature_flag_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('measurement_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('requests_total', sa.Integer, nullable=False),
        sa.Column('requests_success', sa.Integer, nullable=False),
        sa.Column('requests_error', sa.Integer, nullable=False),
        sa.Column('avg_latency_ms', sa.Float, nullable=False),
        sa.Column('p95_latency_ms', sa.Float, nullable=False),
        sa.Column('p99_latency_ms', sa.Float, nullable=False),
        sa.Column('throughput_per_minute', sa.Float, nullable=False),
        sa.Column('error_rate', sa.Float, nullable=False),
        sa.Column('performance_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['feature_flag_id'], ['feature_flags.id'], ondelete='CASCADE')
    )
    
    # Load Prediction Models
    op.create_table(
        'load_prediction_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('model_type', model_type_enum, nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False, default='1.0'),
        sa.Column('model_data', postgresql.JSONB, nullable=False),  # Serialized model parameters
        sa.Column('feature_names', postgresql.ARRAY(sa.String), nullable=False),
        sa.Column('training_data_size', sa.Integer, nullable=False),
        sa.Column('training_period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('training_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('accuracy_score', sa.Float, nullable=False),
        sa.Column('mean_absolute_error', sa.Float, nullable=False),
        sa.Column('mean_squared_error', sa.Float, nullable=False),
        sa.Column('active', sa.Boolean, nullable=False, default=True),
        sa.Column('model_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Load Prediction Results
    op.create_table(
        'load_prediction_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('prediction_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('horizon_minutes', sa.Integer, nullable=False),
        sa.Column('predicted_load', postgresql.JSONB, nullable=False),
        sa.Column('confidence_interval', postgresql.JSONB, nullable=True),
        sa.Column('confidence_score', sa.Float, nullable=False),
        sa.Column('seasonal_pattern', seasonal_pattern_enum, nullable=True),
        sa.Column('trend_direction', sa.String(50), nullable=True),
        sa.Column('actual_load', postgresql.JSONB, nullable=True),  # Filled in when actual data available
        sa.Column('prediction_accuracy', sa.Float, nullable=True),  # Calculated when actual data available
        sa.Column('prediction_error', sa.Float, nullable=True),
        sa.Column('prediction_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['model_id'], ['load_prediction_models.id'], ondelete='CASCADE')
    )
    
    # Load Data Points (for training and validation)
    op.create_table(
        'load_data_points',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cpu_utilization', sa.Float, nullable=False),
        sa.Column('memory_utilization', sa.Float, nullable=False),
        sa.Column('active_agents', sa.Integer, nullable=False),
        sa.Column('pending_tasks', sa.Integer, nullable=False),
        sa.Column('message_queue_depth', sa.Integer, nullable=False),
        sa.Column('response_time_p95', sa.Float, nullable=False),
        sa.Column('error_rate', sa.Float, nullable=False),
        sa.Column('throughput_rps', sa.Float, nullable=False),
        sa.Column('consolidation_effectiveness', sa.Float, nullable=True),
        sa.Column('data_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Performance Monitoring Alerts
    op.create_table(
        'vs7_2_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('alert_type', alert_type_enum, nullable=False),
        sa.Column('severity', alert_severity_enum, nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('component', sa.String(255), nullable=False),
        sa.Column('resolved', sa.Boolean, nullable=False, default=False),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_reason', sa.Text, nullable=True),
        sa.Column('alert_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Performance Metrics Storage
    op.create_table(
        'vs7_2_performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('component', sa.String(255), nullable=False),
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('value', sa.Float, nullable=False),
        sa.Column('labels', postgresql.JSONB, nullable=True),
        sa.Column('measurement_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Efficiency Measurements
    op.create_table(
        'efficiency_measurements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('measurement_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('baseline_period_hours', sa.Integer, nullable=False),
        sa.Column('current_period_hours', sa.Integer, nullable=False),
        sa.Column('baseline_metrics', postgresql.JSONB, nullable=False),
        sa.Column('current_metrics', postgresql.JSONB, nullable=False),
        sa.Column('efficiency_improvement_pct', sa.Float, nullable=False),
        sa.Column('meets_target', sa.Boolean, nullable=False),
        sa.Column('target_pct', sa.Float, nullable=False, default=70.0),
        sa.Column('measurement_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # System Overhead Measurements
    op.create_table(
        'system_overhead_measurements',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('measurement_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cpu_overhead_pct', sa.Float, nullable=False),
        sa.Column('memory_overhead_pct', sa.Float, nullable=False),
        sa.Column('latency_overhead_ms', sa.Float, nullable=False),
        sa.Column('total_overhead_pct', sa.Float, nullable=False),
        sa.Column('meets_target', sa.Boolean, nullable=False),
        sa.Column('target_pct', sa.Float, nullable=False, default=1.0),
        sa.Column('component_breakdown', postgresql.JSONB, nullable=True),
        sa.Column('measurement_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()'))
    )
    
    # Create indexes for performance
    op.create_index('idx_scheduling_decisions_agent_id', 'scheduling_decisions', ['agent_id'])
    op.create_index('idx_scheduling_decisions_created_at', 'scheduling_decisions', ['created_at'])
    op.create_index('idx_scheduling_decisions_decision', 'scheduling_decisions', ['decision'])
    
    op.create_index('idx_automation_tasks_agent_id', 'automation_tasks', ['agent_id'])
    op.create_index('idx_automation_tasks_status', 'automation_tasks', ['status'])
    op.create_index('idx_automation_tasks_priority', 'automation_tasks', ['priority'])
    op.create_index('idx_automation_tasks_created_at', 'automation_tasks', ['created_at'])
    
    op.create_index('idx_feature_flags_name', 'feature_flags', ['name'])
    op.create_index('idx_feature_flags_rollout_stage', 'feature_flags', ['rollout_stage'])
    op.create_index('idx_feature_flags_enabled', 'feature_flags', ['enabled'])
    
    op.create_index('idx_feature_rollout_history_flag_id', 'feature_rollout_history', ['feature_flag_id'])
    op.create_index('idx_feature_rollout_history_created_at', 'feature_rollout_history', ['created_at'])
    
    op.create_index('idx_feature_performance_metrics_flag_id', 'feature_performance_metrics', ['feature_flag_id'])
    op.create_index('idx_feature_performance_metrics_timestamp', 'feature_performance_metrics', ['measurement_timestamp'])
    
    op.create_index('idx_load_prediction_models_type', 'load_prediction_models', ['model_type'])
    op.create_index('idx_load_prediction_models_active', 'load_prediction_models', ['active'])
    
    op.create_index('idx_load_prediction_results_model_id', 'load_prediction_results', ['model_id'])
    op.create_index('idx_load_prediction_results_timestamp', 'load_prediction_results', ['prediction_timestamp'])
    
    op.create_index('idx_load_data_points_timestamp', 'load_data_points', ['timestamp'])
    
    op.create_index('idx_vs7_2_alerts_type', 'vs7_2_alerts', ['alert_type'])
    op.create_index('idx_vs7_2_alerts_severity', 'vs7_2_alerts', ['severity'])
    op.create_index('idx_vs7_2_alerts_resolved', 'vs7_2_alerts', ['resolved'])
    op.create_index('idx_vs7_2_alerts_created_at', 'vs7_2_alerts', ['created_at'])
    
    op.create_index('idx_vs7_2_performance_metrics_component', 'vs7_2_performance_metrics', ['component'])
    op.create_index('idx_vs7_2_performance_metrics_name', 'vs7_2_performance_metrics', ['metric_name'])
    op.create_index('idx_vs7_2_performance_metrics_timestamp', 'vs7_2_performance_metrics', ['measurement_timestamp'])
    
    op.create_index('idx_efficiency_measurements_timestamp', 'efficiency_measurements', ['measurement_timestamp'])
    op.create_index('idx_efficiency_measurements_target', 'efficiency_measurements', ['meets_target'])
    
    op.create_index('idx_system_overhead_measurements_timestamp', 'system_overhead_measurements', ['measurement_timestamp'])
    op.create_index('idx_system_overhead_measurements_target', 'system_overhead_measurements', ['meets_target'])
    
    # Add constraint to ensure only one active configuration per component
    op.execute(text("""
        CREATE UNIQUE INDEX idx_smart_scheduler_config_single_active 
        ON smart_scheduler_config ((true)) 
        WHERE enabled = true
    """))
    
    op.execute(text("""
        CREATE UNIQUE INDEX idx_automation_engine_config_single_active 
        ON automation_engine_config ((true)) 
        WHERE enabled = true
    """))
    
    # Add triggers for updated_at columns
    op.execute(text("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """))
    
    op.execute(text("""
        CREATE TRIGGER update_smart_scheduler_config_updated_at 
        BEFORE UPDATE ON smart_scheduler_config 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """))
    
    op.execute(text("""
        CREATE TRIGGER update_automation_engine_config_updated_at 
        BEFORE UPDATE ON automation_engine_config 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """))
    
    op.execute(text("""
        CREATE TRIGGER update_feature_flags_updated_at 
        BEFORE UPDATE ON feature_flags 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """))
    
    op.execute(text("""
        CREATE TRIGGER update_load_prediction_models_updated_at 
        BEFORE UPDATE ON load_prediction_models 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    """))
    
    # Create partitioning for high-volume tables
    op.execute(text("""
        -- Partition performance metrics by month
        CREATE TABLE vs7_2_performance_metrics_y2025m01 
        PARTITION OF vs7_2_performance_metrics 
        FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
        
        CREATE TABLE vs7_2_performance_metrics_y2025m02 
        PARTITION OF vs7_2_performance_metrics 
        FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
        
        -- Partition load data points by month  
        CREATE TABLE load_data_points_y2025m01 
        PARTITION OF load_data_points 
        FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
        
        CREATE TABLE load_data_points_y2025m02 
        PARTITION OF load_data_points 
        FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
    """))
    
    # Insert default configurations
    op.execute(text("""
        INSERT INTO smart_scheduler_config (
            id, enabled, shadow_mode, safety_level, prediction_enabled,
            max_simultaneous_consolidations_pct, min_agents_awake,
            consolidation_cooldown_minutes, hysteresis_threshold,
            decision_time_target_ms, system_overhead_target_pct,
            configuration_metadata
        ) VALUES (
            gen_random_uuid(), false, true, 'SAFE', true,
            30.0, 2, 10, 0.15, 100, 1.0,
            '{"created_by": "migration", "version": "7.2", "description": "Default VS 7.2 scheduler configuration"}'::jsonb
        );
    """))
    
    op.execute(text("""
        INSERT INTO automation_engine_config (
            id, execution_mode, status, enabled, max_concurrent_tasks,
            max_consolidations_per_minute, emergency_stop_error_threshold,
            rollback_latency_threshold_ms, coordination_lock_timeout_seconds,
            task_distribution_enabled, leader_election_enabled,
            configuration_metadata
        ) VALUES (
            gen_random_uuid(), 'SHADOW', 'IDLE', false, 5,
            10, 0.2, 5000, 300, true, true,
            '{"created_by": "migration", "version": "7.2", "description": "Default VS 7.2 automation engine configuration"}'::jsonb
        );
    """))


def downgrade():
    """Remove VS 7.2 Automated Scheduler database schema."""
    
    # Drop partitioned tables
    op.execute(text("DROP TABLE IF EXISTS vs7_2_performance_metrics_y2025m01"))
    op.execute(text("DROP TABLE IF EXISTS vs7_2_performance_metrics_y2025m02"))
    op.execute(text("DROP TABLE IF EXISTS load_data_points_y2025m01"))
    op.execute(text("DROP TABLE IF EXISTS load_data_points_y2025m02"))
    
    # Drop triggers
    op.execute(text("DROP TRIGGER IF EXISTS update_smart_scheduler_config_updated_at ON smart_scheduler_config"))
    op.execute(text("DROP TRIGGER IF EXISTS update_automation_engine_config_updated_at ON automation_engine_config"))
    op.execute(text("DROP TRIGGER IF EXISTS update_feature_flags_updated_at ON feature_flags"))
    op.execute(text("DROP TRIGGER IF EXISTS update_load_prediction_models_updated_at ON load_prediction_models"))
    
    # Drop function
    op.execute(text("DROP FUNCTION IF EXISTS update_updated_at_column()"))
    
    # Drop tables in reverse dependency order
    op.drop_table('system_overhead_measurements')
    op.drop_table('efficiency_measurements')
    op.drop_table('vs7_2_performance_metrics')
    op.drop_table('vs7_2_alerts')
    op.drop_table('load_data_points')
    op.drop_table('load_prediction_results')
    op.drop_table('load_prediction_models')
    op.drop_table('feature_performance_metrics')
    op.drop_table('feature_rollout_history')
    op.drop_table('feature_flags')
    op.drop_table('automation_tasks')
    op.drop_table('automation_engine_config')
    op.drop_table('scheduling_decisions')
    op.drop_table('smart_scheduler_config')
    
    # Drop enum types
    op.execute(text("DROP TYPE IF EXISTS alert_type_enum"))
    op.execute(text("DROP TYPE IF EXISTS alert_severity_enum"))
    op.execute(text("DROP TYPE IF EXISTS seasonal_pattern_enum"))
    op.execute(text("DROP TYPE IF EXISTS model_type_enum"))
    op.execute(text("DROP TYPE IF EXISTS rollback_trigger_enum"))
    op.execute(text("DROP TYPE IF EXISTS feature_type_enum"))
    op.execute(text("DROP TYPE IF EXISTS rollout_stage_enum"))
    op.execute(text("DROP TYPE IF EXISTS task_priority_enum"))
    op.execute(text("DROP TYPE IF EXISTS task_type_enum"))
    op.execute(text("DROP TYPE IF EXISTS automation_status_enum"))
    op.execute(text("DROP TYPE IF EXISTS execution_mode_enum"))
    op.execute(text("DROP TYPE IF EXISTS safety_level_enum"))
    op.execute(text("DROP TYPE IF EXISTS scheduling_decision_enum"))
    op.execute(text("DROP TYPE IF EXISTS automation_tier_enum"))