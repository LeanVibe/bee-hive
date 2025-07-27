"""Add Sleep-Wake Manager tables for autonomous consolidation and recovery

Revision ID: 006
Revises: 005
Create Date: 2024-01-01 20:00:00.000000

Adds schema for:
- Sleep windows configuration for agent scheduling
- Checkpoints for atomic state preservation and recovery
- Sleep-wake cycle tracking and analytics
- Consolidation metadata and performance metrics
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add Sleep-Wake Manager schema."""
    
    # Create enums for sleep-wake manager
    op.execute("""
        CREATE TYPE sleep_state_enum AS ENUM (
            'AWAKE',
            'PREPARING_SLEEP',
            'SLEEPING',
            'CONSOLIDATING',
            'PREPARING_WAKE',
            'ERROR'
        )
    """)
    
    op.execute("""
        CREATE TYPE checkpoint_type_enum AS ENUM (
            'SCHEDULED',
            'PRE_SLEEP',
            'ERROR_RECOVERY',
            'MANUAL',
            'EMERGENCY'
        )
    """)
    
    op.execute("""
        CREATE TYPE consolidation_status_enum AS ENUM (
            'PENDING',
            'IN_PROGRESS',
            'COMPLETED',
            'FAILED',
            'SKIPPED'
        )
    """)
    
    # Sleep windows table for configurable scheduling
    op.create_table(
        'sleep_windows',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),  # NULL for system-wide default
        sa.Column('start_time', sa.Time, nullable=False),
        sa.Column('end_time', sa.Time, nullable=False),
        sa.Column('timezone', sa.String(64), nullable=False, server_default='UTC'),
        sa.Column('active', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('days_of_week', postgresql.JSON, nullable=False, server_default='[1,2,3,4,5,6,7]'),  # 1=Monday, 7=Sunday
        sa.Column('priority', sa.Integer, nullable=False, server_default='0'),  # Higher priority overrides lower
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        # Ensure valid time ranges
        sa.CheckConstraint('start_time != end_time', name='valid_time_window'),
        # Index for efficient schedule queries
        sa.Index('idx_sleep_windows_active_priority', 'active', 'priority'),
        sa.Index('idx_sleep_windows_agent_active', 'agent_id', 'active')
    )
    
    # Enhanced checkpoints table with integrity validation
    op.create_table(
        'checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),  # NULL for system-wide
        sa.Column('checkpoint_type', sa.String(30), nullable=False, index=True),
        sa.Column('path', sa.Text, nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('size_bytes', sa.BigInteger, nullable=False),
        sa.Column('is_valid', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('validation_errors', postgresql.JSON, nullable=True, server_default='[]'),
        sa.Column('checkpoint_metadata', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('redis_offsets', postgresql.JSON, nullable=True, server_default='{}'),  # Stream offsets snapshot
        sa.Column('database_snapshot_id', sa.String(255), nullable=True),  # Git commit or database backup ID
        sa.Column('compression_ratio', sa.Float, nullable=True),
        sa.Column('creation_time_ms', sa.Float, nullable=True),
        sa.Column('validation_time_ms', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),  # For cleanup policies
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        # Index for efficient checkpoint queries
        sa.Index('idx_checkpoints_agent_type_created', 'agent_id', 'checkpoint_type', 'created_at'),
        sa.Index('idx_checkpoints_valid_created', 'is_valid', 'created_at'),
        sa.Index('idx_checkpoints_expires', 'expires_at')
    )
    
    # Enhanced sleep-wake cycles with detailed state tracking (extend existing table)
    # Add new columns to existing sleep_wake_cycles table
    op.add_column('sleep_wake_cycles', sa.Column('expected_wake_time', sa.DateTime(timezone=True), nullable=True))
    op.add_column('sleep_wake_cycles', sa.Column('pre_sleep_checkpoint_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('sleep_wake_cycles', sa.Column('post_wake_checkpoint_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('sleep_wake_cycles', sa.Column('performance_metrics', postgresql.JSON, nullable=True, server_default='{}'))
    op.add_column('sleep_wake_cycles', sa.Column('error_details', postgresql.JSON, nullable=True, server_default='{}'))
    op.add_column('sleep_wake_cycles', sa.Column('token_reduction_achieved', sa.Float, nullable=True))
    op.add_column('sleep_wake_cycles', sa.Column('consolidation_time_ms', sa.Float, nullable=True))
    op.add_column('sleep_wake_cycles', sa.Column('recovery_time_ms', sa.Float, nullable=True))
    
    # Add foreign key constraints for the new columns
    op.create_foreign_key('fk_sleep_wake_cycles_pre_checkpoint', 'sleep_wake_cycles', 'checkpoints', ['pre_sleep_checkpoint_id'], ['id'], ondelete='SET NULL')
    op.create_foreign_key('fk_sleep_wake_cycles_post_checkpoint', 'sleep_wake_cycles', 'checkpoints', ['post_wake_checkpoint_id'], ['id'], ondelete='SET NULL')
    
    # Add indexes for performance
    op.create_index('idx_sleep_wake_cycles_performance', 'sleep_wake_cycles', ['token_reduction_achieved', 'consolidation_time_ms'])
    
    # Consolidation jobs table for tracking background work
    op.create_table(
        'consolidation_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('cycle_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('job_type', sa.String(50), nullable=False, index=True),  # context_compression, vector_update, etc.
        sa.Column('status', sa.String(30), nullable=False, server_default='PENDING', index=True),
        sa.Column('input_data', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('output_data', postgresql.JSON, nullable=True, server_default='{}'),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('progress_percentage', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('processing_time_ms', sa.Float, nullable=True),
        sa.Column('tokens_processed', sa.Integer, nullable=True),
        sa.Column('tokens_saved', sa.Integer, nullable=True),
        sa.Column('priority', sa.Integer, nullable=False, server_default='0'),
        sa.Column('retry_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer, nullable=False, server_default='3'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['cycle_id'], ['sleep_wake_cycles.id'], ondelete='CASCADE'),
        # Index for job processing and monitoring
        sa.Index('idx_consolidation_jobs_status_priority', 'status', 'priority'),
        sa.Index('idx_consolidation_jobs_type_status', 'job_type', 'status'),
        sa.Index('idx_consolidation_jobs_cycle_status', 'cycle_id', 'status')
    )
    
    # Sleep-wake analytics for performance tracking
    op.create_table(
        'sleep_wake_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),  # NULL for system-wide metrics
        sa.Column('date', sa.Date, nullable=False, index=True),
        sa.Column('total_cycles', sa.Integer, nullable=False, server_default='0'),
        sa.Column('successful_cycles', sa.Integer, nullable=False, server_default='0'),
        sa.Column('failed_cycles', sa.Integer, nullable=False, server_default='0'),
        sa.Column('average_token_reduction', sa.Float, nullable=True),
        sa.Column('average_consolidation_time_ms', sa.Float, nullable=True),
        sa.Column('average_recovery_time_ms', sa.Float, nullable=True),
        sa.Column('total_tokens_saved', sa.BigInteger, nullable=False, server_default='0'),
        sa.Column('total_processing_time_ms', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('uptime_percentage', sa.Float, nullable=True),
        sa.Column('checkpoints_created', sa.Integer, nullable=False, server_default='0'),
        sa.Column('checkpoints_validated', sa.Integer, nullable=False, server_default='0'),
        sa.Column('fallback_recoveries', sa.Integer, nullable=False, server_default='0'),
        sa.Column('manual_interventions', sa.Integer, nullable=False, server_default='0'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ondelete='CASCADE'),
        # Unique constraint for daily analytics per agent
        sa.UniqueConstraint('agent_id', 'date', name='unique_daily_analytics'),
        # Index for analytics queries
        sa.Index('idx_sleep_wake_analytics_date', 'date'),
        sa.Index('idx_sleep_wake_analytics_agent_date', 'agent_id', 'date')
    )
    
    # Add sleep state to agents table for current status tracking
    op.add_column('agents', sa.Column(
        'current_sleep_state', 
        sa.String(30),
        nullable=False,
        server_default='AWAKE'
    ))
    
    op.add_column('agents', sa.Column(
        'current_cycle_id',
        postgresql.UUID(as_uuid=True),
        nullable=True
    ))
    
    op.add_column('agents', sa.Column(
        'last_sleep_time',
        sa.DateTime(timezone=True),
        nullable=True
    ))
    
    op.add_column('agents', sa.Column(
        'last_wake_time',
        sa.DateTime(timezone=True),
        nullable=True
    ))
    
    # Add foreign key constraint for current cycle
    op.create_foreign_key(
        'fk_agents_current_cycle',
        'agents', 'sleep_wake_cycles',
        ['current_cycle_id'], ['id'],
        ondelete='SET NULL'
    )
    
    # Create trigger for automatic analytics updates
    op.execute("""
        CREATE OR REPLACE FUNCTION update_sleep_wake_analytics()
        RETURNS TRIGGER AS $$
        DECLARE
            cycle_date DATE;
            agent_uuid UUID;
        BEGIN
            -- Determine which record to update based on trigger
            IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
                cycle_date := DATE(NEW.created_at);
                agent_uuid := NEW.agent_id;
            ELSE
                cycle_date := DATE(OLD.created_at);
                agent_uuid := OLD.agent_id;
            END IF;
            
            -- Update daily analytics for the agent
            INSERT INTO sleep_wake_analytics (
                agent_id,
                date,
                total_cycles,
                successful_cycles,
                failed_cycles,
                updated_at
            )
            VALUES (
                agent_uuid,
                cycle_date,
                1,
                CASE WHEN NEW.sleep_state = 'AWAKE' AND NEW.wake_time IS NOT NULL THEN 1 ELSE 0 END,
                CASE WHEN NEW.sleep_state = 'ERROR' THEN 1 ELSE 0 END,
                NOW()
            )
            ON CONFLICT (agent_id, date)
            DO UPDATE SET
                total_cycles = sleep_wake_analytics.total_cycles + 
                    CASE WHEN TG_OP = 'INSERT' THEN 1 ELSE 0 END,
                successful_cycles = sleep_wake_analytics.successful_cycles + 
                    CASE WHEN NEW.sleep_state = 'AWAKE' AND NEW.wake_time IS NOT NULL THEN 1 ELSE 0 END,
                failed_cycles = sleep_wake_analytics.failed_cycles + 
                    CASE WHEN NEW.sleep_state = 'ERROR' THEN 1 ELSE 0 END,
                average_token_reduction = 
                    CASE WHEN NEW.token_reduction_achieved IS NOT NULL THEN
                        COALESCE(sleep_wake_analytics.average_token_reduction, 0) * 0.8 + NEW.token_reduction_achieved * 0.2
                    ELSE sleep_wake_analytics.average_token_reduction END,
                average_consolidation_time_ms = 
                    CASE WHEN NEW.consolidation_time_ms IS NOT NULL THEN
                        COALESCE(sleep_wake_analytics.average_consolidation_time_ms, 0) * 0.8 + NEW.consolidation_time_ms * 0.2
                    ELSE sleep_wake_analytics.average_consolidation_time_ms END,
                average_recovery_time_ms = 
                    CASE WHEN NEW.recovery_time_ms IS NOT NULL THEN
                        COALESCE(sleep_wake_analytics.average_recovery_time_ms, 0) * 0.8 + NEW.recovery_time_ms * 0.2
                    ELSE sleep_wake_analytics.average_recovery_time_ms END,
                total_tokens_saved = sleep_wake_analytics.total_tokens_saved + 
                    COALESCE((NEW.performance_metrics->>'tokens_saved')::BIGINT, 0),
                updated_at = NOW();
            
            RETURN COALESCE(NEW, OLD);
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER sleep_wake_cycle_analytics_trigger
        AFTER INSERT OR UPDATE ON sleep_wake_cycles
        FOR EACH ROW
        EXECUTE FUNCTION update_sleep_wake_analytics();
    """)
    
    # Create function for checkpoint cleanup based on retention policies
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_checkpoints(
            max_checkpoints_per_agent INTEGER DEFAULT 10,
            max_age_days INTEGER DEFAULT 30
        )
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER := 0;
            temp_count INTEGER;
        BEGIN
            -- Delete checkpoints older than max_age_days
            DELETE FROM checkpoints 
            WHERE created_at < NOW() - INTERVAL '1 day' * max_age_days
            AND is_valid = false;  -- Only delete invalid old checkpoints
            
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;
            
            -- Keep only the latest N valid checkpoints per agent
            WITH ranked_checkpoints AS (
                SELECT id, 
                       ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY created_at DESC) as rn
                FROM checkpoints 
                WHERE is_valid = true
            )
            DELETE FROM checkpoints 
            WHERE id IN (
                SELECT id FROM ranked_checkpoints WHERE rn > max_checkpoints_per_agent
            );
            
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;
            
            -- Clean up expired checkpoints
            DELETE FROM checkpoints 
            WHERE expires_at IS NOT NULL AND expires_at < NOW();
            
            GET DIAGNOSTICS temp_count = ROW_COUNT;
            deleted_count := deleted_count + temp_count;
            
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    """Remove Sleep-Wake Manager schema."""
    
    # Drop triggers and functions
    op.execute('DROP TRIGGER IF EXISTS sleep_wake_cycle_analytics_trigger ON sleep_wake_cycles;')
    op.execute('DROP FUNCTION IF EXISTS update_sleep_wake_analytics();')
    op.execute('DROP FUNCTION IF EXISTS cleanup_old_checkpoints(INTEGER, INTEGER);')
    
    # Drop foreign key constraint from agents table
    op.drop_constraint('fk_agents_current_cycle', 'agents', type_='foreignkey')
    
    # Remove columns from agents table
    op.drop_column('agents', 'last_wake_time')
    op.drop_column('agents', 'last_sleep_time')
    op.drop_column('agents', 'current_cycle_id')
    op.drop_column('agents', 'current_sleep_state')
    
    # Drop tables in reverse order
    op.drop_table('sleep_wake_analytics')
    op.drop_table('consolidation_jobs')
    op.drop_table('sleep_wake_cycles')
    op.drop_table('checkpoints')
    op.drop_table('sleep_windows')
    
    # Drop custom enums
    op.execute('DROP TYPE IF EXISTS consolidation_status_enum')
    op.execute('DROP TYPE IF EXISTS checkpoint_type_enum')
    op.execute('DROP TYPE IF EXISTS sleep_state_enum')