"""Add project management hierarchy

Revision ID: 008_add_project_management_hierarchy
Revises: 007_add_production_indexes
Create Date: 2025-01-20 10:00:00.000000

This migration adds the comprehensive project management hierarchy:
- Projects (top level containers)
- Epics (within projects)
- PRDs (Product Requirements Documents within epics)
- Enhanced Tasks (within PRDs)
- Kanban workflow support
- Short ID integration
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008_add_project_management_hierarchy'
down_revision = '007_add_production_indexes'
branch_labels = None
depends_on = None


def upgrade():
    """Add project management hierarchy tables and enhancements."""
    
    # Create enum types
    op.execute("CREATE TYPE projectstatus AS ENUM ('planning', 'active', 'on_hold', 'completed', 'archived')")
    op.execute("CREATE TYPE epicstatus AS ENUM ('draft', 'planned', 'in_progress', 'completed', 'cancelled')")
    op.execute("CREATE TYPE prdstatus AS ENUM ('draft', 'review', 'approved', 'in_development', 'completed', 'deprecated')")
    op.execute("CREATE TYPE kanban_state_project AS ENUM ('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled')")
    op.execute("CREATE TYPE kanban_state_epic AS ENUM ('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled')")
    op.execute("CREATE TYPE kanban_state_prd AS ENUM ('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled')")
    op.execute("CREATE TYPE kanban_state_task AS ENUM ('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled')")
    op.execute("CREATE TYPE epic_priority AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')")
    op.execute("CREATE TYPE prd_priority AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')")
    
    # Update TaskType enum to include new types
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'analysis'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'infrastructure'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'security'")
    
    # Create projects table
    op.create_table('projects',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', postgresql.ENUM('planning', 'active', 'on_hold', 'completed', 'archived', name='projectstatus'), 
                  nullable=False, default='planning', index=True),
        sa.Column('kanban_state', postgresql.ENUM('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled', name='kanban_state_project'), 
                  nullable=False, default='backlog', index=True),
        sa.Column('objectives', postgresql.JSON(), nullable=True),
        sa.Column('success_criteria', postgresql.JSON(), nullable=True),
        sa.Column('stakeholders', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('target_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('actual_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('owner_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('assigned_agents', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('configuration', postgresql.JSON(), nullable=True),
        sa.Column('external_links', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        
        # Short ID columns
        sa.Column('short_id', sa.String(length=20), nullable=True, unique=True, index=True),
        sa.Column('short_id_generated_at', sa.DateTime(timezone=True), nullable=True),
        
        # Constraints
        sa.CheckConstraint('start_date IS NULL OR target_end_date IS NULL OR start_date <= target_end_date', name='check_project_dates'),
        sa.ForeignKeyConstraint(['owner_agent_id'], ['agents.id'], name='fk_projects_owner_agent'),
    )
    
    # Create indexes for projects
    op.create_index('ix_projects_status_kanban', 'projects', ['status', 'kanban_state'])
    op.create_index('ix_projects_timeline', 'projects', ['start_date', 'target_end_date'])
    
    # Create epics table
    op.create_table('epics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=255), nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('status', postgresql.ENUM('draft', 'planned', 'in_progress', 'completed', 'cancelled', name='epicstatus'), 
                  nullable=False, default='draft', index=True),
        sa.Column('kanban_state', postgresql.ENUM('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled', name='kanban_state_epic'), 
                  nullable=False, default='backlog', index=True),
        sa.Column('user_stories', postgresql.JSON(), nullable=True),
        sa.Column('acceptance_criteria', postgresql.JSON(), nullable=True),
        sa.Column('business_value', sa.Text(), nullable=True),
        sa.Column('technical_notes', sa.Text(), nullable=True),
        sa.Column('priority', postgresql.ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='epic_priority'), 
                  nullable=False, default='MEDIUM', index=True),
        sa.Column('estimated_story_points', sa.Integer(), nullable=True),
        sa.Column('actual_story_points', sa.Integer(), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('target_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('actual_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('owner_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('assigned_agents', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('dependencies', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('blocking_epics', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('labels', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        
        # Short ID columns
        sa.Column('short_id', sa.String(length=20), nullable=True, unique=True, index=True),
        sa.Column('short_id_generated_at', sa.DateTime(timezone=True), nullable=True),
        
        # Constraints
        sa.CheckConstraint('start_date IS NULL OR target_end_date IS NULL OR start_date <= target_end_date', name='check_epic_dates'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], name='fk_epics_project'),
        sa.ForeignKeyConstraint(['owner_agent_id'], ['agents.id'], name='fk_epics_owner_agent'),
    )
    
    # Create indexes for epics
    op.create_index('ix_epics_project_status', 'epics', ['project_id', 'status'])
    op.create_index('ix_epics_priority_kanban', 'epics', ['priority', 'kanban_state'])
    op.create_index('ix_epics_timeline', 'epics', ['start_date', 'target_end_date'])
    
    # Create PRDs table
    op.create_table('prds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(length=255), nullable=False, index=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('epic_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('status', postgresql.ENUM('draft', 'review', 'approved', 'in_development', 'completed', 'deprecated', name='prdstatus'), 
                  nullable=False, default='draft', index=True),
        sa.Column('kanban_state', postgresql.ENUM('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled', name='kanban_state_prd'), 
                  nullable=False, default='backlog', index=True),
        sa.Column('requirements', postgresql.JSON(), nullable=True),
        sa.Column('technical_requirements', postgresql.JSON(), nullable=True),
        sa.Column('acceptance_criteria', postgresql.JSON(), nullable=True),
        sa.Column('user_flows', postgresql.JSON(), nullable=True),
        sa.Column('mockups_wireframes', postgresql.JSON(), nullable=True),
        sa.Column('priority', postgresql.ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='prd_priority'), 
                  nullable=False, default='MEDIUM', index=True),
        sa.Column('estimated_effort_days', sa.Integer(), nullable=True),
        sa.Column('complexity_score', sa.Integer(), nullable=True),
        sa.Column('reviewers', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('approved_by', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('approval_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('owner_agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('assigned_agents', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('dependencies', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('blocking_prds', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('version', sa.String(length=20), nullable=False, default='1.0'),
        sa.Column('previous_version_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('external_references', postgresql.JSON(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        
        # Short ID columns
        sa.Column('short_id', sa.String(length=20), nullable=True, unique=True, index=True),
        sa.Column('short_id_generated_at', sa.DateTime(timezone=True), nullable=True),
        
        # Constraints
        sa.CheckConstraint('complexity_score IS NULL OR (complexity_score >= 1 AND complexity_score <= 10)', name='check_complexity_range'),
        sa.ForeignKeyConstraint(['epic_id'], ['epics.id'], name='fk_prds_epic'),
        sa.ForeignKeyConstraint(['owner_agent_id'], ['agents.id'], name='fk_prds_owner_agent'),
        sa.ForeignKeyConstraint(['previous_version_id'], ['prds.id'], name='fk_prds_previous_version'),
    )
    
    # Create indexes for PRDs
    op.create_index('ix_prds_epic_status', 'prds', ['epic_id', 'status'])
    op.create_index('ix_prds_priority_kanban', 'prds', ['priority', 'kanban_state'])
    op.create_index('ix_prds_version', 'prds', ['version'])
    
    # Enhance existing tasks table with new project management features
    # Add new columns to tasks table
    op.add_column('tasks', sa.Column('prd_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('tasks', sa.Column('kanban_state', postgresql.ENUM('backlog', 'ready', 'in_progress', 'review', 'done', 'blocked', 'cancelled', name='kanban_state_task'), 
                                     nullable=False, default='backlog', server_default='backlog'))
    op.add_column('tasks', sa.Column('complexity_points', sa.Integer(), nullable=True))
    op.add_column('tasks', sa.Column('acceptance_criteria', postgresql.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('test_requirements', postgresql.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('quality_gates', postgresql.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('due_date', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('scheduled_start', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('actual_start', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('actual_completion', sa.DateTime(timezone=True), nullable=True))
    op.add_column('tasks', sa.Column('state_history', postgresql.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True))
    op.add_column('tasks', sa.Column('external_references', postgresql.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('metadata', postgresql.JSON(), nullable=True))
    
    # Rename estimated_effort to estimated_effort_minutes for clarity
    op.alter_column('tasks', 'estimated_effort', new_column_name='estimated_effort_minutes')
    op.alter_column('tasks', 'actual_effort', new_column_name='actual_effort_minutes')
    
    # Add short ID columns to tasks if they don't exist
    try:
        op.add_column('tasks', sa.Column('short_id', sa.String(length=20), nullable=True, unique=True))
        op.add_column('tasks', sa.Column('short_id_generated_at', sa.DateTime(timezone=True), nullable=True))
    except Exception:
        # Columns might already exist
        pass
    
    # Add foreign key constraints for tasks
    op.create_foreign_key('fk_tasks_prd', 'tasks', 'prds', ['prd_id'], ['id'])
    
    # Add new task indexes
    op.create_index('ix_tasks_prd_kanban', 'tasks', ['prd_id', 'kanban_state'])
    op.create_index('ix_tasks_assigned_kanban', 'tasks', ['assigned_agent_id', 'kanban_state'])
    op.create_index('ix_tasks_priority_due', 'tasks', ['priority', 'due_date'])
    op.create_index('ix_tasks_type_kanban', 'tasks', ['task_type', 'kanban_state'])
    
    # Add check constraints for tasks
    op.create_check_constraint('check_complexity_points_range', 'tasks', 
                              'complexity_points IS NULL OR (complexity_points >= 1 AND complexity_points <= 21)')
    
    # Create short ID tracking table for all entity types
    op.create_table('short_id_registry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=sa.func.gen_random_uuid()),
        sa.Column('short_id', sa.String(length=20), nullable=False, unique=True, index=True),
        sa.Column('entity_type', sa.String(length=50), nullable=False, index=True),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        
        # Unique constraint on entity_type + entity_id
        sa.UniqueConstraint('entity_type', 'entity_id', name='uq_entity_short_id')
    )
    
    # Create indexes for short ID registry
    op.create_index('ix_short_id_registry_entity', 'short_id_registry', ['entity_type', 'entity_id'])
    
    # Create trigger function for automatic short ID generation
    op.execute("""
        CREATE OR REPLACE FUNCTION generate_short_id_trigger()
        RETURNS TRIGGER AS $$
        DECLARE
            entity_prefix VARCHAR(3);
            random_suffix VARCHAR(10);
            new_short_id VARCHAR(20);
            attempt_count INTEGER := 0;
            max_attempts INTEGER := 10;
        BEGIN
            -- Determine entity prefix based on table name
            CASE TG_TABLE_NAME
                WHEN 'projects' THEN entity_prefix := 'PRJ';
                WHEN 'epics' THEN entity_prefix := 'EPC';
                WHEN 'prds' THEN entity_prefix := 'PRD';
                WHEN 'tasks' THEN entity_prefix := 'TSK';
                ELSE entity_prefix := 'UNK';
            END CASE;
            
            -- Generate unique short ID
            WHILE attempt_count < max_attempts LOOP
                -- Generate random suffix using human-friendly characters
                random_suffix := UPPER(SUBSTRING(MD5(RANDOM()::TEXT || CLOCK_TIMESTAMP()::TEXT) FROM 1 FOR 4));
                -- Replace ambiguous characters
                random_suffix := REPLACE(random_suffix, '0', '2');
                random_suffix := REPLACE(random_suffix, '1', '3');
                random_suffix := REPLACE(random_suffix, 'I', 'J');
                random_suffix := REPLACE(random_suffix, 'O', 'P');
                
                new_short_id := entity_prefix || '-' || random_suffix;
                
                -- Check if this short ID already exists across all tables
                IF NOT EXISTS (
                    SELECT 1 FROM short_id_registry WHERE short_id = new_short_id
                ) THEN
                    NEW.short_id := new_short_id;
                    NEW.short_id_generated_at := NOW();
                    
                    -- Insert into registry
                    INSERT INTO short_id_registry (short_id, entity_type, entity_id)
                    VALUES (new_short_id, TG_TABLE_NAME, NEW.id);
                    
                    EXIT;
                END IF;
                
                attempt_count := attempt_count + 1;
            END LOOP;
            
            -- If we couldn't generate a unique ID after max attempts, log warning but continue
            IF NEW.short_id IS NULL THEN
                RAISE WARNING 'Could not generate unique short ID for % after % attempts', TG_TABLE_NAME, max_attempts;
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create triggers for automatic short ID generation
    for table_name in ['projects', 'epics', 'prds', 'tasks']:
        op.execute(f"""
            CREATE TRIGGER trigger_generate_short_id_{table_name}
                BEFORE INSERT ON {table_name}
                FOR EACH ROW
                WHEN (NEW.short_id IS NULL)
                EXECUTE FUNCTION generate_short_id_trigger();
        """)
    
    # Create indexes for performance
    op.create_index('ix_projects_short_id', 'projects', ['short_id'])
    op.create_index('ix_epics_short_id', 'epics', ['short_id'])
    op.create_index('ix_prds_short_id', 'prds', ['short_id'])
    op.create_index('ix_tasks_short_id', 'tasks', ['short_id'])


def downgrade():
    """Remove project management hierarchy tables and enhancements."""
    
    # Drop triggers
    for table_name in ['projects', 'epics', 'prds', 'tasks']:
        op.execute(f"DROP TRIGGER IF EXISTS trigger_generate_short_id_{table_name} ON {table_name}")
    
    # Drop trigger function
    op.execute("DROP FUNCTION IF EXISTS generate_short_id_trigger()")
    
    # Drop short ID registry table
    op.drop_table('short_id_registry')
    
    # Remove enhancements from tasks table
    op.drop_constraint('fk_tasks_prd', 'tasks', type_='foreignkey')
    op.drop_constraint('check_complexity_points_range', 'tasks', type_='check')
    
    # Drop task indexes
    op.drop_index('ix_tasks_prd_kanban', 'tasks')
    op.drop_index('ix_tasks_assigned_kanban', 'tasks')
    op.drop_index('ix_tasks_priority_due', 'tasks')
    op.drop_index('ix_tasks_type_kanban', 'tasks')
    op.drop_index('ix_tasks_short_id', 'tasks')
    
    # Remove new columns from tasks
    columns_to_remove = [
        'prd_id', 'kanban_state', 'complexity_points', 'acceptance_criteria',
        'test_requirements', 'quality_gates', 'due_date', 'scheduled_start',
        'actual_start', 'actual_completion', 'state_history', 'tags',
        'external_references', 'metadata'
    ]
    
    for column in columns_to_remove:
        try:
            op.drop_column('tasks', column)
        except Exception:
            pass  # Column might not exist
    
    # Restore original column names
    try:
        op.alter_column('tasks', 'estimated_effort_minutes', new_column_name='estimated_effort')
        op.alter_column('tasks', 'actual_effort_minutes', new_column_name='actual_effort')
    except Exception:
        pass
    
    # Drop PRDs table
    op.drop_index('ix_prds_version', 'prds')
    op.drop_index('ix_prds_priority_kanban', 'prds')
    op.drop_index('ix_prds_epic_status', 'prds')
    op.drop_index('ix_prds_short_id', 'prds')
    op.drop_table('prds')
    
    # Drop epics table
    op.drop_index('ix_epics_timeline', 'epics')
    op.drop_index('ix_epics_priority_kanban', 'epics')
    op.drop_index('ix_epics_project_status', 'epics')
    op.drop_index('ix_epics_short_id', 'epics')
    op.drop_table('epics')
    
    # Drop projects table
    op.drop_index('ix_projects_timeline', 'projects')
    op.drop_index('ix_projects_status_kanban', 'projects')
    op.drop_index('ix_projects_short_id', 'projects')
    op.drop_table('projects')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS prd_priority")
    op.execute("DROP TYPE IF EXISTS epic_priority")
    op.execute("DROP TYPE IF EXISTS kanban_state_task")
    op.execute("DROP TYPE IF EXISTS kanban_state_prd")
    op.execute("DROP TYPE IF EXISTS kanban_state_epic")
    op.execute("DROP TYPE IF EXISTS kanban_state_project")
    op.execute("DROP TYPE IF EXISTS prdstatus")
    op.execute("DROP TYPE IF EXISTS epicstatus")
    op.execute("DROP TYPE IF EXISTS projectstatus")