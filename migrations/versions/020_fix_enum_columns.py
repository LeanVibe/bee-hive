"""Fix enum column types to use proper enum types

Revision ID: 020_fix_enum_columns
Revises: 019_add_enhanced_orchestrator_metrics
Create Date: 2025-01-15 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '020_fix_enum_columns'
down_revision = '019_add_enhanced_orchestrator_metrics'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema to use proper enum types."""
    
    # First, ensure the enum types are properly created with all required values
    # Update taskstatus enum to include all values from the TaskStatus enum
    op.execute("ALTER TYPE taskstatus ADD VALUE IF NOT EXISTS 'assigned'")
    op.execute("ALTER TYPE taskstatus ADD VALUE IF NOT EXISTS 'blocked'")
    
    # Update tasktype enum to include all values from the TaskType enum  
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'feature_development'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'bug_fix'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'refactoring'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'testing'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'documentation'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'architecture'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'deployment'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'code_review'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'research'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'optimization'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'code_generation'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'coordination'")
    op.execute("ALTER TYPE tasktype ADD VALUE IF NOT EXISTS 'planning'")
    
    # Update taskpriority enum to include all values from the TaskPriority enum
    op.execute("ALTER TYPE taskpriority ADD VALUE IF NOT EXISTS 'low'")
    op.execute("ALTER TYPE taskpriority ADD VALUE IF NOT EXISTS 'medium'")
    op.execute("ALTER TYPE taskpriority ADD VALUE IF NOT EXISTS 'high'")
    op.execute("ALTER TYPE taskpriority ADD VALUE IF NOT EXISTS 'critical'")
    
    # Now convert tasks table columns to use proper enum types
    # Convert status column from varchar to taskstatus enum
    op.execute("""
        ALTER TABLE tasks 
        ALTER COLUMN status TYPE taskstatus 
        USING status::taskstatus
    """)
    
    # Convert task_type column from varchar to tasktype enum
    op.execute("""
        ALTER TABLE tasks 
        ALTER COLUMN task_type TYPE tasktype 
        USING task_type::tasktype
    """)
    
    # Convert priority column from varchar to taskpriority enum
    op.execute("""
        ALTER TABLE tasks 
        ALTER COLUMN priority TYPE taskpriority 
        USING priority::taskpriority
    """)
    
    # Update default values to use enum values instead of strings
    op.execute("ALTER TABLE tasks ALTER COLUMN status SET DEFAULT 'pending'::taskstatus")
    op.execute("ALTER TABLE tasks ALTER COLUMN priority SET DEFAULT 'medium'::taskpriority")
    
    # Similarly fix other tables that might have enum issues
    # Fix agents table status column
    op.execute("""
        ALTER TABLE agents 
        ALTER COLUMN status TYPE agentstatus 
        USING status::agentstatus
    """)
    
    # Fix sessions table status column  
    op.execute("""
        ALTER TABLE sessions 
        ALTER COLUMN status TYPE sessionstatus 
        USING status::sessionstatus
    """)


def downgrade() -> None:
    """Downgrade database schema back to varchar columns."""
    
    # Convert enum columns back to varchar
    op.execute("ALTER TABLE tasks ALTER COLUMN status TYPE character varying(20) USING status::text")
    op.execute("ALTER TABLE tasks ALTER COLUMN task_type TYPE character varying(30) USING task_type::text") 
    op.execute("ALTER TABLE tasks ALTER COLUMN priority TYPE character varying(10) USING priority::text")
    
    # Restore original default values
    op.execute("ALTER TABLE tasks ALTER COLUMN status SET DEFAULT 'PENDING'::character varying")
    op.execute("ALTER TABLE tasks ALTER COLUMN priority SET DEFAULT 'MEDIUM'::character varying")
    
    # Convert other tables back
    op.execute("ALTER TABLE agents ALTER COLUMN status TYPE character varying(20) USING status::text")
    op.execute("ALTER TABLE sessions ALTER COLUMN status TYPE character varying(20) USING status::text")