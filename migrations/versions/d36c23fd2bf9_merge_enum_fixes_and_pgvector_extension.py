"""merge enum fixes and pgvector extension

Revision ID: d36c23fd2bf9
Revises: 020_fix_enum_columns, a47c9cb5af36
Create Date: 2025-08-05 03:24:01.491312

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd36c23fd2bf9'
down_revision = ('020_fix_enum_columns', 'a47c9cb5af36')
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Merge upgrade that creates enum types if they don't exist and ensures pgvector extension."""
    
    # Enable pgvector extension if not already enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create enum types if they don't exist
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE sessionstatus AS ENUM ('active', 'inactive', 'sleeping', 'terminated');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE agentstatus AS ENUM ('inactive', 'active', 'busy', 'error', 'maintenance');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE taskstatus AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'cancelled', 'assigned', 'blocked');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE tasktype AS ENUM ('development', 'testing', 'documentation', 'research', 'feature_development', 'bug_fix', 'refactoring', 'architecture', 'deployment', 'code_review', 'optimization', 'code_generation', 'coordination', 'planning');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE taskpriority AS ENUM ('low', 'medium', 'high', 'critical');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    # Now safely convert columns to enum types if tables exist
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions') THEN
                ALTER TABLE sessions ALTER COLUMN status TYPE sessionstatus USING status::sessionstatus;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agents') THEN
                ALTER TABLE agents ALTER COLUMN status TYPE agentstatus USING status::agentstatus;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tasks') THEN
                ALTER TABLE tasks ALTER COLUMN status TYPE taskstatus USING status::taskstatus;
                ALTER TABLE tasks ALTER COLUMN task_type TYPE tasktype USING task_type::tasktype;
                ALTER TABLE tasks ALTER COLUMN priority TYPE taskpriority USING priority::taskpriority;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Downgrade - convert back to varchar types."""
    
    # Convert enum columns back to varchar
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tasks') THEN
                ALTER TABLE tasks ALTER COLUMN status TYPE character varying(20) USING status::text;
                ALTER TABLE tasks ALTER COLUMN task_type TYPE character varying(30) USING task_type::text;
                ALTER TABLE tasks ALTER COLUMN priority TYPE character varying(10) USING priority::text;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'agents') THEN
                ALTER TABLE agents ALTER COLUMN status TYPE character varying(20) USING status::text;
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions') THEN
                ALTER TABLE sessions ALTER COLUMN status TYPE character varying(20) USING status::text;
            END IF;
        END $$;
    """)
