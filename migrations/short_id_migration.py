"""
Database Migration for Short ID System Implementation

This migration adds short ID support to existing LeanVibe Agent Hive 2.0 tables
while maintaining backward compatibility with existing UUID-based systems.

Migration Strategy:
1. Add short_id columns to existing tables
2. Create central short_id_mappings table  
3. Generate short IDs for existing records
4. Add indexes for performance
5. Create triggers for automatic ID generation
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from datetime import datetime
import uuid
import logging

# Import our short ID generator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.short_id_generator import (
    ShortIdGenerator, EntityType, generate_short_id
)

logger = logging.getLogger(__name__)

# Revision identifiers
revision = 'short_id_system'
down_revision = 'head'  # Replace with actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    """Apply short ID system migration."""
    
    logger.info("Starting short ID system migration...")
    
    # 1. Create short_id_mappings table
    create_short_id_mappings_table()
    
    # 2. Add short_id columns to existing tables
    add_short_id_columns()
    
    # 3. Create indexes for performance
    create_indexes()
    
    # 4. Generate short IDs for existing records
    generate_existing_short_ids()
    
    # 5. Create database functions and triggers
    create_database_functions()
    
    logger.info("Short ID system migration completed successfully")


def downgrade():
    """Remove short ID system migration."""
    
    logger.info("Starting short ID system rollback...")
    
    # Remove triggers
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_projects ON project_indexes")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_tasks ON tasks")  
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_agents ON agents")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_workflows ON workflows")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_files ON file_entries")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_dependencies ON dependency_relationships")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_snapshots ON index_snapshots")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_sessions ON analysis_sessions")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_debt_items ON debt_items")
    op.execute("DROP TRIGGER IF EXISTS auto_generate_short_id_debt_plans ON debt_remediation_plans")
    
    # Remove functions
    op.execute("DROP FUNCTION IF EXISTS generate_short_id(varchar, uuid)")
    op.execute("DROP FUNCTION IF EXISTS auto_generate_short_id()")
    
    # Remove short_id columns
    remove_short_id_columns()
    
    # Remove short_id_mappings table
    op.drop_table('short_id_mappings')
    
    logger.info("Short ID system rollback completed")


def create_short_id_mappings_table():
    """Create the central short_id_mappings table."""
    
    op.create_table(
        'short_id_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                 server_default=sa.text('gen_random_uuid()')),
        sa.Column('short_id', sa.String(20), nullable=False, unique=True),
        sa.Column('entity_uuid', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('entity_type', sa.String(10), nullable=False),
        sa.Column('entity_table', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                 server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                 server_default=sa.func.now()),
        
        # Constraints
        sa.UniqueConstraint('short_id', name='uq_short_id'),
        sa.UniqueConstraint('entity_uuid', 'entity_type', name='uq_entity_uuid_type')
    )
    
    logger.info("Created short_id_mappings table")


def add_short_id_columns():
    """Add short_id columns to existing tables."""
    
    # Tables and their entity types
    tables = [
        ('project_indexes', EntityType.PROJECT),
        ('file_entries', EntityType.FILE),
        ('dependency_relationships', EntityType.DEPENDENCY),
        ('index_snapshots', EntityType.SNAPSHOT),
        ('analysis_sessions', EntityType.SESSION),
        ('debt_items', EntityType.DEBT),
        ('debt_remediation_plans', EntityType.PLAN),
        # Add other tables from different modules
        ('agents', EntityType.AGENT),
        ('tasks', EntityType.TASK), 
        ('workflows', EntityType.WORKFLOW),
    ]
    
    for table_name, entity_type in tables:
        try:
            # Check if table exists before adding columns
            result = op.get_bind().execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                );
            """)).fetchone()
            
            if result[0]:  # Table exists
                # Add short_id column
                op.add_column(table_name, 
                    sa.Column('short_id', sa.String(20), nullable=True, unique=True,
                             comment='Human-friendly short identifier'))
                
                # Add metadata column
                op.add_column(table_name,
                    sa.Column('short_id_generated_at', sa.DateTime(timezone=True), 
                             nullable=True, server_default=sa.func.now(),
                             comment='When the short ID was generated'))
                
                logger.info(f"Added short_id columns to {table_name}")
            else:
                logger.warning(f"Table {table_name} does not exist, skipping")
                
        except Exception as e:
            logger.error(f"Error adding columns to {table_name}: {e}")
            # Continue with other tables


def create_indexes():
    """Create indexes for optimal short ID performance."""
    
    # Indexes on short_id_mappings
    op.create_index('idx_short_id_mappings_short_id', 'short_id_mappings', ['short_id'])
    op.create_index('idx_short_id_mappings_entity_uuid', 'short_id_mappings', ['entity_uuid'])
    op.create_index('idx_short_id_mappings_entity_type', 'short_id_mappings', ['entity_type'])
    op.create_index('idx_short_id_mappings_created_at', 'short_id_mappings', ['created_at'])
    
    # Indexes on individual tables
    tables = [
        'project_indexes', 'file_entries', 'dependency_relationships', 
        'index_snapshots', 'analysis_sessions', 'debt_items', 
        'debt_remediation_plans', 'agents', 'tasks', 'workflows'
    ]
    
    for table_name in tables:
        try:
            # Check if table exists
            result = op.get_bind().execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                );
            """)).fetchone()
            
            if result[0]:  # Table exists
                op.create_index(f'idx_{table_name}_short_id', table_name, ['short_id'])
                logger.info(f"Created index on {table_name}.short_id")
                
        except Exception as e:
            logger.error(f"Error creating index on {table_name}: {e}")


def generate_existing_short_ids():
    """Generate short IDs for existing records."""
    
    bind = op.get_bind()
    generator = ShortIdGenerator()
    
    # Tables and their entity types
    table_mappings = {
        'project_indexes': EntityType.PROJECT,
        'file_entries': EntityType.FILE,
        'dependency_relationships': EntityType.DEPENDENCY,
        'index_snapshots': EntityType.SNAPSHOT,
        'analysis_sessions': EntityType.SESSION,
        'debt_items': EntityType.DEBT,
        'debt_remediation_plans': EntityType.PLAN,
        'agents': EntityType.AGENT,
        'tasks': EntityType.TASK,
        'workflows': EntityType.WORKFLOW,
    }
    
    for table_name, entity_type in table_mappings.items():
        try:
            # Check if table exists and has records
            count_result = bind.execute(text(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE short_id IS NULL
            """)).fetchone()
            
            if count_result[0] > 0:
                logger.info(f"Generating short IDs for {count_result[0]} records in {table_name}")
                
                # Get records without short IDs
                records = bind.execute(text(f"""
                    SELECT id FROM {table_name} 
                    WHERE short_id IS NULL
                    ORDER BY created_at ASC
                """)).fetchall()
                
                # Generate short IDs in batches
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    
                    for record in batch:
                        try:
                            # Generate short ID
                            short_id, _ = generator.generate_id(entity_type, record[0])
                            
                            # Update record
                            bind.execute(text(f"""
                                UPDATE {table_name} 
                                SET short_id = :short_id, 
                                    short_id_generated_at = NOW()
                                WHERE id = :record_id
                            """), {
                                'short_id': short_id,
                                'record_id': record[0]
                            })
                            
                            # Insert into mappings table
                            bind.execute(text("""
                                INSERT INTO short_id_mappings 
                                (short_id, entity_uuid, entity_type, entity_table, created_at, updated_at)
                                VALUES (:short_id, :entity_uuid, :entity_type, :entity_table, NOW(), NOW())
                            """), {
                                'short_id': short_id,
                                'entity_uuid': record[0],
                                'entity_type': entity_type.value,
                                'entity_table': table_name
                            })
                            
                        except Exception as e:
                            logger.error(f"Error generating short ID for {record[0]}: {e}")
                            continue
                    
                    # Commit batch
                    bind.execute(text("COMMIT"))
                    logger.info(f"Processed batch {i//batch_size + 1} of {len(records)//batch_size + 1}")
                
                logger.info(f"Completed generating short IDs for {table_name}")
            else:
                logger.info(f"No records need short IDs in {table_name}")
                
        except Exception as e:
            logger.error(f"Error processing table {table_name}: {e}")
            # Rollback and continue
            bind.execute(text("ROLLBACK"))


def create_database_functions():
    """Create database functions for automatic short ID generation."""
    
    # Create function to generate short IDs
    op.execute(text("""
        CREATE OR REPLACE FUNCTION generate_short_id(entity_type_param VARCHAR, entity_uuid_param UUID)
        RETURNS VARCHAR AS $$
        DECLARE
            prefix VARCHAR(3);
            code VARCHAR(4);
            attempt INTEGER := 0;
            candidate_id VARCHAR(20);
            collision_exists BOOLEAN;
        BEGIN
            -- Map entity types to prefixes
            prefix := CASE entity_type_param
                WHEN 'PROJECT' THEN 'PRJ'
                WHEN 'EPIC' THEN 'EPC' 
                WHEN 'PRD' THEN 'PRD'
                WHEN 'TASK' THEN 'TSK'
                WHEN 'AGENT' THEN 'AGT'
                WHEN 'WORKFLOW' THEN 'WFL'
                WHEN 'FILE' THEN 'FIL'
                WHEN 'DEPENDENCY' THEN 'DEP'
                WHEN 'SNAPSHOT' THEN 'SNP'
                WHEN 'SESSION' THEN 'SES'
                WHEN 'DEBT' THEN 'DBT'
                WHEN 'PLAN' THEN 'PLN'
                ELSE 'UNK'
            END;
            
            -- Generate unique code
            WHILE attempt < 5 LOOP
                -- Generate pseudo-random code based on UUID and attempt
                code := substr(
                    translate(
                        encode(
                            digest(entity_uuid_param::text || attempt::text, 'sha256'), 
                            'base64'
                        ),
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=',
                        '23456789ABCDEFGHJKMNPQRSTVWXYZ23456789ABCDEFGHJKMNPQRST'
                    ),
                    1, 4
                );
                
                candidate_id := prefix || '-' || code;
                
                -- Check for collision
                SELECT EXISTS(
                    SELECT 1 FROM short_id_mappings 
                    WHERE short_id = candidate_id
                ) INTO collision_exists;
                
                IF NOT collision_exists THEN
                    RETURN candidate_id;
                END IF;
                
                attempt := attempt + 1;
            END LOOP;
            
            -- If we get here, something went wrong
            RAISE EXCEPTION 'Unable to generate unique short ID for % after 5 attempts', entity_type_param;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    # Create trigger function for automatic short ID generation
    op.execute(text("""
        CREATE OR REPLACE FUNCTION auto_generate_short_id()
        RETURNS TRIGGER AS $$
        DECLARE
            entity_type_val VARCHAR;
            generated_short_id VARCHAR(20);
        BEGIN
            -- Only generate if short_id is NULL
            IF NEW.short_id IS NULL THEN
                -- Determine entity type based on table name
                entity_type_val := CASE TG_TABLE_NAME
                    WHEN 'project_indexes' THEN 'PROJECT'
                    WHEN 'file_entries' THEN 'FILE'
                    WHEN 'dependency_relationships' THEN 'DEPENDENCY'
                    WHEN 'index_snapshots' THEN 'SNAPSHOT'
                    WHEN 'analysis_sessions' THEN 'SESSION'
                    WHEN 'debt_items' THEN 'DEBT'
                    WHEN 'debt_remediation_plans' THEN 'PLAN'
                    WHEN 'agents' THEN 'AGENT'
                    WHEN 'tasks' THEN 'TASK'
                    WHEN 'workflows' THEN 'WORKFLOW'
                    ELSE 'UNKNOWN'
                END;
                
                -- Generate short ID
                generated_short_id := generate_short_id(entity_type_val, NEW.id);
                
                -- Set in the record
                NEW.short_id := generated_short_id;
                NEW.short_id_generated_at := NOW();
                
                -- Insert into mappings table
                INSERT INTO short_id_mappings 
                (short_id, entity_uuid, entity_type, entity_table, created_at, updated_at)
                VALUES (generated_short_id, NEW.id, entity_type_val, TG_TABLE_NAME, NOW(), NOW());
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """))
    
    # Create triggers for each table
    tables = [
        'project_indexes', 'file_entries', 'dependency_relationships',
        'index_snapshots', 'analysis_sessions', 'debt_items', 
        'debt_remediation_plans', 'agents', 'tasks', 'workflows'
    ]
    
    for table_name in tables:
        try:
            # Check if table exists
            result = op.get_bind().execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                );
            """)).fetchone()
            
            if result[0]:  # Table exists
                op.execute(text(f"""
                    CREATE TRIGGER auto_generate_short_id_{table_name}
                        BEFORE INSERT ON {table_name}
                        FOR EACH ROW
                        EXECUTE FUNCTION auto_generate_short_id();
                """))
                logger.info(f"Created trigger for {table_name}")
                
        except Exception as e:
            logger.error(f"Error creating trigger for {table_name}: {e}")
    
    # Create updated_at trigger for mappings table
    op.execute(text("""
        CREATE OR REPLACE FUNCTION update_short_id_mappings_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE TRIGGER update_short_id_mappings_updated_at
            BEFORE UPDATE ON short_id_mappings
            FOR EACH ROW
            EXECUTE FUNCTION update_short_id_mappings_updated_at();
    """))
    
    logger.info("Created database functions and triggers")


def remove_short_id_columns():
    """Remove short_id columns from tables."""
    
    tables = [
        'project_indexes', 'file_entries', 'dependency_relationships',
        'index_snapshots', 'analysis_sessions', 'debt_items', 
        'debt_remediation_plans', 'agents', 'tasks', 'workflows'
    ]
    
    for table_name in tables:
        try:
            op.drop_column(table_name, 'short_id')
            op.drop_column(table_name, 'short_id_generated_at')
            logger.info(f"Removed short_id columns from {table_name}")
        except Exception as e:
            logger.error(f"Error removing columns from {table_name}: {e}")