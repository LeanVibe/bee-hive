#!/usr/bin/env python3
"""
Fix database enum types for LeanVibe Agent Hive 2.0

This script creates the missing enum types that are causing database errors.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from app.core.database import create_engine
from app.models.task import TaskStatus
from app.models.agent import AgentStatus


async def fix_database_enums():
    """Create missing enum types in PostgreSQL."""
    
    engine = await create_engine()
    
    try:
        async with engine.begin() as conn:
            print("🔧 Fixing database enum types...")
            
            # Create TaskStatus enum type
            task_status_values = [status.value for status in TaskStatus]
            task_status_enum_sql = f"""
            DO $$ BEGIN
                CREATE TYPE taskstatus AS ENUM ({', '.join([f"'{val}'" for val in task_status_values])});
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """
            
            await conn.execute(text(task_status_enum_sql))
            print("✅ TaskStatus enum created/updated")
            
            # Create AgentStatus enum type  
            agent_status_values = [status.value for status in AgentStatus]
            agent_status_enum_sql = f"""
            DO $$ BEGIN
                CREATE TYPE agentstatus AS ENUM ({', '.join([f"'{val}'" for val in agent_status_values])});
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
            """
            
            await conn.execute(text(agent_status_enum_sql))
            print("✅ AgentStatus enum created/updated")
            
            # Fix existing columns to use proper enum types
            print("🔧 Updating existing table columns...")
            
            # Convert tasks.status column to use enum
            await conn.execute(text("""
                DO $$ BEGIN
                    ALTER TABLE tasks ALTER COLUMN status TYPE taskstatus USING status::taskstatus;
                EXCEPTION
                    WHEN others THEN 
                        RAISE NOTICE 'Could not convert tasks.status column: %', SQLERRM;
                END $$;
            """))
            
            # Convert agents.status column to use enum  
            await conn.execute(text("""
                DO $$ BEGIN
                    ALTER TABLE agents ALTER COLUMN status TYPE agentstatus USING status::agentstatus;
                EXCEPTION
                    WHEN others THEN 
                        RAISE NOTICE 'Could not convert agents.status column: %', SQLERRM;
                END $$;
            """))
            
            print("✅ Database enum types fixed successfully!")
            
    except Exception as e:
        print(f"❌ Error fixing database enums: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(fix_database_enums())