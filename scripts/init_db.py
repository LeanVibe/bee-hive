#!/usr/bin/env python3
"""
Idempotent Database Bootstrap Orchestrator
LeanVibe Agent Hive 2.0

This script ensures the database is properly initialized with all prerequisites
before running schema migrations. It can be run multiple times safely.
"""

import os
import sys
import time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import subprocess
from typing import List, Dict

# Database connection parameters
DATABASE_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'leanvibe_agent_hive'),
    'user': os.getenv('POSTGRES_USER', 'leanvibe_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'leanvibe_secure_pass')
}

# Required PostgreSQL extensions
REQUIRED_EXTENSIONS = [
    'vector',  # pgvector for semantic search
    '"uuid-ossp"',  # UUID generation functions (quoted due to hyphen)
    'pg_trgm',  # Trigram matching for text search
]

# Required enum types that must exist before schema migrations
REQUIRED_ENUMS = [
    {
        'name': 'agentstatus',
        'values': ['active', 'inactive', 'busy', 'error', 'maintenance']
    },
    {
        'name': 'taskstatus', 
        'values': ['pending', 'in_progress', 'completed', 'failed', 'cancelled']
    },
    {
        'name': 'taskpriority',
        'values': ['low', 'medium', 'high', 'urgent']
    },
    {
        'name': 'tasktype',
        'values': ['code_generation', 'testing', 'deployment', 'analysis', 'documentation']
    },
    {
        'name': 'contexttype',
        'values': ['conversation', 'code', 'documentation', 'configuration', 'analysis']
    },
    {
        'name': 'workflowstatus',
        'values': ['draft', 'active', 'paused', 'completed', 'failed']
    },
    {
        'name': 'messagetype',
        'values': ['task_assignment', 'status_update', 'error_report', 'completion_notice']
    },
    {
        'name': 'eventtype',
        'values': ['agent_started', 'agent_stopped', 'task_created', 'task_completed', 'error_occurred']
    },
    {
        'name': 'contextstatus',
        'values': ['active', 'archived', 'consolidated', 'deprecated']
    },
    {
        'name': 'consolidationstatus',
        'values': ['pending', 'in_progress', 'completed', 'failed']
    },
    {
        'name': 'repositorystatus',
        'values': ['active', 'archived', 'private', 'public']
    },
    {
        'name': 'issuestate',
        'values': ['open', 'closed', 'in_progress', 'blocked']
    },
    {
        'name': 'branchstatus',
        'values': ['active', 'merged', 'abandoned', 'stale']
    },
    {
        'name': 'modificationsafety',
        'values': ['conservative', 'moderate', 'aggressive', 'experimental']
    },
    {
        'name': 'modificationstatus',
        'values': ['analyzing', 'planning', 'executing', 'completed', 'failed', 'rolled_back']
    },
    {
        'name': 'prompt_status',
        'values': ['draft', 'active', 'archived', 'deprecated']
    }
]


class DatabaseBootstrap:
    """Handles idempotent database initialization."""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Establish database connection with retries."""
        max_retries = 30
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔌 Attempting database connection (attempt {attempt + 1}/{max_retries})...")
                self.connection = psycopg2.connect(**DATABASE_CONFIG)
                self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                self.cursor = self.connection.cursor()
                
                # Test connection
                self.cursor.execute("SELECT version();")
                version = self.cursor.fetchone()[0]
                print(f"✅ Connected to PostgreSQL: {version}")
                return True
                
            except psycopg2.Error as e:
                print(f"❌ Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"💥 Failed to connect after {max_retries} attempts")
                    return False
        
        return False
    
    def ensure_extensions(self) -> bool:
        """Create required PostgreSQL extensions."""
        print("\n📦 Ensuring required PostgreSQL extensions...")
        
        for extension in REQUIRED_EXTENSIONS:
            try:
                self.cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {extension};")
                print(f"✅ Extension '{extension}' ensured")
            except psycopg2.Error as e:
                print(f"❌ Failed to create extension '{extension}': {e}")
                return False
        
        return True
    
    def ensure_enum_types(self) -> bool:
        """Create required enum types."""
        print("\n🔢 Ensuring required enum types...")
        
        for enum_def in REQUIRED_ENUMS:
            enum_name = enum_def['name']
            enum_values = enum_def['values']
            
            try:
                # Check if enum type exists
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_type WHERE typname = %s
                    );
                """, (enum_name,))
                
                exists = self.cursor.fetchone()[0]
                
                if not exists:
                    # Create enum type
                    values_str = "', '".join(enum_values)
                    create_enum_sql = f"CREATE TYPE {enum_name} AS ENUM ('{values_str}');"
                    self.cursor.execute(create_enum_sql)
                    print(f"✅ Created enum type '{enum_name}' with values: {enum_values}")
                else:
                    print(f"✅ Enum type '{enum_name}' already exists")
                    
            except psycopg2.Error as e:
                print(f"❌ Failed to create enum type '{enum_name}': {e}")
                return False
        
        return True
    
    def run_migrations(self) -> bool:
        """Run Alembic database migrations."""
        print("\n📋 Running database schema migrations...")
        
        try:
            # Change to project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.chdir(project_root)
            
            # Run Alembic migrations
            result = subprocess.run(
                ['alembic', 'upgrade', 'head'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Database migrations completed successfully")
                if result.stdout:
                    print(f"📝 Migration output:\n{result.stdout}")
                return True
            else:
                print(f"❌ Migration failed with return code {result.returncode}")
                if result.stderr:
                    print(f"🚨 Migration errors:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Migration timeout after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Migration error: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate that database setup is complete."""
        print("\n🔍 Validating database setup...")
        
        try:
            # Check that key tables exist
            key_tables = ['agents', 'tasks', 'workflows', 'contexts', 'sessions']
            
            for table in key_tables:
                self.cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                
                exists = self.cursor.fetchone()[0]
                if exists:
                    print(f"✅ Table '{table}' exists")
                else:
                    print(f"❌ Table '{table}' missing")
                    return False
            
            # Check that pgvector extension is working
            self.cursor.execute("SELECT 1 WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            if self.cursor.fetchone():
                print("✅ pgvector extension is active")
            else:
                print("❌ pgvector extension not found")
                return False
            
            print("✅ Database validation successful")
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Database validation failed: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    
    def bootstrap(self) -> bool:
        """Run complete database bootstrap process."""
        print("🚀 Starting LeanVibe Agent Hive Database Bootstrap")
        print("=" * 60)
        
        try:
            # Step 1: Connect to database
            if not self.connect():
                return False
            
            # Step 2: Ensure extensions
            if not self.ensure_extensions():
                return False
            
            # Step 3: Ensure enum types  
            if not self.ensure_enum_types():
                return False
            
            # Step 4: Run migrations
            if not self.run_migrations():
                return False
            
            # Step 5: Validate setup
            if not self.validate_setup():
                return False
            
            print("\n🎉 Database bootstrap completed successfully!")
            print("✅ LeanVibe Agent Hive database is ready for use")
            return True
            
        except Exception as e:
            print(f"\n💥 Bootstrap failed with unexpected error: {e}")
            return False
        
        finally:
            self.close()


def main():
    """Main entry point for database bootstrap."""
    print("🏗️  LeanVibe Agent Hive - Database Bootstrap Orchestrator")
    print("🎯 Ensuring idempotent database initialization")
    print()
    
    # Load environment variables from .env.local if it exists
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local')
    if os.path.exists(env_file):
        print(f"📄 Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Initialize and run bootstrap
    bootstrap = DatabaseBootstrap()
    success = bootstrap.bootstrap()
    
    if success:
        print("\n🚀 Database is ready! You can now start the application.")
        sys.exit(0)
    else:
        print("\n💥 Database bootstrap failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()