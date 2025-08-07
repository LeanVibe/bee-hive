"""
Database Compatibility Fix for LeanVibe Agent Hive 2.0 Testing Infrastructure

This module provides utilities to fix SQLite/PostgreSQL compatibility issues
in the testing environment, specifically addressing JSONB type conflicts.
"""

import os
from typing import Optional
from sqlalchemy import MetaData, JSON, Text, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable


class DatabaseCompatibilityFixer:
    """
    Fixes database compatibility issues between PostgreSQL and SQLite for testing.
    
    Key fixes:
    - JSONB -> JSON mapping for SQLite compatibility
    - UUID handling for cross-database compatibility  
    - Enum type handling
    """
    
    @staticmethod
    def fix_jsonb_compatibility(engine: Engine, metadata: MetaData) -> None:
        """
        Replace PostgreSQL JSONB types with SQLite-compatible JSON types.
        
        Args:
            engine: SQLAlchemy engine instance
            metadata: Database metadata to fix
        """
        if engine.dialect.name != 'sqlite':
            # Only apply fixes for SQLite
            return
            
        # Iterate through all tables and columns
        for table_name, table in metadata.tables.items():
            for column_name, column in table.columns.items():
                # Replace JSONB with JSON for SQLite compatibility
                if hasattr(column.type, '__class__') and 'JSONB' in str(column.type):
                    column.type = JSON()
                    
                # Ensure proper column defaults
                if column.default is not None and hasattr(column.default, 'arg'):
                    if column.default.arg in ('{}', '[]'):
                        column.default.arg = None  # Let SQLAlchemy handle defaults
    
    @staticmethod
    def create_test_database_url() -> str:
        """
        Create an appropriate database URL for testing.
        
        Returns:
            Database URL string for test environment
        """
        # Use in-memory SQLite for fast tests
        if os.getenv('TEST_USE_POSTGRES') == 'true':
            # Optional PostgreSQL testing
            return "postgresql+asyncpg://test:test@localhost:5432/test_db"
        else:
            # Default to SQLite for speed and simplicity
            return "sqlite+aiosqlite:///:memory:"
    
    @staticmethod
    def ensure_test_compatibility() -> None:
        """
        Ensure test environment is properly configured for cross-database compatibility.
        """
        # Set test environment variables for compatibility
        os.environ.update({
            "DATABASE_URL": DatabaseCompatibilityFixer.create_test_database_url(),
            "TESTING": "true",
            "SKIP_MIGRATIONS": "true",  # Use direct table creation for tests
            "USE_DATABASE_AGNOSTIC_TYPES": "true"
        })
    
    @staticmethod
    def patch_migration_for_sqlite() -> None:
        """
        Monkey-patch PostgreSQL-specific types in migrations for SQLite compatibility.
        """
        # Replace postgresql.JSONB with standard JSON for SQLite
        import sqlalchemy.dialects.postgresql as pg
        from sqlalchemy import JSON, Text
        
        if not hasattr(pg, '_original_JSONB'):
            pg._original_JSONB = pg.JSONB
            # Replace JSONB constructor with JSON for SQLite compatibility
            def jsonb_replacement(*args, **kwargs):
                # Return JSON type for SQLite compatibility
                return JSON()
            pg.JSONB = jsonb_replacement
            
        # Also patch any direct JSONB references that might be in models
        if hasattr(pg, 'postgresql'):
            if not hasattr(pg.postgresql, '_original_JSONB'):
                pg.postgresql._original_JSONB = pg.postgresql.JSONB
                pg.postgresql.JSONB = jsonb_replacement


def setup_test_database_compatibility():
    """
    One-time setup function to configure database compatibility for tests.
    
    Call this in conftest.py or test setup to ensure proper compatibility.
    """
    fixer = DatabaseCompatibilityFixer()
    
    # Apply environment configuration
    fixer.ensure_test_compatibility()
    
    # Apply migration patches if needed
    if os.getenv('DATABASE_URL', '').startswith('sqlite'):
        fixer.patch_migration_for_sqlite()


def get_test_database_config() -> dict:
    """
    Get test database configuration optimized for the current environment.
    
    Returns:
        Dictionary with database configuration
    """
    return {
        "url": DatabaseCompatibilityFixer.create_test_database_url(),
        "connect_args": {
            "check_same_thread": False
        } if "sqlite" in DatabaseCompatibilityFixer.create_test_database_url() else {},
        "poolclass": None,  # Use default connection pooling
        "echo": False,  # Reduce noise in test output
    }