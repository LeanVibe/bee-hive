"""
Alembic environment configuration for LeanVibe Agent Hive 2.0

Supports async database operations with PostgreSQL and pgvector.
Automatically detects model changes for migration generation.
"""

import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import pool, engine_from_config
from sqlalchemy.engine import Connection

# Import all models to ensure they're registered with SQLAlchemy
from app.core.database import Base
from app.models import *  # noqa: F403,F401

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata

# Override database URL from environment if available
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Convert to sync URL for migrations (replace asyncpg with psycopg2)
    if "postgresql+asyncpg://" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://")
    config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def include_object(object: Any, name: str, type_: str, reflected: bool, compare_to: Any) -> bool:
    """
    Filter objects to include in migrations.
    
    Returns True if object should be included in the migration.
    """
    # Skip system tables and extensions
    if type_ == "table" and name in [
        "spatial_ref_sys",  # PostGIS
        "geography_columns",  # PostGIS
        "geometry_columns",  # PostGIS
    ]:
        return False
    
    # Include everything else
    return True


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Enable autogenerate for foreign keys
        render_as_batch=False,
        # Custom naming convention for constraints
        version_table_schema=target_metadata.schema,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()