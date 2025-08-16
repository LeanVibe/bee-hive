"""
Database configuration and session management for LeanVibe Agent Hive 2.0

Async SQLAlchemy setup with PostgreSQL and pgvector for multi-agent coordination.
Optimized for high-concurrency agent operations with connection pooling.
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

from .config import settings
from .logging_service import get_component_logger

logger = get_component_logger("database")

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_database_url() -> str:
    """Get the database URL with async driver."""
    return settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


async def create_engine() -> AsyncEngine:
    """Create async database engine with optimized settings for multi-agent usage."""
    
    engine = create_async_engine(
        get_database_url(),
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        echo=settings.DEBUG,
        future=True,
    )
    
    return engine


async def create_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create session factory for database operations."""
    global _engine
    
    if _engine is None:
        _engine = await create_engine()
    
    return async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )


async def init_database() -> None:
    """Initialize database connection and create tables if needed."""
    global _engine, _session_factory
    
    try:
        logger.info("🔌 Initializing database connection...")
        
        _engine = await create_engine()
        _session_factory = await create_session_factory()
        
        # Test database connection
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
            
        logger.info("✅ Database connection established")
        
        # Create tables if they don't exist (in production, use Alembic migrations)
        if settings.DEBUG:
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("📊 Database tables created")
    
    except Exception as e:
        logger.error("❌ Failed to initialize database", error=str(e))
        raise


async def close_database() -> None:
    """Close database connections gracefully."""
    global _engine
    
    if _engine:
        await _engine.dispose()
        logger.info("🔌 Database connections closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with automatic cleanup."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_session() as session:
        yield session


# Alias for consistency with other modules
get_async_session = get_session_dependency
get_db = get_session_dependency


async def get_db_session() -> AsyncSession:
    """Get database session for direct use (not as FastAPI dependency)."""
    if _session_factory is None:
        await init_database()
    
    return _session_factory()


class DatabaseHealthCheck:
    """Database health check utilities for monitoring."""
    
    @staticmethod
    async def check_connection() -> bool:
        """Check if database connection is healthy."""
        try:
            async with get_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    @staticmethod
    async def check_extensions() -> dict:
        """Check if required PostgreSQL extensions are available."""
        try:
            async with get_session() as session:
                # Check pgvector extension
                vector_result = await session.execute(
                    text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                )
                
                # Check uuid-ossp extension
                uuid_result = await session.execute(
                    text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp')")
                )
                
                return {
                    "pgvector": vector_result.scalar(),
                    "uuid-ossp": uuid_result.scalar()
                }
        except Exception as e:
            logger.error("Extension check failed", error=str(e))
            return {"pgvector": False, "uuid-ossp": False}
    
    @staticmethod
    async def get_connection_stats() -> dict:
        """Get database connection pool statistics."""
        global _engine
        
        if _engine is None:
            return {"status": "not_initialized"}
        
        pool = _engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalidated": pool.invalidated()
        }