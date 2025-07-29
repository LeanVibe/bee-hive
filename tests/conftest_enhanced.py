"""
Enhanced Pytest configuration for LeanVibe Agent Hive 2.0.

Provides improved fixtures for:
- Real PostgreSQL database testing (with fallback to SQLite)
- Redis test containers
- Better async test support
- Comprehensive mock factories
"""

import asyncio
import os
import sys
from pathlib import Path
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator, Optional
from unittest.mock import AsyncMock, MagicMock

# Test environment detection
POSTGRES_AVAILABLE = os.getenv("TEST_WITH_POSTGRES", "false").lower() == "true"
REDIS_AVAILABLE = os.getenv("TEST_WITH_REDIS", "false").lower() == "true"

# Database URLs
if POSTGRES_AVAILABLE:
    TEST_DATABASE_URL = os.getenv(
        "TEST_DATABASE_URL", 
        "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_leanvibe"
    )
else:
    TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Redis URL
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    yield loop
    
    # Cleanup
    try:
        loop.close()
    except Exception:
        pass


@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine with proper configuration."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool, NullPool
    from app.core.database import Base
    
    if POSTGRES_AVAILABLE:
        engine = create_async_engine(
            TEST_DATABASE_URL,
            poolclass=NullPool,
            echo=False,
        )
    else:
        engine = create_async_engine(
            TEST_DATABASE_URL,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
    except Exception as e:
        print(f"Warning: Database cleanup failed: {e}")


@pytest_asyncio.fixture
async def test_db_session(test_engine) -> AsyncGenerator:
    """Create test database session with proper transaction handling."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        # Start a transaction
        transaction = await session.begin()
        
        try:
            yield session
        finally:
            # Always rollback transaction
            try:
                await transaction.rollback()
            except Exception:
                pass


@pytest_asyncio.fixture
async def mock_redis():
    """Enhanced mock Redis client with better stream simulation."""
    mock_redis = AsyncMock()
    
    # Basic operations
    mock_redis.ping.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1
    mock_redis.publish.return_value = 1
    
    # Stream operations
    mock_redis.xadd.return_value = "1234567890-0"
    mock_redis.xreadgroup.return_value = []
    mock_redis.xlen.return_value = 0
    mock_redis.xgroup_create.return_value = True
    mock_redis.xgroup_destroy.return_value = 1
    
    # Hash operations
    mock_redis.hset.return_value = 1
    mock_redis.hget.return_value = None
    mock_redis.hgetall.return_value = {}
    mock_redis.hdel.return_value = 1
    
    # List operations
    mock_redis.lpush.return_value = 1
    mock_redis.rpush.return_value = 1
    mock_redis.lpop.return_value = None
    mock_redis.rpop.return_value = None
    mock_redis.llen.return_value = 0
    
    # Set operations
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.scard.return_value = 0
    
    return mock_redis


@pytest_asyncio.fixture
async def real_redis():
    """Real Redis connection for integration tests."""
    if not REDIS_AVAILABLE:
        pytest.skip("Redis not available for testing")
    
    import redis.asyncio as redis
    
    redis_client = redis.from_url(TEST_REDIS_URL)
    
    # Test connection
    try:
        await redis_client.ping()
    except Exception:
        pytest.skip("Cannot connect to Redis")
    
    yield redis_client
    
    # Cleanup
    try:
        await redis_client.flushdb()
        await redis_client.close()
    except Exception:
        pass


@pytest_asyncio.fixture
def mock_anthropic_client():
    """Enhanced mock Anthropic client."""
    mock_client = AsyncMock()
    
    # Mock message creation
    mock_response = AsyncMock()
    mock_response.content = [
        AsyncMock(text="Mock response from Claude", type="text")
    ]
    mock_response.model = "claude-3-sonnet-20240229"
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"
    mock_response.usage = AsyncMock(input_tokens=100, output_tokens=50)
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest_asyncio.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        "max_response_time_ms": 2000,
        "max_memory_usage_mb": 512,
        "max_cpu_usage_percent": 80,
        "concurrent_requests": 10,
        "test_duration_seconds": 30,
    }


@pytest_asyncio.fixture
def security_test_config():
    """Configuration for security testing."""
    return {
        "test_tokens": [
            "ghp_1234567890abcdef1234567890abcdef12345678",
            "invalid_token",
            "",
            "bearer_token_123"
        ],
        "test_payloads": [
            {"normal": "data"},
            {"<script>": "alert('xss')"},
            {"../../etc/passwd": "traversal"},
            {"'; DROP TABLE users; --": "sql_injection"}
        ],
        "rate_limit_requests": 100
    }


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration  
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.chaos = pytest.mark.chaos
pytest.mark.slow = pytest.mark.slow
pytest.mark.redis = pytest.mark.redis
pytest.mark.postgres = pytest.mark.postgres
pytest.mark.anthropic = pytest.mark.anthropic


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "unit: Fast unit tests with mocked dependencies"
    )
    config.addinivalue_line(
        "markers", 
        "integration: Tests that integrate multiple components"
    )
    config.addinivalue_line(
        "markers", 
        "e2e: End-to-end tests of complete workflows"
    )
    config.addinivalue_line(
        "markers", 
        "performance: Performance and load testing"
    )
    config.addinivalue_line(
        "markers", 
        "security: Security and authentication testing"  
    )
    config.addinivalue_line(
        "markers", 
        "chaos: Chaos engineering and resilience testing"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "chaos" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif str(item.fspath).endswith("test_*.py"):
            item.add_marker(pytest.mark.unit)
