"""
Simplified pytest configuration for LeanVibe Agent Hive 2.0 tests.

Provides basic fixtures for infrastructure testing with minimal dependencies.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator

# Basic async event loop fixture
@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    class MockSettings:
        DATABASE_URL = "postgresql+asyncpg://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive_test"
        REDIS_URL = "redis://localhost:6379/1"
        DEBUG = True
        ENVIRONMENT = "test"
        SECRET_KEY = "test-secret-key"
        JWT_SECRET_KEY = "test-jwt-secret-key"
        
    return MockSettings()


@pytest_asyncio.fixture
async def test_db_session():
    """Provide a test database session."""
    from app.core.database import init_database, get_session, close_database
    
    # Initialize database for testing
    await init_database()
    
    try:
        async with get_session() as session:
            yield session
    finally:
        await close_database()


@pytest.fixture
def test_client():
    """Provide a test client for API testing."""
    from fastapi.testclient import TestClient
    from app.main import create_app
    
    app = create_app()
    return TestClient(app)