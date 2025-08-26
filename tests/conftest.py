"""
Enhanced pytest configuration for LeanVibe Agent Hive 2.0 tests.

Provides comprehensive fixtures for infrastructure testing with proper isolation and mocking.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path

# Set test environment variables
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["DEBUG"] = "true"

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
        DATABASE_URL = "sqlite+aiosqlite:///:memory:"
        REDIS_URL = "redis://localhost:6379/1"
        DEBUG = True
        ENVIRONMENT = "testing"
        SECRET_KEY = "test-secret-key"
        JWT_SECRET_KEY = "test-jwt-secret-key"
        
    return MockSettings()


@pytest.fixture
def mock_configuration():
    """Provide mock ApplicationConfiguration for testing."""
    from app.core.configuration_service import ApplicationConfiguration
    
    # Create a properly instantiated configuration object
    return ApplicationConfiguration(
        app_name="LeanVibe Agent Hive 2.0 Test",
        version="2.0-test",
        environment="testing",
        debug=True,
        log_level="INFO"
    )


@pytest_asyncio.fixture
async def test_db_session():
    """Provide an isolated test database session with minimal setup."""
    from unittest.mock import AsyncMock
    
    # Create a mock database session for isolated testing
    # This avoids complex SQLAlchemy model compatibility issues
    session = AsyncMock()
    session.add = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.query = AsyncMock()
    
    yield session


@pytest.fixture
def test_client():
    """Provide a test client for API testing."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    # Create minimal FastAPI app for testing
    app = FastAPI(title="Test App", version="1.0.0")
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    return TestClient(app)


@pytest_asyncio.fixture
async def async_test_client():
    """Provide an async test client for API testing."""
    import httpx
    from fastapi import FastAPI
    
    # Create minimal FastAPI app for testing
    app = FastAPI(title="Test App", version="1.0.0")
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    @app.get("/api/v1/agents/")
    def list_agents():
        return {"agents": []}
    
    @app.post("/api/v1/agents/", status_code=201)
    def create_agent(agent_data: dict):
        return {"id": "test-agent-id", **agent_data, "status": "INACTIVE"}
    
    @app.get("/api/v1/agents/{agent_id}")
    def get_agent(agent_id: str):
        if agent_id == "test-agent-id":
            return {"id": agent_id, "name": "Test Agent", "status": "ACTIVE"}
        return {"detail": "Agent not found"}
    
    # Use the correct httpx AsyncClient interface
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client for testing."""
    import fakeredis.aioredis
    
    # Use fakeredis for isolated Redis testing
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_orchestrator():
    """Provide a mock orchestrator for testing."""
    orchestrator = AsyncMock()
    orchestrator.register_agent = AsyncMock(return_value="test-agent-id")
    orchestrator.get_agent = AsyncMock(return_value={"id": "test-agent-id", "name": "Test Agent"})
    orchestrator.list_agents = AsyncMock(return_value=[])
    orchestrator.health_check = AsyncMock(return_value={"status": "healthy"})
    return orchestrator


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up isolated test environment for each test."""
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["WORKSPACE_DIR"] = temp_dir
        os.environ["LOGS_DIR"] = str(Path(temp_dir) / "logs")
        os.environ["CHECKPOINTS_DIR"] = str(Path(temp_dir) / "checkpoints")
        yield temp_dir


@pytest.fixture
def isolated_config():
    """Provide isolated configuration for testing."""
    from app.core.configuration_service import ApplicationConfiguration, Environment, LogLevel
    
    config = ApplicationConfiguration()
    config.environment = Environment.TESTING
    config.debug = True
    config.log_level = LogLevel.DEBUG
    return config


@pytest_asyncio.fixture
async def isolated_orchestrator():
    """Provide a fully isolated orchestrator for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    # Mock all external dependencies
    mock_orchestrator = AsyncMock()
    mock_orchestrator.database = AsyncMock()
    mock_orchestrator.redis = MagicMock()
    mock_orchestrator.message_broker = AsyncMock()
    mock_orchestrator.health_monitor = AsyncMock()
    
    # Mock common orchestrator methods
    mock_orchestrator.register_agent = AsyncMock(return_value="agent-123")
    mock_orchestrator.get_agent = AsyncMock()
    mock_orchestrator.list_agents = AsyncMock(return_value=[])
    mock_orchestrator.start = AsyncMock()
    mock_orchestrator.stop = AsyncMock()
    mock_orchestrator.health_check = AsyncMock(return_value={"status": "healthy"})
    
    return mock_orchestrator


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {
        "max_agents": 50,
        "concurrent_requests": 100,
        "timeout_seconds": 30,
        "memory_limit_mb": 512,
        "cpu_limit_percent": 80
    }


@pytest_asyncio.fixture
async def performance_orchestrator(performance_config):
    """Orchestrator configured for performance testing."""
    from unittest.mock import AsyncMock
    
    orchestrator = AsyncMock()
    orchestrator.config = performance_config
    orchestrator.metrics = {
        "agents_registered": 0,
        "tasks_completed": 0,
        "avg_response_time": 0.0,
        "error_rate": 0.0
    }
    
    return orchestrator


# Contract testing fixtures
@pytest.fixture
def contract_schemas():
    """Provide contract schemas for validation."""
    return {
        "agent_create_schema": {
            "type": "object",
            "required": ["name", "type", "role"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["CLAUDE", "OPENAI", "CUSTOM"]},
                "role": {"type": "string", "minLength": 1},
                "capabilities": {"type": "array"},
                "config": {"type": "object"}
            }
        },
        "agent_response_schema": {
            "type": "object",
            "required": ["id", "name", "type", "status"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "type": {"type": "string"},
                "status": {"type": "string", "enum": ["ACTIVE", "INACTIVE", "ERROR"]},
                "created_at": {"type": "string", "format": "date-time"}
            }
        }
    }


# Chaos testing fixtures
@pytest.fixture
def chaos_scenario_config():
    """Configuration for chaos engineering tests."""
    return {
        "failure_injection_rate": 0.1,  # 10% failure rate
        "recovery_timeout_seconds": 30,
        "max_retries": 3,
        "health_check_interval": 5
    }