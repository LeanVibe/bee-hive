"""
Pytest configuration and fixtures for LeanVibe Agent Hive 2.0 tests.

Provides async database sessions, test clients, and mock data
for comprehensive testing of the multi-agent system.

EMERGENCY FIX: Completely redesigned to avoid all PostgreSQL/SQLite compatibility issues
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock
import random

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import MetaData, create_engine

# SKIP all complex model imports to avoid JSONB issues
# Import only what we absolutely need
from app.main import create_app
from app.core.redis import get_redis, get_message_broker, get_session_cache

# Simple test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine - NO TABLE CREATION to avoid JSONB issues."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # DO NOT create tables - this avoids all JSONB compatibility issues
    # Tests that need specific tables should mock or create minimal schemas
    
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
def mock_redis():
    """Mock Redis client for testing with async pubsub."""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.xadd.return_value = "1234567890-0"
    mock_redis.xreadgroup.return_value = []
    mock_redis.publish.return_value = 1
    mock_redis.setex.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1

    class _PubSub:
        async def subscribe(self, *channels):  # type: ignore[no-redef]
            return True

        async def listen(self):  # async generator
            if False:
                yield None  # pragma: no cover
            return

    mock_redis.pubsub.return_value = _PubSub()
    return mock_redis


@pytest_asyncio.fixture
def mock_message_broker(mock_redis):
    """Mock message broker for testing."""
    from app.core.redis import AgentMessageBroker
    return AgentMessageBroker(mock_redis)


@pytest_asyncio.fixture
def mock_session_cache(mock_redis):
    """Mock session cache for testing."""
    from app.core.redis import SessionCache
    return SessionCache(mock_redis)


@pytest_asyncio.fixture
async def test_app(mock_redis, mock_message_broker, mock_session_cache):
    """Create test FastAPI application with mocked dependencies and minimal surface.
    - Remove enterprise SecurityMiddleware wrapper
    - Seed core redis client to avoid init requirement
    - Stub coordination dashboard data for compat endpoint
    - Monkeypatch SecurityMiddleware.__call__ to pass-through
    """
    app = create_app()

    # Remove SecurityMiddleware even if wrapped via BaseHTTPMiddleware
    try:
        from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTP
        from app.core.enterprise_security_system import SecurityMiddleware as _SecMW  # type: ignore
        filtered = []
        for m in app.user_middleware:
            if m.cls is _BaseHTTP and isinstance(m.options.get('dispatch'), _SecMW):
                continue
            filtered.append(m)
        app.user_middleware = filtered
        app.middleware_stack = app.build_middleware_stack()
    except Exception:
        pass

    # Monkeypatch SecurityMiddleware.__call__ to a no-op in tests
    try:
        from app.core.enterprise_security_system import SecurityMiddleware as _SecMW  # type: ignore

        async def _noop(self, request, call_next):  # type: ignore[no-redef]
            return await call_next(request)

        _SecMW.__call__ = _noop  # type: ignore[assignment]
    except Exception:
        pass

    # Seed core redis client so get_redis() works
    try:
        import app.core.redis as core_redis
        core_redis._redis_client = mock_redis  # type: ignore[attr-defined]
    except Exception:
        pass

    # Override Redis-related dependencies to mocks
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_message_broker] = lambda: mock_message_broker
    app.dependency_overrides[get_session_cache] = lambda: mock_session_cache

    # Stub coordination dashboard data source used by /dashboard/api/live-data
    try:
        from app.core.coordination_dashboard import coordination_dashboard

        async def _fake_dashboard_data():
            return {
                "metrics": {
                    "active_projects": 0,
                    "active_agents": 0,
                    "agent_utilization": 0.0,
                    "completed_tasks": 0,
                    "active_conflicts": 0,
                    "system_efficiency": 0.0,
                    "system_status": "healthy",
                    "last_updated": "1970-01-01T00:00:00Z",
                },
                "agent_activities": [],
                "project_snapshots": [],
            }

        coordination_dashboard.get_dashboard_data = _fake_dashboard_data  # type: ignore[attr-defined]
    except Exception:
        pass

    yield app

    # Clear overrides
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
def test_client(test_app):
    """Create test client for synchronous requests."""
    return TestClient(test_app)


@pytest_asyncio.fixture
async def async_test_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for async requests."""
    from httpx import ASGITransport
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8000", headers={"host": "localhost:8000"}) as client:
        yield client


# Simplified test data fixtures using mocks
@pytest_asyncio.fixture
def sample_agent():
    """Create a mock sample agent for testing."""
    from unittest.mock import Mock
    import uuid
    
    agent = Mock()
    agent.id = uuid.uuid4()
    agent.name = "Test Agent"
    agent.type = "CLAUDE" 
    agent.role = "test_role"
    agent.capabilities = [
        {
            "name": "test_capability",
            "description": "Test capability description",
            "confidence_level": 0.8,
            "specialization_areas": ["testing"]
        }
    ]
    agent.status = "active"
    agent.config = {"test": True}
    
    return agent


# All mock fixtures to avoid database issues
@pytest_asyncio.fixture
def sample_session(sample_agent):
    """Create a mock sample session for testing.""" 
    from unittest.mock import Mock
    import uuid
    
    session = Mock()
    session.id = uuid.uuid4()
    session.name = "Test Session"
    session.description = "Test session description"
    session.session_type = "FEATURE_DEVELOPMENT"
    session.status = "ACTIVE"
    session.participant_agents = [sample_agent.id]
    session.lead_agent_id = sample_agent.id
    session.objectives = ["Test objective 1", "Test objective 2"]
    
    return session


@pytest_asyncio.fixture  
def sample_task(sample_agent):
    """Create a mock sample task for testing."""
    from unittest.mock import Mock
    import uuid
    
    task = Mock()
    task.id = uuid.uuid4()
    task.title = "Test Task"
    task.description = "Test task description"
    task.task_type = "FEATURE_DEVELOPMENT"
    task.status = "PENDING"
    task.priority = "MEDIUM"
    task.assigned_agent_id = sample_agent.id
    task.required_capabilities = ["test_capability"]
    task.estimated_effort = 60
    task.context = {"test": True}
    
    return task


@pytest_asyncio.fixture
def sample_workflow():
    """Create a mock sample workflow for testing."""
    from unittest.mock import Mock
    import uuid
    
    workflow = Mock()
    workflow.id = uuid.uuid4()
    workflow.name = "Test Workflow"
    workflow.description = "Test workflow description"
    workflow.status = "CREATED"
    workflow.priority = "MEDIUM"
    workflow.definition = {"type": "sequential", "steps": ["task1", "task2"]}
    workflow.context = {"project": "test"}
    workflow.variables = {"env": "testing"}
    workflow.estimated_duration = 120
    
    return workflow


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup minimal test environment variables."""
    os.environ.update({
        "DEBUG": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/1",
        "ANTHROPIC_API_KEY": "test-key-12345",
        "SECRET_KEY": "test-secret-key-for-testing-purposes-only",
        "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-purposes-only",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "ERROR",  # Reduce noise
        "TESTING": "true",
        "SKIP_DATABASE_INIT": "true",
        # Ensure deterministic behavior in stochastic tests
        "PYTHONHASHSEED": "0"
    })


@pytest.fixture(autouse=True, scope="session")
def _seed_rng_session():
    """Ensure deterministic RNG across stochastic tests (evolutionary, perf)."""
    random.seed(1337)
    if _np is not None:
        try:
            _np.random.seed(1337)
        except Exception:
            pass


# Test markers for categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance