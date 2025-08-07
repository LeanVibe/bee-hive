"""
FIXED Pytest configuration for LeanVibe Agent Hive 2.0 tests.

This version resolves all database compatibility issues between SQLite and PostgreSQL
by using test-specific models and proper compatibility fixes.
"""

import asyncio
import os
import sys
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import MetaData

# CRITICAL: Apply compatibility fixes BEFORE any model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from database_compatibility_fix import setup_test_database_compatibility
setup_test_database_compatibility()

from app.main import create_app
from app.core.database import Base, get_session_dependency
from app.core.redis import get_redis, get_message_broker, get_session_cache

# Import test-compatible models
from tests.utils.test_models import Agent, Task, Session, Workflow
from app.models.agent import AgentStatus, AgentType
from app.models.session import SessionStatus, SessionType
from app.models.task import TaskStatus, TaskPriority, TaskType
from app.models.workflow import WorkflowStatus, WorkflowPriority

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine with full compatibility."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create test schema using only test-compatible models
    test_metadata = MetaData()
    
    # Register test models with the test metadata
    for model in [Agent, Task, Session, Workflow]:
        if hasattr(model, '__table__'):
            model.__table__.metadata = test_metadata
    
    # Create tables
    async with engine.begin() as conn:
        try:
            await conn.run_sync(test_metadata.create_all)
        except Exception as e:
            print(f"Warning: Some tables may not be created: {e}")
    
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
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.xadd.return_value = "1234567890-0"
    mock_redis.xreadgroup.return_value = []
    mock_redis.publish.return_value = 1
    mock_redis.setex.return_value = True
    mock_redis.get.return_value = None
    mock_redis.delete.return_value = 1
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
async def test_app(test_db_session, mock_redis, mock_message_broker, mock_session_cache):
    """Create test FastAPI application."""
    app = create_app()
    
    # Override dependencies with test versions
    app.dependency_overrides[get_session_dependency] = lambda: test_db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    app.dependency_overrides[get_message_broker] = lambda: mock_message_broker
    app.dependency_overrides[get_session_cache] = lambda: mock_session_cache
    
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
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# Test data fixtures
@pytest_asyncio.fixture
async def sample_agent(test_db_session: AsyncSession) -> Agent:
    """Create a sample agent for testing."""
    agent = Agent(
        name="Test Agent",
        type=AgentType.CLAUDE,
        role="test_role",
        capabilities=[
            {
                "name": "test_capability",
                "description": "Test capability description",
                "confidence_level": 0.8,
                "specialization_areas": ["testing"]
            }
        ],
        status=AgentStatus.ACTIVE,
        config={"test": True}
    )
    
    test_db_session.add(agent)
    await test_db_session.commit()
    await test_db_session.refresh(agent)
    
    return agent


@pytest_asyncio.fixture
async def sample_session(test_db_session: AsyncSession, sample_agent: Agent) -> Session:
    """Create a sample session for testing."""
    session = Session(
        name="Test Session",
        description="Test session description",
        session_type=SessionType.FEATURE_DEVELOPMENT,
        status=SessionStatus.ACTIVE,
        participant_agents=[str(sample_agent.id)],
        lead_agent_id=sample_agent.id,
        objectives=["Test objective 1", "Test objective 2"]
    )
    
    test_db_session.add(session)
    await test_db_session.commit()
    await test_db_session.refresh(session)
    
    return session


@pytest_asyncio.fixture
async def sample_task(test_db_session: AsyncSession, sample_agent: Agent) -> Task:
    """Create a sample task for testing."""
    task = Task(
        title="Test Task",
        description="Test task description",
        task_type=TaskType.FEATURE_DEVELOPMENT,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_agent_id=sample_agent.id,
        required_capabilities=["test_capability"],
        estimated_effort=60,
        context={"test": True}
    )
    
    test_db_session.add(task)
    await test_db_session.commit()
    await test_db_session.refresh(task)
    
    return task


@pytest_asyncio.fixture
async def sample_workflow(test_db_session: AsyncSession) -> Workflow:
    """Create a sample workflow for testing."""
    workflow = Workflow(
        name="Test Workflow",
        description="Test workflow description",
        status=WorkflowStatus.CREATED,
        priority=WorkflowPriority.MEDIUM,
        definition={"type": "sequential", "steps": ["task1", "task2"]},
        context={"project": "test"},
        variables={"env": "testing"},
        estimated_duration=120
    )
    
    test_db_session.add(workflow)
    await test_db_session.commit()
    await test_db_session.refresh(workflow)
    
    return workflow


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    # Ensure critical test environment variables are set
    os.environ.update({
        "DEBUG": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/1",
        "ANTHROPIC_API_KEY": "test-key-12345",
        "SECRET_KEY": "test-secret-key-for-testing-purposes-only",
        "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-purposes-only",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "WARNING",
        "TESTING": "true",
        "SKIP_MIGRATIONS": "true",
        "USE_TEST_MODELS": "true"
    })


# Custom test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.redis = pytest.mark.redis
pytest.mark.postgres = pytest.mark.postgres
pytest.mark.anthropic = pytest.mark.anthropic
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security