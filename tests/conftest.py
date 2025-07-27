"""
Pytest configuration and fixtures for LeanVibe Agent Hive 2.0 tests.

Provides async database sessions, test clients, and mock data
for comprehensive testing of the multi-agent system.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import create_app
from app.core.database import Base, get_session_dependency
from app.core.redis import get_redis, get_message_broker, get_session_cache
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.session import Session, SessionStatus, SessionType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority


# Test database URL (in-memory SQLite for speed)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine."""
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
    async with AsyncClient(app=test_app, base_url="http://test") as client:
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
        participant_agents=[sample_agent.id],
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


@pytest_asyncio.fixture
async def complex_workflow_with_tasks(test_db_session: AsyncSession, sample_agent: Agent) -> tuple[Workflow, list[Task]]:
    """Create a complex workflow with multiple tasks and dependencies."""
    # Create workflow
    workflow = Workflow(
        name="Complex Test Workflow",
        description="A workflow with multiple tasks and dependencies",
        status=WorkflowStatus.READY,
        priority=WorkflowPriority.HIGH,
        definition={"type": "dag", "parallel_execution": True},
        context={"project": "integration_test"},
        estimated_duration=480
    )
    
    test_db_session.add(workflow)
    await test_db_session.flush()  # Get workflow ID
    
    # Create tasks
    tasks = []
    for i in range(4):
        task = Task(
            title=f"Test Task {i+1}",
            description=f"Task {i+1} description",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            assigned_agent_id=sample_agent.id if i % 2 == 0 else None,
            required_capabilities=["test_capability"],
            estimated_effort=120,
            context={"workflow_id": str(workflow.id), "task_index": i}
        )
        tasks.append(task)
        test_db_session.add(task)
    
    await test_db_session.flush()  # Get task IDs
    
    # Add tasks to workflow with dependencies
    task_ids = [task.id for task in tasks]
    workflow.task_ids = task_ids
    workflow.total_tasks = len(task_ids)
    
    # Set up dependencies: task2 depends on task1, task4 depends on task2 and task3
    workflow.dependencies = {
        str(task_ids[1]): [str(task_ids[0])],  # task2 depends on task1
        str(task_ids[3]): [str(task_ids[1]), str(task_ids[2])]  # task4 depends on task2 and task3
    }
    
    await test_db_session.commit()
    await test_db_session.refresh(workflow)
    for task in tasks:
        await test_db_session.refresh(task)
    
    return workflow, tasks


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ.update({
        "DEBUG": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": "redis://localhost:6379/1",
        "ANTHROPIC_API_KEY": "test-key",
        "SECRET_KEY": "test-secret-key",
        "ENVIRONMENT": "test"
    })


# GitHub Integration Test Fixtures
@pytest_asyncio.fixture
def mock_github_client():
    """Create mock GitHub API client."""
    client = AsyncMock()
    
    # Default mock responses
    client.get_user_info.return_value = {
        "login": "test_user",
        "id": 12345,
        "name": "Test User"
    }
    
    client.get_repository.return_value = {
        "id": 123456,
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "default_branch": "main",
        "clone_url": "https://github.com/owner/test-repo.git"
    }
    
    client.create_repository.return_value = {
        "id": 123456,
        "name": "test-repo",
        "full_name": "owner/test-repo",
        "clone_url": "https://github.com/owner/test-repo.git"
    }
    
    client.create_pull_request.return_value = {
        "number": 123,
        "html_url": "https://github.com/owner/test-repo/pull/123",
        "title": "Test PR",
        "state": "open"
    }
    
    client.get_repository_issues.return_value = [
        {
            "number": 1,
            "title": "Test Issue",
            "body": "Test issue description",
            "state": "open",
            "labels": [{"name": "bug"}]
        }
    ]
    
    client.create_issue.return_value = {
        "number": 1,
        "title": "Test Issue",
        "html_url": "https://github.com/owner/test-repo/issues/1"
    }
    
    return client


@pytest_asyncio.fixture
def sample_webhook_payloads():
    """Provide sample webhook payloads for testing."""
    return {
        "pull_request_opened": {
            "action": "opened",
            "pull_request": {
                "number": 123,
                "title": "Add new feature",
                "head": {"ref": "feature-branch"},
                "base": {"ref": "main"},
                "user": {"login": "test_user"}
            },
            "repository": {
                "full_name": "owner/repo",
                "default_branch": "main"
            }
        },
        "push_event": {
            "ref": "refs/heads/main",
            "commits": [
                {
                    "id": "abc123def456",
                    "message": "feat: add new feature",
                    "author": {"name": "Test User", "email": "test@example.com"}
                }
            ],
            "repository": {
                "full_name": "owner/repo"
            }
        },
        "issue_opened": {
            "action": "opened",
            "issue": {
                "number": 456,
                "title": "Bug in login system",
                "body": "Login form is not working properly",
                "labels": [{"name": "bug"}, {"name": "priority:high"}]
            },
            "repository": {
                "full_name": "owner/repo"
            }
        }
    }


@pytest_asyncio.fixture
def performance_benchmarks():
    """Provide performance benchmark targets."""
    return {
        "github_api_success_rate": 0.995,  # >99.5%
        "pull_request_creation_time_seconds": 30,  # <30 seconds
        "branch_merge_success_rate": 0.95,  # >95%
        "work_tree_isolation_effectiveness": 1.0,  # 100%
        "max_agents_concurrent": 50,
        "max_work_trees_per_agent": 5,
        "api_rate_limit_per_hour": 5000,
        "webhook_processing_time_ms": 500  # <500ms
    }


@pytest_asyncio.fixture
def security_test_data():
    """Provide security testing data."""
    return {
        "valid_tokens": [
            "ghp_1234567890abcdef1234567890abcdef12345678",
            "github_pat_11AAAAAAA0123456789abcdef0123456789abcdef",
            "ghs_abcdef1234567890abcdef1234567890abcdef12"
        ],
        "invalid_tokens": [
            "invalid_token_format",
            "bearer_xyz123",
            "ghp_short",
            ""
        ],
        "sensitive_context": {
            "token": "ghp_1234567890abcdef1234567890abcdef12345678",
            "password": "secret123",
            "api_key": "sk-1234567890abcdef",
            "webhook_secret": "webhook_secret_123",
            "user_id": "user_456",  # Not sensitive
            "public_data": "visible_information"  # Not sensitive
        },
        "permission_scenarios": [
            {
                "name": "read_only_agent",
                "permissions": {"read": "read", "metadata": "read"},
                "allowed_operations": ["read_repository"],
                "blocked_operations": ["create_pull_request", "push_commits"]
            },
            {
                "name": "developer_agent", 
                "permissions": {
                    "read": "read", "contents": "write", 
                    "issues": "write", "pull_requests": "write"
                },
                "allowed_operations": [
                    "read_repository", "create_branch", "create_pull_request",
                    "push_commits", "create_issue"
                ],
                "blocked_operations": ["delete_repository", "manage_webhooks"]
            }
        ]
    }


# Custom test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.redis = pytest.mark.redis
pytest.mark.postgres = pytest.mark.postgres
pytest.mark.anthropic = pytest.mark.anthropic
pytest.mark.github = pytest.mark.github
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security