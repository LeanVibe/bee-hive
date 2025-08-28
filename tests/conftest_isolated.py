"""
Isolated pytest configuration for Epic B Phase 2: Test Infrastructure Stabilization.

This configuration provides completely isolated test fixtures that avoid complex dependencies
while enabling comprehensive testing of the LeanVibe Agent Hive 2.0 system.
"""

import asyncio
import os
import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from dataclasses import dataclass, field

# Set test environment variables for isolation
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use test database
os.environ["DEBUG"] = "true"
os.environ["TESTING"] = "true"

# Isolated Test Database Configuration
@pytest_asyncio.fixture
async def isolated_test_db():
    """Provide completely isolated in-memory SQLite database for testing."""
    import aiosqlite
    
    # Create in-memory database
    db = await aiosqlite.connect(":memory:")
    
    # Basic table creation for agent testing
    await db.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            status TEXT NOT NULL,
            current_task TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    await db.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            priority TEXT DEFAULT 'MEDIUM',
            status TEXT DEFAULT 'PENDING',
            assigned_agent_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    await db.commit()
    
    yield db
    
    await db.close()


# Mock Redis Client for Testing
@pytest.fixture
def isolated_redis():
    """Provide isolated Redis client using fakeredis."""
    try:
        import fakeredis.aioredis
        return fakeredis.aioredis.FakeRedis(decode_responses=True)
    except ImportError:
        # Fallback to mock if fakeredis not available
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=True)
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.exists = AsyncMock(return_value=False)
        return mock_redis


# Test Data Factories
@dataclass
class TestAgent:
    """Isolated test agent data class."""
    id: str
    role: str = "BACKEND_DEVELOPER"
    status: str = "ACTIVE"
    current_task: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "status": self.status,
            "current_task": self.current_task,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


@dataclass 
class TestTask:
    """Isolated test task data class."""
    id: str
    title: str
    description: str = ""
    priority: str = "MEDIUM"
    status: str = "PENDING"
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat()
        }


# Test Data Factory Functions
@pytest.fixture
def agent_factory():
    """Factory for creating test agents."""
    def _create_agent(agent_id: str = None, **kwargs) -> TestAgent:
        agent_id = agent_id or f"test-agent-{datetime.utcnow().timestamp()}"
        return TestAgent(id=agent_id, **kwargs)
    return _create_agent


@pytest.fixture
def task_factory():
    """Factory for creating test tasks."""
    def _create_task(task_id: str = None, **kwargs) -> TestTask:
        task_id = task_id or f"test-task-{datetime.utcnow().timestamp()}"
        title = kwargs.get("title", f"Test Task {task_id}")
        return TestTask(id=task_id, title=title, **kwargs)
    return _create_task


# Isolated HTTP Client for API Testing
@pytest_asyncio.fixture
async def isolated_http_client():
    """Provide isolated HTTP client for API testing."""
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI(title="Test API", version="2.0.0-test")
    
    # Basic health endpoint
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "environment": "testing"}
    
    # Basic agents endpoints
    @app.get("/api/v1/agents/")
    def list_agents():
        return {"agents": [], "total": 0}
    
    @app.post("/api/v1/agents/", status_code=201)
    def create_agent(agent_data: dict):
        return {
            "id": f"agent-{len(agent_data)}",
            "status": "ACTIVE",
            **agent_data
        }
    
    # Use httpx AsyncClient for proper async testing
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# Mock Orchestrator for Integration Testing
@pytest.fixture
def isolated_orchestrator():
    """Provide completely isolated orchestrator mock."""
    orchestrator = AsyncMock()
    
    # Mock storage
    orchestrator._agents = {}
    orchestrator._tasks = {}
    orchestrator._metrics = {
        "agents_spawned": 0,
        "tasks_completed": 0,
        "errors": 0,
        "uptime": 0
    }
    
    # Mock methods with realistic behavior
    async def mock_register_agent(role: str, **kwargs) -> str:
        agent_id = f"agent-{len(orchestrator._agents) + 1}"
        orchestrator._agents[agent_id] = {
            "id": agent_id,
            "role": role,
            "status": "ACTIVE",
            **kwargs
        }
        orchestrator._metrics["agents_spawned"] += 1
        return agent_id
    
    async def mock_get_agent(agent_id: str) -> Optional[dict]:
        return orchestrator._agents.get(agent_id)
    
    async def mock_list_agents() -> list:
        return list(orchestrator._agents.values())
    
    async def mock_health_check() -> dict:
        return {
            "status": "healthy",
            "agents": len(orchestrator._agents),
            "tasks": len(orchestrator._tasks),
            "metrics": orchestrator._metrics
        }
    
    # Assign mock methods
    orchestrator.register_agent = mock_register_agent
    orchestrator.get_agent = mock_get_agent
    orchestrator.list_agents = mock_list_agents
    orchestrator.health_check = mock_health_check
    
    return orchestrator


# Performance Testing Configuration
@pytest.fixture
def performance_test_config():
    """Configuration optimized for testing performance."""
    return {
        "max_concurrent_agents": 10,  # Reduced for testing
        "max_concurrent_tasks": 20,   # Reduced for testing
        "timeout_seconds": 5,         # Shorter timeout for tests
        "memory_limit_mb": 100,       # Lower memory limit for tests
        "test_duration_seconds": 30   # Short test duration
    }


# Test Environment Setup
@pytest.fixture(autouse=True)
def isolated_test_environment():
    """Set up completely isolated test environment."""
    with tempfile.TemporaryDirectory(prefix="epic_b_test_") as temp_dir:
        # Set environment variables
        test_env = {
            "WORKSPACE_DIR": temp_dir,
            "LOGS_DIR": str(Path(temp_dir) / "logs"),
            "CHECKPOINTS_DIR": str(Path(temp_dir) / "checkpoints"),
            "TESTING": "true",
            "ENVIRONMENT": "testing"
        }
        
        # Create directories
        Path(test_env["LOGS_DIR"]).mkdir(exist_ok=True)
        Path(test_env["CHECKPOINTS_DIR"]).mkdir(exist_ok=True)
        
        # Set environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        yield temp_dir
        
        # Restore environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# Async Event Loop
@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Provide event loop for async testing."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Coverage and Quality Gates Configuration
@pytest.fixture
def coverage_config():
    """Configuration for coverage testing."""
    return {
        "target_coverage": 90,  # 90% as specified in pyproject.toml
        "critical_modules": [
            "app.core",
            "app.api",
            "app.agents"
        ],
        "exclude_patterns": [
            "*/tests/*",
            "*/migrations/*",
            "*/__pycache__/*"
        ]
    }


# Parallel Testing Support
@pytest.fixture
def parallel_test_config():
    """Configuration for parallel test execution."""
    import multiprocessing
    return {
        "max_workers": min(4, multiprocessing.cpu_count()),  # Limit for stability
        "chunk_size": 10,  # Tests per chunk
        "timeout_per_test": 30  # Seconds
    }