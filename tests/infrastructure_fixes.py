"""
Test Infrastructure Fixes for LeanVibe Agent Hive 2.0

This module contains fixes for common test infrastructure issues:
1. Database compatibility (SQLite vs PostgreSQL)
2. Async test patterns and fixtures
3. Mock improvements for better integration testing
4. Test isolation and cleanup

Usage:
    python tests/infrastructure_fixes.py --check    # Check issues
    python tests/infrastructure_fixes.py --fix      # Apply fixes
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class TestInfrastructureFixer:
    """Fixes common test infrastructure issues."""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def check_database_compatibility(self) -> List[str]:
        """Check for database compatibility issues."""
        issues = []
        
        # Check for ARRAY type usage with SQLite
        db_files = list(Path("app/models").glob("*.py"))
        for file_path in db_files:
            content = file_path.read_text()
            if "ARRAY(" in content and "sqlite" in str(file_path).lower():
                issues.append(f"ARRAY type used in {file_path} - incompatible with SQLite tests")
        
        return issues
    
    def check_async_patterns(self) -> List[str]:
        """Check for problematic async test patterns."""
        issues = []
        
        test_files = list(Path("tests").glob("**/*.py"))
        for file_path in test_files:
            if file_path.name.startswith("test_"):
                content = file_path.read_text()
                
                # Check for missing async/await patterns
                if "def test_" in content and "async def test_" not in content:
                    if "await " in content:
                        issues.append(f"Non-async test function with await in {file_path}")
                
                # Check for improper fixture usage
                if "@pytest.fixture" in content and "async" not in content:
                    if "yield" in content or "await" in content:
                        issues.append(f"Sync fixture with async patterns in {file_path}")
        
        return issues
    
    def check_import_errors(self) -> List[str]:
        """Check for common import errors in tests."""
        issues = []
        
        # Common missing imports/classes identified from test failures
        missing_items = [
            ("app.core.health_monitor", "CheckType"),
            ("app.core.performance_benchmarks", "PerformanceBenchmarkSuite"),
            ("app.models.workflow", "WorkflowStep"),
            ("tests.factories", "create_mock_embedding_service"),
            ("app.observability.hooks", "ObservabilityHook")
        ]
        
        for module_path, item_name in missing_items:
            try:
                # Try to import and check if item exists
                import importlib
                module = importlib.import_module(module_path)
                if not hasattr(module, item_name):
                    issues.append(f"Missing {item_name} in {module_path}")
            except ImportError:
                issues.append(f"Cannot import {module_path}")
        
        return issues
    
    def fix_pydantic_regex_issues(self) -> None:
        """Fix Pydantic regex parameter issues (v1 -> v2 migration)."""
        python_files = list(Path("app").glob("**/*.py"))
        python_files.extend(list(Path("tests").glob("**/*.py")))
        
        fixes_count = 0
        for file_path in python_files:
            content = file_path.read_text()
            if "pattern=" in content:
                new_content = content.replace("pattern=", "pattern=")
                file_path.write_text(new_content)
                fixes_count += 1
                self.fixes_applied.append(f"Fixed Pydantic regex in {file_path}")
        
        logger.info(f"Fixed Pydantic regex issues in {fixes_count} files")
    
    def create_enhanced_conftest(self) -> None:
        """Create enhanced conftest.py with better test infrastructure."""
        conftest_content = '''"""
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
'''
        
        # Write enhanced conftest
        conftest_path = Path("tests/conftest_enhanced.py")
        conftest_path.write_text(conftest_content)
        self.fixes_applied.append("Created enhanced conftest.py")
    
    def create_database_test_utils(self) -> None:
        """Create database testing utilities."""
        utils_content = '''"""
Database testing utilities for LeanVibe Agent Hive 2.0.

Provides utilities for:
- Database schema validation
- Test data factories
- Transaction management
- Performance testing
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Type
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect

from app.models.agent import Agent, AgentType, AgentStatus
from app.models.session import Session, SessionType, SessionStatus
from app.models.task import Task, TaskType, TaskStatus, TaskPriority
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.context import Context, ContextType


class DatabaseTestUtils:
    """Utilities for database testing."""
    
    @staticmethod
    async def create_test_agent(
        session: AsyncSession,
        name: str = "Test Agent",
        agent_type: AgentType = AgentType.CLAUDE,
        **kwargs
    ) -> Agent:
        """Create a test agent with sensible defaults."""
        agent_data = {
            "name": name,
            "type": agent_type,
            "role": kwargs.get("role", "test_role"),
            "capabilities": kwargs.get("capabilities", [
                {
                    "name": "test_capability",
                    "description": "Test capability",
                    "confidence_level": 0.8,
                    "specialization_areas": ["testing"]
                }
            ]),
            "status": kwargs.get("status", AgentStatus.ACTIVE),
            "config": kwargs.get("config", {"test": True}),
            **kwargs
        }
        
        agent = Agent(**agent_data)
        session.add(agent)
        await session.flush()
        return agent
    
    @staticmethod
    async def create_test_session(
        db_session: AsyncSession,
        name: str = "Test Session",
        lead_agent: Optional[Agent] = None,
        **kwargs
    ) -> Session:
        """Create a test session with sensible defaults."""
        if lead_agent is None:
            lead_agent = await DatabaseTestUtils.create_test_agent(db_session)
        
        session_data = {
            "name": name,
            "description": kwargs.get("description", "Test session"),
            "session_type": kwargs.get("session_type", SessionType.FEATURE_DEVELOPMENT),
            "status": kwargs.get("status", SessionStatus.ACTIVE),
            "participant_agents": kwargs.get("participant_agents", [lead_agent.id]),
            "lead_agent_id": lead_agent.id,
            "objectives": kwargs.get("objectives", ["Test objective"]),
            **kwargs
        }
        
        session = Session(**session_data)
        db_session.add(session)
        await db_session.flush()
        return session
    
    @staticmethod
    async def create_test_task(
        session: AsyncSession,
        title: str = "Test Task",
        assigned_agent: Optional[Agent] = None,
        **kwargs
    ) -> Task:
        """Create a test task with sensible defaults."""
        if assigned_agent is None:
            assigned_agent = await DatabaseTestUtils.create_test_agent(session)
        
        task_data = {
            "title": title,
            "description": kwargs.get("description", "Test task description"),
            "task_type": kwargs.get("task_type", TaskType.FEATURE_DEVELOPMENT),
            "status": kwargs.get("status", TaskStatus.PENDING),
            "priority": kwargs.get("priority", TaskPriority.MEDIUM),
            "assigned_agent_id": assigned_agent.id,
            "required_capabilities": kwargs.get("required_capabilities", ["test_capability"]),
            "estimated_effort": kwargs.get("estimated_effort", 60),
            "context": kwargs.get("context", {"test": True}),
            **kwargs
        }
        
        task = Task(**task_data)
        session.add(task)
        await session.flush()
        return task
    
    @staticmethod
    async def create_test_workflow(
        session: AsyncSession,
        name: str = "Test Workflow",
        **kwargs
    ) -> Workflow:
        """Create a test workflow with sensible defaults."""
        workflow_data = {
            "name": name,
            "description": kwargs.get("description", "Test workflow"),
            "status": kwargs.get("status", WorkflowStatus.CREATED),
            "priority": kwargs.get("priority", WorkflowPriority.MEDIUM),
            "definition": kwargs.get("definition", {"type": "sequential", "steps": []}),
            "context": kwargs.get("context", {"test": True}),
            "variables": kwargs.get("variables", {}),
            "estimated_duration": kwargs.get("estimated_duration", 120),
            **kwargs
        }
        
        workflow = Workflow(**workflow_data)
        session.add(workflow)
        await session.flush()
        return workflow
    
    @staticmethod
    async def verify_database_schema(session: AsyncSession) -> Dict[str, Any]:
        """Verify database schema integrity."""
        inspector = inspect(session.bind)
        
        # Get table names
        tables = await session.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            if "postgresql" in str(session.bind.url)
            else "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        
        table_names = [row[0] for row in tables.fetchall()]
        
        schema_info = {
            "tables": table_names,
            "table_count": len(table_names),
            "expected_tables": [
                "agents", "sessions", "tasks", "workflows", "contexts",
                "agent_events", "system_checkpoints", "conversations"
            ]
        }
        
        # Check for missing tables
        missing_tables = set(schema_info["expected_tables"]) - set(table_names)
        schema_info["missing_tables"] = list(missing_tables)
        schema_info["schema_valid"] = len(missing_tables) == 0
        
        return schema_info
    
    @staticmethod
    async def cleanup_test_data(session: AsyncSession, created_objects: List[Any]) -> None:
        """Clean up test data."""
        for obj in reversed(created_objects):  # Reverse order for FK constraints
            try:
                await session.delete(obj)
            except Exception as e:
                print(f"Warning: Failed to delete {obj}: {e}")
        
        try:
            await session.flush()
        except Exception as e:
            print(f"Warning: Failed to flush deletions: {e}")


class PerformanceTestUtils:
    """Utilities for database performance testing."""
    
    @staticmethod
    async def measure_query_time(session: AsyncSession, query_func) -> float:
        """Measure query execution time in milliseconds."""
        start_time = asyncio.get_event_loop().time()
        await query_func(session)
        end_time = asyncio.get_event_loop().time()
        return (end_time - start_time) * 1000  # Convert to ms
    
    @staticmethod
    async def create_bulk_test_data(
        session: AsyncSession, 
        count: int = 1000
    ) -> Dict[str, List[Any]]:
        """Create bulk test data for performance testing."""
        agents = []
        sessions = []
        tasks = []
        
        # Create agents
        for i in range(count // 10):  # 10% agents
            agent = Agent(
                name=f"Perf Agent {i}",
                type=AgentType.CLAUDE,
                role=f"test_role_{i}",
                capabilities=[{
                    "name": f"capability_{i}",
                    "confidence_level": 0.8
                }],
                status=AgentStatus.ACTIVE
            )
            agents.append(agent)
            session.add(agent)
        
        await session.flush()  # Get agent IDs
        
        # Create sessions
        for i in range(count // 5):  # 20% sessions
            test_session = Session(
                name=f"Perf Session {i}",
                session_type=SessionType.FEATURE_DEVELOPMENT,
                status=SessionStatus.ACTIVE,
                lead_agent_id=agents[i % len(agents)].id,
                participant_agents=[agents[i % len(agents)].id]
            )
            sessions.append(test_session)
            session.add(test_session)
        
        await session.flush()
        
        # Create tasks
        for i in range(count):  # 100% tasks
            task = Task(
                title=f"Perf Task {i}",
                description=f"Performance test task {i}",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                assigned_agent_id=agents[i % len(agents)].id,
                estimated_effort=60
            )
            tasks.append(task)
            session.add(task)
        
        await session.flush()
        
        return {
            "agents": agents,
            "sessions": sessions,
            "tasks": tasks
        }
'''
        
        utils_path = Path("tests/utils/database_test_utils.py")
        utils_path.parent.mkdir(exist_ok=True)
        utils_path.write_text(utils_content)
        self.fixes_applied.append("Created database test utilities")
    
    def run_fixes(self) -> None:
        """Run all infrastructure fixes."""
        print("🔧 Applying test infrastructure fixes...")
        
        try:
            self.fix_pydantic_regex_issues()
            self.create_enhanced_conftest()
            self.create_database_test_utils()
            
            print(f"✅ Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"   - {fix}")
                
        except Exception as e:
            print(f"❌ Error applying fixes: {e}")
            raise
    
    def check_all_issues(self) -> Dict[str, List[str]]:
        """Check all infrastructure issues."""
        print("🔍 Checking test infrastructure issues...")
        
        issues = {
            "database": self.check_database_compatibility(),
            "async_patterns": self.check_async_patterns(),  
            "imports": self.check_import_errors()
        }
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        print(f"Found {total_issues} total issues:")
        
        for category, issue_list in issues.items():
            if issue_list:
                print(f"\n{category.upper()} ISSUES ({len(issue_list)}):")
                for issue in issue_list:
                    print(f"   - {issue}")
        
        return issues


def main():
    """Main function for running infrastructure fixes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix test infrastructure issues")
    parser.add_argument("--check", action="store_true", help="Check for issues")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    
    args = parser.parse_args()
    
    fixer = TestInfrastructureFixer()
    
    if args.check:
        issues = fixer.check_all_issues()
        return len(issues) > 0
    
    if args.fix:
        fixer.run_fixes()
        return True
    
    # Default: check and fix
    issues = fixer.check_all_issues()
    if any(issues.values()):
        fixer.run_fixes()
    
    return True


if __name__ == "__main__":
    main()