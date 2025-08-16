"""
Project Index Testing Configuration and Fixtures

Provides shared testing infrastructure for comprehensive Project Index testing,
including database setup, mock data, performance monitoring, and test utilities.
"""

import asyncio
import uuid
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from redis.asyncio import Redis

from app.core.database import get_session
from app.core.redis import get_redis_client
from app.main import app

# Import Base and models from project index separately
from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus, SnapshotType
)
from app.core.database import Base
from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig, AnalysisConfiguration
from app.project_index.cache import CacheManager
from app.project_index.analyzer import CodeAnalyzer
from app.project_index.file_monitor import FileMonitor
from app.project_index.websocket_integration import ProjectIndexWebSocketManager


# ================== DATABASE SETUP ==================

@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create isolated test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    
    # Only create project index tables to avoid JSONB issues with other models
    async with engine.begin() as conn:
        # Create only project index related tables
        project_index_tables = [
            ProjectIndex.__table__,
            FileEntry.__table__,
            DependencyRelationship.__table__,
            AnalysisSession.__table__,
            IndexSnapshot.__table__
        ]
        
        for table in project_index_tables:
            await conn.run_sync(table.create, checkfirst=True)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create isolated test database session."""
    async_session = async_sessionmaker(
        test_db_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def test_redis():
    """Create mock Redis client for testing."""
    redis_mock = Mock(spec=Redis)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.incr = AsyncMock(return_value=1)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.publish = AsyncMock(return_value=1)
    redis_mock.subscribe = AsyncMock()
    redis_mock.unsubscribe = AsyncMock()
    redis_mock.pubsub = Mock()
    redis_mock.pubsub.return_value = Mock()
    
    return redis_mock


# ================== TEST CLIENT SETUP ==================

@pytest_asyncio.fixture(scope="function")
async def test_client(test_session, test_redis):
    """Create test HTTP client with database and Redis overrides."""
    
    def get_test_session():
        return test_session
    
    def get_test_redis():
        return test_redis
    
    app.dependency_overrides[get_session] = get_test_session
    app.dependency_overrides[get_redis_client] = get_test_redis
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    app.dependency_overrides.clear()


# ================== PROJECT FIXTURES ==================

@pytest_asyncio.fixture
async def sample_project_data():
    """Sample project data for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_project_"))
    
    # Create sample project structure
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    
    # Create sample files
    (temp_dir / "src" / "main.py").write_text("""
import os
import sys
from utils import helper_function

def main():
    print("Hello, World!")
    helper_function()

if __name__ == "__main__":
    main()
""")
    
    (temp_dir / "src" / "utils.py").write_text("""
import json
import datetime

def helper_function():
    return {"timestamp": datetime.datetime.now()}

class DataProcessor:
    def process(self, data):
        return json.dumps(data)
""")
    
    (temp_dir / "tests" / "test_main.py").write_text("""
import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Test main function
        pass
""")
    
    (temp_dir / "requirements.txt").write_text("""
requests==2.28.1
numpy==1.21.0
pandas==1.3.3
""")
    
    (temp_dir / "README.md").write_text("""
# Test Project

This is a test project for validating the Project Index system.
""")
    
    data = {
        "name": "Test Project",
        "description": "A test project for Project Index validation",
        "root_path": str(temp_dir),
        "git_repository_url": "https://github.com/test/test-project.git",
        "git_branch": "main",
        "git_commit_hash": "a" * 40,
        "configuration": {
            "languages": ["python"],
            "analysis_depth": 3,
            "enable_ai_analysis": True
        },
        "analysis_settings": {
            "extract_functions": True,
            "extract_classes": True,
            "analyze_imports": True
        },
        "file_patterns": {
            "include": ["**/*.py", "**/*.md", "**/*.txt"],
            "exclude": ["**/__pycache__/**", "**/.*"]
        },
        "ignore_patterns": {
            "directories": ["__pycache__", ".git", "node_modules"],
            "files": ["*.pyc", "*.pyo", ".DS_Store"]
        }
    }
    
    yield data, temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def test_project(test_session, sample_project_data):
    """Create a test project in the database."""
    data, temp_dir = sample_project_data
    
    project = ProjectIndex(
        name=data["name"],
        description=data["description"],
        root_path=data["root_path"],
        git_repository_url=data["git_repository_url"],
        git_branch=data["git_branch"],
        git_commit_hash=data["git_commit_hash"],
        status=ProjectStatus.ACTIVE,
        configuration=data["configuration"],
        analysis_settings=data["analysis_settings"],
        file_patterns=data["file_patterns"],
        ignore_patterns=data["ignore_patterns"],
        file_count=0,
        dependency_count=0
    )
    
    test_session.add(project)
    await test_session.commit()
    await test_session.refresh(project)
    
    return project


@pytest_asyncio.fixture
async def test_files(test_session, test_project):
    """Create test file entries."""
    files = []
    
    file_data = [
        {
            "relative_path": "src/main.py",
            "file_name": "main.py",
            "file_extension": ".py",
            "file_type": FileType.SOURCE,
            "language": "python",
            "file_size": 250,
            "line_count": 12,
            "is_binary": False,
            "content_preview": "import os\nimport sys\nfrom utils import helper_function"
        },
        {
            "relative_path": "src/utils.py", 
            "file_name": "utils.py",
            "file_extension": ".py",
            "file_type": FileType.SOURCE,
            "language": "python",
            "file_size": 180,
            "line_count": 10,
            "is_binary": False,
            "content_preview": "import json\nimport datetime"
        },
        {
            "relative_path": "tests/test_main.py",
            "file_name": "test_main.py",
            "file_extension": ".py",
            "file_type": FileType.TEST,
            "language": "python",
            "file_size": 120,
            "line_count": 8,
            "is_binary": False,
            "content_preview": "import unittest\nfrom src.main import main"
        },
        {
            "relative_path": "README.md",
            "file_name": "README.md",
            "file_extension": ".md",
            "file_type": FileType.DOCUMENTATION,
            "language": "markdown",
            "file_size": 85,
            "line_count": 4,
            "is_binary": False,
            "content_preview": "# Test Project\n\nThis is a test project"
        }
    ]
    
    for file_info in file_data:
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path=str(Path(test_project.root_path) / file_info["relative_path"]),
            relative_path=file_info["relative_path"],
            file_name=file_info["file_name"],
            file_extension=file_info["file_extension"],
            file_type=file_info["file_type"],
            language=file_info["language"],
            file_size=file_info["file_size"],
            line_count=file_info["line_count"],
            is_binary=file_info["is_binary"],
            content_preview=file_info["content_preview"],
            sha256_hash="a" * 64,
            last_modified=datetime.now(timezone.utc),
            indexed_at=datetime.now(timezone.utc)
        )
        
        test_session.add(file_entry)
        files.append(file_entry)
    
    await test_session.commit()
    
    for file_entry in files:
        await test_session.refresh(file_entry)
    
    return files


@pytest_asyncio.fixture
async def test_dependencies(test_session, test_files):
    """Create test dependency relationships."""
    dependencies = []
    
    # main.py imports from utils.py
    main_file = next(f for f in test_files if f.file_name == "main.py")
    utils_file = next(f for f in test_files if f.file_name == "utils.py")
    test_file = next(f for f in test_files if f.file_name == "test_main.py")
    
    dep_data = [
        {
            "source_file": main_file,
            "target_file": utils_file,
            "target_name": "utils",
            "dependency_type": DependencyType.IMPORT,
            "line_number": 3,
            "source_text": "from utils import helper_function",
            "is_external": False
        },
        {
            "source_file": main_file,
            "target_file": None,
            "target_name": "os",
            "dependency_type": DependencyType.IMPORT,
            "line_number": 1,
            "source_text": "import os",
            "is_external": True
        },
        {
            "source_file": main_file,
            "target_file": None,
            "target_name": "sys",
            "dependency_type": DependencyType.IMPORT,
            "line_number": 2,
            "source_text": "import sys",
            "is_external": True
        },
        {
            "source_file": utils_file,
            "target_file": None,
            "target_name": "json",
            "dependency_type": DependencyType.IMPORT,
            "line_number": 1,
            "source_text": "import json",
            "is_external": True
        },
        {
            "source_file": test_file,
            "target_file": main_file,
            "target_name": "src.main",
            "dependency_type": DependencyType.IMPORT,
            "line_number": 2,
            "source_text": "from src.main import main",
            "is_external": False
        }
    ]
    
    for dep_info in dep_data:
        dependency = DependencyRelationship(
            project_id=main_file.project_id,
            source_file_id=dep_info["source_file"].id,
            target_file_id=dep_info["target_file"].id if dep_info["target_file"] else None,
            target_name=dep_info["target_name"],
            dependency_type=dep_info["dependency_type"],
            line_number=dep_info["line_number"],
            source_text=dep_info["source_text"],
            is_external=dep_info["is_external"],
            confidence_score=1.0
        )
        
        test_session.add(dependency)
        dependencies.append(dependency)
    
    await test_session.commit()
    
    for dependency in dependencies:
        await test_session.refresh(dependency)
    
    return dependencies


@pytest_asyncio.fixture
async def test_analysis_session(test_session, test_project):
    """Create a test analysis session."""
    session = AnalysisSession(
        project_id=test_project.id,
        session_name="Test Analysis Session",
        session_type=AnalysisSessionType.FULL_ANALYSIS,
        status=AnalysisStatus.COMPLETED,
        progress_percentage=100.0,
        current_phase="completed",
        files_processed=4,
        files_total=4,
        dependencies_found=5,
        configuration={"test": True},
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc)
    )
    
    test_session.add(session)
    await test_session.commit()
    await test_session.refresh(session)
    
    return session


# ================== SERVICE FIXTURES ==================

@pytest_asyncio.fixture
async def project_indexer(test_session, test_redis):
    """Create ProjectIndexer instance for testing."""
    config = ProjectIndexConfig(
        analysis_timeout=300,
        max_file_size=10_000_000,
        max_files_per_project=10_000,
        cache_ttl=300,
        enable_ai_analysis=False  # Disable for testing
    )
    
    # Mock event publisher
    event_publisher = Mock()
    event_publisher.publish = AsyncMock()
    
    indexer = ProjectIndexer(
        session=test_session,
        redis_client=test_redis,
        config=config,
        event_publisher=event_publisher
    )
    
    return indexer


@pytest_asyncio.fixture
async def cache_manager(test_redis):
    """Create CacheManager instance for testing."""
    from app.project_index.cache import CacheConfig
    
    config = CacheConfig(
        default_ttl=300,
        max_key_length=250,
        compression_threshold=1024,
        enable_compression=True
    )
    
    return CacheManager(test_redis, config)


@pytest_asyncio.fixture
async def code_analyzer():
    """Create CodeAnalyzer instance for testing."""
    analyzer = CodeAnalyzer()
    return analyzer


@pytest_asyncio.fixture
async def file_monitor(test_redis):
    """Create FileMonitor instance for testing."""
    monitor = FileMonitor(test_redis)
    return monitor


@pytest_asyncio.fixture
async def websocket_manager(test_redis):
    """Create ProjectIndexWebSocketManager instance for testing."""
    manager = ProjectIndexWebSocketManager(test_redis)
    return manager


# ================== PERFORMANCE MONITORING ==================

@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for test execution times."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        
        def start_timer(self, name: str):
            self.metrics[name] = {"start": datetime.now()}
        
        def end_timer(self, name: str):
            if name in self.metrics:
                self.metrics[name]["end"] = datetime.now()
                self.metrics[name]["duration"] = (
                    self.metrics[name]["end"] - self.metrics[name]["start"]
                ).total_seconds()
        
        def get_duration(self, name: str) -> float:
            return self.metrics.get(name, {}).get("duration", 0.0)
        
        def assert_performance(self, name: str, max_seconds: float):
            duration = self.get_duration(name)
            assert duration <= max_seconds, f"Performance test failed: {name} took {duration}s, expected <= {max_seconds}s"
    
    return PerformanceMonitor()


# ================== LOAD TESTING FIXTURES ==================

@pytest_asyncio.fixture
async def load_test_data():
    """Generate data for load testing scenarios."""
    
    def generate_projects(count: int = 10):
        projects = []
        for i in range(count):
            projects.append({
                "name": f"Load Test Project {i}",
                "description": f"Project {i} for load testing",
                "root_path": f"/tmp/load_test_{i}",
                "git_repository_url": f"https://github.com/test/project-{i}.git",
                "git_branch": "main"
            })
        return projects
    
    def generate_files(project_count: int = 10, files_per_project: int = 50):
        files = []
        for p in range(project_count):
            for f in range(files_per_project):
                files.append({
                    "project_index": p,
                    "relative_path": f"src/module_{f}.py",
                    "file_name": f"module_{f}.py",
                    "file_type": FileType.SOURCE,
                    "language": "python",
                    "file_size": 1000 + (f * 50),
                    "line_count": 40 + (f * 2)
                })
        return files
    
    return {
        "projects": generate_projects,
        "files": generate_files
    }


# ================== MOCK UTILITIES ==================

@pytest.fixture
def mock_git_integration():
    """Mock Git integration for testing."""
    mock = Mock()
    mock.get_current_commit = Mock(return_value="a" * 40)
    mock.get_current_branch = Mock(return_value="main")
    mock.get_file_changes = Mock(return_value=[])
    mock.get_repository_url = Mock(return_value="https://github.com/test/repo.git")
    return mock


@pytest.fixture
def mock_ai_analyzer():
    """Mock AI analyzer for testing."""
    mock = AsyncMock()
    mock.analyze_code = AsyncMock(return_value={
        "functions": ["main", "helper_function"],
        "classes": ["DataProcessor"],
        "complexity": 3,
        "maintainability": 8.5,
        "suggestions": ["Add docstrings", "Consider type hints"]
    })
    return mock


# ================== TEST UTILITIES ==================

class TestDataBuilder:
    """Builder pattern for creating test data."""
    
    @staticmethod
    def project(**kwargs):
        """Build project data with defaults."""
        defaults = {
            "name": "Test Project",
            "description": "A test project",
            "root_path": "/tmp/test",
            "status": ProjectStatus.ACTIVE,
            "configuration": {},
            "file_count": 0,
            "dependency_count": 0
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def file_entry(project_id: uuid.UUID, **kwargs):
        """Build file entry data with defaults."""
        defaults = {
            "project_id": project_id,
            "file_path": "/tmp/test/src/main.py",
            "relative_path": "src/main.py",
            "file_name": "main.py",
            "file_extension": ".py",
            "file_type": FileType.SOURCE,
            "language": "python",
            "file_size": 1000,
            "line_count": 50,
            "is_binary": False,
            "sha256_hash": "a" * 64
        }
        defaults.update(kwargs)
        return defaults
    
    @staticmethod
    def dependency(project_id: uuid.UUID, source_file_id: uuid.UUID, **kwargs):
        """Build dependency data with defaults."""
        defaults = {
            "project_id": project_id,
            "source_file_id": source_file_id,
            "target_name": "test_module",
            "dependency_type": DependencyType.IMPORT,
            "is_external": False,
            "confidence_score": 1.0
        }
        defaults.update(kwargs)
        return defaults


@pytest.fixture
def test_data_builder():
    """Test data builder fixture."""
    return TestDataBuilder


# ================== ASYNC UTILITIES ==================

async def async_range(count: int):
    """Async range generator for testing."""
    for i in range(count):
        yield i


def assert_valid_uuid(value: str):
    """Assert that a string is a valid UUID."""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return False


def assert_recent_timestamp(timestamp: datetime, tolerance_seconds: int = 60):
    """Assert that a timestamp is recent within tolerance."""
    now = datetime.now(timezone.utc)
    diff = abs((now - timestamp.replace(tzinfo=timezone.utc)).total_seconds())
    assert diff <= tolerance_seconds, f"Timestamp {timestamp} is not recent (diff: {diff}s)"


# ================== CLEANUP UTILITIES ==================

@pytest.fixture(autouse=True)
async def cleanup_test_data(test_session):
    """Automatically cleanup test data after each test."""
    yield
    
    # Clean up any test data that might have been created
    try:
        await test_session.rollback()
    except:
        pass