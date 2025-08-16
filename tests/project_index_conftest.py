"""
Project Index specific pytest fixtures and test configuration.

Provides dedicated fixtures for Project Index testing including
database models, test data, and mock services.
"""

import asyncio
import uuid
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, AsyncGenerator, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus, SnapshotType
)
from app.schemas.project_index import ProjectIndexCreate, ProjectIndexUpdate
from app.core.database import Base
from app.core.redis import RedisClient


# Test database configuration
TEST_PROJECT_INDEX_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
def project_index_event_loop():
    """Event loop for Project Index tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def project_index_engine():
    """Create test database engine for Project Index tests."""
    engine = create_async_engine(
        TEST_PROJECT_INDEX_DB_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all Project Index tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def project_index_session(project_index_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session for Project Index tests."""
    async_session_maker = async_sessionmaker(
        bind=project_index_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
def mock_redis_client():
    """Mock Redis client for Project Index caching and events."""
    mock_client = AsyncMock(spec=RedisClient)
    
    # Configure basic Redis operations
    mock_client.ping.return_value = True
    mock_client.get.return_value = None
    mock_client.set.return_value = True
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = False
    mock_client.expire.return_value = True
    mock_client.publish.return_value = 1
    
    # Hash operations
    mock_client.hget.return_value = None
    mock_client.hset.return_value = 1
    mock_client.hdel.return_value = 1
    mock_client.hgetall.return_value = {}
    
    # List operations
    mock_client.lpush.return_value = 1
    mock_client.rpush.return_value = 1
    mock_client.lrange.return_value = []
    mock_client.llen.return_value = 0
    
    # Set operations
    mock_client.sadd.return_value = 1
    mock_client.srem.return_value = 1
    mock_client.smembers.return_value = set()
    
    return mock_client


@pytest_asyncio.fixture
def mock_event_publisher():
    """Mock event publisher for Project Index events."""
    publisher = AsyncMock()
    publisher.publish.return_value = None
    return publisher


@pytest_asyncio.fixture
def temp_project_directory():
    """Create temporary directory for test projects."""
    temp_dir = tempfile.mkdtemp(prefix="test_project_")
    temp_path = Path(temp_dir)
    
    # Create a sample project structure
    (temp_path / "src").mkdir()
    (temp_path / "tests").mkdir()
    (temp_path / "docs").mkdir()
    
    # Create sample files
    (temp_path / "README.md").write_text("# Test Project\n\nA test project for unit testing.")
    (temp_path / "requirements.txt").write_text("requests==2.28.1\nflask==2.2.2\n")
    (temp_path / "src" / "__init__.py").write_text("")
    (temp_path / "src" / "main.py").write_text("""
import os
import sys
from typing import List

def main() -> None:
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
    (temp_path / "src" / "utils.py").write_text("""
import json
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def save_data(data: dict, output_path: str) -> None:
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
""")
    (temp_path / "tests" / "__init__.py").write_text("")
    (temp_path / "tests" / "test_main.py").write_text("""
import pytest
from src.main import main

def test_main():
    # This would normally test the main function
    assert True
""")
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture
async def sample_project_index(project_index_session: AsyncSession, temp_project_directory: Path):
    """Create a sample project index for testing."""
    project = ProjectIndex(
        name="Test Project",
        description="A test project for unit testing",
        root_path=str(temp_project_directory),
        git_repository_url="https://github.com/test/test-project.git",
        git_branch="main",
        git_commit_hash="abc123def456",
        status=ProjectStatus.ACTIVE,
        configuration={
            "languages": ["python"],
            "analysis_depth": 3,
            "enable_ai_analysis": True
        },
        analysis_settings={
            "parse_ast": True,
            "extract_dependencies": True,
            "calculate_complexity": True
        },
        file_patterns={
            "include": ["**/*.py", "**/*.md"],
            "exclude": ["__pycache__/**", "*.pyc"]
        },
        ignore_patterns={
            "directories": ["node_modules", ".git"],
            "files": ["*.log", "*.tmp"]
        },
        file_count=5,
        dependency_count=3
    )
    
    project_index_session.add(project)
    await project_index_session.commit()
    await project_index_session.refresh(project)
    
    return project


@pytest_asyncio.fixture
async def sample_file_entries(project_index_session: AsyncSession, sample_project_index: ProjectIndex):
    """Create sample file entries for testing."""
    files = [
        FileEntry(
            project_id=sample_project_index.id,
            file_path=str(Path(sample_project_index.root_path) / "README.md"),
            relative_path="README.md",
            file_name="README.md",
            file_extension=".md",
            file_type=FileType.DOCUMENTATION,
            language="markdown",
            file_size=45,
            line_count=3,
            sha256_hash="readme_hash_123",
            content_preview="# Test Project\n\nA test project...",
            analysis_data={
                "word_count": 8,
                "sections": ["heading"]
            },
            tags=["documentation", "readme"],
            is_binary=False,
            is_generated=False
        ),
        FileEntry(
            project_id=sample_project_index.id,
            file_path=str(Path(sample_project_index.root_path) / "src" / "main.py"),
            relative_path="src/main.py",
            file_name="main.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=156,
            line_count=10,
            sha256_hash="main_hash_456",
            content_preview="import os\nimport sys\nfrom typing import List...",
            analysis_data={
                "functions": [{"name": "main", "line": 5, "complexity": 1}],
                "imports": [
                    {"module": "os", "line": 1},
                    {"module": "sys", "line": 2},
                    {"module": "typing", "line": 3, "names": ["List"]}
                ],
                "classes": [],
                "complexity_metrics": {
                    "cyclomatic_complexity": 1,
                    "cognitive_complexity": 1
                }
            },
            tags=["main", "entry_point"],
            is_binary=False,
            is_generated=False
        ),
        FileEntry(
            project_id=sample_project_index.id,
            file_path=str(Path(sample_project_index.root_path) / "src" / "utils.py"),
            relative_path="src/utils.py",
            file_name="utils.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=289,
            line_count=12,
            sha256_hash="utils_hash_789",
            content_preview="import json\nfrom pathlib import Path...",
            analysis_data={
                "functions": [
                    {"name": "load_config", "line": 4, "complexity": 2},
                    {"name": "save_data", "line": 8, "complexity": 2}
                ],
                "imports": [
                    {"module": "json", "line": 1},
                    {"module": "pathlib", "line": 2, "names": ["Path"]}
                ],
                "classes": [],
                "complexity_metrics": {
                    "cyclomatic_complexity": 4,
                    "cognitive_complexity": 3
                }
            },
            tags=["utility", "helper"],
            is_binary=False,
            is_generated=False
        )
    ]
    
    for file_entry in files:
        project_index_session.add(file_entry)
    
    await project_index_session.commit()
    
    for file_entry in files:
        await project_index_session.refresh(file_entry)
    
    return files


@pytest_asyncio.fixture
async def sample_dependencies(
    project_index_session: AsyncSession, 
    sample_project_index: ProjectIndex,
    sample_file_entries: List[FileEntry]
):
    """Create sample dependency relationships for testing."""
    # Find the main.py file
    main_file = next(f for f in sample_file_entries if f.file_name == "main.py")
    utils_file = next(f for f in sample_file_entries if f.file_name == "utils.py")
    
    dependencies = [
        # External dependencies from main.py
        DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=main_file.id,
            target_name="os",
            dependency_type=DependencyType.IMPORT,
            line_number=1,
            source_text="import os",
            is_external=True,
            confidence_score=1.0,
            metadata={"standard_library": True}
        ),
        DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=main_file.id,
            target_name="sys",
            dependency_type=DependencyType.IMPORT,
            line_number=2,
            source_text="import sys",
            is_external=True,
            confidence_score=1.0,
            metadata={"standard_library": True}
        ),
        # External dependencies from utils.py
        DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=utils_file.id,
            target_name="json",
            dependency_type=DependencyType.IMPORT,
            line_number=1,
            source_text="import json",
            is_external=True,
            confidence_score=1.0,
            metadata={"standard_library": True}
        ),
        DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=utils_file.id,
            target_name="pathlib.Path",
            dependency_type=DependencyType.IMPORT,
            line_number=2,
            source_text="from pathlib import Path",
            is_external=True,
            confidence_score=1.0,
            metadata={"standard_library": True}
        )
    ]
    
    for dep in dependencies:
        project_index_session.add(dep)
    
    await project_index_session.commit()
    
    for dep in dependencies:
        await project_index_session.refresh(dep)
    
    return dependencies


@pytest_asyncio.fixture
async def sample_analysis_session(
    project_index_session: AsyncSession,
    sample_project_index: ProjectIndex
):
    """Create a sample analysis session for testing."""
    session = AnalysisSession(
        project_id=sample_project_index.id,
        session_name="Full Project Analysis",
        session_type=AnalysisSessionType.FULL_ANALYSIS,
        status=AnalysisStatus.COMPLETED,
        progress_percentage=100.0,
        current_phase="completed",
        files_processed=5,
        files_total=5,
        dependencies_found=4,
        errors_count=0,
        warnings_count=1,
        session_data={
            "analysis_config": {
                "parse_ast": True,
                "extract_dependencies": True
            }
        },
        error_log=[],
        performance_metrics={
            "total_duration": 2.5,
            "avg_file_analysis_time": 0.5,
            "peak_memory_usage": "50MB"
        },
        configuration={
            "enabled_languages": ["python", "markdown"],
            "max_file_size_mb": 10
        },
        result_data={
            "summary": {
                "files_analyzed": 5,
                "dependencies_extracted": 4,
                "complexity_calculated": True
            }
        },
        started_at=datetime.utcnow() - timedelta(minutes=5),
        completed_at=datetime.utcnow()
    )
    
    project_index_session.add(session)
    await project_index_session.commit()
    await project_index_session.refresh(session)
    
    return session


@pytest_asyncio.fixture
async def sample_index_snapshot(
    project_index_session: AsyncSession,
    sample_project_index: ProjectIndex
):
    """Create a sample index snapshot for testing."""
    snapshot = IndexSnapshot(
        project_id=sample_project_index.id,
        snapshot_name="Pre-Analysis Snapshot",
        description="Snapshot taken before full analysis",
        snapshot_type=SnapshotType.PRE_ANALYSIS,
        git_commit_hash="abc123def456",
        git_branch="main",
        file_count=5,
        dependency_count=4,
        changes_since_last={
            "files_added": 2,
            "files_modified": 1,
            "files_deleted": 0
        },
        analysis_metrics={
            "avg_complexity": 2.0,
            "total_loc": 25,
            "documentation_coverage": 0.6
        },
        data_checksum="snapshot_checksum_123"
    )
    
    project_index_session.add(snapshot)
    await project_index_session.commit()
    await project_index_session.refresh(snapshot)
    
    return snapshot


@pytest_asyncio.fixture
def large_project_structure():
    """Create a large project structure for performance testing."""
    return {
        "name": "Large Test Project",
        "description": "A large project for performance and load testing",
        "files": {
            "src": {
                "main.py": "# Main application entry point",
                "api": {
                    "routes.py": "# API routes",
                    "auth.py": "# Authentication logic",
                    "models.py": "# Database models"
                },
                "services": {
                    "user_service.py": "# User management service",
                    "email_service.py": "# Email service",
                    "notification_service.py": "# Notification service"
                },
                "utils": {
                    "helpers.py": "# Utility functions",
                    "decorators.py": "# Custom decorators",
                    "validators.py": "# Input validators"
                }
            },
            "tests": {
                "test_api.py": "# API tests",
                "test_services.py": "# Service tests",
                "test_utils.py": "# Utility tests"
            },
            "docs": {
                "README.md": "# Project Documentation",
                "API.md": "# API Documentation",
                "DEPLOYMENT.md": "# Deployment Guide"
            }
        },
        "expected_file_count": 15,
        "expected_dependency_count": 25
    }


@pytest_asyncio.fixture
def mock_project_indexer():
    """Mock ProjectIndexer for testing."""
    indexer = AsyncMock()
    
    # Configure create_project method
    async def mock_create_project(**kwargs):
        project_id = uuid.uuid4()
        project = Mock()
        project.id = project_id
        project.name = kwargs.get("name", "Test Project")
        project.root_path = kwargs.get("root_path", "/test/path")
        project.status = ProjectStatus.INACTIVE
        return project
    
    indexer.create_project = mock_create_project
    
    # Configure analyze_project method
    indexer.analyze_project.return_value = None
    
    # Configure other methods
    indexer.get_project_statistics.return_value = {
        "total_files": 5,
        "total_dependencies": 10,
        "analysis_status": "completed"
    }
    
    return indexer


@pytest_asyncio.fixture
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "max_response_time_ms": 500,
        "max_database_query_time_ms": 100,
        "max_websocket_latency_ms": 50,
        "max_concurrent_users": 100,
        "max_file_analysis_time_s": 2.0,
        "max_memory_usage_mb": 1024,
        "performance_targets": {
            "api_endpoints": {
                "GET /api/project-index/{id}": 200,
                "POST /api/project-index/create": 1000,
                "GET /api/project-index/{id}/files": 300,
                "GET /api/project-index/{id}/dependencies": 400
            },
            "database_operations": {
                "project_creation": 100,
                "file_entry_creation": 50,
                "dependency_creation": 30,
                "complex_queries": 200
            },
            "file_system_operations": {
                "file_scan": 1000,
                "directory_traversal": 500,
                "file_hash_calculation": 100
            }
        }
    }


@pytest_asyncio.fixture
def security_test_config():
    """Configuration for security tests."""
    return {
        "sql_injection_payloads": [
            "'; DROP TABLE project_indexes; --",
            "' OR '1'='1",
            "'; UPDATE project_indexes SET name='hacked'; --",
            "' UNION SELECT * FROM project_indexes --"
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ],
        "path_traversal_payloads": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        "max_request_size": 10 * 1024 * 1024,  # 10MB
        "rate_limit_requests_per_minute": 60,
        "auth_token_expiry_hours": 24
    }


@pytest_asyncio.fixture
def websocket_test_config():
    """Configuration for WebSocket tests."""
    return {
        "max_connections": 1000,
        "heartbeat_interval": 30,
        "message_size_limit": 1024 * 1024,  # 1MB
        "connection_timeout": 60,
        "event_types": [
            "analysis_progress",
            "file_change",
            "dependency_update", 
            "project_status_change",
            "session_started",
            "session_completed",
            "error_occurred"
        ],
        "subscription_filters": [
            "project_id",
            "session_id",
            "file_path",
            "event_type"
        ]
    }


# Utility functions for tests
def create_mock_file_analysis_result(file_path: str, **kwargs) -> Dict[str, Any]:
    """Create a mock file analysis result for testing."""
    return {
        "file_path": file_path,
        "relative_path": kwargs.get("relative_path", file_path),
        "file_name": Path(file_path).name,
        "file_extension": Path(file_path).suffix,
        "file_type": kwargs.get("file_type", "source"),
        "language": kwargs.get("language", "python"),
        "file_size": kwargs.get("file_size", 1024),
        "line_count": kwargs.get("line_count", 50),
        "analysis_successful": kwargs.get("analysis_successful", True),
        "complexity_metrics": kwargs.get("complexity_metrics", {
            "cyclomatic_complexity": 2,
            "cognitive_complexity": 2
        }),
        "dependencies": kwargs.get("dependencies", []),
        "tags": kwargs.get("tags", []),
        "metadata": kwargs.get("metadata", {})
    }


def create_mock_dependency_result(source_file: str, target_name: str, **kwargs) -> Dict[str, Any]:
    """Create a mock dependency result for testing."""
    return {
        "source_file_path": source_file,
        "target_name": target_name,
        "dependency_type": kwargs.get("dependency_type", "import"),
        "line_number": kwargs.get("line_number", 1),
        "is_external": kwargs.get("is_external", True),
        "confidence_score": kwargs.get("confidence_score", 1.0),
        "metadata": kwargs.get("metadata", {})
    }


async def wait_for_analysis_completion(
    session: AsyncSession,
    analysis_session_id: uuid.UUID,
    timeout_seconds: int = 30
) -> bool:
    """Wait for analysis session to complete with timeout."""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        # Check session status
        result = await session.get(AnalysisSession, analysis_session_id)
        if result and result.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
            return result.status == AnalysisStatus.COMPLETED
        
        await asyncio.sleep(0.1)  # Check every 100ms
    
    return False  # Timeout


def assert_project_index_valid(project: ProjectIndex) -> None:
    """Assert that a project index is valid."""
    assert project.id is not None
    assert project.name is not None and len(project.name) > 0
    assert project.root_path is not None and len(project.root_path) > 0
    assert project.status in [status.value for status in ProjectStatus]
    assert project.file_count >= 0
    assert project.dependency_count >= 0
    assert project.created_at is not None
    assert project.updated_at is not None


def assert_file_entry_valid(file_entry: FileEntry) -> None:
    """Assert that a file entry is valid."""
    assert file_entry.id is not None
    assert file_entry.project_id is not None
    assert file_entry.file_path is not None
    assert file_entry.relative_path is not None
    assert file_entry.file_name is not None
    assert file_entry.file_type in [ft.value for ft in FileType]
    assert file_entry.file_size is None or file_entry.file_size >= 0
    assert file_entry.line_count is None or file_entry.line_count >= 0


def assert_dependency_valid(dependency: DependencyRelationship) -> None:
    """Assert that a dependency relationship is valid."""
    assert dependency.id is not None
    assert dependency.project_id is not None
    assert dependency.source_file_id is not None
    assert dependency.target_name is not None and len(dependency.target_name) > 0
    assert dependency.dependency_type in [dt.value for dt in DependencyType]
    assert dependency.confidence_score >= 0.0 and dependency.confidence_score <= 1.0