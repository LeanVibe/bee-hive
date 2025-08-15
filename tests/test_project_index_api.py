"""
Comprehensive test suite for Project Index API endpoints.

Tests all API endpoints with various scenarios including:
- CRUD operations
- Error handling
- Rate limiting
- WebSocket functionality
- Performance validation
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.project_index import router
from app.core.database import Base, get_session
from app.core.redis import get_redis_client
from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, DependencyType, AnalysisStatus, AnalysisSessionType
)
from app.schemas.project_index import (
    ProjectIndexCreate, AnalyzeProjectRequest, ProjectIndexResponse
)
from app.project_index.core import ProjectIndexer
from tests.conftest import async_session_maker, redis_client


class TestProjectIndexAPI:
    """Test suite for Project Index API endpoints."""

    @pytest.fixture
    async def sample_project_data(self) -> Dict[str, Any]:
        """Create sample project data for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample files
            project_root = Path(temp_dir)
            (project_root / "main.py").write_text("print('Hello World')")
            (project_root / "utils.py").write_text("def helper(): pass")
            (project_root / "requirements.txt").write_text("fastapi==0.68.0")
            
            return {
                "name": "Test Project",
                "description": "A test project for API testing",
                "root_path": str(project_root),
                "git_repository_url": "https://github.com/test/project.git",
                "git_branch": "main",
                "configuration": {
                    "languages": ["python"],
                    "analysis_depth": 3
                },
                "file_patterns": {
                    "include": ["**/*.py", "**/*.txt"],
                    "exclude": ["**/__pycache__/**"]
                }
            }

    @pytest.fixture
    async def mock_indexer(self):
        """Mock ProjectIndexer for testing."""
        indexer = Mock(spec=ProjectIndexer)
        indexer.create_project = AsyncMock()
        indexer.analyze_project = AsyncMock()
        indexer.get_project = AsyncMock()
        return indexer

    @pytest.fixture
    async def client(self, async_session: AsyncSession, redis_client):
        """Create test client with database session override."""
        from app.main import app
        
        async def override_get_session():
            yield async_session
        
        async def override_get_redis():
            return redis_client
        
        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_redis_client] = override_get_redis
        
        with TestClient(app) as client:
            yield client

    # ================== PROJECT MANAGEMENT ENDPOINT TESTS ==================

    @pytest.mark.asyncio
    async def test_create_project_success(self, client: TestClient, sample_project_data: Dict):
        """Test successful project creation."""
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            response = client.post("/api/project-index/create", json=sample_project_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "data" in data
        assert "meta" in data
        assert "links" in data
        
        project_data = data["data"]
        assert project_data["name"] == sample_project_data["name"]
        assert project_data["description"] == sample_project_data["description"]
        assert project_data["status"] == "inactive"
        
        # Check HATEOAS links
        links = data["links"]
        assert "self" in links
        assert "files" in links
        assert "dependencies" in links
        assert "analyze" in links

    @pytest.mark.asyncio
    async def test_create_project_invalid_path(self, client: TestClient):
        """Test project creation with invalid root path."""
        invalid_data = {
            "name": "Invalid Project",
            "root_path": "/nonexistent/path",
            "description": "Project with invalid path"
        }
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            response = client.post("/api/project-index/create", json=invalid_data)
        
        assert response.status_code == 400
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "INVALID_ROOT_PATH"
        assert "does not exist" in error_data["detail"]["message"]

    @pytest.mark.asyncio
    async def test_get_project_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful project retrieval."""
        # Create test project
        project = ProjectIndex(
            name="Test Project",
            root_path="/test/path",
            status=ProjectStatus.ACTIVE,
            file_count=10,
            dependency_count=25
        )
        async_session.add(project)
        await async_session.commit()
        await async_session.refresh(project)
        
        response = client.get(f"/api/project-index/{project.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        project_data = data["data"]
        assert project_data["id"] == str(project.id)
        assert project_data["name"] == "Test Project"
        assert project_data["status"] == "active"
        assert project_data["file_count"] == 10
        assert project_data["dependency_count"] == 25

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, client: TestClient):
        """Test project retrieval with non-existent ID."""
        non_existent_id = str(uuid.uuid4())
        response = client.get(f"/api/project-index/{non_existent_id}")
        
        assert response.status_code == 404
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "PROJECT_NOT_FOUND"
        assert non_existent_id in error_data["detail"]["message"]

    @pytest.mark.asyncio
    async def test_refresh_project_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful project refresh."""
        project = ProjectIndex(
            name="Test Project",
            root_path="/test/path",
            status=ProjectStatus.ACTIVE
        )
        async_session.add(project)
        await async_session.commit()
        await async_session.refresh(project)
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            with patch('app.api.project_index.rate_limit_analysis', return_value=None):
                response = client.put(f"/api/project-index/{project.id}/refresh")
        
        assert response.status_code == 200
        data = response.json()
        
        refresh_data = data["data"]
        assert refresh_data["project_id"] == str(project.id)
        assert refresh_data["status"] == "scheduled"
        assert "analysis_session_id" in refresh_data

    @pytest.mark.asyncio
    async def test_refresh_project_already_analyzing(self, client: TestClient, async_session: AsyncSession):
        """Test project refresh when analysis is already in progress."""
        project = ProjectIndex(
            name="Test Project",
            root_path="/test/path",
            status=ProjectStatus.ANALYZING
        )
        async_session.add(project)
        await async_session.commit()
        await async_session.refresh(project)
        
        # Create active analysis session
        analysis_session = AnalysisSession(
            project_id=project.id,
            session_name="Active Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            status=AnalysisStatus.RUNNING
        )
        async_session.add(analysis_session)
        await async_session.commit()
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            with patch('app.api.project_index.rate_limit_analysis', return_value=None):
                response = client.put(f"/api/project-index/{project.id}/refresh")
        
        assert response.status_code == 409
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "ANALYSIS_IN_PROGRESS"
        assert str(analysis_session.id) in error_data["detail"]["active_session_id"]

    @pytest.mark.asyncio
    async def test_delete_project_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful project deletion."""
        project = ProjectIndex(
            name="Test Project",
            root_path="/test/path",
            status=ProjectStatus.ACTIVE,
            file_count=5,
            dependency_count=10
        )
        async_session.add(project)
        await async_session.commit()
        await async_session.refresh(project)
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            response = client.delete(f"/api/project-index/{project.id}")
        
        assert response.status_code == 200
        data = response.json()
        
        cleanup_data = data["data"]
        assert cleanup_data["project_id"] == str(project.id)
        assert cleanup_data["project_name"] == "Test Project"
        assert "deleted_at" in cleanup_data

    # ================== FILE ANALYSIS ENDPOINT TESTS ==================

    @pytest.mark.asyncio
    async def test_list_project_files_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful file listing with pagination."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        # Create test files
        files = []
        for i in range(25):
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/test/path/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_type=FileType.SOURCE,
                language="python"
            )
            files.append(file_entry)
            async_session.add(file_entry)
        
        await async_session.commit()
        
        # Test pagination
        response = client.get(f"/api/project-index/{project.id}/files?page=1&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        files_data = data["data"]
        assert len(files_data["files"]) == 10
        assert files_data["total"] == 25
        assert files_data["page"] == 1
        assert files_data["has_next"] is True
        assert files_data["has_prev"] is False

    @pytest.mark.asyncio
    async def test_list_project_files_with_filters(self, client: TestClient, async_session: AsyncSession):
        """Test file listing with language filter."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        # Create files with different languages
        python_file = FileEntry(
            project_id=project.id,
            file_path="/test/path/script.py",
            relative_path="script.py",
            file_name="script.py",
            file_type=FileType.SOURCE,
            language="python"
        )
        js_file = FileEntry(
            project_id=project.id,
            file_path="/test/path/app.js",
            relative_path="app.js",
            file_name="app.js",
            file_type=FileType.SOURCE,
            language="javascript"
        )
        
        async_session.add_all([python_file, js_file])
        await async_session.commit()
        
        # Filter by language
        response = client.get(f"/api/project-index/{project.id}/files?language=python")
        
        assert response.status_code == 200
        data = response.json()
        
        files_data = data["data"]
        assert len(files_data["files"]) == 1
        assert files_data["files"][0]["language"] == "python"

    @pytest.mark.asyncio
    async def test_get_file_analysis_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful file analysis retrieval."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        file_entry = FileEntry(
            project_id=project.id,
            file_path="/test/path/main.py",
            relative_path="main.py",
            file_name="main.py",
            file_type=FileType.SOURCE,
            language="python",
            analysis_data={"functions": ["main"], "classes": []},
            metadata={"complexity": "low"}
        )
        async_session.add(file_entry)
        await async_session.commit()
        
        response = client.get(f"/api/project-index/{project.id}/files/main.py")
        
        assert response.status_code == 200
        data = response.json()
        
        file_data = data["data"]
        assert file_data["relative_path"] == "main.py"
        assert file_data["language"] == "python"
        assert file_data["analysis_data"]["functions"] == ["main"]

    @pytest.mark.asyncio
    async def test_get_file_analysis_not_found(self, client: TestClient, async_session: AsyncSession):
        """Test file analysis retrieval for non-existent file."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        
        response = client.get(f"/api/project-index/{project.id}/files/nonexistent.py")
        
        assert response.status_code == 404
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "FILE_NOT_FOUND"
        assert "nonexistent.py" in error_data["detail"]["message"]

    @pytest.mark.asyncio
    async def test_analyze_project_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful project analysis trigger."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        await async_session.refresh(project)
        
        analysis_request = {
            "analysis_type": "full",
            "force": False,
            "configuration": {"depth": 3}
        }
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            with patch('app.api.project_index.rate_limit_analysis', return_value=None):
                response = client.post(
                    f"/api/project-index/{project.id}/analyze",
                    json=analysis_request
                )
        
        assert response.status_code == 200
        data = response.json()
        
        analysis_data = data["data"]
        assert analysis_data["project_id"] == str(project.id)
        assert analysis_data["analysis_type"] == "full"
        assert analysis_data["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_analyze_project_invalid_type(self, client: TestClient, async_session: AsyncSession):
        """Test project analysis with invalid analysis type."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        
        analysis_request = {
            "analysis_type": "invalid_type",
            "force": False
        }
        
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            with patch('app.api.project_index.rate_limit_analysis', return_value=None):
                response = client.post(
                    f"/api/project-index/{project.id}/analyze",
                    json=analysis_request
                )
        
        assert response.status_code == 400
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "INVALID_ANALYSIS_TYPE"
        assert "invalid_type" in error_data["detail"]["message"]

    # ================== DEPENDENCIES ENDPOINT TESTS ==================

    @pytest.mark.asyncio
    async def test_get_dependencies_success(self, client: TestClient, async_session: AsyncSession):
        """Test successful dependency retrieval."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        # Create source and target files
        source_file = FileEntry(
            project_id=project.id,
            file_path="/test/path/main.py",
            relative_path="main.py",
            file_name="main.py",
            file_type=FileType.SOURCE
        )
        target_file = FileEntry(
            project_id=project.id,
            file_path="/test/path/utils.py",
            relative_path="utils.py",
            file_name="utils.py",
            file_type=FileType.SOURCE
        )
        async_session.add_all([source_file, target_file])
        await async_session.flush()
        
        # Create dependency relationship
        dependency = DependencyRelationship(
            project_id=project.id,
            source_file_id=source_file.id,
            target_file_id=target_file.id,
            target_name="utils",
            dependency_type=DependencyType.IMPORT,
            is_external=False
        )
        async_session.add(dependency)
        await async_session.commit()
        
        response = client.get(f"/api/project-index/{project.id}/dependencies")
        
        assert response.status_code == 200
        data = response.json()
        
        deps_data = data["data"]
        assert len(deps_data["dependencies"]) == 1
        assert deps_data["dependencies"][0]["target_name"] == "utils"
        assert deps_data["dependencies"][0]["is_external"] is False

    @pytest.mark.asyncio
    async def test_get_dependencies_graph_format(self, client: TestClient, async_session: AsyncSession):
        """Test dependency graph format response."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        # Create files and dependencies for graph
        file1 = FileEntry(
            project_id=project.id,
            file_path="/test/path/file1.py",
            relative_path="file1.py",
            file_name="file1.py",
            file_type=FileType.SOURCE,
            language="python"
        )
        file2 = FileEntry(
            project_id=project.id,
            file_path="/test/path/file2.py",
            relative_path="file2.py",
            file_name="file2.py",
            file_type=FileType.SOURCE,
            language="python"
        )
        async_session.add_all([file1, file2])
        await async_session.flush()
        
        dependency = DependencyRelationship(
            project_id=project.id,
            source_file_id=file1.id,
            target_file_id=file2.id,
            target_name="file2",
            dependency_type=DependencyType.IMPORT,
            is_external=False
        )
        async_session.add(dependency)
        await async_session.commit()
        
        response = client.get(f"/api/project-index/{project.id}/dependencies?format=graph")
        
        assert response.status_code == 200
        data = response.json()
        
        graph_data = data["data"]
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "statistics" in graph_data
        
        assert len(graph_data["nodes"]) == 2
        assert len(graph_data["edges"]) == 1

    # ================== RATE LIMITING TESTS ==================

    @pytest.mark.asyncio
    async def test_rate_limiting_analysis(self, client: TestClient, async_session: AsyncSession):
        """Test rate limiting for analysis endpoints."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        
        analysis_request = {"analysis_type": "full", "force": False}
        
        # Mock rate limiter to return False (rate limit exceeded)
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            with patch('app.api.project_index.RateLimiter.check_rate_limit', return_value=False):
                response = client.post(
                    f"/api/project-index/{project.id}/analyze",
                    json=analysis_request
                )
        
        assert response.status_code == 429
        error_data = response.json()
        
        assert error_data["detail"]["error"] == "RATE_LIMIT_EXCEEDED"
        assert "rate limit exceeded" in error_data["detail"]["message"].lower()

    # ================== WEBSOCKET TESTS ==================

    @pytest.mark.asyncio
    async def test_websocket_connection_success(self, client: TestClient):
        """Test successful WebSocket connection."""
        with patch('app.api.project_index_websocket.get_current_user_from_token', return_value="test_user"):
            with client.websocket_connect("/api/project-index/ws?token=valid_token") as websocket:
                # Should receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "welcome"
                assert data["data"]["user_id"] == "test_user"

    @pytest.mark.asyncio
    async def test_websocket_subscription(self, client: TestClient):
        """Test WebSocket subscription to events."""
        with patch('app.api.project_index_websocket.get_current_user_from_token', return_value="test_user"):
            with client.websocket_connect("/api/project-index/ws?token=valid_token") as websocket:
                # Receive welcome message
                websocket.receive_json()
                
                # Send subscription request
                subscription = {
                    "action": "subscribe",
                    "event_types": ["analysis_progress"],
                    "project_id": str(uuid.uuid4())
                }
                websocket.send_json(subscription)
                
                # Should receive acknowledgment
                ack = websocket.receive_json()
                assert ack["type"] == "subscription_ack"
                assert ack["data"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_websocket_stats(self, client: TestClient):
        """Test WebSocket statistics endpoint."""
        response = client.get("/api/project-index/ws/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        stats_data = data["data"]
        assert "active_connections" in stats_data
        assert "project_subscriptions" in stats_data
        assert "session_subscriptions" in stats_data

    # ================== PERFORMANCE TESTS ==================

    @pytest.mark.asyncio
    async def test_api_response_time(self, client: TestClient, async_session: AsyncSession):
        """Test API response time requirements."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        
        import time
        
        # Test project retrieval response time (<200ms)
        start_time = time.time()
        response = client.get(f"/api/project-index/{project.id}")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert response_time < 200, f"Response time {response_time}ms exceeds 200ms limit"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client: TestClient, async_session: AsyncSession):
        """Test handling of concurrent requests."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.commit()
        
        async def make_request():
            return client.get(f"/api/project-index/{project.id}")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    # ================== ERROR HANDLING TESTS ==================

    @pytest.mark.asyncio
    async def test_database_error_handling(self, client: TestClient):
        """Test error handling when database is unavailable."""
        with patch('app.api.project_index.get_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            response = client.get(f"/api/project-index/{uuid.uuid4()}")
            
            assert response.status_code == 500
            error_data = response.json()
            assert error_data["detail"]["error"] == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_invalid_uuid_format(self, client: TestClient):
        """Test handling of invalid UUID format."""
        response = client.get("/api/project-index/invalid-uuid")
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_malformed_json_request(self, client: TestClient):
        """Test handling of malformed JSON in request."""
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            response = client.post(
                "/api/project-index/create",
                data="invalid json",
                headers={"Content-Type": "application/json"}
            )
        
        assert response.status_code == 422  # Validation error

    # ================== INTEGRATION TESTS ==================

    @pytest.mark.asyncio
    async def test_full_project_lifecycle(self, client: TestClient, sample_project_data: Dict):
        """Test complete project lifecycle: create, analyze, query, delete."""
        with patch('app.api.project_index.get_current_user', return_value="test_user"):
            # 1. Create project
            create_response = client.post("/api/project-index/create", json=sample_project_data)
            assert create_response.status_code == 200
            
            project_id = create_response.json()["data"]["id"]
            
            # 2. Trigger analysis
            with patch('app.api.project_index.rate_limit_analysis', return_value=None):
                analysis_response = client.post(
                    f"/api/project-index/{project_id}/analyze",
                    json={"analysis_type": "full", "force": False}
                )
                assert analysis_response.status_code == 200
            
            # 3. Query files
            files_response = client.get(f"/api/project-index/{project_id}/files")
            assert files_response.status_code == 200
            
            # 4. Query dependencies
            deps_response = client.get(f"/api/project-index/{project_id}/dependencies")
            assert deps_response.status_code == 200
            
            # 5. Delete project
            delete_response = client.delete(f"/api/project-index/{project_id}")
            assert delete_response.status_code == 200

    @pytest.mark.asyncio
    async def test_pagination_consistency(self, client: TestClient, async_session: AsyncSession):
        """Test pagination consistency across multiple requests."""
        project = ProjectIndex(name="Test Project", root_path="/test/path")
        async_session.add(project)
        await async_session.flush()
        
        # Create 50 files
        for i in range(50):
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/test/path/file_{i:02d}.py",
                relative_path=f"file_{i:02d}.py",
                file_name=f"file_{i:02d}.py",
                file_type=FileType.SOURCE
            )
            async_session.add(file_entry)
        
        await async_session.commit()
        
        # Test pagination consistency
        page1 = client.get(f"/api/project-index/{project.id}/files?page=1&limit=20")
        page2 = client.get(f"/api/project-index/{project.id}/files?page=2&limit=20")
        page3 = client.get(f"/api/project-index/{project.id}/files?page=3&limit=20")
        
        assert page1.status_code == 200
        assert page2.status_code == 200
        assert page3.status_code == 200
        
        page1_data = page1.json()["data"]
        page2_data = page2.json()["data"]
        page3_data = page3.json()["data"]
        
        # Check counts
        assert len(page1_data["files"]) == 20
        assert len(page2_data["files"]) == 20
        assert len(page3_data["files"]) == 10  # Remaining files
        
        # Check pagination flags
        assert page1_data["has_next"] is True
        assert page1_data["has_prev"] is False
        assert page2_data["has_next"] is True
        assert page2_data["has_prev"] is True
        assert page3_data["has_next"] is False
        assert page3_data["has_prev"] is True

    # ================== CLEANUP AND FIXTURES ==================

    @pytest.fixture(autouse=True)
    async def cleanup_database(self, async_session: AsyncSession):
        """Clean up database after each test."""
        yield
        
        # Clean up test data
        await async_session.execute("DELETE FROM dependency_relationships")
        await async_session.execute("DELETE FROM file_entries")
        await async_session.execute("DELETE FROM analysis_sessions")
        await async_session.execute("DELETE FROM index_snapshots")
        await async_session.execute("DELETE FROM project_indexes")
        await async_session.commit()