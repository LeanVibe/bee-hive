"""
Comprehensive unit tests for Project Index API endpoints.

Tests for all REST API endpoints including request/response validation,
error handling, authentication, rate limiting, and edge cases.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.models.project_index import ProjectStatus, FileType, DependencyType, AnalysisStatus
from tests.project_index_conftest import create_mock_file_analysis_result, create_mock_dependency_result


@pytest.mark.project_index_api
@pytest.mark.unit
class TestProjectIndexCRUDEndpoints:
    """Test Project Index CRUD API endpoints."""
    
    @patch('app.api.project_index.get_project_indexer')
    async def test_create_project_index_success(
        self, 
        mock_get_indexer,
        async_test_client: AsyncClient,
        temp_project_directory
    ):
        """Test successful project index creation."""
        # Mock the indexer
        mock_indexer = AsyncMock()
        mock_project = Mock()
        mock_project.id = uuid.uuid4()
        mock_project.name = "Test Project"
        mock_project.root_path = str(temp_project_directory)
        mock_project.status = ProjectStatus.INACTIVE
        mock_project.file_count = 0
        mock_project.dependency_count = 0
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_indexer.create_project.return_value = mock_project
        mock_get_indexer.return_value = mock_indexer
        
        # Prepare request data
        request_data = {
            "name": "Test Project",
            "description": "A test project for API testing",
            "root_path": str(temp_project_directory),
            "git_repository_url": "https://github.com/test/project.git",
            "git_branch": "main",
            "configuration": {
                "languages": ["python"],
                "analysis_depth": 3
            }
        }
        
        # Make request
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        # Assertions
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        assert "meta" in response_data
        assert "links" in response_data
        
        # Verify response structure
        project_data = response_data["data"]
        assert project_data["name"] == "Test Project"
        assert "id" in project_data
        
        # Verify links
        links = response_data["links"]
        assert "self" in links
        assert "files" in links
        assert "dependencies" in links
        assert "analyze" in links
        
        # Verify indexer was called correctly
        mock_indexer.create_project.assert_called_once()
        call_kwargs = mock_indexer.create_project.call_args.kwargs
        assert call_kwargs["name"] == "Test Project"
        assert call_kwargs["root_path"] == str(temp_project_directory)
    
    async def test_create_project_index_invalid_path(self, async_test_client: AsyncClient):
        """Test project creation with invalid root path."""
        request_data = {
            "name": "Invalid Project",
            "root_path": "/non/existent/path",
            "description": "Project with invalid path"
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        assert response.status_code == 400
        response_data = response.json()
        assert "detail" in response_data
        assert response_data["detail"]["error"] == "INVALID_ROOT_PATH"
    
    async def test_create_project_index_validation_error(self, async_test_client: AsyncClient):
        """Test project creation with validation errors."""
        # Missing required fields
        request_data = {
            "description": "Project missing required fields"
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_project_index_success(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test successful project index retrieval."""
        # Mock project
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Retrieved Project"
        mock_project.status = ProjectStatus.ACTIVE
        mock_project.file_count = 10
        mock_project.dependency_count = 15
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_get_project.return_value = mock_project
        
        response = await async_test_client.get(f"/api/project-index/{project_id}")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify response structure
        assert "data" in response_data
        assert "meta" in response_data
        assert "links" in response_data
        
        project_data = response_data["data"]
        assert project_data["name"] == "Retrieved Project"
        assert project_data["file_count"] == 10
        assert project_data["dependency_count"] == 15
    
    async def test_get_project_index_not_found(self, async_test_client: AsyncClient):
        """Test project retrieval with non-existent ID."""
        non_existent_id = uuid.uuid4()
        
        response = await async_test_client.get(f"/api/project-index/{non_existent_id}")
        
        assert response.status_code == 404
        response_data = response.json()
        assert "detail" in response_data
    
    @patch('app.api.project_index.get_project_or_404')
    @patch('app.api.project_index.get_project_indexer')
    async def test_refresh_project_index_success(
        self, 
        mock_get_indexer,
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test successful project refresh."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        
        mock_get_project.return_value = mock_project
        
        # Mock indexer session to return no active sessions
        mock_indexer = AsyncMock()
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_indexer.session = mock_session
        mock_get_indexer.return_value = mock_indexer
        
        response = await async_test_client.put(f"/api/project-index/{project_id}/refresh")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        refresh_data = response_data["data"]
        assert refresh_data["project_id"] == str(project_id)
        assert refresh_data["status"] == "scheduled"
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_delete_project_index_success(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        project_index_session
    ):
        """Test successful project deletion."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Project to Delete"
        
        mock_get_project.return_value = mock_project
        
        # Mock session operations for cleanup statistics
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.scalar.return_value = 5  # Mock file count
            mock_session.delete = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.delete(f"/api/project-index/{project_id}")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        cleanup_data = response_data["data"]
        assert cleanup_data["project_id"] == str(project_id)
        assert cleanup_data["project_name"] == "Project to Delete"


@pytest.mark.project_index_api
@pytest.mark.unit
class TestFileAnalysisEndpoints:
    """Test file analysis API endpoints."""
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_list_project_files_success(
        self, 
        mock_get_project,
        async_test_client: AsyncClient,
        project_index_session
    ):
        """Test successful file listing with pagination."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        # Mock database session for file query
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock file entries
            mock_files = [
                Mock(
                    id=uuid.uuid4(),
                    relative_path="src/main.py",
                    file_name="main.py",
                    file_type=FileType.SOURCE,
                    language="python",
                    last_modified=datetime.utcnow()
                ),
                Mock(
                    id=uuid.uuid4(),
                    relative_path="tests/test_main.py",
                    file_name="test_main.py",
                    file_type=FileType.TEST,
                    language="python",
                    last_modified=datetime.utcnow()
                )
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_files
            mock_session.execute.return_value.scalar.return_value = len(mock_files)
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files?page=1&limit=10"
            )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        files_data = response_data["data"]
        assert "files" in files_data
        assert "total" in files_data
        assert "page" in files_data
        assert "page_size" in files_data
        assert len(files_data["files"]) == 2
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_list_project_files_with_filters(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test file listing with language and type filters."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_session.execute.return_value.scalar.return_value = 0
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files"
                f"?language=python&file_type=source&page=1&limit=50"
            )
        
        assert response.status_code == 200
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_file_analysis_success(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test successful file analysis retrieval."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        file_path = "src/main.py"
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock file entry with dependencies
            mock_file = Mock()
            mock_file.id = uuid.uuid4()
            mock_file.relative_path = file_path
            mock_file.file_name = "main.py"
            mock_file.file_type = FileType.SOURCE
            mock_file.language = "python"
            mock_file.outgoing_dependencies = []
            mock_file.incoming_dependencies = []
            
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_file
            mock_get_session.return_value = mock_session
            
            # URL encode the file path
            encoded_path = file_path.replace("/", "%2F")
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files/{encoded_path}"
            )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        file_data = response_data["data"]
        assert file_data["relative_path"] == file_path
        assert "outgoing_dependencies" in file_data
        assert "incoming_dependencies" in file_data
    
    async def test_get_file_analysis_not_found(self, async_test_client: AsyncClient):
        """Test file analysis retrieval for non-existent file."""
        project_id = uuid.uuid4()
        file_path = "non/existent/file.py"
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_get_project.return_value = mock_project
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalar_one_or_none.return_value = None
                mock_get_session.return_value = mock_session
                
                encoded_path = file_path.replace("/", "%2F")
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files/{encoded_path}"
                )
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["detail"]["error"] == "FILE_NOT_FOUND"


@pytest.mark.project_index_api
@pytest.mark.unit
class TestAnalysisEndpoints:
    """Test analysis trigger API endpoints."""
    
    @patch('app.api.project_index.get_project_or_404')
    @patch('app.api.project_index.get_project_indexer')
    @patch('app.api.project_index.rate_limit_analysis')
    async def test_analyze_project_success(
        self, 
        mock_rate_limit,
        mock_get_indexer,
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test successful project analysis trigger."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        mock_indexer = AsyncMock()
        mock_get_indexer.return_value = mock_indexer
        
        request_data = {
            "analysis_type": "full",
            "force": False,
            "configuration": {
                "parse_ast": True,
                "extract_dependencies": True
            }
        }
        
        response = await async_test_client.post(
            f"/api/project-index/{project_id}/analyze",
            json=request_data
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        analysis_data = response_data["data"]
        assert analysis_data["project_id"] == str(project_id)
        assert analysis_data["analysis_type"] == "full"
        assert analysis_data["status"] == "scheduled"
        assert "analysis_session_id" in analysis_data
    
    @patch('app.api.project_index.get_project_or_404')
    @patch('app.api.project_index.rate_limit_analysis')
    async def test_analyze_project_invalid_type(
        self, 
        mock_rate_limit,
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test analysis with invalid analysis type."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        request_data = {
            "analysis_type": "invalid_type",
            "force": False
        }
        
        response = await async_test_client.post(
            f"/api/project-index/{project_id}/analyze",
            json=request_data
        )
        
        assert response.status_code == 400
        response_data = response.json()
        assert response_data["detail"]["error"] == "INVALID_ANALYSIS_TYPE"
    
    @patch('app.api.project_index.rate_limit_analysis')
    async def test_analyze_project_rate_limited(
        self, 
        mock_rate_limit,
        async_test_client: AsyncClient
    ):
        """Test analysis rate limiting."""
        from fastapi import HTTPException
        
        # Mock rate limiting to raise exception
        mock_rate_limit.side_effect = HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Analysis rate limit exceeded"
            }
        )
        
        project_id = uuid.uuid4()
        request_data = {
            "analysis_type": "full",
            "force": False
        }
        
        response = await async_test_client.post(
            f"/api/project-index/{project_id}/analyze",
            json=request_data
        )
        
        assert response.status_code == 429
        response_data = response.json()
        assert "rate limit" in response_data["detail"]["message"].lower()


@pytest.mark.project_index_api
@pytest.mark.unit
class TestDependencyEndpoints:
    """Test dependency analysis API endpoints."""
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_dependencies_json_format(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test dependency retrieval in JSON format."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock dependencies
            mock_deps = [
                Mock(
                    id=uuid.uuid4(),
                    source_file_id=uuid.uuid4(),
                    target_name="os",
                    dependency_type=DependencyType.IMPORT,
                    is_external=True,
                    confidence_score=1.0
                ),
                Mock(
                    id=uuid.uuid4(),
                    source_file_id=uuid.uuid4(),
                    target_name="json",
                    dependency_type=DependencyType.IMPORT,
                    is_external=True,
                    confidence_score=1.0
                )
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_deps
            mock_session.execute.return_value.scalar.return_value = len(mock_deps)
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?format=json&page=1&limit=100"
            )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        deps_data = response_data["data"]
        assert "dependencies" in deps_data
        assert "total" in deps_data
        assert len(deps_data["dependencies"]) == 2
    
    @patch('app.api.project_index.get_project_or_404')
    @patch('app.api.project_index._get_dependency_graph')
    async def test_get_dependencies_graph_format(
        self, 
        mock_get_graph,
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test dependency retrieval in graph format."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        # Mock graph response
        mock_graph_response = {
            "data": {
                "nodes": [
                    {
                        "file_id": str(uuid.uuid4()),
                        "file_path": "src/main.py",
                        "file_type": "source",
                        "language": "python",
                        "in_degree": 0,
                        "out_degree": 2
                    }
                ],
                "edges": [
                    {
                        "source_file_id": str(uuid.uuid4()),
                        "target_name": "os",
                        "dependency_type": "import",
                        "is_external": True
                    }
                ],
                "statistics": {
                    "total_nodes": 1,
                    "total_edges": 1,
                    "external_dependencies": 1
                }
            },
            "meta": {
                "format": "graph"
            }
        }
        
        mock_get_graph.return_value = mock_graph_response
        
        response = await async_test_client.get(
            f"/api/project-index/{project_id}/dependencies?format=graph"
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        graph_data = response_data["data"]
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "statistics" in graph_data
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_dependencies_with_file_filter(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test dependency retrieval filtered by specific file."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        file_path = "src/main.py"
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock file entry lookup
            mock_file = Mock()
            mock_file.id = uuid.uuid4()
            mock_session.execute.return_value.scalar_one_or_none.return_value = mock_file
            
            # Mock dependencies for the file
            mock_deps = [
                Mock(
                    id=uuid.uuid4(),
                    source_file_id=mock_file.id,
                    target_name="os",
                    dependency_type=DependencyType.IMPORT,
                    is_external=True
                )
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_deps
            mock_session.execute.return_value.scalar.return_value = len(mock_deps)
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?file_path={file_path}"
            )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        deps_data = response_data["data"]
        assert "dependencies" in deps_data
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_get_dependencies_file_not_found(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test dependency retrieval for non-existent file."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        file_path = "non/existent/file.py"
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalar_one_or_none.return_value = None
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?file_path={file_path}"
            )
        
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["detail"]["error"] == "FILE_NOT_FOUND"


@pytest.mark.project_index_api
@pytest.mark.unit
class TestWebSocketEndpoints:
    """Test WebSocket API endpoints."""
    
    @patch('app.api.project_index.get_websocket_handler')
    @patch('app.api.project_index.websocket_auth_context')
    async def test_websocket_connection_success(
        self, 
        mock_auth_context,
        mock_get_handler,
        test_client: TestClient
    ):
        """Test successful WebSocket connection."""
        # Mock authentication context
        mock_auth_context.return_value.__aenter__.return_value = "test_user"
        mock_auth_context.return_value.__aexit__.return_value = None
        
        # Mock WebSocket handler
        mock_handler = AsyncMock()
        mock_handler.handle_connection = AsyncMock()
        mock_get_handler.return_value = mock_handler
        
        # Test WebSocket connection
        with test_client.websocket_connect("/api/project-index/ws?token=test_token") as websocket:
            # Connection should be established
            # In a real test, you would send/receive messages
            pass
    
    @patch('app.api.project_index.get_subscription_stats')
    async def test_websocket_stats(
        self, 
        mock_get_stats,
        async_test_client: AsyncClient
    ):
        """Test WebSocket statistics endpoint."""
        mock_stats = {
            "total_connections": 5,
            "active_subscriptions": 10,
            "events_sent_today": 150,
            "connection_uptime_hours": 24.5
        }
        mock_get_stats.return_value = mock_stats
        
        response = await async_test_client.get("/api/project-index/ws/stats")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "data" in response_data
        stats_data = response_data["data"]
        assert stats_data["total_connections"] == 5
        assert stats_data["active_subscriptions"] == 10


@pytest.mark.project_index_api
@pytest.mark.unit
class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    async def test_invalid_uuid_parameter(self, async_test_client: AsyncClient):
        """Test API endpoint with invalid UUID parameter."""
        invalid_uuid = "not-a-valid-uuid"
        
        response = await async_test_client.get(f"/api/project-index/{invalid_uuid}")
        
        assert response.status_code == 422  # Validation error
    
    async def test_malformed_json_request(self, async_test_client: AsyncClient):
        """Test API endpoint with malformed JSON."""
        response = await async_test_client.post(
            "/api/project-index/create",
            content="{ invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    async def test_missing_required_fields(self, async_test_client: AsyncClient):
        """Test API endpoint with missing required fields."""
        request_data = {
            # Missing required 'name' and 'root_path' fields
            "description": "Project with missing fields"
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        assert response.status_code == 422
    
    @patch('app.api.project_index.get_project_indexer')
    async def test_internal_server_error(
        self, 
        mock_get_indexer,
        async_test_client: AsyncClient,
        temp_project_directory
    ):
        """Test API error handling for internal server errors."""
        # Mock indexer to raise an exception
        mock_indexer = AsyncMock()
        mock_indexer.create_project.side_effect = Exception("Database connection failed")
        mock_get_indexer.return_value = mock_indexer
        
        request_data = {
            "name": "Error Test Project",
            "root_path": str(temp_project_directory)
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        assert response.status_code == 500
        response_data = response.json()
        assert response_data["detail"]["error"] == "INTERNAL_ERROR"
    
    async def test_large_request_body(self, async_test_client: AsyncClient):
        """Test API handling of large request bodies."""
        # Create a large configuration object
        large_config = {
            "languages": ["python"] * 1000,  # Very large list
            "large_data": "x" * 10000  # Large string
        }
        
        request_data = {
            "name": "Large Config Project",
            "root_path": "/test/path",
            "configuration": large_config
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        # Should handle large requests gracefully (might succeed or fail with proper error)
        assert response.status_code in [200, 413, 422, 500]


@pytest.mark.project_index_api
@pytest.mark.unit
class TestAPIResponseFormats:
    """Test API response formats and standards compliance."""
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_standard_response_format(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test that all responses follow the standard format."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_project.name = "Format Test"
        mock_project.status = ProjectStatus.ACTIVE
        mock_project.created_at = datetime.utcnow()
        mock_project.updated_at = datetime.utcnow()
        
        mock_get_project.return_value = mock_project
        
        response = await async_test_client.get(f"/api/project-index/{project_id}")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify standard response structure
        assert "data" in response_data
        assert "meta" in response_data
        assert "links" in response_data
        
        # Verify meta information
        meta = response_data["meta"]
        assert "timestamp" in meta
        assert "correlation_id" in meta
        
        # Verify HATEOAS links
        links = response_data["links"]
        assert "self" in links
        assert links["self"].endswith(str(project_id))
    
    async def test_error_response_format(self, async_test_client: AsyncClient):
        """Test that error responses follow the standard format."""
        non_existent_id = uuid.uuid4()
        
        response = await async_test_client.get(f"/api/project-index/{non_existent_id}")
        
        assert response.status_code == 404
        response_data = response.json()
        
        # Verify error response structure
        assert "detail" in response_data
        detail = response_data["detail"]
        assert "error" in detail
        assert "message" in detail
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_pagination_metadata(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test pagination metadata in responses."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_session.execute.return_value.scalar.return_value = 0
            mock_get_session.return_value = mock_session
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files?page=2&limit=25"
            )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Verify pagination metadata
        assert "meta" in response_data
        meta = response_data["meta"]
        assert "pagination" in meta
        
        pagination = meta["pagination"]
        assert pagination["page"] == 2
        assert pagination["limit"] == 25
        assert "total" in pagination
        assert "total_pages" in pagination


@pytest.mark.project_index_api
@pytest.mark.unit
class TestAPIParameterValidation:
    """Test API parameter validation and constraints."""
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_pagination_parameter_validation(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test pagination parameter validation."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        # Test invalid page number (negative)
        response = await async_test_client.get(
            f"/api/project-index/{project_id}/files?page=-1&limit=10"
        )
        assert response.status_code == 422
        
        # Test invalid limit (too large)
        response = await async_test_client.get(
            f"/api/project-index/{project_id}/files?page=1&limit=10000"
        )
        assert response.status_code == 422
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_filter_parameter_validation(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test filter parameter validation."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_session.execute.return_value.scalar.return_value = 0
            mock_get_session.return_value = mock_session
            
            # Test valid file type filter
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files?file_type=source"
            )
            assert response.status_code == 200
            
            # Test invalid file type filter
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/files?file_type=invalid_type"
            )
            assert response.status_code == 422
    
    @patch('app.api.project_index.get_project_or_404')
    async def test_dependency_parameter_validation(
        self, 
        mock_get_project,
        async_test_client: AsyncClient
    ):
        """Test dependency endpoint parameter validation."""
        project_id = uuid.uuid4()
        mock_project = Mock()
        mock_project.id = project_id
        mock_get_project.return_value = mock_project
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            mock_session.execute.return_value.scalar.return_value = 0
            mock_get_session.return_value = mock_session
            
            # Test depth parameter validation (too high)
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?depth=10"
            )
            assert response.status_code == 422
            
            # Test depth parameter validation (too low)
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/dependencies?depth=0"
            )
            assert response.status_code == 422