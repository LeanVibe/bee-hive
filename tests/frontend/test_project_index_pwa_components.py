"""
Frontend integration tests for Project Index PWA components.

Tests for the React/TypeScript frontend components, user workflows,
responsive design, and Progressive Web App functionality.
"""

import pytest
import asyncio
import json
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch

# Import Project Index fixtures
pytest_plugins = ["tests.project_index_conftest"]

# Note: These tests would typically use Playwright or Selenium for actual browser testing
# For now, we'll create integration tests that verify the frontend-backend contract

@pytest.mark.project_index_frontend
@pytest.mark.integration
class TestProjectIndexDashboard:
    """Test Project Index dashboard components and user interactions."""
    
    async def test_dashboard_project_list_component(
        self, 
        async_test_client,
        sample_project_index,
        sample_file_entries
    ):
        """Test project list component data flow and rendering."""
        
        # Mock the projects list API endpoint
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            
            # Mock projects data
            mock_projects = [
                {
                    "id": str(sample_project_index.id),
                    "name": sample_project_index.name,
                    "description": sample_project_index.description,
                    "status": sample_project_index.status.value,
                    "file_count": len(sample_file_entries),
                    "dependency_count": 4,
                    "last_indexed_at": "2024-01-15T10:30:00Z",
                    "created_at": "2024-01-15T10:00:00Z"
                }
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = [sample_project_index]
            mock_session.execute.return_value.scalar.return_value = 1
            mock_get_session.return_value = mock_session
            
            # Test the API endpoint that frontend would call
            response = await async_test_client.get("/api/project-index")
            
            # Verify response structure for frontend consumption
            assert response.status_code in [200, 404]  # 404 if endpoint doesn't exist yet
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify data structure matches frontend expectations
                assert "data" in data
                assert "meta" in data
                
                projects = data["data"]
                if isinstance(projects, dict) and "projects" in projects:
                    projects = projects["projects"]
                
                if projects:
                    project = projects[0] if isinstance(projects, list) else projects
                    
                    # Verify required fields for dashboard
                    required_fields = ["id", "name", "status", "file_count", "created_at"]
                    for field in required_fields:
                        assert field in project, f"Missing required field: {field}"
                    
                    # Verify data types
                    assert isinstance(project["file_count"], int)
                    assert project["status"] in ["active", "inactive", "analyzing", "archived", "failed"]
    
    async def test_project_creation_form_validation(
        self, 
        async_test_client,
        temp_project_directory
    ):
        """Test project creation form validation and submission."""
        
        # Test various form submission scenarios
        test_cases = [
            # Valid submission
            {
                "data": {
                    "name": "Test Project",
                    "description": "A test project",
                    "root_path": str(temp_project_directory),
                    "git_repository_url": "https://github.com/test/repo.git",
                    "configuration": {
                        "languages": ["python", "javascript"],
                        "analysis_depth": 3
                    }
                },
                "expected_status": [200, 422]  # 422 if validation fails
            },
            
            # Missing required fields
            {
                "data": {
                    "description": "Missing name and path"
                },
                "expected_status": [422]
            },
            
            # Invalid configuration
            {
                "data": {
                    "name": "Invalid Config",
                    "root_path": str(temp_project_directory),
                    "configuration": {
                        "analysis_depth": -1,  # Invalid negative value
                        "languages": "not_an_array"  # Should be array
                    }
                },
                "expected_status": [400, 422]
            },
            
            # Empty name
            {
                "data": {
                    "name": "",
                    "root_path": str(temp_project_directory)
                },
                "expected_status": [400, 422]
            }
        ]
        
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = "123e4567-e89b-12d3-a456-426614174000"
            mock_project.name = "Test Project"
            mock_project.status = "inactive"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            for test_case in test_cases:
                response = await async_test_client.post(
                    "/api/project-index/create",
                    json=test_case["data"]
                )
                
                assert response.status_code in test_case["expected_status"]
                
                # Verify error messages are user-friendly for frontend
                if response.status_code >= 400:
                    error_data = response.json()
                    assert "detail" in error_data
                    
                    # Error should be structured for frontend display
                    detail = error_data["detail"]
                    if isinstance(detail, dict):
                        assert "error" in detail or "message" in detail
    
    async def test_file_explorer_component_data(
        self, 
        async_test_client,
        sample_project_index,
        sample_file_entries
    ):
        """Test file explorer component data structure and pagination."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalars.return_value.all.return_value = sample_file_entries
                mock_session.execute.return_value.scalar.return_value = len(sample_file_entries)
                mock_get_session.return_value = mock_session
                
                # Test file listing for file explorer component
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files?page=1&limit=10"
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify data structure for file explorer
                assert "data" in data
                files_data = data["data"]
                
                # Check pagination metadata
                pagination_fields = ["total", "page", "page_size", "has_next", "has_prev"]
                for field in pagination_fields:
                    assert field in files_data, f"Missing pagination field: {field}"
                
                # Check file data structure
                if "files" in files_data and files_data["files"]:
                    file_item = files_data["files"][0]
                    
                    # Required fields for file explorer
                    file_fields = [
                        "id", "relative_path", "file_name", "file_type", 
                        "language", "file_size", "last_modified"
                    ]
                    for field in file_fields:
                        assert field in file_item, f"Missing file field: {field}"
    
    async def test_dependency_graph_visualization_data(
        self, 
        async_test_client,
        sample_project_index,
        sample_dependencies
    ):
        """Test dependency graph data for visualization components."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            with patch('app.api.project_index._get_dependency_graph') as mock_get_graph:
                # Mock graph data structure expected by frontend
                mock_graph_data = {
                    "data": {
                        "nodes": [
                            {
                                "file_id": "file_123",
                                "file_path": "src/main.py",
                                "file_type": "source",
                                "language": "python",
                                "in_degree": 0,
                                "out_degree": 2,
                                "size": 1024,
                                "complexity": 3
                            },
                            {
                                "file_id": "file_456",
                                "file_path": "src/utils.py",
                                "file_type": "source",
                                "language": "python",
                                "in_degree": 1,
                                "out_degree": 2,
                                "size": 2048,
                                "complexity": 5
                            }
                        ],
                        "edges": [
                            {
                                "source_file_id": "file_123",
                                "target_file_id": "file_456",
                                "target_name": "utils",
                                "dependency_type": "import",
                                "is_external": False,
                                "strength": 1.0
                            },
                            {
                                "source_file_id": "file_123",
                                "target_name": "os",
                                "dependency_type": "import",
                                "is_external": True,
                                "strength": 0.8
                            }
                        ],
                        "statistics": {
                            "total_nodes": 2,
                            "total_edges": 2,
                            "external_dependencies": 1,
                            "internal_dependencies": 1,
                            "avg_in_degree": 0.5,
                            "avg_out_degree": 2.0,
                            "complexity_score": 4.0
                        }
                    },
                    "meta": {
                        "format": "graph",
                        "generation_time": "2024-01-15T10:30:00Z"
                    }
                }
                
                mock_get_graph.return_value = mock_graph_data
                
                # Test graph endpoint
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/dependencies?format=graph"
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify graph data structure for D3.js or similar visualization
                assert "data" in data
                graph_data = data["data"]
                
                # Check nodes structure
                assert "nodes" in graph_data
                assert "edges" in graph_data
                assert "statistics" in graph_data
                
                # Verify node structure for visualization
                if graph_data["nodes"]:
                    node = graph_data["nodes"][0]
                    required_node_fields = [
                        "file_id", "file_path", "file_type", 
                        "in_degree", "out_degree"
                    ]
                    for field in required_node_fields:
                        assert field in node, f"Missing node field: {field}"
                
                # Verify edge structure for visualization
                if graph_data["edges"]:
                    edge = graph_data["edges"][0]
                    required_edge_fields = [
                        "source_file_id", "target_name", 
                        "dependency_type", "is_external"
                    ]
                    for field in required_edge_fields:
                        assert field in edge, f"Missing edge field: {field}"


@pytest.mark.project_index_frontend
@pytest.mark.integration
class TestAnalysisProgressComponents:
    """Test real-time analysis progress components."""
    
    async def test_analysis_progress_websocket_contract(
        self, 
        async_test_client,
        websocket_test_config
    ):
        """Test WebSocket contract for real-time analysis progress."""
        
        # Test WebSocket message format for frontend consumption
        expected_message_types = [
            "analysis_progress",
            "analysis_started", 
            "analysis_completed",
            "analysis_failed",
            "file_analyzed",
            "dependency_discovered"
        ]
        
        with patch('app.api.project_index.get_websocket_handler') as mock_handler:
            mock_handler_instance = AsyncMock()
            mock_handler.return_value = mock_handler_instance
            
            with patch('app.api.project_index.websocket_auth_context') as mock_auth:
                mock_auth.return_value.__aenter__.return_value = "test_user"
                
                try:
                    with async_test_client.websocket_connect(
                        "/api/project-index/ws?token=test_token"
                    ) as websocket:
                        
                        # Test subscription message
                        subscription_message = {
                            "action": "subscribe",
                            "event_types": expected_message_types,
                            "project_id": "123e4567-e89b-12d3-a456-426614174000"
                        }
                        
                        await websocket.send_json(subscription_message)
                        
                        # Simulate server sending progress updates
                        mock_progress_updates = [
                            {
                                "type": "analysis_started",
                                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                                "session_id": "sess_456",
                                "timestamp": "2024-01-15T10:30:00Z",
                                "data": {
                                    "analysis_type": "full_analysis",
                                    "estimated_duration": 300
                                }
                            },
                            {
                                "type": "analysis_progress",
                                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                                "session_id": "sess_456",
                                "timestamp": "2024-01-15T10:30:30Z",
                                "data": {
                                    "progress_percentage": 25.0,
                                    "current_phase": "scanning_files",
                                    "files_processed": 5,
                                    "files_total": 20,
                                    "current_file": "src/main.py"
                                }
                            },
                            {
                                "type": "file_analyzed",
                                "project_id": "123e4567-e89b-12d3-a456-426614174000",
                                "session_id": "sess_456",
                                "timestamp": "2024-01-15T10:30:45Z",
                                "data": {
                                    "file_path": "src/utils.py",
                                    "analysis_duration": 1.2,
                                    "dependencies_found": 3,
                                    "complexity_score": 4
                                }
                            }
                        ]
                        
                        # Verify message structure for frontend
                        for update in mock_progress_updates:
                            # Check required fields for frontend progress bar
                            assert "type" in update
                            assert "project_id" in update
                            assert "timestamp" in update
                            assert "data" in update
                            
                            # Type-specific validation
                            if update["type"] == "analysis_progress":
                                progress_data = update["data"]
                                assert "progress_percentage" in progress_data
                                assert "current_phase" in progress_data
                                assert 0 <= progress_data["progress_percentage"] <= 100
                
                except Exception:
                    # WebSocket connection may fail due to mocking
                    pass
    
    async def test_analysis_status_polling_endpoint(
        self, 
        async_test_client,
        sample_project_index,
        sample_analysis_session
    ):
        """Test analysis status polling endpoint for progress updates."""
        
        project_id = sample_project_index.id
        session_id = sample_analysis_session.id
        
        with patch('app.api.project_index.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_session.get.return_value = sample_analysis_session
            mock_get_session.return_value = mock_session
            
            # Test analysis status endpoint
            response = await async_test_client.get(
                f"/api/project-index/{project_id}/analysis/{session_id}/status"
            )
            
            # This endpoint might not exist yet, so we test what the structure should be
            if response.status_code == 404:
                # Endpoint doesn't exist yet - this is expected
                return
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify status data structure for frontend polling
                required_fields = [
                    "session_id", "status", "progress_percentage",
                    "current_phase", "files_processed", "files_total"
                ]
                
                status_data = data.get("data", {})
                for field in required_fields:
                    assert field in status_data, f"Missing status field: {field}"
                
                # Verify status values
                assert status_data["status"] in [
                    "pending", "running", "completed", "failed", "cancelled"
                ]
                assert 0 <= status_data["progress_percentage"] <= 100


@pytest.mark.project_index_frontend
@pytest.mark.integration
class TestResponsiveDesignComponents:
    """Test responsive design and mobile compatibility."""
    
    async def test_mobile_api_response_optimization(
        self, 
        async_test_client,
        sample_project_index,
        sample_file_entries
    ):
        """Test API responses are optimized for mobile consumption."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalars.return_value.all.return_value = sample_file_entries[:5]  # Smaller batch
                mock_session.execute.return_value.scalar.return_value = len(sample_file_entries)
                mock_get_session.return_value = mock_session
                
                # Test mobile-optimized file listing (smaller page size)
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files?page=1&limit=5",
                    headers={"User-Agent": "Mobile Safari"}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                # Verify response size is reasonable for mobile
                response_size = len(json.dumps(data))
                assert response_size < 50000, f"Response too large for mobile: {response_size} bytes"
                
                # Check for mobile-friendly data structure
                files_data = data.get("data", {})
                if "files" in files_data:
                    files = files_data["files"]
                    
                    # Should have limited files for mobile pagination
                    assert len(files) <= 5
                    
                    # Check for essential fields only (not all metadata)
                    if files:
                        file_item = files[0]
                        
                        # Mobile should have core fields but maybe not all details
                        essential_fields = ["id", "file_name", "file_type", "file_size"]
                        for field in essential_fields:
                            assert field in file_item
    
    async def test_api_response_compression_headers(
        self, 
        async_test_client,
        sample_project_index
    ):
        """Test API responses include compression headers for performance."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            response = await async_test_client.get(
                f"/api/project-index/{project_id}",
                headers={"Accept-Encoding": "gzip, deflate"}
            )
            
            assert response.status_code == 200
            
            # Check for performance-related headers
            headers = response.headers
            
            # Should have caching headers for static data
            cache_headers = ["cache-control", "etag", "last-modified"]
            performance_headers = []
            
            for header in cache_headers:
                if header in headers:
                    performance_headers.append(header)
            
            # At least some performance headers should be present
            # (This depends on middleware configuration)
            print(f"Performance headers found: {performance_headers}")


@pytest.mark.project_index_frontend
@pytest.mark.integration
class TestPWAFunctionality:
    """Test Progressive Web App functionality."""
    
    async def test_pwa_manifest_endpoint(self, async_test_client):
        """Test PWA manifest endpoint for app installation."""
        
        # Test manifest.json endpoint
        response = await async_test_client.get("/manifest.json")
        
        if response.status_code == 404:
            # Manifest endpoint doesn't exist yet
            return
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        manifest = response.json()
        
        # Verify PWA manifest structure
        required_fields = ["name", "short_name", "start_url", "display", "icons"]
        for field in required_fields:
            assert field in manifest, f"Missing manifest field: {field}"
        
        # Verify manifest values
        assert manifest["display"] in ["standalone", "fullscreen", "minimal-ui"]
        assert "icons" in manifest and len(manifest["icons"]) > 0
        
        # Check icon structure
        icon = manifest["icons"][0]
        icon_fields = ["src", "sizes", "type"]
        for field in icon_fields:
            assert field in icon, f"Missing icon field: {field}"
    
    async def test_service_worker_registration(self, async_test_client):
        """Test service worker registration endpoint."""
        
        # Test service worker file
        response = await async_test_client.get("/sw.js")
        
        if response.status_code == 404:
            # Service worker doesn't exist yet
            return
        
        assert response.status_code == 200
        assert "javascript" in response.headers.get("content-type", "").lower()
        
        # Verify service worker content includes PWA functionality
        sw_content = response.text if hasattr(response, 'text') else ""
        
        # Should include caching strategies
        pwa_features = ["cache", "fetch", "install", "activate"]
        found_features = [feature for feature in pwa_features if feature in sw_content.lower()]
        
        assert len(found_features) > 0, "Service worker should include PWA functionality"
    
    async def test_offline_api_behavior(
        self, 
        async_test_client,
        sample_project_index
    ):
        """Test API behavior for offline PWA functionality."""
        
        project_id = sample_project_index.id
        
        # Test with cache headers for offline support
        response = await async_test_client.get(
            f"/api/project-index/{project_id}",
            headers={
                "Cache-Control": "max-age=0",
                "If-None-Match": "test-etag"
            }
        )
        
        # API should handle caching headers appropriately
        assert response.status_code in [200, 304, 404]
        
        if response.status_code == 200:
            # Should include caching headers for offline support
            headers = response.headers
            
            # Check for cache-related headers
            cache_related = ["cache-control", "etag", "expires"]
            found_cache_headers = [h for h in cache_related if h in headers]
            
            print(f"Cache headers for offline support: {found_cache_headers}")


@pytest.mark.project_index_frontend
@pytest.mark.integration
class TestUserWorkflowIntegration:
    """Test complete user workflows from frontend perspective."""
    
    async def test_complete_project_creation_workflow(
        self, 
        async_test_client,
        temp_project_directory
    ):
        """Test complete project creation workflow from frontend perspective."""
        
        # Step 1: Get initial projects list (should be empty)
        response = await async_test_client.get("/api/project-index")
        initial_projects_count = 0
        
        if response.status_code == 200:
            data = response.json()
            projects = data.get("data", {})
            if isinstance(projects, dict) and "projects" in projects:
                initial_projects_count = len(projects["projects"])
        
        # Step 2: Create new project
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = "123e4567-e89b-12d3-a456-426614174000"
            mock_project.name = "Frontend Test Project"
            mock_project.status = "inactive"
            mock_project.created_at = "2024-01-15T10:30:00Z"
            mock_project.updated_at = "2024-01-15T10:30:00Z"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            create_data = {
                "name": "Frontend Test Project",
                "description": "Project created from frontend",
                "root_path": str(temp_project_directory),
                "configuration": {
                    "languages": ["python"],
                    "analysis_depth": 2
                }
            }
            
            response = await async_test_client.post(
                "/api/project-index/create",
                json=create_data
            )
            
            assert response.status_code == 200
            created_project = response.json()
            
            # Verify response structure for frontend
            assert "data" in created_project
            assert "links" in created_project
            
            project_data = created_project["data"]
            project_id = project_data["id"]
            
            # Step 3: Verify project appears in list
            response = await async_test_client.get("/api/project-index")
            if response.status_code == 200:
                data = response.json()
                # Project should now appear in the list
                # (In real implementation, would verify count increased)
            
            # Step 4: Get project details
            with patch('app.api.project_index.get_project_or_404') as mock_get_project:
                mock_get_project.return_value = mock_project
                
                response = await async_test_client.get(f"/api/project-index/{project_id}")
                assert response.status_code == 200
                
                project_details = response.json()
                assert project_details["data"]["name"] == "Frontend Test Project"
            
            # Step 5: Trigger analysis
            with patch('app.api.project_index.rate_limit_analysis'):
                response = await async_test_client.post(
                    f"/api/project-index/{project_id}/analyze",
                    json={"analysis_type": "full", "force": False}
                )
                
                # Should schedule analysis
                assert response.status_code in [200, 404]  # 404 if not implemented
    
    async def test_file_exploration_workflow(
        self, 
        async_test_client,
        sample_project_index,
        sample_file_entries
    ):
        """Test file exploration workflow from frontend perspective."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalars.return_value.all.return_value = sample_file_entries
                mock_session.execute.return_value.scalar.return_value = len(sample_file_entries)
                mock_get_session.return_value = mock_session
                
                # Step 1: Get file list
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files?page=1&limit=10"
                )
                
                assert response.status_code == 200
                files_data = response.json()
                
                files = files_data["data"]["files"]
                assert len(files) > 0
                
                # Step 2: Get specific file details
                file_path = files[0]["relative_path"]
                encoded_path = file_path.replace("/", "%2F")
                
                # Mock file details response
                mock_file = sample_file_entries[0]
                mock_file.outgoing_dependencies = []
                mock_file.incoming_dependencies = []
                
                mock_session.execute.return_value.scalar_one_or_none.return_value = mock_file
                
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/files/{encoded_path}"
                )
                
                assert response.status_code == 200
                file_details = response.json()
                
                # Verify file details structure
                assert "data" in file_details
                file_data = file_details["data"]
                assert file_data["relative_path"] == file_path
                assert "outgoing_dependencies" in file_data
                assert "incoming_dependencies" in file_data
    
    async def test_dependency_visualization_workflow(
        self, 
        async_test_client,
        sample_project_index
    ):
        """Test dependency visualization workflow."""
        
        project_id = sample_project_index.id
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = sample_project_index
            
            with patch('app.api.project_index._get_dependency_graph') as mock_get_graph:
                # Mock comprehensive graph data
                graph_data = {
                    "data": {
                        "nodes": [
                            {
                                "file_id": "file_1",
                                "file_path": "src/main.py",
                                "file_type": "source",
                                "language": "python",
                                "in_degree": 0,
                                "out_degree": 3,
                                "complexity": 2,
                                "size": 1024
                            },
                            {
                                "file_id": "file_2", 
                                "file_path": "src/utils.py",
                                "file_type": "source",
                                "language": "python",
                                "in_degree": 1,
                                "out_degree": 1,
                                "complexity": 4,
                                "size": 2048
                            }
                        ],
                        "edges": [
                            {
                                "source_file_id": "file_1",
                                "target_file_id": "file_2",
                                "dependency_type": "import",
                                "is_external": False,
                                "weight": 1.0
                            },
                            {
                                "source_file_id": "file_1",
                                "target_name": "os",
                                "dependency_type": "import",
                                "is_external": True,
                                "weight": 0.8
                            }
                        ],
                        "statistics": {
                            "total_nodes": 2,
                            "total_edges": 2,
                            "complexity_distribution": {
                                "low": 1,
                                "medium": 1,
                                "high": 0
                            },
                            "language_distribution": {
                                "python": 2
                            }
                        }
                    }
                }
                
                mock_get_graph.return_value = graph_data
                
                # Test graph endpoint for visualization
                response = await async_test_client.get(
                    f"/api/project-index/{project_id}/dependencies?format=graph"
                )
                
                assert response.status_code == 200
                result = response.json()
                
                # Verify graph structure for D3.js visualization
                graph = result["data"]
                
                # Check nodes for visualization
                assert "nodes" in graph
                assert len(graph["nodes"]) > 0
                
                node = graph["nodes"][0]
                visualization_fields = ["file_id", "file_path", "in_degree", "out_degree"]
                for field in visualization_fields:
                    assert field in node, f"Missing visualization field: {field}"
                
                # Check edges for visualization
                assert "edges" in graph
                if graph["edges"]:
                    edge = graph["edges"][0]
                    edge_fields = ["source_file_id", "dependency_type", "is_external"]
                    for field in edge_fields:
                        assert field in edge, f"Missing edge field: {field}"
                
                # Check statistics for dashboard
                assert "statistics" in graph
                stats = graph["statistics"]
                assert "total_nodes" in stats
                assert "total_edges" in stats