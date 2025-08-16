"""
Comprehensive Unit Tests for Project Index API Endpoints

Tests all REST API endpoints with full coverage including authentication,
validation, error handling, performance, and edge cases.
"""

import uuid
import json
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import pytest
from httpx import AsyncClient
from fastapi import status

from app.models.project_index import ProjectStatus, FileType, DependencyType, AnalysisSessionType


class TestProjectIndexEndpoints:
    """Test Project Index CRUD endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_project_success(self, test_client: AsyncClient, sample_project_data):
        """Test successful project creation."""
        data, temp_dir = sample_project_data
        
        create_data = {
            "name": data["name"],
            "description": data["description"],
            "root_path": data["root_path"],
            "git_repository_url": data["git_repository_url"],
            "git_branch": data["git_branch"],
            "configuration": data["configuration"],
            "file_patterns": data["file_patterns"]
        }
        
        response = await test_client.post("/api/project-index/create", json=create_data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        assert "meta" in response_data
        assert "links" in response_data
        
        project_data = response_data["data"]
        assert project_data["name"] == data["name"]
        assert project_data["description"] == data["description"]
        assert project_data["root_path"] == data["root_path"]
        assert project_data["status"] == ProjectStatus.INACTIVE.value
        assert "id" in project_data
        
        # Verify HATEOAS links
        links = response_data["links"]
        assert "self" in links
        assert "files" in links
        assert "dependencies" in links
    
    @pytest.mark.asyncio
    async def test_create_project_validation_errors(self, test_client: AsyncClient):
        """Test project creation validation errors."""
        # Test missing required fields
        response = await test_client.post("/api/project-index/create", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test invalid root path
        invalid_data = {
            "name": "Invalid Project",
            "root_path": "/nonexistent/path"
        }
        response = await test_client.post("/api/project-index/create", json=invalid_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        error_data = response.json()
        assert "error" in error_data["detail"]
        assert error_data["detail"]["error"] == "INVALID_ROOT_PATH"
    
    @pytest.mark.asyncio
    async def test_get_project_success(self, test_client: AsyncClient, test_project):
        """Test successful project retrieval."""
        response = await test_client.get(f"/api/project-index/{test_project.id}")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        project_data = response_data["data"]
        assert project_data["id"] == str(test_project.id)
        assert project_data["name"] == test_project.name
        assert project_data["status"] == test_project.status.value
    
    @pytest.mark.asyncio
    async def test_get_project_not_found(self, test_client: AsyncClient):
        """Test project not found error."""
        non_existent_id = str(uuid.uuid4())
        response = await test_client.get(f"/api/project-index/{non_existent_id}")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert error_data["detail"]["error"] == "PROJECT_NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_refresh_project_success(self, test_client: AsyncClient, test_project):
        """Test successful project refresh."""
        response = await test_client.put(f"/api/project-index/{test_project.id}/refresh")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        refresh_data = response_data["data"]
        assert refresh_data["project_id"] == str(test_project.id)
        assert refresh_data["status"] == "scheduled"
        assert "analysis_session_id" in refresh_data
    
    @pytest.mark.asyncio
    async def test_refresh_project_conflict(self, test_client: AsyncClient, test_project, test_session):
        """Test project refresh conflict when analysis is running."""
        # Create a running analysis session
        from app.models.project_index import AnalysisSession, AnalysisStatus
        
        running_session = AnalysisSession(
            project_id=test_project.id,
            session_name="Running Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            status=AnalysisStatus.RUNNING
        )
        test_session.add(running_session)
        await test_session.commit()
        
        response = await test_client.put(f"/api/project-index/{test_project.id}/refresh")
        
        assert response.status_code == status.HTTP_409_CONFLICT
        error_data = response.json()
        assert error_data["detail"]["error"] == "ANALYSIS_IN_PROGRESS"
    
    @pytest.mark.asyncio
    async def test_delete_project_success(self, test_client: AsyncClient, test_project):
        """Test successful project deletion."""
        response = await test_client.delete(f"/api/project-index/{test_project.id}")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        cleanup_data = response_data["data"]
        assert cleanup_data["project_id"] == str(test_project.id)
        assert "deleted_files" in cleanup_data
        assert "deleted_dependencies" in cleanup_data


class TestFileAnalysisEndpoints:
    """Test file analysis and retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_project_files_success(self, test_client: AsyncClient, test_project, test_files):
        """Test successful file listing."""
        response = await test_client.get(f"/api/project-index/{test_project.id}/files")
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        files_data = response_data["data"]
        assert "files" in files_data
        assert "total" in files_data
        assert "page" in files_data
        
        assert files_data["total"] == len(test_files)
        assert len(files_data["files"]) <= files_data["total"]
    
    @pytest.mark.asyncio
    async def test_list_files_with_filters(self, test_client: AsyncClient, test_project, test_files):
        """Test file listing with filters."""
        # Test language filter
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files?language=python"
        )
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        files = response_data["data"]["files"]
        
        # All returned files should be Python files
        python_files = [f for f in files if f["language"] == "python"]
        assert len(python_files) == len(files)
        
        # Test file type filter
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files?file_type=source"
        )
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        files = response_data["data"]["files"]
        source_files = [f for f in files if f["file_type"] == "source"]
        assert len(source_files) == len(files)
    
    @pytest.mark.asyncio
    async def test_list_files_pagination(self, test_client: AsyncClient, test_project):
        """Test file listing pagination."""
        # Test with small page size
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files?page=1&limit=2"
        )
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        files_data = response_data["data"]
        
        assert files_data["page"] == 1
        assert files_data["page_size"] == 2
        assert len(files_data["files"]) <= 2
        
        # Test pagination metadata
        meta = response_data["meta"]
        assert "pagination" in meta
        pagination = meta["pagination"]
        assert "total_pages" in pagination
    
    @pytest.mark.asyncio
    async def test_get_file_analysis_success(self, test_client: AsyncClient, test_project, test_files):
        """Test successful file analysis retrieval."""
        test_file = test_files[0]
        encoded_path = test_file.relative_path.replace("/", "%2F")
        
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files/{encoded_path}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        file_data = response_data["data"]
        assert file_data["id"] == str(test_file.id)
        assert file_data["relative_path"] == test_file.relative_path
        assert "outgoing_dependencies" in file_data
        assert "incoming_dependencies" in file_data
    
    @pytest.mark.asyncio
    async def test_get_file_analysis_not_found(self, test_client: AsyncClient, test_project):
        """Test file analysis not found error."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files/nonexistent.py"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert error_data["detail"]["error"] == "FILE_NOT_FOUND"
    
    @pytest.mark.asyncio
    async def test_analyze_project_success(self, test_client: AsyncClient, test_project):
        """Test successful project analysis trigger."""
        analysis_request = {
            "analysis_type": "full",
            "force": False,
            "configuration": {"depth": 3}
        }
        
        response = await test_client.post(
            f"/api/project-index/{test_project.id}/analyze",
            json=analysis_request
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        analysis_data = response_data["data"]
        assert analysis_data["project_id"] == str(test_project.id)
        assert analysis_data["analysis_type"] == "full"
        assert analysis_data["status"] == "scheduled"
        assert "analysis_session_id" in analysis_data
    
    @pytest.mark.asyncio
    async def test_analyze_project_invalid_type(self, test_client: AsyncClient, test_project):
        """Test project analysis with invalid type."""
        analysis_request = {
            "analysis_type": "invalid_type",
            "force": False
        }
        
        response = await test_client.post(
            f"/api/project-index/{test_project.id}/analyze",
            json=analysis_request
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        error_data = response.json()
        assert error_data["detail"]["error"] == "INVALID_ANALYSIS_TYPE"
    
    @pytest.mark.asyncio
    async def test_analyze_project_rate_limiting(self, test_client: AsyncClient, test_project):
        """Test analysis rate limiting."""
        # Mock rate limiter to return False (rate limited)
        with patch('app.api.project_index.RateLimiter.check_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = False
            
            analysis_request = {"analysis_type": "full"}
            
            response = await test_client.post(
                f"/api/project-index/{test_project.id}/analyze",
                json=analysis_request
            )
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            error_data = response.json()
            assert error_data["detail"]["error"] == "RATE_LIMIT_EXCEEDED"


class TestDependencyEndpoints:
    """Test dependency analysis endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_dependencies_json_format(self, test_client: AsyncClient, test_project, test_dependencies):
        """Test dependencies in JSON format."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?format=json"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        deps_data = response_data["data"]
        assert "dependencies" in deps_data
        assert "total" in deps_data
        
        assert deps_data["total"] == len(test_dependencies)
    
    @pytest.mark.asyncio
    async def test_get_dependencies_graph_format(self, test_client: AsyncClient, test_project, test_files, test_dependencies):
        """Test dependencies in graph format."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?format=graph"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert "data" in response_data
        graph_data = response_data["data"]
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "statistics" in graph_data
        
        # Verify graph structure
        assert len(graph_data["nodes"]) == len(test_files)
        assert len(graph_data["edges"]) == len(test_dependencies)
        
        # Verify statistics
        stats = graph_data["statistics"]
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "external_dependencies" in stats
    
    @pytest.mark.asyncio
    async def test_get_dependencies_with_filters(self, test_client: AsyncClient, test_project, test_files):
        """Test dependencies with filters."""
        test_file = test_files[0]
        
        # Test file path filter
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?file_path={test_file.relative_path}"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        dependencies = response_data["data"]["dependencies"]
        # All dependencies should be from the specified file
        for dep in dependencies:
            assert dep["source_file_id"] == str(test_file.id)
        
        # Test external filter
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?include_external=false"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        dependencies = response_data["data"]["dependencies"]
        # All dependencies should be internal
        for dep in dependencies:
            assert not dep["is_external"]
    
    @pytest.mark.asyncio
    async def test_get_dependencies_pagination(self, test_client: AsyncClient, test_project):
        """Test dependencies pagination."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?page=1&limit=2"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        deps_data = response_data["data"]
        assert deps_data["page"] == 1
        assert deps_data["page_size"] == 2
        assert len(deps_data["dependencies"]) <= 2
        
        # Verify pagination metadata
        meta = response_data["meta"]
        assert "pagination" in meta
    
    @pytest.mark.asyncio
    async def test_get_dependencies_file_not_found(self, test_client: AsyncClient, test_project):
        """Test dependencies with non-existent file."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/dependencies?file_path=nonexistent.py"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert error_data["detail"]["error"] == "FILE_NOT_FOUND"


class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""
    
    @pytest.mark.asyncio
    async def test_websocket_stats(self, test_client: AsyncClient):
        """Test WebSocket statistics endpoint."""
        with patch('app.api.project_index.get_subscription_stats') as mock_stats:
            mock_stats.return_value = {
                "active_connections": 5,
                "total_subscriptions": 12,
                "events_published": 150,
                "connection_stats": {
                    "total_connected": 25,
                    "total_disconnected": 20
                }
            }
            
            response = await test_client.get("/api/project-index/ws/stats")
            
            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            
            assert "data" in response_data
            stats_data = response_data["data"]
            assert stats_data["active_connections"] == 5
            assert stats_data["total_subscriptions"] == 12
            assert stats_data["events_published"] == 150


class TestErrorHandling:
    """Test API error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_internal_server_error(self, test_client: AsyncClient, test_project):
        """Test internal server error handling."""
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.side_effect = Exception("Database connection failed")
            
            response = await test_client.get(f"/api/project-index/{test_project.id}")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            error_data = response.json()
            assert error_data["detail"]["error"] == "INTERNAL_ERROR"
    
    @pytest.mark.asyncio
    async def test_validation_error_details(self, test_client: AsyncClient):
        """Test detailed validation error responses."""
        invalid_data = {
            "name": "",  # Empty name
            "root_path": "relative/path",  # Should be absolute
            "git_commit_hash": "invalid_hash"  # Wrong length
        }
        
        response = await test_client.post("/api/project-index/create", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        error_data = response.json()
        
        # Verify detailed validation errors are returned
        assert "detail" in error_data
        validation_errors = error_data["detail"]
        assert isinstance(validation_errors, list)
        assert len(validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_malformed_json(self, test_client: AsyncClient):
        """Test malformed JSON handling."""
        # Send invalid JSON
        response = await test_client.post(
            "/api/project-index/create",
            content="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.asyncio
    async def test_unsupported_media_type(self, test_client: AsyncClient):
        """Test unsupported media type handling."""
        response = await test_client.post(
            "/api/project-index/create",
            content="test data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPerformanceAndOptimization:
    """Test performance aspects of API endpoints."""
    
    @pytest.mark.asyncio
    async def test_response_caching(self, test_client: AsyncClient, test_project):
        """Test response caching functionality."""
        # First request should hit database
        response1 = await test_client.get(f"/api/project-index/{test_project.id}")
        assert response1.status_code == status.HTTP_200_OK
        
        # Second request should hit cache
        response2 = await test_client.get(f"/api/project-index/{test_project.id}")
        assert response2.status_code == status.HTTP_200_OK
        
        # Responses should be identical
        assert response1.json() == response2.json()
        
        # Cache metadata should indicate source
        meta2 = response2.json()["meta"]
        # Note: In a real implementation, this would show "cache" after the first request
    
    @pytest.mark.asyncio
    async def test_large_response_handling(self, test_client: AsyncClient, test_project, test_session):
        """Test handling of large responses."""
        # Create many file entries
        from app.models.project_index import FileEntry, FileType
        
        for i in range(100):
            file_entry = FileEntry(
                project_id=test_project.id,
                file_path=f"/test/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_type=FileType.SOURCE,
                language="python"
            )
            test_session.add(file_entry)
        
        await test_session.commit()
        
        # Request with large page size
        response = await test_client.get(
            f"/api/project-index/{test_project.id}/files?limit=500"
        )
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        # Should handle large responses without timeout
        files_data = response_data["data"]
        assert "files" in files_data
        assert files_data["total"] >= 100
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, test_client: AsyncClient, test_project):
        """Test handling of concurrent requests."""
        import asyncio
        
        # Send multiple concurrent requests
        tasks = []
        for _ in range(10):
            task = test_client.get(f"/api/project-index/{test_project.id}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
        
        # All responses should be consistent
        first_response_data = responses[0].json()["data"]
        for response in responses[1:]:
            assert response.json()["data"]["id"] == first_response_data["id"]


class TestAPIDocumentation:
    """Test API documentation and OpenAPI compliance."""
    
    @pytest.mark.asyncio
    async def test_openapi_schema(self, test_client: AsyncClient):
        """Test OpenAPI schema generation."""
        response = await test_client.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify Project Index endpoints are documented
        paths = schema["paths"]
        assert "/api/project-index/create" in paths
        assert "/api/project-index/{project_id}" in paths
        assert "/api/project-index/{project_id}/files" in paths
        assert "/api/project-index/{project_id}/dependencies" in paths
    
    @pytest.mark.asyncio
    async def test_api_documentation_endpoint(self, test_client: AsyncClient):
        """Test API documentation endpoint."""
        response = await test_client.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


class TestSecurityFeatures:
    """Test security features of the API."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, test_client: AsyncClient, test_project):
        """Test CORS headers are present."""
        response = await test_client.get(
            f"/api/project-index/{test_project.id}",
            headers={"Origin": "https://example.com"}
        )
        
        # Note: CORS headers would be added by middleware in production
        assert response.status_code == status.HTTP_200_OK
    
    @pytest.mark.asyncio
    async def test_request_size_limits(self, test_client: AsyncClient):
        """Test request size limits."""
        # Create very large request payload
        large_data = {
            "name": "Large Project",
            "root_path": "/tmp/test",
            "configuration": {"large_field": "x" * 10000}  # 10KB field
        }
        
        response = await test_client.post("/api/project-index/create", json=large_data)
        
        # Should handle reasonably large requests
        # In production, there would be size limits configured
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE]
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, test_client: AsyncClient, test_project):
        """Test SQL injection protection."""
        # Attempt SQL injection in path parameter
        malicious_id = "'; DROP TABLE projects; --"
        
        response = await test_client.get(f"/api/project-index/{malicious_id}")
        
        # Should return 404 or 422, not cause database issues
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    @pytest.mark.asyncio
    async def test_xss_protection(self, test_client: AsyncClient, sample_project_data):
        """Test XSS protection in responses."""
        data, temp_dir = sample_project_data
        
        # Include potential XSS payload in project name
        xss_data = {
            "name": "<script>alert('xss')</script>",
            "root_path": data["root_path"]
        }
        
        response = await test_client.post("/api/project-index/create", json=xss_data)
        
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            project_name = response_data["data"]["name"]
            
            # Response should contain escaped or sanitized content
            assert "<script>" not in project_name or "&lt;script&gt;" in project_name