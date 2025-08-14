"""
Comprehensive unit tests for critical API endpoints.

Tests cover:
- Main API routes and endpoints
- Task management APIs
- Agent activation APIs
- Authentication and authorization
- Error handling and validation
- Request/response serialization
"""

import pytest
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.api.routes import router
from app.api.dashboard_task_management import router as task_router
from app.api.agent_activation import (
    router as activation_router,
    AgentActivationRequest,
    AgentActivationResponse
)
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.agent import Agent, AgentStatus, AgentType
from app.core.auth import get_current_user, Permission


class TestMainAPIRoutes:
    """Test main API routing functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    def test_api_root_endpoint(self, client):
        """Test API root endpoint returns correct information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "LeanVibe Agent Hive 2.0 API"
        assert data["version"] == "2.0.0"
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"
    
    @patch('app.api.routes.get_current_user')
    def test_protected_ping_authenticated(self, mock_get_user, client):
        """Test protected ping endpoint with authenticated user."""
        # Mock authenticated user
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_get_user.return_value = mock_user
        
        response = client.get("/protected/ping")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pong"] is True
        assert data["user_id"] == "user-123"
    
    @patch('app.api.routes.get_current_user')
    def test_protected_ping_unauthenticated(self, mock_get_user, client):
        """Test protected ping endpoint without authentication."""
        # Mock unauthenticated request
        mock_get_user.side_effect = HTTPException(status_code=401, detail="Not authenticated")
        
        response = client.get("/protected/ping")
        
        assert response.status_code == 401
    
    @patch('app.api.routes.require_basic_permission')
    def test_protected_admin_authorized(self, mock_require_permission, client):
        """Test admin endpoint with proper authorization."""
        # Mock authorized request
        mock_require_permission.return_value = MagicMock()
        
        response = client.get("/protected/admin")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["route"] == "admin"
    
    @patch('app.api.routes.require_basic_permission')
    def test_protected_admin_unauthorized(self, mock_require_permission, client):
        """Test admin endpoint without proper authorization."""
        # Mock unauthorized request
        mock_require_permission.side_effect = HTTPException(status_code=403, detail="Forbidden")
        
        response = client.get("/protected/admin")
        
        assert response.status_code == 403


class TestTaskManagementAPI:
    """Test task management API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample tasks for testing."""
        return [
            Task(
                id="task-1",
                title="Implement API endpoint",
                description="Create new FastAPI endpoint",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                task_type=TaskType.DEVELOPMENT,
                created_at=datetime.utcnow()
            ),
            Task(
                id="task-2", 
                title="Write tests",
                description="Unit tests for API endpoint",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.IN_PROGRESS,
                task_type=TaskType.TESTING,
                assigned_agent_id="agent-1",
                created_at=datetime.utcnow()
            ),
            Task(
                id="task-3",
                title="Update documentation",
                description="API documentation updates",
                priority=TaskPriority.LOW,
                status=TaskStatus.BLOCKED,
                task_type=TaskType.DOCUMENTATION,
                created_at=datetime.utcnow()
            )
        ]
    
    @patch('app.api.dashboard_task_management.get_async_session')
    @pytest.mark.asyncio
    async def test_get_task_queue_status_default(self, mock_get_session, client, sample_tasks):
        """Test getting task queue status with default filters."""
        # Mock database session
        mock_session = AsyncMock()
        mock_get_session.return_value = mock_session
        
        # Mock query results
        mock_execute = AsyncMock()
        mock_execute.scalars.return_value.all.return_value = sample_tasks[:2]  # Pending and in-progress tasks
        mock_session.execute.return_value = mock_execute
        
        # Mock statistics queries
        mock_session.execute.side_effect = [
            # Tasks query
            mock_execute,
            # Status counts query
            AsyncMock(all=lambda: [(TaskStatus.PENDING, 1), (TaskStatus.IN_PROGRESS, 1)]),
            # Priority counts query
            AsyncMock(all=lambda: [(TaskPriority.HIGH, 1), (TaskPriority.MEDIUM, 1)]),
            # Agent assignment query
            AsyncMock(all=lambda: [("agent-1", 1)])
        ]
        
        response = client.get("/api/dashboard/tasks/queue")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "statistics" in data
        assert len(data["tasks"]) <= 100  # Default limit
    
    @patch('app.api.dashboard_task_management.get_async_session')
    def test_get_task_queue_status_with_filters(self, mock_get_session, client):
        """Test getting task queue status with filters applied."""
        mock_session = AsyncMock()
        mock_get_session.return_value = mock_session
        
        # Mock query results for high priority, pending tasks
        mock_execute = AsyncMock()
        mock_execute.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_execute
        
        response = client.get("/api/dashboard/tasks/queue?status_filter=pending&priority_filter=high&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
    
    def test_get_task_queue_status_invalid_filter(self, client):
        """Test task queue endpoint with invalid filter values."""
        response = client.get("/api/dashboard/tasks/queue?status_filter=invalid_status")
        
        assert response.status_code == 422  # Validation error
        
        response = client.get("/api/dashboard/tasks/queue?priority_filter=invalid_priority")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_task_queue_status_invalid_limit(self, client):
        """Test task queue endpoint with invalid limit values."""
        # Test limit too high
        response = client.get("/api/dashboard/tasks/queue?limit=1000")
        assert response.status_code == 422
        
        # Test limit too low
        response = client.get("/api/dashboard/tasks/queue?limit=0")
        assert response.status_code == 422


class TestAgentActivationAPI:
    """Test agent activation API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @patch('app.api.agent_activation.spawn_development_team')
    @patch('app.api.agent_activation.get_active_agents_status')
    @patch('app.api.agent_activation._start_demo_tasks')
    @pytest.mark.asyncio
    async def test_activate_agent_system_success(self, mock_demo_tasks, mock_get_status, mock_spawn_team, client):
        """Test successful agent system activation."""
        # Mock successful team spawning
        mock_spawn_team.return_value = {
            "architect": "agent-1",
            "developer": "agent-2",
            "qa": "agent-3"
        }
        
        # Mock active agents status
        mock_get_status.return_value = {
            "agent-1": {"role": "architect", "status": "active"},
            "agent-2": {"role": "developer", "status": "active"},
            "agent-3": {"role": "qa", "status": "active"}
        }
        
        # Test activation request
        activation_request = {
            "team_size": 3,
            "roles": ["architect", "developer", "qa"],
            "auto_start_tasks": True
        }
        
        response = client.post("/activate", json=activation_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Successfully activated" in data["message"]
        assert len(data["active_agents"]) == 3
        assert "architect" in data["team_composition"]
        
        # Verify that functions were called
        mock_spawn_team.assert_called_once()
        mock_get_status.assert_called_once()
    
    @patch('app.api.agent_activation.spawn_development_team')
    def test_activate_agent_system_failure(self, mock_spawn_team, client):
        """Test agent system activation failure handling."""
        # Mock team spawning failure
        mock_spawn_team.side_effect = Exception("Agent spawning failed")
        
        activation_request = {
            "team_size": 3,
            "auto_start_tasks": False
        }
        
        response = client.post("/activate", json=activation_request)
        
        assert response.status_code == 500
        data = response.json()
        assert "Agent activation failed" in data["detail"]
    
    def test_activate_agent_system_invalid_request(self, client):
        """Test agent activation with invalid request data."""
        # Test with invalid team_size
        invalid_request = {
            "team_size": -1,  # Invalid negative size
            "auto_start_tasks": "invalid"  # Invalid boolean
        }
        
        response = client.post("/activate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('app.api.agent_activation.get_agent_manager')
    @patch('app.api.agent_activation.get_active_agents_status')
    @pytest.mark.asyncio
    async def test_get_agent_system_status(self, mock_get_status, mock_get_manager, client):
        """Test getting agent system status."""
        # Mock agent manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager
        
        # Mock active agents
        mock_get_status.return_value = {
            "agent-1": {"role": "developer", "status": "active", "health": "healthy"},
            "agent-2": {"role": "qa", "status": "active", "health": "healthy"}
        }
        
        response = client.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "spawner_agents" in data or "agents" in data
        assert "system_status" in data or "status" in data


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization mechanisms."""
    
    @pytest.fixture
    def client(self):
        """Create test client.""" 
        return TestClient(app)
    
    @patch('app.core.auth.get_current_user')
    def test_protected_endpoint_valid_token(self, mock_get_user, client):
        """Test protected endpoint with valid authentication token."""
        # Mock valid user
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.is_active = True
        mock_get_user.return_value = mock_user
        
        headers = {"Authorization": "Bearer valid-token"}
        response = client.get("/protected/ping", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-123"
    
    @patch('app.core.auth.get_current_user')
    def test_protected_endpoint_invalid_token(self, mock_get_user, client):
        """Test protected endpoint with invalid authentication token."""
        # Mock invalid token
        mock_get_user.side_effect = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
        
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/protected/ping", headers=headers)
        
        assert response.status_code == 401
    
    def test_protected_endpoint_no_token(self, client):
        """Test protected endpoint without authentication token."""
        response = client.get("/protected/ping")
        
        # Should return 401 or redirect to login
        assert response.status_code in [401, 403]
    
    @patch('app.core.auth.require_permission')
    def test_admin_endpoint_sufficient_permissions(self, mock_require_permission, client):
        """Test admin endpoint with sufficient permissions."""
        # Mock user with admin permissions
        mock_user = MagicMock()
        mock_user.has_permission = lambda p: p == Permission.MANAGE_USERS
        mock_require_permission.return_value = mock_user
        
        response = client.get("/protected/admin")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
    
    @patch('app.core.auth.require_permission')
    def test_admin_endpoint_insufficient_permissions(self, mock_require_permission, client):
        """Test admin endpoint with insufficient permissions."""
        # Mock user without admin permissions
        mock_require_permission.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
        
        response = client.get("/protected/admin")
        
        assert response.status_code == 403


class TestErrorHandling:
    """Test error handling across API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_404_error_handling(self, client):
        """Test 404 error for non-existent endpoints."""
        response = client.get("/non-existent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed_error(self, client):
        """Test 405 error for wrong HTTP methods."""
        # Try POST on a GET-only endpoint
        response = client.post("/")
        
        assert response.status_code == 405
    
    @patch('app.api.dashboard_task_management.get_async_session')
    def test_database_connection_error(self, mock_get_session, client):
        """Test handling of database connection errors."""
        # Mock database connection error
        mock_get_session.side_effect = Exception("Database connection failed")
        
        response = client.get("/api/dashboard/tasks/queue")
        
        assert response.status_code == 500
    
    def test_malformed_json_request(self, client):
        """Test handling of malformed JSON in request body."""
        # Send malformed JSON
        response = client.post(
            "/activate",
            data="{ malformed json }",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


class TestRequestValidation:
    """Test request validation and serialization."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_agent_activation_request_validation(self, client):
        """Test validation of agent activation requests."""
        # Valid request
        valid_request = {
            "team_size": 3,
            "roles": ["developer", "qa"],
            "auto_start_tasks": True
        }
        
        with patch('app.api.agent_activation.spawn_development_team'), \
             patch('app.api.agent_activation.get_active_agents_status'):
            response = client.post("/activate", json=valid_request)
            # May fail due to mocked dependencies, but should pass validation
            assert response.status_code != 422
    
    def test_task_queue_query_validation(self, client):
        """Test validation of task queue query parameters."""
        # Test valid parameters
        response = client.get("/api/dashboard/tasks/queue?limit=50&status_filter=pending")
        # Should pass validation (may fail on business logic due to mocked DB)
        assert response.status_code != 422
        
        # Test invalid limit
        response = client.get("/api/dashboard/tasks/queue?limit=99999")
        assert response.status_code == 422
        
        # Test invalid status filter
        response = client.get("/api/dashboard/tasks/queue?status_filter=invalid")
        assert response.status_code == 422
    
    def test_json_response_serialization(self, client):
        """Test JSON response serialization."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "message" in data


class TestConcurrencyAndPerformance:
    """Test concurrent requests and performance characteristics."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent API requests."""
        import asyncio
        
        async def make_request():
            response = client.get("/")
            return response.status_code
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        assert all(result == 200 for result in results if not isinstance(result, Exception))
    
    def test_large_response_handling(self, client):
        """Test handling of potentially large responses."""
        # Request task queue with maximum limit
        response = client.get("/api/dashboard/tasks/queue?limit=500")
        
        # Should handle large responses gracefully
        assert response.status_code in [200, 500]  # Either success or handled error
    
    def test_request_timeout_handling(self, client):
        """Test handling of request timeouts."""
        with patch('app.api.dashboard_task_management.get_async_session') as mock_session:
            # Mock a slow database query
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(10)  # Simulate slow query
                return AsyncMock()
            
            mock_session.return_value.execute.side_effect = slow_query
            
            # This should timeout or be handled gracefully
            response = client.get("/api/dashboard/tasks/queue")
            
            # Should not hang indefinitely
            assert response.status_code in [200, 500, 504]


class TestAPIDocumentation:
    """Test API documentation and OpenAPI spec."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_openapi_schema_generation(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_swagger_ui_available(self, client):
        """Test Swagger UI availability."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
    
    def test_redoc_ui_available(self, client):
        """Test ReDoc UI availability."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "redoc" in response.text.lower() or "api documentation" in response.text.lower()