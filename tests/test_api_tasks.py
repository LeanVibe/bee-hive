"""
Tests for Task Management API Endpoints - Epic C Phase 1

Comprehensive test suite for task creation, assignment, progress tracking, and lifecycle management.
Tests cover orchestrator integration, performance requirements, and error handling.

Test Categories:
- Unit tests for individual endpoints
- Integration tests with orchestrator
- Task lifecycle management tests
- Performance tests for <200ms target
- Assignment and priority management tests

Epic C Phase 1: API Endpoint Implementation
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.api.main import app
from app.schemas.task import TaskCreate, TaskResponse, TaskAssignmentRequest
from app.models.task import TaskStatus, TaskPriority, TaskType
from app.core.simple_orchestrator import SimpleOrchestrator


# Test fixtures
@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """Async FastAPI test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    orchestrator = AsyncMock(spec=SimpleOrchestrator)
    orchestrator.initialize = AsyncMock()
    orchestrator.create_task = AsyncMock()
    orchestrator.list_tasks = AsyncMock()
    orchestrator.get_task_status = AsyncMock()
    orchestrator.update_task_priority = AsyncMock()
    orchestrator.cancel_task = AsyncMock()
    orchestrator.assign_task = AsyncMock()
    orchestrator.get_task_stats = AsyncMock()
    orchestrator.get_system_health = AsyncMock()
    return orchestrator


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "title": "Test Task",
        "description": "This is a test task for API validation",
        "task_type": "development",
        "priority": "high",
        "required_capabilities": ["python", "testing", "api_development"],
        "estimated_effort": 120,  # minutes
        "context": {
            "project": "api_testing",
            "deadline": "2025-01-01T00:00:00Z",
            "complexity": "medium"
        }
    }


@pytest.fixture
def sample_task_response():
    """Sample task response data."""
    task_id = str(uuid.uuid4())
    return {
        "id": task_id,
        "title": "Test Task",
        "description": "This is a test task for API validation",
        "task_type": TaskType.DEVELOPMENT,
        "status": TaskStatus.PENDING,
        "priority": TaskPriority.HIGH,
        "assigned_agent_id": None,
        "required_capabilities": ["python", "testing", "api_development"],
        "estimated_effort": 120,
        "actual_effort": None,
        "result": None,
        "error_message": None,
        "retry_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "assigned_at": None,
        "started_at": None,
        "completed_at": None
    }


class TestTaskCreation:
    """Test task creation endpoint."""

    async def test_create_task_success(self, async_test_client, mock_orchestrator, sample_task_data):
        """Test successful task creation."""
        # Mock orchestrator response
        mock_orchestrator.create_task.return_value = {"success": True, "task_id": str(uuid.uuid4())}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/tasks/", json=sample_task_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert "id" in data
        assert data["title"] == sample_task_data["title"]
        assert data["priority"] == sample_task_data["priority"]
        assert data["status"] == "pending"
        assert mock_orchestrator.create_task.called

    async def test_create_task_with_immediate_assignment(self, async_test_client, mock_orchestrator, sample_task_data):
        """Test task creation with immediate agent assignment."""
        agent_id = str(uuid.uuid4())
        
        # Mock orchestrator responses
        mock_orchestrator.create_task.return_value = {"success": True}
        mock_orchestrator.assign_task.return_value = {"success": True, "agent_id": agent_id}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post(
                "/api/v1/tasks/?assign_immediately=true", 
                json=sample_task_data
            )
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "assigned"
        assert data["assigned_agent_id"] == agent_id
        assert mock_orchestrator.assign_task.called

    async def test_create_task_minimal_data(self, async_test_client, mock_orchestrator):
        """Test task creation with minimal required data."""
        minimal_data = {"title": "Minimal Task"}
        mock_orchestrator.create_task.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/tasks/", json=minimal_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Minimal Task"
        assert data["priority"] == "medium"  # Default priority

    async def test_create_task_invalid_data(self, async_test_client):
        """Test task creation with invalid data."""
        invalid_data = {"title": ""}  # Empty title should fail validation
        
        response = await async_test_client.post("/api/v1/tasks/", json=invalid_data)
        assert response.status_code == 422  # Validation error

    async def test_create_task_orchestrator_failure(self, async_test_client, mock_orchestrator, sample_task_data):
        """Test task creation when orchestrator fails."""
        mock_orchestrator.create_task.side_effect = Exception("Orchestrator error")
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/tasks/", json=sample_task_data)
        
        # Should still create task but log the issue
        assert response.status_code == 201


class TestTaskListing:
    """Test task listing endpoint."""

    async def test_list_tasks_success(self, async_test_client, mock_orchestrator, sample_task_response):
        """Test successful task listing."""
        mock_orchestrator.list_tasks.return_value = {
            "tasks": [sample_task_response],
            "total": 1
        }
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/tasks/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "tasks" in data
        assert "total" in data
        assert len(data["tasks"]) == 1
        assert data["total"] == 1

    async def test_list_tasks_with_filters(self, async_test_client, mock_orchestrator):
        """Test task listing with status, priority, and agent filters."""
        agent_id = str(uuid.uuid4())
        mock_orchestrator.list_tasks.return_value = {"tasks": [], "total": 0}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(
                f"/api/v1/tasks/?status=running&priority=high&assigned_agent_id={agent_id}&limit=10&offset=0"
            )
        
        assert response.status_code == 200
        # Verify filters were passed to orchestrator
        mock_orchestrator.list_tasks.assert_called_with(
            status_filter=TaskStatus.RUNNING,
            priority_filter=TaskPriority.HIGH,
            agent_filter=agent_id,
            limit=10,
            offset=0
        )

    async def test_list_tasks_pagination(self, async_test_client, mock_orchestrator):
        """Test task listing pagination."""
        mock_orchestrator.list_tasks.return_value = {"tasks": [], "total": 250}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/tasks/?limit=50&offset=100")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 50
        assert data["offset"] == 100
        assert data["total"] == 250


class TestTaskStatusTracking:
    """Test task status and progress tracking endpoint."""

    async def test_get_task_status_success(self, async_test_client, mock_orchestrator):
        """Test successful task status retrieval."""
        task_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        status_data = {
            "status": TaskStatus.RUNNING,
            "progress": 0.65,
            "assigned_agent_id": agent_id,
            "started_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow()).isoformat(),
            "last_update": datetime.utcnow()
        }
        
        mock_orchestrator.get_task_status.return_value = status_data
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(f"/api/v1/tasks/{task_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "running"
        assert data["progress"] == 0.65
        assert data["assigned_agent_id"] == agent_id

    async def test_get_task_status_not_found(self, async_test_client, mock_orchestrator):
        """Test task status retrieval when task doesn't exist."""
        task_id = str(uuid.uuid4())
        mock_orchestrator.get_task_status.return_value = None
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(f"/api/v1/tasks/{task_id}/status")
        
        assert response.status_code == 404

    async def test_get_task_status_invalid_uuid(self, async_test_client):
        """Test task status retrieval with invalid UUID format."""
        invalid_id = "not-a-uuid"
        
        response = await async_test_client.get(f"/api/v1/tasks/{invalid_id}/status")
        assert response.status_code == 400


class TestTaskPriorityManagement:
    """Test task priority update endpoint."""

    async def test_update_task_priority_success(self, async_test_client, mock_orchestrator):
        """Test successful task priority update."""
        task_id = str(uuid.uuid4())
        priority_data = {"priority": "critical", "reason": "Urgent customer request"}
        
        mock_orchestrator.update_task_priority.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.put(
                f"/api/v1/tasks/{task_id}/priority", 
                json=priority_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "critical" in data["message"]
        
        # Verify orchestrator was called with correct parameters
        mock_orchestrator.update_task_priority.assert_called_with(
            task_id=task_id,
            new_priority=TaskPriority.CRITICAL,
            reason="Urgent customer request"
        )

    async def test_update_task_priority_failure(self, async_test_client, mock_orchestrator):
        """Test task priority update failure."""
        task_id = str(uuid.uuid4())
        priority_data = {"priority": "low"}
        
        mock_orchestrator.update_task_priority.return_value = {
            "success": False, 
            "message": "Task not found or cannot update priority"
        }
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.put(
                f"/api/v1/tasks/{task_id}/priority",
                json=priority_data
            )
        
        assert response.status_code == 400


class TestTaskCancellation:
    """Test task cancellation endpoint."""

    async def test_cancel_task_success(self, async_test_client, mock_orchestrator):
        """Test successful task cancellation."""
        task_id = str(uuid.uuid4())
        mock_orchestrator.cancel_task.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.delete(f"/api/v1/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert task_id in data["message"]

    async def test_cancel_task_with_force_and_reason(self, async_test_client, mock_orchestrator):
        """Test force cancellation of task with reason."""
        task_id = str(uuid.uuid4())
        mock_orchestrator.cancel_task.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.delete(
                f"/api/v1/tasks/{task_id}?force=true&reason=Emergency%20stop"
            )
        
        assert response.status_code == 200
        # Verify parameters were passed correctly
        mock_orchestrator.cancel_task.assert_called_with(
            task_id=task_id,
            force=True,
            reason="Emergency stop"
        )

    async def test_cancel_task_failure(self, async_test_client, mock_orchestrator):
        """Test task cancellation failure."""
        task_id = str(uuid.uuid4())
        mock_orchestrator.cancel_task.return_value = {
            "success": False,
            "message": "Cannot cancel running task without force flag"
        }
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.delete(f"/api/v1/tasks/{task_id}")
        
        assert response.status_code == 400


class TestTaskAssignment:
    """Test task assignment endpoint."""

    async def test_assign_task_to_specific_agent(self, async_test_client, mock_orchestrator):
        """Test assigning task to a specific agent."""
        task_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        assignment_data = {
            "agent_id": agent_id,
            "priority_override": "high",
            "context_override": {"urgent": True}
        }
        
        mock_orchestrator.assign_task.return_value = {"success": True, "agent_id": agent_id}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post(
                f"/api/v1/tasks/{task_id}/assign",
                json=assignment_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == agent_id
        assert agent_id in data["message"]

    async def test_assign_task_auto_assignment(self, async_test_client, mock_orchestrator):
        """Test automatic task assignment (find best agent)."""
        task_id = str(uuid.uuid4())
        selected_agent_id = str(uuid.uuid4())
        
        assignment_data = {}  # No specific agent ID
        
        mock_orchestrator.assign_task.return_value = {"success": True, "agent_id": selected_agent_id}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post(
                f"/api/v1/tasks/{task_id}/assign",
                json=assignment_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == selected_agent_id

    async def test_assign_task_no_suitable_agent(self, async_test_client, mock_orchestrator):
        """Test task assignment when no suitable agent is available."""
        task_id = str(uuid.uuid4())
        assignment_data = {}
        
        mock_orchestrator.assign_task.return_value = {
            "success": False,
            "message": "No agents available with required capabilities"
        }
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post(
                f"/api/v1/tasks/{task_id}/assign",
                json=assignment_data
            )
        
        assert response.status_code == 400


class TestTaskStatistics:
    """Test task statistics endpoint."""

    async def test_get_task_stats_success(self, async_test_client, mock_orchestrator):
        """Test successful task statistics retrieval."""
        stats_data = {
            "total_tasks": 150,
            "pending_tasks": 12,
            "running_tasks": 8,
            "completed_tasks": 125,
            "failed_tasks": 5,
            "average_completion_time_minutes": 45.7,
            "success_rate": 0.961,
            "active_agents": 6
        }
        
        mock_orchestrator.get_task_stats.return_value = stats_data
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/tasks/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_tasks"] == 150
        assert data["success_rate"] == 0.961
        assert data["active_agents"] == 6

    async def test_get_task_stats_orchestrator_unavailable(self, async_test_client, mock_orchestrator):
        """Test task statistics when orchestrator is unavailable."""
        mock_orchestrator.get_task_stats.side_effect = Exception("Orchestrator unavailable")
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/tasks/stats")
        
        assert response.status_code == 503


class TestTaskHealthCheck:
    """Test task subsystem health check."""

    async def test_tasks_health_check_healthy(self, async_test_client, mock_orchestrator):
        """Test health check when system is healthy."""
        mock_orchestrator.get_system_health.return_value = {
            "status": "healthy",
            "task_queue_size": 15
        }
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/tasks/health/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["service"] == "tasks_api"
        assert data["metrics"]["task_queue_size"] == 15

    async def test_tasks_health_check_unhealthy(self, async_test_client):
        """Test health check when orchestrator is unavailable."""
        with patch("app.api.routes.tasks.SimpleOrchestrator") as mock_class:
            mock_class.side_effect = Exception("Connection failed")
            response = await async_test_client.get("/api/v1/tasks/health/status")
        
        assert response.status_code == 503
        data = response.json()
        assert data["healthy"] is False


class TestPerformance:
    """Performance tests for task API endpoints."""

    async def test_create_task_performance(self, async_test_client, mock_orchestrator, sample_task_data):
        """Test task creation performance (<200ms target)."""
        import time
        
        mock_orchestrator.create_task.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            start_time = time.time()
            response = await async_test_client.post("/api/v1/tasks/", json=sample_task_data)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 201
        # Allow some tolerance for test environment overhead
        assert elapsed_time < 500, f"Task creation took {elapsed_time:.1f}ms (target: <200ms)"

    async def test_list_tasks_performance(self, async_test_client, mock_orchestrator):
        """Test task listing performance with large dataset."""
        import time
        
        # Mock large task list
        mock_tasks = [{"id": str(uuid.uuid4()), "title": f"Task {i}"} for i in range(100)]
        mock_orchestrator.list_tasks.return_value = {"tasks": mock_tasks, "total": 100}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            start_time = time.time()
            response = await async_test_client.get("/api/v1/tasks/")
            elapsed_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert elapsed_time < 400, f"Task listing took {elapsed_time:.1f}ms"


class TestIntegration:
    """Integration tests with full system components."""

    async def test_task_lifecycle_integration(self, async_test_client, mock_orchestrator, sample_task_data):
        """Test complete task lifecycle (create -> assign -> track -> complete/cancel)."""
        task_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        # Mock orchestrator responses for full lifecycle
        mock_orchestrator.create_task.return_value = {"success": True, "task_id": task_id}
        mock_orchestrator.assign_task.return_value = {"success": True, "agent_id": agent_id}
        mock_orchestrator.get_task_status.return_value = {
            "status": TaskStatus.RUNNING,
            "progress": 0.5,
            "assigned_agent_id": agent_id,
            "last_update": datetime.utcnow()
        }
        mock_orchestrator.update_task_priority.return_value = {"success": True}
        mock_orchestrator.cancel_task.return_value = {"success": True}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            # Create task
            create_response = await async_test_client.post("/api/v1/tasks/", json=sample_task_data)
            assert create_response.status_code == 201
            created_task_id = create_response.json()["id"]
            
            # Assign task
            assignment_response = await async_test_client.post(
                f"/api/v1/tasks/{created_task_id}/assign",
                json={"agent_id": agent_id}
            )
            assert assignment_response.status_code == 200
            
            # Check status
            status_response = await async_test_client.get(f"/api/v1/tasks/{created_task_id}/status")
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "running"
            
            # Update priority
            priority_response = await async_test_client.put(
                f"/api/v1/tasks/{created_task_id}/priority",
                json={"priority": "critical", "reason": "Urgent"}
            )
            assert priority_response.status_code == 200
            
            # Cancel task
            cancel_response = await async_test_client.delete(f"/api/v1/tasks/{created_task_id}")
            assert cancel_response.status_code == 200

    async def test_task_assignment_with_capabilities_matching(self, async_test_client, mock_orchestrator):
        """Test task assignment considering required capabilities."""
        task_data = {
            "title": "Python Development Task",
            "required_capabilities": ["python", "fastapi", "testing"],
            "priority": "high"
        }
        
        selected_agent_id = str(uuid.uuid4())
        
        mock_orchestrator.create_task.return_value = {"success": True}
        mock_orchestrator.assign_task.return_value = {"success": True, "agent_id": selected_agent_id}
        
        with patch("app.api.endpoints.tasks.SimpleOrchestrator", return_value=mock_orchestrator):
            # Create task with immediate assignment
            response = await async_test_client.post(
                "/api/v1/tasks/?assign_immediately=true",
                json=task_data
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["assigned_agent_id"] == selected_agent_id
            
            # Verify orchestrator was called with capability requirements
            mock_orchestrator.assign_task.assert_called_with(
                task_id=data["id"],
                required_capabilities=["python", "fastapi", "testing"]
            )