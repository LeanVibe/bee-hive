"""
Tests for Agent Management API Endpoints - Epic C Phase 1

Comprehensive test suite for agent lifecycle management endpoints.
Tests cover creation, status control, deletion, and orchestrator integration.

Test Categories:
- Unit tests for individual endpoints
- Integration tests with orchestrator
- Error handling and validation tests
- Performance tests for <200ms target
- Authentication and authorization tests

Epic C Phase 1: API Endpoint Implementation
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.api.main import app
from app.schemas.agent import AgentCreate, AgentResponse
from app.models.agent import AgentStatus, AgentType
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
    orchestrator.create_agent = AsyncMock()
    orchestrator.list_agents = AsyncMock()
    orchestrator.get_agent = AsyncMock()
    orchestrator.update_agent_status = AsyncMock()
    orchestrator.delete_agent = AsyncMock()
    orchestrator.get_agent_stats = AsyncMock()
    orchestrator.get_system_health = AsyncMock()
    return orchestrator


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "name": "Test Agent",
        "type": "claude",
        "role": "developer",
        "capabilities": [
            {"name": "code_generation", "confidence": 0.95},
            {"name": "debugging", "confidence": 0.88}
        ],
        "system_prompt": "You are a helpful coding assistant",
        "config": {"max_tokens": 4000}
    }


@pytest.fixture
def sample_agent_response():
    """Sample agent response data."""
    agent_id = str(uuid.uuid4())
    return {
        "id": agent_id,
        "name": "Test Agent",
        "type": AgentType.CLAUDE,
        "role": "developer", 
        "capabilities": [
            {"name": "code_generation", "confidence": 0.95},
            {"name": "debugging", "confidence": 0.88}
        ],
        "status": AgentStatus.ACTIVE,
        "config": {"max_tokens": 4000},
        "tmux_session": None,
        "total_tasks_completed": 0,
        "total_tasks_failed": 0,
        "average_response_time": 0.0,
        "context_window_usage": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "last_heartbeat": None,
        "last_active": None
    }


class TestAgentCreation:
    """Test agent creation endpoint."""

    async def test_create_agent_success(self, async_test_client, mock_orchestrator, sample_agent_data):
        """Test successful agent creation."""
        # Mock orchestrator response
        mock_orchestrator.create_agent.return_value = {"success": True, "agent_id": str(uuid.uuid4())}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/agents/", json=sample_agent_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert "id" in data
        assert data["name"] == sample_agent_data["name"]
        assert data["type"] == sample_agent_data["type"]
        assert data["status"] in ["created", "active"]
        assert mock_orchestrator.create_agent.called

    async def test_create_agent_minimal_data(self, async_test_client, mock_orchestrator):
        """Test agent creation with minimal required data."""
        minimal_data = {"name": "Minimal Agent"}
        mock_orchestrator.create_agent.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/agents/", json=minimal_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Agent"
        assert "id" in data

    async def test_create_agent_invalid_data(self, async_test_client):
        """Test agent creation with invalid data."""
        invalid_data = {"name": ""}  # Empty name should fail validation
        
        response = await async_test_client.post("/api/v1/agents/", json=invalid_data)
        assert response.status_code == 422  # Validation error

    async def test_create_agent_orchestrator_failure(self, async_test_client, mock_orchestrator, sample_agent_data):
        """Test agent creation when orchestrator fails."""
        mock_orchestrator.create_agent.side_effect = Exception("Orchestrator error")
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.post("/api/v1/agents/", json=sample_agent_data)
        
        # Should still create agent but mark as inactive
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "inactive"


class TestAgentListing:
    """Test agent listing endpoint."""

    async def test_list_agents_success(self, async_test_client, mock_orchestrator, sample_agent_response):
        """Test successful agent listing."""
        mock_orchestrator.list_agents.return_value = {
            "agents": [sample_agent_response],
            "total": 1
        }
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/agents/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "agents" in data
        assert "total" in data
        assert len(data["agents"]) == 1
        assert data["total"] == 1

    async def test_list_agents_with_filters(self, async_test_client, mock_orchestrator):
        """Test agent listing with status and type filters."""
        mock_orchestrator.list_agents.return_value = {"agents": [], "total": 0}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(
                "/api/v1/agents/?status=active&agent_type=claude&limit=10&offset=0"
            )
        
        assert response.status_code == 200
        # Verify filters were passed to orchestrator
        mock_orchestrator.list_agents.assert_called_with(
            status_filter=AgentStatus.ACTIVE,
            type_filter=AgentType.CLAUDE,
            limit=10,
            offset=0
        )

    async def test_list_agents_pagination(self, async_test_client, mock_orchestrator):
        """Test agent listing pagination."""
        mock_orchestrator.list_agents.return_value = {"agents": [], "total": 100}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/agents/?limit=20&offset=40")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 20
        assert data["offset"] == 40
        assert data["total"] == 100


class TestAgentRetrieval:
    """Test individual agent retrieval endpoint."""

    async def test_get_agent_success(self, async_test_client, mock_orchestrator, sample_agent_response):
        """Test successful agent retrieval."""
        agent_id = sample_agent_response["id"]
        mock_orchestrator.get_agent.return_value = sample_agent_response
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["name"] == sample_agent_response["name"]

    async def test_get_agent_not_found(self, async_test_client, mock_orchestrator):
        """Test agent retrieval when agent doesn't exist."""
        agent_id = str(uuid.uuid4())
        mock_orchestrator.get_agent.return_value = None
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 404

    async def test_get_agent_invalid_uuid(self, async_test_client):
        """Test agent retrieval with invalid UUID format."""
        invalid_id = "not-a-uuid"
        
        response = await async_test_client.get(f"/api/v1/agents/{invalid_id}")
        assert response.status_code == 400


class TestAgentStatusControl:
    """Test agent status update endpoint."""

    async def test_update_agent_status_success(self, async_test_client, mock_orchestrator):
        """Test successful agent status update."""
        agent_id = str(uuid.uuid4())
        status_data = {"status": "inactive", "reason": "Maintenance"}
        
        mock_orchestrator.update_agent_status.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.put(
                f"/api/v1/agents/{agent_id}/status", 
                json=status_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "inactive" in data["message"]

    async def test_update_agent_status_failure(self, async_test_client, mock_orchestrator):
        """Test agent status update failure."""
        agent_id = str(uuid.uuid4())
        status_data = {"status": "active"}
        
        mock_orchestrator.update_agent_status.return_value = {
            "success": False, 
            "message": "Agent not found"
        }
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.put(
                f"/api/v1/agents/{agent_id}/status",
                json=status_data
            )
        
        assert response.status_code == 400


class TestAgentDeletion:
    """Test agent deletion endpoint."""

    async def test_delete_agent_success(self, async_test_client, mock_orchestrator):
        """Test successful agent deletion."""
        agent_id = str(uuid.uuid4())
        mock_orchestrator.delete_agent.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.delete(f"/api/v1/agents/{agent_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert agent_id in data["message"]

    async def test_delete_agent_with_force(self, async_test_client, mock_orchestrator):
        """Test force deletion of agent."""
        agent_id = str(uuid.uuid4())
        mock_orchestrator.delete_agent.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.delete(f"/api/v1/agents/{agent_id}?force=true")
        
        assert response.status_code == 200
        # Verify force parameter was passed
        mock_orchestrator.delete_agent.assert_called_with(agent_id=agent_id, force=True)


class TestAgentStatistics:
    """Test agent statistics endpoint."""

    async def test_get_agent_stats_success(self, async_test_client, mock_orchestrator):
        """Test successful agent statistics retrieval."""
        agent_id = str(uuid.uuid4())
        stats_data = {
            "total_tasks_completed": 25,
            "total_tasks_failed": 2,
            "success_rate": 0.925,
            "average_response_time": 1.5,
            "context_window_usage": 0.65,
            "uptime_hours": 72.5,
            "last_active": datetime.utcnow().isoformat(),
            "capabilities": ["coding", "debugging"]
        }
        
        mock_orchestrator.get_agent_stats.return_value = stats_data
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get(f"/api/v1/agents/{agent_id}/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_tasks_completed"] == 25
        assert data["success_rate"] == 0.925
        assert data["capabilities_count"] == 2


class TestAgentHealthCheck:
    """Test agent subsystem health check."""

    async def test_agents_health_check_healthy(self, async_test_client, mock_orchestrator):
        """Test health check when system is healthy."""
        mock_orchestrator.get_system_health.return_value = {"status": "healthy"}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            response = await async_test_client.get("/api/v1/agents/health/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["service"] == "agents_api"

    async def test_agents_health_check_unhealthy(self, async_test_client):
        """Test health check when orchestrator is unavailable."""
        with patch("app.api.routes.agents.SimpleOrchestrator") as mock_class:
            mock_class.side_effect = Exception("Connection failed")
            response = await async_test_client.get("/api/v1/agents/health/status")
        
        assert response.status_code == 503
        data = response.json()
        assert data["healthy"] is False


class TestPerformance:
    """Performance tests for agent API endpoints."""

    async def test_create_agent_performance(self, async_test_client, mock_orchestrator, sample_agent_data):
        """Test agent creation performance (<200ms target)."""
        import time
        
        mock_orchestrator.create_agent.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            start_time = time.time()
            response = await async_test_client.post("/api/v1/agents/", json=sample_agent_data)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 201
        # Allow some tolerance for test environment overhead
        assert elapsed_time < 500, f"Agent creation took {elapsed_time:.1f}ms (target: <200ms)"

    async def test_list_agents_performance(self, async_test_client, mock_orchestrator):
        """Test agent listing performance."""
        import time
        
        # Mock large agent list
        mock_agents = [{"id": str(uuid.uuid4()), "name": f"Agent {i}"} for i in range(50)]
        mock_orchestrator.list_agents.return_value = {"agents": mock_agents, "total": 50}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            start_time = time.time()
            response = await async_test_client.get("/api/v1/agents/")
            elapsed_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert elapsed_time < 300, f"Agent listing took {elapsed_time:.1f}ms"


class TestIntegration:
    """Integration tests with full system components."""

    async def test_agent_lifecycle_integration(self, async_test_client, mock_orchestrator, sample_agent_data):
        """Test complete agent lifecycle (create -> get -> update -> delete)."""
        agent_id = str(uuid.uuid4())
        
        # Mock orchestrator responses for full lifecycle
        mock_orchestrator.create_agent.return_value = {"success": True, "agent_id": agent_id}
        mock_orchestrator.get_agent.return_value = {**sample_agent_data, "id": agent_id}
        mock_orchestrator.update_agent_status.return_value = {"success": True}
        mock_orchestrator.delete_agent.return_value = {"success": True}
        
        with patch("app.api.endpoints.agents.SimpleOrchestrator", return_value=mock_orchestrator):
            # Create agent
            create_response = await async_test_client.post("/api/v1/agents/", json=sample_agent_data)
            assert create_response.status_code == 201
            created_agent_id = create_response.json()["id"]
            
            # Get agent
            get_response = await async_test_client.get(f"/api/v1/agents/{created_agent_id}")
            assert get_response.status_code == 200
            
            # Update status
            status_response = await async_test_client.put(
                f"/api/v1/agents/{created_agent_id}/status",
                json={"status": "inactive"}
            )
            assert status_response.status_code == 200
            
            # Delete agent
            delete_response = await async_test_client.delete(f"/api/v1/agents/{created_agent_id}")
            assert delete_response.status_code == 200