"""
Tests for agent management functionality.
"""

import pytest
import uuid
from datetime import datetime

from app.models.agent import Agent, AgentStatus, AgentType
from app.schemas.agent import AgentCreate, AgentUpdate


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_agent_endpoint(async_test_client, test_db_session):
    """Test creating an agent via API."""
    
    agent_data = {
        "name": "Test API Agent",
        "type": "CLAUDE",
        "role": "test_role",
        "capabilities": [
            {
                "name": "test_capability",
                "description": "Test capability",
                "confidence_level": 0.9,
                "specialization_areas": ["testing"]
            }
        ],
        "system_prompt": "You are a test agent",
        "config": {"test": True}
    }
    
    response = await async_test_client.post("/api/v1/agents/", json=agent_data)
    
    assert response.status_code == 201
    data = response.json()
    
    assert data["name"] == agent_data["name"]
    assert data["type"] == agent_data["type"]
    assert data["role"] == agent_data["role"]
    assert data["status"] == "INACTIVE"  # Default status
    assert len(data["capabilities"]) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_agent_endpoint(async_test_client, sample_agent):
    """Test retrieving an agent via API."""
    
    response = await async_test_client.get(f"/api/v1/agents/{sample_agent.id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["id"] == str(sample_agent.id)
    assert data["name"] == sample_agent.name
    assert data["status"] == sample_agent.status.value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_agents_endpoint(async_test_client, sample_agent):
    """Test listing agents via API."""
    
    response = await async_test_client.get("/api/v1/agents/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["total"] >= 1
    assert len(data["agents"]) >= 1
    assert any(agent["id"] == str(sample_agent.id) for agent in data["agents"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_agent_endpoint(async_test_client, sample_agent):
    """Test updating an agent via API."""
    
    update_data = {
        "name": "Updated Test Agent",
        "status": "BUSY"
    }
    
    response = await async_test_client.put(f"/api/v1/agents/{sample_agent.id}", json=update_data)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["name"] == update_data["name"]
    assert data["status"] == update_data["status"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_agent_endpoint(async_test_client, sample_agent):
    """Test deleting (deactivating) an agent via API."""
    
    response = await async_test_client.delete(f"/api/v1/agents/{sample_agent.id}")
    
    assert response.status_code == 204
    
    # Verify agent is deactivated
    get_response = await async_test_client.get(f"/api/v1/agents/{sample_agent.id}")
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["status"] == "INACTIVE"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_heartbeat_endpoint(async_test_client, sample_agent):
    """Test agent heartbeat update."""
    
    response = await async_test_client.post(f"/api/v1/agents/{sample_agent.id}/heartbeat")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "heartbeat_updated"
    assert "timestamp" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_stats_endpoint(async_test_client, sample_agent):
    """Test agent statistics retrieval."""
    
    response = await async_test_client.get(f"/api/v1/agents/{sample_agent.id}/stats")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["agent_id"] == str(sample_agent.id)
    assert "total_tasks_completed" in data
    assert "success_rate" in data
    assert "uptime_hours" in data


@pytest.mark.unit
def test_agent_model_capabilities():
    """Test agent model capability methods."""
    
    agent = Agent(
        name="Test Agent",
        type=AgentType.CLAUDE,
        capabilities=[
            {
                "name": "python_development",
                "description": "Python development",
                "confidence_level": 0.9,
                "specialization_areas": ["backend", "api"]
            }
        ]
    )
    
    # Test capability checking
    assert agent.has_capability("python_development")
    assert not agent.has_capability("frontend_development")
    
    # Test confidence retrieval
    assert agent.get_capability_confidence("python_development") == 0.9
    assert agent.get_capability_confidence("unknown") == 0.0
    
    # Test adding capability
    agent.add_capability("testing", "Test development", 0.8, ["pytest", "unittest"])
    assert agent.has_capability("testing")


@pytest.mark.unit
def test_agent_task_suitability():
    """Test agent task suitability calculation."""
    
    agent = Agent(
        name="Test Agent",
        type=AgentType.CLAUDE,
        status=AgentStatus.ACTIVE,
        context_window_usage="0.5",
        capabilities=[
            {
                "name": "backend_development",
                "description": "Backend development",
                "confidence_level": 0.9,
                "specialization_areas": ["python", "fastapi"]
            }
        ]
    )
    
    # Test suitability calculation
    required_caps = ["backend_development"]
    suitability = agent.calculate_task_suitability("backend task", required_caps)
    
    assert suitability > 0.0
    assert suitability <= 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_not_found(async_test_client):
    """Test handling of non-existent agent."""
    
    fake_id = uuid.uuid4()
    response = await async_test_client.get(f"/api/v1/agents/{fake_id}")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()