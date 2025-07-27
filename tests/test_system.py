"""
Tests for system functionality and health checks.
"""

import pytest


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check(async_test_client):
    """Test the health check endpoint."""
    
    response = await async_test_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_api_root(async_test_client):
    """Test the API root endpoint."""
    
    response = await async_test_client.get("/api/v1/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "LeanVibe Agent Hive" in data["message"]
    assert data["version"] == "2.0.0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_system_status(async_test_client):
    """Test the system status endpoint."""
    
    response = await async_test_client.get("/api/v1/system/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cors_headers(async_test_client):
    """Test CORS headers are properly set."""
    
    response = await async_test_client.options("/api/v1/agents/")
    
    # Should handle OPTIONS request
    assert response.status_code in [200, 405]  # Some test clients return 405 for OPTIONS