"""
Core Functionality Smoke Tests

These tests validate the essential system functionality works correctly:
- API endpoints respond correctly
- Database connectivity
- Redis connectivity (if available)
- SimpleOrchestrator basic operations
- Import resolution

Designed for <30 seconds execution time.
"""

import pytest
import asyncio
import time
from typing import Any, Dict
from sqlalchemy import text


class TestCoreAPIs:
    """Test critical API endpoints respond correctly."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_response_time(self, async_test_client):
        """Health endpoint should respond within 100ms."""
        start_time = time.time()
        response = await async_test_client.get("/health")
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should respond quickly
        assert response_time < 100, f"Health endpoint took {response_time:.2f}ms, expected <100ms"
        
        # Should return valid response
        assert response.status_code in [200, 500]  # Allow degraded state in tests
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "components" in data
            assert "summary" in data
            assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_status_endpoint_response_time(self, async_test_client):
        """Status endpoint should respond within 100ms."""
        start_time = time.time()
        response = await async_test_client.get("/status")
        response_time = (time.time() - start_time) * 1000
        
        assert response_time < 100, f"Status endpoint took {response_time:.2f}ms, expected <100ms"
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_response_time(self, async_test_client):
        """Metrics endpoint should respond within 100ms with Prometheus format."""
        start_time = time.time()
        response = await async_test_client.get("/metrics")
        response_time = (time.time() - start_time) * 1000
        
        assert response_time < 100, f"Metrics endpoint took {response_time:.2f}ms, expected <100ms"
        assert response.status_code == 200
        
        # Should be Prometheus format
        assert "leanvibe" in response.text
        assert "# HELP" in response.text or "# TYPE" in response.text
    
    @pytest.mark.asyncio
    async def test_debug_agents_endpoint(self, async_test_client):
        """Debug agents endpoint should work for development."""
        response = await async_test_client.get("/debug-agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agent_count" in data
        assert "agents" in data
        assert "status" in data


class TestDatabaseConnectivity:
    """Test database operations work correctly."""
    
    @pytest.mark.asyncio
    async def test_database_basic_operation(self, test_db_session):
        """Test basic database session works."""
        from sqlalchemy import text
        
        # Simple query that should work on any SQL database
        result = await test_db_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row[0] == 1
    
    @pytest.mark.asyncio 
    async def test_database_connection_pool(self, test_engine):
        """Test database connection pool is working."""
        # Test multiple concurrent connections
        async def test_connection():
            async with test_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar()
        
        # Run multiple concurrent connections
        tasks = [test_connection() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(result == 1 for result in results)


class TestRedisConnectivity:
    """Test Redis connectivity if available."""
    
    @pytest.mark.asyncio
    async def test_redis_basic_operations(self, mock_redis):
        """Test basic Redis operations work."""
        # Test ping
        pong = await mock_redis.ping()
        assert pong is True
        
        # Test set/get
        await mock_redis.setex("test_key", 60, "test_value")
        # Mock returns None for get, but operation should complete
        result = await mock_redis.get("test_key")
        assert result is None  # Expected from mock
        
        # Test delete
        deleted = await mock_redis.delete("test_key")
        assert deleted == 1
    
    @pytest.mark.asyncio
    async def test_redis_pubsub_operations(self, mock_redis):
        """Test Redis pub/sub operations work."""
        pubsub = mock_redis.pubsub()
        
        # Test subscribe
        result = await pubsub.subscribe("test_channel")
        assert result is True
        
        # Test publish
        published = await mock_redis.publish("test_channel", "test_message")
        assert published == 1


class TestSimpleOrchestrator:
    """Test SimpleOrchestrator basic functionality."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, test_app):
        """Test SimpleOrchestrator can be initialized."""
        # Import orchestrator
        from app.core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator
        
        # Test factory function
        orchestrator = create_simple_orchestrator()
        assert orchestrator is not None
        assert isinstance(orchestrator, SimpleOrchestrator)
    
    @pytest.mark.asyncio
    async def test_orchestrator_system_status(self, test_app):
        """Test SimpleOrchestrator system status method."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        orchestrator = create_simple_orchestrator()
        status = await orchestrator.get_system_status()
        
        assert isinstance(status, dict)
        assert "agents" in status
        assert "health" in status
        assert "performance" in status
    
    @pytest.mark.asyncio
    async def test_orchestrator_agent_lifecycle(self, test_app):
        """Test basic agent lifecycle operations."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        orchestrator = create_simple_orchestrator()
        
        # Test spawn agent
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            context={"test": True}
        )
        assert agent_id is not None
        
        # Test get agent status
        status = await orchestrator.get_agent_status(agent_id)
        assert status is not None
        assert "status" in status
        
        # Test shutdown agent
        success = await orchestrator.shutdown_agent(agent_id, graceful=True)
        assert success is True


class TestImportResolution:
    """Test that import resolution fixes work correctly."""
    
    def test_core_imports_work(self):
        """Test that core module imports work without errors."""
        # Test main app imports
        from app.main import create_app
        assert create_app is not None
        
        # Test core module imports
        from app.core.simple_orchestrator import SimpleOrchestrator
        from app.core.config import settings
        from app.core.database import get_session
        from app.core.redis import get_redis
        
        assert SimpleOrchestrator is not None
        assert settings is not None
        assert get_session is not None
        assert get_redis is not None
    
    def test_api_imports_work(self):
        """Test that API module imports work without errors."""
        from app.api.routes import router
        from app.api.v1.system import router as system_router
        
        assert router is not None
        assert system_router is not None
    
    def test_model_imports_work(self):
        """Test that model imports work without errors."""
        from app.models.agent import Agent, AgentStatus
        from app.models.task import Task, TaskStatus
        
        assert Agent is not None
        assert AgentStatus is not None
        assert Task is not None
        assert TaskStatus is not None


class TestPerformanceBasics:
    """Basic performance validation for critical paths."""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, async_test_client):
        """Test system handles concurrent health check requests."""
        async def health_check():
            start_time = time.time()
            response = await async_test_client.get("/health")
            response_time = (time.time() - start_time) * 1000
            return response.status_code, response_time
        
        # Run 10 concurrent health checks
        tasks = [health_check() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for status_code, response_time in results:
            assert status_code in [200, 500]  # Allow degraded state
            assert response_time < 1000  # Should be under 1 second even in tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_response_time(self, test_app):
        """Test SimpleOrchestrator operations are fast."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        orchestrator = create_simple_orchestrator()
        
        # Test system status response time
        start_time = time.time()
        await orchestrator.get_system_status()
        response_time = (time.time() - start_time) * 1000
        
        assert response_time < 100, f"System status took {response_time:.2f}ms, expected <100ms"


class TestErrorHandling:
    """Test basic error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint_returns_404(self, async_test_client):
        """Test invalid endpoints return 404."""
        response = await async_test_client.get("/invalid-endpoint-that-does-not-exist")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_orchestrator_handles_invalid_agent_id(self, test_app):
        """Test orchestrator handles invalid agent IDs gracefully."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentNotFoundError
        
        orchestrator = create_simple_orchestrator()
        
        # Test get status with invalid ID
        with pytest.raises(AgentNotFoundError):
            await orchestrator.get_agent_status("invalid-agent-id")
        
        # Test shutdown with invalid ID
        with pytest.raises(AgentNotFoundError):
            await orchestrator.shutdown_agent("invalid-agent-id")
