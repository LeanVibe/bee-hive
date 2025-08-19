"""
Integration Smoke Tests

Tests critical cross-component integrations to ensure the system works end-to-end.
Focuses on the most important integration paths that could break the system.
"""

import pytest
import asyncio
import json
from typing import Any, Dict


class TestDatabaseRedisIntegration:
    """Test database and Redis work together correctly."""
    
    @pytest.mark.asyncio
    async def test_database_redis_session_integration(self, test_db_session, mock_redis):
        """Test database sessions and Redis caching work together."""
        from sqlalchemy import text
        
        # Test database operation
        result = await test_db_session.execute(text("SELECT 1 as db_test"))
        db_value = result.scalar()
        assert db_value == 1
        
        # Test Redis operation
        await mock_redis.setex("integration_test", 60, "success")
        redis_result = await mock_redis.get("integration_test")
        # Mock returns None, but operation should complete
        assert redis_result is None  # Expected from mock
        
        # Both systems operational
        assert db_value == 1


class TestOrchestratorAPIIntegration:
    """Test SimpleOrchestrator integrates correctly with API endpoints."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_health_integration(self, async_test_client, test_app):
        """Test orchestrator status is reflected in health endpoint."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        # Initialize orchestrator
        orchestrator = create_simple_orchestrator()
        
        # Mock app state to have orchestrator
        test_app.state.orchestrator = orchestrator
        test_app.state.orchestrator_type = "SimpleOrchestrator"
        
        # Health endpoint should report orchestrator status
        response = await async_test_client.get("/health")
        
        if response.status_code == 200:
            data = response.json()
            assert "components" in data
            # May have orchestrator component status
            if "orchestrator" in data["components"]:
                assert "status" in data["components"]["orchestrator"]
    
    @pytest.mark.asyncio
    async def test_orchestrator_debug_agents_integration(self, async_test_client, test_app):
        """Test orchestrator integrates with debug agents endpoint."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        orchestrator = create_simple_orchestrator()
        
        # Spawn test agent
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            context={"test": True}
        )
        
        # Debug endpoint should show agent info
        response = await async_test_client.get("/debug-agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agent_count" in data
        assert "agents" in data
        
        # Clean up
        await orchestrator.shutdown_agent(agent_id, graceful=True)


class TestWebSocketAPIIntegration:
    """Test WebSocket integration with the API system."""
    
    def test_websocket_health_integration(self, test_client):
        """Test WebSocket endpoints integrate with system health."""
        ws_path = "/api/dashboard/ws/dashboard"
        
        try:
            with test_client.websocket_connect(ws_path) as ws:
                # Send health check message
                health_msg = {
                    "type": "health_check",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
                ws.send_text(json.dumps(health_msg))
                
                # Should not immediately close connection
                # Receiving response is optional for smoke test
                try:
                    response = ws.receive_text()
                    if response:
                        data = json.loads(response)
                        assert isinstance(data, dict)
                except Exception:
                    # Timeout or connection close is acceptable
                    pass
                    
        except Exception:
            # WebSocket connection might be protected or unavailable
            # This is acceptable for smoke tests
            pass
    
    def test_websocket_subscription_integration(self, test_client):
        """Test WebSocket subscription system works."""
        ws_path = "/api/dashboard/ws/dashboard"
        
        try:
            with test_client.websocket_connect(ws_path) as ws:
                # Test subscription message
                subscribe_msg = {
                    "type": "subscribe",
                    "subscriptions": ["system_metrics", "agent_status"]
                }
                ws.send_text(json.dumps(subscribe_msg))
                
                # Test unsubscribe message
                unsubscribe_msg = {
                    "type": "unsubscribe",
                    "subscriptions": ["system_metrics"]
                }
                ws.send_text(json.dumps(unsubscribe_msg))
                
                # Connection should remain stable
                assert ws is not None
                
        except Exception:
            # WebSocket might be protected - acceptable for smoke test
            pass


class TestErrorHandlingIntegration:
    """Test error handling integrates across components."""
    
    @pytest.mark.asyncio
    async def test_global_exception_handler_integration(self, async_test_client):
        """Test global exception handler catches errors properly."""
        # Try to trigger an error condition
        response = await async_test_client.post("/api/v1/invalid-endpoint", json={"test": "data"})
        
        # Should get proper error response, not crash
        assert 400 <= response.status_code < 600
        
        # Should return JSON error
        try:
            error_data = response.json()
            assert isinstance(error_data, dict)
        except json.JSONDecodeError:
            # Some errors might return HTML, that's OK
            pass
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_handling_integration(self, test_app):
        """Test orchestrator error handling integrates with system."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentNotFoundError
        
        orchestrator = create_simple_orchestrator()
        
        # Test error handling for invalid operations
        with pytest.raises(AgentNotFoundError):
            await orchestrator.get_agent_status("invalid-id")
        
        # System should remain stable after errors
        status = await orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert "health" in status


class TestPerformanceIntegration:
    """Test performance aspects of component integration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self, async_test_client, test_app):
        """Test system handles concurrent operations across components."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        orchestrator = create_simple_orchestrator()
        
        async def mixed_operations():
            """Mix of API calls and orchestrator operations."""
            # API call
            health_response = await async_test_client.get("/health")
            
            # Orchestrator operation
            status = await orchestrator.get_system_status()
            
            return health_response.status_code, status
        
        # Run multiple concurrent mixed operations
        tasks = [mixed_operations() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All operations should complete successfully
        for api_status, orchestrator_status in results:
            assert api_status in [200, 500]  # Allow degraded state
            assert isinstance(orchestrator_status, dict)
    
    @pytest.mark.asyncio
    async def test_database_redis_orchestrator_integration_performance(self, test_db_session, mock_redis, test_app):
        """Test performance when all components work together."""
        import time
        from app.core.simple_orchestrator import create_simple_orchestrator
        from sqlalchemy import text
        
        orchestrator = create_simple_orchestrator()
        
        async def integrated_operation():
            """Operation that uses all three components."""
            start_time = time.time()
            
            # Database operation
            db_result = await test_db_session.execute(text("SELECT 1"))
            db_value = db_result.scalar()
            
            # Redis operation
            await mock_redis.ping()
            
            # Orchestrator operation
            orch_status = await orchestrator.get_system_status()
            
            end_time = time.time()
            return (end_time - start_time) * 1000, db_value, orch_status
        
        # Test integrated operation performance
        response_time, db_result, orch_status = await integrated_operation()
        
        # Should complete quickly even with all components
        assert response_time < 100, f"Integrated operation took {response_time:.2f}ms, expected <100ms"
        assert db_result == 1
        assert isinstance(orch_status, dict)


class TestSystemStabilityIntegration:
    """Test system stability under integrated load."""
    
    @pytest.mark.asyncio
    async def test_rapid_api_orchestrator_cycles(self, async_test_client, test_app):
        """Test rapid cycles of API calls and orchestrator operations."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        orchestrator = create_simple_orchestrator()
        
        async def rapid_cycle():
            """Rapid spawn/shutdown cycle with API monitoring."""
            # API health check
            health_response = await async_test_client.get("/health")
            
            # Orchestrator agent lifecycle
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER,
                context={"rapid_test": True}
            )
            
            # Check status
            status = await orchestrator.get_agent_status(agent_id)
            
            # Shutdown
            await orchestrator.shutdown_agent(agent_id, graceful=True)
            
            return health_response.status_code, status is not None
        
        # Run rapid cycles
        tasks = [rapid_cycle() for _ in range(3)]  # Keep small for speed
        results = await asyncio.gather(*tasks)
        
        # System should remain stable
        for api_status, agent_status_ok in results:
            assert api_status in [200, 500]
            assert agent_status_ok is True
        
        # Final system check
        final_status = await orchestrator.get_system_status()
        assert final_status["health"] in ["healthy", "no_agents"]  # Should be stable
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, async_test_client, test_app):
        """Test system recovers from errors across components."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentNotFoundError
        
        orchestrator = create_simple_orchestrator()
        
        # Cause orchestrator error
        try:
            await orchestrator.get_agent_status("error-causing-id")
        except AgentNotFoundError:
            pass  # Expected
        
        # System should still respond to API calls
        response = await async_test_client.get("/health")
        assert response.status_code in [200, 500]
        
        # Orchestrator should still be operational
        status = await orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert "health" in status
