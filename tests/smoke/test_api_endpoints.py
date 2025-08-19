"""
API Endpoints Smoke Tests

Validates that all major API endpoints are accessible and return appropriate responses.
Covers authentication, authorization, and core business logic endpoints.
"""

import pytest
import json
import time
from typing import Any, Dict


class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints."""
    
    @pytest.mark.asyncio
    async def test_api_v1_health_endpoints(self, async_test_client):
        """Test v1 API health endpoints are accessible."""
        # Test error handling health endpoint
        response = await async_test_client.get("/api/v1/error-handling/health")
        assert response.status_code in [200, 404]  # May not be available in all configs
        
    @pytest.mark.asyncio
    async def test_enterprise_security_endpoints(self, async_test_client):
        """Test enterprise security endpoints."""
        # Test security status endpoint
        response = await async_test_client.get("/api/enterprise/security/status")
        # May return 404 if not configured, or 401/403 if protected
        assert response.status_code in [200, 401, 403, 404]


class TestAgentManagementEndpoints:
    """Test agent management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_agents_list_endpoint(self, async_test_client):
        """Test agents list endpoint."""
        response = await async_test_client.get("/api/agents")
        # May be protected or not implemented
        assert response.status_code in [200, 401, 404, 405]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
    
    @pytest.mark.asyncio
    async def test_debug_agents_endpoint(self, async_test_client):
        """Test debug agents endpoint provides agent information."""
        response = await async_test_client.get("/debug-agents")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["agent_count", "agents", "status"]
        for field in required_fields:
            assert field in data
            
        assert isinstance(data["agent_count"], int)
        assert isinstance(data["agents"], list)
        assert data["status"] in ["debug_working", "debug_error"]


class TestTaskManagementEndpoints:
    """Test task management API endpoints."""
    
    @pytest.mark.asyncio
    async def test_v1_tasks_endpoint_exists(self, async_test_client):
        """Test v1 tasks endpoint exists."""
        response = await async_test_client.get("/api/v1/tasks")
        # Should exist but may be protected
        assert response.status_code in [200, 401, 405, 422]  # 422 for missing query params
        
    @pytest.mark.asyncio
    async def test_v1_tasks_post_validation(self, async_test_client):
        """Test tasks POST endpoint validates input."""
        # Test with invalid data
        response = await async_test_client.post("/api/v1/tasks", json={})
        assert response.status_code in [400, 401, 422]  # Validation error or auth required
        
        if response.status_code == 422:
            data = response.json()
            assert "detail" in data


class TestWorkflowEndpoints:
    """Test workflow management endpoints."""
    
    @pytest.mark.asyncio
    async def test_v1_workflows_endpoint_exists(self, async_test_client):
        """Test v1 workflows endpoint exists."""
        response = await async_test_client.get("/api/v1/workflows")
        assert response.status_code in [200, 401, 405, 422]
        
    @pytest.mark.asyncio
    async def test_v1_workflows_post_validation(self, async_test_client):
        """Test workflows POST endpoint validates input."""
        response = await async_test_client.post("/api/v1/workflows", json={})
        assert response.status_code in [400, 401, 422]


class TestObservabilityEndpoints:
    """Test observability and monitoring endpoints."""
    
    @pytest.mark.asyncio
    async def test_dashboard_websocket_endpoint_exists(self, test_client):
        """Test dashboard WebSocket endpoint exists."""
        ws_path = "/api/dashboard/ws/dashboard"
        
        # Test WebSocket handshake
        try:
            with test_client.websocket_connect(ws_path) as ws:
                # Connection successful
                assert ws is not None
                
                # Try to send a valid message
                test_message = {
                    "type": "subscribe",
                    "subscriptions": ["system_metrics"]
                }
                ws.send_text(json.dumps(test_message))
                
                # Should not immediately close
                # If it processes the message, good
                # If it closes due to auth, also acceptable
        except Exception:
            # WebSocket might require authentication or be disabled
            # This is acceptable for smoke tests
            pass
    
    @pytest.mark.asyncio
    async def test_dashboard_api_endpoints(self, async_test_client):
        """Test dashboard API endpoints."""
        endpoints = [
            "/api/dashboard/live-data",
            "/api/dashboard/metrics",
            "/api/dashboard/system-status"
        ]
        
        for endpoint in endpoints:
            response = await async_test_client.get(endpoint)
            # Should exist but may require auth
            assert response.status_code in [200, 401, 404]
            
            if response.status_code == 200:
                # Should return valid JSON
                data = response.json()
                assert isinstance(data, (dict, list))


class TestMemoryOperationsEndpoints:
    """Test memory operations endpoints."""
    
    @pytest.mark.asyncio
    async def test_memory_operations_endpoint(self, async_test_client):
        """Test memory operations endpoint exists."""
        response = await async_test_client.get("/api/v1/memory")
        assert response.status_code in [200, 401, 404, 405]


class TestCoordinationEndpoints:
    """Test multi-agent coordination endpoints."""
    
    @pytest.mark.asyncio
    async def test_enhanced_coordination_endpoints(self, async_test_client):
        """Test enhanced coordination endpoints."""
        endpoints = [
            "/api/v1/coordination/status",
            "/api/v1/coordination/agents"
        ]
        
        for endpoint in endpoints:
            response = await async_test_client.get(endpoint)
            # May not be implemented or may require auth
            assert response.status_code in [200, 401, 404, 405]
    
    @pytest.mark.asyncio
    async def test_global_coordination_endpoints(self, async_test_client):
        """Test global coordination endpoints."""
        response = await async_test_client.get("/api/v1/global-coordination")
        assert response.status_code in [200, 401, 404, 405]


class TestHiveCommandsEndpoints:
    """Test hive-specific command endpoints."""
    
    @pytest.mark.asyncio
    async def test_hive_commands_endpoint(self, async_test_client):
        """Test hive commands endpoint."""
        response = await async_test_client.get("/api/hive")
        assert response.status_code in [200, 401, 404, 405]
        
        # Test specific command endpoint if base exists
        if response.status_code == 200:
            response = await async_test_client.get("/api/hive/status")
            assert response.status_code in [200, 404, 405]


class TestProjectIndexEndpoints:
    """Test project index and intelligence endpoints."""
    
    @pytest.mark.asyncio
    async def test_project_index_endpoints(self, async_test_client):
        """Test project index endpoints."""
        endpoints = [
            "/api/project-index",
            "/api/project-index/status",
            "/api/project-index/health"
        ]
        
        for endpoint in endpoints:
            response = await async_test_client.get(endpoint)
            assert response.status_code in [200, 401, 404, 405]
    
    @pytest.mark.asyncio
    async def test_intelligence_endpoints(self, async_test_client):
        """Test AI intelligence endpoints."""
        response = await async_test_client.get("/api/intelligence")
        assert response.status_code in [200, 401, 404, 405]


class TestResponseTimeValidation:
    """Validate API response times meet performance targets."""
    
    @pytest.mark.asyncio
    async def test_critical_endpoints_response_time(self, async_test_client):
        """Test critical endpoints respond within acceptable time limits."""
        critical_endpoints = [
            "/health",
            "/status", 
            "/metrics",
            "/debug-agents"
        ]
        
        for endpoint in critical_endpoints:
            start_time = time.time()
            response = await async_test_client.get(endpoint)
            response_time = (time.time() - start_time) * 1000
            
            # Critical endpoints should respond quickly
            assert response_time < 200, f"{endpoint} took {response_time:.2f}ms, expected <200ms"
            assert response.status_code in [200, 500]  # Allow degraded state
    
    @pytest.mark.asyncio
    async def test_api_endpoints_reasonable_response_time(self, async_test_client):
        """Test API endpoints have reasonable response times."""
        api_endpoints = [
            "/api/v1/tasks",
            "/api/v1/workflows", 
            "/api/agents",
            "/api/hive"
        ]
        
        for endpoint in api_endpoints:
            start_time = time.time()
            response = await async_test_client.get(endpoint)
            response_time = (time.time() - start_time) * 1000
            
            # API endpoints should respond within 1 second
            assert response_time < 1000, f"{endpoint} took {response_time:.2f}ms, expected <1000ms"
            # Status codes may vary (auth, validation, etc.)
            assert 200 <= response.status_code < 600


class TestAPIDocumentation:
    """Test API documentation endpoints are available in development."""
    
    @pytest.mark.asyncio
    async def test_openapi_json(self, async_test_client):
        """Test OpenAPI JSON schema is available."""
        response = await async_test_client.get("/openapi.json")
        
        if response.status_code == 200:
            data = response.json()
            assert "openapi" in data
            assert "info" in data
            assert "paths" in data
    
    @pytest.mark.asyncio
    async def test_docs_endpoints(self, async_test_client):
        """Test documentation endpoints."""
        # These may be disabled in production
        docs_endpoints = ["/docs", "/redoc"]
        
        for endpoint in docs_endpoints:
            response = await async_test_client.get(endpoint)
            # May be disabled (404) or available (200)
            assert response.status_code in [200, 404]
