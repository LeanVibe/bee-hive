"""
Frontend API Integration Testing
===============================

Comprehensive API integration tests with real HTTP requests for the frontend API server.
Tests complete HTTP request/response cycles, validates JSON schema compliance,
and ensures all endpoints meet performance and reliability requirements.

Key Testing Areas:
- System health and status endpoints
- Agent management CRUD operations
- Task management workflow
- WebSocket real-time communication
- Error handling and HTTP status codes
- Performance validation (<500ms target)
- CORS configuration validation
"""

import pytest
import json
import asyncio
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import httpx
import jsonschema
from jsonschema import validate, ValidationError
import uvicorn
import threading
from contextlib import asynccontextmanager

# Import the frontend API server
from frontend_api_server import app


class TestFrontendAPIIntegration:
    """Integration tests for frontend API endpoints with real HTTP requests."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start the API server for testing."""
        import uvicorn
        import threading
        import time
        
        # Start server in background thread
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8999,  # Use different port for testing
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        yield "http://127.0.0.1:8999"
        
        # Server will be cleaned up automatically due to daemon thread

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for API testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client

    # System Health and Status Endpoints

    async def test_root_endpoint_integration(self, http_client):
        """Test root endpoint returns correct API information."""
        
        response = await http_client.get("/")
        
        # Validate HTTP status
        assert response.status_code == 200
        
        # Validate response structure
        data = response.json()
        
        expected_schema = {
            "type": "object",
            "required": ["message", "version", "endpoints"],
            "properties": {
                "message": {"type": "string"},
                "version": {"type": "string"},
                "endpoints": {
                    "type": "object",
                    "required": ["health", "status", "api_v1", "websocket"],
                    "properties": {
                        "health": {"type": "string"},
                        "status": {"type": "string"},
                        "api_v1": {"type": "string"},
                        "websocket": {"type": "string"}
                    }
                }
            }
        }
        
        jsonschema.validate(data, expected_schema)
        
        # Validate specific values
        assert data["message"] == "LeanVibe Frontend API Server"
        assert data["version"] == "1.0.0"
        assert data["endpoints"]["health"] == "/health"
        assert data["endpoints"]["websocket"] == "/ws/updates"

    async def test_health_check_endpoint_integration(self, http_client):
        """Test health check endpoint returns system health status."""
        
        start_time = time.time()
        response = await http_client.get("/health")
        response_time_ms = (time.time() - start_time) * 1000
        
        # Validate performance (<500ms target)
        assert response_time_ms < 500.0, f"Health check took {response_time_ms}ms, exceeds 500ms target"
        
        # Validate HTTP status
        assert response.status_code == 200
        
        # Validate response schema
        data = response.json()
        
        health_schema = {
            "type": "object",
            "required": ["status", "timestamp", "version"],
            "properties": {
                "status": {"type": "string", "enum": ["healthy"]},
                "timestamp": {"type": "string"},
                "version": {"type": "string"}
            }
        }
        
        jsonschema.validate(data, health_schema)
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    async def test_system_status_endpoint_integration(self, http_client):
        """Test system status endpoint returns detailed status."""
        
        start_time = time.time()
        response = await http_client.get("/status")
        response_time_ms = (time.time() - start_time) * 1000
        
        # Validate performance
        assert response_time_ms < 500.0
        
        # Validate HTTP status
        assert response.status_code == 200
        
        # Validate response schema
        data = response.json()
        
        status_schema = {
            "type": "object",
            "required": ["status", "timestamp", "components", "uptime"],
            "properties": {
                "status": {"type": "string"},
                "timestamp": {"type": "string"},
                "components": {
                    "type": "object",
                    "required": ["api", "websocket", "agents", "tasks"],
                    "properties": {
                        "api": {"type": "string"},
                        "websocket": {"type": "string"},
                        "agents": {"type": "integer", "minimum": 0},
                        "tasks": {"type": "integer", "minimum": 0}
                    }
                },
                "uptime": {"type": "string"}
            }
        }
        
        jsonschema.validate(data, status_schema)

    # API v1 System Endpoints

    async def test_api_v1_system_status_integration(self, http_client):
        """Test API v1 system status endpoint."""
        
        response = await http_client.get("/api/v1/system/status")
        
        assert response.status_code == 200
        
        data = response.json()
        
        api_status_schema = {
            "type": "object",
            "required": ["status", "message", "version", "timestamp", "components"],
            "properties": {
                "status": {"type": "string", "enum": ["healthy"]},
                "message": {"type": "string"},
                "version": {"type": "string"},
                "timestamp": {"type": "string"},
                "components": {
                    "type": "object",
                    "required": ["database", "redis", "orchestrator"],
                    "properties": {
                        "database": {"type": "string", "enum": ["online"]},
                        "redis": {"type": "string", "enum": ["online"]},
                        "orchestrator": {"type": "string", "enum": ["online"]}
                    }
                }
            }
        }
        
        jsonschema.validate(data, api_status_schema)

    # Agent Management API Tests

    async def test_agent_crud_workflow_integration(self, http_client):
        """Test complete agent CRUD workflow."""
        
        # 1. List agents (initially empty)
        response = await http_client.get("/api/v1/agents")
        assert response.status_code == 200
        
        initial_data = response.json()
        initial_count = initial_data["total"]
        
        list_schema = {
            "type": "object",
            "required": ["agents", "total", "offset", "limit"],
            "properties": {
                "agents": {"type": "array"},
                "total": {"type": "integer", "minimum": 0},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1}
            }
        }
        
        jsonschema.validate(initial_data, list_schema)
        
        # 2. Create new agent
        create_payload = {
            "name": "Test Agent",
            "type": "claude",
            "role": "backend_developer",
            "capabilities": ["coding", "testing"]
        }
        
        start_time = time.time()
        response = await http_client.post("/api/v1/agents", json=create_payload)
        create_time_ms = (time.time() - start_time) * 1000
        
        # Validate performance (<500ms target)
        assert create_time_ms < 500.0
        
        assert response.status_code == 200
        
        agent_data = response.json()
        agent_id = agent_data["id"]
        
        # Validate created agent schema
        agent_schema = {
            "type": "object",
            "required": ["id", "name", "type", "status", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "name": {"type": "string"},
                "type": {"type": "string"},
                "status": {"type": "string", "enum": ["active"]},
                "role": {"type": ["string", "null"]},
                "capabilities": {"type": "array"},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"}
            }
        }
        
        jsonschema.validate(agent_data, agent_schema)
        
        assert agent_data["name"] == create_payload["name"]
        assert agent_data["type"] == create_payload["type"]
        assert agent_data["status"] == "active"
        
        # 3. Get specific agent
        response = await http_client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        
        retrieved_agent = response.json()
        jsonschema.validate(retrieved_agent, agent_schema)
        assert retrieved_agent["id"] == agent_id
        
        # 4. Update agent
        update_payload = {
            "status": "inactive",
            "role": "qa_engineer"
        }
        
        response = await http_client.put(f"/api/v1/agents/{agent_id}", json=update_payload)
        assert response.status_code == 200
        
        updated_agent = response.json()
        jsonschema.validate(updated_agent, agent_schema)
        assert updated_agent["status"] == "inactive"
        assert updated_agent["role"] == "qa_engineer"
        
        # 5. List agents (should include new agent)
        response = await http_client.get("/api/v1/agents")
        assert response.status_code == 200
        
        updated_list = response.json()
        assert updated_list["total"] == initial_count + 1
        
        # 6. Delete agent
        response = await http_client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200
        
        delete_response = response.json()
        assert "message" in delete_response
        assert "deleted successfully" in delete_response["message"]
        
        # 7. Verify agent is deleted (404)
        response = await http_client.get(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 404

    async def test_agent_not_found_error_handling(self, http_client):
        """Test agent API error handling for non-existent agents."""
        
        non_existent_id = "non-existent-agent-id"
        
        # Test GET non-existent agent
        response = await http_client.get(f"/api/v1/agents/{non_existent_id}")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"].lower()
        
        # Test UPDATE non-existent agent
        response = await http_client.put(f"/api/v1/agents/{non_existent_id}", json={"status": "active"})
        assert response.status_code == 404
        
        # Test DELETE non-existent agent
        response = await http_client.delete(f"/api/v1/agents/{non_existent_id}")
        assert response.status_code == 404

    # Task Management API Tests

    async def test_task_crud_workflow_integration(self, http_client):
        """Test complete task CRUD workflow."""
        
        # 1. List tasks (initially empty)
        response = await http_client.get("/api/v1/tasks")
        assert response.status_code == 200
        
        initial_data = response.json()
        initial_count = initial_data["total"]
        
        # 2. Create agent first (for task assignment)
        agent_payload = {
            "name": "Task Agent",
            "type": "claude",
            "role": "backend_developer"
        }
        
        response = await http_client.post("/api/v1/agents", json=agent_payload)
        assert response.status_code == 200
        agent_data = response.json()
        agent_id = agent_data["id"]
        
        # 3. Create new task
        task_payload = {
            "title": "Integration Test Task",
            "description": "Test task for integration testing",
            "priority": "high",
            "agent_id": agent_id
        }
        
        start_time = time.time()
        response = await http_client.post("/api/v1/tasks", json=task_payload)
        create_time_ms = (time.time() - start_time) * 1000
        
        # Validate performance
        assert create_time_ms < 500.0
        
        assert response.status_code == 200
        
        task_data = response.json()
        task_id = task_data["id"]
        
        # Validate task schema
        task_schema = {
            "type": "object",
            "required": ["id", "title", "status", "priority", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "title": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "status": {"type": "string", "enum": ["pending"]},
                "priority": {"type": "string"},
                "agent_id": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"}
            }
        }
        
        jsonschema.validate(task_data, task_schema)
        
        assert task_data["title"] == task_payload["title"]
        assert task_data["status"] == "pending"
        assert task_data["agent_id"] == agent_id
        
        # 4. Get specific task
        response = await http_client.get(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        
        retrieved_task = response.json()
        jsonschema.validate(retrieved_task, task_schema)
        
        # 5. Update task
        update_payload = {
            "status": "in_progress",
            "priority": "medium"
        }
        
        response = await http_client.put(f"/api/v1/tasks/{task_id}", json=update_payload)
        assert response.status_code == 200
        
        updated_task = response.json()
        assert updated_task["status"] == "in_progress"
        assert updated_task["priority"] == "medium"
        
        # 6. List tasks (should include new task)
        response = await http_client.get("/api/v1/tasks")
        assert response.status_code == 200
        
        updated_list = response.json()
        assert updated_list["total"] == initial_count + 1
        
        # 7. Delete task
        response = await http_client.delete(f"/api/v1/tasks/{task_id}")
        assert response.status_code == 200
        
        # 8. Clean up: delete agent
        await http_client.delete(f"/api/v1/agents/{agent_id}")

    # Observability Endpoints

    async def test_observability_endpoints_integration(self, http_client):
        """Test observability endpoints for monitoring."""
        
        # Test metrics endpoint
        response = await http_client.get("/observability/metrics")
        assert response.status_code == 200
        
        metrics_data = response.json()
        
        metrics_schema = {
            "type": "object",
            "required": ["agents_total", "tasks_total", "websocket_connections", "timestamp"],
            "properties": {
                "agents_total": {"type": "integer", "minimum": 0},
                "tasks_total": {"type": "integer", "minimum": 0},
                "websocket_connections": {"type": "integer", "minimum": 0},
                "timestamp": {"type": "string"}
            }
        }
        
        jsonschema.validate(metrics_data, metrics_schema)
        
        # Test observability health endpoint
        response = await http_client.get("/observability/health")
        assert response.status_code == 200
        
        health_data = response.json()
        
        obs_health_schema = {
            "type": "object",
            "required": ["status", "components", "timestamp"],
            "properties": {
                "status": {"type": "string", "enum": ["healthy"]},
                "components": {
                    "type": "object",
                    "required": ["api", "websocket", "storage"],
                    "properties": {
                        "api": {"type": "string", "enum": ["online"]},
                        "websocket": {"type": "string", "enum": ["online"]},
                        "storage": {"type": "string", "enum": ["online"]}
                    }
                },
                "timestamp": {"type": "string"}
            }
        }
        
        jsonschema.validate(health_data, obs_health_schema)

    # Development Utilities

    async def test_populate_demo_data_integration(self, http_client):
        """Test demo data population endpoint."""
        
        # Get initial counts
        agents_response = await http_client.get("/api/v1/agents")
        tasks_response = await http_client.get("/api/v1/tasks")
        
        initial_agents = agents_response.json()["total"]
        initial_tasks = tasks_response.json()["total"]
        
        # Populate demo data
        response = await http_client.post("/dev/populate")
        assert response.status_code == 200
        
        populate_data = response.json()
        
        populate_schema = {
            "type": "object",
            "required": ["message", "agents_created", "tasks_created", "agents", "tasks"],
            "properties": {
                "message": {"type": "string"},
                "agents_created": {"type": "integer", "minimum": 1},
                "tasks_created": {"type": "integer", "minimum": 1},
                "agents": {"type": "array", "minItems": 1},
                "tasks": {"type": "array", "minItems": 1}
            }
        }
        
        jsonschema.validate(populate_data, populate_schema)
        
        # Verify agents and tasks were created
        agents_response = await http_client.get("/api/v1/agents")
        tasks_response = await http_client.get("/api/v1/tasks")
        
        final_agents = agents_response.json()["total"]
        final_tasks = tasks_response.json()["total"]
        
        assert final_agents > initial_agents
        assert final_tasks > initial_tasks
        assert final_agents == initial_agents + populate_data["agents_created"]
        assert final_tasks == initial_tasks + populate_data["tasks_created"]

    # Error Handling and Edge Cases

    async def test_invalid_json_payload_handling(self, http_client):
        """Test API error handling for invalid JSON payloads."""
        
        # Test invalid JSON for agent creation
        response = await http_client.post(
            "/api/v1/agents",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
        
        # Test missing required fields
        response = await http_client.post("/api/v1/agents", json={})
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data

    async def test_cors_configuration_integration(self, http_client):
        """Test CORS configuration for frontend integration."""
        
        # Test preflight request
        response = await http_client.options(
            "/api/v1/agents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should not fail due to CORS
        assert response.status_code in [200, 204]
        
        # Test actual request with CORS headers
        response = await http_client.get(
            "/api/v1/agents",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # CORS headers should be present in actual implementation

    # Performance and Load Testing

    async def test_concurrent_requests_performance(self, http_client):
        """Test API performance under concurrent load."""
        
        async def create_agent_request():
            """Single agent creation request."""
            payload = {
                "name": f"Load Test Agent {time.time()}",
                "type": "claude"
            }
            
            start_time = time.time()
            response = await http_client.post("/api/v1/agents", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            return response.status_code, response_time, response.json() if response.status_code == 200 else None
        
        # Execute 10 concurrent requests
        tasks = [create_agent_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all requests succeeded
        success_count = 0
        total_time = 0
        agent_ids = []
        
        for result in results:
            if isinstance(result, tuple):
                status_code, response_time, data = result
                if status_code == 200 and data:
                    success_count += 1
                    total_time += response_time
                    agent_ids.append(data["id"])
        
        # All requests should succeed
        assert success_count == 10
        
        # Average response time should be reasonable
        avg_response_time = total_time / success_count
        assert avg_response_time < 1000.0  # 1 second average under load
        
        # Cleanup: delete created agents
        for agent_id in agent_ids:
            await http_client.delete(f"/api/v1/agents/{agent_id}")

    async def test_api_endpoint_response_times(self, http_client):
        """Test individual endpoint response times meet performance targets."""
        
        endpoint_targets = [
            ("GET", "/", 100),           # Root: 100ms
            ("GET", "/health", 50),      # Health: 50ms
            ("GET", "/status", 100),     # Status: 100ms
            ("GET", "/api/v1/system/status", 100),  # System status: 100ms
            ("GET", "/api/v1/agents", 200),        # List agents: 200ms
            ("GET", "/observability/metrics", 100), # Metrics: 100ms
        ]
        
        for method, endpoint, target_ms in endpoint_targets:
            start_time = time.time()
            
            if method == "GET":
                response = await http_client.get(endpoint)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert response_time_ms < target_ms, f"{endpoint} took {response_time_ms}ms, exceeds {target_ms}ms target"


# WebSocket Integration Tests
class TestWebSocketIntegration:
    """WebSocket integration tests for real-time communication."""

    @pytest.fixture(scope="class")
    def ws_server(self):
        """Start server for WebSocket testing."""
        import uvicorn
        import threading
        import time
        
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8998,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "ws://127.0.0.1:8998"

    async def test_websocket_connection_lifecycle(self, ws_server):
        """Test WebSocket connection establishment and lifecycle."""
        
        try:
            # Test connection establishment
            async with websockets.connect(f"{ws_server}/ws/updates") as websocket:
                
                # Should receive welcome message
                welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                welcome_data = json.loads(welcome_msg)
                
                assert welcome_data["type"] == "connection_established"
                assert "message" in welcome_data
                assert "timestamp" in welcome_data
                
                # Test echo functionality
                test_message = {"test": "ping"}
                await websocket.send(json.dumps(test_message))
                
                echo_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                echo_data = json.loads(echo_msg)
                
                assert echo_data["type"] == "echo"
                assert echo_data["data"] == json.dumps(test_message)
                
        except asyncio.TimeoutError:
            pytest.fail("WebSocket connection or message exchange timed out")
        except Exception as e:
            pytest.fail(f"WebSocket test failed: {e}")

    async def test_websocket_message_format_validation(self, ws_server):
        """Test WebSocket message format compliance."""
        
        message_schema = {
            "type": "object",
            "required": ["type", "timestamp"],
            "properties": {
                "type": {"type": "string"},
                "timestamp": {"type": "string"},
                "data": {},  # Any data type
                "message": {"type": "string"}
            }
        }
        
        try:
            async with websockets.connect(f"{ws_server}/ws/updates") as websocket:
                
                # Collect multiple messages
                messages = []
                
                # Get welcome message
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                messages.append(json.loads(msg))
                
                # Send test message and get echo
                await websocket.send('{"test": "validation"}')
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                messages.append(json.loads(msg))
                
                # Validate all messages follow schema
                for message in messages:
                    jsonschema.validate(message, message_schema)
                
        except Exception as e:
            pytest.fail(f"WebSocket message validation failed: {e}")


# Integration Test Summary
class TestAPIIntegrationSummary:
    """Summary integration test ensuring all API components work together."""
    
    async def test_complete_api_integration_workflow(self):
        """Complete end-to-end API workflow test."""
        
        base_url = "http://127.0.0.1:8999"
        
        # Start server for this test
        import uvicorn
        import threading
        import time
        
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8997,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8997", timeout=30.0) as client:
            
            # 1. Health check
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
            
            # 2. System status
            response = await client.get("/api/v1/system/status")
            assert response.status_code == 200
            
            # 3. Create agent
            agent_payload = {
                "name": "Integration Test Agent",
                "type": "claude",
                "role": "full_stack_developer",
                "capabilities": ["frontend", "backend", "testing"]
            }
            
            response = await client.post("/api/v1/agents", json=agent_payload)
            assert response.status_code == 200
            agent_data = response.json()
            agent_id = agent_data["id"]
            
            # 4. Create task assigned to agent
            task_payload = {
                "title": "Complete Integration Test",
                "description": "End-to-end API integration test task",
                "priority": "high",
                "agent_id": agent_id
            }
            
            response = await client.post("/api/v1/tasks", json=task_payload)
            assert response.status_code == 200
            task_data = response.json()
            task_id = task_data["id"]
            
            # 5. Update task progress
            response = await client.put(f"/api/v1/tasks/{task_id}", json={"status": "in_progress"})
            assert response.status_code == 200
            
            # 6. Check metrics
            response = await client.get("/observability/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert metrics["agents_total"] >= 1
            assert metrics["tasks_total"] >= 1
            
            # 7. Complete task
            response = await client.put(f"/api/v1/tasks/{task_id}", json={"status": "completed"})
            assert response.status_code == 200
            
            # 8. Cleanup
            await client.delete(f"/api/v1/tasks/{task_id}")
            await client.delete(f"/api/v1/agents/{agent_id}")
            
            # 9. Verify cleanup
            response = await client.get(f"/api/v1/agents/{agent_id}")
            assert response.status_code == 404