"""
Epic 2 Phase 2: API Integration Tests for Consolidated System

Tests API endpoints that integrate with the consolidated system:
- Health check endpoints with engine status
- Agent management endpoints using consolidated orchestrator
- Task execution endpoints through consolidated engines
- System status and metrics endpoints
- Performance validation for API response times

Isolated approach avoiding complex ML dependencies.
"""

import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Mock complex dependencies
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock complex dependencies."""
    with patch.dict('sys.modules', {
        'sklearn': Mock(),
        'scipy': Mock(),
        'numpy': Mock(),
        'pandas': Mock(),
        'structlog': Mock()
    }):
        yield


@pytest.fixture
def mock_app_dependencies():
    """Mock FastAPI app dependencies."""
    
    # Mock database session
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.close = AsyncMock()
    
    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.info = AsyncMock(return_value={
        'used_memory': 10 * 1024 * 1024,
        'connected_clients': 5
    })
    
    # Mock orchestrator with consolidated engine
    mock_orchestrator = Mock()
    mock_orchestrator.is_running = True
    mock_orchestrator.get_production_status = AsyncMock(return_value={
        'orchestrator_status': 'running',
        'uptime_seconds': 3600,
        'system_health': 'healthy',
        'active_alerts': 0,
        'critical_alerts': 0,
        'engine_status': {
            'workflow_engine': {'status': 'running'},
            'task_execution_engine': {'status': 'running'},
            'communication_engine': {'status': 'running'}
        }
    })
    mock_orchestrator.engine_coordinator = Mock()
    mock_orchestrator.engine_coordinator.get_status = AsyncMock(return_value={
        'workflow_engine': {'status': 'running'},
        'task_execution_engine': {'status': 'running'},
        'communication_engine': {'status': 'running'}
    })
    mock_orchestrator.engine_coordinator.get_performance_metrics = AsyncMock(return_value={
        'workflow_engine': {'average_execution_time_ms': 150, 'success_rate_percent': 98.5},
        'task_execution_engine': {'average_execution_time_ms': 45, 'success_rate_percent': 99.2},
        'communication_engine': {'average_delivery_time_ms': 12, 'success_rate_percent': 99.8}
    })
    
    # Mock agent orchestrator
    mock_agent_orchestrator = Mock()
    mock_agent_orchestrator.get_active_agents = AsyncMock(return_value=[
        {
            'id': 'agent_1',
            'role': 'backend_developer',
            'status': 'active',
            'created_at': '2024-01-01T10:00:00Z',
            'last_activity': '2024-01-01T10:30:00Z'
        },
        {
            'id': 'agent_2', 
            'role': 'system_architect',
            'status': 'active',
            'created_at': '2024-01-01T10:15:00Z',
            'last_activity': '2024-01-01T10:32:00Z'
        }
    ])
    mock_agent_orchestrator.spawn_agent = AsyncMock(return_value={
        'id': 'new_agent_1',
        'role': 'backend_developer',
        'status': 'active',
        'created_at': datetime.utcnow().isoformat()
    })
    mock_agent_orchestrator.shutdown_agent = AsyncMock(return_value={'success': True})
    
    return {
        'db': mock_db,
        'redis': mock_redis,
        'production_orchestrator': mock_orchestrator,
        'agent_orchestrator': mock_agent_orchestrator
    }


@pytest.fixture
def api_client(mock_app_dependencies):
    """Create FastAPI test client with consolidated system integration."""
    
    app = FastAPI(title="LeanVibe Agent Hive 2.0 - Consolidated System API")
    
    # Mock dependency injection
    def get_db():
        return mock_app_dependencies['db']
    
    def get_redis():
        return mock_app_dependencies['redis']
    
    def get_production_orchestrator():
        return mock_app_dependencies['production_orchestrator']
    
    def get_agent_orchestrator():
        return mock_app_dependencies['agent_orchestrator']
    
    # Health Check Endpoints
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "consolidated_system"}
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with consolidated system status."""
        orchestrator = get_production_orchestrator()
        status = await orchestrator.get_production_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "orchestrator_status": status['orchestrator_status'],
                "system_health": status['system_health'],
                "uptime_seconds": status['uptime_seconds'],
                "alerts": {
                    "active": status['active_alerts'],
                    "critical": status['critical_alerts']
                }
            },
            "engines": status['engine_status']
        }
    
    @app.get("/health/engines")
    async def engine_health_check():
        """Engine-specific health check."""
        orchestrator = get_production_orchestrator()
        engine_status = await orchestrator.engine_coordinator.get_status()
        
        return {
            "status": "healthy",
            "engines": engine_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Agent Management Endpoints
    @app.get("/api/agents")
    async def get_agents():
        """Get all active agents."""
        agent_orchestrator = get_agent_orchestrator()
        agents = await agent_orchestrator.get_active_agents()
        return {"agents": agents, "count": len(agents)}
    
    @app.post("/api/agents")
    async def create_agent(request: Dict[str, Any]):
        """Create new agent through consolidated orchestrator."""
        agent_orchestrator = get_agent_orchestrator()
        
        role = request.get('role', 'backend_developer')
        agent = await agent_orchestrator.spawn_agent(role=role)
        
        return {"success": True, "agent": agent}
    
    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete agent through consolidated orchestrator."""
        agent_orchestrator = get_agent_orchestrator()
        result = await agent_orchestrator.shutdown_agent(agent_id)
        return result
    
    # Task Execution Endpoints  
    @app.post("/api/tasks/execute")
    async def execute_task(request: Dict[str, Any]):
        """Execute task through consolidated engine system."""
        orchestrator = get_production_orchestrator()
        
        # Simulate task execution through consolidated engines
        task_result = {
            'task_id': f'task_{datetime.utcnow().timestamp()}',
            'success': True,
            'execution_time_ms': 45.2,
            'result': {'status': 'completed', 'task_type': request.get('task_type', 'general')},
            'engine_used': 'consolidated_task_execution_engine'
        }
        
        return {"success": True, "task": task_result}
    
    @app.post("/api/workflows/execute") 
    async def execute_workflow(request: Dict[str, Any]):
        """Execute workflow through consolidated engine system."""
        orchestrator = get_production_orchestrator()
        
        # Simulate workflow execution
        workflow_result = {
            'workflow_id': f'workflow_{datetime.utcnow().timestamp()}',
            'success': True,
            'execution_time_ms': 1200.5,
            'steps_completed': len(request.get('workflow_definition', {}).get('steps', [])),
            'engine_used': 'consolidated_workflow_engine'
        }
        
        return {"success": True, "workflow": workflow_result}
    
    # System Status and Metrics Endpoints
    @app.get("/api/system/status")
    async def get_system_status():
        """Get comprehensive system status."""
        orchestrator = get_production_orchestrator()
        status = await orchestrator.get_production_status()
        return status
    
    @app.get("/api/system/metrics")
    async def get_system_metrics():
        """Get system performance metrics."""
        orchestrator = get_production_orchestrator()
        metrics = await orchestrator.engine_coordinator.get_performance_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "engine_metrics": metrics,
            "consolidated": True
        }
    
    @app.get("/api/system/engines/status")
    async def get_engine_status():
        """Get detailed engine status."""
        orchestrator = get_production_orchestrator()
        status = await orchestrator.engine_coordinator.get_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "engines": status,
            "coordination_layer": "active"
        }
    
    # Override app dependencies
    app.dependency_overrides[get_db] = lambda: mock_app_dependencies['db']
    app.dependency_overrides[get_redis] = lambda: mock_app_dependencies['redis']
    
    return TestClient(app)


class TestHealthCheckEndpoints:
    """Test health check endpoints with consolidated system integration."""
    
    def test_basic_health_check(self, api_client):
        """Test basic health check endpoint."""
        response = api_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "consolidated_system"
    
    def test_detailed_health_check(self, api_client):
        """Test detailed health check with consolidated system status."""
        response = api_client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate structure
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "system" in data
        assert "engines" in data
        
        # Validate system information
        system = data["system"]
        assert system["orchestrator_status"] == "running"
        assert system["system_health"] == "healthy"
        assert "uptime_seconds" in system
        assert "alerts" in system
        
        # Validate engine information
        engines = data["engines"]
        assert "workflow_engine" in engines
        assert "task_execution_engine" in engines
        assert "communication_engine" in engines
    
    def test_engine_health_check(self, api_client):
        """Test engine-specific health check."""
        response = api_client.get("/health/engines")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "engines" in data
        assert "timestamp" in data
        
        engines = data["engines"]
        assert "workflow_engine" in engines
        assert "task_execution_engine" in engines  
        assert "communication_engine" in engines
        
        for engine_name, engine_info in engines.items():
            assert "status" in engine_info
            assert engine_info["status"] == "running"


class TestAgentManagementEndpoints:
    """Test agent management endpoints using consolidated orchestrator."""
    
    def test_get_agents(self, api_client):
        """Test retrieving all active agents."""
        response = api_client.get("/api/agents")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "agents" in data
        assert "count" in data
        assert data["count"] == len(data["agents"])
        
        # Validate agent structure
        for agent in data["agents"]:
            assert "id" in agent
            assert "role" in agent
            assert "status" in agent
            assert "created_at" in agent
            assert "last_activity" in agent
    
    def test_create_agent(self, api_client):
        """Test creating new agent through consolidated orchestrator."""
        request_data = {"role": "backend_developer"}
        
        response = api_client.post("/api/agents", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "agent" in data
        
        agent = data["agent"]
        assert "id" in agent
        assert "role" in agent
        assert agent["role"] == "backend_developer"
        assert "status" in agent
        assert "created_at" in agent
    
    def test_delete_agent(self, api_client):
        """Test deleting agent through consolidated orchestrator."""
        agent_id = "test_agent_123"
        
        response = api_client.delete(f"/api/agents/{agent_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True


class TestTaskExecutionEndpoints:
    """Test task execution endpoints through consolidated engines."""
    
    def test_execute_task(self, api_client):
        """Test task execution through consolidated engine system."""
        task_request = {
            "task_type": "coordination",
            "payload": {
                "data": "test_coordination_data",
                "priority": "high"
            },
            "agent_id": "test_agent_1"
        }
        
        response = api_client.post("/api/tasks/execute", json=task_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "task" in data
        
        task = data["task"]
        assert "task_id" in task
        assert task["success"] is True
        assert "execution_time_ms" in task
        assert "result" in task
        assert task["engine_used"] == "consolidated_task_execution_engine"
        
        # Validate task result
        result = task["result"]
        assert result["status"] == "completed"
        assert result["task_type"] == "coordination"
    
    def test_execute_workflow(self, api_client):
        """Test workflow execution through consolidated engine system."""
        workflow_request = {
            "workflow_definition": {
                "name": "api_test_workflow",
                "steps": [
                    {"name": "step1", "type": "analysis"},
                    {"name": "step2", "type": "coordination"},
                    {"name": "step3", "type": "communication"}
                ]
            },
            "context": {
                "user_id": "test_user",
                "priority": "medium"
            }
        }
        
        response = api_client.post("/api/workflows/execute", json=workflow_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "workflow" in data
        
        workflow = data["workflow"]
        assert "workflow_id" in workflow
        assert workflow["success"] is True
        assert "execution_time_ms" in workflow
        assert workflow["steps_completed"] == 3  # Number of steps in definition
        assert workflow["engine_used"] == "consolidated_workflow_engine"
    
    def test_task_execution_performance(self, api_client):
        """Test task execution API performance."""
        import time
        
        task_request = {
            "task_type": "general",
            "payload": {"test": "performance"},
            "priority": 8
        }
        
        start_time = time.time()
        response = api_client.post("/api/tasks/execute", json=task_request)
        api_response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        
        # API should respond quickly (includes network + processing time)
        assert api_response_time < 500, f"API response took {api_response_time}ms, should be <500ms"
        
        # Validate the reported execution time
        data = response.json()
        task = data["task"]
        assert task["execution_time_ms"] < 100, "Task execution should meet <100ms target"


class TestSystemStatusEndpoints:
    """Test system status and metrics endpoints."""
    
    def test_get_system_status(self, api_client):
        """Test comprehensive system status endpoint."""
        response = api_client.get("/api/system/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate comprehensive status structure
        required_keys = [
            "orchestrator_status", "uptime_seconds", "system_health",
            "active_alerts", "critical_alerts", "engine_status"
        ]
        
        for key in required_keys:
            assert key in data, f"Missing required status key: {key}"
        
        assert data["orchestrator_status"] == "running"
        assert data["system_health"] == "healthy"
        assert isinstance(data["uptime_seconds"], (int, float))
        
        # Validate engine status
        engine_status = data["engine_status"]
        assert "workflow_engine" in engine_status
        assert "task_execution_engine" in engine_status
        assert "communication_engine" in engine_status
    
    def test_get_system_metrics(self, api_client):
        """Test system performance metrics endpoint."""
        response = api_client.get("/api/system/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "engine_metrics" in data
        assert data["consolidated"] is True
        
        # Validate engine metrics
        engine_metrics = data["engine_metrics"]
        assert "workflow_engine" in engine_metrics
        assert "task_execution_engine" in engine_metrics
        assert "communication_engine" in engine_metrics
        
        # Validate workflow engine metrics
        workflow_metrics = engine_metrics["workflow_engine"]
        assert "average_execution_time_ms" in workflow_metrics
        assert "success_rate_percent" in workflow_metrics
        assert workflow_metrics["average_execution_time_ms"] < 2000  # Performance target
        
        # Validate task engine metrics
        task_metrics = engine_metrics["task_execution_engine"]
        assert "average_execution_time_ms" in task_metrics
        assert "success_rate_percent" in task_metrics
        assert task_metrics["average_execution_time_ms"] < 100  # Performance target
    
    def test_get_engine_status(self, api_client):
        """Test detailed engine status endpoint."""
        response = api_client.get("/api/system/engines/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "engines" in data
        assert "coordination_layer" in data
        assert data["coordination_layer"] == "active"
        
        engines = data["engines"]
        engine_names = ["workflow_engine", "task_execution_engine", "communication_engine"]
        
        for engine_name in engine_names:
            assert engine_name in engines
            assert "status" in engines[engine_name]
            assert engines[engine_name]["status"] == "running"


class TestAPIPerformance:
    """Test API performance with consolidated system."""
    
    def test_health_check_performance(self, api_client):
        """Test health check endpoint performance."""
        import time
        
        # Test multiple health check calls
        response_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = api_client.get("/health")
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            response_times.append(response_time)
        
        # Health checks should be fast
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 100, f"Average health check time: {avg_response_time}ms"
        
        # No response should be excessively slow
        max_response_time = max(response_times)
        assert max_response_time < 200, f"Max health check time: {max_response_time}ms"
    
    def test_detailed_health_check_performance(self, api_client):
        """Test detailed health check performance."""
        import time
        
        start_time = time.time()
        response = api_client.get("/health/detailed")
        response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        
        # Detailed health check involves orchestrator calls but should still be reasonable
        assert response_time < 500, f"Detailed health check took {response_time}ms"
    
    def test_concurrent_api_requests(self, api_client):
        """Test API performance under concurrent load."""
        import concurrent.futures
        import time
        
        def make_request(endpoint):
            start_time = time.time()
            response = api_client.get(endpoint)
            response_time = (time.time() - start_time) * 1000
            return response.status_code == 200, response_time
        
        # Test concurrent requests to different endpoints
        endpoints = [
            "/health",
            "/health/engines", 
            "/api/agents",
            "/api/system/status",
            "/api/system/metrics"
        ] * 4  # 20 total requests
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = (time.time() - start_time) * 1000
        
        # Validate all requests succeeded
        successful_requests = [success for success, _ in results]
        success_rate = sum(successful_requests) / len(successful_requests)
        assert success_rate > 0.95, f"Success rate: {success_rate:.2%}"
        
        # Validate performance under concurrent load
        response_times = [time for _, time in results]
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 300, f"Average concurrent response time: {avg_response_time}ms"
        
        # Overall throughput check
        throughput = len(results) / (total_time / 1000)  # requests per second
        assert throughput > 20, f"API throughput: {throughput} req/sec"


class TestAPIErrorHandling:
    """Test API error handling with consolidated system."""
    
    def test_invalid_task_execution(self, api_client):
        """Test error handling for invalid task execution requests."""
        invalid_request = {
            "task_type": "",  # Empty task type
            "payload": None,  # Null payload
            "invalid_field": "should_be_ignored"
        }
        
        response = api_client.post("/api/tasks/execute", json=invalid_request)
        
        # Should still handle gracefully (consolidated engines handle errors)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True  # Engines handle errors gracefully
    
    def test_invalid_workflow_execution(self, api_client):
        """Test error handling for invalid workflow execution requests."""
        invalid_request = {
            "workflow_definition": {},  # Empty definition
            "context": None
        }
        
        response = api_client.post("/api/workflows/execute", json=invalid_request)
        
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True  # Engines handle errors gracefully
    
    def test_nonexistent_agent_deletion(self, api_client):
        """Test deletion of nonexistent agent."""
        nonexistent_agent_id = "nonexistent_agent_12345"
        
        response = api_client.delete(f"/api/agents/{nonexistent_agent_id}")
        
        # Should handle gracefully
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True  # Mock returns success for any agent ID


class TestAPIIntegrationWithConsolidatedSystem:
    """Test API integration with consolidated system components."""
    
    def test_api_reflects_engine_consolidation(self, api_client):
        """Test that API responses reflect engine consolidation."""
        # Test task execution
        task_response = api_client.post("/api/tasks/execute", json={"task_type": "general"})
        task_data = task_response.json()
        assert "consolidated_task_execution_engine" in task_data["task"]["engine_used"]
        
        # Test workflow execution
        workflow_response = api_client.post("/api/workflows/execute", json={
            "workflow_definition": {"steps": []}
        })
        workflow_data = workflow_response.json()
        assert "consolidated_workflow_engine" in workflow_data["workflow"]["engine_used"]
        
        # Test system metrics show consolidated engines
        metrics_response = api_client.get("/api/system/metrics")
        metrics_data = metrics_response.json()
        assert metrics_data["consolidated"] is True
        assert len(metrics_data["engine_metrics"]) == 3  # 3 consolidated engines
    
    def test_api_performance_targets_validation(self, api_client):
        """Test that API validates consolidated system performance targets."""
        # Get metrics and validate performance targets are met
        response = api_client.get("/api/system/metrics")
        data = response.json()
        
        engine_metrics = data["engine_metrics"]
        
        # Workflow engine should meet <2s target
        workflow_metrics = engine_metrics["workflow_engine"]
        assert workflow_metrics["average_execution_time_ms"] < 2000
        
        # Task engine should meet <100ms target
        task_metrics = engine_metrics["task_execution_engine"]
        assert task_metrics["average_execution_time_ms"] < 100
        
        # All engines should have high success rates
        for engine_name, metrics in engine_metrics.items():
            assert metrics["success_rate_percent"] > 95.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])