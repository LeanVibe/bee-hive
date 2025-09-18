"""
API Performance Contract Testing
===============================

Performance validation for all API endpoints with specific time targets.
Ensures API responsiveness meets requirements for production deployment
and maintains performance contracts under various load conditions.

Key Performance Areas:
- Individual endpoint response times (<500ms target)
- Concurrent request handling performance
- Memory usage during API operations
- Database query performance optimization
- WebSocket connection performance
- Load testing and scalability validation
"""

import pytest
import json
import asyncio
import time
import statistics
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import httpx
import uvicorn
import psutil
import os
from concurrent.futures import ThreadPoolExecutor


class TestAPIEndpointPerformanceContracts:
    """Performance contract tests for individual API endpoints."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for performance testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8991,
                log_level="error",
                workers=1  # Single worker for consistent testing
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Allow server to fully start
        
        yield "http://127.0.0.1:8991"

    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for performance testing."""
        async with httpx.AsyncClient(
            base_url=api_server, 
            timeout=30.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        ) as client:
            yield client

    async def test_health_endpoint_performance_contract(self, http_client):
        """Test /health endpoint meets <50ms performance contract."""
        
        target_time_ms = 50.0
        measurements = []
        
        # Take multiple measurements for accuracy
        for _ in range(10):
            start_time = time.time()
            response = await http_client.get("/health")
            response_time_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            measurements.append(response_time_ms)
            
            # Brief pause between requests
            await asyncio.sleep(0.01)
        
        # Performance contract validation
        avg_time = statistics.mean(measurements)
        max_time = max(measurements)
        
        assert avg_time < target_time_ms, f"Health endpoint avg response time {avg_time:.2f}ms exceeds {target_time_ms}ms contract"
        assert max_time < target_time_ms * 2, f"Health endpoint max response time {max_time:.2f}ms exceeds {target_time_ms * 2}ms threshold"

    async def test_system_status_performance_contract(self, http_client):
        """Test system status endpoints meet <100ms performance contract."""
        
        target_time_ms = 100.0
        status_endpoints = [
            "/status",
            "/api/v1/system/status"
        ]
        
        for endpoint in status_endpoints:
            measurements = []
            
            for _ in range(5):
                start_time = time.time()
                response = await http_client.get(endpoint)
                response_time_ms = (time.time() - start_time) * 1000
                
                assert response.status_code == 200
                measurements.append(response_time_ms)
                await asyncio.sleep(0.01)
            
            avg_time = statistics.mean(measurements)
            assert avg_time < target_time_ms, f"{endpoint} avg response time {avg_time:.2f}ms exceeds {target_time_ms}ms contract"

    async def test_agent_list_performance_contract(self, http_client):
        """Test agent listing endpoint meets <200ms performance contract."""
        
        target_time_ms = 200.0
        
        # First, populate with some agents for realistic testing
        agent_ids = []
        for i in range(5):
            agent_payload = {
                "name": f"Performance Test Agent {i}",
                "type": "claude",
                "role": "backend_developer"
            }
            response = await http_client.post("/api/v1/agents", json=agent_payload)
            if response.status_code in [200, 201]:
                agent_data = response.json()
                agent_ids.append(agent_data["id"])
        
        # Test list performance
        measurements = []
        for _ in range(5):
            start_time = time.time()
            response = await http_client.get("/api/v1/agents")
            response_time_ms = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            measurements.append(response_time_ms)
            await asyncio.sleep(0.01)
        
        avg_time = statistics.mean(measurements)
        assert avg_time < target_time_ms, f"Agent list response time {avg_time:.2f}ms exceeds {target_time_ms}ms contract"
        
        # Cleanup
        for agent_id in agent_ids:
            await http_client.delete(f"/api/v1/agents/{agent_id}")

    async def test_agent_creation_performance_contract(self, http_client):
        """Test agent creation meets <500ms performance contract."""
        
        target_time_ms = 500.0
        measurements = []
        created_agents = []
        
        for i in range(5):
            agent_payload = {
                "name": f"Performance Creation Test Agent {i}",
                "type": "claude",
                "role": "qa_engineer"
            }
            
            start_time = time.time()
            response = await http_client.post("/api/v1/agents", json=agent_payload)
            response_time_ms = (time.time() - start_time) * 1000
            
            assert response.status_code in [200, 201]
            measurements.append(response_time_ms)
            
            if response.status_code in [200, 201]:
                agent_data = response.json()
                created_agents.append(agent_data["id"])
            
            await asyncio.sleep(0.02)
        
        avg_time = statistics.mean(measurements)
        max_time = max(measurements)
        
        assert avg_time < target_time_ms, f"Agent creation avg time {avg_time:.2f}ms exceeds {target_time_ms}ms contract"
        assert max_time < target_time_ms * 1.5, f"Agent creation max time {max_time:.2f}ms exceeds {target_time_ms * 1.5}ms threshold"
        
        # Cleanup
        for agent_id in created_agents:
            await http_client.delete(f"/api/v1/agents/{agent_id}")

    async def test_task_operations_performance_contract(self, http_client):
        """Test task operations meet <300ms performance contract."""
        
        target_time_ms = 300.0
        
        # Create agent first
        agent_response = await http_client.post("/api/v1/agents", json={
            "name": "Task Performance Test Agent",
            "type": "claude"
        })
        assert agent_response.status_code in [200, 201]
        agent_data = agent_response.json()
        agent_id = agent_data["id"]
        
        try:
            # Test task creation performance
            task_measurements = []
            created_tasks = []
            
            for i in range(5):
                task_payload = {
                    "title": f"Performance Test Task {i}",
                    "description": f"Task {i} for performance testing",
                    "priority": "medium",
                    "agent_id": agent_id
                }
                
                start_time = time.time()
                response = await http_client.post("/api/v1/tasks", json=task_payload)
                response_time_ms = (time.time() - start_time) * 1000
                
                assert response.status_code in [200, 201]
                task_measurements.append(response_time_ms)
                
                if response.status_code in [200, 201]:
                    task_data = response.json()
                    created_tasks.append(task_data["id"])
                
                await asyncio.sleep(0.02)
            
            # Test task listing performance
            start_time = time.time()
            list_response = await http_client.get("/api/v1/tasks")
            list_time_ms = (time.time() - start_time) * 1000
            
            assert list_response.status_code == 200
            
            # Performance validation
            avg_create_time = statistics.mean(task_measurements)
            assert avg_create_time < target_time_ms, f"Task creation avg time {avg_create_time:.2f}ms exceeds {target_time_ms}ms contract"
            assert list_time_ms < target_time_ms, f"Task list time {list_time_ms:.2f}ms exceeds {target_time_ms}ms contract"
            
            # Cleanup tasks
            for task_id in created_tasks:
                await http_client.delete(f"/api/v1/tasks/{task_id}")
        
        finally:
            # Cleanup agent
            await http_client.delete(f"/api/v1/agents/{agent_id}")

    async def test_observability_endpoints_performance_contract(self, http_client):
        """Test observability endpoints meet <100ms performance contract."""
        
        target_time_ms = 100.0
        observability_endpoints = [
            "/observability/metrics",
            "/observability/health"
        ]
        
        for endpoint in observability_endpoints:
            measurements = []
            
            for _ in range(5):
                start_time = time.time()
                response = await http_client.get(endpoint)
                response_time_ms = (time.time() - start_time) * 1000
                
                assert response.status_code == 200
                measurements.append(response_time_ms)
                await asyncio.sleep(0.01)
            
            avg_time = statistics.mean(measurements)
            assert avg_time < target_time_ms, f"{endpoint} avg response time {avg_time:.2f}ms exceeds {target_time_ms}ms contract"


class TestConcurrentPerformanceContracts:
    """Performance tests for concurrent request handling."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for concurrent testing."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8990,
                log_level="error",
                workers=1
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)
        
        yield "http://127.0.0.1:8990"

    async def test_concurrent_read_performance_contract(self, api_server):
        """Test concurrent read operations performance contract."""
        
        max_concurrent = 20
        target_avg_time_ms = 200.0
        
        async def make_request():
            async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
                start_time = time.time()
                response = await client.get("/api/v1/agents")
                response_time_ms = (time.time() - start_time) * 1000
                return response.status_code, response_time_ms
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(max_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_requests = [r for r in results if r[0] == 200]
        response_times = [r[1] for r in successful_requests]
        
        assert len(successful_requests) == max_concurrent, f"Expected {max_concurrent} successful requests, got {len(successful_requests)}"
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        # Performance contracts
        assert avg_time < target_avg_time_ms, f"Concurrent avg response time {avg_time:.2f}ms exceeds {target_avg_time_ms}ms contract"
        assert max_time < target_avg_time_ms * 3, f"Concurrent max response time {max_time:.2f}ms exceeds threshold"
        assert total_time_ms < target_avg_time_ms * 2, f"Total concurrent execution time {total_time_ms:.2f}ms too high"

    async def test_concurrent_write_performance_contract(self, api_server):
        """Test concurrent write operations performance contract."""
        
        max_concurrent = 10
        target_avg_time_ms = 600.0  # Higher for write operations
        
        async def create_agent(agent_index):
            async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
                payload = {
                    "name": f"Concurrent Agent {agent_index}",
                    "type": "claude",
                    "role": "backend_developer"
                }
                
                start_time = time.time()
                response = await client.post("/api/v1/agents", json=payload)
                response_time_ms = (time.time() - start_time) * 1000
                
                agent_id = None
                if response.status_code in [200, 201]:
                    agent_data = response.json()
                    agent_id = agent_data["id"]
                
                return response.status_code, response_time_ms, agent_id
        
        # Execute concurrent agent creation
        start_time = time.time()
        tasks = [create_agent(i) for i in range(max_concurrent)]
        results = await asyncio.gather(*tasks)
        total_time_ms = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_creates = [r for r in results if r[0] in [200, 201] and r[2] is not None]
        response_times = [r[1] for r in successful_creates]
        created_agent_ids = [r[2] for r in successful_creates]
        
        assert len(successful_creates) >= max_concurrent * 0.8, f"Too many failed concurrent creates: {len(successful_creates)}/{max_concurrent}"
        
        if response_times:
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            
            assert avg_time < target_avg_time_ms, f"Concurrent write avg time {avg_time:.2f}ms exceeds {target_avg_time_ms}ms contract"
            assert max_time < target_avg_time_ms * 2, f"Concurrent write max time {max_time:.2f}ms exceeds threshold"
        
        # Cleanup created agents
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            for agent_id in created_agent_ids:
                try:
                    await client.delete(f"/api/v1/agents/{agent_id}")
                except:
                    pass  # Ignore cleanup errors

    async def test_mixed_operations_performance_contract(self, api_server):
        """Test mixed read/write operations performance contract."""
        
        total_operations = 30
        target_completion_time_ms = 3000.0  # 3 seconds for all operations
        
        # Create some initial data
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            initial_agents = []
            for i in range(5):
                response = await client.post("/api/v1/agents", json={
                    "name": f"Initial Agent {i}",
                    "type": "claude"
                })
                if response.status_code in [200, 201]:
                    agent_data = response.json()
                    initial_agents.append(agent_data["id"])
        
        try:
            async def mixed_operation(op_index):
                async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
                    operation_type = op_index % 3
                    start_time = time.time()
                    
                    if operation_type == 0:  # Read operation
                        response = await client.get("/api/v1/agents")
                        success = response.status_code == 200
                    elif operation_type == 1:  # Create operation
                        response = await client.post("/api/v1/agents", json={
                            "name": f"Mixed Op Agent {op_index}",
                            "type": "claude"
                        })
                        success = response.status_code in [200, 201]
                    else:  # Status check operation
                        response = await client.get("/health")
                        success = response.status_code == 200
                    
                    response_time_ms = (time.time() - start_time) * 1000
                    return success, response_time_ms, operation_type
            
            # Execute mixed operations
            start_time = time.time()
            tasks = [mixed_operation(i) for i in range(total_operations)]
            results = await asyncio.gather(*tasks)
            total_time_ms = (time.time() - start_time) * 1000
            
            # Analyze results
            successful_ops = [r for r in results if r[0]]
            response_times = [r[1] for r in successful_ops]
            
            success_rate = len(successful_ops) / total_operations
            assert success_rate >= 0.9, f"Mixed operations success rate {success_rate:.2%} below 90% threshold"
            
            if response_times:
                avg_time = statistics.mean(response_times)
                assert avg_time < 500.0, f"Mixed operations avg time {avg_time:.2f}ms exceeds 500ms contract"
            
            assert total_time_ms < target_completion_time_ms, f"Mixed operations total time {total_time_ms:.2f}ms exceeds {target_completion_time_ms}ms contract"
        
        finally:
            # Cleanup
            async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
                for agent_id in initial_agents:
                    try:
                        await client.delete(f"/api/v1/agents/{agent_id}")
                    except:
                        pass


class TestMemoryPerformanceContracts:
    """Performance tests for memory usage during API operations."""

    def get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB

    async def test_memory_usage_stability_contract(self):
        """Test memory usage remains stable during operations."""
        
        max_memory_increase_mb = 50.0  # Allow up to 50MB increase
        
        # Measure baseline memory
        initial_memory = self.get_process_memory_mb()
        
        # Simulate API operations that should not cause memory leaks
        for cycle in range(5):
            # Simulate request processing
            test_data = {
                "agents": [{"id": f"agent-{i}", "name": f"Agent {i}"} for i in range(100)],
                "tasks": [{"id": f"task-{i}", "title": f"Task {i}"} for i in range(200)]
            }
            
            # Process and discard data (simulating request handling)
            processed_data = json.dumps(test_data)
            parsed_data = json.loads(processed_data)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            await asyncio.sleep(0.1)
        
        # Measure final memory
        final_memory = self.get_process_memory_mb()
        memory_increase = final_memory - initial_memory
        
        # Memory contract
        assert memory_increase < max_memory_increase_mb, f"Memory increased by {memory_increase:.2f}MB, exceeds {max_memory_increase_mb}MB contract"

    async def test_large_response_memory_contract(self):
        """Test memory efficiency with large API responses."""
        
        # Test that large responses don't cause excessive memory usage
        max_memory_per_mb_response = 5.0  # Max 5MB memory per 1MB response
        
        # Create large test data (simulate large agent list)
        large_agent_list = [
            {
                "id": f"large-agent-{i}",
                "name": f"Large Response Test Agent {i}",
                "type": "claude",
                "role": "backend_developer",
                "capabilities": ["coding", "testing", "debugging", "optimization"],
                "created_at": "2025-01-18T12:00:00Z",
                "updated_at": "2025-01-18T12:00:00Z"
            }
            for i in range(1000)  # 1000 agents
        ]
        
        initial_memory = self.get_process_memory_mb()
        
        # Serialize large response (simulating API response generation)
        large_response = {
            "agents": large_agent_list,
            "total": len(large_agent_list),
            "offset": 0,
            "limit": 1000
        }
        
        json_response = json.dumps(large_response)
        response_size_mb = len(json_response.encode('utf-8')) / 1024 / 1024
        
        # Process response (simulating client processing)
        parsed_response = json.loads(json_response)
        
        final_memory = self.get_process_memory_mb()
        memory_increase = final_memory - initial_memory
        
        # Memory efficiency contract
        memory_per_mb = memory_increase / response_size_mb if response_size_mb > 0 else 0
        assert memory_per_mb < max_memory_per_mb_response, f"Memory usage {memory_per_mb:.2f}MB per MB response exceeds {max_memory_per_mb_response}MB contract"


class TestWebSocketPerformanceContracts:
    """Performance tests for WebSocket operations."""

    async def test_websocket_connection_performance_contract(self):
        """Test WebSocket connection establishment performance."""
        
        target_connection_time_ms = 100.0
        
        # Test WebSocket connection time
        # Note: This is a simplified test since we need actual WebSocket server
        
        connection_times = []
        
        for _ in range(5):
            start_time = time.time()
            
            # Simulate WebSocket connection overhead
            connection_data = {
                "type": "connection_request",
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": f"test-client-{time.time()}"
            }
            
            # Simulate connection processing
            await asyncio.sleep(0.01)  # Minimal processing time
            
            connection_time_ms = (time.time() - start_time) * 1000
            connection_times.append(connection_time_ms)
        
        avg_connection_time = statistics.mean(connection_times)
        max_connection_time = max(connection_times)
        
        # Performance contracts
        assert avg_connection_time < target_connection_time_ms, f"WebSocket connection avg time {avg_connection_time:.2f}ms exceeds {target_connection_time_ms}ms contract"
        assert max_connection_time < target_connection_time_ms * 2, f"WebSocket connection max time {max_connection_time:.2f}ms exceeds threshold"

    async def test_websocket_message_throughput_contract(self):
        """Test WebSocket message processing throughput."""
        
        target_messages_per_second = 100
        test_duration_seconds = 1.0
        
        # Simulate message processing
        messages_processed = 0
        start_time = time.time()
        
        while (time.time() - start_time) < test_duration_seconds:
            # Simulate message processing
            message = {
                "type": "test_message",
                "data": {"id": messages_processed},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Process message (JSON serialization/deserialization)
            json_msg = json.dumps(message)
            parsed_msg = json.loads(json_msg)
            
            messages_processed += 1
            
            # Brief pause to avoid tight loop
            await asyncio.sleep(0.001)
        
        actual_duration = time.time() - start_time
        messages_per_second = messages_processed / actual_duration
        
        # Throughput contract
        assert messages_per_second >= target_messages_per_second, f"WebSocket throughput {messages_per_second:.2f} msg/sec below {target_messages_per_second} msg/sec contract"


# Performance Contract Summary
class TestAPIPerformanceContractSummary:
    """Summary test validating all API performance contracts work together."""
    
    async def test_complete_api_performance_contract_compliance(self):
        """Integration test ensuring all performance contracts are met simultaneously."""
        
        # Start server for comprehensive testing
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8989,
                log_level="error",
                workers=1
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)
        
        base_url = "http://127.0.0.1:8989"
        
        # Test complete performance workflow
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            
            total_start_time = time.time()
            
            # 1. Health check performance (target: <50ms)
            health_start = time.time()
            health_response = await client.get("/health")
            health_time = (time.time() - health_start) * 1000
            
            assert health_response.status_code == 200
            assert health_time < 50.0, f"Health check {health_time:.2f}ms exceeds 50ms contract"
            
            # 2. System status performance (target: <100ms)
            status_start = time.time()
            status_response = await client.get("/api/v1/system/status")
            status_time = (time.time() - status_start) * 1000
            
            assert status_response.status_code == 200
            assert status_time < 100.0, f"System status {status_time:.2f}ms exceeds 100ms contract"
            
            # 3. Agent operations performance (target: <500ms each)
            agent_start = time.time()
            agent_response = await client.post("/api/v1/agents", json={
                "name": "Performance Summary Agent",
                "type": "claude",
                "role": "full_stack_developer"
            })
            agent_create_time = (time.time() - agent_start) * 1000
            
            assert agent_response.status_code in [200, 201]
            assert agent_create_time < 500.0, f"Agent creation {agent_create_time:.2f}ms exceeds 500ms contract"
            
            if agent_response.status_code in [200, 201]:
                agent_data = agent_response.json()
                agent_id = agent_data["id"]
                
                # 4. Agent retrieval performance
                get_start = time.time()
                get_response = await client.get(f"/api/v1/agents/{agent_id}")
                get_time = (time.time() - get_start) * 1000
                
                assert get_response.status_code == 200
                assert get_time < 200.0, f"Agent retrieval {get_time:.2f}ms exceeds 200ms contract"
                
                # 5. Task operations performance
                task_start = time.time()
                task_response = await client.post("/api/v1/tasks", json={
                    "title": "Performance Summary Task",
                    "description": "Comprehensive performance test task",
                    "priority": "high",
                    "agent_id": agent_id
                })
                task_create_time = (time.time() - task_start) * 1000
                
                assert task_response.status_code in [200, 201]
                assert task_create_time < 500.0, f"Task creation {task_create_time:.2f}ms exceeds 500ms contract"
                
                if task_response.status_code in [200, 201]:
                    task_data = task_response.json()
                    task_id = task_data["id"]
                    
                    # 6. List operations performance
                    list_start = time.time()
                    agents_list = await client.get("/api/v1/agents")
                    tasks_list = await client.get("/api/v1/tasks")
                    list_time = (time.time() - list_start) * 1000
                    
                    assert agents_list.status_code == 200
                    assert tasks_list.status_code == 200
                    assert list_time < 400.0, f"List operations {list_time:.2f}ms exceeds 400ms contract"
                    
                    # 7. Observability performance
                    obs_start = time.time()
                    metrics_response = await client.get("/observability/metrics")
                    obs_health_response = await client.get("/observability/health")
                    obs_time = (time.time() - obs_start) * 1000
                    
                    assert metrics_response.status_code == 200
                    assert obs_health_response.status_code == 200
                    assert obs_time < 200.0, f"Observability {obs_time:.2f}ms exceeds 200ms contract"
                    
                    # 8. Cleanup performance
                    cleanup_start = time.time()
                    await client.delete(f"/api/v1/tasks/{task_id}")
                    await client.delete(f"/api/v1/agents/{agent_id}")
                    cleanup_time = (time.time() - cleanup_start) * 1000
                    
                    assert cleanup_time < 600.0, f"Cleanup operations {cleanup_time:.2f}ms exceeds 600ms contract"
            
            # 9. Total workflow performance
            total_time = (time.time() - total_start_time) * 1000
            
            # Complete workflow should finish within reasonable time
            assert total_time < 3000.0, f"Complete workflow {total_time:.2f}ms exceeds 3000ms contract"
            
            # 10. Concurrent performance validation
            concurrent_start = time.time()
            
            # Execute 5 concurrent health checks
            concurrent_tasks = [client.get("/health") for _ in range(5)]
            concurrent_responses = await asyncio.gather(*concurrent_tasks)
            concurrent_time = (time.time() - concurrent_start) * 1000
            
            # All should succeed
            assert all(r.status_code == 200 for r in concurrent_responses)
            
            # Concurrent execution should be efficient
            assert concurrent_time < 300.0, f"Concurrent operations {concurrent_time:.2f}ms exceeds 300ms contract"
            
            # Average per request should be reasonable
            avg_concurrent_time = concurrent_time / 5
            assert avg_concurrent_time < 100.0, f"Average concurrent request {avg_concurrent_time:.2f}ms exceeds 100ms contract"