"""
API Performance Testing Suite - Phase 4
Tests API performance requirements following PLAN.md specifications

Performance Requirements:
- Load testing with 50+ concurrent agents
- Stress testing to identify breaking points  
- Response time validation (<200ms standard operations)
- Memory usage under API load
- Concurrent request handling
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
from datetime import datetime, timezone
import structlog
import psutil
import threading

from app.main import create_app
from app.core.simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance

logger = structlog.get_logger()

class TestAPIPerformanceValidation:
    """Comprehensive API performance testing following Phase 4 requirements."""
    
    @pytest.fixture(scope="class")
    def test_app(self):
        """Create test application for performance testing."""
        import os
        os.environ.update({
            'SKIP_STARTUP_INIT': 'true',
            'CI': 'true', 
            'TESTING': 'true'
        })
        return create_app()
    
    @pytest.fixture
    def client(self, test_app):
        """Create test client for performance testing."""
        return TestClient(test_app)
    
    @pytest.fixture
    def mock_orchestrator_for_load(self):
        """Mock orchestrator optimized for load testing."""
        orchestrator = Mock(spec=SimpleOrchestrator)
        
        # Fast mock responses for load testing
        def create_agent_fast(*args, **kwargs):
            return AgentInstance(
                id=f"load-test-agent-{time.time_ns()}",
                role=AgentRole.BACKEND_DEVELOPER,
                status="active",
                created_at=datetime.now(timezone.utc),
                workspace_name="load-test-workspace",
                git_branch="main"
            )
        
        def list_agents_fast(*args, **kwargs):
            # Return variable number of agents for load testing
            agent_count = min(50, len(threading.enumerate()) * 2)  # Scale with load
            return [
                AgentInstance(
                    id=f"load-agent-{i}",
                    role=AgentRole.BACKEND_DEVELOPER,
                    status="active" if i % 2 == 0 else "idle",
                    created_at=datetime.now(timezone.utc),
                    workspace_name=f"workspace-{i}",
                    git_branch="main"
                ) for i in range(agent_count)
            ]
        
        def get_system_status_fast(*args, **kwargs):
            return {
                "status": "healthy",
                "agents": {"total": 25, "active": 15, "idle": 10},
                "performance": {
                    "response_time_ms": 0.01,
                    "memory_usage_mb": 45.2,
                    "cpu_usage_percent": 12.5
                },
                "health": "healthy"
            }
        
        orchestrator.create_agent = AsyncMock(side_effect=create_agent_fast)
        orchestrator.list_agents = AsyncMock(side_effect=list_agents_fast)
        orchestrator.get_system_status = AsyncMock(side_effect=get_system_status_fast)
        orchestrator.get_agent = AsyncMock(side_effect=create_agent_fast)
        orchestrator.shutdown_agent = AsyncMock(return_value=True)
        
        return orchestrator
    
    # === LEVEL 1: Response Time Performance ===
    
    @pytest.mark.performance
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint responds within 50ms requirement."""
        response_times = []
        
        # Warm up
        client.get("/health")
        
        # Measure response times
        for _ in range(10):
            start_time = time.perf_counter()
            response = client.get("/health")
            end_time = time.perf_counter()
            
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            assert response.status_code == 200
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        max_time = max(response_times)
        
        # Health endpoint should respond within 50ms
        assert avg_time < 50, f"Average response time {avg_time:.2f}ms exceeds 50ms limit"
        assert p95_time < 100, f"95th percentile {p95_time:.2f}ms exceeds 100ms limit"
        
        logger.info("Health endpoint performance validation passed",
                   avg_response_time_ms=round(avg_time, 2),
                   p95_response_time_ms=round(p95_time, 2),
                   max_response_time_ms=round(max_time, 2))
    
    @pytest.mark.performance
    def test_agent_listing_response_time(self, client, mock_orchestrator_for_load):
        """Test agent listing endpoint responds within 200ms requirement."""
        with patch('app.api.v2.agents.get_orchestrator', return_value=mock_orchestrator_for_load):
            response_times = []
            
            # Warm up
            client.get("/api/v2/agents")
            
            # Measure response times
            for _ in range(10):
                start_time = time.perf_counter()
                response = client.get("/api/v2/agents")
                end_time = time.perf_counter()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                assert response.status_code == 200
            
            # Calculate statistics
            avg_time = statistics.mean(response_times)
            p95_time = statistics.quantiles(response_times, n=20)[18]
            
            # Agent listing should respond within 200ms for standard operations
            assert avg_time < 200, f"Average response time {avg_time:.2f}ms exceeds 200ms limit"
            assert p95_time < 500, f"95th percentile {p95_time:.2f}ms exceeds 500ms limit"
            
            logger.info("Agent listing performance validation passed",
                       avg_response_time_ms=round(avg_time, 2),
                       p95_response_time_ms=round(p95_time, 2))
    
    # === LEVEL 2: Concurrent Request Performance ===
    
    @pytest.mark.performance
    @pytest.mark.timeout(30)
    def test_concurrent_health_checks(self, client):
        """Test health endpoint under concurrent load."""
        
        def make_health_request():
            start_time = time.perf_counter()
            response = client.get("/health")
            end_time = time.perf_counter()
            
            return {
                'status_code': response.status_code,
                'response_time_ms': (end_time - start_time) * 1000,
                'success': response.status_code == 200
            }
        
        # Test with 20 concurrent requests
        concurrent_requests = 20
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit all requests concurrently
            futures = [executor.submit(make_health_request) for _ in range(concurrent_requests)]
            
            # Collect results
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        response_times = [r['response_time_ms'] for r in results]
        
        success_rate = len(successful_requests) / len(results) * 100
        avg_response_time = statistics.mean(response_times) 
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
        
        # Validate concurrent performance
        assert success_rate >= 95, f"Success rate {success_rate:.1f}% below 95% threshold"
        assert avg_response_time < 100, f"Concurrent avg response time {avg_response_time:.2f}ms too high"
        
        logger.info("Concurrent health check performance validation passed",
                   concurrent_requests=concurrent_requests,
                   success_rate=f"{success_rate:.1f}%",
                   avg_response_time_ms=round(avg_response_time, 2),
                   p95_response_time_ms=round(p95_response_time, 2))
    
    @pytest.mark.performance
    @pytest.mark.timeout(60) 
    def test_concurrent_agent_operations(self, client, mock_orchestrator_for_load):
        """Test agent management endpoints under concurrent load."""
        
        def make_agent_request(operation_type, agent_id=None):
            start_time = time.perf_counter()
            
            if operation_type == "list":
                response = client.get("/api/v2/agents")
            elif operation_type == "create":
                response = client.post("/api/v2/agents", json={
                    "role": "backend_developer",
                    "agent_type": "claude_code",
                    "workspace_name": f"concurrent-test-{time.time_ns()}"
                })
            elif operation_type == "get" and agent_id:
                response = client.get(f"/api/v2/agents/{agent_id}")
            else:
                response = Mock(status_code=400)
            
            end_time = time.perf_counter()
            
            return {
                'operation': operation_type,
                'status_code': response.status_code,
                'response_time_ms': (end_time - start_time) * 1000,
                'success': response.status_code in [200, 201]
            }
        
        with patch('app.api.v2.agents.get_orchestrator', return_value=mock_orchestrator_for_load):
            # Mix of operations: 60% list, 30% create, 10% get
            operations = (
                ['list'] * 30 + 
                ['create'] * 15 + 
                [('get', 'test-agent')] * 5
            )
            
            # Execute concurrent operations
            with ThreadPoolExecutor(max_workers=25) as executor:
                futures = []
                for op in operations:
                    if isinstance(op, tuple):
                        futures.append(executor.submit(make_agent_request, op[0], op[1]))
                    else:
                        futures.append(executor.submit(make_agent_request, op))
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Analyze performance
            successful_operations = [r for r in results if r['success']]
            response_times = [r['response_time_ms'] for r in results]
            
            success_rate = len(successful_operations) / len(results) * 100
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else response_times[0]
            
            # Validate concurrent agent operations performance
            assert success_rate >= 90, f"Agent operations success rate {success_rate:.1f}% below 90%"
            assert avg_response_time < 300, f"Concurrent avg response time {avg_response_time:.2f}ms too high"
            
            logger.info("Concurrent agent operations performance validation passed",
                       total_operations=len(results),
                       success_rate=f"{success_rate:.1f}%", 
                       avg_response_time_ms=round(avg_response_time, 2),
                       p95_response_time_ms=round(p95_response_time, 2))
    
    # === LEVEL 3: Load Testing with 50+ Concurrent Agents ===
    
    @pytest.mark.performance
    @pytest.mark.timeout(120)
    def test_fifty_plus_concurrent_agent_simulation(self, client, mock_orchestrator_for_load):
        """Test system handles 50+ concurrent agent operations as required by PLAN.md."""
        
        def agent_lifecycle_simulation(agent_number):
            """Simulate complete agent lifecycle."""
            try:
                results = []
                agent_id = f"perf-test-agent-{agent_number}"
                
                # Step 1: Create agent
                start_time = time.perf_counter()
                create_response = client.post("/api/v2/agents", json={
                    "role": "backend_developer",
                    "agent_type": "claude_code",
                    "workspace_name": f"perf-workspace-{agent_number}"
                })
                create_time = (time.perf_counter() - start_time) * 1000
                results.append(('create', create_response.status_code, create_time))
                
                # Step 2: List agents (check if agent appears)
                start_time = time.perf_counter()
                list_response = client.get("/api/v2/agents")
                list_time = (time.perf_counter() - start_time) * 1000
                results.append(('list', list_response.status_code, list_time))
                
                # Step 3: Get agent details
                start_time = time.perf_counter()  
                get_response = client.get(f"/api/v2/agents/{agent_id}")
                get_time = (time.perf_counter() - start_time) * 1000
                results.append(('get', get_response.status_code, get_time))
                
                # Step 4: Shutdown agent
                start_time = time.perf_counter()
                shutdown_response = client.delete(f"/api/v2/agents/{agent_id}")
                shutdown_time = (time.perf_counter() - start_time) * 1000
                results.append(('shutdown', shutdown_response.status_code, shutdown_time))
                
                return {
                    'agent_number': agent_number,
                    'operations': results,
                    'total_time_ms': sum(r[2] for r in results),
                    'success': all(r[1] in [200, 201, 404] for r in results)  # 404 ok for shutdown
                }
                
            except Exception as e:
                return {
                    'agent_number': agent_number,
                    'operations': [],
                    'total_time_ms': 0,
                    'success': False,
                    'error': str(e)
                }
        
        with patch('app.api.v2.agents.get_orchestrator', return_value=mock_orchestrator_for_load):
            # Simulate 55 concurrent agents (exceeding 50+ requirement)
            num_agents = 55
            
            logger.info("Starting 50+ concurrent agent simulation",
                       agent_count=num_agents)
            
            # Execute concurrent agent simulations
            with ThreadPoolExecutor(max_workers=min(num_agents, 30)) as executor:
                futures = [
                    executor.submit(agent_lifecycle_simulation, i) 
                    for i in range(num_agents)
                ]
                
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Analyze load test results
            successful_agents = [r for r in results if r['success']]
            failed_agents = [r for r in results if not r['success']]
            
            if successful_agents:
                total_operations = sum(len(r['operations']) for r in successful_agents)
                avg_agent_time = statistics.mean([r['total_time_ms'] for r in successful_agents])
                
                # Extract all operation times for analysis
                all_operation_times = []
                for result in successful_agents:
                    for op_name, status_code, op_time in result['operations']:
                        all_operation_times.append(op_time)
                
                if all_operation_times:
                    avg_operation_time = statistics.mean(all_operation_times)
                    p95_operation_time = statistics.quantiles(all_operation_times, n=20)[18] if len(all_operation_times) > 1 else all_operation_times[0]
                else:
                    avg_operation_time = 0
                    p95_operation_time = 0
            else:
                total_operations = 0
                avg_agent_time = 0
                avg_operation_time = 0
                p95_operation_time = 0
            
            success_rate = len(successful_agents) / len(results) * 100
            
            # Validate 50+ concurrent agent requirements
            assert len(successful_agents) >= 45, f"Only {len(successful_agents)} of {num_agents} agents succeeded (need 45+)"
            assert success_rate >= 80, f"Success rate {success_rate:.1f}% below 80% threshold for load test"
            
            # Performance should remain reasonable under load
            if avg_operation_time > 0:
                assert avg_operation_time < 1000, f"Average operation time {avg_operation_time:.2f}ms too high under load"
            
            logger.info("50+ concurrent agent simulation validation passed",
                       total_agents=num_agents,
                       successful_agents=len(successful_agents),
                       failed_agents=len(failed_agents),
                       success_rate=f"{success_rate:.1f}%",
                       total_operations=total_operations,
                       avg_agent_time_ms=round(avg_agent_time, 2),
                       avg_operation_time_ms=round(avg_operation_time, 2),
                       p95_operation_time_ms=round(p95_operation_time, 2))
    
    # === LEVEL 4: Memory and Resource Usage ===
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, client, mock_orchestrator_for_load):
        """Test memory usage remains acceptable under API load."""
        
        # Get baseline memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        with patch('app.api.v2.agents.get_orchestrator', return_value=mock_orchestrator_for_load):
            # Generate sustained load
            def sustained_requests():
                for _ in range(50):
                    client.get("/health")
                    client.get("/api/v2/agents")
                    client.get("/status")
            
            # Run sustained load for memory measurement
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(sustained_requests) for _ in range(5)]
                
                # Wait for completion and measure peak memory
                peak_memory_mb = initial_memory_mb
                for future in as_completed(futures):
                    future.result()
                    current_memory_mb = process.memory_info().rss / 1024 / 1024
                    peak_memory_mb = max(peak_memory_mb, current_memory_mb)
            
            # Final memory measurement
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_growth_mb = final_memory_mb - initial_memory_mb
        
        # Memory usage should remain reasonable 
        assert peak_memory_mb < 200, f"Peak memory {peak_memory_mb:.1f}MB exceeds 200MB limit"
        assert memory_growth_mb < 50, f"Memory growth {memory_growth_mb:.1f}MB too high"
        
        logger.info("Memory usage under load validation passed",
                   initial_memory_mb=round(initial_memory_mb, 1),
                   peak_memory_mb=round(peak_memory_mb, 1), 
                   final_memory_mb=round(final_memory_mb, 1),
                   memory_growth_mb=round(memory_growth_mb, 1))
    
    # === LEVEL 5: Stress Testing to Breaking Point ===
    
    @pytest.mark.performance
    @pytest.mark.timeout(180)
    def test_stress_test_breaking_point_identification(self, client, mock_orchestrator_for_load):
        """Identify system breaking point through progressive load increase."""
        
        breaking_points = {}
        
        with patch('app.api.v2.agents.get_orchestrator', return_value=mock_orchestrator_for_load):
            # Progressive load levels
            load_levels = [10, 25, 50, 75, 100]
            
            for load_level in load_levels:
                logger.info("Testing load level", concurrent_requests=load_level)
                
                def stress_request():
                    start_time = time.perf_counter()
                    try:
                        response = client.get("/api/v2/agents")
                        end_time = time.perf_counter()
                        
                        return {
                            'success': response.status_code == 200,
                            'response_time_ms': (end_time - start_time) * 1000,
                            'status_code': response.status_code
                        }
                    except Exception as e:
                        end_time = time.perf_counter()
                        return {
                            'success': False,
                            'response_time_ms': (end_time - start_time) * 1000,
                            'status_code': 500,
                            'error': str(e)
                        }
                
                # Execute stress test at current load level
                with ThreadPoolExecutor(max_workers=load_level) as executor:
                    futures = [executor.submit(stress_request) for _ in range(load_level)]
                    results = [future.result() for future in as_completed(futures)]
                
                # Analyze results for this load level
                successful_requests = [r for r in results if r['success']]
                success_rate = len(successful_requests) / len(results) * 100
                
                if successful_requests:
                    avg_response_time = statistics.mean([r['response_time_ms'] for r in successful_requests])
                    p95_response_time = statistics.quantiles(
                        [r['response_time_ms'] for r in successful_requests], 
                        n=20
                    )[18] if len(successful_requests) > 1 else successful_requests[0]['response_time_ms']
                else:
                    avg_response_time = 0
                    p95_response_time = 0
                
                breaking_points[load_level] = {
                    'success_rate': success_rate,
                    'avg_response_time_ms': avg_response_time,
                    'p95_response_time_ms': p95_response_time
                }
                
                logger.info("Load level testing complete",
                           load_level=load_level,
                           success_rate=f"{success_rate:.1f}%",
                           avg_response_time_ms=round(avg_response_time, 2))
                
                # Stop if we hit breaking point (success rate < 50% or avg response > 2 seconds)
                if success_rate < 50 or avg_response_time > 2000:
                    logger.warning("Breaking point identified", 
                                 load_level=load_level,
                                 success_rate=f"{success_rate:.1f}%",
                                 avg_response_time_ms=round(avg_response_time, 2))
                    break
        
        # Validate that system handles at least 25 concurrent requests well
        assert breaking_points[25]['success_rate'] >= 90, "System fails at 25 concurrent requests"
        assert breaking_points[25]['avg_response_time_ms'] < 500, "Response time too high at 25 concurrent requests"
        
        logger.info("Stress test breaking point identification complete",
                   breaking_points=breaking_points)


# Export test classes for pytest discovery
__all__ = ['TestAPIPerformanceValidation']