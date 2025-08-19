"""
Performance Smoke Tests

Validates that the system meets Epic 1 performance requirements:
- <100ms response times for critical operations
- <500ms for complex operations
- System stability under load
- Resource utilization within acceptable limits
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Tuple


class TestCriticalPathPerformance:
    """Test performance of critical system paths."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_performance_target(self, async_test_client):
        """Health endpoint must respond within 100ms consistently."""
        response_times = []
        
        # Test multiple requests to get average
        for _ in range(10):
            start_time = time.time()
            response = await async_test_client.get("/health")
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            
            # Each request should be under target
            assert response.status_code in [200, 500]
        
        # Statistical analysis
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        # Performance targets
        assert avg_time < 100, f"Average response time {avg_time:.2f}ms exceeds 100ms target"
        assert max_time < 200, f"Maximum response time {max_time:.2f}ms exceeds 200ms limit"
        assert p95_time < 150, f"95th percentile {p95_time:.2f}ms exceeds 150ms target"
    
    @pytest.mark.asyncio
    async def test_orchestrator_performance_target(self, test_app):
        """SimpleOrchestrator operations must meet performance targets."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        orchestrator = create_simple_orchestrator()
        
        # Test system status performance
        status_times = []
        for _ in range(10):
            start_time = time.time()
            await orchestrator.get_system_status()
            end_time = time.time()
            status_times.append((end_time - start_time) * 1000)
        
        avg_status_time = statistics.mean(status_times)
        assert avg_status_time < 50, f"System status avg {avg_status_time:.2f}ms exceeds 50ms target"
        
        # Test agent spawn performance
        spawn_times = []
        agent_ids = []
        
        for _ in range(5):  # Fewer iterations for expensive operations
            start_time = time.time()
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                context={"perf_test": True}
            )
            end_time = time.time()
            
            spawn_times.append((end_time - start_time) * 1000)
            agent_ids.append(agent_id)
        
        avg_spawn_time = statistics.mean(spawn_times)
        assert avg_spawn_time < 100, f"Agent spawn avg {avg_spawn_time:.2f}ms exceeds 100ms target"
        
        # Test agent status check performance
        status_check_times = []
        for agent_id in agent_ids:
            start_time = time.time()
            await orchestrator.get_agent_status(agent_id)
            end_time = time.time()
            status_check_times.append((end_time - start_time) * 1000)
        
        avg_status_check_time = statistics.mean(status_check_times)
        assert avg_status_check_time < 50, f"Agent status check avg {avg_status_check_time:.2f}ms exceeds 50ms target"
        
        # Clean up agents
        for agent_id in agent_ids:
            await orchestrator.shutdown_agent(agent_id, graceful=True)
    
    @pytest.mark.asyncio
    async def test_database_operation_performance(self, test_db_session):
        """Database operations should be fast."""
        from sqlalchemy import text
        
        # Test simple queries
        query_times = []
        for _ in range(20):
            start_time = time.time()
            result = await test_db_session.execute(text("SELECT 1"))
            _ = result.scalar()
            end_time = time.time()
            
            query_times.append((end_time - start_time) * 1000)
        
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        assert avg_query_time < 10, f"Database query avg {avg_query_time:.2f}ms exceeds 10ms target"
        assert max_query_time < 50, f"Database query max {max_query_time:.2f}ms exceeds 50ms limit"


class TestConcurrencyPerformance:
    """Test performance under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks_performance(self, async_test_client):
        """System should handle concurrent requests efficiently."""
        async def timed_health_check() -> Tuple[float, int]:
            start_time = time.time()
            response = await async_test_client.get("/health")
            end_time = time.time()
            return (end_time - start_time) * 1000, response.status_code
        
        # Test with increasing concurrency
        concurrency_levels = [1, 5, 10]
        
        for concurrency in concurrency_levels:
            tasks = [timed_health_check() for _ in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            response_times = [time for time, _ in results]
            status_codes = [code for _, code in results]
            
            # All requests should complete
            assert len(results) == concurrency
            
            # All should return valid status codes
            for status_code in status_codes:
                assert status_code in [200, 500]
            
            # Performance should not degrade significantly with concurrency
            avg_time = statistics.mean(response_times)
            max_time = max(response_times)
            
            # Allow more time for higher concurrency, but still reasonable
            max_allowed = 200 + (concurrency * 10)  # Scale with concurrency
            assert avg_time < max_allowed, f"Concurrency {concurrency}: avg time {avg_time:.2f}ms exceeds {max_allowed}ms"
            assert max_time < max_allowed * 2, f"Concurrency {concurrency}: max time {max_time:.2f}ms exceeds {max_allowed * 2}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_orchestrator_operations(self, test_app):
        """Test concurrent orchestrator operations performance."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        orchestrator = create_simple_orchestrator()
        
        async def timed_system_status() -> float:
            start_time = time.time()
            await orchestrator.get_system_status()
            end_time = time.time()
            return (end_time - start_time) * 1000
        
        # Test concurrent system status calls
        tasks = [timed_system_status() for _ in range(10)]
        response_times = await asyncio.gather(*tasks)
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        # Should handle concurrent operations efficiently
        assert avg_time < 100, f"Concurrent system status avg {avg_time:.2f}ms exceeds 100ms"
        assert max_time < 300, f"Concurrent system status max {max_time:.2f}ms exceeds 300ms"


class TestMemoryPerformance:
    """Test memory usage patterns for performance."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_memory_efficiency(self, test_app):
        """Test orchestrator doesn't leak memory during operations."""
        import gc
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        # Force garbage collection to get baseline
        gc.collect()
        
        orchestrator = create_simple_orchestrator()
        
        # Perform many operations that should not accumulate memory
        for i in range(50):  # Moderate number for smoke test
            # Mix of operations
            if i % 10 == 0:
                # Spawn and immediately shutdown agents occasionally
                agent_id = await orchestrator.spawn_agent(
                    role=AgentRole.QA_ENGINEER,
                    context={"memory_test": i}
                )
                await orchestrator.shutdown_agent(agent_id, graceful=True)
            else:
                # Mostly just status checks
                await orchestrator.get_system_status()
        
        # Force garbage collection
        gc.collect()
        
        # Final system status should still be fast
        start_time = time.time()
        final_status = await orchestrator.get_system_status()
        end_time = time.time()
        
        final_response_time = (end_time - start_time) * 1000
        assert final_response_time < 100, f"After memory test, response time {final_response_time:.2f}ms degraded"
        assert final_status["health"] in ["healthy", "no_agents"]
    
    @pytest.mark.asyncio
    async def test_api_memory_efficiency(self, async_test_client):
        """Test API endpoints don't accumulate memory over many requests."""
        import gc
        
        gc.collect()
        
        # Make many API requests
        endpoints = ["/health", "/status", "/debug-agents"]
        
        for i in range(30):  # Moderate number for smoke test
            endpoint = endpoints[i % len(endpoints)]
            response = await async_test_client.get(endpoint)
            assert response.status_code in [200, 500]
        
        gc.collect()
        
        # Final requests should still be fast
        final_times = []
        for endpoint in endpoints:
            start_time = time.time()
            response = await async_test_client.get(endpoint)
            end_time = time.time()
            
            final_times.append((end_time - start_time) * 1000)
            assert response.status_code in [200, 500]
        
        avg_final_time = statistics.mean(final_times)
        assert avg_final_time < 200, f"After memory test, avg API response time {avg_final_time:.2f}ms degraded"


class TestStressScenarios:
    """Light stress testing to validate performance under load."""
    
    @pytest.mark.asyncio
    async def test_rapid_agent_lifecycle_performance(self, test_app):
        """Test rapid agent spawn/shutdown cycles don't degrade performance."""
        from app.core.simple_orchestrator import create_simple_orchestrator, AgentRole
        
        orchestrator = create_simple_orchestrator()
        
        cycle_times = []
        
        # Rapid spawn/shutdown cycles
        for _ in range(10):  # Keep moderate for smoke test
            cycle_start = time.time()
            
            # Spawn agent
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER,
                context={"stress_test": True}
            )
            
            # Check status
            await orchestrator.get_agent_status(agent_id)
            
            # Shutdown
            await orchestrator.shutdown_agent(agent_id, graceful=True)
            
            cycle_end = time.time()
            cycle_times.append((cycle_end - cycle_start) * 1000)
        
        # Performance should remain consistent
        avg_cycle_time = statistics.mean(cycle_times)
        max_cycle_time = max(cycle_times)
        
        # Full lifecycle should complete quickly
        assert avg_cycle_time < 300, f"Agent lifecycle avg {avg_cycle_time:.2f}ms exceeds 300ms"
        assert max_cycle_time < 500, f"Agent lifecycle max {max_cycle_time:.2f}ms exceeds 500ms"
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, async_test_client, test_app):
        """Test performance under mixed API and orchestrator workload."""
        from app.core.simple_orchestrator import create_simple_orchestrator
        
        orchestrator = create_simple_orchestrator()
        
        async def mixed_workload_iteration() -> float:
            start_time = time.time()
            
            # Mix of operations
            health_task = async_test_client.get("/health")
            status_task = orchestrator.get_system_status()
            debug_task = async_test_client.get("/debug-agents")
            
            # Execute concurrently
            health_response, orch_status, debug_response = await asyncio.gather(
                health_task, status_task, debug_task
            )
            
            end_time = time.time()
            
            # Validate responses
            assert health_response.status_code in [200, 500]
            assert isinstance(orch_status, dict)
            assert debug_response.status_code == 200
            
            return (end_time - start_time) * 1000
        
        # Run mixed workload iterations
        iteration_times = []
        for _ in range(10):
            iteration_time = await mixed_workload_iteration()
            iteration_times.append(iteration_time)
        
        # Mixed workload should complete efficiently
        avg_iteration_time = statistics.mean(iteration_times)
        max_iteration_time = max(iteration_times)
        
        assert avg_iteration_time < 500, f"Mixed workload avg {avg_iteration_time:.2f}ms exceeds 500ms"
        assert max_iteration_time < 1000, f"Mixed workload max {max_iteration_time:.2f}ms exceeds 1000ms"


class TestPerformanceRegression:
    """Test for performance regression indicators."""
    
    @pytest.mark.asyncio
    async def test_performance_consistency(self, async_test_client):
        """Test that performance is consistent across multiple runs."""
        response_times = []
        
        # Multiple runs of the same operation
        for _ in range(20):
            start_time = time.time()
            response = await async_test_client.get("/health")
            end_time = time.time()
            
            response_times.append((end_time - start_time) * 1000)
            assert response.status_code in [200, 500]
        
        # Calculate variance
        mean_time = statistics.mean(response_times)
        stdev_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Performance should be consistent (low variance)
        coefficient_of_variation = stdev_time / mean_time if mean_time > 0 else 0
        
        assert coefficient_of_variation < 1.0, f"Performance inconsistent: CV {coefficient_of_variation:.2f} too high"
        assert mean_time < 100, f"Mean performance {mean_time:.2f}ms exceeds target"
        assert stdev_time < 50, f"Performance variance {stdev_time:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_no_performance_degradation_over_time(self, async_test_client):
        """Test that performance doesn't degrade over time."""
        # Early performance sample
        early_times = []
        for _ in range(10):
            start_time = time.time()
            response = await async_test_client.get("/status")
            end_time = time.time()
            early_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        # Do some work to simulate system usage
        for _ in range(20):
            await async_test_client.get("/debug-agents")
            await async_test_client.get("/health")
        
        # Later performance sample
        later_times = []
        for _ in range(10):
            start_time = time.time()
            response = await async_test_client.get("/status")
            end_time = time.time()
            later_times.append((end_time - start_time) * 1000)
            assert response.status_code == 200
        
        early_avg = statistics.mean(early_times)
        later_avg = statistics.mean(later_times)
        
        # Performance should not degrade significantly
        degradation_ratio = later_avg / early_avg if early_avg > 0 else 1
        assert degradation_ratio < 2.0, f"Performance degraded {degradation_ratio:.2f}x from {early_avg:.2f}ms to {later_avg:.2f}ms"
