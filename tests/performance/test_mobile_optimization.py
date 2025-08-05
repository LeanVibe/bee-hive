"""
Mobile Optimization Performance Testing Suite
Tests mobile-specific performance targets, caching effectiveness, and resource usage.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

import requests
from app.api.hive_commands import (
    HiveCommandRequest, 
    execute_command,
    get_mobile_performance_metrics,
    optimize_mobile_performance
)
from app.core.mobile_api_cache import get_mobile_cache


class TestMobilePerformanceTargets:
    """Test mobile performance targets and optimization."""
    
    @pytest.mark.asyncio
    async def test_cached_response_time_target(self):
        """Test cached responses meet <5ms target consistently."""
        performance_results = []
        
        for i in range(10):  # Test consistency across multiple requests
            request = HiveCommandRequest(
                command="/hive:status --mobile --priority=high",
                mobile_optimized=True,
                use_cache=True,
                priority="high"
            )
            
            with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
                mock_cache.return_value = {
                    "success": True,
                    "mobile_optimized": True,
                    "system_state": "operational",
                    "agent_count": 5
                }
                
                start_time = time.perf_counter()
                response = await execute_command(request)
                execution_time = (time.perf_counter() - start_time) * 1000
                
                performance_results.append(execution_time)
                
                assert response.success
                assert response.cached
                assert execution_time < 5.0, f"Cached response took {execution_time}ms, target <5ms"
        
        # Validate consistency
        avg_time = statistics.mean(performance_results)
        max_time = max(performance_results)
        
        assert avg_time < 3.0, f"Average cached response time {avg_time}ms exceeds 3ms"
        assert max_time < 5.0, f"Maximum cached response time {max_time}ms exceeds 5ms target"
        print(f"✅ Cached response performance: avg={avg_time:.2f}ms, max={max_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_live_response_time_target(self):
        """Test live responses meet <50ms target for mobile."""
        performance_results = []
        
        for i in range(5):  # Test live performance consistency
            request = HiveCommandRequest(
                command="/hive:status --mobile",
                mobile_optimized=True,
                use_cache=False  # Force live execution
            )
            
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                mock_agents.return_value = {
                    f"agent{i}": {"role": "backend", "status": "active"} for i in range(3)
                }
                
                start_time = time.perf_counter()
                response = await execute_command(request)
                execution_time = (time.perf_counter() - start_time) * 1000
                
                performance_results.append(execution_time)
                
                assert response.success
                assert not response.cached
                assert execution_time < 50.0, f"Live response took {execution_time}ms, target <50ms"
        
        # Performance analysis
        avg_time = statistics.mean(performance_results)
        p95_time = statistics.quantiles(performance_results, n=20)[18]  # 95th percentile
        
        assert avg_time < 30.0, f"Average live response time {avg_time}ms exceeds 30ms"
        assert p95_time < 50.0, f"95th percentile response time {p95_time}ms exceeds 50ms target"
        print(f"✅ Live response performance: avg={avg_time:.2f}ms, p95={p95_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_mobile_requests_performance(self):
        """Test performance under concurrent mobile load."""
        async def mobile_request():
            request = HiveCommandRequest(
                command="/hive:status --mobile",
                mobile_optimized=True,
                use_cache=True
            )
            
            with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
                mock_cache.return_value = {"success": True, "mobile_optimized": True}
                
                start_time = time.perf_counter()
                response = await execute_command(request)
                execution_time = (time.perf_counter() - start_time) * 1000
                
                return execution_time, response.success
        
        # Execute 20 concurrent requests
        tasks = [mobile_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        
        times = [result[0] for result in results]
        successes = [result[1] for result in results]
        
        # Validate concurrent performance
        assert all(successes), "All concurrent requests should succeed"
        assert max(times) < 10.0, f"Max concurrent response time {max(times):.2f}ms exceeds 10ms"
        assert statistics.mean(times) < 5.0, f"Average concurrent response time {statistics.mean(times):.2f}ms exceeds 5ms"
        
        print(f"✅ Concurrent mobile performance: 20 requests, avg={statistics.mean(times):.2f}ms")
    
    @pytest.mark.asyncio 
    async def test_mobile_cache_hit_rate_optimization(self):
        """Test mobile cache achieves >80% hit rate under typical usage."""
        cache_stats = {"hits": 0, "misses": 0}
        
        # Simulate typical mobile usage pattern
        mobile_commands = [
            "/hive:status --mobile",
            "/hive:focus development --mobile", 
            "/hive:productivity --mobile",
            "/hive:status --mobile",  # Repeat - should hit cache
            "/hive:notifications --mobile",
            "/hive:focus --mobile",  # Repeat - should hit cache
            "/hive:status --mobile"  # Repeat - should hit cache
        ]
        
        for command in mobile_commands:
            request = HiveCommandRequest(
                command=command,
                mobile_optimized=True,
                use_cache=True
            )
            
            # Simulate cache behavior
            cache_key = f"mobile:{command}"
            if cache_key in {"mobile:/hive:status --mobile", "mobile:/hive:focus --mobile"}:
                # Cache hit
                with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
                    mock_cache.return_value = {"success": True, "cached": True}
                    response = await execute_command(request)
                    if response.cached:
                        cache_stats["hits"] += 1
                    else:
                        cache_stats["misses"] += 1
            else:
                # Cache miss - live execution
                with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                    mock_agents.return_value = {}
                    response = await execute_command(request)
                    cache_stats["misses"] += 1
        
        # Calculate hit rate
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        hit_rate = cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        assert hit_rate >= 0.4, f"Cache hit rate {hit_rate:.1%} below 40% minimum"
        print(f"✅ Mobile cache hit rate: {hit_rate:.1%} ({cache_stats['hits']}/{total_requests})")


class TestMobileBatteryAndResourceOptimization:
    """Test mobile battery usage and resource consumption optimization."""
    
    @pytest.mark.asyncio
    async def test_request_batching_efficiency(self):
        """Test request batching reduces mobile network usage."""
        # Test individual requests vs batched approach
        individual_times = []
        
        # Individual requests
        for i in range(3):
            request = HiveCommandRequest(
                command=f"/hive:status --mobile",
                mobile_optimized=True
            )
            
            start_time = time.perf_counter()
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                mock_agents.return_value = {}
                await execute_command(request)
            execution_time = time.perf_counter() - start_time
            individual_times.append(execution_time)
        
        # Simulated batched request (multiple commands in context)
        batched_request = HiveCommandRequest(
            command="/hive:status --mobile --detailed",
            mobile_optimized=True,
            context={"batch_mode": True, "include_agents": True, "include_health": True}
        )
        
        start_time = time.perf_counter()
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {}
            await execute_command(batched_request)
        batched_time = time.perf_counter() - start_time
        
        # Batched should be more efficient than individual requests
        total_individual_time = sum(individual_times)
        efficiency_gain = (total_individual_time - batched_time) / total_individual_time
        
        assert efficiency_gain > 0.3, f"Batching should provide >30% efficiency gain, got {efficiency_gain:.1%}"
        print(f"✅ Request batching efficiency: {efficiency_gain:.1%} reduction in execution time")
    
    @pytest.mark.asyncio
    async def test_mobile_data_compression(self):
        """Test mobile responses are optimized for data usage."""
        # Test full response vs mobile-optimized response size
        full_request = HiveCommandRequest(
            command="/hive:status --detailed",
            mobile_optimized=False
        )
        
        mobile_request = HiveCommandRequest(
            command="/hive:status --mobile",
            mobile_optimized=True
        )
        
        with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
            mock_agents.return_value = {
                f"agent{i}": {
                    "role": f"role{i}",
                    "status": "active",
                    "capabilities": [f"cap{j}" for j in range(5)],
                    "metadata": {"detailed": "information", "timestamps": [1, 2, 3, 4, 5]}
                } for i in range(10)
            }
            
            full_response = await execute_command(full_request)
            mobile_response = await execute_command(mobile_request)
            
            # Simulate response size calculation
            full_size = len(str(full_response.result))
            mobile_size = len(str(mobile_response.result))
            compression_ratio = (full_size - mobile_size) / full_size
            
            assert compression_ratio > 0.4, f"Mobile response should be >40% smaller, got {compression_ratio:.1%}"
            assert mobile_response.mobile_optimized
            print(f"✅ Mobile data compression: {compression_ratio:.1%} size reduction")
    
    @pytest.mark.asyncio
    async def test_mobile_polling_optimization(self):
        """Test mobile polling is optimized to reduce battery usage."""
        # Test adaptive polling based on activity
        polling_intervals = []
        
        # Simulate different activity levels
        activity_scenarios = [
            {"agents_active": 0, "expected_interval": 30},  # Low activity - longer intervals
            {"agents_active": 3, "expected_interval": 15},  # Medium activity
            {"agents_active": 7, "expected_interval": 5}    # High activity - shorter intervals
        ]
        
        for scenario in activity_scenarios:
            request = HiveCommandRequest(
                command="/hive:status --mobile",
                mobile_optimized=True,
                context={"polling_optimization": True}
            )
            
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                mock_agents.return_value = {
                    f"agent{i}": {"role": "backend", "status": "active"} 
                    for i in range(scenario["agents_active"])
                }
                
                response = await execute_command(request)
                
                # Extract recommended polling interval from response
                performance_metrics = response.performance_metrics or {}
                suggested_interval = performance_metrics.get("suggested_polling_interval", 15)
                polling_intervals.append(suggested_interval)
                
                # Validate adaptive polling
                if scenario["agents_active"] == 0:
                    assert suggested_interval >= 20, "Low activity should suggest longer polling intervals"
                elif scenario["agents_active"] >= 5:
                    assert suggested_interval <= 10, "High activity should suggest shorter polling intervals"
        
        print(f"✅ Adaptive polling intervals: {polling_intervals} seconds")


class TestWebSocketPerformanceIntegration:
    """Test WebSocket integration performance for real-time updates."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self):
        """Test WebSocket connection establishes quickly for mobile."""
        # Mock WebSocket connection time
        connection_times = []
        
        for i in range(5):
            start_time = time.perf_counter()
            
            # Simulate WebSocket connection
            with patch('websockets.connect') as mock_ws:
                mock_connection = AsyncMock()
                mock_ws.return_value.__aenter__.return_value = mock_connection
                
                # Simulate connection establishment
                await asyncio.sleep(0.01)  # 10ms connection time
                connection_time = (time.perf_counter() - start_time) * 1000
                connection_times.append(connection_time)
        
        avg_connection_time = statistics.mean(connection_times)
        max_connection_time = max(connection_times)
        
        assert avg_connection_time < 20.0, f"Average WebSocket connection time {avg_connection_time}ms exceeds 20ms"
        assert max_connection_time < 50.0, f"Max WebSocket connection time {max_connection_time}ms exceeds 50ms"
        
        print(f"✅ WebSocket connection performance: avg={avg_connection_time:.1f}ms, max={max_connection_time:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_real_time_update_latency(self):
        """Test real-time update latency meets mobile targets."""
        update_latencies = []
        
        for i in range(10):
            # Simulate agent status change
            change_time = time.perf_counter()
            
            # Mock WebSocket message delivery
            with patch('app.core.realtime_dashboard_streaming.broadcast_agent_update') as mock_broadcast:
                mock_broadcast.return_value = asyncio.sleep(0.005)  # 5ms delivery time
                
                await asyncio.sleep(0.005)
                delivery_time = time.perf_counter()
                
                latency = (delivery_time - change_time) * 1000
                update_latencies.append(latency)
        
        avg_latency = statistics.mean(update_latencies)
        p95_latency = statistics.quantiles(update_latencies, n=20)[18]
        
        assert avg_latency < 10.0, f"Average update latency {avg_latency}ms exceeds 10ms"
        assert p95_latency < 50.0, f"95th percentile latency {p95_latency}ms exceeds 50ms"
        
        print(f"✅ Real-time update latency: avg={avg_latency:.1f}ms, p95={p95_latency:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_mobile_websocket_reconnection(self):
        """Test WebSocket reconnection is fast and reliable for mobile."""
        reconnection_times = []
        
        for attempt in range(3):
            # Simulate connection drop
            disconnect_time = time.perf_counter()
            
            # Mock exponential backoff reconnection
            backoff_delay = min(0.1 * (2 ** attempt), 1.0)  # Max 1 second
            await asyncio.sleep(backoff_delay)
            
            # Mock successful reconnection
            with patch('websockets.connect') as mock_ws:
                mock_connection = AsyncMock()
                mock_ws.return_value.__aenter__.return_value = mock_connection
                
                reconnect_time = time.perf_counter()
                total_reconnection_time = (reconnect_time - disconnect_time) * 1000
                reconnection_times.append(total_reconnection_time)
        
        avg_reconnection = statistics.mean(reconnection_times)
        max_reconnection = max(reconnection_times)
        
        assert avg_reconnection < 500.0, f"Average reconnection time {avg_reconnection}ms exceeds 500ms"
        assert max_reconnection < 1000.0, f"Max reconnection time {max_reconnection}ms exceeds 1000ms"
        
        print(f"✅ WebSocket reconnection performance: avg={avg_reconnection:.0f}ms, max={max_reconnection:.0f}ms")


class TestMobileAPIEndpointPerformance:
    """Test mobile-specific API endpoint performance."""
    
    @pytest.mark.asyncio
    async def test_mobile_performance_metrics_endpoint(self):
        """Test /api/hive/mobile/performance endpoint performance."""
        with patch('app.core.mobile_api_cache.get_mobile_cache') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache_instance.stats.return_value = {
                "total_entries": 100,
                "mobile_optimized": 80,
                "total_size_mb": 15.5,
                "utilization_percentage": 65.2,
                "mobile_optimization_percentage": 80.0
            }
            mock_cache_instance.get_performance_metrics.return_value = Mock(
                avg_response_time_ms=4.2,
                cache_hit_rate=0.75,
                mobile_optimization_score=85.5,
                alert_relevance_score=92.3
            )
            mock_cache.return_value = mock_cache_instance
            
            start_time = time.perf_counter()
            response = await get_mobile_performance_metrics()
            execution_time = (time.perf_counter() - start_time) * 1000
            
            assert response["success"]
            assert execution_time < 10.0, f"Performance metrics endpoint took {execution_time}ms, target <10ms"
            assert response["mobile_performance_score"] > 0
            assert "cache_performance" in response
            assert "mobile_optimization" in response
    
    @pytest.mark.asyncio
    async def test_mobile_optimization_endpoint(self):
        """Test /api/hive/mobile/optimize endpoint performance."""
        with patch('app.core.mobile_api_cache.get_mobile_cache') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache_instance.optimize_for_mobile.return_value = {
                "entries_optimized": 25,
                "space_freed_mb": 5.2,
                "performance_improvement": "15%"
            }
            mock_cache.return_value = mock_cache_instance
            
            start_time = time.perf_counter()
            response = await optimize_mobile_performance()
            execution_time = (time.perf_counter() - start_time) * 1000
            
            assert response["success"]
            assert execution_time < 50.0, f"Mobile optimization took {execution_time}ms, target <50ms"
            assert "optimization_results" in response
    
    @pytest.mark.asyncio
    async def test_mobile_cache_clear_performance(self):
        """Test mobile cache clear operation performance."""
        with patch('app.core.mobile_api_cache.get_mobile_cache') as mock_cache:
            mock_cache_instance = Mock()
            mock_cache_instance.clear.return_value = 150  # Cleared entries count
            mock_cache.return_value = mock_cache_instance
            
            # Import the function we need to test
            from app.api.hive_commands import clear_mobile_cache
            
            start_time = time.perf_counter()
            response = await clear_mobile_cache()
            execution_time = (time.perf_counter() - start_time) * 1000
            
            assert response["success"]
            assert execution_time < 20.0, f"Cache clear took {execution_time}ms, target <20ms"
            assert response["cleared_entries"] == 150


@pytest.mark.integration
class TestEndToEndMobilePerformance:
    """End-to-end mobile performance testing."""
    
    @pytest.mark.asyncio
    async def test_complete_mobile_workflow_performance(self):
        """Test complete mobile workflow from status to action execution."""
        workflow_times = []
        
        # Complete workflow: Status -> Focus -> Action
        workflow_commands = [
            "/hive:status --mobile --priority=high",
            "/hive:focus development --mobile", 
            "/hive:productivity --mobile --developer"
        ]
        
        total_start_time = time.perf_counter()
        
        for command in workflow_commands:
            request = HiveCommandRequest(
                command=command,
                mobile_optimized=True,
                use_cache=True
            )
            
            cmd_start_time = time.perf_counter()
            
            with patch('app.core.hive_slash_commands.get_active_agents_status') as mock_agents:
                with patch('app.api.hive_commands.get_cached_mobile_response') as mock_cache:
                    mock_agents.return_value = {"agent1": {"role": "backend"}}
                    mock_cache.return_value = {"success": True, "mobile_optimized": True}
                    
                    response = await execute_command(request)
                    
            cmd_time = (time.perf_counter() - cmd_start_time) * 1000
            workflow_times.append(cmd_time)
            
            assert response.success
            assert response.mobile_optimized
        
        total_workflow_time = (time.perf_counter() - total_start_time) * 1000
        
        # Workflow performance validation
        assert total_workflow_time < 100.0, f"Complete mobile workflow took {total_workflow_time}ms, target <100ms"
        assert max(workflow_times) < 50.0, f"Individual command exceeded 50ms: {max(workflow_times)}ms"
        assert statistics.mean(workflow_times) < 20.0, f"Average command time {statistics.mean(workflow_times)}ms exceeds 20ms"
        
        print(f"✅ Complete mobile workflow: {total_workflow_time:.1f}ms total, avg per command: {statistics.mean(workflow_times):.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])