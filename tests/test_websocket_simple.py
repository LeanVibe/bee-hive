"""
Simple WebSocket integration tests for Real-Time Monitoring Dashboard

Basic tests to validate WebSocket functionality without complex fixtures.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from app.api.v1.websocket import connection_manager
from app.core.performance_metrics_publisher import PerformanceMetricsPublisher


class TestWebSocketBasic:
    """Basic WebSocket functionality tests."""

    @pytest.mark.asyncio
    async def test_connection_manager_initialization(self):
        """Test that connection manager initializes correctly."""
        assert connection_manager is not None
        assert hasattr(connection_manager, 'observability_connections')
        assert hasattr(connection_manager, 'agent_connections')

    @pytest.mark.asyncio
    async def test_connection_manager_stats(self):
        """Test connection manager statistics."""
        stats = connection_manager.get_connection_stats()
        
        assert isinstance(stats, dict)
        assert "observability_connections" in stats
        assert "agent_connections" in stats
        assert "total_connections" in stats
        
        # Should be integers and non-negative
        assert isinstance(stats["observability_connections"], int)
        assert isinstance(stats["total_connections"], int)
        assert stats["observability_connections"] >= 0
        assert stats["total_connections"] >= 0

    @pytest.mark.asyncio
    async def test_broadcast_event(self):
        """Test event broadcasting without connections."""
        test_event = {
            "event_type": "test_event",
            "agent_id": "test-agent-123",
            "timestamp": "2025-01-28T10:00:00",
            "payload": {"test": "data"}
        }
        
        # Should not raise exception even with no connections
        await connection_manager.broadcast_event(test_event)

    @pytest.mark.asyncio
    async def test_broadcast_agent_lifecycle_event(self):
        """Test agent lifecycle event broadcasting."""
        test_event = {
            "event_type": "agent_registered",
            "agent_id": "test-agent-456",
            "timestamp": "2025-01-28T10:00:00",
            "payload": {"name": "test_agent"}
        }
        
        # Should not raise exception even with no connections
        await connection_manager.broadcast_agent_lifecycle_event(test_event)

    @pytest.mark.asyncio
    async def test_broadcast_performance_update(self):
        """Test performance update broadcasting."""
        test_metrics = {
            "timestamp": "2025-01-28T10:00:00",
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 68.3,
            "active_connections": 5
        }
        
        # Should not raise exception even with no connections
        await connection_manager.broadcast_performance_update(test_metrics)

    @pytest.mark.asyncio
    async def test_broadcast_hook_event(self):
        """Test hook event broadcasting."""
        test_hook = {
            "hook_type": "PreToolUse",
            "agent_id": "test-agent-789",
            "timestamp": "2025-01-28T10:00:00",
            "payload": {"tool": "test_tool"}
        }
        
        # Should not raise exception even with no connections
        await connection_manager.broadcast_hook_event(test_hook)


class TestPerformanceMetricsPublisher:
    """Test performance metrics publisher functionality."""

    @pytest.mark.asyncio
    async def test_publisher_initialization(self):
        """Test performance metrics publisher initialization."""
        publisher = PerformanceMetricsPublisher()
        
        assert publisher is not None
        assert publisher.is_running is False
        assert publisher.collection_interval == 5.0
        assert publisher.metrics_stream == "performance_metrics"

    @pytest.mark.asyncio
    async def test_publisher_start_stop(self):
        """Test starting and stopping the publisher."""
        publisher = PerformanceMetricsPublisher()
        
        # Mock Redis client
        with patch('app.core.performance_metrics_publisher.get_redis') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            # Start publisher
            await publisher.start()
            assert publisher.is_running is True
            
            # Stop publisher
            await publisher.stop()
            assert publisher.is_running is False

    @pytest.mark.asyncio
    async def test_custom_metric_publishing(self):
        """Test publishing custom metrics."""
        publisher = PerformanceMetricsPublisher()
        
        # Mock Redis client
        mock_redis = AsyncMock()
        publisher.redis = mock_redis
        
        # Publish custom metric
        await publisher.publish_custom_metric(
            "test_metric",
            95.5,
            {"component": "test"}
        )
        
        # Verify Redis operations were called
        assert mock_redis.xadd.called
        assert mock_redis.publish.called

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection methods."""
        publisher = PerformanceMetricsPublisher()
        
        # Test system metrics collection
        with patch('psutil.cpu_percent', return_value=45.2), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock memory object
            mock_memory.return_value.total = 8589934592  # 8GB
            mock_memory.return_value.available = 4294967296  # 4GB
            mock_memory.return_value.used = 4294967296  # 4GB
            mock_memory.return_value.percent = 50.0
            
            # Mock disk object
            mock_disk.return_value.total = 1000000000000  # 1TB
            mock_disk.return_value.used = 500000000000   # 500GB
            mock_disk.return_value.free = 500000000000   # 500GB
            
            # Collect system metrics
            metrics = await publisher._collect_system_metrics()
            
            assert "cpu" in metrics
            assert "memory" in metrics
            assert "disk" in metrics
            assert metrics["cpu"]["usage_percent"] == 45.2
            assert metrics["memory"]["usage_percent"] == 50.0


class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""

    @pytest.mark.asyncio
    async def test_websocket_stats_endpoint(self, async_test_client):
        """Test WebSocket stats endpoint."""
        response = await async_test_client.get("/api/v1/ws/websocket/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "stats" in data
        assert "timestamp" in data
        
        stats = data["stats"]
        assert "observability_connections" in stats
        assert "total_connections" in stats
        assert isinstance(stats["observability_connections"], int)
        assert isinstance(stats["total_connections"], int)

    @pytest.mark.asyncio
    async def test_complete_metrics_flow(self):
        """Test complete metrics publishing and broadcasting flow."""
        publisher = PerformanceMetricsPublisher()
        
        # Mock Redis for publisher
        mock_redis = AsyncMock()
        publisher.redis = mock_redis
        
        # Mock system metrics
        with patch('psutil.cpu_percent', return_value=35.7), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.total = 8589934592
            mock_memory.return_value.available = 5368709120
            mock_memory.return_value.used = 3221225472
            mock_memory.return_value.percent = 37.5
            
            # Collect and publish metrics
            metrics = await publisher._collect_all_metrics()
            await publisher._publish_metrics(metrics)
            
            # Verify metrics structure
            assert "timestamp" in metrics
            assert "system" in metrics
            assert "cpu" in metrics["system"]
            assert "memory" in metrics["system"]
            
            # Verify Redis operations
            assert mock_redis.xadd.called
            assert mock_redis.publish.called

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in WebSocket components."""
        publisher = PerformanceMetricsPublisher()
        
        # Test with no Redis client
        publisher.redis = None
        
        # Should not raise exception
        await publisher._publish_metrics({
            "timestamp": "2025-01-28T10:00:00",
            "system": {"cpu": {"usage_percent": 50.0}}
        })
        
        # Should not raise exception
        await publisher.publish_custom_metric("test", 100.0)


# Simple test to verify the basic structure works
def test_websocket_module_import():
    """Test that WebSocket modules can be imported."""
    from app.api.v1.websocket import connection_manager, router
    from app.core.performance_metrics_publisher import PerformanceMetricsPublisher
    
    assert connection_manager is not None
    assert router is not None
    assert PerformanceMetricsPublisher is not None


def test_basic_functionality():
    """Test basic functionality without async."""
    from app.api.v1.websocket import connection_manager
    
    # Test basic properties
    stats = connection_manager.get_connection_stats()
    assert isinstance(stats, dict)
    assert stats["observability_connections"] >= 0