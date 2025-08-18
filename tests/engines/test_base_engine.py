"""
Test Base Engine Architecture and Common Functionality
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

import sys
import os
sys.path.append('/Users/bogdan/work/leanvibe-dev/bee-hive')

from app.core.engines.base_engine import (
    BaseEngine,
    EngineConfig,
    EngineRequest,
    EngineResponse,
    EngineStatus,
    RequestPriority,
    HealthStatus,
    EngineMetrics,
    CircuitBreaker,
    CircuitBreakerState,
    EnginePlugin,
    PluginRegistry,
    PerformanceMonitor
)


class MockEngine(BaseEngine):
    """Mock engine for testing."""
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.process_delay = 0.1  # 100ms default delay
        self.fail_requests = False
    
    async def _engine_initialize(self) -> None:
        """Mock initialization."""
        await asyncio.sleep(0.01)  # Simulate init time
    
    async def _engine_process(self, request: EngineRequest) -> EngineResponse:
        """Mock request processing."""
        await asyncio.sleep(self.process_delay)
        
        if self.fail_requests:
            raise ValueError("Mock failure")
        
        return EngineResponse(
            request_id=request.request_id,
            success=True,
            result={"processed": True, "request_type": request.request_type}
        )


class MockPlugin(EnginePlugin):
    """Mock plugin for testing."""
    
    def get_name(self) -> str:
        return "mock_plugin"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: dict) -> None:
        self.config = config
    
    async def can_handle(self, request: EngineRequest) -> bool:
        return request.request_type == "mock_request"
    
    async def process(self, request: EngineRequest) -> EngineResponse:
        return EngineResponse(
            request_id=request.request_id,
            success=True,
            result={"plugin": "mock_plugin", "handled": True}
        )
    
    async def get_health(self) -> dict:
        return {"status": "healthy"}
    
    async def shutdown(self) -> None:
        pass


@pytest.fixture
def engine_config():
    """Create test engine configuration."""
    return EngineConfig(
        engine_id="test_engine",
        name="Test Engine",
        max_concurrent_requests=100,
        request_timeout_seconds=5,
        circuit_breaker_enabled=True,
        plugins_enabled=True
    )


@pytest.fixture
async def mock_engine(engine_config):
    """Create and initialize mock engine."""
    engine = MockEngine(engine_config)
    await engine.initialize()
    yield engine
    await engine.shutdown()


class TestBaseEngine:
    """Test base engine functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine_config):
        """Test engine initialization."""
        engine = MockEngine(engine_config)
        assert engine.status == EngineStatus.INITIALIZING
        
        await engine.initialize()
        assert engine.status == EngineStatus.HEALTHY
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_basic_request_processing(self, mock_engine):
        """Test basic request processing."""
        request = EngineRequest(
            request_type="test_request",
            payload={"data": "test"}
        )
        
        response = await mock_engine.process(request)
        
        assert response.success
        assert response.request_id == request.request_id
        assert response.result["processed"] is True
        assert response.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_request_timeout(self, mock_engine):
        """Test request timeout handling."""
        mock_engine.process_delay = 10  # 10 seconds
        
        request = EngineRequest(
            request_type="slow_request",
            timeout_seconds=1  # 1 second timeout
        )
        
        response = await mock_engine.process(request)
        
        assert not response.success
        assert response.error_code == "TIMEOUT"
        assert "timeout" in response.error.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_engine):
        """Test error handling in request processing."""
        mock_engine.fail_requests = True
        
        request = EngineRequest(request_type="failing_request")
        response = await mock_engine.process(request)
        
        assert not response.success
        assert response.error_code == "PROCESSING_ERROR"
        assert "Mock failure" in response.error
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, engine_config):
        """Test circuit breaker functionality."""
        # Configure circuit breaker with low threshold
        engine_config.circuit_breaker_failure_threshold = 2
        engine_config.circuit_breaker_recovery_timeout = 1
        
        engine = MockEngine(engine_config)
        await engine.initialize()
        
        try:
            # Cause failures to trip circuit breaker
            engine.fail_requests = True
            
            # First two requests should fail normally
            for _ in range(2):
                request = EngineRequest(request_type="test")
                response = await engine.process(request)
                assert not response.success
                assert response.error_code == "PROCESSING_ERROR"
            
            # Circuit breaker should now be open
            assert engine.circuit_breaker.state == CircuitBreakerState.OPEN
            
            # Next request should be rejected by circuit breaker
            request = EngineRequest(request_type="test")
            response = await engine.process(request)
            assert not response.success
            assert response.error_code == "CIRCUIT_BREAKER_OPEN"
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_engine):
        """Test health monitoring."""
        health = await mock_engine.get_health()
        
        assert isinstance(health, HealthStatus)
        assert health.status == EngineStatus.HEALTHY
        assert health.uptime_seconds > 0
        assert health.total_requests_processed >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_engine):
        """Test metrics collection."""
        # Process a few requests to generate metrics
        for i in range(5):
            request = EngineRequest(request_type=f"test_{i}")
            await mock_engine.process(request)
        
        metrics = await mock_engine.get_metrics()
        
        assert isinstance(metrics, EngineMetrics)
        assert metrics.engine_id == mock_engine.config.engine_id
        assert metrics.requests_per_second >= 0
        assert metrics.average_response_time_ms > 0
        assert metrics.success_rate_percent == 100.0
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_engine):
        """Test concurrent request handling."""
        # Create multiple concurrent requests
        requests = [
            EngineRequest(request_type=f"concurrent_{i}")
            for i in range(10)
        ]
        
        # Process all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*[
            mock_engine.process(request) for request in requests
        ])
        total_time = time.time() - start_time
        
        # All requests should succeed
        assert all(response.success for response in responses)
        
        # Concurrent execution should be faster than sequential
        # (10 requests * 0.1s delay = 1s sequential, but should be ~0.1s concurrent)
        assert total_time < 0.5  # Much less than 1 second
    
    @pytest.mark.asyncio
    async def test_priority_handling(self, mock_engine):
        """Test request priority handling."""
        high_priority_request = EngineRequest(
            request_type="high_priority",
            priority=RequestPriority.HIGH
        )
        
        normal_priority_request = EngineRequest(
            request_type="normal_priority",
            priority=RequestPriority.NORMAL
        )
        
        # Both should process successfully (priority affects scheduling, not processing)
        high_response = await mock_engine.process(high_priority_request)
        normal_response = await mock_engine.process(normal_priority_request)
        
        assert high_response.success
        assert normal_response.success


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
        
        # Initially closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.should_allow_request()
        
        # Record failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED  # Still below threshold
        
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN  # Now open
        assert not cb.should_allow_request()
        
        # Record success should close circuit
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.should_allow_request()


class TestPluginRegistry:
    """Test plugin registry functionality."""
    
    @pytest.mark.asyncio
    async def test_plugin_registration(self):
        """Test plugin registration and management."""
        registry = PluginRegistry()
        plugin = MockPlugin()
        
        await registry.register_plugin(plugin, {"setting": "value"})
        
        assert "mock_plugin" in registry.list_plugins()
        assert registry.get_plugin("mock_plugin") == plugin
    
    @pytest.mark.asyncio
    async def test_plugin_handler_discovery(self):
        """Test finding plugin handlers."""
        registry = PluginRegistry()
        plugin = MockPlugin()
        
        await registry.register_plugin(plugin)
        
        # Plugin should handle mock_request
        mock_request = EngineRequest(request_type="mock_request")
        handler = await registry.find_handler(mock_request)
        assert handler == plugin
        
        # Plugin should not handle other requests
        other_request = EngineRequest(request_type="other_request")
        handler = await registry.find_handler(other_request)
        assert handler is None
    
    @pytest.mark.asyncio
    async def test_plugin_health_monitoring(self):
        """Test plugin health monitoring."""
        registry = PluginRegistry()
        plugin = MockPlugin()
        
        await registry.register_plugin(plugin)
        
        health = await registry.get_plugins_health()
        assert "mock_plugin" in health
        assert health["mock_plugin"]["status"] == "healthy"


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        monitor = PerformanceMonitor(window_size=60)  # 1 minute window
        
        # Record some requests
        monitor.record_request(100.0, True)  # 100ms, success
        monitor.record_request(200.0, True)  # 200ms, success
        monitor.record_request(150.0, False) # 150ms, failure
        
        metrics = monitor.get_metrics()
        
        assert metrics["average_response_time_ms"] == 150.0  # (100+200+150)/3
        assert metrics["error_rate_percent"] == 33.33  # 1/3 failed (rounded)
        assert metrics["success_rate_percent"] == 66.67  # 2/3 succeeded
        assert metrics["requests_per_second"] > 0
    
    def test_empty_metrics(self):
        """Test metrics with no recorded requests."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics()
        
        assert metrics["requests_per_second"] == 0.0
        assert metrics["average_response_time_ms"] == 0.0
        assert metrics["error_rate_percent"] == 0.0
        assert metrics["success_rate_percent"] == 100.0


@pytest.mark.asyncio
async def test_engine_with_plugins(engine_config):
    """Test engine operation with plugins."""
    engine = MockEngine(engine_config)
    await engine.initialize()
    
    try:
        # Register a plugin
        plugin = MockPlugin()
        await engine.register_plugin(plugin)
        
        # Request that plugin can handle
        plugin_request = EngineRequest(request_type="mock_request")
        response = await engine.process(plugin_request)
        
        assert response.success
        assert response.result["plugin"] == "mock_plugin"
        assert response.result["handled"] is True
        
        # Request that plugin cannot handle (should go to engine)
        engine_request = EngineRequest(request_type="engine_request")
        response = await engine.process(engine_request)
        
        assert response.success
        assert response.result["processed"] is True
        
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    # Run tests manually for debugging
    import asyncio
    
    async def run_tests():
        config = EngineConfig(
            engine_id="test",
            name="Test Engine",
            max_concurrent_requests=100
        )
        
        engine = MockEngine(config)
        await engine.initialize()
        
        # Test basic functionality
        request = EngineRequest(request_type="test")
        response = await engine.process(request)
        print(f"Response: {response}")
        
        # Test health
        health = await engine.get_health()
        print(f"Health: {health}")
        
        # Test metrics
        metrics = await engine.get_metrics()
        print(f"Metrics: {metrics}")
        
        await engine.shutdown()
        print("Tests completed successfully!")
    
    asyncio.run(run_tests())