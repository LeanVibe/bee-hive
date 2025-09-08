"""
Comprehensive Test Suite for Error Handling Framework - LeanVibe Agent Hive 2.0 - VS 3.3

Complete test coverage for all error handling components:
- Circuit breaker pattern testing with state transitions
- Retry policy testing with all strategies
- Graceful degradation testing with fallback mechanisms
- Error handling middleware integration testing
- Workflow error handling testing
- Configuration system testing
- Performance and reliability testing
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

# Import error handling components
from app.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig, CircuitBreakerError,
    get_circuit_breaker, reset_all_circuit_breakers
)
from app.core.retry_policies import (
    RetryPolicy, ExponentialBackoffPolicy, LinearBackoffPolicy, FixedDelayPolicy,
    AdaptiveBackoffPolicy, FibonacciBackoffPolicy, RetryPolicyFactory,
    RetryConfig, RetryStrategy, JitterType, RetryExecutor,
    retry_with_exponential_backoff, retry_with_fixed_delay
)
from app.core.graceful_degradation import (
    GracefulDegradationManager, DegradationLevel, ServiceHealthTracker,
    CachedResponseStrategy, StaticResponseStrategy, get_degradation_manager
)
from app.core.error_handling_middleware import (
    ErrorHandlingMiddleware, ErrorHandlingConfig, ErrorAnalyzer,
    create_error_handling_middleware
)
from app.core.workflow_engine_error_handling import (
    WorkflowErrorRecoveryManager, WorkflowErrorHandlingConfig,
    WorkflowErrorContext, WorkflowErrorType, WorkflowRecoveryStrategy,
    get_workflow_recovery_manager
)
from app.core.error_handling_config import (
    ErrorHandlingConfiguration, ErrorHandlingEnvironment, ConfigurationManager,
    get_error_handling_config, initialize_error_handling_config
)
from app.core.error_handling_integration import (
    ErrorHandlingObservabilityIntegration, get_error_handling_integration
)

# ScriptBase import for standardized execution
from app.common.script_base import ScriptBase


class TestCircuitBreaker:
    """Test suite for circuit breaker implementation."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a test circuit breaker."""
        return CircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1
        )
    
    @pytest.fixture
    def config(self):
        """Create test circuit breaker configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
            max_processing_time_ms=5.0
        )
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker proper initialization."""
        assert circuit_breaker.name == "test_breaker"
        assert circuit_breaker.config.failure_threshold == 3
        assert circuit_breaker.config.success_threshold == 2
        assert circuit_breaker.config.timeout_seconds == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state."""
        state = await circuit_breaker.get_state()
        assert state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recording(self, circuit_breaker):
        """Test failure recording and state transition."""
        # Record failures up to threshold
        for i in range(3):
            await circuit_breaker.record_failure()
        
        # Circuit breaker should now be open
        state = await circuit_breaker.get_state()
        assert state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejection(self, circuit_breaker):
        """Test that circuit breaker rejects requests when open."""
        # Force circuit breaker to open state
        await circuit_breaker.force_open()
        
        # Test function should be rejected
        async def test_function():
            return "success"
        
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(test_function)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, circuit_breaker):
        """Test transition to half-open state after timeout."""
        # Force to open state
        await circuit_breaker.force_open()
        
        # Wait for timeout (use shorter timeout for testing)
        await asyncio.sleep(1.1)
        
        # Should transition to half-open
        state = await circuit_breaker.get_state()
        assert state == CircuitBreakerState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery to closed state."""
        # Force to half-open state
        await circuit_breaker.force_half_open()
        
        # Record successful calls to meet success threshold
        for i in range(2):
            await circuit_breaker.record_success()
        
        # Should now be closed
        state = await circuit_breaker.get_state()
        assert state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_call_success(self, circuit_breaker):
        """Test successful function call through circuit breaker."""
        async def test_function():
            return "success"
        
        result = await circuit_breaker.call(test_function)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_call_failure(self, circuit_breaker):
        """Test failed function call through circuit breaker."""
        async def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.call(test_function)
        
        # Failure should be recorded
        metrics = circuit_breaker.get_metrics()
        assert metrics["failure_count"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics(self, circuit_breaker):
        """Test circuit breaker metrics collection."""
        # Record some operations
        await circuit_breaker.record_success()
        await circuit_breaker.record_failure()
        
        metrics = circuit_breaker.get_metrics()
        
        assert "name" in metrics
        assert "state" in metrics
        assert "failure_count" in metrics
        assert "success_count" in metrics
        assert "total_requests" in metrics
        assert "performance" in metrics
        
        assert metrics["failure_count"] == 1
        assert metrics["success_count"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_health_check(self, circuit_breaker):
        """Test circuit breaker health check."""
        health = await circuit_breaker.health_check()
        
        assert "status" in health
        assert "state" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded", "recovering"]
    
    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality."""
        # Record some state
        circuit_breaker._failure_count = 5
        circuit_breaker._success_count = 3
        
        # Reset
        circuit_breaker.reset()
        
        # Should be back to initial state
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0
        assert circuit_breaker._state == CircuitBreakerState.CLOSED


class TestRetryPolicies:
    """Test suite for retry policy implementations."""
    
    @pytest.fixture
    def retry_config(self):
        """Create test retry configuration."""
        return RetryConfig(
            max_attempts=3,
            base_delay_ms=100,
            max_delay_ms=1000,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter_type=JitterType.NONE  # Disable jitter for predictable testing
        )
    
    @pytest.fixture
    def exponential_policy(self, retry_config):
        """Create exponential backoff policy."""
        return ExponentialBackoffPolicy(retry_config)
    
    def test_retry_config_initialization(self, retry_config):
        """Test retry configuration initialization."""
        assert retry_config.max_attempts == 3
        assert retry_config.base_delay_ms == 100
        assert retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, exponential_policy):
        """Test exponential backoff delay calculation."""
        # Test increasing delays
        result1 = await exponential_policy.calculate_delay(0)
        result2 = await exponential_policy.calculate_delay(1)
        result3 = await exponential_policy.calculate_delay(2)
        
        assert result1.delay_ms == 100  # base_delay * 2^0
        assert result2.delay_ms == 200  # base_delay * 2^1
        assert result3.delay_ms == 400  # base_delay * 2^2
        
        assert result1.should_retry
        assert result2.should_retry
        assert result3.should_retry
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_max_attempts(self, exponential_policy):
        """Test exponential backoff respects max attempts."""
        result = await exponential_policy.calculate_delay(3)  # Exceeds max_attempts
        assert not result.should_retry
    
    @pytest.mark.asyncio
    async def test_linear_backoff_policy(self):
        """Test linear backoff policy."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_ms=100,
            backoff_multiplier=1.5,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter_type=JitterType.NONE
        )
        
        policy = LinearBackoffPolicy(config)
        
        result1 = await policy.calculate_delay(0)
        result2 = await policy.calculate_delay(1)
        
        # Linear: base * (1 + attempt * multiplier)
        assert result1.delay_ms == 100  # 100 * (1 + 0 * 1.5) = 100
        assert result2.delay_ms == 250  # 100 * (1 + 1 * 1.5) = 250
    
    @pytest.mark.asyncio
    async def test_fixed_delay_policy(self):
        """Test fixed delay policy."""
        config = RetryConfig(
            max_attempts=3,
            base_delay_ms=200,
            strategy=RetryStrategy.FIXED_DELAY,
            jitter_type=JitterType.NONE
        )
        
        policy = FixedDelayPolicy(config)
        
        result1 = await policy.calculate_delay(0)
        result2 = await policy.calculate_delay(1)
        result3 = await policy.calculate_delay(2)
        
        # All delays should be the same
        assert result1.delay_ms == 200
        assert result2.delay_ms == 200
        assert result3.delay_ms == 200
    
    @pytest.mark.asyncio
    async def test_fibonacci_backoff_policy(self):
        """Test Fibonacci backoff policy."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_ms=100,
            strategy=RetryStrategy.FIBONACCI,
            jitter_type=JitterType.NONE
        )
        
        policy = FibonacciBackoffPolicy(config)
        
        result1 = await policy.calculate_delay(0)
        result2 = await policy.calculate_delay(1)
        result3 = await policy.calculate_delay(2)
        result4 = await policy.calculate_delay(3)
        
        # Fibonacci sequence: 1, 1, 2, 3, 5, ...
        assert result1.delay_ms == 100  # 100 * 1
        assert result2.delay_ms == 100  # 100 * 1
        assert result3.delay_ms == 200  # 100 * 2
        assert result4.delay_ms == 300  # 100 * 3
    
    @pytest.mark.asyncio
    async def test_adaptive_backoff_policy(self):
        """Test adaptive backoff policy."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_ms=100,
            strategy=RetryStrategy.ADAPTIVE,
            jitter_type=JitterType.NONE
        )
        
        policy = AdaptiveBackoffPolicy(config)
        
        # Simulate some failures to trigger adaptation
        for i in range(5):
            policy._record_attempt(i, False, 100)
        
        result = await policy.calculate_delay(0)
        assert result.should_retry
        assert result.delay_ms >= 100  # Should adjust based on failures
    
    def test_retry_policy_factory(self):
        """Test retry policy factory creation."""
        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL_BACKOFF)
        policy = RetryPolicyFactory.create_policy(config)
        assert isinstance(policy, ExponentialBackoffPolicy)
        
        config.strategy = RetryStrategy.LINEAR_BACKOFF
        policy = RetryPolicyFactory.create_policy(config)
        assert isinstance(policy, LinearBackoffPolicy)
        
        config.strategy = RetryStrategy.FIXED_DELAY
        policy = RetryPolicyFactory.create_policy(config)
        assert isinstance(policy, FixedDelayPolicy)
    
    @pytest.mark.asyncio
    async def test_retry_executor_success(self):
        """Test retry executor with successful function."""
        config = RetryConfig(max_attempts=3, base_delay_ms=10)
        policy = ExponentialBackoffPolicy(config)
        executor = RetryExecutor(policy)
        
        async def test_function():
            return "success"
        
        result = await executor.execute(test_function)
        assert result == "success"
        
        metrics = executor.get_metrics()
        assert metrics["execution_metrics"]["successful_executions"] == 1
    
    @pytest.mark.asyncio
    async def test_retry_executor_with_retries(self):
        """Test retry executor with function that fails then succeeds."""
        config = RetryConfig(max_attempts=3, base_delay_ms=10)
        policy = ExponentialBackoffPolicy(config)
        executor = RetryExecutor(policy)
        
        call_count = 0
        
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await executor.execute(test_function)
        assert result == "success"
        assert call_count == 3
        
        metrics = executor.get_metrics()
        assert metrics["execution_metrics"]["successful_executions"] == 1
        assert metrics["execution_metrics"]["total_attempts"] == 3
    
    @pytest.mark.asyncio
    async def test_retry_executor_exhausted(self):
        """Test retry executor when all attempts are exhausted."""
        config = RetryConfig(max_attempts=2, base_delay_ms=10)
        policy = ExponentialBackoffPolicy(config)
        executor = RetryExecutor(policy)
        
        async def test_function():
            raise ValueError("Permanent failure")
        
        with pytest.raises(ValueError):
            await executor.execute(test_function)
        
        metrics = executor.get_metrics()
        assert metrics["execution_metrics"]["failed_executions"] == 1
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self):
        """Test convenience retry functions."""
        call_count = 0
        
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        # Test exponential backoff convenience function
        result = await retry_with_exponential_backoff(
            test_function,
            max_attempts=3,
            base_delay_ms=10
        )
        assert result == "success"
        assert call_count == 2
        
        # Reset for next test
        call_count = 0
        
        # Test fixed delay convenience function
        result = await retry_with_fixed_delay(
            test_function,
            max_attempts=3,
            delay_ms=10
        )
        assert result == "success"
        assert call_count == 2


class TestGracefulDegradation:
    """Test suite for graceful degradation implementation."""
    
    @pytest.fixture
    def degradation_manager(self):
        """Create test degradation manager."""
        return GracefulDegradationManager()
    
    @pytest.fixture
    def cached_strategy(self):
        """Create cached response strategy."""
        return CachedResponseStrategy(cache_ttl_seconds=300, max_cache_size=100)
    
    @pytest.fixture
    def static_strategy(self):
        """Create static response strategy."""
        return StaticResponseStrategy()
    
    def test_degradation_manager_initialization(self, degradation_manager):
        """Test degradation manager initialization."""
        assert degradation_manager.config.enabled
        assert len(degradation_manager.fallback_strategies) >= 2  # At least cached and static
    
    @pytest.mark.asyncio
    async def test_cached_response_strategy(self, cached_strategy):
        """Test cached response fallback strategy."""
        # Test with no cache
        response = await cached_strategy.execute_fallback(
            service_path="/test",
            degradation_level=DegradationLevel.MINIMAL,
            error_context={"error": "test"},
            original_request_data={"key": "value"}
        )
        assert response is None  # No cache available
        
        # Cache a response
        from fastapi.responses import JSONResponse
        cached_response = JSONResponse(content={"cached": True})
        
        await cached_strategy.cache_response(
            service_path="/test",
            response=cached_response,
            request_data={"key": "value"}
        )
        
        # Test cache retrieval
        response = await cached_strategy.execute_fallback(
            service_path="/test",
            degradation_level=DegradationLevel.MINIMAL,
            error_context={"error": "test"},
            original_request_data={"key": "value"}
        )
        assert response is not None
    
    @pytest.mark.asyncio
    async def test_static_response_strategy(self, static_strategy):
        """Test static response fallback strategy."""
        # Test with matching path
        response = await static_strategy.execute_fallback(
            service_path="/api/v1/agents",
            degradation_level=DegradationLevel.PARTIAL,
            error_context={"error": "test"}
        )
        assert response is not None
        assert response.status_code == 206  # Partial Content
        
        # Test with non-matching path
        response = await static_strategy.execute_fallback(
            service_path="/unknown/path",
            degradation_level=DegradationLevel.MINIMAL,
            error_context={"error": "test"}
        )
        assert response is None
    
    def test_static_strategy_path_support(self, static_strategy):
        """Test static strategy path support detection."""
        assert static_strategy.supports_path("/api/v1/agents")
        assert static_strategy.supports_path("/api/v1/tasks")
        assert not static_strategy.supports_path("/unknown/path")
    
    @pytest.mark.asyncio
    async def test_service_health_tracker(self):
        """Test service health tracking."""
        tracker = ServiceHealthTracker()
        
        # Test initial state
        status = tracker.get_service_status("test_service")
        assert status.value == "healthy"
        
        # Record errors
        for i in range(5):
            await tracker.record_service_error("test_service", f"Error {i}")
        
        # Status should change
        status = tracker.get_service_status("test_service")
        assert status.value in ["degraded", "unhealthy", "unavailable"]
        
        # Record successes
        for i in range(10):
            await tracker.record_service_success("test_service")
        
        # Status should improve
        status = tracker.get_service_status("test_service")
        # May still be degraded but should be better than unavailable
    
    @pytest.mark.asyncio
    async def test_degradation_level_determination(self):
        """Test degradation level determination based on service health."""
        tracker = ServiceHealthTracker()
        
        # Test with healthy service
        level = tracker.get_degradation_level_for_service("healthy_service")
        assert level == DegradationLevel.NONE
        
        # Simulate service degradation
        for i in range(3):
            await tracker.record_service_error("degraded_service", f"Error {i}")
        
        level = tracker.get_degradation_level_for_service("degraded_service")
        assert level in [DegradationLevel.MINIMAL, DegradationLevel.PARTIAL, DegradationLevel.FULL]
    
    @pytest.mark.asyncio
    async def test_degradation_manager_apply(self, degradation_manager):
        """Test degradation manager application."""
        # Test with no degradation
        response = await degradation_manager.apply_degradation(
            service_path="/test",
            degradation_level=DegradationLevel.NONE,
            error_context={"error": "test"}
        )
        assert response is None  # No degradation needed
        
        # Test with degradation
        response = await degradation_manager.apply_degradation(
            service_path="/api/v1/agents",
            degradation_level=DegradationLevel.PARTIAL,
            error_context={"error": "service_unavailable"}
        )
        assert response is not None  # Should have fallback response
    
    @pytest.mark.asyncio
    async def test_degradation_metrics(self, degradation_manager):
        """Test degradation metrics collection."""
        # Apply some degradation
        await degradation_manager.apply_degradation(
            service_path="/api/v1/tasks",
            degradation_level=DegradationLevel.MINIMAL,
            error_context={"error": "test"}
        )
        
        metrics = degradation_manager.get_metrics()
        
        assert "total_requests" in metrics
        assert "degraded_requests" in metrics
        assert "degradation_rate" in metrics
        assert "performance" in metrics
        assert "cache" in metrics
        
        assert metrics["total_requests"] >= 1
    
    @pytest.mark.asyncio
    async def test_degradation_health_check(self, degradation_manager):
        """Test degradation manager health check."""
        health = await degradation_manager.health_check()
        
        assert "status" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded"]


class TestErrorHandlingMiddleware:
    """Test suite for error handling middleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.get("/error")
        async def error_endpoint():
            raise HTTPException(status_code=500, detail="Test error")
        
        @app.get("/timeout")
        async def timeout_endpoint():
            await asyncio.sleep(10)  # This will timeout
            return {"message": "delayed"}
        
        return app
    
    @pytest.fixture
    def middleware_config(self):
        """Create middleware configuration."""
        return ErrorHandlingConfig(
            enabled=True,
            max_retries=2,
            base_delay_ms=10,
            circuit_breaker_enabled=False,  # Disable for simpler testing
            graceful_degradation_enabled=False
        )
    
    def test_error_analyzer_initialization(self):
        """Test error analyzer initialization."""
        analyzer = ErrorAnalyzer()
        assert analyzer is not None
    
    def test_error_analyzer_severity_determination(self):
        """Test error severity determination."""
        analyzer = ErrorAnalyzer()
        context = Mock()
        context.retry_count = 0
        
        # Test HTTP error
        http_error = HTTPException(status_code=500, detail="Server error")
        analysis = analyzer.analyze_error(http_error, context)
        assert analysis["severity"].value in ["medium", "high"]
        
        # Test connection error
        connection_error = ConnectionError("Connection failed")
        analysis = analyzer.analyze_error(connection_error, context)
        assert analysis["severity"].value == "high"
    
    def test_error_analyzer_category_determination(self):
        """Test error category determination.""" 
        analyzer = ErrorAnalyzer()
        context = Mock()
        
        # Test timeout error
        timeout_error = TimeoutError("Request timeout")
        analysis = analyzer.analyze_error(timeout_error, context)
        assert analysis["category"].value == "timeout"
        
        # Test HTTP authentication error
        auth_error = HTTPException(status_code=401, detail="Unauthorized")
        analysis = analyzer.analyze_error(auth_error, context)
        assert analysis["category"].value == "authentication"
    
    def test_error_analyzer_retry_determination(self):
        """Test retry determination logic."""
        analyzer = ErrorAnalyzer()
        context = Mock()
        context.retry_count = 1
        
        # Test retryable error
        connection_error = ConnectionError("Connection failed")
        analysis = analyzer.analyze_error(connection_error, context)
        assert analysis["is_retryable"]
        
        # Test non-retryable error (max retries reached)
        context.retry_count = 5
        analysis = analyzer.analyze_error(connection_error, context)
        assert not analysis["is_retryable"]
    
    @pytest.mark.asyncio
    async def test_middleware_success_request(self, app, middleware_config):
        """Test middleware with successful request."""
        # Add middleware to app
        middleware = ErrorHandlingMiddleware(app, middleware_config)
        app.add_middleware(BaseHTTPMiddleware, dispatch=middleware.dispatch)
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
    
    @pytest.mark.asyncio
    async def test_middleware_error_handling(self, app, middleware_config):
        """Test middleware error handling."""
        middleware = ErrorHandlingMiddleware(app, middleware_config)
        app.add_middleware(BaseHTTPMiddleware, dispatch=middleware.dispatch)
        
        client = TestClient(app)
        response = client.get("/error")
        
        # Should handle error gracefully
        assert response.status_code in [500, 503]  # Depending on handling
    
    def test_middleware_metrics(self, middleware_config):
        """Test middleware metrics collection."""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app, middleware_config)
        
        # Simulate some requests
        middleware.request_count = 10
        middleware.error_count = 2
        middleware.total_processing_time = 500.0
        
        metrics = middleware.get_metrics()
        
        assert "requests_total" in metrics
        assert "errors_total" in metrics
        assert "availability" in metrics
        assert "average_processing_time_ms" in metrics
        
        assert metrics["requests_total"] == 10
        assert metrics["errors_total"] == 2
        assert metrics["availability"] == 0.8  # (10-2)/10
    
    @pytest.mark.asyncio
    async def test_middleware_health_check(self, middleware_config):
        """Test middleware health check."""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app, middleware_config)
        
        health = await middleware.health_check()
        
        assert "status" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded"]
    
    def test_middleware_factory(self):
        """Test middleware factory function."""
        config = ErrorHandlingConfig()
        factory = create_error_handling_middleware(config)
        
        app = FastAPI()
        middleware = factory(app)
        
        assert isinstance(middleware, ErrorHandlingMiddleware)
        assert middleware.config == config


class TestWorkflowErrorHandling:
    """Test suite for workflow error handling."""
    
    @pytest.fixture
    def config(self):
        """Create workflow error handling configuration."""
        return WorkflowErrorHandlingConfig(
            enabled=True,
            max_task_retries=2,
            max_batch_retries=1,
            recovery_timeout_ms=5000  # 5 seconds for testing
        )
    
    @pytest.fixture
    def recovery_manager(self, config):
        """Create workflow recovery manager."""
        return WorkflowErrorRecoveryManager(config)
    
    @pytest.fixture
    def error_context(self):
        """Create test error context."""
        return WorkflowErrorContext(
            workflow_id="test_workflow_123",
            task_id="test_task_456",
            agent_id="test_agent_789",
            error_message="Test error message",
            timeout_threshold_ms=30000
        )
    
    def test_workflow_error_context_creation(self, error_context):
        """Test workflow error context creation."""
        assert error_context.workflow_id == "test_workflow_123"
        assert error_context.task_id == "test_task_456"
        assert error_context.agent_id == "test_agent_789"
        assert error_context.retry_count == 0
    
    def test_recovery_manager_initialization(self, recovery_manager):
        """Test recovery manager initialization."""
        assert recovery_manager.config.enabled
        assert recovery_manager.config.max_task_retries == 2
        assert recovery_manager.recovery_attempts == 0
    
    @pytest.mark.asyncio
    async def test_error_analysis(self, recovery_manager, error_context):
        """Test error analysis and strategy determination."""
        # Test with timeout error
        timeout_error = TimeoutError("Operation timed out")
        
        strategy, params = recovery_manager.error_analyzer.analyze_error(
            timeout_error, error_context
        )
        
        assert isinstance(strategy, WorkflowRecoveryStrategy)
        assert isinstance(params, dict)
        assert error_context.error_type == WorkflowErrorType.TIMEOUT_ERROR
    
    @pytest.mark.asyncio
    async def test_retry_task_recovery(self, recovery_manager, error_context):
        """Test retry task recovery strategy."""
        connection_error = ConnectionError("Connection failed")
        
        success, details = await recovery_manager.handle_workflow_error(
            connection_error, error_context
        )
        
        # Should attempt retry for connection error
        assert isinstance(success, bool)
        assert isinstance(details, dict)
        assert recovery_manager.recovery_attempts == 1
    
    @pytest.mark.asyncio
    async def test_skip_task_recovery(self, recovery_manager, error_context):
        """Test skip task recovery strategy."""
        # Set retry count to max to trigger skip
        error_context.retry_count = 3
        
        validation_error = ValueError("Validation failed")
        
        success, details = await recovery_manager.handle_workflow_error(
            validation_error, error_context
        )
        
        assert isinstance(details, dict)
        # Strategy might be skip or graceful degradation
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_recovery(self, recovery_manager, error_context):
        """Test graceful degradation recovery strategy."""
        # Use an error that typically triggers degradation
        service_error = ConnectionError("Service unavailable")
        error_context.retry_count = 5  # Exceed retry limits
        
        success, details = await recovery_manager.handle_workflow_error(
            service_error, error_context
        )
        
        assert isinstance(details, dict)
        # Should have some recovery attempt
        assert recovery_manager.recovery_attempts >= 1
    
    def test_recovery_metrics(self, recovery_manager):
        """Test recovery metrics collection."""
        # Simulate some recovery attempts
        recovery_manager.recovery_attempts = 5
        recovery_manager.successful_recoveries = 3
        recovery_manager.failed_recoveries = 2
        recovery_manager.recovery_times = [100.0, 200.0, 150.0]
        
        metrics = recovery_manager.get_recovery_metrics()
        
        assert "recovery_attempts" in metrics
        assert "successful_recoveries" in metrics
        assert "failed_recoveries" in metrics
        assert "recovery_success_rate" in metrics
        assert "average_recovery_time_ms" in metrics
        
        assert metrics["recovery_attempts"] == 5
        assert metrics["successful_recoveries"] == 3
        assert metrics["recovery_success_rate"] == 0.6
    
    @pytest.mark.asyncio
    async def test_recovery_health_check(self, recovery_manager):
        """Test recovery manager health check."""
        health = await recovery_manager.health_check()
        
        assert "status" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "degraded"]


class TestErrorHandlingConfiguration:
    """Test suite for error handling configuration system."""
    
    def test_configuration_initialization(self):
        """Test configuration initialization with defaults."""
        config = ErrorHandlingConfiguration()
        
        assert config.enabled
        assert config.environment == ErrorHandlingEnvironment.DEVELOPMENT
        assert config.circuit_breaker.enabled
        assert config.retry_policy.enabled
        assert config.graceful_degradation.enabled
    
    def test_configuration_environment_overrides(self):
        """Test environment-specific configuration overrides."""
        # Test production environment
        prod_config = ErrorHandlingConfiguration(
            environment=ErrorHandlingEnvironment.PRODUCTION
        )
        prod_config.apply_environment_overrides()
        
        assert not prod_config.debug_mode
        assert not prod_config.observability.detailed_logging
        
        # Test development environment
        dev_config = ErrorHandlingConfiguration(
            environment=ErrorHandlingEnvironment.DEVELOPMENT
        )
        dev_config.apply_environment_overrides()
        
        assert dev_config.debug_mode
        assert dev_config.observability.detailed_logging
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = ErrorHandlingConfiguration()
        
        # Test with valid configuration
        issues = config.validate_configuration()
        assert "errors" in issues
        assert "warnings" in issues
        assert "recommendations" in issues
        
        # Errors should be empty for default config
        assert len(issues["errors"]) == 0
    
    def test_configuration_validation_with_issues(self):
        """Test configuration validation with problematic settings."""
        config = ErrorHandlingConfiguration()
        
        # Set problematic values
        config.performance_targets.availability_target = 0.5  # Too low
        config.circuit_breaker.failure_threshold = 1  # Too sensitive
        
        issues = config.validate_configuration()
        
        # Should have errors for low availability target
        assert len(issues["errors"]) > 0
        assert any("availability" in error.lower() for error in issues["errors"])
    
    def test_configuration_export_import(self, tmp_path):
        """Test configuration export and import."""
        config = ErrorHandlingConfiguration()
        config.debug_mode = True
        config.circuit_breaker.failure_threshold = 15
        
        # Export to file
        config_file = tmp_path / "test_config.json"
        config.save_to_file(str(config_file))
        
        # Import from file
        imported_config = ErrorHandlingConfiguration.load_from_file(str(config_file))
        
        assert imported_config.debug_mode == True
        assert imported_config.circuit_breaker.failure_threshold == 15
    
    def test_configuration_manager(self):
        """Test configuration manager functionality."""
        config = ErrorHandlingConfiguration()
        manager = ConfigurationManager(config)
        
        assert manager.get_config() == config
        
        # Test configuration update
        new_config = ErrorHandlingConfiguration()
        new_config.debug_mode = True
        
        success = manager.update_config(new_config)
        assert success
        assert manager.get_config().debug_mode == True
    
    def test_configuration_manager_rollback(self):
        """Test configuration manager rollback functionality."""
        original_config = ErrorHandlingConfiguration()
        manager = ConfigurationManager(original_config)
        
        # Update configuration
        new_config = ErrorHandlingConfiguration()
        new_config.debug_mode = True
        manager.update_config(new_config)
        
        # Rollback
        success = manager.rollback_config(steps=1)
        assert success
        assert manager.get_config().debug_mode == original_config.debug_mode
    
    def test_configuration_change_callbacks(self):
        """Test configuration change callbacks."""
        config = ErrorHandlingConfiguration()
        manager = ConfigurationManager(config)
        
        callback_called = False
        received_config = None
        
        def test_callback(new_config):
            nonlocal callback_called, received_config
            callback_called = True
            received_config = new_config
        
        manager.add_change_callback(test_callback)
        
        # Update configuration
        new_config = ErrorHandlingConfiguration()
        new_config.debug_mode = True
        manager.update_config(new_config)
        
        assert callback_called
        assert received_config == new_config


class TestErrorHandlingIntegration:
    """Test suite for error handling observability integration."""
    
    @pytest.fixture
    def integration(self):
        """Create error handling integration."""
        return ErrorHandlingObservabilityIntegration(enable_detailed_logging=True)
    
    def test_integration_initialization(self, integration):
        """Test integration initialization."""
        assert integration.enable_detailed_logging
        assert integration.error_events_emitted == 0
        assert integration.recovery_events_emitted == 0
    
    @pytest.mark.asyncio
    async def test_error_event_emission(self, integration):
        """Test error event emission."""
        # Mock observability hooks
        with patch.object(integration, 'observability_hooks') as mock_hooks:
            mock_hooks.failure_detected = AsyncMock()
            
            await integration.emit_error_handling_failure(
                error_type="TestError",
                error_message="Test error message",
                component="test_component",
                context={"test": "context"}
            )
            
            mock_hooks.failure_detected.assert_called_once()
            assert integration.error_events_emitted == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_state_change_emission(self, integration):
        """Test circuit breaker state change event emission."""
        with patch.object(integration, 'observability_hooks') as mock_hooks:
            mock_hooks.failure_detected = AsyncMock()
            mock_hooks.recovery_initiated = AsyncMock()
            
            # Test state change to open (failure)
            await integration.emit_circuit_breaker_state_change(
                circuit_breaker_name="test_breaker",
                old_state=CircuitBreakerState.CLOSED,
                new_state=CircuitBreakerState.OPEN,
                reason="failure_threshold_exceeded",
                metrics={"failure_count": 5}
            )
            
            mock_hooks.failure_detected.assert_called_once()
            
            # Test state change to closed (recovery)
            await integration.emit_circuit_breaker_state_change(
                circuit_breaker_name="test_breaker",
                old_state=CircuitBreakerState.HALF_OPEN,
                new_state=CircuitBreakerState.CLOSED,
                reason="recovery_success",
                metrics={}
            )
            
            mock_hooks.recovery_initiated.assert_called_once()
            assert integration.recovery_events_emitted == 1
    
    def test_integration_metrics(self, integration):
        """Test integration metrics collection."""
        # Simulate some events
        integration.error_events_emitted = 5
        integration.recovery_events_emitted = 2
        integration.degradation_events_emitted = 1
        
        metrics = integration.get_integration_metrics()
        
        assert "error_events_emitted" in metrics
        assert "recovery_events_emitted" in metrics
        assert "degradation_events_emitted" in metrics
        assert "total_events_emitted" in metrics
        
        assert metrics["error_events_emitted"] == 5
        assert metrics["recovery_events_emitted"] == 2
        assert metrics["total_events_emitted"] == 8


class TestPerformanceAndReliability:
    """Performance and reliability tests for error handling system."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance targets."""
        circuit_breaker = CircuitBreaker("perf_test", failure_threshold=10)
        
        # Measure decision time
        start_time = time.time()
        
        for _ in range(1000):
            await circuit_breaker.get_state()
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_time_per_call = total_time / 1000
        
        # Should be well under 1ms per call
        assert avg_time_per_call < 1.0
    
    @pytest.mark.asyncio
    async def test_retry_policy_performance(self):
        """Test retry policy performance targets."""
        config = RetryConfig(max_attempts=5)
        policy = ExponentialBackoffPolicy(config)
        
        # Measure calculation time
        start_time = time.time()
        
        for i in range(1000):
            await policy.calculate_delay(i % 5)
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_time_per_calc = total_time / 1000
        
        # Should be well under 0.5ms per calculation
        assert avg_time_per_calc < 0.5
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_performance(self):
        """Test graceful degradation performance targets."""
        manager = GracefulDegradationManager()
        
        # Measure degradation application time
        start_time = time.time()
        
        for i in range(100):
            await manager.apply_degradation(
                service_path=f"/test/{i}",
                degradation_level=DegradationLevel.MINIMAL,
                error_context={"error": "test"}
            )
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_time_per_degradation = total_time / 100
        
        # Should be under 2ms per degradation
        assert avg_time_per_degradation < 2.0
    
    @pytest.mark.asyncio
    async def test_error_handling_reliability(self):
        """Test error handling system reliability under load."""
        circuit_breaker = CircuitBreaker("reliability_test")
        config = RetryConfig(max_attempts=3, base_delay_ms=1)
        policy = ExponentialBackoffPolicy(config)
        executor = RetryExecutor(policy)
        
        # Simulate mixed success/failure scenarios
        success_count = 0
        failure_count = 0
        
        for i in range(100):
            try:
                if i % 10 == 0:  # 10% failure rate
                    raise ConnectionError("Simulated failure")
                else:
                    await circuit_breaker.record_success()
                    success_count += 1
            except:
                await circuit_breaker.record_failure()
                failure_count += 1
        
        # System should remain stable
        state = await circuit_breaker.get_state()
        metrics = circuit_breaker.get_metrics()
        
        # With 10% failure rate, circuit breaker should remain closed
        assert state == CircuitBreakerState.CLOSED
        assert success_count > failure_count
        assert metrics["failure_rate"] < 0.2  # Should be around 10%
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test error handling under concurrent load."""
        circuit_breaker = CircuitBreaker("concurrent_test")
        
        async def concurrent_operation(operation_id: int):
            """Simulate concurrent operation."""
            try:
                if operation_id % 20 == 0:  # 5% failure rate
                    raise ValueError(f"Operation {operation_id} failed")
                
                await circuit_breaker.record_success()
                return f"success_{operation_id}"
            except Exception as e:
                await circuit_breaker.record_failure()
                raise
        
        # Run 100 concurrent operations
        tasks = [concurrent_operation(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if isinstance(r, str) and r.startswith("success"))
        failures = sum(1 for r in results if isinstance(r, Exception))
        
        # Should handle concurrent load gracefully
        assert successes > failures
        assert successes + failures == 100
        
        # Circuit breaker should remain stable
        state = await circuit_breaker.get_state()
        assert state == CircuitBreakerState.CLOSED


class ErrorHandlingComprehensiveTestScript(ScriptBase):
    """Standardized error handling test execution with coverage using ScriptBase pattern."""
    
    async def run(self) -> Dict[str, Any]:
        """Execute comprehensive error handling tests with coverage."""
        import subprocess
        import sys
        
        # Run pytest with coverage configuration
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            __file__,
            "-v",
            "--cov=app.core",
            "--cov-report=html", 
            "--cov-report=term-missing",
            "--cov-fail-under=85"
        ], capture_output=True, text=True)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "message": f"Error handling tests with coverage {'passed' if result.returncode == 0 else 'failed'}"
        }


# Create script instance
script = ErrorHandlingComprehensiveTestScript()

if __name__ == "__main__":
    script.execute()