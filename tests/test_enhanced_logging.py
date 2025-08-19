"""
Comprehensive tests for enhanced logging and error handling system.

Tests all aspects of the production-ready logging infrastructure including
correlation IDs, performance monitoring, security event logging, and
error handling with context.
"""

import json
import pytest
import asyncio
import time
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from contextlib import asynccontextmanager

from app.core.enhanced_logging import (
    EnhancedLogger,
    CorrelationContext,
    PerformanceTracker,
    correlation_context,
    correlation_processor,
    performance_processor,
    security_processor,
    with_correlation_id,
    with_performance_logging,
    operation_context,
    log_aggregator,
    set_request_context
)


class TestCorrelationContext:
    """Test correlation context management."""
    
    def test_correlation_id_management(self):
        """Test correlation ID setting and retrieval."""
        context = CorrelationContext()
        
        # Test setting and getting correlation ID
        test_id = "test-correlation-123"
        context.set_correlation_id(test_id)
        assert context.get_correlation_id() == test_id
        
        # Test request ID
        request_id = "req-456" 
        context.set_request_id(request_id)
        assert context.get_request_id() == request_id
        
        # Test full context
        full_context = context.get_full_context()
        assert full_context["correlation_id"] == test_id
        assert full_context["request_id"] == request_id
    
    def test_user_context_management(self):
        """Test user context setting and retrieval."""
        context = CorrelationContext()
        
        context.set_user_context("user123", "admin")
        user_context = context.get_user_context()
        
        assert user_context["user_id"] == "user123"
        assert user_context["user_role"] == "admin"
    
    def test_operation_context_management(self):
        """Test operation context setting and retrieval."""
        context = CorrelationContext()
        
        context.set_operation_context("spawn_agent", "orchestrator")
        full_context = context.get_full_context()
        
        assert full_context["operation"] == "spawn_agent"
        assert full_context["component"] == "orchestrator"
    
    def test_context_clear(self):
        """Test context clearing."""
        context = CorrelationContext()
        
        context.set_correlation_id("test")
        context.set_request_id("test")
        context.set_user_context("user", "role")
        
        context.clear()
        
        assert context.get_correlation_id() is None
        assert context.get_request_id() is None
        assert context.get_full_context() == {}


class TestEnhancedLogger:
    """Test enhanced logger functionality."""
    
    @pytest.fixture
    def logger(self):
        """Create test logger."""
        return EnhancedLogger("test_component")
    
    def test_logger_initialization(self, logger):
        """Test logger initialization."""
        assert logger.component == "test_component"
        assert logger.logger is not None
        assert logger._operation_start_times == {}
    
    def test_operation_tracking(self, logger):
        """Test operation start and end tracking."""
        # Start operation
        operation_id = logger.start_operation("test_operation", param1="value1")
        assert operation_id in logger._operation_start_times
        
        # End operation
        logger.end_operation(operation_id, success=True, result="success")
        assert operation_id not in logger._operation_start_times
    
    def test_request_response_logging(self, logger):
        """Test request and response logging methods."""
        # Mock the underlying logger
        logger.logger = Mock()
        
        # Test request logging
        logger.log_request(
            "POST", 
            "/api/v2/agents",
            client_ip="192.168.1.100",
            user_agent="test-client"
        )
        
        logger.logger.info.assert_called()
        call_args = logger.logger.info.call_args
        assert call_args[0][0] == "api_request"
        assert call_args[1]["http_method"] == "POST"
        assert call_args[1]["path"] == "/api/v2/agents"
        
        # Test response logging
        logger.log_response(201, 150.5, resource_type="agents")
        
        # Should have been called twice now (request + response)
        assert logger.logger.info.call_count >= 2
    
    def test_error_logging(self, logger):
        """Test error logging with context."""
        logger.logger = Mock()
        
        test_error = ValueError("Test error message")
        context = {"operation": "test_op", "param": "value"}
        
        logger.log_error(test_error, context)
        
        logger.logger.error.assert_called_once()
        call_args = logger.logger.error.call_args
        assert call_args[0][0] == "error_occurred"
        assert call_args[1]["error_type"] == "ValueError"
        assert call_args[1]["error_message"] == "Test error message"
        assert "stack_trace" in call_args[1]
        assert call_args[1]["operation"] == "test_op"
    
    def test_security_event_logging(self, logger):
        """Test security event logging."""
        logger.logger = Mock()
        logger.logger.bind = Mock(return_value=logger.logger)
        
        logger.log_security_event(
            "authentication_failed",
            "HIGH",
            user_id="test_user",
            client_ip="192.168.1.100"
        )
        
        logger.logger.bind.assert_called_once()
        bind_args = logger.logger.bind.call_args[1]
        assert bind_args["security_event"] is True
        assert bind_args["severity"] == "HIGH"
        assert bind_args["event_type"] == "authentication_failed"
        
        logger.logger.warning.assert_called_once()
    
    def test_performance_metric_logging(self, logger):
        """Test performance metric logging."""
        logger.logger = Mock()
        logger.logger.bind = Mock(return_value=logger.logger)
        
        logger.log_performance_metric(
            "response_time",
            125.5,
            unit="ms",
            endpoint="agents"
        )
        
        logger.logger.bind.assert_called_once()
        bind_args = logger.logger.bind.call_args[1]
        assert bind_args["metric_type"] == "performance"
        assert bind_args["metric_name"] == "response_time"
        assert bind_args["metric_value"] == 125.5
        assert bind_args["unit"] == "ms"
    
    def test_audit_event_logging(self, logger):
        """Test audit event logging."""
        logger.logger = Mock()
        logger.logger.bind = Mock(return_value=logger.logger)
        
        logger.log_audit_event(
            "agent_created",
            "agent:123",
            success=True,
            user_id="test_user"
        )
        
        logger.logger.bind.assert_called_once()
        bind_args = logger.logger.bind.call_args[1]
        assert bind_args["audit_event"] is True
        assert bind_args["action"] == "agent_created"
        assert bind_args["resource"] == "agent:123"
        assert bind_args["success"] is True


class TestPerformanceTracker:
    """Test performance tracking context manager."""
    
    def test_performance_tracker_context(self):
        """Test performance tracker as context manager."""
        logger = Mock()
        logger.start_operation = Mock(return_value="op_123")
        logger.end_operation = Mock()
        
        with PerformanceTracker(logger, "test_operation", param="value") as tracker:
            assert tracker.operation_id == "op_123"
            assert tracker.start_time is not None
            time.sleep(0.01)  # Small delay for testing
        
        logger.start_operation.assert_called_once_with(
            "test_operation", 
            param="value"
        )
        logger.end_operation.assert_called_once_with(
            "op_123",
            success=True,
            param="value"
        )
    
    def test_performance_tracker_with_exception(self):
        """Test performance tracker with exception."""
        logger = Mock()
        logger.start_operation = Mock(return_value="op_123")
        logger.end_operation = Mock()
        
        with pytest.raises(ValueError):
            with PerformanceTracker(logger, "test_operation") as tracker:
                raise ValueError("Test error")
        
        logger.end_operation.assert_called_once()
        call_args = logger.end_operation.call_args[1]
        assert call_args["success"] is False
        assert "error" in call_args


class TestStructlogProcessors:
    """Test custom structlog processors."""
    
    def test_correlation_processor(self):
        """Test correlation processor adds context."""
        # Set up correlation context
        correlation_context.set_correlation_id("test-corr-123")
        correlation_context.set_request_id("test-req-456")
        correlation_context.set_user_context("user123", "admin")
        
        event_dict = {"event": "test_event", "message": "test"}
        
        # Process event
        result = correlation_processor(None, None, event_dict)
        
        assert result["correlation_id"] == "test-corr-123"
        assert result["request_id"] == "test-req-456"
        assert result["user_id"] == "user123"
        assert result["user_role"] == "admin"
        assert "process_id" in result
        assert "thread_id" in result
        
        # Clean up
        correlation_context.clear()
    
    def test_performance_processor(self):
        """Test performance processor adds classification."""
        event_dict = {"event": "test_event", "duration_ms": 1500.0}
        
        result = performance_processor(None, None, event_dict)
        
        assert result["performance_class"] == "slow"
        assert "log_timestamp_ms" in result
        
        # Test medium performance
        event_dict = {"event": "test_event", "duration_ms": 750.0}
        result = performance_processor(None, None, event_dict)
        assert result["performance_class"] == "medium"
        
        # Test fast performance
        event_dict = {"event": "test_event", "duration_ms": 50.0}
        result = performance_processor(None, None, event_dict)
        assert result["performance_class"] == "fast"
    
    def test_security_processor(self):
        """Test security processor marks security events."""
        # Test security-related event
        event_dict = {"event": "authentication_failed", "user_id": "test"}
        result = security_processor(None, None, event_dict)
        
        assert result["security_relevant"] is True
        assert result["log_category"] == "security"
        
        # Test non-security event
        event_dict = {"event": "data_processed", "count": 100}
        result = security_processor(None, None, event_dict)
        
        assert "security_relevant" not in result
        assert "log_category" not in result


class TestDecorators:
    """Test logging decorators."""
    
    @pytest.mark.asyncio
    async def test_correlation_id_decorator_async(self):
        """Test correlation ID decorator for async functions."""
        @with_correlation_id("test-correlation-789")
        async def async_function():
            return correlation_context.get_correlation_id()
        
        # Test with specified correlation ID
        result = await async_function()
        assert result == "test-correlation-789"
        
        # Test with auto-generated correlation ID
        @with_correlation_id()
        async def async_function_auto():
            return correlation_context.get_correlation_id()
        
        result = await async_function_auto()
        assert result is not None
        assert len(result) > 10  # Should be a UUID-like string
    
    def test_correlation_id_decorator_sync(self):
        """Test correlation ID decorator for sync functions."""
        @with_correlation_id("test-correlation-sync")
        def sync_function():
            return correlation_context.get_correlation_id()
        
        result = sync_function()
        assert result == "test-correlation-sync"
    
    @pytest.mark.asyncio
    async def test_performance_logging_decorator_async(self):
        """Test performance logging decorator for async functions."""
        with patch('app.core.enhanced_logging.EnhancedLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            @with_performance_logging("test_operation")
            async def async_function():
                await asyncio.sleep(0.01)
                return "success"
            
            result = await async_function()
            assert result == "success"
            
            # Verify logger was created and used
            mock_logger_class.assert_called()
    
    def test_performance_logging_decorator_sync(self):
        """Test performance logging decorator for sync functions."""
        with patch('app.core.enhanced_logging.EnhancedLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            @with_performance_logging("test_operation")
            def sync_function():
                time.sleep(0.01)
                return "success"
            
            result = sync_function()
            assert result == "success"
            
            # Verify logger was created and used
            mock_logger_class.assert_called()


class TestLogAggregator:
    """Test log aggregation functionality."""
    
    def test_performance_metric_aggregation(self):
        """Test performance metric aggregation."""
        # Clear existing metrics
        log_aggregator.metrics.clear()
        
        # Add some metrics
        log_aggregator.aggregate_performance_metric("api_response_time", 100.0)
        log_aggregator.aggregate_performance_metric("api_response_time", 150.0)
        log_aggregator.aggregate_performance_metric("api_response_time", 200.0)
        
        # Get summary
        summary = log_aggregator.get_performance_summary()
        
        assert "api_response_time" in summary
        stats = summary["api_response_time"]
        assert stats["count"] == 3
        assert stats["avg"] == 150.0
        assert stats["min"] == 100.0
        assert stats["max"] == 200.0
    
    def test_error_count_aggregation(self):
        """Test error count aggregation."""
        # Clear existing errors
        log_aggregator.error_counts.clear()
        
        # Record some errors
        log_aggregator.record_error("ValidationError")
        log_aggregator.record_error("DatabaseError")
        log_aggregator.record_error("ValidationError")
        
        # Get summary
        summary = log_aggregator.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["by_type"]["ValidationError"] == 2
        assert summary["by_type"]["DatabaseError"] == 1
    
    def test_security_event_aggregation(self):
        """Test security event aggregation."""
        # Clear existing events
        log_aggregator.security_events.clear()
        
        # Record security events
        event1 = {
            "event_type": "authentication_failed",
            "severity": "HIGH",
            "user_id": "test_user"
        }
        event2 = {
            "event_type": "suspicious_activity",
            "severity": "CRITICAL",
            "user_id": "other_user"
        }
        
        log_aggregator.record_security_event(event1)
        log_aggregator.record_security_event(event2)
        
        # Get summary
        summary = log_aggregator.get_security_summary()
        
        assert summary["total_events"] == 2
        assert len(summary["recent_events"]) == 2
        assert "authentication_failed" in summary["event_types"]
        assert "suspicious_activity" in summary["event_types"]


class TestOperationContext:
    """Test operation context manager."""
    
    @pytest.mark.asyncio
    async def test_operation_context_success(self):
        """Test operation context manager with successful operation."""
        logger = Mock()
        logger.start_operation = Mock(return_value="op_123")
        logger.end_operation = Mock()
        
        async with operation_context(logger, "test_operation", param="value") as op_id:
            assert op_id == "op_123"
            # Simulate some work
            await asyncio.sleep(0.001)
        
        logger.start_operation.assert_called_once_with(
            "test_operation",
            param="value"
        )
        logger.end_operation.assert_called_once_with("op_123", success=True)
    
    @pytest.mark.asyncio
    async def test_operation_context_with_exception(self):
        """Test operation context manager with exception."""
        logger = Mock()
        logger.start_operation = Mock(return_value="op_123")
        logger.end_operation = Mock()
        logger.log_error = Mock()
        
        with pytest.raises(ValueError):
            async with operation_context(logger, "test_operation") as op_id:
                raise ValueError("Test error")
        
        logger.start_operation.assert_called_once()
        logger.log_error.assert_called_once()
        logger.end_operation.assert_called_once()
        
        # Check that error was logged with operation_id
        error_args = logger.log_error.call_args[1]
        assert error_args["operation_id"] == "op_123"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_set_request_context(self):
        """Test set_request_context utility function."""
        # Clear context first
        correlation_context.clear()
        
        set_request_context("req_123", "user456", "admin")
        
        assert correlation_context.get_request_id() == "req_123"
        user_context = correlation_context.get_user_context()
        assert user_context["user_id"] == "user456"
        assert user_context["user_role"] == "admin"
    
    def test_get_correlation_id(self):
        """Test get_correlation_id utility function."""
        from app.core.enhanced_logging import get_correlation_id, set_correlation_id
        
        # Clear and set correlation ID
        correlation_context.clear()
        set_correlation_id("test_corr_789")
        
        assert get_correlation_id() == "test_corr_789"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self):
        """Test full request lifecycle with all logging features."""
        # Setup
        correlation_id = "req_" + str(uuid.uuid4())
        logger = EnhancedLogger("integration_test")
        logger.logger = Mock()
        
        # Simulate request start
        set_request_context(correlation_id, "test_user", "admin")
        
        # Simulate operation with performance tracking
        with PerformanceTracker(logger, "full_operation", request_id=correlation_id):
            # Log various events
            logger.log_request("POST", "/api/v2/test", client_ip="127.0.0.1")
            
            # Simulate some processing
            await asyncio.sleep(0.01)
            
            # Log performance metric
            logger.log_performance_metric("processing_time", 10.5, unit="ms")
            
            # Log audit event
            logger.log_audit_event("resource_accessed", "resource:123", success=True)
            
            # Log response
            logger.log_response(200, 15.2)
        
        # Verify all logging calls were made
        assert logger.logger.info.call_count >= 3  # request, audit, response
        
        # Clean up
        correlation_context.clear()
    
    @pytest.mark.asyncio
    async def test_error_handling_with_context(self):
        """Test comprehensive error handling with context."""
        logger = EnhancedLogger("error_test")
        logger.logger = Mock()
        
        correlation_id = "err_" + str(uuid.uuid4())
        set_request_context(correlation_id, "error_user", "user")
        
        try:
            async with operation_context(logger, "error_operation") as op_id:
                # Simulate error condition
                raise DatabaseError("Connection failed")
        except Exception:
            pass  # Expected
        
        # Verify error was logged with proper context
        logger.log_error.assert_called_once()
        error_context = logger.log_error.call_args[1]
        assert "operation_id" in error_context
        
        # Clean up
        correlation_context.clear()
    
    def test_security_event_with_audit_trail(self):
        """Test security event with complete audit trail."""
        logger = EnhancedLogger("security_test")
        logger.logger = Mock()
        logger.logger.bind = Mock(return_value=logger.logger)
        
        # Set user context
        set_request_context("sec_123", "security_user", "admin")
        
        # Log authentication failure
        logger.log_security_event(
            "authentication_failed",
            "HIGH",
            user_id="security_user",
            client_ip="192.168.1.100",
            reason="invalid_password"
        )
        
        # Log related audit event
        logger.log_audit_event(
            "authentication_attempt",
            "user:security_user",
            success=False,
            client_ip="192.168.1.100"
        )
        
        # Verify both events were logged
        assert logger.logger.bind.call_count >= 2
        assert logger.logger.warning.call_count >= 1
        assert logger.logger.info.call_count >= 1
        
        # Clean up
        correlation_context.clear()


# Custom exception for testing
class DatabaseError(Exception):
    """Test database error exception."""
    pass


# Integration test fixtures
@pytest.fixture
def clean_correlation_context():
    """Fixture to ensure clean correlation context for each test."""
    correlation_context.clear()
    yield
    correlation_context.clear()


@pytest.fixture
def clean_log_aggregator():
    """Fixture to ensure clean log aggregator for each test."""
    log_aggregator.metrics.clear()
    log_aggregator.error_counts.clear()
    log_aggregator.security_events.clear()
    yield
    log_aggregator.metrics.clear()
    log_aggregator.error_counts.clear()
    log_aggregator.security_events.clear()


# Performance test
class TestPerformance:
    """Test performance of logging infrastructure."""
    
    def test_logging_performance_overhead(self):
        """Test that logging doesn't add significant overhead."""
        logger = EnhancedLogger("perf_test")
        
        # Baseline: function without logging
        def baseline_function():
            return sum(range(1000))
        
        # Function with logging
        @with_performance_logging("performance_test")
        def logged_function():
            return sum(range(1000))
        
        # Time baseline
        start_time = time.time()
        for _ in range(100):
            baseline_function()
        baseline_time = time.time() - start_time
        
        # Time with logging
        start_time = time.time()
        for _ in range(100):
            logged_function()
        logged_time = time.time() - start_time
        
        # Logging overhead should be less than 200% of baseline
        overhead_ratio = logged_time / baseline_time
        assert overhead_ratio < 3.0, f"Logging overhead too high: {overhead_ratio:.2f}x"
    
    def test_correlation_context_performance(self):
        """Test correlation context performance."""
        context = CorrelationContext()
        
        # Test setting context many times
        start_time = time.time()
        for i in range(1000):
            context.set_correlation_id(f"test_{i}")
            context.set_request_id(f"req_{i}")
            context.get_full_context()
        
        duration = time.time() - start_time
        
        # Should be very fast
        assert duration < 0.1, f"Correlation context too slow: {duration:.3f}s for 1000 operations"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])