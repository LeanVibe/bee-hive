"""
TDD Tests for WebSocket Circuit Breaker Pattern
Sprint 2: WebSocket Resilience & Documentation Foundation

Test-driven development for circuit breaker implementation to prevent
cascading failures and provide graceful degradation in WebSocket connections.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from enum import Enum

from app.api.dashboard_websockets import DashboardWebSocketManager


class CircuitBreakerState(Enum):
    """Circuit breaker states for WebSocket connections."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests  
    HALF_OPEN = "half-open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    timeout_duration: int = 60  # seconds
    success_threshold: int = 3  # for half-open to closed
    max_failures_window: int = 300  # 5 minutes


class TestWebSocketCircuitBreaker:
    """TDD test suite for WebSocket circuit breaker pattern."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Provide circuit breaker configuration for tests."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            timeout_duration=30,
            success_threshold=2,
            max_failures_window=120
        )

    @pytest.fixture
    def mock_websocket_manager_with_circuit_breaker(self, circuit_breaker_config):
        """Create WebSocket manager with circuit breaker capabilities."""
        manager = DashboardWebSocketManager()
        manager.circuit_breaker_config = circuit_breaker_config
        manager.circuit_breakers = {}
        manager.connection_failure_history = {}
        return manager

    # =============== CIRCUIT BREAKER STATE MANAGEMENT TESTS ===============

    @pytest.mark.asyncio
    async def test_should_initialize_circuit_breaker_in_closed_state(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test circuit breaker starts in CLOSED state for new connections."""
        # Given: A new connection
        connection_id = "new-connection-123"
        
        # When: Circuit breaker is initialized
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        # Then: Should be in CLOSED state
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "closed"
        
        # And: Should have zero failure count
        breaker = mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id]
        assert breaker["failure_count"] == 0
        assert breaker["consecutive_successes"] == 0

    @pytest.mark.asyncio
    async def test_should_transition_to_open_after_failure_threshold(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test circuit breaker opens after reaching failure threshold."""
        # Given: A connection with circuit breaker
        connection_id = "failing-connection-456"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        # When: Failures exceed threshold (3 in this config)
        for i in range(4):  # 4 failures > 3 threshold
            await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                connection_id, f"failure_{i}"
            )
        
        # Then: Circuit breaker should be OPEN
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "open"
        
        # And: Should block new connection attempts
        should_allow = await mock_websocket_manager_with_circuit_breaker.should_allow_connection(
            connection_id
        )
        assert should_allow is False

    @pytest.mark.asyncio
    async def test_should_transition_to_half_open_after_timeout(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test circuit breaker moves to HALF_OPEN after timeout period."""
        # Given: A circuit breaker in OPEN state
        connection_id = "timeout-test-connection"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        # Force OPEN state with past timestamp
        mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id] = {
            "state": "open",
            "failure_count": 5,
            "consecutive_successes": 0,
            "last_failure_time": time.time() - 35,  # 35 seconds ago (> 30s timeout)
            "last_success_time": None,
            "opened_at": time.time() - 35
        }
        
        # When: Timeout period has elapsed
        await mock_websocket_manager_with_circuit_breaker.update_circuit_breaker_states()
        
        # Then: Should transition to HALF_OPEN
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "half-open"
        
        # And: Should allow limited connection attempts
        should_allow = await mock_websocket_manager_with_circuit_breaker.should_allow_connection(
            connection_id
        )
        assert should_allow is True

    @pytest.mark.asyncio
    async def test_should_transition_half_open_to_closed_on_success(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test HALF_OPEN transitions to CLOSED after successful operations."""
        # Given: A circuit breaker in HALF_OPEN state
        connection_id = "recovery-success-connection"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id] = {
            "state": "half-open",
            "failure_count": 3,
            "consecutive_successes": 0,
            "last_failure_time": time.time() - 40,
            "last_success_time": None,
            "opened_at": time.time() - 40
        }
        
        # When: Successful operations meet success threshold (2 in config)
        for i in range(2):
            await mock_websocket_manager_with_circuit_breaker.record_connection_success(
                connection_id
            )
        
        # Then: Should transition to CLOSED
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "closed"
        
        # And: Should reset failure count
        breaker = mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id]
        assert breaker["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_should_transition_half_open_to_open_on_failure(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test HALF_OPEN returns to OPEN on failure."""
        # Given: A circuit breaker in HALF_OPEN state
        connection_id = "recovery-failure-connection"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id] = {
            "state": "half-open",
            "failure_count": 3,
            "consecutive_successes": 1,
            "last_failure_time": time.time() - 40,
            "last_success_time": time.time() - 10,
            "opened_at": time.time() - 40
        }
        
        # When: A failure occurs in HALF_OPEN state
        await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
            connection_id, "half_open_failure"
        )
        
        # Then: Should return to OPEN state
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "open"
        
        # And: Should reset consecutive successes
        breaker = mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id]
        assert breaker["consecutive_successes"] == 0

    # =============== FAILURE DETECTION & CLASSIFICATION TESTS ===============

    @pytest.mark.asyncio
    async def test_should_classify_different_failure_types(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test circuit breaker handles different types of connection failures."""
        # Given: A connection with circuit breaker
        connection_id = "failure-classification-test"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        # When: Different failure types occur
        failure_types = [
            "network_timeout",
            "websocket_closed",
            "send_failure", 
            "authentication_error",
            "rate_limit_exceeded"
        ]
        
        for failure_type in failure_types:
            await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                connection_id, failure_type
            )
        
        # Then: Should track failure types in history
        failure_history = mock_websocket_manager_with_circuit_breaker.connection_failure_history[
            connection_id
        ]
        assert len(failure_history) == 5
        
        # And: Should count only circuit breaker relevant failures
        breaker = mock_websocket_manager_with_circuit_breaker.circuit_breakers[connection_id]
        # Authentication errors might not count toward circuit breaker
        assert breaker["failure_count"] >= 4  # Excluding auth errors

    @pytest.mark.asyncio
    async def test_should_implement_sliding_window_for_failure_counting(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test sliding window approach for failure rate calculation."""
        # Given: A connection with historical failures
        connection_id = "sliding-window-test"
        await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(
            connection_id
        )
        
        # When: Failures occur over time with some outside window
        current_time = time.time()
        
        # Old failures (outside 2-minute window)
        old_failures = [
            {"timestamp": current_time - 200, "type": "network_timeout"},
            {"timestamp": current_time - 180, "type": "websocket_closed"}
        ]
        
        # Recent failures (within 2-minute window)
        recent_failures = [
            {"timestamp": current_time - 60, "type": "send_failure"},
            {"timestamp": current_time - 30, "type": "network_timeout"},
            {"timestamp": current_time - 10, "type": "websocket_closed"}
        ]
        
        for failure in old_failures + recent_failures:
            await mock_websocket_manager_with_circuit_breaker.record_failure_with_timestamp(
                connection_id, failure["type"], failure["timestamp"]
            )
        
        # Then: Only recent failures should count
        recent_failure_count = mock_websocket_manager_with_circuit_breaker.get_recent_failure_count(
            connection_id, window_seconds=120
        )
        assert recent_failure_count == 3
        
        # And: Circuit breaker should consider only recent failures
        state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
            connection_id
        )
        assert state == "closed"  # 3 failures = threshold, should be exactly at limit

    # =============== CIRCUIT BREAKER METRICS & MONITORING TESTS ===============

    @pytest.mark.asyncio
    async def test_should_track_comprehensive_circuit_breaker_metrics(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test comprehensive metrics tracking for circuit breaker operations."""
        # Given: Various circuit breaker events
        connections = ["conn-1", "conn-2", "conn-3"]
        
        for conn_id in connections:
            await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(conn_id)
            
            # Simulate different circuit breaker events
            for _ in range(4):  # Force OPEN state
                await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                    conn_id, "network_error"
                )
        
        # When: Circuit breaker metrics are collected
        metrics = await mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_metrics()
        
        # Then: Should include comprehensive metrics
        expected_metrics = [
            "circuit_breakers_total",
            "circuit_breakers_open_total", 
            "circuit_breakers_closed_total",
            "circuit_breakers_half_open_total",
            "circuit_breaker_transitions_total",
            "circuit_breaker_blocked_requests_total",
            "circuit_breaker_allowed_requests_total",
            "average_failure_rate",
            "average_recovery_time_seconds"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # And: Should show correct counts
        assert metrics["circuit_breakers_total"] == 3
        assert metrics["circuit_breakers_open_total"] == 3

    @pytest.mark.asyncio
    async def test_should_expose_circuit_breaker_state_per_connection(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test detailed circuit breaker state exposure for monitoring."""
        # Given: Multiple connections with different circuit breaker states
        connections_config = {
            "healthy-conn": {"failures": 0, "expected_state": "closed"},
            "failing-conn": {"failures": 5, "expected_state": "open"},
            "recovering-conn": {"failures": 3, "expected_state": "closed"}
        }
        
        for conn_id, config in connections_config.items():
            await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(conn_id)
            
            for _ in range(config["failures"]):
                await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                    conn_id, "test_failure"
                )
        
        # When: Circuit breaker states are requested
        circuit_breaker_states = await mock_websocket_manager_with_circuit_breaker.get_all_circuit_breaker_states()
        
        # Then: Should return detailed state for each connection
        for conn_id, config in connections_config.items():
            assert conn_id in circuit_breaker_states
            state_info = circuit_breaker_states[conn_id]
            
            assert "state" in state_info
            assert "failure_count" in state_info
            assert "last_failure_time" in state_info
            assert "time_in_current_state" in state_info

    # =============== INTEGRATION & PERFORMANCE TESTS ===============

    @pytest.mark.asyncio
    async def test_circuit_breaker_should_not_impact_healthy_connections(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test circuit breaker doesn't affect performance of healthy connections."""
        # Given: Mix of healthy and failing connections
        healthy_connections = [f"healthy-{i}" for i in range(10)]
        failing_connections = [f"failing-{i}" for i in range(3)]
        
        # Initialize all circuit breakers
        for conn_id in healthy_connections + failing_connections:
            await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(conn_id)
        
        # Fail the failing connections
        for conn_id in failing_connections:
            for _ in range(5):
                await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                    conn_id, "network_error"
                )
        
        # When: Checking connection permissions
        start_time = time.time()
        
        healthy_results = []
        for conn_id in healthy_connections:
            allowed = await mock_websocket_manager_with_circuit_breaker.should_allow_connection(
                conn_id
            )
            healthy_results.append(allowed)
        
        processing_time = time.time() - start_time
        
        # Then: Healthy connections should not be affected
        assert all(healthy_results)  # All healthy connections allowed
        assert processing_time < 0.1  # Fast processing (< 100ms)
        
        # And: Failing connections should be blocked
        for conn_id in failing_connections:
            allowed = await mock_websocket_manager_with_circuit_breaker.should_allow_connection(
                conn_id
            )
            assert allowed is False

    @pytest.mark.asyncio 
    async def test_should_handle_concurrent_circuit_breaker_operations(
        self, mock_websocket_manager_with_circuit_breaker
    ):
        """Test thread-safety and concurrent circuit breaker operations."""
        # Given: Multiple connections being accessed concurrently
        connection_ids = [f"concurrent-{i}" for i in range(5)]
        
        for conn_id in connection_ids:
            await mock_websocket_manager_with_circuit_breaker.initialize_circuit_breaker(conn_id)
        
        # When: Concurrent operations occur
        async def simulate_failures(conn_id):
            for _ in range(3):
                await mock_websocket_manager_with_circuit_breaker.record_connection_failure(
                    conn_id, "concurrent_failure"
                )
                await asyncio.sleep(0.01)  # Small delay
        
        async def check_states(conn_id):
            for _ in range(10):
                state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
                    conn_id
                )
                await asyncio.sleep(0.005)
            return state
        
        # Execute concurrently
        failure_tasks = [simulate_failures(conn_id) for conn_id in connection_ids]
        check_tasks = [check_states(conn_id) for conn_id in connection_ids]
        
        await asyncio.gather(*failure_tasks, *check_tasks)
        
        # Then: All operations should complete without errors
        # And: Circuit breaker states should be consistent
        for conn_id in connection_ids:
            state = mock_websocket_manager_with_circuit_breaker.get_circuit_breaker_state(
                conn_id
            )
            assert state in ["closed", "open"]
            
            breaker = mock_websocket_manager_with_circuit_breaker.circuit_breakers[conn_id]
            assert breaker["failure_count"] >= 0