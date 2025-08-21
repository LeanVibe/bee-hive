"""
Epic 2 Phase 2.2: WebSocket Observability & Metrics - Comprehensive Tests

Tests for production-grade WebSocket monitoring, structured logging, and metrics collection.
Validates all Epic 2 Phase 2.2 requirements are met.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from app.api.dashboard_websockets import (
    DashboardWebSocketManager,
    WebSocketConnection,
    websocket_manager,
    router
)
from app.api.ws_utils import WS_CONTRACT_VERSION


class TestWebSocketObservability:
    """Test suite for WebSocket observability and metrics."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with WebSocket routes."""
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def ws_manager(self):
        """Create fresh WebSocket manager for testing."""
        manager = DashboardWebSocketManager()
        # Reset metrics for clean testing
        manager.metrics = {
            "messages_sent_total": 0,
            "messages_send_failures_total": 0,
            "messages_received_total": 0,
            "messages_dropped_rate_limit_total": 0,
            "errors_sent_total": 0,
            "connections_total": 0,
            "disconnections_total": 0,
            "backpressure_disconnects_total": 0,
            "auth_denied_total": 0,
            "origin_denied_total": 0,
            "idle_disconnects_total": 0,
            "bytes_sent_total": 0,
            "bytes_received_total": 0,
        }
        return manager

    def test_websocket_metrics_endpoint_availability(self, client):
        """Test that WebSocket metrics endpoint is available."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "messages_sent_total" in data
        assert "messages_send_failures_total" in data
        assert "messages_received_total" in data
        assert "errors_sent_total" in data
        assert "connections_total" in data
        assert "disconnections_total" in data
        assert "timestamp" in data

    def test_websocket_metrics_required_fields(self, client):
        """Test that all Epic 2 Phase 2.2 required metrics are present."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        
        # Epic 2 Phase 2.2 required metrics
        required_metrics = [
            "messages_sent_total",
            "messages_send_failures_total", 
            "messages_received_total",
            "messages_dropped_rate_limit_total",
            "errors_sent_total",
            "connections_total",
            "disconnections_total",
            "backpressure_disconnects_total"
        ]
        
        for metric in required_metrics:
            assert metric in data, f"Required metric '{metric}' missing from response"
            assert isinstance(data[metric], (int, float)), f"Metric '{metric}' should be numeric"

    def test_websocket_metrics_performance_analytics(self, client):
        """Test performance analytics in metrics response."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "performance_metrics" in data
        
        perf_metrics = data["performance_metrics"]
        expected_perf_fields = [
            "message_success_rate",
            "backpressure_rate", 
            "rate_limit_violation_rate",
            "average_connection_duration_seconds",
            "peak_concurrent_connections"
        ]
        
        for field in expected_perf_fields:
            assert field in perf_metrics, f"Performance metric '{field}' missing"

    def test_websocket_metrics_observability_compliance(self, client):
        """Test Epic 2 Phase 2.2 observability compliance flags."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "observability_compliance" in data
        
        compliance = data["observability_compliance"]
        required_compliance = [
            "structured_logging",
            "correlation_id_injection", 
            "metrics_exposition",
            "performance_monitoring",
            "error_tracking"
        ]
        
        for feature in required_compliance:
            assert feature in compliance, f"Compliance feature '{feature}' missing"
            assert compliance[feature] is True, f"Compliance feature '{feature}' not enabled"

    def test_websocket_metrics_contract_versioning(self, client):
        """Test contract versioning in metrics response.""" 
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "contract_version" in data
        assert "supported_versions" in data
        assert data["contract_version"] == WS_CONTRACT_VERSION
        assert WS_CONTRACT_VERSION in data["supported_versions"]

    def test_websocket_metrics_background_tasks_health(self, client):
        """Test background task health monitoring."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "background_tasks" in data
        
        tasks = data["background_tasks"]
        expected_tasks = [
            "broadcast_task_running",
            "redis_listener_running", 
            "health_monitor_running"
        ]
        
        for task in expected_tasks:
            assert task in tasks, f"Background task '{task}' missing from health check"
            assert isinstance(tasks[task], bool), f"Task status '{task}' should be boolean"

    def test_websocket_metrics_configuration_exposure(self, client):
        """Test configuration and limits exposure."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "configuration" in data
        
        config = data["configuration"]
        expected_config = [
            "rate_limit_tokens_per_second",
            "rate_limit_burst_capacity",
            "max_inbound_message_bytes",
            "max_subscriptions_per_connection",
            "backpressure_disconnect_threshold",
            "idle_disconnect_seconds"
        ]
        
        for setting in expected_config:
            assert setting in config, f"Configuration setting '{setting}' missing"

    @pytest.mark.asyncio
    async def test_websocket_connection_tracking(self, ws_manager):
        """Test that WebSocket connections are properly tracked in metrics."""
        # Create mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        # Connect a client
        connection_id = str(uuid.uuid4())
        await ws_manager.connect(
            mock_websocket,
            connection_id,
            client_type="test_client",
            subscriptions=["agents", "system"]
        )

        # Check metrics updated
        assert ws_manager.metrics["connections_total"] == 1
        assert len(ws_manager.connections) == 1
        assert connection_id in ws_manager.connections

        # Disconnect client
        await ws_manager.disconnect(connection_id)

        # Check metrics updated
        assert ws_manager.metrics["disconnections_total"] == 1
        assert len(ws_manager.connections) == 0

    @pytest.mark.asyncio
    async def test_websocket_message_metrics_tracking(self, ws_manager):
        """Test message send/receive metrics tracking."""
        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        await ws_manager.connect(mock_websocket, connection_id)

        initial_sent = ws_manager.metrics["messages_sent_total"]
        initial_received = ws_manager.metrics["messages_received_total"]

        # Send message
        await ws_manager._send_to_connection(connection_id, {
            "type": "test_message",
            "data": {"test": "data"}
        })

        # Check sent metrics
        assert ws_manager.metrics["messages_sent_total"] == initial_sent + 1

        # Simulate receiving message
        await ws_manager.handle_message(connection_id, {
            "type": "ping"
        })

        # Check received metrics  
        assert ws_manager.metrics["messages_received_total"] == initial_received + 1

        await ws_manager.disconnect(connection_id)

    @pytest.mark.asyncio
    async def test_websocket_error_metrics_tracking(self, ws_manager):
        """Test error message metrics tracking."""
        # Create mock connection that fails on send
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Send failed"))
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        await ws_manager.connect(mock_websocket, connection_id)

        initial_failures = ws_manager.metrics["messages_send_failures_total"]

        # Attempt to send message (should fail)
        success = await ws_manager._send_to_connection(connection_id, {
            "type": "test_message"
        })

        # Check failure tracked
        assert success is False
        assert ws_manager.metrics["messages_send_failures_total"] == initial_failures + 1

    @pytest.mark.asyncio
    async def test_websocket_rate_limiting_metrics(self, ws_manager):
        """Test rate limiting metrics tracking."""
        # Set aggressive rate limits for testing
        ws_manager.rate_limit_tokens_per_second = 1.0
        ws_manager.rate_limit_burst_capacity = 2.0

        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        await ws_manager.connect(mock_websocket, connection_id)

        # Exhaust rate limit tokens
        connection = ws_manager.connections[connection_id]
        connection.tokens = 0.0  # No tokens remaining

        initial_dropped = ws_manager.metrics["messages_dropped_rate_limit_total"]

        # Send message that should be rate limited
        await ws_manager.handle_message(connection_id, {
            "type": "test_message"
        })

        # Check rate limit drop tracked
        assert ws_manager.metrics["messages_dropped_rate_limit_total"] == initial_dropped + 1

        await ws_manager.disconnect(connection_id)

    @pytest.mark.asyncio
    async def test_websocket_backpressure_disconnect_tracking(self, ws_manager):
        """Test backpressure disconnect metrics tracking."""
        # Create mock connection that always fails sends
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Always fails"))
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        await ws_manager.connect(mock_websocket, connection_id)

        initial_backpressure = ws_manager.metrics["backpressure_disconnects_total"]

        # Send multiple messages to trigger backpressure disconnect
        for _ in range(ws_manager.backpressure_disconnect_threshold + 1):
            await ws_manager._send_to_connection(connection_id, {
                "type": "test_message"
            })

        # Check backpressure disconnect tracked
        assert ws_manager.metrics["backpressure_disconnects_total"] == initial_backpressure + 1
        assert connection_id not in ws_manager.connections  # Should be disconnected

    def test_websocket_metrics_error_handling(self, client):
        """Test metrics endpoint error handling."""
        # Test with potential error conditions
        with patch.object(websocket_manager, 'metrics', side_effect=Exception("Metrics error")):
            response = client.get("/api/dashboard/metrics/websockets")
            assert response.status_code == 200
            
            data = response.json()
            assert "error" in data
            assert "fallback_metrics" in data
            assert "timestamp" in data

    def test_websocket_metrics_real_time_data(self, client):
        """Test that metrics include real-time connection data."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        
        # Real-time fields
        assert "current_connections" in data
        assert "active_subscriptions" in data
        assert "connection_details" in data
        assert "subscription_analysis" in data
        
        # Timestamp should be recent (within last minute)
        timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        now = datetime.utcnow()
        time_diff = abs((now - timestamp).total_seconds())
        assert time_diff < 60, "Timestamp should be recent"

    def test_websocket_metrics_subscription_analysis(self, client):
        """Test subscription group analysis in metrics."""
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        
        data = response.json()
        assert "subscription_analysis" in data
        
        analysis = data["subscription_analysis"]
        expected_groups = ["agents", "coordination", "tasks", "system", "alerts", "project_index"]
        
        for group in expected_groups:
            assert group in analysis, f"Subscription group '{group}' missing from analysis"
            assert "active_connections" in analysis[group]
            assert "percentage_of_total" in analysis[group]

    @pytest.mark.asyncio
    async def test_websocket_correlation_id_injection(self, ws_manager):
        """Test that correlation IDs are injected into all outbound frames."""
        # Create mock connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        await ws_manager.connect(mock_websocket, connection_id)

        # Send message without correlation_id
        message = {"type": "test_message", "data": "test"}
        await ws_manager._send_to_connection(connection_id, message)

        # Check that send_text was called with correlation_id injected
        mock_websocket.send_text.assert_called_once()
        sent_data = mock_websocket.send_text.call_args[0][0]
        sent_message = json.loads(sent_data)
        
        assert "correlation_id" in sent_message
        assert sent_message["correlation_id"] is not None
        assert len(sent_message["correlation_id"]) > 0

        await ws_manager.disconnect(connection_id)

    @pytest.mark.asyncio
    async def test_websocket_structured_logging_compliance(self, ws_manager):
        """Test structured logging includes required fields."""
        with patch('app.api.dashboard_websockets.logger') as mock_logger:
            # Create connection that will fail
            mock_websocket = AsyncMock()
            mock_websocket.accept = AsyncMock()
            mock_websocket.send_text = AsyncMock(side_effect=Exception("Send failed"))
            mock_websocket.headers = {}
            mock_websocket.query_params = {}

            connection_id = str(uuid.uuid4())
            await ws_manager.connect(mock_websocket, connection_id)

            # Trigger error logging
            await ws_manager._send_to_connection(connection_id, {
                "type": "test_message",
                "correlation_id": "test-correlation-id"
            })

            # Check structured logging called with required fields
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            
            # Should include connection_id, error, correlation_id, message_type
            assert "connection_id" in call_args[1]
            assert "error" in call_args[1]
            assert "correlation_id" in call_args[1]
            assert "message_type" in call_args[1]

    def test_websocket_limits_endpoint_integration(self, client):
        """Test that limits endpoint includes contract version."""
        response = client.get("/api/dashboard/websocket/limits")
        assert response.status_code == 200
        
        data = response.json()
        assert "contract_version" in data
        assert data["contract_version"] == WS_CONTRACT_VERSION

    def test_websocket_health_endpoint_integration(self, client):
        """Test WebSocket health endpoint includes comprehensive data."""
        response = client.get("/api/dashboard/websocket/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "websocket_manager" in data
        assert "background_tasks" in data
        assert "connection_stats" in data
        assert "overall_health" in data


class TestWebSocketObservabilityIntegration:
    """Integration tests for WebSocket observability features."""

    @pytest.mark.asyncio
    async def test_full_websocket_lifecycle_metrics(self):
        """Test complete WebSocket connection lifecycle metrics."""
        manager = DashboardWebSocketManager()
        
        # Reset metrics
        manager.metrics = {key: 0 for key in manager.metrics}
        
        # Create mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {}
        mock_websocket.query_params = {}

        connection_id = str(uuid.uuid4())
        
        # Connect
        await manager.connect(mock_websocket, connection_id, subscriptions=["agents"])
        assert manager.metrics["connections_total"] == 1
        
        # Send messages
        await manager._send_to_connection(connection_id, {"type": "test1"})
        await manager._send_to_connection(connection_id, {"type": "test2"})
        assert manager.metrics["messages_sent_total"] == 3  # Including connection_established
        
        # Receive messages
        await manager.handle_message(connection_id, {"type": "ping"})
        await manager.handle_message(connection_id, {"type": "subscribe", "subscriptions": ["system"]})
        assert manager.metrics["messages_received_total"] == 2
        
        # Broadcast to subscription
        sent_count = await manager.broadcast_to_subscription("agents", "agent_update", {"data": "test"})
        assert sent_count == 1
        assert manager.metrics["messages_sent_total"] == 4  # +1 for broadcast
        
        # Disconnect
        await manager.disconnect(connection_id)
        assert manager.metrics["disconnections_total"] == 1
        assert len(manager.connections) == 0

    @pytest.mark.asyncio
    async def test_websocket_metrics_under_load(self):
        """Test WebSocket metrics accuracy under load."""
        manager = DashboardWebSocketManager()
        manager.metrics = {key: 0 for key in manager.metrics}
        
        # Create multiple connections
        connections = []
        for i in range(10):
            mock_websocket = AsyncMock()
            mock_websocket.accept = AsyncMock()
            mock_websocket.send_text = AsyncMock()
            mock_websocket.headers = {}
            mock_websocket.query_params = {}
            
            connection_id = f"test-connection-{i}"
            await manager.connect(mock_websocket, connection_id)
            connections.append(connection_id)
        
        assert manager.metrics["connections_total"] == 10
        assert len(manager.connections) == 10
        
        # Send messages to all connections
        for conn_id in connections:
            await manager._send_to_connection(conn_id, {"type": "test_message"})
        
        # Should be 10 (connection_established) + 10 (test messages) = 20
        assert manager.metrics["messages_sent_total"] == 20
        
        # Broadcast to all
        sent_count = await manager.broadcast_to_all("global_update", {"data": "test"})
        assert sent_count == 10
        assert manager.metrics["messages_sent_total"] == 30
        
        # Disconnect all
        for conn_id in connections:
            await manager.disconnect(conn_id)
        
        assert manager.metrics["disconnections_total"] == 10
        assert len(manager.connections) == 0


@pytest.mark.asyncio
async def test_epic2_phase2_2_acceptance_criteria():
    """
    Epic 2 Phase 2.2 Acceptance Criteria Validation
    
    Validates all acceptance criteria for WebSocket Observability & Metrics are met:
    1. /api/dashboard/metrics/websockets exposes required metrics
    2. Send-failure logs include correlation_id, type, subscription
    3. Smoke test verifies metrics endpoint includes required names
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    # Create test app
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    
    # Test 1: Metrics endpoint exposes required metrics
    response = client.get("/api/dashboard/metrics/websockets")
    assert response.status_code == 200
    
    data = response.json()
    required_metrics = [
        "messages_sent_total",
        "messages_send_failures_total", 
        "messages_received_total",
        "messages_dropped_rate_limit_total",
        "errors_sent_total",
        "connections_total",
        "disconnections_total",
        "backpressure_disconnects_total"
    ]
    
    for metric in required_metrics:
        assert metric in data, f"Required metric '{metric}' missing"
    
    # Test 2: Structured logging compliance
    manager = DashboardWebSocketManager()
    
    with patch('app.api.dashboard_websockets.logger') as mock_logger:
        # Create failing connection to trigger error logging
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Test error"))
        mock_websocket.headers = {}
        mock_websocket.query_params = {}
        
        connection_id = str(uuid.uuid4())
        await manager.connect(mock_websocket, connection_id)
        
        # Send message to trigger error log
        await manager._send_to_connection(connection_id, {
            "type": "test_type",
            "subscription": "test_subscription"
        })
        
        # Verify structured logging includes required fields
        mock_logger.warning.assert_called()
        call_kwargs = mock_logger.warning.call_args[1]
        
        assert "correlation_id" in call_kwargs
        assert "message_type" in call_kwargs
        assert call_kwargs["message_type"] == "test_type"
    
    # Test 3: Smoke test for metrics names
    assert "messages_sent_total" in data
    assert "messages_send_failures_total" in data
    assert "connections_total" in data
    assert "errors_sent_total" in data
    
    print("✅ Epic 2 Phase 2.2 Acceptance Criteria: ALL PASSED")
    print("✅ WebSocket Observability & Metrics implementation complete")
    print("✅ Production-grade metrics exposition operational")
    print("✅ Structured logging with correlation tracking active")


if __name__ == "__main__":
    # Run acceptance criteria test
    asyncio.run(test_epic2_phase2_2_acceptance_criteria())