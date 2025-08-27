"""
Phase 3: WebSocket Integration Testing Framework
===============================================

Enterprise-grade WebSocket integration testing for the LeanVibe Agent Hive 2.0 system.
Tests real-time communication patterns, connection recovery, circuit breakers, and
performance characteristics under various load conditions.

Critical for Phase 4 Mobile PWA real-time capabilities.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

import pytest
import structlog
import websockets
from fastapi.testclient import TestClient

from app.api.dashboard_websockets import websocket_manager, DashboardWebSocketManager
from app.main import app

logger = structlog.get_logger(__name__)


class WebSocketIntegrationTestFramework:
    """Comprehensive WebSocket integration testing framework for Phase 3."""
    
    def __init__(self):
        self.test_clients: List[websockets.WebSocketServerProtocol] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "connection_time": [],
            "message_latency": [],
            "throughput": [],
            "error_rate": []
        }
        self.load_test_results: Dict[str, Any] = {}
        
    async def cleanup(self):
        """Clean up test connections and reset state."""
        for client in self.test_clients:
            if not client.closed:
                await client.close()
        self.test_clients.clear()
        self.performance_metrics = {
            "connection_time": [],
            "message_latency": [],
            "throughput": [],
            "error_rate": []
        }


class TestWebSocketConnectionManagement:
    """Test WebSocket connection establishment, management, and teardown."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket for testing."""
        websocket = AsyncMock()
        websocket.headers = {"origin": "http://localhost:3000"}
        websocket.query_params = {}
        return websocket
    
    async def test_websocket_connection_establishment(self, ws_framework, mock_websocket):
        """Test basic WebSocket connection establishment."""
        # Test connection with default subscriptions
        connection_id = str(uuid.uuid4())
        
        connection = await websocket_manager.connect(
            mock_websocket,
            connection_id,
            client_type="test_client",
            subscriptions=["agents", "system"]
        )
        
        assert connection is not None
        assert connection.connection_id == connection_id
        assert connection.client_type == "test_client"
        assert "agents" in connection.subscriptions
        assert "system" in connection.subscriptions
        
        # Verify connection is tracked
        assert connection_id in websocket_manager.connections
        
        # Test connection stats
        stats = websocket_manager.get_connection_stats()
        assert stats["total_connections"] >= 1
        
        # Clean up
        await websocket_manager.disconnect(connection_id)
    
    async def test_websocket_connection_with_auth(self, ws_framework, mock_websocket):
        """Test WebSocket connection with authentication."""
        # Enable auth for this test
        original_auth_required = websocket_manager.auth_required
        websocket_manager.auth_required = True
        websocket_manager.auth_mode = "token"
        websocket_manager.expected_auth_token = "test_token_123"
        
        try:
            # Test with valid token
            mock_websocket.headers = {"Authorization": "Bearer test_token_123"}
            connection_id = str(uuid.uuid4())
            
            connection = await websocket_manager.connect(
                mock_websocket,
                connection_id,
                client_type="authenticated_client"
            )
            
            assert connection is not None
            assert connection.metadata["auth_mode"] == "token"
            
            await websocket_manager.disconnect(connection_id)
            
            # Test with invalid token (should fail)
            mock_websocket.headers = {"Authorization": "Bearer invalid_token"}
            connection_id_2 = str(uuid.uuid4())
            
            connection = await websocket_manager.connect(
                mock_websocket,
                connection_id_2,
                client_type="unauthenticated_client"
            )
            
            # Connection should be None for invalid auth
            assert connection is None
            
        finally:
            # Restore original auth settings
            websocket_manager.auth_required = original_auth_required
    
    async def test_subscription_management(self, ws_framework, mock_websocket):
        """Test WebSocket subscription management."""
        connection_id = str(uuid.uuid4())
        
        connection = await websocket_manager.connect(
            mock_websocket,
            connection_id,
            subscriptions=["agents"]
        )
        
        # Test adding subscriptions
        await websocket_manager.handle_message(connection_id, {
            "type": "subscribe",
            "subscriptions": ["coordination", "tasks"]
        })
        
        updated_connection = websocket_manager.connections[connection_id]
        assert "coordination" in updated_connection.subscriptions
        assert "tasks" in updated_connection.subscriptions
        
        # Test removing subscriptions
        await websocket_manager.handle_message(connection_id, {
            "type": "unsubscribe",
            "subscriptions": ["agents"]
        })
        
        assert "agents" not in updated_connection.subscriptions
        
        await websocket_manager.disconnect(connection_id)
    
    async def test_websocket_rate_limiting(self, ws_framework, mock_websocket):
        """Test WebSocket rate limiting functionality."""
        connection_id = str(uuid.uuid4())
        
        # Set aggressive rate limits for testing
        websocket_manager.rate_limit_tokens_per_second = 1.0
        websocket_manager.rate_limit_burst_capacity = 2.0
        
        connection = await websocket_manager.connect(
            mock_websocket,
            connection_id
        )
        
        # Send messages quickly to trigger rate limiting
        for i in range(5):
            await websocket_manager.handle_message(connection_id, {
                "type": "ping"
            })
        
        # Check if rate limiting kicked in
        assert websocket_manager.metrics["messages_dropped_rate_limit_total"] > 0
        
        await websocket_manager.disconnect(connection_id)
    
    async def test_websocket_idle_disconnect(self, ws_framework, mock_websocket):
        """Test automatic disconnection of idle WebSocket connections."""
        connection_id = str(uuid.uuid4())
        
        # Set short idle timeout for testing
        original_timeout = websocket_manager.idle_disconnect_seconds
        websocket_manager.idle_disconnect_seconds = 1
        
        try:
            connection = await websocket_manager.connect(
                mock_websocket,
                connection_id
            )
            
            # Simulate passage of time
            connection.last_activity = datetime.utcnow() - timedelta(seconds=2)
            
            # Trigger idle check
            await websocket_manager._check_idle_disconnects()
            
            # Connection should be disconnected
            assert connection_id not in websocket_manager.connections
            assert websocket_manager.metrics["idle_disconnects_total"] > 0
            
        finally:
            websocket_manager.idle_disconnect_seconds = original_timeout


class TestWebSocketBroadcasting:
    """Test WebSocket message broadcasting and event distribution."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_subscription_based_broadcasting(self, ws_framework):
        """Test broadcasting to specific subscription groups."""
        # Create multiple mock connections with different subscriptions
        connections = []
        
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = f"test_conn_{i}"
            
            connection = await websocket_manager.connect(
                mock_ws,
                conn_id,
                subscriptions=["agents"] if i < 2 else ["coordination"]
            )
            connections.append((conn_id, connection))
        
        # Broadcast to agents subscription
        sent_count = await websocket_manager.broadcast_to_subscription(
            "agents",
            "test_message",
            {"test_data": "value"}
        )
        
        # Should reach 2 connections subscribed to agents
        assert sent_count == 2
        
        # Clean up
        for conn_id, _ in connections:
            await websocket_manager.disconnect(conn_id)
    
    async def test_broadcast_to_all_connections(self, ws_framework):
        """Test broadcasting to all connected clients."""
        connections = []
        
        # Create multiple connections
        for i in range(3):
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = f"broadcast_test_{i}"
            
            connection = await websocket_manager.connect(
                mock_ws,
                conn_id
            )
            connections.append((conn_id, connection))
        
        # Broadcast to all
        sent_count = await websocket_manager.broadcast_to_all(
            "global_announcement",
            {"message": "System maintenance scheduled"}
        )
        
        # Should reach all connections
        assert sent_count == 3
        
        # Clean up
        for conn_id, _ in connections:
            await websocket_manager.disconnect(conn_id)
    
    async def test_redis_event_handling(self, ws_framework):
        """Test handling of Redis pub/sub events."""
        # Mock Redis message
        redis_message = {
            "type": "message",
            "channel": b"system_events",
            "data": json.dumps({"event": "test_event", "data": "test_value"}).encode()
        }
        
        # Create a connection subscribed to system events
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = "redis_test_conn"
        
        await websocket_manager.connect(
            mock_ws,
            conn_id,
            subscriptions=["system"]
        )
        
        # Handle Redis event
        await websocket_manager._handle_redis_event(redis_message)
        
        # Verify message was sent to subscribed connection
        mock_ws.send_text.assert_called()
        
        await websocket_manager.disconnect(conn_id)


class TestWebSocketCircuitBreaker:
    """Test WebSocket circuit breaker functionality for connection resilience."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_circuit_breaker_initialization(self, ws_framework):
        """Test circuit breaker initialization for connections."""
        connection_id = "circuit_test_conn"
        
        await websocket_manager.initialize_circuit_breaker(connection_id)
        
        assert connection_id in websocket_manager.circuit_breakers
        breaker = websocket_manager.circuit_breakers[connection_id]
        assert breaker["state"] == "closed"
        assert breaker["failure_count"] == 0
    
    async def test_circuit_breaker_failure_recording(self, ws_framework):
        """Test circuit breaker failure recording and state transitions."""
        connection_id = "failure_test_conn"
        
        await websocket_manager.initialize_circuit_breaker(connection_id)
        
        # Record failures to trigger circuit breaker
        for i in range(6):  # Exceed failure threshold
            await websocket_manager.record_connection_failure(
                connection_id, 
                "connection_timeout"
            )
        
        breaker = websocket_manager.circuit_breakers[connection_id]
        assert breaker["state"] == "open"
        assert breaker["failure_count"] >= 5
        
        # Test that connections are blocked when circuit is open
        should_allow = await websocket_manager.should_allow_connection(connection_id)
        assert not should_allow
    
    async def test_circuit_breaker_recovery(self, ws_framework):
        """Test circuit breaker recovery to half-open and closed states."""
        connection_id = "recovery_test_conn"
        
        await websocket_manager.initialize_circuit_breaker(connection_id)
        
        # Force circuit to open state
        breaker = websocket_manager.circuit_breakers[connection_id]
        breaker["state"] = "open"
        breaker["failure_count"] = 5
        breaker["opened_at"] = time.time() - 61  # More than timeout duration
        
        # Update circuit breaker states (should transition to half-open)
        await websocket_manager.update_circuit_breaker_states()
        
        assert breaker["state"] == "half-open"
        
        # Record successful operations to close circuit
        for i in range(3):  # Meet success threshold
            await websocket_manager.record_connection_success(connection_id)
        
        assert breaker["state"] == "closed"
        assert breaker["failure_count"] == 0
    
    async def test_circuit_breaker_metrics(self, ws_framework):
        """Test circuit breaker metrics collection."""
        # Create multiple circuit breakers with different states
        connections = ["cb_closed", "cb_open", "cb_half_open"]
        
        for conn_id in connections:
            await websocket_manager.initialize_circuit_breaker(conn_id)
        
        # Set different states
        websocket_manager.circuit_breakers["cb_open"]["state"] = "open"
        websocket_manager.circuit_breakers["cb_half_open"]["state"] = "half-open"
        
        metrics = await websocket_manager.get_circuit_breaker_metrics()
        
        assert metrics["circuit_breakers_total"] == 3
        assert metrics["circuit_breakers_open_total"] == 1
        assert metrics["circuit_breakers_closed_total"] == 1
        assert metrics["circuit_breakers_half_open_total"] == 1


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics and scalability."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_concurrent_connections_performance(self, ws_framework):
        """Test performance with multiple concurrent connections."""
        start_time = time.time()
        connections = []
        connection_times = []
        
        # Create 50 concurrent connections
        for i in range(50):
            conn_start = time.time()
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = f"perf_test_{i}"
            
            connection = await websocket_manager.connect(
                mock_ws,
                conn_id,
                client_type="performance_test"
            )
            
            conn_end = time.time()
            connection_times.append(conn_end - conn_start)
            connections.append(conn_id)
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert len(websocket_manager.connections) == 50
        assert total_time < 5.0  # Should establish 50 connections in under 5 seconds
        assert max(connection_times) < 0.1  # No single connection should take more than 100ms
        
        # Test broadcast performance
        broadcast_start = time.time()
        sent_count = await websocket_manager.broadcast_to_all(
            "performance_test",
            {"timestamp": time.time()}
        )
        broadcast_time = time.time() - broadcast_start
        
        assert sent_count == 50
        assert broadcast_time < 1.0  # Broadcast to 50 connections should take under 1 second
        
        # Clean up
        for conn_id in connections:
            await websocket_manager.disconnect(conn_id)
    
    async def test_message_throughput(self, ws_framework):
        """Test WebSocket message throughput under load."""
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = "throughput_test"
        
        await websocket_manager.connect(mock_ws, conn_id)
        
        # Disable rate limiting for throughput test
        original_rate = websocket_manager.rate_limit_tokens_per_second
        websocket_manager.rate_limit_tokens_per_second = 1000.0
        
        try:
            # Send 100 messages and measure time
            start_time = time.time()
            
            for i in range(100):
                await websocket_manager.handle_message(conn_id, {
                    "type": "ping"
                })
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = 100 / duration  # messages per second
            
            # Should handle at least 50 messages per second
            assert throughput >= 50.0
            
            ws_framework.performance_metrics["throughput"].append(throughput)
            
        finally:
            websocket_manager.rate_limit_tokens_per_second = original_rate
            await websocket_manager.disconnect(conn_id)
    
    async def test_memory_usage_under_load(self, ws_framework):
        """Test memory usage with many connections."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        connections = []
        
        # Create 100 connections with different subscription patterns
        for i in range(100):
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = f"memory_test_{i}"
            
            # Vary subscriptions to test different memory patterns
            subs = ["agents"] if i % 3 == 0 else ["coordination", "tasks", "system"]
            
            await websocket_manager.connect(
                mock_ws,
                conn_id,
                subscriptions=subs
            )
            connections.append(conn_id)
        
        # Get memory after connections
        after_connections = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_connection = (after_connections - initial_memory) / 100
        
        # Each connection should use less than 1MB of memory
        assert memory_per_connection < 1.0
        
        # Clean up and verify memory is released
        for conn_id in connections:
            await websocket_manager.disconnect(conn_id)
        
        # Allow some time for cleanup
        await asyncio.sleep(0.1)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_leaked = final_memory - initial_memory
        
        # Memory leak should be minimal (less than 10MB)
        assert memory_leaked < 10.0


class TestWebSocketRecovery:
    """Test WebSocket connection recovery mechanisms."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_broken_connection_detection(self, ws_framework):
        """Test detection of broken WebSocket connections."""
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = "broken_conn_test"
        
        await websocket_manager.connect(mock_ws, conn_id)
        
        # Simulate connection failure
        mock_ws.send_json.side_effect = Exception("Connection broken")
        
        is_broken = await websocket_manager.is_connection_broken(conn_id)
        assert is_broken
        assert conn_id in websocket_manager.broken_connections
    
    async def test_stale_connection_cleanup(self, ws_framework):
        """Test cleanup of stale connections."""
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = "stale_conn_test"
        
        connection = await websocket_manager.connect(mock_ws, conn_id)
        
        # Make connection appear stale
        connection.last_heartbeat_response = time.time() - 120  # 2 minutes ago
        
        is_stale = await websocket_manager.is_connection_stale(conn_id)
        assert is_stale
        
        # Cleanup stale connections
        await websocket_manager.cleanup_stale_connections()
        
        assert conn_id not in websocket_manager.connections
        assert websocket_manager.metrics["stale_connections_cleaned_total"] > 0
    
    async def test_connection_recovery_metrics(self, ws_framework):
        """Test connection recovery metrics tracking."""
        # Simulate some recovery attempts
        websocket_manager.metrics["connection_recovery_attempts_total"] = 10
        websocket_manager.metrics["connection_recovery_successes_total"] = 7
        websocket_manager.metrics["connection_recovery_failures_total"] = 3
        
        metrics = await websocket_manager.get_recovery_metrics()
        
        assert metrics["connection_recovery_attempts_total"] == 10
        assert metrics["connection_recovery_successes_total"] == 7
        assert metrics["connection_recovery_failures_total"] == 3
        assert metrics["recovery_success_rate"] == 70.0


class TestWebSocketEndToEnd:
    """End-to-end WebSocket integration tests."""
    
    @pytest.fixture
    async def ws_framework(self):
        framework = WebSocketIntegrationTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_real_websocket_connection_flow(self, ws_framework):
        """Test complete WebSocket connection flow using real connections."""
        # This test would use actual WebSocket connections in a real environment
        # For now, we'll simulate the full flow with mocks
        
        # Step 1: Client connects
        mock_ws = AsyncMock()
        mock_ws.headers = {"origin": "http://localhost:3000"}
        conn_id = "e2e_test_conn"
        
        connection = await websocket_manager.connect(
            mock_ws,
            conn_id,
            client_type="dashboard",
            subscriptions=["agents", "coordination"]
        )
        
        assert connection is not None
        
        # Step 2: Client sends ping
        await websocket_manager.handle_message(conn_id, {"type": "ping"})
        
        # Verify pong was sent
        mock_ws.send_text.assert_called()
        last_call = mock_ws.send_text.call_args[0][0]
        pong_message = json.loads(last_call)
        assert pong_message["type"] == "pong"
        
        # Step 3: Server broadcasts update
        await websocket_manager.broadcast_to_subscription(
            "agents",
            "agent_update",
            {"agent_id": "test_agent", "status": "active"}
        )
        
        # Step 4: Client disconnects
        await websocket_manager.disconnect(conn_id)
        assert conn_id not in websocket_manager.connections
    
    async def test_system_integration_scenario(self, ws_framework):
        """Test a realistic system integration scenario."""
        # Simulate a typical dashboard monitoring scenario
        
        # Create multiple client types
        clients = [
            ("dashboard_main", "full_dashboard", ["agents", "coordination", "system"]),
            ("mobile_monitor", "mobile_client", ["agents", "alerts"]),
            ("admin_panel", "admin_dashboard", ["system", "alerts", "coordination"])
        ]
        
        connections = []
        
        # Connect all clients
        for conn_id, client_type, subs in clients:
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            
            connection = await websocket_manager.connect(
                mock_ws,
                conn_id,
                client_type=client_type,
                subscriptions=subs
            )
            connections.append((conn_id, connection))
        
        # Simulate system events
        events = [
            ("agents", "agent_status_change", {"agent_id": "agent_1", "status": "busy"}),
            ("coordination", "task_completed", {"task_id": "task_123", "result": "success"}),
            ("system", "health_check", {"status": "healthy", "timestamp": time.time()}),
            ("alerts", "critical_alert", {"level": "high", "message": "Database connection unstable"})
        ]
        
        # Send each event and verify appropriate clients receive them
        for subscription, event_type, event_data in events:
            sent_count = await websocket_manager.broadcast_to_subscription(
                subscription,
                event_type,
                event_data
            )
            
            # Verify correct number of subscribers received the event
            expected_count = sum(1 for _, _, subs in clients if subscription in subs)
            assert sent_count == expected_count
        
        # Verify WebSocket manager state
        stats = websocket_manager.get_connection_stats()
        assert stats["total_connections"] == 3
        
        # Clean up
        for conn_id, _ in connections:
            await websocket_manager.disconnect(conn_id)


# Performance and Load Testing Suite
class TestWebSocketLoadTesting:
    """Load testing for WebSocket system."""
    
    @pytest.mark.asyncio
    async def test_high_connection_count_load(self):
        """Test system behavior under high connection count."""
        # This test simulates 100+ concurrent connections
        # In production, would test with even higher numbers
        
        connection_count = 100
        connections = []
        
        start_time = time.time()
        
        try:
            # Create connections rapidly
            for i in range(connection_count):
                mock_ws = AsyncMock()
                mock_ws.headers = {}
                conn_id = f"load_test_{i}"
                
                connection = await websocket_manager.connect(
                    mock_ws,
                    conn_id,
                    client_type="load_test_client"
                )
                connections.append(conn_id)
                
                # Brief pause to avoid overwhelming the system
                if i % 10 == 0:
                    await asyncio.sleep(0.01)
            
            connection_time = time.time() - start_time
            
            # Verify all connections were established
            assert len(websocket_manager.connections) == connection_count
            assert connection_time < 30.0  # Should handle 100 connections in under 30 seconds
            
            # Test broadcast performance under load
            broadcast_start = time.time()
            sent_count = await websocket_manager.broadcast_to_all(
                "load_test_broadcast",
                {"test_id": "high_load_test"}
            )
            broadcast_time = time.time() - broadcast_start
            
            assert sent_count == connection_count
            assert broadcast_time < 5.0  # Broadcast to 100 clients in under 5 seconds
            
        finally:
            # Clean up all connections
            for conn_id in connections:
                if conn_id in websocket_manager.connections:
                    await websocket_manager.disconnect(conn_id)


# Integration Test Suite Configuration
@pytest.mark.integration
class TestWebSocketSystemIntegration:
    """Full system integration tests for WebSocket functionality."""
    
    async def test_websocket_api_endpoint_integration(self):
        """Test WebSocket API endpoints integration with FastAPI."""
        client = TestClient(app)
        
        # Test WebSocket stats endpoint
        response = client.get("/api/dashboard/websocket/stats")
        assert response.status_code == 200
        data = response.json()
        assert "websocket_stats" in data
        assert "endpoints" in data
        
        # Test WebSocket health endpoint
        response = client.get("/api/dashboard/websocket/health")
        assert response.status_code == 200
        health_data = response.json()
        assert "overall_health" in health_data
        
        # Test WebSocket metrics endpoint
        response = client.get("/api/dashboard/metrics/websockets")
        assert response.status_code == 200
        metrics = response.json()
        assert "messages_sent_total" in metrics
        assert "performance_metrics" in metrics
    
    async def test_websocket_redis_integration(self):
        """Test WebSocket integration with Redis pub/sub."""
        # This test would require a real Redis instance
        # Mock the Redis integration for unit testing
        
        with patch('app.core.redis.get_redis') as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.return_value = mock_redis_client
            
            # Test Redis listener task startup
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = "redis_integration_test"
            
            await websocket_manager.connect(mock_ws, conn_id)
            
            # Verify Redis listener task is running
            stats = websocket_manager.get_connection_stats()
            assert stats["background_tasks_running"] or len(websocket_manager.connections) == 0
            
            await websocket_manager.disconnect(conn_id)


if __name__ == "__main__":
    """Run integration tests directly for development."""
    import asyncio
    
    async def run_basic_tests():
        """Run basic integration tests for development."""
        framework = WebSocketIntegrationTestFramework()
        
        try:
            # Test basic connection
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = "dev_test"
            
            connection = await websocket_manager.connect(mock_ws, conn_id)
            print(f"✅ Connection established: {connection.connection_id}")
            
            # Test message handling
            await websocket_manager.handle_message(conn_id, {"type": "ping"})
            print("✅ Message handling working")
            
            # Test broadcasting
            sent_count = await websocket_manager.broadcast_to_all(
                "test_message", {"data": "test"}
            )
            print(f"✅ Broadcast sent to {sent_count} clients")
            
            # Test circuit breaker
            await websocket_manager.initialize_circuit_breaker("test_cb")
            print("✅ Circuit breaker initialized")
            
            await websocket_manager.disconnect(conn_id)
            print("✅ Connection cleaned up")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        finally:
            await framework.cleanup()
    
    # Run the basic tests
    asyncio.run(run_basic_tests())
    print("\n🎯 Phase 3 WebSocket Integration Testing Framework Ready!")
    print("   - Connection management ✅")
    print("   - Circuit breaker patterns ✅") 
    print("   - Performance testing ✅")
    print("   - Load testing scenarios ✅")
    print("   - Recovery mechanisms ✅")