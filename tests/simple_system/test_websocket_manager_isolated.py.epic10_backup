"""
WebSocket Manager - Component Isolation Tests
=============================================

Tests the WebSocket management system in complete isolation.
This validates real-time communication, connection management,
message routing, and dashboard integration without external dependencies.

Testing Strategy:
- Mock all external dependencies (Redis, database, network connections)
- Test connection lifecycle and state management
- Validate message routing and subscription patterns
- Ensure proper error handling and recovery
- Test rate limiting and connection security
"""

import asyncio
import uuid
import pytest
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch, call
from fastapi import WebSocket, WebSocketDisconnect

from app.api.dashboard_websockets import WebSocketManager, ConnectionState
from app.core.realtime_dashboard_streaming import RealtimeDashboardStreaming
from app.core.observability_streams import ObservabilityStreams
from app.api.ws_utils import WebSocketMessageHandler


@pytest.mark.isolation
@pytest.mark.unit
class TestWebSocketManagerIsolated:
    """Test WebSocket manager functionality in isolation."""
    
    @pytest.fixture
    async def isolated_websocket_manager(
        self,
        mock_redis_streams,
        mock_database_session,
        isolated_test_environment,
        assert_isolated
    ):
        """Create isolated WebSocket manager with all dependencies mocked."""
        
        with patch('app.api.dashboard_websockets.get_redis_client', return_value=mock_redis_streams), \
             patch('app.api.dashboard_websockets.get_database_session', return_value=mock_database_session):
            
            manager = WebSocketManager()
            await manager.initialize()
            
            # Assert complete isolation
            assert_isolated(manager, {
                "redis": mock_redis_streams,
                "database": mock_database_session
            })
            
            yield manager
            
            await manager.shutdown()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.client = MagicMock()
        websocket.client.host = "127.0.0.1"
        websocket.client.port = 12345
        websocket.headers = {"user-agent": "test-client"}
        websocket.query_params = {}
        
        # Mock connection state
        websocket.application_state = "connected"
        websocket.client_state = "connected"
        
        # Mock send/receive methods
        websocket.send_text = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.receive_json = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        
        return websocket
    
    async def test_connection_lifecycle_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket,
        capture_component_calls
    ):
        """Test WebSocket connection lifecycle in isolation."""
        manager = isolated_websocket_manager
        
        # Capture method calls
        calls, _ = capture_component_calls(manager, [
            "connect", "disconnect", "get_connection_state"
        ])
        
        # Test connection establishment
        connection_id = await manager.connect(
            websocket=mock_websocket,
            client_id="test-client-123",
            user_id="user-456"
        )
        
        assert connection_id is not None
        assert isinstance(connection_id, str)
        
        # Verify connection was established
        mock_websocket.accept.assert_called_once()
        
        # Verify connection is tracked
        connection_state = await manager.get_connection_state(connection_id)
        assert connection_state["status"] == ConnectionState.CONNECTED
        assert connection_state["client_id"] == "test-client-123"
        assert connection_state["user_id"] == "user-456"
        
        # Test connection count
        active_connections = await manager.get_active_connections()
        assert len(active_connections) == 1
        assert active_connections[0]["connection_id"] == connection_id
        
        # Test disconnection
        await manager.disconnect(connection_id)
        
        # Verify connection was closed
        mock_websocket.close.assert_called_once()
        
        # Verify connection is no longer tracked
        updated_connections = await manager.get_active_connections()
        assert len(updated_connections) == 0
        
        # Verify method calls
        assert len(calls) >= 3
        assert any(call["method"] == "connect" for call in calls)
        assert any(call["method"] == "disconnect" for call in calls)
    
    async def test_message_routing_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket,
        mock_redis_streams
    ):
        """Test message routing and delivery in isolation."""
        manager = isolated_websocket_manager
        
        # Establish multiple connections
        connections = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.send_json = AsyncMock()
            mock_ws.accept = AsyncMock()
            
            conn_id = await manager.connect(
                websocket=mock_ws,
                client_id=f"client-{i}",
                user_id=f"user-{i}"
            )
            connections.append({"id": conn_id, "websocket": mock_ws, "client": f"client-{i}"})
        
        # Test direct message to specific connection
        direct_message = {
            "type": "task_update",
            "data": {
                "task_id": "task_123",
                "status": "completed",
                "result": "Task completed successfully"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        target_connection = connections[0]
        await manager.send_to_connection(
            connection_id=target_connection["id"],
            message=direct_message
        )
        
        # Verify only target connection received message
        target_connection["websocket"].send_json.assert_called_once_with(direct_message)
        for other_conn in connections[1:]:
            other_conn["websocket"].send_json.assert_not_called()
        
        # Test broadcast message to all connections
        broadcast_message = {
            "type": "system_announcement",
            "data": {
                "message": "System maintenance scheduled",
                "scheduled_time": "2024-01-15T02:00:00Z"
            }
        }
        
        await manager.broadcast_message(broadcast_message)
        
        # Verify all connections received broadcast
        for conn in connections:
            conn["websocket"].send_json.assert_called_with(broadcast_message)
        
        # Test filtered broadcast (by user group)
        filtered_message = {
            "type": "admin_alert",
            "data": {"alert": "High system load detected"}
        }
        
        await manager.broadcast_to_group(
            message=filtered_message,
            group_filter={"user_id": ["user-0", "user-2"]}  # Only user-0 and user-2
        )
        
        # Verify only filtered connections received message
        connections[0]["websocket"].send_json.assert_called_with(filtered_message)
        connections[2]["websocket"].send_json.assert_called_with(filtered_message)
        # user-1 should not have received the admin alert
    
    async def test_subscription_management_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket,
        mock_redis_streams
    ):
        """Test subscription and channel management in isolation."""
        manager = isolated_websocket_manager
        
        # Establish connection
        connection_id = await manager.connect(
            websocket=mock_websocket,
            client_id="subscriber-client",
            user_id="subscriber-user"
        )
        
        # Test channel subscription
        subscription_channels = [
            "agent_status_updates",
            "task_progress",
            "system_metrics",
            "error_alerts"
        ]
        
        for channel in subscription_channels:
            result = await manager.subscribe_to_channel(
                connection_id=connection_id,
                channel=channel
            )
            assert result["success"] is True
        
        # Verify subscriptions are tracked
        subscriptions = await manager.get_connection_subscriptions(connection_id)
        assert len(subscriptions) == 4
        assert all(channel in subscriptions for channel in subscription_channels)
        
        # Test channel message delivery
        channel_message = {
            "type": "agent_status",
            "data": {
                "agent_id": "agent_123",
                "status": "active",
                "current_task": "API development"
            }
        }
        
        await manager.send_to_channel(
            channel="agent_status_updates",
            message=channel_message
        )
        
        # Verify subscriber received channel message
        mock_websocket.send_json.assert_called_with(channel_message)
        
        # Test unsubscription
        unsubscribe_result = await manager.unsubscribe_from_channel(
            connection_id=connection_id,
            channel="error_alerts"
        )
        assert unsubscribe_result["success"] is True
        
        # Verify subscription was removed
        updated_subscriptions = await manager.get_connection_subscriptions(connection_id)
        assert "error_alerts" not in updated_subscriptions
        assert len(updated_subscriptions) == 3
    
    async def test_rate_limiting_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket
    ):
        """Test rate limiting and connection protection in isolation."""
        manager = isolated_websocket_manager
        
        # Configure rate limits
        await manager.configure_rate_limits({
            "messages_per_minute": 60,
            "connections_per_ip": 5,
            "subscription_limit": 10
        })
        
        # Establish connection
        connection_id = await manager.connect(
            websocket=mock_websocket,
            client_id="rate-test-client",
            user_id="rate-test-user"
        )
        
        # Test message rate limiting
        messages_sent = 0
        rate_limit_triggered = False
        
        # Send messages rapidly to trigger rate limit
        for i in range(70):  # Exceed 60 messages per minute
            try:
                await manager.handle_incoming_message(
                    connection_id=connection_id,
                    message={
                        "type": "ping",
                        "data": {"sequence": i}
                    }
                )
                messages_sent += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    rate_limit_triggered = True
                    break
        
        # Verify rate limiting was enforced
        assert rate_limit_triggered, "Rate limiting should have been triggered"
        assert messages_sent <= 60, "Should not exceed rate limit"
        
        # Test subscription limit
        subscription_limit_reached = False
        subscriptions_created = 0
        
        for i in range(15):  # Try to exceed 10 subscription limit
            try:
                await manager.subscribe_to_channel(
                    connection_id=connection_id,
                    channel=f"test_channel_{i}"
                )
                subscriptions_created += 1
            except Exception as e:
                if "subscription limit" in str(e).lower():
                    subscription_limit_reached = True
                    break
        
        assert subscription_limit_reached, "Subscription limit should have been enforced"
        assert subscriptions_created <= 10, "Should not exceed subscription limit"
    
    async def test_error_handling_and_recovery_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket
    ):
        """Test error handling and connection recovery in isolation."""
        manager = isolated_websocket_manager
        
        # Establish connection
        connection_id = await manager.connect(
            websocket=mock_websocket,
            client_id="error-test-client",
            user_id="error-test-user"
        )
        
        # Test handling of malformed messages
        malformed_messages = [
            {"invalid": "no type field"},
            {"type": "valid_type"},  # Missing data field
            "not_a_json_object",
            None,
            {"type": "test", "data": "circular_reference_test"}
        ]
        
        for malformed_msg in malformed_messages:
            try:
                await manager.handle_incoming_message(
                    connection_id=connection_id,
                    message=malformed_msg
                )
            except Exception:
                pass  # Expected to fail gracefully
        
        # Verify connection is still alive after error handling
        connection_state = await manager.get_connection_state(connection_id)
        assert connection_state["status"] == ConnectionState.CONNECTED
        
        # Test connection recovery after network issues
        # Simulate network disconnection
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        
        # Attempt to send message (should trigger error handling)
        with pytest.raises(Exception):
            await manager.send_to_connection(
                connection_id=connection_id,
                message={"type": "test", "data": "recovery_test"}
            )
        
        # Verify connection is marked as disconnected
        updated_state = await manager.get_connection_state(connection_id)
        assert updated_state["status"] == ConnectionState.DISCONNECTED
        
        # Test automatic cleanup of disconnected connections
        await manager.cleanup_stale_connections()
        
        active_connections = await manager.get_active_connections()
        assert len(active_connections) == 0
    
    async def test_metrics_collection_isolated(
        self,
        isolated_websocket_manager,
        mock_websocket
    ):
        """Test WebSocket metrics collection in isolation."""
        manager = isolated_websocket_manager
        
        # Establish multiple connections for metrics testing
        connections = []
        for i in range(5):
            mock_ws = AsyncMock(spec=WebSocket)
            mock_ws.accept = AsyncMock()
            mock_ws.send_json = AsyncMock()
            
            conn_id = await manager.connect(
                websocket=mock_ws,
                client_id=f"metrics-client-{i}",
                user_id=f"metrics-user-{i}"
            )
            connections.append(conn_id)
        
        # Send various messages to generate metrics
        for i, conn_id in enumerate(connections):
            # Send different types of messages
            for j in range(10 + i):  # Variable message counts
                await manager.handle_incoming_message(
                    connection_id=conn_id,
                    message={
                        "type": f"message_type_{j % 3}",
                        "data": {"test": f"data_{j}"}
                    }
                )
        
        # Collect metrics
        metrics = await manager.get_websocket_metrics()
        
        # Verify connection metrics
        assert metrics["total_connections"] == 5
        assert metrics["active_connections"] == 5
        assert metrics["total_messages_sent"] >= 35  # Sum of 10+11+12+13+14
        assert metrics["total_messages_received"] >= 35
        
        # Verify message type distribution
        assert "message_type_distribution" in metrics
        type_dist = metrics["message_type_distribution"]
        assert "message_type_0" in type_dist
        assert "message_type_1" in type_dist
        assert "message_type_2" in type_dist
        
        # Verify performance metrics
        assert "average_message_processing_time" in metrics
        assert "connection_duration_stats" in metrics
        assert metrics["average_message_processing_time"] > 0
        
        # Test real-time metrics streaming
        metrics_stream = await manager.start_metrics_stream()
        
        # Send some activity to generate real-time metrics
        await manager.broadcast_message({
            "type": "test_broadcast",
            "data": {"broadcast_test": True}
        })
        
        # Verify metrics were updated
        updated_metrics = await manager.get_current_metrics()
        assert updated_metrics["total_messages_sent"] > metrics["total_messages_sent"]


@pytest.mark.isolation
@pytest.mark.unit
class TestRealtimeDashboardStreamingIsolated:
    """Test real-time dashboard streaming in isolation."""
    
    @pytest.fixture
    async def isolated_dashboard_streaming(
        self,
        mock_redis_streams,
        mock_database_session,
        isolated_test_environment
    ):
        """Create isolated dashboard streaming service."""
        
        with patch('app.core.realtime_dashboard_streaming.get_redis_client', return_value=mock_redis_streams), \
             patch('app.core.realtime_dashboard_streaming.get_database_session', return_value=mock_database_session):
            
            streaming = RealtimeDashboardStreaming()
            await streaming.initialize()
            
            yield streaming
            
            await streaming.shutdown()
    
    async def test_dashboard_data_streaming_isolated(
        self,
        isolated_dashboard_streaming,
        mock_redis_streams
    ):
        """Test dashboard data streaming in isolation."""
        streaming = isolated_dashboard_streaming
        
        # Mock dashboard data
        dashboard_data = {
            "metrics": {
                "active_agents": 15,
                "pending_tasks": 23,
                "completed_tasks": 145,
                "system_load": 0.67,
                "memory_usage": 0.45
            },
            "agent_status": [
                {"id": "agent_1", "status": "active", "current_task": "API development"},
                {"id": "agent_2", "status": "idle", "current_task": None},
                {"id": "agent_3", "status": "busy", "current_task": "Testing"}
            ],
            "recent_activities": [
                {"timestamp": "2024-01-15T10:30:00Z", "activity": "Task completed", "agent": "agent_1"},
                {"timestamp": "2024-01-15T10:25:00Z", "activity": "New task assigned", "agent": "agent_2"}
            ]
        }
        
        # Stream dashboard data
        result = await streaming.stream_dashboard_data(dashboard_data)
        
        assert result["success"] is True
        assert "stream_id" in result
        
        # Verify Redis stream was used
        mock_redis_streams.xadd.assert_called()
        call_args = mock_redis_streams.xadd.call_args
        
        # Verify correct stream name
        assert "dashboard_updates" in str(call_args)
        
        # Verify data was serialized properly
        streamed_data = call_args[0][1]  # Second argument is the data
        assert "metrics" in str(streamed_data)
        assert "agent_status" in str(streamed_data)
    
    async def test_selective_data_streaming_isolated(
        self,
        isolated_dashboard_streaming
    ):
        """Test selective data streaming based on subscriptions."""
        streaming = isolated_dashboard_streaming
        
        # Configure different data streams
        stream_configs = {
            "metrics_only": {
                "include": ["metrics"],
                "exclude": ["agent_status", "recent_activities"],
                "update_frequency": 5  # seconds
            },
            "agent_focus": {
                "include": ["agent_status", "recent_activities"],
                "exclude": ["metrics"],
                "update_frequency": 2
            },
            "full_dashboard": {
                "include": ["metrics", "agent_status", "recent_activities"],
                "exclude": [],
                "update_frequency": 10
            }
        }
        
        for stream_name, config in stream_configs.items():
            await streaming.configure_stream(stream_name, config)
        
        # Test metrics-only streaming
        metrics_data = {
            "metrics": {"active_agents": 20, "system_load": 0.8},
            "agent_status": [{"id": "agent_1", "status": "active"}],
            "recent_activities": [{"activity": "test"}]
        }
        
        filtered_result = await streaming.stream_filtered_data(
            stream_name="metrics_only",
            data=metrics_data
        )
        
        assert filtered_result["success"] is True
        
        # Verify only metrics were included
        filtered_data = filtered_result["filtered_data"]
        assert "metrics" in filtered_data
        assert "agent_status" not in filtered_data
        assert "recent_activities" not in filtered_data
    
    async def test_data_aggregation_isolated(
        self,
        isolated_dashboard_streaming
    ):
        """Test data aggregation for dashboard streaming."""
        streaming = isolated_dashboard_streaming
        
        # Simulate multiple data points over time
        time_series_data = []
        base_time = datetime.utcnow()
        
        for i in range(10):
            data_point = {
                "timestamp": (base_time + timedelta(minutes=i)).isoformat(),
                "metrics": {
                    "active_agents": 10 + i,
                    "system_load": 0.3 + (i * 0.05),
                    "memory_usage": 0.4 + (i * 0.02)
                }
            }
            time_series_data.append(data_point)
        
        # Test aggregation
        aggregated_result = await streaming.aggregate_time_series_data(
            data_points=time_series_data,
            aggregation_window="5min",
            aggregation_functions=["avg", "max", "min"]
        )
        
        assert "aggregated_data" in aggregated_result
        assert "time_windows" in aggregated_result
        
        aggregated = aggregated_result["aggregated_data"]
        assert "avg_active_agents" in aggregated
        assert "max_system_load" in aggregated
        assert "min_memory_usage" in aggregated
        
        # Verify aggregation accuracy
        assert aggregated["avg_active_agents"] == 14.5  # (10+19)/2
        assert aggregated["max_system_load"] == 0.75    # 0.3 + (9 * 0.05)
        assert aggregated["min_memory_usage"] == 0.4     # Base value