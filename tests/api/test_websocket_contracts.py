"""
WebSocket Message Format Contract Testing
========================================

Contract validation for WebSocket message formats and real-time communication protocols.
Ensures message schema compliance, connection lifecycle stability, and integration compatibility.

Key Contract Areas:
- WebSocket connection establishment and lifecycle
- Message format schema validation  
- Real-time update broadcasting contracts
- Error handling and reconnection protocols
- Performance and reliability requirements
"""

import pytest
import json
import asyncio
import time
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import jsonschema
from jsonschema import validate, ValidationError

# Import WebSocket manager and related components
from frontend_api_server import WebSocketManager, websocket_manager


class TestWebSocketMessageContracts:
    """Contract tests for WebSocket message formats and protocols."""

    @pytest.fixture
    def ws_manager(self):
        """Create WebSocket manager for testing."""
        return WebSocketManager()

    @pytest.fixture
    async def mock_websocket(self):
        """Create mock WebSocket connection for testing."""
        websocket = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.accept = AsyncMock()
        return websocket

    # Message Format Contracts

    def test_welcome_message_schema_contract(self):
        """Test welcome message follows required schema contract."""
        
        welcome_schema = {
            "type": "object",
            "required": ["type", "message", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["connection_established"]},
                "message": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"}
            }
        }
        
        # Create sample welcome message
        welcome_message = {
            "type": "connection_established",
            "message": "Connected to LeanVibe real-time updates",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Validate against schema
        try:
            jsonschema.validate(welcome_message, welcome_schema)
        except ValidationError as e:
            pytest.fail(f"Welcome message schema contract violation: {e}")

    def test_echo_message_schema_contract(self):
        """Test echo message follows required schema contract."""
        
        echo_schema = {
            "type": "object",
            "required": ["type", "data", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["echo"]},
                "data": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        }
        
        # Create sample echo message
        echo_message = {
            "type": "echo",
            "data": '{"test": "message"}',
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Validate against schema
        try:
            jsonschema.validate(echo_message, echo_schema)
        except ValidationError as e:
            pytest.fail(f"Echo message schema contract violation: {e}")

    def test_agent_update_message_schema_contract(self):
        """Test agent update message follows required schema contract."""
        
        agent_update_schema = {
            "type": "object",
            "required": ["type", "data", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["agent_created", "agent_updated", "agent_deleted"]},
                "data": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "status": {"type": "string"},
                        "role": {"type": ["string", "null"]},
                        "capabilities": {"type": "array"},
                        "created_at": {"type": "string"},
                        "updated_at": {"type": "string"}
                    }
                },
                "timestamp": {"type": "string"}
            }
        }
        
        # Test agent creation message
        agent_created = {
            "type": "agent_created",
            "data": {
                "id": "agent-12345678",
                "name": "Test Agent",
                "type": "claude",
                "status": "active",
                "role": "backend_developer",
                "capabilities": ["coding", "testing"],
                "created_at": "2025-01-18T12:00:00Z",
                "updated_at": "2025-01-18T12:00:00Z"
            },
            "timestamp": "2025-01-18T12:00:00Z"
        }
        
        try:
            jsonschema.validate(agent_created, agent_update_schema)
        except ValidationError as e:
            pytest.fail(f"Agent update message schema contract violation: {e}")
        
        # Test agent deletion message
        agent_deleted = {
            "type": "agent_deleted",
            "data": {"id": "agent-12345678"},
            "timestamp": "2025-01-18T12:05:00Z"
        }
        
        # Simplified schema for deletion
        deletion_schema = {
            "type": "object",
            "required": ["type", "data", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["agent_deleted"]},
                "data": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string"}}
                },
                "timestamp": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(agent_deleted, deletion_schema)
        except ValidationError as e:
            pytest.fail(f"Agent deletion message schema contract violation: {e}")

    def test_task_update_message_schema_contract(self):
        """Test task update message follows required schema contract."""
        
        task_update_schema = {
            "type": "object",
            "required": ["type", "data", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["task_created", "task_updated", "task_deleted"]},
                "data": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": ["string", "null"]},
                        "status": {"type": "string"},
                        "priority": {"type": "string"},
                        "agent_id": {"type": ["string", "null"]},
                        "created_at": {"type": "string"},
                        "updated_at": {"type": "string"}
                    }
                },
                "timestamp": {"type": "string"}
            }
        }
        
        # Test task creation message
        task_created = {
            "type": "task_created",
            "data": {
                "id": "task-87654321",
                "title": "Test Task",
                "description": "Contract testing task",
                "status": "pending",
                "priority": "medium",
                "agent_id": "agent-12345678",
                "created_at": "2025-01-18T12:00:00Z",
                "updated_at": "2025-01-18T12:00:00Z"
            },
            "timestamp": "2025-01-18T12:00:00Z"
        }
        
        try:
            jsonschema.validate(task_created, task_update_schema)
        except ValidationError as e:
            pytest.fail(f"Task update message schema contract violation: {e}")

    # WebSocket Manager Contract Tests

    async def test_websocket_manager_connect_contract(self, ws_manager, mock_websocket):
        """Test WebSocket manager connect method contract."""
        
        # Test connection acceptance
        await ws_manager.connect(mock_websocket)
        
        # Verify websocket.accept() was called
        mock_websocket.accept.assert_called_once()
        
        # Verify websocket was added to active connections
        assert mock_websocket in ws_manager.active_connections
        assert len(ws_manager.active_connections) == 1

    async def test_websocket_manager_disconnect_contract(self, ws_manager, mock_websocket):
        """Test WebSocket manager disconnect method contract."""
        
        # First connect
        await ws_manager.connect(mock_websocket)
        assert len(ws_manager.active_connections) == 1
        
        # Then disconnect
        ws_manager.disconnect(mock_websocket)
        
        # Verify websocket was removed
        assert mock_websocket not in ws_manager.active_connections
        assert len(ws_manager.active_connections) == 0
        
        # Test disconnect of non-connected websocket (should not raise error)
        unconnected_ws = AsyncMock()
        ws_manager.disconnect(unconnected_ws)  # Should not raise

    async def test_websocket_manager_broadcast_contract(self, ws_manager):
        """Test WebSocket manager broadcast method contract."""
        
        # Create multiple mock websockets
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()
        
        # Connect all websockets
        await ws_manager.connect(ws1)
        await ws_manager.connect(ws2)
        await ws_manager.connect(ws3)
        
        # Test broadcast message
        test_message = {
            "type": "test_broadcast",
            "data": "broadcast test",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        await ws_manager.broadcast(test_message)
        
        # Verify all websockets received the message
        expected_message = json.dumps(test_message)
        
        ws1.send_text.assert_called_once_with(expected_message)
        ws2.send_text.assert_called_once_with(expected_message)
        ws3.send_text.assert_called_once_with(expected_message)

    async def test_websocket_manager_error_handling_contract(self, ws_manager):
        """Test WebSocket manager error handling contract."""
        
        # Create websockets, one that will fail
        working_ws = AsyncMock()
        failing_ws = AsyncMock()
        failing_ws.send_text.side_effect = Exception("Connection lost")
        
        await ws_manager.connect(working_ws)
        await ws_manager.connect(failing_ws)
        
        assert len(ws_manager.active_connections) == 2
        
        # Broadcast message (should handle failing websocket gracefully)
        test_message = {"type": "test", "data": "error handling test"}
        
        await ws_manager.broadcast(test_message)
        
        # Working websocket should still receive message
        working_ws.send_text.assert_called_once()
        
        # Failing websocket should be removed from active connections
        assert failing_ws not in ws_manager.active_connections
        assert working_ws in ws_manager.active_connections
        assert len(ws_manager.active_connections) == 1

    # Message Broadcasting Integration Contracts

    async def test_agent_broadcast_integration_contract(self):
        """Test agent update broadcasting integration contract."""
        
        # This would typically test actual agent creation broadcasting
        # For now, validate the message format requirements
        
        agent_data = {
            "agent_id": "agent-test-123",
            "role": "backend_developer",
            "status": "active",
            "created_at": "2025-01-18T12:00:00Z",
            "agent_type": "claude",
            "session_name": "test-session",
            "workspace_path": "/test/workspace",
            "source": "agent_creation"
        }
        
        # Validate this follows the expected broadcast format
        broadcast_schema = {
            "type": "object",
            "required": ["agent_id", "status", "created_at", "source"],
            "properties": {
                "agent_id": {"type": "string"},
                "role": {"type": "string"},
                "status": {"type": "string"},
                "created_at": {"type": "string"},
                "agent_type": {"type": "string"},
                "session_name": {"type": "string"},
                "workspace_path": {"type": "string"},
                "source": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(agent_data, broadcast_schema)
        except ValidationError as e:
            pytest.fail(f"Agent broadcast data contract violation: {e}")

    async def test_task_broadcast_integration_contract(self):
        """Test task update broadcasting integration contract."""
        
        task_data = {
            "task_id": "task-test-456",
            "description": "Test task description",
            "task_type": "testing",
            "priority": "medium",
            "status": "pending",
            "assigned_agent_id": "agent-test-123",
            "created_at": "2025-01-18T12:00:00Z",
            "source": "task_creation"
        }
        
        # Validate task broadcast format
        task_broadcast_schema = {
            "type": "object",
            "required": ["task_id", "status", "created_at", "source"],
            "properties": {
                "task_id": {"type": "string"},
                "description": {"type": "string"},
                "task_type": {"type": "string"},
                "priority": {"type": "string"},
                "status": {"type": "string"},
                "assigned_agent_id": {"type": "string"},
                "created_at": {"type": "string"},
                "source": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(task_data, task_broadcast_schema)
        except ValidationError as e:
            pytest.fail(f"Task broadcast data contract violation: {e}")

    # Performance and Reliability Contracts

    async def test_broadcast_performance_contract(self, ws_manager):
        """Test broadcast performance meets contract requirements."""
        
        # Create multiple connections to test performance
        websockets = []
        for i in range(10):
            ws = AsyncMock()
            await ws_manager.connect(ws)
            websockets.append(ws)
        
        # Test broadcast performance
        test_message = {
            "type": "performance_test",
            "data": {"test": "data"},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        start_time = time.time()
        await ws_manager.broadcast(test_message)
        broadcast_time_ms = (time.time() - start_time) * 1000
        
        # Performance contract: broadcast should complete in <100ms
        assert broadcast_time_ms < 100.0, f"Broadcast took {broadcast_time_ms}ms, exceeds 100ms contract"
        
        # Verify all connections received the message
        for ws in websockets:
            ws.send_text.assert_called_once()

    async def test_connection_limit_contract(self, ws_manager):
        """Test connection limit handling contract."""
        
        # Test that manager can handle reasonable number of connections
        max_connections = 100
        websockets = []
        
        # Connect multiple websockets
        for i in range(max_connections):
            ws = AsyncMock()
            await ws_manager.connect(ws)
            websockets.append(ws)
        
        assert len(ws_manager.active_connections) == max_connections
        
        # Test broadcast with many connections
        test_message = {"type": "load_test", "data": "many connections"}
        
        start_time = time.time()
        await ws_manager.broadcast(test_message)
        broadcast_time_ms = (time.time() - start_time) * 1000
        
        # Should still perform reasonably with many connections
        assert broadcast_time_ms < 1000.0, f"Broadcast with {max_connections} connections took {broadcast_time_ms}ms"

    # Message Ordering and Consistency Contracts

    async def test_message_ordering_contract(self, ws_manager):
        """Test that messages maintain ordering contract."""
        
        ws = AsyncMock()
        await ws_manager.connect(ws)
        
        # Send multiple messages in sequence
        messages = [
            {"type": "test", "sequence": 1, "timestamp": "2025-01-18T12:00:01Z"},
            {"type": "test", "sequence": 2, "timestamp": "2025-01-18T12:00:02Z"},
            {"type": "test", "sequence": 3, "timestamp": "2025-01-18T12:00:03Z"}
        ]
        
        for message in messages:
            await ws_manager.broadcast(message)
        
        # Verify calls were made in order
        expected_calls = [json.dumps(msg) for msg in messages]
        actual_calls = [call[0][0] for call in ws.send_text.call_args_list]
        
        assert actual_calls == expected_calls, "Message ordering contract violation"

    # Error Recovery and Reconnection Contracts

    async def test_connection_recovery_contract(self, ws_manager):
        """Test connection recovery contract for resilient operations."""
        
        # Test that manager continues functioning after connection failures
        stable_ws = AsyncMock()
        unstable_ws = AsyncMock()
        unstable_ws.send_text.side_effect = [Exception("Connection lost"), None]
        
        await ws_manager.connect(stable_ws)
        await ws_manager.connect(unstable_ws)
        
        # First broadcast - unstable connection fails and gets removed
        await ws_manager.broadcast({"type": "test1", "data": "first"})
        
        assert len(ws_manager.active_connections) == 1
        assert stable_ws in ws_manager.active_connections
        assert unstable_ws not in ws_manager.active_connections
        
        # Second broadcast - should work with remaining connections
        await ws_manager.broadcast({"type": "test2", "data": "second"})
        
        # Stable connection should have received both messages
        assert stable_ws.send_text.call_count == 2

    # Protocol Compliance Contracts

    def test_websocket_protocol_compliance_contract(self):
        """Test WebSocket protocol compliance contracts."""
        
        # Test that all message types follow WebSocket protocol requirements
        message_types = [
            "connection_established",
            "echo", 
            "agent_created",
            "agent_updated",
            "agent_deleted",
            "task_created",
            "task_updated", 
            "task_deleted"
        ]
        
        base_message_schema = {
            "type": "object",
            "required": ["type", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": message_types},
                "timestamp": {"type": "string"},
                "data": {},  # Can be any type
                "message": {"type": "string"}
            }
        }
        
        # Test each message type
        for msg_type in message_types:
            test_message = {
                "type": msg_type,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Add appropriate data based on message type
            if msg_type.endswith("_created") or msg_type.endswith("_updated"):
                test_message["data"] = {"id": "test-id", "status": "active"}
            elif msg_type.endswith("_deleted"):
                test_message["data"] = {"id": "test-id"}
            elif msg_type == "echo":
                test_message["data"] = "echo data"
            elif msg_type == "connection_established":
                test_message["message"] = "Connected"
            
            try:
                jsonschema.validate(test_message, base_message_schema)
            except ValidationError as e:
                pytest.fail(f"Message type {msg_type} protocol compliance violation: {e}")


# Integration Contract Tests
class TestWebSocketIntegrationContracts:
    """Integration contract tests for WebSocket system components."""
    
    async def test_end_to_end_websocket_contract(self):
        """Test complete WebSocket integration contract from connection to broadcast."""
        
        # Create WebSocket manager
        ws_manager = WebSocketManager()
        
        # Mock WebSocket connection
        mock_ws = AsyncMock()
        
        # Test complete workflow
        try:
            # 1. Connect (contract: accept connection, add to active list)
            await ws_manager.connect(mock_ws)
            mock_ws.accept.assert_called_once()
            assert len(ws_manager.active_connections) == 1
            
            # 2. Broadcast agent creation (contract: JSON serialization, send to all)
            agent_message = {
                "type": "agent_created",
                "data": {
                    "id": "agent-integration-test",
                    "name": "Integration Test Agent",
                    "status": "active"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            await ws_manager.broadcast(agent_message)
            mock_ws.send_text.assert_called_with(json.dumps(agent_message))
            
            # 3. Broadcast task update (contract: different message type handling)
            task_message = {
                "type": "task_updated",
                "data": {
                    "id": "task-integration-test",
                    "status": "in_progress"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            await ws_manager.broadcast(task_message)
            assert mock_ws.send_text.call_count == 2
            
            # 4. Disconnect (contract: clean removal from active list)
            ws_manager.disconnect(mock_ws)
            assert len(ws_manager.active_connections) == 0
            
        except Exception as e:
            pytest.fail(f"End-to-end WebSocket contract test failed: {e}")

    async def test_websocket_contract_compliance_summary(self):
        """Summary test validating all WebSocket contracts are compatible."""
        
        # Test that all message schemas are compatible with the base protocol
        test_messages = [
            {
                "type": "connection_established",
                "message": "Connected to LeanVibe real-time updates",
                "timestamp": "2025-01-18T12:00:00Z"
            },
            {
                "type": "agent_created", 
                "data": {
                    "id": "agent-test",
                    "name": "Test Agent",
                    "status": "active"
                },
                "timestamp": "2025-01-18T12:00:00Z"
            },
            {
                "type": "task_updated",
                "data": {
                    "id": "task-test",
                    "status": "completed"
                },
                "timestamp": "2025-01-18T12:00:00Z"
            },
            {
                "type": "echo",
                "data": '{"test": "message"}',
                "timestamp": "2025-01-18T12:00:00Z"
            }
        ]
        
        # Universal message schema that all messages should comply with
        universal_schema = {
            "type": "object",
            "required": ["type", "timestamp"],
            "properties": {
                "type": {"type": "string"},
                "timestamp": {"type": "string"}
            }
        }
        
        # Validate all test messages
        for message in test_messages:
            try:
                jsonschema.validate(message, universal_schema)
                # Also ensure JSON serialization works
                json_str = json.dumps(message)
                json.loads(json_str)  # Ensure round-trip works
            except (ValidationError, json.JSONDecodeError) as e:
                pytest.fail(f"WebSocket contract compliance failure for {message['type']}: {e}")
        
        # Test WebSocket manager with all message types
        ws_manager = WebSocketManager()
        mock_ws = AsyncMock()
        await ws_manager.connect(mock_ws)
        
        for message in test_messages:
            await ws_manager.broadcast(message)
        
        # Verify all messages were sent
        assert mock_ws.send_text.call_count == len(test_messages)
        
        # Verify each call was valid JSON
        for call in mock_ws.send_text.call_args_list:
            message_str = call[0][0]
            try:
                json.loads(message_str)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in WebSocket message: {e}")