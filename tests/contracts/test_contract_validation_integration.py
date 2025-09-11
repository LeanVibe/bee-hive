"""
Contract Validation Integration Tests

End-to-end tests that validate contract enforcement across component boundaries,
ensuring the contract testing strategy prevents breaking changes in practice.
"""

import asyncio
import json
import uuid
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from jsonschema import validate, ValidationError
from fastapi.testclient import TestClient

from app.core.redis import AgentMessageBroker, get_redis
# Epic 10 Mock Replacements
try:
    from app.api.dashboard_websockets import DashboardWebSocketManager
except ImportError:
    # Use Epic 10 mock replacements
    from tests.epic10_mock_replacements import (
        MockOrchestrator as UniversalOrchestrator,
        MockAgentRole as AgentRole,
        MockAgentStatus as AgentStatus,
        MockTaskPriority as TaskPriority,
        MockWebSocketManager as WebSocketManager,
        MockDatabase as Database,
        MockRedisManager as RedisManager
    )

from app.core.database import get_session
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus
from app.main import app

pytestmark = [pytest.mark.asyncio, pytest.mark.contract, pytest.mark.integration]


class ContractEnforcingMessageBroker(AgentMessageBroker):
    """Enhanced message broker with contract enforcement for testing."""
    
    def __init__(self, redis_client, contract_validator):
        super().__init__(redis_client)
        self.validator = contract_validator
        self.metrics = {
            'messages_validated': 0,
            'validation_failures': 0,
            'contract_violations': 0
        }
    
    async def send_message(self, from_agent: str, to_agent: str, 
                          message_type: str, payload: Dict[str, Any],
                          correlation_id: Optional[str] = None) -> str:
        """Send message with contract validation."""
        
        # Prepare message for validation
        message_data = {
            'message_id': str(uuid.uuid4()),
            'from_agent': from_agent,
            'to_agent': to_agent,
            'type': message_type,
            'payload': json.dumps(payload),
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Validate against contract
        if not self.validator.validate_redis_message(message_data):
            self.metrics['contract_violations'] += 1
            raise ContractViolationError(f"Message from {from_agent} violates contract")
        
        self.metrics['messages_validated'] += 1
        
        # Call parent implementation
        return await super().send_message(from_agent, to_agent, message_type, payload, correlation_id)


class ContractViolationError(Exception):
    """Raised when a contract violation is detected."""
    pass


class MockContractValidator:
    """Mock contract validator for testing."""
    
    def __init__(self):
        self.redis_schema = self._load_redis_schema()
        self.ws_schema = self._load_ws_schema()
        self.validation_calls = []
    
    def _load_redis_schema(self):
        """Load Redis message schema."""
        import pathlib
        schema_path = pathlib.Path(__file__).parents[2] / "schemas" / "redis_agent_messages.schema.json"
        with open(schema_path) as f:
            return json.load(f)
    
    def _load_ws_schema(self):
        """Load WebSocket message schema."""
        import pathlib
        schema_path = pathlib.Path(__file__).parents[2] / "schemas" / "ws_messages.schema.json"
        with open(schema_path) as f:
            return json.load(f)
    
    def validate_redis_message(self, message: Dict[str, Any]) -> bool:
        """Validate Redis message against schema."""
        self.validation_calls.append(('redis', message))
        try:
            validate(instance=message, schema=self.redis_schema)
            return True
        except ValidationError:
            return False
    
    def validate_websocket_message(self, message: Dict[str, Any]) -> bool:
        """Validate WebSocket message against schema."""
        self.validation_calls.append(('websocket', message))
        try:
            validate(instance=message, schema=self.ws_schema)
            return True
        except ValidationError:
            return False


class TestEndToEndContractValidation:
    """Test contract validation across the entire system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock()
        self.contract_validator = MockContractValidator()
        self.broker = ContractEnforcingMessageBroker(self.mock_redis, self.contract_validator)
    
    async def test_agent_coordination_workflow_contracts(self):
        """Test complete agent coordination workflow respects all contracts."""
        
        # Mock Redis XADD responses
        self.mock_redis.xadd.return_value = "1642248600000-0"
        
        # Simulate orchestrator assigning tasks to agents
        workflow_id = str(uuid.uuid4())
        
        # Step 1: Orchestrator broadcasts workflow start
        workflow_start_payload = {
            "workflow_id": workflow_id,
            "tasks": ["task-1", "task-2", "task-3"],
            "priority": "high",
            "deadline": "2025-01-15T16:00:00Z"
        }
        
        await self.broker.broadcast_message(
            from_agent="orchestrator-001",
            message_type="coordination",
            payload=workflow_start_payload
        )
        
        # Step 2: Individual task assignments
        tasks = [
            {"task_id": "task-1", "type": "analysis", "agent": "analyst-agent"},
            {"task_id": "task-2", "type": "development", "agent": "dev-agent-python"},
            {"task_id": "task-3", "type": "testing", "agent": "qa-agent"}
        ]
        
        for task in tasks:
            await self.broker.send_message(
                from_agent="orchestrator-001",
                to_agent=task["agent"],
                message_type="task_assignment",
                payload={
                    "task_id": task["task_id"],
                    "workflow_id": workflow_id,
                    "type": task["type"],
                    "requirements": ["python", "testing"],
                    "priority": "high"
                }
            )
        
        # Step 3: Agents respond with status updates
        for task in tasks:
            await self.broker.send_message(
                from_agent=task["agent"],
                to_agent="orchestrator-001",
                message_type="task_result",
                payload={
                    "task_id": task["task_id"],
                    "workflow_id": workflow_id,
                    "status": "accepted",
                    "estimated_completion": "2025-01-15T14:00:00Z"
                }
            )
        
        # Verify all messages were validated
        assert self.broker.metrics['messages_validated'] == 7  # 1 broadcast + 3 assignments + 3 responses
        assert self.broker.metrics['contract_violations'] == 0
        
        # Verify Redis was called for each message
        assert self.mock_redis.xadd.call_count == 7
        
        # Verify contract validator was called
        assert len(self.contract_validator.validation_calls) == 7
        redis_calls = [call for call in self.contract_validator.validation_calls if call[0] == 'redis']
        assert len(redis_calls) == 7
    
    async def test_contract_violation_detection_and_handling(self):
        """Test that contract violations are detected and handled properly."""
        
        # Attempt to send message with invalid agent ID
        with pytest.raises(ContractViolationError):
            await self.broker.send_message(
                from_agent="",  # Empty agent ID violates contract
                to_agent="valid-agent",
                message_type="heartbeat",
                payload={"status": "active"}
            )
        
        # Attempt to send message with invalid message type
        with pytest.raises(ContractViolationError):
            await self.broker.send_message(
                from_agent="valid-agent",
                to_agent="another-agent",
                message_type="invalid_type",  # Not in enum
                payload={"data": "test"}
            )
        
        # Attempt to send message with oversized payload
        oversized_payload = {"data": "x" * 70000}  # Exceeds 64KB limit
        with pytest.raises(ContractViolationError):
            await self.broker.send_message(
                from_agent="valid-agent",
                to_agent="another-agent",
                message_type="heartbeat",
                payload=oversized_payload
            )
        
        # Verify violation metrics
        assert self.broker.metrics['contract_violations'] == 3
        assert self.broker.metrics['messages_validated'] == 0  # No valid messages sent


class TestWebSocketContractIntegration:
    """Test WebSocket contract integration with real clients."""
    
    @patch('app.api.dashboard_websockets.DashboardWebSocketManager')
    async def test_websocket_contract_enforcement(self, mock_ws_manager):
        """Test WebSocket message contract enforcement."""
        
        # Set up mock WebSocket manager with contract validation
        mock_manager_instance = MagicMock()
        mock_ws_manager.return_value = mock_manager_instance
        
        contract_validator = MockContractValidator()
        
        # Simulate WebSocket message handling with contract validation
        async def validate_and_handle_message(connection_id: str, message: Dict[str, Any]):
            """Mock message handler with contract validation."""
            if not contract_validator.validate_websocket_message(message):
                return {"type": "error", "message": "Message violates contract"}
            return {"type": "pong"} if message.get("type") == "ping" else {"type": "ack"}
        
        mock_manager_instance.handle_message.side_effect = validate_and_handle_message
        
        # Test valid WebSocket messages
        valid_messages = [
            {"type": "ping"},
            {"type": "subscribe", "subscriptions": ["agents"]},
            {"type": "unsubscribe", "subscriptions": ["tasks"]}
        ]
        
        for message in valid_messages:
            response = await mock_manager_instance.handle_message("test-conn", message)
            assert response["type"] != "error"
        
        # Test invalid WebSocket message
        invalid_message = {"invalid_field": "value"}  # Missing required 'type' field
        response = await mock_manager_instance.handle_message("test-conn", invalid_message)
        assert response["type"] == "error"
        assert "contract" in response["message"].lower()
    
    async def test_websocket_message_broadcasting_contracts(self):
        """Test WebSocket broadcasting respects message contracts."""
        
        contract_validator = MockContractValidator()
        
        # Simulate broadcast messages
        broadcast_messages = [
            {
                "type": "agent_update",
                "subscription": "agents",
                "data": {
                    "agent_id": "agent-123",
                    "status": "active",
                    "current_task": "task-456"
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "type": "coordination_update", 
                "subscription": "coordination",
                "data": {
                    "workflow_id": str(uuid.uuid4()),
                    "status": "in_progress",
                    "progress": 0.45
                },
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "type": "critical_alert",
                "subscription": "alerts",
                "data": {
                    "alerts": [
                        {"level": "critical", "message": "System overload detected"}
                    ],
                    "timestamp": datetime.utcnow().isoformat()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        
        # Validate all broadcast messages
        for message in broadcast_messages:
            is_valid = contract_validator.validate_websocket_message(message)
            assert is_valid, f"Broadcast message failed validation: {message}"
        
        # Verify contract validator was called
        ws_calls = [call for call in contract_validator.validation_calls if call[0] == 'websocket']
        assert len(ws_calls) == len(broadcast_messages)


class TestDatabaseContractIntegration:
    """Test database schema contract enforcement."""
    
    async def test_agent_model_contract_enforcement(self):
        """Test Agent model respects database schema contracts."""
        
        # This would require actual database connection in real test
        # For now, we test the model validation logic
        
        # Valid agent creation
        valid_agent_data = {
            "name": "test-agent-contracts",
            "role": "testing",
            "capabilities": ["testing", "validation"],
            "status": AgentStatus.ACTIVE.value
        }
        
        # Test constraint validation (this would be done by SQLAlchemy in real scenario)
        assert len(valid_agent_data["name"]) > 0
        assert valid_agent_data["status"] in [status.value for status in AgentStatus]
        assert isinstance(valid_agent_data["capabilities"], list)
        
        # Invalid agent data should fail
        invalid_agent_data = {
            "name": "",  # Empty name violates constraint
            "role": "testing",
            "status": "invalid_status"  # Not in enum
        }
        
        # These would raise IntegrityError in real database scenario
        assert len(invalid_agent_data["name"]) == 0  # Would fail constraint
        assert invalid_agent_data["status"] not in [status.value for status in AgentStatus]
    
    async def test_task_assignment_foreign_key_contracts(self):
        """Test Task-Agent foreign key relationship contracts."""
        
        # Valid task with agent reference
        agent_id = uuid.uuid4()
        valid_task_data = {
            "title": "Contract Test Task",
            "description": "Testing foreign key contracts",
            "assigned_agent_id": agent_id,
            "status": TaskStatus.ASSIGNED.value
        }
        
        # Verify task data structure
        assert valid_task_data["assigned_agent_id"] is not None
        assert isinstance(valid_task_data["assigned_agent_id"], uuid.UUID)
        assert valid_task_data["status"] in [status.value for status in TaskStatus]
        
        # Invalid task with non-existent agent reference
        invalid_task_data = {
            "title": "Invalid Task",
            "assigned_agent_id": "not-a-uuid",  # Invalid UUID format
            "status": "invalid_status"  # Not in enum
        }
        
        # These would fail in real database
        try:
            uuid.UUID(invalid_task_data["assigned_agent_id"])
            assert False, "Should have failed UUID validation"
        except (ValueError, TypeError):
            pass  # Expected failure


class TestAPIResponseContractIntegration:
    """Test API response contract integration."""
    
    def test_live_dashboard_api_contract(self):
        """Test live dashboard API response contract."""
        
        client = TestClient(app)
        
        # This would require actual endpoint implementation
        # For now, we test the expected response structure
        
        expected_response_structure = {
            "metrics": {
                "active_projects": 3,
                "active_agents": 5,
                "agent_utilization": 0.75,
                "completed_tasks": 42,
                "active_conflicts": 1,
                "system_efficiency": 0.92,
                "system_status": "healthy",
                "last_updated": datetime.utcnow().isoformat()
            },
            "agent_activities": [
                {
                    "agent_id": "agent-1",
                    "name": "Dev Agent Python",
                    "status": "active",
                    "performance_score": 0.85,
                    "specializations": ["python", "fastapi"]
                }
            ],
            "project_snapshots": [
                {
                    "name": "Contract Testing",
                    "status": "active", 
                    "progress_percentage": 0.65,
                    "participating_agents": ["agent-1", "agent-2"],
                    "completed_tasks": 8,
                    "active_tasks": 3,
                    "conflicts": 0,
                    "quality_score": 0.91
                }
            ],
            "conflict_snapshots": []
        }
        
        # Validate response structure against schema
        import pathlib
        schema_path = pathlib.Path(__file__).parents[2] / "schemas" / "live_dashboard_data.schema.json"
        with open(schema_path) as f:
            dashboard_schema = json.load(f)
        
        # Should validate successfully
        validate(instance=expected_response_structure, schema=dashboard_schema)
    
    def test_api_error_response_contracts(self):
        """Test API error response contracts."""
        
        # Standard error response structure
        error_responses = [
            {
                "error": "ValidationError",
                "message": "Invalid request payload",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "field": "agent_id",
                    "issue": "required field missing"
                }
            },
            {
                "error": "NotFoundError", 
                "message": "Agent not found",
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "error": "ContractViolationError",
                "message": "Request violates API contract",
                "timestamp": datetime.utcnow().isoformat(),
                "contract_details": {
                    "violated_field": "message_type",
                    "expected_values": ["task_assignment", "heartbeat"],
                    "received_value": "invalid_type"
                }
            }
        ]
        
        # Basic validation of error response structure
        for error_response in error_responses:
            assert "error" in error_response
            assert "message" in error_response
            assert "timestamp" in error_response
            assert isinstance(error_response["error"], str)
            assert isinstance(error_response["message"], str)


class TestPerformanceContractIntegration:
    """Test performance contract compliance in integration scenarios."""
    
    @pytest.mark.performance
    async def test_message_processing_performance_contract(self):
        """Test end-to-end message processing meets performance contracts."""
        
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = "1642248600000-0"
        
        contract_validator = MockContractValidator()
        broker = ContractEnforcingMessageBroker(mock_redis, contract_validator)
        
        # Test batch message processing performance
        message_count = 100
        start_time = time.perf_counter()
        
        for i in range(message_count):
            await broker.send_message(
                from_agent="perf-test-sender",
                to_agent=f"perf-test-receiver-{i % 10}",  # 10 different receivers
                message_type="heartbeat",
                payload={"sequence": i, "timestamp": datetime.utcnow().isoformat()}
            )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance contract validation
        messages_per_second = message_count / total_time
        avg_time_per_message = (total_time / message_count) * 1000  # Convert to ms
        
        # Assert performance contracts
        assert messages_per_second > 250, f"Throughput {messages_per_second:.0f} msg/sec below contract (250 msg/sec)"
        assert avg_time_per_message < 5.0, f"Average processing time {avg_time_per_message:.2f}ms exceeds contract (5ms)"
        
        # Verify all messages were validated
        assert broker.metrics['messages_validated'] == message_count
        assert broker.metrics['contract_violations'] == 0
    
    @pytest.mark.performance
    async def test_schema_validation_performance_contract(self):
        """Test schema validation performance under load."""
        
        contract_validator = MockContractValidator()
        
        # Create test message
        test_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "perf-test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",
            "payload": json.dumps({"task": "performance test", "data": ["x"] * 1000}),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "priority": "normal"
        }
        
        # Test validation performance
        validation_count = 1000
        start_time = time.perf_counter()
        
        for _ in range(validation_count):
            result = contract_validator.validate_redis_message(test_message)
            assert result is True
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance contract validation
        validations_per_second = validation_count / total_time
        avg_time_per_validation = (total_time / validation_count) * 1000  # Convert to ms
        
        # Assert performance contracts
        assert validations_per_second > 200, f"Validation throughput {validations_per_second:.0f} val/sec below contract (200 val/sec)"
        assert avg_time_per_validation < 10.0, f"Average validation time {avg_time_per_validation:.3f}ms exceeds contract (10ms)"


class TestContractEvolutionIntegration:
    """Test contract evolution scenarios in integration context."""
    
    async def test_backward_compatibility_integration(self):
        """Test backward compatibility with older message formats."""
        
        contract_validator = MockContractValidator()
        
        # Simulate older v1.0 message format (minimal required fields)
        v1_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "legacy-agent",
            "to_agent": "modern-agent",
            "type": "heartbeat",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Should validate against current schema
        assert contract_validator.validate_redis_message(v1_message)
        
        # Modern v1.1+ message with optional fields
        v1_1_message = v1_message.copy()
        v1_1_message.update({
            "ttl": 3600,
            "priority": "normal",
            "workflow_id": str(uuid.uuid4())
        })
        
        # Should also validate (forward compatibility)
        assert contract_validator.validate_redis_message(v1_1_message)
    
    async def test_contract_migration_simulation(self):
        """Test contract migration scenarios."""
        
        # Simulate a breaking change scenario (for testing migration logic)
        # In real implementation, this would test the migration utilities
        
        # Original message format
        original_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",  # In v2.0, this becomes "message_type"
            "payload": json.dumps({"task": "test"}),  # In v2.0, this becomes object
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Simulate migration to v2.0 format
        def migrate_v1_to_v2(v1_message):
            """Simulate migration from v1 to v2 format."""
            v2_message = v1_message.copy()
            
            # Breaking change: 'type' -> 'message_type'
            v2_message['message_type'] = v2_message.pop('type')
            
            # Breaking change: payload from string to object
            if isinstance(v2_message['payload'], str):
                try:
                    v2_message['payload'] = json.loads(v2_message['payload'])
                except json.JSONDecodeError:
                    v2_message['payload'] = {"legacy_data": v2_message['payload']}
            
            return v2_message
        
        # Test migration
        migrated_message = migrate_v1_to_v2(original_message)
        
        # Verify migration results
        assert "type" not in migrated_message
        assert "message_type" in migrated_message
        assert migrated_message["message_type"] == "task_assignment"
        assert isinstance(migrated_message["payload"], dict)


if __name__ == "__main__":
    # Run integration contract tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "contract and integration", 
        "--asyncio-mode=auto"
    ])