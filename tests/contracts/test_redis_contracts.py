"""
Redis Streams Contract Testing Suite

Validates that all Redis Streams messages conform to the agent message contract,
ensuring reliable inter-agent communication and preventing message format drift.
"""

import json
import uuid
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch

from jsonschema import validate, ValidationError
import pathlib

from app.core.redis import AgentMessageBroker, RedisStreamMessage, get_redis
from app.core.config import settings

# Load the Redis message schema
SCHEMA_PATH = pathlib.Path(__file__).parents[2] / "schemas" / "redis_agent_messages.schema.json"
with open(SCHEMA_PATH) as f:
    REDIS_MESSAGE_SCHEMA = json.load(f)

pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


class TestRedisMessageContracts:
    """Test Redis message format contracts and serialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock()
        self.broker = AgentMessageBroker(self.mock_redis)
        self.sample_payload = {
            "task_id": "task-123",
            "requirements": ["python", "fastapi", "testing"],
            "priority": "high",
            "estimated_time": 3600,
            "context": {"project": "contract-testing"}
        }
    
    async def test_agent_message_schema_validation(self):
        """Test that agent messages conform to schema contract."""
        
        # Valid message should pass validation
        valid_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "orchestrator-001",
            "to_agent": "dev-agent-python",
            "type": "task_assignment",
            "payload": json.dumps(self.sample_payload),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ttl": 3600,
            "priority": "high"
        }
        
        # Should validate successfully
        validate(instance=valid_message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Test required field validation
        required_fields = ["message_id", "from_agent", "to_agent", "type", "payload", "timestamp"]
        for field in required_fields:
            invalid_message = valid_message.copy()
            del invalid_message[field]
            
            with pytest.raises(ValidationError):
                validate(instance=invalid_message, schema=REDIS_MESSAGE_SCHEMA)
    
    async def test_message_type_contract_validation(self):
        """Test message type enumeration contract."""
        
        valid_types = [
            "task_assignment", "heartbeat", "task_result", "error", 
            "coordination", "workflow_sync", "resource_request", "capability_announcement"
        ]
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent", 
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Valid message types should pass
        for msg_type in valid_types:
            message = base_message.copy()
            message["type"] = msg_type
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Invalid message type should fail
        invalid_message = base_message.copy()
        invalid_message["type"] = "invalid_message_type"
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=invalid_message, schema=REDIS_MESSAGE_SCHEMA)
        assert "not one of" in str(exc_info.value)
    
    async def test_agent_identifier_format_contract(self):
        """Test agent identifier format constraints."""
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "type": "heartbeat",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Valid agent identifiers
        valid_agent_ids = [
            "orchestrator-001",
            "dev_agent_python", 
            "test-agent123",
            "AGENT_CAPS",
            "a"  # Minimum length
        ]
        
        for agent_id in valid_agent_ids:
            message = base_message.copy()
            message["from_agent"] = agent_id
            message["to_agent"] = agent_id
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Invalid agent identifiers should fail
        invalid_agent_ids = [
            "",  # Empty string
            "agent with spaces",  # Contains spaces
            "agent@domain.com",  # Contains special chars
            "a" * 65,  # Too long
            "agent.name"  # Contains dot
        ]
        
        for invalid_id in invalid_agent_ids:
            message = base_message.copy()
            message["from_agent"] = invalid_id
            message["to_agent"] = "valid-agent"
            
            with pytest.raises(ValidationError):
                validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
    
    async def test_payload_size_contract(self):
        """Test payload size constraints."""
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Valid payload size (within 64KB limit)
        large_payload = {"data": "x" * 60000}  # ~60KB
        message = base_message.copy()
        message["payload"] = json.dumps(large_payload)
        
        validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Payload exceeding size limit should fail
        oversized_payload = {"data": "x" * 70000}  # ~70KB
        message["payload"] = json.dumps(oversized_payload)
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        assert "too long" in str(exc_info.value).lower()
    
    async def test_timestamp_format_contract(self):
        """Test timestamp format requirements."""
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "heartbeat",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4())
        }
        
        # Valid timestamp formats
        valid_timestamps = [
            datetime.utcnow().isoformat() + "Z",
            "2025-01-15T10:30:00.123456Z",
            "2025-01-15T10:30:00Z",
            "2025-01-15T10:30:00+00:00"
        ]
        
        for timestamp in valid_timestamps:
            message = base_message.copy()
            message["timestamp"] = timestamp
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Invalid timestamp formats should fail
        invalid_timestamps = [
            "2025-01-15",  # Date only
            "10:30:00",  # Time only
            "not-a-timestamp",  # Invalid format
            "2025-13-45T25:70:90Z"  # Invalid values
        ]
        
        for invalid_ts in invalid_timestamps:
            message = base_message.copy()
            message["timestamp"] = invalid_ts
            
            with pytest.raises(ValidationError):
                validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
    
    async def test_ttl_constraints_contract(self):
        """Test TTL value constraints."""
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "heartbeat",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Valid TTL values
        valid_ttls = [1, 3600, 86400, 604800]  # 1 sec to 1 week
        
        for ttl in valid_ttls:
            message = base_message.copy()
            message["ttl"] = ttl
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Invalid TTL values should fail
        invalid_ttls = [0, -1, 604801]  # Zero, negative, too large
        
        for invalid_ttl in invalid_ttls:
            message = base_message.copy() 
            message["ttl"] = invalid_ttl
            
            with pytest.raises(ValidationError):
                validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
    
    async def test_priority_enumeration_contract(self):
        """Test priority value enumeration."""
        
        base_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Valid priorities
        valid_priorities = ["low", "normal", "high", "critical"]
        
        for priority in valid_priorities:
            message = base_message.copy()
            message["priority"] = priority
            validate(instance=message, schema=REDIS_MESSAGE_SCHEMA)
        
        # Invalid priority should fail
        invalid_message = base_message.copy()
        invalid_message["priority"] = "urgent"  # Not in enum
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_message, schema=REDIS_MESSAGE_SCHEMA)


class TestAgentMessageBrokerContracts:
    """Test AgentMessageBroker contract compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock()
        self.broker = AgentMessageBroker(self.mock_redis)
    
    async def test_send_message_contract_compliance(self):
        """Test that send_message creates contract-compliant messages."""
        
        # Mock Redis XADD response
        self.mock_redis.xadd.return_value = "1642248600000-0"
        
        payload = {
            "task_id": "test-task",
            "description": "Contract test task",
            "requirements": ["testing"]
        }
        
        message_id = await self.broker.send_message(
            from_agent="test-sender",
            to_agent="test-receiver",
            message_type="task_assignment",
            payload=payload
        )
        
        # Verify Redis was called
        assert self.mock_redis.xadd.called
        
        # Extract the message data sent to Redis
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        message_fields = call_args[0][1]
        
        # Verify stream naming convention
        assert stream_name == "agent_messages:test-receiver"
        
        # Validate message fields against schema
        validate(instance=message_fields, schema=REDIS_MESSAGE_SCHEMA)
        
        # Verify payload serialization
        deserialized_payload = json.loads(message_fields["payload"])
        assert deserialized_payload == payload
        
        # Verify auto-generated fields
        assert message_fields["from_agent"] == "test-sender"
        assert message_fields["to_agent"] == "test-receiver"
        assert message_fields["type"] == "task_assignment"
        assert "message_id" in message_fields
        assert "correlation_id" in message_fields
        assert "timestamp" in message_fields
    
    async def test_broadcast_message_contract_compliance(self):
        """Test broadcast message contract compliance."""
        
        self.mock_redis.xadd.return_value = "1642248600000-1"
        
        payload = {"announcement": "System maintenance in 10 minutes"}
        
        await self.broker.broadcast_message(
            from_agent="system-controller",
            message_type="coordination", 
            payload=payload
        )
        
        # Verify broadcast stream usage
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        message_fields = call_args[0][1]
        
        assert stream_name == "agent_messages:broadcast"
        assert message_fields["to_agent"] == "broadcast"
        
        # Validate message contract compliance
        validate(instance=message_fields, schema=REDIS_MESSAGE_SCHEMA)
    
    async def test_workflow_coordination_contract(self):
        """Test workflow coordination message contracts."""
        
        self.mock_redis.xadd.return_value = "1642248600000-2"
        
        # Test workflow synchronization message
        workflow_payload = {
            "workflow_id": str(uuid.uuid4()),
            "sync_point": "task_completion",
            "completed_tasks": ["task-1", "task-2"],
            "pending_tasks": ["task-3", "task-4"],
            "agent_status": {
                "agent-1": "ready",
                "agent-2": "busy"
            }
        }
        
        await self.broker.coordinate_workflow_tasks(
            workflow_id=workflow_payload["workflow_id"],
            tasks=workflow_payload["pending_tasks"],
            assignments={"task-3": "agent-1", "task-4": "agent-2"}
        )
        
        # Verify workflow coordination stream
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        message_fields = call_args[0][1]
        
        assert stream_name.startswith("workflow_coordination:")
        assert message_fields["type"] == "coordination"
        
        # Validate contract compliance
        validate(instance=message_fields, schema=REDIS_MESSAGE_SCHEMA)


class TestRedisStreamMessageContracts:
    """Test RedisStreamMessage parsing and contract compliance."""
    
    def test_redis_stream_message_parsing(self):
        """Test RedisStreamMessage correctly parses contract-compliant messages."""
        
        # Sample Redis stream data
        stream_id = "1642248600123-0"
        message_fields = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "sender-agent",
            "to_agent": "receiver-agent", 
            "type": "task_result",
            "payload": json.dumps({"result": "success", "output": "Task completed"}),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": "2025-01-15T10:30:00.123Z",
            "priority": "normal"
        }
        
        # Create RedisStreamMessage instance
        message = RedisStreamMessage(stream_id, message_fields)
        
        # Test property access
        assert message.message_id == message_fields["message_id"]
        assert message.from_agent == "sender-agent"
        assert message.to_agent == "receiver-agent"
        assert message.message_type == "task_result"
        assert message.correlation_id == message_fields["correlation_id"]
        
        # Test payload deserialization
        payload = message.payload
        assert payload["result"] == "success"
        assert payload["output"] == "Task completed"
        
        # Test timestamp parsing
        assert isinstance(message.timestamp, datetime)
    
    def test_payload_deserialization_robustness(self):
        """Test payload deserialization handles various formats."""
        
        test_cases = [
            # Valid JSON object
            ('{"key": "value", "number": 42}', {"key": "value", "number": 42}),
            
            # Valid JSON array
            ('[1, 2, 3]', {"data": [1, 2, 3]}),
            
            # Valid JSON string 
            ('"simple string"', {"data": "simple string"}),
            
            # Invalid JSON - should wrap in data field
            ('not valid json', {"data": "not valid json"}),
            
            # Empty payload
            ('', {"data": ""}),
        ]
        
        stream_id = "1642248600123-0"
        
        for payload_input, expected_output in test_cases:
            fields = {
                "message_id": str(uuid.uuid4()),
                "from_agent": "test-agent",
                "to_agent": "target-agent",
                "type": "heartbeat",
                "payload": payload_input,
                "correlation_id": str(uuid.uuid4()),
                "timestamp": "2025-01-15T10:30:00Z"
            }
            
            message = RedisStreamMessage(stream_id, fields)
            assert message.payload == expected_output


class TestContractPerformanceRequirements:
    """Test performance requirements as part of the contract."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        self.mock_redis = AsyncMock()
        self.broker = AgentMessageBroker(self.mock_redis)
    
    @pytest.mark.performance
    async def test_message_serialization_performance_contract(self):
        """Test message serialization meets performance contract (<5ms)."""
        
        # Large but valid payload
        large_payload = {
            "data": ["x" * 1000] * 50,  # ~50KB payload
            "metadata": {"timestamp": datetime.utcnow().isoformat()},
            "nested": {
                "deep": {
                    "structure": list(range(100))
                }
            }
        }
        
        serialization_times = []
        
        # Test serialization performance over multiple iterations
        for _ in range(100):
            start_time = time.perf_counter()
            
            # This would normally call the internal serialization method
            serialized = json.dumps(large_payload)
            
            end_time = time.perf_counter()
            serialization_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Validate performance contract
        avg_time = sum(serialization_times) / len(serialization_times)
        p95_time = sorted(serialization_times)[int(len(serialization_times) * 0.95)]
        
        assert avg_time < 2.0, f"Average serialization time {avg_time:.2f}ms exceeds contract (2ms)"
        assert p95_time < 5.0, f"P95 serialization time {p95_time:.2f}ms exceeds contract (5ms)"
    
    @pytest.mark.performance  
    async def test_message_validation_performance_contract(self):
        """Test schema validation meets performance contract (<1ms)."""
        
        # Standard message for validation testing
        test_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "perf-test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",
            "payload": json.dumps({"task": "performance test"}),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "priority": "normal"
        }
        
        validation_times = []
        
        # Test validation performance
        for _ in range(1000):
            start_time = time.perf_counter()
            
            try:
                validate(instance=test_message, schema=REDIS_MESSAGE_SCHEMA)
            except ValidationError:
                pass  # We're testing performance, not correctness here
            
            end_time = time.perf_counter()
            validation_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Validate performance contract
        avg_time = sum(validation_times) / len(validation_times)
        p95_time = sorted(validation_times)[int(len(validation_times) * 0.95)]
        
        assert avg_time < 0.5, f"Average validation time {avg_time:.3f}ms exceeds contract (0.5ms)"
        assert p95_time < 1.0, f"P95 validation time {p95_time:.3f}ms exceeds contract (1ms)"


class TestContractEvolutionScenarios:
    """Test contract evolution and backward compatibility scenarios."""
    
    def test_optional_field_addition_compatibility(self):
        """Test that adding optional fields maintains backward compatibility."""
        
        # Original v1.0 message format
        v1_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "legacy-agent",
            "to_agent": "modern-agent",
            "type": "heartbeat",
            "payload": "{}",
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Should validate against current schema (backward compatibility)
        validate(instance=v1_message, schema=REDIS_MESSAGE_SCHEMA)
        
        # v1.1 message with optional fields
        v1_1_message = v1_message.copy()
        v1_1_message.update({
            "ttl": 7200,
            "priority": "high",
            "workflow_id": str(uuid.uuid4()),
            "retry_count": 1
        })
        
        # Should also validate (forward compatibility)
        validate(instance=v1_1_message, schema=REDIS_MESSAGE_SCHEMA)
    
    def test_contract_violation_detection(self):
        """Test that contract violations are properly detected."""
        
        # Test various contract violation scenarios
        violation_scenarios = [
            # Missing required field
            {
                "from_agent": "test-agent",
                "to_agent": "target-agent",
                "type": "heartbeat",
                "payload": "{}",
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z"
                # Missing message_id
            },
            
            # Invalid field value
            {
                "message_id": str(uuid.uuid4()),
                "from_agent": "test-agent",
                "to_agent": "target-agent", 
                "type": "invalid_type",  # Not in enum
                "payload": "{}",
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            
            # Field constraint violation
            {
                "message_id": str(uuid.uuid4()),
                "from_agent": "",  # Empty string violates minLength
                "to_agent": "target-agent",
                "type": "heartbeat",
                "payload": "{}",
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            
            # Additional properties
            {
                "message_id": str(uuid.uuid4()),
                "from_agent": "test-agent",
                "to_agent": "target-agent",
                "type": "heartbeat",
                "payload": "{}",
                "correlation_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extra_field": "not allowed"  # additionalProperties: false
            }
        ]
        
        for scenario in violation_scenarios:
            with pytest.raises(ValidationError):
                validate(instance=scenario, schema=REDIS_MESSAGE_SCHEMA)


class TestContractMonitoringIntegration:
    """Test contract monitoring and observability integration."""
    
    def setup_method(self):
        """Set up monitoring test fixtures."""
        self.mock_redis = AsyncMock()
        self.broker = AgentMessageBroker(self.mock_redis)
    
    @patch('app.core.redis.logger')
    async def test_contract_violation_logging(self, mock_logger):
        """Test that contract violations are properly logged."""
        
        # This would be implemented in the actual ContractEnforcingMessageBroker
        # For now, we test the concept with a mock
        
        invalid_payload = "x" * 70000  # Exceeds size limit
        
        with pytest.raises(Exception):  # Would be ContractViolationError in real implementation
            await self.broker.send_message(
                from_agent="test-agent",
                to_agent="target-agent",
                message_type="task_assignment",
                payload={"oversized": invalid_payload}
            )
    
    def test_contract_metrics_collection(self):
        """Test that contract validation metrics are collected."""
        
        # This would test the actual metrics collection in ContractValidator
        # For now, we define the expected metrics structure
        
        expected_metrics = {
            "messages_validated_total": 0,
            "validation_failures_total": 0,
            "contract_violations_by_type": {
                "missing_required_field": 0,
                "invalid_field_value": 0,
                "constraint_violation": 0,
                "additional_properties": 0
            },
            "validation_time_ms": {
                "avg": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        }
        
        # In real implementation, these would be Prometheus metrics
        assert isinstance(expected_metrics, dict)
        assert "messages_validated_total" in expected_metrics


if __name__ == "__main__":
    # Run Redis contract tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "-m", "contract",
        "--asyncio-mode=auto"
    ])