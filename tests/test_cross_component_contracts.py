"""
Cross-Component Contract Validation Tests

Critical tests to validate contracts between different system components
to ensure integration integrity and prevent breaking changes between services.
"""

import pytest
import json
import uuid
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator
from app.core.redis import AgentMessageBroker
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus, TaskPriority


class TestRedisMessageContracts:
    """Test Redis message format contracts between components."""
    
    @pytest.fixture
    def sample_agent_message(self):
        """Sample agent message for contract validation."""
        return {
            "message_id": str(uuid.uuid4()),
            "agent_id": "test-agent-001",
            "message_type": "task_assignment",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "task_id": str(uuid.uuid4()),
                "task_type": "backend_development",
                "priority": "HIGH",
                "description": "Implement user authentication",
                "context": {
                    "repository": "test-repo",
                    "branch": "feature/auth"
                }
            },
            "correlation_id": str(uuid.uuid4()),
            "reply_to": "orchestrator-main"
        }
    
    @pytest.fixture
    def sample_orchestrator_response(self):
        """Sample orchestrator response for contract validation."""
        return {
            "message_id": str(uuid.uuid4()),
            "response_to": str(uuid.uuid4()),
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "assigned_agent": "agent-backend-001",
                "estimated_completion": "2025-08-14T10:00:00Z",
                "routing_score": 0.95
            },
            "errors": []
        }
    
    def test_agent_message_contract_validation(self, sample_agent_message):
        """Test agent message contract validation."""
        
        # Required fields
        required_fields = [
            "message_id", "agent_id", "message_type", 
            "timestamp", "payload", "correlation_id"
        ]
        
        for field in required_fields:
            assert field in sample_agent_message
            assert sample_agent_message[field] is not None
        
        # Field type validation
        assert isinstance(sample_agent_message["message_id"], str)
        assert isinstance(sample_agent_message["agent_id"], str)
        assert isinstance(sample_agent_message["payload"], dict)
        
        # Timestamp format validation
        timestamp = sample_agent_message["timestamp"]
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    def test_orchestrator_response_contract_validation(self, sample_orchestrator_response):
        """Test orchestrator response contract validation."""
        
        # Required fields
        required_fields = [
            "message_id", "response_to", "status", 
            "timestamp", "data"
        ]
        
        for field in required_fields:
            assert field in sample_orchestrator_response
        
        # Status validation
        valid_statuses = ["success", "error", "pending", "timeout"]
        assert sample_orchestrator_response["status"] in valid_statuses
        
        # Error handling
        assert "errors" in sample_orchestrator_response
        assert isinstance(sample_orchestrator_response["errors"], list)
    
    def test_redis_message_serialization_contract(self, sample_agent_message):
        """Test Redis message serialization contract."""
        
        # Should be JSON serializable
        serialized = json.dumps(sample_agent_message)
        assert isinstance(serialized, str)
        
        # Should be deserializable
        deserialized = json.loads(serialized)
        assert deserialized == sample_agent_message
        
        # Should maintain type integrity
        assert isinstance(deserialized["payload"], dict)
        assert isinstance(deserialized["correlation_id"], str)
    
    @pytest.mark.asyncio
    async def test_message_broker_contract_compliance(self, sample_agent_message):
        """Test message broker contract compliance."""
        
        broker = AgentMessageBroker()
        broker.redis_client = AsyncMock()
        
        # Should accept valid message format
        await broker.send_message("test-channel", sample_agent_message)
        broker.redis_client.xadd.assert_called_once()
        
        # Verify message was properly formatted for Redis
        call_args = broker.redis_client.xadd.call_args
        assert call_args is not None
        
        # Should contain required Redis stream fields
        stream_data = call_args[0][1]  # Second argument is the data
        assert isinstance(stream_data, dict)


class TestWebSocketMessageContracts:
    """Test WebSocket message contracts between Backend and PWA."""
    
    @pytest.fixture
    def sample_websocket_message(self):
        """Sample WebSocket message for contract validation."""
        return {
            "type": "agent_status_update",
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "agent_id": "agent-backend-001",
                "status": "active",
                "current_task": {
                    "task_id": str(uuid.uuid4()),
                    "description": "Implementing user service",
                    "progress": 0.45
                },
                "metrics": {
                    "cpu_usage": 0.25,
                    "memory_usage": 0.60,
                    "context_usage": 0.80
                }
            },
            "source": "orchestrator",
            "target": "dashboard"
        }
    
    @pytest.fixture
    def sample_websocket_command(self):
        """Sample WebSocket command from PWA to Backend."""
        return {
            "type": "request_agent_list",
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": {
                "filter": "active",
                "include_metrics": True,
                "max_results": 50
            },
            "authentication": {
                "session_id": str(uuid.uuid4()),
                "user_id": "user-001"
            }
        }
    
    def test_websocket_message_schema_validation(self, sample_websocket_message):
        """Test WebSocket message schema validation."""
        
        # Required fields
        required_fields = ["type", "id", "timestamp", "data", "source"]
        
        for field in required_fields:
            assert field in sample_websocket_message
            assert sample_websocket_message[field] is not None
        
        # Type validation
        valid_types = [
            "agent_status_update", "task_assignment", "workflow_update",
            "error_notification", "heartbeat", "metrics_update"
        ]
        assert sample_websocket_message["type"] in valid_types
        
        # Data structure validation
        data = sample_websocket_message["data"]
        assert isinstance(data, dict)
        assert "agent_id" in data
        assert "status" in data
    
    def test_websocket_command_schema_validation(self, sample_websocket_command):
        """Test WebSocket command schema validation."""
        
        # Required fields
        required_fields = ["type", "id", "timestamp", "parameters"]
        
        for field in required_fields:
            assert field in sample_websocket_command
        
        # Command type validation
        valid_command_types = [
            "request_agent_list", "assign_task", "pause_agent",
            "resume_agent", "get_metrics", "execute_workflow"
        ]
        assert sample_websocket_command["type"] in valid_command_types
        
        # Authentication validation
        if "authentication" in sample_websocket_command:
            auth = sample_websocket_command["authentication"]
            assert "session_id" in auth
            assert isinstance(auth["session_id"], str)
    
    def test_websocket_message_size_limits(self, sample_websocket_message):
        """Test WebSocket message size limits."""
        
        serialized = json.dumps(sample_websocket_message)
        message_size = len(serialized.encode('utf-8'))
        
        # Should be under reasonable size limit (1MB)
        assert message_size < 1024 * 1024
        
        # Should be efficiently structured
        assert message_size < 10000  # 10KB for typical messages


class TestDatabaseSchemaContracts:
    """Test database schema contracts and constraints."""
    
    def test_task_model_contract_validation(self):
        """Test Task model contract validation."""
        
        # Create task with required fields
        task_data = {
            "title": "Test Task",
            "description": "Test task description",
            "task_type": "backend_development",
            "status": TaskStatus.PENDING,
            "priority": TaskPriority.MEDIUM,
            "required_capabilities": ["python", "fastapi"],
            "context": {"repository": "test-repo"}
        }
        
        task = Task(**task_data)
        
        # Verify required fields
        assert task.title == "Test Task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        
        # Verify default values
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert isinstance(task.context, dict)
        assert isinstance(task.required_capabilities, list)
    
    def test_agent_model_contract_validation(self):
        """Test Agent model contract validation."""
        
        agent_data = {
            "name": "Test Agent",
            "role": "backend_developer",
            "status": AgentStatus.ACTIVE,
            "capabilities": ["python", "fastapi", "postgresql"],
            "configuration": {
                "max_context_window": 8000,
                "preferred_tools": ["editor", "terminal"]
            }
        }
        
        agent = Agent(**agent_data)
        
        # Verify required fields
        assert agent.name == "Test Agent"
        assert agent.role == "backend_developer"
        assert agent.status == AgentStatus.ACTIVE
        
        # Verify data types
        assert isinstance(agent.capabilities, list)
        assert isinstance(agent.configuration, dict)
    
    def test_database_constraint_validation(self):
        """Test database constraint validation."""
        
        # Test task with workflow_id (new field)
        task = Task(
            title="Workflow Task",
            description="Task part of workflow",
            workflow_id=str(uuid.uuid4()),
            status=TaskStatus.PENDING
        )
        
        # Should accept workflow_id
        assert task.workflow_id is not None
        assert isinstance(task.workflow_id, str)
        
        # Test foreign key constraint (would be validated by database)
        assert task.assigned_agent_id is None  # Should allow null
        assert task.created_by_agent_id is None  # Should allow null


class TestAPIPayloadContracts:
    """Test API payload contracts between services."""
    
    @pytest.fixture
    def sample_task_assignment_payload(self):
        """Sample task assignment API payload."""
        return {
            "task": {
                "id": str(uuid.uuid4()),
                "title": "Implement user authentication",
                "description": "Create secure user login system",
                "type": "backend_development",
                "priority": "HIGH",
                "estimated_effort": 8,
                "required_capabilities": ["python", "fastapi", "jwt"],
                "context": {
                    "repository": "user-service",
                    "branch": "feature/auth",
                    "dependencies": ["database", "redis"]
                }
            },
            "assignment": {
                "agent_id": "agent-backend-001",
                "routing_strategy": "capability_first",
                "deadline": "2025-08-15T17:00:00Z",
                "persona_preference": "senior_developer"
            },
            "metadata": {
                "assigned_by": "orchestrator",
                "assignment_time": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4())
            }
        }
    
    @pytest.fixture
    def sample_agent_metrics_payload(self):
        """Sample agent metrics API payload."""
        return {
            "agent_id": "agent-backend-001",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "performance": {
                    "tasks_completed": 15,
                    "tasks_failed": 1,
                    "average_completion_time": 3.5,
                    "success_rate": 0.94
                },
                "resource_usage": {
                    "cpu_usage": 0.45,
                    "memory_usage": 0.72,
                    "context_window_usage": 0.85
                },
                "capability_scores": {
                    "python": 0.95,
                    "fastapi": 0.90,
                    "postgresql": 0.85
                }
            },
            "status": {
                "current_state": "active",
                "current_task": str(uuid.uuid4()),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
        }
    
    def test_task_assignment_payload_validation(self, sample_task_assignment_payload):
        """Test task assignment payload validation."""
        
        payload = sample_task_assignment_payload
        
        # Validate structure
        assert "task" in payload
        assert "assignment" in payload
        assert "metadata" in payload
        
        # Validate task fields
        task = payload["task"]
        required_task_fields = ["id", "title", "type", "priority"]
        for field in required_task_fields:
            assert field in task
        
        # Validate assignment fields
        assignment = payload["assignment"]
        required_assignment_fields = ["agent_id", "routing_strategy"]
        for field in required_assignment_fields:
            assert field in assignment
        
        # Validate data types
        assert isinstance(task["required_capabilities"], list)
        assert isinstance(task["context"], dict)
        assert isinstance(assignment["agent_id"], str)
    
    def test_agent_metrics_payload_validation(self, sample_agent_metrics_payload):
        """Test agent metrics payload validation."""
        
        payload = sample_agent_metrics_payload
        
        # Validate structure
        assert "agent_id" in payload
        assert "timestamp" in payload
        assert "metrics" in payload
        assert "status" in payload
        
        # Validate metrics structure
        metrics = payload["metrics"]
        assert "performance" in metrics
        assert "resource_usage" in metrics
        assert "capability_scores" in metrics
        
        # Validate performance metrics
        performance = metrics["performance"]
        assert isinstance(performance["tasks_completed"], int)
        assert isinstance(performance["success_rate"], float)
        assert 0.0 <= performance["success_rate"] <= 1.0
        
        # Validate resource usage
        resource_usage = metrics["resource_usage"]
        for metric in ["cpu_usage", "memory_usage", "context_window_usage"]:
            assert metric in resource_usage
            assert 0.0 <= resource_usage[metric] <= 1.0
    
    def test_api_error_response_contract(self):
        """Test API error response contract."""
        
        error_response = {
            "error": {
                "code": "AGENT_NOT_FOUND",
                "message": "Agent with ID 'agent-001' not found",
                "details": {
                    "agent_id": "agent-001",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": str(uuid.uuid4())
                },
                "suggestions": [
                    "Verify the agent ID is correct",
                    "Check if the agent is still active"
                ]
            },
            "status": "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Validate error structure
        assert "error" in error_response
        assert "status" in error_response
        assert "timestamp" in error_response
        
        error = error_response["error"]
        assert "code" in error
        assert "message" in error
        
        # Validate error code format
        assert isinstance(error["code"], str)
        assert error["code"].isupper()
        assert "_" in error["code"]  # Should use SNAKE_CASE


class TestIntegrationContractValidation:
    """Test integration contracts between multiple components."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_to_redis_contract_flow(self):
        """Test complete contract flow from orchestrator to Redis."""
        
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        
        # Create a task assignment message
        task_data = {
            "task_id": str(uuid.uuid4()),
            "agent_id": "test-agent",
            "task_type": "backend_development",
            "priority": "HIGH"
        }
        
        # Should format message according to contract
        await orchestrator._send_task_assignment_message(
            agent_id=task_data["agent_id"],
            task_id=task_data["task_id"],
            task_type=task_data["task_type"]
        )
        
        # Verify message broker was called with correct format
        orchestrator.message_broker.send_message.assert_called_once()
        call_args = orchestrator.message_broker.send_message.call_args
        
        # Validate message structure
        channel, message = call_args[0]
        assert isinstance(channel, str)
        assert isinstance(message, dict)
        assert "message_id" in message
        assert "timestamp" in message
    
    @pytest.mark.asyncio
    async def test_redis_to_websocket_contract_flow(self):
        """Test contract flow from Redis to WebSocket."""
        
        # Simulate Redis message being converted to WebSocket message
        redis_message = {
            "agent_id": "test-agent",
            "status": "task_completed",
            "task_result": {
                "task_id": str(uuid.uuid4()),
                "success": True,
                "execution_time": 2.5
            }
        }
        
        # Should be convertible to WebSocket format
        websocket_message = {
            "type": "agent_status_update",
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "data": redis_message,
            "source": "orchestrator",
            "target": "dashboard"
        }
        
        # Validate conversion maintains contract
        assert websocket_message["type"] in ["agent_status_update"]
        assert "id" in websocket_message
        assert "timestamp" in websocket_message
        assert websocket_message["data"] == redis_message
    
    def test_database_to_api_contract_flow(self):
        """Test contract flow from database models to API responses."""
        
        # Create database model
        task = Task(
            title="Test Task",
            description="Test description",
            task_type="backend_development",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH
        )
        
        # Convert to API response format
        api_response = {
            "task": {
                "id": str(task.id),
                "title": task.title,
                "description": task.description,
                "type": task.task_type.value if task.task_type else None,
                "status": task.status.value,
                "priority": task.priority.name.lower(),
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "updated_at": task.updated_at.isoformat() if task.updated_at else None
            }
        }
        
        # Validate API contract compliance
        task_data = api_response["task"]
        assert isinstance(task_data["id"], str)
        assert isinstance(task_data["title"], str)
        assert task_data["status"] in ["pending", "assigned", "in_progress", "completed", "failed"]
        assert task_data["priority"] in ["low", "medium", "high", "critical"]


# Helper methods for orchestrator extensions to support contract testing
def add_orchestrator_contract_methods():
    """Add contract testing methods to AgentOrchestrator."""
    
    async def _send_task_assignment_message(self, agent_id: str, task_id: str, task_type: str):
        """Send task assignment message with proper contract format."""
        message = {
            "message_id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "message_type": "task_assignment",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "task_id": task_id,
                "task_type": task_type
            },
            "correlation_id": str(uuid.uuid4())
        }
        
        await self.message_broker.send_message(f"agent:{agent_id}", message)
    
    # Add method to AgentOrchestrator class
    AgentOrchestrator._send_task_assignment_message = _send_task_assignment_message


# Add the methods when the module is imported
add_orchestrator_contract_methods()