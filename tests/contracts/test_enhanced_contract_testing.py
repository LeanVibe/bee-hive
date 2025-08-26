"""
Level 4: Enhanced Contract Testing for LeanVibe Agent Hive 2.0

Contract testing validates the interfaces between different components and services:
- API request/response schemas
- Message formats between services
- Database model contracts
- Inter-service communication protocols

This ensures that components can integrate without breaking changes.
"""

import pytest
import json
from typing import Dict, Any, List
from datetime import datetime
from unittest.mock import MagicMock
import jsonschema
from jsonschema import validate, ValidationError


class TestAgentAPIContracts:
    """Test contracts for Agent API endpoints."""
    
    @pytest.fixture
    def agent_create_request_schema(self):
        """Schema for agent creation requests."""
        return {
            "type": "object",
            "required": ["name", "type", "role"],
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100
                },
                "type": {
                    "type": "string",
                    "enum": ["CLAUDE", "OPENAI", "CUSTOM", "GEMINI"]
                },
                "role": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 50
                },
                "capabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "description"],
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence_level": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            },
                            "specialization_areas": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                },
                "system_prompt": {"type": "string"},
                "config": {"type": "object"}
            },
            "additionalProperties": False
        }
    
    @pytest.fixture
    def agent_response_schema(self):
        """Schema for agent API responses."""
        return {
            "type": "object",
            "required": ["id", "name", "type", "role", "status", "created_at"],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9-_]+$"
                },
                "name": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["CLAUDE", "OPENAI", "CUSTOM", "GEMINI"]
                },
                "role": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["ACTIVE", "INACTIVE", "ERROR", "PAUSED"]
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "updated_at": {
                    "type": "string",
                    "format": "date-time"
                },
                "capabilities": {
                    "type": "array",
                    "items": {"type": "object"}
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "tasks_completed": {"type": "integer", "minimum": 0},
                        "success_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "avg_response_time": {"type": "number", "minimum": 0.0}
                    }
                }
            },
            "additionalProperties": False
        }
    
    def test_agent_create_request_contract(self, agent_create_request_schema):
        """Test agent creation request validates against schema."""
        # Valid request
        valid_request = {
            "name": "Test Agent",
            "type": "CLAUDE",
            "role": "backend_engineer",
            "capabilities": [
                {
                    "name": "python_development",
                    "description": "Python backend development",
                    "confidence_level": 0.9,
                    "specialization_areas": ["fastapi", "asyncio"]
                }
            ],
            "system_prompt": "You are a backend engineer",
            "config": {"temperature": 0.7}
        }
        
        # Should not raise exception
        validate(instance=valid_request, schema=agent_create_request_schema)
        
        # Test required fields validation
        invalid_request = {
            "name": "Test Agent"
            # Missing required fields: type, role
        }
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_request, schema=agent_create_request_schema)
        
        # Test enum validation
        invalid_type_request = {
            "name": "Test Agent",
            "type": "INVALID_TYPE",  # Not in enum
            "role": "backend_engineer"
        }
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_type_request, schema=agent_create_request_schema)
    
    def test_agent_response_contract(self, agent_response_schema):
        """Test agent response validates against schema."""
        # Valid response
        valid_response = {
            "id": "agent-123",
            "name": "Test Agent",
            "type": "CLAUDE",
            "role": "backend_engineer",
            "status": "ACTIVE",
            "created_at": "2024-08-26T10:00:00Z",
            "updated_at": "2024-08-26T10:00:00Z",
            "capabilities": [
                {
                    "name": "python_development",
                    "description": "Python backend development"
                }
            ],
            "metrics": {
                "tasks_completed": 42,
                "success_rate": 0.95,
                "avg_response_time": 1.2
            }
        }
        
        # Should not raise exception
        validate(instance=valid_response, schema=agent_response_schema)
        
        # Test invalid status
        invalid_status_response = valid_response.copy()
        invalid_status_response["status"] = "INVALID_STATUS"
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_status_response, schema=agent_response_schema)


class TestTaskQueueContracts:
    """Test contracts for task queue message formats."""
    
    @pytest.fixture
    def task_message_schema(self):
        """Schema for task queue messages."""
        return {
            "type": "object",
            "required": ["id", "type", "payload", "metadata"],
            "properties": {
                "id": {
                    "type": "string",
                    "pattern": "^task-[a-zA-Z0-9-]+$"
                },
                "type": {
                    "type": "string",
                    "enum": ["CODE_REVIEW", "DEPLOYMENT", "TESTING", "DOCUMENTATION", "ANALYSIS"]
                },
                "payload": {
                    "type": "object",
                    "required": ["description"],
                    "properties": {
                        "description": {"type": "string", "minLength": 1},
                        "priority": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "context": {"type": "object"},
                        "deadline": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "required": ["created_at", "source"],
                    "properties": {
                        "created_at": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "source": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "retry_count": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "correlation_id": {"type": "string"}
                    }
                }
            },
            "additionalProperties": False
        }
    
    @pytest.fixture
    def task_result_schema(self):
        """Schema for task result messages."""
        return {
            "type": "object",
            "required": ["task_id", "status", "result", "metadata"],
            "properties": {
                "task_id": {
                    "type": "string",
                    "pattern": "^task-[a-zA-Z0-9-]+$"
                },
                "status": {
                    "type": "string",
                    "enum": ["SUCCESS", "FAILURE", "PARTIAL", "RETRY_NEEDED"]
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "output": {"type": "string"},
                        "artifacts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["type", "content"],
                                "properties": {
                                    "type": {"type": "string"},
                                    "content": {"type": "string"},
                                    "metadata": {"type": "object"}
                                }
                            }
                        },
                        "metrics": {
                            "type": "object",
                            "properties": {
                                "execution_time": {"type": "number", "minimum": 0},
                                "resources_used": {"type": "object"},
                                "quality_score": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "required": ["completed_at", "agent_id"],
                    "properties": {
                        "completed_at": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "agent_id": {"type": "string"},
                        "error_details": {"type": "string"},
                        "retry_attempts": {
                            "type": "integer",
                            "minimum": 0
                        }
                    }
                }
            },
            "additionalProperties": False
        }
    
    def test_task_message_contract(self, task_message_schema):
        """Test task queue message validates against schema."""
        # Valid task message
        valid_message = {
            "id": "task-abc-123",
            "type": "CODE_REVIEW",
            "payload": {
                "description": "Review PR #456 for security vulnerabilities",
                "priority": 8,
                "requirements": ["security_scan", "code_analysis"],
                "context": {
                    "repository": "leanvibe/agent-hive",
                    "branch": "feature/security-update",
                    "pr_number": 456
                },
                "deadline": "2024-08-26T18:00:00Z"
            },
            "metadata": {
                "created_at": "2024-08-26T10:00:00Z",
                "source": "github_webhook",
                "correlation_id": "req-xyz-789"
            }
        }
        
        # Should not raise exception
        validate(instance=valid_message, schema=task_message_schema)
        
        # Test invalid task type
        invalid_message = valid_message.copy()
        invalid_message["type"] = "INVALID_TASK_TYPE"
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_message, schema=task_message_schema)
        
        # Test missing required fields
        incomplete_message = {
            "id": "task-abc-123",
            "type": "CODE_REVIEW"
            # Missing payload and metadata
        }
        
        with pytest.raises(ValidationError):
            validate(instance=incomplete_message, schema=task_message_schema)
    
    def test_task_result_contract(self, task_result_schema):
        """Test task result message validates against schema."""
        # Valid task result
        valid_result = {
            "task_id": "task-abc-123",
            "status": "SUCCESS",
            "result": {
                "output": "Code review completed successfully",
                "artifacts": [
                    {
                        "type": "security_report",
                        "content": "No security vulnerabilities found",
                        "metadata": {"scan_duration": 45}
                    }
                ],
                "metrics": {
                    "execution_time": 120.5,
                    "resources_used": {"cpu": 0.3, "memory": "256MB"},
                    "quality_score": 0.92
                }
            },
            "metadata": {
                "completed_at": "2024-08-26T10:02:00Z",
                "agent_id": "agent-security-123",
                "retry_attempts": 0
            }
        }
        
        # Should not raise exception
        validate(instance=valid_result, schema=task_result_schema)
        
        # Test invalid status
        invalid_result = valid_result.copy()
        invalid_result["status"] = "INVALID_STATUS"
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_result, schema=task_result_schema)


class TestWebSocketContracts:
    """Test contracts for WebSocket message formats."""
    
    @pytest.fixture
    def websocket_message_schema(self):
        """Schema for WebSocket messages."""
        return {
            "type": "object",
            "required": ["type", "data", "timestamp"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["AGENT_STATUS", "TASK_UPDATE", "SYSTEM_ALERT", "METRICS_UPDATE"]
                },
                "data": {"type": "object"},
                "timestamp": {
                    "type": "string",
                    "format": "date-time"
                },
                "correlation_id": {"type": "string"},
                "source": {"type": "string"}
            },
            "additionalProperties": False
        }
    
    @pytest.fixture
    def agent_status_data_schema(self):
        """Schema for agent status WebSocket data."""
        return {
            "type": "object",
            "required": ["agent_id", "status", "current_task"],
            "properties": {
                "agent_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["ACTIVE", "IDLE", "BUSY", "ERROR", "OFFLINE"]
                },
                "current_task": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": ["id", "type", "progress"],
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "progress": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "estimated_completion": {
                                    "type": "string",
                                    "format": "date-time"
                                }
                            }
                        }
                    ]
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "cpu_usage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "memory_usage": {"type": "number", "minimum": 0.0},
                        "uptime": {"type": "number", "minimum": 0.0}
                    }
                }
            },
            "additionalProperties": False
        }
    
    def test_websocket_message_contract(self, websocket_message_schema):
        """Test WebSocket message validates against schema."""
        # Valid WebSocket message
        valid_message = {
            "type": "AGENT_STATUS",
            "data": {
                "agent_id": "agent-123",
                "status": "ACTIVE",
                "message": "Processing task"
            },
            "timestamp": "2024-08-26T10:00:00Z",
            "correlation_id": "ws-msg-456",
            "source": "orchestrator"
        }
        
        # Should not raise exception
        validate(instance=valid_message, schema=websocket_message_schema)
        
        # Test invalid message type
        invalid_message = valid_message.copy()
        invalid_message["type"] = "INVALID_TYPE"
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_message, schema=websocket_message_schema)
    
    def test_agent_status_data_contract(self, agent_status_data_schema):
        """Test agent status data validates against schema."""
        # Valid agent status with active task
        valid_status_with_task = {
            "agent_id": "agent-123",
            "status": "BUSY",
            "current_task": {
                "id": "task-abc-456",
                "type": "CODE_REVIEW",
                "progress": 0.65,
                "estimated_completion": "2024-08-26T10:05:00Z"
            },
            "metrics": {
                "cpu_usage": 0.35,
                "memory_usage": 512.0,
                "uptime": 3600.5
            }
        }
        
        # Should not raise exception
        validate(instance=valid_status_with_task, schema=agent_status_data_schema)
        
        # Valid agent status with no current task
        valid_status_idle = {
            "agent_id": "agent-456",
            "status": "IDLE",
            "current_task": None,
            "metrics": {
                "cpu_usage": 0.05,
                "memory_usage": 128.0,
                "uptime": 7200.0
            }
        }
        
        # Should not raise exception
        validate(instance=valid_status_idle, schema=agent_status_data_schema)
        
        # Test invalid progress value
        invalid_progress = valid_status_with_task.copy()
        invalid_progress["current_task"]["progress"] = 1.5  # > 1.0
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_progress, schema=agent_status_data_schema)


class TestDatabaseModelContracts:
    """Test contracts for database model schemas."""
    
    @pytest.fixture
    def agent_model_schema(self):
        """Schema for Agent database model."""
        return {
            "type": "object",
            "required": ["id", "name", "type", "role", "status", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "type": {"type": "string"},
                "role": {"type": "string", "minLength": 1, "maxLength": 50},
                "status": {"type": "string"},
                "capabilities": {"type": "array"},
                "system_prompt": {"type": ["string", "null"]},
                "config": {"type": ["object", "null"]},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": ["string", "null"], "format": "date-time"},
                "last_seen_at": {"type": ["string", "null"], "format": "date-time"}
            },
            "additionalProperties": True  # Allow for additional fields
        }
    
    @pytest.fixture
    def task_model_schema(self):
        """Schema for Task database model."""
        return {
            "type": "object",
            "required": ["id", "type", "status", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string"},
                "status": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "payload": {"type": ["object", "null"]},
                "result": {"type": ["object", "null"]},
                "agent_id": {"type": ["string", "null"]},
                "priority": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
                "created_at": {"type": "string", "format": "date-time"},
                "started_at": {"type": ["string", "null"], "format": "date-time"},
                "completed_at": {"type": ["string", "null"], "format": "date-time"},
                "deadline": {"type": ["string", "null"], "format": "date-time"},
                "retry_count": {"type": ["integer", "null"], "minimum": 0}
            },
            "additionalProperties": True
        }
    
    def test_agent_model_contract(self, agent_model_schema):
        """Test Agent model validates against schema."""
        # Valid agent model
        valid_agent = {
            "id": "agent-123",
            "name": "Test Agent",
            "type": "CLAUDE",
            "role": "backend_engineer",
            "status": "ACTIVE",
            "capabilities": [
                {"name": "python", "level": "expert"},
                {"name": "fastapi", "level": "advanced"}
            ],
            "system_prompt": "You are a backend engineer",
            "config": {"temperature": 0.7, "max_tokens": 1000},
            "created_at": "2024-08-26T10:00:00Z",
            "updated_at": "2024-08-26T10:00:00Z",
            "last_seen_at": "2024-08-26T10:00:00Z",
            "custom_field": "allowed"  # Additional fields allowed
        }
        
        # Should not raise exception
        validate(instance=valid_agent, schema=agent_model_schema)
        
        # Test missing required fields
        invalid_agent = {
            "id": "agent-123",
            "name": "Test Agent"
            # Missing required fields: type, role, status, created_at
        }
        
        with pytest.raises(ValidationError):
            validate(instance=invalid_agent, schema=agent_model_schema)
    
    def test_task_model_contract(self, task_model_schema):
        """Test Task model validates against schema."""
        # Valid task model
        valid_task = {
            "id": "task-abc-123",
            "type": "CODE_REVIEW",
            "status": "COMPLETED",
            "description": "Review security changes in PR #456",
            "payload": {
                "repository": "leanvibe/agent-hive",
                "pr_number": 456
            },
            "result": {
                "summary": "Review completed successfully",
                "findings": []
            },
            "agent_id": "agent-123",
            "priority": 8,
            "created_at": "2024-08-26T10:00:00Z",
            "started_at": "2024-08-26T10:01:00Z",
            "completed_at": "2024-08-26T10:03:00Z",
            "retry_count": 0
        }
        
        # Should not raise exception
        validate(instance=valid_task, schema=task_model_schema)
        
        # Test minimal valid task
        minimal_task = {
            "id": "task-minimal-456",
            "type": "DEPLOYMENT",
            "status": "PENDING",
            "created_at": "2024-08-26T10:00:00Z"
        }
        
        # Should not raise exception
        validate(instance=minimal_task, schema=task_model_schema)


class TestServiceIntegrationContracts:
    """Test contracts between different services."""
    
    def test_orchestrator_agent_contract(self):
        """Test contract between orchestrator and agent services."""
        # Mock orchestrator request to agent
        orchestrator_request = {
            "action": "assign_task",
            "task_id": "task-123",
            "task_data": {
                "type": "CODE_REVIEW",
                "description": "Review changes in feature branch",
                "priority": 7
            },
            "deadline": "2024-08-26T18:00:00Z",
            "correlation_id": "orch-req-456"
        }
        
        # Simulate agent response
        agent_response = {
            "status": "ACCEPTED",
            "task_id": "task-123",
            "agent_id": "agent-789",
            "estimated_completion": "2024-08-26T10:15:00Z",
            "correlation_id": "orch-req-456"
        }
        
        # Verify contract compliance
        assert orchestrator_request["task_id"] == agent_response["task_id"]
        assert orchestrator_request["correlation_id"] == agent_response["correlation_id"]
        assert agent_response["status"] in ["ACCEPTED", "REJECTED", "BUSY"]
        assert "agent_id" in agent_response
    
    def test_agent_task_queue_contract(self):
        """Test contract between agent and task queue services."""
        # Mock task completion notification
        task_completion = {
            "task_id": "task-123",
            "agent_id": "agent-789",
            "status": "COMPLETED",
            "result": {
                "output": "Task completed successfully",
                "execution_time": 45.2
            },
            "timestamp": "2024-08-26T10:15:00Z"
        }
        
        # Mock queue acknowledgment
        queue_ack = {
            "task_id": "task-123",
            "acknowledged": True,
            "next_task": {
                "id": "task-456",
                "type": "TESTING",
                "priority": 5
            },
            "timestamp": "2024-08-26T10:15:01Z"
        }
        
        # Verify contract compliance
        assert task_completion["task_id"] == queue_ack["task_id"]
        assert queue_ack["acknowledged"] == True
        assert "next_task" in queue_ack  # Queue can suggest next task
        
        # Test queue can also respond with no next task
        queue_ack_no_task = {
            "task_id": "task-123",
            "acknowledged": True,
            "next_task": None,
            "timestamp": "2024-08-26T10:15:01Z"
        }
        
        assert queue_ack_no_task["next_task"] is None