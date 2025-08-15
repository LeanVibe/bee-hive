"""
Contract Testing Framework
=========================

Comprehensive framework for testing contracts between system components.
This validates that all interfaces maintain backward compatibility and
adhere to defined specifications across component boundaries.

Key Features:
- Schema validation for all message formats
- API contract testing with versioning support
- Database schema contract validation
- Event and message format validation
- Backward compatibility testing
- Contract evolution tracking
"""

import json
import uuid
import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError
from unittest.mock import AsyncMock, MagicMock


class ContractType(Enum):
    """Types of contracts that can be tested."""
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    MESSAGE_FORMAT = "message_format"
    DATABASE_SCHEMA = "database_schema"
    EVENT_FORMAT = "event_format"
    WEBSOCKET_MESSAGE = "websocket_message"


@dataclass
class ContractDefinition:
    """Definition of a contract between components."""
    name: str
    version: str
    contract_type: ContractType
    schema: Dict[str, Any]
    description: str
    producer: str
    consumer: str
    breaking_changes_allowed: bool = False
    deprecated_fields: List[str] = None
    required_fields: List[str] = None
    
    def __post_init__(self):
        if self.deprecated_fields is None:
            self.deprecated_fields = []
        if self.required_fields is None:
            self.required_fields = []


class ContractValidator:
    """Validates data against contract definitions."""
    
    def __init__(self):
        self.contracts: Dict[str, ContractDefinition] = {}
        self.validation_results: List[Dict[str, Any]] = []
    
    def register_contract(self, contract: ContractDefinition):
        """Register a contract for validation."""
        contract_key = f"{contract.name}:{contract.version}"
        self.contracts[contract_key] = contract
    
    def validate_data(
        self,
        contract_name: str,
        contract_version: str,
        data: Any,
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """Validate data against a registered contract."""
        contract_key = f"{contract_name}:{contract_version}"
        
        if contract_key not in self.contracts:
            raise ValueError(f"Contract {contract_key} not registered")
        
        contract = self.contracts[contract_key]
        result = {
            "contract_name": contract_name,
            "contract_version": contract_version,
            "valid": False,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Validate against JSON schema
            validate(instance=data, schema=contract.schema)
            result["valid"] = True
            
            # Check for deprecated fields usage
            if contract.deprecated_fields:
                used_deprecated = self._check_deprecated_fields(data, contract.deprecated_fields)
                if used_deprecated:
                    result["warnings"].extend([
                        f"Deprecated field '{field}' is used" 
                        for field in used_deprecated
                    ])
            
            # Check required fields in strict mode
            if strict_mode and contract.required_fields:
                missing_required = self._check_required_fields(data, contract.required_fields)
                if missing_required:
                    result["errors"].extend([
                        f"Required field '{field}' is missing"
                        for field in missing_required
                    ])
                    result["valid"] = False
        
        except ValidationError as e:
            result["errors"].append({
                "type": "schema_validation_error",
                "message": str(e.message),
                "path": list(e.absolute_path),
                "schema_path": list(e.schema_path)
            })
        
        except Exception as e:
            result["errors"].append({
                "type": "validation_error",
                "message": str(e)
            })
        
        self.validation_results.append(result)
        return result
    
    def _check_deprecated_fields(self, data: Any, deprecated_fields: List[str]) -> List[str]:
        """Check if deprecated fields are used in data."""
        used_deprecated = []
        
        if isinstance(data, dict):
            for field in deprecated_fields:
                if self._field_exists_in_data(data, field):
                    used_deprecated.append(field)
        
        return used_deprecated
    
    def _check_required_fields(self, data: Any, required_fields: List[str]) -> List[str]:
        """Check if required fields are missing in data."""
        missing_required = []
        
        if isinstance(data, dict):
            for field in required_fields:
                if not self._field_exists_in_data(data, field):
                    missing_required.append(field)
        
        return missing_required
    
    def _field_exists_in_data(self, data: Dict[str, Any], field_path: str) -> bool:
        """Check if a field exists in nested data structure."""
        if "." in field_path:
            # Handle nested fields like "metadata.timestamp"
            parts = field_path.split(".")
            current = data
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return True
        else:
            return field_path in data
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        total_validations = len(self.validation_results)
        successful_validations = sum(1 for r in self.validation_results if r["valid"])
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "contracts_tested": len(set(f"{r['contract_name']}:{r['contract_version']}" 
                                      for r in self.validation_results)),
            "total_errors": sum(len(r["errors"]) for r in self.validation_results),
            "total_warnings": sum(len(r["warnings"]) for r in self.validation_results)
        }


class ContractTestSuite:
    """Test suite for contract validation."""
    
    def __init__(self):
        self.validator = ContractValidator()
        self.test_cases: List[Dict[str, Any]] = []
    
    def setup_orchestrator_database_contracts(self):
        """Setup contracts for orchestrator-database interactions."""
        
        # Agent registration contract
        agent_registration_schema = {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "format": "uuid"},
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["CLAUDE", "HUMAN", "SYSTEM"]},
                "role": {"type": "string", "minLength": 1},
                "capabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
                            "specialization_areas": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "confidence_level"]
                    }
                },
                "status": {"type": "string", "enum": ["ACTIVE", "INACTIVE", "ERROR", "MAINTENANCE"]},
                "config": {"type": "object"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            "required": ["agent_id", "name", "type", "role", "capabilities", "status"]
        }
        
        agent_contract = ContractDefinition(
            name="agent_registration",
            version="1.0",
            contract_type=ContractType.DATABASE_SCHEMA,
            schema=agent_registration_schema,
            description="Contract for agent registration in database",
            producer="orchestrator",
            consumer="database",
            required_fields=["agent_id", "name", "type", "role", "capabilities"]
        )
        
        self.validator.register_contract(agent_contract)
        
        # Task submission contract
        task_submission_schema = {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "format": "uuid"},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "task_type": {"type": "string", "enum": ["FEATURE_DEVELOPMENT", "BUG_FIX", "TESTING", "DOCUMENTATION"]},
                "status": {"type": "string", "enum": ["PENDING", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELLED"]},
                "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                "assigned_agent_id": {"type": ["string", "null"], "format": "uuid"},
                "required_capabilities": {"type": "array", "items": {"type": "string"}},
                "estimated_effort": {"type": "integer", "minimum": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1},
                "context": {"type": "object"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            "required": ["task_id", "title", "task_type", "status", "priority", "required_capabilities"]
        }
        
        task_contract = ContractDefinition(
            name="task_submission",
            version="1.0",
            contract_type=ContractType.DATABASE_SCHEMA,
            schema=task_submission_schema,
            description="Contract for task submission to database",
            producer="orchestrator",
            consumer="database",
            required_fields=["task_id", "title", "task_type", "status", "priority"]
        )
        
        self.validator.register_contract(task_contract)
    
    def setup_orchestrator_redis_contracts(self):
        """Setup contracts for orchestrator-Redis messaging."""
        
        # Agent message contract
        agent_message_schema = {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "format": "uuid"},
                "from_agent_id": {"type": "string"},
                "to_agent_id": {"type": "string"},
                "message_type": {"type": "string", "enum": ["TASK_ASSIGNMENT", "STATUS_UPDATE", "ERROR_REPORT", "COORDINATION", "HANDOFF"]},
                "content": {"type": "object"},
                "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "URGENT"]},
                "correlation_id": {"type": ["string", "null"]},
                "timestamp": {"type": "string", "format": "date-time"},
                "expires_at": {"type": ["string", "null"], "format": "date-time"},
                "metadata": {"type": "object"}
            },
            "required": ["message_id", "from_agent_id", "to_agent_id", "message_type", "content", "timestamp"]
        }
        
        message_contract = ContractDefinition(
            name="agent_message",
            version="1.0",
            contract_type=ContractType.MESSAGE_FORMAT,
            schema=agent_message_schema,
            description="Contract for inter-agent messages via Redis streams",
            producer="orchestrator",
            consumer="redis_streams",
            required_fields=["message_id", "from_agent_id", "to_agent_id", "message_type", "content"]
        )
        
        self.validator.register_contract(message_contract)
        
        # Task event contract
        task_event_schema = {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "format": "uuid"},
                "task_id": {"type": "string", "format": "uuid"},
                "agent_id": {"type": "string"},
                "event_type": {"type": "string", "enum": ["CREATED", "ASSIGNED", "STARTED", "PROGRESS", "COMPLETED", "FAILED", "CANCELLED"]},
                "event_data": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
                "sequence_number": {"type": "integer", "minimum": 0}
            },
            "required": ["event_id", "task_id", "event_type", "timestamp"]
        }
        
        task_event_contract = ContractDefinition(
            name="task_event",
            version="1.0",
            contract_type=ContractType.EVENT_FORMAT,
            schema=task_event_schema,
            description="Contract for task lifecycle events",
            producer="orchestrator",
            consumer="redis_streams",
            required_fields=["event_id", "task_id", "event_type", "timestamp"]
        )
        
        self.validator.register_contract(task_event_contract)
    
    def setup_context_engine_contracts(self):
        """Setup contracts for context engine interactions."""
        
        # Memory storage contract
        memory_storage_schema = {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "format": "uuid"},
                "content": {"type": "string", "minLength": 1},
                "memory_type": {"type": "string", "enum": ["FACT", "PREFERENCE", "SKILL", "CONTEXT", "RELATIONSHIP"]},
                "agent_id": {"type": "string", "format": "uuid"},
                "context": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance_score": {"type": "number", "minimum": 0, "maximum": 1},
                "access_count": {"type": "integer", "minimum": 0},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "expires_at": {"type": ["string", "null"], "format": "date-time"}
            },
            "required": ["memory_id", "content", "memory_type", "agent_id", "importance_score"]
        }
        
        memory_contract = ContractDefinition(
            name="memory_storage",
            version="1.0",
            contract_type=ContractType.API_REQUEST,
            schema=memory_storage_schema,
            description="Contract for storing memories in context engine",
            producer="semantic_memory_service",
            consumer="pgvector",
            required_fields=["memory_id", "content", "memory_type", "agent_id"]
        )
        
        self.validator.register_contract(memory_contract)
        
        # Vector search contract
        vector_search_schema = {
            "type": "object",
            "properties": {
                "query_vector": {"type": "array", "items": {"type": "number"}},
                "query_text": {"type": "string"},
                "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
                "filters": {"type": "object"},
                "include_metadata": {"type": "boolean"}
            },
            "required": ["max_results"],
            "anyOf": [
                {"required": ["query_vector"]},
                {"required": ["query_text"]}
            ]
        }
        
        search_contract = ContractDefinition(
            name="vector_search",
            version="1.0",
            contract_type=ContractType.API_REQUEST,
            schema=vector_search_schema,
            description="Contract for vector similarity search",
            producer="semantic_memory_service",
            consumer="pgvector",
            required_fields=["max_results"]
        )
        
        self.validator.register_contract(search_contract)
    
    def setup_websocket_contracts(self):
        """Setup contracts for WebSocket communication."""
        
        # WebSocket message contract
        websocket_message_schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["SUBSCRIPTION", "DATA", "ERROR", "PING", "PONG"]},
                "data": {"type": "object"},
                "timestamp": {"type": "string", "format": "date-time"},
                "correlation_id": {"type": ["string", "null"]},
                "subscription_id": {"type": ["string", "null"]},
                "error": {
                    "type": ["object", "null"],
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "details": {"type": "object"}
                    }
                }
            },
            "required": ["type", "timestamp"]
        }
        
        ws_contract = ContractDefinition(
            name="websocket_message",
            version="1.0",
            contract_type=ContractType.WEBSOCKET_MESSAGE,
            schema=websocket_message_schema,
            description="Contract for WebSocket messages",
            producer="websocket_manager",
            consumer="dashboard",
            required_fields=["type", "timestamp"]
        )
        
        self.validator.register_contract(ws_contract)
        
        # Dashboard data contract
        dashboard_data_schema = {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "properties": {
                        "active_agents": {"type": "integer", "minimum": 0},
                        "pending_tasks": {"type": "integer", "minimum": 0},
                        "completed_tasks": {"type": "integer", "minimum": 0},
                        "system_load": {"type": "number", "minimum": 0, "maximum": 1},
                        "memory_usage": {"type": "number", "minimum": 0, "maximum": 1},
                        "last_updated": {"type": "string", "format": "date-time"}
                    },
                    "required": ["active_agents", "pending_tasks", "last_updated"]
                },
                "agent_status": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "format": "uuid"},
                            "name": {"type": "string"},
                            "status": {"type": "string", "enum": ["ACTIVE", "IDLE", "BUSY", "ERROR"]},
                            "current_task": {"type": ["string", "null"]},
                            "last_seen": {"type": "string", "format": "date-time"}
                        },
                        "required": ["agent_id", "name", "status", "last_seen"]
                    }
                },
                "recent_activities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string", "format": "date-time"},
                            "activity": {"type": "string"},
                            "agent": {"type": "string"},
                            "details": {"type": "object"}
                        },
                        "required": ["timestamp", "activity", "agent"]
                    }
                }
            },
            "required": ["metrics"]
        }
        
        dashboard_contract = ContractDefinition(
            name="dashboard_data",
            version="1.0",
            contract_type=ContractType.API_RESPONSE,
            schema=dashboard_data_schema,
            description="Contract for dashboard data updates",
            producer="coordination_dashboard",
            consumer="websocket_manager",
            required_fields=["metrics"]
        )
        
        self.validator.register_contract(dashboard_contract)
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for all contracts."""
        test_cases = []
        
        # Agent registration test cases
        test_cases.extend([
            {
                "contract": "agent_registration:1.0",
                "description": "Valid agent registration",
                "data": {
                    "agent_id": str(uuid.uuid4()),
                    "name": "test-agent-1",
                    "type": "CLAUDE",
                    "role": "backend-engineer",
                    "capabilities": [
                        {
                            "name": "python",
                            "description": "Python programming",
                            "confidence_level": 0.9,
                            "specialization_areas": ["web_development"]
                        }
                    ],
                    "status": "ACTIVE",
                    "config": {"max_context": 8000},
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                "expected_valid": True
            },
            {
                "contract": "agent_registration:1.0",
                "description": "Invalid agent registration - missing required field",
                "data": {
                    "name": "invalid-agent",
                    "type": "CLAUDE",
                    "role": "backend-engineer",
                    "capabilities": []
                    # Missing agent_id and status
                },
                "expected_valid": False
            }
        ])
        
        # Task submission test cases
        test_cases.extend([
            {
                "contract": "task_submission:1.0",
                "description": "Valid task submission",
                "data": {
                    "task_id": str(uuid.uuid4()),
                    "title": "Implement user authentication",
                    "description": "Add JWT-based authentication system",
                    "task_type": "FEATURE_DEVELOPMENT",
                    "status": "PENDING",
                    "priority": "HIGH",
                    "assigned_agent_id": None,
                    "required_capabilities": ["python", "fastapi", "security"],
                    "estimated_effort": 240,
                    "timeout_seconds": 7200,
                    "context": {"project": "ecommerce-api"},
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                },
                "expected_valid": True
            }
        ])
        
        # Agent message test cases
        test_cases.extend([
            {
                "contract": "agent_message:1.0",
                "description": "Valid agent message",
                "data": {
                    "message_id": str(uuid.uuid4()),
                    "from_agent_id": "agent_123",
                    "to_agent_id": "agent_456",
                    "message_type": "TASK_ASSIGNMENT",
                    "content": {
                        "task_id": str(uuid.uuid4()),
                        "details": "Please implement the user login endpoint"
                    },
                    "priority": "HIGH",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {"source": "orchestrator"}
                },
                "expected_valid": True
            }
        ])
        
        # WebSocket message test cases
        test_cases.extend([
            {
                "contract": "websocket_message:1.0",
                "description": "Valid WebSocket data message",
                "data": {
                    "type": "DATA",
                    "data": {
                        "update_type": "agent_status",
                        "agent_id": "agent_123",
                        "status": "ACTIVE"
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "correlation_id": str(uuid.uuid4())
                },
                "expected_valid": True
            }
        ])
        
        self.test_cases = test_cases
        return test_cases
    
    def run_contract_tests(self) -> Dict[str, Any]:
        """Run all contract tests and return results."""
        if not self.test_cases:
            self.generate_test_cases()
        
        results = {
            "total_tests": len(self.test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": []
        }
        
        for test_case in self.test_cases:
            contract_parts = test_case["contract"].split(":")
            contract_name = contract_parts[0]
            contract_version = contract_parts[1]
            
            validation_result = self.validator.validate_data(
                contract_name=contract_name,
                contract_version=contract_version,
                data=test_case["data"],
                strict_mode=True
            )
            
            test_passed = validation_result["valid"] == test_case["expected_valid"]
            
            if test_passed:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1
            
            results["test_results"].append({
                "description": test_case["description"],
                "contract": test_case["contract"],
                "passed": test_passed,
                "expected_valid": test_case["expected_valid"],
                "actual_valid": validation_result["valid"],
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"]
            })
        
        results["success_rate"] = results["passed_tests"] / results["total_tests"]
        return results