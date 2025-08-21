"""
Contract Testing Framework for LeanVibe Agent Hive 2.0

This framework implements comprehensive contract testing for maintaining
100% integration success by validating API contracts, component interfaces,
and data flow agreements between system components.

Key Features:
- API-PWA contract validation
- Backend component interface testing
- Real-time WebSocket contract enforcement
- Performance contract monitoring
- Automated regression prevention
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

import jsonschema
from jsonschema import ValidationError, Draft7Validator
from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class ContractType(Enum):
    """Types of contracts in the system."""
    API_ENDPOINT = "api_endpoint"
    WEBSOCKET_MESSAGE = "websocket_message"
    REDIS_MESSAGE = "redis_message"
    DATABASE_SCHEMA = "database_schema"
    COMPONENT_INTERFACE = "component_interface"
    DATA_TRANSFORMATION = "data_transformation"
    PERFORMANCE_SLA = "performance_sla"


class ContractViolationSeverity(Enum):
    """Severity levels for contract violations."""
    CRITICAL = "critical"  # System breaking changes
    HIGH = "high"         # Feature breaking changes
    MEDIUM = "medium"     # Degraded functionality
    LOW = "low"           # Minor inconsistencies
    WARNING = "warning"   # Potential future issues


@dataclass
class ContractViolation:
    """Represents a contract violation with detailed context."""
    contract_id: str
    contract_type: ContractType
    severity: ContractViolationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    component: str
    violated_fields: List[str] = field(default_factory=list)
    expected_value: Any = None
    actual_value: Any = None
    suggestion: Optional[str] = None


@dataclass
class ContractValidationResult:
    """Result of contract validation."""
    contract_id: str
    is_valid: bool
    violations: List[ContractViolation] = field(default_factory=list)
    validation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContractDefinition(BaseModel):
    """Definition of a contract between components."""
    id: str = Field(..., description="Unique contract identifier")
    name: str = Field(..., description="Human-readable contract name")
    version: str = Field("1.0.0", description="Contract version")
    contract_type: ContractType = Field(..., description="Type of contract")
    description: str = Field(..., description="Contract description")
    
    # Contract specification
    schema: Dict[str, Any] = Field(..., description="JSON schema for validation")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Valid examples")
    counter_examples: List[Dict[str, Any]] = Field(default_factory=list, description="Invalid examples")
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Contract dependencies")
    consumers: List[str] = Field(default_factory=list, description="Contract consumers")
    
    # SLA and performance requirements
    performance_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    owner: str = Field(..., description="Contract owner component")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)


class ContractRegistry:
    """Registry for managing system contracts."""
    
    def __init__(self):
        self.contracts: Dict[str, ContractDefinition] = {}
        self.validators: Dict[str, Draft7Validator] = {}
        self._load_system_contracts()
    
    def _load_system_contracts(self):
        """Load system contracts from configuration."""
        # Load API-PWA contracts
        self._register_api_pwa_contracts()
        
        # Load backend component contracts
        self._register_backend_contracts()
        
        # Load WebSocket contracts
        self._register_websocket_contracts()
        
        # Load Redis messaging contracts
        self._register_redis_contracts()
        
        # Load database contracts
        self._register_database_contracts()
    
    def register_contract(self, contract: ContractDefinition):
        """Register a new contract."""
        self.contracts[contract.id] = contract
        
        # Create JSON schema validator
        try:
            self.validators[contract.id] = Draft7Validator(contract.schema)
            logger.info("Contract registered", contract_id=contract.id, type=contract.contract_type.value)
        except Exception as e:
            logger.error("Failed to register contract", contract_id=contract.id, error=str(e))
    
    def get_contract(self, contract_id: str) -> Optional[ContractDefinition]:
        """Get contract by ID."""
        return self.contracts.get(contract_id)
    
    def get_contracts_by_type(self, contract_type: ContractType) -> List[ContractDefinition]:
        """Get contracts by type."""
        return [c for c in self.contracts.values() if c.contract_type == contract_type]
    
    def get_contracts_by_component(self, component: str) -> List[ContractDefinition]:
        """Get contracts owned by a component."""
        return [c for c in self.contracts.values() if c.owner == component]
    
    def _register_api_pwa_contracts(self):
        """Register API-PWA contracts."""
        # Live data endpoint contract
        live_data_contract = ContractDefinition(
            id="api.pwa.live_data",
            name="PWA Live Data Endpoint",
            contract_type=ContractType.API_ENDPOINT,
            description="Contract for /dashboard/api/live-data endpoint",
            owner="pwa_backend",
            schema={
                "type": "object",
                "required": ["metrics", "agent_activities", "project_snapshots", "conflict_snapshots"],
                "properties": {
                    "metrics": {
                        "type": "object",
                        "required": ["active_projects", "active_agents", "agent_utilization", 
                                   "completed_tasks", "active_conflicts", "system_efficiency", 
                                   "system_status", "last_updated"],
                        "properties": {
                            "active_projects": {"type": "integer", "minimum": 0},
                            "active_agents": {"type": "integer", "minimum": 0},
                            "agent_utilization": {"type": "number", "minimum": 0, "maximum": 1},
                            "completed_tasks": {"type": "integer", "minimum": 0},
                            "active_conflicts": {"type": "integer", "minimum": 0},
                            "system_efficiency": {"type": "number", "minimum": 0, "maximum": 1},
                            "system_status": {"enum": ["healthy", "degraded", "critical"]},
                            "last_updated": {"type": "string", "format": "date-time"}
                        }
                    },
                    "agent_activities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["agent_id", "name", "status", "performance_score"],
                            "properties": {
                                "agent_id": {"type": "string"},
                                "name": {"type": "string"},
                                "status": {"enum": ["active", "idle", "busy", "error"]},
                                "performance_score": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    },
                    "project_snapshots": {"type": "array"},
                    "conflict_snapshots": {"type": "array"}
                }
            },
            performance_requirements={
                "max_response_time_ms": 100,
                "min_availability": 0.99,
                "max_payload_size_kb": 500
            },
            consumers=["mobile_pwa"]
        )
        self.register_contract(live_data_contract)
        
        # Agent management contracts
        agent_status_contract = ContractDefinition(
            id="api.agents.status",
            name="Agent Status Endpoint",
            contract_type=ContractType.API_ENDPOINT,
            description="Contract for /api/agents/status endpoint",
            owner="orchestrator",
            schema={
                "type": "object",
                "required": ["agents", "total_count", "by_status"],
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "name", "status", "capabilities"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "status": {"enum": ["active", "idle", "busy", "error", "offline"]},
                                "capabilities": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "total_count": {"type": "integer", "minimum": 0},
                    "by_status": {"type": "object"}
                }
            },
            performance_requirements={
                "max_response_time_ms": 50,
                "min_availability": 0.995
            },
            consumers=["mobile_pwa", "orchestrator"]
        )
        self.register_contract(agent_status_contract)
    
    def _register_backend_contracts(self):
        """Register backend component contracts."""
        # Configuration service contract
        config_service_contract = ContractDefinition(
            id="component.configuration_service",
            name="Configuration Service Interface",
            contract_type=ContractType.COMPONENT_INTERFACE,
            description="Contract for ConfigurationService component",
            owner="configuration_service",
            schema={
                "type": "object",
                "required": ["get_config", "update_config", "validate_config"],
                "properties": {
                    "get_config": {
                        "type": "object",
                        "description": "Get configuration method signature",
                        "properties": {
                            "input": {"type": "object", "properties": {"key": {"type": "string"}}},
                            "output": {"type": "any"}
                        }
                    },
                    "update_config": {
                        "type": "object",
                        "description": "Update configuration method signature"
                    },
                    "validate_config": {
                        "type": "object",
                        "description": "Validate configuration method signature"
                    }
                }
            },
            performance_requirements={
                "max_initialization_time_ms": 1000,
                "max_config_access_time_ms": 10
            },
            consumers=["orchestrator", "messaging_service", "agent_registry"]
        )
        self.register_contract(config_service_contract)
        
        # Messaging service contract
        messaging_service_contract = ContractDefinition(
            id="component.messaging_service",
            name="Messaging Service Interface",
            contract_type=ContractType.COMPONENT_INTERFACE,
            description="Contract for MessagingService component",
            owner="messaging_service",
            schema={
                "type": "object",
                "required": ["send_message", "receive_message", "subscribe", "unsubscribe"],
                "properties": {
                    "send_message": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "object",
                                "required": ["channel", "message"],
                                "properties": {
                                    "channel": {"type": "string"},
                                    "message": {"type": "object"}
                                }
                            },
                            "output": {"type": "boolean"}
                        }
                    },
                    "receive_message": {"type": "object"},
                    "subscribe": {"type": "object"},
                    "unsubscribe": {"type": "object"}
                }
            },
            performance_requirements={
                "max_message_latency_ms": 50,
                "min_throughput_msg_per_sec": 1000,
                "max_memory_usage_mb": 100
            },
            consumers=["orchestrator", "agent_registry", "simple_orchestrator"]
        )
        self.register_contract(messaging_service_contract)
    
    def _register_websocket_contracts(self):
        """Register WebSocket message contracts."""
        websocket_message_contract = ContractDefinition(
            id="websocket.dashboard_messages",
            name="Dashboard WebSocket Messages",
            contract_type=ContractType.WEBSOCKET_MESSAGE,
            description="Contract for dashboard WebSocket messages",
            owner="dashboard_websockets",
            schema={
                "type": "object",
                "required": ["type", "id", "timestamp"],
                "properties": {
                    "type": {"enum": ["agent_update", "task_update", "system_update", "error", "heartbeat"]},
                    "id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "data": {"type": "object"},
                    "source": {"type": "string"},
                    "target": {"type": "string"}
                }
            },
            performance_requirements={
                "max_message_size_kb": 64,
                "max_latency_ms": 50,
                "min_connection_stability": 0.95
            },
            consumers=["mobile_pwa"]
        )
        self.register_contract(websocket_message_contract)
    
    def _register_redis_contracts(self):
        """Register Redis messaging contracts."""
        redis_message_contract = ContractDefinition(
            id="redis.agent_messages",
            name="Redis Agent Messages",
            contract_type=ContractType.REDIS_MESSAGE,
            description="Contract for Redis agent communication messages",
            owner="messaging_service",
            schema={
                "type": "object",
                "required": ["message_id", "from_agent", "to_agent", "type", "payload", "timestamp"],
                "properties": {
                    "message_id": {"type": "string"},
                    "from_agent": {"type": "string", "minLength": 1},
                    "to_agent": {"type": "string", "minLength": 1},
                    "type": {"enum": ["task_assignment", "task_result", "heartbeat", "coordination", "error"]},
                    "payload": {"type": "string", "maxLength": 65536},  # 64KB limit
                    "timestamp": {"type": "string", "format": "date-time"},
                    "correlation_id": {"type": "string"},
                    "ttl": {"type": "integer", "minimum": 0, "maximum": 86400},  # Max 24 hours
                    "priority": {"enum": ["low", "normal", "high", "critical"]}
                }
            },
            performance_requirements={
                "max_message_size_kb": 64,
                "max_processing_time_ms": 5,
                "min_throughput_msg_per_sec": 500
            },
            consumers=["orchestrator", "agent_registry", "simple_orchestrator"]
        )
        self.register_contract(redis_message_contract)
    
    def _register_database_contracts(self):
        """Register database schema contracts."""
        task_model_contract = ContractDefinition(
            id="database.task_model",
            name="Task Model Schema",
            contract_type=ContractType.DATABASE_SCHEMA,
            description="Contract for Task database model",
            owner="database",
            schema={
                "type": "object",
                "required": ["id", "title", "status", "created_at"],
                "properties": {
                    "id": {"type": "string", "format": "uuid"},
                    "title": {"type": "string", "minLength": 1, "maxLength": 255},
                    "description": {"type": "string", "maxLength": 2000},
                    "status": {"enum": ["pending", "assigned", "in_progress", "completed", "failed"]},
                    "priority": {"enum": ["low", "medium", "high", "critical"]},
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"},
                    "assigned_agent_id": {"type": ["string", "null"], "format": "uuid"},
                    "workflow_id": {"type": ["string", "null"], "format": "uuid"}
                }
            },
            performance_requirements={
                "max_query_time_ms": 100,
                "max_insert_time_ms": 50
            },
            consumers=["orchestrator", "task_manager"]
        )
        self.register_contract(task_model_contract)


class ContractValidator:
    """Core contract validation engine."""
    
    def __init__(self, registry: ContractRegistry):
        self.registry = registry
        self.validation_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
    
    async def validate_contract(self, contract_id: str, data: Any, 
                              context: Optional[Dict[str, Any]] = None) -> ContractValidationResult:
        """Validate data against a contract."""
        start_time = time.perf_counter()
        
        contract = self.registry.get_contract(contract_id)
        if not contract:
            return ContractValidationResult(
                contract_id=contract_id,
                is_valid=False,
                violations=[ContractViolation(
                    contract_id=contract_id,
                    contract_type=ContractType.API_ENDPOINT,
                    severity=ContractViolationSeverity.CRITICAL,
                    message=f"Contract not found: {contract_id}",
                    details={"contract_id": contract_id},
                    timestamp=datetime.utcnow(),
                    component="contract_validator"
                )]
            )
        
        violations = []
        
        # Validate against JSON schema
        validator = self.registry.validators.get(contract_id)
        if validator:
            errors = list(validator.iter_errors(data))
            for error in errors:
                violation = ContractViolation(
                    contract_id=contract_id,
                    contract_type=contract.contract_type,
                    severity=self._determine_severity(error, contract),
                    message=error.message,
                    details={
                        "json_path": error.absolute_path,
                        "schema_path": error.schema_path,
                        "invalid_value": error.instance
                    },
                    timestamp=datetime.utcnow(),
                    component=contract.owner,
                    violated_fields=[str(error.absolute_path)] if error.absolute_path else [],
                    actual_value=error.instance
                )
                violations.append(violation)
        
        # Validate performance requirements
        if contract.performance_requirements and context:
            perf_violations = await self._validate_performance(contract, context)
            violations.extend(perf_violations)
        
        validation_time = (time.perf_counter() - start_time) * 1000
        
        # Store performance metrics
        if contract_id not in self.performance_metrics:
            self.performance_metrics[contract_id] = []
        self.performance_metrics[contract_id].append(validation_time)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics[contract_id]) > 100:
            self.performance_metrics[contract_id] = self.performance_metrics[contract_id][-100:]
        
        return ContractValidationResult(
            contract_id=contract_id,
            is_valid=len(violations) == 0,
            violations=violations,
            validation_time_ms=validation_time,
            metadata={
                "contract_version": contract.version,
                "validation_timestamp": datetime.utcnow().isoformat(),
                "context": context or {}
            }
        )
    
    def _determine_severity(self, error: ValidationError, contract: ContractDefinition) -> ContractViolationSeverity:
        """Determine severity of a validation error."""
        # Critical violations
        if "required" in error.message.lower():
            return ContractViolationSeverity.CRITICAL
        
        # High violations
        if any(keyword in error.message.lower() for keyword in ["type", "enum", "format"]):
            return ContractViolationSeverity.HIGH
        
        # Medium violations
        if any(keyword in error.message.lower() for keyword in ["minimum", "maximum", "length"]):
            return ContractViolationSeverity.MEDIUM
        
        # Default to low
        return ContractViolationSeverity.LOW
    
    async def _validate_performance(self, contract: ContractDefinition, 
                                  context: Dict[str, Any]) -> List[ContractViolation]:
        """Validate performance requirements."""
        violations = []
        requirements = contract.performance_requirements
        
        # Response time validation
        if "max_response_time_ms" in requirements and "response_time_ms" in context:
            max_time = requirements["max_response_time_ms"]
            actual_time = context["response_time_ms"]
            
            if actual_time > max_time:
                violations.append(ContractViolation(
                    contract_id=contract.id,
                    contract_type=contract.contract_type,
                    severity=ContractViolationSeverity.HIGH,
                    message=f"Response time {actual_time}ms exceeds limit of {max_time}ms",
                    details={"requirement": "max_response_time_ms", "limit": max_time, "actual": actual_time},
                    timestamp=datetime.utcnow(),
                    component=contract.owner,
                    expected_value=max_time,
                    actual_value=actual_time
                ))
        
        # Payload size validation
        if "max_payload_size_kb" in requirements and "payload_size_kb" in context:
            max_size = requirements["max_payload_size_kb"]
            actual_size = context["payload_size_kb"]
            
            if actual_size > max_size:
                violations.append(ContractViolation(
                    contract_id=contract.id,
                    contract_type=contract.contract_type,
                    severity=ContractViolationSeverity.MEDIUM,
                    message=f"Payload size {actual_size}KB exceeds limit of {max_size}KB",
                    details={"requirement": "max_payload_size_kb", "limit": max_size, "actual": actual_size},
                    timestamp=datetime.utcnow(),
                    component=contract.owner,
                    expected_value=max_size,
                    actual_value=actual_size
                ))
        
        return violations
    
    def get_performance_stats(self, contract_id: str) -> Dict[str, float]:
        """Get performance statistics for a contract."""
        if contract_id not in self.performance_metrics:
            return {}
        
        times = self.performance_metrics[contract_id]
        return {
            "avg_validation_time_ms": sum(times) / len(times),
            "min_validation_time_ms": min(times),
            "max_validation_time_ms": max(times),
            "total_validations": len(times)
        }


class ContractTestingFramework:
    """Main contract testing framework."""
    
    def __init__(self):
        self.registry = ContractRegistry()
        self.validator = ContractValidator(self.registry)
        self.test_results: List[ContractValidationResult] = []
        self.violation_history: List[ContractViolation] = []
        
    async def validate_api_endpoint(self, endpoint: str, response_data: Any, 
                                  response_time_ms: float = 0.0,
                                  payload_size_kb: float = 0.0) -> ContractValidationResult:
        """Validate API endpoint response."""
        # Map endpoint to contract ID
        contract_mapping = {
            "/dashboard/api/live-data": "api.pwa.live_data",
            "/api/agents/status": "api.agents.status"
        }
        
        contract_id = contract_mapping.get(endpoint)
        if not contract_id:
            logger.warning("No contract found for endpoint", endpoint=endpoint)
            return ContractValidationResult(
                contract_id=f"unknown.{endpoint}",
                is_valid=False,
                violations=[]
            )
        
        context = {
            "endpoint": endpoint,
            "response_time_ms": response_time_ms,
            "payload_size_kb": payload_size_kb
        }
        
        result = await self.validator.validate_contract(contract_id, response_data, context)
        self.test_results.append(result)
        self.violation_history.extend(result.violations)
        
        return result
    
    async def validate_websocket_message(self, message: Dict[str, Any]) -> ContractValidationResult:
        """Validate WebSocket message."""
        result = await self.validator.validate_contract("websocket.dashboard_messages", message)
        self.test_results.append(result)
        self.violation_history.extend(result.violations)
        return result
    
    async def validate_redis_message(self, message: Dict[str, Any]) -> ContractValidationResult:
        """Validate Redis message."""
        result = await self.validator.validate_contract("redis.agent_messages", message)
        self.test_results.append(result)
        self.violation_history.extend(result.violations)
        return result
    
    async def validate_component_interface(self, component: str, operation: str, 
                                         input_data: Any, output_data: Any) -> ContractValidationResult:
        """Validate component interface operation."""
        contract_id = f"component.{component}"
        
        # Create validation data structure
        validation_data = {
            "operation": operation,
            "input": input_data,
            "output": output_data
        }
        
        result = await self.validator.validate_contract(contract_id, validation_data)
        self.test_results.append(result)
        self.violation_history.extend(result.violations)
        return result
    
    def get_contract_health_report(self) -> Dict[str, Any]:
        """Generate contract health report."""
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.is_valid])
        
        violations_by_severity = {}
        for violation in self.violation_history:
            severity = violation.severity.value
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        violations_by_component = {}
        for violation in self.violation_history:
            component = violation.component
            violations_by_component[component] = violations_by_component.get(component, 0) + 1
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
                "total_violations": len(self.violation_history)
            },
            "violations_by_severity": violations_by_severity,
            "violations_by_component": violations_by_component,
            "contract_performance": {
                contract_id: self.validator.get_performance_stats(contract_id)
                for contract_id in self.registry.contracts.keys()
            },
            "recent_violations": [
                {
                    "contract_id": v.contract_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "component": v.component,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in sorted(self.violation_history[-10:], key=lambda x: x.timestamp, reverse=True)
            ]
        }
    
    async def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests against all contracts."""
        logger.info("Starting contract regression tests")
        
        results = {
            "total_contracts": len(self.registry.contracts),
            "tested_contracts": 0,
            "passed_contracts": 0,
            "failed_contracts": 0,
            "contract_results": {}
        }
        
        for contract_id, contract in self.registry.contracts.items():
            # Test with valid examples
            contract_passed = True
            
            for example in contract.examples:
                result = await self.validator.validate_contract(contract_id, example)
                if not result.is_valid:
                    contract_passed = False
                    logger.warning("Contract example failed validation", 
                                 contract_id=contract_id, violations=len(result.violations))
            
            # Test with counter-examples (should fail)
            for counter_example in contract.counter_examples:
                result = await self.validator.validate_contract(contract_id, counter_example)
                if result.is_valid:  # Counter-example should fail
                    contract_passed = False
                    logger.warning("Contract counter-example passed validation", 
                                 contract_id=contract_id)
            
            results["tested_contracts"] += 1
            if contract_passed:
                results["passed_contracts"] += 1
            else:
                results["failed_contracts"] += 1
            
            results["contract_results"][contract_id] = {
                "passed": contract_passed,
                "examples_tested": len(contract.examples),
                "counter_examples_tested": len(contract.counter_examples)
            }
        
        logger.info("Contract regression tests completed", 
                   passed=results["passed_contracts"], 
                   failed=results["failed_contracts"])
        
        return results


# Global contract testing framework instance
contract_framework = ContractTestingFramework()