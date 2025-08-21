"""
Contract Testing Framework Integration Tests

These tests validate the contract testing framework itself and demonstrate
how to use it for maintaining 100% integration success in LeanVibe Agent Hive 2.0.
"""

import asyncio
import json
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.core.contract_testing_framework import (
    ContractTestingFramework, ContractType, ContractViolationSeverity,
    ContractDefinition, ContractViolation, ContractValidationResult,
    contract_framework
)


class TestContractTestingFramework:
    """Test the contract testing framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ContractTestingFramework()
    
    def test_framework_initialization(self):
        """Test framework initializes with system contracts."""
        assert len(self.framework.registry.contracts) > 0
        
        # Check that key contracts are loaded
        expected_contracts = [
            "api.pwa.live_data",
            "api.agents.status", 
            "component.configuration_service",
            "component.messaging_service",
            "websocket.dashboard_messages",
            "redis.agent_messages",
            "database.task_model"
        ]
        
        for contract_id in expected_contracts:
            assert contract_id in self.framework.registry.contracts
            contract = self.framework.registry.get_contract(contract_id)
            assert contract is not None
            assert contract.schema is not None
    
    @pytest.mark.asyncio
    async def test_api_pwa_live_data_contract_validation(self):
        """Test API-PWA live data contract validation."""
        # Valid live data response
        valid_response = {
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
                    "agent_id": "agent-001",
                    "name": "Dev Agent Python",
                    "status": "active",
                    "performance_score": 0.85
                }
            ],
            "project_snapshots": [],
            "conflict_snapshots": []
        }
        
        result = await self.framework.validate_api_endpoint(
            "/dashboard/api/live-data", 
            valid_response,
            response_time_ms=45.0,
            payload_size_kb=12.5
        )
        
        assert result.is_valid
        assert len(result.violations) == 0
        assert result.validation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_api_contract_violation_detection(self):
        """Test contract violation detection."""
        # Invalid response - missing required fields
        invalid_response = {
            "metrics": {
                "active_projects": 3,
                # Missing required fields: active_agents, agent_utilization, etc.
                "system_status": "healthy"
            },
            "agent_activities": [],
            # Missing: project_snapshots, conflict_snapshots
        }
        
        result = await self.framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            invalid_response
        )
        
        assert not result.is_valid
        assert len(result.violations) > 0
        
        # Check that violations include missing required fields
        violation_messages = [v.message for v in result.violations]
        assert any("required" in msg.lower() for msg in violation_messages)
    
    @pytest.mark.asyncio
    async def test_performance_contract_violation(self):
        """Test performance contract violation detection."""
        valid_response = {
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
            "agent_activities": [],
            "project_snapshots": [],
            "conflict_snapshots": []
        }
        
        # Simulate slow response time (exceeds 100ms limit)
        result = await self.framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            valid_response,
            response_time_ms=150.0  # Exceeds 100ms limit
        )
        
        # Should be invalid due to performance violation
        assert not result.is_valid
        assert len(result.violations) > 0
        
        # Check for performance violation
        perf_violations = [v for v in result.violations if "response time" in v.message.lower()]
        assert len(perf_violations) > 0
        assert perf_violations[0].severity == ContractViolationSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_websocket_message_contract_validation(self):
        """Test WebSocket message contract validation."""
        # Valid WebSocket message
        valid_message = {
            "type": "agent_update",
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "agent_id": "agent-001",
                "status": "active",
                "current_task": "task-123"
            },
            "source": "orchestrator",
            "target": "dashboard"
        }
        
        result = await self.framework.validate_websocket_message(valid_message)
        
        assert result.is_valid
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_redis_message_contract_validation(self):
        """Test Redis message contract validation."""
        # Valid Redis message
        valid_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "orchestrator-001",
            "to_agent": "agent-backend-001",
            "type": "task_assignment",
            "payload": json.dumps({"task_id": "task-123", "description": "Test task"}),
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(uuid.uuid4()),
            "priority": "normal"
        }
        
        result = await self.framework.validate_redis_message(valid_message)
        
        assert result.is_valid
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_redis_message_size_limit_violation(self):
        """Test Redis message size limit enforcement."""
        # Message with oversized payload (exceeds 64KB)
        oversized_payload = {"data": "x" * 70000}  # 70KB payload
        
        invalid_message = {
            "message_id": str(uuid.uuid4()),
            "from_agent": "test-agent",
            "to_agent": "target-agent",
            "type": "task_assignment",
            "payload": json.dumps(oversized_payload),
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(uuid.uuid4())
        }
        
        result = await self.framework.validate_redis_message(invalid_message)
        
        assert not result.is_valid
        assert len(result.violations) > 0
        
        # Check for size violation
        size_violations = [v for v in result.violations if "maxlength" in v.message.lower()]
        assert len(size_violations) > 0
    
    @pytest.mark.asyncio
    async def test_component_interface_validation(self):
        """Test component interface validation."""
        # Test configuration service interface
        config_input = {"key": "database.connection_string"}
        config_output = "postgresql://localhost:5432/leanvibe"
        
        result = await self.framework.validate_component_interface(
            "configuration_service",
            "get_config",
            config_input,
            config_output
        )
        
        # Note: This will create a basic validation structure
        # In practice, component interfaces would have more detailed contracts
        assert result.contract_id == "component.configuration_service"
    
    def test_contract_health_report(self):
        """Test contract health report generation."""
        # Add some test results
        self.framework.test_results = [
            ContractValidationResult(
                contract_id="test.contract.1",
                is_valid=True,
                violations=[],
                validation_time_ms=5.0
            ),
            ContractValidationResult(
                contract_id="test.contract.2",
                is_valid=False,
                violations=[
                    ContractViolation(
                        contract_id="test.contract.2",
                        contract_type=ContractType.API_ENDPOINT,
                        severity=ContractViolationSeverity.HIGH,
                        message="Test violation",
                        details={},
                        timestamp=datetime.utcnow(),
                        component="test_component"
                    )
                ],
                validation_time_ms=8.0
            )
        ]
        
        self.framework.violation_history = [
            violation for result in self.framework.test_results 
            for violation in result.violations
        ]
        
        report = self.framework.get_contract_health_report()
        
        assert "summary" in report
        assert report["summary"]["total_tests"] == 2
        assert report["summary"]["successful_tests"] == 1
        assert report["summary"]["success_rate"] == 0.5
        assert report["summary"]["total_violations"] == 1
        
        assert "violations_by_severity" in report
        assert "high" in report["violations_by_severity"]
        
        assert "violations_by_component" in report
        assert "test_component" in report["violations_by_component"]
    
    @pytest.mark.asyncio
    async def test_regression_tests(self):
        """Test contract regression testing."""
        # Add example data to contracts for testing
        contract = self.framework.registry.get_contract("api.pwa.live_data")
        if contract:
            # Add valid example
            contract.examples = [{
                "metrics": {
                    "active_projects": 1,
                    "active_agents": 2,
                    "agent_utilization": 0.5,
                    "completed_tasks": 10,
                    "active_conflicts": 0,
                    "system_efficiency": 0.95,
                    "system_status": "healthy",
                    "last_updated": datetime.utcnow().isoformat()
                },
                "agent_activities": [],
                "project_snapshots": [],
                "conflict_snapshots": []
            }]
            
            # Add invalid counter-example
            contract.counter_examples = [{
                "metrics": {
                    "active_projects": "invalid_type",  # Should be integer
                    "system_status": "invalid_status"   # Should be enum value
                },
                "agent_activities": [],
                "project_snapshots": [],
                "conflict_snapshots": []
            }]
        
        results = await self.framework.run_regression_tests()
        
        assert "total_contracts" in results
        assert "tested_contracts" in results
        assert "passed_contracts" in results
        assert "failed_contracts" in results
        assert results["total_contracts"] > 0


class TestContractFrameworkIntegration:
    """Test contract framework integration with existing system."""
    
    @pytest.mark.asyncio
    async def test_api_pwa_integration_contract_validation(self):
        """Test real API-PWA integration contract validation."""
        
        # This would typically be called from the actual API endpoint
        async def mock_api_endpoint():
            """Mock API endpoint that validates its response."""
            
            # Simulate API response generation
            response_data = {
                "metrics": {
                    "active_projects": 2,
                    "active_agents": 3,
                    "agent_utilization": 0.67,
                    "completed_tasks": 15,
                    "active_conflicts": 0,
                    "system_efficiency": 0.89,
                    "system_status": "healthy",
                    "last_updated": datetime.utcnow().isoformat()
                },
                "agent_activities": [
                    {
                        "agent_id": "agent-qa-001",
                        "name": "QA Engineer Agent",
                        "status": "active",
                        "performance_score": 0.92
                    }
                ],
                "project_snapshots": [],
                "conflict_snapshots": []
            }
            
            # Measure response time
            start_time = time.perf_counter()
            
            # Simulate processing time
            await asyncio.sleep(0.02)  # 20ms
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            # Validate contract before returning response
            result = await contract_framework.validate_api_endpoint(
                "/dashboard/api/live-data",
                response_data,
                response_time_ms=response_time_ms,
                payload_size_kb=len(json.dumps(response_data)) / 1024
            )
            
            if not result.is_valid:
                # In production, this would log violations and potentially
                # trigger alerts or return a fallback response
                raise ValueError(f"Contract violation: {[v.message for v in result.violations]}")
            
            return response_data
        
        # Test the integration
        response = await mock_api_endpoint()
        
        assert "metrics" in response
        assert response["metrics"]["system_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_backend_component_contract_enforcement(self):
        """Test backend component contract enforcement."""
        
        class MockConfigurationService:
            """Mock configuration service with contract validation."""
            
            async def get_config(self, key: str) -> Any:
                """Get configuration with contract validation."""
                
                # Simulate configuration retrieval
                config_result = f"config_value_for_{key}"
                
                # Validate interface contract
                result = await contract_framework.validate_component_interface(
                    "configuration_service",
                    "get_config",
                    {"key": key},
                    config_result
                )
                
                if not result.is_valid:
                    raise ValueError(f"Interface contract violation: {result.violations}")
                
                return config_result
        
        # Test the service
        service = MockConfigurationService()
        config_value = await service.get_config("test.setting")
        
        assert config_value == "config_value_for_test.setting"
    
    @pytest.mark.asyncio
    async def test_websocket_contract_middleware(self):
        """Test WebSocket contract validation middleware."""
        
        class MockWebSocketManager:
            """Mock WebSocket manager with contract validation."""
            
            def __init__(self):
                self.connected_clients = set()
            
            async def broadcast_message(self, message_type: str, data: Dict[str, Any]):
                """Broadcast message with contract validation."""
                
                message = {
                    "type": message_type,
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                }
                
                # Validate contract before broadcasting
                result = await contract_framework.validate_websocket_message(message)
                
                if not result.is_valid:
                    # Log violation and don't broadcast invalid message
                    violations = [v.message for v in result.violations]
                    raise ValueError(f"WebSocket message contract violation: {violations}")
                
                # Simulate broadcasting to connected clients
                for client in self.connected_clients:
                    # In practice, this would send the message via WebSocket
                    pass
                
                return len(self.connected_clients)
        
        # Test the WebSocket manager
        ws_manager = MockWebSocketManager()
        ws_manager.connected_clients.add("client-1")
        ws_manager.connected_clients.add("client-2")
        
        # Valid message should broadcast successfully
        client_count = await ws_manager.broadcast_message(
            "agent_update",
            {"agent_id": "agent-001", "status": "active"}
        )
        
        assert client_count == 2
        
        # Invalid message should raise contract violation
        with pytest.raises(ValueError, match="contract violation"):
            await ws_manager.broadcast_message(
                "invalid_type",  # Not in allowed enum values
                {"data": "test"}
            )
    
    def test_contract_framework_performance(self):
        """Test contract framework performance meets requirements."""
        
        # Test validation performance
        validation_times = []
        
        test_data = {
            "metrics": {
                "active_projects": 1,
                "active_agents": 1,
                "agent_utilization": 0.5,
                "completed_tasks": 5,
                "active_conflicts": 0,
                "system_efficiency": 0.9,
                "system_status": "healthy",
                "last_updated": datetime.utcnow().isoformat()
            },
            "agent_activities": [],
            "project_snapshots": [],
            "conflict_snapshots": []
        }
        
        # Run multiple validations to test performance
        async def performance_test():
            for _ in range(100):
                start_time = time.perf_counter()
                
                result = await contract_framework.validate_api_endpoint(
                    "/dashboard/api/live-data",
                    test_data
                )
                
                end_time = time.perf_counter()
                validation_times.append((end_time - start_time) * 1000)
                
                assert result.is_valid
        
        # Run the performance test
        asyncio.run(performance_test())
        
        # Analyze performance
        avg_time = sum(validation_times) / len(validation_times)
        max_time = max(validation_times)
        
        # Contract validation should be fast (< 5ms average, < 10ms max)
        assert avg_time < 5.0, f"Average validation time {avg_time:.2f}ms exceeds 5ms target"
        assert max_time < 10.0, f"Max validation time {max_time:.2f}ms exceeds 10ms target"
    
    def test_contract_violation_monitoring(self):
        """Test contract violation monitoring and alerting."""
        
        # Simulate a series of contract violations
        violations = [
            ContractViolation(
                contract_id="api.pwa.live_data",
                contract_type=ContractType.API_ENDPOINT,
                severity=ContractViolationSeverity.HIGH,
                message="Response time exceeded limit",
                details={"limit": 100, "actual": 150},
                timestamp=datetime.utcnow(),
                component="pwa_backend"
            ),
            ContractViolation(
                contract_id="redis.agent_messages", 
                contract_type=ContractType.REDIS_MESSAGE,
                severity=ContractViolationSeverity.CRITICAL,
                message="Message size exceeded limit",
                details={"limit": 65536, "actual": 70000},
                timestamp=datetime.utcnow(),
                component="messaging_service"
            )
        ]
        
        contract_framework.violation_history.extend(violations)
        
        # Generate monitoring report
        health_report = contract_framework.get_contract_health_report()
        
        # Check that violations are properly categorized
        assert "violations_by_severity" in health_report
        assert "high" in health_report["violations_by_severity"]
        assert "critical" in health_report["violations_by_severity"]
        
        assert "violations_by_component" in health_report
        assert "pwa_backend" in health_report["violations_by_component"]
        assert "messaging_service" in health_report["violations_by_component"]
        
        # Check recent violations
        assert "recent_violations" in health_report
        assert len(health_report["recent_violations"]) > 0


if __name__ == "__main__":
    # Run contract framework tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])