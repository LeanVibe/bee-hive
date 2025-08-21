"""
API-PWA Contract Integration Tests

These tests specifically validate the contracts between the API backend
and PWA frontend, ensuring the 100% integration success is maintained
through automated contract validation.
"""

import asyncio
import json
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import httpx

from app.core.contract_testing_framework import (
    contract_framework, ContractType, ContractViolationSeverity
)
from app.api.pwa_backend import (
    LiveDataResponse, SystemMetrics, AgentActivity,
    ProjectSnapshot, ConflictSnapshot
)


class TestAPIPWAContractIntegration:
    """Test API-PWA contract integration."""
    
    @pytest.mark.asyncio
    async def test_live_data_endpoint_contract_compliance(self):
        """Test /dashboard/api/live-data endpoint contract compliance."""
        
        # Create a valid response using Pydantic models
        system_metrics = SystemMetrics(
            active_projects=3,
            active_agents=5,
            agent_utilization=0.75,
            completed_tasks=42,
            active_conflicts=1,
            system_efficiency=0.92,
            system_status="healthy"
        )
        
        agent_activities = [
            AgentActivity(
                agent_id="agent-qa-001",
                name="QA Engineer Agent",
                status="active",
                current_project="LeanVibe Agent Hive 2.0",
                current_task="Contract Testing Framework Implementation",
                task_progress=0.85,
                performance_score=0.94,
                capabilities=["contract_testing", "qa_engineering", "python"]
            ),
            AgentActivity(
                agent_id="agent-backend-001", 
                name="Backend Developer Agent",
                status="busy",
                current_project="API Integration",
                current_task="API-PWA Contract Validation",
                task_progress=0.72,
                performance_score=0.89,
                capabilities=["fastapi", "python", "postgresql"]
            )
        ]
        
        project_snapshots = [
            ProjectSnapshot(
                project_id="project-contract-testing",
                name="Contract Testing Framework",
                status="active",
                progress=0.85,
                assigned_agents=["agent-qa-001", "agent-backend-001"],
                priority="high"
            )
        ]
        
        conflict_snapshots = []
        
        live_data_response = LiveDataResponse(
            metrics=system_metrics,
            agent_activities=agent_activities,
            project_snapshots=project_snapshots,
            conflict_snapshots=conflict_snapshots
        )
        
        # Convert to dict for contract validation
        response_data = live_data_response.dict()
        
        # Validate contract
        start_time = time.perf_counter()
        result = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            response_data,
            response_time_ms=25.0,  # Well within 100ms limit
            payload_size_kb=len(json.dumps(response_data)) / 1024
        )
        validation_time = (time.perf_counter() - start_time) * 1000
        
        # Assertions
        assert result.is_valid, f"Contract violations: {[v.message for v in result.violations]}"
        assert len(result.violations) == 0
        assert validation_time < 10.0  # Contract validation should be fast
        
        # Verify response structure matches PWA expectations
        assert "metrics" in response_data
        assert "agent_activities" in response_data
        assert "project_snapshots" in response_data
        assert "conflict_snapshots" in response_data
        
        # Verify metrics structure
        metrics = response_data["metrics"]
        assert metrics["system_status"] in ["healthy", "degraded", "critical"]
        assert 0 <= metrics["agent_utilization"] <= 1
        assert 0 <= metrics["system_efficiency"] <= 1
        
        # Verify agent activities structure
        for activity in response_data["agent_activities"]:
            assert "agent_id" in activity
            assert "name" in activity
            assert activity["status"] in ["active", "idle", "busy", "error"]
            assert 0 <= activity["performance_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_agent_status_endpoint_contract_compliance(self):
        """Test /api/agents/status endpoint contract compliance."""
        
        # Mock agent status response
        agent_status_response = {
            "agents": [
                {
                    "id": "agent-qa-001",
                    "name": "QA Engineer Agent",
                    "status": "active",
                    "capabilities": ["contract_testing", "integration_testing", "python"]
                },
                {
                    "id": "agent-backend-001",
                    "name": "Backend Developer Agent", 
                    "status": "busy",
                    "capabilities": ["fastapi", "python", "postgresql", "redis"]
                },
                {
                    "id": "agent-frontend-001",
                    "name": "Frontend Developer Agent",
                    "status": "idle",
                    "capabilities": ["typescript", "react", "pwa"]
                }
            ],
            "total_count": 3,
            "by_status": {
                "active": 1,
                "busy": 1,
                "idle": 1,
                "error": 0,
                "offline": 0
            }
        }
        
        # Validate contract
        result = await contract_framework.validate_api_endpoint(
            "/api/agents/status",
            agent_status_response,
            response_time_ms=15.0,  # Well within 50ms limit
            payload_size_kb=len(json.dumps(agent_status_response)) / 1024
        )
        
        # Assertions
        assert result.is_valid, f"Contract violations: {[v.message for v in result.violations]}"
        assert len(result.violations) == 0
        
        # Verify response structure
        assert "agents" in agent_status_response
        assert "total_count" in agent_status_response
        assert "by_status" in agent_status_response
        
        # Verify agent structure
        for agent in agent_status_response["agents"]:
            assert "id" in agent
            assert "name" in agent
            assert agent["status"] in ["active", "idle", "busy", "error", "offline"]
            assert "capabilities" in agent
            assert isinstance(agent["capabilities"], list)
    
    @pytest.mark.asyncio
    async def test_contract_violation_scenarios(self):
        """Test various contract violation scenarios."""
        
        # Scenario 1: Missing required fields
        invalid_response_1 = {
            "metrics": {
                "active_projects": 3,
                # Missing required fields: active_agents, agent_utilization, etc.
                "system_status": "healthy"
            },
            "agent_activities": [],
            # Missing: project_snapshots, conflict_snapshots
        }
        
        result_1 = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            invalid_response_1
        )
        
        assert not result_1.is_valid
        assert len(result_1.violations) > 0
        assert any(v.severity == ContractViolationSeverity.CRITICAL for v in result_1.violations)
        
        # Scenario 2: Invalid data types
        invalid_response_2 = {
            "metrics": {
                "active_projects": "three",  # Should be integer
                "active_agents": 5,
                "agent_utilization": 1.5,    # Should be <= 1.0
                "completed_tasks": 42,
                "active_conflicts": 1,
                "system_efficiency": 0.92,
                "system_status": "excellent",  # Should be enum value
                "last_updated": datetime.utcnow().isoformat()
            },
            "agent_activities": [],
            "project_snapshots": [],
            "conflict_snapshots": []
        }
        
        result_2 = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            invalid_response_2
        )
        
        assert not result_2.is_valid
        assert len(result_2.violations) >= 3  # Type error, range error, enum error
        
        # Scenario 3: Performance violation
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
        
        result_3 = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            valid_response,
            response_time_ms=250.0  # Exceeds 100ms limit
        )
        
        assert not result_3.is_valid
        perf_violations = [v for v in result_3.violations if "response time" in v.message.lower()]
        assert len(perf_violations) > 0
        assert perf_violations[0].severity == ContractViolationSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_pwa_service_layer_contract_compatibility(self):
        """Test PWA service layer contract compatibility."""
        
        # Simulate the PWA BackendAdapter service expectations
        pwa_expected_structure = {
            "metrics": {
                "active_projects": "number",
                "active_agents": "number", 
                "agent_utilization": "number",
                "completed_tasks": "number",
                "active_conflicts": "number",
                "system_efficiency": "number",
                "system_status": "string",
                "last_updated": "string"
            },
            "agent_activities": [
                {
                    "agent_id": "string",
                    "name": "string",
                    "status": "string",
                    "current_project": "string_optional",
                    "current_task": "string_optional",
                    "task_progress": "number_optional",
                    "performance_score": "number",
                    "specializations": "array"
                }
            ],
            "project_snapshots": "array",
            "conflict_snapshots": "array"
        }
        
        # Create backend response that matches PWA expectations
        backend_response = {
            "metrics": {
                "active_projects": 2,
                "active_agents": 4,
                "agent_utilization": 0.68,
                "completed_tasks": 28,
                "active_conflicts": 0,
                "system_efficiency": 0.91,
                "system_status": "healthy",
                "last_updated": datetime.utcnow().isoformat()
            },
            "agent_activities": [
                {
                    "agent_id": "agent-001",
                    "name": "Contract Testing Agent",
                    "status": "active",
                    "current_project": "LeanVibe Agent Hive 2.0",
                    "current_task": "PWA Contract Validation",
                    "task_progress": 0.78,
                    "performance_score": 0.92,
                    "specializations": ["contract_testing", "integration_testing"]
                }
            ],
            "project_snapshots": [
                {
                    "name": "Contract Testing Project",
                    "status": "active",
                    "progress_percentage": 0.78,
                    "participating_agents": ["agent-001"],
                    "completed_tasks": 12,
                    "active_tasks": 3,
                    "conflicts": 0,
                    "quality_score": 0.94
                }
            ],
            "conflict_snapshots": []
        }
        
        # Validate that backend response meets contract
        result = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            backend_response,
            response_time_ms=35.0
        )
        
        assert result.is_valid, f"Backend-PWA compatibility issue: {[v.message for v in result.violations]}"
        
        # Verify specific PWA service layer requirements
        metrics = backend_response["metrics"]
        assert isinstance(metrics["active_projects"], int)
        assert isinstance(metrics["agent_utilization"], float)
        assert metrics["system_status"] in ["healthy", "degraded", "critical"]
        
        # Verify agent activities match PWA expectations
        for activity in backend_response["agent_activities"]:
            assert isinstance(activity["agent_id"], str)
            assert isinstance(activity["name"], str)
            assert activity["status"] in ["active", "idle", "busy", "error"]
            assert isinstance(activity["specializations"], list)
            if "performance_score" in activity:
                assert 0 <= activity["performance_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_websocket_message_contract_integration(self):
        """Test WebSocket message contract integration with PWA."""
        
        # Test various WebSocket message types expected by PWA
        websocket_messages = [
            {
                "type": "agent_update",
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "agent_id": "agent-001",
                    "status": "active",
                    "current_task": "contract_validation",
                    "progress": 0.85
                },
                "source": "orchestrator",
                "target": "dashboard"
            },
            {
                "type": "system_update",
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "system_status": "healthy",
                    "active_agents": 5,
                    "active_projects": 3
                },
                "source": "system_monitor",
                "target": "dashboard"
            },
            {
                "type": "heartbeat",
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "status": "alive",
                    "uptime": 3600
                },
                "source": "system",
                "target": "dashboard"
            }
        ]
        
        # Validate all WebSocket messages
        for message in websocket_messages:
            result = await contract_framework.validate_websocket_message(message)
            assert result.is_valid, f"WebSocket message validation failed: {result.violations}"
            
            # Verify message structure expected by PWA
            assert "type" in message
            assert "id" in message
            assert "timestamp" in message
            assert "data" in message
    
    @pytest.mark.asyncio
    async def test_error_response_contract_consistency(self):
        """Test error response contract consistency."""
        
        # Test various error response formats
        error_responses = [
            {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "field": "agent_id",
                    "issue": "required field missing"
                }
            },
            {
                "error": "NotFoundError",
                "message": "Agent not found",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "agent_id": "agent-nonexistent"
                }
            },
            {
                "error": "ContractViolationError",
                "message": "Response violates API contract",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "contract_id": "api.pwa.live_data",
                    "violated_fields": ["metrics.system_status"],
                    "expected": ["healthy", "degraded", "critical"],
                    "actual": "invalid_status"
                }
            }
        ]
        
        # Verify error response consistency
        for error_response in error_responses:
            assert "error" in error_response
            assert "message" in error_response
            assert "timestamp" in error_response
            assert isinstance(error_response["error"], str)
            assert isinstance(error_response["message"], str)
            
            # Verify timestamp format
            try:
                datetime.fromisoformat(error_response["timestamp"])
            except ValueError:
                pytest.fail(f"Invalid timestamp format: {error_response['timestamp']}")
    
    @pytest.mark.asyncio
    async def test_contract_performance_under_load(self):
        """Test contract validation performance under load."""
        
        # Create a realistic load test scenario
        test_responses = []
        for i in range(50):
            response = {
                "metrics": {
                    "active_projects": i % 10,
                    "active_agents": (i % 5) + 1,
                    "agent_utilization": (i % 100) / 100.0,
                    "completed_tasks": i * 2,
                    "active_conflicts": i % 3,
                    "system_efficiency": 0.8 + (i % 20) / 100.0,
                    "system_status": ["healthy", "degraded", "critical"][i % 3],
                    "last_updated": datetime.utcnow().isoformat()
                },
                "agent_activities": [
                    {
                        "agent_id": f"agent-{j:03d}",
                        "name": f"Agent {j}",
                        "status": ["active", "idle", "busy"][j % 3],
                        "performance_score": 0.7 + (j % 30) / 100.0
                    }
                    for j in range(i % 5 + 1)
                ],
                "project_snapshots": [],
                "conflict_snapshots": []
            }
            test_responses.append(response)
        
        # Measure validation performance
        start_time = time.perf_counter()
        validation_results = []
        
        for i, response in enumerate(test_responses):
            result = await contract_framework.validate_api_endpoint(
                "/dashboard/api/live-data",
                response,
                response_time_ms=20.0 + (i % 10)  # Vary response times
            )
            validation_results.append(result)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance assertions
        avg_validation_time = (total_time / len(test_responses)) * 1000  # Convert to ms
        success_rate = sum(1 for r in validation_results if r.is_valid) / len(validation_results)
        
        assert avg_validation_time < 5.0, f"Average validation time {avg_validation_time:.2f}ms exceeds 5ms limit"
        assert success_rate == 1.0, f"Success rate {success_rate:.2%} is not 100%"
        assert total_time < 1.0, f"Total validation time {total_time:.2f}s exceeds 1s for 50 validations"
    
    def test_contract_framework_health_monitoring(self):
        """Test contract framework health monitoring."""
        
        # Get health report
        health_report = contract_framework.get_contract_health_report()
        
        # Verify report structure
        assert "summary" in health_report
        assert "violations_by_severity" in health_report
        assert "violations_by_component" in health_report
        assert "contract_performance" in health_report
        assert "recent_violations" in health_report
        
        # Verify summary metrics
        summary = health_report["summary"]
        assert "total_tests" in summary
        assert "successful_tests" in summary
        assert "success_rate" in summary
        assert "total_violations" in summary
        
        # Verify performance metrics are collected
        if summary["total_tests"] > 0:
            assert 0 <= summary["success_rate"] <= 1
            assert summary["successful_tests"] <= summary["total_tests"]


class TestRealTimeContractMonitoring:
    """Test real-time contract monitoring integration."""
    
    @pytest.mark.asyncio
    async def test_continuous_contract_monitoring(self):
        """Test continuous contract monitoring during API operations."""
        
        async def simulate_api_operations():
            """Simulate continuous API operations with contract monitoring."""
            
            operations = []
            
            for i in range(10):
                # Simulate API call
                response_data = {
                    "metrics": {
                        "active_projects": i + 1,
                        "active_agents": (i % 3) + 2,
                        "agent_utilization": 0.5 + (i * 0.05),
                        "completed_tasks": i * 5,
                        "active_conflicts": i % 2,
                        "system_efficiency": 0.85 + (i * 0.01),
                        "system_status": "healthy",
                        "last_updated": datetime.utcnow().isoformat()
                    },
                    "agent_activities": [],
                    "project_snapshots": [],
                    "conflict_snapshots": []
                }
                
                # Validate contract for each operation
                result = await contract_framework.validate_api_endpoint(
                    "/dashboard/api/live-data",
                    response_data,
                    response_time_ms=25.0 + (i * 2)  # Gradually increasing response time
                )
                
                operations.append({
                    "operation_id": i,
                    "result": result,
                    "timestamp": datetime.utcnow()
                })
                
                # Simulate some processing time
                await asyncio.sleep(0.01)
            
            return operations
        
        # Run simulation
        operations = await simulate_api_operations()
        
        # Analyze results
        assert len(operations) == 10
        
        valid_operations = [op for op in operations if op["result"].is_valid]
        assert len(valid_operations) == 10, "All operations should be contract-compliant"
        
        # Check that monitoring captured all operations
        health_report = contract_framework.get_contract_health_report()
        assert health_report["summary"]["total_tests"] >= 10
    
    @pytest.mark.asyncio
    async def test_contract_violation_alerting(self):
        """Test contract violation alerting system."""
        
        # Simulate a contract violation scenario
        invalid_response = {
            "metrics": {
                "active_projects": -1,  # Invalid: negative value
                "active_agents": 5,
                "agent_utilization": 1.5,  # Invalid: > 1.0
                "completed_tasks": 42,
                "active_conflicts": 1,
                "system_efficiency": 0.92,
                "system_status": "broken",  # Invalid: not in enum
                "last_updated": "invalid-date"  # Invalid: bad format
            },
            "agent_activities": [],
            "project_snapshots": [],
            "conflict_snapshots": []
        }
        
        # This would trigger alerts in a real system
        result = await contract_framework.validate_api_endpoint(
            "/dashboard/api/live-data",
            invalid_response,
            response_time_ms=200.0  # Also violates performance contract
        )
        
        assert not result.is_valid
        assert len(result.violations) >= 4  # Multiple violations
        
        # Verify violation severity classification
        severities = [v.severity for v in result.violations]
        assert ContractViolationSeverity.HIGH in severities or ContractViolationSeverity.CRITICAL in severities
        
        # Check that violations are properly logged
        health_report = contract_framework.get_contract_health_report()
        recent_violations = health_report["recent_violations"]
        
        assert len(recent_violations) > 0
        assert any(v["contract_id"] == "api.pwa.live_data" for v in recent_violations)


if __name__ == "__main__":
    # Run API-PWA contract integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])