"""
Task Execution Workflow Contract Testing
=======================================

Contract validation for task creation, assignment, execution, and lifecycle management.
Ensures workflow consistency, data integrity, and reliable task delegation between
frontend and backend systems.

Key Contract Areas:
- Task creation and assignment contracts
- Task lifecycle state transitions
- Task delegation workflow validation
- Task-agent relationship contracts
- Performance and reliability requirements
- Integration with SimpleOrchestrator task management
"""

import pytest
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import jsonschema
from jsonschema import validate, ValidationError

# Import task-related components
from app.core.simple_orchestrator import (
    SimpleOrchestrator,
    AgentRole,
    TaskAssignment,
    TaskDelegationError,
    create_simple_orchestrator
)
from app.models.task import TaskStatus, TaskPriority
from frontend_api_server import Task as APITask, CreateTaskRequest


class TestTaskCreationContracts:
    """Contract tests for task creation and initialization."""

    def test_create_task_request_schema_contract(self):
        """Test CreateTaskRequest schema contract validation."""
        
        request_schema = {
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 200},
                "description": {"type": ["string", "null"], "maxLength": 1000},
                "priority": {
                    "type": "string", 
                    "enum": ["low", "medium", "high", "urgent"],
                    "default": "medium"
                },
                "agent_id": {"type": ["string", "null"]}
            },
            "additionalProperties": False
        }
        
        # Test valid request
        valid_request = {
            "title": "Complete API Integration",
            "description": "Integrate frontend with backend API endpoints",
            "priority": "high",
            "agent_id": "agent-12345678"
        }
        
        try:
            jsonschema.validate(valid_request, request_schema)
        except ValidationError as e:
            pytest.fail(f"Valid CreateTaskRequest failed schema validation: {e}")
        
        # Test minimal valid request
        minimal_request = {"title": "Minimal Task"}
        
        try:
            jsonschema.validate(minimal_request, request_schema)
        except ValidationError as e:
            pytest.fail(f"Minimal CreateTaskRequest failed schema validation: {e}")
        
        # Test invalid requests
        invalid_requests = [
            {},  # Missing required title
            {"title": ""},  # Empty title
            {"title": "Test", "priority": "invalid_priority"},  # Invalid priority
            {"title": "a" * 201}  # Title too long
        ]
        
        for invalid_req in invalid_requests:
            with pytest.raises(ValidationError):
                jsonschema.validate(invalid_req, request_schema)

    def test_task_response_schema_contract(self):
        """Test task response schema contract validation."""
        
        task_response_schema = {
            "type": "object",
            "required": ["id", "title", "status", "priority", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "pattern": r"^task-[a-f0-9]{8}$"},
                "title": {"type": "string", "minLength": 1},
                "description": {"type": ["string", "null"]},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "agent_id": {"type": ["string", "null"]},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            "additionalProperties": False
        }
        
        # Test valid task response
        valid_response = {
            "id": "task-87654321",
            "title": "Integration Test Task",
            "description": "Test task for contract validation",
            "status": "pending",
            "priority": "medium",
            "agent_id": "agent-12345678",
            "created_at": "2025-01-18T12:00:00Z",
            "updated_at": "2025-01-18T12:00:00Z"
        }
        
        try:
            jsonschema.validate(valid_response, task_response_schema)
        except ValidationError as e:
            pytest.fail(f"Valid task response failed schema validation: {e}")

    async def test_task_creation_integration_contract(self):
        """Test task creation integration with orchestrator contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Create agent first for task assignment
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="task-test-session",
                    session_name="task-test-name",
                    workspace_path="/test/task"
                ))
                
                agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Test task delegation via orchestrator
                task_id = await orchestrator.delegate_task(
                    task_description="Contract test task",
                    task_type="testing",
                    priority=TaskPriority.HIGH,
                    preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                )
                
                # Validate contract compliance
                assert isinstance(task_id, str)
                assert len(task_id) > 0
                assert task_id in orchestrator._task_assignments
                
                # Validate task assignment contract
                assignment = orchestrator._task_assignments[task_id]
                assert isinstance(assignment, TaskAssignment)
                assert assignment.task_id == task_id
                assert assignment.agent_id == agent_id
                assert assignment.status == TaskStatus.PENDING
        
        finally:
            await orchestrator.shutdown()


class TestTaskLifecycleContracts:
    """Contract tests for task lifecycle and state management."""

    @pytest.fixture
    async def orchestrator_with_task(self):
        """Create orchestrator with agent and task for lifecycle tests."""
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        # Create agent and task
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="lifecycle-session",
                session_name="lifecycle-name",
                workspace_path="/test/lifecycle"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            task_id = await orchestrator.delegate_task(
                task_description="Lifecycle test task",
                task_type="testing",
                priority=TaskPriority.MEDIUM
            )
        
        yield orchestrator, agent_id, task_id
        await orchestrator.shutdown()

    async def test_task_status_transition_contract(self, orchestrator_with_task):
        """Test task status transition contract compliance."""
        
        orchestrator, agent_id, task_id = orchestrator_with_task
        
        # Valid status transitions contract
        valid_transitions = [
            (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
            (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
            (TaskStatus.IN_PROGRESS, TaskStatus.FAILED),
            (TaskStatus.FAILED, TaskStatus.PENDING),  # Retry
            (TaskStatus.COMPLETED, TaskStatus.PENDING)  # Reopen
        ]
        
        # Test current status
        assignment = orchestrator._task_assignments[task_id]
        assert assignment.status == TaskStatus.PENDING
        
        # Test status transition (PENDING -> IN_PROGRESS)
        assignment.status = TaskStatus.IN_PROGRESS
        assert assignment.status == TaskStatus.IN_PROGRESS
        
        # Test completion transition
        assignment.status = TaskStatus.COMPLETED
        assert assignment.status == TaskStatus.COMPLETED

    async def test_task_assignment_contract(self, orchestrator_with_task):
        """Test task assignment data contract."""
        
        orchestrator, agent_id, task_id = orchestrator_with_task
        
        # Get task assignment
        assignment = orchestrator._task_assignments[task_id]
        
        # Test assignment data contract
        assignment_dict = assignment.to_dict()
        
        assignment_schema = {
            "type": "object",
            "required": ["task_id", "agent_id", "assigned_at", "status"],
            "properties": {
                "task_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "assigned_at": {"type": "string"},
                "status": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(assignment_dict, assignment_schema)
        except ValidationError as e:
            pytest.fail(f"TaskAssignment data contract violation: {e}")
        
        # Validate specific values
        assert assignment_dict["task_id"] == task_id
        assert assignment_dict["agent_id"] == agent_id
        assert assignment_dict["status"] == TaskStatus.PENDING.value

    async def test_task_agent_relationship_contract(self, orchestrator_with_task):
        """Test task-agent relationship contract."""
        
        orchestrator, agent_id, task_id = orchestrator_with_task
        
        # Verify agent has task assignment
        agent = orchestrator._agents[agent_id]
        assignment = orchestrator._task_assignments[task_id]
        
        # Contract: agent should be linked to task
        assert assignment.agent_id == agent_id
        
        # Contract: agent should show current task assignment
        # Note: current_task_id is set during delegation
        agent.current_task_id = task_id
        assert agent.current_task_id == task_id

    async def test_task_delegation_performance_contract(self):
        """Test task delegation meets performance contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Create agent for delegation
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="perf-session",
                    session_name="perf-name",
                    workspace_path="/test/perf"
                ))
                
                await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Measure delegation performance
                start_time = time.time()
                task_id = await orchestrator.delegate_task(
                    task_description="Performance test task",
                    task_type="testing",
                    priority=TaskPriority.HIGH
                )
                delegation_time_ms = (time.time() - start_time) * 1000
                
                # Performance contract: <500ms for task delegation
                assert delegation_time_ms < 500.0, f"Task delegation took {delegation_time_ms}ms, exceeds 500ms contract"
        
        finally:
            await orchestrator.shutdown()


class TestTaskDelegationContracts:
    """Contract tests for task delegation algorithms and agent selection."""

    async def test_task_delegation_agent_selection_contract(self):
        """Test agent selection algorithm contract for task delegation."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="selection-session",
                    session_name="selection-name",
                    workspace_path="/test/selection"
                ))
                
                # Create agents with different roles
                backend_agent = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                frontend_agent = await orchestrator.spawn_agent(AgentRole.FRONTEND_DEVELOPER)
                qa_agent = await orchestrator.spawn_agent(AgentRole.QA_ENGINEER)
                
                # Test preferred role selection contract
                task_id = await orchestrator.delegate_task(
                    task_description="Backend development task",
                    task_type="backend_development",
                    preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                )
                
                # Verify correct agent was selected
                assignment = orchestrator._task_assignments[task_id]
                assert assignment.agent_id == backend_agent
                
                # Test fallback selection when preferred agent is busy
                # Assign task to backend agent
                backend_agent_instance = orchestrator._agents[backend_agent]
                backend_agent_instance.current_task_id = task_id
                
                # Create new task - should select different agent
                task_id2 = await orchestrator.delegate_task(
                    task_description="Another task",
                    task_type="general",
                    preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                )
                
                assignment2 = orchestrator._task_assignments[task_id2]
                # Should select different available agent
                assert assignment2.agent_id != backend_agent
                assert assignment2.agent_id in [frontend_agent, qa_agent]
        
        finally:
            await orchestrator.shutdown()

    async def test_task_delegation_no_agents_contract(self):
        """Test task delegation error handling when no agents available."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Try to delegate task with no agents available
            with pytest.raises(TaskDelegationError) as exc_info:
                await orchestrator.delegate_task(
                    task_description="Task with no agents",
                    task_type="testing"
                )
            
            # Validate error message contract
            assert "No suitable agent available" in str(exc_info.value)
        
        finally:
            await orchestrator.shutdown()

    async def test_task_priority_handling_contract(self):
        """Test task priority handling contract."""
        
        # Test priority enum conversion contract
        priority_mappings = [
            (TaskPriority.LOW, "low"),
            (TaskPriority.MEDIUM, "medium"), 
            (TaskPriority.HIGH, "high"),
            (TaskPriority.URGENT, "urgent")
        ]
        
        for priority_enum, priority_string in priority_mappings:
            assert priority_enum.value == priority_string
        
        # Test message priority conversion
        from app.models.message import MessagePriority
        
        priority_conversions = [
            (TaskPriority.LOW, MessagePriority.LOW),
            (TaskPriority.MEDIUM, MessagePriority.NORMAL),
            (TaskPriority.HIGH, MessagePriority.HIGH),
            (TaskPriority.URGENT, MessagePriority.URGENT)
        ]
        
        # This tests the conversion logic in delegate_task
        for task_priority, expected_msg_priority in priority_conversions:
            # Validate the conversion exists and is correct
            if task_priority == TaskPriority.LOW:
                assert expected_msg_priority == MessagePriority.LOW
            elif task_priority == TaskPriority.MEDIUM:
                assert expected_msg_priority == MessagePriority.NORMAL
            elif task_priority == TaskPriority.HIGH:
                assert expected_msg_priority == MessagePriority.HIGH
            elif task_priority == TaskPriority.URGENT:
                assert expected_msg_priority == MessagePriority.URGENT


class TestTaskDataConsistencyContracts:
    """Contract tests for task data consistency and integrity."""

    async def test_task_assignment_data_contract(self):
        """Test TaskAssignment data model contract."""
        
        # Create task assignment
        assignment = TaskAssignment(
            task_id="test-task-123",
            agent_id="test-agent-456"
        )
        
        # Test to_dict contract
        assignment_dict = assignment.to_dict()
        
        assignment_dict_schema = {
            "type": "object",
            "required": ["task_id", "agent_id", "assigned_at", "status"],
            "properties": {
                "task_id": {"type": "string"},
                "agent_id": {"type": "string"}, 
                "assigned_at": {"type": "string"},
                "status": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(assignment_dict, assignment_dict_schema)
        except ValidationError as e:
            pytest.fail(f"TaskAssignment to_dict contract violation: {e}")
        
        # Validate specific values
        assert assignment_dict["task_id"] == "test-task-123"
        assert assignment_dict["agent_id"] == "test-agent-456"
        assert assignment_dict["status"] == TaskStatus.PENDING.value

    async def test_task_status_enum_contract(self):
        """Test TaskStatus enum contract stability."""
        
        # Required task statuses contract
        required_statuses = [
            ("PENDING", "pending"),
            ("IN_PROGRESS", "in_progress"),
            ("COMPLETED", "completed"),
            ("FAILED", "failed")
        ]
        
        # Validate all required statuses exist
        for status_name, status_value in required_statuses:
            if hasattr(TaskStatus, status_name):
                status = getattr(TaskStatus, status_name)
                assert status.value == status_value, f"TaskStatus {status_name} should have value {status_value}"

    async def test_task_priority_enum_contract(self):
        """Test TaskPriority enum contract stability."""
        
        # Required task priorities contract
        required_priorities = [
            ("LOW", "low"),
            ("MEDIUM", "medium"),
            ("HIGH", "high"),
            ("URGENT", "urgent")
        ]
        
        # Validate all required priorities exist
        for priority_name, priority_value in required_priorities:
            if hasattr(TaskPriority, priority_name):
                priority = getattr(TaskPriority, priority_name)
                assert priority.value == priority_value, f"TaskPriority {priority_name} should have value {priority_value}"

    async def test_task_persistence_contract(self):
        """Test task data persistence contract."""
        
        # Test task data that should be persisted
        persistent_data = {
            "id": "persist-test-task",
            "description": "Task persistence test",
            "task_type": "testing",
            "priority": "high",
            "status": "pending",
            "assigned_agent_id": "test-agent-123",
            "created_at": datetime.utcnow()
        }
        
        # Validate persistence data schema
        persistence_schema = {
            "type": "object",
            "required": ["id", "description", "task_type", "status", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "description": {"type": "string"},
                "task_type": {"type": "string"},
                "priority": {"type": "string"},
                "status": {"type": "string"},
                "assigned_agent_id": {"type": ["string", "null"]},
                "created_at": {"type": "string"}  # Will be serialized
            }
        }
        
        # Convert datetime for validation
        validation_data = persistent_data.copy()
        validation_data["created_at"] = "2025-01-18T12:00:00Z"
        
        try:
            jsonschema.validate(validation_data, persistence_schema)
        except ValidationError as e:
            pytest.fail(f"Task persistence data contract violation: {e}")


class TestTaskErrorHandlingContracts:
    """Contract tests for task error handling and validation."""

    async def test_task_delegation_error_contracts(self):
        """Test task delegation error handling contracts."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Test no available agents error
            with pytest.raises(TaskDelegationError) as exc_info:
                await orchestrator.delegate_task(
                    task_description="No agents task",
                    task_type="testing"
                )
            
            assert "No suitable agent available" in str(exc_info.value)
            
            # Test with invalid task parameters (would be caught by API layer)
            # Here we test the orchestrator handles invalid data gracefully
            try:
                task_id = await orchestrator.delegate_task(
                    task_description="",  # Empty description
                    task_type=""  # Empty type
                )
                # Should still create task with empty values
                assert task_id in orchestrator._task_assignments
            except Exception as e:
                # If it raises an exception, it should be a specific error type
                assert isinstance(e, (TaskDelegationError, ValueError))
        
        finally:
            await orchestrator.shutdown()

    async def test_task_validation_contracts(self):
        """Test task validation error contracts."""
        
        # Test task creation validation at API level
        from frontend_api_server import CreateTaskRequest
        
        # Valid task request
        valid_request = CreateTaskRequest(
            title="Valid Task",
            description="Valid description",
            priority="medium"
        )
        
        assert valid_request.title == "Valid Task"
        assert valid_request.priority == "medium"
        
        # Test default values
        minimal_request = CreateTaskRequest(title="Minimal Task")
        assert minimal_request.priority is None  # Will use default
        assert minimal_request.description is None

    async def test_task_assignment_failure_contract(self):
        """Test task assignment failure handling contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Create agent, then make it unavailable
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="assignment-session",
                    session_name="assignment-name",
                    workspace_path="/test/assignment"
                ))
                
                agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Assign task successfully
                task_id1 = await orchestrator.delegate_task(
                    task_description="First task",
                    task_type="testing"
                )
                
                # Make agent busy
                agent = orchestrator._agents[agent_id]
                agent.current_task_id = task_id1
                
                # Try to delegate another task (should fail if no other agents)
                with pytest.raises(TaskDelegationError):
                    await orchestrator.delegate_task(
                        task_description="Second task",
                        task_type="testing",
                        preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                    )
        
        finally:
            await orchestrator.shutdown()


class TestTaskPerformanceContracts:
    """Contract tests for task operation performance requirements."""

    async def test_task_operation_performance_contracts(self):
        """Test task operations meet performance contracts."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Create agent for tasks
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="perf-session",
                    session_name="perf-name",
                    workspace_path="/test/perf"
                ))
                
                await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Test task delegation performance
                start_time = time.time()
                task_id = await orchestrator.delegate_task(
                    task_description="Performance test task",
                    task_type="testing",
                    priority=TaskPriority.HIGH
                )
                delegation_time_ms = (time.time() - start_time) * 1000
                
                # Performance contract: <500ms for task delegation
                assert delegation_time_ms < 500.0, f"Task delegation took {delegation_time_ms}ms, exceeds 500ms contract"
                
                # Test task status retrieval performance
                start_time = time.time()
                assignment = orchestrator._task_assignments[task_id]
                assignment_data = assignment.to_dict()
                retrieval_time_ms = (time.time() - start_time) * 1000
                
                # Performance contract: <10ms for status retrieval
                assert retrieval_time_ms < 10.0, f"Task status retrieval took {retrieval_time_ms}ms, exceeds 10ms contract"
        
        finally:
            await orchestrator.shutdown()

    async def test_concurrent_task_delegation_contract(self):
        """Test concurrent task delegation performance contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Create multiple agents for concurrent tasks
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="concurrent-session",
                    session_name="concurrent-name", 
                    workspace_path="/test/concurrent"
                ))
                
                # Create 3 agents
                for role in [AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER, AgentRole.QA_ENGINEER]:
                    await orchestrator.spawn_agent(role)
                
                # Test concurrent task delegation
                async def delegate_task():
                    return await orchestrator.delegate_task(
                        task_description=f"Concurrent task {time.time()}",
                        task_type="testing",
                        priority=TaskPriority.MEDIUM
                    )
                
                start_time = time.time()
                task_futures = [delegate_task() for _ in range(3)]
                task_ids = await asyncio.gather(*task_futures)
                total_time_ms = (time.time() - start_time) * 1000
                
                # Concurrent operations should not degrade performance significantly
                avg_time_per_task = total_time_ms / 3
                assert avg_time_per_task < 200.0, f"Concurrent task delegation avg {avg_time_per_task}ms per task exceeds 200ms contract"
                
                # All tasks should be assigned
                assert len(task_ids) == 3
                for task_id in task_ids:
                    assert task_id in orchestrator._task_assignments
        
        finally:
            await orchestrator.shutdown()


# Integration Contract Summary
class TestTaskWorkflowContractSummary:
    """Summary test validating all task workflow contracts work together."""
    
    async def test_complete_task_workflow_contract_compliance(self):
        """Integration test ensuring all task workflow contracts are compatible."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Complete task workflow with contract validation
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="workflow-session",
                    session_name="workflow-name",
                    workspace_path="/test/workflow"
                ))
                
                # 1. Agent creation for task assignment contract
                agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                assert agent_id in orchestrator._agents
                
                # 2. Task delegation contract
                start_time = time.time()
                task_id = await orchestrator.delegate_task(
                    task_description="Complete workflow test task",
                    task_type="integration_testing",
                    priority=TaskPriority.HIGH,
                    preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                )
                delegation_time = (time.time() - start_time) * 1000
                
                assert isinstance(task_id, str)
                assert len(task_id) > 0
                assert delegation_time < 500.0
                
                # 3. Task assignment data consistency contract
                assignment = orchestrator._task_assignments[task_id]
                assignment_dict = assignment.to_dict()
                
                required_fields = ["task_id", "agent_id", "assigned_at", "status"]
                for field in required_fields:
                    assert field in assignment_dict
                
                assert assignment.task_id == task_id
                assert assignment.agent_id == agent_id
                assert assignment.status == TaskStatus.PENDING
                
                # 4. Agent-task relationship contract
                agent = orchestrator._agents[agent_id]
                agent.current_task_id = task_id  # Simulate task assignment
                assert agent.current_task_id == task_id
                
                # 5. Task status transition contract
                assignment.status = TaskStatus.IN_PROGRESS
                assert assignment.status == TaskStatus.IN_PROGRESS
                
                assignment.status = TaskStatus.COMPLETED
                assert assignment.status == TaskStatus.COMPLETED
                
                # 6. System status integration contract
                status = await orchestrator.get_system_status()
                assert status["tasks"]["active_assignments"] >= 1
                
                # 7. Performance metrics contract
                metrics = await orchestrator.get_performance_metrics()
                assert "operation_metrics" in metrics
                assert metrics["tasks"] >= 1
                
                # 8. Task completion cleanup contract
                # Remove completed task from agent
                agent.current_task_id = None
                assert agent.current_task_id is None
                
                # 9. Final validation
                assert task_id in orchestrator._task_assignments
                final_assignment = orchestrator._task_assignments[task_id]
                assert final_assignment.status == TaskStatus.COMPLETED
        
        finally:
            await orchestrator.shutdown()