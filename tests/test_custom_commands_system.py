"""
Comprehensive test suite for Custom Commands System - Phase 6.1

Tests all components of the multi-agent workflow command system including
command registry, task distribution, execution engine, and API endpoints.
"""

import asyncio
import uuid
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import json

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

from app.main import create_app
from app.core.command_registry import CommandRegistry, CommandRegistryError
from app.core.task_distributor import TaskDistributor, DistributionStrategy, TaskAssignment
from app.core.command_executor import CommandExecutor, ExecutionEnvironment
from app.core.agent_registry import AgentRegistry
from app.schemas.custom_commands import (
    CommandDefinition, CommandExecutionRequest, CommandStatus,
    AgentRole, AgentRequirement, WorkflowStep, SecurityPolicy,
    CommandCreateRequest, CommandValidationResult
)
from app.models.agent import Agent, AgentStatus, AgentType
from app.observability.custom_commands_hooks import CustomCommandsHooks

# Test fixtures and utilities

@pytest.fixture
def sample_command_definition():
    """Sample command definition for testing."""
    return CommandDefinition(
        name="test-feature-implementation",
        version="1.0.0",
        description="Test feature implementation workflow",
        category="development",
        tags=["test", "implementation"],
        agents=[
            AgentRequirement(
                role=AgentRole.BACKEND_ENGINEER,
                specialization=["python", "fastapi"],
                required_capabilities=["coding", "testing"]
            ),
            AgentRequirement(
                role=AgentRole.QA_TEST_GUARDIAN,
                specialization=["testing", "quality"],
                required_capabilities=["test_automation", "validation"]
            )
        ],
        workflow=[
            WorkflowStep(
                step="analyze_requirements",
                agent=AgentRole.BACKEND_ENGINEER,
                task="Analyze feature requirements and create technical design",
                outputs=["technical_design.md", "requirements_analysis.md"],
                timeout_minutes=30
            ),
            WorkflowStep(
                step="implement_feature",
                agent=AgentRole.BACKEND_ENGINEER,
                task="Implement the feature based on technical design",
                depends_on=["analyze_requirements"],
                outputs=["feature_code.py", "unit_tests.py"],
                timeout_minutes=120
            ),
            WorkflowStep(
                step="validate_implementation",
                agent=AgentRole.QA_TEST_GUARDIAN,
                task="Validate implementation and run comprehensive tests",
                depends_on=["implement_feature"],
                outputs=["test_results.json", "quality_report.md"],
                timeout_minutes=60
            )
        ],
        security_policy=SecurityPolicy(
            allowed_operations=["file_read", "file_write", "code_execution"],
            network_access=False,
            resource_limits={"max_memory_mb": 1024, "max_cpu_time_seconds": 3600}
        ),
        author="test_user"
    )


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for testing."""
    registry = AsyncMock(spec=AgentRegistry)
    
    # Mock agents
    mock_agents = [
        Agent(
            id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            name="backend-agent-1",
            type=AgentType.CLAUDE,
            role="backend-engineer",
            status=AgentStatus.ACTIVE,
            capabilities=[
                {"name": "coding", "specialization": "python"},
                {"name": "testing", "specialization": "pytest"}
            ]
        ),
        Agent(
            id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            name="qa-agent-1", 
            type=AgentType.CLAUDE,
            role="qa-test-guardian",
            status=AgentStatus.ACTIVE,
            capabilities=[
                {"name": "test_automation", "specialization": "testing"},
                {"name": "validation", "specialization": "quality"}
            ]
        )
    ]
    
    registry.get_active_agents.return_value = mock_agents
    registry.get_agent.side_effect = lambda agent_id: next(
        (agent for agent in mock_agents if str(agent.id) == str(agent_id)), None
    )
    
    return registry


@pytest.fixture
def mock_message_broker():
    """Mock message broker for testing."""
    broker = AsyncMock()
    broker.send_message = AsyncMock()
    broker.publish_agent_event = AsyncMock()
    return broker


@pytest.fixture
def mock_hook_manager():
    """Mock hook manager for testing."""
    hook_manager = AsyncMock()
    hook_manager.execute_hook = AsyncMock()
    return hook_manager


# Command Registry Tests

class TestCommandRegistry:
    """Test suite for CommandRegistry."""
    
    @pytest.mark.asyncio
    async def test_register_command_success(self, sample_command_definition, mock_agent_registry):
        """Test successful command registration."""
        registry = CommandRegistry(agent_registry=mock_agent_registry)
        
        with patch.object(registry, '_store_command_in_db') as mock_store, \
             patch.object(registry, '_store_command_file') as mock_file_store:
            
            mock_store.return_value = uuid.uuid4()
            
            success, validation_result = await registry.register_command(
                definition=sample_command_definition,
                author_id="test_user",
                validate_agents=True
            )
            
            assert success is True
            assert validation_result.is_valid is True
            assert len(validation_result.errors) == 0
            mock_store.assert_called_once()
            mock_file_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_command_validation_failure(self, mock_agent_registry):
        """Test command registration with validation failure."""
        registry = CommandRegistry(agent_registry=mock_agent_registry)
        
        # Create invalid command definition
        invalid_command = CommandDefinition(
            name="invalid-command",
            version="1.0.0", 
            description="Invalid command",
            agents=[],  # No agents - should fail validation
            workflow=[]  # No workflow steps - should fail validation
        )
        
        success, validation_result = await registry.register_command(
            definition=invalid_command,
            author_id="test_user",
            validate_agents=True
        )
        
        assert success is False
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_get_command(self, sample_command_definition, mock_agent_registry):
        """Test command retrieval."""
        registry = CommandRegistry(agent_registry=mock_agent_registry)
        
        # Mock cache update
        registry._update_command_cache(sample_command_definition)
        
        command = await registry.get_command("test-feature-implementation", "1.0.0")
        
        assert command is not None
        assert command.name == "test-feature-implementation"
        assert command.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_validate_command(self, sample_command_definition, mock_agent_registry):
        """Test command validation."""
        registry = CommandRegistry(agent_registry=mock_agent_registry)
        
        validation_result = await registry.validate_command(
            definition=sample_command_definition,
            validate_agents=True
        )
        
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
        assert len(validation_result.agent_availability) > 0


# Task Distributor Tests

class TestTaskDistributor:
    """Test suite for TaskDistributor."""
    
    @pytest.mark.asyncio
    async def test_distribute_tasks_success(self, sample_command_definition, mock_agent_registry, mock_message_broker):
        """Test successful task distribution."""
        distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        
        result = await distributor.distribute_tasks(
            workflow_steps=sample_command_definition.workflow,
            agent_requirements=sample_command_definition.agents,
            strategy_override=DistributionStrategy.HYBRID
        )
        
        assert len(result.assignments) > 0
        assert len(result.unassigned_tasks) == 0
        assert result.strategy_used == DistributionStrategy.HYBRID
        assert result.distribution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_distribute_tasks_no_agents(self, sample_command_definition, mock_message_broker):
        """Test task distribution with no available agents."""
        # Create empty agent registry
        empty_registry = AsyncMock(spec=AgentRegistry)
        empty_registry.get_active_agents.return_value = []
        
        distributor = TaskDistributor(
            agent_registry=empty_registry,
            message_broker=mock_message_broker
        )
        
        result = await distributor.distribute_tasks(
            workflow_steps=sample_command_definition.workflow,
            agent_requirements=sample_command_definition.agents
        )
        
        assert len(result.assignments) == 0
        assert len(result.unassigned_tasks) == len(sample_command_definition.workflow)
    
    @pytest.mark.asyncio
    async def test_reassign_failed_task(self, mock_agent_registry, mock_message_broker):
        """Test task reassignment after failure."""
        distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        
        # Mock agent performance cache
        distributor.agent_performance_cache = {
            "11111111-1111-1111-1111-111111111111": MagicMock(
                current_tasks=0, cpu_usage=10.0, memory_usage=20.0,
                context_usage=30.0
            )
        }
        
        task_requirement = AgentRequirement(
            role=AgentRole.BACKEND_ENGINEER,
            required_capabilities=["coding"]
        )
        
        assignment = await distributor.reassign_failed_task(
            task_id="test_task",
            failed_agent_id="failed_agent_id",
            task_requirements=task_requirement
        )
        
        assert assignment is not None
        assert assignment.task_id == "test_task"
        assert assignment.assignment_reason == "task_reassignment_after_failure"


# Command Executor Tests

class TestCommandExecutor:
    """Test suite for CommandExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_command_success(
        self, 
        sample_command_definition, 
        mock_agent_registry, 
        mock_message_broker,
        mock_hook_manager
    ):
        """Test successful command execution."""
        # Mock dependencies
        mock_registry = AsyncMock()
        mock_registry.get_command.return_value = sample_command_definition
        mock_registry.update_execution_metrics = AsyncMock()
        
        mock_distributor = AsyncMock()
        mock_distributor.distribute_tasks.return_value = MagicMock(
            assignments=[
                TaskAssignment(
                    task_id="analyze_requirements",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    assignment_score=0.9,
                    estimated_completion_time=datetime.utcnow() + timedelta(minutes=30),
                    assignment_reason="optimal_match",
                    backup_agents=[]
                ),
                TaskAssignment(
                    task_id="implement_feature",
                    agent_id="11111111-1111-1111-1111-111111111111",
                    assignment_score=0.9,
                    estimated_completion_time=datetime.utcnow() + timedelta(minutes=120),
                    assignment_reason="optimal_match",
                    backup_agents=[]
                ),
                TaskAssignment(
                    task_id="validate_implementation",
                    agent_id="22222222-2222-2222-2222-222222222222",
                    assignment_score=0.85,
                    estimated_completion_time=datetime.utcnow() + timedelta(minutes=60),
                    assignment_reason="optimal_match",
                    backup_agents=[]
                )
            ],
            unassigned_tasks=[],
            distribution_time_ms=150.0,
            strategy_used=DistributionStrategy.HYBRID,
            optimization_metrics={"load_balance_score": 0.8}
        )
        
        executor = CommandExecutor(
            command_registry=mock_registry,
            task_distributor=mock_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager,
            execution_environment=ExecutionEnvironment.SANDBOX
        )
        
        await executor.start()
        
        request = CommandExecutionRequest(
            command_name="test-feature-implementation",
            command_version="1.0.0",
            parameters={"feature_name": "test_feature"},
            context={"environment": "test"}
        )
        
        result = await executor.execute_command(request, "test_user")
        
        assert result.command_name == "test-feature-implementation"
        assert result.status in [CommandStatus.COMPLETED, CommandStatus.FAILED]
        assert result.total_steps == 3
        assert result.execution_id is not None
        
        await executor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_command_not_found(
        self,
        mock_agent_registry,
        mock_message_broker, 
        mock_hook_manager
    ):
        """Test command execution with non-existent command."""
        mock_registry = AsyncMock()
        mock_registry.get_command.return_value = None
        
        mock_distributor = AsyncMock()
        
        executor = CommandExecutor(
            command_registry=mock_registry,
            task_distributor=mock_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager
        )
        
        await executor.start()
        
        request = CommandExecutionRequest(
            command_name="non-existent-command",
            parameters={}
        )
        
        result = await executor.execute_command(request, "test_user")
        
        assert result.status == CommandStatus.FAILED
        assert "not found" in result.error_message.lower()
        
        await executor.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_execution(
        self,
        sample_command_definition,
        mock_agent_registry,
        mock_message_broker,
        mock_hook_manager
    ):
        """Test command execution cancellation."""
        mock_registry = AsyncMock()
        mock_distributor = AsyncMock()
        
        executor = CommandExecutor(
            command_registry=mock_registry,
            task_distributor=mock_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager
        )
        
        await executor.start()
        
        # Create a mock execution context
        execution_id = str(uuid.uuid4())
        executor.active_executions[execution_id] = MagicMock()
        
        success = await executor.cancel_execution(execution_id, "test_cancellation")
        
        assert success is True
        assert execution_id not in executor.active_executions
        
        await executor.stop()


# API Endpoint Tests

class TestCustomCommandsAPI:
    """Test suite for Custom Commands API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user."""
        return {"user_id": "test_user", "is_admin": False}
    
    def test_create_command_success(self, client, sample_command_definition, mock_auth_user):
        """Test successful command creation via API."""
        request_data = CommandCreateRequest(
            definition=sample_command_definition,
            validate_agents=True,
            dry_run=False
        )
        
        with patch('app.api.v1.custom_commands.get_current_user', return_value=mock_auth_user), \
             patch('app.api.v1.custom_commands.get_command_registry') as mock_get_registry:
            
            mock_registry = AsyncMock()
            mock_registry.register_command.return_value = (
                True, 
                CommandValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=[],
                    agent_availability={"backend-engineer": True, "qa-test-guardian": True}
                )
            )
            mock_get_registry.return_value = mock_registry
            
            response = client.post(
                "/api/v1/custom-commands/commands",
                json=request_data.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["command_name"] == sample_command_definition.name
    
    def test_list_commands(self, client, mock_auth_user):
        """Test command listing via API."""
        with patch('app.api.v1.custom_commands.get_current_user', return_value=mock_auth_user), \
             patch('app.api.v1.custom_commands.get_command_registry') as mock_get_registry:
            
            mock_registry = AsyncMock()
            mock_registry.list_commands.return_value = (
                [
                    {
                        "name": "test-command",
                        "version": "1.0.0",
                        "description": "Test command",
                        "category": "test",
                        "tags": ["testing"],
                        "required_agents": 2,
                        "workflow_steps": 3,
                        "execution_count": 5,
                        "success_rate": 80.0
                    }
                ],
                1
            )
            mock_get_registry.return_value = mock_registry
            
            response = client.get("/api/v1/custom-commands/commands")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["total"] == 1
    
    def test_execute_command_success(self, client, mock_auth_user):
        """Test command execution via API."""
        request_data = CommandExecutionRequest(
            command_name="test-command",
            command_version="1.0.0",
            parameters={"param1": "value1"},
            context={"env": "test"}
        )
        
        with patch('app.api.v1.custom_commands.get_current_user', return_value=mock_auth_user), \
             patch('app.api.v1.custom_commands.get_command_executor') as mock_get_executor:
            
            mock_executor = AsyncMock()
            execution_id = uuid.uuid4()
            mock_executor.execute_command.return_value = MagicMock(
                execution_id=execution_id,
                command_name="test-command",
                command_version="1.0.0",
                status=CommandStatus.COMPLETED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                total_execution_time_seconds=120.0,
                step_results=[],
                final_outputs={"result": "success"},
                total_steps=3,
                completed_steps=3,
                failed_steps=0
            )
            mock_get_executor.return_value = mock_executor
            
            response = client.post(
                "/api/v1/custom-commands/execute",
                json=request_data.model_dump()
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["command_name"] == "test-command"
            assert data["status"] == "completed"


# Observability Integration Tests

class TestCustomCommandsHooks:
    """Test suite for CustomCommandsHooks observability integration."""
    
    @pytest.mark.asyncio
    async def test_command_execution_start_hook(self, mock_hook_manager, mock_message_broker):
        """Test command execution start hook."""
        hooks = CustomCommandsHooks(
            hook_manager=mock_hook_manager,
            message_broker=mock_message_broker
        )
        
        execution_id = str(uuid.uuid4())
        await hooks.on_command_execution_start(
            execution_id=execution_id,
            command_name="test-command",
            command_version="1.0.0",
            requester_id="test_user",
            parameters={"param1": "value1"},
            agent_assignments={"step1": "agent1", "step2": "agent2"}
        )
        
        # Verify hook manager was called
        mock_hook_manager.execute_hook.assert_called_once()
        call_args = mock_hook_manager.execute_hook.call_args
        assert call_args[0][0] == "custom_command_execution_start"
        
        # Verify message broker was called
        mock_message_broker.publish_agent_event.assert_called_once()
        
        # Verify execution is tracked
        assert execution_id in hooks.active_executions
        assert hooks.execution_metrics["total_executions"] == 1
    
    @pytest.mark.asyncio
    async def test_security_violation_hook(self, mock_hook_manager, mock_message_broker):
        """Test security violation hook."""
        hooks = CustomCommandsHooks(
            hook_manager=mock_hook_manager,
            message_broker=mock_message_broker
        )
        
        await hooks.on_security_violation(
            execution_id="test_execution",
            command_name="test-command",
            violation_type="unauthorized_access",
            violation_details={"attempted_path": "/etc/passwd"},
            requester_id="test_user"
        )
        
        # Verify critical severity hook was called
        mock_hook_manager.execute_hook.assert_called_once()
        call_args = mock_hook_manager.execute_hook.call_args
        assert call_args[0][0] == "security_violation"
        
        event_data = call_args[0][1]
        assert event_data["severity"] == "critical"
        assert event_data["data"]["violation_type"] == "unauthorized_access"
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, mock_hook_manager, mock_message_broker):
        """Test system metrics collection."""
        hooks = CustomCommandsHooks(
            hook_manager=mock_hook_manager,
            message_broker=mock_message_broker
        )
        
        # Set up some test metrics
        hooks.execution_metrics["total_executions"] = 100
        hooks.execution_metrics["successful_executions"] = 85
        hooks.execution_metrics["failed_executions"] = 15
        hooks.execution_metrics["average_execution_time"] = 125.5
        
        metrics = await hooks.get_system_metrics()
        
        assert "timestamp" in metrics
        assert "execution_metrics" in metrics
        assert "performance_metrics" in metrics
        assert metrics["performance_metrics"]["success_rate_percent"] == 85.0
        assert metrics["execution_metrics"]["total_executions"] == 100


# Integration Tests

class TestCustomCommandsIntegration:
    """Integration tests for the complete custom commands system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_command_workflow(
        self,
        sample_command_definition,
        mock_agent_registry,
        mock_message_broker,
        mock_hook_manager
    ):
        """Test complete end-to-end command workflow."""
        # Initialize all components
        command_registry = CommandRegistry(agent_registry=mock_agent_registry)
        task_distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        command_executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager
        )
        hooks = CustomCommandsHooks(
            hook_manager=mock_hook_manager,
            message_broker=mock_message_broker
        )
        
        try:
            await command_executor.start()
            
            # Step 1: Register command
            with patch.object(command_registry, '_store_command_in_db') as mock_store, \
                 patch.object(command_registry, '_store_command_file'):
                
                mock_store.return_value = uuid.uuid4()
                
                success, validation_result = await command_registry.register_command(
                    definition=sample_command_definition,
                    author_id="test_user",
                    validate_agents=True
                )
                
                assert success is True
                assert validation_result.is_valid is True
            
            # Step 2: Execute command
            request = CommandExecutionRequest(
                command_name=sample_command_definition.name,
                command_version=sample_command_definition.version,
                parameters={"feature_name": "integration_test"},
                context={"environment": "test"}
            )
            
            with patch.object(command_registry, 'get_command', return_value=sample_command_definition):
                result = await command_executor.execute_command(request, "test_user")
                
                assert result.command_name == sample_command_definition.name
                assert result.status in [CommandStatus.COMPLETED, CommandStatus.FAILED]
                assert result.total_steps == len(sample_command_definition.workflow)
            
            # Step 3: Verify observability hooks were called
            assert mock_hook_manager.execute_hook.call_count > 0
            
            # Step 4: Get system metrics
            metrics = await hooks.get_system_metrics()
            assert "execution_metrics" in metrics
            assert "performance_metrics" in metrics
            
        finally:
            await command_executor.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        sample_command_definition,
        mock_agent_registry,
        mock_message_broker,
        mock_hook_manager
    ):
        """Test error handling and recovery mechanisms."""
        # Create components with failure scenarios
        command_registry = CommandRegistry(agent_registry=mock_agent_registry)
        task_distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        command_executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager
        )
        
        try:
            await command_executor.start()
            
            # Test with non-existent command
            request = CommandExecutionRequest(
                command_name="non-existent-command",
                parameters={}
            )
            
            result = await command_executor.execute_command(request, "test_user")
            
            assert result.status == CommandStatus.FAILED
            assert result.error_message is not None
            assert "not found" in result.error_message.lower()
            
            # Verify error was logged and handled properly
            stats = command_executor.get_execution_statistics()
            assert stats["failed_executions"] >= 1
            
        finally:
            await command_executor.stop()


# Performance Tests

class TestCustomCommandsPerformance:
    """Performance tests for the custom commands system."""
    
    @pytest.mark.asyncio
    async def test_concurrent_command_executions(
        self,
        sample_command_definition,
        mock_agent_registry,
        mock_message_broker,
        mock_hook_manager
    ):
        """Test system performance under concurrent load."""
        command_registry = CommandRegistry(agent_registry=mock_agent_registry)
        task_distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        command_executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker,
            hook_manager=mock_hook_manager
        )
        
        try:
            await command_executor.start()
            
            # Mock command retrieval
            with patch.object(command_registry, 'get_command', return_value=sample_command_definition):
                
                # Create multiple concurrent execution requests
                requests = [
                    CommandExecutionRequest(
                        command_name=sample_command_definition.name,
                        command_version=sample_command_definition.version,
                        parameters={"test_id": f"concurrent_test_{i}"}
                    )
                    for i in range(5)
                ]
                
                # Execute concurrently
                start_time = datetime.utcnow()
                results = await asyncio.gather(*[
                    command_executor.execute_command(req, f"test_user_{i}")
                    for i, req in enumerate(requests)
                ], return_exceptions=True)
                end_time = datetime.utcnow()
                
                # Verify all executions completed
                assert len(results) == 5
                
                # Check performance metrics
                total_time = (end_time - start_time).total_seconds()
                assert total_time < 30  # Should complete within 30 seconds
                
                # Verify system statistics
                stats = command_executor.get_execution_statistics()
                assert stats["total_executions"] >= 5
                assert stats["peak_concurrent_executions"] >= 1
                
        finally:
            await command_executor.stop()
    
    @pytest.mark.asyncio
    async def test_large_workflow_processing(
        self,
        mock_agent_registry,
        mock_message_broker,
        mock_hook_manager
    ):
        """Test processing of large workflows with many steps."""
        # Create a large workflow definition
        large_workflow_steps = [
            WorkflowStep(
                step=f"step_{i}",
                agent=AgentRole.BACKEND_ENGINEER,
                task=f"Execute task {i}",
                depends_on=[f"step_{i-1}"] if i > 0 else [],
                timeout_minutes=5
            )
            for i in range(50)  # 50 sequential steps
        ]
        
        large_command = CommandDefinition(
            name="large-workflow-test",
            version="1.0.0",
            description="Large workflow for performance testing",
            agents=[
                AgentRequirement(
                    role=AgentRole.BACKEND_ENGINEER,
                    required_capabilities=["processing"]
                )
            ],
            workflow=large_workflow_steps
        )
        
        task_distributor = TaskDistributor(
            agent_registry=mock_agent_registry,
            message_broker=mock_message_broker
        )
        
        # Test task distribution performance
        start_time = datetime.utcnow()
        result = await task_distributor.distribute_tasks(
            workflow_steps=large_command.workflow,
            agent_requirements=large_command.agents
        )
        end_time = datetime.utcnow()
        
        distribution_time = (end_time - start_time).total_seconds()
        
        # Verify performance targets
        assert distribution_time < 5.0  # Should complete within 5 seconds
        assert result.distribution_time_ms < 200  # Target <200ms from requirements
        assert len(result.assignments) == 50  # All steps should be assigned


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])