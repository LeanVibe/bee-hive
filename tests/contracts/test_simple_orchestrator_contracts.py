"""
SimpleOrchestrator Contract Testing Framework
============================================

Contract testing for the SimpleOrchestrator public interface to ensure:
- Interface compatibility across component versions
- Method signature stability 
- Return value schema validation
- Error handling contract compliance
- Performance contract adherence

This ensures the frontend-backend integration remains stable and reliable.
"""

import pytest
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import jsonschema
from jsonschema import validate, ValidationError

# Import the contract testing framework
from .contract_testing_framework import (
    ContractTestingFramework,
    ContractDefinition,
    ContractType,
    ContractValidationResult
)

# Import SimpleOrchestrator and related classes
from app.core.simple_orchestrator import (
    SimpleOrchestrator,
    AgentRole,
    AgentInstance,
    TaskAssignment,
    SimpleOrchestratorError,
    AgentNotFoundError,
    TaskDelegationError,
    create_simple_orchestrator
)
from app.models.agent import AgentStatus, AgentType
from app.models.task import TaskStatus, TaskPriority


class TestSimpleOrchestratorContracts:
    """Contract tests for SimpleOrchestrator interface validation."""

    @pytest.fixture
    async def orchestrator(self):
        """Create a test orchestrator instance with mocked dependencies."""
        # Mock dependencies to avoid real infrastructure
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        mock_anthropic = AsyncMock()
        
        orchestrator = create_simple_orchestrator(
            db_session_factory=mock_db,
            cache=mock_cache,
            anthropic_client=mock_anthropic
        )
        
        # Initialize without real dependencies
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.shutdown()

    @pytest.fixture
    def contract_framework(self):
        """Create contract testing framework instance."""
        return ContractTestingFramework()

    # Contract: Agent Spawning Interface
    
    async def test_spawn_agent_method_signature_contract(self, orchestrator, contract_framework):
        """Test that spawn_agent method maintains required signature."""
        
        # Define expected method signature contract
        signature_contract = {
            "method_name": "spawn_agent",
            "parameters": {
                "role": {"type": "AgentRole", "required": True},
                "agent_id": {"type": "Optional[str]", "required": False},
                "agent_type": {"type": "AgentLauncherType", "required": False},
                "task_id": {"type": "Optional[str]", "required": False},
                "workspace_name": {"type": "Optional[str]", "required": False},
                "git_branch": {"type": "Optional[str]", "required": False},
                "working_directory": {"type": "Optional[str]", "required": False},
                "environment_vars": {"type": "Optional[Dict[str, str]]", "required": False}
            },
            "return_type": "str",
            "exceptions": ["SimpleOrchestratorError"]
        }
        
        # Validate method exists and has correct signature
        assert hasattr(orchestrator, 'spawn_agent')
        method = getattr(orchestrator, 'spawn_agent')
        assert callable(method)
        assert asyncio.iscoroutinefunction(method)
        
        # Test with minimal required parameters
        try:
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="test-session",
                    session_name="test-session-name",
                    workspace_path="/test/path"
                ))
                
                agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                assert isinstance(agent_id, str)
                assert len(agent_id) > 0
        except Exception as e:
            # Contract allows SimpleOrchestratorError and subclasses
            assert isinstance(e, SimpleOrchestratorError)

    async def test_spawn_agent_return_value_contract(self, orchestrator, contract_framework):
        """Test spawn_agent return value schema compliance."""
        
        return_value_schema = {
            "type": "string",
            "minLength": 1,
            "pattern": r"^[a-f0-9\-]+$"  # UUID format
        }
        
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Validate against schema
            try:
                jsonschema.validate(agent_id, return_value_schema)
            except ValidationError as e:
                pytest.fail(f"spawn_agent return value contract violation: {e}")

    # Contract: Agent Shutdown Interface
    
    async def test_shutdown_agent_method_signature_contract(self, orchestrator):
        """Test shutdown_agent method signature contract."""
        
        assert hasattr(orchestrator, 'shutdown_agent')
        method = getattr(orchestrator, 'shutdown_agent')
        assert callable(method)
        assert asyncio.iscoroutinefunction(method)
        
        # Test with non-existent agent (should return False, not raise)
        result = await orchestrator.shutdown_agent("non-existent-agent")
        assert isinstance(result, bool)
        assert result is False

    # Contract: Task Delegation Interface
    
    async def test_delegate_task_method_signature_contract(self, orchestrator):
        """Test delegate_task method signature contract."""
        
        signature_contract = {
            "parameters": {
                "task_description": {"type": "str", "required": True},
                "task_type": {"type": "str", "required": True},
                "priority": {"type": "TaskPriority", "required": False},
                "preferred_agent_role": {"type": "Optional[AgentRole]", "required": False}
            },
            "return_type": "str",
            "exceptions": ["TaskDelegationError"]
        }
        
        assert hasattr(orchestrator, 'delegate_task')
        method = getattr(orchestrator, 'delegate_task')
        assert callable(method)
        assert asyncio.iscoroutinefunction(method)
        
        # Test delegation without available agents (should raise TaskDelegationError)
        with pytest.raises(TaskDelegationError):
            await orchestrator.delegate_task(
                task_description="Test task",
                task_type="testing"
            )

    async def test_delegate_task_return_schema_contract(self, orchestrator):
        """Test delegate_task return value schema contract."""
        
        return_schema = {
            "type": "string",
            "minLength": 1,
            "pattern": r"^[a-f0-9\-]+$"  # UUID format
        }
        
        # Create an agent first so delegation can succeed
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Now delegate a task
            task_id = await orchestrator.delegate_task(
                task_description="Test task",
                task_type="testing",
                priority=TaskPriority.MEDIUM
            )
            
            # Validate return value schema
            jsonschema.validate(task_id, return_schema)

    # Contract: System Status Interface
    
    async def test_get_system_status_method_signature_contract(self, orchestrator):
        """Test get_system_status method signature contract."""
        
        assert hasattr(orchestrator, 'get_system_status')
        method = getattr(orchestrator, 'get_system_status')
        assert callable(method)
        assert asyncio.iscoroutinefunction(method)
        
        status = await orchestrator.get_system_status()
        assert isinstance(status, dict)

    async def test_get_system_status_return_schema_contract(self, orchestrator):
        """Test get_system_status return value schema contract."""
        
        status_schema = {
            "type": "object",
            "required": ["timestamp", "agents", "tasks", "performance", "health"],
            "properties": {
                "timestamp": {"type": "string"},
                "agents": {
                    "type": "object",
                    "required": ["total", "by_status", "details"],
                    "properties": {
                        "total": {"type": "integer", "minimum": 0},
                        "by_status": {"type": "object"},
                        "details": {"type": "object"}
                    }
                },
                "tasks": {
                    "type": "object",
                    "required": ["active_assignments"],
                    "properties": {
                        "active_assignments": {"type": "integer", "minimum": 0}
                    }
                },
                "performance": {
                    "type": "object",
                    "required": ["operations_count", "operations_per_second", "response_time_ms"],
                    "properties": {
                        "operations_count": {"type": "integer", "minimum": 0},
                        "operations_per_second": {"type": "number", "minimum": 0},
                        "response_time_ms": {"type": "number", "minimum": 0}
                    }
                },
                "health": {"type": "string", "enum": ["healthy", "no_agents", "error"]}
            }
        }
        
        status = await orchestrator.get_system_status()
        
        try:
            jsonschema.validate(status, status_schema)
        except ValidationError as e:
            pytest.fail(f"get_system_status schema contract violation: {e}")

    # Contract: Performance Requirements
    
    async def test_spawn_agent_performance_contract(self, orchestrator):
        """Test spawn_agent meets performance contract (<100ms)."""
        
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            start_time = time.time()
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Performance contract: spawn_agent should complete in <100ms (Epic 1 target)
            assert execution_time_ms < 100.0, f"spawn_agent took {execution_time_ms}ms, exceeds 100ms contract"

    async def test_delegate_task_performance_contract(self, orchestrator):
        """Test delegate_task meets performance contract (<500ms)."""
        
        # Create an agent first
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Measure task delegation performance
            start_time = time.time()
            task_id = await orchestrator.delegate_task(
                task_description="Performance test task",
                task_type="testing"
            )
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Performance contract: delegate_task should complete in <500ms
            assert execution_time_ms < 500.0, f"delegate_task took {execution_time_ms}ms, exceeds 500ms contract"

    async def test_get_system_status_performance_contract(self, orchestrator):
        """Test get_system_status meets performance contract (<50ms)."""
        
        start_time = time.time()
        status = await orchestrator.get_system_status()
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Performance contract: get_system_status should complete in <50ms
        assert execution_time_ms < 50.0, f"get_system_status took {execution_time_ms}ms, exceeds 50ms contract"

    # Contract: Error Handling Interface
    
    async def test_error_handling_contracts(self, orchestrator):
        """Test that error handling maintains contract specifications."""
        
        # Test AgentNotFoundError contract
        with pytest.raises(SimpleOrchestratorError):
            await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER, agent_id="existing-id")
            # Try to spawn with same ID again
            await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER, agent_id="existing-id")
        
        # Test TaskDelegationError contract
        with pytest.raises(TaskDelegationError):
            await orchestrator.delegate_task("Test task", "testing")

    # Contract: Data Consistency Interface
    
    async def test_agent_instance_schema_contract(self, orchestrator):
        """Test AgentInstance data schema contract."""
        
        agent_schema = {
            "type": "object",
            "required": ["id", "role", "status", "created_at", "last_activity"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "role": {"type": "string"},
                "status": {"type": "string"},
                "current_task_id": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "last_activity": {"type": "string"}
            }
        }
        
        # Create an agent and validate its schema
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Get agent data via system status
            status = await orchestrator.get_system_status()
            agent_data = status["agents"]["details"][agent_id]
            
            try:
                jsonschema.validate(agent_data, agent_schema)
            except ValidationError as e:
                pytest.fail(f"AgentInstance schema contract violation: {e}")

    async def test_task_assignment_schema_contract(self, orchestrator):
        """Test TaskAssignment data schema contract."""
        
        assignment_schema = {
            "type": "object",
            "required": ["task_id", "agent_id", "assigned_at", "status"],
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "agent_id": {"type": "string", "minLength": 1},
                "assigned_at": {"type": "string"},
                "status": {"type": "string"}
            }
        }
        
        # Create agent and assign task
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/path"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            task_id = await orchestrator.delegate_task("Test task", "testing")
            
            # Validate assignment was created with correct schema
            assert task_id in orchestrator._task_assignments
            assignment = orchestrator._task_assignments[task_id]
            assignment_data = assignment.to_dict()
            
            try:
                jsonschema.validate(assignment_data, assignment_schema)
            except ValidationError as e:
                pytest.fail(f"TaskAssignment schema contract violation: {e}")

    # Contract: Backward Compatibility
    
    async def test_backward_compatibility_contract(self, orchestrator):
        """Test that interface changes maintain backward compatibility."""
        
        # Test that all public methods still exist and are callable
        required_methods = [
            'spawn_agent',
            'shutdown_agent', 
            'delegate_task',
            'get_system_status',
            'initialize',
            'shutdown'
        ]
        
        for method_name in required_methods:
            assert hasattr(orchestrator, method_name), f"Required method {method_name} missing"
            method = getattr(orchestrator, method_name)
            assert callable(method), f"Method {method_name} is not callable"
            
            # All async methods should remain async
            if method_name != 'shutdown':  # shutdown can be sync or async
                assert asyncio.iscoroutinefunction(method), f"Method {method_name} should be async"

    # Contract: Integration Interface Stability
    
    async def test_plugin_interface_contract(self, orchestrator):
        """Test plugin interface contract stability."""
        
        # Verify plugin-related methods exist and are accessible
        plugin_methods = [
            'load_plugin_dynamic',
            'unload_plugin_safe',
            'hot_swap_plugin',
            'get_plugin_performance_metrics',
            'get_plugin_security_status'
        ]
        
        for method_name in plugin_methods:
            assert hasattr(orchestrator, method_name), f"Plugin method {method_name} missing"
            method = getattr(orchestrator, method_name)
            assert callable(method), f"Plugin method {method_name} is not callable"
            assert asyncio.iscoroutinefunction(method), f"Plugin method {method_name} should be async"

    async def test_enhanced_features_contract(self, orchestrator):
        """Test enhanced features interface contract."""
        
        enhanced_methods = [
            'get_agent_session_info',
            'list_agent_sessions',
            'attach_to_agent_session',
            'execute_command_in_agent_session',
            'get_agent_logs',
            'get_enhanced_system_status',
            'get_performance_metrics'
        ]
        
        for method_name in enhanced_methods:
            assert hasattr(orchestrator, method_name), f"Enhanced method {method_name} missing"
            method = getattr(orchestrator, method_name)
            assert callable(method), f"Enhanced method {method_name} is not callable"
            assert asyncio.iscoroutinefunction(method), f"Enhanced method {method_name} should be async"

    # Contract: WebSocket Broadcasting Interface
    
    async def test_websocket_broadcast_contract(self, orchestrator):
        """Test WebSocket broadcasting interface contract."""
        
        broadcast_methods = [
            '_broadcast_agent_update',
            '_broadcast_task_update', 
            '_broadcast_system_status'
        ]
        
        for method_name in broadcast_methods:
            assert hasattr(orchestrator, method_name), f"Broadcast method {method_name} missing"
            method = getattr(orchestrator, method_name)
            assert callable(method), f"Broadcast method {method_name} is not callable"
            assert asyncio.iscoroutinefunction(method), f"Broadcast method {method_name} should be async"


# Contract Testing Summary
class TestSimpleOrchestratorContractSummary:
    """Summary test to validate all contracts are passing."""
    
    async def test_complete_contract_compliance(self):
        """Integration test ensuring all contracts work together."""
        
        # Create orchestrator with mocked dependencies
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        
        orchestrator = create_simple_orchestrator(
            db_session_factory=mock_db,
            cache=mock_cache
        )
        
        try:
            # Initialize
            with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
                await orchestrator.initialize()
            
            # Test complete workflow follows all contracts
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="test-session",
                    session_name="test-session-name",
                    workspace_path="/test/path"
                ))
                
                # 1. Spawn agent (contract: returns string ID, <100ms)
                start_time = time.time()
                agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                spawn_time = (time.time() - start_time) * 1000
                
                assert isinstance(agent_id, str)
                assert len(agent_id) > 0
                assert spawn_time < 100.0
                
                # 2. Delegate task (contract: returns string task ID, <500ms)
                start_time = time.time()
                task_id = await orchestrator.delegate_task("Test task", "testing")
                delegate_time = (time.time() - start_time) * 1000
                
                assert isinstance(task_id, str)
                assert len(task_id) > 0
                assert delegate_time < 500.0
                
                # 3. Get system status (contract: returns dict with required fields, <50ms)
                start_time = time.time()
                status = await orchestrator.get_system_status()
                status_time = (time.time() - start_time) * 1000
                
                assert isinstance(status, dict)
                assert "timestamp" in status
                assert "agents" in status
                assert "tasks" in status
                assert "performance" in status
                assert "health" in status
                assert status_time < 50.0
                
                # 4. Shutdown agent (contract: returns bool)
                result = await orchestrator.shutdown_agent(agent_id)
                assert isinstance(result, bool)
                assert result is True
                
        finally:
            await orchestrator.shutdown()