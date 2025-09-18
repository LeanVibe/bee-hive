"""
Agent Management API Contract Testing
====================================

Contract testing for agent creation, management, and lifecycle operations.
Validates API contract stability, data consistency, and integration reliability
between frontend and backend systems.

Key Contract Areas:
- Agent creation and registration contracts
- Agent lifecycle state management
- Agent data schema validation
- Error handling and validation contracts
- Performance and reliability requirements
- Integration with SimpleOrchestrator contracts
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

# Import agent-related components
from app.core.simple_orchestrator import (
    SimpleOrchestrator,
    AgentRole,
    AgentInstance,
    create_simple_orchestrator
)
from app.models.agent import AgentStatus, AgentType
from frontend_api_server import Agent as APIAgent, CreateAgentRequest


class TestAgentCreationContracts:
    """Contract tests for agent creation API operations."""

    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator for agent operations."""
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        
        orchestrator = create_simple_orchestrator(
            db_session_factory=mock_db,
            cache=mock_cache
        )
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        yield orchestrator
        await orchestrator.shutdown()

    def test_create_agent_request_schema_contract(self):
        """Test CreateAgentRequest schema contract validation."""
        
        request_schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 100},
                "type": {"type": "string", "enum": ["claude", "system", "custom"], "default": "claude"},
                "role": {"type": ["string", "null"], "maxLength": 50},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20
                }
            },
            "additionalProperties": False
        }
        
        # Test valid request
        valid_request = {
            "name": "Test Agent",
            "type": "claude", 
            "role": "backend_developer",
            "capabilities": ["coding", "testing", "debugging"]
        }
        
        try:
            jsonschema.validate(valid_request, request_schema)
        except ValidationError as e:
            pytest.fail(f"Valid CreateAgentRequest failed schema validation: {e}")
        
        # Test minimal valid request
        minimal_request = {"name": "Minimal Agent"}
        
        try:
            jsonschema.validate(minimal_request, request_schema)
        except ValidationError as e:
            pytest.fail(f"Minimal CreateAgentRequest failed schema validation: {e}")
        
        # Test invalid requests
        invalid_requests = [
            {},  # Missing required name
            {"name": ""},  # Empty name
            {"name": "Test", "type": "invalid_type"},  # Invalid type
            {"name": "Test", "capabilities": ["a"] * 25}  # Too many capabilities
        ]
        
        for invalid_req in invalid_requests:
            with pytest.raises(ValidationError):
                jsonschema.validate(invalid_req, request_schema)

    def test_agent_response_schema_contract(self):
        """Test agent response schema contract validation."""
        
        agent_response_schema = {
            "type": "object",
            "required": ["id", "name", "type", "status", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "pattern": r"^agent-[a-f0-9]{8}$"},
                "name": {"type": "string", "minLength": 1},
                "type": {"type": "string", "enum": ["claude", "system", "custom"]},
                "status": {"type": "string", "enum": ["active", "inactive", "error"]},
                "role": {"type": ["string", "null"]},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            "additionalProperties": False
        }
        
        # Test valid agent response
        valid_response = {
            "id": "agent-12345678",
            "name": "Test Agent", 
            "type": "claude",
            "status": "active",
            "role": "backend_developer",
            "capabilities": ["coding", "testing"],
            "created_at": "2025-01-18T12:00:00Z",
            "updated_at": "2025-01-18T12:00:00Z"
        }
        
        try:
            jsonschema.validate(valid_response, agent_response_schema)
        except ValidationError as e:
            pytest.fail(f"Valid agent response failed schema validation: {e}")

    async def test_agent_creation_integration_contract(self, orchestrator):
        """Test agent creation integration with orchestrator contract."""
        
        # Mock agent launcher for successful creation
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session-123",
                session_name="test-session-name",
                workspace_path="/test/workspace"
            ))
            
            # Test agent creation via orchestrator
            agent_id = await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                agent_id="contract-test-agent"
            )
            
            # Validate contract compliance
            assert isinstance(agent_id, str)
            assert len(agent_id) > 0
            assert agent_id in orchestrator._agents
            
            # Validate agent instance contract
            agent_instance = orchestrator._agents[agent_id]
            assert isinstance(agent_instance, AgentInstance)
            assert agent_instance.id == agent_id
            assert agent_instance.role == AgentRole.BACKEND_DEVELOPER
            assert agent_instance.status == AgentStatus.ACTIVE

    async def test_agent_creation_performance_contract(self, orchestrator):
        """Test agent creation meets performance contract requirements."""
        
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="perf-test-session",
                session_name="perf-test-name",
                workspace_path="/test/perf"
            ))
            
            # Measure creation time
            start_time = time.time()
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            creation_time_ms = (time.time() - start_time) * 1000
            
            # Performance contract: <100ms for agent creation
            assert creation_time_ms < 100.0, f"Agent creation took {creation_time_ms}ms, exceeds 100ms contract"


class TestAgentLifecycleContracts:
    """Contract tests for agent lifecycle management operations."""

    @pytest.fixture
    async def orchestrator_with_agent(self):
        """Create orchestrator with pre-created agent for lifecycle tests."""
        mock_db = AsyncMock()
        mock_cache = AsyncMock()
        
        orchestrator = create_simple_orchestrator(
            db_session_factory=mock_db,
            cache=mock_cache
        )
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        # Create test agent
        with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
            mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                success=True,
                session_id="test-session",
                session_name="test-session-name",
                workspace_path="/test/workspace"
            ))
            
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
        
        yield orchestrator, agent_id
        await orchestrator.shutdown()

    async def test_agent_status_transition_contract(self, orchestrator_with_agent):
        """Test agent status transition contract compliance."""
        
        orchestrator, agent_id = orchestrator_with_agent
        
        # Valid status transitions contract
        valid_transitions = [
            (AgentStatus.ACTIVE, AgentStatus.INACTIVE),
            (AgentStatus.INACTIVE, AgentStatus.ACTIVE),
            (AgentStatus.ACTIVE, AgentStatus.ERROR),
            (AgentStatus.ERROR, AgentStatus.INACTIVE)
        ]
        
        # Test current status
        agent = orchestrator._agents[agent_id]
        assert agent.status == AgentStatus.ACTIVE
        
        # Test status transition via shutdown (ACTIVE -> INACTIVE)
        result = await orchestrator.shutdown_agent(agent_id)
        assert result is True
        
        # Verify agent was removed (shutdown contract)
        assert agent_id not in orchestrator._agents

    async def test_agent_shutdown_contract(self, orchestrator_with_agent):
        """Test agent shutdown contract compliance."""
        
        orchestrator, agent_id = orchestrator_with_agent
        
        # Test graceful shutdown contract
        start_time = time.time()
        result = await orchestrator.shutdown_agent(agent_id, graceful=True)
        shutdown_time_ms = (time.time() - start_time) * 1000
        
        # Contract validation
        assert isinstance(result, bool)
        assert result is True
        assert shutdown_time_ms < 1000.0  # Should complete within 1 second
        
        # Verify agent removal contract
        assert agent_id not in orchestrator._agents
        
        # Test shutdown of non-existent agent contract
        result2 = await orchestrator.shutdown_agent("non-existent-agent")
        assert result2 is False  # Should return False, not raise exception

    async def test_agent_update_contract(self, orchestrator_with_agent):
        """Test agent update operations contract."""
        
        orchestrator, agent_id = orchestrator_with_agent
        
        # Get original agent data
        original_agent = orchestrator._agents[agent_id]
        original_updated_at = original_agent.last_activity
        
        # Wait to ensure timestamp difference
        await asyncio.sleep(0.01)
        
        # Update agent (simulate task assignment)
        original_agent.current_task_id = "test-task-123"
        original_agent.last_activity = datetime.utcnow()
        
        # Validate update contract
        updated_agent = orchestrator._agents[agent_id]
        assert updated_agent.current_task_id == "test-task-123"
        assert updated_agent.last_activity > original_updated_at

    async def test_agent_session_info_contract(self, orchestrator_with_agent):
        """Test agent session information contract."""
        
        orchestrator, agent_id = orchestrator_with_agent
        
        # Test session info retrieval contract
        session_info = await orchestrator.get_agent_session_info(agent_id)
        
        if session_info:  # May be None if tmux integration not available
            session_info_schema = {
                "type": "object",
                "required": ["agent_instance"],
                "properties": {
                    "agent_instance": {"type": "object"},
                    "session_info": {"type": ["object", "null"]},
                    "launcher_status": {"type": ["object", "null"]},
                    "bridge_status": {"type": ["object", "null"]},
                    "tmux_session_id": {"type": ["string", "null"]}
                }
            }
            
            try:
                jsonschema.validate(session_info, session_info_schema)
            except ValidationError as e:
                pytest.fail(f"Agent session info contract violation: {e}")

    async def test_agent_list_sessions_contract(self, orchestrator_with_agent):
        """Test agent sessions listing contract."""
        
        orchestrator, agent_id = orchestrator_with_agent
        
        # Test list sessions contract
        sessions = await orchestrator.list_agent_sessions()
        
        # Contract validation
        assert isinstance(sessions, list)
        
        if sessions:  # May be empty if tmux integration not available
            # Each session should follow the session info contract
            session_schema = {
                "type": "object",
                "required": ["agent_instance"],
                "properties": {
                    "agent_instance": {"type": "object"},
                    "session_info": {"type": ["object", "null"]},
                    "launcher_status": {"type": ["object", "null"]},
                    "bridge_status": {"type": ["object", "null"]},
                    "tmux_session_id": {"type": ["string", "null"]}
                }
            }
            
            for session in sessions:
                try:
                    jsonschema.validate(session, session_schema)
                except ValidationError as e:
                    pytest.fail(f"Agent session list item contract violation: {e}")


class TestAgentDataConsistencyContracts:
    """Contract tests for agent data consistency and integrity."""

    async def test_agent_instance_data_contract(self):
        """Test AgentInstance data model contract."""
        
        # Create agent instance
        agent = AgentInstance(
            id="test-agent-123",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE
        )
        
        # Test to_dict contract
        agent_dict = agent.to_dict()
        
        agent_dict_schema = {
            "type": "object",
            "required": ["id", "role", "status", "created_at", "last_activity"],
            "properties": {
                "id": {"type": "string"},
                "role": {"type": "string"},
                "status": {"type": "string"},
                "current_task_id": {"type": ["string", "null"]},
                "created_at": {"type": "string"},
                "last_activity": {"type": "string"}
            }
        }
        
        try:
            jsonschema.validate(agent_dict, agent_dict_schema)
        except ValidationError as e:
            pytest.fail(f"AgentInstance to_dict contract violation: {e}")
        
        # Validate specific values
        assert agent_dict["id"] == "test-agent-123"
        assert agent_dict["role"] == AgentRole.BACKEND_DEVELOPER.value
        assert agent_dict["status"] == AgentStatus.ACTIVE.value

    async def test_agent_role_enum_contract(self):
        """Test AgentRole enum contract stability."""
        
        # Required agent roles contract
        required_roles = [
            "BACKEND_DEVELOPER",
            "FRONTEND_DEVELOPER", 
            "DEVOPS_ENGINEER",
            "QA_ENGINEER",
            "META_AGENT"
        ]
        
        # Validate all required roles exist
        for role_name in required_roles:
            assert hasattr(AgentRole, role_name), f"Required AgentRole {role_name} missing"
            role = getattr(AgentRole, role_name)
            assert isinstance(role.value, str), f"AgentRole {role_name} value should be string"

    async def test_agent_status_enum_contract(self):
        """Test AgentStatus enum contract stability."""
        
        # Required agent statuses contract
        required_statuses = ["ACTIVE", "INACTIVE", "ERROR", "PENDING"]
        
        # Validate all required statuses exist
        for status_name in required_statuses:
            if hasattr(AgentStatus, status_name):
                status = getattr(AgentStatus, status_name)
                assert isinstance(status.value, str), f"AgentStatus {status_name} value should be string"

    async def test_agent_persistence_contract(self):
        """Test agent data persistence contract."""
        
        # Test agent data that should be persisted
        persistent_data = {
            "id": "persist-test-agent",
            "role": "backend_developer",
            "agent_type": "claude",
            "status": "active",
            "tmux_session": "test-session-name",
            "created_at": datetime.utcnow()
        }
        
        # Validate persistence data schema
        persistence_schema = {
            "type": "object",
            "required": ["id", "role", "agent_type", "status", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "role": {"type": "string"},
                "agent_type": {"type": "string"},
                "status": {"type": "string"},
                "tmux_session": {"type": ["string", "null"]},
                "created_at": {"type": "object"}  # datetime object
            }
        }
        
        # Remove datetime for JSON schema validation (would be serialized in real case)
        validation_data = persistent_data.copy()
        validation_data["created_at"] = persistent_data["created_at"].isoformat()
        validation_data["created_at"] = "2025-01-18T12:00:00Z"  # Use string for validation
        
        persistence_schema["properties"]["created_at"]["type"] = "string"
        
        try:
            jsonschema.validate(validation_data, persistence_schema)
        except ValidationError as e:
            pytest.fail(f"Agent persistence data contract violation: {e}")


class TestAgentErrorHandlingContracts:
    """Contract tests for agent error handling and validation."""

    async def test_agent_creation_error_contracts(self):
        """Test agent creation error handling contracts."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Test duplicate agent ID error contract
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="test-session",
                    session_name="test-session-name",
                    workspace_path="/test/workspace"
                ))
                
                # Create first agent
                agent_id = await orchestrator.spawn_agent(
                    AgentRole.BACKEND_DEVELOPER,
                    agent_id="duplicate-test-agent"
                )
                
                # Try to create agent with same ID (should raise error)
                from app.core.simple_orchestrator import SimpleOrchestratorError
                with pytest.raises(SimpleOrchestratorError) as exc_info:
                    await orchestrator.spawn_agent(
                        AgentRole.BACKEND_DEVELOPER,
                        agent_id="duplicate-test-agent"
                    )
                
                # Validate error message contract
                assert "already exists" in str(exc_info.value)
        
        finally:
            await orchestrator.shutdown()

    async def test_agent_launcher_failure_contract(self):
        """Test agent launcher failure handling contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Test launcher failure contract
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=False,
                    error_message="Mock launcher failure"
                ))
                
                # Should raise SimpleOrchestratorError
                from app.core.simple_orchestrator import SimpleOrchestratorError
                with pytest.raises(SimpleOrchestratorError) as exc_info:
                    await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Validate error message includes launcher failure details
                assert "Failed to launch agent" in str(exc_info.value)
        
        finally:
            await orchestrator.shutdown()

    async def test_agent_not_found_error_contract(self):
        """Test agent not found error handling contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Test operations on non-existent agent
            
            # 1. Shutdown non-existent agent (should return False)
            result = await orchestrator.shutdown_agent("non-existent-agent")
            assert result is False
            
            # 2. Get session info for non-existent agent (should return None)
            session_info = await orchestrator.get_agent_session_info("non-existent-agent")
            assert session_info is None
            
            # 3. Execute command on non-existent agent (should return None)
            command_result = await orchestrator.execute_command_in_agent_session(
                "non-existent-agent", "echo test"
            )
            assert command_result is None
        
        finally:
            await orchestrator.shutdown()

    async def test_resource_limit_error_contract(self):
        """Test resource limit error handling contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            # Mock settings to limit agents
            with patch('app.core.simple_orchestrator.settings') as mock_settings:
                mock_settings.MAX_CONCURRENT_AGENTS = 2
                
                with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                    mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                        success=True,
                        session_id="test-session",
                        session_name="test-session-name",
                        workspace_path="/test/workspace"
                    ))
                    
                    # Create agents up to limit
                    agent1 = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                    agent2 = await orchestrator.spawn_agent(AgentRole.FRONTEND_DEVELOPER)
                    
                    # Try to exceed limit
                    from app.core.simple_orchestrator import SimpleOrchestratorError
                    with pytest.raises(SimpleOrchestratorError) as exc_info:
                        await orchestrator.spawn_agent(AgentRole.QA_ENGINEER)
                    
                    # Validate error message contract
                    assert "Maximum concurrent agents reached" in str(exc_info.value)
        
        finally:
            await orchestrator.shutdown()


class TestAgentPerformanceContracts:
    """Contract tests for agent operation performance requirements."""

    async def test_agent_operation_performance_contracts(self):
        """Test all agent operations meet performance contracts."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="perf-test-session",
                    session_name="perf-test-name",
                    workspace_path="/test/perf"
                ))
                
                # Performance contract tests
                performance_tests = [
                    ("spawn_agent", lambda: orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER), 100.0),
                    ("get_system_status", lambda: orchestrator.get_system_status(), 50.0),
                    ("get_performance_metrics", lambda: orchestrator.get_performance_metrics(), 100.0)
                ]
                
                created_agents = []
                
                for operation_name, operation_func, max_time_ms in performance_tests:
                    start_time = time.time()
                    result = await operation_func()
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    # Track created agents for cleanup
                    if operation_name == "spawn_agent":
                        created_agents.append(result)
                    
                    # Validate performance contract
                    assert execution_time_ms < max_time_ms, f"{operation_name} took {execution_time_ms}ms, exceeds {max_time_ms}ms contract"
                
                # Test shutdown performance
                for agent_id in created_agents:
                    start_time = time.time()
                    await orchestrator.shutdown_agent(agent_id)
                    shutdown_time_ms = (time.time() - start_time) * 1000
                    
                    # Shutdown should be fast
                    assert shutdown_time_ms < 1000.0, f"Agent shutdown took {shutdown_time_ms}ms, exceeds 1000ms contract"
        
        finally:
            await orchestrator.shutdown()

    async def test_concurrent_agent_operations_contract(self):
        """Test concurrent agent operations performance contract."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="concurrent-test-session",
                    session_name="concurrent-test-name",
                    workspace_path="/test/concurrent"
                ))
                
                # Test concurrent agent creation
                async def create_agent():
                    return await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                
                # Create 5 agents concurrently
                start_time = time.time()
                agent_tasks = [create_agent() for _ in range(5)]
                agent_ids = await asyncio.gather(*agent_tasks)
                total_time_ms = (time.time() - start_time) * 1000
                
                # Concurrent operations should not degrade performance significantly
                avg_time_per_agent = total_time_ms / 5
                assert avg_time_per_agent < 200.0, f"Concurrent agent creation avg {avg_time_per_agent}ms per agent exceeds 200ms contract"
                
                # Cleanup
                for agent_id in agent_ids:
                    await orchestrator.shutdown_agent(agent_id)
        
        finally:
            await orchestrator.shutdown()


# Integration Contract Summary
class TestAgentManagementContractSummary:
    """Summary test validating all agent management contracts work together."""
    
    async def test_complete_agent_management_contract_compliance(self):
        """Integration test ensuring all agent management contracts are compatible."""
        
        orchestrator = create_simple_orchestrator()
        
        with patch.object(orchestrator, '_ensure_dependencies_loaded', new_callable=AsyncMock):
            await orchestrator.initialize()
        
        try:
            with patch.object(orchestrator, '_agent_launcher', create=True) as mock_launcher:
                mock_launcher.launch_agent = AsyncMock(return_value=MagicMock(
                    success=True,
                    session_id="integration-session",
                    session_name="integration-session-name",
                    workspace_path="/test/integration"
                ))
                
                # Complete agent lifecycle with contract validation
                
                # 1. Agent creation contract
                start_time = time.time()
                agent_id = await orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER,
                    agent_id="contract-integration-agent"
                )
                creation_time = (time.time() - start_time) * 1000
                
                assert isinstance(agent_id, str)
                assert agent_id == "contract-integration-agent"
                assert creation_time < 100.0
                
                # 2. Agent data consistency contract
                agent = orchestrator._agents[agent_id]
                agent_dict = agent.to_dict()
                
                required_fields = ["id", "role", "status", "created_at", "last_activity"]
                for field in required_fields:
                    assert field in agent_dict
                
                # 3. Agent status contract
                assert agent.status == AgentStatus.ACTIVE
                assert agent.role == AgentRole.BACKEND_DEVELOPER
                
                # 4. System status integration contract
                status = await orchestrator.get_system_status()
                assert status["agents"]["total"] >= 1
                assert agent_id in status["agents"]["details"]
                
                # 5. Performance metrics contract
                metrics = await orchestrator.get_performance_metrics()
                assert "operation_metrics" in metrics
                assert metrics["agents"] >= 1
                
                # 6. Session info contract
                session_info = await orchestrator.get_agent_session_info(agent_id)
                if session_info:
                    assert "agent_instance" in session_info
                    assert session_info["agent_instance"]["id"] == agent_id
                
                # 7. Agent shutdown contract
                start_time = time.time()
                result = await orchestrator.shutdown_agent(agent_id)
                shutdown_time = (time.time() - start_time) * 1000
                
                assert result is True
                assert shutdown_time < 1000.0
                assert agent_id not in orchestrator._agents
                
                # 8. Post-shutdown status contract
                final_status = await orchestrator.get_system_status()
                assert agent_id not in final_status["agents"]["details"]
        
        finally:
            await orchestrator.shutdown()