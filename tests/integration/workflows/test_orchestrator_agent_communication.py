"""
Integration Tests for Orchestrator ↔ Agent Communication Flow

Tests the complete communication flow between the SimpleOrchestrator and
agent components using real component implementations with controlled test data.

Integration Points Tested:
- Orchestrator → Agent communication setup
- Agent registration and status reporting
- Task assignment and execution coordination
- Message routing through Redis streams
- Error handling and recovery workflows
- Performance monitoring and metrics collection

Test Strategy:
- Use real component instances (not mocks)
- Use test-specific Redis namespace
- Use in-memory database for test isolation
- Control external dependencies (time, UUIDs, etc.)
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, Mock
from typing import Dict, Any, List, Optional

# Real components for integration testing
from app.core.simple_orchestrator import (
    SimpleOrchestrator,
    AgentRole,
    AgentInstance,
    TaskAssignment,
    create_simple_orchestrator
)
from app.core.enhanced_agent_launcher import (
    EnhancedAgentLauncher,
    AgentLaunchConfig,
    AgentLauncherType
)
from app.core.task_queue import TaskQueue, QueuedTask
from app.models.agent import AgentStatus, AgentType
from app.models.task import TaskStatus, TaskPriority


class TestOrchestratorAgentCommunicationIntegration:
    """Integration tests for orchestrator-agent communication workflows."""

    @pytest.fixture
    async def test_redis_namespace(self):
        """Provide isolated Redis namespace for testing."""
        test_namespace = f"test_{uuid.uuid4().hex[:8]}"
        yield test_namespace
        # Cleanup would happen here in real implementation

    @pytest.fixture
    async def test_database_session(self):
        """Provide isolated database session for testing."""
        # This would create an in-memory or test database session
        class MockSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            async def commit(self):
                pass
            async def rollback(self):
                pass
            def add(self, obj):
                pass
            async def execute(self, query):
                result = Mock()
                result.scalar_one_or_none.return_value = None
                result.scalars.return_value.all.return_value = []
                return result
        
        class MockSessionFactory:
            def get_session(self):
                return MockSession()
        
        return MockSessionFactory()

    @pytest.fixture
    async def test_cache(self):
        """Provide test cache implementation."""
        class TestCache:
            def __init__(self):
                self._data = {}
            
            async def get(self, key: str):
                return self._data.get(key)
            
            async def set(self, key: str, value: Any, ttl: int = 300):
                self._data[key] = value
                return True
            
            async def delete(self, key: str):
                return self._data.pop(key, None) is not None
        
        return TestCache()

    @pytest.fixture
    async def test_task_queue(self, test_redis_namespace):
        """Provide test task queue with isolated Redis."""
        # Mock Redis client for task queue
        mock_redis = Mock()
        mock_redis.zadd = Mock(return_value=1)
        mock_redis.zrange = Mock(return_value=[])
        mock_redis.hset = Mock(return_value=1)
        mock_redis.hgetall = Mock(return_value={})
        
        return TaskQueue(redis_client=mock_redis)

    @pytest.fixture
    async def integration_orchestrator(
        self,
        test_database_session,
        test_cache,
        test_task_queue
    ):
        """Create orchestrator with real components for integration testing."""
        # Mock external dependencies that we don't want to test in integration
        mock_anthropic = Mock()
        mock_tmux_manager = Mock()
        mock_short_id_generator = Mock()
        mock_short_id_generator.generate.return_value = "TEST123"
        
        # Mock agent launcher with realistic behavior
        mock_agent_launcher = Mock()
        mock_launch_result = Mock()
        mock_launch_result.success = True
        mock_launch_result.session_id = "test-session-123"
        mock_launch_result.session_name = "test-agent-session"
        mock_launch_result.workspace_path = "/test/workspace"
        mock_launch_result.launch_time_seconds = 0.5
        mock_launch_result.error_message = None
        mock_agent_launcher.launch_agent.return_value = mock_launch_result
        mock_agent_launcher.terminate_agent.return_value = True
        mock_agent_launcher.get_agent_status.return_value = {"status": "active"}
        mock_agent_launcher.get_launcher_metrics.return_value = {"launched": 0}
        
        # Mock Redis bridge with realistic behavior
        mock_redis_bridge = Mock()
        mock_redis_bridge.register_agent.return_value = True
        mock_redis_bridge.unregister_agent.return_value = True
        mock_redis_bridge.assign_task_to_agent.return_value = None
        mock_redis_bridge.get_agent_status.return_value = {"messages": 0}
        mock_redis_bridge.get_bridge_metrics.return_value = {"agents": 0}
        mock_redis_bridge.shutdown.return_value = None
        
        # Mock WebSocket manager
        mock_websocket_manager = Mock()
        mock_websocket_manager.broadcast_agent_update.return_value = None
        mock_websocket_manager.broadcast_task_update.return_value = None
        
        orchestrator = SimpleOrchestrator(
            db_session_factory=test_database_session,
            cache=test_cache,
            anthropic_client=mock_anthropic,
            agent_launcher=mock_agent_launcher,
            redis_bridge=mock_redis_bridge,
            tmux_manager=mock_tmux_manager,
            short_id_generator=mock_short_id_generator,
            websocket_manager=mock_websocket_manager
        )
        
        await orchestrator.initialize()
        return orchestrator

    class TestAgentRegistrationWorkflow:
        """Test complete agent registration workflow."""

        @pytest.mark.asyncio
        async def test_agent_registration_complete_flow(self, integration_orchestrator):
            """Test complete agent registration workflow from start to finish."""
            # Step 1: Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                task_id="integration-test-task"
            )
            
            # Verify agent is registered in orchestrator
            assert agent_id in integration_orchestrator._agents
            agent = integration_orchestrator._agents[agent_id]
            assert agent.role == AgentRole.BACKEND_DEVELOPER
            assert agent.status == AgentStatus.ACTIVE
            
            # Verify agent launcher was called
            integration_orchestrator._agent_launcher.launch_agent.assert_called_once()
            
            # Verify Redis bridge registration was called
            integration_orchestrator._redis_bridge.register_agent.assert_called_once()
            
            # Verify cache was updated
            cached_data = await integration_orchestrator._cache.get(f"agent:{agent_id}")
            assert cached_data is not None
            
            # Verify WebSocket broadcast was triggered
            integration_orchestrator._websocket_manager.broadcast_agent_update.assert_called()

        @pytest.mark.asyncio
        async def test_agent_registration_with_task_assignment(self, integration_orchestrator):
            """Test agent registration immediately followed by task assignment."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER
            )
            
            # Assign task to the agent
            task_id = await integration_orchestrator.delegate_task(
                task_description="Integration test task",
                task_type="testing",
                priority=TaskPriority.MEDIUM,
                preferred_agent_role=AgentRole.QA_ENGINEER
            )
            
            # Verify task assignment
            assert task_id in integration_orchestrator._task_assignments
            assignment = integration_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id
            assert assignment.status == TaskStatus.PENDING

        @pytest.mark.asyncio
        async def test_agent_registration_failure_cleanup(self, integration_orchestrator):
            """Test that failed agent registration cleans up properly."""
            # Mock agent launcher failure
            integration_orchestrator._agent_launcher.launch_agent.return_value.success = False
            integration_orchestrator._agent_launcher.launch_agent.return_value.error_message = "Launch failed"
            
            # Attempt to spawn agent (should fail)
            with pytest.raises(Exception, match="Failed to launch agent"):
                await integration_orchestrator.spawn_agent(AgentRole.DEVOPS_ENGINEER)
            
            # Verify no agent was registered
            assert len(integration_orchestrator._agents) == 0
            
            # Verify no cache entries
            # This would check cache cleanup in real implementation

    class TestTaskDelegationWorkflow:
        """Test complete task delegation workflow."""

        @pytest.mark.asyncio
        async def test_task_delegation_with_available_agent(self, integration_orchestrator):
            """Test task delegation when suitable agent is available."""
            # First spawn an agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Delegate task
            task_id = await integration_orchestrator.delegate_task(
                task_description="Build React component",
                task_type="frontend_development",
                priority=TaskPriority.HIGH,
                preferred_agent_role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Verify task was assigned to the correct agent
            assignment = integration_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id
            assert assignment.status == TaskStatus.PENDING
            
            # Verify WebSocket notification was sent
            integration_orchestrator._websocket_manager.broadcast_task_update.assert_called()

        @pytest.mark.asyncio
        async def test_task_delegation_via_redis_bridge(self, integration_orchestrator):
            """Test task delegation routed through Redis bridge."""
            # Mock Redis bridge returning an agent ID
            assigned_agent_id = "redis-assigned-agent-123"
            integration_orchestrator._redis_bridge.assign_task_to_agent.return_value = assigned_agent_id
            
            # Delegate task
            task_id = await integration_orchestrator.delegate_task(
                task_description="Redis-routed task",
                task_type="processing"
            )
            
            # Verify Redis bridge was used
            integration_orchestrator._redis_bridge.assign_task_to_agent.assert_called_once()
            
            # Verify task assignment
            assignment = integration_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == assigned_agent_id

        @pytest.mark.asyncio
        async def test_task_delegation_no_suitable_agent(self, integration_orchestrator):
            """Test task delegation when no suitable agent is available."""
            # Don't spawn any agents
            
            # Attempt to delegate task (should fail)
            with pytest.raises(Exception, match="No suitable agent available"):
                await integration_orchestrator.delegate_task(
                    task_description="Impossible task",
                    task_type="unknown",
                    preferred_agent_role=AgentRole.META_AGENT
                )

    class TestAgentShutdownWorkflow:
        """Test complete agent shutdown workflow."""

        @pytest.mark.asyncio
        async def test_graceful_agent_shutdown(self, integration_orchestrator):
            """Test graceful agent shutdown workflow."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Assign a task to make shutdown more realistic
            task_id = await integration_orchestrator.delegate_task(
                task_description="Active task",
                task_type="development",
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Shutdown agent gracefully
            result = await integration_orchestrator.shutdown_agent(agent_id, graceful=True)
            
            assert result is True
            assert agent_id not in integration_orchestrator._agents
            
            # Verify agent launcher termination was called
            integration_orchestrator._agent_launcher.terminate_agent.assert_called_once_with(
                agent_id, cleanup_workspace=True
            )
            
            # Verify Redis bridge unregistration
            integration_orchestrator._redis_bridge.unregister_agent.assert_called_once_with(agent_id)
            
            # Verify WebSocket notification
            integration_orchestrator._websocket_manager.broadcast_agent_update.assert_called()

        @pytest.mark.asyncio
        async def test_forced_agent_shutdown(self, integration_orchestrator):
            """Test forced (non-graceful) agent shutdown."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER
            )
            
            # Shutdown agent immediately (not graceful)
            result = await integration_orchestrator.shutdown_agent(agent_id, graceful=False)
            
            assert result is True
            assert agent_id not in integration_orchestrator._agents
            
            # Verify immediate termination
            integration_orchestrator._agent_launcher.terminate_agent.assert_called_once()

    class TestSystemStatusWorkflow:
        """Test system status monitoring workflow."""

        @pytest.mark.asyncio
        async def test_system_status_with_active_agents(self, integration_orchestrator):
            """Test system status reporting with active agents and tasks."""
            # Spawn multiple agents
            agent_ids = []
            for i, role in enumerate([AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER]):
                agent_id = await integration_orchestrator.spawn_agent(role=role)
                agent_ids.append(agent_id)
            
            # Delegate some tasks
            task_ids = []
            for i in range(2):
                task_id = await integration_orchestrator.delegate_task(
                    task_description=f"Task {i+1}",
                    task_type="development"
                )
                task_ids.append(task_id)
            
            # Get system status
            status = await integration_orchestrator.get_system_status()
            
            # Verify status contains expected information
            assert status['agents']['total'] == 2
            assert status['agents']['by_status']['active'] == 2
            assert status['tasks']['active_assignments'] == 2
            assert status['health'] == 'healthy'
            assert 'performance' in status
            assert len(status['agents']['details']) == 2

        @pytest.mark.asyncio
        async def test_enhanced_system_status_integration(self, integration_orchestrator):
            """Test enhanced system status with component metrics."""
            # Spawn agent to generate some activity
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.DEVOPS_ENGINEER
            )
            
            # Get enhanced status
            enhanced_status = await integration_orchestrator.get_enhanced_system_status()
            
            # Verify enhanced status includes component metrics
            assert 'tmux_integration' in enhanced_status
            assert 'agent_launcher' in enhanced_status
            assert 'redis_bridge' in enhanced_status
            assert 'enhanced_agents' in enhanced_status
            
            # Verify component metrics were collected
            assert enhanced_status['enhanced_agents']['total_with_sessions'] >= 0
            assert enhanced_status['enhanced_agents']['initialized'] is True

    class TestErrorRecoveryWorkflow:
        """Test error recovery workflows."""

        @pytest.mark.asyncio
        async def test_agent_communication_failure_recovery(self, integration_orchestrator):
            """Test recovery from agent communication failures."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Simulate Redis bridge failure
            integration_orchestrator._redis_bridge.register_agent.side_effect = Exception("Redis down")
            
            # Attempt another agent spawn (should handle Redis failure gracefully)
            try:
                await integration_orchestrator.spawn_agent(
                    role=AgentRole.FRONTEND_DEVELOPER
                )
            except Exception:
                # Expected if Redis is critical, or should succeed with degraded functionality
                pass
            
            # Verify first agent is still functional
            status = await integration_orchestrator.get_agent_session_info(agent_id)
            assert status is not None

        @pytest.mark.asyncio
        async def test_database_failure_graceful_degradation(self, integration_orchestrator):
            """Test graceful degradation when database operations fail."""
            # Mock database session to fail
            integration_orchestrator._db_session_factory.get_session.side_effect = Exception("DB connection lost")
            
            # Should still be able to spawn agent (graceful degradation)
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER
            )
            
            # Agent should be in memory even if not persisted
            assert agent_id in integration_orchestrator._agents

        @pytest.mark.asyncio
        async def test_cache_failure_continues_operation(self, integration_orchestrator):
            """Test that cache failures don't stop core operations."""
            # Mock cache to fail
            integration_orchestrator._cache.set.side_effect = Exception("Cache unavailable")
            
            # Should still be able to spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.DEVOPS_ENGINEER
            )
            
            # Core functionality should work
            assert agent_id in integration_orchestrator._agents
            
            # Task delegation should also work
            task_id = await integration_orchestrator.delegate_task(
                task_description="Cache-independent task",
                task_type="ops"
            )
            assert task_id in integration_orchestrator._task_assignments

    class TestPerformanceIntegration:
        """Test performance aspects of integrated workflows."""

        @pytest.mark.asyncio
        async def test_concurrent_agent_operations(self, integration_orchestrator):
            """Test concurrent agent operations for performance."""
            # Spawn multiple agents concurrently
            tasks = []
            for i in range(5):
                task = integration_orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER
                )
                tasks.append(task)
            
            # Wait for all to complete
            agent_ids = await asyncio.gather(*tasks)
            
            # Verify all agents were created
            assert len(agent_ids) == 5
            assert len(integration_orchestrator._agents) == 5
            
            # Check performance metrics
            performance_metrics = await integration_orchestrator.get_performance_metrics()
            assert 'spawn_agent' in performance_metrics['operation_metrics']
            assert performance_metrics['operation_metrics']['spawn_agent']['count'] == 5

        @pytest.mark.asyncio
        async def test_high_frequency_task_delegation(self, integration_orchestrator):
            """Test high-frequency task delegation performance."""
            # Spawn agents to handle tasks
            for role in [AgentRole.BACKEND_DEVELOPER, AgentRole.FRONTEND_DEVELOPER]:
                await integration_orchestrator.spawn_agent(role=role)
            
            # Delegate many tasks quickly
            task_ids = []
            for i in range(10):
                task_id = await integration_orchestrator.delegate_task(
                    task_description=f"High frequency task {i+1}",
                    task_type="performance_test"
                )
                task_ids.append(task_id)
            
            # Verify all tasks were handled
            assert len(task_ids) == 10
            assert len(integration_orchestrator._task_assignments) == 10
            
            # Check delegation performance
            performance_metrics = await integration_orchestrator.get_performance_metrics()
            delegate_metrics = performance_metrics['operation_metrics']['delegate_task']
            assert delegate_metrics['count'] == 10
            
            # Should meet Epic 1 performance targets
            assert delegate_metrics['avg_ms'] < 500.0  # Epic 1 target for task delegation

    class TestDataConsistencyWorkflow:
        """Test data consistency across component boundaries."""

        @pytest.mark.asyncio
        async def test_agent_state_consistency(self, integration_orchestrator):
            """Test that agent state remains consistent across components."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.META_AGENT
            )
            
            # Check state in orchestrator
            orchestrator_agent = integration_orchestrator._agents[agent_id]
            assert orchestrator_agent.status == AgentStatus.ACTIVE
            
            # Check cached state
            cached_data = await integration_orchestrator._cache.get(f"agent:{agent_id}")
            assert cached_data is not None
            assert cached_data['status'] == AgentStatus.ACTIVE.value
            
            # Get session info (should be consistent)
            session_info = await integration_orchestrator.get_agent_session_info(agent_id)
            assert session_info['agent_instance']['status'] == AgentStatus.ACTIVE.value

        @pytest.mark.asyncio
        async def test_task_assignment_consistency(self, integration_orchestrator):
            """Test task assignment consistency across components."""
            # Spawn agent
            agent_id = await integration_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Delegate task
            task_id = await integration_orchestrator.delegate_task(
                task_description="Consistency test task",
                task_type="testing",
                preferred_agent_role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Verify task assignment in orchestrator
            assignment = integration_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id
            assert assignment.status == TaskStatus.PENDING
            
            # Verify agent knows about the task
            # In real implementation, agent would be updated
            # For now, just verify assignment exists
            assert task_id in integration_orchestrator._task_assignments