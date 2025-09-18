"""
Unit Tests for SimpleOrchestrator - Complete Component Isolation

Tests the SimpleOrchestrator component in complete isolation with all external
dependencies mocked. This ensures we test only the orchestrator's business logic
without any external system dependencies.

Testing Focus:
- Agent lifecycle management 
- Task delegation logic
- Resource management and limits
- Error handling and validation
- Performance tracking
- Memory efficiency

All external dependencies are mocked:
- Database sessions
- Redis operations  
- Anthropic API
- Agent launcher
- Tmux manager
- WebSocket manager
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Component under test
from app.core.simple_orchestrator import (
    SimpleOrchestrator,
    AgentRole,
    AgentInstance,
    TaskAssignment,
    SimpleOrchestratorError,
    AgentNotFoundError,
    TaskDelegationError,
    create_simple_orchestrator,
    create_enhanced_simple_orchestrator
)

# Models for mocking
from app.models.agent import AgentStatus, AgentType
from app.models.task import TaskStatus, TaskPriority
from app.models.message import MessagePriority


class TestSimpleOrchestratorUnit:
    """Unit tests for SimpleOrchestrator component in isolation."""

    @pytest.fixture
    def mock_db_session_factory(self):
        """Mock database session factory."""
        mock_factory = Mock()
        mock_session = AsyncMock()
        mock_factory.get_session.return_value.__aenter__.return_value = mock_session
        return mock_factory

    @pytest.fixture
    def mock_cache(self):
        """Mock cache dependency."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        return mock_cache

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        return AsyncMock()

    @pytest.fixture
    def mock_agent_launcher(self):
        """Mock enhanced agent launcher."""
        mock_launcher = AsyncMock()
        
        # Mock successful launch result
        mock_result = Mock()
        mock_result.success = True
        mock_result.session_id = "test-session-123"
        mock_result.session_name = "test-agent-session"
        mock_result.workspace_path = "/test/workspace"
        mock_result.launch_time_seconds = 0.5
        mock_result.error_message = None
        
        mock_launcher.launch_agent.return_value = mock_result
        mock_launcher.terminate_agent.return_value = True
        mock_launcher.get_agent_status.return_value = {"status": "active", "recent_logs": []}
        mock_launcher.get_launcher_metrics.return_value = {"launched": 0, "terminated": 0}
        
        return mock_launcher

    @pytest.fixture
    def mock_redis_bridge(self):
        """Mock agent Redis bridge."""
        mock_bridge = AsyncMock()
        mock_bridge.register_agent.return_value = True
        mock_bridge.unregister_agent.return_value = True
        mock_bridge.assign_task_to_agent.return_value = None
        mock_bridge.get_agent_status.return_value = {"messages": 0}
        mock_bridge.get_bridge_metrics.return_value = {"agents": 0}
        mock_bridge.shutdown.return_value = None
        return mock_bridge

    @pytest.fixture
    def mock_tmux_manager(self):
        """Mock tmux session manager."""
        mock_tmux = AsyncMock()
        mock_tmux.initialize.return_value = None
        mock_tmux.get_session_info.return_value = Mock(to_dict=lambda: {"session_name": "test"})
        mock_tmux.get_session_metrics.return_value = {"active_sessions": 0}
        mock_tmux.execute_command.return_value = {"output": "test"}
        mock_tmux.shutdown.return_value = None
        return mock_tmux

    @pytest.fixture
    def mock_short_id_generator(self):
        """Mock short ID generator."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "TEST123"
        return mock_generator

    @pytest.fixture
    def mock_websocket_manager(self):
        """Mock WebSocket connection manager."""
        mock_ws = AsyncMock()
        mock_ws.broadcast_agent_update.return_value = None
        mock_ws.broadcast_task_update.return_value = None
        mock_ws.broadcast_system_status.return_value = None
        return mock_ws

    @pytest.fixture
    def isolated_orchestrator(
        self,
        mock_db_session_factory,
        mock_cache,
        mock_anthropic_client,
        mock_agent_launcher,
        mock_redis_bridge,
        mock_tmux_manager,
        mock_short_id_generator,
        mock_websocket_manager
    ):
        """Create SimpleOrchestrator with all dependencies mocked."""
        return SimpleOrchestrator(
            db_session_factory=mock_db_session_factory,
            cache=mock_cache,
            anthropic_client=mock_anthropic_client,
            agent_launcher=mock_agent_launcher,
            redis_bridge=mock_redis_bridge,
            tmux_manager=mock_tmux_manager,
            short_id_generator=mock_short_id_generator,
            websocket_manager=mock_websocket_manager
        )

    class TestInitialization:
        """Test orchestrator initialization."""

        def test_orchestrator_creation_lightweight(self):
            """Test that orchestrator creation is lightweight without heavy dependencies."""
            start_time = datetime.utcnow()
            orchestrator = SimpleOrchestrator()
            creation_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Should initialize quickly (Epic 1 requirement: <50ms response times)
            assert creation_time < 0.1  # 100ms tolerance for unit test environment
            assert orchestrator._agents == {}
            assert orchestrator._tasks == {}
            assert not orchestrator._initialized

        def test_orchestrator_with_dependencies(self, isolated_orchestrator):
            """Test orchestrator creation with mocked dependencies."""
            assert isolated_orchestrator._db_session_factory is not None
            assert isolated_orchestrator._cache is not None
            assert isolated_orchestrator._agent_launcher is not None
            assert not isolated_orchestrator._initialized

        @pytest.mark.asyncio
        async def test_initialization_process(self, isolated_orchestrator):
            """Test the initialization process."""
            # Should not be initialized initially
            assert not isolated_orchestrator._initialized
            
            # Initialize
            await isolated_orchestrator.initialize()
            
            # Should be initialized after initialization
            assert isolated_orchestrator._initialized

        @pytest.mark.asyncio
        async def test_initialization_idempotent(self, isolated_orchestrator):
            """Test that multiple initialization calls are safe."""
            await isolated_orchestrator.initialize()
            await isolated_orchestrator.initialize()  # Second call should be safe
            
            assert isolated_orchestrator._initialized

    class TestAgentLifecycle:
        """Test agent spawning and shutdown."""

        @pytest.mark.asyncio
        async def test_spawn_agent_success(self, isolated_orchestrator):
            """Test successful agent spawning."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent
            agent_id = await isolated_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                task_id="test-task-123"
            )
            
            # Verify agent was created
            assert agent_id in isolated_orchestrator._agents
            agent = isolated_orchestrator._agents[agent_id]
            assert agent.role == AgentRole.BACKEND_DEVELOPER
            assert agent.status == AgentStatus.ACTIVE
            assert agent.current_task_id is None  # Set by Redis bridge
            
            # Verify agent launcher was called
            isolated_orchestrator._agent_launcher.launch_agent.assert_called_once()

        @pytest.mark.asyncio
        async def test_spawn_agent_with_specific_id(self, isolated_orchestrator):
            """Test spawning agent with specific ID."""
            await isolated_orchestrator.initialize()
            
            specific_id = "custom-agent-id-123"
            agent_id = await isolated_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER,
                agent_id=specific_id
            )
            
            assert agent_id == specific_id
            assert specific_id in isolated_orchestrator._agents

        @pytest.mark.asyncio
        async def test_spawn_agent_duplicate_id_fails(self, isolated_orchestrator):
            """Test that spawning agent with duplicate ID fails."""
            await isolated_orchestrator.initialize()
            
            agent_id = "duplicate-test-id"
            
            # First spawn should succeed
            await isolated_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER,
                agent_id=agent_id
            )
            
            # Second spawn with same ID should fail
            with pytest.raises(SimpleOrchestratorError, match="already exists"):
                await isolated_orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER,
                    agent_id=agent_id
                )

        @pytest.mark.asyncio
        async def test_spawn_agent_resource_limit(self, isolated_orchestrator):
            """Test that resource limits are enforced."""
            await isolated_orchestrator.initialize()
            
            # Mock settings to have low limit
            with patch('app.core.simple_orchestrator.settings') as mock_settings:
                mock_settings.MAX_CONCURRENT_AGENTS = 2
                
                # Spawn up to limit
                agent1 = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
                agent2 = await isolated_orchestrator.spawn_agent(AgentRole.FRONTEND_DEVELOPER)
                
                # Third agent should fail
                with pytest.raises(SimpleOrchestratorError, match="Maximum concurrent agents reached"):
                    await isolated_orchestrator.spawn_agent(AgentRole.QA_ENGINEER)

        @pytest.mark.asyncio
        async def test_spawn_agent_launcher_failure(self, isolated_orchestrator):
            """Test handling of agent launcher failure."""
            await isolated_orchestrator.initialize()
            
            # Mock launcher failure
            mock_result = Mock()
            mock_result.success = False
            mock_result.error_message = "Launch failed"
            isolated_orchestrator._agent_launcher.launch_agent.return_value = mock_result
            
            with pytest.raises(SimpleOrchestratorError, match="Failed to launch agent"):
                await isolated_orchestrator.spawn_agent(AgentRole.DEVOPS_ENGINEER)

        @pytest.mark.asyncio
        async def test_shutdown_agent_success(self, isolated_orchestrator):
            """Test successful agent shutdown."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent first
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            assert agent_id in isolated_orchestrator._agents
            
            # Shutdown agent
            result = await isolated_orchestrator.shutdown_agent(agent_id)
            
            assert result is True
            assert agent_id not in isolated_orchestrator._agents
            isolated_orchestrator._agent_launcher.terminate_agent.assert_called_once()

        @pytest.mark.asyncio
        async def test_shutdown_agent_not_found(self, isolated_orchestrator):
            """Test shutting down non-existent agent."""
            await isolated_orchestrator.initialize()
            
            result = await isolated_orchestrator.shutdown_agent("non-existent-id")
            assert result is False

        @pytest.mark.asyncio
        async def test_shutdown_agent_graceful(self, isolated_orchestrator):
            """Test graceful agent shutdown with active task."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent and assign task
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            agent = isolated_orchestrator._agents[agent_id]
            agent.current_task_id = "active-task-123"
            
            # Mock sleep to verify graceful shutdown
            with patch('asyncio.sleep') as mock_sleep:
                result = await isolated_orchestrator.shutdown_agent(agent_id, graceful=True)
                mock_sleep.assert_called_once_with(1)
            
            assert result is True

    class TestTaskDelegation:
        """Test task delegation logic."""

        @pytest.mark.asyncio
        async def test_delegate_task_success(self, isolated_orchestrator):
            """Test successful task delegation."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent first
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Delegate task
            task_id = await isolated_orchestrator.delegate_task(
                task_description="Test task",
                task_type="development",
                priority=TaskPriority.MEDIUM,
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Verify task was created
            assert task_id in isolated_orchestrator._task_assignments
            assignment = isolated_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id
            assert assignment.status == TaskStatus.PENDING

        @pytest.mark.asyncio
        async def test_delegate_task_via_redis_bridge(self, isolated_orchestrator):
            """Test task delegation via Redis bridge."""
            await isolated_orchestrator.initialize()
            
            # Mock Redis bridge returning assigned agent
            isolated_orchestrator._redis_bridge.assign_task_to_agent.return_value = "redis-agent-123"
            
            task_id = await isolated_orchestrator.delegate_task(
                task_description="Redis task",
                task_type="testing"
            )
            
            # Verify Redis bridge was used
            isolated_orchestrator._redis_bridge.assign_task_to_agent.assert_called_once()
            
            # Verify task assignment
            assignment = isolated_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == "redis-agent-123"

        @pytest.mark.asyncio
        async def test_delegate_task_no_suitable_agent(self, isolated_orchestrator):
            """Test task delegation when no suitable agent available."""
            await isolated_orchestrator.initialize()
            
            # No agents spawned, should fail
            with pytest.raises(TaskDelegationError, match="No suitable agent available"):
                await isolated_orchestrator.delegate_task(
                    task_description="Impossible task",
                    task_type="testing"
                )

        @pytest.mark.asyncio
        async def test_delegate_task_priority_mapping(self, isolated_orchestrator):
            """Test that task priorities are mapped correctly to message priorities."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent
            await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Test high priority mapping
            await isolated_orchestrator.delegate_task(
                task_description="High priority task",
                task_type="urgent_fix",
                priority=TaskPriority.HIGH
            )
            
            # Verify Redis bridge was called with correct priority
            call_args = isolated_orchestrator._redis_bridge.assign_task_to_agent.call_args
            assert call_args[1]['priority'] == MessagePriority.HIGH

    class TestSystemStatus:
        """Test system status monitoring."""

        @pytest.mark.asyncio
        async def test_get_system_status_empty(self, isolated_orchestrator):
            """Test system status with no agents."""
            status = await isolated_orchestrator.get_system_status()
            
            assert status['agents']['total'] == 0
            assert status['agents']['by_status'] == {}
            assert status['health'] == 'no_agents'
            assert 'performance' in status
            assert isinstance(status['performance']['operations_per_second'], float)

        @pytest.mark.asyncio
        async def test_get_system_status_with_agents(self, isolated_orchestrator):
            """Test system status with active agents."""
            await isolated_orchestrator.initialize()
            
            # Spawn agents
            agent1 = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            agent2 = await isolated_orchestrator.spawn_agent(AgentRole.FRONTEND_DEVELOPER)
            
            status = await isolated_orchestrator.get_system_status()
            
            assert status['agents']['total'] == 2
            assert status['agents']['by_status']['active'] == 2
            assert status['health'] == 'healthy'
            assert len(status['agents']['details']) == 2

        @pytest.mark.asyncio
        async def test_get_system_status_performance_metrics(self, isolated_orchestrator):
            """Test that performance metrics are included in status."""
            status = await isolated_orchestrator.get_system_status()
            
            performance = status['performance']
            assert 'operations_count' in performance
            assert 'operations_per_second' in performance
            assert 'response_time_ms' in performance
            assert isinstance(performance['response_time_ms'], float)

    class TestPerformanceTracking:
        """Test performance tracking and Epic 1 compliance."""

        @pytest.mark.asyncio
        async def test_performance_tracking_spawn_agent(self, isolated_orchestrator):
            """Test performance tracking for agent spawning."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent and check performance tracking
            await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            metrics = await isolated_orchestrator.get_performance_metrics()
            assert 'spawn_agent' in metrics['operation_metrics']
            spawn_metrics = metrics['operation_metrics']['spawn_agent']
            assert 'avg_ms' in spawn_metrics
            assert 'count' in spawn_metrics
            assert spawn_metrics['count'] == 1

        @pytest.mark.asyncio
        async def test_performance_tracking_delegate_task(self, isolated_orchestrator):
            """Test performance tracking for task delegation."""
            await isolated_orchestrator.initialize()
            
            # Spawn agent and delegate task
            await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            await isolated_orchestrator.delegate_task("Test task", "development")
            
            metrics = await isolated_orchestrator.get_performance_metrics()
            assert 'delegate_task' in metrics['operation_metrics']
            delegate_metrics = metrics['operation_metrics']['delegate_task']
            assert delegate_metrics['count'] == 1

        @pytest.mark.asyncio
        async def test_epic1_compliance_tracking(self, isolated_orchestrator):
            """Test Epic 1 performance compliance tracking."""
            await isolated_orchestrator.initialize()
            
            # Spawn multiple agents to get performance data
            for i in range(3):
                await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            metrics = await isolated_orchestrator.get_performance_metrics()
            spawn_metrics = metrics['operation_metrics']['spawn_agent']
            
            # Check Epic 1 compliance field exists
            assert 'epic1_compliant' in spawn_metrics
            # Should be compliant for fast operations
            assert spawn_metrics['epic1_compliant'] is not None

    class TestErrorHandling:
        """Test error handling and validation."""

        @pytest.mark.asyncio
        async def test_operation_without_initialization_fails(self, isolated_orchestrator):
            """Test that operations without initialization are handled properly."""
            # Try to spawn agent without initialization
            # The spawn_agent method calls initialize() automatically, so this tests that behavior
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            assert agent_id is not None
            assert isolated_orchestrator._initialized

        @pytest.mark.asyncio
        async def test_database_persistence_error_handling(self, isolated_orchestrator, mock_db_session_factory):
            """Test graceful handling of database persistence errors."""
            await isolated_orchestrator.initialize()
            
            # Mock database error
            mock_session = mock_db_session_factory.get_session.return_value.__aenter__.return_value
            mock_session.add.side_effect = Exception("Database error")
            
            # Should still succeed (graceful degradation)
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            assert agent_id in isolated_orchestrator._agents

        @pytest.mark.asyncio
        async def test_cache_error_handling(self, isolated_orchestrator, mock_cache):
            """Test graceful handling of cache errors."""
            await isolated_orchestrator.initialize()
            
            # Mock cache error
            mock_cache.set.side_effect = Exception("Cache error")
            
            # Should still succeed (graceful degradation)
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            assert agent_id in isolated_orchestrator._agents

    class TestMemoryEfficiency:
        """Test memory efficiency and resource management."""

        def test_operation_time_history_limit(self, isolated_orchestrator):
            """Test that operation time history is kept within memory limits."""
            # Add many operation times
            for i in range(100):
                isolated_orchestrator._record_operation_time("test_op", float(i))
            
            # Should only keep last 50 measurements
            assert len(isolated_orchestrator._operation_times["test_op"]) == 50
            # Should keep the most recent ones
            assert isolated_orchestrator._operation_times["test_op"][-1] == 99.0

        @pytest.mark.asyncio
        async def test_lazy_loading_anthropic_client(self, isolated_orchestrator):
            """Test that Anthropic client is lazy loaded."""
            # Initially should be None
            assert isolated_orchestrator._anthropic_client is None
            
            # After calling ensure method, should be loaded
            with patch('app.core.simple_orchestrator.AsyncAnthropic') as mock_anthropic:
                client = await isolated_orchestrator._ensure_anthropic_client()
                mock_anthropic.assert_called_once()
                assert client is not None

    class TestWebSocketIntegration:
        """Test WebSocket broadcasting integration."""

        @pytest.mark.asyncio
        async def test_agent_creation_broadcast(self, isolated_orchestrator, mock_websocket_manager):
            """Test that agent creation triggers WebSocket broadcast."""
            await isolated_orchestrator.initialize()
            
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            
            # Verify WebSocket broadcast was called
            mock_websocket_manager.broadcast_agent_update.assert_called_once()
            call_args = mock_websocket_manager.broadcast_agent_update.call_args
            assert call_args[0][0] == agent_id  # First argument is agent_id
            assert 'source' in call_args[0][1]  # Second argument contains update data

        @pytest.mark.asyncio
        async def test_agent_shutdown_broadcast(self, isolated_orchestrator, mock_websocket_manager):
            """Test that agent shutdown triggers WebSocket broadcast."""
            await isolated_orchestrator.initialize()
            
            # Spawn and shutdown agent
            agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            mock_websocket_manager.reset_mock()  # Reset to ignore spawn broadcast
            
            await isolated_orchestrator.shutdown_agent(agent_id)
            
            # Verify shutdown broadcast
            mock_websocket_manager.broadcast_agent_update.assert_called_once()


class TestFactoryFunctions:
    """Test factory functions for orchestrator creation."""

    def test_create_simple_orchestrator(self):
        """Test simple orchestrator factory function."""
        orchestrator = create_simple_orchestrator()
        assert isinstance(orchestrator, SimpleOrchestrator)
        assert orchestrator._db_session_factory is None

    @pytest.mark.asyncio
    async def test_create_enhanced_simple_orchestrator(self):
        """Test enhanced orchestrator factory with full initialization."""
        # Mock all the components that get initialized
        with patch('app.core.simple_orchestrator.TmuxSessionManager') as mock_tmux, \
             patch('app.core.simple_orchestrator.create_enhanced_agent_launcher') as mock_launcher, \
             patch('app.core.simple_orchestrator.create_agent_redis_bridge') as mock_bridge, \
             patch('app.core.simple_orchestrator.create_advanced_plugin_manager') as mock_plugin:
            
            mock_tmux.return_value.initialize = AsyncMock()
            mock_launcher.return_value = AsyncMock()
            mock_bridge.return_value = AsyncMock()
            mock_plugin.return_value = None
            
            orchestrator = await create_enhanced_simple_orchestrator()
            assert isinstance(orchestrator, SimpleOrchestrator)
            assert orchestrator._initialized


class TestIntegrationHelpers:
    """Test helper methods for component integration."""

    @pytest.mark.asyncio
    async def test_find_suitable_agent_logic(self, isolated_orchestrator):
        """Test the internal agent selection logic."""
        await isolated_orchestrator.initialize()
        
        # Spawn agents of different roles
        backend_agent = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
        frontend_agent = await isolated_orchestrator.spawn_agent(AgentRole.FRONTEND_DEVELOPER)
        
        # Test finding suitable agent with preferred role
        suitable_agent = await isolated_orchestrator._find_suitable_agent(
            preferred_role=AgentRole.FRONTEND_DEVELOPER
        )
        
        assert suitable_agent is not None
        assert suitable_agent.role == AgentRole.FRONTEND_DEVELOPER
        assert suitable_agent.id == frontend_agent

    @pytest.mark.asyncio
    async def test_find_suitable_agent_no_preference(self, isolated_orchestrator):
        """Test agent selection without role preference."""
        await isolated_orchestrator.initialize()
        
        # Spawn any agent
        agent_id = await isolated_orchestrator.spawn_agent(AgentRole.QA_ENGINEER)
        
        # Should find the available agent
        suitable_agent = await isolated_orchestrator._find_suitable_agent()
        assert suitable_agent is not None
        assert suitable_agent.id == agent_id

    @pytest.mark.asyncio
    async def test_find_suitable_agent_busy_agents(self, isolated_orchestrator):
        """Test agent selection when agents are busy."""
        await isolated_orchestrator.initialize()
        
        # Spawn agent and make it busy
        agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
        agent = isolated_orchestrator._agents[agent_id]
        agent.current_task_id = "busy-task"
        
        # Should not find suitable agent (all busy)
        suitable_agent = await isolated_orchestrator._find_suitable_agent()
        assert suitable_agent is None