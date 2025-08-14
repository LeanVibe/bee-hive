"""
Comprehensive unit tests for the core orchestrator system.

Tests cover:
- Agent spawning and lifecycle management
- Task routing and assignment
- Load balancing and coordination  
- Error recovery and resilience
- Sleep-wake cycle management
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Optional

from app.core.orchestrator import (
    AgentOrchestrator, 
    AgentRole, 
    AgentCapability,
    AgentInstance
)
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.workflow import Workflow, WorkflowStatus
from app.core.workflow_engine import WorkflowResult, TaskExecutionState
from app.core.intelligent_task_router import TaskRoutingContext, RoutingStrategy, AgentSuitabilityScore


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator core functionality."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance with mocked dependencies."""
        with patch('app.core.orchestrator.get_message_broker'), \
             patch('app.core.orchestrator.get_session_cache'), \
             patch('app.core.orchestrator.get_session'), \
             patch('app.core.orchestrator.get_container_orchestrator'), \
             patch('app.core.orchestrator.get_agent_persona_system'):
            
            orchestrator = AgentOrchestrator()
            
            # Mock key dependencies
            orchestrator.message_broker = AsyncMock()
            orchestrator.session_cache = AsyncMock()
            orchestrator.db_session = AsyncMock()
            orchestrator.container_orchestrator = AsyncMock()
            orchestrator.persona_system = AsyncMock()
            orchestrator.workflow_engine = AsyncMock()
            orchestrator.task_router = AsyncMock()
            orchestrator.capability_matcher = AsyncMock()
            orchestrator.communication_service = AsyncMock()
            orchestrator.message_processor = AsyncMock()
            
            # Initialize agent instances storage
            orchestrator.agent_instances = {}
            orchestrator.agent_tasks = {}
            orchestrator.task_queue = asyncio.Queue()
            orchestrator.agent_performance = {}
            orchestrator._shutdown_event = asyncio.Event()
            orchestrator._monitoring_task = None
            
            await orchestrator._initialize()
            return orchestrator
    
    @pytest.fixture
    def sample_agent(self):
        """Create a sample agent for testing."""
        return AgentInstance(
            id="test-agent-123",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session="tmux-session-123",
            capabilities=[
                AgentCapability(
                    name="python_development",
                    description="Python backend development",
                    confidence_level=0.9,
                    specialization_areas=["FastAPI", "SQLAlchemy", "pytest"]
                )
            ],
            current_task=None,
            context_window_usage=0.3,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=AsyncMock()
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="task-456",
            title="Implement API endpoint",
            description="Create new FastAPI endpoint for user management",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            requirements=["python", "fastapi", "database"]
        )


class TestAgentLifecycleManagement:
    """Tests for agent spawning, monitoring, and shutdown."""
    
    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, orchestrator, sample_agent):
        """Test successful agent spawning."""
        # Mock container orchestrator response
        orchestrator.container_orchestrator.spawn_agent.return_value = {
            'container_id': 'container-123',
            'tmux_session': 'tmux-session-123',
            'status': 'running'
        }
        
        # Mock persona system response
        orchestrator.persona_system.assign_persona.return_value = MagicMock(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=[sample_agent.capabilities[0]]
        )
        
        # Spawn agent
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python_development"]
        )
        
        # Verify agent was created
        assert agent_id is not None
        assert agent_id in orchestrator.agent_instances
        agent = orchestrator.agent_instances[agent_id]
        assert agent.role == AgentRole.BACKEND_DEVELOPER
        assert agent.status == AgentStatus.ACTIVE
        assert len(agent.capabilities) == 1
        
        # Verify container orchestrator was called
        orchestrator.container_orchestrator.spawn_agent.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_spawn_agent_failure(self, orchestrator):
        """Test agent spawning failure handling."""
        # Mock container orchestrator failure
        orchestrator.container_orchestrator.spawn_agent.side_effect = Exception("Container spawn failed")
        
        # Attempt to spawn agent should raise exception
        with pytest.raises(Exception, match="Container spawn failed"):
            await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                capabilities=["python_development"]
            )
        
        # Verify no agent was created
        assert len(orchestrator.agent_instances) == 0
    
    @pytest.mark.asyncio
    async def test_monitor_agent_health_healthy(self, orchestrator, sample_agent):
        """Test monitoring of healthy agent."""
        # Add agent to orchestrator
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        
        # Mock container health check
        orchestrator.container_orchestrator.check_agent_health.return_value = True
        
        # Monitor agent health
        is_healthy = await orchestrator._monitor_agent_health(sample_agent.id)
        
        assert is_healthy is True
        assert sample_agent.status == AgentStatus.ACTIVE
        
    @pytest.mark.asyncio
    async def test_monitor_agent_health_unhealthy(self, orchestrator, sample_agent):
        """Test monitoring of unhealthy agent."""
        # Add agent to orchestrator
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        
        # Mock container health check failure
        orchestrator.container_orchestrator.check_agent_health.return_value = False
        
        # Monitor agent health
        is_healthy = await orchestrator._monitor_agent_health(sample_agent.id)
        
        assert is_healthy is False
        # Agent status should be updated to ERROR
        assert sample_agent.status == AgentStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_shutdown_agent(self, orchestrator, sample_agent):
        """Test graceful agent shutdown."""
        # Add agent to orchestrator
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        orchestrator.agent_tasks[sample_agent.id] = []
        
        # Mock container shutdown
        orchestrator.container_orchestrator.shutdown_agent.return_value = True
        
        # Shutdown agent
        success = await orchestrator.shutdown_agent(sample_agent.id)
        
        assert success is True
        assert sample_agent.id not in orchestrator.agent_instances
        assert sample_agent.id not in orchestrator.agent_tasks
        
        # Verify container orchestrator was called
        orchestrator.container_orchestrator.shutdown_agent.assert_called_once_with(
            sample_agent.tmux_session
        )


class TestTaskRoutingAndAssignment:
    """Tests for task routing, assignment, and load balancing."""
    
    @pytest.mark.asyncio
    async def test_assign_task_to_best_agent(self, orchestrator, sample_agent, sample_task):
        """Test task assignment to most suitable agent."""
        # Add agent to orchestrator
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        orchestrator.agent_tasks[sample_agent.id] = []
        
        # Mock task router response
        mock_score = AgentSuitabilityScore(
            agent_id=sample_agent.id,
            score=0.9,
            reasoning="Perfect match for Python development",
            confidence=0.95
        )
        orchestrator.task_router.find_best_agent.return_value = mock_score
        
        # Assign task
        assigned_agent_id = await orchestrator.assign_task(sample_task)
        
        assert assigned_agent_id == sample_agent.id
        assert sample_agent.current_task == sample_task.id
        assert sample_task.id in orchestrator.agent_tasks[sample_agent.id]
        
        # Verify task router was called
        orchestrator.task_router.find_best_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assign_task_no_suitable_agent(self, orchestrator, sample_task):
        """Test task assignment when no suitable agent available."""
        # Mock task router returning no suitable agent
        orchestrator.task_router.find_best_agent.return_value = None
        
        # Task assignment should return None
        assigned_agent_id = await orchestrator.assign_task(sample_task)
        
        assert assigned_agent_id is None
        
        # Task should be added to queue for later processing
        assert not orchestrator.task_queue.empty()
        queued_task = await orchestrator.task_queue.get()
        assert queued_task.id == sample_task.id
    
    @pytest.mark.asyncio
    async def test_load_balancing_multiple_agents(self, orchestrator, sample_task):
        """Test load balancing across multiple agents."""
        # Create multiple agents with different loads
        agents = []
        for i in range(3):
            agent = AgentInstance(
                id=f"agent-{i}",
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=f"tmux-{i}",
                capabilities=[
                    AgentCapability(
                        name="python_development",
                        description="Python backend development",
                        confidence_level=0.8,
                        specialization_areas=["FastAPI"]
                    )
                ],
                current_task=None,
                context_window_usage=0.2 + (i * 0.3),  # Different usage levels
                last_heartbeat=datetime.utcnow(),
                anthropic_client=AsyncMock()
            )
            agents.append(agent)
            orchestrator.agent_instances[agent.id] = agent
            orchestrator.agent_tasks[agent.id] = [f"existing-task-{j}" for j in range(i)]  # Different task loads
        
        # Mock task router to consider load balancing
        def mock_find_best_agent(task, routing_context):
            # Return agent with lowest load (agent-0)
            return AgentSuitabilityScore(
                agent_id="agent-0",
                score=0.9,
                reasoning="Best load balance",
                confidence=0.8
            )
        
        orchestrator.task_router.find_best_agent.side_effect = mock_find_best_agent
        
        # Assign task
        assigned_agent_id = await orchestrator.assign_task(sample_task)
        
        # Should assign to least loaded agent
        assert assigned_agent_id == "agent-0"
        assert len(orchestrator.agent_tasks["agent-0"]) == 1  # Was 0, now 1
        assert len(orchestrator.agent_tasks["agent-1"]) == 1  # Still 1
        assert len(orchestrator.agent_tasks["agent-2"]) == 2  # Still 2


class TestErrorRecoveryAndResilience:
    """Tests for error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_agent_crash_recovery(self, orchestrator, sample_agent, sample_task):
        """Test recovery from agent crash."""
        # Add agent with assigned task
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        orchestrator.agent_tasks[sample_agent.id] = [sample_task.id]
        sample_agent.current_task = sample_task.id
        
        # Simulate agent crash
        sample_agent.status = AgentStatus.ERROR
        orchestrator.container_orchestrator.check_agent_health.return_value = False
        
        # Mock recovery mechanisms
        orchestrator.container_orchestrator.restart_agent.return_value = True
        orchestrator.task_router.find_best_agent.return_value = AgentSuitabilityScore(
            agent_id=sample_agent.id,
            score=0.8,
            reasoning="Recovered agent",
            confidence=0.7
        )
        
        # Trigger recovery
        await orchestrator._handle_agent_failure(sample_agent.id)
        
        # Verify agent was restarted
        orchestrator.container_orchestrator.restart_agent.assert_called_once()
        
        # Verify task was reassigned (in this case back to same agent after restart)
        assert sample_task.id in orchestrator.agent_tasks[sample_agent.id]
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, orchestrator, sample_agent, sample_task):
        """Test handling of task timeouts."""
        # Add agent with long-running task
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        orchestrator.agent_tasks[sample_agent.id] = [sample_task.id]
        sample_agent.current_task = sample_task.id
        
        # Simulate task timeout
        sample_task.status = TaskStatus.RUNNING
        sample_task.started_at = datetime.utcnow() - timedelta(hours=2)  # Task running for 2 hours
        
        # Mock timeout detection
        orchestrator.workflow_engine.check_task_timeout.return_value = True
        
        # Handle timeout
        await orchestrator._handle_task_timeout(sample_task.id)
        
        # Verify task was cancelled and agent freed
        orchestrator.workflow_engine.cancel_task.assert_called_once_with(sample_task.id)
        assert sample_agent.current_task is None
        assert sample_task.id not in orchestrator.agent_tasks[sample_agent.id]
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, orchestrator, sample_agent):
        """Test handling of memory pressure situations."""
        # Add agent with high context usage
        sample_agent.context_window_usage = 0.95  # 95% usage - critical level
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        
        # Mock sleep-wake manager
        orchestrator.sleep_wake_manager = AsyncMock()
        orchestrator.sleep_wake_manager.should_sleep.return_value = True
        orchestrator.sleep_wake_manager.initiate_sleep.return_value = True
        
        # Check memory pressure
        await orchestrator._check_memory_pressure(sample_agent.id)
        
        # Verify sleep was initiated
        orchestrator.sleep_wake_manager.should_sleep.assert_called_once_with(
            sample_agent.id, sample_agent.context_window_usage
        )
        orchestrator.sleep_wake_manager.initiate_sleep.assert_called_once_with(sample_agent.id)


class TestPerformanceMetrics:
    """Tests for performance monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_collect_agent_metrics(self, orchestrator, sample_agent):
        """Test collection of agent performance metrics."""
        # Add agent to orchestrator
        orchestrator.agent_instances[sample_agent.id] = sample_agent
        
        # Mock metrics collection
        mock_metrics = {
            'tasks_completed': 5,
            'average_task_time': 120.5,
            'context_efficiency': 0.85,
            'error_rate': 0.02
        }
        
        # Collect metrics
        metrics = await orchestrator._collect_agent_metrics(sample_agent.id)
        
        assert isinstance(metrics, dict)
        assert 'agent_id' in metrics
        assert 'timestamp' in metrics
        assert metrics['agent_id'] == sample_agent.id
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, orchestrator):
        """Test performance-based optimization decisions."""
        # Create agents with different performance profiles
        high_performer = AgentInstance(
            id="high-perf-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session="tmux-high",
            capabilities=[],
            current_task=None,
            context_window_usage=0.4,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=AsyncMock()
        )
        
        low_performer = AgentInstance(
            id="low-perf-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session="tmux-low",
            capabilities=[],
            current_task=None,
            context_window_usage=0.8,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=AsyncMock()
        )
        
        orchestrator.agent_instances[high_performer.id] = high_performer
        orchestrator.agent_instances[low_performer.id] = low_performer
        
        # Mock performance history
        orchestrator.agent_performance[high_performer.id] = {
            'success_rate': 0.95,
            'avg_completion_time': 60,
            'context_efficiency': 0.9
        }
        orchestrator.agent_performance[low_performer.id] = {
            'success_rate': 0.7,
            'avg_completion_time': 180,
            'context_efficiency': 0.6
        }
        
        # Test optimization recommendations
        recommendations = await orchestrator._get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend actions for low performer
        low_perf_recommendations = [r for r in recommendations if r['agent_id'] == low_performer.id]
        assert len(low_perf_recommendations) > 0


class TestConcurrentOperations:
    """Tests for concurrent operations and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_assignment(self, orchestrator, sample_agent):
        """Test concurrent task assignments to multiple agents."""
        # Add multiple agents
        agents = []
        for i in range(3):
            agent = AgentInstance(
                id=f"concurrent-agent-{i}",
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=f"tmux-concurrent-{i}",
                capabilities=[],
                current_task=None,
                context_window_usage=0.2,
                last_heartbeat=datetime.utcnow(),
                anthropic_client=AsyncMock()
            )
            agents.append(agent)
            orchestrator.agent_instances[agent.id] = agent
            orchestrator.agent_tasks[agent.id] = []
        
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = Task(
                id=f"concurrent-task-{i}",
                title=f"Task {i}",
                description=f"Concurrent test task {i}",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
        
        # Mock task router to distribute tasks across agents
        def mock_find_agent(task, context):
            agent_idx = int(task.id.split('-')[-1]) % 3
            return AgentSuitabilityScore(
                agent_id=f"concurrent-agent-{agent_idx}",
                score=0.8,
                reasoning="Distributed assignment",
                confidence=0.8
            )
        
        orchestrator.task_router.find_best_agent.side_effect = mock_find_agent
        
        # Assign all tasks concurrently
        assignments = await asyncio.gather(
            *[orchestrator.assign_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Verify all tasks were assigned successfully
        assert len(assignments) == 5
        assert all(assignment is not None for assignment in assignments if not isinstance(assignment, Exception))
        
        # Verify tasks were distributed across agents
        total_assigned_tasks = sum(len(tasks) for tasks in orchestrator.agent_tasks.values())
        assert total_assigned_tasks == 5


class TestConfigurationAndSettings:
    """Tests for configuration management and settings."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_with_config(self, orchestrator):
        """Test orchestrator initialization with custom configuration."""
        # Verify default configuration was applied
        assert hasattr(orchestrator, 'config')
        assert hasattr(orchestrator, 'max_agents_per_role')
        assert hasattr(orchestrator, 'heartbeat_interval')
        
    @pytest.mark.asyncio
    async def test_dynamic_configuration_update(self, orchestrator):
        """Test dynamic configuration updates during runtime."""
        # Update configuration
        new_config = {
            'max_agents_per_role': 5,
            'heartbeat_interval': 30,
            'task_timeout_minutes': 60
        }
        
        await orchestrator.update_configuration(new_config)
        
        # Verify configuration was updated
        assert orchestrator.max_agents_per_role == 5
        assert orchestrator.heartbeat_interval == 30