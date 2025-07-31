"""
Comprehensive tests for Agent Orchestrator functionality.

Tests cover agent lifecycle management, task delegation, scheduling,
health monitoring, and workflow coordination with >90% coverage.
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentCapability, AgentInstance
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority


class TestAgentOrchestrator:
    """Test suite for Agent Orchestrator core functionality."""
    
    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create a test orchestrator instance."""
        orchestrator = AgentOrchestrator()
        
        # Mock dependencies
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        orchestrator.anthropic_client = AsyncMock()
        
        return orchestrator
    
    @pytest.fixture
    def sample_agent_capability(self):
        """Create sample agent capability for testing."""
        return AgentCapability(
            name="python_development",
            description="Python backend development",
            confidence_level=0.9,
            specialization_areas=["fastapi", "sqlalchemy", "pytest"]
        )
    
    @pytest.fixture
    def sample_agent_instance(self, sample_agent_capability):
        """Create sample agent instance for testing."""
        return AgentInstance(
            id="test-agent-001",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session=None,
            capabilities=[sample_agent_capability],
            current_task=None,
            context_window_usage=0.3,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator proper initialization."""
        assert orchestrator.agents == {}
        assert orchestrator.active_sessions == {}
        assert not orchestrator.is_running
        assert orchestrator.metrics is not None
        assert 'tasks_completed' in orchestrator.metrics
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, mock_get_session, orchestrator):
        """Test successful agent spawning."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock message broker
        orchestrator.message_broker.create_consumer_group = AsyncMock()
        
        # Test agent spawning
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            agent_id="test-backend-001"
        )
        
        assert agent_id == "test-backend-001"
        assert agent_id in orchestrator.agents
        assert orchestrator.agents[agent_id].role == AgentRole.BACKEND_DEVELOPER
        assert orchestrator.agents[agent_id].status == AgentStatus.INITIALIZING
        
        # Verify database interaction
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_spawn_agent_duplicate_error(self, orchestrator):
        """Test error when spawning duplicate agent."""
        orchestrator.agents["existing-agent"] = MagicMock()
        
        with pytest.raises(ValueError, match="Agent existing-agent already exists"):
            await orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                agent_id="existing-agent"
            )
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_shutdown_agent_success(self, mock_get_session, orchestrator, sample_agent_instance):
        """Test successful agent shutdown."""
        # Setup
        agent_id = sample_agent_instance.id
        orchestrator.agents[agent_id] = sample_agent_instance
        
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.get.return_value = MagicMock()
        
        # Test shutdown
        result = await orchestrator.shutdown_agent(agent_id)
        
        assert result is True
        assert agent_id not in orchestrator.agents
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_nonexistent_agent(self, orchestrator):
        """Test shutdown of non-existent agent."""
        result = await orchestrator.shutdown_agent("nonexistent-agent")
        assert result is False
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_delegate_task_success(self, mock_get_session, orchestrator, sample_agent_instance):
        """Test successful task delegation."""
        # Setup
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        orchestrator._schedule_task = AsyncMock(return_value=sample_agent_instance.id)
        
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Test task delegation
        task_id = await orchestrator.delegate_task(
            task_description="Implement user authentication",
            task_type="backend_development",
            priority=TaskPriority.HIGH,
            required_capabilities=["python", "authentication"]
        )
        
        assert task_id is not None
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        orchestrator._schedule_task.assert_called_once()
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_delegate_task_no_agent_available(self, mock_get_session, orchestrator):
        """Test task delegation when no agents are available."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        orchestrator._schedule_task = AsyncMock(return_value=None)
        
        # Test task delegation
        task_id = await orchestrator.delegate_task(
            task_description="Test task",
            task_type="testing"
        )
        
        assert task_id is not None  # Task is created but not assigned
        orchestrator._schedule_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_candidate_agents_by_role(self, orchestrator, sample_agent_instance):
        """Test finding candidate agents by role."""
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        candidates = await orchestrator._find_candidate_agents(
            task_type="backend",
            preferred_role=AgentRole.BACKEND_DEVELOPER
        )
        
        assert len(candidates) == 1
        assert candidates[0].id == sample_agent_instance.id
    
    @pytest.mark.asyncio
    async def test_find_candidate_agents_by_capabilities(self, orchestrator, sample_agent_instance):
        """Test finding candidate agents by required capabilities."""
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        candidates = await orchestrator._find_candidate_agents(
            task_type="python",
            required_capabilities=["python_development"]
        )
        
        assert len(candidates) == 1
        assert candidates[0].id == sample_agent_instance.id
    
    @pytest.mark.asyncio
    async def test_find_candidate_agents_no_match(self, orchestrator, sample_agent_instance):
        """Test finding candidate agents with no matching capabilities."""
        sample_agent_instance.status = AgentStatus.BUSY
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        candidates = await orchestrator._find_candidate_agents(
            task_type="frontend",
            required_capabilities=["react", "typescript"]
        )
        
        assert len(candidates) == 0
    
    def test_agent_has_required_capabilities_success(self, orchestrator, sample_agent_instance):
        """Test capability matching - positive case."""
        result = orchestrator._agent_has_required_capabilities(
            sample_agent_instance,
            ["python"]
        )
        assert result is True
    
    def test_agent_has_required_capabilities_failure(self, orchestrator, sample_agent_instance):
        """Test capability matching - negative case."""
        result = orchestrator._agent_has_required_capabilities(
            sample_agent_instance,
            ["javascript", "react"]
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_calculate_agent_suitability_score(self, orchestrator, sample_agent_instance):
        """Test agent suitability scoring algorithm."""
        score = await orchestrator._calculate_agent_suitability_score(
            sample_agent_instance,
            "python_development",
            TaskPriority.HIGH,
            ["python_development"]
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be a good match
    
    def test_calculate_priority_score_high_priority(self, orchestrator, sample_agent_instance):
        """Test priority scoring for high priority tasks."""
        score = orchestrator._calculate_priority_score(
            sample_agent_instance,
            TaskPriority.HIGH
        )
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.6  # Should prefer recently active agents
    
    def test_calculate_priority_score_low_priority(self, orchestrator, sample_agent_instance):
        """Test priority scoring for low priority tasks."""
        score = orchestrator._calculate_priority_score(
            sample_agent_instance,
            TaskPriority.LOW
        )
        
        assert score == 0.8  # Standard score for low priority
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_assign_task_to_agent_success(self, mock_get_session, orchestrator, sample_agent_instance):
        """Test successful task assignment to agent."""
        # Setup
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Test assignment
        result = await orchestrator._assign_task_to_agent("task-123", sample_agent_instance.id)
        
        assert result is True
        assert sample_agent_instance.current_task == "task-123"
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
        orchestrator.message_broker.send_message.assert_called_once()
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_process_task_queue(self, mock_get_session, orchestrator):
        """Test task queue processing."""
        # Mock database session with pending tasks
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Create mock tasks
        mock_task = MagicMock()
        mock_task.id = uuid.uuid4()
        mock_task.task_type = TaskType.FEATURE_DEVELOPMENT
        mock_task.priority = TaskPriority.MEDIUM
        mock_task.required_capabilities = ["python"]
        
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = [mock_task]
        orchestrator._schedule_task = AsyncMock(return_value="agent-123")
        
        # Test queue processing
        assigned_count = await orchestrator.process_task_queue()
        
        assert assigned_count == 1
        orchestrator._schedule_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initiate_sleep_cycle_high_context_usage(self, orchestrator, sample_agent_instance):
        """Test sleep cycle initiation for high context usage."""
        sample_agent_instance.context_window_usage = 0.95
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        result = await orchestrator.initiate_sleep_cycle(sample_agent_instance.id)
        
        assert result is True
        assert sample_agent_instance.status == AgentStatus.SLEEPING
        orchestrator.message_broker.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initiate_sleep_cycle_low_context_usage(self, orchestrator, sample_agent_instance):
        """Test sleep cycle not initiated for low context usage."""
        sample_agent_instance.context_window_usage = 0.3
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.CONSOLIDATION_THRESHOLD = 0.8
            result = await orchestrator.initiate_sleep_cycle(sample_agent_instance.id)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, orchestrator, sample_agent_instance):
        """Test system status reporting."""
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        orchestrator.is_running = True
        orchestrator._check_system_health = AsyncMock(return_value={"overall": True})
        
        status = await orchestrator.get_system_status()
        
        assert status["orchestrator_status"] == "running"
        assert status["total_agents"] == 1
        assert status["active_agents"] == 1
        assert "agents" in status
        assert "metrics" in status
        assert "system_health" in status
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_handle_agent_timeout(self, mock_get_session, orchestrator, sample_agent_instance):
        """Test agent timeout handling."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock settings for auto-restart
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.AUTO_RESTART_AGENTS = False
            
            await orchestrator._handle_agent_timeout(sample_agent_instance.id, sample_agent_instance)
        
        assert sample_agent_instance.status == AgentStatus.ERROR
        mock_db_session.execute.assert_called_once()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_monitor_agent_health_critical_context(self, mock_get_session, orchestrator, sample_agent_instance):
        """Test health monitoring for critical context usage."""
        sample_agent_instance.context_window_usage = 0.98
        orchestrator._record_health_issues = AsyncMock()
        
        await orchestrator._monitor_agent_health(sample_agent_instance.id, sample_agent_instance)
        
        orchestrator._record_health_issues.assert_called_once()
        call_args = orchestrator._record_health_issues.call_args[0]
        assert "critical_context_usage" in call_args[1]
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_check_database_health_success(self, mock_get_session, orchestrator):
        """Test successful database health check."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar.return_value = 1
        mock_db_session.execute.return_value.scalars.return_value.first.return_value = uuid.uuid4()
        
        result = await orchestrator._check_database_health()
        
        assert result is True
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_check_database_health_failure(self, mock_get_session, orchestrator):
        """Test database health check failure."""
        # Mock database session that raises exception
        mock_get_session.return_value.__aenter__.side_effect = Exception("Database connection failed")
        
        result = await orchestrator._check_database_health()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_redis_health_success(self, orchestrator):
        """Test successful Redis health check."""
        orchestrator.message_broker = MagicMock()  # Simulate active broker
        
        result = await orchestrator._check_redis_health()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_redis_health_failure(self, orchestrator):
        """Test Redis health check failure."""
        orchestrator.message_broker = None
        
        result = await orchestrator._check_redis_health()
        
        assert result is False
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_check_task_processing_health_success(self, mock_get_session, orchestrator):
        """Test successful task processing health check."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock query results - no stuck tasks, reasonable queue size
        mock_db_session.execute.return_value.scalar.side_effect = [0, 5]  # stuck_tasks, pending_tasks
        
        result = await orchestrator._check_task_processing_health()
        
        assert result is True
    
    @patch('app.core.orchestrator.get_session')
    @pytest.mark.asyncio
    async def test_check_task_processing_health_failure(self, mock_get_session, orchestrator):
        """Test task processing health check failure."""
        # Mock database session
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        
        # Mock query results - stuck tasks present
        mock_db_session.execute.return_value.scalar.side_effect = [5, 10]  # stuck_tasks, pending_tasks
        
        result = await orchestrator._check_task_processing_health()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_system_resources_success(self, orchestrator, sample_agent_instance):
        """Test successful system resources health check."""
        # Add multiple agents with good health
        orchestrator.agents["agent-1"] = sample_agent_instance
        
        agent_2 = sample_agent_instance.copy()
        agent_2.id = "agent-2"
        agent_2.context_window_usage = 0.4
        orchestrator.agents["agent-2"] = agent_2
        
        result = await orchestrator._check_system_resources()
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_system_resources_failure(self, orchestrator, sample_agent_instance):
        """Test system resources health check failure."""
        # Add agent with high memory usage
        sample_agent_instance.context_window_usage = 0.95
        orchestrator.agents[sample_agent_instance.id] = sample_agent_instance
        
        result = await orchestrator._check_system_resources()
        
        assert result is False  # High memory usage should fail check
    
    def test_get_default_capabilities_backend_developer(self, orchestrator):
        """Test default capabilities for backend developer role."""
        capabilities = orchestrator._get_default_capabilities(AgentRole.BACKEND_DEVELOPER)
        
        assert len(capabilities) > 0
        capability_names = [cap.name for cap in capabilities]
        assert "api_development" in capability_names
        assert "database_design" in capability_names
    
    def test_get_default_capabilities_unknown_role(self, orchestrator):
        """Test default capabilities for unknown role."""
        capabilities = orchestrator._get_default_capabilities(AgentRole.META_AGENT)
        
        assert len(capabilities) > 0  # Should have default capabilities


class TestAgentOrchestratorIntegration:
    """Integration tests for orchestrator with real components."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self):
        """Test complete agent lifecycle from spawn to shutdown."""
        orchestrator = AgentOrchestrator()
        
        # Mock dependencies
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        orchestrator.message_broker.create_consumer_group = AsyncMock()
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Spawn agent
            agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            assert agent_id is not None
            assert agent_id in orchestrator.agents
            
            # Verify agent state
            agent = orchestrator.agents[agent_id]
            assert agent.status == AgentStatus.INITIALIZING
            assert agent.role == AgentRole.BACKEND_DEVELOPER
            
            # Shutdown agent
            result = await orchestrator.shutdown_agent(agent_id)
            assert result is True
            assert agent_id not in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_task_delegation_workflow(self):
        """Test complete task delegation workflow."""
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        
        # Create a mock agent
        agent_capability = AgentCapability(
            name="python_development",
            description="Python development",
            confidence_level=0.9,
            specialization_areas=["fastapi"]
        )
        
        agent_instance = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session=None,
            capabilities=[agent_capability],
            current_task=None,
            context_window_usage=0.3,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None
        )
        
        orchestrator.agents["test-agent"] = agent_instance
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Create mock task for scheduling
            mock_task = MagicMock()
            mock_task.id = "task-123"
            mock_db_session.get.return_value = mock_task
            
            # Delegate task
            task_id = await orchestrator.delegate_task(
                task_description="Implement API endpoint",
                task_type="api_development",
                required_capabilities=["python_development"]
            )
            
            assert task_id is not None
            # Verify task was created in database
            mock_db_session.add.assert_called()
            mock_db_session.commit.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.core.orchestrator", "--cov-report=term-missing"])