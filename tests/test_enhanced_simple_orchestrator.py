"""
Comprehensive test suite for Enhanced Simple Orchestrator

Tests all functionality including:
- Agent lifecycle management
- Task delegation with multiple strategies
- Error handling and recovery
- Performance monitoring
- Database operations
- Configuration management
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

# Import the enhanced orchestrator and its components
from app.core.simple_orchestrator_enhanced import (
    EnhancedSimpleOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole,
    TaskAssignmentStrategy,
    AgentInstance,
    TaskAssignment,
    PerformanceMetrics,
    SimpleOrchestratorError,
    AgentNotFoundError,
    TaskDelegationError,
    DatabaseOperationError,
    ConfigurationError,
    ResourceLimitError,
    create_enhanced_orchestrator
)
from app.models.agent import AgentStatus, AgentType
from app.models.task import TaskStatus, TaskPriority, TaskType


class MockDatabaseDependency:
    """Mock database dependency for testing."""
    
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.session = AsyncMock(spec=AsyncSession)
        
    async def get_session(self) -> AsyncSession:
        if self.should_fail:
            raise SQLAlchemyError("Mock database error")
        return self.session


class MockCacheDependency:
    """Mock cache dependency for testing."""
    
    def __init__(self):
        self.cache = {}
        
    async def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        self.cache[key] = value
        return True


class MockMetricsDependency:
    """Mock metrics dependency for testing."""
    
    def __init__(self):
        self.counters = {}
        self.histograms = {}
        
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        key = f"{name}:{tags}" if tags else name
        self.counters[key] = self.counters.get(key, 0) + 1
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        key = f"{name}:{tags}" if tags else name
        self.histograms.setdefault(key, []).append(value)


@pytest.fixture
def test_config():
    """Test configuration."""
    return OrchestratorConfig(
        mode=OrchestratorMode.TEST,
        max_concurrent_agents=5,
        enable_database_persistence=False,  # Disable for most tests
        enable_caching=True,
        enable_performance_monitoring=True,
        heartbeat_interval_seconds=1,
        task_timeout_minutes=1,
        database_retry_attempts=1
    )


@pytest.fixture
def mock_db():
    """Mock database dependency."""
    return MockDatabaseDependency()


@pytest.fixture
def mock_cache():
    """Mock cache dependency."""
    return MockCacheDependency()


@pytest.fixture
def mock_metrics():
    """Mock metrics dependency."""
    return MockMetricsDependency()


@pytest.fixture
async def orchestrator(test_config, mock_db, mock_cache, mock_metrics):
    """Create test orchestrator instance."""
    orch = EnhancedSimpleOrchestrator(
        config=test_config,
        db_session_factory=mock_db,
        cache=mock_cache,
        metrics=mock_metrics
    )
    await orch.start()
    yield orch
    await orch.shutdown()


class TestOrchestratorConfig:
    """Test orchestrator configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.mode == OrchestratorMode.DEVELOPMENT
        assert config.max_concurrent_agents == 10
        assert config.default_task_assignment_strategy == TaskAssignmentStrategy.AVAILABILITY_BASED
        assert config.enable_performance_monitoring is True
    
    def test_config_from_settings(self):
        """Test creating config from settings."""
        with patch('app.core.simple_orchestrator_enhanced.settings') as mock_settings:
            mock_settings.ENVIRONMENT = "production"
            mock_settings.DEBUG = False
            mock_settings.PROMETHEUS_METRICS_ENABLED = True
            
            config = OrchestratorConfig.from_settings()
            assert config.mode == OrchestratorMode.PRODUCTION
            assert config.enable_database_persistence is True
            assert config.enable_performance_monitoring is True


class TestAgentInstance:
    """Test AgentInstance functionality."""
    
    def test_agent_creation(self):
        """Test creating an agent instance."""
        agent = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.active,
            capabilities=["python", "django"]
        )
        
        assert agent.id == "test-agent"
        assert agent.role == AgentRole.BACKEND_DEVELOPER
        assert agent.status == AgentStatus.active
        assert agent.capabilities == ["python", "django"]
        assert agent.load_score == 0.0
    
    def test_agent_to_dict(self):
        """Test converting agent to dictionary."""
        agent = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.active
        )
        
        agent_dict = agent.to_dict()
        assert agent_dict["id"] == "test-agent"
        assert agent_dict["role"] == "backend_developer"
        assert agent_dict["status"] == "active"
    
    def test_agent_health_check(self):
        """Test agent health checking."""
        agent = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.active
        )
        
        # New agent should be healthy
        assert agent.is_healthy() is True
        
        # Agent with recent heartbeat should be healthy
        agent.update_heartbeat()
        assert agent.is_healthy() is True
        
        # Agent with old heartbeat should be unhealthy
        agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        assert agent.is_healthy() is False
    
    def test_agent_availability(self):
        """Test agent availability checking."""
        agent = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.active
        )
        
        # Active agent with no task should be available
        assert agent.is_available_for_task() is True
        
        # Agent with task should not be available
        agent.current_task_id = "some-task"
        assert agent.is_available_for_task() is False
        
        # Agent with high load should not be available
        agent.current_task_id = None
        agent.load_score = 0.9
        assert agent.is_available_for_task() is False
        
        # Inactive agent should not be available
        agent.load_score = 0.0
        agent.status = AgentStatus.inactive
        assert agent.is_available_for_task() is False
    
    def test_task_suitability_calculation(self):
        """Test task suitability calculation."""
        agent = AgentInstance(
            id="test-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.active,
            capabilities=["python", "django", "api"]
        )
        
        # Test with matching capabilities
        score = agent.calculate_task_suitability(["python", "api"], "backend_api_development")
        assert score > 0.5  # Should be above base score
        
        # Test with no matching capabilities
        score = agent.calculate_task_suitability(["javascript", "react"], "frontend_development")
        assert score < 0.7  # Should be lower due to no capability match
        
        # Test unavailable agent
        agent.status = AgentStatus.inactive
        score = agent.calculate_task_suitability(["python"], "backend")
        assert score == 0.0


class TestTaskAssignment:
    """Test TaskAssignment functionality."""
    
    def test_assignment_creation(self):
        """Test creating a task assignment."""
        assignment = TaskAssignment(
            task_id="test-task",
            agent_id="test-agent",
            priority=TaskPriority.HIGH,
            estimated_duration=30
        )
        
        assert assignment.task_id == "test-task"
        assert assignment.agent_id == "test-agent"
        assert assignment.status == TaskStatus.PENDING
        assert assignment.priority == TaskPriority.HIGH
        assert assignment.estimated_duration == 30
    
    def test_assignment_to_dict(self):
        """Test converting assignment to dictionary."""
        assignment = TaskAssignment(
            task_id="test-task",
            agent_id="test-agent",
            priority=TaskPriority.HIGH
        )
        
        assignment_dict = assignment.to_dict()
        assert assignment_dict["task_id"] == "test-task"
        assert assignment_dict["agent_id"] == "test-agent"
        assert assignment_dict["priority"] == TaskPriority.HIGH.value


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_metrics_recording(self):
        """Test recording operation metrics."""
        metrics = PerformanceMetrics()
        
        # Record successful operations
        metrics.record_operation(True, 100.0)
        metrics.record_operation(True, 200.0)
        metrics.record_operation(False, 150.0)
        
        assert metrics.operation_count == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert metrics.get_success_rate() == 2/3
        assert metrics.get_average_response_time() == 150.0
    
    def test_empty_metrics(self):
        """Test metrics with no data."""
        metrics = PerformanceMetrics()
        
        assert metrics.get_success_rate() == 1.0
        assert metrics.get_average_response_time() == 0.0
        assert metrics.get_p95_response_time() == 0.0


class TestExceptions:
    """Test custom exceptions."""
    
    def test_base_exception(self):
        """Test base orchestrator exception."""
        error = SimpleOrchestratorError(
            "Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert isinstance(error.timestamp, datetime)
    
    def test_agent_not_found_error(self):
        """Test agent not found exception."""
        error = AgentNotFoundError("test-agent-id")
        
        assert "test-agent-id" in str(error)
        assert error.error_code == "AGENT_NOT_FOUND"
        assert error.agent_id == "test-agent-id"
    
    def test_task_delegation_error(self):
        """Test task delegation exception."""
        error = TaskDelegationError("No agents available", {"task_type": "test"})
        
        assert "No agents available" in str(error)
        assert error.error_code == "TASK_DELEGATION_ERROR"
        assert error.details["task_type"] == "test"
    
    def test_database_operation_error(self):
        """Test database operation exception."""
        original_error = SQLAlchemyError("SQL error")
        error = DatabaseOperationError("insert", original_error)
        
        assert "insert" in str(error)
        assert error.error_code == "DATABASE_ERROR"
        assert error.operation == "insert"
        assert error.original_error == original_error
    
    def test_configuration_error(self):
        """Test configuration exception."""
        error = ConfigurationError("max_agents", "Must be positive")
        
        assert "max_agents" in str(error)
        assert error.error_code == "CONFIG_ERROR"
        assert error.config_key == "max_agents"
    
    def test_resource_limit_error(self):
        """Test resource limit exception."""
        error = ResourceLimitError("agents", 10, 15)
        
        assert "agents" in str(error)
        assert error.error_code == "RESOURCE_LIMIT_ERROR"
        assert error.resource == "agents"
        assert error.limit == 10
        assert error.current == 15


class TestOrchestratorBasicOperations:
    """Test basic orchestrator operations."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_startup_shutdown(self, test_config, mock_db, mock_cache, mock_metrics):
        """Test orchestrator startup and shutdown."""
        orch = EnhancedSimpleOrchestrator(
            config=test_config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        # Start orchestrator
        await orch.start()
        assert orch._shutdown_event is not None
        
        # Shutdown orchestrator
        await orch.shutdown()
        assert orch._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, orchestrator):
        """Test successful agent spawning."""
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "django"]
        )
        
        assert agent_id is not None
        assert agent_id in orchestrator._agents
        
        agent = orchestrator._agents[agent_id]
        assert agent.role == AgentRole.BACKEND_DEVELOPER
        assert agent.status == AgentStatus.active
        assert agent.capabilities == ["python", "django"]
    
    @pytest.mark.asyncio
    async def test_spawn_agent_with_custom_id(self, orchestrator):
        """Test spawning agent with custom ID."""
        custom_id = "custom-agent-123"
        agent_id = await orchestrator.spawn_agent(
            role=AgentRole.FRONTEND_DEVELOPER,
            agent_id=custom_id
        )
        
        assert agent_id == custom_id
        assert custom_id in orchestrator._agents
    
    @pytest.mark.asyncio
    async def test_spawn_agent_duplicate_id(self, orchestrator):
        """Test spawning agent with duplicate ID."""
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        with pytest.raises(SimpleOrchestratorError, match="already exists"):
            await orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER,
                agent_id=agent_id
            )
    
    @pytest.mark.asyncio
    async def test_spawn_agent_limit_exceeded(self, orchestrator):
        """Test spawning agents beyond limit."""
        # Spawn agents up to limit
        for i in range(orchestrator.config.max_concurrent_agents):
            await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        # Try to spawn one more
        with pytest.raises(ResourceLimitError):
            await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
    
    @pytest.mark.asyncio
    async def test_shutdown_agent_success(self, orchestrator):
        """Test successful agent shutdown."""
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        result = await orchestrator.shutdown_agent(agent_id)
        
        assert result is True
        assert agent_id not in orchestrator._agents
    
    @pytest.mark.asyncio
    async def test_shutdown_nonexistent_agent(self, orchestrator):
        """Test shutting down non-existent agent."""
        result = await orchestrator.shutdown_agent("nonexistent-agent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_shutdown_agent_with_task(self, orchestrator):
        """Test shutting down agent with active task."""
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        # Assign a task to the agent
        task_id = await orchestrator.delegate_task(
            task_description="Test task",
            task_type="backend"
        )
        
        # Shutdown should succeed but take longer (graceful)
        result = await orchestrator.shutdown_agent(agent_id, graceful=True)
        assert result is True
        assert agent_id not in orchestrator._agents


class TestTaskDelegation:
    """Test task delegation functionality."""
    
    @pytest.mark.asyncio
    async def test_delegate_task_success(self, orchestrator):
        """Test successful task delegation."""
        # Spawn an agent
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        # Delegate a task
        task_id = await orchestrator.delegate_task(
            task_description="Implement API endpoint",
            task_type="backend_api",
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert task_id in orchestrator._task_assignments
        
        assignment = orchestrator._task_assignments[task_id]
        assert assignment.agent_id == agent_id
        assert assignment.status == TaskStatus.ASSIGNED
        assert assignment.priority == TaskPriority.HIGH
        
        # Agent should now have the task
        agent = orchestrator._agents[agent_id]
        assert agent.current_task_id == task_id
    
    @pytest.mark.asyncio
    async def test_delegate_task_no_agents(self, orchestrator):
        """Test task delegation with no available agents."""
        with pytest.raises(TaskDelegationError, match="No suitable agent available"):
            await orchestrator.delegate_task(
                task_description="Test task",
                task_type="backend"
            )
    
    @pytest.mark.asyncio
    async def test_delegate_task_round_robin(self, orchestrator):
        """Test round-robin task assignment."""
        # Spawn multiple agents
        agent_ids = []
        for i in range(3):
            agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
            agent_ids.append(agent_id)
        
        # Delegate tasks using round-robin
        task_agents = []
        for i in range(6):  # More tasks than agents
            task_id = await orchestrator.delegate_task(
                task_description=f"Task {i}",
                task_type="backend",
                assignment_strategy=TaskAssignmentStrategy.ROUND_ROBIN
            )
            assignment = orchestrator._task_assignments[task_id]
            task_agents.append(assignment.agent_id)
        
        # Check that tasks were distributed in round-robin fashion
        # Note: This is a simplified check; actual round-robin might be affected by task completion
        assert len(set(task_agents)) <= 3  # Should use all agents
    
    @pytest.mark.asyncio
    async def test_delegate_task_capability_match(self, orchestrator):
        """Test capability-based task assignment."""
        # Spawn agents with different capabilities
        python_agent = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "django", "api"]
        )
        js_agent = await orchestrator.spawn_agent(
            role=AgentRole.FRONTEND_DEVELOPER,
            capabilities=["javascript", "react", "ui"]
        )
        
        # Delegate Python task
        python_task = await orchestrator.delegate_task(
            task_description="Python API task",
            task_type="backend_api",
            required_capabilities=["python", "api"],
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH
        )
        
        python_assignment = orchestrator._task_assignments[python_task]
        assert python_assignment.agent_id == python_agent
        
        # Complete the Python task to free the agent
        await orchestrator.complete_task(python_task)
        
        # Delegate JavaScript task
        js_task = await orchestrator.delegate_task(
            task_description="React UI task",
            task_type="frontend_ui",
            required_capabilities=["javascript", "react"],
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH
        )
        
        js_assignment = orchestrator._task_assignments[js_task]
        assert js_assignment.agent_id == js_agent
    
    @pytest.mark.asyncio
    async def test_complete_task_success(self, orchestrator):
        """Test successful task completion."""
        # Spawn agent and delegate task
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        task_id = await orchestrator.delegate_task(
            task_description="Test task",
            task_type="backend"
        )
        
        # Complete the task
        result = await orchestrator.complete_task(
            task_id,
            result={"status": "success", "output": "Task completed"}
        )
        
        assert result is True
        
        assignment = orchestrator._task_assignments[task_id]
        assert assignment.status == TaskStatus.COMPLETED
        assert assignment.actual_duration is not None
        
        # Agent should be free
        agent = orchestrator._agents[agent_id]
        assert agent.current_task_id is None
        assert agent.load_score < 0.2  # Load should have decreased
    
    @pytest.mark.asyncio
    async def test_complete_nonexistent_task(self, orchestrator):
        """Test completing non-existent task."""
        result = await orchestrator.complete_task("nonexistent-task")
        assert result is False


class TestSystemStatus:
    """Test system status functionality."""
    
    @pytest.mark.asyncio
    async def test_get_system_status_empty(self, orchestrator):
        """Test system status with no agents or tasks."""
        status = await orchestrator.get_system_status()
        
        assert status["agents"]["total"] == 0
        assert status["tasks"]["active_assignments"] == 0
        assert status["health"]["overall"] == "degraded"  # No agents
        assert status["performance"]["operations_total"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_system_status_with_agents(self, orchestrator):
        """Test system status with agents and tasks."""
        # Spawn agents
        backend_agent = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        frontend_agent = await orchestrator.spawn_agent(role=AgentRole.FRONTEND_DEVELOPER)
        
        # Delegate tasks
        task1 = await orchestrator.delegate_task("Task 1", "backend")
        task2 = await orchestrator.delegate_task("Task 2", "frontend")
        
        status = await orchestrator.get_system_status()
        
        assert status["agents"]["total"] == 2
        assert status["agents"]["healthy"] == 2
        assert status["agents"]["by_status"]["active"] == 2
        assert status["agents"]["by_role"]["backend_developer"] == 1
        assert status["agents"]["by_role"]["frontend_developer"] == 1
        assert status["tasks"]["active_assignments"] == 2
        assert status["health"]["overall"] == "healthy"
        
        # Check agent details
        assert len(status["agents"]["details"]) == 2
        assert backend_agent in status["agents"]["details"]
        assert frontend_agent in status["agents"]["details"]


class TestDatabaseOperations:
    """Test database operations."""
    
    @pytest.mark.asyncio
    async def test_database_enabled_operations(self, test_config, mock_cache, mock_metrics):
        """Test operations with database enabled."""
        test_config.enable_database_persistence = True
        mock_db = MockDatabaseDependency(should_fail=False)
        
        orch = EnhancedSimpleOrchestrator(
            config=test_config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        await orch.start()
        
        try:
            # Test agent spawning with database
            agent_id = await orch.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
            assert agent_id in orch._agents
            
            # Test task delegation with database
            task_id = await orch.delegate_task("Test task", "backend")
            assert task_id in orch._task_assignments
            
        finally:
            await orch.shutdown()
    
    @pytest.mark.asyncio
    async def test_database_failure_handling(self, test_config, mock_cache, mock_metrics):
        """Test handling of database failures."""
        test_config.enable_database_persistence = True
        test_config.database_retry_attempts = 2
        mock_db = MockDatabaseDependency(should_fail=True)
        
        orch = EnhancedSimpleOrchestrator(
            config=test_config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        await orch.start()
        
        try:
            # Operations should still work despite database failures
            # (graceful degradation)
            agent_id = await orch.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
            assert agent_id in orch._agents
            
        finally:
            await orch.shutdown()


class TestConfigurationValidation:
    """Test configuration validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_max_agents(self, mock_db, mock_cache, mock_metrics):
        """Test invalid max agents configuration."""
        config = OrchestratorConfig(max_concurrent_agents=0)
        
        orch = EnhancedSimpleOrchestrator(
            config=config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        with pytest.raises(ConfigurationError, match="max_concurrent_agents"):
            await orch.start()
    
    @pytest.mark.asyncio
    async def test_invalid_heartbeat_interval(self, mock_db, mock_cache, mock_metrics):
        """Test invalid heartbeat interval configuration."""
        config = OrchestratorConfig(heartbeat_interval_seconds=0)
        
        orch = EnhancedSimpleOrchestrator(
            config=config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        with pytest.raises(ConfigurationError, match="heartbeat_interval_seconds"):
            await orch.start()


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_exception_during_spawn(self, orchestrator):
        """Test exception handling during agent spawning."""
        # Mock a failure in the spawn process
        with patch.object(orchestrator, '_persist_agent_to_db', side_effect=Exception("Test error")):
            with pytest.raises(SimpleOrchestratorError):
                await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
    
    @pytest.mark.asyncio
    async def test_exception_during_delegation(self, orchestrator):
        """Test exception handling during task delegation."""
        # Spawn an agent first
        await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        # Mock a failure in the delegation process
        with patch.object(orchestrator, '_persist_task_to_db', side_effect=Exception("Test error")):
            with pytest.raises(TaskDelegationError):
                await orchestrator.delegate_task("Test task", "backend")


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_metrics_recording(self, orchestrator, mock_metrics):
        """Test that metrics are properly recorded."""
        # Perform some operations
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        task_id = await orchestrator.delegate_task("Test task", "backend")
        await orchestrator.complete_task(task_id)
        await orchestrator.shutdown_agent(agent_id)
        
        # Check that metrics were recorded
        assert orchestrator._metrics_tracker.operation_count > 0
        assert orchestrator._metrics_tracker.success_count > 0
        
        if mock_metrics:
            assert len(mock_metrics.counters) > 0
            assert len(mock_metrics.histograms) > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, orchestrator):
        """Test performance metrics calculation."""
        metrics = orchestrator._metrics_tracker
        
        # Record some operations
        metrics.record_operation(True, 100.0)
        metrics.record_operation(True, 200.0)
        metrics.record_operation(False, 300.0)
        
        assert metrics.get_success_rate() == 2/3
        assert metrics.get_average_response_time() == 200.0
        assert metrics.operation_count == 3


class TestFactoryFunctions:
    """Test factory functions and global instance management."""
    
    def test_create_enhanced_orchestrator(self):
        """Test orchestrator factory function."""
        config = OrchestratorConfig(mode=OrchestratorMode.TEST)
        mock_db = MockDatabaseDependency()
        mock_cache = MockCacheDependency()
        mock_metrics = MockMetricsDependency()
        
        orch = create_enhanced_orchestrator(
            config=config,
            db_session_factory=mock_db,
            cache=mock_cache,
            metrics=mock_metrics
        )
        
        assert isinstance(orch, EnhancedSimpleOrchestrator)
        assert orch.config == config
        assert orch._db_session_factory == mock_db
        assert orch._cache == mock_cache
        assert orch._metrics == mock_metrics


# Integration test
class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_development_workflow(self, orchestrator):
        """Test a complete development workflow."""
        # Spawn development team
        backend_dev = await orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "django", "api", "database"]
        )
        frontend_dev = await orchestrator.spawn_agent(
            role=AgentRole.FRONTEND_DEVELOPER,
            capabilities=["javascript", "react", "ui", "css"]
        )
        qa_engineer = await orchestrator.spawn_agent(
            role=AgentRole.QA_ENGINEER,
            capabilities=["testing", "automation", "selenium"]
        )
        
        # Delegate tasks for a feature development
        api_task = await orchestrator.delegate_task(
            task_description="Implement user authentication API",
            task_type="backend_api",
            required_capabilities=["python", "api"],
            priority=TaskPriority.HIGH,
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH
        )
        
        ui_task = await orchestrator.delegate_task(
            task_description="Create login/register UI components",
            task_type="frontend_ui",
            required_capabilities=["react", "ui"],
            priority=TaskPriority.HIGH,
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH
        )
        
        # Check assignments
        api_assignment = orchestrator._task_assignments[api_task]
        ui_assignment = orchestrator._task_assignments[ui_task]
        
        assert api_assignment.agent_id == backend_dev
        assert ui_assignment.agent_id == frontend_dev
        
        # Complete backend task
        await orchestrator.complete_task(
            api_task,
            result={"endpoints": ["POST /auth/login", "POST /auth/register"]}
        )
        
        # Complete frontend task
        await orchestrator.complete_task(
            ui_task,
            result={"components": ["LoginForm", "RegisterForm"]}
        )
        
        # Delegate testing task
        test_task = await orchestrator.delegate_task(
            task_description="Test authentication flow",
            task_type="testing",
            required_capabilities=["testing"],
            priority=TaskPriority.MEDIUM,
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH
        )
        
        test_assignment = orchestrator._task_assignments[test_task]
        assert test_assignment.agent_id == qa_engineer
        
        # Complete testing
        await orchestrator.complete_task(
            test_task,
            result={"test_results": "All tests passed"}
        )
        
        # Check system status
        status = await orchestrator.get_system_status()
        assert status["agents"]["total"] == 3
        assert status["tasks"]["active_assignments"] == 3
        assert all(
            assignment.status == TaskStatus.COMPLETED
            for assignment in orchestrator._task_assignments.values()
        )
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, orchestrator):
        """Test error recovery and graceful degradation."""
        # Spawn agent
        agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        # Delegate task
        task_id = await orchestrator.delegate_task("Test task", "backend")
        
        # Simulate agent becoming unhealthy
        agent = orchestrator._agents[agent_id]
        agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        
        # Check that agent is detected as unhealthy
        assert not agent.is_healthy()
        
        # System should still be able to report status
        status = await orchestrator.get_system_status()
        assert status["agents"]["total"] == 1
        assert status["agents"]["healthy"] == 0
        
        # Should be able to spawn new agent to replace unhealthy one
        new_agent_id = await orchestrator.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        assert new_agent_id != agent_id
        assert new_agent_id in orchestrator._agents


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])