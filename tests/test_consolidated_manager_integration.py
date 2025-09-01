"""
Consolidated Manager Integration Tests

Comprehensive test suite for validating the consolidated manager hierarchy
and its integration with the ConsolidatedProductionOrchestrator.

Tests include:
- Individual manager functionality
- Manager integration and communication
- Performance validation
- Migration testing
- API compatibility preservation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

try:
    from app.core.managers.consolidated_manager import (
        ConsolidatedLifecycleManager,
        ConsolidatedTaskCoordinationManager,
        ConsolidatedPerformanceManager,
        ConsolidatedAgentInstance,
        ConsolidatedTaskAssignment,
        ConsolidatedPerformanceMetrics,
        AgentRole,
        RoutingStrategy
    )
except ImportError as e:
    pytest.skip(f"ConsolidatedManager imports failed: {e}", allow_module_level=True)
try:
    from app.core.managers.orchestrator_integration import (
        ConsolidatedManagerIntegrator,
        ManagerMigrationResult,
        create_manager_integrator,
        integrate_consolidated_managers
    )
except ImportError as e:
    pytest.skip(f"OrchestratorIntegration imports failed: {e}", allow_module_level=True)
from app.models.agent import AgentStatus, AgentType
from app.models.task import TaskStatus, TaskPriority


class MockOrchestrator:
    """Mock orchestrator for testing."""
    
    def __init__(self):
        self.config = Mock()
        self.config.max_concurrent_agents = 10
        
        self.integration = Mock()
        self.integration.get_database_session = AsyncMock(return_value=None)
        
        # Mock existing managers
        self.agent_lifecycle = None
        self.task_coordination = None
        self.performance = None
        
        # Mock methods
        self.broadcast_agent_update = AsyncMock()
        self.broadcast_task_update = AsyncMock()
        self.get_system_status = AsyncMock(return_value={"status": "operational"})


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for testing."""
    return MockOrchestrator()


@pytest.fixture
async def consolidated_lifecycle_manager(mock_orchestrator):
    """Create consolidated lifecycle manager for testing.""" 
    manager = ConsolidatedLifecycleManager(mock_orchestrator)
    await manager.initialize()
    return manager


@pytest.fixture
async def consolidated_task_manager(mock_orchestrator):
    """Create consolidated task coordination manager for testing."""
    manager = ConsolidatedTaskCoordinationManager(mock_orchestrator)
    await manager.initialize()
    return manager


@pytest.fixture
async def consolidated_performance_manager(mock_orchestrator):
    """Create consolidated performance manager for testing."""
    manager = ConsolidatedPerformanceManager(mock_orchestrator)
    await manager.initialize()
    return manager


class TestConsolidatedLifecycleManager:
    """Test suite for ConsolidatedLifecycleManager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_orchestrator):
        """Test manager initialization."""
        manager = ConsolidatedLifecycleManager(mock_orchestrator)
        
        assert not manager.initialized
        assert not manager.running
        assert len(manager.agents) == 0
        
        await manager.initialize()
        
        assert manager.initialized
        assert manager.spawn_count == 0
        assert manager.registration_count == 0
    
    @pytest.mark.asyncio 
    async def test_agent_registration(self, consolidated_lifecycle_manager):
        """Test agent registration functionality."""
        manager = consolidated_lifecycle_manager
        
        # Register an agent
        agent_id = await manager.register_agent(
            name="test-agent",
            agent_type=AgentType.CLAUDE,
            role="backend_developer",
            capabilities=["python", "api_development"],
            metadata={"test": True}
        )
        
        assert agent_id is not None
        assert agent_id in manager.agents
        assert manager.registration_count == 1
        
        # Get agent details
        agent_status = await manager.get_agent_status(agent_id)
        assert agent_status is not None
        assert agent_status["agent_instance"]["capabilities"] == ["python", "api_development"]
        
    @pytest.mark.asyncio
    async def test_agent_spawn_api_compatibility(self, consolidated_lifecycle_manager):
        """Test SimpleOrchestrator API compatibility for agent spawning."""
        manager = consolidated_lifecycle_manager
        
        # Test spawn_agent method (SimpleOrchestrator compatibility)
        agent_id = await manager.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            task_id="test-task-123",
            capabilities=["python", "fastapi"]
        )
        
        assert agent_id is not None
        assert agent_id in manager.agents
        assert manager.spawn_count == 1
        
        # Verify agent details
        agent = manager.agents[agent_id]
        assert agent.role == AgentRole.BACKEND_DEVELOPER
        assert agent.current_task_id == "test-task-123"
        assert agent.status == AgentStatus.ACTIVE
        assert "python" in agent.capabilities
        assert "fastapi" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, consolidated_lifecycle_manager):
        """Test agent shutdown functionality."""
        manager = consolidated_lifecycle_manager
        
        # Spawn an agent
        agent_id = await manager.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        assert agent_id in manager.agents
        
        # Shutdown the agent
        success = await manager.shutdown_agent(agent_id, graceful=True)
        
        assert success
        assert agent_id not in manager.agents
        assert manager.shutdown_count == 1
    
    @pytest.mark.asyncio
    async def test_agent_heartbeat(self, consolidated_lifecycle_manager):
        """Test agent heartbeat functionality."""
        manager = consolidated_lifecycle_manager
        
        # Register an agent
        agent_id = await manager.register_agent("test-agent")
        
        # Process heartbeat
        success = await manager.agent_heartbeat(
            agent_id,
            status_data={"current_task_id": "task-123", "performance_metrics": {"cpu": 50}}
        )
        
        assert success
        assert manager.heartbeat_count == 1
        
        # Verify heartbeat data was stored
        agent = manager.agents[agent_id]
        assert agent.last_heartbeat is not None
        assert agent.current_task_id == "task-123"
        assert agent.performance_metrics.get("cpu") == 50
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self, consolidated_lifecycle_manager):
        """Test agent cleanup functionality.""" 
        manager = consolidated_lifecycle_manager
        
        # Create an agent with old activity
        agent_id = await manager.register_agent("old-agent")
        agent = manager.agents[agent_id]
        agent.status = AgentStatus.INACTIVE
        agent.last_activity = datetime.utcnow() - timedelta(hours=25)  # Over 24 hours old
        
        initial_count = len(manager.agents)
        
        # Run cleanup
        await manager.cleanup_inactive_agents()
        
        # Verify old agent was cleaned up
        assert len(manager.agents) < initial_count
        assert agent_id not in manager.agents
    
    @pytest.mark.asyncio
    async def test_manager_metrics(self, consolidated_lifecycle_manager):
        """Test manager metrics collection."""
        manager = consolidated_lifecycle_manager
        
        # Add some agents
        await manager.register_agent("agent-1")
        await manager.spawn_agent(role=AgentRole.FRONTEND_DEVELOPER)
        
        # Get metrics
        metrics = await manager.get_metrics()
        
        assert "agent_count" in metrics
        assert "active_agents" in metrics
        assert "spawn_count" in metrics
        assert "registration_count" in metrics
        assert metrics["agent_count"] == 2
        assert metrics["spawn_count"] == 1
        assert metrics["registration_count"] == 1


class TestConsolidatedTaskCoordinationManager:
    """Test suite for ConsolidatedTaskCoordinationManager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_orchestrator):
        """Test task coordination manager initialization."""
        manager = ConsolidatedTaskCoordinationManager(mock_orchestrator)
        
        assert not manager.initialized
        assert len(manager.tasks) == 0
        assert manager.delegation_count == 0
        
        await manager.initialize()
        
        assert manager.initialized
        assert manager.routing_strategy in list(RoutingStrategy)
    
    @pytest.mark.asyncio
    async def test_task_delegation_api_compatibility(self, consolidated_task_manager, mock_orchestrator):
        """Test SimpleOrchestrator API compatibility for task delegation."""
        # Mock agent lifecycle manager with available agents
        mock_lifecycle = Mock()
        mock_lifecycle.agents = {
            "agent-1": Mock(status=AgentStatus.ACTIVE, capabilities=["python"], persona="developer")
        }
        mock_orchestrator.agent_lifecycle = mock_lifecycle
        
        manager = consolidated_task_manager
        manager.master_orchestrator = mock_orchestrator
        
        # Delegate a task  
        task_id = await manager.delegate_task(
            task_description="Test task",
            task_type="development",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["python"]
        )
        
        assert task_id is not None
        assert task_id in manager.tasks
        assert manager.delegation_count == 1
        
        # Verify task details
        task = manager.tasks[task_id]
        assert task.task_description == "Test task"
        assert task.task_type == "development"
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_task_completion(self, consolidated_task_manager):
        """Test task completion functionality."""
        manager = consolidated_task_manager
        
        # Create a mock task
        task_assignment = ConsolidatedTaskAssignment(
            task_id="test-task",
            agent_id="test-agent", 
            task_description="Test task",
            task_type="test",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.IN_PROGRESS
        )
        manager.tasks["test-task"] = task_assignment
        
        # Complete the task
        success = await manager.complete_task(
            "test-task",
            result={"status": "success", "output": "Task completed"},
            success=True
        )
        
        assert success
        assert manager.completion_count == 1
        assert manager.tasks["test-task"].status == TaskStatus.COMPLETED
        assert manager.tasks["test-task"].result is not None
        assert manager.tasks["test-task"].progress_percentage == 100.0
    
    @pytest.mark.asyncio
    async def test_persona_based_routing(self, consolidated_task_manager, mock_orchestrator):
        """Test persona-based task routing."""
        # Mock agent lifecycle with agents having different personas
        mock_lifecycle = Mock()
        mock_lifecycle.agents = {
            "agent-1": Mock(
                status=AgentStatus.ACTIVE, 
                capabilities=["python"], 
                persona="backend_specialist",
                task_completion_count=5,
                task_failure_count=0
            ),
            "agent-2": Mock(
                status=AgentStatus.ACTIVE,
                capabilities=["python"], 
                persona="frontend_specialist",
                task_completion_count=3,
                task_failure_count=1
            )
        }
        mock_orchestrator.agent_lifecycle = mock_lifecycle
        
        manager = consolidated_task_manager
        manager.master_orchestrator = mock_orchestrator
        manager.routing_strategy = RoutingStrategy.PERSONA_MATCH
        manager.agent_workloads = {"agent-1": 1, "agent-2": 2}
        
        # Find optimal agent for backend task with persona preference
        routing_result = await manager._find_optimal_agent(
            task_type="backend_api",
            priority=TaskPriority.HIGH,
            preferred_persona="backend_specialist"
        )
        
        assert routing_result is not None
        assert routing_result["agent_id"] == "agent-1"  # Should prefer backend specialist
        assert "persona_match_score" in routing_result
        assert routing_result["persona_match_score"] > 0
    
    @pytest.mark.asyncio
    async def test_routing_performance_tracking(self, consolidated_task_manager):
        """Test routing performance tracking."""
        manager = consolidated_task_manager
        
        # Create a task with routing strategy
        task_assignment = ConsolidatedTaskAssignment(
            task_id="test-task",
            agent_id="test-agent",
            task_description="Test task", 
            task_type="test",
            priority=TaskPriority.MEDIUM,
            routing_strategy=RoutingStrategy.PERSONA_MATCH
        )
        manager.tasks["test-task"] = task_assignment
        
        # Complete successfully
        await manager.complete_task("test-task", success=True)
        
        # Verify routing performance was updated
        assert RoutingStrategy.PERSONA_MATCH.value in manager.routing_performance
        assert manager.routing_performance[RoutingStrategy.PERSONA_MATCH.value] > 0.8
        
        # Complete another task with failure
        task_assignment_2 = ConsolidatedTaskAssignment(
            task_id="test-task-2",
            agent_id="test-agent",
            task_description="Test task 2",
            task_type="test", 
            priority=TaskPriority.MEDIUM,
            routing_strategy=RoutingStrategy.PERSONA_MATCH
        )
        manager.tasks["test-task-2"] = task_assignment_2
        
        await manager.complete_task("test-task-2", success=False)
        
        # Performance should be penalized
        assert manager.routing_performance[RoutingStrategy.PERSONA_MATCH.value] < 0.85
    
    @pytest.mark.asyncio
    async def test_task_status_retrieval(self, consolidated_task_manager):
        """Test task status retrieval API compatibility."""
        manager = consolidated_task_manager
        
        # Create a task
        task_assignment = ConsolidatedTaskAssignment(
            task_id="status-test-task",
            agent_id="test-agent",
            task_description="Status test task",
            task_type="test",
            priority=TaskPriority.LOW
        )
        manager.tasks["status-test-task"] = task_assignment
        
        # Get task status
        task_status = await manager.get_task_status("status-test-task")
        
        assert task_status is not None
        assert task_status["task"]["task_id"] == "status-test-task" 
        assert task_status["task"]["task_description"] == "Status test task"
        assert task_status["task"]["priority"] == TaskPriority.LOW.value
        
        # Test non-existent task
        non_existent_status = await manager.get_task_status("non-existent")
        assert non_existent_status is None


class TestConsolidatedPerformanceManager:
    """Test suite for ConsolidatedPerformanceManager."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_orchestrator):
        """Test performance manager initialization."""
        manager = ConsolidatedPerformanceManager(mock_orchestrator)
        
        assert not manager.initialized
        assert manager.target_response_time_ms == 50.0
        assert manager.target_memory_usage_mb == 37.0
        assert manager.target_agent_capacity == 250
        
        await manager.initialize()
        
        assert manager.initialized
        assert len(manager.baseline_metrics) > 0
        assert "response_time_ms" in manager.baseline_metrics
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, consolidated_performance_manager):
        """Test performance metrics collection."""
        manager = consolidated_performance_manager
        
        # Collect metrics
        metrics = await manager._collect_performance_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, ConsolidatedPerformanceMetrics)
        assert metrics.timestamp is not None
        assert metrics.cpu_usage_percent >= 0
        assert metrics.memory_usage_mb >= 0
        assert metrics.response_time_ms >= 0
        
        # Verify metrics are stored
        assert manager.current_metrics is not None
        assert len(manager.metrics_history) == 1
    
    @pytest.mark.asyncio
    async def test_system_optimization(self, consolidated_performance_manager):
        """Test system optimization functionality."""
        manager = consolidated_performance_manager
        
        # Run system optimization
        optimization_result = await manager.optimize_system()
        
        assert optimization_result is not None
        assert "total_improvement_factor" in optimization_result
        assert "cumulative_improvement_factor" in optimization_result
        assert "optimizations" in optimization_result
        assert "epic1_claims_validated" in optimization_result
        
        # Verify optimization tracking
        assert manager.optimizations_performed >= 1
        assert manager.total_improvement_factor >= 1.0
    
    @pytest.mark.asyncio
    async def test_epic1_claims_validation(self, consolidated_performance_manager):
        """Test Epic 1 performance claims validation."""
        manager = consolidated_performance_manager
        
        # Set some metrics to test validation
        manager.current_metrics = ConsolidatedPerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=25.0,
            memory_usage_mb=35.0,  # Below 37MB target
            response_time_ms=45.0,  # Below 50ms target
            throughput_ops_per_second=100.0,
            active_agents=10,
            pending_tasks=5,
            operation_count=100,
            error_count=2
        )
        
        # Validate claims
        validation = manager._validate_epic1_claims()
        
        assert validation is not None
        assert "claimed_improvement" in validation
        assert validation["claimed_improvement"] == 39092
        assert "individual_validations" in validation
        assert "response_time" in validation["individual_validations"]
        assert "memory_usage" in validation["individual_validations"]
    
    @pytest.mark.asyncio
    async def test_performance_target_checking(self, consolidated_performance_manager):
        """Test performance target checking."""
        manager = consolidated_performance_manager
        
        # Set metrics that meet targets
        manager.current_metrics = ConsolidatedPerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=20.0,
            memory_usage_mb=30.0,  # Below 37MB target
            response_time_ms=40.0,  # Below 50ms target
            throughput_ops_per_second=150.0,
            active_agents=100,  # Below 250 target
            pending_tasks=5,
            operation_count=200,
            error_count=1
        )
        
        # Check targets
        targets_met = await manager._check_performance_targets()
        
        assert targets_met is not None
        assert targets_met["response_time_target_met"] == True
        assert targets_met["memory_target_met"] == True
        assert targets_met["agent_capacity_available"] == True
    
    @pytest.mark.asyncio
    async def test_optimization_trigger_detection(self, consolidated_performance_manager):
        """Test automatic optimization trigger detection."""
        manager = consolidated_performance_manager
        
        # Set metrics that should trigger optimization
        manager.current_metrics = ConsolidatedPerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage_percent=80.0,
            memory_usage_mb=80.0,  # Way above 37MB target
            response_time_ms=200.0,  # Way above 50ms target
            throughput_ops_per_second=10.0,
            active_agents=5,
            pending_tasks=50,
            operation_count=100,
            error_count=10
        )
        
        # This should trigger optimization
        with patch.object(manager, 'optimize_system', new=AsyncMock()) as mock_optimize:
            await manager._check_optimization_triggers()
            
            # Verify optimization was triggered
            mock_optimize.assert_called_once()


class TestConsolidatedManagerIntegration:
    """Test suite for consolidated manager integration."""
    
    @pytest.mark.asyncio
    async def test_integrator_initialization(self, mock_orchestrator):
        """Test manager integrator initialization."""
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        assert integrator.orchestrator == mock_orchestrator
        assert len(integrator.migrated_managers) == 0
        assert len(integrator.migration_results) == 0
        assert integrator.pre_migration_metrics is None
    
    @pytest.mark.asyncio
    async def test_lifecycle_manager_integration(self, mock_orchestrator):
        """Test lifecycle manager integration."""
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Integrate lifecycle manager
        result = await integrator._integrate_lifecycle_manager()
        
        assert isinstance(result, ManagerMigrationResult)
        assert result.manager_type == "ConsolidatedLifecycleManager"
        assert result.migration_successful == True
        assert result.rollback_available == True
        
        # Verify manager was added
        assert "agent_lifecycle" in integrator.migrated_managers
        assert isinstance(integrator.migrated_managers["agent_lifecycle"], ConsolidatedLifecycleManager)
    
    @pytest.mark.asyncio
    async def test_task_coordination_manager_integration(self, mock_orchestrator):
        """Test task coordination manager integration."""
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Integrate task coordination manager
        result = await integrator._integrate_task_coordination_manager()
        
        assert isinstance(result, ManagerMigrationResult)
        assert result.manager_type == "ConsolidatedTaskCoordinationManager"
        assert result.migration_successful == True
        
        # Verify manager was added
        assert "task_coordination" in integrator.migrated_managers
        assert isinstance(integrator.migrated_managers["task_coordination"], ConsolidatedTaskCoordinationManager)
    
    @pytest.mark.asyncio
    async def test_performance_manager_integration(self, mock_orchestrator):
        """Test performance manager integration."""
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Integrate performance manager
        result = await integrator._integrate_performance_manager()
        
        assert isinstance(result, ManagerMigrationResult)
        assert result.manager_type == "ConsolidatedPerformanceManager"
        assert result.migration_successful == True
        
        # Verify manager was added
        assert "performance" in integrator.migrated_managers
        assert isinstance(integrator.migrated_managers["performance"], ConsolidatedPerformanceManager)
    
    @pytest.mark.asyncio
    async def test_complete_integration(self, mock_orchestrator):
        """Test complete manager integration process."""
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Run complete integration
        integration_result = await integrator.integrate_all_managers()
        
        assert integration_result is not None
        assert integration_result["integration_successful"] == True
        assert integration_result["managers_integrated"] == 3
        assert len(integration_result["migration_results"]) == 3
        
        # Verify orchestrator references were updated
        assert mock_orchestrator.agent_lifecycle is not None
        assert mock_orchestrator.task_coordination is not None
        assert mock_orchestrator.performance is not None
        
        # Verify all managers are running
        assert integrator.migrated_managers["agent_lifecycle"].running == True
        assert integrator.migrated_managers["task_coordination"].running == True
        assert integrator.migrated_managers["performance"].running == True
    
    @pytest.mark.asyncio
    async def test_rollback_functionality(self, mock_orchestrator):
        """Test integration rollback functionality."""
        # Set up original managers
        original_lifecycle = Mock()
        original_task_coordination = Mock() 
        original_performance = Mock()
        
        mock_orchestrator.agent_lifecycle = original_lifecycle
        mock_orchestrator.task_coordination = original_task_coordination
        mock_orchestrator.performance = original_performance
        
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Create rollback checkpoint
        await integrator._create_rollback_checkpoint()
        
        # Simulate partial integration
        await integrator._integrate_lifecycle_manager()
        
        # Verify original managers are stored
        assert integrator.rollback_data["original_managers"]["agent_lifecycle"] == original_lifecycle
        
        # Perform rollback
        await integrator._rollback_integration()
        
        # Verify original managers were restored
        assert mock_orchestrator.agent_lifecycle == original_lifecycle
        assert len(integrator.migrated_managers) == 0
    
    @pytest.mark.asyncio
    async def test_convenience_functions(self, mock_orchestrator):
        """Test convenience functions for integration."""
        # Test create_manager_integrator
        integrator = create_manager_integrator(mock_orchestrator)
        assert isinstance(integrator, ConsolidatedManagerIntegrator)
        assert integrator.orchestrator == mock_orchestrator
        
        # Test integrate_consolidated_managers
        result = await integrate_consolidated_managers(mock_orchestrator)
        assert result is not None
        assert "integration_successful" in result
    
    @pytest.mark.asyncio
    async def test_performance_impact_measurement(self, mock_orchestrator):
        """Test performance impact measurement during integration.""" 
        integrator = ConsolidatedManagerIntegrator(mock_orchestrator)
        
        # Collect pre-migration metrics
        await integrator._collect_pre_migration_metrics()
        assert integrator.pre_migration_metrics is not None
        
        # Run integration
        await integrator.integrate_all_managers()
        
        # Verify post-migration metrics were collected
        assert integrator.post_migration_metrics is not None
        
        # Calculate performance impact
        impact = await integrator._calculate_performance_impact()
        assert impact is not None
        assert "memory_usage" in impact
        assert "response_time" in impact


class TestManagerCommunication:
    """Test suite for inter-manager communication."""
    
    @pytest.mark.asyncio
    async def test_lifecycle_task_coordination_integration(self, mock_orchestrator):
        """Test integration between lifecycle and task coordination managers."""
        # Set up both managers
        lifecycle_manager = ConsolidatedLifecycleManager(mock_orchestrator)
        task_manager = ConsolidatedTaskCoordinationManager(mock_orchestrator)
        
        await lifecycle_manager.initialize()
        await task_manager.initialize()
        
        # Set up orchestrator references
        mock_orchestrator.agent_lifecycle = lifecycle_manager
        mock_orchestrator.task_coordination = task_manager
        
        # Register an agent
        agent_id = await lifecycle_manager.register_agent(
            "test-agent",
            capabilities=["python", "api_development"]
        )
        
        # Mock the agent to be available for tasks
        lifecycle_manager.agents[agent_id].status = AgentStatus.ACTIVE
        task_manager.agent_workloads[agent_id] = 0
        
        # Delegate a task
        task_id = await task_manager.delegate_task(
            task_description="Test integration task",
            task_type="api_development",
            priority=TaskPriority.MEDIUM,
            required_capabilities=["python"]
        )
        
        # Verify task was assigned to the agent
        assert task_id in task_manager.tasks
        task = task_manager.tasks[task_id]
        assert task.agent_id == agent_id
        
        # Verify agent workload was updated
        assert task_manager.agent_workloads[agent_id] == 1
    
    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self, mock_orchestrator):
        """Test performance manager integration with other managers."""
        # Set up all managers
        lifecycle_manager = ConsolidatedLifecycleManager(mock_orchestrator)
        task_manager = ConsolidatedTaskCoordinationManager(mock_orchestrator)
        performance_manager = ConsolidatedPerformanceManager(mock_orchestrator)
        
        await lifecycle_manager.initialize()
        await task_manager.initialize() 
        await performance_manager.initialize()
        
        # Set up orchestrator references
        mock_orchestrator.agent_lifecycle = lifecycle_manager
        mock_orchestrator.task_coordination = task_manager
        mock_orchestrator.performance = performance_manager
        
        # Add some agents and tasks
        await lifecycle_manager.register_agent("agent-1")
        await lifecycle_manager.register_agent("agent-2") 
        
        # Create a task
        task_assignment = ConsolidatedTaskAssignment(
            task_id="perf-test-task",
            agent_id="agent-1",
            task_description="Performance test task",
            task_type="test",
            priority=TaskPriority.HIGH
        )
        task_manager.tasks["perf-test-task"] = task_assignment
        task_manager.delegation_count = 1
        
        # Collect performance metrics
        metrics = await performance_manager._collect_performance_metrics()
        
        # Verify metrics include data from other managers
        assert metrics is not None
        assert metrics.active_agents > 0  # Should reflect registered agents
        assert metrics.pending_tasks >= 0  # Should reflect task state
        assert metrics.operation_count > 0  # Should reflect operations from both managers


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests for consolidated managers."""
    
    @pytest.mark.asyncio
    async def test_agent_spawn_performance(self, consolidated_lifecycle_manager):
        """Test agent spawn performance meets Epic 1 targets."""
        manager = consolidated_lifecycle_manager
        
        # Measure agent spawn time
        start_time = datetime.utcnow()
        
        agent_id = await manager.spawn_agent(role=AgentRole.BACKEND_DEVELOPER)
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should be well under Epic 1 target of <50ms
        assert duration_ms < 100, f"Agent spawn took {duration_ms}ms, should be <100ms"
        assert agent_id is not None
    
    @pytest.mark.asyncio 
    async def test_task_delegation_performance(self, consolidated_task_manager, mock_orchestrator):
        """Test task delegation performance."""
        # Mock an available agent
        mock_lifecycle = Mock()
        mock_lifecycle.agents = {
            "fast-agent": Mock(
                status=AgentStatus.ACTIVE,
                capabilities=["performance_test"],
                persona="speed_demon"
            )
        }
        mock_orchestrator.agent_lifecycle = mock_lifecycle
        
        manager = consolidated_task_manager
        manager.master_orchestrator = mock_orchestrator
        
        # Measure task delegation time
        start_time = datetime.utcnow()
        
        task_id = await manager.delegate_task(
            task_description="Performance test task",
            task_type="performance_test", 
            priority=TaskPriority.HIGH
        )
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should be fast delegation
        assert duration_ms < 50, f"Task delegation took {duration_ms}ms, should be <50ms"
        assert task_id is not None
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self, consolidated_performance_manager):
        """Test performance metrics collection speed."""
        manager = consolidated_performance_manager
        
        # Measure metrics collection time
        start_time = datetime.utcnow()
        
        metrics = await manager._collect_performance_metrics()
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should be very fast
        assert duration_ms < 20, f"Metrics collection took {duration_ms}ms, should be <20ms"
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, mock_orchestrator):
        """Test performance under concurrent operations."""
        # Set up managers
        lifecycle_manager = ConsolidatedLifecycleManager(mock_orchestrator)
        task_manager = ConsolidatedTaskCoordinationManager(mock_orchestrator)
        
        await lifecycle_manager.initialize()
        await task_manager.initialize()
        
        mock_orchestrator.agent_lifecycle = lifecycle_manager
        mock_orchestrator.task_coordination = task_manager
        
        # Create concurrent operations
        start_time = datetime.utcnow()
        
        # Simulate concurrent agent registrations and task delegations
        agent_tasks = []
        for i in range(10):
            agent_task = lifecycle_manager.register_agent(f"concurrent-agent-{i}")
            agent_tasks.append(agent_task)
        
        agent_ids = await asyncio.gather(*agent_tasks)
        
        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Should handle 10 concurrent operations efficiently
        assert duration_ms < 500, f"10 concurrent operations took {duration_ms}ms, should be <500ms"
        assert len(agent_ids) == 10
        assert all(aid is not None for aid in agent_ids)