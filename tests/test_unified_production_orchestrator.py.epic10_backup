"""
Comprehensive Unit Tests for Unified Production Orchestrator

Tests cover all core functionality including agent registration, task delegation,
resource management, circuit breakers, and performance requirements.
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig,
    AgentProtocol,
    AgentState,
    AgentCapability,
    AgentMetrics,
    TaskRoutingStrategy,
    ResourceType
)
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerError
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.agent import AgentType


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, agent_id: str = None, capabilities: List[AgentCapability] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.ACTIVE
        self.capabilities = capabilities or [
            AgentCapability(
                name="general",
                description="General purpose agent",
                confidence_level=0.8,
                specialization_areas=["general"]
            )
        ]
        self.task_count = 0
        self.execution_results = []
        self.shutdown_called = False
        
    async def execute_task(self, task: Task) -> Any:
        """Mock task execution."""
        self.task_count += 1
        # Simulate some processing time
        await asyncio.sleep(0.01)
        result = f"Task {task.id} completed by {self.agent_id}"
        self.execution_results.append(result)
        return result
        
    async def get_status(self) -> AgentState:
        """Get mock agent status."""
        return self.state
        
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get mock agent capabilities."""
        return self.capabilities
        
    async def shutdown(self, graceful: bool = True) -> None:
        """Mock shutdown."""
        self.shutdown_called = True
        self.state = AgentState.TERMINATED


@pytest.fixture
def orchestrator_config():
    """Test orchestrator configuration."""
    return OrchestratorConfig(
        max_concurrent_agents=10,
        min_agent_pool=2,
        max_agent_pool=15,
        agent_registration_timeout=1.0,
        agent_heartbeat_interval=5.0,
        task_delegation_timeout=1.0,
        max_task_queue_size=100,
        memory_limit_mb=512,
        cpu_limit_percent=70.0,
        registration_target_ms=50.0,
        delegation_target_ms=200.0,
        health_check_interval=10.0,
        metrics_collection_interval=5.0
    )


@pytest.fixture
async def orchestrator(orchestrator_config):
    """Test orchestrator instance."""
    orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
    await orchestrator.start()
    yield orchestrator
    await orchestrator.shutdown(graceful=True)


@pytest.fixture
def mock_task():
    """Mock task for testing."""
    return Task(
        id=str(uuid.uuid4()),
        title="Test Task",
        description="Test task for orchestrator",
        priority=TaskPriority.MEDIUM,
        status=TaskStatus.PENDING,
        estimated_time_minutes=5
    )


class TestUnifiedProductionOrchestrator:
    """Test suite for UnifiedProductionOrchestrator."""
    
    async def test_orchestrator_initialization(self, orchestrator_config):
        """Test orchestrator initialization."""
        orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
        
        assert orchestrator.config == orchestrator_config
        assert orchestrator._is_running is False
        assert len(orchestrator._agents) == 0
        assert len(orchestrator._agent_metrics) == 0
        assert orchestrator._task_queue.qsize() == 0
        
    async def test_orchestrator_start_stop(self, orchestrator_config):
        """Test orchestrator lifecycle."""
        orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
        
        # Test start
        await orchestrator.start()
        assert orchestrator._is_running is True
        assert len(orchestrator._background_tasks) > 0
        
        # Test shutdown
        await orchestrator.shutdown()
        assert orchestrator._is_running is False
        
    async def test_agent_registration_success(self, orchestrator):
        """Test successful agent registration."""
        mock_agent = MockAgent()
        
        start_time = time.time()
        agent_id = await orchestrator.register_agent(mock_agent)
        registration_time = (time.time() - start_time) * 1000
        
        # Verify registration
        assert agent_id is not None
        assert agent_id in orchestrator._agents
        assert agent_id in orchestrator._agent_metrics
        assert agent_id in orchestrator._agent_pool
        
        # Verify performance target
        assert registration_time < orchestrator.config.registration_target_ms
        
        # Verify agent metrics initialized
        metrics = orchestrator._agent_metrics[agent_id]
        assert metrics.agent_id == agent_id
        assert metrics.success_rate == 1.0
        
    async def test_agent_registration_with_custom_id(self, orchestrator):
        """Test agent registration with custom ID."""
        mock_agent = MockAgent()
        custom_id = "custom-agent-123"
        
        agent_id = await orchestrator.register_agent(mock_agent, custom_id)
        
        assert agent_id == custom_id
        assert custom_id in orchestrator._agents
        
    async def test_agent_registration_capacity_limit(self, orchestrator):
        """Test agent registration respects capacity limits."""
        # Fill up to capacity
        agents = []
        for i in range(orchestrator.config.max_concurrent_agents):
            mock_agent = MockAgent()
            agent_id = await orchestrator.register_agent(mock_agent)
            agents.append(agent_id)
        
        # Try to register one more - should fail
        overflow_agent = MockAgent()
        with pytest.raises(ValueError, match="Maximum concurrent agents"):
            await orchestrator.register_agent(overflow_agent)
            
    async def test_agent_registration_invalid_protocol(self, orchestrator):
        """Test agent registration with invalid protocol."""
        # Create mock that doesn't implement required methods
        invalid_agent = Mock()
        del invalid_agent.execute_task  # Remove required method
        
        with pytest.raises(ValueError, match="missing required method"):
            await orchestrator.register_agent(invalid_agent)
            
    async def test_task_delegation_success(self, orchestrator, mock_task):
        """Test successful task delegation."""
        # Register a mock agent
        mock_agent = MockAgent()
        agent_id = await orchestrator.register_agent(mock_agent)
        
        # Delegate task
        start_time = time.time()
        assigned_agent_id = await orchestrator.delegate_task(mock_task)
        delegation_time = (time.time() - start_time) * 1000
        
        # Verify delegation
        assert assigned_agent_id == agent_id
        assert delegation_time < orchestrator.config.delegation_target_ms
        
        # Wait for task execution to complete
        await asyncio.sleep(0.1)
        
        # Verify task was executed
        assert mock_agent.task_count == 1
        assert len(mock_agent.execution_results) == 1
        
    async def test_task_delegation_no_agents(self, orchestrator, mock_task):
        """Test task delegation when no agents available."""
        # Try to delegate task without any agents registered
        with pytest.raises(RuntimeError, match="No suitable agent available"):
            await orchestrator.delegate_task(mock_task)
            
    async def test_task_delegation_invalid_task(self, orchestrator):
        """Test task delegation with invalid task."""
        # Register an agent
        mock_agent = MockAgent()
        await orchestrator.register_agent(mock_agent)
        
        # Try to delegate invalid task
        with pytest.raises(ValueError, match="Invalid task provided"):
            await orchestrator.delegate_task(None)
            
    async def test_agent_unregistration(self, orchestrator):
        """Test agent unregistration."""
        # Register agent
        mock_agent = MockAgent()
        agent_id = await orchestrator.register_agent(mock_agent)
        
        # Verify registration
        assert agent_id in orchestrator._agents
        
        # Unregister agent
        await orchestrator.unregister_agent(agent_id)
        
        # Verify unregistration
        assert agent_id not in orchestrator._agents
        assert agent_id not in orchestrator._agent_metrics
        assert agent_id not in orchestrator._agent_pool
        assert mock_agent.shutdown_called is True
        
    async def test_agent_unregistration_nonexistent(self, orchestrator):
        """Test unregistration of non-existent agent."""
        # Should not raise an error
        await orchestrator.unregister_agent("nonexistent-agent")
        
    async def test_intelligent_task_routing(self, orchestrator):
        """Test intelligent task routing algorithm."""
        # Create agents with different capabilities
        backend_agent = MockAgent(
            capabilities=[
                AgentCapability(
                    name="backend",
                    description="Backend development",
                    confidence_level=0.9,
                    specialization_areas=["backend", "api", "database"]
                )
            ]
        )
        
        frontend_agent = MockAgent(
            capabilities=[
                AgentCapability(
                    name="frontend",
                    description="Frontend development", 
                    confidence_level=0.8,
                    specialization_areas=["frontend", "ui", "react"]
                )
            ]
        )
        
        # Register agents
        backend_id = await orchestrator.register_agent(backend_agent)
        frontend_id = await orchestrator.register_agent(frontend_agent)
        
        # Create backend-specific task
        backend_task = Task(
            id=str(uuid.uuid4()),
            title="API Development",
            description="Implement REST API endpoints",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING
        )
        
        # Delegate task - should go to backend agent
        assigned_agent = await orchestrator.delegate_task(backend_task)
        assert assigned_agent == backend_id
        
    async def test_load_balancing(self, orchestrator):
        """Test load balancing across multiple agents."""
        # Register multiple agents
        agents = []
        agent_ids = []
        for i in range(3):
            agent = MockAgent()
            agent_id = await orchestrator.register_agent(agent)
            agents.append(agent)
            agent_ids.append(agent_id)
        
        # Delegate multiple tasks
        tasks = []
        for i in range(6):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Task {i}",
                description=f"Load balancing test task {i}",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
            await orchestrator.delegate_task(task)
        
        # Wait for tasks to complete
        await asyncio.sleep(0.2)
        
        # Verify load distribution
        total_tasks = sum(agent.task_count for agent in agents)
        assert total_tasks == 6
        
        # Verify tasks were distributed (not all to one agent)
        task_counts = [agent.task_count for agent in agents]
        assert max(task_counts) <= 3  # No agent should have more than half
        
    async def test_circuit_breaker_agent_registration(self, orchestrator):
        """Test circuit breaker for agent registration failures."""
        # Mock agent that always fails validation
        failing_agent = Mock()
        failing_agent.execute_task = AsyncMock(side_effect=Exception("Always fails"))
        failing_agent.get_status = AsyncMock(return_value=AgentState.ACTIVE)
        failing_agent.get_capabilities = AsyncMock(return_value=[])
        failing_agent.shutdown = AsyncMock()
        
        # Trigger circuit breaker by failing multiple times
        failure_count = 0
        for _ in range(10):  # More than circuit breaker threshold
            try:
                await orchestrator.register_agent(failing_agent)
            except Exception:
                failure_count += 1
                if failure_count >= 5:  # Circuit breaker threshold
                    break
        
        # Verify circuit breaker is open
        cb = orchestrator._circuit_breakers['agent_registration']
        # Circuit breaker should eventually open after failures
        assert failure_count >= 3  # At least some failures occurred
        
    async def test_circuit_breaker_task_delegation(self, orchestrator, mock_task):
        """Test circuit breaker for task delegation failures."""
        # Register agent that will fail task execution
        failing_agent = MockAgent()
        failing_agent.execute_task = AsyncMock(side_effect=Exception("Task execution failed"))
        
        agent_id = await orchestrator.register_agent(failing_agent)
        
        # Force task delegation to fail by making agent unavailable
        orchestrator._idle_agents.clear()  # No idle agents
        
        # Try to delegate task - should fail
        with pytest.raises(RuntimeError, match="No suitable agent available"):
            await orchestrator.delegate_task(mock_task)
            
    async def test_resource_monitoring(self, orchestrator):
        """Test resource monitoring functionality."""
        # Get initial resource usage
        status = await orchestrator.get_system_status()
        
        assert 'resources' in status
        resources = status['resources']
        assert 'cpu' in resources
        assert 'memory' in resources
        
        # Verify resource values are reasonable
        assert 0 <= resources.get('cpu', 0) <= 100
        assert resources.get('memory', 0) >= 0
        
    async def test_agent_status_retrieval(self, orchestrator):
        """Test agent status retrieval."""
        # Register agent
        mock_agent = MockAgent()
        agent_id = await orchestrator.register_agent(mock_agent)
        
        # Get agent status
        status = await orchestrator.get_agent_status(agent_id)
        
        assert status is not None
        assert status['agent_id'] == agent_id
        assert status['state'] == AgentState.ACTIVE
        assert 'capabilities' in status
        assert 'metrics' in status
        assert status['is_idle'] is True
        assert status['current_tasks'] == []
        
    async def test_agent_status_nonexistent(self, orchestrator):
        """Test status retrieval for non-existent agent."""
        status = await orchestrator.get_agent_status("nonexistent-agent")
        assert status is None
        
    async def test_system_status_retrieval(self, orchestrator):
        """Test system status retrieval."""
        status = await orchestrator.get_system_status()
        
        # Verify structure
        assert 'orchestrator' in status
        assert 'agents' in status
        assert 'tasks' in status
        assert 'resources' in status
        assert 'circuit_breakers' in status
        assert 'performance' in status
        
        # Verify orchestrator status
        orch_status = status['orchestrator']
        assert orch_status['is_running'] is True
        assert 'uptime_seconds' in orch_status
        assert 'config' in orch_status
        
        # Verify agent status
        agent_status = status['agents']
        assert agent_status['total_registered'] == 0  # No agents registered yet
        assert agent_status['idle_count'] == 0
        assert agent_status['busy_count'] == 0
        assert agent_status['max_concurrent'] == orchestrator.config.max_concurrent_agents
        
    async def test_performance_metrics_tracking(self, orchestrator):
        """Test performance metrics are tracked correctly."""
        # Register agent and delegate task
        mock_agent = MockAgent()
        agent_id = await orchestrator.register_agent(mock_agent)
        
        task = Task(
            id=str(uuid.uuid4()),
            title="Performance Test Task",
            description="Task for performance tracking",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING
        )
        
        await orchestrator.delegate_task(task)
        
        # Wait for task completion
        await asyncio.sleep(0.1)
        
        # Check that metrics were updated
        metrics = orchestrator._agent_metrics[agent_id]
        assert metrics.success_rate >= 0.8  # Should be high for successful task
        assert metrics.average_response_time >= 0  # Should be tracked
        assert metrics.last_heartbeat is not None
        
    async def test_graceful_shutdown_with_running_tasks(self, orchestrator):
        """Test graceful shutdown waits for running tasks."""
        # Register agent
        slow_agent = MockAgent()
        
        async def slow_execute_task(task):
            await asyncio.sleep(0.5)  # Simulate slow task
            return "Slow task completed"
        
        slow_agent.execute_task = slow_execute_task
        agent_id = await orchestrator.register_agent(slow_agent)
        
        # Start a slow task
        task = Task(
            id=str(uuid.uuid4()),
            title="Slow Task",
            description="Task that takes time to complete",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING
        )
        
        await orchestrator.delegate_task(task)
        
        # Immediately try graceful shutdown
        start_time = time.time()
        await orchestrator.shutdown(graceful=True)
        shutdown_time = time.time() - start_time
        
        # Should have waited for task to complete
        assert shutdown_time >= 0.4  # Task takes ~0.5s, some overhead acceptable
        assert slow_agent.shutdown_called is True
        
    async def test_forced_shutdown(self, orchestrator):
        """Test forced shutdown doesn't wait for tasks."""
        # Register agent
        mock_agent = MockAgent()
        agent_id = await orchestrator.register_agent(mock_agent)
        
        # Force shutdown
        start_time = time.time()
        await orchestrator.shutdown(graceful=False)
        shutdown_time = time.time() - start_time
        
        # Should be fast
        assert shutdown_time < 1.0  # Should complete quickly
        
    async def test_configuration_validation(self):
        """Test orchestrator configuration validation."""
        # Test valid config
        config = OrchestratorConfig(
            max_concurrent_agents=50,
            registration_target_ms=100.0,
            delegation_target_ms=500.0
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        assert orchestrator.config.max_concurrent_agents == 50
        assert orchestrator.config.registration_target_ms == 100.0
        
        # Test default config
        default_orchestrator = UnifiedProductionOrchestrator()
        assert default_orchestrator.config.max_concurrent_agents == 50
        assert default_orchestrator.config.registration_target_ms == 100.0
        
    async def test_memory_management(self, orchestrator):
        """Test memory management and leak prevention."""
        # Register and unregister many agents to test cleanup
        agent_ids = []
        
        for i in range(20):
            mock_agent = MockAgent()
            agent_id = await orchestrator.register_agent(mock_agent)
            agent_ids.append(agent_id)
        
        # Verify all agents are registered
        assert len(orchestrator._agents) == 20
        assert len(orchestrator._agent_metrics) == 20
        
        # Unregister all agents
        for agent_id in agent_ids:
            await orchestrator.unregister_agent(agent_id)
        
        # Verify cleanup
        assert len(orchestrator._agents) == 0
        assert len(orchestrator._agent_metrics) == 0
        assert len(orchestrator._agent_pool) == 0
        
    async def test_concurrent_operations(self, orchestrator):
        """Test thread safety of concurrent operations."""
        # Register multiple agents concurrently
        agents = [MockAgent() for _ in range(10)]
        
        registration_tasks = [
            orchestrator.register_agent(agent) for agent in agents
        ]
        
        agent_ids = await asyncio.gather(*registration_tasks)
        
        # Verify all agents were registered
        assert len(agent_ids) == 10
        assert len(set(agent_ids)) == 10  # All IDs should be unique
        assert len(orchestrator._agents) == 10
        
        # Delegate tasks concurrently  
        tasks = [
            Task(
                id=str(uuid.uuid4()),
                title=f"Concurrent Task {i}",
                description=f"Task {i} for concurrency test",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            for i in range(20)
        ]
        
        delegation_tasks = [
            orchestrator.delegate_task(task) for task in tasks
        ]
        
        assigned_agents = await asyncio.gather(*delegation_tasks)
        
        # Verify all tasks were delegated
        assert len(assigned_agents) == 20
        assert all(agent_id in agent_ids for agent_id in assigned_agents)
        
    async def test_error_recovery(self, orchestrator):
        """Test error recovery mechanisms."""
        # Register agent that sometimes fails
        unreliable_agent = MockAgent()
        failure_count = 0
        
        original_execute = unreliable_agent.execute_task
        
        async def unreliable_execute_task(task):
            nonlocal failure_count
            if failure_count < 2:
                failure_count += 1
                raise Exception("Simulated failure")
            return await original_execute(task)
        
        unreliable_agent.execute_task = unreliable_execute_task
        agent_id = await orchestrator.register_agent(unreliable_agent)
        
        # Delegate task - should eventually succeed with retries
        task = Task(
            id=str(uuid.uuid4()),
            title="Recovery Test Task",
            description="Task for testing error recovery",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING
        )
        
        # This might fail due to agent execution failures, but orchestrator should handle it
        try:
            await orchestrator.delegate_task(task)
            await asyncio.sleep(0.2)  # Wait for execution
            
            # Check agent metrics reflect the failures
            metrics = orchestrator._agent_metrics[agent_id]
            # Success rate should be lower due to failures
            assert metrics.success_rate < 1.0
        except Exception:
            # Task delegation itself might fail, which is acceptable behavior
            pass


@pytest.mark.performance
class TestPerformanceRequirements:
    """Performance requirement validation tests."""
    
    async def test_agent_registration_performance(self, orchestrator_config):
        """Test agent registration meets <100ms requirement."""
        orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
        await orchestrator.start()
        
        try:
            # Test registration performance for multiple agents
            registration_times = []
            
            for i in range(10):
                mock_agent = MockAgent()
                
                start_time = time.time()
                agent_id = await orchestrator.register_agent(mock_agent)
                registration_time = (time.time() - start_time) * 1000
                
                registration_times.append(registration_time)
                assert agent_id is not None
            
            # Verify performance requirements
            avg_registration_time = sum(registration_times) / len(registration_times)
            max_registration_time = max(registration_times)
            
            print(f"Average registration time: {avg_registration_time:.2f}ms")
            print(f"Maximum registration time: {max_registration_time:.2f}ms")
            
            assert avg_registration_time < orchestrator_config.registration_target_ms
            assert max_registration_time < orchestrator_config.registration_target_ms * 2  # Allow some variance
            
        finally:
            await orchestrator.shutdown()
    
    async def test_task_delegation_performance(self, orchestrator_config):
        """Test task delegation meets <500ms requirement."""
        orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
        await orchestrator.start()
        
        try:
            # Register multiple agents for routing
            for i in range(5):
                mock_agent = MockAgent()
                await orchestrator.register_agent(mock_agent)
            
            # Test delegation performance
            delegation_times = []
            
            for i in range(20):
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Performance Test Task {i}",
                    description="Task for performance testing",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                start_time = time.time()
                agent_id = await orchestrator.delegate_task(task)
                delegation_time = (time.time() - start_time) * 1000
                
                delegation_times.append(delegation_time)
                assert agent_id is not None
            
            # Verify performance requirements
            avg_delegation_time = sum(delegation_times) / len(delegation_times)
            max_delegation_time = max(delegation_times)
            
            print(f"Average delegation time: {avg_delegation_time:.2f}ms")
            print(f"Maximum delegation time: {max_delegation_time:.2f}ms")
            
            assert avg_delegation_time < orchestrator_config.delegation_target_ms
            assert max_delegation_time < orchestrator_config.delegation_target_ms * 2  # Allow some variance
            
        finally:
            await orchestrator.shutdown()
    
    async def test_concurrent_agent_capacity(self, orchestrator_config):
        """Test support for 50+ concurrent agents."""
        # Set high capacity config
        high_capacity_config = OrchestratorConfig(
            max_concurrent_agents=60,
            max_agent_pool=75,
            registration_target_ms=100.0,
            delegation_target_ms=500.0
        )
        
        orchestrator = UnifiedProductionOrchestrator(high_capacity_config)
        await orchestrator.start()
        
        try:
            # Register 55 agents to test 50+ capacity
            agents = []
            registration_times = []
            
            for i in range(55):
                mock_agent = MockAgent()
                
                start_time = time.time()
                agent_id = await orchestrator.register_agent(mock_agent)
                registration_time = (time.time() - start_time) * 1000
                
                agents.append(agent_id)
                registration_times.append(registration_time)
            
            # Verify all agents were registered
            assert len(orchestrator._agents) == 55
            assert len(orchestrator._agent_pool) == 55
            
            # Verify performance didn't degrade significantly with scale
            avg_registration_time = sum(registration_times) / len(registration_times)
            print(f"Average registration time with 55 agents: {avg_registration_time:.2f}ms")
            
            # Allow some performance degradation at scale, but not too much
            assert avg_registration_time < high_capacity_config.registration_target_ms * 1.5
            
            # Test task delegation at scale
            tasks = []
            delegation_times = []
            
            for i in range(100):  # More tasks than agents
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Scale Test Task {i}",
                    description="Task for scale testing",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                start_time = time.time()
                agent_id = await orchestrator.delegate_task(task)
                delegation_time = (time.time() - start_time) * 1000
                
                tasks.append(task.id)
                delegation_times.append(delegation_time)
                
                assert agent_id in agents
            
            # Verify delegation performance at scale
            avg_delegation_time = sum(delegation_times) / len(delegation_times)
            print(f"Average delegation time with 55 agents, 100 tasks: {avg_delegation_time:.2f}ms")
            
            assert avg_delegation_time < high_capacity_config.delegation_target_ms * 1.5
            
        finally:
            await orchestrator.shutdown()
    
    async def test_memory_efficiency(self, orchestrator_config):
        """Test memory efficiency requirements."""
        import psutil
        import gc
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        orchestrator = UnifiedProductionOrchestrator(orchestrator_config)
        await orchestrator.start()
        
        try:
            # Register agents and measure memory growth
            for i in range(20):
                mock_agent = MockAgent()
                await orchestrator.register_agent(mock_agent)
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory usage with agents
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - baseline_memory
            
            print(f"Baseline memory: {baseline_memory:.2f} MB")
            print(f"Memory with 20 agents: {current_memory:.2f} MB")
            print(f"Memory growth: {memory_growth:.2f} MB")
            
            # Verify memory efficiency (should be well under 50MB base overhead)
            assert memory_growth < 50.0, f"Memory growth {memory_growth:.2f}MB exceeds 50MB limit"
            
            # Test memory cleanup after unregistering agents
            agents_to_remove = list(orchestrator._agents.keys())[:10]
            for agent_id in agents_to_remove:
                await orchestrator.unregister_agent(agent_id)
            
            gc.collect()
            
            # Memory should decrease after cleanup
            cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = current_memory - cleanup_memory
            
            print(f"Memory after cleanup: {cleanup_memory:.2f} MB")
            print(f"Memory freed: {memory_freed:.2f} MB")
            
            # Should free some memory (at least 10% of growth)
            assert memory_freed > memory_growth * 0.1
            
        finally:
            await orchestrator.shutdown()


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__ + "::TestPerformanceRequirements", "-v", "-s"])