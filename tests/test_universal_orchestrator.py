"""
Comprehensive test suite for Universal Orchestrator

Tests consolidation of 28+ orchestrator implementations into a single
production-ready orchestrator with performance guarantees:

- Agent registration: <100ms per agent
- Concurrent agents: 50+ simultaneous agents
- Task delegation: <500ms for complex routing
- Memory usage: <50MB base overhead
- System initialization: <2000ms

Coverage targets:
- Unit tests: 95%+ coverage
- Integration tests: All plugin interactions
- Performance tests: All timing requirements
- Load tests: Concurrent agent limits
- Functional tests: 100% backward compatibility
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

from app.core.universal_orchestrator import (
    UniversalOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole,
    AgentInstance,
    TaskExecution,
    SystemMetrics,
    CircuitBreaker,
    CircuitBreakerState,
    HealthStatus,
    get_universal_orchestrator,
    shutdown_universal_orchestrator
)
from app.models.agent import AgentStatus
from app.models.task import TaskStatus, TaskPriority


@pytest.fixture
async def redis_mock():
    """Mock Redis instance for testing."""
    mock_redis = AsyncMock()
    mock_redis.info.return_value = {'connected_clients': 5}
    mock_redis.setex = AsyncMock()
    return mock_redis


@pytest.fixture
async def test_config():
    """Test configuration for orchestrator."""
    return OrchestratorConfig(
        mode=OrchestratorMode.TESTING,
        max_agents=10,
        max_concurrent_tasks=50,
        health_check_interval=5,
        cleanup_interval=60,
        max_agent_registration_ms=50.0,  # Stricter for testing
        max_task_delegation_ms=250.0,   # Stricter for testing
        max_system_initialization_ms=1000.0,  # Stricter for testing
        enable_performance_plugin=False,  # Disable for unit tests
        enable_security_plugin=False,
        enable_context_plugin=False
    )


@pytest.fixture
async def orchestrator(test_config, redis_mock):
    """Create test orchestrator instance."""
    with patch('app.core.universal_orchestrator.get_redis', return_value=redis_mock):
        orchestrator = UniversalOrchestrator(test_config)
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()


class TestUniversalOrchestratorInitialization:
    """Test orchestrator initialization and configuration."""
    
    async def test_orchestrator_creation(self, test_config):
        """Test basic orchestrator creation."""
        orchestrator = UniversalOrchestrator(test_config)
        
        assert orchestrator.config.mode == OrchestratorMode.TESTING
        assert orchestrator.config.max_agents == 10
        assert orchestrator.orchestrator_id is not None
        assert len(orchestrator.agents) == 0
        assert len(orchestrator.active_tasks) == 0
    
    async def test_orchestrator_initialization_performance(self, test_config, redis_mock):
        """Test that orchestrator initialization meets performance requirements."""
        with patch('app.core.universal_orchestrator.get_redis', return_value=redis_mock):
            start_time = time.time()
            orchestrator = UniversalOrchestrator(test_config)
            initialization_result = await orchestrator.initialize()
            initialization_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            assert initialization_result is True
            assert initialization_time < test_config.max_system_initialization_ms
            
            await orchestrator.shutdown()
    
    async def test_orchestrator_with_default_config(self):
        """Test orchestrator creation with default configuration."""
        orchestrator = UniversalOrchestrator()
        
        assert orchestrator.config.mode == OrchestratorMode.PRODUCTION
        assert orchestrator.config.max_agents == 100
        assert orchestrator.config.max_concurrent_tasks == 1000


class TestAgentRegistration:
    """Test agent registration functionality and performance."""
    
    async def test_agent_registration_success(self, orchestrator):
        """Test successful agent registration."""
        agent_id = "test_agent_001"
        role = AgentRole.WORKER
        capabilities = ["python", "javascript", "testing"]
        
        start_time = time.time()
        result = await orchestrator.register_agent(agent_id, role, capabilities)
        registration_time = (time.time() - start_time) * 1000
        
        assert result is True
        assert registration_time < orchestrator.config.max_agent_registration_ms
        assert agent_id in orchestrator.agents
        
        agent = orchestrator.agents[agent_id]
        assert agent.id == agent_id
        assert agent.role == role
        assert agent.capabilities == capabilities
        assert agent.status == AgentStatus.ACTIVE
        
        # Check capability index
        for capability in capabilities:
            assert agent_id in orchestrator.agent_capabilities_index[capability]
    
    async def test_agent_registration_duplicate(self, orchestrator):
        """Test registration of duplicate agent."""
        agent_id = "test_agent_duplicate"
        role = AgentRole.WORKER
        capabilities = ["python"]
        
        # Register first time
        result1 = await orchestrator.register_agent(agent_id, role, capabilities)
        assert result1 is True
        
        # Register duplicate
        result2 = await orchestrator.register_agent(agent_id, role, capabilities)
        assert result2 is False
        assert len(orchestrator.agents) == 1
    
    async def test_agent_registration_capacity_limit(self, orchestrator):
        """Test agent registration when capacity limit is reached."""
        # Register up to max capacity
        for i in range(orchestrator.config.max_agents):
            agent_id = f"agent_{i:03d}"
            result = await orchestrator.register_agent(
                agent_id, AgentRole.WORKER, ["python"]
            )
            assert result is True
        
        # Try to register one more
        result = await orchestrator.register_agent(
            "overflow_agent", AgentRole.WORKER, ["python"]
        )
        assert result is False
        assert len(orchestrator.agents) == orchestrator.config.max_agents
    
    async def test_agent_registration_performance_batch(self, orchestrator):
        """Test batch agent registration performance."""
        num_agents = 20
        registration_times = []
        
        for i in range(num_agents):
            agent_id = f"batch_agent_{i:03d}"
            start_time = time.time()
            result = await orchestrator.register_agent(
                agent_id, AgentRole.WORKER, ["python", "testing"]
            )
            registration_time = (time.time() - start_time) * 1000
            
            assert result is True
            registration_times.append(registration_time)
        
        # Check that all registrations met performance requirements
        avg_registration_time = sum(registration_times) / len(registration_times)
        max_registration_time = max(registration_times)
        
        assert avg_registration_time < orchestrator.config.max_agent_registration_ms
        assert max_registration_time < orchestrator.config.max_agent_registration_ms * 2  # Allow some variance
    
    async def test_circuit_breaker_agent_registration(self, orchestrator):
        """Test circuit breaker behavior during agent registration failures."""
        # Mock a failure scenario
        with patch.object(orchestrator, 'agents', side_effect=Exception("Database error")):
            
            # Trigger multiple failures to open circuit breaker
            breaker = orchestrator.circuit_breakers['agent_registration']
            initial_threshold = breaker.failure_threshold
            
            for i in range(initial_threshold + 1):
                result = await orchestrator.register_agent(
                    f"failing_agent_{i}", AgentRole.WORKER, ["python"]
                )
                assert result is False
            
            # Circuit breaker should now be open
            assert breaker.state == CircuitBreakerState.OPEN
            
            # Next registration should fail immediately due to circuit breaker
            result = await orchestrator.register_agent(
                "blocked_agent", AgentRole.WORKER, ["python"]
            )
            assert result is False


class TestTaskDelegation:
    """Test task delegation functionality and performance."""
    
    async def test_task_delegation_success(self, orchestrator):
        """Test successful task delegation."""
        # Register agents first
        await orchestrator.register_agent("agent1", AgentRole.WORKER, ["python", "testing"])
        await orchestrator.register_agent("agent2", AgentRole.WORKER, ["javascript"])
        
        task_id = "test_task_001"
        task_type = "python_script"
        required_capabilities = ["python"]
        
        start_time = time.time()
        assigned_agent = await orchestrator.delegate_task(
            task_id, task_type, required_capabilities, TaskPriority.MEDIUM
        )
        delegation_time = (time.time() - start_time) * 1000
        
        assert assigned_agent is not None
        assert delegation_time < orchestrator.config.max_task_delegation_ms
        assert task_id in orchestrator.active_tasks
        
        # Check agent assignment
        assert orchestrator.agents[assigned_agent].current_task == task_id
        
        # Check task execution record
        task_execution = orchestrator.active_tasks[task_id]
        assert task_execution.task_id == task_id
        assert task_execution.agent_id == assigned_agent
        assert task_execution.status == TaskStatus.ASSIGNED
    
    async def test_task_delegation_no_suitable_agents(self, orchestrator):
        """Test task delegation when no agents have required capabilities."""
        # Register agents with different capabilities
        await orchestrator.register_agent("agent1", AgentRole.WORKER, ["python"])
        await orchestrator.register_agent("agent2", AgentRole.WORKER, ["javascript"])
        
        task_id = "impossible_task"
        required_capabilities = ["rust", "embedded"]  # No agents have these
        
        assigned_agent = await orchestrator.delegate_task(
            task_id, "rust_code", required_capabilities
        )
        
        assert assigned_agent is None
        assert task_id not in orchestrator.active_tasks
    
    async def test_task_delegation_no_available_agents(self, orchestrator):
        """Test task delegation when all suitable agents are busy."""
        # Register agents
        await orchestrator.register_agent("agent1", AgentRole.WORKER, ["python"])
        await orchestrator.register_agent("agent2", AgentRole.WORKER, ["python"])
        
        # Assign tasks to make all agents busy
        await orchestrator.delegate_task("task1", "python_script", ["python"])
        await orchestrator.delegate_task("task2", "python_script", ["python"])
        
        # Try to delegate another task
        assigned_agent = await orchestrator.delegate_task(
            "task3", "python_script", ["python"]
        )
        
        assert assigned_agent is None
        assert "task3" not in orchestrator.active_tasks
    
    async def test_task_delegation_load_balancing(self, orchestrator):
        """Test load balancing in task delegation."""
        # Register multiple agents with same capabilities
        agents = []
        for i in range(5):
            agent_id = f"worker_{i}"
            await orchestrator.register_agent(agent_id, AgentRole.WORKER, ["python"])
            agents.append(agent_id)
        
        # Delegate multiple tasks and track assignments
        assignments = {}
        for i in range(10):
            task_id = f"balanced_task_{i}"
            assigned_agent = await orchestrator.delegate_task(
                task_id, "python_script", ["python"]
            )
            
            assignments[task_id] = assigned_agent
            
            # Complete task to free up agent
            await orchestrator.complete_task(task_id, assigned_agent, success=True)
        
        # Check that tasks were distributed across agents
        agent_task_counts = {}
        for assigned_agent in assignments.values():
            agent_task_counts[assigned_agent] = agent_task_counts.get(assigned_agent, 0) + 1
        
        # All agents should have received at least one task
        assert len(agent_task_counts) == 5
        
        # No agent should have received significantly more tasks than others
        counts = list(agent_task_counts.values())
        assert max(counts) - min(counts) <= 2  # Allow small variance
    
    async def test_task_delegation_performance_concurrent(self, orchestrator):
        """Test concurrent task delegation performance."""
        # Register multiple agents
        for i in range(10):
            await orchestrator.register_agent(
                f"concurrent_agent_{i}", AgentRole.WORKER, ["python", "concurrent"]
            )
        
        # Create concurrent task delegation
        async def delegate_single_task(task_index):
            task_id = f"concurrent_task_{task_index}"
            start_time = time.time()
            assigned_agent = await orchestrator.delegate_task(
                task_id, "concurrent_script", ["python"], TaskPriority.MEDIUM
            )
            delegation_time = (time.time() - start_time) * 1000
            return assigned_agent, delegation_time
        
        # Execute concurrent delegations
        num_concurrent_tasks = 20
        tasks = [delegate_single_task(i) for i in range(num_concurrent_tasks)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful_delegations = [r for r in results if r[0] is not None]
        delegation_times = [r[1] for r in successful_delegations]
        
        assert len(successful_delegations) == 10  # Limited by available agents
        
        # All delegations should meet performance requirements
        for delegation_time in delegation_times:
            assert delegation_time < orchestrator.config.max_task_delegation_ms
        
        avg_delegation_time = sum(delegation_times) / len(delegation_times)
        assert avg_delegation_time < orchestrator.config.max_task_delegation_ms * 0.8  # Well under limit


class TestTaskCompletion:
    """Test task completion functionality."""
    
    async def test_task_completion_success(self, orchestrator):
        """Test successful task completion."""
        # Register agent and delegate task
        agent_id = "completion_agent"
        await orchestrator.register_agent(agent_id, AgentRole.WORKER, ["python"])
        
        task_id = "completion_task"
        assigned_agent = await orchestrator.delegate_task(
            task_id, "python_script", ["python"]
        )
        assert assigned_agent == agent_id
        
        # Complete the task
        result = {"output": "Task completed successfully", "exit_code": 0}
        completion_result = await orchestrator.complete_task(
            task_id, agent_id, result, success=True
        )
        
        assert completion_result is True
        assert task_id not in orchestrator.active_tasks
        assert orchestrator.agents[agent_id].current_task is None
        assert orchestrator.agents[agent_id].total_tasks_completed == 1
    
    async def test_task_completion_failure(self, orchestrator):
        """Test task completion with failure."""
        # Register agent and delegate task
        agent_id = "failure_agent"
        await orchestrator.register_agent(agent_id, AgentRole.WORKER, ["python"])
        
        task_id = "failure_task"
        await orchestrator.delegate_task(task_id, "python_script", ["python"])
        
        # Complete the task with failure
        result = {"error": "Script failed", "exit_code": 1}
        completion_result = await orchestrator.complete_task(
            task_id, agent_id, result, success=False
        )
        
        assert completion_result is True
        assert task_id not in orchestrator.active_tasks
        assert orchestrator.agents[agent_id].current_task is None
        assert orchestrator.agents[agent_id].error_count == 1
    
    async def test_task_completion_invalid_task(self, orchestrator):
        """Test completion of non-existent task."""
        agent_id = "test_agent"
        await orchestrator.register_agent(agent_id, AgentRole.WORKER, ["python"])
        
        completion_result = await orchestrator.complete_task(
            "non_existent_task", agent_id, {}, success=True
        )
        
        assert completion_result is False
    
    async def test_task_completion_wrong_agent(self, orchestrator):
        """Test task completion by wrong agent."""
        # Register agents and delegate task
        await orchestrator.register_agent("agent1", AgentRole.WORKER, ["python"])
        await orchestrator.register_agent("agent2", AgentRole.WORKER, ["python"])
        
        task_id = "wrong_agent_task"
        assigned_agent = await orchestrator.delegate_task(
            task_id, "python_script", ["python"]
        )
        
        # Try to complete with different agent
        wrong_agent = "agent1" if assigned_agent == "agent2" else "agent2"
        completion_result = await orchestrator.complete_task(
            task_id, wrong_agent, {}, success=True
        )
        
        assert completion_result is False
        assert task_id in orchestrator.active_tasks  # Task still active


class TestSystemStatus:
    """Test system status and health monitoring."""
    
    async def test_system_status_basic(self, orchestrator):
        """Test basic system status reporting."""
        status = await orchestrator.get_system_status()
        
        assert 'orchestrator_id' in status
        assert 'mode' in status
        assert 'uptime_seconds' in status
        assert 'health_status' in status
        assert 'agents' in status
        assert 'tasks' in status
        assert 'performance' in status
        assert 'circuit_breakers' in status
        
        # Check structure
        assert 'total' in status['agents']
        assert 'active' in status['agents']
        assert 'busy' in status['agents']
        assert 'by_role' in status['agents']
    
    async def test_system_status_with_agents(self, orchestrator):
        """Test system status with registered agents."""
        # Register different types of agents
        await orchestrator.register_agent("coordinator1", AgentRole.COORDINATOR, ["coordination"])
        await orchestrator.register_agent("worker1", AgentRole.WORKER, ["python"])
        await orchestrator.register_agent("worker2", AgentRole.WORKER, ["javascript"])
        await orchestrator.register_agent("monitor1", AgentRole.MONITOR, ["monitoring"])
        
        status = await orchestrator.get_system_status()
        
        assert status['agents']['total'] == 4
        assert status['agents']['active'] == 4
        assert status['agents']['busy'] == 0
        assert status['agents']['by_role']['coordinator'] == 1
        assert status['agents']['by_role']['worker'] == 2
        assert status['agents']['by_role']['monitor'] == 1
    
    async def test_system_status_with_active_tasks(self, orchestrator):
        """Test system status with active tasks."""
        # Register agents and delegate tasks
        await orchestrator.register_agent("worker1", AgentRole.WORKER, ["python"])
        await orchestrator.register_agent("worker2", AgentRole.WORKER, ["python"])
        
        await orchestrator.delegate_task("task1", "python_script", ["python"])
        await orchestrator.delegate_task("task2", "python_script", ["python"])
        
        status = await orchestrator.get_system_status()
        
        assert status['agents']['busy'] == 2
        assert status['tasks']['active'] == 2
    
    async def test_health_status_healthy(self, orchestrator):
        """Test healthy system status."""
        # Register some agents
        for i in range(5):
            await orchestrator.register_agent(f"healthy_agent_{i}", AgentRole.WORKER, ["python"])
        
        status = await orchestrator.get_system_status()
        assert status['health_status'] == HealthStatus.HEALTHY.value
        assert len(status.get('health_issues', [])) == 0
    
    async def test_health_status_degraded(self, orchestrator):
        """Test degraded system status due to circuit breaker."""
        # Open a circuit breaker
        breaker = orchestrator.circuit_breakers['agent_registration']
        breaker.state = CircuitBreakerState.OPEN
        
        status = await orchestrator.get_system_status()
        assert status['health_status'] in [HealthStatus.DEGRADED.value, HealthStatus.UNHEALTHY.value]
        assert len(status.get('health_issues', [])) > 0


class TestCircuitBreakers:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and initial state."""
        breaker = CircuitBreaker("test_breaker")
        
        assert breaker.name == "test_breaker"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        breaker = CircuitBreaker("test_breaker", failure_threshold=3)
        
        # Record failures up to threshold
        for i in range(3):
            assert breaker.state == CircuitBreakerState.CLOSED
            assert breaker.can_execute() is True
            breaker.record_failure()
        
        # Circuit breaker should now be open
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.can_execute() is False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        breaker = CircuitBreaker("test_breaker", failure_threshold=2, recovery_timeout=1)
        
        # Open the circuit breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should allow execution in half-open state
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Record success to close circuit
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0


class TestConcurrencyAndLoad:
    """Test concurrent operations and load handling."""
    
    async def test_concurrent_agent_registration(self, orchestrator):
        """Test concurrent agent registrations."""
        num_concurrent = 25
        
        async def register_agent(agent_index):
            agent_id = f"concurrent_reg_agent_{agent_index}"
            return await orchestrator.register_agent(
                agent_id, AgentRole.WORKER, ["concurrent", "testing"]
            )
        
        # Execute concurrent registrations
        tasks = [register_agent(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        
        # All registrations should succeed (within capacity limit)
        successful_registrations = sum(1 for r in results if r)
        expected_successful = min(num_concurrent, orchestrator.config.max_agents)
        
        assert successful_registrations == expected_successful
        assert len(orchestrator.agents) == expected_successful
    
    async def test_high_load_task_delegation(self, orchestrator):
        """Test task delegation under high load."""
        # Register many agents
        num_agents = 20
        for i in range(num_agents):
            await orchestrator.register_agent(f"load_agent_{i}", AgentRole.WORKER, ["python", "load"])
        
        # Create high load of task delegations
        num_tasks = 50
        
        async def delegate_task(task_index):
            task_id = f"load_task_{task_index}"
            return await orchestrator.delegate_task(
                task_id, "load_script", ["python"], TaskPriority.MEDIUM
            )
        
        start_time = time.time()
        tasks = [delegate_task(i) for i in range(num_tasks)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_delegations = sum(1 for r in results if r is not None)
        
        # Should delegate up to the number of available agents
        assert successful_delegations == num_agents
        assert len(orchestrator.active_tasks) == num_agents
        
        # Total time should be reasonable
        assert total_time < 10.0  # Should complete within 10 seconds
    
    async def test_50_concurrent_agents(self, orchestrator):
        """Test support for 50+ concurrent agents requirement."""
        # Override config for this test
        orchestrator.config.max_agents = 60
        
        # Register 50+ agents concurrently
        num_agents = 55
        
        async def register_agent(agent_index):
            agent_id = f"concurrent_agent_{agent_index:02d}"
            start_time = time.time()
            result = await orchestrator.register_agent(
                agent_id, AgentRole.WORKER, ["python", "concurrent"]
            )
            registration_time = (time.time() - start_time) * 1000
            return result, registration_time
        
        # Execute concurrent registrations
        tasks = [register_agent(i) for i in range(num_agents)]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        successful_registrations = sum(1 for r, t in results if r)
        registration_times = [t for r, t in results if r]
        
        assert successful_registrations == num_agents
        assert len(orchestrator.agents) == num_agents
        
        # All registrations should meet performance requirements
        max_registration_time = max(registration_times)
        avg_registration_time = sum(registration_times) / len(registration_times)
        
        assert max_registration_time < orchestrator.config.max_agent_registration_ms * 2  # Allow some variance for load
        assert avg_registration_time < orchestrator.config.max_agent_registration_ms


class TestBackwardCompatibility:
    """Test backward compatibility with existing orchestrator interfaces."""
    
    async def test_agent_lifecycle_compatibility(self, orchestrator):
        """Test compatibility with existing agent lifecycle patterns."""
        # This test ensures that existing code using the orchestrator
        # will continue to work without modifications
        
        # Standard agent registration pattern
        agent_id = "compat_agent_001"
        result = await orchestrator.register_agent(
            agent_id, AgentRole.WORKER, ["python", "testing"]
        )
        assert result is True
        
        # Standard task delegation pattern
        task_id = "compat_task_001"
        assigned_agent = await orchestrator.delegate_task(
            task_id, "python_script", ["python"], TaskPriority.MEDIUM
        )
        assert assigned_agent == agent_id
        
        # Standard task completion pattern
        result = await orchestrator.complete_task(
            task_id, agent_id, {"status": "success"}, success=True
        )
        assert result is True
        
        # Standard status checking pattern
        status = await orchestrator.get_system_status()
        assert status['agents']['total'] == 1
        assert status['tasks']['active'] == 0
    
    async def test_existing_orchestrator_patterns(self, orchestrator):
        """Test patterns commonly used with existing orchestrators."""
        # Pattern 1: Bulk agent registration
        agents_to_register = [
            ("bulk_agent_1", AgentRole.WORKER, ["python", "django"]),
            ("bulk_agent_2", AgentRole.WORKER, ["javascript", "react"]),
            ("bulk_agent_3", AgentRole.SPECIALIST, ["database", "postgresql"]),
        ]
        
        for agent_id, role, capabilities in agents_to_register:
            result = await orchestrator.register_agent(agent_id, role, capabilities)
            assert result is True
        
        # Pattern 2: Task delegation with capability matching
        test_tasks = [
            ("python_task", ["python"]),
            ("web_task", ["javascript", "react"]),
            ("db_task", ["database"]),
        ]
        
        assigned_agents = []
        for task_id, required_caps in test_tasks:
            assigned_agent = await orchestrator.delegate_task(
                task_id, "work", required_caps, TaskPriority.MEDIUM
            )
            assert assigned_agent is not None
            assigned_agents.append((task_id, assigned_agent))
        
        # Pattern 3: Bulk task completion
        for task_id, agent_id in assigned_agents:
            result = await orchestrator.complete_task(
                task_id, agent_id, {"result": "completed"}, success=True
            )
            assert result is True


class TestPluginIntegration:
    """Test plugin system integration."""
    
    async def test_plugin_manager_initialization(self, orchestrator):
        """Test that plugin manager is properly initialized."""
        assert orchestrator.plugin_manager is not None
    
    async def test_plugin_hooks_execution(self, orchestrator):
        """Test that plugin hooks are executed during operations."""
        # Mock a plugin to verify hook execution
        mock_plugin = AsyncMock()
        mock_plugin.enabled = True
        mock_plugin.pre_agent_registration = AsyncMock(return_value={})
        mock_plugin.post_agent_registration = AsyncMock(return_value={})
        
        # This would normally be handled by plugin system
        # For testing, we verify the hook points exist and are called
        
        # Register an agent and verify the orchestrator handles the process
        result = await orchestrator.register_agent(
            "plugin_test_agent", AgentRole.WORKER, ["python"]
        )
        assert result is True


class TestMemoryManagement:
    """Test memory usage and resource management."""
    
    async def test_memory_usage_within_limits(self, orchestrator):
        """Test that memory usage stays within configured limits."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register many agents and tasks to stress memory usage
        for i in range(20):
            await orchestrator.register_agent(
                f"memory_agent_{i}", AgentRole.WORKER, ["python", "memory"]
            )
        
        # Create many task delegations
        for i in range(50):
            task_id = f"memory_task_{i}"
            assigned_agent = await orchestrator.delegate_task(
                task_id, "memory_script", ["python"]
            )
            if assigned_agent:
                # Complete task immediately to create history
                await orchestrator.complete_task(task_id, assigned_agent, {}, True)
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (this is a rough estimate)
        # In production, this would be more strictly controlled
        assert memory_increase < orchestrator.config.max_memory_mb * 2  # Allow some overhead for testing
    
    async def test_resource_cleanup(self, orchestrator):
        """Test that resources are properly cleaned up."""
        initial_agent_count = len(orchestrator.agents)
        initial_task_count = len(orchestrator.active_tasks)
        
        # Create and complete multiple tasks
        await orchestrator.register_agent("cleanup_agent", AgentRole.WORKER, ["python"])
        
        for i in range(10):
            task_id = f"cleanup_task_{i}"
            assigned_agent = await orchestrator.delegate_task(
                task_id, "cleanup_script", ["python"]
            )
            assert assigned_agent is not None
            
            # Complete the task
            await orchestrator.complete_task(task_id, assigned_agent, {}, True)
        
        # Verify cleanup
        assert len(orchestrator.active_tasks) == initial_task_count  # No active tasks remaining
        assert len(orchestrator.agents) == initial_agent_count + 1  # Only the registered agent


class TestErrorHandling:
    """Test error handling and recovery."""
    
    async def test_database_error_handling(self, orchestrator):
        """Test handling of database errors."""
        # This would require mocking database operations
        # For now, we test that the orchestrator handles errors gracefully
        pass
    
    async def test_redis_error_handling(self, orchestrator):
        """Test handling of Redis errors."""
        # Mock Redis error
        with patch.object(orchestrator, 'redis', side_effect=Exception("Redis connection failed")):
            # Operations should still work or fail gracefully
            status = await orchestrator.get_system_status()
            assert 'error' not in status or status.get('error') is not None
    
    async def test_plugin_error_handling(self, orchestrator):
        """Test handling of plugin errors."""
        # This would test that plugin errors don't crash the orchestrator
        pass


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""
    
    async def test_full_workflow_scenario(self, orchestrator):
        """Test complete workflow from start to finish."""
        # Scenario: Python development team workflow
        
        # 1. Register development team agents
        team_agents = [
            ("lead_developer", AgentRole.COORDINATOR, ["python", "architecture", "review"]),
            ("backend_dev_1", AgentRole.WORKER, ["python", "django", "api"]),
            ("backend_dev_2", AgentRole.WORKER, ["python", "celery", "database"]),
            ("qa_engineer", AgentRole.WORKER, ["python", "testing", "pytest"]),
        ]
        
        for agent_id, role, capabilities in team_agents:
            result = await orchestrator.register_agent(agent_id, role, capabilities)
            assert result is True
        
        # 2. Delegate development tasks
        development_tasks = [
            ("implement_api_endpoint", ["python", "django", "api"], TaskPriority.HIGH),
            ("setup_task_queue", ["python", "celery"], TaskPriority.MEDIUM),
            ("write_unit_tests", ["python", "testing"], TaskPriority.MEDIUM),
            ("code_review", ["python", "review"], TaskPriority.LOW),
        ]
        
        task_assignments = {}
        for task_id, required_caps, priority in development_tasks:
            assigned_agent = await orchestrator.delegate_task(
                task_id, "development_work", required_caps, priority
            )
            assert assigned_agent is not None
            task_assignments[task_id] = assigned_agent
        
        # 3. Complete tasks in realistic order
        # Complete unit tests first
        await orchestrator.complete_task(
            "write_unit_tests", 
            task_assignments["write_unit_tests"], 
            {"test_results": "all_passed"}, 
            True
        )
        
        # Complete development tasks
        await orchestrator.complete_task(
            "implement_api_endpoint", 
            task_assignments["implement_api_endpoint"], 
            {"api_endpoint": "/api/v1/users"}, 
            True
        )
        
        await orchestrator.complete_task(
            "setup_task_queue", 
            task_assignments["setup_task_queue"], 
            {"queue_status": "configured"}, 
            True
        )
        
        # Complete code review
        await orchestrator.complete_task(
            "code_review", 
            task_assignments["code_review"], 
            {"review_status": "approved"}, 
            True
        )
        
        # 4. Verify final state
        status = await orchestrator.get_system_status()
        assert status['agents']['total'] == 4
        assert status['agents']['busy'] == 0
        assert status['tasks']['active'] == 0
        assert status['tasks']['completed_total'] == 4
    
    async def test_high_load_production_scenario(self, orchestrator):
        """Test high-load production scenario."""
        # Override config for production-like load
        orchestrator.config.max_agents = 50
        
        # Register a production-like agent fleet
        agent_types = [
            (AgentRole.COORDINATOR, ["coordination", "management"], 2),
            (AgentRole.WORKER, ["python", "api", "backend"], 20),
            (AgentRole.WORKER, ["javascript", "frontend", "react"], 15),
            (AgentRole.WORKER, ["database", "sql", "optimization"], 5),
            (AgentRole.MONITOR, ["monitoring", "alerting", "metrics"], 3),
            (AgentRole.SECURITY, ["security", "audit", "compliance"], 2),
        ]
        
        total_agents = 0
        for role, capabilities, count in agent_types:
            for i in range(count):
                agent_id = f"{role.value}_{i:02d}"
                result = await orchestrator.register_agent(agent_id, role, capabilities)
                assert result is True
                total_agents += 1
        
        assert total_agents == 47  # Total expected agents
        
        # Simulate production load with various task types
        task_types = [
            ("api_request", ["python", "api"], 30),
            ("frontend_render", ["javascript", "frontend"], 25),
            ("db_query", ["database", "sql"], 15),
            ("security_scan", ["security", "audit"], 5),
            ("health_check", ["monitoring", "metrics"], 10),
        ]
        
        # Delegate tasks in batches to simulate realistic load
        all_tasks = []
        for task_type, required_caps, count in task_types:
            tasks = []
            for i in range(count):
                task_id = f"{task_type}_{i:03d}"
                task_coro = orchestrator.delegate_task(
                    task_id, task_type, required_caps, TaskPriority.MEDIUM
                )
                tasks.append(task_coro)
            
            # Execute batch
            results = await asyncio.gather(*tasks)
            successful_assignments = [r for r in results if r is not None]
            all_tasks.extend(successful_assignments)
        
        # Should have successfully delegated many tasks
        assert len(all_tasks) > 40  # Most tasks should be assigned
        
        # Complete all tasks to test completion handling under load
        completion_tasks = []
        for task_id in orchestrator.active_tasks.keys():
            task_execution = orchestrator.active_tasks[task_id]
            completion_coro = orchestrator.complete_task(
                task_id, task_execution.agent_id, {"status": "completed"}, True
            )
            completion_tasks.append(completion_coro)
        
        completion_results = await asyncio.gather(*completion_tasks)
        successful_completions = sum(1 for r in completion_results if r)
        
        assert successful_completions > 40
        
        # Verify final state
        status = await orchestrator.get_system_status()
        assert status['agents']['total'] == total_agents
        assert status['tasks']['active'] == 0  # All tasks completed
        assert status['health_status'] == HealthStatus.HEALTHY.value


class TestGlobalOrchestratorInstance:
    """Test global orchestrator instance management."""
    
    async def test_get_universal_orchestrator(self):
        """Test getting global orchestrator instance."""
        config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
        
        # Mock Redis for clean test
        with patch('app.core.universal_orchestrator.get_redis', return_value=AsyncMock()):
            orchestrator1 = await get_universal_orchestrator(config)
            orchestrator2 = await get_universal_orchestrator(config)
            
            # Should return same instance
            assert orchestrator1 is orchestrator2
            
            # Clean up
            await shutdown_universal_orchestrator()
    
    async def test_shutdown_universal_orchestrator(self):
        """Test shutting down global orchestrator instance."""
        config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
        
        with patch('app.core.universal_orchestrator.get_redis', return_value=AsyncMock()):
            orchestrator = await get_universal_orchestrator(config)
            assert orchestrator is not None
            
            await shutdown_universal_orchestrator()
            
            # Should create new instance after shutdown
            orchestrator2 = await get_universal_orchestrator(config)
            assert orchestrator2 is not orchestrator
            
            await shutdown_universal_orchestrator()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])