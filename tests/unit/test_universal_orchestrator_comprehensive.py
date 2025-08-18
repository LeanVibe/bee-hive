"""
Comprehensive Unit Tests for UniversalOrchestrator

Tests all functionality of the consolidated UniversalOrchestrator with
performance validation, error handling, and regression detection.
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tests.consolidated.test_framework_base import ConsolidatedTestBase, TestScenario
from tests.performance.performance_benchmarking_framework import (
    PerformanceBenchmarkFramework, 
    BenchmarkConfiguration,
    get_universal_orchestrator_benchmarks
)

from app.core.universal_orchestrator import (
    UniversalOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole,
    HealthStatus,
    AgentInstance,
    TaskExecution,
    SystemMetrics
)
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority


class TestUniversalOrchestrator(ConsolidatedTestBase):
    """Comprehensive test suite for UniversalOrchestrator."""
    
    async def setup_component(self) -> UniversalOrchestrator:
        """Setup UniversalOrchestrator for testing."""
        config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=100,
            max_concurrent_tasks=1000,
            health_check_interval=1,  # Faster for testing
            cleanup_interval=5,
            auto_scaling_enabled=True,
            max_agent_registration_ms=100.0,
            max_task_delegation_ms=500.0,
            max_system_initialization_ms=2000.0,
            max_memory_mb=50.0
        )
        
        orchestrator = UniversalOrchestrator(config, "test_orchestrator")
        
        # Mock external dependencies
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            with patch('app.core.universal_orchestrator.get_session') as mock_session:
                mock_session.return_value = self.test_db
                
                success = await orchestrator.initialize()
                assert success, "Orchestrator initialization should succeed"
        
        self.add_cleanup_task(orchestrator.shutdown)
        return orchestrator
    
    async def cleanup_component(self) -> None:
        """Cleanup orchestrator after testing."""
        # Cleanup is handled by add_cleanup_task
        pass
    
    def get_performance_scenarios(self) -> List[TestScenario]:
        """Get performance test scenarios for UniversalOrchestrator."""
        scenarios = []
        
        # Agent registration performance scenario
        registration_scenario = TestScenario(
            name="agent_registration_performance",
            description="Test agent registration meets <100ms latency requirement",
            tags={"performance", "registration"}
        )
        registration_scenario.add_performance_threshold("registration_latency_ms", 100.0, 150.0)
        scenarios.append(registration_scenario)
        
        # Task delegation performance scenario
        delegation_scenario = TestScenario(
            name="task_delegation_performance", 
            description="Test task delegation meets <500ms latency requirement",
            tags={"performance", "delegation"}
        )
        delegation_scenario.add_performance_threshold("delegation_latency_ms", 500.0, 750.0)
        scenarios.append(delegation_scenario)
        
        # Concurrent agent coordination scenario
        coordination_scenario = TestScenario(
            name="concurrent_agent_coordination",
            description="Test 50+ concurrent agent coordination",
            tags={"performance", "concurrency"}
        )
        coordination_scenario.add_performance_threshold("coordination_latency_ms", 1000.0, 2000.0)
        scenarios.append(coordination_scenario)
        
        return scenarios
    
    # === INITIALIZATION AND LIFECYCLE TESTS ===
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization_performance(self):
        """Test orchestrator initialization meets <2000ms requirement."""
        config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
        
        async def test_initialization():
            orchestrator = UniversalOrchestrator(config)
            
            with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
                mock_redis.return_value = self.test_redis
                start_time = time.time()
                success = await orchestrator.initialize()
                duration_ms = (time.time() - start_time) * 1000
                
                assert success
                assert duration_ms < 2000.0, f"Initialization took {duration_ms:.2f}ms, exceeds 2000ms limit"
                
                await orchestrator.shutdown()
                return duration_ms
        
        scenario = TestScenario(
            name="system_initialization_performance",
            description="Test system initialization meets <2000ms requirement"
        )
        scenario.add_performance_threshold("initialization_time_ms", 2000.0, 3000.0)
        
        metrics = await self.run_performance_test(
            "test_orchestrator_initialization_performance",
            test_initialization,
            scenario
        )
        
        assert metrics.success, f"Test failed: {metrics.errors}"
    
    @pytest.mark.asyncio
    async def test_orchestrator_graceful_shutdown(self):
        """Test orchestrator shutdown completes gracefully."""
        orchestrator = await self.setup_component()
        
        # Register some agents first
        for i in range(5):
            success = await orchestrator.register_agent(
                f"test_agent_{i}",
                AgentRole.WORKER,
                [f"capability_{i}"]
            )
            assert success
        
        # Test shutdown
        start_time = time.time()
        await orchestrator.shutdown()
        shutdown_duration = (time.time() - start_time) * 1000
        
        # Verify shutdown completed in reasonable time
        assert shutdown_duration < 5000.0, f"Shutdown took {shutdown_duration:.2f}ms, too long"
    
    # === AGENT REGISTRATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_agent_registration_performance_requirement(self):
        """Test agent registration meets <100ms performance requirement."""
        orchestrator = await self.setup_component()
        
        async def register_single_agent():
            agent_id = str(uuid.uuid4())
            
            async with self.performance_monitor("test_agent_registration", "registration_latency_ms"):
                success = await orchestrator.register_agent(
                    agent_id,
                    AgentRole.WORKER,
                    ["test_capability"]
                )
            
            assert success
            return agent_id
        
        scenario = self.get_performance_scenarios()[0]  # Registration scenario
        
        metrics = await self.run_performance_test(
            "test_agent_registration",
            register_single_agent,
            scenario
        )
        
        assert metrics.success, f"Agent registration performance test failed: {metrics.errors}"
    
    @pytest.mark.asyncio
    async def test_agent_registration_capacity_limits(self):
        """Test agent registration respects capacity limits."""
        config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=5  # Small limit for testing
        )
        orchestrator = UniversalOrchestrator(config)
        
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            await orchestrator.initialize()
        
        self.add_cleanup_task(orchestrator.shutdown)
        
        # Register up to capacity
        for i in range(5):
            success = await orchestrator.register_agent(
                f"agent_{i}",
                AgentRole.WORKER,
                ["capability"]
            )
            assert success
        
        # Attempt to exceed capacity
        success = await orchestrator.register_agent(
            "excess_agent",
            AgentRole.WORKER,
            ["capability"]
        )
        assert not success, "Should reject agent registration when at capacity"
    
    @pytest.mark.asyncio
    async def test_agent_registration_duplicate_prevention(self):
        """Test prevention of duplicate agent registration."""
        orchestrator = await self.setup_component()
        
        agent_id = "duplicate_test_agent"
        
        # First registration should succeed
        success = await orchestrator.register_agent(
            agent_id,
            AgentRole.WORKER,
            ["capability"]
        )
        assert success
        
        # Second registration of same agent should fail
        success = await orchestrator.register_agent(
            agent_id,
            AgentRole.WORKER,
            ["capability"]
        )
        assert not success, "Should prevent duplicate agent registration"
    
    @pytest.mark.asyncio
    async def test_agent_registration_different_roles(self):
        """Test agent registration with different roles and capabilities."""
        orchestrator = await self.setup_component()
        
        # Test each agent role
        roles_and_capabilities = [
            (AgentRole.COORDINATOR, ["coordination", "delegation"]),
            (AgentRole.SPECIALIST, ["expert_knowledge", "analysis"]),
            (AgentRole.WORKER, ["task_execution"]),
            (AgentRole.MONITOR, ["health_monitoring", "metrics"]),
            (AgentRole.SECURITY, ["security_validation", "threat_detection"]),
            (AgentRole.OPTIMIZER, ["performance_tuning", "resource_optimization"])
        ]
        
        for role, capabilities in roles_and_capabilities:
            agent_id = f"test_agent_{role.value}"
            success = await orchestrator.register_agent(agent_id, role, capabilities)
            assert success, f"Failed to register agent with role {role.value}"
            
            # Verify agent is in orchestrator state
            assert agent_id in orchestrator.agents
            assert orchestrator.agents[agent_id].role == role
            assert orchestrator.agents[agent_id].capabilities == capabilities
    
    # === TASK DELEGATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_task_delegation_performance_requirement(self):
        """Test task delegation meets <500ms performance requirement."""
        orchestrator = await self.setup_component()
        
        # Register an agent first
        await orchestrator.register_agent(
            "worker_agent",
            AgentRole.WORKER,
            ["test_capability"]
        )
        
        async def delegate_single_task():
            task_id = str(uuid.uuid4())
            
            async with self.performance_monitor("test_task_delegation", "delegation_latency_ms"):
                selected_agent = await orchestrator.delegate_task(
                    task_id,
                    "test_task",
                    ["test_capability"],
                    TaskPriority.MEDIUM
                )
            
            assert selected_agent == "worker_agent"
            return task_id
        
        scenario = self.get_performance_scenarios()[1]  # Delegation scenario
        
        metrics = await self.run_performance_test(
            "test_task_delegation",
            delegate_single_task,
            scenario
        )
        
        assert metrics.success, f"Task delegation performance test failed: {metrics.errors}"
    
    @pytest.mark.asyncio
    async def test_task_delegation_capability_matching(self):
        """Test task delegation matches required capabilities correctly."""
        orchestrator = await self.setup_component()
        
        # Register agents with different capabilities
        await orchestrator.register_agent("python_agent", AgentRole.WORKER, ["python", "web_dev"])
        await orchestrator.register_agent("db_agent", AgentRole.WORKER, ["database", "sql"])
        await orchestrator.register_agent("ml_agent", AgentRole.SPECIALIST, ["machine_learning", "python"])
        
        # Test delegation to python agent
        selected = await orchestrator.delegate_task(
            "python_task",
            "code_task",
            ["python"],
            TaskPriority.MEDIUM
        )
        assert selected in ["python_agent", "ml_agent"], "Should select agent with python capability"
        
        # Test delegation to database agent
        selected = await orchestrator.delegate_task(
            "db_task",
            "query_task", 
            ["database"],
            TaskPriority.MEDIUM
        )
        assert selected == "db_agent", "Should select agent with database capability"
        
        # Test delegation requiring multiple capabilities
        selected = await orchestrator.delegate_task(
            "ml_task",
            "ml_task",
            ["machine_learning", "python"],
            TaskPriority.MEDIUM
        )
        assert selected == "ml_agent", "Should select agent with all required capabilities"
    
    @pytest.mark.asyncio
    async def test_task_delegation_load_balancing(self):
        """Test task delegation implements load balancing."""
        orchestrator = await self.setup_component()
        
        # Register multiple agents with same capabilities
        for i in range(3):
            await orchestrator.register_agent(
                f"worker_{i}",
                AgentRole.WORKER,
                ["common_capability"]
            )
        
        # Delegate multiple tasks and track distribution
        agent_task_counts = {"worker_0": 0, "worker_1": 0, "worker_2": 0}
        
        for i in range(9):  # 3 tasks per agent
            selected = await orchestrator.delegate_task(
                f"task_{i}",
                "test_task",
                ["common_capability"],
                TaskPriority.MEDIUM
            )
            
            # Complete the task to free up the agent
            await orchestrator.complete_task(f"task_{i}", selected, success=True)
            agent_task_counts[selected] += 1
        
        # Verify load is distributed reasonably (each agent should have gotten some tasks)
        for agent, count in agent_task_counts.items():
            assert count > 0, f"Agent {agent} received no tasks - load balancing failed"
            assert count <= 5, f"Agent {agent} received {count} tasks - poor load balancing"
    
    @pytest.mark.asyncio
    async def test_task_delegation_priority_handling(self):
        """Test task delegation handles priority correctly."""
        orchestrator = await self.setup_component()
        
        # Register one agent
        await orchestrator.register_agent(
            "priority_agent",
            AgentRole.WORKER,
            ["priority_capability"]
        )
        
        # Test different priority levels
        priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL]
        
        for priority in priorities:
            selected = await orchestrator.delegate_task(
                f"task_{priority.value}",
                "priority_task",
                ["priority_capability"],
                priority
            )
            assert selected == "priority_agent"
            
            # Verify task is recorded with correct priority
            task_execution = orchestrator.active_tasks[f"task_{priority.value}"]
            assert task_execution.priority == priority
            
            # Complete task to free agent
            await orchestrator.complete_task(f"task_{priority.value}", selected, success=True)
    
    @pytest.mark.asyncio
    async def test_task_delegation_no_suitable_agents(self):
        """Test task delegation handles no suitable agents gracefully."""
        orchestrator = await self.setup_component()
        
        # Register agent with different capabilities
        await orchestrator.register_agent(
            "mismatched_agent",
            AgentRole.WORKER,
            ["different_capability"]
        )
        
        # Try to delegate task requiring unavailable capability
        selected = await orchestrator.delegate_task(
            "impossible_task",
            "test_task",
            ["unavailable_capability"],
            TaskPriority.MEDIUM
        )
        
        assert selected is None, "Should return None when no suitable agents available"
    
    # === TASK COMPLETION TESTS ===
    
    @pytest.mark.asyncio
    async def test_task_completion_success_path(self):
        """Test successful task completion updates state correctly."""
        orchestrator = await self.setup_component()
        
        # Register agent and delegate task
        await orchestrator.register_agent("completion_agent", AgentRole.WORKER, ["test_capability"])
        selected = await orchestrator.delegate_task(
            "completion_task",
            "test_task",
            ["test_capability"],
            TaskPriority.MEDIUM
        )
        assert selected == "completion_agent"
        
        # Verify task is active
        assert "completion_task" in orchestrator.active_tasks
        assert orchestrator.agents["completion_agent"].current_task == "completion_task"
        
        # Complete task successfully
        result = {"output": "task completed successfully"}
        success = await orchestrator.complete_task(
            "completion_task",
            "completion_agent", 
            result,
            True
        )
        
        assert success
        
        # Verify state updates
        assert "completion_task" not in orchestrator.active_tasks
        assert orchestrator.agents["completion_agent"].current_task is None
        assert orchestrator.agents["completion_agent"].total_tasks_completed == 1
        assert orchestrator.agents["completion_agent"].error_count == 0
    
    @pytest.mark.asyncio
    async def test_task_completion_failure_path(self):
        """Test failed task completion updates error counts."""
        orchestrator = await self.setup_component()
        
        # Register agent and delegate task
        await orchestrator.register_agent("error_agent", AgentRole.WORKER, ["test_capability"])
        selected = await orchestrator.delegate_task("error_task", "test_task", ["test_capability"])
        
        # Complete task with failure
        success = await orchestrator.complete_task(
            "error_task",
            "error_agent",
            {"error": "task failed"},
            False
        )
        
        assert success  # Completion processing succeeded
        
        # Verify error tracking
        assert orchestrator.agents["error_agent"].error_count == 1
        assert "error_task" not in orchestrator.active_tasks
    
    @pytest.mark.asyncio 
    async def test_task_completion_invalid_scenarios(self):
        """Test task completion handles invalid scenarios."""
        orchestrator = await self.setup_component()
        
        # Test completing non-existent task
        success = await orchestrator.complete_task(
            "nonexistent_task",
            "nonexistent_agent",
            success=True
        )
        assert not success
        
        # Register agent and delegate task
        await orchestrator.register_agent("valid_agent", AgentRole.WORKER, ["test_capability"])
        selected = await orchestrator.delegate_task("valid_task", "test_task", ["test_capability"])
        
        # Test wrong agent completing task
        success = await orchestrator.complete_task(
            "valid_task",
            "wrong_agent",
            success=True
        )
        assert not success
        
        # Task should still be active
        assert "valid_task" in orchestrator.active_tasks
    
    # === CONCURRENT OPERATIONS TESTS ===
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_registration_50_plus(self):
        """Test concurrent registration of 50+ agents meets performance requirements."""
        orchestrator = await self.setup_component()
        
        async def register_agents_concurrently():
            """Register 55 agents concurrently."""
            tasks = []
            
            for i in range(55):
                task = orchestrator.register_agent(
                    f"concurrent_agent_{i}",
                    AgentRole.WORKER,
                    [f"capability_{i%5}"]
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent registration test
        scenario = self.get_performance_scenarios()[2]  # Coordination scenario
        
        metrics = await self.run_performance_test(
            "test_concurrent_agent_registration",
            register_agents_concurrently,
            scenario
        )
        
        assert metrics.success, f"Concurrent agent registration failed: {metrics.errors}"
        
        # Verify all agents were registered
        assert len(orchestrator.agents) == 55, f"Expected 55 agents, got {len(orchestrator.agents)}"
    
    @pytest.mark.asyncio
    async def test_concurrent_task_delegation(self):
        """Test concurrent task delegation performance."""
        orchestrator = await self.setup_component()
        
        # Register multiple agents
        for i in range(20):
            await orchestrator.register_agent(
                f"delegation_agent_{i}",
                AgentRole.WORKER,
                ["concurrent_capability"]
            )
        
        async def delegate_tasks_concurrently():
            """Delegate 50 tasks concurrently."""
            tasks = []
            
            for i in range(50):
                task = orchestrator.delegate_task(
                    f"concurrent_task_{i}",
                    "concurrent_test",
                    ["concurrent_capability"],
                    TaskPriority.MEDIUM
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            successful_delegations = sum(1 for r in results if r is not None)
            return successful_delegations
        
        async with self.performance_monitor("concurrent_delegation", "delegation_latency_ms"):
            successful_count = await delegate_tasks_concurrently()
        
        # Should successfully delegate most tasks (some agents may be busy)
        assert successful_count >= 20, f"Only {successful_count} tasks delegated successfully"
    
    # === SYSTEM STATUS AND MONITORING TESTS ===
    
    @pytest.mark.asyncio
    async def test_system_status_comprehensive(self):
        """Test system status provides comprehensive information."""
        orchestrator = await self.setup_component()
        
        # Setup test state
        for i in range(5):
            await orchestrator.register_agent(f"status_agent_{i}", AgentRole.WORKER, ["status_cap"])
        
        # Delegate some tasks
        for i in range(3):
            await orchestrator.delegate_task(f"status_task_{i}", "test", ["status_cap"])
        
        # Get system status
        status = await orchestrator.get_system_status()
        
        # Verify comprehensive status information
        assert "orchestrator_id" in status
        assert "uptime_seconds" in status
        assert "health_status" in status
        assert "agents" in status
        assert "tasks" in status
        assert "performance" in status
        assert "circuit_breakers" in status
        
        # Verify agent statistics
        agent_stats = status["agents"]
        assert agent_stats["total"] == 5
        assert agent_stats["active"] == 5
        assert agent_stats["busy"] == 3
        
        # Verify task statistics
        task_stats = status["tasks"]
        assert task_stats["active"] == 3
    
    @pytest.mark.asyncio
    async def test_health_monitoring_agent_heartbeats(self):
        """Test health monitoring detects stale agent heartbeats."""
        config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            health_check_interval=1  # Fast for testing
        )
        orchestrator = UniversalOrchestrator(config)
        
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            await orchestrator.initialize()
        
        self.add_cleanup_task(orchestrator.shutdown)
        
        # Register agent
        await orchestrator.register_agent("heartbeat_agent", AgentRole.WORKER, ["test_cap"])
        
        # Simulate stale heartbeat by backdating
        agent = orchestrator.agents["heartbeat_agent"]
        agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=5)  # 5 minutes ago
        
        # Wait for health check to run
        await asyncio.sleep(2)
        
        # Verify agent marked as inactive
        assert orchestrator.agents["heartbeat_agent"].status == AgentStatus.INACTIVE
    
    # === CIRCUIT BREAKER AND FAULT TOLERANCE TESTS ===
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_agent_registration(self):
        """Test circuit breaker for agent registration failures."""
        orchestrator = await self.setup_component()
        
        # Force failures by mocking registration process
        with patch.object(orchestrator, 'register_agent') as mock_register:
            mock_register.side_effect = Exception("Simulated failure")
            
            # Trigger failures to open circuit breaker
            for i in range(6):  # Exceed failure threshold
                try:
                    await orchestrator.register_agent(f"failing_agent_{i}", AgentRole.WORKER, ["test"])
                except:
                    pass
        
        # Verify circuit breaker state
        cb_state = orchestrator.circuit_breakers['agent_registration']
        assert cb_state.state.value in ["open", "half_open"], "Circuit breaker should be open after failures"
    
    @pytest.mark.asyncio 
    async def test_circuit_breaker_task_delegation(self):
        """Test circuit breaker for task delegation failures."""
        orchestrator = await self.setup_component()
        
        # Force failures by mocking delegation
        original_method = orchestrator.delegate_task
        
        async def failing_delegate(*args, **kwargs):
            raise Exception("Simulated delegation failure")
        
        orchestrator.delegate_task = failing_delegate
        
        # Trigger failures
        for i in range(6):
            try:
                await orchestrator.delegate_task(f"failing_task_{i}", "test", ["capability"])
            except:
                pass
        
        # Restore original method
        orchestrator.delegate_task = original_method
        
        # Verify circuit breaker
        cb_state = orchestrator.circuit_breakers['task_delegation']
        assert cb_state.state.value in ["open", "half_open"], "Task delegation circuit breaker should be open"
    
    # === ERROR HANDLING AND RECOVERY TESTS ===
    
    @pytest.mark.asyncio
    async def test_error_recovery_from_redis_failure(self):
        """Test recovery from Redis connection failures."""
        orchestrator = await self.setup_component()
        
        # Simulate Redis failure
        self.test_redis.ping.side_effect = Exception("Redis connection lost")
        
        # Operations should handle Redis failure gracefully
        success = await orchestrator.register_agent("redis_test_agent", AgentRole.WORKER, ["test"])
        
        # May succeed or fail, but should not crash
        status = await orchestrator.get_system_status()
        assert "error" not in status or status.get("error") is None
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self):
        """Test enforcement of resource limits."""
        # Test with very low limits
        config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=2,
            max_concurrent_tasks=3
        )
        orchestrator = UniversalOrchestrator(config)
        
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            await orchestrator.initialize()
        
        self.add_cleanup_task(orchestrator.shutdown)
        
        # Register up to agent limit
        for i in range(2):
            success = await orchestrator.register_agent(f"limit_agent_{i}", AgentRole.WORKER, ["test"])
            assert success
        
        # Next registration should fail
        success = await orchestrator.register_agent("excess_agent", AgentRole.WORKER, ["test"])
        assert not success
        
        # Task limit testing would require more complex setup
        # but the principle is the same


# === PERFORMANCE INTEGRATION TESTS ===

class TestUniversalOrchestratorPerformance(ConsolidatedTestBase):
    """Performance-focused tests for UniversalOrchestrator."""
    
    async def setup_component(self) -> UniversalOrchestrator:
        """Setup orchestrator for performance testing."""
        config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=200,
            max_concurrent_tasks=5000,
            health_check_interval=10,
            cleanup_interval=30
        )
        
        orchestrator = UniversalOrchestrator(config, "performance_test_orchestrator")
        
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            success = await orchestrator.initialize()
            assert success
        
        self.add_cleanup_task(orchestrator.shutdown)
        return orchestrator
    
    async def cleanup_component(self) -> None:
        """Cleanup performance test orchestrator."""
        pass
    
    def get_performance_scenarios(self) -> List[TestScenario]:
        """Get performance scenarios for orchestrator."""
        return []  # Uses benchmark framework instead
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_comprehensive_performance_benchmarks(self):
        """Run comprehensive performance benchmarks for UniversalOrchestrator."""
        orchestrator = await self.setup_component()
        benchmark_framework = PerformanceBenchmarkFramework()
        
        # Get orchestrator benchmark configurations
        benchmark_configs = get_universal_orchestrator_benchmarks()
        
        results = []
        for config in benchmark_configs:
            if config.name == "agent_registration_performance":
                async def registration_benchmark():
                    agent_id = str(uuid.uuid4())
                    return await orchestrator.register_agent(
                        agent_id, 
                        AgentRole.WORKER, 
                        ["benchmark_capability"]
                    )
                
                result = await benchmark_framework.run_benchmark(
                    config, 
                    registration_benchmark
                )
                
            elif config.name == "task_delegation_performance":
                # Register agents for delegation testing
                for i in range(50):
                    await orchestrator.register_agent(
                        f"delegation_agent_{i}",
                        AgentRole.WORKER,
                        ["delegation_capability"]
                    )
                
                async def delegation_benchmark():
                    task_id = str(uuid.uuid4())
                    selected = await orchestrator.delegate_task(
                        task_id,
                        "benchmark_task",
                        ["delegation_capability"],
                        TaskPriority.MEDIUM
                    )
                    
                    if selected:
                        # Complete task to free agent
                        await orchestrator.complete_task(task_id, selected, success=True)
                    
                    return selected
                
                result = await benchmark_framework.run_benchmark(
                    config,
                    delegation_benchmark
                )
                
            elif config.name == "concurrent_agent_coordination":
                async def coordination_benchmark():
                    # Register multiple agents concurrently
                    tasks = []
                    for i in range(10):  # Smaller batch for duration test
                        task = orchestrator.register_agent(
                            f"coord_agent_{uuid.uuid4()}",
                            AgentRole.WORKER,
                            ["coordination_capability"]
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    return sum(1 for r in results if r)
                
                result = await benchmark_framework.run_benchmark(
                    config,
                    coordination_benchmark  
                )
            
            results.append(result)
        
        # Validate all benchmarks passed
        for result in results:
            assert not result.regression_detected, f"Performance regression in {result.benchmark_name}"
            assert result.error_rate_percent <= 5.0, f"High error rate in {result.benchmark_name}: {result.error_rate_percent}%"
        
        # Print summary
        summary = benchmark_framework.get_benchmark_summary()
        print(f"\n📊 Performance Benchmark Summary:")
        print(f"   Total Benchmarks: {summary['total_benchmarks']}")
        print(f"   Regressions Detected: {summary['benchmarks_with_regressions']}")
        
        assert summary['benchmarks_with_regressions'] == 0, "Performance regressions detected"