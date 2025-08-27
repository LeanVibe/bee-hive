"""
Isolated Integration Tests for Epic 1 UnifiedProductionOrchestrator

Tests orchestrator integration without external dependencies, focusing on
internal component integration and realistic workflow validation.
"""

import asyncio
import pytest
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig,
    AgentState,
    AgentCapability,
    TaskRoutingStrategy
)
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.agent import AgentType


class IntegrationTestAgent:
    """Integration test agent that simulates realistic behavior."""
    
    def __init__(self, agent_id: str = None, capabilities: List[AgentCapability] = None, failure_rate: float = 0.0):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.ACTIVE
        self.capabilities = capabilities or [
            AgentCapability(
                name="integration_test",
                description="Integration test agent",
                confidence_level=0.9,
                specialization_areas=["testing", "integration"]
            )
        ]
        self.task_count = 0
        self.execution_results = []
        self.shutdown_called = False
        self.execution_times = []
        self.failure_rate = failure_rate
        self.total_executions = 0
        
    async def execute_task(self, task: Task) -> Any:
        """Execute task with realistic processing simulation."""
        start_time = time.time()
        self.task_count += 1
        self.total_executions += 1
        
        # Simulate failure rate
        if self.failure_rate > 0 and (self.total_executions * self.failure_rate) >= 1:
            self.total_executions = 0  # Reset counter
            raise Exception(f"Simulated agent failure (rate: {self.failure_rate})")
        
        # Simulate realistic processing time
        processing_time = 0.02 + (0.03 * len(task.description or "")) / 100
        await asyncio.sleep(processing_time)
        
        result = {
            "task_id": task.id,
            "agent_id": self.agent_id,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time,
            "task_title": task.title
        }
        
        self.execution_results.append(result)
        self.execution_times.append(time.time() - start_time)
        return result
        
    async def get_status(self) -> AgentState:
        """Get agent status."""
        return self.state
        
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities
        
    async def shutdown(self, graceful: bool = True) -> None:
        """Shutdown agent."""
        if graceful:
            await asyncio.sleep(0.01)  # Simulate graceful shutdown time
        self.shutdown_called = True
        self.state = AgentState.TERMINATED


@pytest.fixture
def integration_config():
    """Configuration optimized for integration testing."""
    return OrchestratorConfig(
        max_concurrent_agents=25,
        min_agent_pool=3,
        max_agent_pool=30,
        agent_registration_timeout=1.0,
        agent_heartbeat_interval=5.0,
        task_delegation_timeout=1.0,
        max_task_queue_size=100,
        memory_limit_mb=512,
        cpu_limit_percent=80.0,
        registration_target_ms=50.0,
        delegation_target_ms=200.0,
        health_check_interval=10.0,
        metrics_collection_interval=5.0
    )


@pytest.fixture
async def isolated_orchestrator(integration_config):
    """Isolated orchestrator for integration testing."""
    # Mock external dependencies
    with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
         patch('app.core.unified_production_orchestrator.get_session') as mock_db:
        
        # Configure mocks
        mock_redis.return_value = AsyncMock()
        mock_db.return_value.__aenter__ = AsyncMock()
        mock_db.return_value.__aexit__ = AsyncMock()
        
        orchestrator = UnifiedProductionOrchestrator(integration_config)
        await orchestrator.start()
        yield orchestrator
        await orchestrator.shutdown(graceful=True)


class TestOrchestratorIsolatedIntegration:
    """Isolated integration tests for orchestrator functionality."""
    
    async def test_multi_agent_workflow_integration(self, isolated_orchestrator):
        """Test complete multi-agent workflow integration."""
        # Register diverse agents
        agents = []
        
        # Backend specialist
        backend_agent = IntegrationTestAgent(
            agent_id="backend_specialist",
            capabilities=[
                AgentCapability(
                    name="backend",
                    description="Backend development",
                    confidence_level=0.95,
                    specialization_areas=["python", "api", "database"]
                )
            ]
        )
        
        # Frontend specialist  
        frontend_agent = IntegrationTestAgent(
            agent_id="frontend_specialist",
            capabilities=[
                AgentCapability(
                    name="frontend",
                    description="Frontend development",
                    confidence_level=0.90,
                    specialization_areas=["javascript", "react", "ui"]
                )
            ]
        )
        
        # QA specialist
        qa_agent = IntegrationTestAgent(
            agent_id="qa_specialist",
            capabilities=[
                AgentCapability(
                    name="qa",
                    description="Quality assurance",
                    confidence_level=0.88,
                    specialization_areas=["testing", "validation", "automation"]
                )
            ]
        )
        
        agents = [backend_agent, frontend_agent, qa_agent]
        
        # Register all agents
        agent_ids = []
        for agent in agents:
            agent_id = await isolated_orchestrator.register_agent(agent)
            agent_ids.append(agent_id)
        
        assert len(agent_ids) == 3
        assert len(isolated_orchestrator._agents) == 3
        
        # Create workflow tasks
        workflow_tasks = [
            Task(
                id=str(uuid.uuid4()),
                title="Database Schema Design",
                description="Design database schema for user management",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                estimated_effort=30
            ),
            Task(
                id=str(uuid.uuid4()),
                title="API Endpoint Implementation", 
                description="Implement REST API endpoints for user CRUD",
                priority=TaskPriority.HIGH,
                status=TaskStatus.PENDING,
                estimated_effort=60
            ),
            Task(
                id=str(uuid.uuid4()),
                title="User Interface Components",
                description="Create React components for user management UI",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                estimated_effort=45
            ),
            Task(
                id=str(uuid.uuid4()),
                title="Integration Testing",
                description="Write integration tests for user management flow",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                estimated_effort=25
            ),
            Task(
                id=str(uuid.uuid4()),
                title="Performance Testing",
                description="Load test user API endpoints",
                priority=TaskPriority.LOW,
                status=TaskStatus.PENDING,
                estimated_effort=15
            )
        ]
        
        # Delegate all tasks
        delegated_agents = []
        delegation_times = []
        
        for task in workflow_tasks:
            start_time = time.time()
            assigned_agent = await isolated_orchestrator.delegate_task(task)
            delegation_time = (time.time() - start_time) * 1000
            
            delegated_agents.append(assigned_agent)
            delegation_times.append(delegation_time)
            
            assert assigned_agent in agent_ids
        
        # Verify delegation performance (allow slight overhead for integration tests)
        avg_delegation_time = sum(delegation_times) / len(delegation_times)
        assert avg_delegation_time < isolated_orchestrator.config.delegation_target_ms * 1.1
        
        # Wait for task execution
        await asyncio.sleep(1.0)
        
        # Verify all tasks were executed
        total_executed = sum(agent.task_count for agent in agents)
        assert total_executed == len(workflow_tasks)
        
        # Verify load distribution
        task_distribution = [agent.task_count for agent in agents]
        assert max(task_distribution) <= len(workflow_tasks)  # No agent overloaded
        assert min(task_distribution) >= 0  # All agents could participate
        
        # Verify execution results
        all_results = []
        for agent in agents:
            all_results.extend(agent.execution_results)
        
        assert len(all_results) == len(workflow_tasks)
        
        # Verify all original tasks are represented in results
        result_task_ids = {result["task_id"] for result in all_results}
        original_task_ids = {task.id for task in workflow_tasks}
        assert result_task_ids == original_task_ids
    
    async def test_agent_lifecycle_integration(self, isolated_orchestrator):
        """Test complete agent lifecycle integration."""
        # Phase 1: Agent registration
        initial_agents = []
        for i in range(5):
            agent = IntegrationTestAgent(agent_id=f"lifecycle_agent_{i}")
            agent_id = await isolated_orchestrator.register_agent(agent)
            initial_agents.append((agent, agent_id))
        
        assert len(isolated_orchestrator._agents) == 5
        
        # Phase 2: Task execution
        tasks = []
        for i in range(10):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Lifecycle Task {i}",
                description=f"Task {i} for lifecycle testing",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
            await isolated_orchestrator.delegate_task(task)
        
        await asyncio.sleep(0.5)
        
        # Verify initial execution
        total_executed_phase1 = sum(agent.task_count for agent, _ in initial_agents)
        assert total_executed_phase1 == 10
        
        # Phase 3: Dynamic agent scaling (add more agents)
        additional_agents = []
        for i in range(3):
            agent = IntegrationTestAgent(agent_id=f"additional_agent_{i}")
            agent_id = await isolated_orchestrator.register_agent(agent)
            additional_agents.append((agent, agent_id))
        
        assert len(isolated_orchestrator._agents) == 8
        
        # Phase 4: More task execution with expanded pool
        additional_tasks = []
        for i in range(15):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Additional Task {i}",
                description=f"Additional task {i} for expanded pool",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            additional_tasks.append(task)
            await isolated_orchestrator.delegate_task(task)
        
        await asyncio.sleep(0.8)
        
        # Verify expanded execution
        total_executed_phase2 = sum(agent.task_count for agent, _ in initial_agents + additional_agents)
        assert total_executed_phase2 == 25  # 10 + 15
        
        # Phase 5: Agent removal
        agents_to_remove = initial_agents[:2]  # Remove first 2 agents
        for agent, agent_id in agents_to_remove:
            await isolated_orchestrator.unregister_agent(agent_id)
            assert agent.shutdown_called
        
        assert len(isolated_orchestrator._agents) == 6  # 8 - 2
        
        # Phase 6: Continued operation with reduced pool
        final_tasks = []
        for i in range(5):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Final Task {i}",
                description=f"Final task {i} with reduced pool",
                priority=TaskPriority.LOW,
                status=TaskStatus.PENDING
            )
            final_tasks.append(task)
            await isolated_orchestrator.delegate_task(task)
        
        await asyncio.sleep(0.3)
        
        # Verify continued operation
        remaining_agents = initial_agents[2:] + additional_agents
        final_executed = sum(agent.task_count for agent, _ in remaining_agents) - 20  # Only new tasks
        assert final_executed == 5
        
        # Verify system health
        status = await isolated_orchestrator.get_system_status()
        assert status['agents']['total_registered'] == 6
        assert status['orchestrator']['is_running'] is True
    
    async def test_error_recovery_integration(self, isolated_orchestrator):
        """Test comprehensive error recovery integration."""
        # Register mix of reliable and unreliable agents
        reliable_agents = []
        unreliable_agents = []
        
        # Reliable agents (no failures)
        for i in range(3):
            agent = IntegrationTestAgent(
                agent_id=f"reliable_{i}",
                failure_rate=0.0
            )
            agent_id = await isolated_orchestrator.register_agent(agent)
            reliable_agents.append((agent, agent_id))
        
        # Unreliable agents (20% failure rate)
        for i in range(2):
            agent = IntegrationTestAgent(
                agent_id=f"unreliable_{i}",
                failure_rate=0.2
            )
            agent_id = await isolated_orchestrator.register_agent(agent)
            unreliable_agents.append((agent, agent_id))
        
        assert len(isolated_orchestrator._agents) == 5
        
        # Create tasks that will encounter failures
        error_recovery_tasks = []
        for i in range(20):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Error Recovery Task {i}",
                description=f"Task {i} for error recovery testing",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            error_recovery_tasks.append(task)
        
        # Delegate tasks - some will fail, system should recover
        successful_delegations = 0
        failed_delegations = 0
        
        for task in error_recovery_tasks:
            try:
                assigned_agent = await isolated_orchestrator.delegate_task(task)
                successful_delegations += 1
                assert assigned_agent in isolated_orchestrator._agents
            except Exception as e:
                failed_delegations += 1
                print(f"Task delegation failed: {e}")
        
        # All delegations should succeed (delegation != execution)
        assert successful_delegations == 20
        assert failed_delegations == 0
        
        # Wait for task execution (some will fail during execution)
        await asyncio.sleep(1.5)
        
        # Count successful executions
        reliable_executions = sum(agent.task_count for agent, _ in reliable_agents)
        unreliable_executions = sum(agent.task_count for agent, _ in unreliable_agents)
        total_executions = reliable_executions + unreliable_executions
        
        print(f"Reliable executions: {reliable_executions}")
        print(f"Unreliable executions: {unreliable_executions}")
        print(f"Total executions: {total_executions}")
        
        # Should have completed most tasks despite some failures
        assert total_executions >= 15  # At least 75% success rate
        
        # Reliable agents should have higher success
        assert reliable_executions >= unreliable_executions
        
        # System should still be healthy
        status = await isolated_orchestrator.get_system_status()
        assert status['orchestrator']['is_running'] is True
        assert status['agents']['total_registered'] == 5
    
    async def test_performance_under_load_integration(self, isolated_orchestrator):
        """Test performance degradation under increasing load."""
        # Register agents for load testing
        load_agents = []
        for i in range(8):
            agent = IntegrationTestAgent(agent_id=f"load_agent_{i}")
            agent_id = await isolated_orchestrator.register_agent(agent)
            load_agents.append((agent, agent_id))
        
        # Test escalating load levels
        load_levels = [10, 25, 50, 75]
        performance_metrics = []
        
        for load_level in load_levels:
            print(f"Testing load level: {load_level} tasks")
            
            # Create tasks for this load level
            load_tasks = []
            for i in range(load_level):
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Load Test {load_level}-{i}",
                    description=f"Load test task {i} at level {load_level}",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                load_tasks.append(task)
            
            # Measure delegation performance
            start_time = time.time()
            delegation_times = []
            
            for task in load_tasks:
                task_start = time.time()
                assigned_agent = await isolated_orchestrator.delegate_task(task)
                task_time = (time.time() - task_start) * 1000
                delegation_times.append(task_time)
                assert assigned_agent in isolated_orchestrator._agents
            
            total_delegation_time = time.time() - start_time
            
            # Wait for execution
            execution_start = time.time()
            await asyncio.sleep(max(1.0, load_level * 0.02))  # Scale wait time with load
            execution_wait_time = time.time() - execution_start
            
            # Collect metrics
            avg_delegation_time = sum(delegation_times) / len(delegation_times)
            max_delegation_time = max(delegation_times)
            total_executed = sum(agent.task_count for agent, _ in load_agents)
            
            metrics = {
                'load_level': load_level,
                'avg_delegation_time_ms': avg_delegation_time,
                'max_delegation_time_ms': max_delegation_time,
                'total_delegation_time_s': total_delegation_time,
                'tasks_executed': total_executed,
                'execution_wait_time_s': execution_wait_time
            }
            performance_metrics.append(metrics)
            
            print(f"Load {load_level}: avg={avg_delegation_time:.2f}ms, max={max_delegation_time:.2f}ms")
            
            # Reset agent counters for next iteration
            for agent, _ in load_agents:
                agent.task_count = 0
                agent.execution_results.clear()
        
        # Analyze performance degradation
        for i, metrics in enumerate(performance_metrics):
            load_level = metrics['load_level']
            avg_time = metrics['avg_delegation_time_ms']
            
            # Performance should remain reasonable even under load
            if load_level <= 25:
                assert avg_time < isolated_orchestrator.config.delegation_target_ms
            else:
                # Allow some degradation at higher loads, but not excessive
                assert avg_time < isolated_orchestrator.config.delegation_target_ms * 2
            
            print(f"Load {load_level}: {metrics}")
        
        # Verify system is still healthy after load testing
        final_status = await isolated_orchestrator.get_system_status()
        assert final_status['orchestrator']['is_running'] is True
        assert final_status['agents']['total_registered'] == 8
    
    async def test_concurrent_operations_integration(self, isolated_orchestrator):
        """Test concurrent orchestrator operations integration."""
        # Test concurrent agent registration
        registration_tasks = []
        agents_to_register = []
        
        for i in range(15):
            agent = IntegrationTestAgent(agent_id=f"concurrent_agent_{i}")
            agents_to_register.append(agent)
            registration_tasks.append(isolated_orchestrator.register_agent(agent))
        
        # Execute concurrent registrations
        start_time = time.time()
        agent_ids = await asyncio.gather(*registration_tasks)
        registration_time = time.time() - start_time
        
        # Verify all registrations succeeded
        assert len(agent_ids) == 15
        assert len(set(agent_ids)) == 15  # All unique
        assert len(isolated_orchestrator._agents) == 15
        
        print(f"Concurrent registration time: {registration_time:.2f}s")
        
        # Test concurrent task delegation
        concurrent_tasks = []
        delegation_tasks = []
        
        for i in range(30):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Concurrent Task {i}",
                description=f"Concurrent delegation test task {i}",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            concurrent_tasks.append(task)
            delegation_tasks.append(isolated_orchestrator.delegate_task(task))
        
        # Execute concurrent delegations
        start_time = time.time()
        assigned_agents = await asyncio.gather(*delegation_tasks)
        delegation_time = time.time() - start_time
        
        # Verify all delegations succeeded
        assert len(assigned_agents) == 30
        assert all(agent_id in agent_ids for agent_id in assigned_agents)
        
        print(f"Concurrent delegation time: {delegation_time:.2f}s")
        
        # Wait for execution
        await asyncio.sleep(2.0)
        
        # Verify task distribution
        total_executed = sum(agent.task_count for agent in agents_to_register)
        assert total_executed == 30
        
        # Verify reasonable load distribution
        execution_counts = [agent.task_count for agent in agents_to_register]
        max_tasks = max(execution_counts)
        min_tasks = min(execution_counts)
        
        # No agent should be severely overloaded
        assert max_tasks <= (30 // 15) + 3  # Allow some imbalance
        
        print(f"Task distribution: min={min_tasks}, max={max_tasks}")
        
        # Test concurrent unregistration
        agents_to_remove = agent_ids[:5]
        unregistration_tasks = [
            isolated_orchestrator.unregister_agent(agent_id)
            for agent_id in agents_to_remove
        ]
        
        start_time = time.time()
        await asyncio.gather(*unregistration_tasks)
        unregistration_time = time.time() - start_time
        
        # Verify unregistrations
        assert len(isolated_orchestrator._agents) == 10
        for agent_id in agents_to_remove:
            assert agent_id not in isolated_orchestrator._agents
        
        print(f"Concurrent unregistration time: {unregistration_time:.2f}s")
        
        # Verify system stability after concurrent operations
        final_status = await isolated_orchestrator.get_system_status()
        assert final_status['orchestrator']['is_running'] is True
        assert final_status['agents']['total_registered'] == 10


@pytest.mark.performance
class TestOrchestratorPerformanceIntegration:
    """Performance-focused integration tests."""
    
    async def test_registration_performance_at_scale(self, integration_config):
        """Test agent registration performance at scale."""
        # Use higher capacity configuration
        scale_config = OrchestratorConfig(
            max_concurrent_agents=50,
            max_agent_pool=60,
            registration_target_ms=100.0,
            delegation_target_ms=500.0
        )
        
        with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
             patch('app.core.unified_production_orchestrator.get_session') as mock_db:
            
            mock_redis.return_value = AsyncMock()
            mock_db.return_value.__aenter__ = AsyncMock()
            mock_db.return_value.__aexit__ = AsyncMock()
            
            orchestrator = UnifiedProductionOrchestrator(scale_config)
            await orchestrator.start()
            
            try:
                # Test registration at different scales
                scale_levels = [10, 25, 40]
                registration_metrics = []
                
                for scale in scale_levels:
                    print(f"Testing registration scale: {scale} agents")
                    
                    # Create agents for this scale
                    agents = [
                        IntegrationTestAgent(agent_id=f"scale_{scale}_{i}")
                        for i in range(scale)
                    ]
                    
                    # Measure registration performance
                    start_time = time.time()
                    registration_times = []
                    
                    for agent in agents:
                        agent_start = time.time()
                        agent_id = await orchestrator.register_agent(agent)
                        agent_time = (time.time() - agent_start) * 1000
                        registration_times.append(agent_time)
                        assert agent_id is not None
                    
                    total_time = time.time() - start_time
                    
                    # Calculate metrics
                    avg_time = sum(registration_times) / len(registration_times)
                    max_time = max(registration_times)
                    min_time = min(registration_times)
                    
                    metrics = {
                        'scale': scale,
                        'total_time_s': total_time,
                        'avg_time_ms': avg_time,
                        'max_time_ms': max_time,
                        'min_time_ms': min_time,
                        'registrations_per_second': scale / total_time
                    }
                    registration_metrics.append(metrics)
                    
                    print(f"Scale {scale}: avg={avg_time:.2f}ms, throughput={metrics['registrations_per_second']:.1f}/s")
                    
                    # Verify performance requirements
                    assert avg_time < scale_config.registration_target_ms * 1.2  # Allow slight degradation
                    assert max_time < scale_config.registration_target_ms * 2.0
                    
                    # Clear for next iteration
                    for agent_id in list(orchestrator._agents.keys()):
                        await orchestrator.unregister_agent(agent_id)
                
                # Verify performance doesn't degrade significantly with scale
                for i in range(1, len(registration_metrics)):
                    current = registration_metrics[i]
                    previous = registration_metrics[i-1]
                    
                    # Performance degradation should be reasonable
                    degradation_factor = current['avg_time_ms'] / previous['avg_time_ms']
                    assert degradation_factor < 2.0, f"Performance degraded by {degradation_factor:.2f}x"
                
            finally:
                await orchestrator.shutdown(graceful=True)
    
    async def test_delegation_performance_at_scale(self, integration_config):
        """Test task delegation performance at scale."""
        scale_config = OrchestratorConfig(
            max_concurrent_agents=30,
            max_agent_pool=35,
            registration_target_ms=100.0,
            delegation_target_ms=500.0
        )
        
        with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
             patch('app.core.unified_production_orchestrator.get_session') as mock_db:
            
            mock_redis.return_value = AsyncMock()
            mock_db.return_value.__aenter__ = AsyncMock()
            mock_db.return_value.__aexit__ = AsyncMock()
            
            orchestrator = UnifiedProductionOrchestrator(scale_config)
            await orchestrator.start()
            
            try:
                # Register agents for delegation testing
                agents = []
                for i in range(20):
                    agent = IntegrationTestAgent(agent_id=f"delegation_agent_{i}")
                    agent_id = await orchestrator.register_agent(agent)
                    agents.append(agent)
                
                # Test delegation at different task volumes
                task_volumes = [50, 100, 200]
                delegation_metrics = []
                
                for volume in task_volumes:
                    print(f"Testing delegation volume: {volume} tasks")
                    
                    # Create tasks
                    tasks = []
                    for i in range(volume):
                        task = Task(
                            id=str(uuid.uuid4()),
                            title=f"Volume Test {volume}-{i}",
                            description=f"Delegation volume test task {i}",
                            priority=TaskPriority.MEDIUM,
                            status=TaskStatus.PENDING
                        )
                        tasks.append(task)
                    
                    # Measure delegation performance
                    start_time = time.time()
                    delegation_times = []
                    
                    for task in tasks:
                        task_start = time.time()
                        assigned_agent = await orchestrator.delegate_task(task)
                        task_time = (time.time() - task_start) * 1000
                        delegation_times.append(task_time)
                        assert assigned_agent is not None
                    
                    total_time = time.time() - start_time
                    
                    # Calculate metrics
                    avg_time = sum(delegation_times) / len(delegation_times)
                    max_time = max(delegation_times)
                    throughput = volume / total_time
                    
                    metrics = {
                        'volume': volume,
                        'total_time_s': total_time,
                        'avg_time_ms': avg_time,
                        'max_time_ms': max_time,
                        'delegations_per_second': throughput
                    }
                    delegation_metrics.append(metrics)
                    
                    print(f"Volume {volume}: avg={avg_time:.2f}ms, throughput={throughput:.1f}/s")
                    
                    # Verify performance requirements
                    assert avg_time < scale_config.delegation_target_ms
                    assert max_time < scale_config.delegation_target_ms * 2
                    
                    # Wait for task execution to complete before next iteration
                    await asyncio.sleep(min(2.0, volume * 0.01))
                    
                    # Reset agent state
                    for agent in agents:
                        agent.task_count = 0
                        agent.execution_results.clear()
                
                # Verify throughput scales reasonably
                for metrics in delegation_metrics:
                    volume = metrics['volume']
                    throughput = metrics['delegations_per_second']
                    
                    # Should achieve reasonable throughput
                    assert throughput > 10.0, f"Throughput {throughput:.1f}/s too low for volume {volume}"
                
            finally:
                await orchestrator.shutdown(graceful=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])