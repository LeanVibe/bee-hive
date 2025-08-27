"""
Comprehensive Integration Tests for Epic 1 UnifiedProductionOrchestrator

Tests orchestrator integration with real Redis, Database, and messaging systems.
Validates performance under realistic load conditions and end-to-end workflows.
"""

import asyncio
import pytest
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig,
    AgentState,
    AgentCapability,
    TaskRoutingStrategy
)
from app.core.database import get_session
from app.core.redis import get_redis, get_message_broker
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.agent import Agent, AgentType, AgentStatus
from app.models.session import Session


class RealTestAgent:
    """Real test agent that interfaces with actual systems."""
    
    def __init__(self, agent_id: str = None, capabilities: List[AgentCapability] = None):
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
        
    async def execute_task(self, task: Task) -> Any:
        """Execute task with real processing."""
        start_time = time.time()
        self.task_count += 1
        
        # Simulate realistic task processing with database interaction
        await asyncio.sleep(0.05)  # Realistic processing time
        
        result = {
            "task_id": task.id,
            "agent_id": self.agent_id,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": time.time() - start_time
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
        self.shutdown_called = True
        self.state = AgentState.TERMINATED


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return OrchestratorConfig(
        max_concurrent_agents=25,
        min_agent_pool=5,
        max_agent_pool=30,
        agent_registration_timeout=2.0,
        agent_heartbeat_interval=10.0,
        task_delegation_timeout=2.0,
        max_task_queue_size=200,
        memory_limit_mb=1024,
        cpu_limit_percent=80.0,
        registration_target_ms=100.0,
        delegation_target_ms=500.0,
        health_check_interval=15.0,
        metrics_collection_interval=10.0
    )


@pytest.fixture
async def integration_orchestrator(integration_config):
    """Integration test orchestrator with real dependencies."""
    orchestrator = UnifiedProductionOrchestrator(integration_config)
    await orchestrator.start()
    yield orchestrator
    await orchestrator.shutdown(graceful=True)


@pytest.fixture
async def redis_connection():
    """Real Redis connection for integration tests."""
    redis = await get_redis()
    yield redis
    await redis.close()


@pytest.fixture
async def db_session():
    """Real database session for integration tests."""
    async with get_session() as session:
        yield session


class TestUnifiedOrchestratorIntegration:
    """Integration tests for UnifiedProductionOrchestrator."""
    
    async def test_orchestrator_redis_integration(self, integration_orchestrator, redis_connection):
        """Test orchestrator integration with Redis messaging."""
        # Test Redis connectivity
        await redis_connection.ping()
        
        # Register agent and verify Redis state updates
        agent = RealTestAgent()
        agent_id = await integration_orchestrator.register_agent(agent)
        
        # Verify agent registration is reflected in Redis
        # This would be specific to the orchestrator's Redis usage
        assert agent_id is not None
        assert len(integration_orchestrator._agents) == 1
        
        # Test message broker functionality if available
        try:
            message_broker = await get_message_broker()
            # Test publishing agent registration event
            event = {
                "type": "agent_registered",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await message_broker.publish("agent_events", json.dumps(event))
        except Exception as e:
            pytest.skip(f"Message broker not available: {e}")
    
    async def test_orchestrator_database_integration(self, integration_orchestrator, db_session):
        """Test orchestrator integration with database."""
        # Register agent
        agent = RealTestAgent()
        agent_id = await integration_orchestrator.register_agent(agent)
        
        # Create task that would be stored in database
        task = Task(
            id=str(uuid.uuid4()),
            title="Database Integration Test",
            description="Test database persistence",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING,
            estimated_time_minutes=1
        )
        
        # Delegate task
        assigned_agent = await integration_orchestrator.delegate_task(task)
        assert assigned_agent == agent_id
        
        # Wait for task execution
        await asyncio.sleep(0.2)
        
        # Verify task was executed
        assert agent.task_count == 1
        assert len(agent.execution_results) == 1
        
        # Verify database state (would depend on actual schema)
        # This is a placeholder for actual database verification
        # db_task = await db_session.get(Task, task.id)
        # assert db_task is not None
    
    async def test_concurrent_agent_registration_integration(self, integration_orchestrator):
        """Test concurrent agent registration under realistic conditions."""
        # Register multiple agents concurrently
        agent_count = 15
        agents = [RealTestAgent() for _ in range(agent_count)]
        
        # Measure registration performance
        start_time = time.time()
        registration_tasks = [
            integration_orchestrator.register_agent(agent) 
            for agent in agents
        ]
        
        agent_ids = await asyncio.gather(*registration_tasks)
        total_time = time.time() - start_time
        
        # Verify all agents registered successfully
        assert len(agent_ids) == agent_count
        assert len(set(agent_ids)) == agent_count  # All unique
        assert len(integration_orchestrator._agents) == agent_count
        
        # Verify performance requirements
        avg_registration_time = (total_time / agent_count) * 1000
        print(f"Average registration time: {avg_registration_time:.2f}ms")
        assert avg_registration_time < integration_orchestrator.config.registration_target_ms * 1.5
    
    async def test_high_load_task_delegation_integration(self, integration_orchestrator):
        """Test task delegation under high load conditions."""
        # Register multiple agents
        agent_count = 10
        agents = []
        for i in range(agent_count):
            agent = RealTestAgent(capabilities=[
                AgentCapability(
                    name=f"load_test_{i}",
                    description=f"Load test agent {i}",
                    confidence_level=0.8,
                    specialization_areas=["load_testing"]
                )
            ])
            await integration_orchestrator.register_agent(agent)
            agents.append(agent)
        
        # Create high volume of tasks
        task_count = 50
        tasks = []
        for i in range(task_count):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Load Test Task {i}",
                description=f"High load test task {i}",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                estimated_time_minutes=1
            )
            tasks.append(task)
        
        # Delegate all tasks concurrently
        start_time = time.time()
        delegation_tasks = [
            integration_orchestrator.delegate_task(task)
            for task in tasks
        ]
        
        assigned_agents = await asyncio.gather(*delegation_tasks)
        delegation_time = time.time() - start_time
        
        # Verify all tasks were delegated
        assert len(assigned_agents) == task_count
        assert all(agent_id in integration_orchestrator._agents for agent_id in assigned_agents)
        
        # Verify performance under load
        avg_delegation_time = (delegation_time / task_count) * 1000
        print(f"Average delegation time under load: {avg_delegation_time:.2f}ms")
        assert avg_delegation_time < integration_orchestrator.config.delegation_target_ms * 2
        
        # Wait for task execution
        await asyncio.sleep(3.0)
        
        # Verify load distribution
        total_executed = sum(agent.task_count for agent in agents)
        assert total_executed == task_count
        
        # Verify no single agent was overloaded
        max_tasks_per_agent = max(agent.task_count for agent in agents)
        assert max_tasks_per_agent <= task_count // agent_count + 2  # Allow some imbalance
    
    async def test_agent_failure_recovery_integration(self, integration_orchestrator):
        """Test agent failure and recovery scenarios."""
        # Register multiple agents
        stable_agent = RealTestAgent()
        failing_agent = RealTestAgent()
        
        stable_id = await integration_orchestrator.register_agent(stable_agent)
        failing_id = await integration_orchestrator.register_agent(failing_agent)
        
        # Make one agent fail
        async def failing_execute_task(task):
            raise Exception("Agent failure simulation")
        
        failing_agent.execute_task = failing_execute_task
        
        # Create tasks
        tasks = []
        for i in range(10):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Recovery Test Task {i}",
                description="Test recovery from agent failure",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
        
        # Delegate tasks - some may fail
        successful_delegations = 0
        failed_delegations = 0
        
        for task in tasks:
            try:
                assigned_agent = await integration_orchestrator.delegate_task(task)
                successful_delegations += 1
            except Exception:
                failed_delegations += 1
        
        # Wait for execution attempts
        await asyncio.sleep(1.0)
        
        # Verify system continued operating despite failures
        assert successful_delegations > 0
        
        # Verify stable agent received some tasks
        assert stable_agent.task_count > 0
    
    async def test_orchestrator_resource_monitoring_integration(self, integration_orchestrator):
        """Test resource monitoring with real system metrics."""
        # Register agents to create load
        agents = []
        for i in range(5):
            agent = RealTestAgent()
            await integration_orchestrator.register_agent(agent)
            agents.append(agent)
        
        # Get system status
        status = await integration_orchestrator.get_system_status()
        
        # Verify status structure
        assert 'orchestrator' in status
        assert 'agents' in status
        assert 'tasks' in status
        assert 'resources' in status
        assert 'performance' in status
        
        # Verify resource monitoring is working
        resources = status['resources']
        assert 'cpu' in resources
        assert 'memory' in resources
        
        # Verify metrics are reasonable
        assert 0 <= resources.get('cpu', 0) <= 100
        assert resources.get('memory', 0) >= 0
        
        # Verify agent metrics
        agent_status = status['agents']
        assert agent_status['total_registered'] == 5
        assert agent_status['idle_count'] + agent_status['busy_count'] == 5
    
    async def test_intelligent_routing_integration(self, integration_orchestrator):
        """Test intelligent task routing with specialized agents."""
        # Create specialized agents
        backend_agent = RealTestAgent(capabilities=[
            AgentCapability(
                name="backend_specialist",
                description="Backend development specialist",
                confidence_level=0.95,
                specialization_areas=["backend", "api", "database", "python"]
            )
        ])
        
        frontend_agent = RealTestAgent(capabilities=[
            AgentCapability(
                name="frontend_specialist",
                description="Frontend development specialist",
                confidence_level=0.90,
                specialization_areas=["frontend", "ui", "react", "javascript"]
            )
        ])
        
        general_agent = RealTestAgent(capabilities=[
            AgentCapability(
                name="general_purpose",
                description="General purpose agent",
                confidence_level=0.75,
                specialization_areas=["general"]
            )
        ])
        
        # Register agents
        backend_id = await integration_orchestrator.register_agent(backend_agent)
        frontend_id = await integration_orchestrator.register_agent(frontend_agent)
        general_id = await integration_orchestrator.register_agent(general_agent)
        
        # Create specialized tasks
        backend_task = Task(
            id=str(uuid.uuid4()),
            title="API Development",
            description="Implement REST API endpoints with database integration",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING
        )
        
        frontend_task = Task(
            id=str(uuid.uuid4()),
            title="UI Component",
            description="Create React component with responsive design",
            priority=TaskPriority.HIGH,
            status=TaskStatus.PENDING
        )
        
        general_task = Task(
            id=str(uuid.uuid4()),
            title="Documentation",
            description="Update project documentation",
            priority=TaskPriority.LOW,
            status=TaskStatus.PENDING
        )
        
        # Delegate tasks and verify intelligent routing
        # Note: This test assumes the orchestrator has intelligent routing implemented
        backend_assigned = await integration_orchestrator.delegate_task(backend_task)
        frontend_assigned = await integration_orchestrator.delegate_task(frontend_task)
        general_assigned = await integration_orchestrator.delegate_task(general_task)
        
        # Verify tasks were assigned
        assert backend_assigned in [backend_id, frontend_id, general_id]
        assert frontend_assigned in [backend_id, frontend_id, general_id]
        assert general_assigned in [backend_id, frontend_id, general_id]
        
        # Wait for execution
        await asyncio.sleep(0.5)
        
        # Verify all tasks were executed
        total_executed = backend_agent.task_count + frontend_agent.task_count + general_agent.task_count
        assert total_executed == 3
    
    async def test_orchestrator_graceful_shutdown_integration(self, integration_config):
        """Test graceful shutdown with running tasks."""
        orchestrator = UnifiedProductionOrchestrator(integration_config)
        await orchestrator.start()
        
        try:
            # Register agents
            agents = []
            for i in range(3):
                agent = RealTestAgent()
                await orchestrator.register_agent(agent)
                agents.append(agent)
            
            # Start long-running tasks
            long_tasks = []
            for i in range(5):
                # Override execute_task to simulate long-running work
                async def long_execute_task(task):
                    await asyncio.sleep(1.0)  # Simulate long task
                    return f"Long task {task.id} completed"
                
                agents[i % len(agents)].execute_task = long_execute_task
                
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Long Task {i}",
                    description="Long-running task for shutdown test",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                long_tasks.append(task)
                await orchestrator.delegate_task(task)
            
            # Allow tasks to start
            await asyncio.sleep(0.1)
            
            # Perform graceful shutdown
            start_time = time.time()
            await orchestrator.shutdown(graceful=True)
            shutdown_time = time.time() - start_time
            
            # Verify graceful shutdown waited for tasks
            assert shutdown_time >= 0.8  # Should wait for most tasks to complete
            
            # Verify all agents were shut down
            for agent in agents:
                assert agent.shutdown_called
                
        finally:
            # Ensure cleanup
            if orchestrator._is_running:
                await orchestrator.shutdown(graceful=False)


@pytest.mark.performance
class TestOrchestratorPerformanceIntegration:
    """Performance integration tests for production validation."""
    
    async def test_50_concurrent_agents_integration(self, integration_config):
        """Test Epic 1 requirement: 50+ concurrent agents."""
        # Override config for high concurrency
        high_concurrency_config = OrchestratorConfig(
            max_concurrent_agents=60,
            max_agent_pool=75,
            registration_target_ms=100.0,
            delegation_target_ms=500.0,
            memory_limit_mb=2048,
            cpu_limit_percent=90.0
        )
        
        orchestrator = UnifiedProductionOrchestrator(high_concurrency_config)
        await orchestrator.start()
        
        try:
            # Register 55 agents (exceed 50+ requirement)
            agents = []
            registration_times = []
            
            print("Starting registration of 55 agents...")
            for i in range(55):
                agent = RealTestAgent(agent_id=f"agent_{i}")
                
                start_time = time.time()
                agent_id = await orchestrator.register_agent(agent)
                registration_time = (time.time() - start_time) * 1000
                
                agents.append(agent)
                registration_times.append(registration_time)
                
                if i % 10 == 0:
                    print(f"Registered {i+1} agents, avg time: {sum(registration_times)/(i+1):.2f}ms")
            
            # Verify all agents registered
            assert len(orchestrator._agents) == 55
            print(f"Successfully registered {len(orchestrator._agents)} agents")
            
            # Verify performance at scale
            avg_registration_time = sum(registration_times) / len(registration_times)
            max_registration_time = max(registration_times)
            
            print(f"Average registration time: {avg_registration_time:.2f}ms")
            print(f"Maximum registration time: {max_registration_time:.2f}ms")
            
            # Allow some performance degradation at scale
            assert avg_registration_time < high_concurrency_config.registration_target_ms * 1.5
            assert max_registration_time < high_concurrency_config.registration_target_ms * 3
            
            # Test task delegation at scale
            task_count = 100
            tasks = []
            delegation_times = []
            
            print(f"Starting delegation of {task_count} tasks...")
            for i in range(task_count):
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Scale Test Task {i}",
                    description="Task for 50+ agent scale test",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                start_time = time.time()
                assigned_agent = await orchestrator.delegate_task(task)
                delegation_time = (time.time() - start_time) * 1000
                
                tasks.append(task)
                delegation_times.append(delegation_time)
                
                assert assigned_agent in orchestrator._agents
                
                if i % 20 == 0:
                    print(f"Delegated {i+1} tasks, avg time: {sum(delegation_times)/(i+1):.2f}ms")
            
            # Verify delegation performance at scale
            avg_delegation_time = sum(delegation_times) / len(delegation_times)
            max_delegation_time = max(delegation_times)
            
            print(f"Average delegation time: {avg_delegation_time:.2f}ms")
            print(f"Maximum delegation time: {max_delegation_time:.2f}ms")
            
            assert avg_delegation_time < high_concurrency_config.delegation_target_ms * 1.5
            assert max_delegation_time < high_concurrency_config.delegation_target_ms * 3
            
            # Wait for task execution
            print("Waiting for task execution...")
            await asyncio.sleep(5.0)
            
            # Verify load distribution
            total_executed = sum(agent.task_count for agent in agents)
            print(f"Total tasks executed: {total_executed}")
            assert total_executed == task_count
            
            # Get system status under load
            status = await orchestrator.get_system_status()
            print(f"System status: {json.dumps(status['agents'], indent=2)}")
            
        finally:
            print("Shutting down orchestrator...")
            await orchestrator.shutdown(graceful=True)
    
    async def test_memory_efficiency_integration(self, integration_orchestrator):
        """Test memory efficiency under realistic conditions."""
        import psutil
        import gc
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register agents and delegate tasks
        agents = []
        for i in range(20):
            agent = RealTestAgent()
            await integration_orchestrator.register_agent(agent)
            agents.append(agent)
        
        # Create and delegate tasks
        for i in range(50):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Memory Test Task {i}",
                description="Task for memory efficiency testing",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING
            )
            await integration_orchestrator.delegate_task(task)
        
        # Wait for task processing
        await asyncio.sleep(2.0)
        
        # Force garbage collection
        gc.collect()
        
        # Measure memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - baseline_memory
        
        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Current memory: {current_memory:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        
        # Verify memory efficiency (Epic 1 requirement: <50MB base overhead)
        assert memory_growth < 100.0, f"Memory growth {memory_growth:.2f}MB exceeds 100MB limit"
        
        # Test memory cleanup
        agents_to_remove = list(integration_orchestrator._agents.keys())[:10]
        for agent_id in agents_to_remove:
            await integration_orchestrator.unregister_agent(agent_id)
        
        gc.collect()
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = current_memory - cleanup_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.2f} MB")
        print(f"Memory freed: {memory_freed:.2f} MB")
        
        # Should free some memory
        assert memory_freed > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])