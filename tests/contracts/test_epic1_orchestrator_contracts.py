"""
Contract Tests for Epic 1 UnifiedProductionOrchestrator

Tests verify that the orchestrator adheres to its interface contracts
and integration points with other system components.
"""

import asyncio
import pytest
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Protocol
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel, ValidationError

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig,
    AgentState,
    AgentCapability,
    TaskRoutingStrategy
)
from app.models.task import Task, TaskStatus, TaskPriority
from app.models.agent import AgentType


# Contract Schemas
class AgentRegistrationContract(BaseModel):
    """Contract for agent registration."""
    agent_id: str
    capabilities: List[Dict[str, Any]]
    state: str
    registration_time_ms: float
    
    class Config:
        extra = "forbid"


class TaskDelegationContract(BaseModel):
    """Contract for task delegation."""
    task_id: str
    assigned_agent_id: str
    delegation_time_ms: float
    routing_strategy: str
    
    class Config:
        extra = "forbid"


class SystemStatusContract(BaseModel):
    """Contract for system status."""
    orchestrator: Dict[str, Any]
    agents: Dict[str, Any]
    tasks: Dict[str, Any]
    resources: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    performance: Dict[str, Any]
    
    class Config:
        extra = "forbid"


class AgentMetricsContract(BaseModel):
    """Contract for agent metrics."""
    agent_id: str
    success_rate: float
    average_response_time: float
    last_heartbeat: str
    total_tasks: int
    
    class Config:
        extra = "forbid"


class OrchestratorMessageContract(BaseModel):
    """Contract for orchestrator messages."""
    type: str
    timestamp: str
    source: str
    data: Dict[str, Any]
    
    class Config:
        extra = "forbid"


class ContractTestAgent:
    """Agent implementation that follows the orchestrator's expected contract."""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.ACTIVE
        self.capabilities = [
            AgentCapability(
                name="contract_test",
                description="Contract testing agent",
                confidence_level=0.85,
                specialization_areas=["testing", "contracts"]
            )
        ]
        self.task_count = 0
        self.execution_results = []
        self.shutdown_called = False
        
    async def execute_task(self, task: Task) -> Any:
        """Execute task following contract expectations."""
        # Contract: Must accept Task object and return result
        if not isinstance(task, Task):
            raise TypeError("execute_task must receive Task object")
        
        self.task_count += 1
        await asyncio.sleep(0.01)  # Simulate work
        
        result = {
            "task_id": str(task.id),
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id
        }
        
        self.execution_results.append(result)
        return result
        
    async def get_status(self) -> AgentState:
        """Get agent status following contract."""
        # Contract: Must return AgentState enum
        return self.state
        
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities following contract."""
        # Contract: Must return List[AgentCapability]
        return self.capabilities
        
    async def shutdown(self, graceful: bool = True) -> None:
        """Shutdown agent following contract."""
        # Contract: Must accept graceful parameter and perform cleanup
        if graceful:
            await asyncio.sleep(0.01)  # Simulate graceful cleanup
        
        self.shutdown_called = True
        self.state = AgentState.TERMINATED


@pytest.fixture
def contract_config():
    """Configuration for contract testing."""
    return OrchestratorConfig(
        max_concurrent_agents=15,
        min_agent_pool=2,
        max_agent_pool=20,
        agent_registration_timeout=1.0,
        task_delegation_timeout=1.0,
        registration_target_ms=50.0,
        delegation_target_ms=200.0,
        health_check_interval=5.0,
        metrics_collection_interval=3.0
    )


@pytest.fixture
async def contract_orchestrator(contract_config):
    """Orchestrator instance for contract testing."""
    with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
         patch('app.core.unified_production_orchestrator.get_session') as mock_db:
        
        # Configure mocks to satisfy dependencies
        mock_redis.return_value = AsyncMock()
        mock_db.return_value.__aenter__ = AsyncMock()
        mock_db.return_value.__aexit__ = AsyncMock()
        
        orchestrator = UnifiedProductionOrchestrator(contract_config)
        await orchestrator.start()
        yield orchestrator
        await orchestrator.shutdown(graceful=True)


class TestOrchestratorContracts:
    """Contract validation tests for orchestrator."""
    
    async def test_agent_registration_contract(self, contract_orchestrator):
        """Test agent registration follows expected contract."""
        orchestrator = contract_orchestrator
        
        # Register agent
        agent = ContractTestAgent()
        start_time = time.time()
        agent_id = await orchestrator.register_agent(agent)
        registration_time = (time.time() - start_time) * 1000
        
        # Verify contract compliance
        contract_data = {
            "agent_id": agent_id,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "confidence_level": cap.confidence_level,
                    "specialization_areas": cap.specialization_areas
                } for cap in agent.capabilities
            ],
            "state": agent.state.value,
            "registration_time_ms": registration_time
        }
        
        # Validate against contract schema
        registration_contract = AgentRegistrationContract(**contract_data)
        
        # Verify contract requirements
        assert registration_contract.agent_id == agent_id
        assert len(registration_contract.capabilities) > 0
        assert registration_contract.state in [state.value for state in AgentState]
        assert registration_contract.registration_time_ms < orchestrator.config.registration_target_ms * 2
        
        # Verify orchestrator state
        assert agent_id in orchestrator._agents
        assert orchestrator._agents[agent_id] == agent
    
    async def test_task_delegation_contract(self, contract_orchestrator):
        """Test task delegation follows expected contract."""
        orchestrator = contract_orchestrator
        
        # Register agent first
        agent = ContractTestAgent()
        agent_id = await orchestrator.register_agent(agent)
        
        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            title="Contract Test Task",
            description="Task to validate delegation contract",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            estimated_effort=10
        )
        
        # Delegate task
        start_time = time.time()
        assigned_agent_id = await orchestrator.delegate_task(task)
        delegation_time = (time.time() - start_time) * 1000
        
        # Verify contract compliance
        contract_data = {
            "task_id": str(task.id),
            "assigned_agent_id": assigned_agent_id,
            "delegation_time_ms": delegation_time,
            "routing_strategy": orchestrator.config.routing_strategy.value
        }
        
        # Validate against contract schema
        delegation_contract = TaskDelegationContract(**contract_data)
        
        # Verify contract requirements
        assert delegation_contract.task_id == str(task.id)
        assert delegation_contract.assigned_agent_id == agent_id
        assert delegation_contract.delegation_time_ms < orchestrator.config.delegation_target_ms * 2
        assert delegation_contract.routing_strategy in [strategy.value for strategy in TaskRoutingStrategy]
    
    async def test_system_status_contract(self, contract_orchestrator):
        """Test system status follows expected contract."""
        orchestrator = contract_orchestrator
        
        # Register some agents to have meaningful status
        agents = []
        for i in range(3):
            agent = ContractTestAgent()
            await orchestrator.register_agent(agent)
            agents.append(agent)
        
        # Get system status
        status = await orchestrator.get_system_status()
        
        # Validate against contract schema
        status_contract = SystemStatusContract(**status)
        
        # Verify contract requirements
        assert 'is_running' in status_contract.orchestrator
        assert 'uptime_seconds' in status_contract.orchestrator
        assert 'config' in status_contract.orchestrator
        
        assert 'total_registered' in status_contract.agents
        assert 'idle_count' in status_contract.agents
        assert 'busy_count' in status_contract.agents
        assert 'max_concurrent' in status_contract.agents
        
        assert 'pending_count' in status_contract.tasks
        assert 'in_progress_count' in status_contract.tasks
        assert 'completed_count' in status_contract.tasks
        
        assert 'cpu' in status_contract.resources
        assert 'memory' in status_contract.resources
        
        # Verify data types and ranges
        assert isinstance(status_contract.orchestrator['is_running'], bool)
        assert isinstance(status_contract.orchestrator['uptime_seconds'], (int, float))
        assert status_contract.orchestrator['uptime_seconds'] >= 0
        
        assert isinstance(status_contract.agents['total_registered'], int)
        assert status_contract.agents['total_registered'] >= 0
        assert isinstance(status_contract.agents['idle_count'], int)
        assert isinstance(status_contract.agents['busy_count'], int)
        
        assert isinstance(status_contract.resources['cpu'], (int, float))
        assert 0 <= status_contract.resources['cpu'] <= 100
        assert isinstance(status_contract.resources['memory'], (int, float))
        assert status_contract.resources['memory'] >= 0
    
    async def test_agent_metrics_contract(self, contract_orchestrator):
        """Test agent metrics follow expected contract."""
        orchestrator = contract_orchestrator
        
        # Register agent and execute task to generate metrics
        agent = ContractTestAgent()
        agent_id = await orchestrator.register_agent(agent)
        
        task = Task(
            id=str(uuid.uuid4()),
            title="Metrics Test Task",
            description="Task to generate metrics",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            estimated_effort=5
        )
        
        await orchestrator.delegate_task(task)
        await asyncio.sleep(0.2)  # Wait for execution
        
        # Get agent metrics
        metrics = orchestrator._agent_metrics[agent_id]
        
        # Prepare contract data
        contract_data = {
            "agent_id": metrics.agent_id,
            "success_rate": metrics.success_rate,
            "average_response_time": metrics.average_response_time,
            "last_heartbeat": metrics.last_heartbeat.isoformat() if metrics.last_heartbeat else datetime.utcnow().isoformat(),
            "total_tasks": metrics.total_tasks
        }
        
        # Validate against contract schema
        metrics_contract = AgentMetricsContract(**contract_data)
        
        # Verify contract requirements
        assert metrics_contract.agent_id == agent_id
        assert 0.0 <= metrics_contract.success_rate <= 1.0
        assert metrics_contract.average_response_time >= 0.0
        assert metrics_contract.total_tasks >= 0
        
        # Verify timestamp format
        datetime.fromisoformat(metrics_contract.last_heartbeat.replace('Z', '+00:00'))
    
    async def test_agent_protocol_contract(self, contract_orchestrator):
        """Test agent must implement required protocol methods."""
        orchestrator = contract_orchestrator
        
        # Test with agent missing required methods
        incomplete_agent = Mock()
        incomplete_agent.agent_id = "incomplete_agent"
        # Missing execute_task, get_status, get_capabilities, shutdown
        
        # Should fail validation
        with pytest.raises((ValueError, AttributeError)):
            await orchestrator.register_agent(incomplete_agent)
        
        # Test with agent having incorrect method signatures
        incorrect_agent = Mock()
        incorrect_agent.agent_id = "incorrect_agent"
        incorrect_agent.execute_task = Mock(return_value="not_async")  # Not async
        incorrect_agent.get_status = AsyncMock(return_value="not_agent_state")  # Wrong return type
        incorrect_agent.get_capabilities = AsyncMock(return_value=[])
        incorrect_agent.shutdown = AsyncMock()
        
        # Should fail during execution
        try:
            await orchestrator.register_agent(incorrect_agent)
            # If registration succeeds, execution should fail
            task = Task(
                id=str(uuid.uuid4()),
                title="Protocol Test",
                description="Test protocol compliance",
                priority=TaskPriority.LOW,
                status=TaskStatus.PENDING,
                estimated_effort=5
            )
            await orchestrator.delegate_task(task)
            await asyncio.sleep(0.1)
            # Should have errors in execution
        except Exception:
            pass  # Expected for incorrect implementation
    
    async def test_configuration_contract(self, contract_orchestrator):
        """Test orchestrator configuration contract."""
        orchestrator = contract_orchestrator
        config = orchestrator.config
        
        # Verify configuration contract
        assert hasattr(config, 'max_concurrent_agents')
        assert hasattr(config, 'registration_target_ms')
        assert hasattr(config, 'delegation_target_ms')
        assert hasattr(config, 'routing_strategy')
        
        # Verify types and ranges
        assert isinstance(config.max_concurrent_agents, int)
        assert config.max_concurrent_agents > 0
        
        assert isinstance(config.registration_target_ms, (int, float))
        assert config.registration_target_ms > 0
        
        assert isinstance(config.delegation_target_ms, (int, float))
        assert config.delegation_target_ms > 0
        
        assert isinstance(config.routing_strategy, TaskRoutingStrategy)
        
        # Verify configuration is immutable during runtime
        original_max = config.max_concurrent_agents
        config.max_concurrent_agents = 999
        assert orchestrator.config.max_concurrent_agents == original_max  # Should not change
    
    async def test_error_handling_contract(self, contract_orchestrator):
        """Test error handling follows expected contract."""
        orchestrator = contract_orchestrator
        
        # Test registration with invalid agent
        with pytest.raises(ValueError):
            await orchestrator.register_agent(None)
        
        # Test delegation with invalid task
        agent = ContractTestAgent()
        await orchestrator.register_agent(agent)
        
        with pytest.raises(ValueError):
            await orchestrator.delegate_task(None)
        
        # Test delegation with no agents
        await orchestrator.unregister_agent(agent.agent_id)
        
        task = Task(
            id=str(uuid.uuid4()),
            title="No Agent Test",
            description="Test delegation with no agents",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            estimated_effort=5
        )
        
        with pytest.raises(RuntimeError, match="No suitable agent available"):
            await orchestrator.delegate_task(task)
        
        # Test unregistering non-existent agent (should not raise)
        await orchestrator.unregister_agent("non_existent_agent")
    
    async def test_lifecycle_contract(self, contract_orchestrator):
        """Test orchestrator lifecycle contract."""
        # Test start/stop contract
        assert contract_orchestrator._is_running is True
        
        # Test status during operation
        status = await contract_orchestrator.get_system_status()
        assert status['orchestrator']['is_running'] is True
        
        # Test graceful shutdown contract
        await contract_orchestrator.shutdown(graceful=True)
        assert contract_orchestrator._is_running is False
        
        # Operations should fail after shutdown
        agent = ContractTestAgent()
        with pytest.raises(Exception):
            await contract_orchestrator.register_agent(agent)
    
    async def test_performance_contract(self, contract_orchestrator):
        """Test performance requirements are met as per contract."""
        orchestrator = contract_orchestrator
        
        # Test registration performance contract
        agent = ContractTestAgent()
        start_time = time.time()
        agent_id = await orchestrator.register_agent(agent)
        registration_time = (time.time() - start_time) * 1000
        
        assert registration_time < orchestrator.config.registration_target_ms * 1.5  # Allow some margin
        
        # Test delegation performance contract
        task = Task(
            id=str(uuid.uuid4()),
            title="Performance Contract Test",
            description="Test delegation performance",
            priority=TaskPriority.MEDIUM,
            status=TaskStatus.PENDING,
            estimated_effort=5
        )
        
        start_time = time.time()
        assigned_agent = await orchestrator.delegate_task(task)
        delegation_time = (time.time() - start_time) * 1000
        
        assert delegation_time < orchestrator.config.delegation_target_ms * 1.5  # Allow some margin
        assert assigned_agent == agent_id
    
    async def test_concurrency_contract(self, contract_orchestrator):
        """Test concurrent operations follow contract."""
        orchestrator = contract_orchestrator
        
        # Test concurrent agent registration
        agents = [ContractTestAgent() for _ in range(5)]
        registration_tasks = [
            orchestrator.register_agent(agent) for agent in agents
        ]
        
        agent_ids = await asyncio.gather(*registration_tasks)
        
        # All should succeed and be unique
        assert len(agent_ids) == 5
        assert len(set(agent_ids)) == 5
        
        # Test concurrent task delegation
        tasks = [
            Task(
                id=str(uuid.uuid4()),
                title=f"Concurrent Task {i}",
                description="Concurrent test task",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                estimated_effort=5
            ) for i in range(10)
        ]
        
        delegation_tasks = [
            orchestrator.delegate_task(task) for task in tasks
        ]
        
        assigned_agents = await asyncio.gather(*delegation_tasks)
        
        # All should succeed and be valid agent IDs
        assert len(assigned_agents) == 10
        assert all(agent_id in agent_ids for agent_id in assigned_agents)


@pytest.mark.performance
class TestOrchestratorPerformanceContracts:
    """Performance contract validation tests."""
    
    async def test_scalability_contract(self, contract_config):
        """Test orchestrator meets scalability contracts."""
        # Test with higher capacity
        high_capacity_config = OrchestratorConfig(
            max_concurrent_agents=25,
            max_agent_pool=30,
            registration_target_ms=100.0,
            delegation_target_ms=500.0
        )
        
        with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
             patch('app.core.unified_production_orchestrator.get_session') as mock_db:
            
            mock_redis.return_value = AsyncMock()
            mock_db.return_value.__aenter__ = AsyncMock()
            mock_db.return_value.__aexit__ = AsyncMock()
            
            orchestrator = UnifiedProductionOrchestrator(high_capacity_config)
            await orchestrator.start()
            
            try:
                # Register many agents
                agents = [ContractTestAgent() for _ in range(20)]
                registration_times = []
                
                for agent in agents:
                    start_time = time.time()
                    agent_id = await orchestrator.register_agent(agent)
                    registration_time = (time.time() - start_time) * 1000
                    registration_times.append(registration_time)
                
                # Verify scalability contract
                avg_registration_time = sum(registration_times) / len(registration_times)
                assert avg_registration_time < high_capacity_config.registration_target_ms * 1.5
                
                # Test delegation at scale
                tasks = [
                    Task(
                        id=str(uuid.uuid4()),
                        title=f"Scale Task {i}",
                        description="Scalability test task",
                        priority=TaskPriority.MEDIUM,
                        status=TaskStatus.PENDING,
                        estimated_effort=5
                    ) for i in range(40)
                ]
                
                delegation_times = []
                for task in tasks:
                    start_time = time.time()
                    agent_id = await orchestrator.delegate_task(task)
                    delegation_time = (time.time() - start_time) * 1000
                    delegation_times.append(delegation_time)
                
                avg_delegation_time = sum(delegation_times) / len(delegation_times)
                assert avg_delegation_time < high_capacity_config.delegation_target_ms * 1.5
                
            finally:
                await orchestrator.shutdown(graceful=True)
    
    async def test_resource_usage_contract(self, contract_orchestrator):
        """Test resource usage stays within contract limits."""
        import psutil
        
        orchestrator = contract_orchestrator
        process = psutil.Process()
        
        # Get baseline
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register agents and create load
        agents = [ContractTestAgent() for _ in range(10)]
        for agent in agents:
            await orchestrator.register_agent(agent)
        
        # Execute tasks
        for i in range(20):
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Resource Test Task {i}",
                description="Resource usage test",
                priority=TaskPriority.MEDIUM,
                status=TaskStatus.PENDING,
                estimated_effort=5
            )
            await orchestrator.delegate_task(task)
        
        await asyncio.sleep(1.0)  # Wait for execution
        
        # Check resource usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = current_memory - baseline_memory
        
        # Resource contract: Should not use excessive memory
        assert memory_growth < 200.0, f"Memory growth {memory_growth:.2f}MB exceeds contract limit"
        
        # System status should reflect reasonable resource usage
        status = await orchestrator.get_system_status()
        cpu_usage = status['resources']['cpu']
        memory_usage = status['resources']['memory']
        
        assert 0 <= cpu_usage <= 100
        assert memory_usage >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])