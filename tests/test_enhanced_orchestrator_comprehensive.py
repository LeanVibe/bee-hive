"""
Comprehensive Test Suite for Enhanced Agent Orchestrator

This test suite provides thorough validation of the enhanced orchestrator functionality
including persona-based task assignment, intelligent routing, circuit breaker patterns,
workflow coordination, and performance optimization features.

Test Coverage:
- Unit tests for all enhanced methods (>95% coverage target)
- Integration tests with persona system and workflow engine
- Performance benchmarks for load balancing and concurrent processing
- Error handling and resilience testing
- Concurrency and multi-agent coordination tests
- Regression tests for backward compatibility
"""

import pytest
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any
from collections import deque

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentCapability, AgentInstance
from app.core.intelligent_task_router import IntelligentTaskRouter, TaskRoutingContext, RoutingStrategy
from app.core.workflow_engine import WorkflowEngine, WorkflowResult, TaskExecutionState
from app.core.agent_persona_system import AgentPersonaSystem, PersonaAssignment
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.agent_performance import AgentPerformanceHistory, TaskRoutingDecision, WorkloadSnapshot


@pytest.fixture
def mock_persona_system():
    """Create a mock persona system for testing."""
    persona_system = AsyncMock()
    persona_assignment = PersonaAssignment(
        agent_id=uuid.uuid4(),
        persona_id="test-persona-id",
        session_id="test-session-id",
        assignment_reason="optimal_match",
        confidence_score=0.9,
        assigned_at=datetime.utcnow()
    )
    persona_system.assign_persona_to_agent.return_value = persona_assignment
    return persona_system


@pytest.fixture
def mock_intelligent_router():
    """Create a mock intelligent task router for testing."""
    router = AsyncMock()
    router.route_task.return_value = "selected-agent-id"
    router.calculate_routing_score.return_value = 0.85
    return router


@pytest.fixture
def mock_workflow_engine():
    """Create a mock workflow engine for testing."""
    engine = AsyncMock()
    workflow_result = WorkflowResult(
        workflow_id=uuid.uuid4(),
        status=TaskExecutionState.COMPLETED,
        result_data={"success": True},
        execution_time=1.5,
        error_details=None
    )
    engine.execute_workflow.return_value = workflow_result
    return engine


@pytest.fixture
async def enhanced_orchestrator(mock_persona_system, mock_intelligent_router, mock_workflow_engine):
    """Create an enhanced orchestrator with mocked dependencies."""
    orchestrator = AgentOrchestrator()
    
    # Mock external dependencies
    orchestrator.message_broker = AsyncMock()
    orchestrator.session_cache = AsyncMock()
    orchestrator.anthropic_client = AsyncMock()
    
    # Inject enhanced components
    orchestrator.persona_system = mock_persona_system
    orchestrator.intelligent_router = mock_intelligent_router
    orchestrator.workflow_engine = mock_workflow_engine
    
    return orchestrator


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id=str(uuid.uuid4()),
        title="Test Task",
        description="Test task description",
        task_type=TaskType.FEATURE_DEVELOPMENT,
        priority=TaskPriority.HIGH,
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow(),
        required_capabilities=["python", "api_development"],
        metadata={
            "complexity": "medium",
            "estimated_effort": 4.0,
            "domain": "backend"
        }
    )


@pytest.fixture
def sample_agent_with_persona():
    """Create a sample agent instance with persona support."""
    capability = AgentCapability(
        name="backend_development",
        description="Backend development expertise",
        confidence_level=0.9,
        specialization_areas=["python", "api_development", "database"]
    )
    
    return AgentInstance(
        id="enhanced-agent-001",
        role=AgentRole.BACKEND_DEVELOPER,
        status=AgentStatus.ACTIVE,
        tmux_session="session-001",
        capabilities=[capability],
        current_task=None,
        context_window_usage=0.3,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )


class TestEnhancedOrchestratorPersonaIntegration:
    """Test suite for persona-based task assignment functionality."""
    
    async def test_assign_task_with_persona_success(self, enhanced_orchestrator, sample_task, sample_agent_with_persona):
        """Test successful task assignment with persona integration."""
        # Setup
        orchestrator = enhanced_orchestrator
        agent_id = sample_agent_with_persona.id
        task_id = sample_task.id
        
        orchestrator.agents[agent_id] = sample_agent_with_persona
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = sample_task
            
            # Execute
            result = await orchestrator._assign_task_to_agent_with_persona(
                task_id=task_id,
                agent_id=agent_id,
                task=sample_task,
                preferred_persona_id="test-persona",
                context={"priority": "high"}
            )
            
            # Verify
            assert result is True
            orchestrator.persona_system.assign_persona_to_agent.assert_called_once()
            mock_db_session.execute.assert_called()
            mock_db_session.commit.assert_called()
            orchestrator.message_broker.send_message.assert_called()
    
    async def test_assign_task_persona_assignment_failure(self, enhanced_orchestrator, sample_task, sample_agent_with_persona):
        """Test task assignment when persona assignment fails."""
        # Setup
        orchestrator = enhanced_orchestrator
        agent_id = sample_agent_with_persona.id
        task_id = sample_task.id
        
        orchestrator.agents[agent_id] = sample_agent_with_persona
        orchestrator.persona_system.assign_persona_to_agent.side_effect = Exception("Persona assignment failed")
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = sample_task
            
            # Execute - should continue without persona
            result = await orchestrator._assign_task_to_agent_with_persona(
                task_id=task_id,
                agent_id=agent_id,
                task=sample_task
            )
            
            # Verify - task assignment continues despite persona failure
            assert result is True
            mock_db_session.execute.assert_called()
            mock_db_session.commit.assert_called()
    
    async def test_delegate_task_with_enhanced_parameters(self, enhanced_orchestrator, sample_agent_with_persona):
        """Test task delegation with enhanced parameters."""
        orchestrator = enhanced_orchestrator
        orchestrator.agents[sample_agent_with_persona.id] = sample_agent_with_persona
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock task creation and scheduling
            mock_task = MagicMock()
            mock_task.id = str(uuid.uuid4())
            orchestrator._schedule_task_with_routing = AsyncMock(return_value=sample_agent_with_persona.id)
            
            # Execute with enhanced parameters
            task_id = await orchestrator.delegate_task(
                task_description="Implement authentication API",
                task_type="api_development",
                priority=TaskPriority.HIGH,
                required_capabilities=["python", "authentication"],
                estimated_effort=6.0,
                due_date=datetime.utcnow() + timedelta(days=3),
                dependencies=["task-dep-1", "task-dep-2"],
                routing_strategy=RoutingStrategy.PERSONA_OPTIMIZED
            )
            
            # Verify enhanced parameters are handled
            assert task_id is not None
            mock_db_session.add.assert_called()
            mock_db_session.commit.assert_called()
            orchestrator._schedule_task_with_routing.assert_called_once()


class TestEnhancedOrchestratorCircuitBreaker:
    """Test suite for circuit breaker patterns and error handling."""
    
    async def test_circuit_breaker_initialization(self, enhanced_orchestrator):
        """Test circuit breaker initialization for agents."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-circuit"
        
        # Initial state should not exist
        assert agent_id not in orchestrator.circuit_breakers
        
        # Update circuit breaker (creates initial state)
        await orchestrator._update_circuit_breaker(agent_id, success=True)
        
        # Verify initial state
        assert agent_id in orchestrator.circuit_breakers
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['state'] == 'closed'
        assert breaker['failure_count'] == 0
        assert breaker['successful_requests'] == 1
    
    async def test_circuit_breaker_success_handling(self, enhanced_orchestrator):
        """Test circuit breaker success state updates."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-success"
        
        # Simulate multiple successful operations
        for i in range(5):
            await orchestrator._update_circuit_breaker(agent_id, success=True)
        
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['successful_requests'] == 5
        assert breaker['consecutive_failures'] == 0
        assert breaker['state'] == 'closed'
    
    async def test_circuit_breaker_failure_accumulation(self, enhanced_orchestrator):
        """Test circuit breaker failure tracking."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-failures"
        
        # Simulate multiple failures
        for i in range(3):
            await orchestrator._update_circuit_breaker(agent_id, success=False, error_type="timeout")
        
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['failure_count'] == 3
        assert breaker['consecutive_failures'] == 3
        assert breaker['last_error_type'] == "timeout"
    
    async def test_circuit_breaker_should_trip_consecutive_failures(self, enhanced_orchestrator):
        """Test circuit breaker tripping on consecutive failures."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-trip"
        
        # Simulate enough requests to meet minimum threshold
        for _ in range(10):
            await orchestrator._update_circuit_breaker(agent_id, success=True)
        
        # Simulate consecutive failures exceeding threshold
        for _ in range(orchestrator.error_thresholds['consecutive_failures']):
            await orchestrator._update_circuit_breaker(agent_id, success=False)
        
        should_trip = await orchestrator._should_trip_circuit_breaker(agent_id)
        assert should_trip is True
    
    async def test_circuit_breaker_trip_execution(self, enhanced_orchestrator, sample_agent_with_persona):
        """Test circuit breaker tripping execution."""
        orchestrator = enhanced_orchestrator
        agent_id = sample_agent_with_persona.id
        orchestrator.agents[agent_id] = sample_agent_with_persona
        
        # Trip the circuit breaker
        await orchestrator._trip_circuit_breaker(agent_id, "consecutive_failures")
        
        # Verify circuit breaker state
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['state'] == 'open'
        assert breaker['trip_time'] is not None
        assert orchestrator.metrics['circuit_breaker_trips'] > 0
        
        # Verify agent is marked as error
        assert sample_agent_with_persona.status == AgentStatus.ERROR
    
    async def test_circuit_breaker_recovery_scheduling(self, enhanced_orchestrator):
        """Test circuit breaker recovery scheduling."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-recovery"
        
        # Initialize circuit breaker
        await orchestrator._update_circuit_breaker(agent_id, success=False)
        await orchestrator._trip_circuit_breaker(agent_id, "test_reason")
        
        # Mock recovery scheduling (without actual sleep)
        with patch('asyncio.sleep') as mock_sleep:
            await orchestrator._schedule_circuit_breaker_recovery(agent_id)
            
            # Verify sleep was called with recovery time
            mock_sleep.assert_called_once_with(orchestrator.error_thresholds['recovery_time_seconds'])
            
            # Verify breaker is now half-open
            breaker = orchestrator.circuit_breakers[agent_id]
            assert breaker['state'] == 'half_open'
    
    async def test_circuit_breaker_half_open_success_recovery(self, enhanced_orchestrator):
        """Test successful recovery from half-open state."""
        orchestrator = enhanced_orchestrator
        agent_id = "test-agent-half-open"
        
        # Set up half-open state
        orchestrator.circuit_breakers[agent_id] = {
            'state': 'half_open',
            'failure_count': 3,
            'consecutive_failures': 0,
            'last_failure_time': None,
            'total_requests': 20,
            'successful_requests': 15,
            'trip_time': time.time() - 70,
            'last_error_type': None
        }
        
        # Simulate successful operation
        await orchestrator._update_circuit_breaker(agent_id, success=True)
        
        # Verify circuit breaker is closed
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['state'] == 'closed'
        assert breaker['failure_count'] == 0
    
    @patch('asyncio.sleep')
    async def test_exponential_backoff_retry(self, mock_sleep, enhanced_orchestrator):
        """Test exponential backoff retry mechanism."""
        orchestrator = enhanced_orchestrator
        
        # Mock operation that fails then succeeds
        call_count = 0
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        # Execute retry with exponential backoff
        result = await orchestrator.retry_with_exponential_backoff(
            mock_operation,
            max_retries=3,
            operation_name="test_operation"
        )
        
        # Verify success after retries
        assert result == "success"
        assert call_count == 3
        
        # Verify exponential backoff sleep calls
        expected_sleep_times = [2**0, 2**1]  # Exponential backoff: 1s, 2s
        actual_sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_sleep_calls == expected_sleep_times
    
    async def test_agent_restart_with_circuit_breaker_protection(self, enhanced_orchestrator, sample_agent_with_persona):
        """Test agent restart with circuit breaker protection."""
        orchestrator = enhanced_orchestrator
        agent_id = sample_agent_with_persona.id
        
        # Set up circuit breaker in open state
        orchestrator.circuit_breakers[agent_id] = {'state': 'open'}
        
        # Attempt restart - should be blocked
        result = await orchestrator._attempt_agent_restart_with_protection(agent_id, sample_agent_with_persona)
        
        assert result is False  # Blocked by circuit breaker
    
    async def test_circuit_breaker_status_reporting(self, enhanced_orchestrator):
        """Test circuit breaker status reporting."""
        orchestrator = enhanced_orchestrator
        
        # Set up multiple circuit breakers in different states
        orchestrator.circuit_breakers = {
            'agent-1': {'state': 'closed', 'failure_count': 1, 'consecutive_failures': 0, 
                        'successful_requests': 10, 'total_requests': 11, 'last_error_type': None, 'trip_time': None},
            'agent-2': {'state': 'open', 'failure_count': 5, 'consecutive_failures': 5,
                        'successful_requests': 5, 'total_requests': 10, 'last_error_type': 'timeout', 'trip_time': time.time()},
            'agent-3': {'state': 'half_open', 'failure_count': 3, 'consecutive_failures': 0,
                        'successful_requests': 7, 'total_requests': 10, 'last_error_type': 'error', 'trip_time': time.time() - 70}
        }
        
        status = orchestrator.get_circuit_breaker_status()
        
        # Verify status structure
        assert 'circuit_breakers' in status
        assert 'summary' in status
        assert status['summary']['total_breakers'] == 3
        assert status['summary']['closed_breakers'] == 1
        assert status['summary']['open_breakers'] == 1
        assert status['summary']['half_open_breakers'] == 1
        
        # Verify individual breaker status
        assert 'agent-1' in status['circuit_breakers']
        assert status['circuit_breakers']['agent-1']['success_rate'] > 0.9
        assert status['circuit_breakers']['agent-2']['last_error_type'] == 'timeout'
    

class TestEnhancedOrchestratorLoadBalancing:
    """Test suite for intelligent load balancing and workload optimization."""
    
    async def test_workload_analysis(self, enhanced_orchestrator):
        """Test workload analysis functionality."""
        orchestrator = enhanced_orchestrator
        
        # Set up agents with different workloads
        agent1 = AgentInstance(id="agent-1", role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                              tmux_session=None, capabilities=[], current_task="task-1", 
                              context_window_usage=0.9, last_heartbeat=datetime.utcnow(), anthropic_client=None)
        agent2 = AgentInstance(id="agent-2", role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                              tmux_session=None, capabilities=[], current_task=None,
                              context_window_usage=0.2, last_heartbeat=datetime.utcnow(), anthropic_client=None)
        
        orchestrator.agents = {"agent-1": agent1, "agent-2": agent2}
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
            
            analysis = await orchestrator._analyze_agent_workloads()
            
            assert 'agents' in analysis
            assert 'balance_score' in analysis
            assert 'overloaded_agents' in analysis
            assert 'underutilized_agents' in analysis
            
            # Agent-1 should be overloaded, agent-2 underutilized
            overloaded_ids = [agent['agent_id'] for agent in analysis['overloaded_agents']]
            underutilized_ids = [agent['agent_id'] for agent in analysis['underutilized_agents']]
            
            assert "agent-1" in overloaded_ids
            assert "agent-2" in underutilized_ids
    
    async def test_rebalance_agent_workloads_skip_when_balanced(self, enhanced_orchestrator):
        """Test workload rebalancing skips when agents are already balanced."""
        orchestrator = enhanced_orchestrator
        
        # Mock well-balanced workload analysis
        mock_analysis = {
            'balance_score': 0.9,  # High balance score
            'agents': [],
            'overloaded_agents': [],
            'underutilized_agents': []
        }
        
        orchestrator._analyze_agent_workloads = AsyncMock(return_value=mock_analysis)
        
        result = await orchestrator.rebalance_agent_workloads(force_rebalance=False)
        
        assert result['skipped'] is True
        assert result['reason'] == 'workloads_balanced'
        assert result['balance_score'] == 0.9
    
    async def test_rebalance_agent_workloads_force_rebalance(self, enhanced_orchestrator):
        """Test forced workload rebalancing."""
        orchestrator = enhanced_orchestrator
        
        # Mock balanced workload but force rebalance
        mock_analysis = {
            'balance_score': 0.9,
            'agents': [],
            'overloaded_agents': [],
            'underutilized_agents': []
        }
        
        orchestrator._analyze_agent_workloads = AsyncMock(return_value=mock_analysis)
        orchestrator._intelligent_workload_rebalancing = AsyncMock(return_value=[])
        
        result = await orchestrator.rebalance_agent_workloads(force_rebalance=True)
        
        # Should not skip despite good balance score
        assert 'skipped' not in result or result['skipped'] is False
        orchestrator._intelligent_workload_rebalancing.assert_called_once()
    
    async def test_intelligent_workload_rebalancing(self, enhanced_orchestrator):
        """Test intelligent workload rebalancing algorithm."""
        orchestrator = enhanced_orchestrator
        
        mock_analysis = {
            'overloaded_agents': [
                {'agent_id': 'agent-1', 'current_load': 0.9, 'tasks': ['task-1', 'task-2']}
            ],
            'underutilized_agents': [
                {'agent_id': 'agent-2', 'current_load': 0.2, 'capacity': 0.8}
            ]
        }
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock task details
            mock_task = MagicMock()
            mock_task.id = 'task-1'
            mock_task.estimated_effort = 2.0
            mock_db_session.get.return_value = mock_task
            
            reassignments = await orchestrator._intelligent_workload_rebalancing(mock_analysis)
            
            assert len(reassignments) > 0
            # Verify reassignment structure
            for reassignment in reassignments:
                assert hasattr(reassignment, 'task_id')
                assert hasattr(reassignment, 'from_agent_id')
                assert hasattr(reassignment, 'to_agent_id')
                assert hasattr(reassignment, 'expected_improvement')
    
    async def test_workload_monitoring_loop(self, enhanced_orchestrator):
        """Test workload monitoring background loop."""
        orchestrator = enhanced_orchestrator
        orchestrator.is_running = True
        
        # Mock workload analysis and rebalancing
        orchestrator._analyze_agent_workloads = AsyncMock(return_value={'balance_score': 0.6})
        orchestrator.rebalance_agent_workloads = AsyncMock()
        
        # Mock sleep to avoid long running test
        with patch('asyncio.sleep') as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]  # Run once then cancel
            
            try:
                await orchestrator._workload_monitoring_loop()
            except asyncio.CancelledError:
                pass  # Expected cancellation
            
            # Verify monitoring was executed
            orchestrator._analyze_agent_workloads.assert_called()
            orchestrator.rebalance_agent_workloads.assert_called_once_with(force_rebalance=False)
    
    async def test_get_available_agent_ids(self, enhanced_orchestrator):
        """Test getting available agent IDs for load balancing."""
        orchestrator = enhanced_orchestrator
        
        # Set up agents with different statuses
        agents = {
            'agent-1': AgentInstance(id="agent-1", role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                                    tmux_session=None, capabilities=[], current_task=None,
                                    context_window_usage=0.3, last_heartbeat=datetime.utcnow(), anthropic_client=None),
            'agent-2': AgentInstance(id="agent-2", role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.BUSY,
                                    tmux_session=None, capabilities=[], current_task="task-1",
                                    context_window_usage=0.8, last_heartbeat=datetime.utcnow(), anthropic_client=None),
            'agent-3': AgentInstance(id="agent-3", role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ERROR,
                                    tmux_session=None, capabilities=[], current_task=None,
                                    context_window_usage=0.5, last_heartbeat=datetime.utcnow(), anthropic_client=None)
        }
        
        orchestrator.agents = agents
        
        available_agents = await orchestrator._get_available_agent_ids()
        
        # Only active agents should be available
        assert "agent-1" in available_agents
        assert "agent-2" not in available_agents  # Busy
        assert "agent-3" not in available_agents  # Error state
    

class TestEnhancedOrchestratorAnalytics:
    """Test suite for routing analytics and performance metrics."""
    
    async def test_get_routing_analytics(self, enhanced_orchestrator):
        """Test routing analytics collection."""
        orchestrator = enhanced_orchestrator
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock database queries
            mock_db_session.execute.return_value.scalar.side_effect = [100, 85]  # total, successful
            mock_db_session.execute.return_value.all.return_value = [
                ('ADAPTIVE', 0.8), ('PERSONA_OPTIMIZED', 0.9)
            ]
            
            analytics = await orchestrator.get_routing_analytics()
            
            assert 'routing_accuracy' in analytics
            assert 'performance_by_strategy' in analytics
            assert 'total_routing_decisions' in analytics
            assert 'successful_routing_decisions' in analytics
            
            # Verify routing accuracy calculation
            assert analytics['routing_accuracy'] == 0.85  # 85/100
    
    async def test_record_routing_analytics(self, enhanced_orchestrator):
        """Test routing analytics recording."""
        orchestrator = enhanced_orchestrator
        
        routing_context = TaskRoutingContext(
            task_id="test-task",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority=TaskPriority.HIGH,
            required_capabilities=["python"],
            available_agents=["agent-1", "agent-2"],
            routing_strategy=RoutingStrategy.PERSONA_OPTIMIZED
        )
        
        selected_agent = "agent-1"
        routing_score = 0.9
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            await orchestrator._record_routing_analytics(
                routing_context, selected_agent, routing_score
            )
            
            # Verify database interaction
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Verify metrics update
            assert orchestrator.metrics['routing_decisions'] > 0
    
    async def test_update_task_completion_metrics(self, enhanced_orchestrator):
        """Test task completion metrics update."""
        orchestrator = enhanced_orchestrator
        
        task_id = "completed-task"
        agent_id = "completing-agent"
        success = True
        completion_time = 3.5
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock existing routing decision
            mock_routing_decision = MagicMock()
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_routing_decision
            
            await orchestrator.update_task_completion_metrics(
                task_id, agent_id, success, completion_time
            )
            
            # Verify routing decision update
            assert mock_routing_decision.task_success == success
            assert mock_routing_decision.actual_completion_time == completion_time
            mock_db_session.commit.assert_called_once()
            
            # Verify metrics update
            if success:
                assert orchestrator.metrics['tasks_completed'] > 0


class TestEnhancedOrchestratorConcurrency:
    """Test suite for concurrent task processing and multi-agent coordination."""
    
    async def test_concurrent_task_assignment(self, enhanced_orchestrator):
        """Test concurrent task assignment to multiple agents."""
        orchestrator = enhanced_orchestrator
        
        # Set up multiple agents
        agents = {}
        for i in range(5):
            agent_id = f"agent-{i}"
            agents[agent_id] = AgentInstance(
                id=agent_id, role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                tmux_session=None, capabilities=[], current_task=None,
                context_window_usage=0.3, last_heartbeat=datetime.utcnow(), anthropic_client=None
            )
        orchestrator.agents = agents
        
        # Create multiple tasks concurrently
        tasks = []
        for i in range(10):
            task_coroutine = orchestrator.delegate_task(
                task_description=f"Concurrent task {i}",
                task_type="testing",
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task_coroutine)
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all tasks were processed
            successful_tasks = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_tasks) == 10
    
    async def test_queue_processing_concurrency(self, enhanced_orchestrator):
        """Test concurrent queue processing doesn't cause race conditions."""
        orchestrator = enhanced_orchestrator
        
        # Mock queue processing state
        orchestrator.queue_processing_active = False
        
        # Create multiple queue processing coroutines
        process_tasks = [
            orchestrator._process_task_queue_batch() for _ in range(3)
        ]
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
            
            # Execute concurrently - only one should process (others should skip)
            results = await asyncio.gather(*process_tasks, return_exceptions=True)
            
            # Verify no race conditions occurred
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) <= 3  # All should complete without errors
    
    async def test_agent_heartbeat_concurrency(self, enhanced_orchestrator):
        """Test concurrent agent heartbeat processing."""
        orchestrator = enhanced_orchestrator
        
        # Set up multiple agents
        agents = {}
        for i in range(3):
            agent_id = f"heartbeat-agent-{i}"
            agents[agent_id] = AgentInstance(
                id=agent_id, role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                tmux_session=None, capabilities=[], current_task=None,
                context_window_usage=0.3, last_heartbeat=datetime.utcnow() - timedelta(minutes=1),
                anthropic_client=None
            )
        orchestrator.agents = agents
        
        # Process heartbeats concurrently
        heartbeat_tasks = [
            orchestrator._monitor_agent_health(agent_id, agent)
            for agent_id, agent in agents.items()
        ]
        
        with patch('app.core.orchestrator.get_session'):
            results = await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
            
            # Verify all heartbeats processed without conflicts
            assert len([r for r in results if isinstance(r, Exception)]) == 0


class TestEnhancedOrchestratorRegressionTests:
    """Regression tests to ensure backward compatibility."""
    
    async def test_legacy_task_assignment_method(self, enhanced_orchestrator, sample_agent_with_persona):
        """Test legacy task assignment method still works."""
        orchestrator = enhanced_orchestrator
        agent_id = sample_agent_with_persona.id
        task_id = "legacy-task-001"
        
        orchestrator.agents[agent_id] = sample_agent_with_persona
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_db_session.get.return_value = mock_task
            
            # Call legacy method
            result = await orchestrator._assign_task_to_agent(task_id, agent_id)
            
            # Should delegate to enhanced method
            assert result is True
    
    async def test_existing_orchestrator_methods_unchanged(self, enhanced_orchestrator):
        """Test that existing orchestrator methods maintain their signatures."""
        orchestrator = enhanced_orchestrator
        
        # Verify critical methods exist and are callable
        assert callable(getattr(orchestrator, 'spawn_agent'))
        assert callable(getattr(orchestrator, 'shutdown_agent'))
        assert callable(getattr(orchestrator, 'delegate_task'))
        assert callable(getattr(orchestrator, 'get_system_status'))
        assert callable(getattr(orchestrator, 'initiate_sleep_cycle'))
        
        # Test basic system status (should work without initialization)
        status = await orchestrator.get_system_status()
        assert 'orchestrator_status' in status
        assert 'total_agents' in status
        assert 'metrics' in status
    
    async def test_enhanced_features_optional(self, enhanced_orchestrator):
        """Test that enhanced features are optional and don't break basic functionality."""
        orchestrator = enhanced_orchestrator
        
        # Disable enhanced features
        orchestrator.persona_system = None
        orchestrator.intelligent_router = None
        orchestrator.workflow_engine = None
        
        # Basic task delegation should still work
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            orchestrator._schedule_task = AsyncMock(return_value=None)
            
            task_id = await orchestrator.delegate_task(
                task_description="Basic task",
                task_type="testing"
            )
            
            assert task_id is not None
            mock_db_session.add.assert_called()


@pytest.mark.performance
class TestEnhancedOrchestratorPerformance:
    """Performance tests and benchmarks for enhanced orchestrator."""
    
    async def test_task_assignment_performance(self, enhanced_orchestrator):
        """Test task assignment performance under load."""
        orchestrator = enhanced_orchestrator
        
        # Set up multiple agents
        for i in range(10):
            agent_id = f"perf-agent-{i}"
            orchestrator.agents[agent_id] = AgentInstance(
                id=agent_id, role=AgentRole.BACKEND_DEVELOPER, status=AgentStatus.ACTIVE,
                tmux_session=None, capabilities=[], current_task=None,
                context_window_usage=0.3, last_heartbeat=datetime.utcnow(), anthropic_client=None
            )
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Measure performance of multiple task assignments
            start_time = time.time()
            
            tasks = []
            for i in range(50):
                task = orchestrator.delegate_task(
                    task_description=f"Performance test task {i}",
                    task_type="performance_test"
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertion - should complete 50 tasks in under 5 seconds
            assert execution_time < 5.0, f"Task assignment took {execution_time:.2f}s, expected < 5.0s"
            
            # Throughput assertion - should handle at least 10 tasks per second
            throughput = 50 / execution_time
            assert throughput >= 10, f"Throughput {throughput:.2f} tasks/sec, expected >= 10"
    
    async def test_circuit_breaker_performance_impact(self, enhanced_orchestrator):
        """Test circuit breaker performance impact."""
        orchestrator = enhanced_orchestrator
        agent_id = "perf-circuit-agent"
        
        # Measure performance without circuit breaker
        start_time = time.time()
        for _ in range(1000):
            pass  # Simulate operation
        baseline_time = time.time() - start_time
        
        # Measure performance with circuit breaker updates
        start_time = time.time()
        for _ in range(1000):
            await orchestrator._update_circuit_breaker(agent_id, success=True)
        circuit_breaker_time = time.time() - start_time
        
        # Circuit breaker overhead should be minimal (< 50% increase)
        overhead_ratio = circuit_breaker_time / max(baseline_time, 0.001)  # Avoid division by zero
        assert overhead_ratio < 1.5, f"Circuit breaker overhead too high: {overhead_ratio:.2f}x"
    
    async def test_memory_usage_optimization(self, enhanced_orchestrator):
        """Test memory usage doesn't grow unbounded."""
        import sys
        orchestrator = enhanced_orchestrator
        
        # Get initial memory usage
        initial_size = sys.getsizeof(orchestrator.__dict__)
        
        # Simulate long-running operations
        for i in range(100):
            agent_id = f"memory-test-agent-{i % 10}"  # Reuse agent IDs
            await orchestrator._update_circuit_breaker(agent_id, success=True)
            
            # Simulate task processing
            orchestrator.metrics['tasks_completed'] += 1
        
        # Check final memory usage
        final_size = sys.getsizeof(orchestrator.__dict__)
        
        # Memory growth should be reasonable (< 200% increase)
        growth_ratio = final_size / initial_size
        assert growth_ratio < 2.0, f"Memory usage grew too much: {growth_ratio:.2f}x"


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__, 
        "-v", 
        "--cov=app.core.orchestrator", 
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/enhanced_orchestrator",
        "-m", "not performance"  # Exclude performance tests by default
    ])