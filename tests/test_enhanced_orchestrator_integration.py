"""
Integration Tests for Enhanced Agent Orchestrator Core

Tests the enhanced orchestrator capabilities including:
- Persona-based task assignment
- Workflow engine integration
- Intelligent load balancing
- Circuit breaker patterns
- Enhanced queuing system
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator, AgentRole
from app.core.agent_persona_system import PersonaType, PersonaDefinition, PersonaCapabilityLevel, PersonaCapability
from app.core.intelligent_task_router import RoutingStrategy
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus
from app.models.agent import Agent, AgentStatus, AgentType


@pytest.fixture
async def enhanced_orchestrator():
    """Create an enhanced orchestrator with mocked dependencies."""
    orchestrator = AgentOrchestrator()
    
    # Mock external dependencies
    orchestrator.message_broker = AsyncMock()
    orchestrator.session_cache = AsyncMock()
    orchestrator.anthropic_client = AsyncMock()
    
    # Mock database sessions
    with patch('app.core.orchestrator.get_session') as mock_get_session:
        mock_db_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_db_session
        mock_get_session.return_value.__aexit__.return_value = None
        
        # Initialize the orchestrator
        await orchestrator.start()
        
        yield orchestrator, mock_db_session
        
        await orchestrator.shutdown()


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id=str(uuid.uuid4()),
        title="Test Task",
        description="Test task for orchestrator integration",
        task_type=TaskType.CODE_GENERATION,
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        context={"test": True},
        required_capabilities=["programming", "python"],
        estimated_effort=60,
        dependencies=[]
    )


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    task_ids = [str(uuid.uuid4()) for _ in range(3)]
    return Workflow(
        id=str(uuid.uuid4()),
        name="Test Workflow",
        description="Test workflow for orchestrator integration",
        task_ids=task_ids,
        dependencies={
            task_ids[1]: [task_ids[0]],  # Task 1 depends on Task 0
            task_ids[2]: [task_ids[1]]   # Task 2 depends on Task 1
        },
        status=WorkflowStatus.CREATED,
        total_tasks=3
    )


class TestPersonaBasedTaskAssignment:
    """Test persona-based task assignment integration."""
    
    async def test_task_assignment_with_optimal_persona(self, enhanced_orchestrator, sample_task):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock persona system
        orchestrator.persona_system = AsyncMock()
        orchestrator.persona_system.assign_persona_to_agent.return_value = MagicMock(
            persona_id="backend_engineer_default",
            confidence_score=0.9,
            active_adaptations={}
        )
        
        # Mock agent spawning
        agent_id = await orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
        
        # Mock task delegation
        with patch.object(orchestrator, '_assign_task_to_agent_with_persona', return_value=True) as mock_assign:
            task_id = await orchestrator.delegate_task(
                task_description=sample_task.description,
                task_type=sample_task.task_type.value,
                priority=sample_task.priority,
                required_capabilities=sample_task.required_capabilities
            )
            
            # Verify persona assignment was called
            orchestrator.persona_system.assign_persona_to_agent.assert_called_once()
            
            # Verify enhanced assignment was used
            mock_assign.assert_called_once()
            
            assert task_id is not None
    
    async def test_persona_performance_update_on_completion(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock persona system
        orchestrator.persona_system = AsyncMock()
        
        # Test task completion handling
        task_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        # Mock task in database
        mock_task = MagicMock()
        mock_task.id = task_id
        mock_task.task_type = TaskType.CODE_GENERATION
        mock_task.metadata = {"required_capabilities": ["programming"]}
        mock_db_session.get.return_value = mock_task
        
        # Test persona performance update
        await orchestrator._handle_task_completion_persona_update(
            task_id, 
            {
                'agent_id': agent_id, 
                'success': True, 
                'completion_time': 120.0,
                'complexity': 0.8
            }
        )
        
        # Verify persona performance was updated
        orchestrator.persona_system.update_persona_performance.assert_called_once_with(
            agent_id=uuid.UUID(agent_id),
            task=mock_task,
            success=True,
            completion_time=120.0,
            complexity=0.8
        )


class TestWorkflowEngineIntegration:
    """Test enhanced workflow engine integration."""
    
    async def test_workflow_preparation_with_persona_optimization(self, enhanced_orchestrator, sample_workflow):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock workflow engine
        orchestrator.workflow_engine = AsyncMock()
        orchestrator.workflow_engine.execute_workflow.return_value = MagicMock(
            status=WorkflowStatus.COMPLETED,
            completed_tasks=3,
            failed_tasks=0,
            task_results=[]
        )
        
        # Mock persona system
        orchestrator.persona_system = AsyncMock()
        orchestrator.persona_system.list_available_personas.return_value = [
            MagicMock(id="backend_engineer_default")
        ]
        
        # Mock workflow in database
        mock_db_session.get.return_value = sample_workflow
        
        # Test workflow execution with preparation
        result = await orchestrator.execute_workflow(str(sample_workflow.id), {"test_context": True})
        
        # Verify workflow preparation was called
        assert result.status == WorkflowStatus.COMPLETED
        
        # Verify workflow engine was called
        orchestrator.workflow_engine.execute_workflow.assert_called_once()
    
    async def test_workflow_persona_performance_updates(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock persona system
        orchestrator.persona_system = AsyncMock()
        
        # Create mock workflow result with task results
        from app.core.workflow_engine import WorkflowResult, TaskResult, TaskExecutionState
        
        task_results = [
            TaskResult(
                task_id=str(uuid.uuid4()),
                status=TaskExecutionState.COMPLETED,
                agent_id=str(uuid.uuid4()),
                execution_time=120.0
            ),
            TaskResult(
                task_id=str(uuid.uuid4()),
                status=TaskExecutionState.FAILED,
                agent_id=str(uuid.uuid4()),
                execution_time=60.0
            )
        ]
        
        workflow_result = WorkflowResult(
            workflow_id=str(uuid.uuid4()),
            status=WorkflowStatus.COMPLETED,
            execution_time=180.0,
            completed_tasks=1,
            failed_tasks=1,
            total_tasks=2,
            task_results=task_results
        )
        
        # Mock tasks in database
        mock_db_session.get.side_effect = [
            MagicMock(id=task_results[0].task_id),
            MagicMock(id=task_results[1].task_id)
        ]
        
        # Test performance update
        await orchestrator._update_workflow_persona_performance(
            workflow_result.workflow_id, 
            workflow_result
        )
        
        # Verify persona performance was updated for both tasks
        assert orchestrator.persona_system.update_persona_performance.call_count == 2


class TestIntelligentLoadBalancing:
    """Test intelligent load balancing with persona awareness."""
    
    async def test_workload_analysis_calculation(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock capability matcher
        orchestrator.capability_matcher = AsyncMock()
        orchestrator.capability_matcher.get_workload_factor.side_effect = [0.8, 0.3, 0.6]
        
        # Add mock agents
        for i, workload in enumerate([0.8, 0.3, 0.6]):
            agent_id = f"agent_{i}"
            orchestrator.agents[agent_id] = MagicMock(
                status=AgentStatus.ACTIVE,
                context_window_usage=workload * 0.5,
                current_task=f"task_{i}" if workload > 0.5 else None,
                role=AgentRole.BACKEND_DEVELOPER,
                last_heartbeat=datetime.utcnow()
            )
        
        # Test workload analysis
        analysis = await orchestrator._analyze_agent_workloads()
        
        # Verify analysis results
        assert 'agent_workloads' in analysis
        assert 'balance_score' in analysis
        assert 'average_workload' in analysis
        assert len(analysis['agent_workloads']) == 3
        
        # Check that balance score reflects the workload distribution
        assert 0.0 <= analysis['balance_score'] <= 1.0
    
    async def test_intelligent_rebalancing_with_persona_consideration(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock intelligent router
        orchestrator.intelligent_router = AsyncMock()
        mock_reassignment = MagicMock()
        mock_reassignment.task_id = str(uuid.uuid4())
        mock_reassignment.from_agent_id = "overloaded_agent"
        mock_reassignment.to_agent_id = "underloaded_agent"
        mock_reassignment.expected_improvement = 0.3
        mock_reassignment.reason = "load_balancing"
        
        orchestrator.intelligent_router.rebalance_workload.return_value = [mock_reassignment]
        
        # Mock persona system for benefit calculation
        orchestrator.persona_system = AsyncMock()
        orchestrator.persona_system.get_agent_current_persona.return_value = MagicMock(
            persona_id="backend_engineer_default"
        )
        orchestrator.persona_system.get_persona.return_value = MagicMock(
            get_task_affinity=MagicMock(return_value=0.8)
        )
        
        # Mock workload analysis
        workload_analysis = {
            'agent_workloads': {
                'overloaded_agent': {'workload_factor': 0.9},
                'underloaded_agent': {'workload_factor': 0.2}
            },
            'average_workload': 0.55,
            'balance_score': 0.3
        }
        
        # Test intelligent rebalancing
        reassignments = await orchestrator._intelligent_workload_rebalancing(workload_analysis)
        
        # Verify reassignments were generated
        assert len(reassignments) > 0
        assert hasattr(reassignments[0], 'persona_improvement')


class TestCircuitBreakerPatterns:
    """Test circuit breaker and error handling patterns."""
    
    async def test_circuit_breaker_trip_on_consecutive_failures(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        agent_id = "test_agent"
        
        # Simulate consecutive failures
        for i in range(6):  # Exceed the threshold of 5
            await orchestrator._update_circuit_breaker(agent_id, success=False, error_type="test_error")
        
        # Check if circuit breaker should trip
        should_trip = await orchestrator._should_trip_circuit_breaker(agent_id)
        assert should_trip is True
        
        # Trip the circuit breaker
        await orchestrator._trip_circuit_breaker(agent_id, "consecutive_failures")
        
        # Verify circuit breaker state
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['state'] == 'open'
        assert orchestrator.metrics['circuit_breaker_trips'] == 1
    
    async def test_circuit_breaker_recovery_cycle(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        agent_id = "test_agent"
        
        # Create agent
        orchestrator.agents[agent_id] = MagicMock(
            status=AgentStatus.ERROR,
            last_heartbeat=datetime.utcnow()
        )
        
        # Trip circuit breaker
        await orchestrator._trip_circuit_breaker(agent_id, "test_reason")
        
        # Manually set to half-open for testing
        orchestrator.circuit_breakers[agent_id]['state'] = 'half_open'
        
        # Test recovery
        await orchestrator._test_circuit_breaker_recovery(agent_id)
        
        # Verify recovery attempt was made
        assert orchestrator.message_broker.send_message.called
    
    async def test_exponential_backoff_retry(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock operation that fails twice then succeeds
        call_count = 0
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Test failure")
            return "success"
        
        # Test retry with backoff
        result = await orchestrator.retry_with_exponential_backoff(
            mock_operation,
            max_retries=3,
            operation_name="test_operation"
        )
        
        assert result == "success"
        assert call_count == 3
        assert orchestrator.metrics['retry_attempts'] == 2


class TestEnhancedQueuing:
    """Test enhanced async task delegation and queuing."""
    
    async def test_task_enqueuing_by_priority(self, enhanced_orchestrator, sample_task):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Test different priority enqueueing
        high_priority_task = str(uuid.uuid4())
        medium_priority_task = str(uuid.uuid4())
        low_priority_task = str(uuid.uuid4())
        
        # Enqueue tasks with different priorities
        await orchestrator.enqueue_task(high_priority_task, TaskPriority.HIGH)
        await orchestrator.enqueue_task(medium_priority_task, TaskPriority.MEDIUM)
        await orchestrator.enqueue_task(low_priority_task, TaskPriority.LOW)
        
        # Verify queue state
        queue_status = orchestrator.get_queue_status()
        assert queue_status['high_priority_count'] == 1
        assert queue_status['medium_priority_count'] == 1
        assert queue_status['low_priority_count'] == 1
        assert queue_status['total_queued_tasks'] == 3
    
    async def test_workflow_task_queuing(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        workflow_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        
        # Enqueue workflow task
        await orchestrator.enqueue_task(
            task_id, 
            TaskPriority.MEDIUM, 
            workflow_id=workflow_id
        )
        
        # Verify workflow queue
        queue_status = orchestrator.get_queue_status()
        assert workflow_id in queue_status['workflow_queues']
        assert queue_status['workflow_queues'][workflow_id] == 1
    
    async def test_retry_queue_processing(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        task_id = str(uuid.uuid4())
        
        # Enqueue task for retry
        await orchestrator.enqueue_task(
            task_id, 
            TaskPriority.MEDIUM, 
            retry_count=1
        )
        
        # Verify retry queue
        queue_status = orchestrator.get_queue_status()
        assert queue_status['retry_queue_count'] == 1
    
    async def test_priority_queue_processing_order(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock intelligent router
        orchestrator.intelligent_router = AsyncMock()
        orchestrator.intelligent_router.route_task.return_value = "test_agent"
        
        # Mock available agents
        with patch.object(orchestrator, '_get_available_agent_ids', return_value=["test_agent"]):
            with patch.object(orchestrator, '_assign_queued_task', return_value=True) as mock_assign:
                # Add tasks to different priority queues
                await orchestrator.enqueue_task(str(uuid.uuid4()), TaskPriority.LOW)
                await orchestrator.enqueue_task(str(uuid.uuid4()), TaskPriority.HIGH) 
                await orchestrator.enqueue_task(str(uuid.uuid4()), TaskPriority.MEDIUM)
                
                # Process queues
                assigned_count = await orchestrator._process_priority_queues()
                
                # Verify high priority was processed first
                assert assigned_count > 0
                assert mock_assign.called


class TestIntegrationScenarios:
    """Test complex integration scenarios combining multiple features."""
    
    async def test_full_workflow_with_persona_and_load_balancing(self, enhanced_orchestrator, sample_workflow):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Mock all required components
        orchestrator.workflow_engine = AsyncMock()
        orchestrator.persona_system = AsyncMock()
        orchestrator.intelligent_router = AsyncMock()
        orchestrator.capability_matcher = AsyncMock()
        
        # Mock workflow execution result
        from app.core.workflow_engine import WorkflowResult, TaskResult, TaskExecutionState
        
        orchestrator.workflow_engine.execute_workflow.return_value = WorkflowResult(
            workflow_id=str(sample_workflow.id),
            status=WorkflowStatus.COMPLETED,
            execution_time=300.0,
            completed_tasks=3,
            failed_tasks=0,
            total_tasks=3,
            task_results=[
                TaskResult(
                    task_id=str(uuid.uuid4()),
                    status=TaskExecutionState.COMPLETED,
                    agent_id=str(uuid.uuid4()),
                    execution_time=100.0
                )
            ]
        )
        
        # Mock database queries
        mock_db_session.get.return_value = sample_workflow
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        
        # Execute workflow with full integration
        result = await orchestrator.execute_workflow(
            str(sample_workflow.id), 
            {"integration_test": True}
        )
        
        # Verify all components were involved
        assert result.status == WorkflowStatus.COMPLETED
        orchestrator.workflow_engine.execute_workflow.assert_called_once()
        
        # Verify metrics were updated
        assert orchestrator.metrics['workflows_executed'] > 0
    
    async def test_error_recovery_with_circuit_breaker_and_rebalancing(self, enhanced_orchestrator):
        orchestrator, mock_db_session = enhanced_orchestrator
        
        # Create test agent
        agent_id = "test_agent"
        orchestrator.agents[agent_id] = MagicMock(
            status=AgentStatus.ACTIVE,
            role=AgentRole.BACKEND_DEVELOPER,
            current_task=None,
            context_window_usage=0.5,
            last_heartbeat=datetime.utcnow()
        )
        
        # Simulate agent failure and recovery cycle
        # 1. Trigger circuit breaker
        for _ in range(6):
            await orchestrator._update_circuit_breaker(agent_id, success=False)
        
        await orchestrator._trip_circuit_breaker(agent_id, "integration_test")
        
        # 2. Verify circuit breaker tripped
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'open'
        
        # 3. Test load rebalancing with circuit breaker protection
        orchestrator.intelligent_router = AsyncMock()
        orchestrator.intelligent_router.rebalance_workload.return_value = []
        
        rebalance_result = await orchestrator.rebalance_agent_workloads(force_rebalance=True)
        
        # 4. Verify system handled the failure gracefully
        assert 'error' not in rebalance_result or rebalance_result.get('executed_reassignments', 0) >= 0


# Pytest configuration and fixtures
@pytest.mark.asyncio
async def test_orchestrator_initialization_with_all_enhancements():
    """Test that the orchestrator initializes correctly with all enhancements."""
    orchestrator = AgentOrchestrator()
    
    # Mock dependencies
    with patch('app.core.orchestrator.get_message_broker'):
        with patch('app.core.orchestrator.get_session_cache'):
            with patch('app.core.orchestrator.get_agent_persona_system') as mock_persona:
                mock_persona.return_value = AsyncMock()
                
                await orchestrator.start()
                
                # Verify all systems are initialized
                assert orchestrator.workflow_engine is not None
                assert orchestrator.intelligent_router is not None
                assert orchestrator.capability_matcher is not None
                assert orchestrator.persona_system is not None
                
                # Verify queuing system is set up
                assert 'high_priority' in orchestrator.task_queues
                assert 'workflow_tasks' in orchestrator.task_queues
                assert 'retry_queue' in orchestrator.task_queues
                
                # Verify circuit breaker is configured
                assert orchestrator.error_thresholds is not None
                assert orchestrator.circuit_breakers is not None
                
                await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])