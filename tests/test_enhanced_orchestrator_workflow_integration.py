"""
Integration Tests for Enhanced Orchestrator with Workflow Engine

This test suite focuses on testing the integration between the enhanced orchestrator
and the workflow engine, ensuring proper coordination, state management, and
workflow execution within the multi-agent environment.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentInstance
from app.core.workflow_engine import WorkflowEngine, WorkflowResult, TaskExecutionState
from app.core.intelligent_task_router import TaskRoutingContext, RoutingStrategy
from app.models.agent import AgentStatus
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus, WorkflowStep, WorkflowPriority


@pytest.fixture
def mock_workflow():
    """Create a mock workflow for testing."""
    return Workflow(
        id=str(uuid.uuid4()),
        name="Test Development Workflow",
        description="A test workflow for development tasks",
        steps=[
            WorkflowStep(
                id="step-1",
                name="Requirements Analysis",
                task_type=TaskType.ANALYSIS,
                dependencies=[],
                estimated_duration=2.0,
                required_capabilities=["analysis", "documentation"],
                metadata={"complexity": "medium"}
            ),
            WorkflowStep(
                id="step-2", 
                name="Implementation",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                dependencies=["step-1"],
                estimated_duration=6.0,
                required_capabilities=["python", "api_development"],
                metadata={"complexity": "high"}
            ),
            WorkflowStep(
                id="step-3",
                name="Testing",
                task_type=TaskType.TESTING,
                dependencies=["step-2"],
                estimated_duration=3.0,
                required_capabilities=["testing", "pytest"],
                metadata={"complexity": "medium"}
            )
        ],
        priority=WorkflowPriority.HIGH,
        status=WorkflowStatus.PENDING,
        created_at=datetime.utcnow(),
        metadata={"project": "test-project", "sprint": "sprint-1"}
    )


@pytest.fixture
def orchestrator_with_workflow_engine():
    """Create orchestrator with mocked workflow engine."""
    orchestrator = AgentOrchestrator()
    
    # Mock dependencies
    orchestrator.message_broker = AsyncMock()
    orchestrator.session_cache = AsyncMock()
    orchestrator.anthropic_client = AsyncMock()
    
    # Mock workflow engine
    workflow_engine = AsyncMock(spec=WorkflowEngine)
    orchestrator.workflow_engine = workflow_engine
    
    # Mock other enhanced components
    orchestrator.persona_system = AsyncMock()
    orchestrator.intelligent_router = AsyncMock()
    
    return orchestrator


@pytest.fixture
def sample_agents():
    """Create sample agents for workflow testing."""
    agents = {}
    
    # Backend Developer Agent
    agents["backend-dev-001"] = AgentInstance(
        id="backend-dev-001",
        role=AgentRole.BACKEND_DEVELOPER,
        status=AgentStatus.ACTIVE,
        tmux_session="backend-session",
        capabilities=[],
        current_task=None,
        context_window_usage=0.3,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )
    
    # QA Engineer Agent
    agents["qa-eng-001"] = AgentInstance(
        id="qa-eng-001",
        role=AgentRole.QA_ENGINEER,
        status=AgentStatus.ACTIVE,
        tmux_session="qa-session",
        capabilities=[],
        current_task=None,
        context_window_usage=0.2,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )
    
    # Product Manager Agent
    agents["pm-001"] = AgentInstance(
        id="pm-001",
        role=AgentRole.PRODUCT_MANAGER,
        status=AgentStatus.ACTIVE,
        tmux_session="pm-session", 
        capabilities=[],
        current_task=None,
        context_window_usage=0.4,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )
    
    return agents


class TestWorkflowOrchestrationIntegration:
    """Test suite for workflow orchestration integration."""
    
    async def test_execute_workflow_success(self, orchestrator_with_workflow_engine, mock_workflow, sample_agents):
        """Test successful workflow execution through orchestrator."""
        orchestrator = orchestrator_with_workflow_engine
        orchestrator.agents = sample_agents
        
        # Mock workflow engine to return success
        workflow_result = WorkflowResult(
            workflow_id=uuid.UUID(mock_workflow.id),
            status=TaskExecutionState.COMPLETED,
            result_data={"steps_completed": 3, "total_duration": 11.0},
            execution_time=11.0,
            error_details=None
        )
        orchestrator.workflow_engine.execute_workflow.return_value = workflow_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            # Execute workflow
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify workflow execution
            assert result is not None
            assert result.status == TaskExecutionState.COMPLETED
            orchestrator.workflow_engine.execute_workflow.assert_called_once_with(
                mock_workflow, orchestrator
            )
    
    async def test_execute_workflow_with_agent_assignment(self, orchestrator_with_workflow_engine, mock_workflow, sample_agents):
        """Test workflow execution with automatic agent assignment."""
        orchestrator = orchestrator_with_workflow_engine
        orchestrator.agents = sample_agents
        
        # Mock intelligent router for agent selection
        orchestrator.intelligent_router.route_task_to_agent.side_effect = [
            "pm-001",        # Requirements Analysis -> Product Manager
            "backend-dev-001", # Implementation -> Backend Developer  
            "qa-eng-001"     # Testing -> QA Engineer
        ]
        
        # Mock workflow engine with step-by-step execution
        async def mock_execute_workflow(workflow, orchestrator_ref):
            # Simulate workflow engine requesting agent assignments for each step
            for step in workflow.steps:
                routing_context = TaskRoutingContext(
                    task_id=f"task-{step.id}",
                    task_type=step.task_type,
                    priority=TaskPriority.HIGH,
                    required_capabilities=step.required_capabilities,
                    available_agents=list(orchestrator_ref.agents.keys()),
                    routing_strategy=RoutingStrategy.WORKFLOW_OPTIMIZED
                )
                
                selected_agent = await orchestrator_ref.intelligent_router.route_task_to_agent(routing_context)
                assert selected_agent in orchestrator_ref.agents
            
            return WorkflowResult(
                workflow_id=uuid.UUID(workflow.id),
                status=TaskExecutionState.COMPLETED,
                result_data={"agent_assignments": 3},
                execution_time=8.5,
                error_details=None
            )
        
        orchestrator.workflow_engine.execute_workflow.side_effect = mock_execute_workflow
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify all steps were assigned to appropriate agents
            assert orchestrator.intelligent_router.route_task_to_agent.call_count == 3
            assert result.status == TaskExecutionState.COMPLETED
    
    async def test_workflow_failure_handling(self, orchestrator_with_workflow_engine, mock_workflow, sample_agents):
        """Test workflow failure handling and recovery."""
        orchestrator = orchestrator_with_workflow_engine
        orchestrator.agents = sample_agents
        
        # Mock workflow engine to return failure
        workflow_result = WorkflowResult(
            workflow_id=uuid.UUID(mock_workflow.id),
            status=TaskExecutionState.FAILED,
            result_data={"failed_step": "step-2", "completed_steps": 1},
            execution_time=4.0,
            error_details="Agent backend-dev-001 became unavailable during implementation"
        )
        orchestrator.workflow_engine.execute_workflow.return_value = workflow_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify failure is properly handled
            assert result.status == TaskExecutionState.FAILED
            assert "backend-dev-001" in result.error_details
            
            # Verify error metrics are updated
            assert 'workflow_failures' in orchestrator.metrics
    
    async def test_workflow_step_retry_mechanism(self, orchestrator_with_workflow_engine, mock_workflow, sample_agents):
        """Test workflow step retry mechanism on agent failure."""
        orchestrator = orchestrator_with_workflow_engine
        orchestrator.agents = sample_agents
        
        # Mark one agent as having circuit breaker issues
        orchestrator.circuit_breakers["backend-dev-001"] = {
            'state': 'open',
            'failure_count': 5,
            'consecutive_failures': 5
        }
        
        # Mock intelligent router to reassign from failed agent
        orchestrator.intelligent_router.route_task_to_agent.side_effect = [
            "backend-dev-001",  # First attempt - will fail due to circuit breaker
            "pm-001"           # Retry attempt - reassign to available agent
        ]
        
        async def mock_execute_with_retry(workflow, orchestrator_ref):
            # Simulate workflow engine detecting agent failure and retrying
            for step in workflow.steps:
                if step.task_type == TaskType.FEATURE_DEVELOPMENT:
                    # First assignment fails
                    first_agent = await orchestrator_ref.intelligent_router.route_task_to_agent(
                        TaskRoutingContext(
                            task_id=f"task-{step.id}",
                            task_type=step.task_type,
                            priority=TaskPriority.HIGH,
                            required_capabilities=step.required_capabilities,
                            available_agents=list(orchestrator_ref.agents.keys()),
                            routing_strategy=RoutingStrategy.FAULT_TOLERANT
                        )
                    )
                    
                    # Check if agent is available (circuit breaker)
                    if orchestrator_ref.circuit_breakers.get(first_agent, {}).get('state') == 'open':
                        # Retry with different agent
                        available_agents = [
                            aid for aid in orchestrator_ref.agents.keys() 
                            if orchestrator_ref.circuit_breakers.get(aid, {}).get('state') != 'open'
                        ]
                        
                        retry_agent = await orchestrator_ref.intelligent_router.route_task_to_agent(
                            TaskRoutingContext(
                                task_id=f"task-{step.id}-retry",
                                task_type=step.task_type,
                                priority=TaskPriority.HIGH,
                                required_capabilities=step.required_capabilities,
                                available_agents=available_agents,
                                routing_strategy=RoutingStrategy.FAULT_TOLERANT
                            )
                        )
            
            return WorkflowResult(
                workflow_id=uuid.UUID(workflow.id),
                status=TaskExecutionState.COMPLETED,
                result_data={"retries_performed": 1},
                execution_time=9.0,
                error_details=None
            )
        
        orchestrator.workflow_engine.execute_workflow.side_effect = mock_execute_with_retry
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify retry mechanism worked
            assert result.status == TaskExecutionState.COMPLETED
            assert orchestrator.intelligent_router.route_task_to_agent.call_count == 2
            assert result.result_data["retries_performed"] == 1


class TestWorkflowQueueIntegration:
    """Test suite for workflow queue integration with task queues."""
    
    async def test_workflow_task_queueing(self, orchestrator_with_workflow_engine, mock_workflow):
        """Test workflow tasks are properly queued in workflow-specific queues."""
        orchestrator = orchestrator_with_workflow_engine
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            # Queue workflow for execution 
            await orchestrator.queue_workflow_execution(mock_workflow.id, priority=WorkflowPriority.HIGH)
            
            # Verify workflow is added to appropriate queue
            assert mock_workflow.id in orchestrator.task_queues['workflow_tasks']
    
    async def test_workflow_queue_processing(self, orchestrator_with_workflow_engine, mock_workflow, sample_agents):
        """Test workflow queue processing with agent coordination."""
        orchestrator = orchestrator_with_workflow_engine
        orchestrator.agents = sample_agents
        
        # Add workflow to queue
        orchestrator.task_queues['workflow_tasks'][mock_workflow.id] = {
            'workflow_id': mock_workflow.id,
            'priority': WorkflowPriority.HIGH,
            'queued_at': datetime.utcnow(),
            'retry_count': 0
        }
        
        # Mock successful workflow execution
        workflow_result = WorkflowResult(
            workflow_id=uuid.UUID(mock_workflow.id),
            status=TaskExecutionState.COMPLETED,
            result_data={"queue_processed": True},
            execution_time=7.0,
            error_details=None
        )
        orchestrator.workflow_engine.execute_workflow.return_value = workflow_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            # Process workflow queue
            processed_count = await orchestrator._process_workflow_queue()
            
            # Verify workflow was processed and removed from queue
            assert processed_count == 1
            assert mock_workflow.id not in orchestrator.task_queues['workflow_tasks']
    
    async def test_workflow_queue_priority_ordering(self, orchestrator_with_workflow_engine):
        """Test workflow queue processes high priority workflows first."""
        orchestrator = orchestrator_with_workflow_engine
        
        # Add workflows with different priorities
        workflows = []
        for i, priority in enumerate([WorkflowPriority.LOW, WorkflowPriority.HIGH, WorkflowPriority.CRITICAL]):
            workflow_id = f"workflow-{i}"
            orchestrator.task_queues['workflow_tasks'][workflow_id] = {
                'workflow_id': workflow_id,
                'priority': priority,
                'queued_at': datetime.utcnow() - timedelta(minutes=i),  # Earlier queued = higher i
                'retry_count': 0
            }
            workflows.append(workflow_id)
        
        # Mock workflow execution to track order
        execution_order = []
        
        async def track_execution_order(workflow, orchestrator_ref):
            execution_order.append(workflow.id)
            return WorkflowResult(
                workflow_id=uuid.UUID(workflow.id),
                status=TaskExecutionState.COMPLETED,
                result_data={},
                execution_time=1.0,
                error_details=None
            )
        
        orchestrator.workflow_engine.execute_workflow.side_effect = track_execution_order
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock workflow objects
            for workflow_id in workflows:
                mock_workflow = MagicMock()
                mock_workflow.id = workflow_id
                mock_db_session.get.return_value = mock_workflow
                
                # Process one workflow at a time to test ordering
                await orchestrator._process_workflow_queue()
        
        # Verify high priority workflows were processed first
        # Expected order: CRITICAL, HIGH, LOW (regardless of queue time)
        assert execution_order == ["workflow-2", "workflow-1", "workflow-0"]


class TestWorkflowMetricsIntegration:
    """Test suite for workflow metrics and analytics integration."""
    
    async def test_workflow_completion_metrics(self, orchestrator_with_workflow_engine, mock_workflow):
        """Test workflow completion metrics are properly recorded."""
        orchestrator = orchestrator_with_workflow_engine
        
        # Execute workflow successfully
        workflow_result = WorkflowResult(
            workflow_id=uuid.UUID(mock_workflow.id),
            status=TaskExecutionState.COMPLETED,
            result_data={"steps_completed": 3},
            execution_time=12.5,
            error_details=None
        )
        orchestrator.workflow_engine.execute_workflow.return_value = workflow_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify metrics are updated
            assert orchestrator.metrics['workflows_completed'] > 0
            assert 'workflow_execution_times' in orchestrator.metrics
            assert 'workflow_success_rate' in orchestrator.metrics
    
    async def test_workflow_failure_metrics(self, orchestrator_with_workflow_engine, mock_workflow):
        """Test workflow failure metrics are properly recorded."""
        orchestrator = orchestrator_with_workflow_engine
        
        # Execute workflow with failure
        workflow_result = WorkflowResult(
            workflow_id=uuid.UUID(mock_workflow.id),
            status=TaskExecutionState.FAILED,
            result_data={"steps_completed": 1, "failed_step": "step-2"},
            execution_time=5.0,
            error_details="Implementation step failed due to missing dependencies"
        )
        orchestrator.workflow_engine.execute_workflow.return_value = workflow_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(mock_workflow.id)
            
            # Verify failure metrics are updated
            assert orchestrator.metrics['workflow_failures'] > 0
            assert 'workflow_failure_reasons' in orchestrator.metrics
    
    async def test_workflow_performance_analytics(self, orchestrator_with_workflow_engine):
        """Test workflow performance analytics collection."""
        orchestrator = orchestrator_with_workflow_engine
        
        # Add some workflow performance data
        orchestrator.metrics['workflow_execution_times'] = [10.0, 12.5, 8.3, 15.2, 9.7]
        orchestrator.metrics['workflows_completed'] = 5
        orchestrator.metrics['workflow_failures'] = 1
        
        analytics = await orchestrator.get_workflow_analytics()
        
        # Verify analytics structure
        assert 'average_execution_time' in analytics
        assert 'success_rate' in analytics
        assert 'total_workflows' in analytics
        assert 'performance_trend' in analytics
        
        # Verify calculations
        assert analytics['average_execution_time'] == 11.14  # Average of execution times
        assert analytics['success_rate'] == 5/6  # 5 successes out of 6 total (5 completed + 1 failed)
        assert analytics['total_workflows'] == 6


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--cov=app.core.orchestrator",
        "--cov=app.core.workflow_engine", 
        "--cov-report=term-missing"
    ])