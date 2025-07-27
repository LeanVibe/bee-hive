"""
Integration tests for Workflow execution with Agent Orchestrator.

Tests the complete workflow execution pipeline including database operations,
agent coordination, and API endpoints.
"""

import pytest
import uuid
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator
from app.core.workflow_engine import WorkflowEngine, WorkflowResult
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.agent import Agent, AgentStatus, AgentType


@pytest.mark.asyncio
class TestWorkflowIntegration:
    """Integration tests for complete workflow execution pipeline."""
    
    @pytest.fixture
    async def mock_orchestrator(self):
        """Create a mock orchestrator with workflow engine."""
        orchestrator = AgentOrchestrator()
        
        # Mock external dependencies
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        orchestrator.is_running = True
        
        # Initialize workflow engine with mocked dependencies
        with patch('app.core.workflow_engine.get_message_broker') as mock_broker:
            mock_broker.return_value = AsyncMock()
            orchestrator.workflow_engine = WorkflowEngine(orchestrator=orchestrator)
            await orchestrator.workflow_engine.initialize()
        
        # Add a mock agent
        from app.core.orchestrator import AgentInstance, AgentRole, AgentCapability, AgentStatus
        
        mock_agent = AgentInstance(
            id="test-agent-1",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session=None,
            capabilities=[
                AgentCapability("backend_development", "Backend development", 0.9, ["python", "fastapi"])
            ],
            current_task=None,
            context_window_usage=0.3,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None
        )
        
        orchestrator.agents["test-agent-1"] = mock_agent
        
        return orchestrator
    
    @pytest.fixture
    def sample_workflow_with_tasks(self):
        """Create a complete workflow with tasks for integration testing."""
        workflow_id = uuid.uuid4()
        task1_id = uuid.uuid4()
        task2_id = uuid.uuid4()
        task3_id = uuid.uuid4()
        
        # Create workflow
        workflow = Workflow(
            id=workflow_id,
            name="Integration Test Workflow",
            description="Complete workflow for integration testing",
            status=WorkflowStatus.READY,
            priority=WorkflowPriority.HIGH,
            task_ids=[task1_id, task2_id, task3_id],
            dependencies={
                str(task2_id): [str(task1_id)],  # task2 depends on task1
                str(task3_id): [str(task1_id)]   # task3 depends on task1 (parallel with task2)
            },
            total_tasks=3,
            estimated_duration=90
        )
        
        # Create tasks
        tasks = [
            Task(
                id=task1_id,
                title="Setup Database",
                description="Initialize database schema",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.HIGH,
                estimated_effort=30,
                required_capabilities=["backend_development"]
            ),
            Task(
                id=task2_id,
                title="Create API Endpoints",
                description="Implement REST API endpoints",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.HIGH,
                estimated_effort=45,
                required_capabilities=["backend_development"]
            ),
            Task(
                id=task3_id,
                title="Add Authentication",
                description="Implement JWT authentication",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                estimated_effort=60,
                required_capabilities=["backend_development"]
            )
        ]
        
        return workflow, tasks
    
    async def test_complete_workflow_execution_flow(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test complete workflow execution from start to finish."""
        
        workflow, tasks = sample_workflow_with_tasks
        workflow_id = str(workflow.id)
        
        # Mock database operations
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock workflow loading
            mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [
                workflow,  # Initial workflow load
                tasks[0],  # Task 1 execution
                tasks[1],  # Task 2 execution  
                tasks[2]   # Task 3 execution
            ]
            
            # Mock task execution - simulate successful completion
            with patch.object(mock_orchestrator.workflow_engine, '_send_task_to_agent') as mock_send_task:
                mock_send_task.return_value = {"status": "completed", "result": {"success": True}}
                
                # Execute workflow
                result = await mock_orchestrator.execute_workflow(workflow_id)
                
                # Verify workflow execution result
                assert result.workflow_id == workflow_id
                assert result.status == WorkflowStatus.COMPLETED
                assert result.completed_tasks == 3
                assert result.failed_tasks == 0
                assert result.total_tasks == 3
                
                # Verify orchestrator metrics were updated
                assert mock_orchestrator.metrics['workflows_executed'] == 1
                assert mock_orchestrator.metrics['workflows_completed'] == 1
                
                # Verify task execution was called for all tasks
                assert mock_send_task.call_count == 3
    
    async def test_workflow_execution_with_task_failure(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test workflow execution when some tasks fail."""
        
        workflow, tasks = sample_workflow_with_tasks
        workflow_id = str(workflow.id)
        
        # Set fail_fast to False to test partial completion
        workflow.context = {"fail_fast": False}
        
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock workflow and task loading
            mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [
                workflow,  # Workflow load
                tasks[0],  # Task 1 - will succeed
                tasks[1],  # Task 2 - will fail
                tasks[2]   # Task 3 - will succeed
            ]
            
            # Mock task execution with mixed results
            with patch.object(mock_orchestrator.workflow_engine, '_send_task_to_agent') as mock_send_task:
                def mock_task_execution(task_id, agent_id, task):
                    if str(task.id) == str(tasks[1].id):  # Task 2 fails
                        raise Exception("Task execution failed")
                    return {"status": "completed", "result": {"success": True}}
                
                mock_send_task.side_effect = mock_task_execution
                
                # Execute workflow
                result = await mock_orchestrator.execute_workflow(workflow_id)
                
                # Verify partial completion
                assert result.workflow_id == workflow_id
                assert result.status == WorkflowStatus.FAILED  # Overall failed due to task failure
                assert result.completed_tasks == 2  # Task 1 and 3 completed
                assert result.failed_tasks == 1   # Task 2 failed
                assert result.total_tasks == 3
    
    async def test_workflow_pause_and_resume(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test workflow pause and resume functionality."""
        
        workflow, tasks = sample_workflow_with_tasks
        workflow_id = str(workflow.id)
        
        # Mock a running workflow
        mock_execution_task = AsyncMock()
        mock_orchestrator.workflow_engine.active_workflows[workflow_id] = mock_execution_task
        
        with patch.object(mock_orchestrator.workflow_engine, '_update_workflow_status') as mock_update_status:
            # Test pause
            result = await mock_orchestrator.pause_workflow(workflow_id)
            assert result is True
            mock_execution_task.cancel.assert_called_once()
            mock_update_status.assert_called_with(workflow_id, WorkflowStatus.PAUSED)
            
            # Verify workflow was removed from active workflows
            assert workflow_id not in mock_orchestrator.workflow_engine.active_workflows
    
    async def test_workflow_status_tracking(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test workflow execution status tracking."""
        
        workflow, tasks = sample_workflow_with_tasks
        workflow_id = str(workflow.id)
        
        # Initialize workflow state
        mock_orchestrator.workflow_engine.workflow_states[workflow_id] = {
            'workflow_id': workflow_id,
            'task_ids': [str(task.id) for task in tasks],
            'start_time': datetime.utcnow(),
            'completed_tasks': 1,
            'failed_tasks': 0,
            'total_tasks': 3
        }
        
        # Add some task execution results
        mock_orchestrator.workflow_engine.task_executions[str(tasks[0].id)] = {
            'task_id': str(tasks[0].id),
            'status': 'completed',
            'execution_time': 30.5,
            'retry_count': 0
        }
        
        # Get execution status
        status = await mock_orchestrator.get_workflow_execution_status(workflow_id)
        
        assert status['workflow_id'] == workflow_id
        assert status['is_active'] is False  # Not in active workflows
        assert status['workflow_state']['completed_tasks'] == 1
        assert str(tasks[0].id) in status['task_statuses']
    
    async def test_dependency_resolution_with_real_workflow(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test dependency resolution with a realistic workflow structure."""
        
        workflow, tasks = sample_workflow_with_tasks
        
        # Test dependency resolution
        execution_batches = await mock_orchestrator.workflow_engine.resolve_dependencies(workflow)
        
        # Should have 2 batches:
        # Batch 1: Task 1 (no dependencies)
        # Batch 2: Task 2 and Task 3 (both depend on Task 1, can run in parallel)
        assert len(execution_batches) == 2
        
        # First batch should contain only task 1
        assert len(execution_batches[0]) == 1
        assert str(tasks[0].id) in execution_batches[0]
        
        # Second batch should contain task 2 and task 3
        assert len(execution_batches[1]) == 2
        assert str(tasks[1].id) in execution_batches[1]
        assert str(tasks[2].id) in execution_batches[1]
    
    async def test_agent_assignment_integration(self, mock_orchestrator, sample_workflow_with_tasks):
        """Test that tasks are properly assigned to suitable agents."""
        
        workflow, tasks = sample_workflow_with_tasks
        
        # Test task assignment for backend development task
        task = tasks[0]  # Setup Database task
        
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = task
            
            # Test task assignment
            agent_id = await mock_orchestrator.workflow_engine._assign_task_to_agent(str(task.id), task)
            
            # Should assign to our mock backend developer agent
            assert agent_id == "test-agent-1"
    
    async def test_metrics_and_observability(self, mock_orchestrator):
        """Test that workflow execution properly updates metrics and observability."""
        
        # Get initial metrics
        initial_metrics = mock_orchestrator.workflow_engine.get_metrics()
        assert initial_metrics['workflows_executed'] == 0
        
        # Mock a completed workflow result
        from app.core.workflow_engine import WorkflowResult
        
        result = WorkflowResult(
            workflow_id="test-workflow",
            status=WorkflowStatus.COMPLETED,
            execution_time=120.0,
            completed_tasks=5,
            failed_tasks=0,
            total_tasks=5,
            task_results=[]
        )
        
        # Update metrics
        mock_orchestrator.workflow_engine._update_execution_metrics(result, 120.0)
        
        # Verify metrics were updated
        updated_metrics = mock_orchestrator.workflow_engine.get_metrics()
        assert updated_metrics['workflows_executed'] == 1
        assert updated_metrics['workflows_completed'] == 1
        assert updated_metrics['average_execution_time'] == 120.0
        
        # Test system status includes workflow metrics
        with patch.object(mock_orchestrator, '_check_system_health') as mock_health:
            mock_health.return_value = {"overall": True}
            
            system_status = await mock_orchestrator.get_system_status()
            
            assert system_status['workflow_engine_active'] is True
            assert 'workflow_metrics' in system_status
            assert system_status['workflow_metrics']['workflows_executed'] == 1


@pytest.mark.asyncio
class TestWorkflowFailureScenarios:
    """Test workflow engine behavior under failure conditions."""
    
    @pytest.fixture
    async def workflow_engine(self):
        """Create workflow engine for failure testing."""
        engine = WorkflowEngine()
        
        with patch('app.core.workflow_engine.get_message_broker') as mock_broker:
            mock_broker.return_value = AsyncMock()
            await engine.initialize()
        
        return engine
    
    async def test_task_retry_exhaustion(self, workflow_engine):
        """Test behavior when task retries are exhausted."""
        
        task_id = str(uuid.uuid4())
        workflow_engine.max_retries = 2
        workflow_engine.retry_delay_seconds = 0.01
        
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock task that always fails
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.task_type = TaskType.FEATURE_DEVELOPMENT
            mock_task.priority = TaskPriority.MEDIUM
            mock_task.required_capabilities = []
            mock_task.to_dict.return_value = {"id": task_id}
            
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
            
            with patch.object(workflow_engine, '_update_task_status'):
                with patch.object(workflow_engine, '_send_task_to_agent') as mock_send_task:
                    # Always fail
                    mock_send_task.side_effect = Exception("Persistent failure")
                    
                    result = await workflow_engine._execute_single_task(task_id)
                    
                    assert result.task_id == task_id
                    assert result.status.value == "failed"
                    assert "Failed after" in result.error
                    assert result.retry_count > workflow_engine.max_retries
    
    async def test_workflow_cancellation_during_execution(self, workflow_engine):
        """Test workflow cancellation while tasks are executing."""
        
        workflow_id = str(uuid.uuid4())
        
        # Create a long-running mock task
        async def long_running_task():
            await asyncio.sleep(10)  # Long task
            return MagicMock()
        
        execution_task = asyncio.create_task(long_running_task())
        workflow_engine.active_workflows[workflow_id] = execution_task
        
        with patch.object(workflow_engine, '_update_workflow_status') as mock_update_status:
            # Cancel the workflow
            result = await workflow_engine.cancel_workflow(workflow_id, "User requested")
            
            assert result is True
            assert execution_task.cancelled()
            mock_update_status.assert_called_with(
                workflow_id, 
                WorkflowStatus.CANCELLED, 
                "Cancelled: User requested"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.core.workflow_engine", "--cov=app.core.orchestrator", "--cov-report=term-missing"])