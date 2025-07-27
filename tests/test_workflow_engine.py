"""
Comprehensive tests for WorkflowEngine functionality.

Tests cover workflow execution, dependency resolution, parallel processing,
failure handling, and integration with the orchestrator.
"""

import pytest
import uuid
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.workflow_engine import (
    WorkflowEngine, ExecutionMode, TaskExecutionState, TaskResult, 
    WorkflowResult, ExecutionPlan
)
from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.core.orchestrator import AgentOrchestrator


class TestWorkflowEngine:
    """Test suite for WorkflowEngine core functionality."""
    
    @pytest.fixture
    async def workflow_engine(self):
        """Create a workflow engine for testing."""
        engine = WorkflowEngine()
        
        # Mock Redis initialization to avoid external dependencies
        with patch('app.core.workflow_engine.get_message_broker') as mock_broker:
            mock_broker.return_value = AsyncMock()
            await engine.initialize()
        
        return engine
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing."""
        workflow_id = uuid.uuid4()
        task1_id = uuid.uuid4()
        task2_id = uuid.uuid4()
        task3_id = uuid.uuid4()
        
        workflow = Workflow(
            id=workflow_id,
            name="Test Workflow",
            description="A test workflow for dependency resolution",
            status=WorkflowStatus.READY,
            priority=WorkflowPriority.MEDIUM,
            task_ids=[task1_id, task2_id, task3_id],
            dependencies={
                str(task2_id): [str(task1_id)],  # task2 depends on task1
                str(task3_id): [str(task1_id), str(task2_id)]  # task3 depends on both
            },
            total_tasks=3
        )
        return workflow
    
    @pytest.fixture
    def sample_tasks(self, sample_workflow):
        """Create sample tasks corresponding to the workflow."""
        task_ids = sample_workflow.task_ids
        tasks = []
        
        for i, task_id in enumerate(task_ids):
            task = Task(
                id=task_id,
                title=f"Test Task {i+1}",
                description=f"Description for task {i+1}",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                status=TaskStatus.PENDING,
                priority=TaskPriority.MEDIUM,
                estimated_effort=30  # 30 minutes
            )
            tasks.append(task)
        
        return tasks
    
    @pytest.mark.asyncio
    async def test_workflow_engine_initialization(self, workflow_engine):
        """Test workflow engine initialization."""
        assert workflow_engine.active_workflows == {}
        assert workflow_engine.workflow_states == {}
        assert workflow_engine.task_executions == {}
        assert workflow_engine.execution_metrics['workflows_executed'] == 0
    
    @pytest.mark.asyncio
    async def test_dependency_resolution_sequential(self, workflow_engine, sample_workflow):
        """Test dependency resolution with sequential dependencies."""
        
        execution_batches = await workflow_engine.resolve_dependencies(sample_workflow)
        
        # Should have 3 batches for sequential execution
        assert len(execution_batches) == 3
        
        # First batch should contain only task1 (no dependencies)
        task1_id = str(sample_workflow.task_ids[0])
        assert execution_batches[0] == [task1_id]
        
        # Second batch should contain task2 (depends on task1)
        task2_id = str(sample_workflow.task_ids[1])
        assert execution_batches[1] == [task2_id]
        
        # Third batch should contain task3 (depends on task1 and task2)
        task3_id = str(sample_workflow.task_ids[2])
        assert execution_batches[2] == [task3_id]
    
    @pytest.mark.asyncio
    async def test_dependency_resolution_parallel(self, workflow_engine):
        """Test dependency resolution with parallel tasks."""
        
        # Create workflow with parallel tasks
        workflow_id = uuid.uuid4()
        task1_id = uuid.uuid4()
        task2_id = uuid.uuid4()
        task3_id = uuid.uuid4()
        
        workflow = Workflow(
            id=workflow_id,
            name="Parallel Test Workflow",
            task_ids=[task1_id, task2_id, task3_id],
            dependencies={},  # No dependencies - all can run in parallel
            total_tasks=3
        )
        
        execution_batches = await workflow_engine.resolve_dependencies(workflow)
        
        # Should have 1 batch with all tasks
        assert len(execution_batches) == 1
        assert len(execution_batches[0]) == 3
        assert set(execution_batches[0]) == {str(task1_id), str(task2_id), str(task3_id)}
    
    @pytest.mark.asyncio
    async def test_dependency_resolution_circular_dependency(self, workflow_engine):
        """Test detection of circular dependencies."""
        
        # Create workflow with circular dependency
        workflow_id = uuid.uuid4()
        task1_id = uuid.uuid4()
        task2_id = uuid.uuid4()
        
        workflow = Workflow(
            id=workflow_id,
            name="Circular Dependency Workflow",
            task_ids=[task1_id, task2_id],
            dependencies={
                str(task1_id): [str(task2_id)],  # task1 depends on task2
                str(task2_id): [str(task1_id)]   # task2 depends on task1 (circular)
            },
            total_tasks=2
        )
        
        # Should raise ValueError for circular dependency
        with pytest.raises(ValueError, match="Circular dependency detected"):
            await workflow_engine.resolve_dependencies(workflow)
    
    @pytest.mark.asyncio
    async def test_task_batch_execution_success(self, workflow_engine):
        """Test successful execution of a task batch."""
        
        # Mock task execution
        with patch.object(workflow_engine, '_execute_single_task') as mock_execute:
            task_ids = [str(uuid.uuid4()) for _ in range(3)]
            
            # Mock successful task results
            mock_execute.side_effect = [
                TaskResult(task_id=task_ids[0], status=TaskExecutionState.COMPLETED),
                TaskResult(task_id=task_ids[1], status=TaskExecutionState.COMPLETED),
                TaskResult(task_id=task_ids[2], status=TaskExecutionState.COMPLETED)
            ]
            
            results = await workflow_engine.execute_task_batch(task_ids)
            
            assert len(results) == 3
            assert all(r.status == TaskExecutionState.COMPLETED for r in results)
            assert mock_execute.call_count == 3
    
    @pytest.mark.asyncio
    async def test_task_batch_execution_with_failures(self, workflow_engine):
        """Test task batch execution with some failures."""
        
        with patch.object(workflow_engine, '_execute_single_task') as mock_execute:
            task_ids = [str(uuid.uuid4()) for _ in range(3)]
            
            # Mock mixed results
            mock_execute.side_effect = [
                TaskResult(task_id=task_ids[0], status=TaskExecutionState.COMPLETED),
                TaskResult(task_id=task_ids[1], status=TaskExecutionState.FAILED, error="Task failed"),
                TaskResult(task_id=task_ids[2], status=TaskExecutionState.COMPLETED)
            ]
            
            results = await workflow_engine.execute_task_batch(task_ids)
            
            assert len(results) == 3
            assert results[0].status == TaskExecutionState.COMPLETED
            assert results[1].status == TaskExecutionState.FAILED
            assert results[2].status == TaskExecutionState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_task_batch_execution_timeout(self, workflow_engine):
        """Test task batch execution with timeout."""
        
        # Set short timeout for testing
        workflow_engine.task_timeout_seconds = 0.1
        
        with patch.object(workflow_engine, '_execute_single_task') as mock_execute:
            task_ids = [str(uuid.uuid4())]
            
            # Mock slow task execution
            async def slow_task(task_id):
                await asyncio.sleep(1)  # Longer than timeout
                return TaskResult(task_id=task_id, status=TaskExecutionState.COMPLETED)
            
            mock_execute.side_effect = slow_task
            
            results = await workflow_engine.execute_task_batch(task_ids)
            
            assert len(results) == 1
            assert results[0].status == TaskExecutionState.FAILED
            assert "timeout" in results[0].error.lower()
    
    @pytest.mark.asyncio
    async def test_single_task_execution_success(self, workflow_engine):
        """Test successful single task execution."""
        
        task_id = str(uuid.uuid4())
        
        # Mock database session and task
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.task_type = TaskType.FEATURE_DEVELOPMENT
            mock_task.priority = TaskPriority.MEDIUM
            mock_task.required_capabilities = []
            mock_task.to_dict.return_value = {"id": task_id, "title": "Test Task"}
            
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
            
            # Mock task status updates
            with patch.object(workflow_engine, '_update_task_status') as mock_update_status:
                with patch.object(workflow_engine, '_send_task_to_agent') as mock_send_task:
                    mock_send_task.return_value = {"status": "completed", "result": {"success": True}}
                    
                    result = await workflow_engine._execute_single_task(task_id)
                    
                    assert result.task_id == task_id
                    assert result.status == TaskExecutionState.COMPLETED
                    assert result.result["status"] == "completed"
                    assert mock_update_status.call_count >= 2  # IN_PROGRESS and COMPLETED
    
    @pytest.mark.asyncio
    async def test_single_task_execution_with_retries(self, workflow_engine):
        """Test single task execution with retry logic."""
        
        task_id = str(uuid.uuid4())
        workflow_engine.max_retries = 2
        workflow_engine.retry_delay_seconds = 0.01  # Fast retry for testing
        
        # Mock database session and task
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.task_type = TaskType.FEATURE_DEVELOPMENT
            mock_task.priority = TaskPriority.MEDIUM
            mock_task.required_capabilities = []
            mock_task.to_dict.return_value = {"id": task_id, "title": "Test Task"}
            
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
            
            # Mock task status updates and task execution
            with patch.object(workflow_engine, '_update_task_status') as mock_update_status:
                with patch.object(workflow_engine, '_send_task_to_agent') as mock_send_task:
                    # First two attempts fail, third succeeds
                    mock_send_task.side_effect = [
                        Exception("Network error"),
                        Exception("Temporary failure"),
                        {"status": "completed", "result": {"success": True}}
                    ]
                    
                    result = await workflow_engine._execute_single_task(task_id)
                    
                    assert result.task_id == task_id
                    assert result.status == TaskExecutionState.COMPLETED
                    assert result.retry_count == 2
                    assert mock_send_task.call_count == 3
    
    @pytest.mark.asyncio
    async def test_single_task_execution_exhausted_retries(self, workflow_engine):
        """Test single task execution with exhausted retries."""
        
        task_id = str(uuid.uuid4())
        workflow_engine.max_retries = 1
        workflow_engine.retry_delay_seconds = 0.01
        
        # Mock database session and task
        with patch('app.core.workflow_engine.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_task = MagicMock()
            mock_task.id = task_id
            mock_task.task_type = TaskType.FEATURE_DEVELOPMENT
            mock_task.priority = TaskPriority.MEDIUM
            mock_task.required_capabilities = []
            mock_task.to_dict.return_value = {"id": task_id, "title": "Test Task"}
            
            mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
            
            # Mock task status updates and failing task execution
            with patch.object(workflow_engine, '_update_task_status') as mock_update_status:
                with patch.object(workflow_engine, '_send_task_to_agent') as mock_send_task:
                    mock_send_task.side_effect = Exception("Persistent failure")
                    
                    result = await workflow_engine._execute_single_task(task_id)
                    
                    assert result.task_id == task_id
                    assert result.status == TaskExecutionState.FAILED
                    assert "Failed after" in result.error
                    assert result.retry_count > workflow_engine.max_retries
    
    @pytest.mark.asyncio
    async def test_workflow_execution_plan_creation(self, workflow_engine, sample_workflow):
        """Test creation of workflow execution plan."""
        
        with patch.object(workflow_engine, 'resolve_dependencies') as mock_resolve:
            mock_resolve.return_value = [
                [str(sample_workflow.task_ids[0])],
                [str(sample_workflow.task_ids[1])],
                [str(sample_workflow.task_ids[2])]
            ]
            
            plan = await workflow_engine._create_execution_plan(sample_workflow)
            
            assert plan.workflow_id == str(sample_workflow.id)
            assert plan.total_tasks == 3
            assert len(plan.execution_batches) == 3
            assert plan.estimated_duration == sample_workflow.estimated_duration
    
    @pytest.mark.asyncio
    async def test_task_completion_handling(self, workflow_engine):
        """Test handling of task completion events."""
        
        task_id = str(uuid.uuid4())
        result = {"success": True, "output": "Task completed successfully"}
        
        with patch.object(workflow_engine, '_update_task_status') as mock_update_status:
            with patch.object(workflow_engine, '_emit_task_completion_event') as mock_emit:
                await workflow_engine.handle_task_completion(task_id, result)
                
                mock_update_status.assert_called_once_with(task_id, TaskStatus.COMPLETED)
                mock_emit.assert_called_once_with(task_id, result)
                
                # Check that task result is stored
                assert task_id in workflow_engine.task_executions
                stored_result = workflow_engine.task_executions[task_id]
                assert stored_result.task_id == task_id
                assert stored_result.status == TaskExecutionState.COMPLETED
                assert stored_result.result == result
    
    @pytest.mark.asyncio
    async def test_workflow_pause_and_cancel(self, workflow_engine):
        """Test workflow pause and cancel functionality."""
        
        workflow_id = str(uuid.uuid4())
        
        # Mock active workflow
        mock_task = AsyncMock()
        workflow_engine.active_workflows[workflow_id] = mock_task
        
        with patch.object(workflow_engine, '_update_workflow_status') as mock_update_status:
            # Test pause
            result = await workflow_engine.pause_workflow(workflow_id)
            assert result is True
            mock_task.cancel.assert_called_once()
            mock_update_status.assert_called_with(workflow_id, WorkflowStatus.PAUSED)
            
            # Reset mock
            mock_task.reset_mock()
            mock_update_status.reset_mock()
            
            # Add workflow back for cancel test
            workflow_engine.active_workflows[workflow_id] = mock_task
            
            # Test cancel
            result = await workflow_engine.cancel_workflow(workflow_id, "User requested")
            assert result is True
            mock_task.cancel.assert_called_once()
            mock_update_status.assert_called_with(workflow_id, WorkflowStatus.CANCELLED, "Cancelled: User requested")
    
    @pytest.mark.asyncio
    async def test_execution_metrics_tracking(self, workflow_engine):
        """Test that execution metrics are properly tracked."""
        
        initial_metrics = workflow_engine.get_metrics()
        assert initial_metrics['workflows_executed'] == 0
        assert initial_metrics['workflows_completed'] == 0
        
        # Simulate successful workflow completion
        result = WorkflowResult(
            workflow_id="test-workflow",
            status=WorkflowStatus.COMPLETED,
            execution_time=120.5,
            completed_tasks=5,
            failed_tasks=0,
            total_tasks=5,
            task_results=[]
        )
        
        workflow_engine._update_execution_metrics(result, 120.5)
        
        updated_metrics = workflow_engine.get_metrics()
        assert updated_metrics['workflows_executed'] == 1
        assert updated_metrics['workflows_completed'] == 1
        assert updated_metrics['average_execution_time'] == 120.5
        
        # Simulate failed workflow
        failed_result = WorkflowResult(
            workflow_id="test-workflow-2",
            status=WorkflowStatus.FAILED,
            execution_time=60.0,
            completed_tasks=2,
            failed_tasks=3,
            total_tasks=5,
            task_results=[]
        )
        
        workflow_engine._update_execution_metrics(failed_result, 60.0)
        
        final_metrics = workflow_engine.get_metrics()
        assert final_metrics['workflows_executed'] == 2
        assert final_metrics['workflows_completed'] == 1
        assert final_metrics['workflows_failed'] == 1
        assert final_metrics['average_execution_time'] == 90.25  # (120.5 + 60.0) / 2


class TestWorkflowEngineIntegration:
    """Integration tests for WorkflowEngine with orchestrator."""
    
    @pytest.fixture
    async def orchestrator_with_workflow_engine(self):
        """Create orchestrator with workflow engine for integration testing."""
        orchestrator = AgentOrchestrator()
        
        # Mock the initialization to avoid external dependencies
        with patch.object(orchestrator, 'message_broker', AsyncMock()):
            with patch.object(orchestrator, 'session_cache', AsyncMock()):
                workflow_engine = WorkflowEngine(orchestrator=orchestrator)
                await workflow_engine.initialize()
                orchestrator.workflow_engine = workflow_engine
                
                return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_workflow_execution_integration(self, orchestrator_with_workflow_engine):
        """Test workflow execution through orchestrator."""
        
        orchestrator = orchestrator_with_workflow_engine
        workflow_id = str(uuid.uuid4())
        
        # Mock workflow engine execution
        mock_result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.COMPLETED,
            execution_time=60.0,
            completed_tasks=3,
            failed_tasks=0,
            total_tasks=3,
            task_results=[]
        )
        
        with patch.object(orchestrator.workflow_engine, 'execute_workflow') as mock_execute:
            mock_execute.return_value = mock_result
            
            result = await orchestrator.execute_workflow(workflow_id)
            
            assert result.workflow_id == workflow_id
            assert result.status == WorkflowStatus.COMPLETED
            assert orchestrator.metrics['workflows_executed'] == 1
            assert orchestrator.metrics['workflows_completed'] == 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_workflow_status_integration(self, orchestrator_with_workflow_engine):
        """Test workflow status retrieval through orchestrator."""
        
        orchestrator = orchestrator_with_workflow_engine
        workflow_id = str(uuid.uuid4())
        
        mock_status = {
            'workflow_id': workflow_id,
            'is_active': True,
            'workflow_state': {'status': 'running'},
            'task_statuses': {},
            'metrics': {}
        }
        
        with patch.object(orchestrator.workflow_engine, 'get_execution_status') as mock_get_status:
            mock_get_status.return_value = mock_status
            
            status = await orchestrator.get_workflow_execution_status(workflow_id)
            
            assert status['workflow_id'] == workflow_id
            assert status['is_active'] is True
    
    @pytest.mark.asyncio
    async def test_system_status_includes_workflow_metrics(self, orchestrator_with_workflow_engine):
        """Test that system status includes workflow engine metrics."""
        
        orchestrator = orchestrator_with_workflow_engine
        
        # Mock health checks
        with patch.object(orchestrator, '_check_system_health') as mock_health:
            mock_health.return_value = {"overall": True}
            
            status = await orchestrator.get_system_status()
            
            assert "workflow_engine_active" in status
            assert status["workflow_engine_active"] is True
            assert "workflow_metrics" in status
            assert isinstance(status["workflow_metrics"], dict)


@pytest.mark.asyncio
class TestWorkflowEnginePerformance:
    """Performance tests for WorkflowEngine."""
    
    async def test_large_workflow_dependency_resolution_performance(self):
        """Test performance of dependency resolution with large workflows."""
        
        workflow_engine = WorkflowEngine()
        
        # Mock Redis initialization
        with patch('app.core.workflow_engine.get_message_broker') as mock_broker:
            mock_broker.return_value = AsyncMock()
            await workflow_engine.initialize()
        
        # Create large workflow with complex dependencies
        num_tasks = 100
        task_ids = [uuid.uuid4() for _ in range(num_tasks)]
        
        # Create dependency chain: each task depends on the previous one
        dependencies = {}
        for i in range(1, num_tasks):
            dependencies[str(task_ids[i])] = [str(task_ids[i-1])]
        
        workflow = Workflow(
            id=uuid.uuid4(),
            name="Large Performance Test Workflow",
            task_ids=task_ids,
            dependencies=dependencies,
            total_tasks=num_tasks
        )
        
        start_time = datetime.utcnow()
        execution_batches = await workflow_engine.resolve_dependencies(workflow)
        resolution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Should resolve dependencies quickly even for large workflows
        assert resolution_time < 1.0  # Less than 1 second
        assert len(execution_batches) == num_tasks  # Sequential execution
        assert all(len(batch) == 1 for batch in execution_batches)  # One task per batch
    
    async def test_concurrent_workflow_execution_performance(self):
        """Test performance with multiple concurrent workflows."""
        
        workflow_engine = WorkflowEngine()
        
        # Mock Redis initialization
        with patch('app.core.workflow_engine.get_message_broker') as mock_broker:
            mock_broker.return_value = AsyncMock()
            await workflow_engine.initialize()
        
        # Create multiple small workflows
        num_workflows = 5
        workflows = []
        
        for i in range(num_workflows):
            task_ids = [uuid.uuid4() for _ in range(3)]
            workflow = Workflow(
                id=uuid.uuid4(),
                name=f"Concurrent Test Workflow {i}",
                task_ids=task_ids,
                dependencies={},  # Parallel tasks
                total_tasks=3
            )
            workflows.append(workflow)
        
        # Mock workflow loading and validation
        with patch.object(workflow_engine, '_load_and_validate_workflow') as mock_load:
            with patch.object(workflow_engine, '_execute_workflow_internal') as mock_execute:
                # Mock successful execution
                def create_mock_result(workflow):
                    return WorkflowResult(
                        workflow_id=str(workflow.id),
                        status=WorkflowStatus.COMPLETED,
                        execution_time=30.0,
                        completed_tasks=3,
                        failed_tasks=0,
                        total_tasks=3,
                        task_results=[]
                    )
                
                mock_load.side_effect = workflows
                mock_execute.side_effect = [create_mock_result(w) for w in workflows]
                
                # Execute all workflows concurrently
                start_time = datetime.utcnow()
                tasks = [
                    workflow_engine.execute_workflow(str(workflow.id))
                    for workflow in workflows
                ]
                results = await asyncio.gather(*tasks)
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # All workflows should complete successfully
                assert len(results) == num_workflows
                assert all(r.status == WorkflowStatus.COMPLETED for r in results)
                
                # Concurrent execution should be faster than sequential
                assert execution_time < num_workflows * 30.0  # Much faster than sequential


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.core.workflow_engine", "--cov-report=term-missing"])