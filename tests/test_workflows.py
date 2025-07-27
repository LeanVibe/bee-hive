"""
Comprehensive tests for Workflow management functionality.

Tests cover workflow creation, execution, dependency management,
progress tracking, and validation with >90% coverage.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.workflow import Workflow, WorkflowStatus, WorkflowPriority
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.api.v1.workflows import *
from app.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowTaskAssignment


class TestWorkflowModel:
    """Test suite for Workflow model functionality."""
    
    @pytest.fixture
    def sample_workflow(self):
        """Create a sample workflow for testing."""
        return Workflow(
            id=uuid.uuid4(),
            name="Test Workflow",
            description="A test workflow for development",
            priority=WorkflowPriority.MEDIUM,
            definition={"type": "sequential", "steps": ["task1", "task2"]},
            context={"project": "test"},
            variables={"env": "testing"},
            estimated_duration=120
        )
    
    def test_workflow_initialization(self, sample_workflow):
        """Test workflow model initialization."""
        assert sample_workflow.name == "Test Workflow"
        assert sample_workflow.status == WorkflowStatus.CREATED
        assert sample_workflow.priority == WorkflowPriority.MEDIUM
        assert sample_workflow.total_tasks == 0
        assert sample_workflow.completed_tasks == 0
        assert sample_workflow.failed_tasks == 0
    
    def test_workflow_to_dict(self, sample_workflow):
        """Test workflow serialization to dictionary."""
        workflow_dict = sample_workflow.to_dict()
        
        assert workflow_dict["name"] == "Test Workflow"
        assert workflow_dict["status"] == "created"
        assert workflow_dict["priority"] == "medium"
        assert "definition" in workflow_dict
        assert "context" in workflow_dict
    
    def test_start_execution(self, sample_workflow):
        """Test workflow execution start."""
        sample_workflow.start_execution()
        
        assert sample_workflow.status == WorkflowStatus.RUNNING
        assert sample_workflow.started_at is not None
    
    def test_complete_successfully(self, sample_workflow):
        """Test successful workflow completion."""
        sample_workflow.start_execution()
        
        result = {"success": True, "output": "All tasks completed"}
        sample_workflow.complete_successfully(result)
        
        assert sample_workflow.status == WorkflowStatus.COMPLETED
        assert sample_workflow.completed_at is not None
        assert sample_workflow.result == result
        assert sample_workflow.actual_duration is not None
    
    def test_fail_with_error(self, sample_workflow):
        """Test workflow failure with error."""
        sample_workflow.start_execution()
        
        error_message = "Task execution failed"
        sample_workflow.fail_with_error(error_message)
        
        assert sample_workflow.status == WorkflowStatus.FAILED
        assert sample_workflow.completed_at is not None
        assert sample_workflow.error_message == error_message
        assert sample_workflow.actual_duration is not None
    
    def test_pause_and_resume_execution(self, sample_workflow):
        """Test workflow pause and resume functionality."""
        sample_workflow.start_execution()
        
        # Pause workflow
        sample_workflow.pause_execution()
        assert sample_workflow.status == WorkflowStatus.PAUSED
        
        # Resume workflow
        sample_workflow.resume_execution()
        assert sample_workflow.status == WorkflowStatus.RUNNING
    
    def test_cancel_execution(self, sample_workflow):
        """Test workflow cancellation."""
        sample_workflow.start_execution()
        
        reason = "User requested cancellation"
        sample_workflow.cancel_execution(reason)
        
        assert sample_workflow.status == WorkflowStatus.CANCELLED
        assert sample_workflow.completed_at is not None
        assert reason in sample_workflow.error_message
    
    def test_add_task_to_workflow(self, sample_workflow):
        """Test adding tasks to workflow."""
        task_id = uuid.uuid4()
        dependencies = [uuid.uuid4(), uuid.uuid4()]
        
        sample_workflow.add_task(task_id, dependencies)
        
        assert task_id in sample_workflow.task_ids
        assert sample_workflow.total_tasks == 1
        assert str(task_id) in sample_workflow.dependencies
        assert sample_workflow.dependencies[str(task_id)] == [str(dep) for dep in dependencies]
    
    def test_remove_task_from_workflow(self, sample_workflow):
        """Test removing tasks from workflow."""
        task_id = uuid.uuid4()
        sample_workflow.add_task(task_id)
        
        sample_workflow.remove_task(task_id)
        
        assert task_id not in sample_workflow.task_ids
        assert sample_workflow.total_tasks == 0
        assert str(task_id) not in sample_workflow.dependencies
    
    def test_get_ready_tasks_no_dependencies(self, sample_workflow):
        """Test getting ready tasks with no dependencies."""
        task1 = uuid.uuid4()
        task2 = uuid.uuid4()
        
        sample_workflow.add_task(task1)
        sample_workflow.add_task(task2)
        
        ready_tasks = sample_workflow.get_ready_tasks([])
        
        assert len(ready_tasks) == 2
        assert str(task1) in ready_tasks
        assert str(task2) in ready_tasks
    
    def test_get_ready_tasks_with_dependencies(self, sample_workflow):
        """Test getting ready tasks with dependencies."""
        task1 = uuid.uuid4()
        task2 = uuid.uuid4()
        task3 = uuid.uuid4()
        
        sample_workflow.add_task(task1)  # No dependencies
        sample_workflow.add_task(task2, [task1])  # Depends on task1
        sample_workflow.add_task(task3, [task1, task2])  # Depends on task1 and task2
        
        # Initially, only task1 should be ready
        ready_tasks = sample_workflow.get_ready_tasks([])
        assert ready_tasks == [str(task1)]
        
        # After task1 completes, task2 should be ready
        ready_tasks = sample_workflow.get_ready_tasks([str(task1)])
        assert ready_tasks == [str(task2)]
        
        # After both task1 and task2 complete, task3 should be ready
        ready_tasks = sample_workflow.get_ready_tasks([str(task1), str(task2)])
        assert ready_tasks == [str(task3)]
    
    def test_validate_dependencies_no_cycles(self, sample_workflow):
        """Test dependency validation with no circular dependencies."""
        task1 = uuid.uuid4()
        task2 = uuid.uuid4()
        task3 = uuid.uuid4()
        
        sample_workflow.add_task(task1)
        sample_workflow.add_task(task2, [task1])
        sample_workflow.add_task(task3, [task2])
        
        errors = sample_workflow.validate_dependencies()
        assert len(errors) == 0
    
    def test_validate_dependencies_with_cycles(self, sample_workflow):
        """Test dependency validation with circular dependencies."""
        task1 = uuid.uuid4()
        task2 = uuid.uuid4()
        
        # Create circular dependency
        sample_workflow.dependencies = {
            str(task1): [str(task2)],
            str(task2): [str(task1)]
        }
        
        errors = sample_workflow.validate_dependencies()
        assert len(errors) > 0
        assert "Circular dependency detected" in errors[0]
    
    def test_update_progress(self, sample_workflow):
        """Test workflow progress updates."""
        # Add tasks to workflow
        for i in range(5):
            sample_workflow.add_task(uuid.uuid4())
        
        # Update progress - complete all tasks with some failures
        sample_workflow.update_progress(completed_tasks=3, failed_tasks=2)
        
        assert sample_workflow.completed_tasks == 3
        assert sample_workflow.failed_tasks == 2
        
        # Should auto-complete since all 5 tasks are done (3 completed + 2 failed)
        assert sample_workflow.status == WorkflowStatus.FAILED  # Because there are failed tasks
    
    def test_get_completion_percentage(self, sample_workflow):
        """Test completion percentage calculation."""
        # Add tasks and update progress
        for i in range(10):
            sample_workflow.add_task(uuid.uuid4())
        
        sample_workflow.completed_tasks = 7
        
        percentage = sample_workflow.get_completion_percentage()
        assert percentage == 70.0
    
    def test_estimate_completion_time(self, sample_workflow):
        """Test completion time estimation."""
        sample_workflow.start_execution()
        sample_workflow.total_tasks = 10
        sample_workflow.completed_tasks = 3
        
        # Mock some elapsed time
        sample_workflow.started_at = datetime.utcnow() - timedelta(minutes=30)
        
        estimated_completion = sample_workflow.estimate_completion_time()
        
        assert estimated_completion is not None
        assert estimated_completion > datetime.utcnow()


class TestWorkflowAPI:
    """Test suite for Workflow API endpoints."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return AsyncMock()
    
    @pytest.fixture
    def sample_workflow_create(self):
        """Create sample workflow creation data."""
        return WorkflowCreate(
            name="Test API Workflow",
            description="Testing workflow API",
            priority=WorkflowPriority.HIGH,
            definition={"type": "parallel"},
            estimated_duration=90
        )
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_create_workflow_success(self, mock_get_session, sample_workflow_create):
        """Test successful workflow creation via API."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock database operations
        mock_workflow = MagicMock()
        mock_workflow.id = uuid.uuid4()
        mock_workflow.name = sample_workflow_create.name
        mock_db_session.refresh = AsyncMock()
        
        with patch('app.api.v1.workflows.Workflow') as mock_workflow_class:
            mock_workflow_class.return_value = mock_workflow
            
            result = await create_workflow(sample_workflow_create, mock_db_session)
            
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            mock_db_session.refresh.assert_called_once()
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_list_workflows_with_filters(self, mock_get_session):
        """Test workflow listing with status and priority filters."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock database query results
        mock_workflows = [MagicMock() for _ in range(3)]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_workflows
        mock_db_session.execute.return_value.scalar.return_value = 3
        
        with patch('app.api.v1.workflows.select') as mock_select:
            result = await list_workflows(
                status=WorkflowStatus.RUNNING,
                priority=WorkflowPriority.HIGH,
                limit=10,
                offset=0,
                db=mock_db_session
            )
            
            assert len(result.workflows) == 3
            assert result.total == 3
            mock_db_session.execute.assert_called()
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_get_workflow_success(self, mock_get_session):
        """Test successful workflow retrieval."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        
        result = await get_workflow(workflow_id, mock_db_session)
        
        mock_db_session.execute.assert_called_once()
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_get_workflow_not_found(self, mock_get_session):
        """Test workflow retrieval when workflow doesn't exist."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        workflow_id = uuid.uuid4()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_workflow(workflow_id, mock_db_session)
        
        assert exc_info.value.status_code == 404
        assert "Workflow not found" in str(exc_info.value.detail)
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_add_task_to_workflow_success(self, mock_get_session):
        """Test successful task addition to workflow."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        task_id = uuid.uuid4()
        
        # Mock workflow and task existence
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.task_ids = []
        mock_workflow.dependencies = {}
        
        mock_task = MagicMock()
        mock_task.id = task_id
        
        mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [mock_workflow, mock_task]
        mock_db_session.execute.return_value.all.return_value = [(task_id,)]
        
        task_assignment = WorkflowTaskAssignment(
            task_id=task_id,
            dependencies=[]
        )
        
        result = await add_task_to_workflow(workflow_id, task_assignment, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_execute_workflow_success(self, mock_get_session):
        """Test successful workflow execution."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.status = WorkflowStatus.READY
        mock_workflow.validate_dependencies.return_value = []  # No validation errors
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        
        result = await execute_workflow(workflow_id, None, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called()
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_execute_workflow_validation_error(self, mock_get_session):
        """Test workflow execution with validation errors."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.status = WorkflowStatus.READY
        mock_workflow.validate_dependencies.return_value = ["Circular dependency detected"]
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        
        with pytest.raises(HTTPException) as exc_info:
            await execute_workflow(workflow_id, None, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "validation failed" in str(exc_info.value.detail)
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_get_workflow_progress(self, mock_get_session):
        """Test workflow progress retrieval."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        task_id = uuid.uuid4()
        
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.name = "Test Workflow"
        mock_workflow.status = WorkflowStatus.RUNNING
        mock_workflow.task_ids = [task_id]
        mock_workflow.total_tasks = 1
        mock_workflow.completed_tasks = 0
        mock_workflow.failed_tasks = 0
        mock_workflow.get_completion_percentage.return_value = 50.0
        mock_workflow.estimate_completion_time.return_value = datetime.utcnow() + timedelta(hours=1)
        mock_workflow.get_ready_tasks.return_value = [str(task_id)]
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        mock_db_session.execute.return_value.all.return_value = [(task_id, TaskStatus.PENDING)]
        
        result = await get_workflow_progress(workflow_id, mock_db_session)
        
        assert result.workflow_id == workflow_id
        assert result.completion_percentage == 50.0
        assert len(result.ready_tasks) == 1
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_validate_workflow_success(self, mock_get_session):
        """Test successful workflow validation."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        task_id = uuid.uuid4()
        
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.task_ids = [task_id]
        mock_workflow.validate_dependencies.return_value = []  # No errors
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = [task_id]
        
        result = await validate_workflow(workflow_id, mock_db_session)
        
        assert result.workflow_id == workflow_id
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    @patch('app.api.v1.workflows.get_session_dependency')
    async def test_validate_workflow_missing_tasks(self, mock_get_session):
        """Test workflow validation with missing tasks."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        workflow_id = uuid.uuid4()
        task_id = uuid.uuid4()
        
        mock_workflow = MagicMock()
        mock_workflow.id = workflow_id
        mock_workflow.task_ids = [task_id]
        mock_workflow.validate_dependencies.return_value = []
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_workflow
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []  # No tasks found
        
        result = await validate_workflow(workflow_id, mock_db_session)
        
        assert result.workflow_id == workflow_id
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "Missing tasks" in result.errors[0]


@pytest.mark.asyncio
class TestWorkflowIntegration:
    """Integration tests for workflow functionality."""
    
    async def test_complete_workflow_lifecycle(self):
        """Test complete workflow lifecycle from creation to completion."""
        # This would be an integration test with real database
        # For now, we'll test the logic flow
        
        workflow = Workflow(
            name="Integration Test Workflow",
            description="Testing complete lifecycle"
        )
        
        # Add tasks with dependencies
        task1 = uuid.uuid4()
        task2 = uuid.uuid4()
        task3 = uuid.uuid4()
        
        workflow.add_task(task1)  # Independent task
        workflow.add_task(task2, [task1])  # Depends on task1
        workflow.add_task(task3, [task2])  # Depends on task2
        
        # Validate dependencies
        errors = workflow.validate_dependencies()
        assert len(errors) == 0
        
        # Start execution
        workflow.start_execution()
        assert workflow.status == WorkflowStatus.RUNNING
        
        # Simulate task completion
        ready_tasks = workflow.get_ready_tasks([])
        assert str(task1) in ready_tasks
        
        ready_tasks = workflow.get_ready_tasks([str(task1)])
        assert str(task2) in ready_tasks
        
        ready_tasks = workflow.get_ready_tasks([str(task1), str(task2)])
        assert str(task3) in ready_tasks
        
        # Complete workflow
        workflow.update_progress(3, 0)
        assert workflow.status == WorkflowStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.models.workflow", "--cov=app.api.v1.workflows", "--cov-report=term-missing"])