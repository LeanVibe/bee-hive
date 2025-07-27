"""
Comprehensive tests for Task API functionality.

Tests cover task CRUD operations, assignment, lifecycle management,
and agent interaction with >90% coverage.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from app.api.v1.tasks import *
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.agent import Agent, AgentStatus, AgentType
from app.schemas.task import TaskCreate, TaskUpdate


class TestTaskAPI:
    """Test suite for Task API endpoints."""
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return AsyncMock()
    
    @pytest.fixture
    def sample_task_create(self):
        """Create sample task creation data."""
        return TaskCreate(
            title="Implement user authentication",
            description="Add JWT-based authentication system",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority=TaskPriority.HIGH,
            required_capabilities=["python", "fastapi", "jwt"],
            estimated_effort=240,
            context={"project": "api-service", "module": "auth"}
        )
    
    @pytest.fixture
    def sample_task_update(self):
        """Create sample task update data."""
        return TaskUpdate(
            title="Updated task title",
            description="Updated description",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.MEDIUM,
            result={"progress": "50%"}
        )
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock task instance."""
        task = MagicMock()
        task.id = uuid.uuid4()
        task.title = "Test Task"
        task.status = TaskStatus.PENDING
        task.priority = TaskPriority.MEDIUM
        task.assigned_agent_id = None
        task.retry_count = 0
        task.max_retries = 3
        task.started_at = None
        return task
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent instance."""
        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.name = "Test Agent"
        agent.status = AgentStatus.ACTIVE
        return agent
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_create_task_success(self, mock_get_session, sample_task_create):
        """Test successful task creation."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock task creation
        with patch('app.api.v1.tasks.Task') as mock_task_class:
            mock_task = MagicMock()
            mock_task.id = uuid.uuid4()
            mock_task.title = sample_task_create.title
            mock_task_class.return_value = mock_task
            
            result = await create_task(sample_task_create, mock_db_session)
            
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            mock_db_session.refresh.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_create_task_failure(self, mock_get_session, sample_task_create):
        """Test task creation failure with database error."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.commit.side_effect = Exception("Database error")
        
        with pytest.raises(HTTPException) as exc_info:
            await create_task(sample_task_create, mock_db_session)
        
        assert exc_info.value.status_code == 500
        assert "Failed to create task" in str(exc_info.value.detail)
        mock_db_session.rollback.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_list_tasks_with_filters(self, mock_get_session):
        """Test task listing with various filters."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock database query results
        mock_tasks = [MagicMock() for _ in range(5)]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_tasks
        mock_db_session.execute.return_value.scalar.return_value = 5
        
        result = await list_tasks(
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            task_type=TaskType.FEATURE_DEVELOPMENT,
            assigned_agent_id=uuid.uuid4(),
            limit=10,
            offset=0,
            db=mock_db_session
        )
        
        assert len(result.tasks) == 5
        assert result.total == 5
        assert result.limit == 10
        assert result.offset == 0
        mock_db_session.execute.assert_called()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_get_task_success(self, mock_get_session, mock_task):
        """Test successful task retrieval."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        
        result = await get_task(mock_task.id, mock_db_session)
        
        mock_db_session.execute.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_get_task_not_found(self, mock_get_session):
        """Test task retrieval when task doesn't exist."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        task_id = uuid.uuid4()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_task(task_id, mock_db_session)
        
        assert exc_info.value.status_code == 404
        assert "Task not found" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_update_task_success(self, mock_get_session, mock_task, sample_task_update):
        """Test successful task update."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock task existence check and update
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        mock_db_session.execute.return_value.scalar_one.return_value = mock_task
        
        result = await update_task(mock_task.id, sample_task_update, mock_db_session)
        
        assert mock_db_session.execute.call_count >= 2  # One for check, one for update, one for fetch
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_update_task_not_found(self, mock_get_session, sample_task_update):
        """Test task update when task doesn't exist."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        task_id = uuid.uuid4()
        
        with pytest.raises(HTTPException) as exc_info:
            await update_task(task_id, sample_task_update, mock_db_session)
        
        assert exc_info.value.status_code == 404
        assert "Task not found" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_delete_task_success(self, mock_get_session, mock_task):
        """Test successful task deletion (cancellation)."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        
        await delete_task(mock_task.id, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_assign_task_to_agent_success(self, mock_get_session, mock_task, mock_agent):
        """Test successful task assignment to agent."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Mock task and agent existence
        mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [mock_task, mock_agent, mock_task]
        
        result = await assign_task_to_agent(mock_task.id, mock_agent.id, mock_db_session)
        
        assert mock_db_session.execute.call_count >= 3
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_assign_task_invalid_status(self, mock_get_session, mock_task, mock_agent):
        """Test task assignment with invalid task status."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task status to not assignable
        mock_task.status = TaskStatus.COMPLETED
        mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [mock_task, mock_agent]
        
        with pytest.raises(HTTPException) as exc_info:
            await assign_task_to_agent(mock_task.id, mock_agent.id, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "cannot be assigned" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_assign_task_agent_not_available(self, mock_get_session, mock_task, mock_agent):
        """Test task assignment when agent is not available."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set agent status to not available
        mock_agent.status = AgentStatus.BUSY
        mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [mock_task, mock_agent]
        
        with pytest.raises(HTTPException) as exc_info:
            await assign_task_to_agent(mock_task.id, mock_agent.id, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "not available" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_start_task_success(self, mock_get_session, mock_task):
        """Test successful task start."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task status to assigned
        mock_task.status = TaskStatus.ASSIGNED
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        mock_db_session.execute.return_value.scalar_one.return_value = mock_task
        
        result = await start_task(mock_task.id, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_start_task_invalid_status(self, mock_get_session, mock_task):
        """Test task start with invalid status."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task status to not startable
        mock_task.status = TaskStatus.PENDING
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        
        with pytest.raises(HTTPException) as exc_info:
            await start_task(mock_task.id, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "cannot be started" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_complete_task_success(self, mock_get_session, mock_task):
        """Test successful task completion."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task status and start time
        mock_task.status = TaskStatus.IN_PROGRESS
        mock_task.started_at = datetime.utcnow() - timedelta(minutes=30)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        mock_db_session.execute.return_value.scalar_one.return_value = mock_task
        
        result_data = {"status": "success", "output": "Task completed successfully"}
        
        result = await complete_task(mock_task.id, result_data, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_complete_task_invalid_status(self, mock_get_session, mock_task):
        """Test task completion with invalid status."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task status to not completable
        mock_task.status = TaskStatus.PENDING
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        
        result_data = {"status": "success"}
        
        with pytest.raises(HTTPException) as exc_info:
            await complete_task(mock_task.id, result_data, mock_db_session)
        
        assert exc_info.value.status_code == 400
        assert "cannot be completed" in str(exc_info.value.detail)
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_fail_task_with_retry(self, mock_get_session, mock_task):
        """Test task failure with retry capability."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task retry settings
        mock_task.retry_count = 1
        mock_task.max_retries = 3
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        mock_db_session.execute.return_value.scalar_one.return_value = mock_task
        
        error_message = "Task execution failed due to network error"
        
        result = await fail_task(mock_task.id, error_message, True, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_fail_task_max_retries_reached(self, mock_get_session, mock_task):
        """Test task failure when max retries are reached."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        # Set task at max retries
        mock_task.retry_count = 2
        mock_task.max_retries = 3
        mock_task.started_at = datetime.utcnow() - timedelta(minutes=15)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_task
        mock_db_session.execute.return_value.scalar_one.return_value = mock_task
        
        error_message = "Final failure after max retries"
        
        result = await fail_task(mock_task.id, error_message, True, mock_db_session)
        
        mock_db_session.execute.assert_called()
        mock_db_session.commit.assert_called_once()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_get_agent_tasks(self, mock_get_session):
        """Test retrieving tasks assigned to a specific agent."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        
        agent_id = uuid.uuid4()
        mock_tasks = [MagicMock() for _ in range(3)]
        
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = mock_tasks
        mock_db_session.execute.return_value.scalar.return_value = 3
        
        result = await get_agent_tasks(
            agent_id=agent_id,
            status=TaskStatus.IN_PROGRESS,
            limit=10,
            offset=0,
            db=mock_db_session
        )
        
        assert len(result.tasks) == 3
        assert result.total == 3
        mock_db_session.execute.assert_called()
    
    @patch('app.api.v1.tasks.get_session_dependency')
    async def test_list_tasks_database_error(self, mock_get_session):
        """Test task listing with database error."""
        mock_db_session = AsyncMock()
        mock_get_session.return_value = mock_db_session
        mock_db_session.execute.side_effect = Exception("Database connection failed")
        
        with pytest.raises(HTTPException) as exc_info:
            await list_tasks(db=mock_db_session)
        
        assert exc_info.value.status_code == 500
        assert "Failed to retrieve tasks" in str(exc_info.value.detail)


class TestTaskModel:
    """Test suite for Task model methods."""
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id=uuid.uuid4(),
            title="Test Task",
            description="A test task for unit testing",
            task_type=TaskType.FEATURE_DEVELOPMENT,
            status=TaskStatus.PENDING,
            priority=TaskPriority.MEDIUM,
            required_capabilities=["python", "testing"],
            estimated_effort=120
        )
    
    def test_task_initialization(self, sample_task):
        """Test task model initialization."""
        assert sample_task.title == "Test Task"
        assert sample_task.status == TaskStatus.PENDING
        assert sample_task.priority == TaskPriority.MEDIUM
        assert sample_task.retry_count == 0
        assert sample_task.max_retries == 3
    
    def test_task_to_dict(self, sample_task):
        """Test task serialization to dictionary."""
        task_dict = sample_task.to_dict()
        
        assert task_dict["title"] == "Test Task"
        assert task_dict["status"] == "pending"
        assert task_dict["priority"] == "medium"
        assert task_dict["required_capabilities"] == ["python", "testing"]
    
    def test_can_be_started_no_dependencies(self, sample_task):
        """Test task can be started when it has no dependencies."""
        result = sample_task.can_be_started([])
        assert result is True
    
    def test_can_be_started_with_dependencies(self, sample_task):
        """Test task can be started when dependencies are met."""
        dep1 = uuid.uuid4()
        dep2 = uuid.uuid4()
        
        sample_task.dependencies = [dep1, dep2]
        
        # Dependencies not met
        result = sample_task.can_be_started([str(dep1)])
        assert result is False
        
        # All dependencies met
        result = sample_task.can_be_started([str(dep1), str(dep2)])
        assert result is True
    
    def test_assign_to_agent(self, sample_task):
        """Test task assignment to agent."""
        agent_id = uuid.uuid4()
        
        sample_task.assign_to_agent(agent_id)
        
        assert sample_task.assigned_agent_id == agent_id
        assert sample_task.status == TaskStatus.ASSIGNED
        assert sample_task.assigned_at is not None
    
    def test_start_execution(self, sample_task):
        """Test task execution start."""
        sample_task.start_execution()
        
        assert sample_task.status == TaskStatus.IN_PROGRESS
        assert sample_task.started_at is not None
    
    def test_complete_successfully(self, sample_task):
        """Test successful task completion."""
        sample_task.start_execution()
        
        result = {"status": "success", "output": "Task completed"}
        sample_task.complete_successfully(result)
        
        assert sample_task.status == TaskStatus.COMPLETED
        assert sample_task.completed_at is not None
        assert sample_task.result == result
        assert sample_task.actual_effort is not None
    
    def test_fail_with_error_retry_available(self, sample_task):
        """Test task failure with retry available."""
        error_message = "Network timeout"
        
        sample_task.fail_with_error(error_message, can_retry=True)
        
        assert sample_task.error_message == error_message
        assert sample_task.retry_count == 1
        assert sample_task.status == TaskStatus.PENDING  # Reset for retry
    
    def test_fail_with_error_max_retries_reached(self, sample_task):
        """Test task failure when max retries are reached."""
        sample_task.retry_count = 2  # One less than max
        sample_task.start_execution()
        
        error_message = "Final failure"
        sample_task.fail_with_error(error_message, can_retry=True)
        
        assert sample_task.status == TaskStatus.FAILED
        assert sample_task.completed_at is not None
        assert sample_task.actual_effort is not None
    
    def test_block_with_reason(self, sample_task):
        """Test task blocking."""
        reason = "Waiting for external dependency"
        
        sample_task.block_with_reason(reason)
        
        assert sample_task.status == TaskStatus.BLOCKED
        assert sample_task.error_message == reason
    
    def test_calculate_urgency_score_no_due_date(self, sample_task):
        """Test urgency score calculation without due date."""
        score = sample_task.calculate_urgency_score()
        
        expected_score = TaskPriority.MEDIUM.value / 10.0
        assert score == expected_score
    
    def test_calculate_urgency_score_with_due_date(self, sample_task):
        """Test urgency score calculation with due date."""
        # Set due date to tomorrow
        sample_task.due_date = datetime.utcnow() + timedelta(days=1)
        
        score = sample_task.calculate_urgency_score()
        
        base_score = TaskPriority.MEDIUM.value / 10.0
        assert score >= base_score  # Should have urgency boost
    
    def test_calculate_urgency_score_overdue(self, sample_task):
        """Test urgency score calculation for overdue task."""
        # Set due date to yesterday
        sample_task.due_date = datetime.utcnow() - timedelta(days=1)
        
        score = sample_task.calculate_urgency_score()
        
        assert score == 1.0  # Maximum urgency for overdue tasks
    
    def test_add_and_remove_dependency(self, sample_task):
        """Test adding and removing task dependencies."""
        dep1 = uuid.uuid4()
        dep2 = uuid.uuid4()
        
        # Add dependencies
        sample_task.add_dependency(dep1)
        sample_task.add_dependency(dep2)
        
        assert dep1 in sample_task.dependencies
        assert dep2 in sample_task.dependencies
        
        # Remove dependency
        sample_task.remove_dependency(dep1)
        
        assert dep1 not in sample_task.dependencies
        assert dep2 in sample_task.dependencies
    
    def test_add_blocking_task(self, sample_task):
        """Test adding blocking task relationship."""
        blocking_task_id = uuid.uuid4()
        
        sample_task.add_blocking_task(blocking_task_id)
        
        assert blocking_task_id in sample_task.blocking_tasks
    
    def test_get_estimated_completion_time(self, sample_task):
        """Test estimated completion time calculation."""
        sample_task.estimated_effort = 60  # 1 hour
        sample_task.start_execution()
        
        estimated_completion = sample_task.get_estimated_completion_time()
        
        assert estimated_completion is not None
        assert estimated_completion > datetime.utcnow()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app.api.v1.tasks", "--cov=app.models.task", "--cov-report=term-missing"])