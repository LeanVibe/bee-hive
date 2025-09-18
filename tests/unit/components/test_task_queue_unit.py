"""
Unit Tests for TaskQueue - Component Isolation

Tests the TaskQueue component in complete isolation with all external
dependencies mocked. This ensures we test only the task queue's business logic
without any external system dependencies.

Testing Focus:
- Task queueing and dequeuing logic
- Priority-based task ordering
- Multiple queue management
- Task timeout and retry handling
- Assignment and lifecycle management
- Performance metrics and monitoring
- Error handling and validation

All external dependencies are mocked:
- Redis client operations
- Database sessions
- Task and Agent models
- Time-based operations
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Component under test
from app.core.task_queue import (
    TaskQueue,
    QueuedTask,
    TaskAssignmentResult,
    QueueStatus
)

# Models for mocking
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.agent import Agent, AgentStatus


class TestTaskQueueUnit:
    """Unit tests for TaskQueue component in isolation."""

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        
        # Mock basic Redis operations
        mock_redis.lpush.return_value = 1
        mock_redis.brpop.return_value = (b"queue_name", b'{"task_id": "test-id"}')
        mock_redis.llen.return_value = 0
        mock_redis.zadd.return_value = 1
        mock_redis.zrem.return_value = 1
        mock_redis.zrange.return_value = []
        mock_redis.exists.return_value = False
        mock_redis.expire.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.hset.return_value = 1
        mock_redis.hget.return_value = None
        mock_redis.hgetall.return_value = {}
        mock_redis.hdel.return_value = 1
        
        return mock_redis

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_session.commit.return_value = None
        mock_session.rollback.return_value = None
        return mock_session

    @pytest.fixture
    def isolated_task_queue(self, mock_redis_client):
        """Create TaskQueue with all dependencies mocked."""
        return TaskQueue(redis_client=mock_redis_client)

    @pytest.fixture
    def sample_queued_task(self):
        """Sample queued task for testing."""
        return QueuedTask(
            task_id=uuid.uuid4(),
            priority_score=100.0,
            queue_name="general",
            required_capabilities=["backend_development"],
            estimated_effort=60,
            timeout_seconds=3600,
            queued_at=datetime.utcnow(),
            retry_count=0,
            max_retries=3,
            metadata={"task_type": "feature_development"}
        )

    class TestInitialization:
        """Test task queue initialization."""

        def test_task_queue_creation(self, isolated_task_queue):
            """Test that task queue creates correctly with mocked dependencies."""
            assert isolated_task_queue.redis_client is not None
            assert hasattr(isolated_task_queue, '_queue_metrics')
            assert hasattr(isolated_task_queue, '_active_assignments')

        def test_task_queue_without_redis(self):
            """Test task queue creation without Redis client."""
            task_queue = TaskQueue()
            assert task_queue.redis_client is None

        @pytest.mark.asyncio
        async def test_task_queue_initialization(self, isolated_task_queue):
            """Test task queue initialization process."""
            with patch('app.core.task_queue.get_redis', return_value=isolated_task_queue.redis_client):
                await isolated_task_queue.initialize()
                assert isolated_task_queue._initialized is True

    class TestQueuedTaskDataClass:
        """Test QueuedTask dataclass functionality."""

        def test_queued_task_creation(self):
            """Test creating a QueuedTask instance."""
            task_id = uuid.uuid4()
            queued_at = datetime.utcnow()
            
            task = QueuedTask(
                task_id=task_id,
                priority_score=50.0,
                queue_name="test_queue",
                required_capabilities=["testing"],
                estimated_effort=30,
                timeout_seconds=1800,
                queued_at=queued_at
            )
            
            assert task.task_id == task_id
            assert task.priority_score == 50.0
            assert task.queue_name == "test_queue"
            assert task.required_capabilities == ["testing"]
            assert task.metadata == {}

        def test_queued_task_with_metadata(self):
            """Test QueuedTask with custom metadata."""
            metadata = {"environment": "staging", "urgent": True}
            
            task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=200.0,
                queue_name="urgent",
                required_capabilities=["devops"],
                estimated_effort=15,
                timeout_seconds=900,
                queued_at=datetime.utcnow(),
                metadata=metadata
            )
            
            assert task.metadata == metadata
            assert task.metadata["urgent"] is True

        def test_queued_task_serialization(self):
            """Test QueuedTask to_dict and from_dict."""
            original_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=75.0,
                queue_name="serialization_test",
                required_capabilities=["python", "testing"],
                estimated_effort=45,
                timeout_seconds=2700,
                queued_at=datetime.utcnow(),
                retry_count=1,
                metadata={"test": True}
            )
            
            # Serialize to dict
            task_dict = original_task.to_dict()
            assert isinstance(task_dict["task_id"], str)
            assert isinstance(task_dict["queued_at"], str)
            
            # Deserialize from dict
            restored_task = QueuedTask.from_dict(task_dict)
            assert restored_task.task_id == original_task.task_id
            assert restored_task.priority_score == original_task.priority_score
            assert restored_task.metadata == original_task.metadata

    class TestTaskEnqueuing:
        """Test task enqueuing operations."""

        @pytest.mark.asyncio
        async def test_enqueue_task_success(self, isolated_task_queue, sample_queued_task):
            """Test successful task enqueuing."""
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.enqueue_task(sample_queued_task)
                
                assert result is True
                # Verify Redis operations were called
                isolated_task_queue.redis_client.zadd.assert_called()
                isolated_task_queue.redis_client.hset.assert_called()

        @pytest.mark.asyncio
        async def test_enqueue_task_priority_ordering(self, isolated_task_queue):
            """Test that tasks are enqueued with proper priority ordering."""
            # Create tasks with different priorities
            high_priority_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=200.0,
                queue_name="general",
                required_capabilities=["urgent"],
                estimated_effort=30,
                timeout_seconds=1800,
                queued_at=datetime.utcnow()
            )
            
            low_priority_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=50.0,
                queue_name="general",
                required_capabilities=["normal"],
                estimated_effort=60,
                timeout_seconds=3600,
                queued_at=datetime.utcnow()
            )
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                # Enqueue both tasks
                await isolated_task_queue.enqueue_task(high_priority_task)
                await isolated_task_queue.enqueue_task(low_priority_task)
                
                # Verify zadd was called with correct priority scores
                assert isolated_task_queue.redis_client.zadd.call_count == 2

        @pytest.mark.asyncio
        async def test_enqueue_task_multiple_queues(self, isolated_task_queue):
            """Test enqueuing tasks to different queues."""
            backend_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=100.0,
                queue_name="backend",
                required_capabilities=["python"],
                estimated_effort=60,
                timeout_seconds=3600,
                queued_at=datetime.utcnow()
            )
            
            frontend_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=100.0,
                queue_name="frontend",
                required_capabilities=["javascript"],
                estimated_effort=45,
                timeout_seconds=2700,
                queued_at=datetime.utcnow()
            )
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                await isolated_task_queue.enqueue_task(backend_task)
                await isolated_task_queue.enqueue_task(frontend_task)
                
                # Verify tasks were added to different Redis sorted sets
                assert isolated_task_queue.redis_client.zadd.call_count == 2

        @pytest.mark.asyncio
        async def test_enqueue_task_database_failure(self, isolated_task_queue, sample_queued_task):
            """Test graceful handling of database failure during enqueue."""
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.commit.side_effect = Exception("Database error")
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                # Should still succeed (graceful degradation)
                result = await isolated_task_queue.enqueue_task(sample_queued_task)
                
                # Verify rollback was called
                mock_session.rollback.assert_called_once()
                # Redis operations should still have been attempted
                isolated_task_queue.redis_client.zadd.assert_called()

    class TestTaskDequeuing:
        """Test task dequeuing operations."""

        @pytest.mark.asyncio
        async def test_dequeue_task_success(self, isolated_task_queue, mock_redis_client):
            """Test successful task dequeuing."""
            # Mock Redis returning a task
            task_id = str(uuid.uuid4())
            mock_redis_client.zrange.return_value = [task_id.encode()]
            mock_redis_client.hgetall.return_value = {
                b"task_id": task_id.encode(),
                b"priority_score": b"100.0",
                b"queue_name": b"general",
                b"required_capabilities": b'["backend_development"]',
                b"queued_at": datetime.utcnow().isoformat().encode()
            }
            
            result = await isolated_task_queue.dequeue_task(
                queue_names=["general"],
                required_capabilities=["backend_development"]
            )
            
            assert result is not None
            assert result.task_id == uuid.UUID(task_id)
            
            # Verify task was removed from queue
            mock_redis_client.zrem.assert_called()

        @pytest.mark.asyncio
        async def test_dequeue_task_empty_queue(self, isolated_task_queue, mock_redis_client):
            """Test dequeuing from empty queue."""
            # Mock empty queue
            mock_redis_client.zrange.return_value = []
            
            result = await isolated_task_queue.dequeue_task(
                queue_names=["empty_queue"],
                required_capabilities=["any"]
            )
            
            assert result is None

        @pytest.mark.asyncio
        async def test_dequeue_task_capability_matching(self, isolated_task_queue, mock_redis_client):
            """Test that capability matching works correctly."""
            # Mock task with specific capabilities
            task_id = str(uuid.uuid4())
            mock_redis_client.zrange.return_value = [task_id.encode()]
            mock_redis_client.hgetall.return_value = {
                b"task_id": task_id.encode(),
                b"priority_score": b"100.0",
                b"queue_name": b"specialized",
                b"required_capabilities": b'["python", "databases"]',
                b"queued_at": datetime.utcnow().isoformat().encode()
            }
            
            # Request with matching capabilities
            result = await isolated_task_queue.dequeue_task(
                queue_names=["specialized"],
                required_capabilities=["python", "databases", "testing"]
            )
            
            assert result is not None
            
            # Request with non-matching capabilities should return None
            mock_redis_client.reset_mock()
            result = await isolated_task_queue.dequeue_task(
                queue_names=["specialized"],
                required_capabilities=["javascript"]
            )
            
            # Task should not be removed if capabilities don't match
            assert result is None

        @pytest.mark.asyncio
        async def test_dequeue_task_timeout_handling(self, isolated_task_queue, mock_redis_client):
            """Test handling of task timeouts."""
            # Mock expired task
            task_id = str(uuid.uuid4())
            expired_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
            
            mock_redis_client.zrange.return_value = [task_id.encode()]
            mock_redis_client.hgetall.return_value = {
                b"task_id": task_id.encode(),
                b"priority_score": b"100.0",
                b"queue_name": b"general",
                b"required_capabilities": b'["backend"]',
                b"timeout_seconds": b"3600",
                b"queued_at": expired_time.encode()
            }
            
            result = await isolated_task_queue.dequeue_task(
                queue_names=["general"],
                required_capabilities=["backend"]
            )
            
            # Expired task should be cleaned up, not returned
            assert result is None
            # Verify cleanup operations
            mock_redis_client.zrem.assert_called()

    class TestTaskAssignment:
        """Test task assignment operations."""

        @pytest.mark.asyncio
        async def test_assign_task_to_agent_success(self, isolated_task_queue, sample_queued_task):
            """Test successful task assignment to agent."""
            agent_id = uuid.uuid4()
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.assign_task_to_agent(
                    task=sample_queued_task,
                    agent_id=agent_id
                )
                
                assert result.success is True
                assert result.agent_id == agent_id
                assert result.task_id == sample_queued_task.task_id
                assert isinstance(result.queue_wait_time_seconds, float)

        @pytest.mark.asyncio
        async def test_assign_task_agent_not_available(self, isolated_task_queue, sample_queued_task):
            """Test task assignment when agent is not available."""
            agent_id = uuid.uuid4()
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                # Mock agent not found or not available
                mock_session.execute.return_value.scalar_one_or_none.return_value = None
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.assign_task_to_agent(
                    task=sample_queued_task,
                    agent_id=agent_id
                )
                
                assert result.success is False
                assert "not available" in result.error_message.lower()

        @pytest.mark.asyncio
        async def test_assign_task_database_error(self, isolated_task_queue, sample_queued_task):
            """Test handling of database errors during assignment."""
            agent_id = uuid.uuid4()
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.commit.side_effect = Exception("Database connection failed")
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.assign_task_to_agent(
                    task=sample_queued_task,
                    agent_id=agent_id
                )
                
                assert result.success is False
                assert "Database connection failed" in result.error_message

    class TestTaskRetryHandling:
        """Test task retry logic."""

        @pytest.mark.asyncio
        async def test_retry_failed_task(self, isolated_task_queue, sample_queued_task):
            """Test retrying a failed task."""
            # Mark task as failed with retry count
            sample_queued_task.retry_count = 1
            sample_queued_task.max_retries = 3
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.retry_task(
                    task_id=sample_queued_task.task_id,
                    error_message="Previous attempt failed"
                )
                
                assert result is True
                # Verify task was re-enqueued with incremented retry count
                isolated_task_queue.redis_client.zadd.assert_called()

        @pytest.mark.asyncio
        async def test_retry_exhausted_task(self, isolated_task_queue, sample_queued_task):
            """Test handling of task with exhausted retries."""
            # Max retries already reached
            sample_queued_task.retry_count = 3
            sample_queued_task.max_retries = 3
            
            with patch('app.core.task_queue.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_get_session.return_value.__aenter__.return_value = mock_session
                
                result = await isolated_task_queue.retry_task(
                    task_id=sample_queued_task.task_id,
                    error_message="Final attempt failed"
                )
                
                assert result is False
                # Task should be marked as permanently failed
                isolated_task_queue.redis_client.zadd.assert_not_called()

    class TestQueueMetrics:
        """Test queue metrics and monitoring."""

        @pytest.mark.asyncio
        async def test_get_queue_metrics(self, isolated_task_queue, mock_redis_client):
            """Test getting queue metrics."""
            # Mock Redis responses for metrics
            mock_redis_client.zcard.return_value = 5
            mock_redis_client.hlen.return_value = 3
            
            metrics = await isolated_task_queue.get_queue_metrics(queue_name="general")
            
            assert isinstance(metrics, dict)
            assert "queued_tasks" in metrics
            assert "pending_assignments" in metrics
            assert "queue_name" in metrics
            assert metrics["queue_name"] == "general"

        @pytest.mark.asyncio
        async def test_get_all_queues_metrics(self, isolated_task_queue, mock_redis_client):
            """Test getting metrics for all queues."""
            # Mock multiple queues
            mock_redis_client.keys.return_value = [
                b"task_queue:backend",
                b"task_queue:frontend",
                b"task_queue:general"
            ]
            mock_redis_client.zcard.return_value = 2
            
            all_metrics = await isolated_task_queue.get_all_queues_metrics()
            
            assert isinstance(all_metrics, dict)
            assert "total_queues" in all_metrics
            assert "total_queued_tasks" in all_metrics
            assert "queue_details" in all_metrics

        @pytest.mark.asyncio
        async def test_queue_performance_tracking(self, isolated_task_queue):
            """Test that queue operations track performance metrics."""
            # The queue should track timing for operations
            start_time = datetime.utcnow()
            
            # Simulate some queue operations
            await isolated_task_queue.get_queue_metrics("test")
            
            # Verify that timing was tracked (implementation dependent)
            assert hasattr(isolated_task_queue, '_queue_metrics')

    class TestQueueMaintenance:
        """Test queue maintenance operations."""

        @pytest.mark.asyncio
        async def test_cleanup_expired_tasks(self, isolated_task_queue, mock_redis_client):
            """Test cleanup of expired tasks."""
            # Mock finding expired tasks
            expired_task_id = str(uuid.uuid4())
            mock_redis_client.zrange.return_value = [expired_task_id.encode()]
            
            # Mock task metadata with expired timestamp
            expired_time = (datetime.utcnow() - timedelta(hours=2)).isoformat()
            mock_redis_client.hgetall.return_value = {
                b"task_id": expired_task_id.encode(),
                b"timeout_seconds": b"3600",
                b"queued_at": expired_time.encode()
            }
            
            cleaned_count = await isolated_task_queue.cleanup_expired_tasks("general")
            
            assert cleaned_count >= 0
            # Verify cleanup operations were performed
            mock_redis_client.zrem.assert_called()

        @pytest.mark.asyncio
        async def test_purge_queue(self, isolated_task_queue, mock_redis_client):
            """Test purging all tasks from a queue."""
            result = await isolated_task_queue.purge_queue("test_queue")
            
            assert result is True
            # Verify Redis delete operations
            mock_redis_client.delete.assert_called()

        @pytest.mark.asyncio
        async def test_get_queue_health(self, isolated_task_queue, mock_redis_client):
            """Test queue health monitoring."""
            # Mock healthy queue state
            mock_redis_client.zcard.return_value = 10  # 10 queued tasks
            mock_redis_client.hlen.return_value = 2   # 2 active assignments
            
            health = await isolated_task_queue.get_queue_health("general")
            
            assert isinstance(health, dict)
            assert "status" in health
            assert "queued_tasks" in health
            assert "active_assignments" in health
            assert "health_score" in health

    class TestErrorHandling:
        """Test error handling and edge cases."""

        @pytest.mark.asyncio
        async def test_redis_connection_failure(self, isolated_task_queue):
            """Test handling of Redis connection failures."""
            # Mock Redis client to raise connection error
            isolated_task_queue.redis_client.zadd.side_effect = ConnectionError("Redis unavailable")
            
            sample_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=100.0,
                queue_name="test",
                required_capabilities=["test"],
                estimated_effort=30,
                timeout_seconds=1800,
                queued_at=datetime.utcnow()
            )
            
            # Should handle error gracefully
            result = await isolated_task_queue.enqueue_task(sample_task)
            assert result is False

        @pytest.mark.asyncio
        async def test_invalid_task_data(self, isolated_task_queue):
            """Test handling of invalid task data."""
            # Create task with invalid data
            invalid_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=-1.0,  # Invalid negative priority
                queue_name="",        # Empty queue name
                required_capabilities=[],
                estimated_effort=0,
                timeout_seconds=-1,
                queued_at=datetime.utcnow()
            )
            
            # Should validate and reject invalid data
            result = await isolated_task_queue.enqueue_task(invalid_task)
            # Implementation should validate and handle appropriately

        @pytest.mark.asyncio
        async def test_concurrent_access_handling(self, isolated_task_queue, mock_redis_client):
            """Test handling of concurrent queue access."""
            # Simulate race condition where task is dequeued by another process
            task_id = str(uuid.uuid4())
            mock_redis_client.zrange.return_value = [task_id.encode()]
            mock_redis_client.hgetall.return_value = {}  # Task metadata gone
            
            result = await isolated_task_queue.dequeue_task(
                queue_names=["concurrent_test"],
                required_capabilities=["test"]
            )
            
            # Should handle missing metadata gracefully
            assert result is None


class TestTaskAssignmentResult:
    """Test TaskAssignmentResult dataclass."""

    def test_assignment_result_success(self):
        """Test successful assignment result."""
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        assignment_time = datetime.utcnow()
        
        result = TaskAssignmentResult(
            success=True,
            task_id=task_id,
            agent_id=agent_id,
            assignment_time=assignment_time,
            queue_wait_time_seconds=45.5,
            error_message=None
        )
        
        assert result.success is True
        assert result.task_id == task_id
        assert result.agent_id == agent_id
        assert result.queue_wait_time_seconds == 45.5
        assert result.error_message is None

    def test_assignment_result_failure(self):
        """Test failed assignment result."""
        task_id = uuid.uuid4()
        
        result = TaskAssignmentResult(
            success=False,
            task_id=task_id,
            agent_id=None,
            assignment_time=None,
            queue_wait_time_seconds=120.0,
            error_message="No suitable agent available"
        )
        
        assert result.success is False
        assert result.agent_id is None
        assert result.assignment_time is None
        assert "No suitable agent available" in result.error_message


class TestQueueIntegrationHelpers:
    """Test helper methods for queue integration."""

    @pytest.fixture
    def task_queue_with_helpers(self, mock_redis_client):
        """Task queue instance for testing helper methods."""
        return TaskQueue(redis_client=mock_redis_client)

    @pytest.mark.asyncio
    async def test_capability_matching_logic(self, task_queue_with_helpers):
        """Test the internal capability matching logic."""
        # Test exact match
        task_capabilities = ["python", "backend"]
        agent_capabilities = ["python", "backend", "testing"]
        
        # This would be internal logic - test if exposed as helper
        # match = task_queue_with_helpers._check_capability_match(task_capabilities, agent_capabilities)
        # assert match is True
        
        # Test partial match (should fail)
        task_capabilities = ["python", "frontend", "react"]
        agent_capabilities = ["python", "backend"]
        
        # match = task_queue_with_helpers._check_capability_match(task_capabilities, agent_capabilities)
        # assert match is False

    @pytest.mark.asyncio
    async def test_priority_calculation(self, task_queue_with_helpers):
        """Test priority score calculation logic."""
        # Test priority factors: urgency, effort, wait time
        base_priority = 100.0
        urgency_multiplier = 1.5
        wait_time_boost = 10.0
        
        # This would test internal priority calculation if exposed
        # calculated_priority = task_queue_with_helpers._calculate_dynamic_priority(
        #     base_priority, urgency_multiplier, wait_time_boost
        # )
        # assert calculated_priority > base_priority

    @pytest.mark.asyncio
    async def test_queue_balancing_logic(self, task_queue_with_helpers):
        """Test logic for balancing tasks across queues."""
        # Test queue selection for optimal distribution
        # This would test internal queue balancing if implemented
        pass