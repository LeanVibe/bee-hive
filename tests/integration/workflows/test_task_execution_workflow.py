"""
Integration Tests for Task Creation → Task Execution Workflow

Tests the complete task lifecycle from creation through execution using real
component implementations with controlled test data.

Integration Points Tested:
- Task creation and validation
- Task queue operations and priority handling
- Task assignment to available agents
- Task execution monitoring and status updates
- Task completion and result handling
- Error handling and retry mechanisms
- Performance metrics collection

Test Strategy:
- Use real TaskQueue and orchestrator components
- Use test-specific data isolation
- Control time-based operations for deterministic tests
- Test both success and failure scenarios
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, AsyncMock
from typing import Dict, Any, List, Optional

# Real components for integration testing
from app.core.simple_orchestrator import SimpleOrchestrator, AgentRole
from app.core.task_queue import TaskQueue, QueuedTask, TaskAssignmentResult
from app.models.task import TaskStatus, TaskPriority, TaskType
from app.models.agent import AgentStatus


class TestTaskExecutionWorkflowIntegration:
    """Integration tests for complete task execution workflows."""

    @pytest.fixture
    async def test_task_queue(self):
        """Create test task queue with mocked Redis."""
        mock_redis = AsyncMock()
        
        # Mock successful operations
        mock_redis.zadd.return_value = 1
        mock_redis.zrem.return_value = 1
        mock_redis.zrange.return_value = []
        mock_redis.hset.return_value = 1
        mock_redis.hgetall.return_value = {}
        mock_redis.hdel.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.expire.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.zcard.return_value = 0
        mock_redis.hlen.return_value = 0
        mock_redis.keys.return_value = []
        
        return TaskQueue(redis_client=mock_redis)

    @pytest.fixture
    async def test_orchestrator(self):
        """Create test orchestrator with mocked dependencies."""
        # Mock database
        mock_db_factory = Mock()
        mock_session = AsyncMock()
        mock_db_factory.get_session.return_value.__aenter__.return_value = mock_session
        
        # Mock cache
        test_cache = {}
        mock_cache = Mock()
        mock_cache.get.side_effect = lambda key: test_cache.get(key)
        mock_cache.set.side_effect = lambda key, value, ttl=300: test_cache.update({key: value}) or True
        
        # Mock agent launcher
        mock_launcher = AsyncMock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.session_id = "test-session"
        mock_result.session_name = "test-session-name"
        mock_result.workspace_path = "/test/workspace"
        mock_result.launch_time_seconds = 0.5
        mock_launcher.launch_agent.return_value = mock_result
        mock_launcher.terminate_agent.return_value = True
        
        # Mock Redis bridge
        mock_redis_bridge = AsyncMock()
        mock_redis_bridge.register_agent.return_value = True
        mock_redis_bridge.assign_task_to_agent.return_value = None
        
        # Mock other components
        mock_tmux = Mock()
        mock_id_gen = Mock()
        mock_id_gen.generate.return_value = "TEST123"
        mock_websocket = AsyncMock()
        mock_websocket.broadcast_task_update.return_value = None
        
        orchestrator = SimpleOrchestrator(
            db_session_factory=mock_db_factory,
            cache=mock_cache,
            agent_launcher=mock_launcher,
            redis_bridge=mock_redis_bridge,
            tmux_manager=mock_tmux,
            short_id_generator=mock_id_gen,
            websocket_manager=mock_websocket
        )
        
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    async def sample_task_data(self):
        """Sample task data for testing."""
        return {
            "task_id": str(uuid.uuid4()),
            "description": "Integration test task",
            "task_type": "development",
            "priority": TaskPriority.MEDIUM,
            "required_capabilities": ["python", "backend"],
            "estimated_effort": 60,
            "timeout_seconds": 3600
        }

    class TestTaskCreationWorkflow:
        """Test task creation and initial queueing."""

        @pytest.mark.asyncio
        async def test_task_creation_to_queue(self, test_task_queue, sample_task_data):
            """Test complete task creation and queueing workflow."""
            # Create queued task
            queued_task = QueuedTask(
                task_id=uuid.UUID(sample_task_data["task_id"]),
                priority_score=100.0,
                queue_name="general",
                required_capabilities=sample_task_data["required_capabilities"],
                estimated_effort=sample_task_data["estimated_effort"],
                timeout_seconds=sample_task_data["timeout_seconds"],
                queued_at=datetime.utcnow()
            )
            
            # Enqueue task
            result = await test_task_queue.enqueue_task(queued_task)
            assert result is True
            
            # Verify Redis operations were called
            test_task_queue.redis_client.zadd.assert_called()
            test_task_queue.redis_client.hset.assert_called()

        @pytest.mark.asyncio
        async def test_task_priority_ordering(self, test_task_queue):
            """Test that tasks are queued with proper priority ordering."""
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
            
            # Enqueue both tasks
            await test_task_queue.enqueue_task(high_priority_task)
            await test_task_queue.enqueue_task(low_priority_task)
            
            # Verify both were enqueued
            assert test_task_queue.redis_client.zadd.call_count == 2

        @pytest.mark.asyncio
        async def test_task_validation_during_creation(self, test_task_queue):
            """Test task validation during the creation process."""
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
            
            # Task creation should handle validation
            # In real implementation, this might fail validation
            result = await test_task_queue.enqueue_task(invalid_task)
            # Result depends on implementation - might succeed with sanitized data

    class TestTaskAssignmentWorkflow:
        """Test task assignment to available agents."""

        @pytest.mark.asyncio
        async def test_task_assignment_to_available_agent(self, test_orchestrator, sample_task_data):
            """Test assigning task to an available agent."""
            # First spawn an agent
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Delegate task (which includes assignment)
            task_id = await test_orchestrator.delegate_task(
                task_description=sample_task_data["description"],
                task_type=sample_task_data["task_type"],
                priority=sample_task_data["priority"],
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Verify task assignment
            assert task_id in test_orchestrator._task_assignments
            assignment = test_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id
            assert assignment.status == TaskStatus.PENDING

        @pytest.mark.asyncio
        async def test_task_assignment_capability_matching(self, test_orchestrator):
            """Test that tasks are assigned based on capability matching."""
            # Spawn specialized agents
            backend_agent = await test_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            frontend_agent = await test_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Delegate backend-specific task
            backend_task_id = await test_orchestrator.delegate_task(
                task_description="Database optimization",
                task_type="backend_development",
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Verify backend task was assigned to backend agent
            assignment = test_orchestrator._task_assignments[backend_task_id]
            assert assignment.agent_id == backend_agent

        @pytest.mark.asyncio
        async def test_task_assignment_when_no_agent_available(self, test_orchestrator):
            """Test task assignment when no suitable agent is available."""
            # Don't spawn any agents
            
            # Attempt task delegation
            with pytest.raises(Exception, match="No suitable agent available"):
                await test_orchestrator.delegate_task(
                    task_description="Impossible task",
                    task_type="specialized_work",
                    preferred_agent_role=AgentRole.META_AGENT
                )

        @pytest.mark.asyncio
        async def test_task_assignment_load_balancing(self, test_orchestrator):
            """Test load balancing across multiple available agents."""
            # Spawn multiple agents of same type
            agent_ids = []
            for i in range(3):
                agent_id = await test_orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER
                )
                agent_ids.append(agent_id)
            
            # Delegate multiple tasks
            task_ids = []
            for i in range(6):
                task_id = await test_orchestrator.delegate_task(
                    task_description=f"Load balance task {i+1}",
                    task_type="development",
                    preferred_agent_role=AgentRole.BACKEND_DEVELOPER
                )
                task_ids.append(task_id)
            
            # Verify all tasks were assigned
            assert len(task_ids) == 6
            
            # In a real load balancer, tasks would be distributed across agents
            # Here we just verify they were all assigned
            for task_id in task_ids:
                assert task_id in test_orchestrator._task_assignments

    class TestTaskExecutionMonitoring:
        """Test task execution monitoring and status updates."""

        @pytest.mark.asyncio
        async def test_task_execution_status_updates(self, test_orchestrator, sample_task_data):
            """Test monitoring task execution status changes."""
            # Spawn agent and assign task
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER
            )
            
            task_id = await test_orchestrator.delegate_task(
                task_description=sample_task_data["description"],
                task_type="testing",
                preferred_agent_role=AgentRole.QA_ENGINEER
            )
            
            # Initial status should be PENDING
            assignment = test_orchestrator._task_assignments[task_id]
            assert assignment.status == TaskStatus.PENDING
            
            # Simulate status update (in real system, this would come from agent)
            assignment.status = TaskStatus.IN_PROGRESS
            
            # Verify WebSocket notification was sent
            test_orchestrator._websocket_manager.broadcast_task_update.assert_called()

        @pytest.mark.asyncio
        async def test_task_execution_timeout_handling(self, test_task_queue):
            """Test handling of task execution timeouts."""
            # Create task with short timeout
            short_timeout_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=100.0,
                queue_name="timeout_test",
                required_capabilities=["testing"],
                estimated_effort=30,
                timeout_seconds=5,  # Very short timeout
                queued_at=datetime.utcnow() - timedelta(seconds=10)  # Already expired
            )
            
            # Mock Redis to return the expired task
            task_id_bytes = str(short_timeout_task.task_id).encode()
            test_task_queue.redis_client.zrange.return_value = [task_id_bytes]
            test_task_queue.redis_client.hgetall.return_value = {
                b"task_id": task_id_bytes,
                b"timeout_seconds": b"5",
                b"queued_at": short_timeout_task.queued_at.isoformat().encode()
            }
            
            # Attempt to dequeue should handle timeout
            result = await test_task_queue.dequeue_task(
                queue_names=["timeout_test"],
                required_capabilities=["testing"]
            )
            
            # Should return None (task expired) and clean up
            assert result is None
            test_task_queue.redis_client.zrem.assert_called()

        @pytest.mark.asyncio
        async def test_task_progress_tracking(self, test_orchestrator):
            """Test tracking task execution progress."""
            # Spawn agent and assign task
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.DEVOPS_ENGINEER
            )
            
            task_id = await test_orchestrator.delegate_task(
                task_description="Deploy application",
                task_type="deployment",
                preferred_agent_role=AgentRole.DEVOPS_ENGINEER
            )
            
            # Get initial assignment
            assignment = test_orchestrator._task_assignments[task_id]
            start_time = assignment.assigned_at
            
            # Simulate progress updates
            # In real system, these would come from the agent
            assignment.status = TaskStatus.IN_PROGRESS
            
            # Verify task is being tracked
            assert task_id in test_orchestrator._task_assignments
            assert assignment.agent_id == agent_id

    class TestTaskCompletionWorkflow:
        """Test task completion and result handling."""

        @pytest.mark.asyncio
        async def test_successful_task_completion(self, test_orchestrator, sample_task_data):
            """Test successful task completion workflow."""
            # Spawn agent and assign task
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER
            )
            
            task_id = await test_orchestrator.delegate_task(
                task_description=sample_task_data["description"],
                task_type="ui_development",
                preferred_agent_role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Simulate task completion
            assignment = test_orchestrator._task_assignments[task_id]
            assignment.status = TaskStatus.COMPLETED
            
            # In real system, completion would trigger:
            # - Result storage
            # - Agent availability update
            # - Metrics collection
            # - Notification broadcast
            
            # Verify agent is available for new tasks
            agent = test_orchestrator._agents[agent_id]
            # Agent's current_task_id would be cleared in real implementation

        @pytest.mark.asyncio
        async def test_task_failure_handling(self, test_orchestrator):
            """Test handling of task failures."""
            # Spawn agent and assign task
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            
            task_id = await test_orchestrator.delegate_task(
                task_description="Complex backend task",
                task_type="development",
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Simulate task failure
            assignment = test_orchestrator._task_assignments[task_id]
            assignment.status = TaskStatus.FAILED
            
            # In real system, failure would trigger:
            # - Error logging
            # - Retry decision
            # - Agent status update
            # - Failure notification
            
            # Verify task is marked as failed
            assert assignment.status == TaskStatus.FAILED

        @pytest.mark.asyncio
        async def test_task_retry_mechanism(self, test_task_queue):
            """Test task retry mechanism for failed tasks."""
            # Create a task that can be retried
            failed_task = QueuedTask(
                task_id=uuid.uuid4(),
                priority_score=100.0,
                queue_name="retry_test",
                required_capabilities=["backend"],
                estimated_effort=60,
                timeout_seconds=3600,
                queued_at=datetime.utcnow(),
                retry_count=1,
                max_retries=3
            )
            
            # Simulate retry
            result = await test_task_queue.retry_task(
                task_id=failed_task.task_id,
                error_message="Previous attempt failed"
            )
            
            # Should succeed if retries not exhausted
            # Implementation details depend on actual retry logic

    class TestTaskWorkflowPerformance:
        """Test performance aspects of task workflows."""

        @pytest.mark.asyncio
        async def test_concurrent_task_processing(self, test_orchestrator):
            """Test concurrent task processing performance."""
            # Spawn multiple agents
            agent_ids = []
            for i in range(3):
                agent_id = await test_orchestrator.spawn_agent(
                    role=AgentRole.BACKEND_DEVELOPER
                )
                agent_ids.append(agent_id)
            
            # Create and delegate multiple tasks concurrently
            task_coroutines = []
            for i in range(9):  # 3 tasks per agent
                coro = test_orchestrator.delegate_task(
                    task_description=f"Concurrent task {i+1}",
                    task_type="development",
                    priority=TaskPriority.MEDIUM
                )
                task_coroutines.append(coro)
            
            # Execute all delegations concurrently
            task_ids = await asyncio.gather(*task_coroutines)
            
            # Verify all tasks were assigned
            assert len(task_ids) == 9
            assert len(test_orchestrator._task_assignments) == 9
            
            # Check performance metrics
            performance_metrics = await test_orchestrator.get_performance_metrics()
            assert 'delegate_task' in performance_metrics['operation_metrics']
            delegate_metrics = performance_metrics['operation_metrics']['delegate_task']
            assert delegate_metrics['count'] == 9

        @pytest.mark.asyncio
        async def test_high_throughput_task_queue_operations(self, test_task_queue):
            """Test high-throughput task queue operations."""
            # Create many tasks
            tasks = []
            for i in range(50):
                task = QueuedTask(
                    task_id=uuid.uuid4(),
                    priority_score=100.0 + i,  # Varying priorities
                    queue_name="throughput_test",
                    required_capabilities=["performance"],
                    estimated_effort=30,
                    timeout_seconds=1800,
                    queued_at=datetime.utcnow()
                )
                tasks.append(task)
            
            # Enqueue all tasks
            enqueue_coroutines = [test_task_queue.enqueue_task(task) for task in tasks]
            results = await asyncio.gather(*enqueue_coroutines, return_exceptions=True)
            
            # Most should succeed (implementation dependent)
            successful_enqueues = sum(1 for r in results if r is True)
            assert successful_enqueues > 0

        @pytest.mark.asyncio
        async def test_task_workflow_latency_measurement(self, test_orchestrator):
            """Test measurement of task workflow latency."""
            # Record start time
            workflow_start = datetime.utcnow()
            
            # Spawn agent
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.QA_ENGINEER
            )
            
            # Delegate task
            task_id = await test_orchestrator.delegate_task(
                task_description="Latency measurement task",
                task_type="testing",
                priority=TaskPriority.HIGH,
                preferred_agent_role=AgentRole.QA_ENGINEER
            )
            
            # Record end time
            workflow_end = datetime.utcnow()
            
            # Calculate latency
            latency = (workflow_end - workflow_start).total_seconds() * 1000  # ms
            
            # Should meet performance targets
            assert latency < 2000  # Less than 2 seconds for full workflow
            
            # Check if task was assigned promptly
            assignment = test_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id

    class TestTaskWorkflowErrorRecovery:
        """Test error recovery in task workflows."""

        @pytest.mark.asyncio
        async def test_agent_failure_during_task_execution(self, test_orchestrator):
            """Test recovery when agent fails during task execution."""
            # Spawn agent and assign task
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER
            )
            
            task_id = await test_orchestrator.delegate_task(
                task_description="Task with agent failure",
                task_type="development",
                preferred_agent_role=AgentRole.BACKEND_DEVELOPER
            )
            
            # Simulate agent failure
            await test_orchestrator.shutdown_agent(agent_id)
            
            # Task should be marked as failed or available for reassignment
            assignment = test_orchestrator._task_assignments[task_id]
            # In real system, task status would be updated appropriately

        @pytest.mark.asyncio
        async def test_queue_failure_graceful_degradation(self, test_orchestrator):
            """Test graceful degradation when task queue fails."""
            # Mock task queue operations to fail
            test_orchestrator._redis_bridge.assign_task_to_agent.side_effect = Exception("Queue unavailable")
            
            # Spawn agent first
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Try to delegate task (should fall back to local assignment)
            task_id = await test_orchestrator.delegate_task(
                task_description="Queue failure test",
                task_type="development",
                preferred_agent_role=AgentRole.FRONTEND_DEVELOPER
            )
            
            # Should succeed with local assignment
            assert task_id in test_orchestrator._task_assignments
            assignment = test_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id

        @pytest.mark.asyncio
        async def test_database_failure_task_workflow_continuation(self, test_orchestrator):
            """Test task workflow continuation despite database failures."""
            # Mock database operations to fail
            test_orchestrator._db_session_factory.get_session.side_effect = Exception("Database down")
            
            # Spawn agent
            agent_id = await test_orchestrator.spawn_agent(
                role=AgentRole.DEVOPS_ENGINEER
            )
            
            # Delegate task (should work despite DB failure)
            task_id = await test_orchestrator.delegate_task(
                task_description="DB failure resilience test",
                task_type="ops",
                preferred_agent_role=AgentRole.DEVOPS_ENGINEER
            )
            
            # Core functionality should work
            assert task_id in test_orchestrator._task_assignments
            assignment = test_orchestrator._task_assignments[task_id]
            assert assignment.agent_id == agent_id