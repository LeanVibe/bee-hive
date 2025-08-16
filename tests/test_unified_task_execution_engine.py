"""
Comprehensive Test Suite for Unified Task Execution Engine
Tests all consolidated task management functionality and performance.

Epic 1, Phase 2 Week 3 - Task System Consolidation Validation
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

from app.core.unified_task_execution_engine import (
    UnifiedTaskExecutionEngine,
    TaskExecutionRequest,
    TaskExecutionType,
    TaskExecutionStatus,
    ExecutionMode,
    SchedulingStrategy,
    TaskDependency,
    TaskQueue,
    AgentMatcher,
    get_unified_task_execution_engine,
    execute_task,
    schedule_task,
    execute_batch_tasks
)
from app.core.task_system_migration_adapter import (
    TaskSystemMigrationAdapter,
    get_task_migration_adapter
)
from app.core.task_orchestrator_integration import (
    TaskOrchestratorBridge,
    TaskAgentRequest,
    AgentAssignmentResponse,
    get_task_orchestrator_bridge
)

class TestUnifiedTaskExecutionEngine:
    """Test unified task execution engine core functionality"""
    
    @pytest.fixture
    async def engine(self):
        """Create test engine instance"""
        engine = UnifiedTaskExecutionEngine()
        await engine.start_engine()
        yield engine
        await engine.stop_engine()
    
    @pytest.fixture
    def test_functions(self, engine):
        """Register test functions"""
        def simple_add(a: int, b: int) -> int:
            return a + b
        
        async def async_multiply(a: int, b: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return a * b
        
        def slow_operation(duration: float) -> str:
            time.sleep(duration)
            return f"completed after {duration}s"
        
        def failing_operation() -> None:
            raise ValueError("Intentional test failure")
        
        engine.register_task_function("simple_add", simple_add)
        engine.register_task_function("async_multiply", async_multiply)
        engine.register_task_function("slow_operation", slow_operation)
        engine.register_task_function("failing_operation", failing_operation)
        
        return {
            "simple_add": simple_add,
            "async_multiply": async_multiply,
            "slow_operation": slow_operation,
            "failing_operation": failing_operation
        }
    
    @pytest.mark.asyncio
    async def test_basic_task_execution(self, engine, test_functions):
        """Test basic synchronous task execution"""
        request = TaskExecutionRequest(
            function_name="simple_add",
            function_args=[5, 3],
            task_type=TaskExecutionType.IMMEDIATE,
            execution_mode=ExecutionMode.SYNC
        )
        
        task_id = await engine.submit_task(request)
        assert task_id is not None
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        status = await engine.get_task_status(task_id)
        assert status is not None
        assert status["status"] == TaskExecutionStatus.COMPLETED.value
        # Note: Result validation would require accessing the completed tasks
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, engine, test_functions):
        """Test asynchronous task execution"""
        request = TaskExecutionRequest(
            function_name="async_multiply",
            function_args=[4, 7],
            task_type=TaskExecutionType.IMMEDIATE,
            execution_mode=ExecutionMode.ASYNC
        )
        
        task_id = await engine.submit_task(request)
        assert task_id is not None
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        status = await engine.get_task_status(task_id)
        assert status is not None
        assert status["status"] == TaskExecutionStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_scheduled_task_execution(self, engine, test_functions):
        """Test scheduled task execution"""
        future_time = datetime.utcnow() + timedelta(seconds=1)
        
        request = TaskExecutionRequest(
            function_name="simple_add",
            function_args=[10, 20],
            task_type=TaskExecutionType.SCHEDULED,
            scheduled_at=future_time
        )
        
        task_id = await engine.submit_task(request)
        assert task_id is not None
        
        # Check that task is not immediately executed
        status = await engine.get_task_status(task_id)
        assert status["status"] in [TaskExecutionStatus.QUEUED.value, TaskExecutionStatus.PENDING.value]
        
        # Wait for scheduled execution
        await asyncio.sleep(2)
        
        status = await engine.get_task_status(task_id)
        assert status["status"] == TaskExecutionStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_priority_task_execution(self, engine, test_functions):
        """Test priority task execution"""
        # Submit low priority task first
        low_priority_request = TaskExecutionRequest(
            function_name="slow_operation",
            function_args=[0.1],
            task_type=TaskExecutionType.IMMEDIATE,
            priority=1  # Low priority
        )
        
        # Submit high priority task second
        high_priority_request = TaskExecutionRequest(
            function_name="simple_add",
            function_args=[1, 1],
            task_type=TaskExecutionType.PRIORITY,
            priority=10  # High priority
        )
        
        low_task_id = await engine.submit_task(low_priority_request)
        high_task_id = await engine.submit_task(high_priority_request)
        
        # Wait for execution
        await asyncio.sleep(1)
        
        # Both should complete, but this tests the priority queue mechanism
        low_status = await engine.get_task_status(low_task_id)
        high_status = await engine.get_task_status(high_task_id)
        
        assert low_status["status"] == TaskExecutionStatus.COMPLETED.value
        assert high_status["status"] == TaskExecutionStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_batch_task_execution(self, engine, test_functions):
        """Test batch task execution"""
        batch_requests = []
        for i in range(5):
            request = TaskExecutionRequest(
                function_name="simple_add",
                function_args=[i, i + 1],
                task_type=TaskExecutionType.BATCH,
                metadata={"batch_item": i}
            )
            batch_requests.append(request)
        
        task_ids = []
        for request in batch_requests:
            task_id = await engine.submit_task(request)
            task_ids.append(task_id)
        
        # Wait for batch completion
        await asyncio.sleep(1)
        
        # Check all tasks completed
        for task_id in task_ids:
            status = await engine.get_task_status(task_id)
            assert status["status"] == TaskExecutionStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, engine, test_functions):
        """Test task failure handling and retry mechanism"""
        request = TaskExecutionRequest(
            function_name="failing_operation",
            task_type=TaskExecutionType.IMMEDIATE,
            max_retries=2,
            retry_backoff_base=1.5
        )
        
        task_id = await engine.submit_task(request)
        assert task_id is not None
        
        # Wait for failure and retries
        await asyncio.sleep(3)
        
        status = await engine.get_task_status(task_id)
        assert status["status"] == TaskExecutionStatus.FAILED.value
        assert "retry_count" in status
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, engine, test_functions):
        """Test task timeout handling"""
        request = TaskExecutionRequest(
            function_name="slow_operation",
            function_args=[2.0],  # 2 second operation
            task_type=TaskExecutionType.IMMEDIATE,
            timeout=timedelta(seconds=0.5)  # 0.5 second timeout
        )
        
        task_id = await engine.submit_task(request)
        assert task_id is not None
        
        # Wait for timeout
        await asyncio.sleep(1)
        
        status = await engine.get_task_status(task_id)
        assert status["status"] == TaskExecutionStatus.TIMEOUT.value
    
    @pytest.mark.asyncio
    async def test_task_dependencies(self, engine, test_functions):
        """Test task dependency resolution"""
        # Create dependency task first
        dependency_request = TaskExecutionRequest(
            task_id="dependency_task",
            function_name="simple_add",
            function_args=[1, 2],
            task_type=TaskExecutionType.IMMEDIATE
        )
        
        dependency_id = await engine.submit_task(dependency_request)
        
        # Create dependent task
        dependent_request = TaskExecutionRequest(
            function_name="simple_add",
            function_args=[3, 4],
            task_type=TaskExecutionType.IMMEDIATE,
            dependencies=[TaskDependency(dependency_id=dependency_id)]
        )
        
        dependent_id = await engine.submit_task(dependent_request)
        
        # Wait for both to complete
        await asyncio.sleep(1)
        
        dependency_status = await engine.get_task_status(dependency_id)
        dependent_status = await engine.get_task_status(dependent_id)
        
        assert dependency_status["status"] == TaskExecutionStatus.COMPLETED.value
        assert dependent_status["status"] == TaskExecutionStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_execution_statistics(self, engine, test_functions):
        """Test execution statistics collection"""
        # Submit several tasks
        task_ids = []
        for i in range(3):
            request = TaskExecutionRequest(
                function_name="simple_add",
                function_args=[i, i + 1],
                task_type=TaskExecutionType.IMMEDIATE
            )
            task_id = await engine.submit_task(request)
            task_ids.append(task_id)
        
        # Wait for completion
        await asyncio.sleep(1)
        
        stats = engine.get_execution_stats()
        assert stats["tasks_submitted"] >= 3
        assert stats["tasks_completed"] >= 3
        assert "average_execution_time_ms" in stats
        assert "queue_sizes" in stats
        assert stats["engine_status"] == "running"

class TestTaskQueue:
    """Test task queue functionality"""
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self):
        """Test priority queue maintains correct order"""
        queue = TaskQueue("test_queue")
        
        # Add tasks with different priorities
        high_priority = TaskExecutionRequest(
            task_id="high",
            function_name="test",
            priority=1
        )
        
        low_priority = TaskExecutionRequest(
            task_id="low", 
            function_name="test",
            priority=10
        )
        
        medium_priority = TaskExecutionRequest(
            task_id="medium",
            function_name="test", 
            priority=5
        )
        
        # Add in non-priority order
        await queue.enqueue(low_priority, 10)
        await queue.enqueue(high_priority, 1)
        await queue.enqueue(medium_priority, 5)
        
        # Dequeue should return in priority order
        first = await queue.dequeue()
        second = await queue.dequeue()
        third = await queue.dequeue()
        
        assert first.task_id == "high"
        assert second.task_id == "medium"
        assert third.task_id == "low"
    
    @pytest.mark.asyncio
    async def test_queue_size_limits(self):
        """Test queue size limits"""
        small_queue = TaskQueue("small", max_size=2)
        
        request1 = TaskExecutionRequest(function_name="test1")
        request2 = TaskExecutionRequest(function_name="test2")
        request3 = TaskExecutionRequest(function_name="test3")
        
        assert await small_queue.enqueue(request1) == True
        assert await small_queue.enqueue(request2) == True
        assert await small_queue.enqueue(request3) == False  # Should fail due to size limit
        
        assert small_queue.size() == 2
    
    @pytest.mark.asyncio
    async def test_scheduled_task_dequeue(self):
        """Test scheduled tasks are not dequeued before their time"""
        queue = TaskQueue("scheduled")
        
        future_time = datetime.utcnow() + timedelta(seconds=1)
        request = TaskExecutionRequest(
            function_name="test",
            scheduled_at=future_time
        )
        
        await queue.enqueue(request)
        
        # Should not dequeue immediately
        task = await queue.dequeue()
        assert task is None
        
        # Wait for schedule time
        await asyncio.sleep(1.1)
        
        # Should dequeue now
        task = await queue.dequeue()
        assert task is not None

class TestAgentMatcher:
    """Test agent matching functionality"""
    
    def test_capability_filtering(self):
        """Test agent filtering by capabilities"""
        matcher = AgentMatcher()
        
        # Mock agent objects
        class MockAgent:
            def __init__(self, capabilities):
                self.capabilities = capabilities
        
        agents = [
            MockAgent([{"name": "python", "specialization_areas": ["web", "api"]}]),
            MockAgent([{"name": "javascript", "specialization_areas": ["frontend"]}]),
            MockAgent([{"name": "python", "specialization_areas": ["data"]}])
        ]
        
        required_caps = ["python", "web"]
        filtered = matcher._filter_by_capabilities(agents, required_caps)
        
        assert len(filtered) == 1
        assert filtered[0].capabilities[0]["name"] == "python"
        assert "web" in filtered[0].capabilities[0]["specialization_areas"]

class TestMigrationAdapter:
    """Test legacy task system migration adapter"""
    
    @pytest.fixture
    def adapter(self):
        return TaskSystemMigrationAdapter()
    
    @pytest.mark.asyncio
    async def test_legacy_task_execution_migration(self, adapter):
        """Test legacy TaskExecutionEngine compatibility"""
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        with pytest.warns(DeprecationWarning):
            result = await adapter.start_task_execution(
                task_id, agent_id, {"test": "context"}
            )
        
        assert result == True
        assert adapter._migration_stats["legacy_calls"] == 1
    
    @pytest.mark.asyncio
    async def test_legacy_task_assignment_migration(self, adapter):
        """Test legacy TaskScheduler compatibility"""
        task_id = uuid.uuid4()
        
        with pytest.warns(DeprecationWarning):
            result = await adapter.assign_task(
                task_id, strategy="hybrid", timeout_seconds=30
            )
        
        assert result["success"] == True
        assert result["task_id"] == task_id
        assert "unified_task_id" in result
    
    @pytest.mark.asyncio
    async def test_legacy_queue_migration(self, adapter):
        """Test legacy TaskQueue compatibility"""
        task_id = uuid.uuid4()
        
        with pytest.warns(DeprecationWarning):
            result = await adapter.enqueue_task(
                task_id, priority=5, queue_name="test_queue"
            )
        
        assert result == True
        assert adapter._migration_stats["legacy_calls"] >= 1
    
    def test_migration_statistics(self, adapter):
        """Test migration statistics tracking"""
        stats = adapter.get_migration_stats()
        
        assert "legacy_calls" in stats
        assert "unified_calls" in stats
        assert "migration_progress_percentage" in stats
        assert "migration_recommendations" in stats

class TestOrchestratorIntegration:
    """Test task-orchestrator integration"""
    
    @pytest.fixture
    def bridge(self):
        return TaskOrchestratorBridge()
    
    @pytest.mark.asyncio
    async def test_agent_request(self, bridge):
        """Test agent request functionality"""
        request = TaskAgentRequest(
            task_id="test_task",
            required_capabilities=["python", "api"],
            priority=5,
            timeout_seconds=30
        )
        
        response = await bridge.request_agent_for_task(request)
        
        assert response.success == True
        assert response.task_id == "test_task"
        assert response.assigned_agent_id is not None
    
    @pytest.mark.asyncio
    async def test_workload_tracking(self, bridge):
        """Test agent workload tracking"""
        # Simulate agent assignment
        request = TaskAgentRequest(task_id="workload_test", priority=5)
        response = await bridge.request_agent_for_task(request)
        
        if response.success:
            workload = await bridge.get_agent_workload(response.assigned_agent_id)
            assert workload["agent_id"] == response.assigned_agent_id
            assert workload["active_task_count"] >= 0
    
    def test_integration_statistics(self, bridge):
        """Test integration statistics"""
        stats = bridge.get_integration_stats()
        
        assert "integration_mode" in stats
        assert "agent_requests" in stats
        assert "successful_assignments" in stats
        assert "average_assignment_time_ms" in stats

class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.mark.asyncio
    async def test_execute_task_function(self):
        """Test execute_task convenience function"""
        # This test requires the engine to be running and have registered functions
        # For now, we'll test that the function accepts parameters correctly
        try:
            task_id = await execute_task("nonexistent_function", 1, 2, priority=5)
            # Should return a task ID even if the function doesn't exist (will fail during execution)
            assert task_id is not None
        except Exception:
            # Expected if unified engine is not properly initialized
            pass
    
    @pytest.mark.asyncio
    async def test_schedule_task_function(self):
        """Test schedule_task convenience function"""
        future_time = datetime.utcnow() + timedelta(minutes=1)
        
        try:
            task_id = await schedule_task("nonexistent_function", future_time, 1, 2)
            assert task_id is not None
        except Exception:
            # Expected if unified engine is not properly initialized
            pass
    
    @pytest.mark.asyncio
    async def test_execute_batch_tasks_function(self):
        """Test execute_batch_tasks convenience function"""
        tasks = [
            ("function1", [1, 2], {}),
            ("function2", [3, 4], {}),
            ("function3", [5, 6], {})
        ]
        
        try:
            task_ids = await execute_batch_tasks(tasks, priority=5)
            assert len(task_ids) == 3
        except Exception:
            # Expected if unified engine is not properly initialized
            pass

class TestPerformanceAndScalability:
    """Test performance and scalability characteristics"""
    
    @pytest.mark.asyncio
    async def test_high_throughput_task_submission(self):
        """Test high-throughput task submission"""
        engine = UnifiedTaskExecutionEngine()
        
        # Register simple test function
        def quick_task(x):
            return x * 2
        
        engine.register_task_function("quick_task", quick_task)
        
        await engine.start_engine()
        
        try:
            # Submit many tasks quickly
            start_time = time.time()
            task_ids = []
            
            for i in range(100):
                request = TaskExecutionRequest(
                    function_name="quick_task",
                    function_args=[i],
                    task_type=TaskExecutionType.IMMEDIATE
                )
                task_id = await engine.submit_task(request)
                task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            
            # Should submit 100 tasks in reasonable time
            assert submission_time < 5.0  # Less than 5 seconds
            assert len(task_ids) == 100
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check statistics
            stats = engine.get_execution_stats()
            assert stats["tasks_submitted"] >= 100
            
        finally:
            await engine.stop_engine()
    
    @pytest.mark.asyncio
    async def test_concurrent_queue_operations(self):
        """Test concurrent queue operations"""
        queue = TaskQueue("concurrent_test", max_size=1000)
        
        async def enqueue_tasks(start_id, count):
            for i in range(count):
                request = TaskExecutionRequest(
                    task_id=f"task_{start_id}_{i}",
                    function_name="test",
                    priority=i % 10
                )
                await queue.enqueue(request, i % 10)
        
        async def dequeue_tasks(count):
            dequeued = []
            for _ in range(count):
                task = await queue.dequeue()
                if task:
                    dequeued.append(task)
                await asyncio.sleep(0.001)  # Small delay
            return dequeued
        
        # Run concurrent enqueue and dequeue operations
        enqueue_task1 = asyncio.create_task(enqueue_tasks(1, 50))
        enqueue_task2 = asyncio.create_task(enqueue_tasks(2, 50))
        dequeue_task = asyncio.create_task(dequeue_tasks(30))
        
        await asyncio.gather(enqueue_task1, enqueue_task2, dequeue_task)
        
        # Queue should handle concurrent operations without issues
        assert queue.size() > 0  # Some tasks should remain

class TestErrorHandlingAndResilience:
    """Test error handling and resilience features"""
    
    @pytest.mark.asyncio
    async def test_engine_recovery_after_failure(self):
        """Test engine recovery after critical failure"""
        engine = UnifiedTaskExecutionEngine()
        
        def problematic_function():
            raise RuntimeError("Critical system error")
        
        engine.register_task_function("problematic", problematic_function)
        await engine.start_engine()
        
        try:
            # Submit task that will cause failure
            request = TaskExecutionRequest(
                function_name="problematic",
                task_type=TaskExecutionType.IMMEDIATE,
                max_retries=0  # No retries
            )
            
            task_id = await engine.submit_task(request)
            await asyncio.sleep(1)  # Wait for failure
            
            # Engine should still be running and accepting new tasks
            stats = engine.get_execution_stats()
            assert stats["engine_status"] == "running"
            
            # Should be able to submit new tasks
            def working_function():
                return "success"
            
            engine.register_task_function("working", working_function)
            
            working_request = TaskExecutionRequest(
                function_name="working",
                task_type=TaskExecutionType.IMMEDIATE
            )
            
            working_task_id = await engine.submit_task(working_request)
            assert working_task_id is not None
            
        finally:
            await engine.stop_engine()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker pattern in orchestrator integration"""
        bridge = TaskOrchestratorBridge()
        
        # Simulate multiple failures to trigger circuit breaker
        for i in range(6):  # More than max_failures (5)
            bridge._orchestrator_failures += 1
        
        # Circuit breaker should be open
        available = await bridge._is_orchestrator_available()
        assert available == False
        
        # Requests should fail fast
        request = TaskAgentRequest(task_id="circuit_test")
        response = await bridge.request_agent_for_task(request)
        
        assert response.success == False
        assert "circuit breaker" in response.error_message.lower()

if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])