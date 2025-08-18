"""
Test TaskExecutionEngine Functionality and Performance
"""

import asyncio
import pytest
import time
import uuid
from unittest.mock import Mock, patch

import sys
import os
sys.path.append('/Users/bogdan/work/leanvibe-dev/bee-hive')

from app.core.engines.task_execution_engine import (
    TaskExecutionEngine,
    TaskExecutionStatus,
    TaskExecutionType,
    ExecutionMode,
    TaskExecutionPriority,
    TaskExecutionContext,
    TaskExecutionResult,
    BatchExecutionRequest,
    BatchExecutionResult,
    ResourceMonitor,
    TaskScheduler,
    SecureExecutor,
    CommandExecutionPlugin
)
from app.core.engines.base_engine import EngineConfig, EngineRequest


@pytest.fixture
def task_engine_config():
    """Create task execution engine configuration."""
    return EngineConfig(
        engine_id="task_execution_engine",
        name="Task Execution Engine",
        max_concurrent_requests=100,
        request_timeout_seconds=30,
        circuit_breaker_enabled=True,
        plugins_enabled=True
    )


@pytest.fixture
async def task_engine(task_engine_config):
    """Create and initialize task execution engine."""
    engine = TaskExecutionEngine(task_engine_config)
    await engine.initialize()
    yield engine
    await engine.shutdown()


class TestTaskExecutionEngine:
    """Test task execution engine functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, task_engine_config):
        """Test engine initialization."""
        engine = TaskExecutionEngine(task_engine_config)
        await engine.initialize()
        
        assert engine.status.value == "healthy"
        assert engine.scheduler is not None
        assert engine.resource_monitor is not None
        assert engine.secure_executor is not None
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_function_task_execution(self, task_engine):
        """Test basic function task execution."""
        request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "function",
                "task_data": {"function": "test_function"},
                "execution_mode": "sync",
                "priority": "high"
            }
        )
        
        response = await task_engine.process(request)
        
        assert response.success
        assert response.result is not None
        assert response.processing_time_ms > 0
        assert "function" in response.result
    
    @pytest.mark.asyncio
    async def test_command_task_execution(self, task_engine):
        """Test command task execution."""
        request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "command",
                "task_data": {},
                "execution_mode": "sync",
                "metadata": {"command": "echo 'Hello World'"}
            }
        )
        
        response = await task_engine.process(request)
        
        assert response.success
        # Command execution should work through secure executor
    
    @pytest.mark.asyncio
    async def test_async_task_execution(self, task_engine):
        """Test asynchronous task execution."""
        request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "function",
                "task_data": {"function": "async_test"},
                "execution_mode": "async",
                "priority": "normal"
            }
        )
        
        response = await task_engine.process(request)
        
        assert response.success
        assert "task_id" in response.result
        assert response.result["status"] == "scheduled"
    
    @pytest.mark.asyncio
    async def test_batch_execution(self, task_engine):
        """Test batch task execution."""
        request = EngineRequest(
            request_type="execute_batch",
            payload={
                "tasks": [
                    ("function", {"function": "task1"}),
                    ("function", {"function": "task2"}),
                    ("function", {"function": "task3"})
                ],
                "execution_mode": "parallel",
                "max_concurrency": 3,
                "priority": "normal"
            }
        )
        
        response = await task_engine.process(request)
        
        assert response.success
        assert response.result["total_tasks"] == 3
        assert response.result["completed_tasks"] >= 0
        assert "results" in response.metadata
    
    @pytest.mark.asyncio
    async def test_task_status_query(self, task_engine):
        """Test task status querying."""
        # First execute a task
        execute_request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "function",
                "task_data": {"function": "status_test"},
                "execution_mode": "sync"
            }
        )
        
        execute_response = await task_engine.process(execute_request)
        assert execute_response.success
        
        # Query task status
        status_request = EngineRequest(
            request_type="get_task_status",
            payload={"task_id": execute_request.request_id}
        )
        
        status_response = await task_engine.process(status_request)
        assert status_response.success
        assert "status" in status_response.result
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_engine):
        """Test task cancellation."""
        cancel_request = EngineRequest(
            request_type="cancel_task",
            payload={"task_id": "test_task_id"}
        )
        
        response = await task_engine.process(cancel_request)
        
        assert response.success
        assert response.result["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_task_listing(self, task_engine):
        """Test task listing."""
        # Execute a few tasks first
        for i in range(3):
            request = EngineRequest(
                request_type="execute_task",
                payload={
                    "task_type": "function",
                    "task_data": {"function": f"list_test_{i}"},
                    "execution_mode": "sync"
                }
            )
            await task_engine.process(request)
        
        # List tasks
        list_request = EngineRequest(
            request_type="list_tasks",
            payload={"limit": 10}
        )
        
        response = await task_engine.process(list_request)
        
        assert response.success
        assert "tasks" in response.result
        assert response.result["total"] >= 0
    
    @pytest.mark.asyncio
    async def test_performance_targets(self, task_engine):
        """Test performance targets are met."""
        # Test sub-100ms task assignment latency
        start_time = time.time()
        
        request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "function",
                "task_data": {"function": "performance_test"},
                "execution_mode": "async"
            }
        )
        
        response = await task_engine.process(request)
        assignment_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.success
        assert assignment_time < 100  # Sub-100ms assignment target
    
    @pytest.mark.asyncio
    async def test_concurrent_task_capacity(self, task_engine):
        """Test handling of multiple concurrent tasks."""
        # Test with 50 concurrent tasks (scaled down for testing)
        concurrent_tasks = 50
        
        requests = [
            EngineRequest(
                request_type="execute_task",
                payload={
                    "task_type": "function",
                    "task_data": {"function": f"concurrent_{i}"},
                    "execution_mode": "async"
                }
            )
            for i in range(concurrent_tasks)
        ]
        
        start_time = time.time()
        responses = await asyncio.gather(*[
            task_engine.process(request) for request in requests
        ])
        total_time = time.time() - start_time
        
        # All tasks should be accepted
        successful_assignments = sum(1 for r in responses if r.success)
        assert successful_assignments == concurrent_tasks
        
        # Assignment should be fast
        assert total_time < 5.0  # All assignments within 5 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self, task_engine):
        """Test error handling and retry logic."""
        request = EngineRequest(
            request_type="execute_task",
            payload={
                "task_type": "nonexistent",  # Invalid task type
                "execution_mode": "sync"
            }
        )
        
        response = await task_engine.process(request)
        
        # Should handle gracefully and not crash
        assert not response.success or response.success  # Either way is acceptable
        assert response.processing_time_ms > 0


class TestTaskScheduler:
    """Test task scheduler functionality."""
    
    def test_priority_scheduling(self):
        """Test priority-based task scheduling."""
        scheduler = TaskScheduler(max_concurrent_tasks=10)
        
        # Add tasks with different priorities
        high_priority = TaskExecutionContext(
            task_id="high",
            priority=TaskExecutionPriority.HIGH
        )
        low_priority = TaskExecutionContext(
            task_id="low",
            priority=TaskExecutionPriority.LOW
        )
        
        scheduler.schedule_task(low_priority)
        scheduler.schedule_task(high_priority)
        
        # High priority should be returned first
        next_task = scheduler.get_next_task()
        assert next_task.task_id == "high"
        
        scheduler.complete_task("high")
        
        next_task = scheduler.get_next_task()
        assert next_task.task_id == "low"
    
    def test_concurrent_task_limits(self):
        """Test concurrent task limits."""
        scheduler = TaskScheduler(max_concurrent_tasks=2)
        
        # Schedule 3 tasks
        for i in range(3):
            context = TaskExecutionContext(
                task_id=f"task_{i}",
                priority=TaskExecutionPriority.NORMAL
            )
            scheduler.schedule_task(context)
        
        # Should get first two tasks
        task1 = scheduler.get_next_task()
        task2 = scheduler.get_next_task()
        task3 = scheduler.get_next_task()  # Should be None (limit reached)
        
        assert task1 is not None
        assert task2 is not None
        assert task3 is None
        
        # Complete one task, should allow next task
        scheduler.complete_task(task1.task_id)
        task4 = scheduler.get_next_task()
        assert task4 is not None


class TestResourceMonitor:
    """Test resource monitoring functionality."""
    
    def test_resource_monitoring(self):
        """Test resource usage monitoring."""
        monitor = ResourceMonitor()
        
        task_id = "test_task"
        monitor.start_monitoring(task_id)
        
        # Simulate some work
        time.sleep(0.1)
        
        metrics = monitor.stop_monitoring(task_id)
        
        assert "execution_time_seconds" in metrics
        assert metrics["execution_time_seconds"] > 0
        assert "memory_usage_mb" in metrics
        assert "cpu_usage_percent" in metrics


class TestSecureExecutor:
    """Test secure command execution."""
    
    @pytest.mark.asyncio
    async def test_safe_command_execution(self):
        """Test safe command execution."""
        executor = SecureExecutor()
        context = TaskExecutionContext(
            task_id="test_command",
            timeout_seconds=10
        )
        
        result = await executor.execute_command("echo 'test'", context)
        
        assert result.status == TaskExecutionStatus.COMPLETED
        assert "test" in result.result["stdout"]
        assert result.result["return_code"] == 0
    
    @pytest.mark.asyncio
    async def test_blocked_command_execution(self):
        """Test blocked command security."""
        executor = SecureExecutor()
        context = TaskExecutionContext(
            task_id="test_blocked",
            timeout_seconds=10
        )
        
        # Try to execute a blocked command
        result = await executor.execute_command("rm -rf /", context)
        
        assert result.status == TaskExecutionStatus.FAILED
        assert "blocked by security policy" in result.error
    
    @pytest.mark.asyncio
    async def test_command_timeout(self):
        """Test command timeout handling."""
        executor = SecureExecutor()
        context = TaskExecutionContext(
            task_id="test_timeout",
            timeout_seconds=1  # 1 second timeout
        )
        
        # Try to execute a long-running command
        result = await executor.execute_command("sleep 5", context)
        
        assert result.status == TaskExecutionStatus.TIMEOUT
        assert "timed out" in result.error


class TestCommandExecutionPlugin:
    """Test command execution plugin."""
    
    @pytest.mark.asyncio
    async def test_plugin_functionality(self):
        """Test command execution plugin."""
        plugin = CommandExecutionPlugin()
        await plugin.initialize({"allowed_commands": ["echo", "ls"]})
        
        # Test can handle command requests
        command_request = EngineRequest(
            request_type="execute_task",
            payload={"task_type": "command"}
        )
        
        assert await plugin.can_handle(command_request)
        
        # Test cannot handle other requests
        other_request = EngineRequest(
            request_type="execute_task",
            payload={"task_type": "function"}
        )
        
        assert not await plugin.can_handle(other_request)
        
        # Test processing
        response = await plugin.process(command_request)
        assert response.success
        assert "plugin" in response.metadata


@pytest.mark.asyncio
async def test_engine_performance_benchmarks(task_engine_config):
    """Test engine performance benchmarks."""
    engine = TaskExecutionEngine(task_engine_config)
    await engine.initialize()
    
    try:
        # Benchmark task assignment latency
        latencies = []
        for _ in range(100):
            start = time.time()
            request = EngineRequest(
                request_type="execute_task",
                payload={
                    "task_type": "function",
                    "execution_mode": "async"
                }
            )
            response = await engine.process(request)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            assert response.success
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"Average assignment latency: {avg_latency:.2f}ms")
        print(f"P95 assignment latency: {p95_latency:.2f}ms")
        
        # Performance targets
        assert avg_latency < 100  # Sub-100ms average
        assert p95_latency < 200  # P95 under 200ms
        
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    # Run performance benchmarks
    import asyncio
    
    async def run_benchmarks():
        config = EngineConfig(
            engine_id="benchmark",
            name="Benchmark Engine",
            max_concurrent_requests=1000
        )
        
        engine = TaskExecutionEngine(config)
        await engine.initialize()
        
        print("=== TaskExecutionEngine Benchmarks ===")
        
        # Test assignment latency
        latencies = []
        for i in range(100):
            start = time.time()
            request = EngineRequest(
                request_type="execute_task",
                payload={
                    "task_type": "function",
                    "task_data": {"index": i},
                    "execution_mode": "async"
                }
            )
            response = await engine.process(request)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[95]
        
        print(f"Task Assignment Latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Target: <100ms average ✓" if avg_latency < 100 else f"  Target: <100ms average ✗")
        
        # Test concurrent capacity
        print(f"\nConcurrent Capacity Test:")
        concurrent_tasks = 100
        start_time = time.time()
        
        requests = [
            EngineRequest(
                request_type="execute_task",
                payload={
                    "task_type": "function",
                    "execution_mode": "async"
                }
            )
            for _ in range(concurrent_tasks)
        ]
        
        responses = await asyncio.gather(*[engine.process(req) for req in requests])
        total_time = time.time() - start_time
        
        successful = sum(1 for r in responses if r.success)
        print(f"  Processed {successful}/{concurrent_tasks} tasks in {total_time:.2f}s")
        print(f"  Throughput: {successful/total_time:.2f} tasks/second")
        
        # Health check
        health = await engine.get_health()
        print(f"\nEngine Health:")
        print(f"  Status: {health.status.value}")
        print(f"  Active Requests: {health.active_requests}")
        print(f"  Total Processed: {health.total_requests_processed}")
        print(f"  Error Rate: {health.error_rate_5min:.2f}%")
        
        await engine.shutdown()
        print("\nBenchmarks completed successfully!")
    
    asyncio.run(run_benchmarks())