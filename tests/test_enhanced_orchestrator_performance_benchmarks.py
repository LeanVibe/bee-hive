"""
Performance Benchmarks for Enhanced Agent Orchestrator

This test suite provides comprehensive performance benchmarks and stress tests
for the enhanced orchestrator functionality, ensuring it meets production
performance requirements under various load conditions.

Performance Targets:
- Task assignment: < 100ms per task
- Agent coordination: < 50ms per agent operation  
- Circuit breaker operations: < 10ms overhead
- Concurrent task processing: > 100 tasks/second
- Memory usage: < 500MB for 100 agents
- Load balancing: < 200ms per rebalancing operation
"""

import pytest
import asyncio
import time
import uuid
import statistics
import gc
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentInstance, AgentCapability
from app.core.intelligent_task_router import TaskRoutingContext, RoutingStrategy
from app.models.agent import AgentStatus
from app.models.task import Task, TaskStatus, TaskPriority, TaskType


@pytest.fixture
def performance_orchestrator():
    """Create orchestrator optimized for performance testing."""
    orchestrator = AgentOrchestrator()
    
    # Mock external dependencies to eliminate network/IO latency
    orchestrator.message_broker = AsyncMock()
    orchestrator.session_cache = AsyncMock()
    orchestrator.anthropic_client = AsyncMock()
    orchestrator.persona_system = AsyncMock()
    orchestrator.intelligent_router = AsyncMock()
    orchestrator.workflow_engine = AsyncMock()
    
    # Configure for optimal performance
    orchestrator.intelligent_router.route_task_to_agent.return_value = "perf-agent-001"
    orchestrator.intelligent_router.calculate_routing_score.return_value = 0.85
    
    return orchestrator


@pytest.fixture 
def performance_agents():
    """Create a large set of agents for performance testing."""
    agents = {}
    
    for i in range(100):  # 100 agents for stress testing
        agent_id = f"perf-agent-{i:03d}"
        
        capability = AgentCapability(
            name=f"capability_{i % 10}",  # 10 different capability types
            description=f"Performance test capability {i % 10}",
            confidence_level=0.8 + (i % 20) * 0.01,  # Vary confidence levels
            specialization_areas=[f"area_{i % 5}", f"area_{(i + 1) % 5}"]
        )
        
        agents[agent_id] = AgentInstance(
            id=agent_id,
            role=list(AgentRole)[i % len(AgentRole)],  # Distribute roles evenly
            status=AgentStatus.ACTIVE if i % 10 != 9 else AgentStatus.BUSY,  # 90% active
            tmux_session=f"session-{i:03d}",
            capabilities=[capability],
            current_task=f"task-{i}" if i % 10 == 9 else None,
            context_window_usage=(i % 80) / 100.0,  # 0% to 79% usage
            last_heartbeat=datetime.utcnow() - timedelta(seconds=i % 300),
            anthropic_client=None
        )
    
    return agents


@pytest.fixture
def mock_database_session():
    """Create optimized mock database session for performance tests."""
    mock_session = AsyncMock()
    
    # Pre-configure common responses to minimize mock overhead
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.execute.return_value.scalar.return_value = 0
    mock_session.execute.return_value.scalars.return_value.all.return_value = []
    mock_session.get.return_value = None
    
    return mock_session


class PerformanceBenchmark:
    """Utility class for performance measurement and analysis."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.measurements = []
        self.memory_before = None
        self.memory_after = None
    
    def start(self):
        """Start performance measurement."""
        gc.collect()  # Clean up before measurement
        self.memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop performance measurement."""
        self.end_time = time.perf_counter()
        self.memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def add_measurement(self, value: float):
        """Add a single measurement."""
        self.measurements.append(value)
    
    @property
    def duration(self) -> float:
        """Get total duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def memory_delta(self) -> float:
        """Get memory usage change in MB."""
        if self.memory_before is None or self.memory_after is None:
            return 0.0
        return self.memory_after - self.memory_before
    
    @property
    def statistics(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.measurements:
            return {}
        
        return {
            'count': len(self.measurements),
            'mean': statistics.mean(self.measurements),
            'median': statistics.median(self.measurements),
            'min': min(self.measurements),
            'max': max(self.measurements),
            'stdev': statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0,
            'percentile_95': statistics.quantiles(self.measurements, n=20)[18] if len(self.measurements) > 20 else max(self.measurements)
        }
    
    def assert_performance(self, max_duration: float = None, max_memory_mb: float = None, 
                          max_mean_ms: float = None, max_p95_ms: float = None):
        """Assert performance requirements are met."""
        if max_duration and self.duration > max_duration:
            raise AssertionError(f"{self.name}: Duration {self.duration:.3f}s exceeds limit {max_duration}s")
        
        if max_memory_mb and self.memory_delta > max_memory_mb:
            raise AssertionError(f"{self.name}: Memory usage {self.memory_delta:.1f}MB exceeds limit {max_memory_mb}MB")
        
        stats = self.statistics
        if max_mean_ms and stats.get('mean', 0) * 1000 > max_mean_ms:
            raise AssertionError(f"{self.name}: Mean {stats['mean']*1000:.1f}ms exceeds limit {max_mean_ms}ms")
        
        if max_p95_ms and stats.get('percentile_95', 0) * 1000 > max_p95_ms:
            raise AssertionError(f"{self.name}: P95 {stats['percentile_95']*1000:.1f}ms exceeds limit {max_p95_ms}ms")


@pytest.mark.performance
class TestTaskAssignmentPerformance:
    """Performance tests for task assignment operations."""
    
    async def test_single_task_assignment_latency(self, performance_orchestrator, performance_agents):
        """Test latency of single task assignment."""
        orchestrator = performance_orchestrator
        orchestrator.agents = performance_agents
        
        benchmark = PerformanceBenchmark("Single Task Assignment")
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = AsyncMock()
            
            # Measure 100 individual task assignments
            for i in range(100):
                start_time = time.perf_counter()
                
                task_id = await orchestrator.delegate_task(
                    task_description=f"Performance test task {i}",
                    task_type="performance_test",
                    priority=TaskPriority.MEDIUM
                )
                
                end_time = time.perf_counter()
                benchmark.add_measurement(end_time - start_time)
                
                assert task_id is not None
        
        # Performance assertions
        benchmark.assert_performance(
            max_mean_ms=100.0,  # Average < 100ms
            max_p95_ms=200.0    # 95th percentile < 200ms
        )
        
        print(f"Task Assignment Performance: {benchmark.statistics}")
    
    async def test_concurrent_task_assignment_throughput(self, performance_orchestrator, performance_agents):
        """Test throughput of concurrent task assignments."""
        orchestrator = performance_orchestrator
        orchestrator.agents = performance_agents
        
        benchmark = PerformanceBenchmark("Concurrent Task Assignment")
        benchmark.start()
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_get_session.return_value.__aenter__.return_value = AsyncMock()
            
            # Create 500 concurrent task assignments
            tasks = []
            for i in range(500):
                task = orchestrator.delegate_task(
                    task_description=f"Concurrent task {i}",
                    task_type="concurrent_test",
                    priority=TaskPriority.MEDIUM
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        benchmark.stop()
        
        # Verify all tasks completed successfully
        successful_tasks = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_tasks) == 500
        
        # Calculate throughput
        throughput = len(successful_tasks) / benchmark.duration
        assert throughput >= 100, f"Throughput {throughput:.1f} tasks/sec below requirement 100 tasks/sec"
        
        benchmark.assert_performance(
            max_duration=5.0,   # Complete 500 tasks in under 5 seconds
            max_memory_mb=100.0  # Memory growth under 100MB
        )
        
        print(f"Concurrent Assignment Throughput: {throughput:.1f} tasks/sec")
    
    async def test_agent_selection_performance(self, performance_orchestrator, performance_agents):
        """Test performance of agent selection algorithm."""
        orchestrator = performance_orchestrator
        orchestrator.agents = performance_agents
        
        # Use real intelligent router for this test
        from app.core.intelligent_task_router import IntelligentTaskRouter
        orchestrator.intelligent_router = IntelligentTaskRouter()
        
        benchmark = PerformanceBenchmark("Agent Selection")
        
        # Test agent selection with various task types
        task_types = [TaskType.FEATURE_DEVELOPMENT, TaskType.BUG_FIX, TaskType.TESTING, 
                     TaskType.DOCUMENTATION, TaskType.ANALYSIS]
        
        for task_type in task_types:
            for i in range(20):  # 20 selections per task type
                routing_context = TaskRoutingContext(
                    task_id=f"perf-task-{task_type.value}-{i}",
                    task_type=task_type,
                    priority=TaskPriority.MEDIUM,
                    required_capabilities=["python", "testing"],
                    available_agents=list(orchestrator.agents.keys()),
                    routing_strategy=RoutingStrategy.ADAPTIVE
                )
                
                start_time = time.perf_counter()
                selected_agent = await orchestrator.intelligent_router.route_task_to_agent(routing_context)
                end_time = time.perf_counter()
                
                benchmark.add_measurement(end_time - start_time)
                assert selected_agent in orchestrator.agents
        
        benchmark.assert_performance(
            max_mean_ms=50.0,   # Average < 50ms
            max_p95_ms=100.0    # 95th percentile < 100ms
        )
        
        print(f"Agent Selection Performance: {benchmark.statistics}")


@pytest.mark.performance
class TestCircuitBreakerPerformance:
    """Performance tests for circuit breaker operations."""
    
    async def test_circuit_breaker_update_overhead(self, performance_orchestrator):
        """Test overhead of circuit breaker updates."""
        orchestrator = performance_orchestrator
        
        benchmark = PerformanceBenchmark("Circuit Breaker Updates")
        
        # Test 10,000 circuit breaker updates (success and failure)
        for i in range(10000):
            agent_id = f"cb-agent-{i % 50}"  # 50 different agents
            success = i % 3 != 0  # 33% failure rate
            
            start_time = time.perf_counter()
            await orchestrator._update_circuit_breaker(agent_id, success=success)
            end_time = time.perf_counter()
            
            benchmark.add_measurement(end_time - start_time)
        
        benchmark.assert_performance(
            max_mean_ms=10.0,   # Average < 10ms overhead
            max_p95_ms=20.0     # 95th percentile < 20ms
        )
        
        print(f"Circuit Breaker Update Performance: {benchmark.statistics}")
    
    async def test_circuit_breaker_decision_performance(self, performance_orchestrator):
        """Test performance of circuit breaker trip decisions."""
        orchestrator = performance_orchestrator
        
        # Setup circuit breakers in various states
        for i in range(100):
            agent_id = f"decision-agent-{i}"
            orchestrator.circuit_breakers[agent_id] = {
                'state': 'closed',
                'failure_count': i % 15,  # Vary failure counts
                'consecutive_failures': i % 8,
                'total_requests': 100 + i,
                'successful_requests': 80 + (i % 20),
                'last_failure_time': time.time() - (i % 300)
            }
        
        benchmark = PerformanceBenchmark("Circuit Breaker Decisions")
        
        # Test decision performance
        for i in range(1000):
            agent_id = f"decision-agent-{i % 100}"
            
            start_time = time.perf_counter()
            should_trip = await orchestrator._should_trip_circuit_breaker(agent_id)
            end_time = time.perf_counter()
            
            benchmark.add_measurement(end_time - start_time)
            assert isinstance(should_trip, bool)
        
        benchmark.assert_performance(
            max_mean_ms=5.0,    # Average < 5ms
            max_p95_ms=15.0     # 95th percentile < 15ms
        )
        
        print(f"Circuit Breaker Decision Performance: {benchmark.statistics}")


@pytest.mark.performance  
class TestLoadBalancingPerformance:
    """Performance tests for load balancing operations."""
    
    async def test_workload_analysis_performance(self, performance_orchestrator, performance_agents):
        """Test performance of workload analysis."""
        orchestrator = performance_orchestrator
        orchestrator.agents = performance_agents
        
        benchmark = PerformanceBenchmark("Workload Analysis")
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            
            # Perform workload analysis multiple times
            for i in range(50):
                start_time = time.perf_counter()
                analysis = await orchestrator._analyze_agent_workloads()
                end_time = time.perf_counter()
                
                benchmark.add_measurement(end_time - start_time)
                
                # Verify analysis structure
                assert 'agents' in analysis
                assert 'balance_score' in analysis
                assert 'overloaded_agents' in analysis
                assert 'underutilized_agents' in analysis
        
        benchmark.assert_performance(
            max_mean_ms=200.0,  # Average < 200ms
            max_p95_ms=500.0    # 95th percentile < 500ms
        )
        
        print(f"Workload Analysis Performance: {benchmark.statistics}")
    
    async def test_rebalancing_operation_performance(self, performance_orchestrator, performance_agents):
        """Test performance of workload rebalancing operations."""
        orchestrator = performance_orchestrator
        orchestrator.agents = performance_agents
        
        # Configure some agents as overloaded
        overloaded_agents = list(performance_agents.keys())[:10]
        for agent_id in overloaded_agents:
            orchestrator.agents[agent_id].context_window_usage = 0.95
            orchestrator.agents[agent_id].current_task = f"heavy-task-{agent_id}"
        
        benchmark = PerformanceBenchmark("Workload Rebalancing")
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            mock_session.execute.return_value.scalars.return_value.all.return_value = []
            
            # Mock intelligent rebalancing to avoid database operations
            orchestrator._intelligent_workload_rebalancing = AsyncMock(return_value=[])
            
            # Perform rebalancing operations
            for i in range(20):
                start_time = time.perf_counter()
                result = await orchestrator.rebalance_agent_workloads(force_rebalance=True)
                end_time = time.perf_counter()
                
                benchmark.add_measurement(end_time - start_time)
                assert 'rebalanced' in result or 'skipped' in result
        
        benchmark.assert_performance(
            max_mean_ms=300.0,  # Average < 300ms
            max_p95_ms=800.0    # 95th percentile < 800ms
        )
        
        print(f"Workload Rebalancing Performance: {benchmark.statistics}")


@pytest.mark.performance
class TestMemoryUsagePerformance:
    """Performance tests for memory usage optimization."""
    
    async def test_orchestrator_memory_growth(self, performance_orchestrator):
        """Test orchestrator memory usage doesn't grow unbounded."""
        orchestrator = performance_orchestrator
        
        benchmark = PerformanceBenchmark("Memory Usage Growth")
        benchmark.start()
        
        # Simulate long-running orchestrator operations
        for i in range(1000):
            # Add agents
            agent_id = f"memory-agent-{i % 100}"  # Reuse agent IDs to test cleanup
            agent = AgentInstance(
                id=agent_id,
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=None,
                capabilities=[],
                current_task=None,
                context_window_usage=0.3,
                last_heartbeat=datetime.utcnow(),
                anthropic_client=None
            )
            orchestrator.agents[agent_id] = agent
            
            # Update circuit breakers
            await orchestrator._update_circuit_breaker(agent_id, success=i % 4 != 0)
            
            # Update metrics
            orchestrator.metrics['tasks_completed'] += 1
            orchestrator.metrics['routing_decisions'] += 1
            
            # Periodically clean up to simulate realistic usage
            if i % 100 == 99:
                # Remove some agents to simulate cleanup
                agents_to_remove = list(orchestrator.agents.keys())[:10]
                for aid in agents_to_remove:
                    del orchestrator.agents[aid]
                
                # Clean up old circuit breaker data
                old_breakers = [k for k in orchestrator.circuit_breakers.keys() if k not in orchestrator.agents]
                for breaker_id in old_breakers[:5]:
                    del orchestrator.circuit_breakers[breaker_id]
        
        benchmark.stop()
        
        # Memory growth should be reasonable
        benchmark.assert_performance(max_memory_mb=200.0)  # Less than 200MB growth
        
        print(f"Memory Usage: {benchmark.memory_delta:.1f}MB growth over 1000 operations")
    
    async def test_circuit_breaker_memory_efficiency(self, performance_orchestrator):
        """Test circuit breaker memory efficiency with many agents."""
        orchestrator = performance_orchestrator
        
        benchmark = PerformanceBenchmark("Circuit Breaker Memory")
        benchmark.start()
        
        # Create circuit breakers for 1000 agents
        for i in range(1000):
            agent_id = f"cb-memory-agent-{i}"
            
            # Simulate realistic circuit breaker usage
            for j in range(50):  # 50 operations per agent
                success = j % 5 != 0  # 80% success rate
                await orchestrator._update_circuit_breaker(agent_id, success=success)
        
        benchmark.stop()
        
        # Verify circuit breaker data structure is reasonable
        assert len(orchestrator.circuit_breakers) == 1000
        
        benchmark.assert_performance(max_memory_mb=50.0)  # Less than 50MB for all circuit breakers
        
        print(f"Circuit Breaker Memory: {benchmark.memory_delta:.1f}MB for 1000 agents")


@pytest.mark.performance
class TestScalabilityLimits:
    """Test scalability limits and breaking points."""
    
    async def test_maximum_concurrent_agents(self, performance_orchestrator):
        """Test maximum number of concurrent agents the system can handle."""
        orchestrator = performance_orchestrator
        
        benchmark = PerformanceBenchmark("Maximum Concurrent Agents") 
        benchmark.start()
        
        max_agents = 0
        try:
            # Gradually increase agent count until performance degrades
            for agent_count in [100, 500, 1000, 2000, 5000]:
                # Clear previous agents
                orchestrator.agents.clear()
                
                # Create agents
                for i in range(agent_count):
                    agent_id = f"scale-agent-{i}"
                    orchestrator.agents[agent_id] = AgentInstance(
                        id=agent_id,
                        role=AgentRole.BACKEND_DEVELOPER,
                        status=AgentStatus.ACTIVE,
                        tmux_session=None,
                        capabilities=[],
                        current_task=None,
                        context_window_usage=0.3,
                        last_heartbeat=datetime.utcnow(),
                        anthropic_client=None
                    )
                
                # Test performance with this many agents
                start_time = time.perf_counter()
                available_agents = await orchestrator._get_available_agent_ids()
                end_time = time.perf_counter()
                
                operation_time = end_time - start_time
                
                # Performance should remain reasonable (< 1 second)
                if operation_time < 1.0:
                    max_agents = agent_count
                else:
                    break
                    
        except MemoryError:
            # Hit memory limit
            pass
        
        benchmark.stop()
        
        print(f"Maximum Agents: {max_agents} (operation time remained < 1s)")
        assert max_agents >= 1000, f"Should handle at least 1000 agents, only achieved {max_agents}"
    
    async def test_task_queue_scalability(self, performance_orchestrator):
        """Test task queue handling under high load."""
        orchestrator = performance_orchestrator
        
        benchmark = PerformanceBenchmark("Task Queue Scalability")
        benchmark.start()
        
        # Add many tasks to different priority queues
        import heapq
        
        # High priority tasks (heap-based)
        for i in range(1000):
            task_entry = (TaskPriority.HIGH.value, time.time(), {
                'task_id': f'high-task-{i}',
                'queued_at': datetime.utcnow()
            })
            heapq.heappush(orchestrator.task_queues['high_priority'], task_entry)
        
        # Medium and low priority tasks (deque-based)
        for i in range(2000):
            orchestrator.task_queues['medium_priority'].append({
                'task_id': f'medium-task-{i}',
                'queued_at': datetime.utcnow()
            })
            
        for i in range(3000):
            orchestrator.task_queues['low_priority'].append({
                'task_id': f'low-task-{i}',
                'queued_at': datetime.utcnow()
            })
        
        # Test queue processing performance
        start_time = time.perf_counter()
        
        # Process a batch of high priority tasks
        processed = 0
        while orchestrator.task_queues['high_priority'] and processed < 100:
            heapq.heappop(orchestrator.task_queues['high_priority'])
            processed += 1
        
        end_time = time.perf_counter()
        
        benchmark.stop()
        
        processing_time = end_time - start_time
        throughput = processed / processing_time
        
        print(f"Queue Processing: {throughput:.1f} tasks/sec from 6000 total queued tasks")
        assert throughput >= 1000, f"Queue processing too slow: {throughput:.1f} tasks/sec"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v", 
        "-m", "performance",
        "--tb=short",
        "-x"  # Stop on first failure for performance tests
    ])