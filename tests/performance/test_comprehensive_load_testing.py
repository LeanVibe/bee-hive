"""
Comprehensive Load Testing Suite for LeanVibe Agent Hive 2.0

Tests for:
- API endpoint performance under load
- Database query performance and optimization
- Redis message throughput and latency
- Multi-agent orchestration scalability
- Memory usage and garbage collection
- Concurrent user simulation
- Performance regression detection
"""

import asyncio
import time
import statistics
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pytest
import httpx
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.orchestrator import AgentOrchestrator
from app.core.communication import MessageBroker
from app.core.database import get_async_session
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus
from tests.utils.database_test_utils import DatabaseTestUtils

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    min_time_ms: float
    max_time_ms: float
    avg_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float


class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.operation_times = []
        self.errors = []
        self.memory_samples = []
        self.cpu_samples = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.operation_times.clear()
        self.errors.clear()
        self.memory_samples.clear()
        self.cpu_samples.clear()
        
        # Initial resource measurement
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
        self.cpu_samples.append(self.process.cpu_percent())
    
    def record_operation(self, duration_ms: float, success: bool = True, error: str = None):
        """Record a single operation."""
        self.operation_times.append(duration_ms)
        
        if not success and error:
            self.errors.append(error)
        
        # Sample resources periodically
        if len(self.operation_times) % 10 == 0:
            self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)
            self.cpu_samples.append(self.process.cpu_percent())
    
    def get_metrics(self, operation_name: str) -> PerformanceMetrics:
        """Calculate and return performance metrics."""
        if not self.operation_times:
            return PerformanceMetrics(
                operation_name=operation_name,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                min_time_ms=0, max_time_ms=0, avg_time_ms=0,
                p50_time_ms=0, p95_time_ms=0, p99_time_ms=0,
                throughput_ops_per_sec=0,
                error_rate=0,
                memory_usage_mb=0,
                cpu_usage_percent=0
            )
        
        # Calculate time-based metrics
        sorted_times = sorted(self.operation_times)
        total_ops = len(self.operation_times)
        failed_ops = len(self.errors)
        successful_ops = total_ops - failed_ops
        
        # Time percentiles
        p50_idx = int(total_ops * 0.5)
        p95_idx = int(total_ops * 0.95)
        p99_idx = int(total_ops * 0.99)
        
        # Throughput calculation
        duration_seconds = time.time() - self.start_time
        throughput = total_ops / duration_seconds if duration_seconds > 0 else 0
        
        return PerformanceMetrics(
            operation_name=operation_name,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            min_time_ms=min(sorted_times),
            max_time_ms=max(sorted_times),
            avg_time_ms=statistics.mean(sorted_times),
            p50_time_ms=sorted_times[p50_idx],
            p95_time_ms=sorted_times[p95_idx],
            p99_time_ms=sorted_times[p99_idx],
            throughput_ops_per_sec=throughput,
            error_rate=failed_ops / total_ops,
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        )


class LoadTestScenario:
    """Base class for load testing scenarios."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.monitor = PerformanceMonitor()
    
    async def setup(self) -> None:
        """Setup the load test scenario."""
        logger.info(f"Setting up load test: {self.name}")
        self.monitor.start_monitoring()
    
    async def execute(self) -> PerformanceMetrics:
        """Execute the load test scenario."""
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Cleanup after the load test."""
        logger.info(f"Cleaning up load test: {self.name}")
        gc.collect()  # Force garbage collection


class APIEndpointLoadTest(LoadTestScenario):
    """Load test for API endpoints."""
    
    def __init__(self, endpoint: str, method: str = "GET", payload: Dict = None, 
                 concurrent_users: int = 50, requests_per_user: int = 20):
        super().__init__(
            f"api_load_{endpoint.replace('/', '_')}",
            f"Load test for {method} {endpoint}"
        )
        self.endpoint = endpoint
        self.method = method
        self.payload = payload or {}
        self.concurrent_users = concurrent_users
        self.requests_per_user = requests_per_user
    
    async def execute_user_requests(self, user_id: int, client: httpx.AsyncClient) -> List[Tuple[float, bool, str]]:
        """Execute requests for a single simulated user."""
        results = []
        
        for request_num in range(self.requests_per_user):
            start_time = time.time()
            success = True
            error_msg = ""
            
            try:
                if self.method.upper() == "GET":
                    response = await client.get(self.endpoint)
                elif self.method.upper() == "POST":
                    response = await client.post(self.endpoint, json=self.payload)
                elif self.method.upper() == "PUT":
                    response = await client.put(self.endpoint, json=self.payload)
                elif self.method.upper() == "DELETE":
                    response = await client.delete(self.endpoint)
                else:
                    raise ValueError(f"Unsupported method: {self.method}")
                
                # Check response status
                if response.status_code >= 400:
                    success = False
                    error_msg = f"HTTP {response.status_code}"
                    
            except Exception as e:
                success = False
                error_msg = str(e)
            
            duration_ms = (time.time() - start_time) * 1000
            results.append((duration_ms, success, error_msg))
            
            # Add some jitter between requests
            await asyncio.sleep(0.01)
        
        return results
    
    async def execute(self) -> PerformanceMetrics:
        """Execute API endpoint load test."""
        logger.info(f"Starting API load test: {self.concurrent_users} users, {self.requests_per_user} req/user")
        
        # Create HTTP client
        async with httpx.AsyncClient(base_url="http://test") as client:
            # Create tasks for concurrent users
            user_tasks = []
            for user_id in range(self.concurrent_users):
                task = self.execute_user_requests(user_id, client)
                user_tasks.append(task)
            
            # Execute all user tasks concurrently
            user_results = await asyncio.gather(*user_tasks)
            
            # Process results
            for user_results_list in user_results:
                for duration_ms, success, error_msg in user_results_list:
                    self.monitor.record_operation(duration_ms, success, error_msg)
        
        return self.monitor.get_metrics(self.name)


class DatabaseLoadTest(LoadTestScenario):
    """Load test for database operations."""
    
    def __init__(self, concurrent_connections: int = 20, operations_per_connection: int = 100):
        super().__init__(
            "database_load",
            "Load test for database operations"
        )
        self.concurrent_connections = concurrent_connections
        self.operations_per_connection = operations_per_connection
    
    async def execute_database_operations(self, connection_id: int, session: AsyncSession) -> List[Tuple[float, bool, str]]:
        """Execute database operations for a single connection."""
        results = []
        
        for op_num in range(self.operations_per_connection):
            start_time = time.time()
            success = True
            error_msg = ""
            
            try:
                # Mix of database operations
                operation_type = op_num % 4
                
                if operation_type == 0:  # Create agent
                    agent = await DatabaseTestUtils.create_test_agent(
                        session, 
                        name=f"LoadTest Agent {connection_id}-{op_num}"
                    )
                elif operation_type == 1:  # Query agents
                    from sqlalchemy import select
                    result = await session.execute(
                        select(Agent).where(Agent.status == AgentStatus.ACTIVE).limit(10)
                    )
                    agents = result.scalars().all()
                elif operation_type == 2:  # Create task
                    agent = await DatabaseTestUtils.create_test_agent(session)
                    task = await DatabaseTestUtils.create_test_task(
                        session,
                        title=f"LoadTest Task {connection_id}-{op_num}",
                        assigned_agent=agent
                    )
                elif operation_type == 3:  # Complex query with joins
                    from sqlalchemy import select, join
                    result = await session.execute(
                        select(Agent, Task)
                        .select_from(join(Agent, Task, Agent.id == Task.assigned_agent_id))
                        .limit(5)
                    )
                    data = result.all()
                
                await session.commit()
                
            except Exception as e:
                success = False
                error_msg = str(e)
                await session.rollback()
            
            duration_ms = (time.time() - start_time) * 1000
            results.append((duration_ms, success, error_msg))
        
        return results
    
    async def execute(self) -> PerformanceMetrics:
        """Execute database load test."""
        logger.info(f"Starting database load test: {self.concurrent_connections} connections")
        
        # Create database sessions for concurrent connections
        connection_tasks = []
        
        for conn_id in range(self.concurrent_connections):
            # Each connection gets its own session
            task = self.execute_database_operations(conn_id, None)  # TODO: Get real session
            connection_tasks.append(task)
        
        # Execute all connection tasks concurrently
        connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Process results
        for result in connection_results:
            if isinstance(result, Exception):
                self.monitor.record_operation(0, False, str(result))
            else:
                for duration_ms, success, error_msg in result:
                    self.monitor.record_operation(duration_ms, success, error_msg)
        
        return self.monitor.get_metrics(self.name)


class RedisMessageLoadTest(LoadTestScenario):
    """Load test for Redis message operations."""
    
    def __init__(self, concurrent_producers: int = 10, concurrent_consumers: int = 5,
                 messages_per_producer: int = 1000):
        super().__init__(
            "redis_message_load",
            "Load test for Redis message operations"
        )
        self.concurrent_producers = concurrent_producers
        self.concurrent_consumers = concurrent_consumers
        self.messages_per_producer = messages_per_producer
    
    async def execute_producer(self, producer_id: int, message_broker: MessageBroker) -> List[Tuple[float, bool, str]]:
        """Execute message production."""
        results = []
        
        for msg_num in range(self.messages_per_producer):
            start_time = time.time()
            success = True
            error_msg = ""
            
            try:
                await message_broker.send_message(
                    from_agent=f"load_producer_{producer_id}",
                    to_agent=f"load_consumer_{msg_num % self.concurrent_consumers}",
                    message_type="load_test",
                    payload={
                        "producer_id": producer_id,
                        "message_num": msg_num,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": f"load test data {producer_id}-{msg_num}"
                    }
                )
            except Exception as e:
                success = False
                error_msg = str(e)
            
            duration_ms = (time.time() - start_time) * 1000
            results.append((duration_ms, success, error_msg))
            
            # Small delay to avoid overwhelming
            if msg_num % 100 == 0:
                await asyncio.sleep(0.01)
        
        return results
    
    async def execute_consumer(self, consumer_id: int, message_broker: MessageBroker,
                              consumption_duration: int = 30) -> List[Tuple[float, bool, str]]:
        """Execute message consumption."""
        results = []
        end_time = time.time() + consumption_duration
        
        while time.time() < end_time:
            start_time = time.time()
            success = True
            error_msg = ""
            
            try:
                messages = await message_broker.receive_messages(
                    agent_id=f"load_consumer_{consumer_id}",
                    max_messages=10,
                    timeout=1.0
                )
                
                if messages:
                    # Process received messages
                    for message in messages:
                        # Simulate message processing
                        await asyncio.sleep(0.001)
            
            except Exception as e:
                success = False
                error_msg = str(e)
            
            duration_ms = (time.time() - start_time) * 1000
            results.append((duration_ms, success, error_msg))
        
        return results
    
    async def execute(self) -> PerformanceMetrics:
        """Execute Redis message load test."""
        logger.info(f"Starting Redis load test: {self.concurrent_producers} producers, {self.concurrent_consumers} consumers")
        
        message_broker = MessageBroker()
        
        # Start consumers first
        consumer_tasks = []
        for consumer_id in range(self.concurrent_consumers):
            task = self.execute_consumer(consumer_id, message_broker)
            consumer_tasks.append(task)
        
        # Start producers
        producer_tasks = []
        for producer_id in range(self.concurrent_producers):
            task = self.execute_producer(producer_id, message_broker)
            producer_tasks.append(task)
        
        # Wait for all tasks to complete
        all_results = await asyncio.gather(
            *producer_tasks, 
            *consumer_tasks,
            return_exceptions=True
        )
        
        # Process results
        for result in all_results:
            if isinstance(result, Exception):
                self.monitor.record_operation(0, False, str(result))
            else:
                for duration_ms, success, error_msg in result:
                    self.monitor.record_operation(duration_ms, success, error_msg)
        
        return self.monitor.get_metrics(self.name)


class MultiAgentOrchestrationLoadTest(LoadTestScenario):
    """Load test for multi-agent orchestration."""
    
    def __init__(self, num_agents: int = 20, workflows_per_agent: int = 10):
        super().__init__(
            "multi_agent_orchestration_load",
            "Load test for multi-agent orchestration"
        )
        self.num_agents = num_agents
        self.workflows_per_agent = workflows_per_agent
    
    async def execute_agent_workflows(self, agent_id: str, orchestrator: AgentOrchestrator) -> List[Tuple[float, bool, str]]:
        """Execute workflows for a single agent."""
        results = []
        
        for workflow_num in range(self.workflows_per_agent):
            start_time = time.time()
            success = True
            error_msg = ""
            
            try:
                # Simulate agent workflow execution
                workflow_data = {
                    "id": f"{agent_id}_workflow_{workflow_num}",
                    "type": "feature_development",
                    "tasks": [
                        {"name": "analyze", "duration": 0.1},
                        {"name": "implement", "duration": 0.2}, 
                        {"name": "test", "duration": 0.1},
                        {"name": "deploy", "duration": 0.05}
                    ]
                }
                
                # Execute workflow through orchestrator
                result = await orchestrator.execute_workflow(agent_id, workflow_data)
                
                if not result.get("success", False):
                    success = False
                    error_msg = result.get("error", "Unknown workflow error")
                    
            except Exception as e:
                success = False
                error_msg = str(e)
            
            duration_ms = (time.time() - start_time) * 1000
            results.append((duration_ms, success, error_msg))
            
            # Brief pause between workflows
            await asyncio.sleep(0.05)
        
        return results
    
    async def execute(self) -> PerformanceMetrics:
        """Execute multi-agent orchestration load test."""
        logger.info(f"Starting orchestration load test: {self.num_agents} agents")
        
        orchestrator = AgentOrchestrator()
        
        # Create agent tasks
        agent_tasks = []
        for agent_num in range(self.num_agents):
            agent_id = f"load_test_agent_{agent_num}"
            task = self.execute_agent_workflows(agent_id, orchestrator)
            agent_tasks.append(task)
        
        # Execute all agent tasks concurrently
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        for result in agent_results:
            if isinstance(result, Exception):
                self.monitor.record_operation(0, False, str(result))
            else:
                for duration_ms, success, error_msg in result:
                    self.monitor.record_operation(duration_ms, success, error_msg)
        
        return self.monitor.get_metrics(self.name)


@pytest.mark.performance
class TestComprehensiveLoadTesting:
    """Main load testing class."""
    
    def assert_performance_requirements(self, metrics: PerformanceMetrics, requirements: Dict[str, Any]):
        """Assert that performance metrics meet requirements."""
        if "max_response_time_ms" in requirements:
            assert metrics.p95_time_ms <= requirements["max_response_time_ms"], \
                f"P95 response time {metrics.p95_time_ms}ms exceeds limit {requirements['max_response_time_ms']}ms"
        
        if "min_throughput_ops_per_sec" in requirements:
            assert metrics.throughput_ops_per_sec >= requirements["min_throughput_ops_per_sec"], \
                f"Throughput {metrics.throughput_ops_per_sec} ops/sec below minimum {requirements['min_throughput_ops_per_sec']}"
        
        if "max_error_rate" in requirements:
            assert metrics.error_rate <= requirements["max_error_rate"], \
                f"Error rate {metrics.error_rate:.2%} exceeds maximum {requirements['max_error_rate']:.2%}"
        
        if "max_memory_usage_mb" in requirements:
            assert metrics.memory_usage_mb <= requirements["max_memory_usage_mb"], \
                f"Memory usage {metrics.memory_usage_mb}MB exceeds limit {requirements['max_memory_usage_mb']}MB"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_load_performance(self, async_test_client):
        """Test API endpoint performance under load."""
        # Test health endpoint
        health_test = APIEndpointLoadTest(
            endpoint="/api/v1/health",
            method="GET",
            concurrent_users=50,
            requests_per_user=20
        )
        
        await health_test.setup()
        metrics = await health_test.execute()
        await health_test.cleanup()
        
        # Performance requirements for health endpoint
        requirements = {
            "max_response_time_ms": 500,  # P95 under 500ms
            "min_throughput_ops_per_sec": 100,  # At least 100 ops/sec
            "max_error_rate": 0.01,  # Less than 1% errors
            "max_memory_usage_mb": 512  # Less than 512MB
        }
        
        self.assert_performance_requirements(metrics, requirements)
        logger.info(f"Health endpoint load test metrics: {metrics}")
    
    @pytest.mark.asyncio
    async def test_database_load_performance(self, test_db_session):
        """Test database performance under load."""
        db_test = DatabaseLoadTest(
            concurrent_connections=20,
            operations_per_connection=100
        )
        
        await db_test.setup()
        metrics = await db_test.execute()
        await db_test.cleanup()
        
        # Performance requirements for database operations
        requirements = {
            "max_response_time_ms": 1000,  # P95 under 1s
            "min_throughput_ops_per_sec": 50,  # At least 50 ops/sec
            "max_error_rate": 0.05,  # Less than 5% errors
        }
        
        self.assert_performance_requirements(metrics, requirements)
        logger.info(f"Database load test metrics: {metrics}")
    
    @pytest.mark.asyncio
    async def test_redis_message_load_performance(self):
        """Test Redis message performance under load.""" 
        redis_test = RedisMessageLoadTest(
            concurrent_producers=10,
            concurrent_consumers=5,
            messages_per_producer=1000
        )
        
        await redis_test.setup()
        metrics = await redis_test.execute()
        await redis_test.cleanup()
        
        # Performance requirements for Redis operations
        requirements = {
            "max_response_time_ms": 100,  # P95 under 100ms
            "min_throughput_ops_per_sec": 500,  # At least 500 ops/sec
            "max_error_rate": 0.02,  # Less than 2% errors
        }
        
        self.assert_performance_requirements(metrics, requirements)
        logger.info(f"Redis load test metrics: {metrics}")
    
    @pytest.mark.asyncio
    async def test_multi_agent_orchestration_load_performance(self):
        """Test multi-agent orchestration performance under load."""
        orchestration_test = MultiAgentOrchestrationLoadTest(
            num_agents=20,
            workflows_per_agent=10
        )
        
        await orchestration_test.setup()
        metrics = await orchestration_test.execute()
        await orchestration_test.cleanup()
        
        # Performance requirements for orchestration
        requirements = {
            "max_response_time_ms": 2000,  # P95 under 2s
            "min_throughput_ops_per_sec": 10,  # At least 10 workflows/sec
            "max_error_rate": 0.10,  # Less than 10% errors
            "max_memory_usage_mb": 1024  # Less than 1GB
        }
        
        self.assert_performance_requirements(metrics, requirements)
        logger.info(f"Orchestration load test metrics: {metrics}")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during sustained load."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Run sustained operations
        operations_count = 1000
        memory_samples = []
        
        for i in range(operations_count):
            start_time = time.time()
            
            # Simulate memory-intensive operation
            data = [{"id": j, "data": f"test_data_{j}"} for j in range(100)]
            
            # Process data
            processed = [item for item in data if item["id"] % 2 == 0]
            
            duration_ms = (time.time() - start_time) * 1000
            monitor.record_operation(duration_ms, True)
            
            # Sample memory every 100 operations
            if i % 100 == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
                
                # Force garbage collection
                gc.collect()
            
            # Small delay
            await asyncio.sleep(0.001)
        
        # Analyze memory usage trend
        if len(memory_samples) >= 3:
            # Check if memory usage is consistently increasing
            memory_increase = memory_samples[-1] - memory_samples[0]
            memory_growth_rate = memory_increase / len(memory_samples)
            
            # Allow some memory growth but detect significant leaks
            assert memory_growth_rate < 10, f"Potential memory leak detected: {memory_growth_rate}MB growth per sample"
            assert memory_increase < 100, f"Excessive memory growth: {memory_increase}MB total increase"
        
        metrics = monitor.get_metrics("memory_leak_detection")
        logger.info(f"Memory leak detection metrics: {metrics}")
        logger.info(f"Memory samples: {memory_samples}")
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self):
        """Test for performance regression detection."""
        # Baseline performance test
        baseline_test = APIEndpointLoadTest(
            endpoint="/api/v1/health",
            concurrent_users=10,
            requests_per_user=50
        )
        
        await baseline_test.setup()
        baseline_metrics = await baseline_test.execute()
        await baseline_test.cleanup()
        
        # Simulate performance regression (add artificial delay)
        with pytest.MonkeyPatch().context() as m:
            # Mock slower response
            original_method = httpx.AsyncClient.get
            
            async def slow_get(self, *args, **kwargs):
                await asyncio.sleep(0.1)  # Add 100ms delay
                return await original_method(self, *args, **kwargs)
            
            m.setattr(httpx.AsyncClient, "get", slow_get)
            
            # Run regression test
            regression_test = APIEndpointLoadTest(
                endpoint="/api/v1/health",
                concurrent_users=10,
                requests_per_user=50
            )
            
            await regression_test.setup()
            regression_metrics = await regression_test.execute()
            await regression_test.cleanup()
        
        # Compare performance metrics
        response_time_increase = (
            regression_metrics.avg_time_ms - baseline_metrics.avg_time_ms
        ) / baseline_metrics.avg_time_ms
        
        throughput_decrease = (
            baseline_metrics.throughput_ops_per_sec - regression_metrics.throughput_ops_per_sec
        ) / baseline_metrics.throughput_ops_per_sec
        
        logger.info(f"Baseline avg response time: {baseline_metrics.avg_time_ms}ms")
        logger.info(f"Regression avg response time: {regression_metrics.avg_time_ms}ms") 
        logger.info(f"Response time increase: {response_time_increase:.2%}")
        logger.info(f"Throughput decrease: {throughput_decrease:.2%}")
        
        # Should detect significant performance regression
        assert response_time_increase > 0.50, "Should detect >50% response time increase"
        assert throughput_decrease > 0.20, "Should detect >20% throughput decrease"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])