"""
Performance and Load Validation for LeanVibe Agent Hive 2.0

This module implements comprehensive performance testing under realistic production loads.
Tests validate system behavior under various load conditions and ensure performance
targets are met across all critical operations.

Performance Test Categories:
1. Concurrent Agent Operations (50-200 agents)
2. High-Frequency Message Processing (1000+ messages/second)
3. Real-time Dashboard Load (100+ concurrent users)
4. GitHub Integration Load (50+ simultaneous repositories)
5. Database Performance (10k+ operations/second)
6. Memory and Resource Efficiency
7. Long-running Stability Tests

Each test validates:
- Response time targets
- Throughput requirements
- Resource utilization limits
- Scalability characteristics
- Performance degradation patterns
- Memory leak detection
"""

import asyncio
import pytest
import time
import uuid
import json
import statistics
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import httpx
import websockets
import logging
import resource

# Test infrastructure
from tests.integration.comprehensive_integration_testing_strategy import (
    IntegrationTestOrchestrator,
    IntegrationTestEnvironment,
    PerformanceMetrics
)

# Core system components
from app.core.orchestrator import AgentOrchestrator
from app.core.coordination_dashboard import CoordinationDashboard
from app.core.redis import AgentMessageBroker, SessionCache
from app.core.github_api_client import GitHubAPIClient
from app.core.work_tree_manager import WorkTreeManager


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    test_name: str
    description: str
    duration_seconds: int
    concurrent_operations: int
    operations_per_second: int
    target_response_time_ms: float
    max_error_rate: float
    resource_limits: Dict[str, float]  # cpu_percent, memory_mb, etc.


@dataclass
class PerformanceTarget:
    """Performance target definitions."""
    operation_name: str
    max_response_time_ms: float
    min_throughput_ops_per_sec: float
    max_cpu_percent: float
    max_memory_mb: float
    max_error_rate: float


@dataclass
class LoadTestResult:
    """Results from a load test execution."""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    operations_per_second: float
    error_rate: float
    cpu_usage_stats: Dict[str, float]
    memory_usage_stats: Dict[str, float]
    resource_utilization: Dict[str, Any]
    performance_targets_met: bool
    detailed_metrics: Dict[str, Any]


class PerformanceMonitor:
    """
    Monitors system performance during load testing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
    
    async def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """Start performance monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics_history.clear()
        
        while self.monitoring:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        
        if not self.metrics_history:
            return {}
        
        # Aggregate metrics
        cpu_values = [m["cpu_percent"] for m in self.metrics_history]
        memory_values = [m["memory_mb"] for m in self.metrics_history]
        
        aggregated = {
            "duration_seconds": time.time() - (self.start_time or 0),
            "sample_count": len(self.metrics_history),
            "cpu_stats": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": statistics.mean(cpu_values),
                "p95": statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) > 20 else max(cpu_values)
            },
            "memory_stats": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": statistics.mean(memory_values),
                "p95": statistics.quantiles(memory_values, n=20)[18] if len(memory_values) > 20 else max(memory_values)
            },
            "raw_metrics": self.metrics_history
        }
        
        return aggregated
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        # Get process-specific metrics
        process = psutil.Process()
        
        # CPU and memory
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # System-wide metrics
        system_cpu = psutil.cpu_percent()
        system_memory = psutil.virtual_memory()
        
        # Network and disk I/O
        try:
            io_counters = process.io_counters()
            network_io = psutil.net_io_counters()
        except (psutil.AccessDenied, AttributeError):
            io_counters = None
            network_io = None
        
        # File descriptors and connections
        try:
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            connections = len(process.connections())
        except (psutil.AccessDenied, OSError):
            num_fds = 0
            connections = 0
        
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "memory_percent": memory_info.percent if hasattr(memory_info, 'percent') else 0,
            "system_cpu_percent": system_cpu,
            "system_memory_percent": system_memory.percent,
            "num_threads": process.num_threads(),
            "num_file_descriptors": num_fds,
            "num_connections": connections
        }
        
        if io_counters:
            metrics.update({
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count
            })
        
        if network_io:
            metrics.update({
                "network_bytes_sent": network_io.bytes_sent,
                "network_bytes_recv": network_io.bytes_recv,
                "network_packets_sent": network_io.packets_sent,
                "network_packets_recv": network_io.packets_recv
            })
        
        return metrics


class LoadTestExecutor:
    """
    Executes various types of load tests with performance monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor()
        
        # Define performance targets
        self.performance_targets = {
            "agent_spawn": PerformanceTarget(
                operation_name="agent_spawn",
                max_response_time_ms=5000.0,
                min_throughput_ops_per_sec=10.0,
                max_cpu_percent=80.0,
                max_memory_mb=2048.0,
                max_error_rate=0.05
            ),
            "message_publish": PerformanceTarget(
                operation_name="message_publish",
                max_response_time_ms=100.0,
                min_throughput_ops_per_sec=1000.0,
                max_cpu_percent=70.0,
                max_memory_mb=1024.0,
                max_error_rate=0.01
            ),
            "dashboard_request": PerformanceTarget(
                operation_name="dashboard_request",
                max_response_time_ms=2000.0,
                min_throughput_ops_per_sec=50.0,
                max_cpu_percent=60.0,
                max_memory_mb=512.0,
                max_error_rate=0.02
            ),
            "github_operation": PerformanceTarget(
                operation_name="github_operation",
                max_response_time_ms=10000.0,
                min_throughput_ops_per_sec=5.0,
                max_cpu_percent=50.0,
                max_memory_mb=1024.0,
                max_error_rate=0.10
            )
        }
    
    async def execute_load_test(
        self,
        config: LoadTestConfig,
        test_function: Callable,
        *args,
        **kwargs
    ) -> LoadTestResult:
        """Execute a load test with performance monitoring."""
        
        print(f"🚀 Starting load test: {config.test_name}")
        print(f"📊 Config: {config.concurrent_operations} concurrent ops, {config.duration_seconds}s duration")
        
        # Start performance monitoring
        monitor_task = asyncio.create_task(self.performance_monitor.start_monitoring())
        
        # Track operation metrics
        operation_results = []
        start_time = datetime.utcnow()
        
        try:
            # Execute the load test
            operation_results = await test_function(config, *args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Load test failed: {e}")
            raise
        finally:
            # Stop monitoring
            performance_metrics = await self.performance_monitor.stop_monitoring()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        end_time = datetime.utcnow()
        
        # Analyze results
        result = self._analyze_results(
            config, start_time, end_time, operation_results, performance_metrics
        )
        
        print(f"✅ Load test completed: {config.test_name}")
        print(f"📈 Results: {result.operations_per_second:.1f} ops/sec, {result.average_response_time_ms:.1f}ms avg")
        print(f"🎯 Targets met: {result.performance_targets_met}")
        
        return result
    
    def _analyze_results(
        self,
        config: LoadTestConfig,
        start_time: datetime,
        end_time: datetime,
        operation_results: List[Dict[str, Any]],
        performance_metrics: Dict[str, Any]
    ) -> LoadTestResult:
        """Analyze load test results and check against targets."""
        
        if not operation_results:
            return LoadTestResult(
                test_name=config.test_name,
                start_time=start_time,
                end_time=end_time,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                operations_per_second=0.0,
                error_rate=1.0,
                cpu_usage_stats={},
                memory_usage_stats={},
                resource_utilization={},
                performance_targets_met=False,
                detailed_metrics={}
            )
        
        # Calculate operation metrics
        successful_ops = [op for op in operation_results if op.get("success", False)]
        failed_ops = [op for op in operation_results if not op.get("success", False)]
        
        response_times = [op.get("response_time_ms", 0) for op in successful_ops]
        
        total_operations = len(operation_results)
        successful_operations = len(successful_ops)
        failed_operations = len(failed_ops)
        error_rate = failed_operations / total_operations if total_operations > 0 else 1.0
        
        duration_seconds = (end_time - start_time).total_seconds()
        operations_per_second = total_operations / duration_seconds if duration_seconds > 0 else 0
        
        # Response time statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)
            
            p95_response_time = sorted_times[min(p95_index, n-1)]
            p99_response_time = sorted_times[min(p99_index, n-1)]
        else:
            avg_response_time = 0.0
            min_response_time = 0.0
            max_response_time = 0.0
            p95_response_time = 0.0
            p99_response_time = 0.0
        
        # Check performance targets
        target_name = config.test_name.split("_")[0]  # Extract operation type
        target = self.performance_targets.get(target_name, self.performance_targets["agent_spawn"])
        
        targets_met = (
            avg_response_time <= target.max_response_time_ms and
            operations_per_second >= target.min_throughput_ops_per_sec and
            error_rate <= target.max_error_rate and
            performance_metrics.get("cpu_stats", {}).get("avg", 0) <= target.max_cpu_percent and
            performance_metrics.get("memory_stats", {}).get("avg", 0) <= target.max_memory_mb
        )
        
        return LoadTestResult(
            test_name=config.test_name,
            start_time=start_time,
            end_time=end_time,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            operations_per_second=operations_per_second,
            error_rate=error_rate,
            cpu_usage_stats=performance_metrics.get("cpu_stats", {}),
            memory_usage_stats=performance_metrics.get("memory_stats", {}),
            resource_utilization=performance_metrics,
            performance_targets_met=targets_met,
            detailed_metrics={
                "target": target.__dict__,
                "operation_results": operation_results[:100],  # Sample of results
                "performance_samples": performance_metrics.get("raw_metrics", [])[:60]  # Sample of metrics
            }
        )
    
    async def concurrent_agent_load_test(self, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Execute concurrent agent spawning load test."""
        orchestrator = AgentOrchestrator()
        operation_results = []
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(config.concurrent_operations)
        
        async def spawn_agent(agent_id: int) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    agent_config = {
                        "name": f"Load Test Agent {agent_id}",
                        "type": "CLAUDE",
                        "role": f"load_test_role_{agent_id % 10}",
                        "capabilities": ["python", "testing", "load_simulation"],
                        "context": {"load_test": True, "agent_id": agent_id}
                    }
                    
                    agent = await orchestrator.spawn_agent(agent_config)
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": agent is not None,
                        "response_time_ms": response_time,
                        "agent_id": agent,
                        "operation": "agent_spawn",
                        "timestamp": time.time()
                    }
                    
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    return {
                        "success": False,
                        "response_time_ms": response_time,
                        "error": str(e),
                        "operation": "agent_spawn",
                        "timestamp": time.time()
                    }
        
        # Calculate total operations based on duration and target rate
        total_agents = min(
            config.duration_seconds * config.operations_per_second,
            config.concurrent_operations * 10  # Reasonable upper limit
        )
        
        # Create tasks for agent spawning
        spawn_tasks = []
        for i in range(total_agents):
            task = spawn_agent(i)
            spawn_tasks.append(task)
            
            # Add delay to achieve target rate
            if config.operations_per_second > 0:
                delay = 1.0 / config.operations_per_second
                await asyncio.sleep(delay)
        
        # Wait for all agent spawns to complete
        operation_results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for result in operation_results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "response_time_ms": 0,
                    "error": str(result),
                    "operation": "agent_spawn",
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def message_throughput_load_test(self, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Execute high-frequency message processing load test."""
        message_broker = AgentMessageBroker()
        operation_results = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.concurrent_operations)
        
        async def publish_message(message_id: int) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    message = {
                        "type": "load_test_message",
                        "id": message_id,
                        "timestamp": time.time(),
                        "data": {
                            "payload": f"Load test message {message_id}",
                            "size": "medium",
                            "priority": "normal"
                        },
                        "routing_key": f"load_test_{message_id % 100}"
                    }
                    
                    channel = f"load_test_channel_{message_id % 10}"
                    message_result = await message_broker.publish_agent_message(channel, message)
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": message_result is not None,
                        "response_time_ms": response_time,
                        "message_id": message_result,
                        "operation": "message_publish",
                        "timestamp": time.time()
                    }
                    
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    return {
                        "success": False,
                        "response_time_ms": response_time,
                        "error": str(e),
                        "operation": "message_publish",
                        "timestamp": time.time()
                    }
        
        # Calculate message count based on duration and target rate
        total_messages = config.duration_seconds * config.operations_per_second
        
        # Publish messages at target rate
        publish_tasks = []
        start_time = time.time()
        
        for i in range(total_messages):
            task = publish_message(i)
            publish_tasks.append(task)
            
            # Rate limiting
            if config.operations_per_second > 0:
                expected_time = start_time + (i / config.operations_per_second)
                current_time = time.time()
                sleep_time = max(0, expected_time - current_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Wait for all messages to be published
        operation_results = await asyncio.gather(*publish_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in operation_results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "response_time_ms": 0,
                    "error": str(result),
                    "operation": "message_publish",
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def dashboard_concurrent_users_test(self, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Execute concurrent dashboard users load test."""
        operation_results = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(config.concurrent_operations)
        
        async def simulate_dashboard_user(user_id: int) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    # Simulate dashboard API requests
                    async with httpx.AsyncClient() as client:
                        # Get dashboard data
                        response = await client.get(
                            "http://localhost:8001/dashboard/api/live-data",
                            timeout=10.0
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        return {
                            "success": response.status_code == 200,
                            "response_time_ms": response_time,
                            "status_code": response.status_code,
                            "operation": "dashboard_request",
                            "timestamp": time.time(),
                            "user_id": user_id
                        }
                        
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    return {
                        "success": False,
                        "response_time_ms": response_time,
                        "error": str(e),
                        "operation": "dashboard_request",
                        "timestamp": time.time(),
                        "user_id": user_id
                    }
        
        # Simulate concurrent users over the test duration
        total_requests = config.duration_seconds * config.operations_per_second
        
        request_tasks = []
        start_time = time.time()
        
        for i in range(total_requests):
            user_id = i % config.concurrent_operations  # Rotate users
            task = simulate_dashboard_user(user_id)
            request_tasks.append(task)
            
            # Rate limiting
            if config.operations_per_second > 0:
                expected_time = start_time + (i / config.operations_per_second)
                current_time = time.time()
                sleep_time = max(0, expected_time - current_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Wait for all requests to complete
        operation_results = await asyncio.gather(*request_tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in operation_results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "response_time_ms": 0,
                    "error": str(result),
                    "operation": "dashboard_request",
                    "timestamp": time.time()
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def memory_leak_detection_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Execute long-running test to detect memory leaks."""
        
        print(f"🔍 Starting memory leak detection test for {config.duration_seconds}s")
        
        # Initial memory snapshot
        gc.collect()  # Force garbage collection
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_snapshots = []
        operation_count = 0
        
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        orchestrator = AgentOrchestrator()
        message_broker = AgentMessageBroker()
        
        while time.time() < end_time:
            cycle_start = time.time()
            
            # Perform various operations
            try:
                # Spawn and cleanup agents
                agent_configs = [
                    {
                        "name": f"Memory Test Agent {i}",
                        "type": "CLAUDE",
                        "role": "memory_test"
                    }
                    for i in range(5)
                ]
                
                agents = []
                for config_item in agent_configs:
                    agent = await orchestrator.spawn_agent(config_item)
                    if agent:
                        agents.append(agent)
                
                # Publish messages
                for i in range(10):
                    message = {
                        "type": "memory_test",
                        "data": "x" * 1000,  # 1KB message
                        "timestamp": time.time()
                    }
                    await message_broker.publish_agent_message("memory_test", message)
                
                operation_count += len(agents) + 10
                
                # Memory snapshot
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_snapshots.append({
                    "timestamp": time.time(),
                    "memory_mb": current_memory,
                    "operations_completed": operation_count
                })
                
                # Clean up (simulate proper resource cleanup)
                agents.clear()
                
            except Exception as e:
                self.logger.warning(f"Error in memory test cycle: {e}")
            
            # Maintain target rate
            cycle_duration = time.time() - cycle_start
            target_cycle_time = 1.0  # 1 second per cycle
            if cycle_duration < target_cycle_time:
                await asyncio.sleep(target_cycle_time - cycle_duration)
        
        # Final memory snapshot
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze memory usage trend
        memory_growth = final_memory - initial_memory
        memory_growth_rate = memory_growth / config.duration_seconds  # MB per second
        
        # Check for memory leaks (arbitrary threshold: >1MB/minute growth)
        leak_threshold = 1.0 / 60.0  # 1MB per minute = ~0.017 MB/second
        potential_leak = memory_growth_rate > leak_threshold
        
        # Calculate memory efficiency
        operations_per_mb = operation_count / max(memory_growth, 1)
        
        memory_analysis = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": memory_growth,
            "memory_growth_rate_mb_per_sec": memory_growth_rate,
            "potential_memory_leak": potential_leak,
            "total_operations": operation_count,
            "operations_per_mb_growth": operations_per_mb,
            "memory_snapshots": memory_snapshots,
            "test_duration_seconds": config.duration_seconds
        }
        
        print(f"📊 Memory analysis complete:")
        print(f"   Initial: {initial_memory:.1f}MB")
        print(f"   Final: {final_memory:.1f}MB")
        print(f"   Growth: {memory_growth:.1f}MB ({memory_growth_rate:.3f}MB/s)")
        print(f"   Potential leak: {potential_leak}")
        
        return memory_analysis


@pytest.mark.asyncio
class TestPerformanceLoadValidation:
    """
    Comprehensive performance and load validation test suite.
    """
    
    @pytest.fixture
    async def performance_test_environment(self, integration_orchestrator) -> str:
        """Setup environment optimized for performance testing."""
        env_config = IntegrationTestEnvironment(
            name="performance_testing",
            services=["postgres", "redis", "api", "frontend"],
            monitoring_enabled=True,
            resource_limits={
                "postgres": {"memory": "1G", "cpus": "1.0"},
                "redis": {"memory": "512M", "cpus": "0.5"},
                "api": {"memory": "2G", "cpus": "2.0"},
                "frontend": {"memory": "512M", "cpus": "0.5"}
            }
        )
        
        env_id = await integration_orchestrator.setup_test_environment(env_config)
        yield env_id
        await integration_orchestrator.cleanup_environment(env_id)
    
    @pytest.fixture
    def load_test_executor(self) -> LoadTestExecutor:
        """Create load test executor instance."""
        return LoadTestExecutor()
    
    async def test_concurrent_agent_operations_performance(
        self,
        performance_test_environment: str,
        load_test_executor: LoadTestExecutor
    ):
        """Test performance under concurrent agent operations."""
        
        # Test Configuration
        config = LoadTestConfig(
            test_name="agent_spawn_concurrent_50",
            description="50 concurrent agent spawning operations",
            duration_seconds=60,
            concurrent_operations=50,
            operations_per_second=10,  # 10 agents per second
            target_response_time_ms=5000.0,
            max_error_rate=0.05,
            resource_limits={"cpu_percent": 80.0, "memory_mb": 2048.0}
        )
        
        result = await load_test_executor.execute_load_test(
            config,
            load_test_executor.concurrent_agent_load_test
        )
        
        # Validate performance targets
        assert result.performance_targets_met, f"Performance targets not met for {config.test_name}"
        assert result.error_rate <= config.max_error_rate, f"Error rate {result.error_rate:.2%} exceeds limit {config.max_error_rate:.2%}"
        assert result.average_response_time_ms <= config.target_response_time_ms, \
            f"Average response time {result.average_response_time_ms:.1f}ms exceeds target {config.target_response_time_ms}ms"
        
        # Validate throughput
        assert result.operations_per_second >= 8.0, \
            f"Throughput {result.operations_per_second:.1f} ops/sec below minimum 8.0"
        
        # Validate resource utilization
        max_cpu = result.cpu_usage_stats.get("max", 0)
        max_memory = result.memory_usage_stats.get("max", 0)
        
        assert max_cpu <= config.resource_limits["cpu_percent"], \
            f"CPU usage {max_cpu:.1f}% exceeds limit {config.resource_limits['cpu_percent']}%"
        assert max_memory <= config.resource_limits["memory_mb"], \
            f"Memory usage {max_memory:.1f}MB exceeds limit {config.resource_limits['memory_mb']}MB"
        
        print(f"✅ Concurrent agent operations performance test passed")
        print(f"📊 Results: {result.operations_per_second:.1f} ops/sec, {result.error_rate:.2%} error rate")
        print(f"⚡ Response times: avg={result.average_response_time_ms:.1f}ms, p95={result.p95_response_time_ms:.1f}ms")
    
    async def test_high_frequency_message_processing(
        self,
        performance_test_environment: str,
        load_test_executor: LoadTestExecutor
    ):
        """Test high-frequency message processing performance."""
        
        config = LoadTestConfig(
            test_name="message_publish_high_frequency",
            description="High-frequency message publishing (1000 msg/sec)",
            duration_seconds=30,
            concurrent_operations=100,
            operations_per_second=1000,
            target_response_time_ms=100.0,
            max_error_rate=0.01,
            resource_limits={"cpu_percent": 70.0, "memory_mb": 1024.0}
        )
        
        result = await load_test_executor.execute_load_test(
            config,
            load_test_executor.message_throughput_load_test
        )
        
        # Validate performance targets
        assert result.performance_targets_met, f"Performance targets not met for {config.test_name}"
        assert result.error_rate <= config.max_error_rate, \
            f"Error rate {result.error_rate:.3%} exceeds limit {config.max_error_rate:.3%}"
        
        # High-frequency operations should have low latency
        assert result.p95_response_time_ms <= 200.0, \
            f"P95 response time {result.p95_response_time_ms:.1f}ms too high for message publishing"
        
        # Should achieve high throughput
        assert result.operations_per_second >= 800.0, \
            f"Throughput {result.operations_per_second:.1f} ops/sec below minimum 800.0"
        
        # Messages should be processed consistently
        response_time_variance = result.max_response_time_ms - result.min_response_time_ms
        assert response_time_variance <= 1000.0, \
            f"Response time variance {response_time_variance:.1f}ms too high"
        
        print(f"✅ High-frequency message processing test passed")
        print(f"🚀 Achieved: {result.operations_per_second:.1f} messages/sec")
        print(f"⚡ Latency: avg={result.average_response_time_ms:.1f}ms, p95={result.p95_response_time_ms:.1f}ms")
    
    async def test_dashboard_concurrent_users_load(
        self,
        performance_test_environment: str,
        load_test_executor: LoadTestExecutor
    ):
        """Test dashboard performance under concurrent user load."""
        
        config = LoadTestConfig(
            test_name="dashboard_request_concurrent_users",
            description="Dashboard requests from 100 concurrent users",
            duration_seconds=45,
            concurrent_operations=100,
            operations_per_second=50,  # 50 requests per second
            target_response_time_ms=2000.0,
            max_error_rate=0.02,
            resource_limits={"cpu_percent": 60.0, "memory_mb": 512.0}
        )
        
        result = await load_test_executor.execute_load_test(
            config,
            load_test_executor.dashboard_concurrent_users_test
        )
        
        # Validate performance targets
        assert result.performance_targets_met, f"Performance targets not met for {config.test_name}"
        assert result.error_rate <= config.max_error_rate, \
            f"Error rate {result.error_rate:.2%} exceeds limit {config.max_error_rate:.2%}"
        
        # Dashboard should respond quickly even under load
        assert result.p95_response_time_ms <= 3000.0, \
            f"P95 response time {result.p95_response_time_ms:.1f}ms too slow for dashboard"
        
        # Should handle concurrent users efficiently
        assert result.operations_per_second >= 40.0, \
            f"Throughput {result.operations_per_second:.1f} req/sec below minimum 40.0"
        
        print(f"✅ Dashboard concurrent users test passed")
        print(f"👥 Handled: {result.operations_per_second:.1f} requests/sec from concurrent users")
        print(f"⚡ Response times: avg={result.average_response_time_ms:.1f}ms, p95={result.p95_response_time_ms:.1f}ms")
    
    async def test_memory_leak_detection(
        self,
        performance_test_environment: str,
        load_test_executor: LoadTestExecutor
    ):
        """Test for memory leaks during extended operation."""
        
        config = LoadTestConfig(
            test_name="memory_leak_detection",
            description="Long-running test to detect memory leaks",
            duration_seconds=300,  # 5 minutes
            concurrent_operations=10,
            operations_per_second=1,
            target_response_time_ms=1000.0,
            max_error_rate=0.05,
            resource_limits={"memory_growth_mb": 50.0}  # Max 50MB growth
        )
        
        memory_analysis = await load_test_executor.memory_leak_detection_test(config)
        
        # Validate memory usage
        assert not memory_analysis["potential_memory_leak"], \
            f"Potential memory leak detected: {memory_analysis['memory_growth_rate_mb_per_sec']:.3f} MB/sec growth"
        
        # Memory growth should be reasonable
        max_acceptable_growth = 50.0  # 50MB for 5-minute test
        assert memory_analysis["memory_growth_mb"] <= max_acceptable_growth, \
            f"Memory growth {memory_analysis['memory_growth_mb']:.1f}MB exceeds limit {max_acceptable_growth}MB"
        
        # Operations should be memory-efficient
        min_ops_per_mb = 100  # At least 100 operations per MB of growth
        assert memory_analysis["operations_per_mb_growth"] >= min_ops_per_mb, \
            f"Memory efficiency {memory_analysis['operations_per_mb_growth']:.1f} ops/MB below minimum {min_ops_per_mb}"
        
        print(f"✅ Memory leak detection test passed")
        print(f"📊 Memory growth: {memory_analysis['memory_growth_mb']:.1f}MB over {config.duration_seconds}s")
        print(f"🔍 Growth rate: {memory_analysis['memory_growth_rate_mb_per_sec']:.3f}MB/sec")
        print(f"⚡ Efficiency: {memory_analysis['operations_per_mb_growth']:.1f} ops/MB")
    
    async def test_system_stability_under_load(
        self,
        performance_test_environment: str,
        load_test_executor: LoadTestExecutor
    ):
        """Test system stability under sustained load."""
        
        # Run multiple load tests in sequence to test stability
        test_configs = [
            LoadTestConfig(
                test_name="stability_test_phase_1",
                description="Initial load phase",
                duration_seconds=60,
                concurrent_operations=25,
                operations_per_second=5,
                target_response_time_ms=3000.0,
                max_error_rate=0.02,
                resource_limits={"cpu_percent": 70.0}
            ),
            LoadTestConfig(
                test_name="stability_test_phase_2",
                description="Increased load phase",
                duration_seconds=90,
                concurrent_operations=50,
                operations_per_second=10,
                target_response_time_ms=4000.0,
                max_error_rate=0.03,
                resource_limits={"cpu_percent": 80.0}
            ),
            LoadTestConfig(
                test_name="stability_test_phase_3",
                description="Peak load phase",
                duration_seconds=60,
                concurrent_operations=75,
                operations_per_second=15,
                target_response_time_ms=5000.0,
                max_error_rate=0.05,
                resource_limits={"cpu_percent": 85.0}
            )
        ]
        
        stability_results = []
        
        for config in test_configs:
            print(f"🔄 Starting stability test phase: {config.description}")
            
            result = await load_test_executor.execute_load_test(
                config,
                load_test_executor.concurrent_agent_load_test
            )
            
            stability_results.append(result)
            
            # Validate each phase
            assert result.performance_targets_met, \
                f"Stability test failed in phase: {config.test_name}"
            
            # Brief recovery period between phases
            await asyncio.sleep(10)
        
        # Analyze stability across phases
        error_rates = [result.error_rate for result in stability_results]
        response_times = [result.average_response_time_ms for result in stability_results]
        throughputs = [result.operations_per_second for result in stability_results]
        
        # Error rates should not increase dramatically
        max_error_rate_increase = max(error_rates) - min(error_rates)
        assert max_error_rate_increase <= 0.03, \
            f"Error rate increased by {max_error_rate_increase:.3f} during load progression"
        
        # Response times should remain reasonable
        max_response_time_degradation = max(response_times) - min(response_times)
        assert max_response_time_degradation <= 2000.0, \
            f"Response time degraded by {max_response_time_degradation:.1f}ms during load progression"
        
        # Throughput should scale reasonably
        throughput_ratio = max(throughputs) / min(throughputs)
        assert throughput_ratio >= 1.5, \
            f"Throughput only scaled by {throughput_ratio:.1f}x, expected at least 1.5x"
        
        print(f"✅ System stability under load test passed")
        print(f"📊 Phases completed: {len(stability_results)}")
        print(f"🔄 Error rate range: {min(error_rates):.2%} - {max(error_rates):.2%}")
        print(f"⚡ Response time range: {min(response_times):.1f} - {max(response_times):.1f}ms")
        print(f"🚀 Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} ops/sec")


if __name__ == "__main__":
    # Run performance and load validation tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-k", "test_performance_load_validation"
    ])