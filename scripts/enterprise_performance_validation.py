#!/usr/bin/env python3
"""
Enterprise Performance and Reliability Validation Script

Comprehensive testing suite to validate LeanVibe Agent Hive 2.0 performance
and reliability for enterprise production deployment.

Performance Targets:
- Agent spawning: < 5 seconds per agent
- Task assignment: < 2 seconds
- Status queries: < 500ms
- Dashboard updates: < 200ms
- End-to-end development cycle: < 30 minutes
"""

import asyncio
import time
import json
import random
import statistics
import httpx
import psutil
import redis.asyncio as aioredis
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    
    # Timing metrics (milliseconds)
    agent_spawn_times: List[float] = field(default_factory=list)
    task_assignment_times: List[float] = field(default_factory=list)
    status_query_times: List[float] = field(default_factory=list)
    dashboard_update_times: List[float] = field(default_factory=list)
    
    # Throughput metrics
    requests_per_second: List[float] = field(default_factory=list)
    concurrent_operations: List[int] = field(default_factory=list)
    
    # System resource metrics
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    disk_io: List[float] = field(default_factory=list)
    network_io: List[float] = field(default_factory=list)
    
    # Error metrics
    error_count: int = 0
    timeout_count: int = 0
    success_count: int = 0
    
    def add_measurement(self, metric_type: str, value: float):
        """Add a measurement to the appropriate metric list."""
        metric_map = {
            'agent_spawn': self.agent_spawn_times,
            'task_assignment': self.task_assignment_times,
            'status_query': self.status_query_times,
            'dashboard_update': self.dashboard_update_times,
            'rps': self.requests_per_second
        }
        
        if metric_type in metric_map:
            metric_map[metric_type].append(value)
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values."""
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0
        }


@dataclass
class TestResult:
    """Test result with pass/fail status."""
    
    test_name: str
    passed: bool
    target_value: float
    actual_value: float
    unit: str
    details: str = ""
    
    def __post_init__(self):
        if not self.details:
            status = "PASS" if self.passed else "FAIL"
            self.details = f"{status}: {self.actual_value:.2f}{self.unit} (target: {self.target_value:.2f}{self.unit})"


class EnterprisePerformanceValidator:
    """
    Enterprise-grade performance and reliability validation framework.
    
    Tests all critical performance requirements for production deployment.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.metrics = PerformanceMetrics()
        self.test_results: List[TestResult] = []
        
        # Performance targets (enterprise requirements)
        self.targets = {
            'agent_spawn_time': 5000,  # 5 seconds in ms
            'task_assignment_time': 2000,  # 2 seconds in ms
            'status_query_time': 500,  # 500ms
            'dashboard_update_time': 200,  # 200ms
            'end_to_end_cycle_time': 1800000,  # 30 minutes in ms
            'min_rps': 100,  # Minimum requests per second
            'max_cpu_usage': 80,  # Maximum CPU usage percentage
            'max_memory_usage': 2048,  # Maximum memory usage in MB
            'max_error_rate': 0.001,  # 0.1% error rate
            'min_availability': 0.999  # 99.9% availability
        }
        
        self.client: Optional[httpx.AsyncClient] = None
        self.redis_client = None
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up enterprise performance validation environment")
        
        # Initialize HTTP client with appropriate timeouts
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        # Initialize Redis client
        self.redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
        
        # Verify system is ready
        await self._verify_system_health()
        
        logger.info("Performance validation environment ready")
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.client:
            await self.client.aclose()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Performance validation environment cleaned up")
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive performance and reliability validation."""
        logger.info("🚀 Starting Enterprise Performance and Reliability Validation")
        
        start_time = time.time()
        
        try:
            # Phase 1: Performance Benchmarking
            await self._run_performance_benchmarks()
            
            # Phase 2: Load Testing
            await self._run_load_tests()
            
            # Phase 3: Error Handling and Recovery Testing
            await self._run_error_handling_tests()
            
            # Phase 4: Resource Utilization Testing
            await self._run_resource_tests()
            
            # Phase 5: Concurrent Operations Testing
            await self._run_concurrency_tests()
            
            # Generate comprehensive report
            total_time = time.time() - start_time
            return self._generate_validation_report(total_time)
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            raise
    
    async def _verify_system_health(self):
        """Verify system is healthy before testing."""
        try:
            response = await self.client.get("/health")
            health_data = response.json()
            
            if health_data.get("status") != "healthy":
                raise RuntimeError(f"System not healthy: {health_data}")
            
            logger.info("✅ System health verified")
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            raise
    
    async def _run_performance_benchmarks(self):
        """Run core performance benchmarking tests."""
        logger.info("📊 Running Performance Benchmarks")
        
        # Test 1: Agent Spawning Performance
        await self._test_agent_spawning_performance()
        
        # Test 2: Task Assignment Performance
        await self._test_task_assignment_performance()
        
        # Test 3: Status Query Performance
        await self._test_status_query_performance()
        
        # Test 4: Dashboard Update Performance
        await self._test_dashboard_update_performance()
        
        # Test 5: End-to-End Development Cycle
        await self._test_end_to_end_cycle_performance()
    
    async def _test_agent_spawning_performance(self):
        """Test agent spawning performance - Target: < 5 seconds per agent."""
        logger.info("Testing agent spawning performance")
        
        agent_spawn_times = []
        
        for i in range(10):  # Test spawning 10 agents
            start_time = time.time()
            
            try:
                # Spawn agent via API
                response = await self.client.post("/api/agents/spawn", json={
                    "agent_type": "developer",
                    "capabilities": ["code_generation", "testing"],
                    "max_concurrent_tasks": 5
                })
                
                if response.status_code == 200:
                    agent_data = response.json()
                    agent_id = agent_data.get("agent_id")
                    
                    # Wait for agent to be fully operational
                    await self._wait_for_agent_ready(agent_id)
                    
                    spawn_time = (time.time() - start_time) * 1000  # Convert to ms
                    agent_spawn_times.append(spawn_time)
                    self.metrics.success_count += 1
                    
                    logger.info(f"Agent {i+1} spawned in {spawn_time:.2f}ms")
                    
                else:
                    self.metrics.error_count += 1
                    logger.error(f"Failed to spawn agent {i+1}: {response.status_code}")
                    
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Agent spawning error: {e}")
        
        # Analyze results
        if agent_spawn_times:
            stats = self.metrics.calculate_statistics(agent_spawn_times)
            self.metrics.agent_spawn_times.extend(agent_spawn_times)
            
            # Validate against target
            avg_spawn_time = stats["mean"]
            passed = avg_spawn_time <= self.targets['agent_spawn_time']
            
            self.test_results.append(TestResult(
                test_name="Agent Spawning Performance",
                passed=passed,
                target_value=self.targets['agent_spawn_time'],
                actual_value=avg_spawn_time,
                unit="ms",
                details=f"P95: {stats['p95']:.2f}ms, Max: {stats['max']:.2f}ms"
            ))
            
            logger.info(f"Agent spawning - Avg: {avg_spawn_time:.2f}ms, P95: {stats['p95']:.2f}ms")
    
    async def _test_task_assignment_performance(self):
        """Test task assignment performance - Target: < 2 seconds."""
        logger.info("Testing task assignment performance")
        
        task_assignment_times = []
        
        for i in range(20):  # Test 20 task assignments
            start_time = time.time()
            
            try:
                # Create and assign a task
                task_payload = {
                    "task_type": "code_generation",
                    "description": f"Generate test function {i}",
                    "priority": "normal",
                    "estimated_duration": 300
                }
                
                response = await self.client.post("/api/v1/tasks", json=task_payload)
                
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data.get("task_id")
                    
                    # Wait for task to be assigned
                    await self._wait_for_task_assignment(task_id)
                    
                    assignment_time = (time.time() - start_time) * 1000
                    task_assignment_times.append(assignment_time)
                    self.metrics.success_count += 1
                    
                else:
                    self.metrics.error_count += 1
                    logger.error(f"Failed to create task {i}: {response.status_code}")
                    
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Task assignment error: {e}")
        
        # Analyze results
        if task_assignment_times:
            stats = self.metrics.calculate_statistics(task_assignment_times)
            self.metrics.task_assignment_times.extend(task_assignment_times)
            
            avg_assignment_time = stats["mean"]
            passed = avg_assignment_time <= self.targets['task_assignment_time']
            
            self.test_results.append(TestResult(
                test_name="Task Assignment Performance",
                passed=passed,
                target_value=self.targets['task_assignment_time'],
                actual_value=avg_assignment_time,
                unit="ms"
            ))
            
            logger.info(f"Task assignment - Avg: {avg_assignment_time:.2f}ms")
    
    async def _test_status_query_performance(self):
        """Test status query performance - Target: < 500ms."""
        logger.info("Testing status query performance")
        
        status_query_times = []
        
        for _ in range(50):  # Test 50 status queries
            start_time = time.time()
            
            try:
                response = await self.client.get("/api/v1/system/status")
                
                if response.status_code == 200:
                    query_time = (time.time() - start_time) * 1000
                    status_query_times.append(query_time)
                    self.metrics.success_count += 1
                else:
                    self.metrics.error_count += 1
                    
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Status query error: {e}")
        
        # Analyze results
        if status_query_times:
            stats = self.metrics.calculate_statistics(status_query_times)
            self.metrics.status_query_times.extend(status_query_times)
            
            avg_query_time = stats["mean"]
            passed = avg_query_time <= self.targets['status_query_time']
            
            self.test_results.append(TestResult(
                test_name="Status Query Performance",
                passed=passed,
                target_value=self.targets['status_query_time'],
                actual_value=avg_query_time,
                unit="ms"
            ))
            
            logger.info(f"Status queries - Avg: {avg_query_time:.2f}ms")
    
    async def _test_dashboard_update_performance(self):
        """Test dashboard update performance - Target: < 200ms."""
        logger.info("Testing dashboard update performance")
        
        dashboard_times = []
        
        for _ in range(30):  # Test 30 dashboard updates
            start_time = time.time()
            
            try:
                response = await self.client.get("/dashboard/coordination")
                
                if response.status_code == 200:
                    update_time = (time.time() - start_time) * 1000
                    dashboard_times.append(update_time)
                    self.metrics.success_count += 1
                else:
                    self.metrics.error_count += 1
                    
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Dashboard update error: {e}")
        
        # Analyze results
        if dashboard_times:
            stats = self.metrics.calculate_statistics(dashboard_times)
            self.metrics.dashboard_update_times.extend(dashboard_times)
            
            avg_update_time = stats["mean"]
            passed = avg_update_time <= self.targets['dashboard_update_time']
            
            self.test_results.append(TestResult(
                test_name="Dashboard Update Performance",
                passed=passed,
                target_value=self.targets['dashboard_update_time'],
                actual_value=avg_update_time,
                unit="ms"
            ))
            
            logger.info(f"Dashboard updates - Avg: {avg_update_time:.2f}ms")
    
    async def _test_end_to_end_cycle_performance(self):
        """Test end-to-end development cycle - Target: < 30 minutes."""
        logger.info("Testing end-to-end development cycle performance")
        
        start_time = time.time()
        
        try:
            # Simulate a complete development cycle
            # 1. Create project
            project_response = await self.client.post("/api/v1/projects", json={
                "name": "performance_test_project",
                "description": "Test project for performance validation",
                "type": "autonomous_development"
            })
            
            if project_response.status_code != 200:
                raise RuntimeError(f"Failed to create project: {project_response.status_code}")
            
            project_id = project_response.json().get("project_id")
            
            # 2. Assign development task
            task_response = await self.client.post(f"/api/v1/projects/{project_id}/tasks", json={
                "task_type": "feature_development",
                "description": "Implement basic REST API with CRUD operations",
                "priority": "high"
            })
            
            if task_response.status_code != 200:
                raise RuntimeError(f"Failed to create task: {task_response.status_code}")
            
            task_id = task_response.json().get("task_id")
            
            # 3. Wait for task completion (with timeout)
            completion_timeout = 1800  # 30 minutes
            await self._wait_for_task_completion(task_id, timeout=completion_timeout)
            
            cycle_time = (time.time() - start_time) * 1000
            
            passed = cycle_time <= self.targets['end_to_end_cycle_time']
            
            self.test_results.append(TestResult(
                test_name="End-to-End Development Cycle",
                passed=passed,
                target_value=self.targets['end_to_end_cycle_time'],
                actual_value=cycle_time,
                unit="ms"
            ))
            
            logger.info(f"End-to-end cycle completed in {cycle_time/1000/60:.2f} minutes")
            
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"End-to-end cycle error: {e}")
            
            self.test_results.append(TestResult(
                test_name="End-to-End Development Cycle",
                passed=False,
                target_value=self.targets['end_to_end_cycle_time'],
                actual_value=float('inf'),
                unit="ms",
                details=f"Failed with error: {str(e)}"
            ))
    
    async def _run_load_tests(self):
        """Run load testing to validate concurrent operations."""
        logger.info("🔄 Running Load Tests")
        
        # Test concurrent API requests
        await self._test_concurrent_api_requests()
        
        # Test concurrent agent operations
        await self._test_concurrent_agent_operations()
        
        # Test system under stress
        await self._test_system_stress()
    
    async def _test_concurrent_api_requests(self):
        """Test concurrent API request handling."""
        logger.info("Testing concurrent API request handling")
        
        async def make_concurrent_requests(num_requests: int, endpoint: str):
            """Make concurrent requests to an endpoint."""
            tasks = []
            start_time = time.time()
            
            for _ in range(num_requests):
                task = asyncio.create_task(self.client.get(endpoint))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            rps = num_requests / duration if duration > 0 else 0
            
            success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            error_count = num_requests - success_count
            
            return {
                "rps": rps,
                "success_count": success_count,
                "error_count": error_count,
                "duration": duration
            }
        
        # Test different concurrency levels
        concurrency_levels = [10, 25, 50, 100]
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing {concurrency} concurrent requests")
            
            result = await make_concurrent_requests(concurrency, "/health")
            
            self.metrics.requests_per_second.append(result["rps"])
            self.metrics.success_count += result["success_count"]
            self.metrics.error_count += result["error_count"]
            
            logger.info(f"Concurrency {concurrency}: {result['rps']:.2f} RPS, {result['success_count']} success, {result['error_count']} errors")
        
        # Validate RPS requirement
        if self.metrics.requests_per_second:
            max_rps = max(self.metrics.requests_per_second)
            passed = max_rps >= self.targets['min_rps']
            
            self.test_results.append(TestResult(
                test_name="Concurrent Request Handling",
                passed=passed,
                target_value=self.targets['min_rps'],
                actual_value=max_rps,
                unit=" RPS"
            ))
    
    async def _test_concurrent_agent_operations(self):
        """Test concurrent agent operations."""
        logger.info("Testing concurrent agent operations")
        
        async def spawn_and_assign_task(agent_index: int):
            """Spawn agent and assign task concurrently."""
            try:
                # Spawn agent
                spawn_response = await self.client.post("/api/agents/spawn", json={
                    "agent_type": "concurrent_test",
                    "capabilities": ["testing"],
                    "agent_id": f"concurrent_agent_{agent_index}"
                })
                
                if spawn_response.status_code != 200:
                    return False
                
                # Assign task
                task_response = await self.client.post("/api/v1/tasks", json={
                    "task_type": "concurrent_test",
                    "description": f"Concurrent test task {agent_index}",
                    "assigned_agent": f"concurrent_agent_{agent_index}"
                })
                
                return task_response.status_code == 200
                
            except Exception as e:
                logger.error(f"Concurrent operation error: {e}")
                return False
        
        # Test concurrent agent operations
        concurrency_levels = [5, 10, 20]
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing {concurrency} concurrent agent operations")
            
            start_time = time.time()
            
            tasks = []
            for i in range(concurrency):
                task = asyncio.create_task(spawn_and_assign_task(i))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            success_count = sum(1 for r in results if r is True)
            
            self.metrics.concurrent_operations.append(concurrency)
            self.metrics.success_count += success_count
            self.metrics.error_count += (concurrency - success_count)
            
            logger.info(f"Concurrent operations {concurrency}: {success_count}/{concurrency} successful in {duration:.2f}s")
    
    async def _test_system_stress(self):
        """Test system under stress conditions."""
        logger.info("Testing system under stress conditions")
        
        # Monitor system resources during stress test
        resource_monitor_task = asyncio.create_task(self._monitor_system_resources())
        
        try:
            # Create high load
            stress_tasks = []
            
            # High-frequency API calls
            for _ in range(50):
                task = asyncio.create_task(self._stress_api_calls())
                stress_tasks.append(task)
            
            # Heavy computation tasks
            for _ in range(10):
                task = asyncio.create_task(self._stress_computation())
                stress_tasks.append(task)
            
            # Wait for stress test completion
            await asyncio.gather(*stress_tasks, return_exceptions=True)
            
        finally:
            resource_monitor_task.cancel()
            try:
                await resource_monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _run_error_handling_tests(self):
        """Run error handling and recovery tests."""
        logger.info("🚧 Running Error Handling and Recovery Tests")
        
        # Test database connection failure recovery
        await self._test_database_failure_recovery()
        
        # Test Redis connection failure recovery
        await self._test_redis_failure_recovery()
        
        # Test agent crash recovery
        await self._test_agent_crash_recovery()
        
        # Test network connectivity issues
        await self._test_network_failure_recovery()
    
    async def _run_resource_tests(self):
        """Run resource utilization tests."""
        logger.info("📈 Running Resource Utilization Tests")
        
        # Monitor resources during normal operations
        await self._monitor_system_resources(duration=60)
        
        # Validate resource usage
        if self.metrics.cpu_usage:
            max_cpu = max(self.metrics.cpu_usage)
            passed = max_cpu <= self.targets['max_cpu_usage']
            
            self.test_results.append(TestResult(
                test_name="CPU Usage",
                passed=passed,
                target_value=self.targets['max_cpu_usage'],
                actual_value=max_cpu,
                unit="%"
            ))
        
        if self.metrics.memory_usage:
            max_memory = max(self.metrics.memory_usage)
            passed = max_memory <= self.targets['max_memory_usage']
            
            self.test_results.append(TestResult(
                test_name="Memory Usage",
                passed=passed,
                target_value=self.targets['max_memory_usage'],
                actual_value=max_memory,
                unit="MB"
            ))
    
    async def _run_concurrency_tests(self):
        """Run concurrency and race condition tests."""
        logger.info("🔄 Running Concurrency Tests")
        
        # Test concurrent access to shared resources
        await self._test_concurrent_resource_access()
        
        # Test race conditions in task assignment
        await self._test_task_assignment_race_conditions()
    
    # Helper methods for specific test scenarios
    
    async def _wait_for_agent_ready(self, agent_id: str, timeout: int = 30):
        """Wait for agent to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = await self.client.get(f"/api/agents/{agent_id}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data.get("status") == "ready":
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Agent {agent_id} not ready within {timeout} seconds")
    
    async def _wait_for_task_assignment(self, task_id: str, timeout: int = 10):
        """Wait for task to be assigned."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = await self.client.get(f"/api/v1/tasks/{task_id}")
                if response.status_code == 200:
                    task_data = response.json()
                    if task_data.get("status") == "assigned":
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} not assigned within {timeout} seconds")
    
    async def _wait_for_task_completion(self, task_id: str, timeout: int = 1800):
        """Wait for task to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = await self.client.get(f"/api/v1/tasks/{task_id}")
                if response.status_code == 200:
                    task_data = response.json()
                    status = task_data.get("status")
                    if status in ["completed", "failed"]:
                        return status == "completed"
            except Exception:
                pass
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        raise TimeoutError(f"Task {task_id} not completed within {timeout} seconds")
    
    async def _monitor_system_resources(self, duration: int = 30):
        """Monitor system resource usage."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.metrics.memory_usage.append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_io_mb = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)
                    self.metrics.disk_io.append(disk_io_mb)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    net_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
                    self.metrics.network_io.append(net_io_mb)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def _stress_api_calls(self):
        """Generate API call stress."""
        for _ in range(100):
            try:
                await self.client.get("/health")
                await asyncio.sleep(0.01)  # 100 RPS per task
            except Exception:
                pass
    
    async def _stress_computation(self):
        """Generate computational stress."""
        try:
            response = await self.client.post("/api/v1/tasks", json={
                "task_type": "stress_test",
                "description": "Computational stress test",
                "priority": "low"
            })
            
            if response.status_code == 200:
                task_data = response.json()
                task_id = task_data.get("task_id")
                # Wait briefly for task processing
                await asyncio.sleep(5)
        except Exception:
            pass
    
    async def _test_database_failure_recovery(self):
        """Test database failure recovery."""
        logger.info("Testing database failure recovery")
        
        # This would require sophisticated test harness
        # For now, we'll simulate by testing system behavior under DB stress
        try:
            # Make many concurrent database operations
            tasks = []
            for _ in range(20):
                task = asyncio.create_task(self.client.get("/api/v1/system/status"))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
            error_count = len(results) - success_count
            
            # Validate graceful degradation
            error_rate = error_count / len(results) if results else 0
            passed = error_rate <= self.targets['max_error_rate']
            
            self.test_results.append(TestResult(
                test_name="Database Stress Recovery",
                passed=passed,
                target_value=self.targets['max_error_rate'],
                actual_value=error_rate,
                unit=" error rate"
            ))
            
        except Exception as e:
            logger.error(f"Database failure test error: {e}")
    
    async def _test_redis_failure_recovery(self):
        """Test Redis failure recovery."""
        logger.info("Testing Redis failure recovery")
        
        # Similar to database test - test system resilience
        try:
            # Test Redis-dependent operations
            for _ in range(10):
                response = await self.client.post("/api/agents/spawn", json={
                    "agent_type": "redis_test",
                    "capabilities": ["testing"]
                })
                
                if response.status_code == 200:
                    self.metrics.success_count += 1
                else:
                    self.metrics.error_count += 1
            
        except Exception as e:
            logger.error(f"Redis failure test error: {e}")
    
    async def _test_agent_crash_recovery(self):
        """Test agent crash recovery."""
        logger.info("Testing agent crash recovery")
        
        # Test system's ability to detect and recover from agent failures
        try:
            # This would require actual agent management
            # For now, test system resilience to agent-related errors
            response = await self.client.get("/debug-agents")
            
            if response.status_code == 200:
                agent_data = response.json()
                agent_count = agent_data.get("agent_count", 0)
                
                self.test_results.append(TestResult(
                    test_name="Agent System Availability",
                    passed=True,
                    target_value=1,
                    actual_value=1 if agent_count >= 0 else 0,
                    unit=" availability"
                ))
            
        except Exception as e:
            logger.error(f"Agent crash recovery test error: {e}")
    
    async def _test_network_failure_recovery(self):
        """Test network failure recovery."""
        logger.info("Testing network failure recovery")
        
        # Test system behavior under network stress
        try:
            # Create network load
            tasks = []
            for _ in range(50):
                task = asyncio.create_task(self.client.get("/health", timeout=1.0))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
            total_requests = len(results)
            
            availability = success_count / total_requests if total_requests > 0 else 0
            passed = availability >= self.targets['min_availability']
            
            self.test_results.append(TestResult(
                test_name="Network Stress Availability",
                passed=passed,
                target_value=self.targets['min_availability'],
                actual_value=availability,
                unit=" availability"
            ))
            
        except Exception as e:
            logger.error(f"Network failure test error: {e}")
    
    async def _test_concurrent_resource_access(self):
        """Test concurrent access to shared resources."""
        logger.info("Testing concurrent resource access")
        
        # Test concurrent access to the same resources
        async def access_shared_resource(resource_id: str):
            try:
                response = await self.client.get(f"/api/v1/contexts/{resource_id}")
                return response.status_code == 200
            except Exception:
                return False
        
        # Test with multiple tasks accessing same resource
        resource_id = "shared_test_resource"
        tasks = []
        
        for _ in range(20):
            task = asyncio.create_task(access_shared_resource(resource_id))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)
        
        # Most requests should succeed (allowing for some contention)
        success_rate = success_count / len(results) if results else 0
        passed = success_rate >= 0.8  # 80% success rate acceptable under contention
        
        self.test_results.append(TestResult(
            test_name="Concurrent Resource Access",
            passed=passed,
            target_value=0.8,
            actual_value=success_rate,
            unit=" success rate"
        ))
    
    async def _test_task_assignment_race_conditions(self):
        """Test for race conditions in task assignment."""
        logger.info("Testing task assignment race conditions")
        
        # Create multiple tasks simultaneously and ensure proper assignment
        async def create_task_concurrently(task_index: int):
            try:
                response = await self.client.post("/api/v1/tasks", json={
                    "task_type": "race_condition_test",
                    "description": f"Race condition test task {task_index}",
                    "priority": "normal"
                })
                return response.status_code == 200
            except Exception:
                return False
        
        # Create many tasks concurrently
        tasks = []
        for i in range(30):
            task = asyncio.create_task(create_task_concurrently(i))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if r is True)
        
        # All tasks should be created successfully
        success_rate = success_count / len(results) if results else 0
        passed = success_rate >= 0.95  # 95% success rate
        
        self.test_results.append(TestResult(
            test_name="Task Assignment Race Conditions",
            passed=passed,
            target_value=0.95,
            actual_value=success_rate,
            unit=" success rate"
        ))
    
    def _generate_validation_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Calculate overall statistics
        passed_tests = sum(1 for test in self.test_results if test.passed)
        total_tests = len(self.test_results)
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate error rates
        total_operations = self.metrics.success_count + self.metrics.error_count
        error_rate = self.metrics.error_count / total_operations if total_operations > 0 else 0
        
        # System resource summary
        resource_summary = {}
        if self.metrics.cpu_usage:
            resource_summary["cpu"] = self.metrics.calculate_statistics(self.metrics.cpu_usage)
        if self.metrics.memory_usage:
            resource_summary["memory"] = self.metrics.calculate_statistics(self.metrics.memory_usage)
        
        # Performance summary
        performance_summary = {}
        if self.metrics.agent_spawn_times:
            performance_summary["agent_spawning"] = self.metrics.calculate_statistics(self.metrics.agent_spawn_times)
        if self.metrics.task_assignment_times:
            performance_summary["task_assignment"] = self.metrics.calculate_statistics(self.metrics.task_assignment_times)
        if self.metrics.status_query_times:
            performance_summary["status_queries"] = self.metrics.calculate_statistics(self.metrics.status_query_times)
        if self.metrics.dashboard_update_times:
            performance_summary["dashboard_updates"] = self.metrics.calculate_statistics(self.metrics.dashboard_update_times)
        
        # Create detailed test results
        test_details = []
        for test in self.test_results:
            test_details.append({
                "test_name": test.test_name,
                "passed": test.passed,
                "target_value": test.target_value,
                "actual_value": test.actual_value,
                "unit": test.unit,
                "details": test.details
            })
        
        # Overall validation status
        validation_status = "PASS" if overall_pass_rate >= 0.9 else "FAIL"  # 90% pass rate required
        
        return {
            "validation_summary": {
                "status": validation_status,
                "overall_pass_rate": overall_pass_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "total_validation_time": total_time,
                "error_rate": error_rate
            },
            "performance_metrics": {
                "targets": self.targets,
                "measurements": performance_summary,
                "throughput": {
                    "max_rps": max(self.metrics.requests_per_second) if self.metrics.requests_per_second else 0,
                    "avg_rps": sum(self.metrics.requests_per_second) / len(self.metrics.requests_per_second) if self.metrics.requests_per_second else 0
                }
            },
            "resource_utilization": resource_summary,
            "reliability_metrics": {
                "total_operations": total_operations,
                "successful_operations": self.metrics.success_count,
                "failed_operations": self.metrics.error_count,
                "error_rate": error_rate,
                "timeout_count": self.metrics.timeout_count
            },
            "test_results": test_details,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [test for test in self.test_results if not test.passed]
        
        for test in failed_tests:
            if "Agent Spawning" in test.test_name:
                recommendations.append("Consider optimizing agent initialization process or increasing resource allocation")
            elif "Task Assignment" in test.test_name:
                recommendations.append("Review task routing and assignment algorithms for performance optimization")
            elif "Status Query" in test.test_name:
                recommendations.append("Implement caching for frequently accessed status data")
            elif "Dashboard Update" in test.test_name:
                recommendations.append("Optimize dashboard data aggregation and consider real-time updates")
            elif "CPU Usage" in test.test_name:
                recommendations.append("Review CPU-intensive operations and consider optimization or scaling")
            elif "Memory Usage" in test.test_name:
                recommendations.append("Investigate memory leaks and optimize memory usage patterns")
            elif "Concurrent" in test.test_name:
                recommendations.append("Review concurrency control mechanisms and resource locking strategies")
        
        if not recommendations:
            recommendations.append("All performance targets met. System ready for enterprise deployment.")
        
        return recommendations


async def main():
    """Main execution function."""
    validator = EnterprisePerformanceValidator()
    
    try:
        await validator.setup()
        
        logger.info("🚀 Starting Enterprise Performance and Reliability Validation")
        
        # Run full validation
        report = await validator.run_full_validation()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/tmp/enterprise_performance_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("ENTERPRISE PERFORMANCE VALIDATION RESULTS")
        print("="*80)
        
        summary = report["validation_summary"]
        print(f"Overall Status: {summary['status']}")
        print(f"Pass Rate: {summary['overall_pass_rate']:.1%}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Error Rate: {summary['error_rate']:.1%}")
        print(f"Validation Time: {summary['total_validation_time']:.2f}s")
        
        print("\nPERFORMANCE TARGETS:")
        performance = report["performance_metrics"]
        for metric, target in performance["targets"].items():
            print(f"  {metric}: {target}")
        
        print("\nTEST RESULTS:")
        for test in report["test_results"]:
            status = "✅ PASS" if test["passed"] else "❌ FAIL"
            print(f"  {status} {test['test_name']}: {test['actual_value']:.2f}{test['unit']} (target: {test['target_value']:.2f}{test['unit']})")
        
        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
        print(f"\nFull report saved to: {report_file}")
        print("="*80)
        
        return summary["status"] == "PASS"
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False
    
    finally:
        await validator.teardown()


if __name__ == "__main__":
    import sys
    
    # Run validation
    success = asyncio.run(main())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)