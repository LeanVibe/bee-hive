"""
Consolidated Testing Framework Base Classes

This module provides the foundation for testing all consolidated components
with comprehensive coverage, performance validation, and regression detection.
"""

import asyncio
import time
import uuid
import pytest
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager
import json
import sys
import os

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from app.core.universal_orchestrator import (
    UniversalOrchestrator, 
    OrchestratorConfig, 
    OrchestratorMode,
    AgentRole,
    HealthStatus
)
from app.core.unified_manager_base import (
    UnifiedManagerBase,
    ManagerConfig,
    ManagerStatus,
    ManagerMetrics
)
from app.core.communication_hub.communication_hub import (
    CommunicationHub,
    CommunicationConfig,
    RoutingStrategy
)
from app.core.engines.base_engine import BaseEngine
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration for testing."""
    metric_name: str
    target_value: float
    max_acceptable_value: float
    unit: str = "ms"
    regression_threshold_percent: float = 5.0  # 5% regression threshold
    
    def validate(self, actual_value: float) -> Tuple[bool, str]:
        """Validate actual value against threshold."""
        if actual_value <= self.target_value:
            return True, f"✅ {self.metric_name}: {actual_value:.2f}{self.unit} (target: {self.target_value}{self.unit})"
        elif actual_value <= self.max_acceptable_value:
            return True, f"⚠️ {self.metric_name}: {actual_value:.2f}{self.unit} (acceptable: {self.max_acceptable_value}{self.unit})"
        else:
            return False, f"❌ {self.metric_name}: {actual_value:.2f}{self.unit} exceeds threshold ({self.max_acceptable_value}{self.unit})"


@dataclass 
class TestScenario:
    """Standard test scenario structure."""
    name: str
    description: str
    setup_data: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    performance_thresholds: List[PerformanceThreshold] = field(default_factory=list)
    cleanup_actions: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def add_performance_threshold(
        self,
        metric_name: str,
        target_value: float,
        max_acceptable_value: float,
        unit: str = "ms"
    ) -> None:
        """Add a performance threshold to the scenario."""
        self.performance_thresholds.append(
            PerformanceThreshold(metric_name, target_value, max_acceptable_value, unit)
        )


@dataclass
class TestMetrics:
    """Test execution metrics."""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    performance_results: Dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def complete(self) -> None:
        """Mark test as completed and calculate duration."""
        self.end_time = datetime.utcnow()
        if self.end_time and self.start_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class TestMetricsCollector:
    """Collects and analyzes test metrics."""
    
    def __init__(self):
        self.test_metrics: Dict[str, TestMetrics] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def start_test(self, test_name: str) -> TestMetrics:
        """Start tracking metrics for a test."""
        metrics = TestMetrics(
            test_name=test_name,
            start_time=datetime.utcnow()
        )
        self.test_metrics[test_name] = metrics
        return metrics
        
    def complete_test(self, test_name: str, success: bool = True) -> TestMetrics:
        """Complete test metrics tracking."""
        if test_name in self.test_metrics:
            metrics = self.test_metrics[test_name]
            metrics.success = success
            metrics.complete()
            return metrics
        else:
            raise KeyError(f"Test {test_name} was not started")
    
    def record_performance_metric(self, test_name: str, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        if test_name in self.test_metrics:
            self.test_metrics[test_name].performance_results[metric_name] = value
            
            # Update performance history
            history_key = f"{test_name}.{metric_name}"
            if history_key not in self.performance_history:
                self.performance_history[history_key] = []
            self.performance_history[history_key].append(value)
    
    def detect_regression(self, test_name: str, threshold_percent: float = 5.0) -> List[str]:
        """Detect performance regression compared to baseline."""
        regressions = []
        
        if test_name not in self.test_metrics:
            return regressions
            
        current_metrics = self.test_metrics[test_name].performance_results
        baseline = self.baseline_metrics.get(test_name, {})
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                regression_percent = ((current_value - baseline_value) / baseline_value) * 100
                
                if regression_percent > threshold_percent:
                    regressions.append(
                        f"{metric_name}: {regression_percent:.1f}% regression "
                        f"({current_value:.2f} vs baseline {baseline_value:.2f})"
                    )
        
        return regressions
    
    def load_baseline_metrics(self, filepath: str) -> bool:
        """Load baseline metrics from file."""
        try:
            with open(filepath, 'r') as f:
                self.baseline_metrics = json.load(f)
            return True
        except Exception:
            return False
    
    def save_baseline_metrics(self, filepath: str) -> bool:
        """Save current metrics as baseline."""
        try:
            baseline_data = {}
            for test_name, metrics in self.test_metrics.items():
                baseline_data[test_name] = metrics.performance_results
            
            with open(filepath, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            return True
        except Exception:
            return False


class ConsolidatedTestBase(ABC):
    """
    Base class for all consolidated component tests.
    
    Provides:
    - Consistent test environment setup
    - Performance monitoring and validation
    - Regression detection
    - Resource cleanup
    - Mock management
    """
    
    def __init__(self):
        self.metrics_collector = TestMetricsCollector()
        self.test_db = None
        self.test_redis = None
        self.cleanup_tasks: List[Callable] = []
        self._test_start_time = None
        
    @pytest.fixture(autouse=True) 
    async def setup_test_environment(self):
        """Setup consistent test environment."""
        # Load baseline metrics
        baseline_file = os.path.join(
            os.path.dirname(__file__), 
            "../baselines/performance_baselines.json"
        )
        self.metrics_collector.load_baseline_metrics(baseline_file)
        
        # Setup test resources
        await self._setup_test_database()
        await self._setup_test_redis()
        await self._setup_test_metrics()
        
        yield
        
        # Cleanup
        await self.cleanup_test_environment()
    
    async def _setup_test_database(self):
        """Setup test database with mock or in-memory instance."""
        self.test_db = MagicMock()
        # Configure mock database responses
        self.test_db.execute.return_value = AsyncMock()
        self.test_db.fetchall.return_value = []
    
    async def _setup_test_redis(self):
        """Setup test Redis with mock instance."""
        self.test_redis = AsyncMock()
        self.test_redis.ping.return_value = True
        self.test_redis.get.return_value = None
        self.test_redis.set.return_value = True
        self.test_redis.publish.return_value = 1
    
    async def _setup_test_metrics(self):
        """Setup test metrics collection."""
        pass
    
    async def cleanup_test_environment(self):
        """Clean teardown of test resources."""
        # Execute cleanup tasks
        for cleanup_task in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_task):
                    await cleanup_task()
                else:
                    cleanup_task()
            except Exception as e:
                print(f"Cleanup task failed: {e}")
        
        # Clear cleanup tasks
        self.cleanup_tasks.clear()
    
    def add_cleanup_task(self, task: Callable):
        """Add cleanup task to be executed during teardown."""
        self.cleanup_tasks.append(task)
    
    async def run_performance_test(
        self,
        test_name: str,
        test_func: Callable,
        scenario: TestScenario,
        *args, **kwargs
    ) -> TestMetrics:
        """
        Run a performance test with comprehensive monitoring.
        
        Args:
            test_name: Name of the test
            test_func: Test function to execute
            scenario: Test scenario with performance thresholds
            *args, **kwargs: Arguments for test function
            
        Returns:
            TestMetrics: Test execution metrics
        """
        # Start metrics collection
        metrics = self.metrics_collector.start_test(test_name)
        
        try:
            # Execute test function
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(*args, **kwargs)
            else:
                result = test_func(*args, **kwargs)
            
            # Validate performance thresholds
            for threshold in scenario.performance_thresholds:
                actual_value = metrics.performance_results.get(threshold.metric_name)
                if actual_value is not None:
                    is_valid, message = threshold.validate(actual_value)
                    if not is_valid:
                        metrics.errors.append(message)
                    else:
                        print(message)  # Log successful validation
            
            # Check for regressions
            regressions = self.metrics_collector.detect_regression(test_name)
            if regressions:
                metrics.errors.extend([f"REGRESSION: {r}" for r in regressions])
            
            # Mark as successful if no errors
            success = len(metrics.errors) == 0
            
        except Exception as e:
            success = False
            metrics.errors.append(f"Test execution failed: {str(e)}")
            result = None
        
        # Complete metrics collection
        self.metrics_collector.complete_test(test_name, success)
        
        return metrics
    
    @asynccontextmanager
    async def performance_monitor(self, test_name: str, metric_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_performance_metric(test_name, metric_name, duration_ms)
    
    async def validate_performance_thresholds(
        self, 
        test_name: str,
        thresholds: List[PerformanceThreshold]
    ) -> Tuple[bool, List[str]]:
        """Validate performance against established thresholds."""
        if test_name not in self.metrics_collector.test_metrics:
            return False, ["Test metrics not found"]
        
        metrics = self.metrics_collector.test_metrics[test_name]
        errors = []
        
        for threshold in thresholds:
            actual_value = metrics.performance_results.get(threshold.metric_name)
            if actual_value is not None:
                is_valid, message = threshold.validate(actual_value)
                if not is_valid:
                    errors.append(message)
        
        return len(errors) == 0, errors
    
    def create_mock_agent(
        self,
        agent_id: str = None,
        role: AgentRole = AgentRole.WORKER,
        capabilities: List[str] = None
    ) -> Dict[str, Any]:
        """Create a mock agent for testing."""
        return {
            "id": agent_id or str(uuid.uuid4()),
            "role": role.value,
            "status": AgentStatus.ACTIVE.value,
            "capabilities": capabilities or ["test_capability"],
            "registration_time": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "total_tasks_completed": 0,
            "error_count": 0
        }
    
    def create_mock_task(
        self,
        task_id: str = None,
        task_type: str = "test_task",
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict[str, Any]:
        """Create a mock task for testing."""
        return {
            "id": task_id or str(uuid.uuid4()),
            "type": task_type,
            "priority": priority.value,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "required_capabilities": ["test_capability"]
        }
    
    # Abstract methods for component-specific implementation
    
    @abstractmethod
    async def setup_component(self) -> Any:
        """Setup the component under test."""
        pass
    
    @abstractmethod
    async def cleanup_component(self) -> None:
        """Cleanup the component after testing."""
        pass
    
    @abstractmethod  
    def get_performance_scenarios(self) -> List[TestScenario]:
        """Get performance test scenarios for the component."""
        pass


class LoadTestBase(ConsolidatedTestBase):
    """Base class for load testing consolidated components."""
    
    def __init__(self):
        super().__init__()
        self.concurrent_tasks: List[asyncio.Task] = []
        self.load_test_results: Dict[str, Any] = {}
    
    async def run_load_test(
        self,
        test_name: str,
        concurrent_operations: int,
        operation_func: Callable,
        duration_seconds: int = 60,
        *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Run load test with specified concurrency and duration.
        
        Args:
            test_name: Name of the load test
            concurrent_operations: Number of concurrent operations
            operation_func: Operation function to execute
            duration_seconds: Duration of the test
            *args, **kwargs: Arguments for operation function
        
        Returns:
            Dict containing load test results
        """
        metrics = self.metrics_collector.start_test(test_name)
        
        # Track operations
        completed_operations = 0
        failed_operations = 0
        operation_latencies = []
        
        async def run_operation():
            nonlocal completed_operations, failed_operations
            operation_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
                completed_operations += 1
            except Exception as e:
                failed_operations += 1
                metrics.errors.append(f"Operation failed: {str(e)}")
            finally:
                operation_latency = (time.time() - operation_start) * 1000
                operation_latencies.append(operation_latency)
        
        # Start load test
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Launch concurrent operations
        while time.time() < end_time:
            # Maintain concurrent operation count
            if len(self.concurrent_tasks) < concurrent_operations:
                task = asyncio.create_task(run_operation())
                self.concurrent_tasks.append(task)
            
            # Clean up completed tasks
            self.concurrent_tasks = [t for t in self.concurrent_tasks if not t.done()]
            
            # Small delay to prevent tight loop
            await asyncio.sleep(0.01)
        
        # Wait for remaining tasks to complete
        if self.concurrent_tasks:
            await asyncio.gather(*self.concurrent_tasks, return_exceptions=True)
        
        # Calculate results
        actual_duration = time.time() - start_time
        throughput = completed_operations / actual_duration
        avg_latency = sum(operation_latencies) / len(operation_latencies) if operation_latencies else 0
        max_latency = max(operation_latencies) if operation_latencies else 0
        error_rate = (failed_operations / max(completed_operations + failed_operations, 1)) * 100
        
        # Record metrics
        self.metrics_collector.record_performance_metric(test_name, "throughput_ops_per_sec", throughput)
        self.metrics_collector.record_performance_metric(test_name, "average_latency_ms", avg_latency)
        self.metrics_collector.record_performance_metric(test_name, "max_latency_ms", max_latency)
        self.metrics_collector.record_performance_metric(test_name, "error_rate_percent", error_rate)
        
        results = {
            "concurrent_operations": concurrent_operations,
            "duration_seconds": actual_duration,
            "completed_operations": completed_operations,
            "failed_operations": failed_operations,
            "throughput_ops_per_sec": throughput,
            "average_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "error_rate_percent": error_rate
        }
        
        self.load_test_results[test_name] = results
        self.metrics_collector.complete_test(test_name, error_rate < 5.0)  # 5% error threshold
        
        return results
    
    async def validate_load_test_requirements(
        self,
        test_name: str,
        min_throughput: float,
        max_latency_ms: float,
        max_error_rate_percent: float = 5.0
    ) -> Tuple[bool, List[str]]:
        """Validate load test results against requirements."""
        if test_name not in self.load_test_results:
            return False, ["Load test results not found"]
        
        results = self.load_test_results[test_name]
        errors = []
        
        if results["throughput_ops_per_sec"] < min_throughput:
            errors.append(f"Throughput {results['throughput_ops_per_sec']:.2f} ops/sec below minimum {min_throughput}")
        
        if results["average_latency_ms"] > max_latency_ms:
            errors.append(f"Average latency {results['average_latency_ms']:.2f}ms exceeds maximum {max_latency_ms}ms")
        
        if results["error_rate_percent"] > max_error_rate_percent:
            errors.append(f"Error rate {results['error_rate_percent']:.2f}% exceeds maximum {max_error_rate_percent}%")
        
        return len(errors) == 0, errors


# Utility functions for test data generation

def generate_test_agents(count: int, roles: List[AgentRole] = None) -> List[Dict[str, Any]]:
    """Generate test agents with varying roles and capabilities."""
    if roles is None:
        roles = [AgentRole.WORKER, AgentRole.SPECIALIST, AgentRole.COORDINATOR]
    
    agents = []
    for i in range(count):
        role = roles[i % len(roles)]
        capabilities = [f"capability_{i%5}", f"skill_{i%3}"]
        
        if role == AgentRole.COORDINATOR:
            capabilities.extend(["coordination", "delegation"])
        elif role == AgentRole.SPECIALIST:
            capabilities.extend(["expert_knowledge", "analysis"])
        
        agent = {
            "id": f"test_agent_{i}",
            "role": role.value,
            "status": AgentStatus.ACTIVE.value,
            "capabilities": capabilities,
            "registration_time": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        agents.append(agent)
    
    return agents


def generate_test_tasks(count: int, priorities: List[TaskPriority] = None) -> List[Dict[str, Any]]:
    """Generate test tasks with varying priorities and requirements."""
    if priorities is None:
        priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL]
    
    tasks = []
    for i in range(count):
        priority = priorities[i % len(priorities)]
        
        task = {
            "id": f"test_task_{i}",
            "type": f"task_type_{i%3}",
            "priority": priority.value,
            "status": TaskStatus.PENDING.value,
            "required_capabilities": [f"capability_{i%5}"],
            "created_at": datetime.utcnow().isoformat(),
            "estimated_duration": 1000 + (i % 5000)  # 1-6 seconds
        }
        tasks.append(task)
    
    return tasks