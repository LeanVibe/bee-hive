"""
Comprehensive Testing Framework for LeanVibe Plugin SDK.

Provides complete testing utilities including:
- Plugin test framework
- Mock implementations
- Test runners
- Validation suites
- Performance testing
"""

import asyncio
import uuid
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, AsyncMock
import logging
from enum import Enum

from .interfaces import (
    PluginBase, AgentInterface, TaskInterface, OrchestratorInterface,
    MonitoringInterface, CoordinationInterface, SecurityInterface
)
from .models import (
    TaskResult, CoordinationResult, PluginConfig, PluginContext,
    PluginEvent, EventSeverity, TaskStatus, CoordinationStrategy
)
from .exceptions import PluginTestError, PluginExecutionError


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(Enum):
    """Test failure severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: TestStatus
    execution_time_ms: float = 0.0
    
    # Result details
    message: str = ""
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Test data
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Metadata
    severity: TestSeverity = TestSeverity.MEDIUM
    category: str = "general"
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def add_assertion(self, description: str, passed: bool, expected: Any = None, actual: Any = None) -> None:
        """Add an assertion result."""
        self.assertions.append({
            "description": description,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a performance metric."""
        self.metrics[name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status.value,
            "execution_time_ms": self.execution_time_ms,
            "message": self.message,
            "error": self.error,
            "stack_trace": self.stack_trace,
            "assertions": self.assertions,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "severity": self.severity.value,
            "category": self.category,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class TestSuite:
    """Collection of test results."""
    suite_name: str
    plugin_id: str
    results: List[TestResult] = field(default_factory=list)
    
    # Summary statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_execution_time_ms: float = 0.0
    
    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)
        self.total_tests += 1
        
        if result.status == TestStatus.PASSED:
            self.passed_tests += 1
        elif result.status == TestStatus.FAILED:
            self.failed_tests += 1
        elif result.status == TestStatus.ERROR:
            self.error_tests += 1
        elif result.status == TestStatus.SKIPPED:
            self.skipped_tests += 1
    
    def get_success_rate(self) -> float:
        """Get test success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def complete(self) -> None:
        """Mark test suite as completed."""
        self.completed_at = datetime.utcnow()
        self.total_execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "plugin_id": self.plugin_id,
            "results": [r.to_dict() for r in self.results],
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "error_tests": self.error_tests,
            "skipped_tests": self.skipped_tests,
            "success_rate": self.get_success_rate(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_execution_time_ms": self.total_execution_time_ms
        }


class MockAgent:
    """Mock implementation of AgentInterface for testing."""
    
    def __init__(self, agent_id: str = None, capabilities: List[str] = None):
        self.agent_id = agent_id or f"mock_agent_{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities or ["mock_capability"]
        self.status = "active"
        self._context = {}
        self._message_log = []
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Mock task execution."""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        return TaskResult(
            success=True,
            plugin_id="mock_plugin",
            task_id=task.get("task_id", "mock_task"),
            execution_time_ms=10.0,
            data={"result": "mock_success"}
        )
    
    async def get_context(self) -> Dict[str, Any]:
        """Get mock context."""
        return self._context.copy()
    
    async def send_message(self, message: str, target: Optional[str] = None) -> bool:
        """Mock message sending."""
        self._message_log.append({
            "message": message,
            "target": target,
            "timestamp": datetime.utcnow().isoformat()
        })
        return True
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set mock context."""
        self._context.update(context)
    
    def get_message_log(self) -> List[Dict[str, Any]]:
        """Get logged messages."""
        return self._message_log.copy()


class MockTask:
    """Mock implementation of TaskInterface for testing."""
    
    def __init__(
        self,
        task_id: str = None,
        task_type: str = "mock_task",
        parameters: Dict[str, Any] = None,
        priority: int = 1
    ):
        self.task_id = task_id or f"mock_task_{uuid.uuid4().hex[:8]}"
        self.task_type = task_type
        self.parameters = parameters or {}
        self.priority = priority
        self._status = TaskStatus.PENDING
        self._progress = 0.0
        self._results = {}
        self._dependency_results = {}
    
    async def update_status(self, status: str, progress: float = 0.0) -> None:
        """Mock status update."""
        self._status = TaskStatus(status)
        self._progress = progress
    
    async def add_result(self, key: str, value: Any) -> None:
        """Mock result addition."""
        self._results[key] = value
    
    async def get_dependency_results(self) -> Dict[str, Any]:
        """Get mock dependency results."""
        return self._dependency_results.copy()
    
    def set_dependency_results(self, results: Dict[str, Any]) -> None:
        """Set mock dependency results."""
        self._dependency_results.update(results)


class MockOrchestrator:
    """Mock implementation of OrchestratorInterface for testing."""
    
    def __init__(self):
        self._agents = []
        self._tasks = []
        self._metrics = {
            "cpu_usage": 45.2,
            "memory_usage": 67.8,
            "active_tasks": 12,
            "agent_count": 5
        }
        self._events = []
    
    async def get_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[AgentInterface]:
        """Get mock agents."""
        if not filters:
            return self._agents
        
        # Simple filtering by capabilities
        if "capabilities" in filters:
            required_caps = filters["capabilities"]
            return [
                agent for agent in self._agents
                if any(cap in agent.capabilities for cap in required_caps)
            ]
        
        return self._agents
    
    async def create_task(self, task_type: str, parameters: Dict[str, Any]) -> TaskInterface:
        """Create mock task."""
        task = MockTask(task_type=task_type, parameters=parameters)
        self._tasks.append(task)
        return task
    
    async def schedule_task(self, task: TaskInterface, delay_seconds: int = 0) -> str:
        """Mock task scheduling."""
        schedule_id = f"schedule_{uuid.uuid4().hex[:8]}"
        await asyncio.sleep(0.01)  # Simulate scheduling delay
        return schedule_id
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get mock system metrics."""
        return self._metrics.copy()
    
    async def broadcast_event(self, event: PluginEvent) -> None:
        """Mock event broadcasting."""
        self._events.append(event)
    
    def add_agent(self, agent: MockAgent) -> None:
        """Add mock agent."""
        self._agents.append(agent)
    
    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set mock metrics."""
        self._metrics.update(metrics)
    
    def get_events(self) -> List[PluginEvent]:
        """Get broadcasted events."""
        return self._events.copy()


class MockMonitoring:
    """Mock implementation of MonitoringInterface for testing."""
    
    def __init__(self):
        self._metrics = []
        self._events = []
        self._alerts = []
        self._performance_data = {}
    
    async def log_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Mock metric logging."""
        self._metrics.append({
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def log_event(self, event: PluginEvent) -> None:
        """Mock event logging."""
        self._events.append(event)
    
    async def create_alert(self, message: str, severity: str = "info") -> None:
        """Mock alert creation."""
        self._alerts.append({
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def get_performance_data(self, plugin_id: str) -> Dict[str, Any]:
        """Get mock performance data."""
        return self._performance_data.get(plugin_id, {})
    
    def get_logged_metrics(self) -> List[Dict[str, Any]]:
        """Get logged metrics."""
        return self._metrics.copy()
    
    def get_logged_events(self) -> List[PluginEvent]:
        """Get logged events."""
        return self._events.copy()
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get created alerts."""
        return self._alerts.copy()


class MockCoordination:
    """Mock implementation of CoordinationInterface for testing."""
    
    def __init__(self):
        self._groups = {}
        self._coordination_results = []
    
    async def coordinate_agents(
        self,
        agents: List[AgentInterface],
        strategy: str = "parallel"
    ) -> CoordinationResult:
        """Mock agent coordination."""
        coordination_id = f"coord_{uuid.uuid4().hex[:8]}"
        
        result = CoordinationResult(
            success=True,
            coordination_id=coordination_id,
            strategy=CoordinationStrategy(strategy),
            total_agents=len(agents)
        )
        
        # Simulate coordination
        await asyncio.sleep(0.05)
        
        # Mock successful results for all agents
        for agent in agents:
            task_result = TaskResult(
                success=True,
                plugin_id="mock_plugin",
                task_id=f"task_{agent.agent_id}",
                execution_time_ms=20.0
            )
            result.add_agent_result(agent.agent_id, task_result)
        
        result.complete()
        self._coordination_results.append(result)
        
        return result
    
    async def create_agent_group(self, agent_ids: List[str], group_name: str) -> str:
        """Mock group creation."""
        group_id = f"group_{uuid.uuid4().hex[:8]}"
        self._groups[group_id] = {
            "name": group_name,
            "agent_ids": agent_ids,
            "created_at": datetime.utcnow().isoformat()
        }
        return group_id
    
    async def sync_agents(self, group_id: str, timeout_seconds: int = 30) -> bool:
        """Mock agent synchronization."""
        await asyncio.sleep(0.02)
        return group_id in self._groups
    
    async def distribute_work(
        self,
        work_items: List[Dict[str, Any]],
        agent_group: str
    ) -> List[TaskResult]:
        """Mock work distribution."""
        results = []
        
        for i, work_item in enumerate(work_items):
            result = TaskResult(
                success=True,
                plugin_id="mock_plugin",
                task_id=f"work_{i}",
                execution_time_ms=15.0,
                data=work_item
            )
            results.append(result)
        
        return results
    
    def get_coordination_results(self) -> List[CoordinationResult]:
        """Get coordination results."""
        return self._coordination_results.copy()


class MockSecurity:
    """Mock implementation of SecurityInterface for testing."""
    
    def __init__(self):
        self._permissions = {}
        self._audit_log = []
        self._encryption_key = "mock_key_123"
    
    async def validate_permissions(self, operation: str, context: Dict[str, Any]) -> bool:
        """Mock permission validation."""
        # Default to allowing operations in test mode
        return self._permissions.get(operation, True)
    
    async def encrypt_data(self, data: str) -> str:
        """Mock data encryption."""
        # Simple mock encryption (not for production)
        return f"encrypted_{len(data)}_{hash(data + self._encryption_key)}"
    
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Mock data decryption."""
        # Simple mock decryption
        if encrypted_data.startswith("encrypted_"):
            return "decrypted_mock_data"
        return encrypted_data
    
    async def audit_log(self, action: str, details: Dict[str, Any]) -> None:
        """Mock audit logging."""
        self._audit_log.append({
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def set_permission(self, operation: str, allowed: bool) -> None:
        """Set mock permission."""
        self._permissions[operation] = allowed
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self._audit_log.copy()


class PluginTestFramework:
    """
    Comprehensive testing framework for plugins.
    
    Provides utilities for testing plugin functionality, performance,
    and integration with the LeanVibe system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_test_framework")
        self._test_suites: List[TestSuite] = []
        self._current_suite: Optional[TestSuite] = None
        
        # Mock interfaces
        self.mock_orchestrator = MockOrchestrator()
        self.mock_monitoring = MockMonitoring()
        self.mock_coordination = MockCoordination()
        self.mock_security = MockSecurity()
        
        # Test configuration
        self.timeout_seconds = 30
        self.performance_threshold_ms = 1000
        self.memory_threshold_mb = 100
    
    def create_test_suite(self, suite_name: str, plugin_id: str) -> TestSuite:
        """Create a new test suite."""
        suite = TestSuite(suite_name=suite_name, plugin_id=plugin_id)
        self._test_suites.append(suite)
        self._current_suite = suite
        return suite
    
    async def test_plugin_initialization(self, plugin: PluginBase) -> TestResult:
        """Test plugin initialization."""
        test_result = TestResult(
            test_name="plugin_initialization",
            status=TestStatus.RUNNING,
            category="initialization"
        )
        
        start_time = time.time()
        
        try:
            # Test initialization
            success = await plugin.initialize(self.mock_orchestrator)
            
            execution_time = (time.time() - start_time) * 1000
            test_result.execution_time_ms = execution_time
            
            # Assertions
            test_result.add_assertion("Plugin initialized successfully", success)
            test_result.add_assertion("Plugin status is initialized", plugin.status.value == "initialized")
            test_result.add_assertion("Plugin has valid ID", bool(plugin.plugin_id))
            test_result.add_assertion("Plugin has valid name", bool(plugin.name))
            
            # Performance check
            test_result.add_metric("initialization_time_ms", execution_time)
            test_result.add_assertion(
                "Initialization time under threshold",
                execution_time < self.performance_threshold_ms
            )
            
            # Determine test status
            failed_assertions = [a for a in test_result.assertions if not a["passed"]]
            if failed_assertions:
                test_result.status = TestStatus.FAILED
                test_result.message = f"Failed {len(failed_assertions)} assertions"
            else:
                test_result.status = TestStatus.PASSED
                test_result.message = "Plugin initialization successful"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
            test_result.execution_time_ms = (time.time() - start_time) * 1000
        
        test_result.completed_at = datetime.utcnow()
        
        if self._current_suite:
            self._current_suite.add_result(test_result)
        
        return test_result
    
    async def test_plugin_task_execution(
        self,
        plugin: PluginBase,
        test_task: Optional[MockTask] = None
    ) -> TestResult:
        """Test plugin task execution."""
        test_result = TestResult(
            test_name="plugin_task_execution",
            status=TestStatus.RUNNING,
            category="execution"
        )
        
        start_time = time.time()
        
        try:
            # Create test task if not provided
            if not test_task:
                test_task = MockTask(
                    task_type="test_task",
                    parameters={"test_data": "mock_value"}
                )
            
            # Execute task
            result = await plugin.execute(test_task)
            
            execution_time = (time.time() - start_time) * 1000
            test_result.execution_time_ms = execution_time
            
            # Assertions
            test_result.add_assertion("Task execution returned result", result is not None)
            test_result.add_assertion("Result has correct plugin ID", result.plugin_id == plugin.plugin_id)
            test_result.add_assertion("Result has correct task ID", result.task_id == test_task.task_id)
            
            # Performance check
            test_result.add_metric("execution_time_ms", execution_time)
            test_result.add_assertion(
                "Execution time under threshold",
                execution_time < self.performance_threshold_ms
            )
            
            # Check result validity
            if result.success:
                test_result.add_assertion("Task completed successfully", True)
            else:
                test_result.add_assertion("Task failed with error", False, expected=True, actual=result.error)
            
            # Determine test status
            failed_assertions = [a for a in test_result.assertions if not a["passed"]]
            if failed_assertions:
                test_result.status = TestStatus.FAILED
                test_result.message = f"Failed {len(failed_assertions)} assertions"
            else:
                test_result.status = TestStatus.PASSED
                test_result.message = "Task execution successful"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
            test_result.execution_time_ms = (time.time() - start_time) * 1000
        
        test_result.completed_at = datetime.utcnow()
        
        if self._current_suite:
            self._current_suite.add_result(test_result)
        
        return test_result
    
    async def test_plugin_cleanup(self, plugin: PluginBase) -> TestResult:
        """Test plugin cleanup."""
        test_result = TestResult(
            test_name="plugin_cleanup",
            status=TestStatus.RUNNING,
            category="cleanup"
        )
        
        start_time = time.time()
        
        try:
            # Test cleanup
            success = await plugin.cleanup()
            
            execution_time = (time.time() - start_time) * 1000
            test_result.execution_time_ms = execution_time
            
            # Assertions
            test_result.add_assertion("Plugin cleanup successful", success)
            test_result.add_metric("cleanup_time_ms", execution_time)
            
            # Determine test status
            if success:
                test_result.status = TestStatus.PASSED
                test_result.message = "Plugin cleanup successful"
            else:
                test_result.status = TestStatus.FAILED
                test_result.message = "Plugin cleanup failed"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
            test_result.execution_time_ms = (time.time() - start_time) * 1000
        
        test_result.completed_at = datetime.utcnow()
        
        if self._current_suite:
            self._current_suite.add_result(test_result)
        
        return test_result
    
    async def performance_test(
        self,
        plugin: PluginBase,
        iterations: int = 10,
        concurrent_tasks: int = 1
    ) -> TestResult:
        """Run performance tests on plugin."""
        test_result = TestResult(
            test_name="performance_test",
            status=TestStatus.RUNNING,
            category="performance"
        )
        
        start_time = time.time()
        
        try:
            execution_times = []
            memory_usage = []
            
            for i in range(iterations):
                # Create test task
                test_task = MockTask(
                    task_type="performance_test",
                    parameters={"iteration": i}
                )
                
                # Measure execution
                iter_start = time.time()
                result = await plugin.execute(test_task)
                iter_time = (time.time() - iter_start) * 1000
                
                execution_times.append(iter_time)
                
                # Check success
                test_result.add_assertion(
                    f"Iteration {i} successful",
                    result.success,
                    expected=True,
                    actual=result.success
                )
            
            total_time = (time.time() - start_time) * 1000
            test_result.execution_time_ms = total_time
            
            # Calculate performance metrics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            
            test_result.add_metric("avg_execution_time_ms", avg_time)
            test_result.add_metric("min_execution_time_ms", min_time)
            test_result.add_metric("max_execution_time_ms", max_time)
            test_result.add_metric("total_iterations", iterations)
            test_result.add_metric("throughput_per_second", 1000 / avg_time if avg_time > 0 else 0)
            
            # Performance assertions
            test_result.add_assertion(
                "Average execution time under threshold",
                avg_time < self.performance_threshold_ms
            )
            test_result.add_assertion(
                "Maximum execution time reasonable",
                max_time < self.performance_threshold_ms * 2
            )
            
            # Determine test status
            failed_assertions = [a for a in test_result.assertions if not a["passed"]]
            if failed_assertions:
                test_result.status = TestStatus.FAILED
                test_result.message = f"Performance test failed {len(failed_assertions)} assertions"
            else:
                test_result.status = TestStatus.PASSED
                test_result.message = f"Performance test passed with {avg_time:.2f}ms avg execution time"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
            test_result.execution_time_ms = (time.time() - start_time) * 1000
        
        test_result.completed_at = datetime.utcnow()
        
        if self._current_suite:
            self._current_suite.add_result(test_result)
        
        return test_result
    
    async def integration_test(self, plugin: PluginBase) -> TestResult:
        """Test plugin integration with system interfaces."""
        test_result = TestResult(
            test_name="integration_test",
            status=TestStatus.RUNNING,
            category="integration"
        )
        
        start_time = time.time()
        
        try:
            # Add mock agents for testing
            mock_agent1 = MockAgent("agent1", ["capability1", "capability2"])
            mock_agent2 = MockAgent("agent2", ["capability2", "capability3"])
            self.mock_orchestrator.add_agent(mock_agent1)
            self.mock_orchestrator.add_agent(mock_agent2)
            
            # Test orchestrator integration
            agents = await plugin.get_available_agents(["capability1"])
            test_result.add_assertion("Can retrieve agents", len(agents) > 0)
            
            # Test subtask creation
            subtask = await plugin.create_subtask("test_subtask", {"param": "value"})
            test_result.add_assertion("Can create subtasks", subtask is not None)
            
            # Test event logging
            await plugin.log_info("Test info message", test_param="value")
            logged_events = self.mock_monitoring.get_logged_events()
            test_result.add_assertion("Can log events", len(logged_events) > 0)
            
            # Test metrics logging
            if hasattr(plugin, '_monitoring') and plugin._monitoring:
                await plugin._monitoring.log_metric("test_metric", 42.0)
                metrics = self.mock_monitoring.get_logged_metrics()
                test_result.add_assertion("Can log metrics", len(metrics) > 0)
            
            execution_time = (time.time() - start_time) * 1000
            test_result.execution_time_ms = execution_time
            
            # Determine test status
            failed_assertions = [a for a in test_result.assertions if not a["passed"]]
            if failed_assertions:
                test_result.status = TestStatus.FAILED
                test_result.message = f"Integration test failed {len(failed_assertions)} assertions"
            else:
                test_result.status = TestStatus.PASSED
                test_result.message = "Integration test passed"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
            test_result.execution_time_ms = (time.time() - start_time) * 1000
        
        test_result.completed_at = datetime.utcnow()
        
        if self._current_suite:
            self._current_suite.add_result(test_result)
        
        return test_result
    
    async def run_full_test_suite(self, plugin: PluginBase) -> TestSuite:
        """Run complete test suite for a plugin."""
        suite = self.create_test_suite(f"full_test_{plugin.name}", plugin.plugin_id)
        
        try:
            # Initialize plugin for testing
            await plugin.initialize(self.mock_orchestrator)
            
            # Run all tests
            await self.test_plugin_initialization(plugin)
            await self.test_plugin_task_execution(plugin)
            await self.integration_test(plugin)
            await self.performance_test(plugin, iterations=5)
            await self.test_plugin_cleanup(plugin)
            
        except Exception as e:
            # Add error test result
            error_result = TestResult(
                test_name="test_suite_error",
                status=TestStatus.ERROR,
                error=str(e),
                stack_trace=traceback.format_exc(),
                category="suite"
            )
            suite.add_result(error_result)
        
        suite.complete()
        return suite
    
    def get_test_report(self, suite: Optional[TestSuite] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if suite:
            suites = [suite]
        else:
            suites = self._test_suites
        
        report = {
            "test_suites": [suite.to_dict() for suite in suites],
            "summary": {
                "total_suites": len(suites),
                "total_tests": sum(suite.total_tests for suite in suites),
                "total_passed": sum(suite.passed_tests for suite in suites),
                "total_failed": sum(suite.failed_tests for suite in suites),
                "total_errors": sum(suite.error_tests for suite in suites),
                "overall_success_rate": 0.0
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Calculate overall success rate
        total_tests = report["summary"]["total_tests"]
        if total_tests > 0:
            report["summary"]["overall_success_rate"] = report["summary"]["total_passed"] / total_tests
        
        return report


class TestRunner:
    """Test runner for executing plugin tests."""
    
    def __init__(self):
        self.framework = PluginTestFramework()
        self.logger = logging.getLogger("plugin_test_runner")
    
    async def run_tests(self, plugins: List[PluginBase]) -> List[TestSuite]:
        """Run tests for multiple plugins."""
        results = []
        
        for plugin in plugins:
            try:
                self.logger.info(f"Running tests for plugin: {plugin.name}")
                suite = await self.framework.run_full_test_suite(plugin)
                results.append(suite)
                self.logger.info(f"Completed tests for {plugin.name}: {suite.get_success_rate():.2%} success rate")
            except Exception as e:
                self.logger.error(f"Failed to test plugin {plugin.name}: {e}")
        
        return results
    
    def generate_html_report(self, suites: List[TestSuite]) -> str:
        """Generate HTML test report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LeanVibe Plugin Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .suite { border: 1px solid #ddd; margin: 10px 0; padding: 10px; }
                .passed { color: green; }
                .failed { color: red; }
                .error { color: orange; }
                .metrics { background: #f5f5f5; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>LeanVibe Plugin Test Report</h1>
        """
        
        for suite in suites:
            html += f"""
            <div class="suite">
                <h2>{suite.suite_name}</h2>
                <p>Plugin: {suite.plugin_id}</p>
                <p>Success Rate: {suite.get_success_rate():.2%}</p>
                <p>Total Tests: {suite.total_tests}</p>
                <p class="passed">Passed: {suite.passed_tests}</p>
                <p class="failed">Failed: {suite.failed_tests}</p>
                <p class="error">Errors: {suite.error_tests}</p>
                
                <h3>Test Results</h3>
            """
            
            for result in suite.results:
                status_class = result.status.value
                html += f"""
                <div class="{status_class}">
                    <h4>{result.test_name} - {result.status.value.upper()}</h4>
                    <p>Execution Time: {result.execution_time_ms:.2f}ms</p>
                    <p>{result.message}</p>
                    {f'<p>Error: {result.error}</p>' if result.error else ''}
                </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


class ValidationSuite:
    """Validation suite for plugin compliance checking."""
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_validation_suite")
    
    async def validate_plugin_compliance(self, plugin: PluginBase) -> TestResult:
        """Validate plugin compliance with SDK requirements."""
        test_result = TestResult(
            test_name="compliance_validation",
            status=TestStatus.RUNNING,
            category="compliance"
        )
        
        try:
            # Check required methods
            required_methods = ["handle_task", "initialize", "cleanup"]
            for method in required_methods:
                has_method = hasattr(plugin, method) and callable(getattr(plugin, method))
                test_result.add_assertion(f"Has required method: {method}", has_method)
            
            # Check configuration
            has_config = hasattr(plugin, "config") and plugin.config is not None
            test_result.add_assertion("Has valid configuration", has_config)
            
            if has_config:
                config_errors = plugin.config.validate()
                test_result.add_assertion("Configuration is valid", len(config_errors) == 0)
            
            # Check plugin type
            has_plugin_type = hasattr(plugin, "plugin_type") and plugin.plugin_type is not None
            test_result.add_assertion("Has valid plugin type", has_plugin_type)
            
            # Determine status
            failed_assertions = [a for a in test_result.assertions if not a["passed"]]
            if failed_assertions:
                test_result.status = TestStatus.FAILED
                test_result.message = f"Compliance validation failed: {len(failed_assertions)} issues"
            else:
                test_result.status = TestStatus.PASSED
                test_result.message = "Plugin is compliant with SDK requirements"
            
        except Exception as e:
            test_result.status = TestStatus.ERROR
            test_result.error = str(e)
            test_result.stack_trace = traceback.format_exc()
        
        test_result.completed_at = datetime.utcnow()
        return test_result