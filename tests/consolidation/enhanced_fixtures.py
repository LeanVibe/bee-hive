"""
Enhanced Test Fixtures for Epic 1 Consolidation Testing
=======================================================

This module provides specialized test fixtures and utilities for testing
consolidated components, particularly focusing on manager consolidation
and integration with ConsolidatedProductionOrchestrator.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
import tempfile
import json
from pathlib import Path

from tests.consolidation.consolidation_framework import ConsolidationTarget
from tests.consolidation.manager_consolidation_test_plan import (
    ManagerConsolidationTarget,
    MANAGER_CONSOLIDATION_TEST_PLAN
)


@dataclass
class ConsolidatedComponentMock:
    """Mock for consolidated components with enhanced capabilities."""
    component_name: str
    public_api: List[str] = field(default_factory=list)
    internal_methods: List[str] = field(default_factory=list) 
    integration_points: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    mock_instance: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize mock instance after dataclass initialization."""
        self.mock_instance = MagicMock()
        
        # Add public API methods
        for method in self.public_api:
            setattr(self.mock_instance, method, AsyncMock() if method.startswith('async_') else MagicMock())
        
        # Add performance simulation
        self.mock_instance.get_performance_metrics = MagicMock(return_value=self.performance_metrics)


@pytest.fixture(scope="session")
def consolidated_orchestrator_mock():
    """Mock of ConsolidatedProductionOrchestrator with realistic behavior."""
    mock = ConsolidatedComponentMock(
        component_name="ConsolidatedProductionOrchestrator",
        public_api=[
            "start", "stop", "delegate_task", "spawn_agent", "shutdown_agent",
            "get_system_status", "process_task_queue", "monitor_health"
        ],
        integration_points=[
            "task_completed", "agent_status_changed", "workflow_finished",
            "resource_alert", "performance_warning"
        ],
        performance_metrics={
            "startup_time": 0.5,
            "task_throughput": 50.0,
            "memory_usage": 100000000,  # 100MB
            "agent_count": 0,
            "active_tasks": 0
        }
    )
    
    # Configure realistic behavior with AsyncMock for async methods
    mock.mock_instance.start = AsyncMock()
    mock.mock_instance.stop = AsyncMock()
    mock.mock_instance.delegate_task = AsyncMock(return_value="task-123")
    mock.mock_instance.spawn_agent = AsyncMock(return_value="agent-456")
    mock.mock_instance.shutdown_agent = AsyncMock()
    mock.mock_instance.process_task_queue = AsyncMock()
    mock.mock_instance.monitor_health = AsyncMock()
    mock.mock_instance.get_system_status = MagicMock(return_value={
        "status": "healthy",
        "agents": 0,
        "tasks": 0,
        "uptime": 0.0
    })
    
    return mock.mock_instance


@pytest.fixture(scope="function")
def task_manager_mock():
    """Mock of consolidated TaskManager."""
    mock = ConsolidatedComponentMock(
        component_name="TaskManager",
        public_api=[
            "route_task", "execute_task", "get_queue_status", "process_queue",
            "register_executor", "get_task_history"
        ],
        integration_points=[
            "orchestrator.delegate_task", "agent_manager.assign_agent",
            "workflow_manager.start_workflow"
        ],
        performance_metrics={
            "avg_routing_time": 0.05,
            "queue_length": 0,
            "processing_rate": 100.0,
            "success_rate": 0.95
        }
    )
    
    # Configure async methods
    mock.mock_instance.route_task = AsyncMock()
    mock.mock_instance.execute_task = AsyncMock() 
    mock.mock_instance.process_queue = AsyncMock()
    mock.mock_instance.initialize = AsyncMock()
    
    return mock.mock_instance


@pytest.fixture(scope="function")  
def agent_manager_mock():
    """Mock of consolidated AgentManager."""
    mock = ConsolidatedComponentMock(
        component_name="AgentManager",
        public_api=[
            "create_agent", "destroy_agent", "get_agent", "list_agents",
            "monitor_health", "update_capabilities", "find_suitable_agent"
        ],
        integration_points=[
            "orchestrator.spawn_agent", "orchestrator.shutdown_agent",
            "task_manager.assign_task", "resource_manager.allocate"
        ],
        performance_metrics={
            "agent_spawn_time": 2.0,
            "health_check_interval": 30.0,
            "active_agents": 0,
            "capability_match_time": 0.1
        }
    )
    
    # Configure async methods
    mock.mock_instance.create_agent = AsyncMock()
    mock.mock_instance.destroy_agent = AsyncMock()
    mock.mock_instance.monitor_health = AsyncMock()
    mock.mock_instance.update_capabilities = AsyncMock()
    mock.mock_instance.find_suitable_agent = AsyncMock()
    mock.mock_instance.initialize = AsyncMock()
    
    return mock.mock_instance


@pytest.fixture(scope="function")
def workflow_manager_mock():
    """Mock of consolidated WorkflowManager.""" 
    mock = ConsolidatedComponentMock(
        component_name="WorkflowManager",
        public_api=[
            "create_workflow", "execute_workflow", "pause_workflow", "resume_workflow",
            "get_workflow_status", "validate_dag", "get_execution_history"
        ],
        integration_points=[
            "task_manager.execute_task", "agent_manager.assign_agents",
            "orchestrator.workflow_callback"
        ],
        performance_metrics={
            "workflow_start_time": 0.5,
            "avg_execution_time": 10.0,
            "success_rate": 0.92,
            "active_workflows": 0
        }
    )
    
    # Configure async methods
    mock.mock_instance.create_workflow = AsyncMock()
    mock.mock_instance.execute_workflow = AsyncMock()
    mock.mock_instance.pause_workflow = AsyncMock()
    mock.mock_instance.resume_workflow = AsyncMock()
    mock.mock_instance.initialize = AsyncMock()
    
    return mock.mock_instance


@pytest.fixture(scope="function")
def resource_manager_mock():
    """Mock of consolidated ResourceManager."""
    return ConsolidatedComponentMock(
        component_name="ResourceManager", 
        public_api=[
            "allocate_memory", "release_memory", "create_session", "cleanup_session",
            "monitor_usage", "set_limits", "get_resource_stats"
        ],
        integration_points=[
            "agent_manager.resource_request", "task_manager.resource_check",
            "orchestrator.resource_alert"
        ],
        performance_metrics={
            "allocation_time": 0.1,
            "memory_usage": 200000000,  # 200MB
            "session_count": 0,
            "cleanup_efficiency": 0.98
        }
    ).mock_instance


@pytest.fixture(scope="function")
def communication_manager_mock():
    """Mock of consolidated CommunicationManager."""
    return ConsolidatedComponentMock(
        component_name="CommunicationManager",
        public_api=[
            "broadcast", "subscribe", "unsubscribe", "send_notification",
            "create_channel", "get_message_history", "setup_websocket"
        ],
        integration_points=[
            "orchestrator.broadcast_status", "agent_manager.agent_updates", 
            "task_manager.task_notifications"
        ],
        performance_metrics={
            "message_latency": 0.01,
            "throughput": 1000.0,  # messages/sec
            "active_connections": 0,
            "delivery_success_rate": 0.99
        }
    ).mock_instance


@pytest.fixture(scope="function")
def consolidated_managers_suite(
    task_manager_mock,
    agent_manager_mock,
    workflow_manager_mock,
    resource_manager_mock,
    communication_manager_mock
):
    """Complete suite of consolidated manager mocks."""
    return {
        "task_manager": task_manager_mock,
        "agent_manager": agent_manager_mock,
        "workflow_manager": workflow_manager_mock,
        "resource_manager": resource_manager_mock,
        "communication_manager": communication_manager_mock
    }


@pytest.fixture(scope="function")
def integrated_system_mock(consolidated_orchestrator_mock, consolidated_managers_suite):
    """Mock of fully integrated system with orchestrator and managers."""
    
    class IntegratedSystemMock:
        def __init__(self, orchestrator, managers):
            self.orchestrator = orchestrator
            self.managers = managers
            self.is_running = False
            self.metrics = {}
            
        async def start_system(self):
            """Start the integrated system."""
            self.is_running = True
            # Start orchestrator
            await self.orchestrator.start()
            
            # Initialize all managers
            for manager_name, manager in self.managers.items():
                if hasattr(manager, 'initialize'):
                    await manager.initialize()
            
            return True
        
        async def stop_system(self):
            """Stop the integrated system."""
            self.is_running = False
            
            # Stop all managers first
            for manager_name, manager in self.managers.items():
                if hasattr(manager, 'shutdown'):
                    await manager.shutdown()
            
            # Stop orchestrator last
            await self.orchestrator.stop()
            
            return True
        
        def get_system_health(self):
            """Get overall system health."""
            return {
                "status": "healthy" if self.is_running else "stopped",
                "orchestrator": "running" if self.is_running else "stopped",
                "managers": {
                    name: "running" if self.is_running else "stopped"
                    for name in self.managers.keys()
                },
                "integration_points": len(self.managers) * 3,  # Mock calculation
                "performance_score": 0.95 if self.is_running else 0.0
            }
        
        async def execute_workflow(self, workflow_definition):
            """Execute a test workflow through the integrated system."""
            if not self.is_running:
                raise RuntimeError("System not running")
            
            # Simulate workflow execution across components
            task_id = await self.orchestrator.delegate_task(workflow_definition)
            workflow_id = await self.managers["workflow_manager"].create_workflow(task_id)
            agent_id = await self.managers["agent_manager"].find_suitable_agent(task_id)
            
            return {
                "task_id": task_id,
                "workflow_id": workflow_id, 
                "agent_id": agent_id,
                "status": "started"
            }
    
    return IntegratedSystemMock(consolidated_orchestrator_mock, consolidated_managers_suite)


@pytest.fixture(scope="session")
def consolidation_test_environment():
    """Test environment for consolidation validation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_env = {
            "workspace_dir": temp_dir,
            "config_file": Path(temp_dir) / "test_config.json",
            "logs_dir": Path(temp_dir) / "logs",
            "checkpoints_dir": Path(temp_dir) / "checkpoints"
        }
        
        # Create directories
        test_env["logs_dir"].mkdir(exist_ok=True)
        test_env["checkpoints_dir"].mkdir(exist_ok=True)
        
        # Create test config
        config = {
            "environment": "testing",
            "debug": True,
            "consolidation": {
                "enabled": True,
                "validation_level": "strict",
                "performance_thresholds": {
                    "max_startup_time": 2.0,
                    "max_memory_mb": 500,
                    "min_success_rate": 0.95
                }
            }
        }
        
        with open(test_env["config_file"], "w") as f:
            json.dump(config, f, indent=2)
        
        yield test_env


@pytest.fixture(scope="function")
def performance_monitor():
    """Performance monitoring utilities for consolidation tests."""
    
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.baselines = {}
        
        def set_baseline(self, component: str, metric: str, value: float):
            """Set performance baseline for comparison."""
            if component not in self.baselines:
                self.baselines[component] = {}
            self.baselines[component][metric] = value
        
        def record_metric(self, component: str, metric: str, value: float):
            """Record a performance metric."""
            if component not in self.metrics:
                self.metrics[component] = {}
            if metric not in self.metrics[component]:
                self.metrics[component][metric] = []
            self.metrics[component][metric].append(value)
        
        def get_average(self, component: str, metric: str) -> Optional[float]:
            """Get average value for a metric."""
            if component in self.metrics and metric in self.metrics[component]:
                values = self.metrics[component][metric]
                return sum(values) / len(values) if values else None
            return None
        
        def check_regression(self, component: str, metric: str, threshold: float = 0.1) -> bool:
            """Check if there's a performance regression."""
            if component not in self.baselines or metric not in self.baselines[component]:
                return True  # No baseline, assume OK
            
            baseline = self.baselines[component][metric]
            current = self.get_average(component, metric)
            
            if current is None:
                return False  # No data, assume regression
            
            if baseline > 0:
                regression = (current - baseline) / baseline
                return regression <= threshold
            
            return current <= baseline
        
        def generate_report(self) -> Dict[str, Any]:
            """Generate performance report."""
            report = {
                "components_tested": len(self.metrics),
                "baselines_defined": len(self.baselines),
                "regressions": [],
                "improvements": [],
                "metrics_summary": {}
            }
            
            for component in self.metrics:
                report["metrics_summary"][component] = {}
                for metric in self.metrics[component]:
                    avg_value = self.get_average(component, metric)
                    baseline = self.baselines.get(component, {}).get(metric)
                    
                    report["metrics_summary"][component][metric] = {
                        "current": avg_value,
                        "baseline": baseline,
                        "measurements": len(self.metrics[component][metric])
                    }
                    
                    if baseline and avg_value:
                        change = (avg_value - baseline) / baseline if baseline > 0 else 0
                        if change > 0.1:  # 10% regression threshold
                            report["regressions"].append({
                                "component": component,
                                "metric": metric,
                                "regression": change
                            })
                        elif change < -0.1:  # 10% improvement
                            report["improvements"].append({
                                "component": component,
                                "metric": metric,
                                "improvement": abs(change)
                            })
            
            return report
    
    return PerformanceMonitor()


@pytest.fixture(scope="function")
def consolidation_validator():
    """Validator for consolidation testing."""
    
    class ConsolidationValidator:
        def __init__(self):
            self.test_plan = MANAGER_CONSOLIDATION_TEST_PLAN
        
        async def validate_manager_consolidation(self, manager_name: str, 
                                               mock_manager: Any) -> Dict[str, Any]:
            """Validate a specific manager consolidation."""
            validation_result = {
                "manager": manager_name,
                "api_compatibility": True,
                "functionality_preserved": True, 
                "performance_acceptable": True,
                "integration_points_working": True,
                "errors": [],
                "warnings": []
            }
            
            try:
                # Check API availability
                expected_methods = getattr(mock_manager, '_expected_api', [])
                for method in expected_methods:
                    if not hasattr(mock_manager, method):
                        validation_result["api_compatibility"] = False
                        validation_result["errors"].append(f"Missing API method: {method}")
                
                # Check integration points
                integration_points = getattr(mock_manager, '_integration_points', [])
                for point in integration_points:
                    if not hasattr(mock_manager, point.split('.')[-1]):
                        validation_result["integration_points_working"] = False
                        validation_result["warnings"].append(f"Integration point not found: {point}")
                
                # Overall success
                validation_result["success"] = (
                    validation_result["api_compatibility"] and
                    validation_result["functionality_preserved"] and
                    validation_result["performance_acceptable"] and
                    validation_result["integration_points_working"]
                )
                
            except Exception as e:
                validation_result["success"] = False
                validation_result["errors"].append(f"Validation failed: {str(e)}")
            
            return validation_result
        
        async def validate_system_integration(self, integrated_system: Any) -> Dict[str, Any]:
            """Validate overall system integration."""
            try:
                # Test system startup
                await integrated_system.start_system()
                health = integrated_system.get_system_health()
                
                # Test workflow execution
                test_workflow = {"type": "test", "steps": ["validate", "report"]}
                result = await integrated_system.execute_workflow(test_workflow)
                
                # Test system shutdown
                await integrated_system.stop_system()
                
                return {
                    "integration_test": "passed",
                    "startup": "success",
                    "workflow_execution": "success", 
                    "shutdown": "success",
                    "health_status": health,
                    "workflow_result": result
                }
                
            except Exception as e:
                return {
                    "integration_test": "failed",
                    "error": str(e),
                    "startup": "unknown",
                    "workflow_execution": "unknown",
                    "shutdown": "unknown"
                }
    
    return ConsolidationValidator()


# Quality gate fixtures
@pytest.fixture(scope="function")
def quality_gate_checker():
    """Quality gate checker for consolidation validation."""
    
    class QualityGateChecker:
        def __init__(self):
            self.requirements = {
                "min_test_pass_rate": 0.8,  # 80%
                "max_performance_regression": 0.1,  # 10%
                "required_api_coverage": 0.95,  # 95%
                "max_integration_failures": 2
            }
        
        def check_test_pass_rate(self, passed: int, total: int) -> bool:
            """Check if test pass rate meets requirements."""
            if total == 0:
                return False
            pass_rate = passed / total
            return pass_rate >= self.requirements["min_test_pass_rate"]
        
        def check_performance_regression(self, performance_report: Dict[str, Any]) -> bool:
            """Check if performance regressions are within limits."""
            regressions = performance_report.get("regressions", [])
            max_regression = max([r["regression"] for r in regressions], default=0)
            return max_regression <= self.requirements["max_performance_regression"]
        
        def check_api_coverage(self, expected_apis: List[str], available_apis: List[str]) -> bool:
            """Check API coverage meets requirements.""" 
            if not expected_apis:
                return True
            coverage = len(set(available_apis) & set(expected_apis)) / len(expected_apis)
            return coverage >= self.requirements["required_api_coverage"]
        
        def evaluate_quality_gates(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate all quality gates."""
            gates = {
                "test_pass_rate": self.check_test_pass_rate(
                    test_results.get("passed", 0),
                    test_results.get("total", 1)
                ),
                "performance_regression": self.check_performance_regression(
                    test_results.get("performance_report", {})
                ),
                "api_coverage": self.check_api_coverage(
                    test_results.get("expected_apis", []),
                    test_results.get("available_apis", [])
                )
            }
            
            gates["all_gates_passed"] = all(gates.values())
            gates["consolidation_approved"] = gates["all_gates_passed"]
            
            return gates
    
    return QualityGateChecker()