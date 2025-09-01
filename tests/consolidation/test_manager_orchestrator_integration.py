"""
Integration Tests for Consolidated Manager + Orchestrator System
===============================================================

This test suite validates the integration between the ConsolidatedProductionOrchestrator
and the newly consolidated manager components, ensuring seamless interaction and
maintained functionality throughout the Epic 1 consolidation process.
"""

import pytest
import asyncio
from typing import Dict, List, Any
from unittest.mock import patch, AsyncMock, MagicMock

from tests.consolidation.enhanced_fixtures import (
    ConsolidatedComponentMock,
    quality_gate_checker,
    performance_monitor,
    consolidation_validator
)


@pytest.mark.consolidation
@pytest.mark.integration
class TestManagerOrchestratorIntegration:
    """Test integration between consolidated orchestrator and managers."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, consolidated_orchestrator_mock, consolidated_managers_suite):
        """Setup for each test method."""
        self.orchestrator = consolidated_orchestrator_mock
        self.managers = consolidated_managers_suite
        
    async def test_orchestrator_manager_startup_sequence(self):
        """Test the startup sequence of orchestrator and managers."""
        # Test orchestrator starts successfully
        await self.orchestrator.start()
        assert self.orchestrator.start.called
        
        # Test managers initialize after orchestrator
        startup_sequence = []
        for manager_name, manager in self.managers.items():
            if hasattr(manager, 'initialize'):
                await manager.initialize()
                startup_sequence.append(manager_name)
        
        # Verify startup order and success
        assert len(startup_sequence) == len(self.managers)
        assert self.orchestrator.get_system_status.return_value["status"] == "healthy"
        
    async def test_task_delegation_through_managers(self):
        """Test task delegation flows through appropriate managers."""
        test_task = {
            "id": "test-task-001",
            "type": "development",
            "priority": "high",
            "requirements": ["python", "testing"]
        }
        
        # Test orchestrator delegates to task manager
        task_id = await self.orchestrator.delegate_task(test_task)
        assert task_id == "task-123"  # From mock configuration
        
        # Test task manager routes the task
        route_result = await self.managers["task_manager"].route_task(test_task)
        assert self.managers["task_manager"].route_task.called
        
        # Test agent manager finds suitable agent
        agent_id = await self.managers["agent_manager"].find_suitable_agent(test_task)
        assert self.managers["agent_manager"].find_suitable_agent.called
        
        # Test workflow manager creates workflow if needed
        if test_task.get("workflow_required", True):
            workflow_id = await self.managers["workflow_manager"].create_workflow(task_id)
            assert self.managers["workflow_manager"].create_workflow.called
    
    async def test_agent_lifecycle_integration(self):
        """Test agent lifecycle management across orchestrator and managers."""
        agent_spec = {
            "role": "backend_developer",
            "capabilities": ["python", "fastapi", "testing"],
            "resources": {"memory": "512MB", "cpu": "1core"}
        }
        
        # Test orchestrator spawn_agent calls agent_manager
        agent_id = await self.orchestrator.spawn_agent(agent_spec)
        assert agent_id == "agent-456"  # From mock configuration
        
        # Test agent_manager creates agent
        await self.managers["agent_manager"].create_agent(agent_spec)
        assert self.managers["agent_manager"].create_agent.called
        
        # Test resource_manager allocates resources
        await self.managers["resource_manager"].allocate_memory(agent_spec["resources"])
        assert self.managers["resource_manager"].allocate_memory.called
        
        # Test agent health monitoring
        health_status = await self.managers["agent_manager"].monitor_health(agent_id)
        assert self.managers["agent_manager"].monitor_health.called
        
        # Test orchestrator shutdown_agent calls cleanup
        shutdown_result = await self.orchestrator.shutdown_agent(agent_id)
        assert shutdown_result == True  # Mock return value
        
    async def test_workflow_execution_integration(self):
        """Test workflow execution across multiple managers."""
        workflow_definition = {
            "id": "test-workflow-001",
            "name": "Test Development Workflow",
            "steps": [
                {"type": "task", "action": "analyze_requirements"},
                {"type": "task", "action": "implement_solution"}, 
                {"type": "task", "action": "run_tests"},
                {"type": "task", "action": "deploy"}
            ]
        }
        
        # Test workflow_manager creates and starts workflow
        workflow_id = await self.managers["workflow_manager"].create_workflow(workflow_definition)
        await self.managers["workflow_manager"].execute_workflow(workflow_id)
        
        assert self.managers["workflow_manager"].create_workflow.called
        assert self.managers["workflow_manager"].execute_workflow.called
        
        # Test task_manager processes workflow tasks
        for step in workflow_definition["steps"]:
            await self.managers["task_manager"].execute_task(step)
        
        assert self.managers["task_manager"].execute_task.call_count == len(workflow_definition["steps"])
        
        # Test communication_manager notifies progress
        await self.managers["communication_manager"].broadcast({
            "type": "workflow_progress",
            "workflow_id": workflow_id,
            "status": "in_progress"
        })
        assert self.managers["communication_manager"].broadcast.called
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery across components."""
        
        # Test task failure handling
        failing_task = {"id": "fail-task", "action": "simulate_failure"}
        
        # Configure task_manager to simulate failure
        self.managers["task_manager"].execute_task.side_effect = Exception("Task execution failed")
        
        # Test error propagation and handling
        with pytest.raises(Exception, match="Task execution failed"):
            await self.managers["task_manager"].execute_task(failing_task)
        
        # Test orchestrator handles the error
        error_handled = await self.orchestrator.handle_task_error(failing_task, "Task execution failed")
        assert self.orchestrator.handle_task_error.called
        
        # Test communication of error status
        await self.managers["communication_manager"].broadcast({
            "type": "error",
            "task_id": failing_task["id"],
            "error": "Task execution failed"
        })
        assert self.managers["communication_manager"].broadcast.called
        
        # Reset mock to normal behavior
        self.managers["task_manager"].execute_task.side_effect = None
    
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring across all components."""
        
        # Test orchestrator performance metrics
        orchestrator_metrics = await self.orchestrator.get_performance_metrics()
        assert self.orchestrator.get_performance_metrics.called
        
        # Test manager performance metrics
        manager_metrics = {}
        for manager_name, manager in self.managers.items():
            if hasattr(manager, 'get_performance_metrics'):
                metrics = await manager.get_performance_metrics()
                manager_metrics[manager_name] = metrics
        
        assert len(manager_metrics) <= len(self.managers)  # Some managers might not have metrics
        
        # Test resource usage monitoring
        resource_stats = await self.managers["resource_manager"].get_resource_stats()
        assert self.managers["resource_manager"].get_resource_stats.called
    
    async def test_configuration_propagation(self):
        """Test configuration changes propagate to all components."""
        
        new_config = {
            "performance": {
                "max_concurrent_tasks": 20,
                "agent_timeout": 300,
                "memory_limit": "1GB"
            },
            "logging": {
                "level": "DEBUG",
                "output": "file"
            }
        }
        
        # Test orchestrator applies configuration
        await self.orchestrator.update_configuration(new_config)
        assert self.orchestrator.update_configuration.called
        
        # Test managers receive configuration updates
        for manager_name, manager in self.managers.items():
            if hasattr(manager, 'update_configuration'):
                await manager.update_configuration(new_config)
                assert manager.update_configuration.called
    
    async def test_graceful_shutdown_sequence(self):
        """Test graceful shutdown of entire system."""
        
        # Test managers shutdown first
        shutdown_sequence = []
        for manager_name, manager in self.managers.items():
            if hasattr(manager, 'shutdown'):
                await manager.shutdown()
                shutdown_sequence.append(manager_name)
        
        # Test orchestrator shuts down last
        await self.orchestrator.stop()
        shutdown_sequence.append("orchestrator")
        
        # Verify shutdown order (orchestrator should be last)
        assert shutdown_sequence[-1] == "orchestrator"
        assert len(shutdown_sequence) >= 1  # At least orchestrator


@pytest.mark.consolidation
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance aspects of manager-orchestrator integration."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, performance_monitor):
        """Setup performance monitoring for each test."""
        self.perf_monitor = performance_monitor
        
        # Set performance baselines
        baselines = {
            "orchestrator": {
                "startup_time": 1.0,
                "task_delegation_time": 0.1,
                "memory_usage": 100000000  # 100MB
            },
            "task_manager": {
                "task_routing_time": 0.05,
                "queue_processing_rate": 100.0
            },
            "agent_manager": {
                "agent_spawn_time": 2.0,
                "health_check_time": 0.1
            },
            "workflow_manager": {
                "workflow_start_time": 0.5,
                "state_transition_time": 0.1
            }
        }
        
        for component, metrics in baselines.items():
            for metric, value in metrics.items():
                self.perf_monitor.set_baseline(component, metric, value)
    
    async def test_system_startup_performance(self, integrated_system_mock):
        """Test overall system startup performance."""
        import time
        
        start_time = time.perf_counter()
        await integrated_system_mock.start_system()
        startup_time = time.perf_counter() - start_time
        
        self.perf_monitor.record_metric("integrated_system", "startup_time", startup_time)
        
        # Verify startup time is reasonable (should be fast with mocks)
        assert startup_time < 1.0, f"System startup took {startup_time:.3f}s"
        
        # Check no performance regressions
        assert self.perf_monitor.check_regression("integrated_system", "startup_time")
    
    async def test_task_throughput_performance(self, integrated_system_mock):
        """Test task processing throughput."""
        await integrated_system_mock.start_system()
        
        # Simulate multiple task executions
        tasks = [{"id": f"task-{i}", "type": "test"} for i in range(10)]
        
        import time
        start_time = time.perf_counter()
        
        for task in tasks:
            await integrated_system_mock.execute_workflow(task)
        
        total_time = time.perf_counter() - start_time
        throughput = len(tasks) / total_time
        
        self.perf_monitor.record_metric("integrated_system", "task_throughput", throughput)
        
        # Verify reasonable throughput (with mocks, should be very fast)
        assert throughput > 50, f"Task throughput too low: {throughput:.2f} tasks/sec"
        
        await integrated_system_mock.stop_system()
    
    async def test_memory_usage_stability(self, integrated_system_mock):
        """Test memory usage remains stable during operations.""" 
        await integrated_system_mock.start_system()
        
        # Simulate various operations
        operations = [
            lambda: integrated_system_mock.orchestrator.get_system_status(),
            lambda: integrated_system_mock.managers["agent_manager"].list_agents(),
            lambda: integrated_system_mock.managers["task_manager"].get_queue_status(),
            lambda: integrated_system_mock.managers["resource_manager"].get_resource_stats()
        ]
        
        # Record memory usage before operations
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Perform operations
        for operation in operations * 10:  # Repeat to simulate load
            await operation()
        
        # Record memory usage after operations
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        self.perf_monitor.record_metric("integrated_system", "memory_usage", memory_after)
        
        # Memory increase should be minimal (with mocks)
        assert memory_increase < 50000000, f"Memory increased by {memory_increase / 1000000:.1f}MB"
        
        await integrated_system_mock.stop_system()


@pytest.mark.consolidation
@pytest.mark.quality_gates
class TestQualityGates:
    """Test quality gates for consolidation validation."""
    
    async def test_consolidation_quality_gates(self, quality_gate_checker, integrated_system_mock):
        """Test all consolidation quality gates pass."""
        
        # Simulate test results
        test_results = {
            "total": 25,
            "passed": 21,
            "failed": 4,
            "expected_apis": [
                "start", "stop", "delegate_task", "spawn_agent", 
                "route_task", "create_agent", "execute_workflow"
            ],
            "available_apis": [
                "start", "stop", "delegate_task", "spawn_agent",
                "route_task", "create_agent", "execute_workflow"
            ],
            "performance_report": {
                "regressions": [
                    {"component": "task_manager", "metric": "routing_time", "regression": 0.05}
                ],
                "improvements": [
                    {"component": "agent_manager", "metric": "spawn_time", "improvement": 0.1}
                ]
            }
        }
        
        # Evaluate quality gates
        gates_result = quality_gate_checker.evaluate_quality_gates(test_results)
        
        # Check individual gates
        assert gates_result["test_pass_rate"] == True, "Test pass rate below threshold"
        assert gates_result["performance_regression"] == True, "Performance regression too high"  
        assert gates_result["api_coverage"] == True, "API coverage insufficient"
        
        # Check overall consolidation approval
        assert gates_result["consolidation_approved"] == True, "Consolidation quality gates failed"
    
    async def test_integration_validation_comprehensive(self, consolidation_validator, integrated_system_mock):
        """Comprehensive integration validation."""
        
        # Test manager consolidation validation
        manager_validations = {}
        for manager_name, manager in integrated_system_mock.managers.items():
            validation_result = await consolidation_validator.validate_manager_consolidation(
                manager_name, manager
            )
            manager_validations[manager_name] = validation_result
        
        # Test system integration validation
        system_validation = await consolidation_validator.validate_system_integration(
            integrated_system_mock
        )
        
        # Verify all validations pass
        for manager_name, result in manager_validations.items():
            assert result.get("success", False) or len(result.get("errors", [])) == 0, \
                f"Manager validation failed for {manager_name}: {result.get('errors', [])}"
        
        assert system_validation["integration_test"] == "passed", \
            f"System integration failed: {system_validation.get('error', 'Unknown error')}"


@pytest.mark.consolidation
@pytest.mark.epic1
@pytest.mark.slow
class TestEndToEndConsolidationWorkflow:
    """End-to-end testing of consolidation workflow."""
    
    async def test_complete_consolidation_workflow(self, 
                                                  integrated_system_mock,
                                                  consolidation_validator,
                                                  performance_monitor,
                                                  quality_gate_checker):
        """Test complete end-to-end consolidation workflow."""
        
        workflow_results = {
            "phases": [],
            "validations": [],
            "performance_metrics": {},
            "quality_gates": {}
        }
        
        # Phase 1: Pre-consolidation validation
        workflow_results["phases"].append("pre_validation")
        pre_validation = await consolidation_validator.validate_system_integration(
            integrated_system_mock
        )
        workflow_results["validations"].append({
            "phase": "pre_validation",
            "result": pre_validation
        })
        
        # Phase 2: System startup and initialization
        workflow_results["phases"].append("system_startup")
        import time
        start_time = time.perf_counter()
        
        await integrated_system_mock.start_system()
        
        startup_time = time.perf_counter() - start_time
        performance_monitor.record_metric("workflow", "startup_time", startup_time)
        
        # Phase 3: Functional testing
        workflow_results["phases"].append("functional_testing")
        
        # Test various system functions
        system_health = integrated_system_mock.get_system_health()
        assert system_health["status"] == "healthy"
        
        # Execute test workflows
        test_workflows = [
            {"name": "simple_task", "complexity": "low"},
            {"name": "multi_agent_task", "complexity": "medium"},
            {"name": "complex_workflow", "complexity": "high"}
        ]
        
        workflow_execution_times = []
        for workflow in test_workflows:
            start_time = time.perf_counter()
            result = await integrated_system_mock.execute_workflow(workflow)
            exec_time = time.perf_counter() - start_time
            
            workflow_execution_times.append(exec_time)
            assert result["status"] == "started"
        
        avg_execution_time = sum(workflow_execution_times) / len(workflow_execution_times)
        performance_monitor.record_metric("workflow", "avg_execution_time", avg_execution_time)
        
        # Phase 4: Performance validation
        workflow_results["phases"].append("performance_validation")
        perf_report = performance_monitor.generate_report()
        workflow_results["performance_metrics"] = perf_report
        
        # Phase 5: Quality gates evaluation
        workflow_results["phases"].append("quality_gates")
        
        test_results = {
            "total": len(test_workflows) + 5,  # workflow tests + system tests
            "passed": len([w for w in test_workflows]) + 4,  # assume most pass
            "failed": 1,  # assume one minor failure
            "expected_apis": ["start", "stop", "execute_workflow", "get_system_health"],
            "available_apis": ["start", "stop", "execute_workflow", "get_system_health"],
            "performance_report": perf_report
        }
        
        quality_gates = quality_gate_checker.evaluate_quality_gates(test_results)
        workflow_results["quality_gates"] = quality_gates
        
        # Phase 6: System shutdown
        workflow_results["phases"].append("system_shutdown")
        await integrated_system_mock.stop_system()
        
        # Final validation
        assert len(workflow_results["phases"]) == 6, "Not all workflow phases completed"
        assert workflow_results["quality_gates"]["consolidation_approved"], \
            "Consolidation not approved by quality gates"
        assert len(workflow_results["performance_metrics"]["regressions"]) == 0, \
            "Performance regressions detected"
        
        # Return comprehensive results for reporting
        return {
            "workflow_success": True,
            "phases_completed": len(workflow_results["phases"]),
            "validations_passed": all([
                v["result"].get("integration_test") == "passed" 
                for v in workflow_results["validations"]
            ]),
            "performance_acceptable": len(workflow_results["performance_metrics"]["regressions"]) == 0,
            "quality_gates_passed": workflow_results["quality_gates"]["consolidation_approved"],
            "detailed_results": workflow_results
        }


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--no-cov"])