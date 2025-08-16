#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Production Orchestrator
Epic 1, Phase 2 Week 3: Orchestrator Consolidation Validation

This test suite validates the unified production orchestrator functionality:
- Agent registration and management
- Task submission and routing
- Performance monitoring and metrics
- Legacy compatibility
- Error handling and resilience
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

from app.core.production_orchestrator_unified import (
    get_production_orchestrator,
    UnifiedProductionOrchestrator,
    AgentCapability,
    OrchestrationTask,
    TaskPriority,
    AgentState,
    OrchestrationStrategy,
    submit_orchestration_task,
    register_orchestration_agent
)

from app.core.orchestrator_migration_adapter import (
    AgentOrchestrator as LegacyAgentOrchestrator,
    ProductionOrchestrator as LegacyProductionOrchestrator,
    AgentRole,
    TaskPriority as LegacyTaskPriority
)

from app.core.logging_service import get_component_logger

logger = get_component_logger("orchestrator_test")


class OrchestrationTestSuite:
    """Comprehensive test suite for unified production orchestrator"""
    
    def __init__(self):
        self.orchestrator = get_production_orchestrator()
        self.test_results: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "agent_registration_times": [],
            "task_assignment_times": [],
            "orchestration_cycles": []
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all orchestrator tests"""
        logger.info("🧪 Starting comprehensive orchestrator test suite...")
        
        test_methods = [
            self.test_orchestrator_initialization,
            self.test_agent_registration_performance,
            self.test_task_submission_and_routing,
            self.test_intelligent_agent_selection,
            self.test_performance_monitoring,
            self.test_legacy_compatibility,
            self.test_error_handling,
            self.test_concurrent_operations,
            self.test_metrics_collection,
            self.cleanup_test_environment
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"🔬 Running test: {test_name}")
            
            start_time = time.time()
            try:
                result = await test_method()
                execution_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    "status": "PASSED",
                    "execution_time": execution_time,
                    "result": result
                }
                logger.info(f"✅ Test passed: {test_name} ({execution_time:.3f}s)")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.test_results[test_name] = {
                    "status": "FAILED",
                    "execution_time": execution_time,
                    "error": str(e)
                }
                logger.error(f"❌ Test failed: {test_name} ({execution_time:.3f}s) - {str(e)}")
        
        # Generate final report
        return self.generate_test_report()
    
    async def test_orchestrator_initialization(self) -> Dict[str, Any]:
        """Test orchestrator initialization and startup"""
        # Test singleton pattern
        orchestrator1 = get_production_orchestrator()
        orchestrator2 = get_production_orchestrator()
        assert orchestrator1 is orchestrator2, "Orchestrator should be singleton"
        
        # Test startup
        await self.orchestrator.start_orchestrator()
        
        # Verify initial state
        metrics = self.orchestrator.get_orchestration_metrics()
        assert metrics["active_agents"] == 0, "Should start with no agents"
        assert metrics["pending_tasks"] == 0, "Should start with no pending tasks"
        
        return {
            "singleton_verified": True,
            "startup_successful": True,
            "initial_metrics": metrics
        }
    
    async def test_agent_registration_performance(self) -> Dict[str, Any]:
        """Test agent registration performance and functionality"""
        registration_times = []
        agent_ids = []
        
        # Test multiple agent registrations
        for i in range(5):
            start_time = time.time()
            
            capabilities = [
                AgentCapability(
                    capability_type=f"test_capability_{i}",
                    skill_level=5 + i,
                    max_concurrent_tasks=2
                )
            ]
            
            agent_id = f"test_agent_{i}"
            success = await self.orchestrator.register_agent(
                agent_id=agent_id,
                agent_type="test_agent",
                capabilities=capabilities
            )
            
            registration_time = (time.time() - start_time) * 1000  # Convert to ms
            registration_times.append(registration_time)
            
            assert success, f"Agent registration failed for {agent_id}"
            agent_ids.append(agent_id)
            
            # Verify agent status
            agent_status = self.orchestrator.get_agent_status(agent_id)
            assert agent_status is not None, f"Agent status not found for {agent_id}"
            assert agent_status["agent_id"] == agent_id, "Agent ID mismatch"
            assert agent_status["state"] == AgentState.READY.value, "Agent should be ready"
        
        # Performance validation
        avg_registration_time = sum(registration_times) / len(registration_times)
        max_registration_time = max(registration_times)
        
        # Epic requirement: <100ms registration time
        assert avg_registration_time < 100, f"Average registration time {avg_registration_time}ms exceeds 100ms target"
        assert max_registration_time < 200, f"Max registration time {max_registration_time}ms exceeds 200ms threshold"
        
        self.performance_metrics["agent_registration_times"].extend(registration_times)
        
        return {
            "agents_registered": len(agent_ids),
            "avg_registration_time_ms": avg_registration_time,
            "max_registration_time_ms": max_registration_time,
            "performance_target_met": avg_registration_time < 100,
            "agent_ids": agent_ids
        }
    
    async def test_task_submission_and_routing(self) -> Dict[str, Any]:
        """Test task submission and intelligent routing"""
        assignment_times = []
        task_ids = []
        
        # Submit multiple tasks with different priorities
        test_tasks = [
            {
                "task_type": "test_task_low",
                "priority": TaskPriority.LOW,
                "required_capabilities": ["test_capability_0"]
            },
            {
                "task_type": "test_task_high",
                "priority": TaskPriority.HIGH,
                "required_capabilities": ["test_capability_1"]
            },
            {
                "task_type": "test_task_critical",
                "priority": TaskPriority.CRITICAL,
                "required_capabilities": ["test_capability_2"]
            }
        ]
        
        for task_data in test_tasks:
            start_time = time.time()
            
            task = OrchestrationTask(
                task_type=task_data["task_type"],
                priority=task_data["priority"],
                required_capabilities=task_data["required_capabilities"],
                payload={"test_data": "test_value"}
            )
            
            success = await self.orchestrator.submit_task(task)
            assert success, f"Task submission failed for {task.task_id}"
            
            # Wait for task assignment (with timeout)
            timeout = 5.0  # 5 second timeout
            start_wait = time.time()
            task_assigned = False
            
            while time.time() - start_wait < timeout:
                task_status = self.orchestrator.get_task_status(task.task_id)
                if task_status and task_status["status"] == "assigned":
                    task_assigned = True
                    assignment_time = (time.time() - start_time) * 1000  # Convert to ms
                    assignment_times.append(assignment_time)
                    break
                await asyncio.sleep(0.1)
            
            assert task_assigned, f"Task {task.task_id} was not assigned within timeout"
            task_ids.append(task.task_id)
        
        # Performance validation
        avg_assignment_time = sum(assignment_times) / len(assignment_times)
        max_assignment_time = max(assignment_times)
        
        # Epic requirement: <50ms task assignment time
        assert avg_assignment_time < 50, f"Average assignment time {avg_assignment_time}ms exceeds 50ms target"
        
        self.performance_metrics["task_assignment_times"].extend(assignment_times)
        
        return {
            "tasks_submitted": len(task_ids),
            "avg_assignment_time_ms": avg_assignment_time,
            "max_assignment_time_ms": max_assignment_time,
            "performance_target_met": avg_assignment_time < 50,
            "task_ids": task_ids
        }
    
    async def test_intelligent_agent_selection(self) -> Dict[str, Any]:
        """Test intelligent agent selection strategies"""
        # Test different routing strategies
        strategies_tested = []
        
        for strategy in [OrchestrationStrategy.ROUND_ROBIN, 
                        OrchestrationStrategy.LOAD_BALANCED,
                        OrchestrationStrategy.CAPABILITY_BASED,
                        OrchestrationStrategy.INTELLIGENT]:
            
            self.orchestrator.set_orchestration_strategy(strategy)
            
            # Submit a test task
            task = OrchestrationTask(
                task_type="strategy_test",
                priority=TaskPriority.NORMAL,
                required_capabilities=["test_capability_0"],
                payload={"strategy": strategy.value}
            )
            
            success = await self.orchestrator.submit_task(task)
            assert success, f"Task submission failed for strategy {strategy.value}"
            
            # Wait for assignment
            await asyncio.sleep(0.5)
            
            task_status = self.orchestrator.get_task_status(task.task_id)
            assert task_status is not None, f"Task status not found for strategy {strategy.value}"
            
            strategies_tested.append({
                "strategy": strategy.value,
                "task_id": task.task_id,
                "assigned": task_status.get("status") == "assigned"
            })
        
        return {
            "strategies_tested": len(strategies_tested),
            "results": strategies_tested
        }
    
    async def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring and metrics collection"""
        # Get initial metrics
        initial_metrics = self.orchestrator.get_orchestration_metrics()
        
        # Simulate some activity by updating heartbeats
        for i in range(3):
            await self.orchestrator.update_agent_heartbeat(f"test_agent_{i}", context_usage=0.3)
        
        # Wait for metrics update
        await asyncio.sleep(1)
        
        # Get updated metrics
        updated_metrics = self.orchestrator.get_orchestration_metrics()
        
        # Verify metrics structure
        required_metrics = [
            "timestamp", "active_agents", "pending_tasks", "tasks_processed",
            "system_health_score", "average_agent_registration_time",
            "average_task_assignment_time"
        ]
        
        for metric in required_metrics:
            assert metric in updated_metrics, f"Required metric '{metric}' missing from metrics"
        
        # Verify metrics are updated
        assert updated_metrics["active_agents"] >= initial_metrics["active_agents"], "Active agents should not decrease"
        assert 0 <= updated_metrics["system_health_score"] <= 1, "System health score should be between 0 and 1"
        
        return {
            "metrics_structure_valid": True,
            "initial_active_agents": initial_metrics["active_agents"],
            "updated_active_agents": updated_metrics["active_agents"],
            "system_health_score": updated_metrics["system_health_score"],
            "metrics_keys": list(updated_metrics.keys())
        }
    
    async def test_legacy_compatibility(self) -> Dict[str, Any]:
        """Test legacy orchestrator compatibility"""
        # Test legacy AgentOrchestrator wrapper
        legacy_orchestrator = LegacyAgentOrchestrator()
        await legacy_orchestrator.start()
        
        # Test legacy agent spawning
        agent_id = await legacy_orchestrator.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["backend_development", "api_design"]
        )
        
        assert agent_id, "Legacy agent spawning should return agent ID"
        
        # Test legacy task delegation
        task_id = await legacy_orchestrator.delegate_task(
            task_data={"type": "code_review", "payload": "test"},
            priority=LegacyTaskPriority.HIGH
        )
        
        assert task_id, "Legacy task delegation should return task ID"
        
        # Test legacy status retrieval
        agent_status = legacy_orchestrator.get_agent_status(agent_id)
        assert agent_status is not None, "Legacy agent status should be retrievable"
        
        stats = legacy_orchestrator.get_orchestration_stats()
        assert "total_agents" in stats, "Legacy stats should include total_agents"
        
        await legacy_orchestrator.stop()
        
        return {
            "legacy_agent_spawned": bool(agent_id),
            "legacy_task_delegated": bool(task_id),
            "legacy_status_retrieved": agent_status is not None,
            "legacy_stats_available": "total_agents" in stats,
            "agent_id": agent_id,
            "task_id": task_id
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and resilience"""
        error_scenarios = []
        
        # Test invalid agent registration
        try:
            success = await self.orchestrator.register_agent(
                agent_id="",  # Invalid empty ID
                agent_type="test",
                capabilities=[]
            )
            error_scenarios.append({"scenario": "empty_agent_id", "handled": not success})
        except Exception as e:
            error_scenarios.append({"scenario": "empty_agent_id", "handled": True, "error": str(e)})
        
        # Test duplicate agent registration
        try:
            # Register same agent twice
            await self.orchestrator.register_agent(
                agent_id="test_agent_0",  # This should already exist
                agent_type="test",
                capabilities=[AgentCapability("test", 5)]
            )
            error_scenarios.append({"scenario": "duplicate_agent", "handled": True})
        except Exception as e:
            error_scenarios.append({"scenario": "duplicate_agent", "handled": True, "error": str(e)})
        
        # Test invalid task submission
        try:
            invalid_task = OrchestrationTask(
                task_id="",  # Invalid empty task ID
                task_type="",  # Invalid empty task type
                required_capabilities=None  # Invalid capabilities
            )
            success = await self.orchestrator.submit_task(invalid_task)
            error_scenarios.append({"scenario": "invalid_task", "handled": not success})
        except Exception as e:
            error_scenarios.append({"scenario": "invalid_task", "handled": True, "error": str(e)})
        
        return {
            "error_scenarios_tested": len(error_scenarios),
            "scenarios": error_scenarios,
            "all_errors_handled": all(scenario.get("handled", False) for scenario in error_scenarios)
        }
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent operations and thread safety"""
        concurrent_tasks = []
        
        # Create multiple concurrent agent registrations
        for i in range(10):
            task = self.orchestrator.register_agent(
                agent_id=f"concurrent_agent_{i}",
                agent_type="concurrent_test",
                capabilities=[AgentCapability(f"concurrent_capability_{i}", 5)]
            )
            concurrent_tasks.append(task)
        
        # Execute all registrations concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Count successful registrations
        successful_registrations = sum(1 for result in results if result is True)
        
        # Test concurrent task submissions
        task_submission_tasks = []
        for i in range(5):
            task = OrchestrationTask(
                task_type=f"concurrent_task_{i}",
                required_capabilities=[f"concurrent_capability_{i}"],
                payload={"concurrent_test": True}
            )
            task_submission_tasks.append(self.orchestrator.submit_task(task))
        
        task_results = await asyncio.gather(*task_submission_tasks, return_exceptions=True)
        successful_submissions = sum(1 for result in task_results if result is True)
        
        return {
            "concurrent_registrations": len(concurrent_tasks),
            "successful_registrations": successful_registrations,
            "concurrent_submissions": len(task_submission_tasks),
            "successful_submissions": successful_submissions,
            "total_execution_time": execution_time,
            "registration_success_rate": successful_registrations / len(concurrent_tasks),
            "submission_success_rate": successful_submissions / len(task_submission_tasks)
        }
    
    async def test_metrics_collection(self) -> Dict[str, Any]:
        """Test comprehensive metrics collection"""
        # Force metrics update
        await asyncio.sleep(1)
        
        metrics = self.orchestrator.get_orchestration_metrics()
        
        # Verify all expected metrics are present and valid
        metric_validations = {}
        
        # Numeric metrics validation
        numeric_metrics = [
            "tasks_processed", "tasks_failed", "agents_spawned", "agents_terminated",
            "active_agents", "busy_agents", "pending_tasks", "active_tasks",
            "cpu_usage", "memory_usage", "system_health_score"
        ]
        
        for metric in numeric_metrics:
            if metric in metrics:
                value = metrics[metric]
                metric_validations[metric] = {
                    "present": True,
                    "numeric": isinstance(value, (int, float)),
                    "valid_range": 0 <= value <= 100 if metric.endswith("_usage") or metric == "system_health_score" else value >= 0
                }
            else:
                metric_validations[metric] = {"present": False}
        
        # String metrics validation
        string_metrics = ["timestamp", "orchestration_strategy"]
        
        for metric in string_metrics:
            if metric in metrics:
                metric_validations[metric] = {
                    "present": True,
                    "valid": isinstance(metrics[metric], str) and len(metrics[metric]) > 0
                }
            else:
                metric_validations[metric] = {"present": False}
        
        return {
            "metrics_count": len(metrics),
            "metric_validations": metric_validations,
            "all_metrics_valid": all(
                validation.get("present", False) and 
                validation.get("numeric", validation.get("valid", True))
                for validation in metric_validations.values()
            ),
            "sample_metrics": {key: metrics.get(key) for key in list(metrics.keys())[:10]}
        }
    
    async def cleanup_test_environment(self) -> Dict[str, Any]:
        """Clean up test environment"""
        # Get all agents
        all_agents = self.orchestrator.get_all_agents_status()
        
        # Deregister test agents
        cleanup_count = 0
        for agent_id in list(all_agents.keys()):
            if agent_id.startswith(("test_agent_", "concurrent_agent_")):
                success = await self.orchestrator.deregister_agent(agent_id)
                if success:
                    cleanup_count += 1
        
        # Stop orchestrator
        await self.orchestrator.stop_orchestrator()
        
        return {
            "agents_cleaned": cleanup_count,
            "orchestrator_stopped": True,
            "cleanup_successful": True
        }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        
        # Calculate performance statistics
        performance_stats = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                performance_stats[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests,
                "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
            },
            "performance_metrics": performance_stats,
            "detailed_results": self.test_results,
            "epic_requirements_validation": {
                "agent_registration_under_100ms": (
                    performance_stats.get("agent_registration_times", {}).get("average", 999) < 100
                ),
                "task_assignment_under_50ms": (
                    performance_stats.get("task_assignment_times", {}).get("average", 999) < 50
                ),
                "unified_orchestrator_functional": passed_tests >= 8,
                "legacy_compatibility_maintained": (
                    self.test_results.get("test_legacy_compatibility", {}).get("status") == "PASSED"
                )
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return report


async def main():
    """Run the comprehensive orchestrator test suite"""
    print("🚀 Epic 1, Phase 2 Week 3: Unified Production Orchestrator Test Suite")
    print("=" * 80)
    
    test_suite = OrchestrationTestSuite()
    
    try:
        report = await test_suite.run_all_tests()
        
        # Print summary
        summary = report["test_summary"]
        performance = report["performance_metrics"]
        epic_validation = report["epic_requirements_validation"]
        
        print(f"\n📊 TEST SUMMARY")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall Status: {summary['overall_status']}")
        
        print(f"\n⚡ PERFORMANCE METRICS")
        if "agent_registration_times" in performance:
            reg_stats = performance["agent_registration_times"]
            print(f"Agent Registration - Avg: {reg_stats['average']:.1f}ms, Max: {reg_stats['max']:.1f}ms")
        
        if "task_assignment_times" in performance:
            task_stats = performance["task_assignment_times"]
            print(f"Task Assignment - Avg: {task_stats['average']:.1f}ms, Max: {task_stats['max']:.1f}ms")
        
        print(f"\n🎯 EPIC REQUIREMENTS VALIDATION")
        print(f"Agent Registration <100ms: {'✅' if epic_validation['agent_registration_under_100ms'] else '❌'}")
        print(f"Task Assignment <50ms: {'✅' if epic_validation['task_assignment_under_50ms'] else '❌'}")
        print(f"Unified Orchestrator Functional: {'✅' if epic_validation['unified_orchestrator_functional'] else '❌'}")
        print(f"Legacy Compatibility Maintained: {'✅' if epic_validation['legacy_compatibility_maintained'] else '❌'}")
        
        # Save detailed report
        with open("orchestrator_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: orchestrator_test_report.json")
        
        if summary["overall_status"] == "PASSED":
            print(f"\n🎉 ALL TESTS PASSED! Unified Production Orchestrator is ready for production.")
            return True
        else:
            print(f"\n❌ SOME TESTS FAILED. Review the detailed report for issues.")
            return False
            
    except Exception as e:
        print(f"❌ Test suite execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)