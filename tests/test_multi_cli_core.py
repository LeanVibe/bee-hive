#!/usr/bin/env python3
"""
Multi-CLI System Core Validation Test

Simple validation test that exercises the core multi-CLI system components
without external dependencies like testcontainers or Redis.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our core components
from app.core.agents.universal_agent_interface import (
    UniversalAgentInterface,
    AgentTask,
    AgentResult,
    ExecutionContext,
    AgentCapability,
    HealthStatus,
    HealthState,
    CapabilityType,
    AgentType,
    TaskStatus
)
from app.core.agents.agent_registry import AgentRegistry
from app.core.agents.models import AgentConfiguration
from app.core.communication.context_preserver import ProductionContextPreserver


class MockCLIAdapter(UniversalAgentInterface):
    """Mock CLI adapter for testing multi-agent coordination."""
    
    def __init__(self, agent_type: AgentType, success_rate: float = 1.0):
        agent_id = f"mock_{agent_type.value}_{str(uuid.uuid4())[:8]}"
        super().__init__(agent_id, agent_type)
        self.success_rate = success_rate
        self.executed_tasks = []
        self.call_count = 0
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Mock task execution with configurable success rate."""
        self.executed_tasks.append(task)
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # Simulate success/failure based on success rate
        import random
        if random.random() <= self.success_rate:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.COMPLETED,
                output_data={
                    "success": True,
                    "mock_output": f"Completed {task.type} task",
                    "execution_time": 0.05,
                    "agent_type": self.agent_type.value
                }
            )
        else:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                status=TaskStatus.FAILED,
                error_message="Mock failure for testing"
            )
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return mock capabilities based on agent type."""
        capabilities_map = {
            AgentType.CLAUDE_CODE: [
                AgentCapability(type=CapabilityType.CODE_ANALYSIS, confidence=0.95, performance_score=0.9),
                AgentCapability(type=CapabilityType.CODE_IMPLEMENTATION, confidence=0.85, performance_score=0.8),
                AgentCapability(type=CapabilityType.CODE_REVIEW, confidence=0.90, performance_score=0.85),
                AgentCapability(type=CapabilityType.DOCUMENTATION, confidence=0.80, performance_score=0.75)
            ],
            AgentType.CURSOR: [
                AgentCapability(type=CapabilityType.CODE_IMPLEMENTATION, confidence=0.90, performance_score=0.95),
                AgentCapability(type=CapabilityType.REFACTORING, confidence=0.85, performance_score=0.85),
                AgentCapability(type=CapabilityType.TESTING, confidence=0.70, performance_score=0.75)
            ],
            AgentType.GITHUB_COPILOT: [
                AgentCapability(type=CapabilityType.CODE_IMPLEMENTATION, confidence=0.80, performance_score=0.90),
                AgentCapability(type=CapabilityType.TESTING, confidence=0.75, performance_score=0.80),
                AgentCapability(type=CapabilityType.CODE_REVIEW, confidence=0.70, performance_score=0.75)
            ]
        }
        return capabilities_map.get(self.agent_type, [AgentCapability(type=CapabilityType.CODE_ANALYSIS, confidence=0.5, performance_score=0.5)])
    
    async def health_check(self) -> HealthStatus:
        """Mock health check."""
        return HealthStatus(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            state=HealthState.HEALTHY,
            response_time_ms=20.0,
            cpu_usage_percent=25.0,
            memory_usage_mb=128.0,
            active_tasks=0,
            completed_tasks=len(self.executed_tasks),
            failed_tasks=0,
            last_activity=datetime.now(),
            error_rate=1.0 - self.success_rate,
            throughput_tasks_per_minute=60.0
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the mock agent."""
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        pass


async def find_and_execute_best_agent(registry: AgentRegistry, task: AgentTask) -> AgentResult:
    """Helper function to find and execute the best agent for a task."""
    # Find best agent
    best_agent_id = await registry.find_best_agent(task)
    
    if not best_agent_id:
        return AgentResult(
            task_id=task.id,
            agent_id="none",
            agent_type=AgentType.PYTHON_AGENT,
            status=TaskStatus.FAILED,
            error_message="No suitable agent found"
        )
    
    # Get the agent
    agent = registry.get_agent(best_agent_id)
    if not agent:
        return AgentResult(
            task_id=task.id,
            agent_id=best_agent_id,
            agent_type=AgentType.PYTHON_AGENT,
            status=TaskStatus.FAILED,
            error_message="Agent not found"
        )
    
    # Execute the task
    return await agent.execute_task(task)


async def test_agent_registration():
    """Test basic agent registration and discovery."""
    logger.info("🧪 Testing Agent Registration...")
    
    registry = AgentRegistry()
    
    # Create test agents
    claude_agent = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=0.95)
    cursor_agent = MockCLIAdapter(AgentType.CURSOR, success_rate=0.90)
    copilot_agent = MockCLIAdapter(AgentType.GITHUB_COPILOT, success_rate=0.85)
    
    # Register agents and measure performance
    registration_times = []
    agents = [claude_agent, cursor_agent, copilot_agent]
    
    for agent in agents:
        start_time = time.time()
        config = AgentConfiguration(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            cli_path=f"/mock/{agent.agent_type.value}",
            working_directory="/tmp/test"
        )
        await registry.register_agent(agent, config)
        registration_time = (time.time() - start_time) * 1000
        registration_times.append(registration_time)
        logger.info(f"  Registered {agent.agent_type.value} in {registration_time:.1f}ms")
    
    # Test discovery (using available methods)
    system_status = registry.get_system_status()
    claude_agents = registry.get_agents_by_type(AgentType.CLAUDE_CODE)
    analysis_agents = registry.get_agents_by_capability(CapabilityType.CODE_ANALYSIS)
    
    # Get system health (use correct method)
    health = {"system_status": system_status.overall_health, "total_agents": system_status.total_agents}
    
    await registry.shutdown()
    
    avg_registration_time = sum(registration_times) / len(registration_times)
    
    results = {
        "agents_registered": len(agents),
        "system_status_retrieved": True,
        "claude_agents_found": len(claude_agents),
        "analysis_capable_agents": len(analysis_agents),
        "avg_registration_time_ms": avg_registration_time,
        "performance_ok": avg_registration_time < 100,
        "system_health": health.get("system_status", "unknown"),
        "success": len(claude_agents) > 0 and avg_registration_time < 100
    }
    
    logger.info(f"  ✅ Registration test: {'PASSED' if results['success'] else 'FAILED'}")
    return results


async def test_task_execution():
    """Test task routing and execution."""
    logger.info("🧪 Testing Task Execution...")
    
    registry = AgentRegistry()
    
    # Register agents
    claude_agent = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=1.0)
    cursor_agent = MockCLIAdapter(AgentType.CURSOR, success_rate=1.0)
    
    claude_config = AgentConfiguration(
        agent_id=claude_agent.agent_id,
        agent_type=claude_agent.agent_type,
        cli_path="/mock/claude_code",
        working_directory="/tmp/test"
    )
    cursor_config = AgentConfiguration(
        agent_id=cursor_agent.agent_id,
        agent_type=cursor_agent.agent_type,
        cli_path="/mock/cursor",
        working_directory="/tmp/test"
    )
    
    await registry.register_agent(claude_agent, claude_config)
    await registry.register_agent(cursor_agent, cursor_config)
    
    # Create test tasks
    tasks = [
        AgentTask(
            id=str(uuid.uuid4()),
            type=CapabilityType.CODE_ANALYSIS,
            description="Test code analysis task",
            input_data={"file": "test.py", "language": "python"},
            requirements=["analysis"],
            context=ExecutionContext(worktree_path="/tmp/test", git_branch="main"),
            priority=1
        ),
        AgentTask(
            id=str(uuid.uuid4()),
            type=CapabilityType.CODE_IMPLEMENTATION,
            description="Test implementation task",
            input_data={"feature": "test_feature", "framework": "fastapi"},
            requirements=["implementation"],
            context=ExecutionContext(worktree_path="/tmp/test", git_branch="main"),
            priority=1
        )
    ]
    
    # Execute tasks
    results = []
    execution_times = []
    
    for task in tasks:
        start_time = time.time()
        result = await find_and_execute_best_agent(registry, task)
        execution_time = (time.time() - start_time) * 1000
        
        results.append(result)
        execution_times.append(execution_time)
        logger.info(f"  Task {task.type.value}: {'✅' if result.status == TaskStatus.COMPLETED else '❌'} ({execution_time:.1f}ms)")
    
    await registry.shutdown()
    
    successful_tasks = [r for r in results if r.status == TaskStatus.COMPLETED]
    avg_execution_time = sum(execution_times) / len(execution_times)
    
    test_results = {
        "tasks_executed": len(tasks),
        "successful_tasks": len(successful_tasks),
        "avg_execution_time_ms": avg_execution_time,
        "performance_ok": avg_execution_time < 500,
        "task_distribution": {
            "claude_tasks": len(claude_agent.executed_tasks),
            "cursor_tasks": len(cursor_agent.executed_tasks)
        },
        "success": len(successful_tasks) == len(tasks) and avg_execution_time < 500
    }
    
    logger.info(f"  ✅ Task execution test: {'PASSED' if test_results['success'] else 'FAILED'}")
    return test_results


async def test_context_preservation():
    """Test context preservation system."""
    logger.info("🧪 Testing Context Preservation...")
    
    context_preserver = ProductionContextPreserver()
    
    # Create comprehensive test context
    test_context = {
        "variables": {
            "project_name": "multi_cli_test",
            "version": "1.0.0",
            "environment": "development",
            "complexity_score": 7.5
        },
        "current_state": {
            "phase": "implementation",
            "completion_percentage": 65.3,
            "active_files": ["main.py", "utils.py", "tests.py"],
            "build_status": "passing",
            "test_coverage": 87.2
        },
        "task_history": [
            {"task": "project_setup", "status": "completed", "duration": 3.2},
            {"task": "core_implementation", "status": "completed", "duration": 15.7},
            {"task": "testing_setup", "status": "in_progress"}
        ],
        "files_created": ["src/main.py", "src/utils.py", "tests/test_main.py"],
        "files_modified": ["requirements.txt", "setup.py"],
        "intermediate_results": [
            {"performance_benchmark": {"avg_response_time": 45, "memory_usage": 128}},
            {"code_quality": {"cyclomatic_complexity": 2.1, "maintainability_index": 85}}
        ]
    }
    
    # Test different compression levels
    compression_results = []
    
    for compression_level in [0, 6, 9]:
        # Package context
        start_time = time.time()
        package = await context_preserver.package_context(
            execution_context=test_context,
            target_agent_type=AgentType.CLAUDE_CODE,
            compression_level=compression_level
        )
        packaging_time = (time.time() - start_time) * 1000
        
        # Validate integrity
        validation = await context_preserver.validate_context_integrity(package)
        
        # Restore context
        start_time = time.time()
        restored_context = await context_preserver.restore_context(package)
        restoration_time = (time.time() - start_time) * 1000
        
        # Verify data integrity
        data_integrity = (
            restored_context["variables"]["project_name"] == test_context["variables"]["project_name"] and
            restored_context["current_state"]["completion_percentage"] == test_context["current_state"]["completion_percentage"] and
            len(restored_context["task_history"]) == len(test_context["task_history"])
        )
        
        compression_results.append({
            "level": compression_level,
            "package_size": package.package_size_bytes,
            "packaging_time_ms": packaging_time,
            "restoration_time_ms": restoration_time,
            "integrity_valid": validation["is_valid"],
            "data_integrity": data_integrity,
            "performance_ok": packaging_time < 1000 and restoration_time < 500
        })
        
        logger.info(f"  Compression {compression_level}: {package.package_size_bytes}B, "
                   f"pack: {packaging_time:.1f}ms, restore: {restoration_time:.1f}ms, "
                   f"valid: {'✅' if validation['is_valid'] and data_integrity else '❌'}")
    
    all_successful = all(r["integrity_valid"] and r["data_integrity"] and r["performance_ok"] 
                        for r in compression_results)
    
    test_results = {
        "compression_tests": compression_results,
        "context_size": len(str(test_context)),
        "all_tests_passed": all_successful,
        "success": all_successful
    }
    
    logger.info(f"  ✅ Context preservation test: {'PASSED' if test_results['success'] else 'FAILED'}")
    return test_results


async def test_multi_agent_coordination():
    """Test multi-agent coordination workflow."""
    logger.info("🧪 Testing Multi-Agent Coordination...")
    
    registry = AgentRegistry()
    context_preserver = ProductionContextPreserver()
    
    # Set up agent ecosystem
    claude_agent = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=0.95)
    cursor_agent = MockCLIAdapter(AgentType.CURSOR, success_rate=0.90)
    copilot_agent = MockCLIAdapter(AgentType.GITHUB_COPILOT, success_rate=0.85)
    
    agents = [claude_agent, cursor_agent, copilot_agent]
    for agent in agents:
        config = AgentConfiguration(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            cli_path=f"/mock/{agent.agent_type.value}",
            working_directory="/tmp/test"
        )
        await registry.register_agent(agent, config)
    
    # Phase 1: Code Analysis
    analysis_task = AgentTask(
        id=str(uuid.uuid4()),
        type=CapabilityType.CODE_ANALYSIS,
        description="Analyze codebase structure",
        input_data={"files": ["main.py", "utils.py"], "analysis_type": "comprehensive"},
        requirements=["static_analysis"],
        context=ExecutionContext(worktree_path="/project/src", git_branch="main"),
        priority=3
    )
    
    analysis_result = await find_and_execute_best_agent(registry, analysis_task)
    
    # Create context for handoff
    handoff_context = {
        "variables": {
            "workflow_phase": "analysis_complete",
            "analysis_findings": analysis_result.output_data,
            "next_phase": "implementation"
        },
        "current_state": {
            "files_analyzed": analysis_task.input_data["files"],
            "analysis_complete": True
        },
        "task_history": [{
            "task": "code_analysis",
            "status": "completed",
            "result": analysis_result.output_data
        }],
        "files_created": ["analysis_report.json"],
        "files_modified": []
    }
    
    # Package context for handoff
    context_package = await context_preserver.package_context(
        execution_context=handoff_context,
        target_agent_type=AgentType.CURSOR,
        compression_level=6
    )
    
    # Phase 2: Implementation (with context)
    restored_context = await context_preserver.restore_context(context_package)
    
    implementation_task = AgentTask(
        id=str(uuid.uuid4()),
        type=CapabilityType.CODE_IMPLEMENTATION,
        description="Implement improvements",
        input_data={
            "analysis_context": restored_context["variables"]["analysis_findings"],
            "target_improvements": ["reduce_complexity"]
        },
        requirements=["implementation"],
        context=ExecutionContext(worktree_path="/project/src", git_branch="main"),
        priority=2
    )
    
    implementation_result = await find_and_execute_best_agent(registry, implementation_task)
    
    # Phase 3: Testing
    testing_task = AgentTask(
        id=str(uuid.uuid4()),
        type=CapabilityType.TESTING,
        description="Create test suite",
        input_data={"test_types": ["unit", "integration"], "coverage_target": 90},
        requirements=["test_generation"],
        context=ExecutionContext(worktree_path="/project/tests", git_branch="main"),
        priority=1
    )
    
    testing_result = await find_and_execute_best_agent(registry, testing_task)
    
    await registry.shutdown()
    
    # Analyze results
    workflow_results = [analysis_result, implementation_result, testing_result]
    successful_phases = [r for r in workflow_results if r.status == TaskStatus.COMPLETED]
    
    agent_utilization = {
        "claude_tasks": len(claude_agent.executed_tasks),
        "cursor_tasks": len(cursor_agent.executed_tasks),
        "copilot_tasks": len(copilot_agent.executed_tasks)
    }
    
    test_results = {
        "total_phases": 3,
        "successful_phases": len(successful_phases),
        "context_handoff_successful": len(restored_context) > 0,
        "agent_utilization": agent_utilization,
        "workflow_complete": len(successful_phases) == 3,
        "success": len(successful_phases) == 3 and len(restored_context) > 0
    }
    
    logger.info(f"  Workflow phases: {len(successful_phases)}/3 completed")
    logger.info(f"  Agent utilization: Claude({agent_utilization['claude_tasks']}) "
               f"Cursor({agent_utilization['cursor_tasks']}) Copilot({agent_utilization['copilot_tasks']})")
    logger.info(f"  ✅ Multi-agent coordination test: {'PASSED' if test_results['success'] else 'FAILED'}")
    
    return test_results


async def test_performance_requirements():
    """Test system performance requirements."""
    logger.info("🧪 Testing Performance Requirements...")
    
    registry = AgentRegistry()
    agent = MockCLIAdapter(AgentType.CLAUDE_CODE)
    
    # Test 1: Agent Registration Performance (<100ms)
    registration_times = []
    for _ in range(5):
        test_agent = MockCLIAdapter(AgentType.CURSOR)
        config = AgentConfiguration(
            agent_id=test_agent.agent_id,
            agent_type=test_agent.agent_type,
            cli_path="/mock/cursor",
            working_directory="/tmp/test"
        )
        start_time = time.time()
        await registry.register_agent(test_agent, config)
        registration_time = (time.time() - start_time) * 1000
        registration_times.append(registration_time)
    
    avg_registration_time = sum(registration_times) / len(registration_times)
    
    # Test 2: Task Execution Performance (<500ms)
    task = AgentTask(
        id=str(uuid.uuid4()),
        type=CapabilityType.CODE_ANALYSIS,
        description="Performance test task",
        input_data={},
        requirements=[],
        context=ExecutionContext(worktree_path="/tmp/test", git_branch="main"),
        priority=1
    )
    
    execution_times = []
    for _ in range(5):
        start_time = time.time()
        result = await agent.execute_task(task)
        execution_time = (time.time() - start_time) * 1000
        execution_times.append(execution_time)
    
    avg_execution_time = sum(execution_times) / len(execution_times)
    
    # Test 3: Context Operations Performance
    context_preserver = ProductionContextPreserver()
    simple_context = {
        "variables": {"test": "value"},
        "current_state": {"status": "active"},
        "task_history": [],
        "files_created": [],
        "files_modified": []
    }
    
    start_time = time.time()
    package = await context_preserver.package_context(
        execution_context=simple_context,
        target_agent_type=AgentType.CLAUDE_CODE
    )
    packaging_time = (time.time() - start_time) * 1000
    
    await registry.shutdown()
    
    performance_tests = {
        "agent_registration": {
            "avg_time_ms": avg_registration_time,
            "target_ms": 100,
            "meets_requirement": avg_registration_time < 100
        },
        "task_execution": {
            "avg_time_ms": avg_execution_time,
            "target_ms": 500,
            "meets_requirement": avg_execution_time < 500
        },
        "context_packaging": {
            "avg_time_ms": packaging_time,
            "target_ms": 1000,
            "meets_requirement": packaging_time < 1000
        }
    }
    
    all_requirements_met = all(test["meets_requirement"] for test in performance_tests.values())
    
    for test_name, test_data in performance_tests.items():
        status = "✅" if test_data["meets_requirement"] else "❌"
        logger.info(f"  {test_name}: {test_data['avg_time_ms']:.1f}ms (target: <{test_data['target_ms']}ms) {status}")
    
    test_results = {
        "performance_tests": performance_tests,
        "all_requirements_met": all_requirements_met,
        "success": all_requirements_met
    }
    
    logger.info(f"  ✅ Performance test: {'PASSED' if test_results['success'] else 'FAILED'}")
    return test_results


async def run_integration_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting Multi-CLI System Core Validation")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Agent Registration & Discovery", test_agent_registration),
        ("Task Routing & Execution", test_task_execution),
        ("Context Preservation", test_context_preservation),
        ("Multi-Agent Coordination", test_multi_agent_coordination),
        ("Performance Requirements", test_performance_requirements)
    ]
    
    results = {}
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        
        try:
            result = await test_func()
            results[test_name] = result
            
            if result["success"]:
                passed_tests += 1
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            logger.error(f"💥 {test_name} - ERROR: {str(e)}")
            results[test_name] = {"success": False, "error": str(e)}
    
    total_time = time.time() - start_time
    success_rate = (passed_tests / len(tests)) * 100
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Tests Passed: {passed_tests}/{len(tests)}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Overall Status: {'✅ PASSED' if success_rate >= 80 else '❌ FAILED'}")
    
    print("\n📋 DETAILED RESULTS:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"  {status} - {test_name}")
    
    if success_rate >= 80:
        print("\n🎉 Multi-CLI System Core Validation Successful!")
        print("\n💡 RECOMMENDATIONS:")
        print("  • System is ready for advanced integration testing")
        print("  • Consider adding Redis and external service integration")
        print("  • Run load testing with higher concurrency")
        print("  • Test with real CLI tools (Claude Code, Cursor, etc.)")
    else:
        print("\n⚠️ Issues Found - Review Failed Tests")
        print("\n🔧 NEXT STEPS:")
        print("  • Debug failed test components")
        print("  • Check performance bottlenecks")
        print("  • Verify component integrations")
        print("  • Re-run tests after fixes")
    
    return success_rate >= 80


if __name__ == "__main__":
    try:
        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Critical test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)