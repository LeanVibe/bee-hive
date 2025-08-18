#!/usr/bin/env python3
"""
Multi-CLI Integration Tests

Comprehensive integration testing for the complete multi-CLI agent coordination system.
Tests the full pipeline from task assignment through agent coordination to completion.
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Test framework imports
from testcontainers.redis import RedisContainer

# System imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.agents.universal_agent_interface import (
    UniversalAgentInterface,
    AgentTask,
    AgentResult,
    ExecutionContext,
    CapabilityType,
    AgentType,
    TaskStatus
)
from app.core.agents.agent_registry import AgentRegistry
from app.core.agents.adapters.claude_code_adapter import ClaudeCodeAdapter
from app.core.orchestration.universal_orchestrator import ProductionUniversalOrchestrator
from app.core.communication.multi_cli_protocol import ProductionMultiCLIProtocol
from app.core.communication.context_preserver import ProductionContextPreserver
from app.core.isolation.worktree_manager import WorktreeManager

# Test configuration
TEST_TIMEOUT = 30  # seconds
INTEGRATION_TEST_CONFIG = {
    "redis_url": "redis://localhost:6379/15",  # Test database
    "git_workspace": "/tmp/test_workspace",
    "agent_timeout": 10.0,
    "max_concurrent_agents": 5
}


class MockCLIAdapter(UniversalAgentInterface):
    """Mock CLI adapter for testing multi-agent coordination."""
    
    def __init__(self, agent_type: AgentType, success_rate: float = 1.0):
        super().__init__(agent_type)
        self.success_rate = success_rate
        self.executed_tasks = []
        self.call_count = 0
    
    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Mock task execution with configurable success rate."""
        self.executed_tasks.append(task)
        self.call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate success/failure based on success rate
        import random
        if random.random() <= self.success_rate:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.COMPLETED,
                result={
                    "success": True,
                    "mock_output": f"Completed {task.task_type} task",
                    "execution_time": 0.1
                },
                execution_time=0.1,
                metadata={"mock": True}
            )
        else:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                status=TaskStatus.FAILED,
                error="Mock failure for testing",
                execution_time=0.1,
                metadata={"mock": True}
            )
    
    async def get_capabilities(self) -> List[CapabilityType]:
        """Return mock capabilities."""
        return [
            CapabilityType.CODE_ANALYSIS,
            CapabilityType.CODE_IMPLEMENTATION,
            CapabilityType.TESTING
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": "healthy",
            "response_time": 0.05,
            "last_task_time": datetime.now().isoformat(),
            "total_tasks": len(self.executed_tasks),
            "mock": True
        }
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize mock adapter."""
        return True
    
    async def shutdown(self) -> None:
        """Shutdown mock adapter."""
        pass


class TestMultiCLIIntegration:
    """Integration tests for multi-CLI system."""
    
    @pytest.fixture(scope="class")
    async def redis_container(self):
        """Start Redis container for testing."""
        with RedisContainer("redis:7-alpine") as redis:
            yield redis
    
    @pytest.fixture
    async def test_components(self, redis_container):
        """Initialize all system components for testing."""
        # Configuration
        config = {
            **INTEGRATION_TEST_CONFIG,
            "redis_url": redis_container.get_connection_url()
        }
        
        # Initialize components
        agent_registry = AgentRegistry()
        orchestrator = ProductionUniversalOrchestrator(config)
        multi_cli_protocol = ProductionMultiCLIProtocol("test-protocol")
        context_preserver = ProductionContextPreserver()
        worktree_system = WorktreeManager(config["git_workspace"])
        
        # Create test agents
        claude_mock = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=0.9)
        cursor_mock = MockCLIAdapter(AgentType.CURSOR, success_rate=0.8)
        copilot_mock = MockCLIAdapter(AgentType.GITHUB_COPILOT, success_rate=0.85)
        
        # Register agents
        await agent_registry.register_agent(claude_mock)
        await agent_registry.register_agent(cursor_mock)
        await agent_registry.register_agent(copilot_mock)
        
        components = {
            "registry": agent_registry,
            "orchestrator": orchestrator,
            "protocol": multi_cli_protocol,
            "context_preserver": context_preserver,
            "worktree": worktree_system,
            "agents": {
                "claude": claude_mock,
                "cursor": cursor_mock,
                "copilot": copilot_mock
            },
            "config": config
        }
        
        yield components
        
        # Cleanup
        await agent_registry.shutdown()
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_single_agent_task_execution(self, test_components):
        """Test basic single-agent task execution through the system."""
        registry = test_components["registry"]
        claude_agent = test_components["agents"]["claude"]
        
        # Create test task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.CODE_ANALYSIS,
            description="Analyze code for integration test",
            parameters={"file_path": "/test/example.py"},
            requirements=["python"],
            context=ExecutionContext(
                workspace_path="/tmp/test",
                environment_vars={},
                resource_limits={"memory_mb": 512, "timeout_seconds": 30}
            ),
            priority=1,
            timeout_seconds=10
        )
        
        # Execute through registry
        result = await claude_agent.execute_task(task)
        
        # Validate result
        assert result.task_id == task.task_id
        assert result.status == TaskStatus.COMPLETED
        assert result.result["success"] is True
        assert "mock_output" in result.result
        
        # Verify task was tracked
        assert len(claude_agent.executed_tasks) == 1
        assert claude_agent.executed_tasks[0].task_id == task.task_id
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 2)
    async def test_multi_agent_coordination(self, test_components):
        """Test coordination between multiple agents."""
        registry = test_components["registry"]
        protocol = test_components["protocol"]
        
        # Create tasks for different agent types
        tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.CODE_ANALYSIS,
                description="Code analysis task",
                parameters={"language": "python"},
                requirements=["analysis"],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1
            ),
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.CODE_IMPLEMENTATION,
                description="Code implementation task",
                parameters={"feature": "new_api"},
                requirements=["implementation"],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1
            ),
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.TESTING,
                description="Testing task",
                parameters={"test_type": "unit"},
                requirements=["testing"],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1
            )
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[registry.find_and_execute_best_agent(task) for task in tasks],
            return_exceptions=True
        )
        
        # Validate all tasks completed successfully
        successful_results = [r for r in results if isinstance(r, AgentResult)]
        assert len(successful_results) == 3
        
        for result in successful_results:
            assert result.status == TaskStatus.COMPLETED
            assert result.result["success"] is True
        
        # Verify task distribution across agents
        total_tasks_executed = sum(
            len(agent.executed_tasks) 
            for agent in test_components["agents"].values()
        )
        assert total_tasks_executed == 3
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_context_preservation_handoff(self, test_components):
        """Test context preservation during agent handoffs."""
        context_preserver = test_components["context_preserver"]
        claude_agent = test_components["agents"]["claude"]
        cursor_agent = test_components["agents"]["cursor"]
        
        # Create execution context
        execution_context = {
            "variables": {
                "project_name": "test_project",
                "current_phase": "implementation",
                "progress": 0.6
            },
            "current_state": {
                "active_files": ["main.py", "tests.py"],
                "build_status": "passing"
            },
            "task_history": [
                {"task": "setup", "status": "completed", "duration": 5.2},
                {"task": "implement_core", "status": "in_progress"}
            ],
            "files_created": ["main.py", "config.py"],
            "files_modified": ["requirements.txt"],
            "intermediate_results": [
                {"test_coverage": 85.4},
                {"performance_score": 92.1}
            ]
        }
        
        # Package context for handoff
        package = await context_preserver.package_context(
            execution_context=execution_context,
            target_agent_type=AgentType.CURSOR,
            compression_level=6
        )
        
        # Validate package integrity
        validation = await context_preserver.validate_context_integrity(package)
        assert validation["is_valid"] is True
        assert validation["validation_checks_passed"] == 8
        
        # Restore context
        restored_context = await context_preserver.restore_context(package)
        
        # Verify context preservation
        assert restored_context["variables"]["project_name"] == "test_project"
        assert restored_context["current_state"]["build_status"] == "passing"
        assert len(restored_context["task_history"]) == 2
        assert len(restored_context["files_created"]) == 2
        
        # Verify compression effectiveness
        assert package.metadata["compression_ratio"] < 1.0
        assert package.metadata["packaging_time_ms"] < 1000
        assert restored_context["restoration_metadata"]["restoration_time_ms"] < 500
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 3)
    async def test_workflow_coordination_with_failures(self, test_components):
        """Test workflow coordination with agent failures and recovery."""
        registry = test_components["registry"]
        
        # Create agent with low success rate to trigger failures
        unreliable_agent = MockCLIAdapter(AgentType.GEMINI_CLI, success_rate=0.3)
        await registry.register_agent(unreliable_agent)
        
        # Create tasks that may fail
        tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.CODE_ANALYSIS,
                description=f"Analysis task {i}",
                parameters={"retry_test": True},
                requirements=["analysis"],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1,
                max_retries=3
            )
            for i in range(5)
        ]
        
        # Execute tasks with retry logic
        results = []
        for task in tasks:
            max_attempts = task.max_retries + 1
            for attempt in range(max_attempts):
                try:
                    result = await registry.find_and_execute_best_agent(task)
                    if result.status == TaskStatus.COMPLETED:
                        results.append(result)
                        break
                    elif attempt == max_attempts - 1:
                        # Final attempt failed
                        results.append(result)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        # Record final failure
                        results.append(AgentResult(
                            task_id=task.task_id,
                            agent_id="unknown",
                            status=TaskStatus.FAILED,
                            error=str(e),
                            execution_time=0
                        ))
                
                await asyncio.sleep(0.1)  # Brief delay between retries
        
        # Analyze results
        completed_tasks = [r for r in results if r.status == TaskStatus.COMPLETED]
        failed_tasks = [r for r in results if r.status == TaskStatus.FAILED]
        
        # We should have some successes despite low success rate due to retries
        assert len(completed_tasks) >= 1, "At least one task should succeed with retries"
        assert len(results) == 5, "Should have results for all tasks"
        
        print(f"Completed: {len(completed_tasks)}, Failed: {len(failed_tasks)}")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_agent_health_monitoring(self, test_components):
        """Test agent health monitoring and registry management."""
        registry = test_components["registry"]
        agents = test_components["agents"]
        
        # Get initial health status
        health_report = await registry.get_system_health()
        
        # Validate health report structure
        assert "total_agents" in health_report
        assert "healthy_agents" in health_report
        assert "agent_details" in health_report
        assert "system_status" in health_report
        
        # Verify all agents are initially healthy
        assert health_report["healthy_agents"] == health_report["total_agents"]
        assert health_report["system_status"] == "healthy"
        
        # Test individual agent health checks
        for agent_name, agent in agents.items():
            health = await agent.health_check()
            assert health["status"] == "healthy"
            assert "response_time" in health
            assert "total_tasks" in health
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_performance_requirements(self, test_components):
        """Test system performance meets requirements."""
        registry = test_components["registry"]
        claude_agent = test_components["agents"]["claude"]
        
        # Test agent registration performance
        new_agent = MockCLIAdapter(AgentType.CURSOR)
        
        start_time = asyncio.get_event_loop().time()
        await registry.register_agent(new_agent)
        registration_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        assert registration_time < 100, f"Agent registration took {registration_time}ms (>100ms)"
        
        # Test task execution performance
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.CODE_ANALYSIS,
            description="Performance test task",
            parameters={},
            requirements=[],
            context=ExecutionContext(workspace_path="/tmp/test"),
            priority=1
        )
        
        start_time = asyncio.get_event_loop().time()
        result = await claude_agent.execute_task(task)
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        assert execution_time < 500, f"Task execution took {execution_time}ms (>500ms)"
        assert result.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 2)
    async def test_concurrent_agent_handling(self, test_components):
        """Test system handling of concurrent agents and tasks."""
        registry = test_components["registry"]
        
        # Create multiple concurrent tasks
        num_concurrent_tasks = 10
        tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.CODE_ANALYSIS,
                description=f"Concurrent task {i}",
                parameters={"task_number": i},
                requirements=[],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1
            )
            for i in range(num_concurrent_tasks)
        ]
        
        # Execute all tasks concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[registry.find_and_execute_best_agent(task) for task in tasks],
            return_exceptions=True
        )
        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Validate results
        successful_results = [
            r for r in results 
            if isinstance(r, AgentResult) and r.status == TaskStatus.COMPLETED
        ]
        
        assert len(successful_results) >= 8, f"Only {len(successful_results)}/{num_concurrent_tasks} tasks succeeded"
        assert total_time < 2000, f"Concurrent execution took {total_time}ms (>2s)"
        
        # Verify load distribution
        agent_task_counts = {}
        for agent_name, agent in test_components["agents"].items():
            agent_task_counts[agent_name] = len(agent.executed_tasks)
        
        # Tasks should be distributed across agents
        agents_with_tasks = sum(1 for count in agent_task_counts.values() if count > 0)
        assert agents_with_tasks >= 2, "Tasks should be distributed across multiple agents"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_system_recovery_after_failure(self, test_components):
        """Test system recovery capabilities after component failures."""
        registry = test_components["registry"]
        agents = test_components["agents"]
        
        # Simulate agent failure by replacing with failing mock
        original_claude = agents["claude"]
        failing_agent = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=0.0)
        
        # Replace agent in registry
        await registry.unregister_agent(original_claude.agent_id)
        await registry.register_agent(failing_agent)
        
        # Create test task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.CODE_ANALYSIS,
            description="Recovery test task",
            parameters={},
            requirements=[],
            context=ExecutionContext(workspace_path="/tmp/test"),
            priority=1
        )
        
        # Task should fail with failing agent
        result = await failing_agent.execute_task(task)
        assert result.status == TaskStatus.FAILED
        
        # Restore healthy agent
        await registry.unregister_agent(failing_agent.agent_id)
        healthy_agent = MockCLIAdapter(AgentType.CLAUDE_CODE, success_rate=1.0)
        await registry.register_agent(healthy_agent)
        
        # Task should now succeed
        recovery_result = await healthy_agent.execute_task(task)
        assert recovery_result.status == TaskStatus.COMPLETED
        
        # Verify system health is restored
        health_report = await registry.get_system_health()
        assert health_report["system_status"] == "healthy"


class TestMultiCLIRealWorld:
    """Real-world scenario tests for multi-CLI coordination."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT * 2)
    async def test_code_review_workflow(self, test_components):
        """Test a complete code review workflow across multiple agents."""
        registry = test_components["registry"]
        context_preserver = test_components["context_preserver"]
        
        # Phase 1: Code Analysis (Claude Code)
        analysis_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.CODE_ANALYSIS,
            description="Analyze codebase for review",
            parameters={
                "files": ["main.py", "utils.py", "tests.py"],
                "focus": ["complexity", "maintainability", "security"]
            },
            requirements=["static_analysis", "security_scan"],
            context=ExecutionContext(workspace_path="/project/src"),
            priority=2
        )
        
        analysis_result = await registry.find_and_execute_best_agent(analysis_task)
        assert analysis_result.status == TaskStatus.COMPLETED
        
        # Create context for handoff
        analysis_context = {
            "variables": {
                "review_phase": "analysis_complete",
                "files_analyzed": 3,
                "issues_found": 2
            },
            "current_state": {
                "analysis_results": analysis_result.result,
                "next_phase": "implementation_review"
            },
            "task_history": [{
                "task": "code_analysis",
                "status": "completed",
                "findings": ["complexity_warning", "security_issue"]
            }],
            "files_created": [],
            "files_modified": ["analysis_report.json"]
        }
        
        # Phase 2: Implementation Review (Cursor)
        context_package = await context_preserver.package_context(
            execution_context=analysis_context,
            target_agent_type=AgentType.CURSOR,
            compression_level=6
        )
        
        restored_context = await context_preserver.restore_context(context_package)
        
        implementation_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.CODE_REVIEW,
            description="Review implementation based on analysis",
            parameters={
                "previous_analysis": restored_context["current_state"]["analysis_results"],
                "focus_areas": restored_context["task_history"][0]["findings"]
            },
            requirements=["code_review", "refactoring_suggestions"],
            context=ExecutionContext(workspace_path="/project/src"),
            priority=2
        )
        
        review_result = await registry.find_and_execute_best_agent(implementation_task)
        assert review_result.status == TaskStatus.COMPLETED
        
        # Phase 3: Testing Recommendations (GitHub Copilot)
        testing_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=CapabilityType.TESTING,
            description="Generate testing strategy",
            parameters={
                "code_issues": restored_context["task_history"][0]["findings"],
                "implementation_feedback": review_result.result
            },
            requirements=["test_generation", "coverage_analysis"],
            context=ExecutionContext(workspace_path="/project/tests"),
            priority=2
        )
        
        testing_result = await registry.find_and_execute_best_agent(testing_task)
        assert testing_result.status == TaskStatus.COMPLETED
        
        # Verify complete workflow
        assert all(result.status == TaskStatus.COMPLETED for result in [
            analysis_result, review_result, testing_result
        ])
        
        # Verify context preservation worked
        assert restored_context["variables"]["review_phase"] == "analysis_complete"
        assert restored_context["variables"]["issues_found"] == 2
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_TIMEOUT)
    async def test_load_balancing_under_pressure(self, test_components):
        """Test load balancing when system is under pressure."""
        registry = test_components["registry"]
        
        # Create high load scenario
        num_tasks = 20
        tasks = [
            AgentTask(
                task_id=str(uuid.uuid4()),
                task_type=CapabilityType.CODE_ANALYSIS,
                description=f"Load test task {i}",
                parameters={"load_test": True, "task_id": i},
                requirements=[],
                context=ExecutionContext(workspace_path="/tmp/test"),
                priority=1
            )
            for i in range(num_tasks)
        ]
        
        # Execute with high concurrency
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[registry.find_and_execute_best_agent(task) for task in tasks],
            return_exceptions=True
        )
        execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Analyze load distribution
        agent_loads = {}
        for agent_name, agent in test_components["agents"].items():
            agent_loads[agent_name] = len(agent.executed_tasks)
        
        # Verify results
        successful_results = [
            r for r in results 
            if isinstance(r, AgentResult) and r.status == TaskStatus.COMPLETED
        ]
        
        success_rate = len(successful_results) / num_tasks
        assert success_rate >= 0.8, f"Success rate {success_rate} too low under load"
        assert execution_time < 5000, f"Load test took {execution_time}ms (>5s)"
        
        # Verify load distribution
        max_load = max(agent_loads.values())
        min_load = min(agent_loads.values())
        load_imbalance = max_load - min_load
        
        # Load should be reasonably balanced
        assert load_imbalance <= num_tasks // 2, f"Load imbalance too high: {load_imbalance}"


if __name__ == "__main__":
    import pytest
    import sys
    
    # Run integration tests
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-x"  # Stop on first failure for debugging
    ])
    
    sys.exit(exit_code)