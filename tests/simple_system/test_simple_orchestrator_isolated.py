"""
Agent Orchestrator Core - Component Isolation Tests
===================================================

Tests the agent orchestrator system in complete isolation from external dependencies.
This validates core business logic, state management, and error handling without
relying on database, Redis, or Anthropic API connections.

Testing Strategy:
- Mock all external dependencies (DB, Redis, Anthropic)
- Test core orchestration logic and state transitions
- Validate error handling and recovery mechanisms
- Ensure proper resource allocation and lifecycle management
- Test agent registration, task assignment, and workflow coordination
"""

import asyncio
import uuid
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.orchestrator import AgentOrchestrator
from app.core.agent_registry import AgentRegistry
from app.core.task_queue import TaskQueue  
from app.core.task_scheduler import TaskScheduler
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType


@pytest.mark.isolation
@pytest.mark.unit
class TestAgentOrchestratorIsolated:
    """Test agent orchestrator core functionality in isolation."""
    
    @pytest.fixture
    async def isolated_orchestrator(
        self,
        mock_database_session,
        mock_redis_streams,
        mock_anthropic_client,
        isolated_agent_config,
        assert_isolated
    ):
        """Create isolated orchestrator instance with all dependencies mocked."""
        
        # Mock all external service dependencies
        with patch('app.core.orchestrator.get_session', return_value=mock_database_session), \
             patch('app.core.orchestrator.get_message_broker', return_value=mock_redis_streams), \
             patch('app.core.orchestrator.AsyncAnthropic', return_value=mock_anthropic_client):
            
            orchestrator = AgentOrchestrator()
            # For isolation testing, we'll just initialize without starting background tasks
            orchestrator.is_running = True
            
            # Assert complete isolation by checking mocks are used
            assert_isolated(orchestrator, {
                "database": mock_database_session,
                "redis": mock_redis_streams,
                "anthropic": mock_anthropic_client
            })
            
            yield orchestrator
            
            # Clean shutdown
            orchestrator.is_running = False
    
    async def test_orchestrator_initialization_isolated(self, isolated_orchestrator):
        """Test orchestrator initializes properly without external dependencies."""
        orchestrator = isolated_orchestrator
        
        # Verify core components are initialized
        assert orchestrator.agent_registry is not None
        assert orchestrator.task_queue is not None
        assert orchestrator.task_scheduler is not None
        
        # Verify state is properly initialized
        assert orchestrator.is_running is True
        assert orchestrator.system_status == "healthy"
        
        # Verify no real connections were made
        assert not hasattr(orchestrator, "_real_db_connection")
        assert not hasattr(orchestrator, "_real_redis_connection")
    
    async def test_agent_registration_lifecycle_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        capture_component_calls
    ):
        """Test complete agent registration lifecycle in isolation."""
        orchestrator = isolated_orchestrator
        agent_config = isolated_agent_config()
        
        # Capture orchestrator method calls
        calls, _ = capture_component_calls(orchestrator, [
            "register_agent", "get_agent_status", "unregister_agent"
        ])
        
        # Test agent registration
        result = await orchestrator.register_agent(**agent_config)
        
        assert result["success"] is True
        assert "agent_id" in result
        agent_id = result["agent_id"]
        
        # Verify agent is registered
        status = await orchestrator.get_agent_status(agent_id)
        assert status == AgentStatus.ACTIVE
        
        # Test agent unregistration
        unregister_result = await orchestrator.unregister_agent(agent_id)
        assert unregister_result["success"] is True
        
        # Verify agent is no longer active
        final_status = await orchestrator.get_agent_status(agent_id)
        assert final_status == AgentStatus.INACTIVE
        
        # Verify method calls were captured
        assert len(calls) == 3
        assert calls[0]["method"] == "register_agent"
        assert calls[1]["method"] == "get_agent_status"
        assert calls[2]["method"] == "unregister_agent"
    
    async def test_task_submission_and_assignment_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test task submission and assignment logic in isolation."""
        orchestrator = isolated_orchestrator
        
        # Register an agent
        agent_config = isolated_agent_config(capabilities=["python", "testing"])
        agent_result = await orchestrator.register_agent(**agent_config)
        agent_id = agent_result["agent_id"]
        
        # Submit a compatible task
        task_config = isolated_task_config(
            required_capabilities=["python"],
            priority="high"
        )
        task_result = await orchestrator.submit_task(**task_config)
        
        assert task_result["success"] is True
        assert "task_id" in task_result
        task_id = task_result["task_id"]
        
        # Wait for task assignment (should happen quickly in isolation)
        await asyncio.sleep(0.1)
        
        # Verify task was assigned
        task_status = await orchestrator.get_task_status(task_id)
        assert task_status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]
        
        # Verify agent has task assigned
        agent_tasks = await orchestrator.get_agent_tasks(agent_id)
        assert len(agent_tasks) > 0
        assert any(task["id"] == task_id for task in agent_tasks)
    
    async def test_multi_agent_coordination_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test multi-agent coordination logic in isolation."""
        orchestrator = isolated_orchestrator
        
        # Register multiple agents with different capabilities
        backend_agent = await orchestrator.register_agent(
            **isolated_agent_config(
                role="backend-engineer",
                capabilities=["python", "fastapi", "postgresql"]
            )
        )
        frontend_agent = await orchestrator.register_agent(
            **isolated_agent_config(
                role="frontend-developer", 
                capabilities=["react", "typescript", "css"]
            )
        )
        qa_agent = await orchestrator.register_agent(
            **isolated_agent_config(
                role="qa-engineer",
                capabilities=["testing", "pytest", "playwright"]
            )
        )
        
        # Submit tasks requiring different capabilities
        backend_task = await orchestrator.submit_task(
            **isolated_task_config(
                title="Backend API Development",
                required_capabilities=["python", "fastapi"],
                priority="high"
            )
        )
        
        frontend_task = await orchestrator.submit_task(
            **isolated_task_config(
                title="Frontend Component Development",
                required_capabilities=["react", "typescript"],
                priority="medium"
            )
        )
        
        qa_task = await orchestrator.submit_task(
            **isolated_task_config(
                title="Test Suite Development",
                required_capabilities=["testing", "pytest"],
                priority="low"
            )
        )
        
        # Wait for task assignment
        await asyncio.sleep(0.2)
        
        # Verify proper task distribution
        backend_tasks = await orchestrator.get_agent_tasks(backend_agent["agent_id"])
        frontend_tasks = await orchestrator.get_agent_tasks(frontend_agent["agent_id"])
        qa_tasks = await orchestrator.get_agent_tasks(qa_agent["agent_id"])
        
        # Each agent should have appropriate tasks
        assert any("Backend API" in task["title"] for task in backend_tasks)
        assert any("Frontend Component" in task["title"] for task in frontend_tasks)
        assert any("Test Suite" in task["title"] for task in qa_tasks)
        
        # Verify load balancing
        total_tasks = len(backend_tasks) + len(frontend_tasks) + len(qa_tasks)
        assert total_tasks == 3  # All tasks should be assigned
    
    async def test_error_handling_and_recovery_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test error handling and recovery mechanisms in isolation."""
        orchestrator = isolated_orchestrator
        
        # Register an agent
        agent_config = isolated_agent_config()
        agent_result = await orchestrator.register_agent(**agent_config)
        agent_id = agent_result["agent_id"]
        
        # Submit a task
        task_config = isolated_task_config()
        task_result = await orchestrator.submit_task(**task_config)
        task_id = task_result["task_id"]
        
        # Simulate agent failure
        await orchestrator.report_agent_failure(agent_id, "Connection timeout")
        
        # Verify agent status changes
        agent_status = await orchestrator.get_agent_status(agent_id)
        assert agent_status == AgentStatus.ERROR
        
        # Verify task is reassigned or queued for retry
        task_status = await orchestrator.get_task_status(task_id)
        assert task_status in [TaskStatus.PENDING, TaskStatus.REASSIGNED, TaskStatus.RETRY]
        
        # Test recovery
        recovery_result = await orchestrator.recover_agent(agent_id)
        assert recovery_result["success"] is True
        
        # Verify agent is active again
        recovered_status = await orchestrator.get_agent_status(agent_id)
        assert recovered_status == AgentStatus.ACTIVE
    
    async def test_resource_allocation_and_limits_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test resource allocation and limit enforcement in isolation."""
        orchestrator = isolated_orchestrator
        
        # Set resource limits
        await orchestrator.set_resource_limits({
            "max_agents": 3,
            "max_concurrent_tasks_per_agent": 2,
            "max_total_concurrent_tasks": 5
        })
        
        # Register agents up to limit
        agents = []
        for i in range(3):
            agent_result = await orchestrator.register_agent(
                **isolated_agent_config(name=f"agent-{i}")
            )
            agents.append(agent_result["agent_id"])
        
        # Try to register one more agent (should fail)
        over_limit_result = await orchestrator.register_agent(
            **isolated_agent_config(name="agent-over-limit")
        )
        assert over_limit_result["success"] is False
        assert "resource limit" in over_limit_result["error"].lower()
        
        # Submit tasks up to concurrent limit
        tasks = []
        for i in range(5):
            task_result = await orchestrator.submit_task(
                **isolated_task_config(title=f"Task {i}")
            )
            tasks.append(task_result["task_id"])
        
        # Submit one more task (should be queued)
        queued_task = await orchestrator.submit_task(
            **isolated_task_config(title="Queued Task")
        )
        
        await asyncio.sleep(0.1)
        
        # Verify resource limits are enforced
        system_status = await orchestrator.get_system_status()
        assert system_status["active_agents"] <= 3
        assert system_status["concurrent_tasks"] <= 5
        
        # Verify extra task is queued
        queued_status = await orchestrator.get_task_status(queued_task["task_id"])
        assert queued_status == TaskStatus.QUEUED
    
    async def test_workflow_coordination_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_workflow_config
    ):
        """Test workflow coordination and dependency management in isolation."""
        orchestrator = isolated_orchestrator
        
        # Register agents
        backend_agent = await orchestrator.register_agent(
            **isolated_agent_config(
                role="backend-engineer",
                capabilities=["python", "fastapi"]
            )
        )
        qa_agent = await orchestrator.register_agent(
            **isolated_agent_config(
                role="qa-engineer", 
                capabilities=["testing", "pytest"]
            )
        )
        
        # Create workflow with dependencies
        workflow_config = isolated_workflow_config(
            name="Feature Development Workflow",
            definition={
                "type": "sequential",
                "steps": [
                    {
                        "id": "implementation",
                        "task_type": "feature_development",
                        "required_capabilities": ["python", "fastapi"],
                        "depends_on": []
                    },
                    {
                        "id": "testing",
                        "task_type": "testing",
                        "required_capabilities": ["testing", "pytest"],
                        "depends_on": ["implementation"]
                    }
                ]
            }
        )
        
        # Submit workflow
        workflow_result = await orchestrator.submit_workflow(**workflow_config)
        assert workflow_result["success"] is True
        workflow_id = workflow_result["workflow_id"]
        
        # Wait for workflow processing
        await asyncio.sleep(0.2)
        
        # Verify workflow status
        workflow_status = await orchestrator.get_workflow_status(workflow_id)
        assert workflow_status["status"] in ["in_progress", "active"]
        
        # Verify step dependencies are respected
        steps = workflow_status["steps"]
        implementation_step = next(s for s in steps if s["id"] == "implementation")
        testing_step = next(s for s in steps if s["id"] == "testing")
        
        # Implementation should start first
        assert implementation_step["status"] in ["assigned", "in_progress", "completed"]
        
        # Testing should wait for implementation (unless implementation is completed)
        if implementation_step["status"] != "completed":
            assert testing_step["status"] in ["pending", "waiting_for_dependencies"]
    
    async def test_performance_monitoring_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test performance monitoring and metrics collection in isolation."""
        orchestrator = isolated_orchestrator
        
        # Register agent
        agent_result = await orchestrator.register_agent(**isolated_agent_config())
        agent_id = agent_result["agent_id"]
        
        # Submit and complete several tasks to generate metrics
        for i in range(5):
            task_result = await orchestrator.submit_task(
                **isolated_task_config(title=f"Perf Test Task {i}")
            )
            task_id = task_result["task_id"]
            
            # Simulate task completion
            await orchestrator.report_task_completion(task_id, {
                "success": True,
                "execution_time": 30.0 + i * 5,
                "output": f"Task {i} completed successfully"
            })
        
        # Get performance metrics
        metrics = await orchestrator.get_performance_metrics()
        
        # Verify metrics are collected
        assert "tasks_completed" in metrics
        assert metrics["tasks_completed"] == 5
        
        assert "average_execution_time" in metrics
        assert metrics["average_execution_time"] > 0
        
        assert "agent_utilization" in metrics
        assert 0 <= metrics["agent_utilization"] <= 1
        
        # Get agent-specific metrics
        agent_metrics = await orchestrator.get_agent_metrics(agent_id)
        assert "tasks_assigned" in agent_metrics
        assert "tasks_completed" in agent_metrics
        assert "average_task_time" in agent_metrics
    
    async def test_system_health_monitoring_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config
    ):
        """Test system health monitoring in isolation."""
        orchestrator = isolated_orchestrator
        
        # Initial health check
        health = await orchestrator.get_system_health()
        assert health["status"] == "healthy"
        assert health["active_agents"] == 0
        assert health["pending_tasks"] == 0
        
        # Register agents and verify health updates
        agents = []
        for i in range(3):
            agent_result = await orchestrator.register_agent(
                **isolated_agent_config(name=f"health-agent-{i}")
            )
            agents.append(agent_result["agent_id"])
        
        updated_health = await orchestrator.get_system_health()
        assert updated_health["active_agents"] == 3
        
        # Simulate system stress
        await orchestrator.simulate_high_load()  # Mock high CPU/memory usage
        
        stressed_health = await orchestrator.get_system_health()
        assert stressed_health["status"] in ["degraded", "warning"]
        assert "load" in stressed_health
        
        # Test health alerts
        alerts = await orchestrator.get_health_alerts()
        assert len(alerts) > 0
        assert any("high load" in alert["message"].lower() for alert in alerts)


@pytest.mark.isolation
@pytest.mark.unit
class TestOrchestratorStateManagement:
    """Test orchestrator state management in isolation."""
    
    async def test_state_persistence_isolated(
        self,
        isolated_orchestrator,
        mock_database_session,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test state persistence without real database."""
        orchestrator = isolated_orchestrator
        
        # Create some state
        agent_result = await orchestrator.register_agent(**isolated_agent_config())
        task_result = await orchestrator.submit_task(**isolated_task_config())
        
        # Verify state is tracked in memory
        state_snapshot = await orchestrator.get_state_snapshot()
        assert len(state_snapshot["agents"]) == 1
        assert len(state_snapshot["tasks"]) == 1
        
        # Test state restoration after restart
        await orchestrator.save_state()
        await orchestrator.shutdown()
        
        # Create new orchestrator instance
        new_orchestrator = AgentOrchestrator()
        await new_orchestrator.initialize()
        await new_orchestrator.restore_state()
        
        # Verify state was restored
        restored_state = await new_orchestrator.get_state_snapshot()
        assert len(restored_state["agents"]) == 1
        assert len(restored_state["tasks"]) == 1
        
        await new_orchestrator.shutdown()
    
    async def test_concurrent_state_modifications_isolated(
        self,
        isolated_orchestrator,
        isolated_agent_config,
        isolated_task_config
    ):
        """Test concurrent state modifications in isolation."""
        orchestrator = isolated_orchestrator
        
        # Simulate concurrent agent registrations
        registration_tasks = []
        for i in range(10):
            task = asyncio.create_task(
                orchestrator.register_agent(**isolated_agent_config(name=f"concurrent-agent-{i}"))
            )
            registration_tasks.append(task)
        
        results = await asyncio.gather(*registration_tasks, return_exceptions=True)
        
        # Verify all registrations succeeded without conflicts
        successful_registrations = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_registrations) == 10
        
        # Verify state consistency
        state = await orchestrator.get_state_snapshot()
        assert len(state["agents"]) == 10
        
        # Test concurrent task submissions
        task_submission_tasks = []
        for i in range(20):
            task = asyncio.create_task(
                orchestrator.submit_task(**isolated_task_config(title=f"concurrent-task-{i}"))
            )
            task_submission_tasks.append(task)
        
        task_results = await asyncio.gather(*task_submission_tasks, return_exceptions=True)
        
        # Verify all task submissions succeeded
        successful_submissions = [r for r in task_results if isinstance(r, dict) and r.get("success")]
        assert len(successful_submissions) == 20