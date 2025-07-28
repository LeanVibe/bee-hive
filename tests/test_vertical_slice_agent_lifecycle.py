"""
Integration tests for Vertical Slice 1.1: Agent Lifecycle

Tests the complete end-to-end agent lifecycle flow including:
- Agent registration with persona assignment
- Task assignment with intelligent routing
- Task execution with hook integration
- Redis messaging and database persistence
- Performance benchmarking (<500ms assignment target)
"""

import asyncio
import pytest
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from sqlalchemy import select, func
from fastapi.testclient import TestClient

from app.core.vertical_slice_orchestrator import VerticalSliceOrchestrator
from app.core.agent_lifecycle_manager import AgentLifecycleManager, AgentRegistrationResult
from app.core.task_execution_engine import TaskExecutionEngine, ExecutionOutcome
from app.core.agent_messaging_service import AgentMessagingService, MessageType
from app.core.agent_lifecycle_hooks import AgentLifecycleHooks, SecurityAction
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskType, TaskPriority
from app.models.observability import AgentEvent
from app.core.database import get_async_session
from app.main import app


@pytest.fixture
async def orchestrator():
    """Create a test orchestrator instance."""
    orchestrator = VerticalSliceOrchestrator()
    await orchestrator.start_system()
    yield orchestrator
    await orchestrator.stop_system()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestAgentLifecycleManager:
    """Test the Agent Lifecycle Manager component."""
    
    @pytest.mark.asyncio
    async def test_agent_registration_with_persona(self, orchestrator):
        """Test agent registration with persona assignment."""
        # Test data
        agent_data = {
            "name": "test_backend_agent",
            "agent_type": AgentType.CLAUDE,
            "role": "backend_developer",
            "capabilities": [
                {
                    "name": "python_development",
                    "description": "Python backend development",
                    "confidence_level": 0.9,
                    "specialization_areas": ["FastAPI", "SQLAlchemy"]
                }
            ],
            "system_prompt": "You are a backend developer specializing in Python."
        }
        
        # Register agent
        start_time = time.time()
        result = await orchestrator.lifecycle_manager.register_agent(**agent_data)
        registration_time_ms = (time.time() - start_time) * 1000
        
        # Assertions
        assert result.success
        assert result.agent_id is not None
        assert result.capabilities_assigned == ["python_development"]
        assert registration_time_ms < 1000  # Should complete in under 1 second
        
        # Verify in database
        async with get_async_session() as db:
            agent_result = await db.execute(select(Agent).where(Agent.id == result.agent_id))
            agent = agent_result.scalar_one_or_none()
            
            assert agent is not None
            assert agent.name == "test_backend_agent"
            assert agent.status == AgentStatus.ACTIVE
            assert agent.role == "backend_developer"
            assert len(agent.capabilities) == 1
            assert agent.capabilities[0]["name"] == "python_development"
    
    @pytest.mark.asyncio
    async def test_agent_deregistration(self, orchestrator):
        """Test agent deregistration and cleanup."""
        # First register an agent
        result = await orchestrator.lifecycle_manager.register_agent(
            name="test_agent_to_deregister",
            agent_type=AgentType.CLAUDE,
            role="qa_engineer"
        )
        assert result.success
        agent_id = result.agent_id
        
        # Deregister agent
        start_time = time.time()
        success = await orchestrator.lifecycle_manager.deregister_agent(agent_id)
        deregistration_time_ms = (time.time() - start_time) * 1000
        
        # Assertions
        assert success
        assert deregistration_time_ms < 500  # Should complete quickly
        
        # Verify agent status in database
        async with get_async_session() as db:
            agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = agent_result.scalar_one_or_none()
            
            assert agent is not None
            assert agent.status == AgentStatus.INACTIVE
    
    @pytest.mark.asyncio
    async def test_task_assignment_performance(self, orchestrator):
        """Test task assignment performance (<500ms target)."""
        # Register multiple agents with different capabilities
        agents = []
        for i, role in enumerate(["backend_developer", "frontend_developer", "qa_engineer"]):
            result = await orchestrator.lifecycle_manager.register_agent(
                name=f"test_agent_{i}",
                role=role,
                capabilities=[{
                    "name": f"{role}_skills",
                    "confidence_level": 0.8,
                    "specialization_areas": [role]
                }]
            )
            assert result.success
            agents.append(result.agent_id)
        
        # Create a test task
        async with get_async_session() as db:
            task = Task(
                title="Test task for assignment",
                description="A test task to measure assignment performance",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                priority=TaskPriority.HIGH,
                required_capabilities=["backend_developer_skills"]
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)
        
        # Test assignment performance
        assignment_times = []
        for _ in range(10):  # Test multiple assignments for consistency
            start_time = time.time()
            result = await orchestrator.lifecycle_manager.assign_task_to_agent(
                task_id=task.id,
                max_assignment_time_ms=500.0
            )
            assignment_time_ms = (time.time() - start_time) * 1000
            assignment_times.append(assignment_time_ms)
            
            # Assertions
            assert result.success
            assert result.agent_id in agents
            assert result.confidence_score > 0.0
            assert assignment_time_ms < 500  # Performance target
            
            # Reset task for next iteration
            async with get_async_session() as db:
                await db.execute(
                    select(Task).where(Task.id == task.id)
                    .values(status=TaskStatus.PENDING, assigned_agent_id=None)
                )
                await db.commit()
        
        # Check average performance
        avg_assignment_time = sum(assignment_times) / len(assignment_times)
        assert avg_assignment_time < 250  # Should average well under target
        print(f"Average assignment time: {avg_assignment_time:.2f}ms")


class TestTaskExecutionEngine:
    """Test the Task Execution Engine component."""
    
    @pytest.mark.asyncio
    async def test_task_execution_lifecycle(self, orchestrator):
        """Test complete task execution lifecycle."""
        # Register an agent
        agent_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_execution_agent",
            role="backend_developer"
        )
        assert agent_result.success
        agent_id = agent_result.agent_id
        
        # Create and assign a task
        async with get_async_session() as db:
            task = Task(
                title="Test execution task",
                description="Task for testing execution lifecycle",
                task_type=TaskType.FEATURE_DEVELOPMENT,
                priority=TaskPriority.MEDIUM,
                estimated_effort=30  # 30 minutes
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)
            
            # Assign task
            task.assign_to_agent(agent_id)
            await db.commit()
        
        # Start task execution
        start_success = await orchestrator.execution_engine.start_task_execution(
            task_id=task.id,
            agent_id=agent_id,
            execution_context={"test_mode": True}
        )
        assert start_success
        
        # Update progress
        progress_success = await orchestrator.execution_engine.update_execution_progress(
            task_id=task.id,
            phase=orchestrator.execution_engine.ExecutionPhase.EXECUTION,
            progress_percentage=50.0,
            metadata={"current_step": "Implementing feature"}
        )
        assert progress_success
        
        # Complete task
        execution_result = await orchestrator.execution_engine.complete_task_execution(
            task_id=task.id,
            outcome=ExecutionOutcome.SUCCESS,
            result_data={
                "implementation": "Feature completed successfully",
                "files_created": ["feature.py", "test_feature.py"],
                "test_results": {"passed": 5, "failed": 0}
            }
        )
        
        # Assertions
        assert execution_result.outcome == ExecutionOutcome.SUCCESS
        assert execution_result.execution_time_ms > 0
        assert "implementation" in execution_result.result_data
        
        # Verify task status in database
        async with get_async_session() as db:
            task_result = await db.execute(select(Task).where(Task.id == task.id))
            updated_task = task_result.scalar_one_or_none()
            
            assert updated_task.status == TaskStatus.COMPLETED
            assert updated_task.completed_at is not None
            assert updated_task.actual_effort is not None


class TestAgentMessagingService:
    """Test the Agent Messaging Service component."""
    
    @pytest.mark.asyncio
    async def test_lifecycle_message_flow(self, orchestrator):
        """Test lifecycle message sending and handling."""
        # Register agents
        agent1_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_sender_agent",
            role="backend_developer"
        )
        agent2_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_receiver_agent", 
            role="frontend_developer"
        )
        assert agent1_result.success and agent2_result.success
        
        # Send lifecycle message
        message_id = await orchestrator.messaging_service.send_lifecycle_message(
            message_type=MessageType.HEARTBEAT_REQUEST,
            from_agent=str(agent1_result.agent_id),
            to_agent=str(agent2_result.agent_id),
            payload={"timestamp": datetime.utcnow().isoformat()},
            priority=orchestrator.messaging_service.MessagePriority.HIGH
        )
        
        assert message_id is not None
        assert len(message_id) > 0
        
        # Verify message metrics
        metrics = await orchestrator.messaging_service.get_messaging_metrics()
        assert metrics["message_counts_by_type"][MessageType.HEARTBEAT_REQUEST.value] >= 1
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, orchestrator):
        """Test broadcast messaging to all agents."""
        # Register multiple agents
        agents = []
        for i in range(3):
            result = await orchestrator.lifecycle_manager.register_agent(
                name=f"broadcast_test_agent_{i}",
                role="qa_engineer"
            )
            assert result.success
            agents.append(result.agent_id)
        
        # Send broadcast message
        message_id = await orchestrator.messaging_service.send_system_shutdown(
            reason="Test shutdown message"
        )
        
        assert len(message_id) > 0
        
        # Verify broadcast was sent
        metrics = await orchestrator.messaging_service.get_messaging_metrics()
        assert "system_shutdown" in str(metrics)


class TestAgentLifecycleHooks:
    """Test the Agent Lifecycle Hooks component."""
    
    @pytest.mark.asyncio
    async def test_pre_tool_hook_execution(self, orchestrator):
        """Test PreToolUse hook execution with security validation."""
        # Register an agent
        agent_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_hook_agent",
            role="backend_developer"
        )
        assert agent_result.success
        agent_id = agent_result.agent_id
        
        # Test safe command
        safe_result = await orchestrator.lifecycle_hooks.execute_pre_tool_hooks(
            agent_id=agent_id,
            session_id=uuid.uuid4(),
            tool_name="python_interpreter",
            parameters={"code": "print('Hello, World!')"},
            metadata={"test_mode": True}
        )
        
        assert safe_result.success
        assert safe_result.security_action == SecurityAction.ALLOW
        assert safe_result.execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_security_validation_blocking(self, orchestrator):
        """Test security validation blocks dangerous commands."""
        # Register an agent
        agent_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_security_agent",
            role="devops_engineer"
        )
        assert agent_result.success
        agent_id = agent_result.agent_id
        
        # Test dangerous command
        dangerous_result = await orchestrator.lifecycle_hooks.execute_pre_tool_hooks(
            agent_id=agent_id,
            session_id=uuid.uuid4(),
            tool_name="bash",
            parameters={"command": "rm -rf /important/data"},
            metadata={"test_mode": True}
        )
        
        assert not dangerous_result.success or dangerous_result.security_action == SecurityAction.BLOCK
        assert dangerous_result.blocked_reason is not None
    
    @pytest.mark.asyncio
    async def test_post_tool_hook_execution(self, orchestrator):
        """Test PostToolUse hook execution with result processing."""
        # Register an agent
        agent_result = await orchestrator.lifecycle_manager.register_agent(
            name="test_post_hook_agent",
            role="qa_engineer"
        )
        assert agent_result.success
        agent_id = agent_result.agent_id
        
        # Execute PostToolUse hook
        post_result = await orchestrator.lifecycle_hooks.execute_post_tool_hooks(
            agent_id=agent_id,
            session_id=uuid.uuid4(),
            tool_name="test_runner",
            parameters={"test_suite": "unit_tests"},
            result={"tests_passed": 10, "tests_failed": 0, "coverage": 95.5},
            success=True,
            execution_time_ms=2500.0,
            metadata={"test_mode": True}
        )
        
        assert post_result.success
        assert post_result.execution_time_ms >= 0


class TestVerticalSliceOrchestrator:
    """Test the complete Vertical Slice Orchestrator."""
    
    @pytest.mark.asyncio
    async def test_complete_lifecycle_demonstration(self, orchestrator):
        """Test the complete lifecycle demonstration."""
        # Run the complete demonstration
        demo_results = await orchestrator.demonstrate_complete_lifecycle()
        
        # Verify demonstration success
        assert demo_results["success"]
        assert len(demo_results["steps_completed"]) > 0
        assert demo_results["duration_seconds"] > 0
        
        # Verify metrics were collected
        assert "metrics" in demo_results
        assert "demonstration" in demo_results["metrics"]
        
        demo_metrics = demo_results["metrics"]["demonstration"]
        assert demo_metrics["agents_registered"] > 0
        assert demo_metrics["tasks_assigned"] > 0
        assert demo_metrics["hooks_executed"] > 0
        
        # Verify performance targets
        if demo_metrics["average_assignment_time_ms"] > 0:
            assert demo_metrics["average_assignment_time_ms"] < 500
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, orchestrator):
        """Test comprehensive system metrics collection."""
        # Get system status
        status = await orchestrator.get_comprehensive_status()
        
        # Verify status structure
        assert "system" in status
        assert "components" in status
        assert "metrics" in status
        
        # Verify system info
        assert status["system"]["is_running"]
        assert status["system"]["start_time"] is not None
        
        # Verify component metrics
        if orchestrator.is_running:
            assert "lifecycle_manager" in status["components"]
            assert "execution_engine" in status["components"]
            assert "messaging_service" in status["components"]
            assert "hooks" in status["components"]
    
    @pytest.mark.asyncio
    async def test_graceful_system_shutdown(self, orchestrator):
        """Test graceful system shutdown."""
        # Verify system is running
        assert orchestrator.is_running
        
        # Stop system
        success = await orchestrator.stop_system()
        
        # Verify shutdown
        assert success
        assert not orchestrator.is_running
        
        # Verify uptime was recorded
        assert orchestrator.metrics.system_uptime_seconds > 0


class TestAPIEndpoints:
    """Test the API endpoints for agent lifecycle."""
    
    def test_register_agent_lifecycle_endpoint(self, client):
        """Test agent registration via API endpoint."""
        agent_data = {
            "name": "api_test_agent",
            "type": "claude",
            "role": "backend_developer",
            "capabilities": [
                {
                    "name": "python_development",
                    "confidence_level": 0.9,
                    "specialization_areas": ["FastAPI"]
                }
            ]
        }
        
        response = client.post("/api/v1/agents/lifecycle/register", json=agent_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"]
        assert "agent_id" in data
        assert "capabilities_assigned" in data
    
    def test_lifecycle_system_start_endpoint(self, client):
        """Test system start via API endpoint."""
        response = client.post("/api/v1/agents/lifecycle/system/start")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "timestamp" in data
    
    def test_system_metrics_endpoint(self, client):
        """Test system metrics via API endpoint."""
        # Start system first
        client.post("/api/v1/agents/lifecycle/system/start")
        
        response = client.get("/api/v1/agents/lifecycle/system/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "metrics" in data
    
    def test_complete_lifecycle_demo_endpoint(self, client):
        """Test complete lifecycle demonstration via API endpoint."""
        response = client.post("/api/v1/agents/lifecycle/demo/complete-flow")
        
        assert response.status_code == 200
        data = response.json()
        assert "demonstration" in data
        assert "timestamp" in data


class TestPerformanceBenchmarks:
    """Test performance benchmarks and targets."""
    
    @pytest.mark.asyncio
    async def test_agent_registration_performance(self, orchestrator):
        """Benchmark agent registration performance."""
        registration_times = []
        
        for i in range(10):
            start_time = time.time()
            result = await orchestrator.lifecycle_manager.register_agent(
                name=f"perf_test_agent_{i}",
                role="backend_developer"
            )
            registration_time = (time.time() - start_time) * 1000
            registration_times.append(registration_time)
            
            assert result.success
            assert registration_time < 2000  # 2 second max
        
        avg_time = sum(registration_times) / len(registration_times)
        max_time = max(registration_times)
        min_time = min(registration_times)
        
        print(f"Registration performance - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms, Min: {min_time:.2f}ms")
        assert avg_time < 1000  # Average under 1 second
    
    @pytest.mark.asyncio
    async def test_task_assignment_performance_benchmark(self, orchestrator):
        """Benchmark task assignment performance against 500ms target."""
        # Register agents
        agents = []
        for i in range(5):
            result = await orchestrator.lifecycle_manager.register_agent(
                name=f"benchmark_agent_{i}",
                role="backend_developer",
                capabilities=[{
                    "name": "python_development",
                    "confidence_level": 0.8 + (i * 0.05),
                    "specialization_areas": ["FastAPI", "SQLAlchemy"]
                }]
            )
            assert result.success
            agents.append(result.agent_id)
        
        # Create tasks
        tasks = []
        async with get_async_session() as db:
            for i in range(20):  # Test with 20 tasks
                task = Task(
                    title=f"Benchmark task {i}",
                    task_type=TaskType.FEATURE_DEVELOPMENT,
                    priority=TaskPriority.HIGH,
                    required_capabilities=["python_development"]
                )
                db.add(task)
                await db.commit()
                await db.refresh(task)
                tasks.append(task.id)
        
        # Benchmark assignments
        assignment_times = []
        successful_assignments = 0
        
        for task_id in tasks:
            start_time = time.time()
            result = await orchestrator.lifecycle_manager.assign_task_to_agent(
                task_id=task_id,
                max_assignment_time_ms=500.0
            )
            assignment_time = (time.time() - start_time) * 1000
            assignment_times.append(assignment_time)
            
            if result.success:
                successful_assignments += 1
                assert assignment_time < 500  # Target performance
        
        # Performance analysis
        avg_time = sum(assignment_times) / len(assignment_times)
        max_time = max(assignment_times)
        min_time = min(assignment_times)
        success_rate = successful_assignments / len(tasks)
        
        print(f"Assignment benchmark - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms, Min: {min_time:.2f}ms")
        print(f"Success rate: {success_rate * 100:.1f}%")
        
        # Assertions
        assert avg_time < 250  # Average well under target
        assert max_time < 500  # All assignments under target
        assert success_rate > 0.95  # 95%+ success rate
    
    @pytest.mark.asyncio
    async def test_hook_execution_performance(self, orchestrator):
        """Benchmark hook execution performance."""
        # Register an agent
        agent_result = await orchestrator.lifecycle_manager.register_agent(
            name="hook_perf_agent",
            role="backend_developer"
        )
        assert agent_result.success
        agent_id = agent_result.agent_id
        
        # Benchmark PreToolUse hooks
        pre_hook_times = []
        for i in range(50):
            start_time = time.time()
            result = await orchestrator.lifecycle_hooks.execute_pre_tool_hooks(
                agent_id=agent_id,
                session_id=uuid.uuid4(),
                tool_name="python_interpreter",
                parameters={"code": f"print('Test {i}')"}
            )
            hook_time = (time.time() - start_time) * 1000
            pre_hook_times.append(hook_time)
            
            assert result.success
            assert hook_time < 100  # Hooks should be very fast
        
        avg_hook_time = sum(pre_hook_times) / len(pre_hook_times)
        print(f"Hook execution performance - Avg: {avg_hook_time:.2f}ms")
        
        assert avg_hook_time < 50  # Average under 50ms


# Performance test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.redis,
    pytest.mark.postgres,
    pytest.mark.performance
]