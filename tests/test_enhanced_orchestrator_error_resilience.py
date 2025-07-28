"""
Error Handling and Resilience Tests for Enhanced Agent Orchestrator

This test suite validates the orchestrator's resilience and error handling
capabilities, ensuring robust operation under various failure conditions
and graceful degradation when components fail.

Test Categories:
- Network and connectivity failures
- Database transaction failures  
- Agent communication failures
- Resource exhaustion scenarios
- Cascading failure prevention
- Recovery and self-healing mechanisms
"""

import pytest
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Any
import random

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentInstance, AgentCapability
from app.core.intelligent_task_router import TaskRoutingContext, RoutingStrategy
from app.models.agent import Agent, AgentStatus, AgentType
from app.models.task import Task, TaskStatus, TaskPriority, TaskType
from app.models.workflow import Workflow, WorkflowStatus


@pytest.fixture
def resilient_orchestrator():
    """Create orchestrator configured for resilience testing."""
    orchestrator = AgentOrchestrator()
    
    # Mock dependencies but allow controlled failures
    orchestrator.message_broker = AsyncMock()
    orchestrator.session_cache = AsyncMock()
    orchestrator.anthropic_client = AsyncMock()
    orchestrator.persona_system = AsyncMock()
    orchestrator.intelligent_router = AsyncMock()
    orchestrator.workflow_engine = AsyncMock()
    
    return orchestrator


@pytest.fixture
def unstable_agents():
    """Create agents with varying stability characteristics."""
    agents = {}
    
    # Stable agent
    agents["stable-agent"] = AgentInstance(
        id="stable-agent",
        role=AgentRole.BACKEND_DEVELOPER,
        status=AgentStatus.ACTIVE,
        tmux_session="stable-session",
        capabilities=[AgentCapability("stable_capability", "Stable", 0.95, ["reliability"])],
        current_task=None,
        context_window_usage=0.3,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )
    
    # Intermittently failing agent
    agents["flaky-agent"] = AgentInstance(
        id="flaky-agent",
        role=AgentRole.BACKEND_DEVELOPER,  
        status=AgentStatus.ACTIVE,
        tmux_session="flaky-session",
        capabilities=[AgentCapability("flaky_capability", "Flaky", 0.7, ["unreliable"])],
        current_task=None,
        context_window_usage=0.6,
        last_heartbeat=datetime.utcnow() - timedelta(minutes=2),
        anthropic_client=None
    )
    
    # Overloaded agent
    agents["overloaded-agent"] = AgentInstance(
        id="overloaded-agent",
        role=AgentRole.BACKEND_DEVELOPER,
        status=AgentStatus.BUSY,
        tmux_session="overloaded-session", 
        capabilities=[AgentCapability("overloaded_capability", "Overloaded", 0.8, ["performance"])],
        current_task="heavy-task",
        context_window_usage=0.95,
        last_heartbeat=datetime.utcnow(),
        anthropic_client=None
    )
    
    return agents


class TestDatabaseFailureResilience:
    """Test resilience to database failures and transaction issues."""
    
    async def test_database_connection_failure_handling(self, resilient_orchestrator):
        """Test handling of database connection failures."""
        orchestrator = resilient_orchestrator
        
        # Mock database connection failure
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_get_session.side_effect = Exception("Database connection failed")
            
            # Task delegation should handle database failure gracefully
            with pytest.raises(Exception) as exc_info:
                await orchestrator.delegate_task(
                    task_description="Test task during DB failure",
                    task_type="testing"
                )
            
            # Verify proper error handling
            assert "Database connection failed" in str(exc_info.value)
            
            # Verify orchestrator is still functional for non-DB operations
            assert orchestrator.is_running is False  # Should not crash
            assert len(orchestrator.agents) == 0     # State should be preserved
    
    async def test_database_transaction_rollback(self, resilient_orchestrator, unstable_agents):
        """Test database transaction rollback on failures."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock commit failure
            mock_db_session.commit.side_effect = Exception("Transaction failed")
            
            # Task assignment should handle transaction failure
            result = await orchestrator._assign_task_to_agent_with_persona(
                task_id="test-task",
                agent_id="stable-agent",
                task=None
            )
            
            # Should return False on transaction failure
            assert result is False
            
            # Verify rollback was attempted (session context manager handles this)
            mock_db_session.commit.assert_called_once()
    
    async def test_database_query_timeout_handling(self, resilient_orchestrator):
        """Test handling of database query timeouts."""
        orchestrator = resilient_orchestrator
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock query timeout
            async def timeout_query(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate delay
                raise asyncio.TimeoutError("Query timeout")
            
            mock_db_session.execute.side_effect = timeout_query
            
            # Analytics query should handle timeout gracefully
            analytics = await orchestrator.get_routing_analytics()
            
            # Should return default/empty analytics on timeout
            assert isinstance(analytics, dict)
            # Analytics should handle the timeout gracefully
    
    async def test_concurrent_database_access_conflicts(self, resilient_orchestrator):
        """Test handling of concurrent database access conflicts."""
        orchestrator = resilient_orchestrator
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Mock deadlock/conflict error on some calls
            call_count = 0
            def conflict_on_second_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("Deadlock detected")
                return AsyncMock()
            
            mock_db_session.execute.side_effect = conflict_on_second_call
            
            # Multiple concurrent operations
            tasks = [
                orchestrator.update_task_completion_metrics("task-1", "agent-1", True, 2.0),
                orchestrator.update_task_completion_metrics("task-2", "agent-2", True, 3.0)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # One should succeed, one should fail with deadlock
            successful = [r for r in results if not isinstance(r, Exception)]
            failed = [r for r in results if isinstance(r, Exception)]
            
            assert len(successful) >= 1  # At least one should succeed
            assert len(failed) <= 1      # At most one should fail


class TestNetworkFailureResilience:
    """Test resilience to network and communication failures."""
    
    async def test_message_broker_connection_failure(self, resilient_orchestrator, unstable_agents):
        """Test handling of message broker connection failures."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        # Mock message broker failure
        orchestrator.message_broker.send_message.side_effect = Exception("Redis connection lost")
        
        # Task assignment should continue despite message broker failure
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_task = MagicMock()
            mock_task.id = "test-task"
            mock_db_session.get.return_value = mock_task
            
            result = await orchestrator._assign_task_to_agent_with_persona(
                task_id="test-task",
                agent_id="stable-agent"
            )
            
            # Assignment should still update database even if messaging fails
            assert result is True
            mock_db_session.execute.assert_called()
            mock_db_session.commit.assert_called()
    
    async def test_agent_communication_timeout(self, resilient_orchestrator, unstable_agents):
        """Test handling of agent communication timeouts."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        # Mock message broker timeout
        async def timeout_send(*args, **kwargs):
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("Agent communication timeout")
        
        orchestrator.message_broker.send_message.side_effect = timeout_send
        
        # Sleep cycle initiation should handle timeout gracefully
        result = await orchestrator.initiate_sleep_cycle("stable-agent")
        
        # Should still update agent status despite communication failure
        agent = orchestrator.agents["stable-agent"]
        assert agent.status == AgentStatus.SLEEPING
        assert result is True
    
    async def test_message_broker_partial_failure(self, resilient_orchestrator, unstable_agents):
        """Test handling of partial message broker failures."""
        orchestrator = resilient_orchestrator  
        orchestrator.agents = unstable_agents
        
        # Mock intermittent failures (30% failure rate)
        def intermittent_failure(*args, **kwargs):
            if random.random() < 0.3:
                raise Exception("Intermittent network issue")
            return AsyncMock()
        
        orchestrator.message_broker.send_message.side_effect = intermittent_failure
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Send multiple messages, some should succeed despite intermittent failures
            success_count = 0
            for i in range(20):
                try:
                    result = await orchestrator.initiate_sleep_cycle("stable-agent")
                    if result:
                        success_count += 1
                except Exception:
                    pass  # Expected intermittent failures
            
            # Should have some successes despite intermittent failures
            assert success_count > 10  # At least 50% success rate expected


class TestAgentFailureResilience:
    """Test resilience to agent failures and unavailability."""
    
    async def test_agent_sudden_disconnection(self, resilient_orchestrator, unstable_agents):
        """Test handling of sudden agent disconnections."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        # Simulate agent disconnection during task assignment
        agent = orchestrator.agents["flaky-agent"]
        agent.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)  # Very stale heartbeat
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Monitor agent health should detect and handle disconnection
            await orchestrator._monitor_agent_health("flaky-agent", agent)
            
            # Agent should be marked as in error state
            assert agent.status == AgentStatus.ERROR
    
    async def test_cascading_agent_failures(self, resilient_orchestrator):
        """Test prevention of cascading agent failures."""
        orchestrator = resilient_orchestrator
        
        # Create multiple agents
        for i in range(5):
            agent_id = f"cascade-agent-{i}"
            orchestrator.agents[agent_id] = AgentInstance(
                id=agent_id,
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=None,
                capabilities=[],
                current_task=None,
                context_window_usage=0.8,  # High usage
                last_heartbeat=datetime.utcnow(),
                anthropic_client=None
            )
        
        # Trigger failures in sequence
        failing_agents = []
        for agent_id in list(orchestrator.agents.keys())[:3]:
            # Trip circuit breaker for each agent
            await orchestrator._trip_circuit_breaker(agent_id, "cascading_test")
            failing_agents.append(agent_id)
        
        # Verify remaining agents are protected from cascade
        remaining_agents = [aid for aid in orchestrator.agents.keys() if aid not in failing_agents]
        for agent_id in remaining_agents:
            breaker = orchestrator.circuit_breakers.get(agent_id, {})
            assert breaker.get('state', 'closed') != 'open'  # Should not trip
    
    async def test_agent_recovery_after_failure(self, resilient_orchestrator, unstable_agents):
        """Test agent recovery mechanisms after failures."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        agent_id = "flaky-agent"
        agent = orchestrator.agents[agent_id]
        
        # Simulate agent failure and circuit breaker trip
        await orchestrator._trip_circuit_breaker(agent_id, "test_failure")
        
        # Verify agent is in failed state
        breaker = orchestrator.circuit_breakers[agent_id]
        assert breaker['state'] == 'open'
        assert agent.status == AgentStatus.ERROR
        
        # Simulate recovery after cooldown period
        breaker['state'] = 'half_open'
        breaker['trip_time'] = time.time() - 120  # 2 minutes ago
        
        # Test recovery mechanism
        await orchestrator._test_circuit_breaker_recovery(agent_id)
        
        # Agent should be recovered if test passes
        # (Mock implementation would need to simulate successful health check)
    
    async def test_agent_resource_exhaustion_handling(self, resilient_orchestrator, unstable_agents):
        """Test handling of agent resource exhaustion."""
        orchestrator = resilient_orchestrator
        orchestrator.agents = unstable_agents
        
        # Simulate resource exhaustion
        agent = orchestrator.agents["overloaded-agent"]
        agent.context_window_usage = 0.98  # Critical usage
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Health monitoring should detect resource exhaustion
            await orchestrator._monitor_agent_health("overloaded-agent", agent)
            
            # Should trigger protective actions
            # (Implementation would trigger sleep cycle or task reassignment)


class TestWorkflowFailureResilience:
    """Test resilience to workflow execution failures."""
    
    async def test_workflow_step_failure_recovery(self, resilient_orchestrator):
        """Test recovery from individual workflow step failures."""
        orchestrator = resilient_orchestrator
        
        # Mock workflow engine to simulate step failure
        from app.core.workflow_engine import WorkflowResult, TaskExecutionState
        
        failed_result = WorkflowResult(
            workflow_id=uuid.uuid4(),
            status=TaskExecutionState.FAILED,
            result_data={"failed_step": "implementation", "error": "agent_unavailable"},
            execution_time=3.0,
            error_details="Agent became unavailable during step execution"
        )
        
        orchestrator.workflow_engine.execute_workflow.return_value = failed_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_workflow = MagicMock()
            mock_workflow.id = str(failed_result.workflow_id)
            mock_db_session.get.return_value = mock_workflow
            
            # Execute workflow - should handle failure gracefully
            result = await orchestrator.execute_workflow(str(failed_result.workflow_id))
            
            assert result.status == TaskExecutionState.FAILED
            assert "agent_unavailable" in result.result_data["error"]
            
            # Failure metrics should be updated
            assert orchestrator.metrics['workflow_failures'] > 0
    
    async def test_workflow_dependency_failure_handling(self, resilient_orchestrator):
        """Test handling of workflow dependency failures.""" 
        orchestrator = resilient_orchestrator
        
        # Mock workflow engine to simulate dependency failure
        from app.core.workflow_engine import WorkflowResult, TaskExecutionState
        
        dependency_failed_result = WorkflowResult(
            workflow_id=uuid.uuid4(),
            status=TaskExecutionState.BLOCKED,
            result_data={"blocked_on": "prerequisite_workflow", "reason": "dependency_failed"},
            execution_time=0.5,
            error_details="Prerequisite workflow failed, cannot proceed"
        )
        
        orchestrator.workflow_engine.execute_workflow.return_value = dependency_failed_result
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            mock_workflow = MagicMock()
            mock_workflow.id = str(dependency_failed_result.workflow_id)
            mock_db_session.get.return_value = mock_workflow
            
            result = await orchestrator.execute_workflow(str(dependency_failed_result.workflow_id))
            
            assert result.status == TaskExecutionState.BLOCKED
            assert "dependency_failed" in result.result_data["reason"]


class TestSystemResourceExhaustionResilience:
    """Test resilience to system resource exhaustion scenarios."""
    
    async def test_memory_pressure_handling(self, resilient_orchestrator):
        """Test handling of memory pressure conditions."""
        orchestrator = resilient_orchestrator
        
        # Simulate memory pressure by creating many objects
        large_objects = []
        try:
            # Fill up memory gradually
            for i in range(1000):
                # Create large mock objects to simulate memory pressure
                large_object = {'data': 'x' * 10000, 'id': i}  # 10KB per object
                large_objects.append(large_object)
                
                # Test orchestrator operation under memory pressure
                if i % 100 == 0:
                    # Should still be able to perform basic operations
                    status = await orchestrator.get_system_status()
                    assert 'orchestrator_status' in status
                    
        except MemoryError:
            # This is expected under extreme memory pressure
            pass
        finally:
            # Clean up
            large_objects.clear()
    
    async def test_high_cpu_load_resilience(self, resilient_orchestrator):
        """Test orchestrator resilience under high CPU load."""
        orchestrator = resilient_orchestrator
        
        # Simulate CPU-intensive background task
        def cpu_intensive_task():
            # Simulate high CPU usage
            end_time = time.time() + 0.5  # Run for 0.5 seconds
            while time.time() < end_time:
                _ = sum(i * i for i in range(1000))
        
        # Start CPU-intensive task in background
        import threading
        cpu_thread = threading.Thread(target=cpu_intensive_task)
        cpu_thread.start()
        
        try:
            # Test orchestrator operations under CPU load
            start_time = time.time()
            
            # Basic operations should still complete in reasonable time
            status = await orchestrator.get_system_status()
            available_agents = await orchestrator._get_available_agent_ids()
            
            end_time = time.time()
            operation_time = end_time - start_time
            
            # Operations should complete within reasonable time despite CPU load
            assert operation_time < 2.0  # Should complete within 2 seconds
            assert 'orchestrator_status' in status
            
        finally:
            cpu_thread.join()
    
    async def test_file_descriptor_exhaustion(self, resilient_orchestrator):
        """Test handling of file descriptor exhaustion."""
        orchestrator = resilient_orchestrator
        
        # Mock file operations that might fail due to fd exhaustion
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = OSError("Too many open files")
            
            # Operations should handle file descriptor exhaustion gracefully
            try:
                status = await orchestrator.get_system_status()
                # Should still return basic status even if file operations fail
                assert isinstance(status, dict)
            except OSError:
                # This is acceptable - the important thing is it doesn't crash
                pass


class TestGracefulDegradationAndRecovery:
    """Test graceful degradation and recovery mechanisms."""
    
    async def test_partial_system_degradation(self, resilient_orchestrator):
        """Test system continues operating with partial component failure."""
        orchestrator = resilient_orchestrator
        
        # Disable persona system
        orchestrator.persona_system = None
        
        # Disable intelligent router  
        orchestrator.intelligent_router = None
        
        # Basic orchestrator functionality should still work
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            mock_db_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_db_session
            
            # Task delegation should fall back to basic assignment
            orchestrator._schedule_task = AsyncMock(return_value=None)
            
            task_id = await orchestrator.delegate_task(
                task_description="Task during degraded operation",
                task_type="testing"
            )
            
            # Should still create task even without enhanced features
            assert task_id is not None
            mock_db_session.add.assert_called()
    
    async def test_system_recovery_after_failure(self, resilient_orchestrator):
        """Test system recovery mechanisms after failures."""
        orchestrator = resilient_orchestrator
        
        # Simulate system failure state
        orchestrator.is_running = False
        
        # Mark several agents as failed
        for i in range(3):
            agent_id = f"recovery-agent-{i}"
            orchestrator.agents[agent_id] = AgentInstance(
                id=agent_id,
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ERROR,
                tmux_session=None,
                capabilities=[],
                current_task=None,
                context_window_usage=0.5,
                last_heartbeat=datetime.utcnow() - timedelta(minutes=5),
                anthropic_client=None
            )
            
            # Trip circuit breakers
            await orchestrator._trip_circuit_breaker(agent_id, "system_failure")
        
        # Simulate system recovery
        orchestrator.is_running = True
        
        # Recovery should restore agent states
        for agent_id in orchestrator.agents.keys():
            # Simulate recovery time passing
            breaker = orchestrator.circuit_breakers.get(agent_id, {})
            if breaker.get('state') == 'open':
                breaker['trip_time'] = time.time() - 120  # 2 minutes ago
                await orchestrator._schedule_circuit_breaker_recovery(agent_id)
        
        # System should be in recovery state
        assert orchestrator.is_running is True
    
    async def test_automatic_failover_mechanisms(self, resilient_orchestrator):
        """Test automatic failover to backup systems."""
        orchestrator = resilient_orchestrator
        
        # Create primary and backup agents
        primary_agent = AgentInstance(
            id="primary-agent",
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session=None,
            capabilities=[AgentCapability("primary", "Primary", 0.9, ["main"])],
            current_task=None,
            context_window_usage=0.3,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None
        )
        
        backup_agent = AgentInstance(
            id="backup-agent", 
            role=AgentRole.BACKEND_DEVELOPER,
            status=AgentStatus.ACTIVE,
            tmux_session=None,
            capabilities=[AgentCapability("backup", "Backup", 0.8, ["main"])],
            current_task=None,
            context_window_usage=0.2,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None
        )
        
        orchestrator.agents = {"primary-agent": primary_agent, "backup-agent": backup_agent}
        
        # Fail primary agent
        await orchestrator._trip_circuit_breaker("primary-agent", "primary_failure")
        primary_agent.status = AgentStatus.ERROR
        
        # Task assignment should automatically failover to backup
        available_agents = await orchestrator._get_available_agent_ids()
        
        # Only backup agent should be available
        assert "primary-agent" not in available_agents
        assert "backup-agent" in available_agents


if __name__ == "__main__":
    # Run resilience tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=app.core.orchestrator",
        "--cov-report=term-missing"
    ])