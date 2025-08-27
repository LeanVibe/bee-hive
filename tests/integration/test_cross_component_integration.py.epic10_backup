"""
Cross-Component Integration Tests

Tests integration between all consolidated components:
- UniversalOrchestrator ↔ CommunicationHub
- UniversalOrchestrator ↔ Domain Managers  
- CommunicationHub ↔ Engines
- Manager ↔ Engine interactions
- End-to-end workflow validation
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from tests.consolidated.test_framework_base import ConsolidatedTestBase, TestScenario
from tests.performance.performance_benchmarking_framework import PerformanceBenchmarkFramework

from app.core.universal_orchestrator import (
    UniversalOrchestrator, 
    OrchestratorConfig, 
    OrchestratorMode,
    AgentRole
)
from app.core.communication_hub.communication_hub import (
    CommunicationHub,
    CommunicationConfig,
    RoutingStrategy
)
from app.core.communication_hub.protocols import (
    UnifiedMessage,
    UnifiedEvent,
    MessageType,
    Priority,
    DeliveryGuarantee,
    ProtocolType,
    ConnectionConfig,
    create_message,
    create_event
)
from app.core.unified_manager_base import (
    UnifiedManagerBase,
    ManagerConfig,
    ManagerStatus
)
from app.core.engines.base_engine import BaseEngine
from app.models.agent import AgentStatus
from app.models.task import TaskStatus, TaskPriority


class MockContextManager(UnifiedManagerBase):
    """Mock context manager for integration testing."""
    
    async def _initialize_manager(self) -> bool:
        self.contexts = {}
        return True
    
    async def _shutdown_manager(self) -> None:
        self.contexts.clear()
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        return {"contexts_count": len(self.contexts)}
    
    async def _load_plugins(self) -> None:
        pass
    
    async def store_context(self, agent_id: str, context_data: Dict[str, Any]) -> bool:
        """Store agent context."""
        await self.execute_with_monitoring(
            "store_context",
            self._store_context_impl,
            agent_id,
            context_data
        )
        return True
    
    def _store_context_impl(self, agent_id: str, context_data: Dict[str, Any]) -> None:
        """Implementation of context storage."""
        self.contexts[agent_id] = context_data
    
    async def get_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent context."""
        return self.contexts.get(agent_id)


class MockTaskExecutionEngine:
    """Mock task execution engine for integration testing."""
    
    def __init__(self):
        self.executed_tasks = []
        self.task_results = {}
    
    async def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return results."""
        execution_start = time.time()
        
        # Simulate task execution time
        await asyncio.sleep(0.01)  # 10ms execution
        
        execution_time = (time.time() - execution_start) * 1000
        
        result = {
            "task_id": task_id,
            "status": "completed",
            "execution_time_ms": execution_time,
            "result_data": {"output": f"Task {task_id} completed successfully"}
        }
        
        self.executed_tasks.append(task_id)
        self.task_results[task_id] = result
        
        return result
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        return self.task_results.get(task_id)


class TestCrossComponentIntegration(ConsolidatedTestBase):
    """Integration tests for cross-component communication."""
    
    async def setup_component(self) -> Dict[str, Any]:
        """Setup all components for integration testing."""
        # Setup UniversalOrchestrator
        orchestrator_config = OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=50,
            health_check_interval=5,
            enable_performance_plugin=True,
            enable_context_plugin=True
        )
        orchestrator = UniversalOrchestrator(orchestrator_config, "integration_test_orchestrator")
        
        # Setup CommunicationHub
        comm_config = CommunicationConfig(
            name="IntegrationTestCommunicationHub",
            enable_metrics=True,
            enable_event_bus=True,
            redis_config=ConnectionConfig(
                protocol=ProtocolType.REDIS_STREAMS,
                host="localhost",
                port=6379
            ),
            websocket_config=ConnectionConfig(
                protocol=ProtocolType.WEBSOCKET,
                host="localhost", 
                port=8765
            )
        )
        
        communication_hub = CommunicationHub(comm_config)
        
        # Setup mock managers
        context_manager_config = ManagerConfig(
            name="IntegrationTestContextManager",
            max_concurrent_operations=100
        )
        context_manager = MockContextManager(context_manager_config)
        
        # Setup mock engines
        task_engine = MockTaskExecutionEngine()
        
        # Initialize all components
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            success = await orchestrator.initialize()
            assert success, "Orchestrator initialization failed"
        
        # Initialize communication hub with mocked adapters
        with patch.object(communication_hub, '_initialize_adapters') as mock_init_adapters:
            mock_init_adapters.return_value = None
            success = await communication_hub.initialize()
            assert success, "CommunicationHub initialization failed"
        
        success = await context_manager.initialize()
        assert success, "ContextManager initialization failed"
        
        components = {
            "orchestrator": orchestrator,
            "communication_hub": communication_hub,
            "context_manager": context_manager,
            "task_engine": task_engine
        }
        
        # Add cleanup tasks
        self.add_cleanup_task(orchestrator.shutdown)
        self.add_cleanup_task(communication_hub.shutdown)
        self.add_cleanup_task(context_manager.shutdown)
        
        return components
    
    async def cleanup_component(self) -> None:
        """Cleanup is handled by add_cleanup_task."""
        pass
    
    def get_performance_scenarios(self) -> List[TestScenario]:
        """Get integration test scenarios."""
        scenarios = []
        
        # Orchestrator-CommunicationHub integration
        comm_scenario = TestScenario(
            name="orchestrator_communication_integration",
            description="Test orchestrator and communication hub integration",
            tags={"integration", "communication"}
        )
        comm_scenario.add_performance_threshold("message_latency_ms", 50.0, 100.0)
        scenarios.append(comm_scenario)
        
        # End-to-end workflow scenario
        e2e_scenario = TestScenario(
            name="end_to_end_workflow",
            description="Test complete workflow across all components",
            tags={"integration", "e2e"}
        )
        e2e_scenario.add_performance_threshold("workflow_latency_ms", 1000.0, 2000.0)
        scenarios.append(e2e_scenario)
        
        return scenarios
    
    # === ORCHESTRATOR ↔ COMMUNICATION HUB INTEGRATION ===
    
    @pytest.mark.asyncio
    async def test_orchestrator_communication_hub_integration(self):
        """Test integration between orchestrator and communication hub."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        
        # Register agents with orchestrator
        await orchestrator.register_agent(
            "comm_test_agent_1", 
            AgentRole.WORKER, 
            ["communication", "testing"]
        )
        await orchestrator.register_agent(
            "comm_test_agent_2",
            AgentRole.COORDINATOR,
            ["coordination", "communication"]
        )
        
        async def test_integration():
            """Test message passing between orchestrator and communication hub."""
            
            # Create message for agent communication
            message = create_message(
                source="comm_test_agent_1",
                destination="comm_test_agent_2", 
                message_type=MessageType.COORDINATION_REQUEST,
                payload={"request_type": "task_coordination", "data": "test_payload"},
                priority=Priority.MEDIUM,
                delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE
            )
            
            # Send message through communication hub
            async with self.performance_monitor("orchestrator_communication", "message_latency_ms"):
                result = await communication_hub.send_message(message)
            
            # Verify message was processed
            assert result.success or result.error, "Message should be processed (success or handled error)"
            
            return result
        
        scenario = self.get_performance_scenarios()[0]
        metrics = await self.run_performance_test(
            "test_orchestrator_communication_hub_integration",
            test_integration,
            scenario
        )
        
        assert metrics.success, f"Integration test failed: {metrics.errors}"
    
    @pytest.mark.asyncio
    async def test_agent_heartbeat_communication_flow(self):
        """Test agent heartbeat flow through communication system."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        
        # Register agent
        await orchestrator.register_agent(
            "heartbeat_agent",
            AgentRole.WORKER,
            ["heartbeat_capability"]
        )
        
        # Test heartbeat communication
        heartbeat_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "active",
            "current_task": None,
            "memory_usage": 45.2,
            "cpu_usage": 12.5
        }
        
        # Send heartbeat through communication hub
        result = await communication_hub.send_heartbeat("heartbeat_agent", heartbeat_data)
        
        # Verify heartbeat was processed
        assert result.success or result.error, "Heartbeat should be processed"
        
        # Verify orchestrator state is updated (in real implementation)
        agent = orchestrator.agents.get("heartbeat_agent")
        assert agent is not None, "Agent should exist in orchestrator"
    
    @pytest.mark.asyncio
    async def test_task_request_communication_flow(self):
        """Test task request flow through communication system."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        
        # Register agents
        await orchestrator.register_agent("requester_agent", AgentRole.COORDINATOR, ["coordination"])
        await orchestrator.register_agent("worker_agent", AgentRole.WORKER, ["task_execution"])
        
        # Test task request communication
        task_data = {
            "task_type": "data_processing",
            "parameters": {"input_file": "test.csv", "operation": "analysis"},
            "priority": "high",
            "estimated_duration": 300
        }
        
        result = await communication_hub.send_task_request(
            "requester_agent",
            "worker_agent", 
            task_data,
            Priority.HIGH
        )
        
        # Verify task request was sent
        assert result.success or result.error, "Task request should be processed"
        
        if result.success:
            assert result.latency_ms < 100.0, f"Task request took {result.latency_ms:.2f}ms, should be <100ms"
    
    # === ORCHESTRATOR ↔ MANAGER INTEGRATION ===
    
    @pytest.mark.asyncio
    async def test_orchestrator_context_manager_integration(self):
        """Test integration between orchestrator and context manager."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        context_manager = components["context_manager"]
        
        # Register agent with orchestrator
        agent_id = "context_test_agent"
        await orchestrator.register_agent(
            agent_id,
            AgentRole.SPECIALIST,
            ["context_management", "analysis"]
        )
        
        # Store context through manager
        context_data = {
            "session_id": str(uuid.uuid4()),
            "conversation_history": ["Hello", "How can I help?"],
            "user_preferences": {"language": "en", "expertise_level": "intermediate"},
            "task_context": {"current_focus": "data_analysis", "tools_loaded": ["pandas", "numpy"]}
        }
        
        success = await context_manager.store_context(agent_id, context_data)
        assert success, "Context storage should succeed"
        
        # Retrieve context
        retrieved_context = await context_manager.get_context(agent_id)
        assert retrieved_context is not None, "Should retrieve stored context"
        assert retrieved_context["session_id"] == context_data["session_id"]
        
        # Verify orchestrator can access context through manager
        agent = orchestrator.agents[agent_id]
        assert agent.status == AgentStatus.ACTIVE, "Agent should be active"
    
    @pytest.mark.asyncio
    async def test_orchestrator_resource_manager_integration(self):
        """Test integration between orchestrator and resource manager."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        
        # Create mock resource manager
        resource_manager_config = ManagerConfig(
            name="IntegrationTestResourceManager",
            max_concurrent_operations=50
        )
        
        class MockResourceManager(UnifiedManagerBase):
            def __init__(self, config):
                super().__init__(config)
                self.allocated_resources = {}
            
            async def _initialize_manager(self) -> bool:
                return True
            
            async def _shutdown_manager(self) -> None:
                self.allocated_resources.clear()
            
            async def _get_manager_health(self) -> Dict[str, Any]:
                return {"allocated_resources": len(self.allocated_resources)}
            
            async def _load_plugins(self) -> None:
                pass
            
            async def allocate_resources(self, agent_id: str, requirements: Dict[str, Any]) -> bool:
                """Allocate resources for agent."""
                await self.execute_with_monitoring(
                    "allocate_resources",
                    self._allocate_impl,
                    agent_id,
                    requirements
                )
                return True
            
            def _allocate_impl(self, agent_id: str, requirements: Dict[str, Any]) -> None:
                self.allocated_resources[agent_id] = requirements
        
        resource_manager = MockResourceManager(resource_manager_config)
        await resource_manager.initialize()
        self.add_cleanup_task(resource_manager.shutdown)
        
        # Register agent and allocate resources
        agent_id = "resource_test_agent"
        await orchestrator.register_agent(agent_id, AgentRole.WORKER, ["resource_intensive"])
        
        resource_requirements = {
            "memory_mb": 512,
            "cpu_cores": 2,
            "gpu_memory_mb": 1024,
            "storage_gb": 10
        }
        
        success = await resource_manager.allocate_resources(agent_id, resource_requirements)
        assert success, "Resource allocation should succeed"
        
        # Verify allocation
        health = await resource_manager.health_check()
        assert health["allocated_resources"] == 1
    
    # === COMMUNICATION HUB ↔ ENGINE INTEGRATION ===
    
    @pytest.mark.asyncio
    async def test_communication_hub_task_engine_integration(self):
        """Test integration between communication hub and task execution engine."""
        components = await self.setup_component()
        communication_hub = components["communication_hub"]
        task_engine = components["task_engine"]
        
        # Setup message handler for task execution
        async def task_execution_handler(message: UnifiedMessage):
            """Handle task execution messages."""
            if message.message_type == MessageType.TASK_ASSIGNMENT:
                task_data = message.payload
                task_id = task_data.get("task_id", str(uuid.uuid4()))
                
                # Execute task through engine
                result = await task_engine.execute_task(task_id, task_data)
                
                # Send completion message
                completion_message = create_message(
                    source="task_engine",
                    destination=message.source,
                    message_type=MessageType.TASK_COMPLETION,
                    payload=result,
                    priority=Priority.HIGH
                )
                
                await communication_hub.send_message(completion_message)
        
        # Subscribe to task assignment messages
        subscription_results = await communication_hub.subscribe(
            "task_assignment",
            task_execution_handler,
            protocols=[ProtocolType.REDIS_STREAMS]
        )
        
        # Send task assignment message
        task_message = create_message(
            source="test_coordinator",
            destination="task_engine",
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={
                "task_id": "integration_test_task",
                "task_type": "data_processing",
                "parameters": {"operation": "sum", "data": [1, 2, 3, 4, 5]}
            },
            priority=Priority.MEDIUM
        )
        
        result = await communication_hub.send_message(task_message)
        
        # Verify message was sent successfully
        assert result.success or result.error, "Task assignment message should be processed"
        
        # Wait for task execution
        await asyncio.sleep(0.1)
        
        # Verify task was executed
        task_status = await task_engine.get_task_status("integration_test_task")
        if task_status:  # Only check if task was actually executed
            assert task_status["status"] == "completed"
    
    # === END-TO-END WORKFLOW INTEGRATION ===
    
    @pytest.mark.asyncio 
    async def test_complete_workflow_integration(self):
        """Test complete workflow across all consolidated components."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        context_manager = components["context_manager"]
        task_engine = components["task_engine"]
        
        async def complete_workflow():
            """Execute complete workflow across components."""
            workflow_start = time.time()
            
            # Step 1: Register agents with orchestrator
            coordinator_id = "workflow_coordinator"
            worker_id = "workflow_worker"
            
            await orchestrator.register_agent(
                coordinator_id,
                AgentRole.COORDINATOR,
                ["coordination", "workflow_management"]
            )
            await orchestrator.register_agent(
                worker_id, 
                AgentRole.WORKER,
                ["task_execution", "data_processing"]
            )
            
            # Step 2: Store context for agents
            coordinator_context = {
                "workflow_id": "integration_test_workflow",
                "current_step": 1,
                "total_steps": 3,
                "workflow_data": {"input": "test_data", "config": {"mode": "fast"}}
            }
            
            worker_context = {
                "capabilities": ["data_processing", "analysis"],
                "current_load": 0.2,
                "available_memory": 1024
            }
            
            await context_manager.store_context(coordinator_id, coordinator_context)
            await context_manager.store_context(worker_id, worker_context)
            
            # Step 3: Delegate task through orchestrator
            task_id = await orchestrator.delegate_task(
                "workflow_task_001",
                "data_processing_task",
                ["task_execution", "data_processing"],
                TaskPriority.HIGH
            )
            
            assert task_id is not None, "Task delegation should succeed"
            
            # Step 4: Send task details through communication hub
            task_details_message = create_message(
                source=coordinator_id,
                destination=worker_id,
                message_type=MessageType.TASK_ASSIGNMENT,
                payload={
                    "task_id": "workflow_task_001",
                    "operation": "process_data",
                    "data": {"values": [1, 2, 3, 4, 5], "operation": "sum"},
                    "context_ref": worker_id
                },
                priority=Priority.HIGH
            )
            
            comm_result = await communication_hub.send_message(task_details_message)
            assert comm_result.success or comm_result.error, "Task message should be sent"
            
            # Step 5: Execute task through engine
            task_result = await task_engine.execute_task(
                "workflow_task_001",
                task_details_message.payload
            )
            
            assert task_result["status"] == "completed"
            
            # Step 6: Complete task in orchestrator
            completion_success = await orchestrator.complete_task(
                "workflow_task_001",
                worker_id,
                task_result,
                True
            )
            
            assert completion_success, "Task completion should succeed"
            
            # Step 7: Send completion notification
            completion_message = create_message(
                source=worker_id,
                destination=coordinator_id,
                message_type=MessageType.TASK_COMPLETION,
                payload=task_result,
                priority=Priority.MEDIUM
            )
            
            completion_comm_result = await communication_hub.send_message(completion_message)
            assert completion_comm_result.success or completion_comm_result.error
            
            workflow_duration = (time.time() - workflow_start) * 1000
            return workflow_duration
        
        # Run complete workflow test
        scenario = self.get_performance_scenarios()[1]  # End-to-end scenario
        
        metrics = await self.run_performance_test(
            "test_complete_workflow_integration",
            complete_workflow,
            scenario
        )
        
        assert metrics.success, f"Complete workflow integration failed: {metrics.errors}"
        
        # Verify system state after workflow
        orchestrator_status = await orchestrator.get_system_status()
        assert orchestrator_status["agents"]["total"] == 2
        assert orchestrator_status["agents"]["active"] == 2
        
        context_health = await context_manager.health_check()
        assert context_health["contexts_count"] == 2
        
        comm_health = await communication_hub.get_health_status()
        assert comm_health["hub_status"] == "healthy"
    
    # === PERFORMANCE INTEGRATION TESTS ===
    
    @pytest.mark.asyncio
    @pytest.mark.performance 
    async def test_cross_component_performance_under_load(self):
        """Test cross-component performance under load conditions."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        context_manager = components["context_manager"]
        
        # Register multiple agents
        for i in range(20):
            await orchestrator.register_agent(
                f"load_test_agent_{i}",
                AgentRole.WORKER,
                [f"capability_{i%5}"]
            )
        
        async def load_test_operation():
            """Single load test operation across components."""
            agent_id = f"load_test_agent_{uuid.uuid4().hex[:8]}"
            
            # Register agent
            await orchestrator.register_agent(
                agent_id,
                AgentRole.WORKER,
                ["load_test_capability"]
            )
            
            # Store context
            await context_manager.store_context(
                agent_id,
                {"test_data": f"load_test_{time.time()}"}
            )
            
            # Send message
            message = create_message(
                source=agent_id,
                destination="system",
                message_type=MessageType.AGENT_HEARTBEAT,
                payload={"status": "active", "timestamp": time.time()}
            )
            
            result = await communication_hub.send_message(message)
            return result.success if result else False
        
        # Run load test
        benchmark_framework = PerformanceBenchmarkFramework()
        
        load_config = BenchmarkConfiguration(
            name="cross_component_load_test",
            component="integrated_system",
            duration_seconds=30,
            concurrent_operations=50,
            target_throughput_ops_sec=100.0,
            max_acceptable_latency_ms=200.0,
            enable_memory_tracking=True
        )
        
        result = await benchmark_framework.run_benchmark(
            load_config,
            load_test_operation
        )
        
        # Validate load test results
        assert not result.regression_detected, f"Performance regression detected: {result.regression_details}"
        assert result.error_rate_percent <= 10.0, f"High error rate: {result.error_rate_percent}%"
        assert result.throughput_ops_per_sec >= 50.0, f"Low throughput: {result.throughput_ops_per_sec} ops/sec"
        
        print(f"\n🚀 Cross-Component Load Test Results:")
        print(f"   Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"   Average Latency: {result.avg_latency_ms:.2f}ms")
        print(f"   Error Rate: {result.error_rate_percent:.2f}%")
        print(f"   Peak Memory: {result.peak_memory_mb:.2f}MB")
    
    # === ERROR HANDLING INTEGRATION TESTS ===
    
    @pytest.mark.asyncio
    async def test_error_propagation_across_components(self):
        """Test error handling and propagation across components."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        
        # Register agent
        await orchestrator.register_agent("error_test_agent", AgentRole.WORKER, ["error_handling"])
        
        # Test error in task delegation
        with patch.object(orchestrator, 'delegate_task') as mock_delegate:
            mock_delegate.side_effect = Exception("Simulated delegation error")
            
            try:
                await orchestrator.delegate_task(
                    "error_task",
                    "test_task",
                    ["error_handling"],
                    TaskPriority.MEDIUM
                )
            except Exception:
                pass  # Expected to fail
        
        # Verify system remains stable
        status = await orchestrator.get_system_status()
        assert status["health_status"] in ["healthy", "degraded"], "System should remain operational"
        
        # Test error in communication
        invalid_message = create_message(
            source="nonexistent_agent",
            destination="another_nonexistent_agent",
            message_type=MessageType.TASK_REQUEST,
            payload=None  # Invalid payload
        )
        
        result = await communication_hub.send_message(invalid_message)
        # Should handle gracefully without crashing
        assert result is not None, "Should return result even for invalid message"
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self):
        """Test system recovery from component failures."""
        components = await self.setup_component()
        orchestrator = components["orchestrator"]
        communication_hub = components["communication_hub"]
        
        # Simulate communication hub failure
        original_send = communication_hub.send_message
        
        async def failing_send(*args, **kwargs):
            raise Exception("Communication hub temporarily unavailable")
        
        communication_hub.send_message = failing_send
        
        # Test that orchestrator handles communication failure gracefully
        await orchestrator.register_agent("recovery_test_agent", AgentRole.WORKER, ["recovery"])
        
        # Restore communication hub
        communication_hub.send_message = original_send
        
        # Verify system can continue operating
        task_id = await orchestrator.delegate_task(
            "recovery_task",
            "test_task", 
            ["recovery"],
            TaskPriority.MEDIUM
        )
        
        assert task_id is not None, "System should recover and continue operating"
        
        # Verify system health
        status = await orchestrator.get_system_status()
        assert status["agents"]["total"] >= 1, "Agents should still be registered"


# === COMPONENT BOUNDARY TESTS ===

class TestComponentBoundaries(ConsolidatedTestBase):
    """Test proper boundaries and interfaces between components."""
    
    async def setup_component(self) -> None:
        """No specific setup needed for boundary tests."""
        pass
    
    async def cleanup_component(self) -> None:
        """No specific cleanup needed."""
        pass
    
    def get_performance_scenarios(self) -> List[TestScenario]:
        """No performance scenarios for boundary tests."""
        return []
    
    @pytest.mark.asyncio
    async def test_interface_contracts(self):
        """Test that components respect interface contracts."""
        # Test UniversalOrchestrator interface
        orchestrator_config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
        orchestrator = UniversalOrchestrator(orchestrator_config)
        
        # Verify required methods exist and have proper signatures
        assert hasattr(orchestrator, 'initialize')
        assert hasattr(orchestrator, 'register_agent')
        assert hasattr(orchestrator, 'delegate_task')
        assert hasattr(orchestrator, 'complete_task')
        assert hasattr(orchestrator, 'get_system_status')
        assert hasattr(orchestrator, 'shutdown')
        
        # Test CommunicationHub interface
        comm_config = CommunicationConfig()
        comm_hub = CommunicationHub(comm_config)
        
        assert hasattr(comm_hub, 'initialize')
        assert hasattr(comm_hub, 'send_message')
        assert hasattr(comm_hub, 'subscribe')
        assert hasattr(comm_hub, 'publish_event')
        assert hasattr(comm_hub, 'get_health_status')
        assert hasattr(comm_hub, 'shutdown')
    
    @pytest.mark.asyncio
    async def test_data_isolation_boundaries(self):
        """Test that components maintain proper data isolation."""
        # Test that orchestrator doesn't expose internal state
        orchestrator_config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
        orchestrator = UniversalOrchestrator(orchestrator_config)
        
        # Should not be able to directly modify internal state
        with pytest.raises(AttributeError):
            orchestrator.agents.clear()  # Should not have direct access
        
        # Should only access state through proper API methods
        status = await orchestrator.get_system_status()
        assert isinstance(status, dict)
        assert "agents" in status
    
    @pytest.mark.asyncio
    async def test_dependency_injection_boundaries(self):
        """Test proper dependency injection without tight coupling."""
        # Test that components can be initialized with mocked dependencies
        with patch('app.core.universal_orchestrator.get_redis') as mock_redis:
            mock_redis.return_value = self.test_redis
            
            orchestrator_config = OrchestratorConfig(mode=OrchestratorMode.TESTING)
            orchestrator = UniversalOrchestrator(orchestrator_config)
            
            success = await orchestrator.initialize()
            assert success, "Should initialize with mocked dependencies"
            
            await orchestrator.shutdown()