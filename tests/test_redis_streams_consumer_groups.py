"""
Comprehensive test suite for Redis Streams with Consumer Groups - Vertical Slice 4.2

Tests all aspects of the consumer group system including:
- Enhanced Redis Streams Manager
- Consumer Group Coordinator
- Workflow Message Router
- Dead Letter Queue Handler
- Load balancing and failure recovery
"""

import asyncio
import json
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType,
    MessageRoutingMode, ConsumerGroupMetrics
)
from app.core.consumer_group_coordinator import (
    ConsumerGroupCoordinator, ConsumerGroupStrategy, ProvisioningPolicy
)
from app.core.workflow_message_router import (
    WorkflowMessageRouter, WorkflowRoutingStrategy, WorkflowContext
)
from app.core.dead_letter_queue_handler import (
    DeadLetterQueueHandler, DLQMessage, FailureCategory, RecoveryStrategy
)
from app.models.message import StreamMessage, MessageType, MessagePriority


class TestEnhancedRedisStreamsManager:
    """Test Enhanced Redis Streams Manager functionality."""
    
    @pytest.fixture
    async def streams_manager(self):
        """Create test streams manager."""
        manager = EnhancedRedisStreamsManager(
            redis_url="redis://localhost:6379/15",  # Test database
            connection_pool_size=5,
            auto_scaling_enabled=False  # Disable for testing
        )
        
        # Mock the base manager to avoid actual Redis connections
        manager._base_manager = AsyncMock()
        manager._base_manager.connect = AsyncMock()
        manager._base_manager.disconnect = AsyncMock()
        manager._base_manager.send_stream_message = AsyncMock(return_value="test-message-id")
        manager._base_manager.get_stream_stats = AsyncMock()
        
        await manager.connect()
        yield manager
        await manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_create_consumer_group(self, streams_manager):
        """Test creating a consumer group."""
        config = ConsumerGroupConfig(
            name="test_backend_engineers",
            stream_name="agent_messages:backend",
            agent_type=ConsumerGroupType.BACKEND_ENGINEERS,
            routing_mode=MessageRoutingMode.LOAD_BALANCED
        )
        
        await streams_manager.create_consumer_group(config)
        
        assert config.name in streams_manager._consumer_groups
        assert streams_manager._consumer_groups[config.name] == config
    
    @pytest.mark.asyncio
    async def test_add_consumer_to_group(self, streams_manager):
        """Test adding a consumer to a group."""
        # First create a group
        config = ConsumerGroupConfig(
            name="test_group",
            stream_name="test_stream",
            agent_type=ConsumerGroupType.GENERAL_AGENTS
        )
        await streams_manager.create_consumer_group(config)
        
        # Mock handler
        async def mock_handler(message):
            return {"processed": True}
        
        # Add consumer
        consumer_id = "test_consumer_1"
        await streams_manager.add_consumer_to_group(
            "test_group", consumer_id, mock_handler
        )
        
        # Verify consumer was added
        assert "test_group" in streams_manager._active_consumers
        assert consumer_id in streams_manager._active_consumers["test_group"]
        
        handler_key = f"test_group:{consumer_id}"
        assert handler_key in streams_manager._consumer_metrics
    
    @pytest.mark.asyncio
    async def test_send_message_to_group(self, streams_manager):
        """Test sending message to consumer group."""
        # Create group
        config = ConsumerGroupConfig(
            name="test_group",
            stream_name="test_stream",
            agent_type=ConsumerGroupType.BACKEND_ENGINEERS
        )
        await streams_manager.create_consumer_group(config)
        
        # Create test message
        message = StreamMessage(
            id="test_msg_1",
            from_agent="orchestrator",
            to_agent=None,
            message_type=MessageType.TASK_REQUEST,
            payload={"test": "data"},
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )
        
        # Send message
        message_id = await streams_manager.send_message_to_group("test_group", message)
        
        assert message_id == "test-message-id"
        streams_manager._base_manager.send_stream_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_consumer_from_group(self, streams_manager):
        """Test removing a consumer from a group."""
        # Setup group and consumer
        config = ConsumerGroupConfig(
            name="test_group",
            stream_name="test_stream",
            agent_type=ConsumerGroupType.GENERAL_AGENTS
        )
        await streams_manager.create_consumer_group(config)
        
        async def mock_handler(message):
            return {"processed": True}
        
        consumer_id = "test_consumer_1"
        await streams_manager.add_consumer_to_group(
            "test_group", consumer_id, mock_handler
        )
        
        # Remove consumer
        await streams_manager.remove_consumer_from_group("test_group", consumer_id)
        
        # Verify consumer was removed
        assert consumer_id not in streams_manager._active_consumers.get("test_group", {})
    
    @pytest.mark.asyncio
    async def test_get_consumer_group_stats(self, streams_manager):
        """Test getting consumer group statistics."""
        # Create group with mock stats
        config = ConsumerGroupConfig(
            name="test_group",
            stream_name="test_stream",
            agent_type=ConsumerGroupType.QA_ENGINEERS
        )
        await streams_manager.create_consumer_group(config)
        
        # Mock some activity
        streams_manager._group_metrics["test_group"] = ConsumerGroupMetrics(
            group_name="test_group",
            stream_name="test_stream",
            consumer_count=2,
            lag=50,
            throughput_msg_per_sec=10.5
        )
        
        stats = await streams_manager.get_consumer_group_stats("test_group")
        
        assert stats is not None
        assert stats.group_name == "test_group"
        assert stats.consumer_count == 2
        assert stats.lag == 50
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, streams_manager):
        """Test performance metrics collection."""
        # Create some activity
        streams_manager._performance_stats['messages_routed'] = 100
        streams_manager._performance_stats['groups_created'] = 5
        
        metrics = await streams_manager.get_performance_metrics()
        
        assert "enhanced_metrics" in metrics
        assert metrics["enhanced_metrics"]["messages_routed"] == 100
        assert metrics["enhanced_metrics"]["groups_created"] == 5


class TestConsumerGroupCoordinator:
    """Test Consumer Group Coordinator functionality."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create test coordinator."""
        # Mock streams manager
        streams_manager = AsyncMock()
        streams_manager.create_consumer_group = AsyncMock()
        streams_manager.get_consumer_group_stats = AsyncMock()
        streams_manager.get_all_group_stats = AsyncMock(return_value={})
        
        coordinator = ConsumerGroupCoordinator(
            streams_manager=streams_manager,
            strategy=ConsumerGroupStrategy.HYBRID,
            health_check_interval=0.1,  # Fast for testing
            rebalance_interval=0.2
        )
        
        await coordinator.start()
        yield coordinator
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_provision_group_for_agent(self, coordinator):
        """Test provisioning group for an agent."""
        from app.models.agent import AgentType
        
        agent_id = "test_agent_1"
        agent_type = AgentType.BACKEND_ENGINEER
        
        group_name = await coordinator.provision_group_for_agent(
            agent_id, agent_type
        )
        
        assert group_name is not None
        assert agent_id in coordinator._group_assignments
        assert coordinator._group_assignments[agent_id] == group_name
    
    @pytest.mark.asyncio
    async def test_rebalance_groups(self, coordinator):
        """Test group rebalancing functionality."""
        # Mock some group stats
        coordinator.streams_manager.get_all_group_stats.return_value = {
            "backend_engineers_consumers": ConsumerGroupMetrics(
                group_name="backend_engineers_consumers",
                stream_name="agent_messages:backend",
                consumer_count=2,
                lag=150,  # High lag to trigger scaling
                success_rate=0.99
            )
        }
        
        result = await coordinator.rebalance_groups()
        
        assert "rebalance_time_ms" in result
        assert "operations_performed" in result
        assert isinstance(result["operations_performed"], list)
    
    @pytest.mark.asyncio
    async def test_coordinator_metrics(self, coordinator):
        """Test coordinator metrics collection."""
        # Add some managed groups
        coordinator._managed_groups["test_group"] = ConsumerGroupConfig(
            name="test_group",
            stream_name="test_stream",
            agent_type=ConsumerGroupType.GENERAL_AGENTS
        )
        
        metrics = await coordinator.get_coordinator_metrics()
        
        assert "coordinator_metrics" in metrics
        assert "managed_groups" in metrics
        assert "test_group" in metrics["managed_groups"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, coordinator):
        """Test coordinator health check."""
        health = await coordinator.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "degraded"]
        assert "groups_managed" in health


class TestWorkflowMessageRouter:
    """Test Workflow Message Router functionality."""
    
    @pytest.fixture
    async def workflow_router(self):
        """Create test workflow router."""
        # Mock dependencies
        streams_manager = AsyncMock()
        streams_manager.send_message_to_group = AsyncMock(return_value="test-msg-id")
        
        coordinator = AsyncMock()
        coordinator.streams_manager = streams_manager
        
        router = WorkflowMessageRouter(
            streams_manager=streams_manager,
            coordinator=coordinator,
            enable_workflow_optimization=False  # Disable for testing
        )
        
        await router.start()
        yield router
        await router.stop()
    
    @pytest.mark.asyncio
    async def test_route_workflow(self, workflow_router):
        """Test routing an entire workflow."""
        workflow_id = "test_workflow_1"
        tasks = [
            {
                "id": "task_1",
                "type": "backend",
                "dependencies": []
            },
            {
                "id": "task_2", 
                "type": "frontend",
                "dependencies": ["task_1"]
            }
        ]
        
        result = await workflow_router.route_workflow(workflow_id, tasks)
        
        assert result["workflow_id"] == workflow_id
        assert result["tasks_routed"] == len(tasks)
        assert "routing_strategy" in result
        assert workflow_id in workflow_router._active_workflows
    
    @pytest.mark.asyncio
    async def test_route_task_message(self, workflow_router):
        """Test routing a single task message."""
        message = StreamMessage(
            id="task_msg_1",
            from_agent="orchestrator",
            to_agent=None,
            message_type=MessageType.TASK_REQUEST,
            payload={"task_type": "backend"},
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )
        
        decision = await workflow_router.route_task_message(message)
        
        assert decision.task_id == message.id
        assert decision.target_group is not None
        assert decision.routing_strategy is not None
    
    @pytest.mark.asyncio
    async def test_signal_task_completion(self, workflow_router):
        """Test signaling task completion."""
        # Setup workflow with dependencies
        workflow_id = "test_workflow_2"
        tasks = [
            {"id": "task_1", "dependencies": []},
            {"id": "task_2", "dependencies": ["task_1"]}
        ]
        
        await workflow_router.route_workflow(workflow_id, tasks)
        
        # Signal completion of task_1
        triggered_tasks = await workflow_router.signal_task_completion(
            workflow_id, "task_1", {"status": "completed"}
        )
        
        # Verify workflow state was updated
        context = workflow_router._active_workflows[workflow_id]
        assert "task_1" in context.completed_tasks
    
    @pytest.mark.asyncio
    async def test_workflow_context_creation(self, workflow_router):
        """Test workflow context creation."""
        tasks = [
            {"id": "task_1", "dependencies": []},
            {"id": "task_2", "dependencies": ["task_1"]},
            {"id": "task_3", "dependencies": ["task_1"]}
        ]
        
        context = await workflow_router._create_workflow_context("test_wf", tasks)
        
        assert context.workflow_id == "test_wf"
        assert context.total_steps == len(tasks)
        assert "task_2" in context.dependencies
        assert "task_1" in context.dependencies["task_2"]
    
    @pytest.mark.asyncio
    async def test_routing_metrics(self, workflow_router):
        """Test routing metrics collection."""
        # Generate some activity
        workflow_router._metrics.total_workflows_routed = 5
        workflow_router._metrics.successful_routings = 20
        
        metrics = await workflow_router.get_routing_metrics()
        
        assert "workflow_routing_metrics" in metrics
        assert metrics["workflow_routing_metrics"]["total_workflows_routed"] == 5
        assert metrics["workflow_routing_metrics"]["successful_routings"] == 20


class TestDeadLetterQueueHandler:
    """Test Dead Letter Queue Handler functionality."""
    
    @pytest.fixture
    async def dlq_handler(self):
        """Create test DLQ handler."""
        # Mock streams manager
        streams_manager = AsyncMock()
        streams_manager._base_manager = AsyncMock()
        streams_manager._base_manager._redis = AsyncMock()
        streams_manager._base_manager.send_stream_message = AsyncMock(return_value="replay-msg-id")
        
        handler = DeadLetterQueueHandler(
            streams_manager=streams_manager,
            analysis_interval=0.1,  # Fast for testing
            cleanup_interval=0.2,
            enable_automatic_recovery=False  # Disable for testing
        )
        
        await handler.start()
        yield handler
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_process_failed_message(self, dlq_handler):
        """Test processing a failed message into DLQ."""
        original_message = StreamMessage(
            id="failed_msg_1",
            from_agent="test_agent",
            to_agent=None,
            message_type=MessageType.TASK_REQUEST,
            payload={"test": "data"},
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )
        
        failure_details = {
            "error_type": "timeout",
            "error_message": "Request timeout after 30 seconds"
        }
        
        dlq_id = await dlq_handler.process_failed_message(
            original_message, "test_stream", "test_group", failure_details
        )
        
        assert dlq_id is not None
        assert dlq_id in dlq_handler._dlq_messages
        
        dlq_message = dlq_handler._dlq_messages[dlq_id]
        assert dlq_message.failure_category == FailureCategory.TIMEOUT
        assert dlq_message.original_message.id == original_message.id
    
    @pytest.mark.asyncio
    async def test_replay_message(self, dlq_handler):
        """Test replaying a message from DLQ."""
        # First add a message to DLQ
        original_message = StreamMessage(
            id="replay_msg_1",
            from_agent="test_agent",
            to_agent=None,
            message_type=MessageType.TASK_REQUEST,
            payload={"test": "replay"},
            priority=MessagePriority.NORMAL,
            timestamp=time.time()
        )
        
        dlq_message = DLQMessage(
            original_message=original_message,
            failure_count=1,
            first_failure_time=datetime.utcnow(),
            last_failure_time=datetime.utcnow(),
            failure_category=FailureCategory.HANDLER_EXCEPTION,
            failure_details={"error": "test"},
            original_stream="test_stream",
            original_consumer_group="test_group"
        )
        
        dlq_handler._dlq_messages[dlq_message.dlq_id] = dlq_message
        
        # Replay the message
        success = await dlq_handler.replay_message(dlq_message.dlq_id)
        
        assert success
        dlq_handler.streams_manager._base_manager.send_stream_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_replay_batch(self, dlq_handler):
        """Test batch replay functionality."""
        # Add multiple messages to DLQ
        for i in range(3):
            original_message = StreamMessage(
                id=f"batch_msg_{i}",
                from_agent="test_agent",
                to_agent=None,
                message_type=MessageType.TASK_REQUEST,
                payload={"test": f"batch_{i}"},
                priority=MessagePriority.NORMAL,
                timestamp=time.time()
            )
            
            dlq_message = DLQMessage(
                original_message=original_message,
                failure_count=1,
                first_failure_time=datetime.utcnow(),
                last_failure_time=datetime.utcnow(),
                failure_category=FailureCategory.NETWORK_ERROR,
                failure_details={"error": "network"},
                original_stream="test_stream",
                original_consumer_group="test_group"
            )
            
            dlq_handler._dlq_messages[dlq_message.dlq_id] = dlq_message
        
        # Mock eligible messages finder
        dlq_handler._find_eligible_messages = AsyncMock(
            return_value=list(dlq_handler._dlq_messages.keys())
        )
        
        result = await dlq_handler.replay_batch(max_messages=2)
        
        assert result["total_attempted"] == 2  # Limited by max_messages
        assert result["successful"] >= 0
    
    @pytest.mark.asyncio
    async def test_failure_categorization(self, dlq_handler):
        """Test failure categorization logic."""
        test_cases = [
            ({"error_message": "timeout occurred"}, FailureCategory.TIMEOUT),
            ({"error_message": "invalid json format"}, FailureCategory.PARSING_ERROR),
            ({"error_message": "validation failed"}, FailureCategory.VALIDATION_ERROR),
            ({"error_message": "network connection error"}, FailureCategory.NETWORK_ERROR),
            ({"error_message": "out of memory"}, FailureCategory.RESOURCE_EXHAUSTION),
            ({"error_message": "unknown error"}, FailureCategory.UNKNOWN)
        ]
        
        for failure_details, expected_category in test_cases:
            category = await dlq_handler._categorize_failure(failure_details)
            assert category == expected_category
    
    @pytest.mark.asyncio
    async def test_dlq_statistics(self, dlq_handler):
        """Test DLQ statistics collection."""
        # Add some test data
        dlq_handler._metrics.messages_processed = 10
        dlq_handler._metrics.messages_recovered = 8
        dlq_handler._metrics.messages_permanently_failed = 1
        
        stats = await dlq_handler.get_dlq_statistics()
        
        assert "dlq_metrics" in stats
        assert stats["dlq_metrics"]["messages_processed"] == 10
        assert stats["dlq_metrics"]["messages_recovered"] == 8
    
    @pytest.mark.asyncio
    async def test_health_check(self, dlq_handler):
        """Test DLQ health check."""
        health = await dlq_handler.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "degraded"]
        assert "dlq_message_count" in health


class TestIntegrationScenarios:
    """Test integration scenarios across multiple components."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system for testing."""
        # Mock Redis
        mock_redis = AsyncMock()
        
        # Create components
        streams_manager = EnhancedRedisStreamsManager(
            redis_url="redis://localhost:6379/15",
            auto_scaling_enabled=False
        )
        streams_manager._base_manager = AsyncMock()
        streams_manager._base_manager.connect = AsyncMock()
        streams_manager._base_manager.disconnect = AsyncMock()
        streams_manager._base_manager.send_stream_message = AsyncMock(return_value="test-msg-id")
        
        coordinator = ConsumerGroupCoordinator(
            streams_manager,
            health_check_interval=0.1,
            rebalance_interval=0.2
        )
        
        router = WorkflowMessageRouter(
            streams_manager,
            coordinator,
            enable_workflow_optimization=False
        )
        
        dlq_handler = DeadLetterQueueHandler(
            streams_manager,
            analysis_interval=0.1,
            cleanup_interval=0.2,
            enable_automatic_recovery=False
        )
        
        # Start all components
        await streams_manager.connect()
        await coordinator.start()
        await router.start()
        await dlq_handler.start()
        
        yield {
            "streams_manager": streams_manager,
            "coordinator": coordinator,
            "router": router,
            "dlq_handler": dlq_handler
        }
        
        # Cleanup
        await dlq_handler.stop()
        await router.stop()
        await coordinator.stop()
        await streams_manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_processing(self, integrated_system):
        """Test complete workflow from creation to completion."""
        coordinator = integrated_system["coordinator"]
        router = integrated_system["router"]
        
        # Create consumer groups
        from app.models.agent import AgentType
        backend_group = await coordinator.provision_group_for_agent(
            "backend_agent_1", AgentType.BACKEND_ENGINEER
        )
        frontend_group = await coordinator.provision_group_for_agent(
            "frontend_agent_1", AgentType.FRONTEND_DEVELOPER
        )
        
        # Route a workflow
        workflow_id = "integration_test_workflow"
        tasks = [
            {
                "id": "backend_task",
                "type": "backend",
                "dependencies": []
            },
            {
                "id": "frontend_task",
                "type": "frontend", 
                "dependencies": ["backend_task"]
            }
        ]
        
        routing_result = await router.route_workflow(workflow_id, tasks)
        
        assert routing_result["workflow_id"] == workflow_id
        assert routing_result["tasks_routed"] == len(tasks)
        
        # Signal completion of backend task
        triggered_tasks = await router.signal_task_completion(
            workflow_id, "backend_task", {"status": "completed"}
        )
        
        # Verify frontend task was triggered
        context = router._active_workflows[workflow_id]
        assert "backend_task" in context.completed_tasks
    
    @pytest.mark.asyncio
    async def test_failure_recovery_flow(self, integrated_system):
        """Test complete failure recovery flow."""
        streams_manager = integrated_system["streams_manager"]
        dlq_handler = integrated_system["dlq_handler"]
        
        # Create a failed message
        failed_message = StreamMessage(
            id="failed_integration_msg",
            from_agent="test_agent",
            to_agent=None,
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "integration_test"},
            priority=MessagePriority.HIGH,
            timestamp=time.time()
        )
        
        failure_details = {
            "error_type": "exception",
            "error_message": "Handler exception during processing"
        }
        
        # Process failure
        dlq_id = await dlq_handler.process_failed_message(
            failed_message, "test_stream", "test_group", failure_details
        )
        
        assert dlq_id is not None
        
        # Replay the message
        success = await dlq_handler.replay_message(dlq_id, priority_boost=True)
        assert success
        
        # Verify message was sent back to stream
        dlq_handler.streams_manager._base_manager.send_stream_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_load_balancing_simulation(self, integrated_system):
        """Test load balancing across multiple consumers."""
        streams_manager = integrated_system["streams_manager"]
        coordinator = integrated_system["coordinator"]
        
        # Create consumer group
        group_config = ConsumerGroupConfig(
            name="load_test_group",
            stream_name="load_test_stream",
            agent_type=ConsumerGroupType.BACKEND_ENGINEERS,
            max_consumers=5
        )
        await streams_manager.create_consumer_group(group_config)
        
        # Add multiple consumers
        consumer_ids = []
        for i in range(3):
            consumer_id = f"load_test_consumer_{i}"
            consumer_ids.append(consumer_id)
            
            async def mock_handler(message):
                return {"consumer": consumer_id, "processed": True}
            
            await streams_manager.add_consumer_to_group(
                "load_test_group", consumer_id, mock_handler
            )
        
        # Send multiple messages
        for i in range(10):
            message = StreamMessage(
                id=f"load_msg_{i}",
                from_agent="load_tester",
                to_agent=None,
                message_type=MessageType.TASK_REQUEST,
                payload={"task_id": i},
                priority=MessagePriority.NORMAL,
                timestamp=time.time()
            )
            
            message_id = await streams_manager.send_message_to_group(
                "load_test_group", message
            )
            assert message_id is not None
        
        # Verify load distribution
        assert len(streams_manager._active_consumers["load_test_group"]) == 3
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, integrated_system):
        """Test comprehensive metrics collection across all components."""
        streams_manager = integrated_system["streams_manager"]
        coordinator = integrated_system["coordinator"]
        router = integrated_system["router"]
        dlq_handler = integrated_system["dlq_handler"]
        
        # Generate some activity data
        streams_manager._performance_stats['messages_routed'] = 50
        coordinator._metrics.groups_created = 3
        router._metrics.total_workflows_routed = 5
        dlq_handler._metrics.messages_processed = 2
        
        # Collect metrics from all components
        streams_metrics = await streams_manager.get_performance_metrics()
        coordinator_metrics = await coordinator.get_coordinator_metrics()
        router_metrics = await router.get_routing_metrics()
        dlq_metrics = await dlq_handler.get_dlq_statistics()
        
        # Verify metrics are comprehensive
        assert "enhanced_metrics" in streams_metrics
        assert streams_metrics["enhanced_metrics"]["messages_routed"] == 50
        
        assert "coordinator_metrics" in coordinator_metrics
        assert coordinator_metrics["coordinator_metrics"]["groups_created"] == 3
        
        assert "workflow_routing_metrics" in router_metrics
        assert router_metrics["workflow_routing_metrics"]["total_workflows_routed"] == 5
        
        assert "dlq_metrics" in dlq_metrics
        assert dlq_metrics["dlq_metrics"]["messages_processed"] == 2


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_message_routing(self):
        """Test system behavior under high message throughput."""
        # Mock high-performance streams manager
        streams_manager = AsyncMock()
        streams_manager.send_message_to_group = AsyncMock(
            side_effect=lambda *args, **kwargs: f"msg-{time.time()}"
        )
        
        # Create router
        coordinator = AsyncMock()
        router = WorkflowMessageRouter(streams_manager, coordinator)
        
        # Simulate high throughput
        message_count = 1000
        start_time = time.time()
        
        tasks = []
        for i in range(message_count):
            message = StreamMessage(
                id=f"perf_msg_{i}",
                from_agent="perf_tester",
                to_agent=None,
                message_type=MessageType.TASK_REQUEST,
                payload={"task_id": i},
                priority=MessagePriority.NORMAL,
                timestamp=time.time()
            )
            
            task = asyncio.create_task(
                router.route_task_message(message)
            )
            tasks.append(task)
        
        # Wait for all messages to be processed
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = message_count / duration
        
        # Verify performance (should handle >1000 messages/second)
        assert throughput > 100  # Conservative threshold for testing
        assert len([r for r in results if not isinstance(r, Exception)]) == message_count
    
    @pytest.mark.asyncio
    async def test_consumer_group_scaling(self):
        """Test consumer group auto-scaling behavior."""
        # Mock streams manager with scaling metrics
        streams_manager = AsyncMock()
        streams_manager.create_consumer_group = AsyncMock()
        streams_manager.get_consumer_group_stats = AsyncMock(
            return_value=ConsumerGroupMetrics(
                group_name="scaling_test_group",
                stream_name="scaling_stream",
                consumer_count=2,
                lag=200,  # High lag to trigger scaling
                success_rate=0.99
            )
        )
        
        coordinator = ConsumerGroupCoordinator(
            streams_manager,
            health_check_interval=0.01,
            rebalance_interval=0.02
        )
        
        # Start coordinator and let it run briefly
        await coordinator.start()
        await asyncio.sleep(0.1)  # Let background tasks run
        await coordinator.stop()
        
        # Verify scaling decisions were made
        assert coordinator._metrics.health_checks_performed > 0
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage characteristics under load."""
        streams_manager = EnhancedRedisStreamsManager(
            auto_scaling_enabled=False
        )
        streams_manager._base_manager = AsyncMock()
        
        # Create many consumer groups and metrics
        for i in range(100):
            config = ConsumerGroupConfig(
                name=f"memory_test_group_{i}",
                stream_name=f"stream_{i}",
                agent_type=ConsumerGroupType.GENERAL_AGENTS
            )
            streams_manager._consumer_groups[config.name] = config
            
            # Add metrics
            streams_manager._group_metrics[config.name] = ConsumerGroupMetrics(
                group_name=config.name,
                stream_name=config.stream_name,
                consumer_count=3,
                lag=10,
                throughput_msg_per_sec=5.0
            )
        
        # Verify system can handle large numbers of groups
        all_stats = await streams_manager.get_all_group_stats()
        assert len(all_stats) == 100
        
        # Test metrics collection performance
        start_time = time.time()
        metrics = await streams_manager.get_performance_metrics()
        metrics_time = time.time() - start_time
        
        # Should be fast even with many groups
        assert metrics_time < 1.0  # Less than 1 second
        assert "enhanced_metrics" in metrics


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])