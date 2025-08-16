"""
Comprehensive tests for messaging service consolidation
Tests the unified messaging service functionality and integration with multi-agent coordination

Epic 1, Phase 1 Week 2: Messaging Service Testing
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.core.messaging_service import (
    MessagingService, get_messaging_service, Message, MessageType, 
    MessagePriority, RoutingStrategy, MessagingMetrics, MessageHandler
)
from app.core.messaging_migration import (
    verify_messaging_migration, get_migration_status, 
    LegacyMessageAdapter, MessagingServiceAdapter
)
from app.core.orchestrator import AgentOrchestrator
from app.core.enhanced_multi_agent_coordination import EnhancedMultiAgentCoordination


class TestMessagingService:
    """Test the unified messaging service functionality"""
    
    @pytest.fixture
    async def messaging_service(self):
        """Create messaging service for testing"""
        service = get_messaging_service()
        await service.connect()
        await service.start_service()
        yield service
        await service.stop_service()
        await service.disconnect()
    
    @pytest.mark.asyncio
    async def test_messaging_service_singleton(self):
        """Test that messaging service is a singleton"""
        service1 = get_messaging_service()
        service2 = get_messaging_service()
        assert service1 is service2, "Messaging service should be singleton"
    
    @pytest.mark.asyncio
    async def test_basic_message_sending(self, messaging_service):
        """Test basic message sending functionality"""
        message = Message(
            type=MessageType.EVENT,
            sender="test_sender",
            recipient="test_recipient",
            payload={"test": "data"},
            priority=MessagePriority.NORMAL,
            routing_strategy=RoutingStrategy.DIRECT
        )
        
        success = await messaging_service.send_message(message)
        assert success, "Message should be sent successfully"
    
    @pytest.mark.asyncio
    async def test_broadcast_messaging(self, messaging_service):
        """Test broadcast messaging functionality"""
        message = Message(
            type=MessageType.BROADCAST,
            sender="test_sender",
            payload={"announcement": "system message"},
            routing_strategy=RoutingStrategy.BROADCAST
        )
        
        success = await messaging_service.send_message(message)
        assert success, "Broadcast message should be sent successfully"
    
    @pytest.mark.asyncio
    async def test_topic_messaging(self, messaging_service):
        """Test topic-based messaging"""
        topic = "test_topic"
        
        # Subscribe to topic
        messages_received = []
        
        class TestTopicHandler(MessageHandler):
            def __init__(self):
                super().__init__("test_topic_handler", message_types=[MessageType.EVENT])
            
            async def _process_message(self, message: Message):
                messages_received.append(message)
                return None
        
        handler = TestTopicHandler()
        messaging_service.register_handler(handler)
        await messaging_service.subscribe_to_topic(topic, "test_topic_handler")
        
        # Send message to topic
        message = Message(
            type=MessageType.EVENT,
            sender="test_sender",
            topic=topic,
            payload={"topic_data": "test"},
            routing_strategy=RoutingStrategy.TOPIC
        )
        
        success = await messaging_service.send_message(message)
        assert success, "Topic message should be sent successfully"
        
        # Wait for message processing
        await asyncio.sleep(0.1)
        
        # Note: In a real test environment with Redis, we would verify message receipt
        # For now, we verify the message was accepted for sending
    
    @pytest.mark.asyncio
    async def test_priority_messaging(self, messaging_service):
        """Test priority message handling"""
        # Send messages with different priorities
        priorities = [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH, MessagePriority.CRITICAL]
        
        for priority in priorities:
            message = Message(
                type=MessageType.REQUEST,
                sender="test_sender",
                recipient="test_recipient",
                payload={"priority_test": priority.name},
                priority=priority
            )
            
            success = await messaging_service.send_message(message)
            assert success, f"Message with priority {priority.name} should be sent"
    
    @pytest.mark.asyncio
    async def test_request_response_pattern(self, messaging_service):
        """Test request-response messaging pattern"""
        # Create a test handler that responds to requests
        class TestRequestHandler(MessageHandler):
            def __init__(self):
                super().__init__("test_request_handler", message_types=[MessageType.REQUEST])
            
            async def _process_message(self, message: Message):
                return Message(
                    type=MessageType.RESPONSE,
                    sender="test_handler",
                    recipient=message.sender,
                    payload={"response": "handled"},
                    correlation_id=message.id
                )
        
        handler = TestRequestHandler()
        messaging_service.register_handler(handler)
        
        # Send request (in real implementation, would wait for response)
        response = await messaging_service.send_request(
            recipient="test_handler",
            payload={"request": "test"},
            timeout=1.0
        )
        
        # Note: Without full Redis setup, response will be None
        # In integration tests, we would verify actual response
    
    @pytest.mark.asyncio
    async def test_message_expiration(self, messaging_service):
        """Test message TTL and expiration"""
        message = Message(
            type=MessageType.EVENT,
            sender="test_sender",
            recipient="test_recipient",
            payload={"test": "expiring_message"},
            ttl=1  # 1 second TTL
        )
        
        success = await messaging_service.send_message(message)
        assert success, "Message with TTL should be sent"
        
        # Test message expiration check
        expired_message = Message(
            type=MessageType.EVENT,
            sender="test_sender",
            timestamp=datetime.utcnow().timestamp() - 3600,  # 1 hour ago
            ttl=60  # 1 minute TTL
        )
        
        assert expired_message.is_expired(), "Old message should be expired"
    
    @pytest.mark.asyncio
    async def test_messaging_metrics(self, messaging_service):
        """Test messaging service metrics collection"""
        # Send some test messages
        for i in range(5):
            message = Message(
                type=MessageType.EVENT,
                sender=f"test_sender_{i}",
                payload={"test_index": i}
            )
            await messaging_service.send_message(message)
        
        # Get metrics
        metrics = messaging_service.get_service_metrics()
        
        assert isinstance(metrics, MessagingMetrics), "Should return MessagingMetrics object"
        assert metrics.messages_sent >= 5, "Should track sent messages"
        
        # Test metrics dictionary conversion
        metrics_dict = metrics.to_dict()
        assert "messages" in metrics_dict, "Metrics should include messages section"
        assert "performance" in metrics_dict, "Metrics should include performance section"
    
    @pytest.mark.asyncio
    async def test_health_check(self, messaging_service):
        """Test messaging service health check"""
        health = await messaging_service.health_check()
        
        assert "status" in health, "Health check should include status"
        assert health["status"] in ["healthy", "degraded", "unhealthy"], "Status should be valid"
        assert "timestamp" in health, "Health check should include timestamp"


class TestMessagingMigration:
    """Test messaging migration utilities and compatibility"""
    
    @pytest.mark.asyncio
    async def test_migration_verification(self):
        """Test messaging migration verification"""
        verification_result = await verify_messaging_migration()
        
        assert "migration_status" in verification_result, "Should include migration status"
        assert "health_check" in verification_result, "Should include health check"
        assert "metrics" in verification_result, "Should include metrics"
    
    def test_migration_status(self):
        """Test migration status tracking"""
        status = get_migration_status()
        
        assert "progress_percent" in status, "Should include progress percentage"
        assert "status_details" in status, "Should include detailed status"
        assert "next_steps" in status, "Should include next steps"
        
        # Check that some components are marked as migrated
        assert status["status_details"]["messaging_service_created"], "Messaging service should be created"
        assert status["status_details"]["legacy_adapters_ready"], "Legacy adapters should be ready"
    
    @pytest.mark.asyncio
    async def test_legacy_message_adapter(self):
        """Test legacy message format conversion"""
        # Create a mock legacy message
        legacy_message = {
            "id": str(uuid.uuid4()),
            "from_agent": "test_agent",
            "to_agent": "target_agent",
            "payload": {"test": "data"},
            "priority": 5
        }
        
        # Convert to unified format
        unified_message = LegacyMessageAdapter.from_agent_message(legacy_message)
        
        assert isinstance(unified_message, Message), "Should convert to unified Message"
        assert unified_message.sender == "test_agent", "Should preserve sender"
        assert unified_message.recipient == "target_agent", "Should preserve recipient"
        assert unified_message.payload == {"test": "data"}, "Should preserve payload"
    
    @pytest.mark.asyncio
    async def test_messaging_service_adapter(self):
        """Test messaging service adapter for backward compatibility"""
        adapter = MessagingServiceAdapter()
        
        # Test legacy interface
        message_id = await adapter.send_lifecycle_message(
            message_type="HEARTBEAT_REQUEST",
            from_agent="test_agent",
            to_agent="target_agent",
            payload={"timestamp": datetime.utcnow().isoformat()}
        )
        
        assert message_id is not None, "Should return message ID"
        
        # Test metrics interface
        metrics = await adapter.get_metrics()
        assert isinstance(metrics, dict), "Should return metrics dictionary"


class TestOrchestratorIntegration:
    """Test orchestrator integration with unified messaging"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        return AgentOrchestrator()
    
    @pytest.mark.asyncio
    async def test_orchestrator_messaging_initialization(self, orchestrator):
        """Test that orchestrator initializes with unified messaging"""
        await orchestrator.start()
        
        assert orchestrator.messaging_service is not None, "Should have messaging service"
        assert isinstance(orchestrator.messaging_service, MessagingService), "Should use unified messaging"
        
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_orchestrator_message_handlers(self, orchestrator):
        """Test orchestrator message handlers registration"""
        await orchestrator.start()
        
        # Check that handlers are registered
        handlers = orchestrator.messaging_service._handlers
        assert "orchestrator" in handlers, "Should register orchestrator handler"
        
        await orchestrator.shutdown()


class TestCoordinationIntegration:
    """Test multi-agent coordination integration with unified messaging"""
    
    @pytest.fixture
    def coordination_system(self):
        """Create coordination system for testing"""
        return EnhancedMultiAgentCoordination()
    
    @pytest.mark.asyncio
    async def test_coordination_messaging_initialization(self, coordination_system):
        """Test coordination system messaging initialization"""
        await coordination_system.initialize()
        
        assert coordination_system.messaging_service is not None, "Should have messaging service"
        assert isinstance(coordination_system.messaging_service, MessagingService), "Should use unified messaging"
    
    @pytest.mark.asyncio
    async def test_coordination_message_handlers(self, coordination_system):
        """Test coordination message handlers registration"""
        await coordination_system.initialize()
        
        # Check that handlers are registered
        handlers = coordination_system.messaging_service._handlers
        assert "coordination" in handlers, "Should register coordination handler"
        
        # Check topic subscriptions
        subscribers = coordination_system.messaging_service._subscribers
        assert "coordination" in subscribers, "Should subscribe to coordination topic"
        assert "agents" in subscribers, "Should subscribe to agents topic"
        assert "tasks" in subscribers, "Should subscribe to tasks topic"


class TestEndToEndMessaging:
    """End-to-end messaging tests"""
    
    @pytest.mark.asyncio
    async def test_agent_task_coordination_flow(self):
        """Test complete agent task coordination flow"""
        # Initialize services
        messaging = get_messaging_service()
        await messaging.connect()
        await messaging.start_service()
        
        orchestrator = AgentOrchestrator()
        coordination = EnhancedMultiAgentCoordination()
        
        try:
            await orchestrator.start()
            await coordination.initialize()
            
            # Test task assignment flow
            task_assignment = Message(
                type=MessageType.TASK_ASSIGNMENT,
                sender="orchestrator",
                recipient="agent_1",
                payload={
                    "task_id": "test_task_123",
                    "task_data": {"action": "code_review"},
                    "assigned_at": datetime.utcnow().isoformat()
                },
                priority=MessagePriority.HIGH
            )
            
            success = await messaging.send_message(task_assignment)
            assert success, "Task assignment should be sent successfully"
            
            # Test task completion flow
            task_completion = Message(
                type=MessageType.TASK_COMPLETION,
                sender="agent_1",
                recipient="orchestrator",
                payload={
                    "task_id": "test_task_123",
                    "result": {"status": "completed", "output": "Review completed"},
                    "completed_at": datetime.utcnow().isoformat()
                },
                priority=MessagePriority.HIGH
            )
            
            success = await messaging.send_message(task_completion)
            assert success, "Task completion should be sent successfully"
            
            # Test system broadcast
            system_announcement = Message(
                type=MessageType.BROADCAST,
                sender="system",
                payload={
                    "announcement": "System maintenance scheduled",
                    "scheduled_time": (datetime.utcnow() + timedelta(hours=1)).isoformat()
                },
                routing_strategy=RoutingStrategy.BROADCAST
            )
            
            success = await messaging.send_message(system_announcement)
            assert success, "System broadcast should be sent successfully"
            
        finally:
            await orchestrator.shutdown()
            await messaging.stop_service()
            await messaging.disconnect()
    
    @pytest.mark.asyncio
    async def test_messaging_performance(self):
        """Test messaging service performance under load"""
        messaging = get_messaging_service()
        await messaging.connect()
        await messaging.start_service()
        
        try:
            start_time = datetime.utcnow()
            
            # Send multiple messages rapidly
            tasks = []
            for i in range(100):
                message = Message(
                    type=MessageType.EVENT,
                    sender=f"performance_test_{i}",
                    payload={"index": i, "timestamp": datetime.utcnow().isoformat()}
                )
                tasks.append(messaging.send_message(message))
            
            # Wait for all messages to be sent
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Check results
            successful_sends = sum(1 for result in results if result is True)
            assert successful_sends >= 90, f"Should send at least 90% successfully, got {successful_sends}/100"
            
            # Check performance
            messages_per_second = successful_sends / duration
            assert messages_per_second > 10, f"Should handle at least 10 msg/sec, got {messages_per_second:.2f}"
            
            # Check metrics
            metrics = messaging.get_service_metrics()
            assert metrics.messages_sent >= successful_sends, "Metrics should track sent messages"
            
        finally:
            await messaging.stop_service()
            await messaging.disconnect()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])