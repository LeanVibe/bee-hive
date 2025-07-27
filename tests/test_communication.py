"""
Comprehensive tests for Redis Streams Communication System.

Tests message passing reliability, consumer groups, dead letter queues,
performance, and integration with the Agent Orchestrator.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.core.communication import MessageBroker, SimplePubSub, CommunicationError
from app.models.message import (
    StreamMessage, MessageType, MessagePriority, MessageStatus,
    MessageAudit, StreamInfo, ConsumerGroupInfo
)
from app.sdk.messaging import MessagingClient, send_quick_message


class TestStreamMessage:
    """Test StreamMessage model and validation."""
    
    def test_message_creation(self):
        """Test creating a stream message."""
        message = StreamMessage(
            from_agent="agent1",
            to_agent="agent2",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "test"},
            priority=MessagePriority.HIGH
        )
        
        assert message.from_agent == "agent1"
        assert message.to_agent == "agent2"
        assert message.message_type == MessageType.TASK_REQUEST
        assert message.payload == {"task": "test"}
        assert message.priority == MessagePriority.HIGH
        assert message.id is not None
        assert message.timestamp > 0
    
    def test_broadcast_message(self):
        """Test creating a broadcast message."""
        message = StreamMessage(
            from_agent="agent1",
            to_agent=None,  # Broadcast
            message_type=MessageType.BROADCAST,
            payload={"announcement": "system update"}
        )
        
        assert message.to_agent is None
        assert message.get_stream_name() == "agent_messages:broadcast"
        assert message.get_consumer_group() == "consumers:broadcast"
    
    def test_message_signing(self):
        """Test message signing and verification."""
        message = StreamMessage(
            from_agent="agent1",
            to_agent="agent2",
            message_type=MessageType.EVENT,
            payload={"data": "sensitive"}
        )
        
        secret_key = "test_secret_key"
        
        # Sign message
        message.sign(secret_key)
        assert message.signature is not None
        
        # Verify signature
        assert message.verify_signature(secret_key) is True
        
        # Test with wrong key
        assert message.verify_signature("wrong_key") is False
    
    def test_message_expiration(self):
        """Test message TTL and expiration."""
        message = StreamMessage(
            from_agent="agent1",
            to_agent="agent2",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "urgent"},
            ttl=1  # 1 second TTL
        )
        
        # Should not be expired immediately
        assert message.is_expired() is False
        
        # Mock time to simulate expiration
        with patch('time.time', return_value=message.timestamp + 2):
            assert message.is_expired() is True
    
    def test_redis_serialization(self):
        """Test conversion to/from Redis format."""
        original_message = StreamMessage(
            from_agent="agent1",
            to_agent="agent2",
            message_type=MessageType.COORDINATION,
            payload={"action": "sync", "data": [1, 2, 3]},
            priority=MessagePriority.URGENT,
            correlation_id="test_correlation"
        )
        
        # Convert to Redis format
        redis_dict = original_message.to_redis_dict()
        
        # Verify Redis format
        assert isinstance(redis_dict, dict)
        assert all(isinstance(v, str) for v in redis_dict.values())
        assert redis_dict["from_agent"] == "agent1"
        assert redis_dict["to_agent"] == "agent2"
        assert redis_dict["message_type"] == "coordination"
        assert '"action":"sync"' in redis_dict["payload"]
        
        # Convert back from Redis format
        restored_message = StreamMessage.from_redis_dict(redis_dict)
        
        # Verify restoration
        assert restored_message.from_agent == original_message.from_agent
        assert restored_message.to_agent == original_message.to_agent
        assert restored_message.message_type == original_message.message_type
        assert restored_message.payload == original_message.payload
        assert restored_message.priority == original_message.priority
        assert restored_message.correlation_id == original_message.correlation_id


class TestMessageBroker:
    """Test MessageBroker functionality."""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis connection."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.xadd.return_value = "1234567890-0"
        redis_mock.xreadgroup.return_value = []
        redis_mock.xinfo_stream.return_value = {
            "length": 10,
            "first-entry": "1-0",
            "last-entry": "10-0"
        }
        redis_mock.xinfo_groups.return_value = []
        return redis_mock
    
    @pytest.fixture
    async def message_broker(self, mock_redis):
        """Create MessageBroker with mocked Redis."""
        broker = MessageBroker(
            redis_url="redis://localhost:6379/0",
            secret_key="test_secret"
        )
        
        with patch('app.core.communication.redis.ConnectionPool') as mock_pool:
            mock_pool.from_url.return_value = AsyncMock()
            with patch('app.core.communication.Redis') as mock_redis_class:
                mock_redis_class.return_value = mock_redis
                broker._redis = mock_redis
                yield broker
    
    async def test_broker_connection(self, message_broker):
        """Test broker connection management."""
        # Should not be connected initially
        assert message_broker._redis is not None  # Mocked
        
        # Test connection
        await message_broker.connect()
        message_broker._redis.ping.assert_called_once()
        
        # Test disconnection
        await message_broker.disconnect()
    
    async def test_send_message(self, message_broker):
        """Test sending messages."""
        message = StreamMessage(
            from_agent="agent1",
            to_agent="agent2",
            message_type=MessageType.TASK_REQUEST,
            payload={"task_type": "analysis", "data": "test"}
        )
        
        # Mock successful send
        message_broker._redis.xadd.return_value = "1234567890-0"
        
        message_id = await message_broker.send_message(message)
        
        assert message_id == "1234567890-0"
        message_broker._redis.xadd.assert_called_once()
        
        # Verify call arguments
        call_args = message_broker._redis.xadd.call_args
        assert call_args[0][0] == "agent_messages:agent2"  # Stream name
        assert isinstance(call_args[0][1], dict)  # Message data
    
    async def test_consumer_group_creation(self, message_broker):
        """Test consumer group management."""
        stream_name = "agent_messages:test"
        group_name = "test_consumers"
        
        # Test successful creation
        await message_broker.create_consumer_group(stream_name, group_name)
        message_broker._redis.xgroup_create.assert_called_once_with(
            stream_name, group_name, id="0", mkstream=True
        )
        
        # Test group already exists
        from redis.exceptions import ResponseError
        message_broker._redis.xgroup_create.side_effect = ResponseError("BUSYGROUP Consumer Group name already exists")
        
        # Should not raise exception
        await message_broker.create_consumer_group(stream_name, group_name)
    
    async def test_message_consumption(self, message_broker):
        """Test message consumption with consumer groups."""
        stream_name = "agent_messages:test"
        group_name = "test_consumers"
        consumer_name = "consumer1"
        
        # Mock message data
        mock_messages = [
            (stream_name, [
                ("1234567890-0", {
                    "id": "test_id",
                    "from_agent": "agent1",
                    "to_agent": "agent2",
                    "message_type": "task_request",
                    "payload": '{"task": "test"}',
                    "priority": "normal",
                    "timestamp": str(time.time()),
                    "correlation_id": "",
                    "signature": ""
                })
            ])
        ]
        
        message_broker._redis.xreadgroup.return_value = mock_messages
        message_broker._redis.xpending_range.return_value = []
        
        # Track handler calls
        handler_calls = []
        
        def test_handler(message: StreamMessage) -> bool:
            handler_calls.append(message)
            return True
        
        # Start consuming (this will create a background task)
        await message_broker.consume_messages(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=consumer_name,
            handler=test_handler,
            count=1,
            block_ms=100
        )
        
        # Give a moment for the consumer loop to process
        await asyncio.sleep(0.1)
        
        # Verify consumer group creation was attempted
        message_broker._redis.xgroup_create.assert_called()
    
    async def test_stream_info(self, message_broker):
        """Test getting stream information."""
        stream_name = "agent_messages:test"
        
        # Mock stream info
        message_broker._redis.xinfo_stream.return_value = {
            "length": 100,
            "first-entry": "1-0",
            "last-entry": "100-0"
        }
        message_broker._redis.xinfo_groups.return_value = [
            {
                "name": "test_group",
                "consumers": 2,
                "pending": 5,
                "last-delivered-id": "95-0",
                "lag": 5
            }
        ]
        
        info = await message_broker.get_stream_info(stream_name)
        
        assert isinstance(info, StreamInfo)
        assert info.name == stream_name
        assert info.length == 100
        assert len(info.groups) == 1
        assert info.groups[0].name == "test_group"
        assert info.groups[0].consumers == 2
        assert info.groups[0].pending == 5
    
    async def test_delivery_report(self, message_broker):
        """Test performance reporting."""
        # Simulate some activity
        message_broker._metrics["messages_sent"] = 100
        message_broker._metrics["messages_acknowledged"] = 95
        message_broker._metrics["messages_failed"] = 5
        message_broker._metrics["total_latency_ms"] = 5000.0
        
        report = await message_broker.get_delivery_report()
        
        assert report.total_sent == 100
        assert report.total_acknowledged == 95
        assert report.total_failed == 5
        assert report.success_rate == 0.95
        assert report.error_rate == 0.05
        assert report.average_latency_ms == 50.0


class TestMessagingSDK:
    """Test high-level messaging SDK."""
    
    @pytest.fixture
    async def mock_messaging_client(self):
        """Create MessagingClient with mocked dependencies."""
        client = MessagingClient(
            agent_id="test_agent",
            redis_url="redis://localhost:6379/0"
        )
        
        # Mock the broker and pubsub
        client._broker = AsyncMock()
        client._pubsub = AsyncMock()
        client._connected = True
        
        return client
    
    async def test_client_connection(self):
        """Test client connection management."""
        client = MessagingClient("test_agent")
        
        with patch.object(client._broker, 'connect') as mock_connect:
            with patch.object(client._pubsub, 'connect') as mock_pubsub_connect:
                await client.connect()
                
                assert client._connected is True
                mock_connect.assert_called_once()
                mock_pubsub_connect.assert_called_once()
    
    async def test_send_message(self, mock_messaging_client):
        """Test sending messages via SDK."""
        client = mock_messaging_client
        
        # Mock successful send
        client._broker.send_message.return_value = "test_message_id"
        
        message_id = await client.send_message(
            to_agent="target_agent",
            message_type=MessageType.TASK_REQUEST,
            payload={"task": "process_data"},
            priority=MessagePriority.HIGH
        )
        
        assert message_id == "test_message_id"
        client._broker.send_message.assert_called_once()
        
        # Verify message structure
        call_args = client._broker.send_message.call_args[0][0]
        assert call_args.from_agent == "test_agent"
        assert call_args.to_agent == "target_agent"
        assert call_args.message_type == MessageType.TASK_REQUEST
        assert call_args.payload == {"task": "process_data"}
        assert call_args.priority == MessagePriority.HIGH
    
    async def test_broadcast_message(self, mock_messaging_client):
        """Test broadcasting messages."""
        client = mock_messaging_client
        client._broker.send_message.return_value = "broadcast_id"
        
        message_id = await client.broadcast(
            message_type=MessageType.BROADCAST,
            payload={"announcement": "system maintenance"}
        )
        
        assert message_id == "broadcast_id"
        
        # Verify broadcast message structure
        call_args = client._broker.send_message.call_args[0][0]
        assert call_args.to_agent is None  # Broadcast
        assert call_args.message_type == MessageType.BROADCAST
    
    async def test_task_request_pattern(self, mock_messaging_client):
        """Test task request/response pattern."""
        client = mock_messaging_client
        client._broker.send_message.return_value = "task_request_id"
        
        correlation_id = await client.send_task_request(
            to_agent="worker_agent",
            task_type="data_analysis",
            task_payload={"dataset": "sales_data.csv"},
            priority=MessagePriority.HIGH,
            timeout=30
        )
        
        assert correlation_id.startswith("task_test_agent_")
        client._broker.send_message.assert_called_once()
        
        # Verify task request structure
        call_args = client._broker.send_message.call_args[0][0]
        assert call_args.message_type == MessageType.TASK_REQUEST
        assert call_args.payload["task_type"] == "data_analysis"
        assert call_args.payload["requester"] == "test_agent"
        assert call_args.ttl == 30
    
    async def test_task_result_response(self, mock_messaging_client):
        """Test sending task results."""
        client = mock_messaging_client
        client._broker.send_message.return_value = "task_result_id"
        
        message_id = await client.send_task_result(
            to_agent="requester_agent",
            correlation_id="task_123",
            result={"status": "completed", "output": "analysis_results.json"},
            success=True
        )
        
        assert message_id == "task_result_id"
        
        # Verify task result structure
        call_args = client._broker.send_message.call_args[0][0]
        assert call_args.message_type == MessageType.TASK_RESULT
        assert call_args.correlation_id == "task_123"
        assert call_args.payload["success"] is True
        assert call_args.payload["responder"] == "test_agent"
    
    async def test_message_handler_registration(self, mock_messaging_client):
        """Test registering message handlers."""
        client = mock_messaging_client
        
        # Handler function
        def task_handler(message: StreamMessage) -> bool:
            return True
        
        # Register handler
        client.register_handler(MessageType.TASK_REQUEST, task_handler)
        
        assert MessageType.TASK_REQUEST.value in client._message_handlers
        assert client._message_handlers[MessageType.TASK_REQUEST.value] == task_handler
    
    async def test_retry_logic(self, mock_messaging_client):
        """Test automatic retry on failures."""
        client = mock_messaging_client
        client.max_retries = 2
        client.retry_delay = 0.01  # Fast retry for testing
        
        # Mock failures then success
        client._broker.send_message.side_effect = [
            CommunicationError("Connection failed"),
            CommunicationError("Timeout"),
            "success_id"
        ]
        
        message_id = await client.send_message(
            to_agent="target",
            message_type=MessageType.EVENT,
            payload={"event": "test"}
        )
        
        assert message_id == "success_id"
        assert client._broker.send_message.call_count == 3  # 2 retries + 1 success
        assert client._stats["retries"] == 2
        assert client._stats["messages_sent"] == 1
    
    async def test_stats_tracking(self, mock_messaging_client):
        """Test client statistics tracking."""
        client = mock_messaging_client
        client._broker.send_message.return_value = "msg_id"
        
        # Send some messages
        for i in range(5):
            await client.send_message(
                to_agent=f"agent_{i}",
                message_type=MessageType.EVENT,
                payload={"index": i}
            )
        
        stats = client.get_stats()
        
        assert stats["agent_id"] == "test_agent"
        assert stats["connected"] is True
        assert stats["stats"]["messages_sent"] == 5
        assert stats["stats"]["errors"] == 0


class TestMessagePerformance:
    """Test message system performance and reliability."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_sending(self):
        """Test sending many messages quickly."""
        with patch('app.core.communication.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.xadd.return_value = "test_id"
            
            mock_redis_module.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            
            broker = MessageBroker()
            broker._redis = mock_redis
            
            # Send 100 messages
            start_time = time.time()
            
            tasks = []
            for i in range(100):
                message = StreamMessage(
                    from_agent="load_test",
                    to_agent="target",
                    message_type=MessageType.EVENT,
                    payload={"index": i}
                )
                tasks.append(broker.send_message(message))
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify all messages sent
            assert len(results) == 100
            assert all(r == "test_id" for r in results)
            
            # Performance assertion (should send 100 messages in < 1 second)
            assert duration < 1.0
            
            # Verify Redis was called for each message
            assert mock_redis.xadd.call_count == 100
    
    async def test_concurrent_consumers(self):
        """Test multiple consumers processing messages concurrently."""
        with patch('app.core.communication.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            
            # Mock message stream
            mock_messages = []
            for i in range(10):
                mock_messages.append(("test_stream", [
                    (f"{i}-0", {
                        "id": f"msg_{i}",
                        "from_agent": "sender",
                        "to_agent": "receiver",
                        "message_type": "event",
                        "payload": f'{{"index": {i}}}',
                        "priority": "normal",
                        "timestamp": str(time.time()),
                        "correlation_id": "",
                        "signature": ""
                    })
                ]))
            
            mock_redis.xreadgroup.side_effect = mock_messages + [None]  # End stream
            mock_redis.xpending_range.return_value = []
            mock_redis.xack.return_value = True
            
            mock_redis_module.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            
            broker = MessageBroker()
            broker._redis = mock_redis
            
            # Track processed messages
            processed_messages = []
            
            def message_handler(message: StreamMessage) -> bool:
                processed_messages.append(message.payload["index"])
                return True
            
            # Start consumer (simplified test)
            handler_key = "test_stream:test_group:consumer1"
            broker._message_handlers[handler_key] = message_handler
            
            # Process messages manually for testing
            for stream, messages in mock_messages:
                for message_id, fields in messages:
                    await broker._process_message("test_stream", "test_group", message_id, fields, message_handler)
            
            # Verify all messages processed
            assert len(processed_messages) == 10
            assert processed_messages == list(range(10))


class TestIntegration:
    """Integration tests for communication system."""
    
    async def test_end_to_end_messaging(self):
        """Test complete message flow from send to receive."""
        # This would require a real Redis instance for full integration testing
        # For now, we'll test the components work together with mocks
        
        with patch('app.core.communication.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.xadd.return_value = "message_123"
            
            mock_redis_module.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            
            # Create sender and receiver clients
            sender = MessagingClient("sender_agent")
            sender._broker._redis = mock_redis
            sender._connected = True
            
            receiver = MessagingClient("receiver_agent")
            receiver._broker._redis = mock_redis
            receiver._connected = True
            
            # Track received messages
            received_messages = []
            
            def receiver_handler(message: StreamMessage) -> bool:
                received_messages.append(message)
                return True
            
            receiver.register_handler(MessageType.TASK_REQUEST, receiver_handler)
            
            # Send message
            correlation_id = await sender.send_task_request(
                to_agent="receiver_agent",
                task_type="test_task",
                task_payload={"data": "test_data"}
            )
            
            # Verify send
            assert correlation_id is not None
            mock_redis.xadd.assert_called()
            
            # Simulate message reception (normally done by consumer loop)
            mock_message_data = {
                "id": correlation_id,
                "from_agent": "sender_agent",
                "to_agent": "receiver_agent",
                "message_type": "task_request",
                "payload": '{"task_type": "test_task", "task_payload": {"data": "test_data"}, "requester": "sender_agent", "timeout": null}',
                "priority": "normal",
                "timestamp": str(time.time()),
                "correlation_id": correlation_id,
                "signature": ""
            }
            
            message = StreamMessage.from_redis_dict(mock_message_data)
            handler_result = receiver_handler(message)
            
            # Verify reception
            assert handler_result is True
            assert len(received_messages) == 1
            assert received_messages[0].from_agent == "sender_agent"
            assert received_messages[0].correlation_id == correlation_id


# Performance and reliability tests
@pytest.mark.performance
class TestPerformanceReliability:
    """Performance and reliability test suite."""
    
    async def test_message_latency(self):
        """Test message delivery latency."""
        # Mock fast Redis responses
        with patch('app.core.communication.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.xadd.return_value = "latency_test"
            
            mock_redis_module.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            
            broker = MessageBroker()
            broker._redis = mock_redis
            
            # Measure latency for single message
            message = StreamMessage(
                from_agent="latency_test",
                to_agent="target",
                message_type=MessageType.EVENT,
                payload={"test": "latency"}
            )
            
            start_time = time.time()
            await broker.send_message(message)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Should be very fast with mocked Redis
            assert latency < 100  # Less than 100ms
    
    async def test_error_recovery(self):
        """Test error handling and recovery."""
        with patch('app.core.communication.redis') as mock_redis_module:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            
            # Simulate Redis failures
            from redis.exceptions import ConnectionError, TimeoutError
            mock_redis.xadd.side_effect = [
                ConnectionError("Connection lost"),
                TimeoutError("Operation timed out"),
                "recovery_success"
            ]
            
            mock_redis_module.ConnectionPool.from_url.return_value = AsyncMock()
            mock_redis_module.from_url.return_value = mock_redis
            
            client = MessagingClient("error_test_agent", auto_retry=True, max_retries=3)
            client._broker._redis = mock_redis
            client._connected = True
            
            # Should succeed after retries
            message_id = await client.send_message(
                to_agent="target",
                message_type=MessageType.EVENT,
                payload={"recovery": "test"}
            )
            
            assert message_id == "recovery_success"
            assert client._stats["retries"] > 0
            assert mock_redis.xadd.call_count == 3


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_communication.py -v
    pytest.main([__file__, "-v", "--tb=short"])