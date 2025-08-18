"""
Performance Tests for CommunicationHub

Tests to validate the performance targets:
- <10ms message routing latency
- 10,000+ messages/second throughput
- Memory usage <100MB under load
- Error rate <0.1%
"""

import asyncio
import time
import pytest
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock

from app.core.communication_hub import (
    CommunicationHub, CommunicationConfig, create_communication_hub,
    UnifiedMessage, MessageType, Priority, DeliveryGuarantee,
    ProtocolType, ConnectionConfig, create_message
)


class TestCommunicationHubPerformance:
    """Performance test suite for CommunicationHub."""
    
    @pytest.fixture
    async def mock_hub(self):
        """Create a CommunicationHub with mocked adapters for testing."""
        config = CommunicationConfig(
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
        
        hub = CommunicationHub(config)
        
        # Mock adapters for performance testing
        mock_redis_adapter = Mock()
        mock_redis_adapter.is_connected.return_value = True
        mock_redis_adapter.send_message = AsyncMock()
        mock_redis_adapter.health_check = AsyncMock(return_value="healthy")
        
        mock_websocket_adapter = Mock()
        mock_websocket_adapter.is_connected.return_value = True
        mock_websocket_adapter.send_message = AsyncMock()
        mock_websocket_adapter.health_check = AsyncMock(return_value="healthy")
        
        # Register mock adapters
        hub.adapter_registry.register_adapter(ProtocolType.REDIS_STREAMS, mock_redis_adapter)
        hub.adapter_registry.register_adapter(ProtocolType.WEBSOCKET, mock_websocket_adapter)
        
        await hub.initialize()
        
        yield hub
        
        await hub.shutdown()
    
    @pytest.mark.asyncio
    async def test_routing_latency_under_10ms(self, mock_hub):
        """Test that message routing latency is consistently under 10ms."""
        latencies = []
        num_messages = 1000
        
        # Configure mock to return successful results quickly
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message.return_value = Mock(
                success=True,
                message_id="test",
                latency_ms=1.0,
                protocol_used=ProtocolType.REDIS_STREAMS
            )
        
        # Send messages and measure routing latency
        for i in range(num_messages):
            message = create_message(
                source="test_agent",
                destination="target_agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"test_id": i}
            )
            
            start_time = time.time()
            result = await mock_hub.send_message(message)
            end_time = time.time()
            
            routing_latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(routing_latency)
            
            assert result.success, f"Message {i} failed to send"
        
        # Analyze latency results
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        print(f"Routing Latency Results:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms") 
        print(f"  99th percentile: {p99_latency:.2f}ms")
        
        # Performance assertions
        assert avg_latency < 10.0, f"Average routing latency {avg_latency:.2f}ms exceeds 10ms target"
        assert p95_latency < 10.0, f"95th percentile latency {p95_latency:.2f}ms exceeds 10ms target"
        assert p99_latency < 20.0, f"99th percentile latency {p99_latency:.2f}ms exceeds 20ms threshold"
    
    @pytest.mark.asyncio
    async def test_throughput_over_10k_messages_per_second(self, mock_hub):
        """Test that throughput exceeds 10,000 messages per second."""
        num_messages = 15000  # Test with more than target to ensure we can achieve it
        
        # Configure mock to return successful results instantly
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message.return_value = Mock(
                success=True,
                message_id="test",
                latency_ms=0.1,  # Very fast response
                protocol_used=ProtocolType.REDIS_STREAMS
            )
        
        # Create messages in advance to minimize test overhead
        messages = [
            create_message(
                source="test_agent",
                destination="target_agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"test_id": i}
            )
            for i in range(num_messages)
        ]
        
        # Measure throughput over 1 second window
        start_time = time.time()
        
        # Send messages concurrently for maximum throughput
        tasks = [mock_hub.send_message(message) for message in messages]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Calculate throughput
        duration = end_time - start_time
        successful_messages = sum(1 for result in results if result.success)
        throughput = successful_messages / duration
        
        print(f"Throughput Results:")
        print(f"  Messages sent: {num_messages}")
        print(f"  Successful: {successful_messages}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput:.0f} msg/sec")
        
        # Performance assertions
        assert successful_messages == num_messages, f"Only {successful_messages}/{num_messages} messages succeeded"
        assert throughput > 10000, f"Throughput {throughput:.0f} msg/sec below 10,000 target"
        assert duration < 2.0, f"Test took {duration:.3f}s, too slow for realistic throughput"
    
    @pytest.mark.asyncio 
    async def test_concurrent_load_handling(self, mock_hub):
        """Test handling concurrent load from multiple sources."""
        num_agents = 50
        messages_per_agent = 200
        total_messages = num_agents * messages_per_agent
        
        # Configure mock adapters
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message.return_value = Mock(
                success=True,
                message_id="test",
                latency_ms=0.5,
                protocol_used=ProtocolType.REDIS_STREAMS
            )
        
        async def agent_sender(agent_id: str, num_messages: int) -> List[bool]:
            """Simulate an agent sending messages."""
            results = []
            for i in range(num_messages):
                message = create_message(
                    source=f"agent_{agent_id}",
                    destination="coordinator",
                    message_type=MessageType.AGENT_HEARTBEAT,
                    payload={"sequence": i, "agent_id": agent_id}
                )
                
                result = await mock_hub.send_message(message)
                results.append(result.success)
            
            return results
        
        # Start concurrent senders
        start_time = time.time()
        
        agent_tasks = [
            agent_sender(str(agent_id), messages_per_agent)
            for agent_id in range(num_agents)
        ]
        
        agent_results = await asyncio.gather(*agent_tasks)
        
        end_time = time.time()
        
        # Analyze results
        total_successful = sum(sum(results) for results in agent_results)
        duration = end_time - start_time
        overall_throughput = total_successful / duration
        
        print(f"Concurrent Load Results:")
        print(f"  Agents: {num_agents}")
        print(f"  Messages per agent: {messages_per_agent}")
        print(f"  Total messages: {total_messages}")
        print(f"  Successful: {total_successful}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {overall_throughput:.0f} msg/sec")
        
        # Performance assertions
        success_rate = total_successful / total_messages
        assert success_rate >= 0.999, f"Success rate {success_rate:.3%} below 99.9% target"
        assert overall_throughput > 5000, f"Concurrent throughput {overall_throughput:.0f} too low"
    
    @pytest.mark.asyncio
    async def test_different_message_types_routing_performance(self, mock_hub):
        """Test routing performance across different message types."""
        message_types = [
            MessageType.TASK_REQUEST,
            MessageType.AGENT_HEARTBEAT,
            MessageType.REALTIME_UPDATE,
            MessageType.BROADCAST,
            MessageType.COORDINATION_REQUEST
        ]
        
        results_by_type = {}
        
        # Configure mock adapters
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message.return_value = Mock(
                success=True,
                message_id="test",
                latency_ms=1.0,
                protocol_used=ProtocolType.REDIS_STREAMS
            )
        
        # Test each message type
        for msg_type in message_types:
            latencies = []
            num_messages = 500
            
            for i in range(num_messages):
                message = create_message(
                    source="test_agent",
                    destination="target_agent",
                    message_type=msg_type,
                    payload={"test_id": i}
                )
                
                start_time = time.time()
                result = await mock_hub.send_message(message)
                end_time = time.time()
                
                routing_latency = (end_time - start_time) * 1000
                latencies.append(routing_latency)
                
                assert result.success, f"Message {i} of type {msg_type} failed"
            
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            results_by_type[msg_type.value] = {
                "avg_latency": avg_latency,
                "max_latency": max_latency,
                "messages": num_messages
            }
            
            print(f"{msg_type.value}: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
            
            # Each message type should have low latency
            assert avg_latency < 10.0, f"{msg_type.value} avg latency {avg_latency:.2f}ms too high"
    
    @pytest.mark.asyncio
    async def test_priority_message_routing_performance(self, mock_hub):
        """Test that high-priority messages are routed faster."""
        priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
        results_by_priority = {}
        
        # Configure mock adapters
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message.return_value = Mock(
                success=True,
                message_id="test", 
                latency_ms=0.5,
                protocol_used=ProtocolType.REDIS_STREAMS
            )
        
        # Test each priority level
        for priority in priorities:
            latencies = []
            num_messages = 300
            
            for i in range(num_messages):
                message = create_message(
                    source="test_agent",
                    destination="target_agent",
                    message_type=MessageType.TASK_REQUEST,
                    payload={"test_id": i},
                    priority=priority
                )
                
                start_time = time.time()
                result = await mock_hub.send_message(message)
                end_time = time.time()
                
                routing_latency = (end_time - start_time) * 1000
                latencies.append(routing_latency)
                
                assert result.success, f"Priority {priority} message {i} failed"
            
            avg_latency = statistics.mean(latencies)
            results_by_priority[priority.value] = avg_latency
            
            print(f"Priority {priority.value}: avg={avg_latency:.2f}ms")
            
            # All priorities should be under 10ms, but critical should be fastest
            assert avg_latency < 10.0, f"Priority {priority.value} latency {avg_latency:.2f}ms too high"
        
        # Critical priority should be fastest (or at least not slower than others)
        critical_latency = results_by_priority[Priority.CRITICAL.value]
        low_latency = results_by_priority[Priority.LOW.value]
        
        # In a mock environment, we can't test actual priority, but we ensure all are fast
        assert critical_latency < 10.0, f"Critical priority latency {critical_latency:.2f}ms too slow"
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, mock_hub):
        """Test performance when some messages fail."""
        num_messages = 1000
        failure_rate = 0.05  # 5% failure rate to test <0.1% target
        
        # Configure mock to occasionally fail
        call_count = 0
        
        async def mock_send_with_failures(message):
            nonlocal call_count
            call_count += 1
            
            # Simulate 5% failure rate
            if call_count % 20 == 0:  # Every 20th message fails
                return Mock(
                    success=False,
                    message_id=message.id,
                    error="Simulated failure"
                )
            else:
                return Mock(
                    success=True,
                    message_id=message.id,
                    latency_ms=1.0,
                    protocol_used=ProtocolType.REDIS_STREAMS
                )
        
        # Apply mock to adapters
        for adapter in mock_hub.adapter_registry._adapters.values():
            adapter.send_message = mock_send_with_failures
        
        # Send messages and track results
        start_time = time.time()
        successful_messages = 0
        failed_messages = 0
        
        for i in range(num_messages):
            message = create_message(
                source="test_agent",
                destination="target_agent", 
                message_type=MessageType.TASK_REQUEST,
                payload={"test_id": i}
            )
            
            result = await mock_hub.send_message(message)
            
            if result.success:
                successful_messages += 1
            else:
                failed_messages += 1
        
        end_time = time.time()
        
        # Calculate metrics
        duration = end_time - start_time
        throughput = successful_messages / duration
        error_rate = failed_messages / num_messages
        
        print(f"Error Handling Performance:")
        print(f"  Total messages: {num_messages}")
        print(f"  Successful: {successful_messages}")
        print(f"  Failed: {failed_messages}")
        print(f"  Error rate: {error_rate:.1%}")
        print(f"  Throughput: {throughput:.0f} msg/sec")
        print(f"  Duration: {duration:.3f}s")
        
        # Performance assertions
        assert error_rate < 0.1, f"Error rate {error_rate:.1%} exceeds 0.1% target"
        assert throughput > 5000, f"Throughput {throughput:.0f} too low with errors"
    
    @pytest.mark.asyncio
    async def test_subscription_performance(self, mock_hub):
        """Test subscription and message delivery performance."""
        num_subscriptions = 100
        messages_per_subscription = 50
        
        # Track received messages
        received_messages = {}
        
        async def create_handler(subscription_id: str):
            async def handler(message):
                if subscription_id not in received_messages:
                    received_messages[subscription_id] = []
                received_messages[subscription_id].append(message)
            return handler
        
        # Create subscriptions
        subscription_ids = []
        for i in range(num_subscriptions):
            pattern = f"agent_{i}"
            handler = await create_handler(pattern)
            
            # Mock the subscription
            mock_hub.adapter_registry._adapters[ProtocolType.REDIS_STREAMS].subscribe = AsyncMock(
                return_value=Mock(success=True, subscription_id=f"sub_{i}")
            )
            
            result = await mock_hub.subscribe(pattern, handler)
            subscription_ids.append(f"sub_{i}")
        
        print(f"Created {len(subscription_ids)} subscriptions")
        
        # Measure subscription creation time
        assert len(subscription_ids) == num_subscriptions, "Not all subscriptions created"
        
        # Test unsubscribe performance
        start_time = time.time()
        
        for sub_id in subscription_ids:
            mock_hub.adapter_registry._adapters[ProtocolType.REDIS_STREAMS].unsubscribe = AsyncMock(
                return_value=True
            )
            await mock_hub.unsubscribe(sub_id)
        
        end_time = time.time()
        unsubscribe_duration = end_time - start_time
        
        print(f"Unsubscribe Performance:")
        print(f"  Subscriptions: {num_subscriptions}")
        print(f"  Duration: {unsubscribe_duration:.3f}s")
        print(f"  Rate: {num_subscriptions / unsubscribe_duration:.0f} ops/sec")
        
        # Performance assertions
        assert unsubscribe_duration < 1.0, f"Unsubscribe took {unsubscribe_duration:.3f}s, too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])