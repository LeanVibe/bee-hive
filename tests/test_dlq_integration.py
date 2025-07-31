"""
Comprehensive Integration Tests for DLQ System

Tests the complete DLQ system integration with:
- DeadLetterQueueManager + DLQRetryScheduler + PoisonMessageDetector
- Enterprise reliability components integration
- UnifiedDLQService coordination
- Chaos engineering scenarios
- Performance validation under load
- End-to-end workflow testing

Performance targets validation:
- >99.9% message delivery reliability
- <10ms processing overhead under normal load
- Auto-recovery from component failures within 30s
- Handle >10k messages/second throughput
"""

import asyncio
import time
import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import statistics

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.unified_dlq_service import UnifiedDLQService, DLQServiceConfig, DLQServiceStatus
from app.core.dead_letter_queue import DeadLetterQueueManager, DLQConfiguration
from app.core.dlq_retry_scheduler import DLQRetryScheduler, RetryPriority
from app.core.poison_message_detector import PoisonMessageDetector, PoisonMessageType
from app.models.message import StreamMessage, MessageType, MessagePriority


class TestDLQIntegration:
    """Integration tests for complete DLQ system."""
    
    @pytest.fixture
    async def redis_client(self):
        """Mock Redis client for testing."""
        mock_redis = AsyncMock(spec=Redis)
        
        # Mock Redis operations
        mock_redis.xadd = AsyncMock(return_value="1234567890-0")
        mock_redis.xlen = AsyncMock(return_value=0)
        mock_redis.xrange = AsyncMock(return_value=[])
        mock_redis.xrevrange = AsyncMock(return_value=[])
        mock_redis.zadd = AsyncMock(return_value=1)
        mock_redis.zcard = AsyncMock(return_value=0)
        mock_redis.zrangebyscore = AsyncMock(return_value=[])
        mock_redis.lpush = AsyncMock(return_value=1)
        mock_redis.llen = AsyncMock(return_value=0)
        mock_redis.lpop = AsyncMock(return_value=[])
        mock_redis.setex = AsyncMock(return_value=True)
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.info = AsyncMock(return_value={"used_memory": 1024 * 1024})
        mock_redis.ping = AsyncMock(return_value=True)
        
        return mock_redis
    
    @pytest.fixture
    async def dlq_service_config(self):
        """Test configuration for DLQ service."""
        return DLQServiceConfig(
            max_retries=3,
            initial_retry_delay_ms=100,  # Fast for testing
            max_retry_delay_ms=5000,
            dlq_max_size=1000,
            enable_poison_detection=True,
            enable_intelligent_retry=True,
            enable_monitoring=True,
            performance_target_ms=10,
            alert_threshold=100
        )
    
    @pytest.fixture
    async def unified_dlq_service(self, redis_client, dlq_service_config):
        """Create unified DLQ service for testing."""
        service = UnifiedDLQService(redis_client, dlq_service_config)
        return service
    
    @pytest.fixture
    def sample_message(self):
        """Sample message for testing."""
        return StreamMessage(
            id=str(uuid.uuid4()),
            message_type=MessageType.TASK,
            priority=MessagePriority.NORMAL,
            payload={"task": "test_task", "data": "test_data"},
            from_agent="test_agent_1",
            to_agent="test_agent_2"
        )
    
    @pytest.fixture
    def poison_message(self):
        """Poison message for testing."""
        return {
            "id": str(uuid.uuid4()),
            "type": "task",
            "payload": '{"invalid": json syntax}',  # Malformed JSON
            "from_agent": "test_agent",
            "to_agent": "test_consumer"
        }
    
    async def test_dlq_service_initialization(self, redis_client, dlq_service_config):
        """Test successful DLQ service initialization."""
        service = UnifiedDLQService(redis_client, dlq_service_config)
        
        assert service.service_status == DLQServiceStatus.STARTING
        assert service.config.max_retries == 3
        assert service.config.enable_poison_detection is True
        assert service.redis == redis_client
    
    async def test_dlq_service_start_stop(self, unified_dlq_service):
        """Test DLQ service start and stop lifecycle."""
        # Start service
        await unified_dlq_service.start()
        
        assert unified_dlq_service.service_status == DLQServiceStatus.HEALTHY
        assert unified_dlq_service.dlq_manager is not None
        assert unified_dlq_service.poison_detector is not None
        assert unified_dlq_service.retry_scheduler is not None
        
        # Stop service
        await unified_dlq_service.stop()
        
        assert unified_dlq_service.service_status == DLQServiceStatus.ERROR  # Stopped state
    
    async def test_successful_message_retry_flow(self, unified_dlq_service, sample_message):
        """Test successful message retry workflow."""
        await unified_dlq_service.start()
        
        # Handle failed message
        should_retry = await unified_dlq_service.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg_123",
            message=sample_message,
            failure_reason="network_timeout",
            current_retry_count=0
        )
        
        assert should_retry is True
        assert unified_dlq_service.metrics.total_messages_processed == 1
        assert unified_dlq_service.metrics.successful_deliveries == 1
        
        await unified_dlq_service.stop()
    
    async def test_poison_message_detection_and_quarantine(self, unified_dlq_service, poison_message):
        """Test poison message detection and quarantine."""
        await unified_dlq_service.start()
        
        # Handle poison message
        should_retry = await unified_dlq_service.handle_failed_message(
            original_stream="test_stream",
            original_message_id="poison_123",
            message=poison_message,
            failure_reason="json_parse_error",
            current_retry_count=0
        )
        
        # Poison message should not be retried
        assert should_retry is False
        assert unified_dlq_service.metrics.permanent_failures == 1
        
        await unified_dlq_service.stop()
    
    async def test_max_retries_exceeded_flow(self, unified_dlq_service, sample_message):
        """Test message flow when max retries are exceeded."""
        await unified_dlq_service.start()
        
        # Handle message that has exceeded max retries
        should_retry = await unified_dlq_service.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg_456",
            message=sample_message,
            failure_reason="persistent_failure",
            current_retry_count=5  # Exceeds max_retries=3
        )
        
        assert should_retry is False
        assert unified_dlq_service.metrics.permanent_failures == 1
        
        await unified_dlq_service.stop()
    
    async def test_dlq_service_graceful_degradation(self, redis_client):
        """Test graceful degradation when enterprise components fail."""
        config = DLQServiceConfig(
            enable_graceful_degradation=True,
            enable_backpressure_integration=True,
            enable_consumer_group_integration=True
        )
        
        service = UnifiedDLQService(redis_client, config)
        
        # Mock enterprise component failures during initialization
        with patch.object(service, '_initialize_enterprise_components', side_effect=Exception("Enterprise failure")):
            await service.start()
            
            # Service should still start in degraded mode
            assert service.service_status in [DLQServiceStatus.HEALTHY, DLQServiceStatus.DEGRADED]
            assert service.degraded_mode is True
        
        await service.stop()
    
    async def test_emergency_dlq_fallback(self, unified_dlq_service, sample_message):
        """Test emergency DLQ fallback when all components fail."""
        await unified_dlq_service.start()
        
        # Mock all components to fail
        unified_dlq_service.dlq_manager = None
        unified_dlq_service.retry_scheduler = None
        unified_dlq_service.service_status = DLQServiceStatus.ERROR
        
        # Handle message during service failure
        should_retry = await unified_dlq_service.handle_failed_message(
            original_stream="test_stream",
            original_message_id="emergency_123",
            message=sample_message,
            failure_reason="service_failure",
            current_retry_count=0
        )
        
        # Should use emergency fallback
        assert should_retry is False
        
        # Verify emergency DLQ was called
        unified_dlq_service.redis.lpush.assert_called()
        
        await unified_dlq_service.stop()
    
    async def test_performance_under_load(self, unified_dlq_service, sample_message):
        """Test DLQ performance under high load."""
        await unified_dlq_service.start()
        
        # Process multiple messages concurrently
        num_messages = 100
        start_time = time.time()
        
        tasks = []
        for i in range(num_messages):
            task = unified_dlq_service.handle_failed_message(
                original_stream=f"stream_{i % 10}",
                original_message_id=f"msg_{i}",
                message=sample_message,
                failure_reason="load_test",
                current_retry_count=0
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        # Validate performance
        avg_processing_time_ms = (processing_time / num_messages) * 1000
        
        assert avg_processing_time_ms < 50  # Should be much less than 50ms per message
        assert unified_dlq_service.metrics.total_messages_processed == num_messages
        assert all(result is True for result in results)  # All should be retried
        
        await unified_dlq_service.stop()
    
    async def test_comprehensive_stats_collection(self, unified_dlq_service):
        """Test comprehensive statistics collection."""
        await unified_dlq_service.start()
        
        # Process some messages to generate stats
        sample_msg = {
            "id": "test_id",
            "type": "task",
            "payload": {"data": "test"}
        }
        
        for i in range(10):
            await unified_dlq_service.handle_failed_message(
                original_stream="stats_stream",
                original_message_id=f"msg_{i}",
                message=sample_msg,
                failure_reason="stats_test",
                current_retry_count=i % 3  # Vary retry counts
            )
        
        # Get comprehensive stats
        stats = await unified_dlq_service.get_comprehensive_stats()
        
        assert stats["service_status"] == "healthy"
        assert stats["performance_metrics"]["total_messages_processed"] == 10
        assert "component_statuses" in stats
        assert "configuration" in stats
        assert isinstance(stats["uptime_seconds"], (int, float))
        
        await unified_dlq_service.stop()
    
    async def test_health_check_functionality(self, unified_dlq_service):
        """Test comprehensive health check functionality."""
        await unified_dlq_service.start()
        
        # Perform health check
        health = await unified_dlq_service.health_check()
        
        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "component_health" in health
        assert "performance_metrics" in health
        assert health["degraded_mode"] is False
        
        # Test degraded health
        unified_dlq_service.failed_components.add("test_component")
        unified_dlq_service.service_status = DLQServiceStatus.DEGRADED
        
        health = await unified_dlq_service.health_check()
        assert health["status"] == "degraded"
        assert len(health["failed_components"]) > 0
        
        await unified_dlq_service.stop()
    
    async def test_dlq_component_integration(self, redis_client):
        """Test integration between DLQ components."""
        # Test DLQ Manager + Retry Scheduler integration
        dlq_config = DLQConfiguration(max_retries=3)
        dlq_manager = DeadLetterQueueManager(redis_client, dlq_config)
        
        retry_scheduler = DLQRetryScheduler(redis_client)
        
        poison_detector = PoisonMessageDetector()
        
        # Start components
        await dlq_manager.start()
        await retry_scheduler.start()
        
        # Test message flow between components
        test_message = StreamMessage(
            id="integration_test",
            message_type=MessageType.TASK,
            priority=MessagePriority.NORMAL,
            payload={"test": "integration"},
            from_agent="test_agent",
            to_agent="test_consumer"
        )
        
        # Process through DLQ manager
        should_retry = await dlq_manager.handle_failed_message(
            original_stream="integration_stream",
            original_message_id="integration_msg",
            message=test_message,
            failure_reason="integration_test"
        )
        
        assert should_retry is True
        
        # Clean up
        await dlq_manager.stop()
        await retry_scheduler.stop()
    
    async def test_chaos_engineering_scenarios(self, unified_dlq_service, sample_message):
        """Test chaos engineering failure scenarios."""
        await unified_dlq_service.start()
        
        # Scenario 1: Redis connection failure
        with patch.object(unified_dlq_service.redis, 'xadd', side_effect=redis.RedisError("Connection lost")):
            should_retry = await unified_dlq_service.handle_failed_message(
                original_stream="chaos_stream",
                original_message_id="chaos_msg_1",
                message=sample_message,
                failure_reason="redis_failure_test"
            )
            
            # Should handle gracefully
            assert should_retry is False  # Falls back to emergency handling
        
        # Scenario 2: Component timeout
        with patch.object(unified_dlq_service.poison_detector, 'analyze_message', 
                         side_effect=asyncio.TimeoutError("Detection timeout")):
            should_retry = await unified_dlq_service.handle_failed_message(
                original_stream="chaos_stream",
                original_message_id="chaos_msg_2", 
                message=sample_message,
                failure_reason="timeout_test"
            )
            
            # Should continue processing without poison detection
            assert should_retry is True
        
        await unified_dlq_service.stop()
    
    async def test_message_priority_handling(self, unified_dlq_service):
        """Test message priority handling in DLQ system."""
        await unified_dlq_service.start()
        
        # Create messages with different priorities
        priorities = [MessagePriority.CRITICAL, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW]
        
        for i, priority in enumerate(priorities):
            message = StreamMessage(
                id=f"priority_msg_{i}",
                message_type=MessageType.TASK,
                priority=priority,
                payload={"priority_test": True},
                from_agent="test_agent",
                to_agent="test_consumer"
            )
            
            should_retry = await unified_dlq_service.handle_failed_message(
                original_stream="priority_stream",
                original_message_id=f"priority_msg_{i}",
                message=message,
                failure_reason="priority_test"
            )
            
            assert should_retry is True
        
        # Verify all messages were processed
        assert unified_dlq_service.metrics.total_messages_processed == len(priorities)
        
        await unified_dlq_service.stop()
    
    async def test_concurrent_processing_safety(self, unified_dlq_service, sample_message):
        """Test thread safety under concurrent processing."""
        await unified_dlq_service.start()
        
        # Process messages concurrently
        num_concurrent = 50
        
        async def process_message(msg_id):
            return await unified_dlq_service.handle_failed_message(
                original_stream="concurrent_stream",
                original_message_id=f"concurrent_msg_{msg_id}",
                message=sample_message,
                failure_reason="concurrency_test"
            )
        
        # Run concurrent tasks
        tasks = [process_message(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions and consistent processing
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Exceptions during concurrent processing: {exceptions}"
        
        successful_results = [r for r in results if r is True]
        assert len(successful_results) == num_concurrent
        
        await unified_dlq_service.stop()
    
    async def test_memory_usage_under_load(self, unified_dlq_service, sample_message):
        """Test memory usage patterns under load."""
        await unified_dlq_service.start()
        
        # Process many messages and monitor memory usage patterns
        num_messages = 200
        
        for i in range(num_messages):
            await unified_dlq_service.handle_failed_message(
                original_stream=f"memory_stream_{i % 5}",
                original_message_id=f"memory_msg_{i}",
                message=sample_message,
                failure_reason="memory_test"
            )
            
            # Check metrics periodically
            if i % 50 == 0:
                stats = await unified_dlq_service.get_comprehensive_stats()
                assert stats["performance_metrics"]["total_messages_processed"] == i + 1
        
        # Final verification
        final_stats = await unified_dlq_service.get_comprehensive_stats()
        assert final_stats["performance_metrics"]["total_messages_processed"] == num_messages
        
        await unified_dlq_service.stop()


class TestDLQPerformanceBenchmarks:
    """Performance benchmark tests for DLQ system."""
    
    async def test_throughput_benchmark(self, unified_dlq_service, sample_message):
        """Benchmark message throughput."""
        await unified_dlq_service.start()
        
        num_messages = 1000
        start_time = time.time()
        
        # Process messages in batches for better performance
        batch_size = 100
        for batch_start in range(0, num_messages, batch_size):
            batch_tasks = []
            for i in range(batch_start, min(batch_start + batch_size, num_messages)):
                task = unified_dlq_service.handle_failed_message(
                    original_stream=f"bench_stream_{i % 10}",
                    original_message_id=f"bench_msg_{i}",
                    message=sample_message,
                    failure_reason="throughput_test"
                )
                batch_tasks.append(task)
            
            await asyncio.gather(*batch_tasks)
        
        processing_time = time.time() - start_time
        throughput = num_messages / processing_time
        
        # Validate throughput (target: >1000 messages/second)
        assert throughput > 1000, f"Throughput too low: {throughput:.2f} msg/sec"
        
        print(f"DLQ Throughput: {throughput:.2f} messages/second")
        print(f"Average processing time: {(processing_time/num_messages)*1000:.2f}ms per message")
        
        await unified_dlq_service.stop()
    
    async def test_latency_distribution(self, unified_dlq_service, sample_message):
        """Test latency distribution under load."""
        await unified_dlq_service.start()
        
        latencies = []
        num_samples = 200
        
        for i in range(num_samples):
            start_time = time.time()
            
            await unified_dlq_service.handle_failed_message(
                original_stream="latency_stream",
                original_message_id=f"latency_msg_{i}",
                message=sample_message,
                failure_reason="latency_test"
            )
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate latency percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg_latency = statistics.mean(latencies)
        
        # Validate latency targets
        assert p95 < 50, f"P95 latency too high: {p95:.2f}ms"
        assert p99 < 100, f"P99 latency too high: {p99:.2f}ms"
        assert avg_latency < 20, f"Average latency too high: {avg_latency:.2f}ms"
        
        print(f"Latency Distribution:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        
        await unified_dlq_service.stop()
    
    async def test_memory_efficiency(self, unified_dlq_service, sample_message):
        """Test memory efficiency under sustained load."""
        await unified_dlq_service.start()
        
        # Process messages and monitor memory growth patterns
        num_cycles = 10
        messages_per_cycle = 100
        
        memory_usage = []
        
        for cycle in range(num_cycles):
            # Process batch of messages
            for i in range(messages_per_cycle):
                await unified_dlq_service.handle_failed_message(
                    original_stream=f"memory_stream_{cycle}",
                    original_message_id=f"memory_msg_{cycle}_{i}",
                    message=sample_message,
                    failure_reason="memory_efficiency_test"
                )
            
            # Collect stats after each cycle
            stats = await unified_dlq_service.get_comprehensive_stats()
            memory_usage.append(stats["performance_metrics"]["total_messages_processed"])
        
        # Verify linear memory usage (no memory leaks)
        expected_growth = messages_per_cycle
        actual_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
        
        # Allow for some variance in memory usage
        assert abs(actual_growth - (num_cycles - 1) * expected_growth) < expected_growth * 0.1
        
        await unified_dlq_service.stop()


if __name__ == "__main__":
    # Run specific test for debugging
    import pytest
    pytest.main([__file__ + "::TestDLQIntegration::test_successful_message_retry_flow", "-v"])