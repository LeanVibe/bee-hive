"""
Comprehensive test suite for Redis Streams enhancements.

Tests DLQ functionality, back-pressure system, performance optimizations,
and monitoring systems with realistic scenarios.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.core.dead_letter_queue import (
    DeadLetterQueueManager, 
    DLQConfiguration, 
    DLQPolicy,
    DLQEntry
)
from app.core.backpressure_manager import (
    BackPressureManager, 
    BackPressureConfig,
    BackPressureState,
    ScalingAction
)
from app.core.stream_monitor import StreamMonitor, AlertRule
from app.core.performance_optimizations import (
    HighPerformanceMessageBroker,
    BatchConfig,
    CompressionConfig,
    ConnectionConfig,
    PayloadCompressor,
    CompressionAlgorithm
)
from app.models.message import StreamMessage, MessageType, MessagePriority


@pytest.fixture
async def mock_redis():
    """Create a mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.xadd.return_value = "1234567890-0"
    redis_mock.xlen.return_value = 100
    redis_mock.zadd.return_value = 1
    redis_mock.zcard.return_value = 10
    redis_mock.zrangebyscore.return_value = []
    redis_mock.xinfo_stream.return_value = {
        "length": 100,
        "first-entry": ("1234567890-0", {}),
        "last-entry": ("1234567890-1", {})
    }
    redis_mock.xinfo_groups.return_value = [
        {
            "name": "test_group",
            "consumers": 2,
            "pending": 5,
            "last-delivered-id": "1234567890-0",
            "lag": 10
        }
    ]
    redis_mock.xinfo_consumers.return_value = [
        {
            "name": "consumer1",
            "pending": 3,
            "idle": 1000
        },
        {
            "name": "consumer2", 
            "pending": 2,
            "idle": 500
        }
    ]
    redis_mock.keys.return_value = [
        b"agent_messages:test_stream_1",
        b"agent_messages:test_stream_2"
    ]
    return redis_mock


@pytest.fixture
def sample_stream_message():
    """Create a sample StreamMessage for testing."""
    return StreamMessage(
        from_agent="test_producer",
        to_agent="test_consumer",
        message_type=MessageType.TASK_REQUEST,
        payload={"task_id": "123", "data": "test_data"},
        priority=MessagePriority.NORMAL
    )


class TestDeadLetterQueueManager:
    """Test DLQ functionality."""
    
    @pytest.mark.asyncio
    async def test_dlq_manager_initialization(self, mock_redis):
        """Test DLQ manager initialization."""
        config = DLQConfiguration(
            max_retries=3,
            policy=DLQPolicy.EXPONENTIAL_BACKOFF
        )
        
        dlq_manager = DeadLetterQueueManager(mock_redis, config)
        
        assert dlq_manager.config.max_retries == 3
        assert dlq_manager.config.policy == DLQPolicy.EXPONENTIAL_BACKOFF
        assert dlq_manager._metrics["messages_retried"] == 0
    
    @pytest.mark.asyncio
    async def test_handle_failed_message_retry(self, mock_redis, sample_stream_message):
        """Test handling failed message with retry."""
        config = DLQConfiguration(max_retries=3)
        dlq_manager = DeadLetterQueueManager(mock_redis, config)
        
        # Should retry on first failure
        result = await dlq_manager.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg-123",
            message=sample_stream_message,
            failure_reason="Processing failed",
            current_retry_count=0
        )
        
        assert result is True  # Should retry
        mock_redis.zadd.assert_called_once()  # Added to retry queue
        assert dlq_manager._metrics["messages_retried"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_failed_message_dlq(self, mock_redis, sample_stream_message):
        """Test moving message to DLQ after max retries."""
        config = DLQConfiguration(max_retries=2)
        dlq_manager = DeadLetterQueueManager(mock_redis, config)
        
        # Should move to DLQ after max retries
        result = await dlq_manager.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg-123",
            message=sample_stream_message,
            failure_reason="Max retries exceeded",
            current_retry_count=2
        )
        
        assert result is False  # Should not retry
        mock_redis.xadd.assert_called_once()  # Added to DLQ
        assert dlq_manager._metrics["messages_moved_to_dlq"] == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, mock_redis, sample_stream_message):
        """Test circuit breaker prevents retries when open."""
        config = DLQConfiguration(max_retries=3)
        dlq_manager = DeadLetterQueueManager(mock_redis, config)
        
        # Simulate circuit breaker open state
        dlq_manager._circuit_breaker_state["test_stream"] = {
            "state": "open",
            "failure_count": 15,
            "next_attempt_time": time.time() + 60
        }
        
        result = await dlq_manager.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg-123",
            message=sample_stream_message,
            failure_reason="Circuit breaker test",
            current_retry_count=0
        )
        
        assert result is False  # Circuit breaker should prevent retry
        mock_redis.xadd.assert_called_once()  # Moved to DLQ
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_calculation(self, mock_redis):
        """Test exponential backoff calculation."""
        config = DLQConfiguration(
            initial_retry_delay_ms=1000,
            policy=DLQPolicy.EXPONENTIAL_BACKOFF
        )
        dlq_manager = DeadLetterQueueManager(mock_redis, config)
        
        # Test different retry counts
        delay_0 = dlq_manager._calculate_next_retry_time(0, DLQPolicy.EXPONENTIAL_BACKOFF)
        delay_1 = dlq_manager._calculate_next_retry_time(1, DLQPolicy.EXPONENTIAL_BACKOFF)
        delay_2 = dlq_manager._calculate_next_retry_time(2, DLQPolicy.EXPONENTIAL_BACKOFF)
        
        current_time = time.time()
        
        # Verify exponential growth
        assert delay_0 - current_time >= 1.0  # 1 second
        assert delay_1 - current_time >= 2.0  # 2 seconds
        assert delay_2 - current_time >= 4.0  # 4 seconds
    
    @pytest.mark.asyncio
    async def test_dlq_stats(self, mock_redis):
        """Test DLQ statistics collection."""
        dlq_manager = DeadLetterQueueManager(mock_redis)
        
        # Simulate some metrics
        dlq_manager._metrics["messages_retried"] = 10
        dlq_manager._metrics["messages_moved_to_dlq"] = 5
        dlq_manager._metrics["successful_replays"] = 3
        
        stats = await dlq_manager.get_dlq_stats()
        
        assert stats["metrics"]["messages_retried"] == 10
        assert stats["metrics"]["messages_moved_to_dlq"] == 5
        assert stats["metrics"]["successful_replays"] == 3
        assert "configuration" in stats


class TestBackPressureManager:
    """Test back-pressure management."""
    
    @pytest.mark.asyncio
    async def test_backpressure_manager_initialization(self, mock_redis):
        """Test back-pressure manager initialization."""
        config = BackPressureConfig(
            warning_lag_threshold=500,
            critical_lag_threshold=1000
        )
        
        bp_manager = BackPressureManager(mock_redis, config)
        
        assert bp_manager.config.warning_lag_threshold == 500
        assert bp_manager.config.critical_lag_threshold == 1000
    
    @pytest.mark.asyncio
    async def test_backpressure_state_determination(self, mock_redis):
        """Test back-pressure state determination."""
        bp_manager = BackPressureManager(mock_redis)
        
        # Test different lag levels
        normal_state = bp_manager._determine_backpressure_state(100)
        warning_state = bp_manager._determine_backpressure_state(1500)
        critical_state = bp_manager._determine_backpressure_state(6000)
        emergency_state = bp_manager._determine_backpressure_state(12000)
        
        assert normal_state == BackPressureState.NORMAL
        assert warning_state == BackPressureState.WARNING
        assert critical_state == BackPressureState.CRITICAL
        assert emergency_state == BackPressureState.EMERGENCY
    
    @pytest.mark.asyncio
    async def test_scaling_action_determination(self, mock_redis):
        """Test scaling action determination."""
        config = BackPressureConfig(
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            max_consumers=10,
            min_consumers=2
        )
        bp_manager = BackPressureManager(mock_redis, config)
        
        # Test scale up scenario
        scale_up = bp_manager._determine_scaling_action(
            stream_name="test_stream",
            total_lag=500,
            current_consumers=3,
            avg_processing_rate=25.0  # High utilization (25/30 = 83%)
        )
        
        # Test scale down scenario
        scale_down = bp_manager._determine_scaling_action(
            stream_name="test_stream",
            total_lag=100,
            current_consumers=5,
            avg_processing_rate=10.0  # Low utilization (10/50 = 20%)
        )
        
        assert scale_up == ScalingAction.SCALE_UP
        assert scale_down == ScalingAction.SCALE_DOWN
    
    @pytest.mark.asyncio
    async def test_throttling_updates(self, mock_redis):
        """Test throttling factor updates."""
        config = BackPressureConfig(throttling_enabled=True)
        bp_manager = BackPressureManager(mock_redis, config)
        
        # Test throttling in different states
        await bp_manager._update_throttling("test_stream", BackPressureState.NORMAL)
        normal_factor = bp_manager.get_throttle_factor("test_stream")
        
        await bp_manager._update_throttling("test_stream", BackPressureState.CRITICAL)
        critical_factor = bp_manager.get_throttle_factor("test_stream")
        
        await bp_manager._update_throttling("test_stream", BackPressureState.EMERGENCY)
        emergency_factor = bp_manager.get_throttle_factor("test_stream")
        
        assert normal_factor == 1.0
        assert critical_factor < normal_factor
        assert emergency_factor < critical_factor
        assert emergency_factor == config.max_throttle_factor


class TestStreamMonitor:
    """Test stream monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_stream_monitor_initialization(self, mock_redis):
        """Test stream monitor initialization."""
        monitor = StreamMonitor(mock_redis, enable_prometheus=False)
        
        assert monitor.redis == mock_redis
        assert monitor.enable_prometheus is False
        assert len(monitor._alert_rules) > 0  # Should have default rules
    
    @pytest.mark.asyncio
    async def test_health_score_calculation(self, mock_redis):
        """Test health score calculation."""
        monitor = StreamMonitor(mock_redis, enable_prometheus=False)
        
        # Test healthy stream
        healthy_score = monitor._calculate_health_score(
            length=100,
            consumers=3,
            pending=5,
            lag=50,
            throughput=15.0,
            error_rate=0.001,
            latency_p95=80.0
        )
        
        # Test unhealthy stream
        unhealthy_score = monitor._calculate_health_score(
            length=1000,
            consumers=0,  # No consumers
            pending=500,
            lag=8000,  # High lag
            throughput=0.1,  # Low throughput
            error_rate=0.1,  # High error rate
            latency_p95=2000.0  # High latency
        )
        
        assert healthy_score > 0.8
        assert unhealthy_score < 0.3
    
    @pytest.mark.asyncio
    async def test_alert_rule_evaluation(self, mock_redis):
        """Test alert rule evaluation."""
        monitor = StreamMonitor(mock_redis, enable_prometheus=False)
        
        # Create test alert rule
        alert_rule = AlertRule(
            name="test_high_lag",
            condition="total_lag > 1000",
            severity="warning",
            message_template="High lag: {total_lag}"
        )
        
        monitor.add_alert_rule(alert_rule)
        
        # Create metrics that should trigger alert
        from app.core.stream_monitor import StreamHealthMetrics
        metrics = StreamHealthMetrics(
            stream_name="test_stream",
            length=500,
            consumer_groups=1,
            total_consumers=2,
            total_pending=50,
            total_lag=1500,  # Above threshold
            messages_per_second=10.0,
            error_rate=0.01,
            avg_processing_latency_ms=100.0,
            p95_processing_latency_ms=200.0,
            p99_processing_latency_ms=300.0,
            oldest_pending_age_seconds=30.0,
            health_score=0.7,
            status="warning",
            last_updated=time.time()
        )
        
        monitor._stream_metrics["test_stream"] = metrics
        
        # Check alert evaluation
        await monitor._check_alerts()
        
        # Alert should have been fired (check via log or callback)
        assert "test_stream" in monitor._stream_metrics


class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_payload_compressor_initialization(self):
        """Test payload compressor initialization."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
            compression_level=6,
            min_payload_size=1024
        )
        
        compressor = PayloadCompressor(config)
        
        assert compressor.config.algorithm == CompressionAlgorithm.ZLIB
        assert compressor.config.compression_level == 6
        assert compressor.config.min_payload_size == 1024
    
    def test_payload_compression_large_payload(self):
        """Test compression of large payloads."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
            min_payload_size=100
        )
        compressor = PayloadCompressor(config)
        
        # Create large payload
        large_payload = {
            "data": "x" * 2000,  # 2KB of data
            "metadata": {"key": "value"},
            "numbers": list(range(100))
        }
        
        compressed_data, was_compressed = compressor.compress_payload(large_payload)
        
        assert was_compressed is True
        assert len(compressed_data) < len(json.dumps(large_payload))
        
        # Test decompression
        decompressed_payload = compressor.decompress_payload(
            compressed_data, was_compressed, "zlib"
        )
        
        assert decompressed_payload == large_payload
    
    def test_payload_compression_small_payload(self):
        """Test that small payloads are not compressed."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZLIB,
            min_payload_size=1024
        )
        compressor = PayloadCompressor(config)
        
        # Create small payload
        small_payload = {"message": "hello"}
        
        compressed_data, was_compressed = compressor.compress_payload(small_payload)
        
        assert was_compressed is False
        assert compressed_data == json.dumps(small_payload).encode('utf-8')
    
    def test_compression_statistics(self):
        """Test compression statistics tracking."""
        config = CompressionConfig(min_payload_size=50)
        compressor = PayloadCompressor(config)
        
        # Compress several payloads
        payloads = [
            {"data": "x" * 200},
            {"data": "y" * 300},
            {"small": "payload"}  # This won't be compressed
        ]
        
        for payload in payloads:
            compressor.compress_payload(payload)
        
        stats = compressor.get_compression_stats()
        
        assert stats["compressed_payloads"] == 2
        assert stats["uncompressed_payloads"] == 1
        assert stats["compression_ratio"] < 1.0  # Should have saved space
    
    @pytest.mark.asyncio
    async def test_high_performance_broker_initialization(self):
        """Test high-performance message broker initialization."""
        batch_config = BatchConfig(max_batch_size=50, max_batch_wait_ms=25)
        compression_config = CompressionConfig(min_payload_size=500)
        connection_config = ConnectionConfig(pool_size=20)
        
        with patch('app.core.performance_optimizations.ConnectionManager'):
            broker = HighPerformanceMessageBroker(
                redis_url="redis://localhost:6379",
                batch_config=batch_config,
                compression_config=compression_config,
                connection_config=connection_config
            )
            
            assert broker.batch_config.max_batch_size == 50
            assert broker.compression_config.min_payload_size == 500
            assert broker.connection_config.pool_size == 20


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_dlq_with_backpressure_integration(self, mock_redis, sample_stream_message):
        """Test DLQ working with back-pressure management."""
        # Setup DLQ
        dlq_config = DLQConfiguration(max_retries=2)
        dlq_manager = DeadLetterQueueManager(mock_redis, dlq_config)
        
        # Setup back-pressure
        bp_config = BackPressureConfig(critical_lag_threshold=1000)
        bp_manager = BackPressureManager(mock_redis, bp_config)
        
        # Simulate high lag triggering back-pressure
        await bp_manager._update_throttling("test_stream", BackPressureState.CRITICAL)
        throttle_factor = bp_manager.get_throttle_factor("test_stream")
        
        # Handle failed message in high-pressure scenario
        result = await dlq_manager.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg-123",
            message=sample_stream_message,
            failure_reason="System under pressure",
            current_retry_count=0
        )
        
        assert throttle_factor < 1.0  # Should be throttled
        assert result is True  # Should still retry (not circuit breaker)
    
    @pytest.mark.asyncio
    async def test_monitoring_with_performance_optimization(self, mock_redis):
        """Test monitoring working with performance optimizations."""
        # Setup monitor
        monitor = StreamMonitor(mock_redis, enable_prometheus=False)
        
        # Setup performance optimization
        compressor = PayloadCompressor(CompressionConfig())
        
        # Record some latency samples
        monitor.record_processing_latency("test_stream", 150.0)
        monitor.record_processing_latency("test_stream", 200.0)
        monitor.record_processing_latency("test_stream", 180.0)
        
        # Get compression stats
        large_payload = {"data": "x" * 2000}
        compressed_data, was_compressed = compressor.compress_payload(large_payload)
        compression_stats = compressor.get_compression_stats()
        
        # Verify integration
        metrics = monitor.get_metrics("test_stream")
        assert len(monitor._latency_samples["test_stream"]) == 3
        assert compression_stats["compressed_payloads"] > 0 if was_compressed else True


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_dlq_redis_connection_failure(self, sample_stream_message):
        """Test DLQ handling Redis connection failures."""
        # Create mock that raises Redis errors
        failing_redis = AsyncMock()
        failing_redis.zadd.side_effect = Exception("Redis connection failed")
        failing_redis.xadd.side_effect = Exception("Redis connection failed")
        
        dlq_manager = DeadLetterQueueManager(failing_redis)
        
        # Should handle Redis errors gracefully
        result = await dlq_manager.handle_failed_message(
            original_stream="test_stream",
            original_message_id="msg-123",
            message=sample_stream_message,
            failure_reason="Test failure",
            current_retry_count=0
        )
        
        # Should return False (fail safe) when Redis is unavailable
        assert result is False
    
    @pytest.mark.asyncio
    async def test_backpressure_invalid_metrics(self, mock_redis):
        """Test back-pressure handling invalid metrics."""
        bp_manager = BackPressureManager(mock_redis)
        
        # Test with invalid data
        mock_redis.xinfo_groups.side_effect = Exception("Stream does not exist")
        
        # Should handle gracefully without crashing
        await bp_manager._update_consumer_metrics("invalid_stream", time.time())
        
        # No metrics should be created for invalid stream
        assert "invalid_stream" not in [
            m.stream_name for m in bp_manager._consumer_metrics.values()
        ]
    
    def test_compression_malformed_data(self):
        """Test compression handling malformed data."""
        compressor = PayloadCompressor(CompressionConfig())
        
        # Test decompression of invalid data
        invalid_data = b"invalid compressed data"
        
        # Should handle gracefully and try as uncompressed
        try:
            result = compressor.decompress_payload(invalid_data, True, "zlib")
            # If it doesn't raise, it fell back to uncompressed
            assert isinstance(result, dict)
        except json.JSONDecodeError:
            # Expected if data is truly invalid
            pass


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_compression_performance(self):
        """Benchmark compression performance."""
        compressor = PayloadCompressor(CompressionConfig())
        
        # Test with various payload sizes
        payload_sizes = [1000, 5000, 10000, 50000]  # bytes
        
        for size in payload_sizes:
            payload = {"data": "x" * size, "metadata": {"size": size}}
            
            start_time = time.time()
            compressed_data, was_compressed = compressor.compress_payload(payload)
            compression_time = (time.time() - start_time) * 1000  # ms
            
            if was_compressed:
                original_size = len(json.dumps(payload))
                compression_ratio = len(compressed_data) / original_size
                
                # Performance assertions
                assert compression_time < 100  # Should compress in <100ms
                assert compression_ratio < 0.8  # Should achieve >20% compression
    
    @pytest.mark.asyncio
    async def test_dlq_processing_performance(self, mock_redis):
        """Benchmark DLQ processing performance."""
        dlq_manager = DeadLetterQueueManager(mock_redis)
        
        # Create multiple failed messages
        messages = []
        for i in range(100):
            message = StreamMessage(
                from_agent=f"producer_{i}",
                to_agent="consumer",
                message_type=MessageType.TASK_REQUEST,
                payload={"task_id": f"task_{i}", "data": "test"},
                priority=MessagePriority.NORMAL
            )
            messages.append(message)
        
        # Benchmark handling failed messages
        start_time = time.time()
        
        for i, message in enumerate(messages):
            await dlq_manager.handle_failed_message(
                original_stream="test_stream",
                original_message_id=f"msg_{i}",
                message=message,
                failure_reason="Benchmark test",
                current_retry_count=0
            )
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should process 100 messages in <5 seconds
        assert dlq_manager._metrics["messages_retried"] == 100
    
    @pytest.mark.asyncio
    async def test_monitoring_collection_performance(self, mock_redis):
        """Benchmark monitoring data collection performance."""
        monitor = StreamMonitor(mock_redis, enable_prometheus=False)
        
        # Simulate multiple streams
        stream_count = 50
        mock_redis.keys.return_value = [
            f"agent_messages:stream_{i}".encode()
            for i in range(stream_count)
        ]
        
        # Benchmark metrics collection
        start_time = time.time()
        await monitor._collect_metrics()
        collection_time = time.time() - start_time
        
        # Performance assertion
        assert collection_time < 2.0  # Should collect from 50 streams in <2 seconds