"""
Simple Test Suite for Observability Hooks System

Basic tests to validate the observability infrastructure including event processing,
serialization performance, and API functionality.
"""

import asyncio
import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.observability import EventType
from app.core.event_serialization import serialize_for_stream, deserialize_from_stream
from app.schemas.observability import BaseObservabilityEvent


@pytest.fixture
def sample_event():
    """Create a sample event for testing."""
    return BaseObservabilityEvent(
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        event_type="PreToolUse",
        event_category="tool",
        session_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        payload={
            "tool_name": "Read",
            "parameters": {"file_path": "/test/file.py"},
            "correlation_id": str(uuid.uuid4())
        }
    )


@pytest.fixture
def mock_redis():
    """Create mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.xadd = AsyncMock(return_value="test-stream-id")
    mock_client.xinfo_stream = AsyncMock(return_value={"length": 100, "groups": 1})
    mock_client.xgroup_create = AsyncMock()
    mock_client.xreadgroup = AsyncMock(return_value=[])
    mock_client.xack = AsyncMock()
    return mock_client


class TestEventSerialization:
    """Test high-performance event serialization system."""
    
    def test_serialize_for_stream_performance(self, sample_event):
        """Test that serialization meets <5ms performance requirement."""
        start_time = time.perf_counter()
        
        serialized_data, metadata = serialize_for_stream(sample_event)
        
        end_time = time.perf_counter()
        serialization_time_ms = (end_time - start_time) * 1000
        
        # Assert performance requirement
        assert serialization_time_ms < 5.0, f"Serialization took {serialization_time_ms}ms, exceeds 5ms limit"
        
        # Verify serialized data
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        assert metadata["serialization_time_ms"] < 5.0
    
    def test_deserialize_from_stream_performance(self, sample_event):
        """Test that deserialization meets <5ms performance requirement."""
        # First serialize
        serialized_data, metadata = serialize_for_stream(sample_event)
        
        # Then deserialize and measure
        start_time = time.perf_counter()
        
        deserialized_data = deserialize_from_stream(serialized_data)
        
        end_time = time.perf_counter()
        deserialization_time_ms = (end_time - start_time) * 1000
        
        # Assert performance requirement
        assert deserialization_time_ms < 5.0, f"Deserialization took {deserialization_time_ms}ms, exceeds 5ms limit"
        
        # Verify deserialized data integrity
        assert str(deserialized_data["event_id"]) == str(sample_event.event_id)
        assert deserialized_data["event_type"] == sample_event.event_type
    
    def test_batch_serialization_throughput(self, sample_event):
        """Test batch serialization meets >200 events/second throughput."""
        # Create batch of events
        batch_events = [sample_event] * 300  # 300 events
        
        start_time = time.perf_counter()
        
        # Serialize all events
        for event in batch_events:
            serialize_for_stream(event)
        
        end_time = time.perf_counter()
        total_time_seconds = end_time - start_time
        
        events_per_second = len(batch_events) / total_time_seconds
        
        # Assert throughput requirement
        assert events_per_second > 200, f"Throughput {events_per_second} events/sec is below 200 requirement"


class TestRealTimeEventProcessor:
    """Test the real-time event processor."""
    
    @pytest.mark.asyncio
    async def test_event_processing_performance(self, mock_redis, sample_event):
        """Test event processing meets performance requirements."""
        from app.observability.real_time_processor import RealTimeEventProcessor
        
        processor = RealTimeEventProcessor(redis_client=mock_redis)
        
        start_time = time.perf_counter()
        
        event_id = await processor.process_event(
            session_id=sample_event.session_id,
            agent_id=sample_event.agent_id,
            event_type=EventType.PRE_TOOL_USE,
            payload=sample_event.payload,
            latency_ms=None
        )
        
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Assert performance requirement
        assert processing_time_ms < 5.0, f"Processing took {processing_time_ms}ms, exceeds 5ms limit"
        assert isinstance(event_id, str)
        assert len(event_id) > 0
    
    @pytest.mark.asyncio
    async def test_redis_streaming_integration(self, mock_redis):
        """Test Redis streaming integration."""
        from app.observability.real_time_processor import RealTimeEventProcessor
        
        processor = RealTimeEventProcessor(redis_client=mock_redis)
        
        event_id = await processor.process_event(
            session_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            event_type=EventType.PRE_TOOL_USE,
            payload={"tool_name": "Test", "parameters": {}},
            latency_ms=100
        )
        
        # Verify Redis calls
        assert mock_redis.xadd.called
        call_args = mock_redis.xadd.call_args
        
        assert call_args[0][0] == "observability_events"  # stream name
        stream_data = call_args[0][1]
        
        assert "event_data" in stream_data
        assert "event_type" in stream_data
        assert "event_id" in stream_data
        assert stream_data["event_type"] == "PreToolUse"


class TestHookInterceptor:
    """Test the enhanced hook interceptor."""
    
    @pytest.mark.asyncio
    async def test_pre_tool_use_capture(self, mock_redis):
        """Test PreToolUse event capture."""
        from app.observability.real_time_processor import RealTimeEventProcessor
        from app.observability.hooks import HookInterceptor
        
        processor = RealTimeEventProcessor(redis_client=mock_redis)
        interceptor = HookInterceptor(processor)
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        tool_data = {
            "tool_name": "Read",
            "parameters": {"file_path": "/test.py"},
            "correlation_id": str(uuid.uuid4())
        }
        
        event_id = await interceptor.capture_pre_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_data=tool_data
        )
        
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert mock_redis.xadd.called
    
    @pytest.mark.asyncio
    async def test_post_tool_use_capture(self, mock_redis):
        """Test PostToolUse event capture."""
        from app.observability.real_time_processor import RealTimeEventProcessor
        from app.observability.hooks import HookInterceptor
        
        processor = RealTimeEventProcessor(redis_client=mock_redis)
        interceptor = HookInterceptor(processor)
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        tool_result = {
            "tool_name": "Read",
            "success": True,
            "result": "File content",
            "execution_time_ms": 150.5,
            "correlation_id": str(uuid.uuid4())
        }
        
        event_id = await interceptor.capture_post_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_result=tool_result,
            latency_ms=155
        )
        
        assert isinstance(event_id, str)
        assert len(event_id) > 0
        assert mock_redis.xadd.called


class TestPerformanceBenchmarks:
    """Performance benchmark tests to validate system requirements."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_benchmark(self, mock_redis, sample_event):
        """Test complete end-to-end performance from event capture to streaming."""
        from app.observability.real_time_processor import RealTimeEventProcessor
        from app.observability.hooks import HookInterceptor
        
        # Setup complete system
        processor = RealTimeEventProcessor(redis_client=mock_redis)
        interceptor = HookInterceptor(processor)
        
        # Test batch of events
        events = [sample_event] * 100  # 100 events
        
        # Benchmark end-to-end processing
        start_time = time.perf_counter()
        
        for event in events:
            # Capture event
            event_id = await interceptor.capture_pre_tool_use(
                session_id=event.session_id,
                agent_id=event.agent_id,
                tool_data={"tool_name": "Test", "parameters": {}}
            )
            
            assert event_id is not None
        
        end_time = time.perf_counter()
        total_time_seconds = end_time - start_time
        
        events_per_second = len(events) / total_time_seconds
        avg_time_per_event_ms = (total_time_seconds * 1000) / len(events)
        
        # Assert performance requirements
        assert events_per_second > 200, f"End-to-end throughput {events_per_second} events/sec is below 200 requirement"
        assert avg_time_per_event_ms < 5.0, f"Average processing time {avg_time_per_event_ms}ms exceeds 5ms requirement"
        
        print(f"\n✅ Performance Benchmark Results:")
        print(f"   Throughput: {events_per_second:.1f} events/second (target: >200)")
        print(f"   Avg processing time: {avg_time_per_event_ms:.2f}ms (target: <5ms)")
        print(f"   Total events processed: {len(events)}")
        print(f"   Total time: {total_time_seconds:.3f}s")


if __name__ == "__main__":
    # Run performance benchmarks when called directly
    pytest.main([__file__, "-v", "-s"])