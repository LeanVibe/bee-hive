"""
Tests for Redis message broker functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock

from app.core.redis import AgentMessageBroker, SessionCache, RedisStreamMessage


@pytest.mark.unit
@pytest.mark.asyncio
async def test_message_broker_send_message(mock_redis):
    """Test sending messages via message broker."""
    
    broker = AgentMessageBroker(mock_redis)
    
    stream_id = await broker.send_message(
        from_agent="agent1",
        to_agent="agent2", 
        message_type="task_assignment",
        payload={"task_id": "123", "description": "Test task"}
    )
    
    # Verify Redis calls
    mock_redis.xadd.assert_called_once()
    mock_redis.publish.assert_called_once()
    
    assert stream_id == "1234567890-0"  # Mock return value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_message_broker_broadcast(mock_redis):
    """Test broadcasting messages."""
    
    broker = AgentMessageBroker(mock_redis)
    
    stream_id = await broker.broadcast_message(
        from_agent="orchestrator",
        message_type="system_shutdown",
        payload={"reason": "maintenance"}
    )
    
    # Verify broadcast calls
    mock_redis.xadd.assert_called_once()
    mock_redis.publish.assert_called_once()
    
    assert stream_id == "1234567890-0"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_cache_operations(mock_redis):
    """Test session cache operations."""
    
    cache = SessionCache(mock_redis)
    
    # Test setting session state
    result = await cache.set_session_state(
        "session123",
        {"current_task": "task456", "progress": 0.5}
    )
    
    assert result is True
    mock_redis.setex.assert_called_once()
    
    # Test getting session state
    mock_redis.get.return_value = json.dumps({"current_task": "task456"})
    
    state = await cache.get_session_state("session123")
    
    assert state["current_task"] == "task456"
    mock_redis.get.assert_called_once()


@pytest.mark.unit
def test_redis_stream_message():
    """Test RedisStreamMessage parsing."""
    
    fields = {
        "message_id": "test123",
        "from_agent": "agent1",
        "to_agent": "agent2",
        "type": "completion",
        "payload": json.dumps({"result": "success"}),
        "correlation_id": "corr123"
    }
    
    message = RedisStreamMessage("1234567890-0", fields)
    
    assert message.message_id == "test123"
    assert message.from_agent == "agent1"
    assert message.to_agent == "agent2"
    assert message.message_type == "completion"
    assert message.payload["result"] == "success"
    assert message.correlation_id == "corr123"


@pytest.mark.redis
@pytest.mark.asyncio
async def test_redis_health_check():
    """Test Redis health check functionality."""
    
    from app.core.redis import RedisHealthCheck
    
    # This test requires actual Redis connection
    # Skip if Redis not available
    try:
        result = await RedisHealthCheck.check_connection()
        # If Redis is running, should return True
        # If not running, will raise exception and be skipped
        assert isinstance(result, bool)
    except Exception:
        pytest.skip("Redis not available for integration test")