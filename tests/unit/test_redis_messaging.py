"""
Redis Messaging and Communication Testing for LeanVibe Agent Hive 2.0

Tests Redis connection management, message broker functionality, stream processing,
and caching to increase coverage from 21% to 40%.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError


class TestRedisStreamMessage:
    """Test RedisStreamMessage functionality."""
    
    def test_redis_stream_message_creation(self):
        """Test basic RedisStreamMessage creation."""
        from app.core.redis import RedisStreamMessage
        
        stream_id = "1234567890-0"
        fields = {
            "message_id": "msg-123",
            "from_agent": "agent-1",
            "to_agent": "agent-2",
            "type": "task_request",
            "payload": '{"task": "test"}',
            "correlation_id": "corr-123"
        }
        
        message = RedisStreamMessage(stream_id, fields)
        
        assert message.id == stream_id
        assert message.fields == fields
        assert message.message_id == "msg-123"
        assert message.from_agent == "agent-1"
        assert message.to_agent == "agent-2"
        assert message.message_type == "task_request"
        assert message.correlation_id == "corr-123"
    
    def test_redis_stream_message_timestamp_parsing(self):
        """Test timestamp parsing from stream ID."""
        from app.core.redis import RedisStreamMessage
        
        # Use timestamp that corresponds to a known date
        timestamp_ms = 1640995200000  # 2022-01-01 00:00:00 UTC
        stream_id = f"{timestamp_ms}-0"
        fields = {"test": "data"}
        
        message = RedisStreamMessage(stream_id, fields)
        
        assert message.timestamp.year == 2022
        assert message.timestamp.month == 1
        assert message.timestamp.day == 1
    
    def test_redis_stream_message_payload_json_parsing(self):
        """Test payload JSON parsing."""
        from app.core.redis import RedisStreamMessage
        
        # Test valid JSON payload
        fields = {"payload": '{"task": "test", "priority": "high"}'}
        message = RedisStreamMessage("123-0", fields)
        
        payload = message.payload
        assert isinstance(payload, dict)
        assert payload["task"] == "test"
        assert payload["priority"] == "high"
    
    def test_redis_stream_message_payload_string_fallback(self):
        """Test payload parsing with invalid JSON."""
        from app.core.redis import RedisStreamMessage
        
        # Test invalid JSON payload
        fields = {"payload": "not valid json"}
        message = RedisStreamMessage("123-0", fields)
        
        payload = message.payload
        assert isinstance(payload, dict)
        assert payload["data"] == "not valid json"
    
    def test_redis_stream_message_payload_non_string(self):
        """Test payload with non-string data."""
        from app.core.redis import RedisStreamMessage
        
        # Test non-string payload
        fields = {"payload": 12345}
        message = RedisStreamMessage("123-0", fields)
        
        payload = message.payload
        assert isinstance(payload, dict)
        assert payload["data"] == 12345
    
    def test_redis_stream_message_default_values(self):
        """Test default values for missing fields."""
        from app.core.redis import RedisStreamMessage
        
        fields = {}  # Empty fields
        message = RedisStreamMessage("123-0", fields)
        
        assert message.from_agent == "unknown"
        assert message.to_agent == "broadcast"
        assert message.message_type == "unknown"
        assert isinstance(message.correlation_id, str)
        assert len(message.correlation_id) > 0  # Should generate UUID
        assert isinstance(message.message_id, str)
        assert len(message.message_id) > 0  # Should generate UUID


class TestAgentMessageBroker:
    """Test AgentMessageBroker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock(spec=redis.Redis)
        
    def test_agent_message_broker_creation(self):
        """Test AgentMessageBroker creation."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        assert broker.redis == self.mock_redis
        assert isinstance(broker.consumer_groups, dict)
        assert isinstance(broker.stream_processors, dict)
        assert isinstance(broker.message_handlers, dict)
        assert isinstance(broker.active_agents, set)
        assert broker.coordination_enabled is True
    
    async def test_send_message_basic(self):
        """Test basic message sending functionality."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock Redis XADD response
        self.mock_redis.xadd.return_value = "1234567890-0"
        self.mock_redis.publish.return_value = 1
        
        stream_id = await broker.send_message(
            from_agent="agent-1",
            to_agent="agent-2",
            message_type="task_request",
            payload={"task": "test"},
            correlation_id="corr-123"
        )
        
        assert stream_id == "1234567890-0"
        
        # Verify Redis calls
        self.mock_redis.xadd.assert_called_once()
        self.mock_redis.publish.assert_called_once()
        
        # Check xadd call arguments
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        message_data = call_args[0][1]
        
        assert stream_name == "agent_messages:agent-2"
        assert message_data["from_agent"] == "agent-1"
        assert message_data["to_agent"] == "agent-2"
        assert message_data["type"] == "task_request"
        assert message_data["correlation_id"] == "corr-123"
    
    async def test_send_message_with_retry(self):
        """Test message sending with retry logic."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock Redis to fail twice, then succeed
        self.mock_redis.xadd.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            "1234567890-0"
        ]
        self.mock_redis.publish.return_value = 1
        
        stream_id = await broker.send_message(
            from_agent="agent-1",
            to_agent="agent-2",
            message_type="task_request",
            payload={"task": "test"}
        )
        
        assert stream_id == "1234567890-0"
        assert self.mock_redis.xadd.call_count == 3  # 2 failures + 1 success
    
    async def test_send_message_max_retries_exceeded(self):
        """Test message sending when max retries are exceeded."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock Redis to always fail
        self.mock_redis.xadd.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await broker.send_message(
                from_agent="agent-1",
                to_agent="agent-2",
                message_type="task_request",
                payload={"task": "test"}
            )
        
        assert self.mock_redis.xadd.call_count == 3  # Max retries
    
    async def test_broadcast_message(self):
        """Test broadcast message functionality."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        self.mock_redis.xadd.return_value = "1234567890-0"
        self.mock_redis.publish.return_value = 1
        
        stream_id = await broker.broadcast_message(
            from_agent="orchestrator",
            message_type="system_update",
            payload={"update": "system restart"}
        )
        
        assert stream_id == "1234567890-0"
        
        # Check broadcast stream
        call_args = self.mock_redis.xadd.call_args
        stream_name = call_args[0][0]
        message_data = call_args[0][1]
        
        assert stream_name == "agent_messages:broadcast"
        assert message_data["to_agent"] == "broadcast"
        assert message_data["type"] == "system_update"
    
    async def test_create_consumer_group_new(self):
        """Test creating a new consumer group."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock successful group creation
        self.mock_redis.xgroup_create.return_value = True
        
        result = await broker.create_consumer_group(
            "test_stream", "test_group", "test_consumer"
        )
        
        assert result is True
        assert broker.consumer_groups["test_stream"] == "test_group"
        
        self.mock_redis.xgroup_create.assert_called_once_with(
            "test_stream", "test_group", id='0', mkstream=True
        )
    
    async def test_create_consumer_group_exists(self):
        """Test creating a consumer group that already exists."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock group already exists error
        self.mock_redis.xgroup_create.side_effect = Exception("BUSYGROUP Consumer Group name already exists")
        
        result = await broker.create_consumer_group(
            "test_stream", "test_group", "test_consumer"
        )
        
        assert result is True  # Should handle existing group gracefully
        assert broker.consumer_groups["test_stream"] == "test_group"
    
    async def test_read_messages(self):
        """Test reading messages from a stream."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock consumer group creation and message reading
        self.mock_redis.xgroup_create.return_value = True
        
        # Mock stream messages
        mock_messages = [
            ("test_stream", [
                (b"1234567890-0", {
                    b"message_id": b"msg-123",
                    b"from_agent": b"agent-1",
                    b"to_agent": b"agent-2",
                    b"type": b"task_request",
                    b"payload": b'{"task": "test"}'
                })
            ])
        ]
        self.mock_redis.xreadgroup.return_value = mock_messages
        
        messages = await broker.read_messages("agent-2", "consumer-1")
        
        assert len(messages) == 1
        message = messages[0]
        assert message.id == "1234567890-0"
        assert message.from_agent == "agent-1"
        assert message.to_agent == "agent-2"
        assert message.message_type == "task_request"
    
    async def test_read_messages_error_handling(self):
        """Test error handling in message reading."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock error in reading messages
        self.mock_redis.xgroup_create.return_value = True
        self.mock_redis.xreadgroup.side_effect = Exception("Connection error")
        
        messages = await broker.read_messages("agent-2", "consumer-1")
        
        assert messages == []  # Should return empty list on error
    
    async def test_acknowledge_message(self):
        """Test message acknowledgment."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        self.mock_redis.xack.return_value = 1
        
        result = await broker.acknowledge_message("agent-2", "1234567890-0")
        
        assert result is True
        self.mock_redis.xack.assert_called_once_with(
            "agent_messages:agent-2", "group_agent-2", "1234567890-0"
        )
    
    async def test_register_agent(self):
        """Test agent registration for multi-agent coordination."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Mock Redis operations
        self.mock_redis.hset.return_value = True
        self.mock_redis.expire.return_value = True
        self.mock_redis.xgroup_create.return_value = True
        
        result = await broker.register_agent(
            "agent-1", ["coding", "testing"], "developer"
        )
        
        assert result is True
        assert "agent-1" in broker.active_agents
        
        # Verify Redis calls
        self.mock_redis.hset.assert_called()
        self.mock_redis.expire.assert_called()
    
    async def test_coordinate_workflow_tasks(self):
        """Test workflow task coordination."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        broker.active_agents = {"agent-1", "agent-2"}
        
        # Mock Redis operations
        self.mock_redis.hset.return_value = True
        self.mock_redis.expire.return_value = True
        self.mock_redis.xadd.return_value = "1234567890-0"
        self.mock_redis.publish.return_value = 1
        
        tasks = [
            {"id": "task-1", "type": "coding", "description": "Write tests"},
            {"id": "task-2", "type": "review", "description": "Code review"}
        ]
        agent_assignments = {"task-1": "agent-1", "task-2": "agent-2"}
        
        result = await broker.coordinate_workflow_tasks(
            "workflow-123", tasks, agent_assignments
        )
        
        assert result is True
        
        # Should create coordination data
        self.mock_redis.hset.assert_called()
        self.mock_redis.expire.assert_called()
        
        # Should send messages to assigned agents
        assert self.mock_redis.xadd.call_count == 2  # One message per task
    
    async def test_handle_agent_failure(self):
        """Test agent failure handling."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        broker.active_agents = {"agent-1", "agent-2"}
        
        # Mock Redis operations
        self.mock_redis.hset.return_value = True
        self.mock_redis.xadd.return_value = "1234567890-0"
        self.mock_redis.publish.return_value = 1
        
        result = await broker.handle_agent_failure("agent-1", "workflow-123")
        
        assert result is True
        assert "agent-1" not in broker.active_agents
        
        # Should update agent status and broadcast notification
        self.mock_redis.hset.assert_called()
        self.mock_redis.xadd.assert_called()  # For broadcast
    
    def test_serialize_for_redis_simple_types(self):
        """Test Redis serialization for simple types."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Test simple types
        assert broker._serialize_for_redis(None) == ""
        assert broker._serialize_for_redis("hello") == "hello"
        assert broker._serialize_for_redis(123) == "123"
        assert broker._serialize_for_redis(3.14) == "3.14"
        assert broker._serialize_for_redis(True) == "True"
    
    def test_serialize_for_redis_complex_types(self):
        """Test Redis serialization for complex types."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Test complex types
        test_dict = {"key": "value", "number": 42}
        result = broker._serialize_for_redis(test_dict)
        assert isinstance(result, str)
        assert json.loads(result) == test_dict
        
        test_list = [1, 2, 3, "hello"]
        result = broker._serialize_for_redis(test_list)
        assert isinstance(result, str)
        assert json.loads(result) == test_list
    
    def test_serialize_for_redis_datetime(self):
        """Test Redis serialization for datetime objects."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        test_datetime = datetime(2023, 1, 1, 12, 0, 0)
        test_data = {"timestamp": test_datetime}
        
        result = broker._serialize_for_redis(test_data)
        parsed = json.loads(result)
        
        # Datetime should be serialized as ISO string
        assert "2023-01-01T12:00:00" in parsed["timestamp"]
    
    def test_deserialize_from_redis(self):
        """Test Redis deserialization."""
        from app.core.redis import AgentMessageBroker
        
        broker = AgentMessageBroker(self.mock_redis)
        
        # Test empty data
        assert broker._deserialize_from_redis("") is None
        
        # Test JSON data
        test_data = {"key": "value", "number": 42}
        json_str = json.dumps(test_data)
        result = broker._deserialize_from_redis(json_str)
        assert result == test_data
        
        # Test non-JSON data
        result = broker._deserialize_from_redis("plain string")
        assert result == "plain string"


class TestSessionCache:
    """Test SessionCache functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock(spec=redis.Redis)
    
    def test_session_cache_creation(self):
        """Test SessionCache creation."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        assert cache.redis == self.mock_redis
        assert cache.default_ttl == 3600
    
    async def test_set_session_state(self):
        """Test setting session state."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        self.mock_redis.setex.return_value = True
        
        state = {"user_id": "user-123", "preferences": {"theme": "dark"}}
        result = await cache.set_session_state("session-123", state)
        
        assert result is True
        self.mock_redis.setex.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_redis.setex.call_args
        assert call_args[0][0] == "session_state:session-123"
        assert call_args[0][1] == 3600  # Default TTL
    
    async def test_set_session_state_custom_ttl(self):
        """Test setting session state with custom TTL."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        self.mock_redis.setex.return_value = True
        
        state = {"user_id": "user-123"}
        result = await cache.set_session_state("session-123", state, ttl=1800)
        
        assert result is True
        
        call_args = self.mock_redis.setex.call_args
        assert call_args[0][1] == 1800  # Custom TTL
    
    async def test_get_session_state(self):
        """Test getting session state."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        # Mock Redis response
        state = {"user_id": "user-123", "preferences": {"theme": "dark"}}
        self.mock_redis.get.return_value = json.dumps(state)
        
        result = await cache.get_session_state("session-123")
        
        assert result == state
        self.mock_redis.get.assert_called_once_with("session_state:session-123")
    
    async def test_get_session_state_not_found(self):
        """Test getting non-existent session state."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        self.mock_redis.get.return_value = None
        
        result = await cache.get_session_state("session-123")
        
        assert result is None
    
    async def test_delete_session_state(self):
        """Test deleting session state."""
        from app.core.redis import SessionCache
        
        cache = SessionCache(self.mock_redis)
        
        self.mock_redis.delete.return_value = 1
        
        result = await cache.delete_session_state("session-123")
        
        assert result is True
        self.mock_redis.delete.assert_called_once_with("session_state:session-123")


class TestRedisConnectionFunctions:
    """Test Redis connection and initialization functions."""
    
    @patch('app.core.redis.redis.ConnectionPool.from_url')
    async def test_create_redis_pool(self, mock_pool_from_url):
        """Test Redis connection pool creation."""
        from app.core.redis import create_redis_pool
        
        mock_pool = Mock()
        mock_pool_from_url.return_value = mock_pool
        
        with patch('app.core.redis.settings') as mock_settings:
            mock_settings.REDIS_URL = "redis://localhost:6379/0"
            
            pool = await create_redis_pool()
            
            assert pool == mock_pool
            mock_pool_from_url.assert_called_once()
    
    @patch('app.core.redis.create_redis_pool')
    @patch('app.core.redis.redis.Redis')
    async def test_init_redis(self, mock_redis_class, mock_create_pool):
        """Test Redis initialization."""
        from app.core.redis import init_redis
        
        # Mock pool and client
        mock_pool = Mock()
        mock_client = AsyncMock()
        mock_create_pool.return_value = mock_pool
        mock_redis_class.return_value = mock_client
        mock_client.ping.return_value = True
        
        await init_redis()
        
        # Should create pool and client, then test connection
        mock_create_pool.assert_called_once()
        mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
        mock_client.ping.assert_called_once()
    
    async def test_get_redis_not_initialized(self):
        """Test getting Redis client when not initialized."""
        from app.core.redis import get_redis
        
        # Reset global state
        with patch('app.core.redis._redis_client', None):
            with pytest.raises(RuntimeError, match="Redis not initialized"):
                get_redis()
    
    def test_get_message_broker(self):
        """Test getting message broker instance."""
        from app.core.redis import get_message_broker, AgentMessageBroker
        
        with patch('app.core.redis.get_redis') as mock_get_redis:
            mock_redis = Mock()
            mock_get_redis.return_value = mock_redis
            
            broker = get_message_broker()
            
            assert isinstance(broker, AgentMessageBroker)
            assert broker.redis == mock_redis
    
    def test_get_session_cache(self):
        """Test getting session cache instance."""
        from app.core.redis import get_session_cache, SessionCache
        
        with patch('app.core.redis.get_redis') as mock_get_redis:
            mock_redis = Mock()
            mock_get_redis.return_value = mock_redis
            
            cache = get_session_cache()
            
            assert isinstance(cache, SessionCache)
            assert cache.redis == mock_redis


class TestRedisHealthCheck:
    """Test Redis health check utilities."""
    
    @patch('app.core.redis.get_redis')
    async def test_check_connection_healthy(self, mock_get_redis):
        """Test healthy Redis connection check."""
        from app.core.redis import RedisHealthCheck
        
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_get_redis.return_value = mock_client
        
        result = await RedisHealthCheck.check_connection()
        
        assert result is True
        mock_client.ping.assert_called_once()
    
    @patch('app.core.redis.get_redis')
    async def test_check_connection_unhealthy(self, mock_get_redis):
        """Test unhealthy Redis connection check."""
        from app.core.redis import RedisHealthCheck
        
        mock_client = AsyncMock()
        mock_client.ping.side_effect = ConnectionError("Connection failed")
        mock_get_redis.return_value = mock_client
        
        result = await RedisHealthCheck.check_connection()
        
        assert result is False
    
    @patch('app.core.redis.get_redis')
    async def test_get_info(self, mock_get_redis):
        """Test getting Redis server information."""
        from app.core.redis import RedisHealthCheck
        
        mock_client = AsyncMock()
        mock_info = {
            "redis_version": "6.2.0",
            "used_memory_human": "1.5M",
            "connected_clients": 10,
            "uptime_in_seconds": 3600
        }
        mock_client.info.return_value = mock_info
        mock_get_redis.return_value = mock_client
        
        result = await RedisHealthCheck.get_info()
        
        assert result["version"] == "6.2.0"
        assert result["memory_used"] == "1.5M"
        assert result["connected_clients"] == 10
        assert result["uptime"] == 3600


class TestRedisClient:
    """Test RedisClient wrapper functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_redis = AsyncMock(spec=redis.Redis)
    
    def test_redis_client_creation(self):
        """Test RedisClient creation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        assert client._redis == self.mock_redis
    
    def test_redis_client_creation_with_default(self):
        """Test RedisClient creation with default Redis instance."""
        from app.core.redis import RedisClient
        
        with patch('app.core.redis.get_redis') as mock_get_redis:
            mock_redis = Mock()
            mock_get_redis.return_value = mock_redis
            
            client = RedisClient()
            assert client._redis == mock_redis
    
    async def test_redis_client_get(self):
        """Test Redis GET operation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        self.mock_redis.get.return_value = "test_value"
        
        result = await client.get("test_key")
        
        assert result == "test_value"
        self.mock_redis.get.assert_called_once_with("test_key")
    
    async def test_redis_client_set(self):
        """Test Redis SET operation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        self.mock_redis.set.return_value = True
        
        result = await client.set("test_key", "test_value", expire=3600)
        
        assert result is True
        self.mock_redis.set.assert_called_once_with("test_key", "test_value", ex=3600)
    
    async def test_redis_client_delete(self):
        """Test Redis DELETE operation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        self.mock_redis.delete.return_value = 1
        
        result = await client.delete("test_key")
        
        assert result is True
        self.mock_redis.delete.assert_called_once_with("test_key")
    
    async def test_redis_client_exists(self):
        """Test Redis EXISTS operation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        self.mock_redis.exists.return_value = 1
        
        result = await client.exists("test_key")
        
        assert result is True
        self.mock_redis.exists.assert_called_once_with("test_key")
    
    async def test_redis_client_hash_operations(self):
        """Test Redis hash operations."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        # Test HGET
        self.mock_redis.hget.return_value = "field_value"
        result = await client.hget("hash_key", "field")
        assert result == "field_value"
        
        # Test HSET
        self.mock_redis.hset.return_value = 1
        result = await client.hset("hash_key", "field", "value")
        assert result is True
        
        # Test HDEL
        self.mock_redis.hdel.return_value = 1
        result = await client.hdel("hash_key", "field")
        assert result is True
    
    async def test_redis_client_error_handling(self):
        """Test Redis client error handling."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        # Test GET with error
        self.mock_redis.get.side_effect = ConnectionError("Connection failed")
        result = await client.get("test_key")
        assert result is None
        
        # Test SET with error
        self.mock_redis.set.side_effect = ConnectionError("Connection failed")
        result = await client.set("test_key", "value")
        assert result is False
    
    async def test_redis_client_close(self):
        """Test Redis client close operation."""
        from app.core.redis import RedisClient
        
        client = RedisClient(self.mock_redis)
        
        await client.close()
        
        self.mock_redis.close.assert_called_once()
    
    def test_get_redis_client_function(self):
        """Test get_redis_client function."""
        from app.core.redis import get_redis_client, RedisClient
        
        client = get_redis_client()
        assert isinstance(client, RedisClient)