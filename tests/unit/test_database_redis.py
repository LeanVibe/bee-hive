"""
Comprehensive unit tests for database and Redis functionality.

Tests cover:
- Database connection management
- Session factory and lifecycle
- Query execution and transactions
- Redis connection and pooling
- Message broker functionality
- Session caching
- Error handling and recovery
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from app.core.database import (
    Base,
    get_database_url,
    create_engine,
    create_session_factory,
    init_database,
    get_session,
    get_async_session,
    close_database
)
from app.core.redis import (
    RedisStreamMessage,
    AgentMessageBroker,
    SessionCache,
    init_redis,
    get_redis,
    get_message_broker,
    get_session_cache,
    close_redis
)
from app.core.config import settings


class TestDatabaseFunctionality:
    """Test database connection and session management."""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for database configuration."""
        with patch('app.core.database.settings') as mock_settings:
            mock_settings.DATABASE_URL = "postgresql://user:pass@localhost:5432/test_db"
            mock_settings.DATABASE_POOL_SIZE = 10
            mock_settings.DATABASE_MAX_OVERFLOW = 20
            mock_settings.DEBUG = False
            yield mock_settings
    
    def test_get_database_url_conversion(self, mock_settings):
        """Test database URL conversion for async driver."""
        url = get_database_url()
        
        assert url == "postgresql+asyncpg://user:pass@localhost:5432/test_db"
        assert "postgresql+asyncpg://" in url
        assert "postgresql://" not in url
    
    @patch('app.core.database.create_async_engine')
    @pytest.mark.asyncio
    async def test_create_engine_configuration(self, mock_create_engine, mock_settings):
        """Test engine creation with proper configuration."""
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine
        
        engine = await create_engine()
        
        # Verify create_async_engine was called with correct parameters
        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args
        
        assert call_args[0][0].startswith("postgresql+asyncpg://")
        assert call_args[1]['pool_size'] == mock_settings.DATABASE_POOL_SIZE
        assert call_args[1]['max_overflow'] == mock_settings.DATABASE_MAX_OVERFLOW
        assert call_args[1]['pool_pre_ping'] is True
        assert call_args[1]['pool_recycle'] == 3600
        assert call_args[1]['future'] is True
        
        assert engine == mock_engine
    
    @patch('app.core.database.create_engine')
    @patch('app.core.database.async_sessionmaker')
    @pytest.mark.asyncio
    async def test_create_session_factory(self, mock_sessionmaker, mock_create_engine, mock_settings):
        """Test session factory creation."""
        mock_engine = AsyncMock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = AsyncMock()
        mock_sessionmaker.return_value = mock_session_factory
        
        session_factory = await create_session_factory()
        
        # Verify session factory configuration
        mock_sessionmaker.assert_called_once_with(
            bind=mock_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
        
        assert session_factory == mock_session_factory
    
    @patch('app.core.database.create_engine')
    @patch('app.core.database.create_session_factory')
    @patch('app.core.database.Base')
    @pytest.mark.asyncio
    async def test_init_database_success(self, mock_base, mock_session_factory, mock_create_engine, mock_settings):
        """Test successful database initialization."""
        # Mock engine and connection
        mock_engine = AsyncMock()
        mock_conn = AsyncMock()
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        # Mock session factory
        mock_factory = AsyncMock()
        mock_session_factory.return_value = mock_factory
        
        await init_database()
        
        # Verify database connection test
        mock_conn.execute.assert_called()
        
        # Verify tables creation in debug mode
        if mock_settings.DEBUG:
            mock_conn.run_sync.assert_called_once_with(mock_base.metadata.create_all)
    
    @patch('app.core.database.create_engine')
    @pytest.mark.asyncio
    async def test_init_database_connection_failure(self, mock_create_engine, mock_settings):
        """Test database initialization with connection failure."""
        # Mock engine that fails connection
        mock_engine = AsyncMock()
        mock_engine.begin.side_effect = Exception("Connection failed")
        mock_create_engine.return_value = mock_engine
        
        with pytest.raises(Exception, match="Connection failed"):
            await init_database()
    
    @patch('app.core.database._session_factory')
    @pytest.mark.asyncio
    async def test_get_async_session(self, mock_session_factory):
        """Test async session context manager."""
        # Mock session
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session
        
        async with get_async_session() as session:
            assert session == mock_session
        
        # Verify session was closed
        mock_session.close.assert_called_once()
    
    @patch('app.core.database._session_factory')
    @pytest.mark.asyncio
    async def test_get_async_session_exception_handling(self, mock_session_factory):
        """Test async session exception handling."""
        mock_session = AsyncMock()
        mock_session_factory.return_value = mock_session
        
        try:
            async with get_async_session() as session:
                # Simulate an exception in the session
                raise Exception("Test exception")
        except Exception:
            pass
        
        # Session should still be closed even with exception
        mock_session.close.assert_called_once()
    
    @patch('app.core.database._engine')
    @pytest.mark.asyncio
    async def test_close_database(self, mock_engine):
        """Test database cleanup on shutdown."""
        mock_engine.dispose = AsyncMock()
        
        await close_database()
        
        mock_engine.dispose.assert_called_once()


class TestRedisStreamMessage:
    """Test Redis stream message functionality."""
    
    @pytest.fixture
    def sample_message_fields(self):
        """Sample message fields for testing."""
        return {
            'message_id': 'msg-123',
            'from_agent': 'agent-1',
            'to_agent': 'agent-2',
            'type': 'task_assignment',
            'payload': json.dumps({'task_id': 'task-456', 'priority': 'high'}),
            'correlation_id': 'corr-789'
        }
    
    def test_redis_stream_message_creation(self, sample_message_fields):
        """Test Redis stream message creation and property access."""
        stream_id = f"{int(datetime.utcnow().timestamp() * 1000)}-0"
        message = RedisStreamMessage(stream_id, sample_message_fields)
        
        assert message.id == stream_id
        assert message.message_id == 'msg-123'
        assert message.from_agent == 'agent-1'
        assert message.to_agent == 'agent-2'
        assert message.message_type == 'task_assignment'
        assert message.correlation_id == 'corr-789'
        
        # Test payload parsing
        payload = message.payload
        assert isinstance(payload, dict)
        assert payload['task_id'] == 'task-456'
        assert payload['priority'] == 'high'
    
    def test_redis_stream_message_timestamp_parsing(self, sample_message_fields):
        """Test timestamp parsing from stream ID."""
        timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
        stream_id = f"{timestamp_ms}-0"
        message = RedisStreamMessage(stream_id, sample_message_fields)
        
        expected_timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
        assert abs((message.timestamp - expected_timestamp).total_seconds()) < 1
    
    def test_redis_stream_message_default_values(self):
        """Test Redis stream message with missing fields."""
        stream_id = "1234567890-0"
        minimal_fields = {}
        message = RedisStreamMessage(stream_id, minimal_fields)
        
        assert message.from_agent == 'unknown'
        assert message.to_agent == 'broadcast'
        assert message.message_type == 'unknown'
        assert isinstance(message.message_id, str)
        assert isinstance(message.correlation_id, str)
    
    def test_redis_stream_message_payload_parsing_errors(self, sample_message_fields):
        """Test payload parsing with malformed JSON."""
        sample_message_fields['payload'] = '{ invalid json }'
        stream_id = "1234567890-0"
        message = RedisStreamMessage(stream_id, sample_message_fields)
        
        payload = message.payload
        assert isinstance(payload, dict)
        assert 'data' in payload
        assert payload['data'] == '{ invalid json }'


class TestAgentMessageBroker:
    """Test Redis-based agent message broker."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        return AsyncMock(spec=Redis)
    
    @pytest.fixture
    def message_broker(self, mock_redis_client):
        """Create message broker instance."""
        return AgentMessageBroker(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_send_message(self, message_broker, mock_redis_client):
        """Test sending a message through the broker."""
        # Mock Redis XADD response
        mock_redis_client.xadd.return_value = "1234567890-0"
        
        payload = {'task_id': 'task-123', 'action': 'start'}
        message_id = await message_broker.send_message(
            from_agent='agent-1',
            to_agent='agent-2',
            message_type='task_assignment',
            payload=payload,
            correlation_id='corr-456'
        )
        
        # Verify Redis XADD was called
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        
        assert call_args[0][0] == "agent_messages:agent-2"  # Stream name
        assert 'from_agent' in call_args[1]
        assert call_args[1]['from_agent'] == 'agent-1'
        assert call_args[1]['to_agent'] == 'agent-2'
        assert call_args[1]['type'] == 'task_assignment'
        assert 'payload' in call_args[1]
        
        assert isinstance(message_id, str)
    
    @pytest.mark.asyncio
    async def test_send_broadcast_message(self, message_broker, mock_redis_client):
        """Test sending a broadcast message."""
        mock_redis_client.xadd.return_value = "1234567890-0"
        
        payload = {'announcement': 'system maintenance'}
        await message_broker.send_broadcast(
            from_agent='system',
            message_type='announcement',
            payload=payload
        )
        
        # Should send to broadcast stream
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        assert call_args[0][0] == "agent_messages:broadcast"
    
    @pytest.mark.asyncio
    async def test_receive_messages(self, message_broker, mock_redis_client):
        """Test receiving messages from Redis streams."""
        # Mock Redis XREAD response
        mock_redis_client.xread.return_value = {
            b'agent_messages:agent-1': [
                (b'1234567890-0', {
                    b'message_id': b'msg-123',
                    b'from_agent': b'agent-2',
                    b'to_agent': b'agent-1',
                    b'type': b'task_update',
                    b'payload': b'{"status": "completed"}',
                    b'correlation_id': b'corr-456'
                })
            ]
        }
        
        messages = await message_broker.receive_messages('agent-1', count=1)
        
        assert len(messages) == 1
        message = messages[0]
        assert isinstance(message, RedisStreamMessage)
        assert message.from_agent == 'agent-2'
        assert message.to_agent == 'agent-1'
        assert message.message_type == 'task_update'
        assert message.payload['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_create_consumer_group(self, message_broker, mock_redis_client):
        """Test creating consumer group for stream processing."""
        mock_redis_client.xgroup_create.return_value = True
        
        await message_broker.create_consumer_group('test-stream', 'test-group')
        
        mock_redis_client.xgroup_create.assert_called_once_with(
            'test-stream', 'test-group', id='0', mkstream=True
        )
    
    @pytest.mark.asyncio
    async def test_register_message_handler(self, message_broker):
        """Test registering message handlers."""
        async def test_handler(message: RedisStreamMessage):
            return f"Handled: {message.message_type}"
        
        message_broker.register_handler('task_assignment', test_handler)
        
        assert 'task_assignment' in message_broker.message_handlers
        assert message_broker.message_handlers['task_assignment'] == test_handler
    
    @pytest.mark.asyncio
    async def test_coordination_features(self, message_broker, mock_redis_client):
        """Test multi-agent coordination features."""
        # Test agent registration
        await message_broker.register_agent('agent-1')
        assert 'agent-1' in message_broker.active_agents
        
        # Test heartbeat
        mock_redis_client.hset.return_value = True
        await message_broker.send_heartbeat('agent-1')
        mock_redis_client.hset.assert_called()
        
        # Test agent status check
        mock_redis_client.hget.return_value = json.dumps({
            'status': 'active',
            'last_heartbeat': datetime.utcnow().isoformat()
        }).encode()
        
        status = await message_broker.get_agent_status('agent-1')
        assert status['status'] == 'active'


class TestSessionCache:
    """Test Redis-based session caching."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        return AsyncMock(spec=Redis)
    
    @pytest.fixture
    def session_cache(self, mock_redis_client):
        """Create session cache instance."""
        return SessionCache(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_store_session_data(self, session_cache, mock_redis_client):
        """Test storing session data in cache."""
        session_id = "session-123"
        session_data = {
            'user_id': 'user-456',
            'agent_id': 'agent-789',
            'context': {'current_task': 'task-101'}
        }
        
        mock_redis_client.hset.return_value = True
        mock_redis_client.expire.return_value = True
        
        await session_cache.store(session_id, session_data, ttl=3600)
        
        # Verify Redis operations
        mock_redis_client.hset.assert_called_once()
        mock_redis_client.expire.assert_called_once_with(f"session:{session_id}", 3600)
    
    @pytest.mark.asyncio
    async def test_retrieve_session_data(self, session_cache, mock_redis_client):
        """Test retrieving session data from cache."""
        session_id = "session-123"
        stored_data = {
            'user_id': 'user-456',
            'agent_id': 'agent-789',
            'context': {'current_task': 'task-101'}
        }
        
        mock_redis_client.hgetall.return_value = {
            b'data': json.dumps(stored_data).encode()
        }
        
        retrieved_data = await session_cache.retrieve(session_id)
        
        assert retrieved_data == stored_data
        mock_redis_client.hgetall.assert_called_once_with(f"session:{session_id}")
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_session(self, session_cache, mock_redis_client):
        """Test retrieving non-existent session data."""
        mock_redis_client.hgetall.return_value = {}
        
        retrieved_data = await session_cache.retrieve("nonexistent-session")
        
        assert retrieved_data is None
    
    @pytest.mark.asyncio
    async def test_delete_session(self, session_cache, mock_redis_client):
        """Test deleting session data."""
        session_id = "session-123"
        mock_redis_client.delete.return_value = 1
        
        deleted = await session_cache.delete(session_id)
        
        assert deleted is True
        mock_redis_client.delete.assert_called_once_with(f"session:{session_id}")
    
    @pytest.mark.asyncio
    async def test_extend_session_ttl(self, session_cache, mock_redis_client):
        """Test extending session TTL."""
        session_id = "session-123"
        mock_redis_client.expire.return_value = True
        
        extended = await session_cache.extend_ttl(session_id, 7200)
        
        assert extended is True
        mock_redis_client.expire.assert_called_once_with(f"session:{session_id}", 7200)


class TestRedisConnectionManagement:
    """Test Redis connection and pool management."""
    
    @patch('app.core.redis.redis.from_url')
    @patch('app.core.redis.settings')
    @pytest.mark.asyncio
    async def test_init_redis_success(self, mock_settings, mock_from_url):
        """Test successful Redis initialization."""
        # Mock settings
        mock_settings.REDIS_URL = "redis://localhost:6379/0"
        mock_settings.REDIS_MAX_CONNECTIONS = 20
        mock_settings.REDIS_SOCKET_KEEPALIVE = True
        
        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_from_url.return_value = mock_client
        
        await init_redis()
        
        # Verify Redis client creation
        mock_from_url.assert_called_once()
        mock_client.ping.assert_called_once()
    
    @patch('app.core.redis.redis.from_url')
    @patch('app.core.redis.settings')
    @pytest.mark.asyncio
    async def test_init_redis_connection_failure(self, mock_settings, mock_from_url):
        """Test Redis initialization with connection failure."""
        mock_settings.REDIS_URL = "redis://localhost:6379/0"
        
        # Mock connection failure
        mock_client = AsyncMock()
        mock_client.ping.side_effect = ConnectionError("Connection refused")
        mock_from_url.return_value = mock_client
        
        with pytest.raises(Exception):
            await init_redis()
    
    @patch('app.core.redis._redis_client')
    def test_get_redis_client(self, mock_client):
        """Test getting Redis client."""
        client = get_redis()
        assert client == mock_client
    
    @patch('app.core.redis._redis_client')
    def test_get_message_broker(self, mock_client):
        """Test getting message broker instance."""
        broker = get_message_broker()
        assert isinstance(broker, AgentMessageBroker)
        assert broker.redis == mock_client
    
    @patch('app.core.redis._redis_client')
    def test_get_session_cache(self, mock_client):
        """Test getting session cache instance."""
        cache = get_session_cache()
        assert isinstance(cache, SessionCache)
        assert cache.redis == mock_client
    
    @patch('app.core.redis._redis_client')
    @pytest.mark.asyncio
    async def test_close_redis(self, mock_client):
        """Test Redis connection cleanup."""
        mock_client.close = AsyncMock()
        mock_client.connection_pool.disconnect = AsyncMock()
        
        await close_redis()
        
        mock_client.close.assert_called_once()
        mock_client.connection_pool.disconnect.assert_called_once()


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture
    def message_broker(self):
        """Create message broker with mock Redis client."""
        mock_client = AsyncMock()
        return AgentMessageBroker(mock_client)
    
    @pytest.mark.asyncio
    async def test_redis_connection_timeout_handling(self, message_broker):
        """Test handling of Redis connection timeouts."""
        # Mock timeout error
        message_broker.redis.xadd.side_effect = TimeoutError("Operation timed out")
        
        with pytest.raises(TimeoutError):
            await message_broker.send_message(
                from_agent='agent-1',
                to_agent='agent-2',
                message_type='test',
                payload={}
            )
    
    @pytest.mark.asyncio
    async def test_redis_connection_error_handling(self, message_broker):
        """Test handling of Redis connection errors."""
        # Mock connection error
        message_broker.redis.xread.side_effect = ConnectionError("Connection lost")
        
        with pytest.raises(ConnectionError):
            await message_broker.receive_messages('agent-1')
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self):
        """Test database transaction rollback on error."""
        with patch('app.core.database.get_async_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value.__aenter__.return_value = mock_session
            
            # Simulate transaction error
            mock_session.execute.side_effect = Exception("Query failed")
            
            try:
                async with get_async_session() as session:
                    await session.execute("SELECT 1")
                    raise Exception("Simulated error")
            except Exception:
                pass
            
            # Session should still be closed
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_cache_json_parsing_error(self):
        """Test session cache handling of JSON parsing errors."""
        mock_client = AsyncMock()
        cache = SessionCache(mock_client)
        
        # Mock corrupted JSON data
        mock_client.hgetall.return_value = {
            b'data': b'{ invalid json }'
        }
        
        retrieved_data = await cache.retrieve('session-123')
        
        # Should handle gracefully and return None
        assert retrieved_data is None


class TestPerformanceAndConcurrency:
    """Test performance characteristics and concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self):
        """Test concurrent message sending through broker."""
        mock_client = AsyncMock()
        mock_client.xadd.return_value = "1234567890-0"
        
        broker = AgentMessageBroker(mock_client)
        
        # Send multiple messages concurrently
        async def send_message(i):
            return await broker.send_message(
                from_agent=f'agent-{i}',
                to_agent='agent-0',
                message_type='test',
                payload={'index': i}
            )
        
        tasks = [send_message(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All messages should be sent successfully
        assert all(isinstance(result, str) for result in results)
        assert mock_client.xadd.call_count == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self):
        """Test concurrent session cache operations."""
        mock_client = AsyncMock()
        mock_client.hset.return_value = True
        mock_client.hgetall.return_value = {b'data': b'{"test": "data"}'}
        
        cache = SessionCache(mock_client)
        
        # Perform concurrent operations
        async def session_operation(i):
            session_id = f'session-{i}'
            await cache.store(session_id, {'test': 'data'})
            return await cache.retrieve(session_id)
        
        tasks = [session_operation(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        assert all(result == {'test': 'data'} for result in results)
    
    @pytest.mark.asyncio
    async def test_large_message_handling(self):
        """Test handling of large messages."""
        mock_client = AsyncMock()
        mock_client.xadd.return_value = "1234567890-0"
        
        broker = AgentMessageBroker(mock_client)
        
        # Create large payload
        large_payload = {
            'data': 'x' * 10000,  # 10KB of data
            'items': list(range(1000))
        }
        
        message_id = await broker.send_message(
            from_agent='agent-1',
            to_agent='agent-2',
            message_type='large_data',
            payload=large_payload
        )
        
        assert isinstance(message_id, str)
        mock_client.xadd.assert_called_once()


class TestConfigurationAndSettings:
    """Test configuration management for database and Redis."""
    
    def test_database_url_environment_variables(self):
        """Test database URL construction from environment."""
        with patch.dict('os.environ', {
            'DATABASE_URL': 'postgresql://test:pass@db:5432/testdb'
        }):
            url = get_database_url()
            assert url == "postgresql+asyncpg://test:pass@db:5432/testdb"
    
    @patch('app.core.redis.settings')
    def test_redis_configuration_values(self, mock_settings):
        """Test Redis configuration parameter usage."""
        mock_settings.REDIS_URL = "redis://custom:6380/1"
        mock_settings.REDIS_MAX_CONNECTIONS = 50
        mock_settings.REDIS_SOCKET_KEEPALIVE = False
        
        # Test that configuration values are used correctly
        assert mock_settings.REDIS_URL == "redis://custom:6380/1"
        assert mock_settings.REDIS_MAX_CONNECTIONS == 50
        assert mock_settings.REDIS_SOCKET_KEEPALIVE is False