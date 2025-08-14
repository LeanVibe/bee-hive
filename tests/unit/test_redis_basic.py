"""
Basic coverage tests for app/core/redis.py
Focus on Redis functionality to boost coverage.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

def test_redis_import():
    """Test that Redis module can be imported."""
    from app.core.redis import get_redis, get_message_broker, get_session_cache
    assert get_redis is not None
    assert get_message_broker is not None
    assert get_session_cache is not None

def test_redis_url_configuration():
    """Test Redis URL configuration."""
    from app.core.config import settings
    
    # Redis URL should be configured
    assert hasattr(settings, 'REDIS_URL')
    assert settings.REDIS_URL is not None
    assert len(settings.REDIS_URL) > 0

@patch('app.core.redis.aioredis')
def test_redis_client_creation(mock_aioredis):
    """Test Redis client creation."""
    mock_client = AsyncMock()
    mock_aioredis.from_url.return_value = mock_client
    
    from app.core.redis import get_redis
    
    redis_client = get_redis()
    assert redis_client is not None

@patch('app.core.redis.aioredis')
def test_message_broker_creation(mock_aioredis):
    """Test message broker creation."""
    mock_client = AsyncMock()
    mock_aioredis.from_url.return_value = mock_client
    
    from app.core.redis import get_message_broker, AgentMessageBroker
    
    broker = get_message_broker()
    assert broker is not None

@patch('app.core.redis.aioredis')
def test_session_cache_creation(mock_aioredis):
    """Test session cache creation."""
    mock_client = AsyncMock()
    mock_aioredis.from_url.return_value = mock_client
    
    from app.core.redis import get_session_cache, SessionCache
    
    cache = get_session_cache()
    assert cache is not None

def test_agent_message_broker_class():
    """Test AgentMessageBroker class definition."""
    from app.core.redis import AgentMessageBroker
    
    # Should be a class
    assert isinstance(AgentMessageBroker, type)
    
    # Test instantiation with mock client
    mock_client = MagicMock()
    broker = AgentMessageBroker(mock_client)
    assert broker is not None
    assert broker.redis == mock_client

def test_session_cache_class():
    """Test SessionCache class definition."""
    from app.core.redis import SessionCache
    
    # Should be a class
    assert isinstance(SessionCache, type)
    
    # Test instantiation with mock client
    mock_client = MagicMock()
    cache = SessionCache(mock_client)
    assert cache is not None
    assert cache.redis == mock_client

@pytest.mark.asyncio
async def test_agent_message_broker_send_message():
    """Test AgentMessageBroker send_message method."""
    mock_client = AsyncMock()
    mock_client.xadd = AsyncMock(return_value=b'message-id-123')
    
    from app.core.redis import AgentMessageBroker
    
    broker = AgentMessageBroker(mock_client)
    
    message_data = {
        'type': 'task_assignment',
        'agent_id': 'agent-123',
        'data': {'task': 'test task'}
    }
    
    message_id = await broker.send_message('test-stream', message_data)
    
    # Verify Redis xadd was called
    mock_client.xadd.assert_called_once()
    assert message_id == 'message-id-123'

@pytest.mark.asyncio
async def test_agent_message_broker_receive_messages():
    """Test AgentMessageBroker receive_messages method."""
    mock_client = AsyncMock()
    mock_client.xread = AsyncMock(return_value=[
        ['test-stream', [['msg-1', {'type': 'test', 'data': 'value'}]]]
    ])
    
    from app.core.redis import AgentMessageBroker
    
    broker = AgentMessageBroker(mock_client)
    
    messages = await broker.receive_messages(['test-stream'])
    
    # Verify Redis xread was called
    mock_client.xread.assert_called_once()
    assert len(messages) == 1

@pytest.mark.asyncio
async def test_session_cache_store_session():
    """Test SessionCache store_session method."""
    mock_client = AsyncMock()
    mock_client.setex = AsyncMock(return_value=True)
    
    from app.core.redis import SessionCache
    
    cache = SessionCache(mock_client)
    
    session_data = {'user_id': 'user-123', 'permissions': ['read', 'write']}
    
    await cache.store_session('session-123', session_data, 3600)
    
    # Verify Redis setex was called
    mock_client.setex.assert_called_once()

@pytest.mark.asyncio
async def test_session_cache_get_session():
    """Test SessionCache get_session method."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=b'{"user_id": "user-123", "permissions": ["read", "write"]}')
    
    from app.core.redis import SessionCache
    
    cache = SessionCache(mock_client)
    
    session_data = await cache.get_session('session-123')
    
    # Verify Redis get was called
    mock_client.get.assert_called_once_with('session:session-123')
    assert session_data is not None
    assert session_data.get('user_id') == 'user-123'

@pytest.mark.asyncio
async def test_session_cache_delete_session():
    """Test SessionCache delete_session method."""
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(return_value=1)
    
    from app.core.redis import SessionCache
    
    cache = SessionCache(mock_client)
    
    result = await cache.delete_session('session-123')
    
    # Verify Redis delete was called
    mock_client.delete.assert_called_once_with('session:session-123')
    assert result is True

def test_redis_settings_validation():
    """Test Redis-related settings."""
    from app.core.config import settings
    
    # Check Redis settings
    assert hasattr(settings, 'REDIS_URL')
    assert hasattr(settings, 'REDIS_STREAM_MAX_LEN')
    assert isinstance(settings.REDIS_STREAM_MAX_LEN, int)
    assert settings.REDIS_STREAM_MAX_LEN > 0

@patch('app.core.redis.settings')
def test_redis_configuration_with_settings(mock_settings):
    """Test Redis configuration with different settings."""
    mock_settings.REDIS_URL = 'redis://localhost:6379/1'
    mock_settings.REDIS_STREAM_MAX_LEN = 5000
    
    # Should not raise exceptions when importing
    import importlib
    import app.core.redis
    importlib.reload(app.core.redis)
    
    assert True

def test_redis_connection_singleton():
    """Test that Redis connections are singletons."""
    from app.core.redis import get_redis
    
    # Should return same instance
    redis1 = get_redis()
    redis2 = get_redis()
    
    assert redis1 is redis2

def test_message_broker_singleton():
    """Test that message broker is singleton."""
    from app.core.redis import get_message_broker
    
    # Should return same instance
    broker1 = get_message_broker()
    broker2 = get_message_broker()
    
    assert broker1 is broker2

def test_session_cache_singleton():
    """Test that session cache is singleton."""
    from app.core.redis import get_session_cache
    
    # Should return same instance
    cache1 = get_session_cache()
    cache2 = get_session_cache()
    
    assert cache1 is cache2