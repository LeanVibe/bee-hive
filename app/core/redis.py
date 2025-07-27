"""
Redis configuration and messaging infrastructure for LeanVibe Agent Hive 2.0

Provides Redis Streams for reliable agent communication, pub/sub for real-time events,
and caching for session state. Optimized for multi-agent coordination patterns.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from .config import settings

logger = structlog.get_logger()

# Global Redis connection pool
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[Redis] = None


class RedisStreamMessage:
    """Represents a message in a Redis Stream."""
    
    def __init__(self, stream_id: str, fields: Dict[str, Any]):
        self.id = stream_id
        self.timestamp = datetime.fromtimestamp(int(stream_id.split('-')[0]) / 1000)
        self.fields = fields
    
    @property
    def message_id(self) -> str:
        """Get the unique message ID."""
        return self.fields.get('message_id', str(uuid.uuid4()))
    
    @property
    def from_agent(self) -> str:
        """Get the sender agent ID."""
        return self.fields.get('from_agent', 'unknown')
    
    @property
    def to_agent(self) -> str:
        """Get the recipient agent ID."""
        return self.fields.get('to_agent', 'broadcast')
    
    @property
    def message_type(self) -> str:
        """Get the message type."""
        return self.fields.get('type', 'unknown')
    
    @property
    def payload(self) -> Dict[str, Any]:
        """Get the message payload."""
        payload_str = self.fields.get('payload', '{}')
        try:
            return json.loads(payload_str) if isinstance(payload_str, str) else payload_str
        except json.JSONDecodeError:
            return {}
    
    @property
    def correlation_id(self) -> str:
        """Get the correlation ID for request tracking."""
        return self.fields.get('correlation_id', str(uuid.uuid4()))


class AgentMessageBroker:
    """Redis-based message broker for agent communication."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.consumer_groups: Dict[str, str] = {}
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """Send a message from one agent to another."""
        
        message_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        message_data = {
            'message_id': message_id,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'type': message_type,
            'payload': json.dumps(payload),
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to agent-specific stream
        stream_name = f"agent_messages:{to_agent}"
        stream_id = await self.redis.xadd(
            stream_name,
            message_data,
            maxlen=settings.REDIS_STREAM_MAX_LEN
        )
        
        # Also publish to real-time pub/sub for immediate notifications
        await self.redis.publish(f"agent_events:{to_agent}", json.dumps(message_data))
        
        logger.info(
            "📧 Message sent",
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            message_id=message_id,
            stream_id=stream_id
        )
        
        return stream_id
    
    async def broadcast_message(
        self,
        from_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """Broadcast a message to all agents."""
        
        message_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        message_data = {
            'message_id': message_id,
            'from_agent': from_agent,
            'to_agent': 'broadcast',
            'type': message_type,
            'payload': json.dumps(payload),
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to broadcast stream
        stream_id = await self.redis.xadd(
            "agent_messages:broadcast",
            message_data,
            maxlen=settings.REDIS_STREAM_MAX_LEN
        )
        
        # Publish to system events
        await self.redis.publish("system_events", json.dumps(message_data))
        
        logger.info(
            "📢 Broadcast message sent",
            from_agent=from_agent,
            message_type=message_type,
            message_id=message_id
        )
        
        return stream_id
    
    async def create_consumer_group(self, stream_name: str, group_name: str, consumer_name: str) -> bool:
        """Create a consumer group for stream processing."""
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id='0',
                mkstream=True
            )
            self.consumer_groups[stream_name] = group_name
            logger.info(f"✅ Created consumer group {group_name} for stream {stream_name}")
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                self.consumer_groups[stream_name] = group_name
                return True
            logger.error(f"❌ Failed to create consumer group", error=str(e))
            return False
    
    async def read_messages(
        self,
        agent_id: str,
        consumer_name: str,
        count: int = 10,
        block: int = 1000
    ) -> List[RedisStreamMessage]:
        """Read messages for an agent with consumer group semantics."""
        
        stream_name = f"agent_messages:{agent_id}"
        group_name = f"group_{agent_id}"
        
        # Ensure consumer group exists
        await self.create_consumer_group(stream_name, group_name, consumer_name)
        
        try:
            # Read new messages
            messages = await self.redis.xreadgroup(
                group_name,
                consumer_name,
                {stream_name: '>'},
                count=count,
                block=block
            )
            
            parsed_messages = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Convert bytes to strings
                    str_fields = {k.decode(): v.decode() for k, v in fields.items()}
                    parsed_messages.append(RedisStreamMessage(msg_id.decode(), str_fields))
            
            return parsed_messages
            
        except Exception as e:
            logger.error(f"❌ Failed to read messages for agent {agent_id}", error=str(e))
            return []
    
    async def acknowledge_message(
        self,
        agent_id: str,
        message_id: str
    ) -> bool:
        """Acknowledge message processing completion."""
        try:
            stream_name = f"agent_messages:{agent_id}"
            group_name = f"group_{agent_id}"
            
            await self.redis.xack(stream_name, group_name, message_id)
            logger.debug(f"✅ Acknowledged message {message_id} for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to acknowledge message", error=str(e))
            return False


class SessionCache:
    """Redis-based session state caching for agents."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def set_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store session state in Redis."""
        try:
            key = f"session_state:{session_id}"
            ttl = ttl or self.default_ttl
            
            await self.redis.setex(
                key,
                ttl,
                json.dumps(state, default=str)
            )
            
            logger.debug(f"💾 Stored session state for {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to store session state", error=str(e))
            return False
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session state from Redis."""
        try:
            key = f"session_state:{session_id}"
            state_json = await self.redis.get(key)
            
            if state_json:
                return json.loads(state_json)
            return None
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session state", error=str(e))
            return None
    
    async def delete_session_state(self, session_id: str) -> bool:
        """Delete session state from Redis."""
        try:
            key = f"session_state:{session_id}"
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete session state", error=str(e))
            return False


async def create_redis_pool() -> ConnectionPool:
    """Create Redis connection pool."""
    return redis.ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=20,
        retry_on_timeout=True,
        socket_keepalive=True,
        socket_keepalive_options={},
        health_check_interval=30
    )


async def init_redis() -> None:
    """Initialize Redis connection and services."""
    global _redis_pool, _redis_client
    
    try:
        logger.info("🔗 Initializing Redis connection...")
        
        _redis_pool = await create_redis_pool()
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        
        # Test Redis connection
        await _redis_client.ping()
        
        logger.info("✅ Redis connection established")
        
    except Exception as e:
        logger.error("❌ Failed to initialize Redis", error=str(e))
        raise


async def close_redis() -> None:
    """Close Redis connections gracefully."""
    global _redis_client, _redis_pool
    
    if _redis_client:
        await _redis_client.close()
    
    if _redis_pool:
        await _redis_pool.disconnect()
    
    logger.info("🔗 Redis connections closed")


def get_redis() -> Redis:
    """Get Redis client instance."""
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client


def get_message_broker() -> AgentMessageBroker:
    """Get message broker instance."""
    return AgentMessageBroker(get_redis())


def get_session_cache() -> SessionCache:
    """Get session cache instance."""
    return SessionCache(get_redis())


class RedisHealthCheck:
    """Redis health check utilities."""
    
    @staticmethod
    async def check_connection() -> bool:
        """Check if Redis connection is healthy."""
        try:
            redis_client = get_redis()
            await redis_client.ping()
            return True
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False
    
    @staticmethod
    async def get_info() -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            redis_client = get_redis()
            info = await redis_client.info()
            return {
                "version": info.get("redis_version"),
                "memory_used": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime": info.get("uptime_in_seconds")
            }
        except Exception as e:
            logger.error("Failed to get Redis info", error=str(e))
            return {}