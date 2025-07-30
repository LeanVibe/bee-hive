"""
Redis configuration and messaging infrastructure for LeanVibe Agent Hive 2.0

Provides Redis Streams for reliable agent communication, pub/sub for real-time events,
and caching for session state. Optimized for multi-agent coordination patterns.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Set, Callable
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
            if isinstance(payload_str, str):
                # Try JSON parsing first
                result = json.loads(payload_str)
                return result if isinstance(result, dict) else {'data': result}
            else:
                return {'data': payload_str}
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as string data
            return {'data': payload_str}
    
    @property
    def correlation_id(self) -> str:
        """Get the correlation ID for request tracking."""
        return self.fields.get('correlation_id', str(uuid.uuid4()))


class AgentMessageBroker:
    """Redis-based message broker for agent communication with enhanced multi-agent coordination."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.consumer_groups: Dict[str, str] = {}
        self.stream_processors: Dict[str, asyncio.Task] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.active_agents: Set[str] = set()
        self.coordination_enabled = True
    
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
        
        # Properly serialize payload to ensure Redis compatibility
        try:
            serialized_payload = self._serialize_for_redis(payload)
        except Exception as e:
            logger.error(f"Failed to serialize payload for Redis: {e}", payload_type=type(payload))
            raise ValueError(f"Invalid payload for Redis serialization: {e}")
        
        message_data = {
            'message_id': message_id,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'type': message_type,
            'payload': serialized_payload,
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to agent-specific stream with retry logic
        stream_name = f"agent_messages:{to_agent}"
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                stream_id = await self.redis.xadd(
                    stream_name,
                    message_data,
                    maxlen=settings.REDIS_STREAM_MAX_LEN
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Redis stream write attempt {attempt + 1} failed, retrying: {e}",
                        stream_name=stream_name,
                        retry_delay=retry_delay
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to add message to Redis stream after {max_retries} attempts: {e}", stream_name=stream_name)
                    raise
        
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
        
        # Properly serialize payload to ensure Redis compatibility
        try:
            serialized_payload = self._serialize_for_redis(payload)
        except Exception as e:
            logger.error(f"Failed to serialize broadcast payload for Redis: {e}", payload_type=type(payload))
            raise ValueError(f"Invalid payload for Redis serialization: {e}")
        
        message_data = {
            'message_id': message_id,
            'from_agent': from_agent,
            'to_agent': 'broadcast',
            'type': message_type,
            'payload': serialized_payload,
            'correlation_id': correlation_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to broadcast stream with retry logic
        max_retries = 3
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                stream_id = await self.redis.xadd(
                    "agent_messages:broadcast",
                    message_data,
                    maxlen=settings.REDIS_STREAM_MAX_LEN
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Redis broadcast write attempt {attempt + 1} failed, retrying: {e}",
                        retry_delay=retry_delay
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to add broadcast message to Redis stream after {max_retries} attempts: {e}")
                    raise
        
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
    
    # Enhanced Multi-Agent Coordination Methods
    
    async def register_agent(self, agent_id: str, capabilities: List[str], role: str) -> bool:
        """Register an agent for multi-agent coordination."""
        try:
            self.active_agents.add(agent_id)
            
            # Store agent metadata
            agent_key = f"agent_metadata:{agent_id}"
            agent_data = {
                'agent_id': agent_id,
                'capabilities': json.dumps(capabilities),
                'role': role,
                'registered_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            await self.redis.hset(agent_key, mapping=agent_data)
            await self.redis.expire(agent_key, 3600)  # 1 hour TTL
            
            # Create agent-specific consumer group
            stream_name = f"agent_messages:{agent_id}"
            group_name = f"group_{agent_id}"
            await self.create_consumer_group(stream_name, group_name, agent_id)
            
            logger.info(f"🤖 Registered agent {agent_id} with role {role}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register agent {agent_id}", error=str(e))
            return False
    
    async def coordinate_workflow_tasks(
        self, 
        workflow_id: str, 
        tasks: List[Dict[str, Any]], 
        agent_assignments: Dict[str, str]
    ) -> bool:
        """Coordinate task distribution for multi-agent workflow execution."""
        try:
            coordination_key = f"workflow_coordination:{workflow_id}"
            
            # Store workflow coordination data with proper serialization
            coordination_data = {
                'workflow_id': workflow_id,
                'tasks': self._serialize_for_redis(tasks),
                'agent_assignments': self._serialize_for_redis(agent_assignments),
                'created_at': datetime.utcnow().isoformat(),
                'status': 'coordinating'
            }
            
            await self.redis.hset(coordination_key, mapping=coordination_data)
            await self.redis.expire(coordination_key, 7200)  # 2 hours TTL
            
            # Send coordination messages to assigned agents
            for task in tasks:
                task_id = task['id']
                assigned_agent = agent_assignments.get(task_id)
                
                if assigned_agent and assigned_agent in self.active_agents:
                    await self.send_message(
                        from_agent="orchestrator",
                        to_agent=assigned_agent,
                        message_type="workflow_task_assignment",
                        payload={
                            'workflow_id': workflow_id,
                            'task': task,
                            'coordination_context': {
                                'total_tasks': len(tasks),
                                'assigned_agents': list(agent_assignments.values())
                            }
                        }
                    )
            
            logger.info(f"🔄 Coordinated workflow {workflow_id} across {len(agent_assignments)} agents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to coordinate workflow tasks", error=str(e))
            return False
    
    async def synchronize_agent_states(self, workflow_id: str, sync_point: str) -> Dict[str, Any]:
        """Synchronize agent states at workflow coordination points."""
        try:
            sync_key = f"workflow_sync:{workflow_id}:{sync_point}"
            
            # Broadcast synchronization request
            await self.broadcast_message(
                from_agent="orchestrator",
                message_type="synchronization_request",
                payload={
                    'workflow_id': workflow_id,
                    'sync_point': sync_point,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Wait for agent responses (with timeout)
            sync_responses = {}
            timeout = 30  # 30 seconds
            start_time = datetime.utcnow()
            
            while (datetime.utcnow() - start_time).total_seconds() < timeout:
                # Check for sync responses
                sync_stream = f"workflow_sync_responses:{workflow_id}"
                messages = await self.redis.xrange(sync_stream, '-', '+', count=100)
                
                for msg_id, fields in messages:
                    if fields.get('sync_point') == sync_point:
                        agent_id = fields.get('agent_id')
                        if agent_id:
                            sync_responses[agent_id] = {
                                'status': fields.get('status'),
                                'data': self._deserialize_from_redis(fields.get('data', '{}')),
                                'timestamp': fields.get('timestamp')
                            }
                
                # Check if all agents have responded
                coordination_key = f"workflow_coordination:{workflow_id}"
                coordination_data = await self.redis.hgetall(coordination_key)
                if coordination_data:
                    agent_assignments = self._deserialize_from_redis(coordination_data.get('agent_assignments', '{}'))
                    expected_agents = set(agent_assignments.values()) if isinstance(agent_assignments, dict) else set()
                    
                    if set(sync_responses.keys()) >= expected_agents:
                        break
                
                await asyncio.sleep(1)
            
            logger.info(f"🔄 Synchronized {len(sync_responses)} agents at {sync_point}")
            return sync_responses
            
        except Exception as e:
            logger.error(f"❌ Failed to synchronize agent states", error=str(e))
            return {}
    
    async def handle_agent_failure(self, failed_agent_id: str, workflow_id: Optional[str] = None) -> bool:
        """Handle agent failure in multi-agent coordination context."""
        try:
            # Remove from active agents
            self.active_agents.discard(failed_agent_id)
            
            # Update agent status
            agent_key = f"agent_metadata:{failed_agent_id}"
            await self.redis.hset(agent_key, 'status', 'failed')
            
            # If part of workflow, initiate recovery
            if workflow_id:
                recovery_key = f"workflow_recovery:{workflow_id}"
                recovery_data = {
                    'failed_agent': failed_agent_id,
                    'failure_time': datetime.utcnow().isoformat(),
                    'recovery_status': 'initiated'
                }
                
                await self.redis.hset(recovery_key, mapping=recovery_data)
                
                # Broadcast failure notification
                await self.broadcast_message(
                    from_agent="orchestrator",
                    message_type="agent_failure_notification",
                    payload={
                        'failed_agent_id': failed_agent_id,
                        'workflow_id': workflow_id,
                        'recovery_initiated': True
                    }
                )
            
            logger.warning(f"🚨 Handling failure of agent {failed_agent_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to handle agent failure", error=str(e))
            return False
    
    def _serialize_for_redis(self, data: Any) -> str:
        """
        Serialize data for Redis storage with proper type handling.
        
        Handles complex Python data types that need to be serialized as strings
        for Redis compatibility.
        
        Args:
            data: Data to serialize (dict, list, str, int, float, bool, None)
            
        Returns:
            String representation suitable for Redis storage
            
        Raises:
            ValueError: If data contains unsupported types
        """
        try:
            # Handle None
            if data is None:
                return ""
            
            # Handle simple types that Redis accepts directly
            if isinstance(data, (str, int, float, bool)):
                return str(data)
            
            # Handle complex types that need JSON serialization
            if isinstance(data, (dict, list, tuple)):
                return json.dumps(data, default=self._json_serializer, separators=(',', ':'))
            
            # For other types, try to convert to string
            return str(data)
            
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization error: {e}", data_type=type(data))
            raise ValueError(f"Cannot serialize data of type {type(data)}: {e}")
    
    def _json_serializer(self, obj: Any) -> str:
        """
        Custom JSON serializer for complex Python objects.
        
        Handles datetime, UUID, and other common objects that need
        special serialization for Redis storage.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            # Handle objects with dictionaries (like dataclasses)
            return obj.__dict__
        elif hasattr(obj, '_asdict'):
            # Handle namedtuples
            return obj._asdict()
        else:
            # Fallback to string representation
            return str(obj)
    
    def _deserialize_from_redis(self, data: str) -> Any:
        """
        Deserialize data retrieved from Redis.
        
        Attempts to parse JSON first, falls back to string if parsing fails.
        
        Args:
            data: String data from Redis
            
        Returns:
            Deserialized Python object
        """
        try:
            if not data:
                return None
            
            # Try JSON deserialization first
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, return as string
            return data


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


class RedisClient:
    """Redis client wrapper for caching and data operations."""
    
    def __init__(self, redis_instance: Optional[Redis] = None):
        """Initialize Redis client."""
        self._redis = redis_instance or get_redis()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            return await self._redis.get(key)
        except Exception as e:
            logger.error("Redis GET failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration."""
        try:
            await self._redis.set(key, value, ex=expire)
            return True
        except Exception as e:
            logger.error("Redis SET failed", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error("Redis DELETE failed", key=key, error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(await self._redis.exists(key))
        except Exception as e:
            logger.error("Redis EXISTS failed", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        try:
            await self._redis.expire(key, seconds)
            return True
        except Exception as e:
            logger.error("Redis EXPIRE failed", key=key, error=str(e))
            return False
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field value."""
        try:
            return await self._redis.hget(name, key)
        except Exception as e:
            logger.error("Redis HGET failed", name=name, key=key, error=str(e))
            return None
    
    async def hset(self, name: str, key: str, value: str) -> bool:
        """Set hash field value."""
        try:
            await self._redis.hset(name, key, value)
            return True
        except Exception as e:
            logger.error("Redis HSET failed", name=name, key=key, error=str(e))
            return False
    
    async def hdel(self, name: str, key: str) -> bool:
        """Delete hash field."""
        try:
            await self._redis.hdel(name, key)
            return True
        except Exception as e:
            logger.error("Redis HDEL failed", name=name, key=key, error=str(e))
            return False
    
    async def close(self):
        """Close Redis connection."""
        try:
            await self._redis.close()
        except Exception as e:
            logger.error("Redis close failed", error=str(e))


def get_redis_client() -> RedisClient:
    """Get Redis client instance."""
    return RedisClient()