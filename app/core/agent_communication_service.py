"""
Agent Communication Service for LeanVibe Agent Hive 2.0

Implements Redis Pub/Sub based communication system for orchestrator-to-agent messaging.
Provides reliable, low-latency message passing with comprehensive monitoring and error handling.

This is Phase 1 of comprehensive communication system implementation, establishing foundation
for Phase 2 Redis Streams upgrade.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from ..models.message import StreamMessage, MessageType, MessagePriority
from .config import settings
from .redis_pubsub_manager import RedisPubSubManager, StreamStats, MessageProcessingResult

logger = logging.getLogger(__name__)


class CommunicationPattern(Enum):
    """Communication patterns supported by the service."""
    POINT_TO_POINT = "point_to_point"
    BROADCAST = "broadcast"
    FILTERED = "filtered"
    ACKNOWLEDGMENT = "acknowledgment"


class DeliveryStatus(Enum):
    """Message delivery status tracking."""
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class MessageMetrics:
    """Metrics for message delivery performance."""
    total_sent: int = 0
    total_delivered: int = 0
    total_acknowledged: int = 0
    total_failed: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    delivery_rate: float = 0.0
    error_rate: float = 0.0
    throughput_msg_per_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ChannelInfo:
    """Information about a Redis channel."""
    name: str
    subscriber_count: int
    message_count: int
    last_activity: Optional[datetime]
    pattern: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class AgentMessage:
    """Enhanced message model for Pub/Sub communication."""
    id: str
    from_agent: str
    to_agent: Optional[str]  # None for broadcast
    type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[int] = None  # TTL in seconds
    correlation_id: Optional[str] = None
    acknowledgment_required: bool = False
    
    @classmethod
    def from_stream_message(cls, stream_msg: StreamMessage) -> "AgentMessage":
        """Create AgentMessage from StreamMessage."""
        return cls(
            id=stream_msg.id,
            from_agent=stream_msg.from_agent,
            to_agent=stream_msg.to_agent,
            type=stream_msg.message_type,
            payload=stream_msg.payload,
            timestamp=stream_msg.timestamp,
            priority=stream_msg.priority,
            ttl=stream_msg.ttl,
            correlation_id=stream_msg.correlation_id,
            acknowledgment_required=False
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "ttl": self.ttl,
            "correlation_id": self.correlation_id,
            "acknowledgment_required": self.acknowledgment_required
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create AgentMessage from dictionary."""
        return cls(
            id=data["id"],
            from_agent=data["from"],
            to_agent=data.get("to"),
            type=MessageType(data["type"]),
            payload=data["payload"],
            timestamp=data["timestamp"],
            priority=MessagePriority(data.get("priority", "normal")),
            ttl=data.get("ttl"),
            correlation_id=data.get("correlation_id"),
            acknowledgment_required=data.get("acknowledgment_required", False)
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if not self.ttl:
            return False
        return time.time() > (self.timestamp + self.ttl)
    
    def get_channel_name(self) -> str:
        """Get Redis channel name for this message."""
        if self.to_agent:
            return f"agent:{self.to_agent}"
        else:
            return "broadcast"


class AgentCommunicationError(Exception):
    """Base exception for agent communication errors."""
    pass


class MessageValidationError(AgentCommunicationError):
    """Error in message validation."""
    pass


class DeliveryError(AgentCommunicationError):
    """Error in message delivery."""
    pass


class ConnectionError(AgentCommunicationError):
    """Error in Redis connection."""
    pass


class MessageValidator:
    """Validates messages according to schema requirements."""
    
    @staticmethod
    def validate_message(message: AgentMessage) -> None:
        """Validate message according to schema requirements."""
        if not message.id:
            raise MessageValidationError("Message ID is required")
        
        if not message.from_agent or len(message.from_agent.strip()) == 0:
            raise MessageValidationError("from_agent is required and cannot be empty")
        
        if len(message.from_agent) > 255:
            raise MessageValidationError("from_agent cannot exceed 255 characters")
        
        if message.to_agent and len(message.to_agent) > 255:
            raise MessageValidationError("to_agent cannot exceed 255 characters")
        
        if not isinstance(message.type, MessageType):
            raise MessageValidationError("message type must be valid MessageType enum")
        
        if not isinstance(message.payload, dict):
            raise MessageValidationError("payload must be a dictionary")
        
        if message.ttl is not None and message.ttl <= 0:
            raise MessageValidationError("TTL must be positive if specified")
        
        if message.timestamp <= 0:
            raise MessageValidationError("timestamp must be positive")
        
        # Check message size (JSON serialized)
        serialized = json.dumps(message.to_dict())
        if len(serialized.encode('utf-8')) > 1024 * 1024:  # 1MB limit
            raise MessageValidationError("Message size exceeds 1MB limit")


class AgentCommunicationService:
    """
    Core Redis Pub/Sub communication service for agent orchestration.
    
    Provides reliable message passing between orchestrator and agents with:
    - Point-to-point messaging
    - Broadcast capabilities
    - Optional acknowledgments
    - Message persistence for reliability
    - Performance monitoring and diagnostics
    - Connection resilience and failover
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        connection_pool_size: int = 10,
        enable_persistence: bool = True,
        message_ttl_seconds: int = 3600,
        ack_timeout_seconds: int = 30,
        enable_streams: bool = True,
        consumer_name: Optional[str] = None
    ):
        """
        Initialize the communication service.
        
        Args:
            redis_url: Redis connection URL
            connection_pool_size: Size of Redis connection pool
            enable_persistence: Whether to persist messages for reliability
            message_ttl_seconds: Default TTL for messages
            ack_timeout_seconds: Timeout for acknowledgments
            enable_streams: Whether to use Redis Streams for durable messaging
            consumer_name: Unique consumer identifier for streams
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.connection_pool_size = connection_pool_size
        self.enable_persistence = enable_persistence
        self.message_ttl_seconds = message_ttl_seconds
        self.ack_timeout_seconds = ack_timeout_seconds
        self.enable_streams = enable_streams
        
        # Connection management
        self._redis: Optional[Redis] = None
        self._connection_pool = None
        self._connected = False
        
        # Redis Streams manager for durable messaging
        self._streams_manager: Optional[RedisPubSubManager] = None
        if enable_streams:
            self._streams_manager = RedisPubSubManager(
                redis_url=redis_url,
                connection_pool_size=connection_pool_size,
                consumer_name=consumer_name
            )
        
        # Subscription management
        self._subscriptions: Dict[str, asyncio.Task] = {}
        self._message_handlers: Dict[str, Callable[[AgentMessage], None]] = {}
        self._subscriber_counts: Dict[str, int] = {}
        
        # Performance tracking
        self._metrics = MessageMetrics()
        self._latency_samples: List[float] = []
        self._message_history: List[Dict[str, Any]] = []
        self._start_time = time.time()
        
        # Acknowledgment tracking
        self._pending_acks: Dict[str, Dict[str, Any]] = {}
        self._ack_handlers: Dict[str, Callable[[str, bool], None]] = {}
        
        # Validation
        self._validator = MessageValidator()
        
        # Monitoring
        self._last_health_check = time.time()
        self._health_check_interval = 30  # seconds
        
    async def connect(self) -> None:
        """Establish Redis connection with resilience."""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=self.connection_pool_size,
                retry_on_timeout=True,
                retry_on_error=[RedisConnectionError],
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            self._redis = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            
            # Connect streams manager if enabled
            if self._streams_manager:
                await self._streams_manager.connect()
            
            logger.info(
                "Connected to Redis for agent communication",
                extra={
                    "redis_url": self.redis_url,
                    "pool_size": self.connection_pool_size,
                    "persistence_enabled": self.enable_persistence,
                    "streams_enabled": self.enable_streams
                }
            )
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources."""
        # Stop all subscription tasks
        for task in self._subscriptions.values():
            if not task.done():
                task.cancel()
        
        if self._subscriptions:
            await asyncio.gather(*self._subscriptions.values(), return_exceptions=True)
        
        # Disconnect streams manager if enabled
        if self._streams_manager:
            await self._streams_manager.disconnect()
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        self._connected = False
        logger.info("Disconnected from Redis agent communication service")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for communication service session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    def _ensure_connected(self) -> None:
        """Ensure Redis connection is active."""
        if not self._connected or not self._redis:
            raise ConnectionError("Not connected to Redis. Call connect() first.")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send message to agent or broadcast channel.
        
        Args:
            message: AgentMessage to send
            
        Returns:
            True if message was sent successfully
            
        Raises:
            MessageValidationError: If message validation fails
            DeliveryError: If message delivery fails
        """
        self._ensure_connected()
        
        # Validate message
        self._validator.validate_message(message)
        
        # Check if expired
        if message.is_expired():
            logger.warning(f"Message {message.id} expired before sending")
            return False
        
        try:
            start_time = time.time()
            
            # Determine channel
            channel = message.get_channel_name()
            
            # Serialize message
            serialized_message = json.dumps(message.to_dict())
            
            # Publish to Redis
            subscriber_count = await self._redis.publish(channel, serialized_message)
            
            # Track metrics
            delivery_time_ms = (time.time() - start_time) * 1000
            self._update_send_metrics(delivery_time_ms, subscriber_count > 0)
            
            # Persist message if enabled
            if self.enable_persistence:
                await self._persist_message(message, channel, subscriber_count)
            
            # Handle acknowledgment if required
            if message.acknowledgment_required:
                await self._track_acknowledgment(message)
            
            logger.debug(
                f"Message sent to {channel}",
                extra={
                    "message_id": message.id,
                    "channel": channel,
                    "subscribers": subscriber_count,
                    "latency_ms": delivery_time_ms
                }
            )
            
            return subscriber_count > 0
            
        except RedisError as e:
            self._metrics.total_failed += 1
            logger.error(f"Failed to send message {message.id}: {e}")
            raise DeliveryError(f"Message delivery failed: {e}")
    
    async def broadcast_message(self, message: AgentMessage) -> int:
        """
        Broadcast message to all agents.
        
        Args:
            message: Message to broadcast (to_agent will be set to None)
            
        Returns:
            Number of subscribers that received the message
        """
        # Ensure it's a broadcast message
        message.to_agent = None
        
        success = await self.send_message(message)
        return self._subscriber_counts.get("broadcast", 0) if success else 0
    
    async def subscribe_agent(
        self,
        agent_id: str,
        callback: Callable[[AgentMessage], None]
    ) -> None:
        """
        Subscribe agent to its dedicated channel.
        
        Args:
            agent_id: Unique agent identifier
            callback: Function to call when message is received
        """
        self._ensure_connected()
        
        channel = f"agent:{agent_id}"
        
        if channel in self._subscriptions:
            logger.warning(f"Agent {agent_id} already subscribed to {channel}")
            return
        
        # Register handler
        self._message_handlers[channel] = callback
        
        # Start subscription task
        task = asyncio.create_task(
            self._subscriber_loop(channel, callback)
        )
        self._subscriptions[channel] = task
        
        logger.info(f"Agent {agent_id} subscribed to channel {channel}")
    
    async def subscribe_broadcast(
        self,
        callback: Callable[[AgentMessage], None]
    ) -> None:
        """
        Subscribe to broadcast channel for system-wide messages.
        
        Args:
            callback: Function to call when broadcast message is received
        """
        self._ensure_connected()
        
        channel = "broadcast"
        
        if channel in self._subscriptions:
            logger.warning("Already subscribed to broadcast channel")
            return
        
        # Register handler
        self._message_handlers[channel] = callback
        
        # Start subscription task
        task = asyncio.create_task(
            self._subscriber_loop(channel, callback)
        )
        self._subscriptions[channel] = task
        
        logger.info("Subscribed to broadcast channel")
    
    async def unsubscribe_agent(self, agent_id: str) -> None:
        """Unsubscribe agent from its channel."""
        channel = f"agent:{agent_id}"
        await self._unsubscribe_channel(channel)
    
    async def unsubscribe_broadcast(self) -> None:
        """Unsubscribe from broadcast channel."""
        await self._unsubscribe_channel("broadcast")
    
    async def _unsubscribe_channel(self, channel: str) -> None:
        """Unsubscribe from a specific channel."""
        if channel in self._subscriptions:
            task = self._subscriptions[channel]
            if not task.done():
                task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._subscriptions[channel]
            del self._message_handlers[channel]
            
            logger.info(f"Unsubscribed from channel {channel}")
    
    async def _subscriber_loop(
        self,
        channel: str,
        callback: Callable[[AgentMessage], None]
    ) -> None:
        """Main subscriber loop for a channel."""
        pubsub = self._redis.pubsub()
        
        try:
            await pubsub.subscribe(channel)
            self._subscriber_counts[channel] = self._subscriber_counts.get(channel, 0) + 1
            
            logger.info(f"Started subscriber loop for channel {channel}")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse message
                        data = json.loads(message["data"])
                        agent_message = AgentMessage.from_dict(data)
                        
                        # Check expiration
                        if agent_message.is_expired():
                            logger.warning(f"Received expired message {agent_message.id}")
                            continue
                        
                        # Track metrics
                        self._update_receive_metrics(agent_message)
                        
                        # Handle acknowledgment
                        if agent_message.acknowledgment_required:
                            await self._send_acknowledgment(agent_message)
                        
                        # Call handler
                        await asyncio.get_event_loop().run_in_executor(
                            None, callback, agent_message
                        )
                        
                        logger.debug(f"Processed message {agent_message.id} on {channel}")
                        
                    except Exception as e:
                        logger.error(f"Error processing message on {channel}: {e}")
                        
        except asyncio.CancelledError:
            logger.info(f"Subscriber loop for {channel} cancelled")
            
        except Exception as e:
            logger.error(f"Error in subscriber loop for {channel}: {e}")
            
        finally:
            try:
                await pubsub.close()
                if channel in self._subscriber_counts:
                    self._subscriber_counts[channel] -= 1
                    if self._subscriber_counts[channel] <= 0:
                        del self._subscriber_counts[channel]
            except Exception as e:
                logger.error(f"Error closing pubsub for {channel}: {e}")
    
    async def _persist_message(
        self,
        message: AgentMessage,
        channel: str,
        subscriber_count: int
    ) -> None:
        """Persist message for reliability and audit purposes."""
        if not self.enable_persistence:
            return
        
        try:
            # Store in Redis list for durability
            persistence_key = f"messages:{channel}:history"
            
            message_record = {
                **message.to_dict(),
                "channel": channel,
                "subscriber_count": subscriber_count,
                "persisted_at": time.time()
            }
            
            # Add to list with TTL
            await self._redis.lpush(persistence_key, json.dumps(message_record))
            await self._redis.expire(persistence_key, self.message_ttl_seconds)
            
            # Trim list to prevent memory issues (keep last 1000 messages)
            await self._redis.ltrim(persistence_key, 0, 999)
            
        except RedisError as e:
            logger.error(f"Failed to persist message {message.id}: {e}")
    
    async def _track_acknowledgment(self, message: AgentMessage) -> None:
        """Track message acknowledgment requirements."""
        if not message.acknowledgment_required:
            return
        
        ack_key = f"ack:{message.id}"
        self._pending_acks[ack_key] = {
            "message_id": message.id,
            "sent_at": time.time(),
            "timeout_at": time.time() + self.ack_timeout_seconds,
            "acknowledged": False
        }
        
        # Schedule timeout cleanup
        async def cleanup_ack():
            await asyncio.sleep(self.ack_timeout_seconds)
            if ack_key in self._pending_acks and not self._pending_acks[ack_key]["acknowledged"]:
                logger.warning(f"Acknowledgment timeout for message {message.id}")
                if message.id in self._ack_handlers:
                    self._ack_handlers[message.id](message.id, False)
                del self._pending_acks[ack_key]
        
        asyncio.create_task(cleanup_ack())
    
    async def _send_acknowledgment(self, message: AgentMessage) -> None:
        """Send acknowledgment for received message."""
        if not message.acknowledgment_required:
            return
        
        try:
            ack_message = AgentMessage(
                id=str(uuid.uuid4()),
                from_agent="system",
                to_agent=message.from_agent,
                type=MessageType.EVENT,
                payload={
                    "type": "acknowledgment",
                    "original_message_id": message.id,
                    "acknowledged_at": time.time()
                },
                timestamp=time.time()
            )
            
            await self.send_message(ack_message)
            
        except Exception as e:
            logger.error(f"Failed to send acknowledgment for {message.id}: {e}")
    
    def _update_send_metrics(self, latency_ms: float, delivered: bool) -> None:
        """Update metrics for sent messages."""
        self._metrics.total_sent += 1
        
        if delivered:
            self._metrics.total_delivered += 1
        else:
            self._metrics.total_failed += 1
        
        # Update latency tracking
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 1000:  # Keep last 1000 samples
            self._latency_samples = self._latency_samples[-1000:]
        
        # Recalculate metrics
        self._recalculate_metrics()
    
    def _update_receive_metrics(self, message: AgentMessage) -> None:
        """Update metrics for received messages."""
        self._metrics.total_acknowledged += 1
        
        # Track receive latency (time since message was sent)
        if message.timestamp:
            receive_latency_ms = (time.time() - message.timestamp) * 1000
            self._latency_samples.append(receive_latency_ms)
            if len(self._latency_samples) > 1000:
                self._latency_samples = self._latency_samples[-1000:]
        
        self._recalculate_metrics()
    
    def _recalculate_metrics(self) -> None:
        """Recalculate performance metrics."""
        if self._metrics.total_sent > 0:
            self._metrics.delivery_rate = self._metrics.total_delivered / self._metrics.total_sent
            self._metrics.error_rate = self._metrics.total_failed / self._metrics.total_sent
        
        if self._latency_samples:
            self._latency_samples.sort()
            n = len(self._latency_samples)
            
            self._metrics.average_latency_ms = sum(self._latency_samples) / n
            self._metrics.p95_latency_ms = self._latency_samples[int(n * 0.95)]
            self._metrics.p99_latency_ms = self._latency_samples[int(n * 0.99)]
        
        # Calculate throughput
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            self._metrics.throughput_msg_per_sec = self._metrics.total_sent / elapsed_time
    
    async def get_channel_info(self, channel: str) -> ChannelInfo:
        """Get information about a channel."""
        self._ensure_connected()
        
        try:
            # Get subscriber count (this is approximate in Redis Pub/Sub)
            subscriber_count = self._subscriber_counts.get(channel, 0)
            
            # Get message count from persistence
            message_count = 0
            if self.enable_persistence:
                persistence_key = f"messages:{channel}:history"
                message_count = await self._redis.llen(persistence_key)
            
            return ChannelInfo(
                name=channel,
                subscriber_count=subscriber_count,
                message_count=message_count,
                last_activity=datetime.utcnow()  # Simplified
            )
            
        except RedisError as e:
            raise AgentCommunicationError(f"Failed to get channel info: {e}")
    
    async def get_metrics(self) -> MessageMetrics:
        """Get current performance metrics."""
        self._recalculate_metrics()
        return self._metrics
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics."""
        metrics = await self.get_metrics()
        
        return {
            "message_metrics": metrics.to_dict(),
            "connection_status": {
                "connected": self._connected,
                "redis_url": self.redis_url,
                "pool_size": self.connection_pool_size
            },
            "subscription_status": {
                "active_subscriptions": len(self._subscriptions),
                "channels": list(self._subscriptions.keys()),
                "subscriber_counts": self._subscriber_counts.copy()
            },
            "acknowledgment_status": {
                "pending_acks": len(self._pending_acks),
                "ack_timeout_seconds": self.ack_timeout_seconds
            },
            "service_status": {
                "uptime_seconds": time.time() - self._start_time,
                "persistence_enabled": self.enable_persistence,
                "last_health_check": self._last_health_check
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            self._ensure_connected()
            
            # Test Redis connection
            start_time = time.time()
            await self._redis.ping()
            ping_latency_ms = (time.time() - start_time) * 1000
            
            # Update health check timestamp
            self._last_health_check = time.time()
            
            metrics = await self.get_metrics()
            
            # Determine health status
            is_healthy = (
                self._connected and 
                ping_latency_ms < 1000 and  # < 1 second
                metrics.error_rate < 0.1  # < 10% error rate
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "connected": self._connected,
                "ping_latency_ms": ping_latency_ms,
                "error_rate": metrics.error_rate,
                "delivery_rate": metrics.delivery_rate,
                "active_subscriptions": len(self._subscriptions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Redis Streams Integration Methods
    
    async def send_durable_message(self, message: AgentMessage) -> Optional[str]:
        """
        Send message via Redis Streams for durability and ordering guarantees.
        
        Args:
            message: AgentMessage to send
            
        Returns:
            Stream message ID if successful, None otherwise
        """
        if not self._streams_manager:
            logger.warning("Streams not enabled, falling back to pub/sub")
            success = await self.send_message(message)
            return str(uuid.uuid4()) if success else None
        
        try:
            # Convert AgentMessage to StreamMessage
            stream_message = StreamMessage(
                id=message.id,
                from_agent=message.from_agent,
                to_agent=message.to_agent,
                message_type=message.type,
                payload=message.payload,
                priority=message.priority,
                timestamp=message.timestamp,
                ttl=message.ttl,
                correlation_id=message.correlation_id
            )
            
            # Determine stream name
            stream_name = stream_message.get_stream_name()
            
            # Send to stream
            message_id = await self._streams_manager.send_stream_message(
                stream_name, stream_message
            )
            
            logger.debug(
                f"Sent durable message to stream {stream_name}",
                extra={
                    "message_id": message_id,
                    "original_id": message.id,
                    "stream_name": stream_name
                }
            )
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send durable message: {e}")
            return None
    
    async def subscribe_agent_stream(
        self,
        agent_id: str,
        callback: Callable[[StreamMessage], Any],
        consumer_group: Optional[str] = None
    ) -> None:
        """
        Subscribe agent to its dedicated stream with consumer group support.
        
        Args:
            agent_id: Unique agent identifier
            callback: Function to call when message is received
            consumer_group: Consumer group name (defaults to agent type)
        """
        if not self._streams_manager:
            logger.warning("Streams not enabled, falling back to pub/sub subscription")
            await self.subscribe_agent(agent_id, lambda msg: callback(
                StreamMessage.from_stream_message(msg)
            ))
            return
        
        stream_name = f"agent_messages:{agent_id}"
        group_name = consumer_group or f"consumers:{agent_id}"
        
        try:
            await self._streams_manager.consume_stream_messages(
                stream_name=stream_name,
                group_name=group_name,
                handler=callback,
                auto_ack=True,
                claim_stalled=True
            )
            
            logger.info(
                f"Agent {agent_id} subscribed to stream {stream_name} with group {group_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to subscribe agent to stream: {e}")
            raise AgentCommunicationError(f"Stream subscription failed: {e}")
    
    async def subscribe_broadcast_stream(
        self,
        callback: Callable[[StreamMessage], Any],
        consumer_group: str = "broadcast_consumers"
    ) -> None:
        """
        Subscribe to broadcast stream for system-wide messages.
        
        Args:
            callback: Function to call when broadcast message is received
            consumer_group: Consumer group name for broadcast messages
        """
        if not self._streams_manager:
            logger.warning("Streams not enabled, falling back to pub/sub subscription")
            await self.subscribe_broadcast(lambda msg: callback(
                StreamMessage.from_stream_message(msg)
            ))
            return
        
        try:
            await self._streams_manager.consume_stream_messages(
                stream_name="agent_messages:broadcast",
                group_name=consumer_group,
                handler=callback,
                auto_ack=True,
                claim_stalled=True
            )
            
            logger.info(f"Subscribed to broadcast stream with group {consumer_group}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to broadcast stream: {e}")
            raise AgentCommunicationError(f"Broadcast stream subscription failed: {e}")
    
    async def get_stream_stats(self, stream_name: str) -> Optional[StreamStats]:
        """
        Get statistics for a specific stream.
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            StreamStats object or None if streams not enabled
        """
        if not self._streams_manager:
            return None
        
        try:
            return await self._streams_manager.get_stream_stats(stream_name)
        except Exception as e:
            logger.error(f"Failed to get stream stats: {e}")
            return None
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics including streams data."""
        metrics = await super().get_comprehensive_metrics()
        
        # Add streams metrics if available
        if self._streams_manager:
            try:
                streams_metrics = await self._streams_manager.get_performance_metrics()
                metrics["streams_metrics"] = streams_metrics
                
                # Get stats for common streams
                agent_stream_stats = await self.get_stream_stats("agent_messages:broadcast")
                if agent_stream_stats:
                    metrics["broadcast_stream_stats"] = agent_stream_stats.to_dict()
                    
            except Exception as e:
                logger.error(f"Failed to get streams metrics: {e}")
                metrics["streams_error"] = str(e)
        
        return metrics
    
    async def replay_messages(
        self,
        stream_name: str,
        start_id: str = "0-0",
        end_id: str = "+",
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Replay messages from a stream for debugging or recovery.
        
        Args:
            stream_name: Name of the stream
            start_id: Starting message ID
            end_id: Ending message ID
            count: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        if not self._streams_manager:
            logger.warning("Streams not enabled, cannot replay messages")
            return []
        
        try:
            self._streams_manager._ensure_connected()
            
            # Use XRANGE to get messages
            messages = await self._streams_manager._redis.xrange(
                stream_name, min=start_id, max=end_id, count=count
            )
            
            replayed_messages = []
            for message_id, fields in messages:
                try:
                    stream_message = StreamMessage.from_redis_dict(fields)
                    replayed_messages.append({
                        "stream_message_id": message_id,
                        "message": stream_message.dict(),
                        "timestamp": fields.get("timestamp", "unknown")
                    })
                except Exception as parse_error:
                    logger.error(f"Failed to parse replayed message: {parse_error}")
                    replayed_messages.append({
                        "stream_message_id": message_id,
                        "raw_fields": fields,
                        "parse_error": str(parse_error)
                    })
            
            logger.info(f"Replayed {len(replayed_messages)} messages from {stream_name}")
            return replayed_messages
            
        except Exception as e:
            logger.error(f"Failed to replay messages: {e}")
            return []
    
    async def get_dead_letter_messages(
        self,
        original_stream: str,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get messages from dead letter queue for a stream.
        
        Args:
            original_stream: Original stream name
            count: Maximum number of DLQ messages to retrieve
            
        Returns:
            List of dead letter message dictionaries
        """
        if not self._streams_manager:
            return []
        
        dlq_stream = f"{original_stream}{self._streams_manager.dead_letter_stream_suffix}"
        
        try:
            return await self.replay_messages(dlq_stream, count=count)
        except Exception as e:
            logger.error(f"Failed to get dead letter messages: {e}")
            return []