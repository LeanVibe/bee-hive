"""
Agent Communication Service for LeanVibe Agent Hive 2.0 - Upgraded for Vertical Slice 4.2

Enhanced communication service now supporting both Redis Pub/Sub (VS 4.1) and 
Redis Streams with Consumer Groups (VS 4.2) for comprehensive message handling.

Provides unified interface for:
- Traditional Pub/Sub messaging for real-time notifications
- Consumer Groups for reliable task distribution and load balancing  
- Workflow-aware message routing with dependency management
- Dead letter queue handling for poison messages
- Seamless migration from Pub/Sub to Streams architecture
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
from .enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType, MessageRoutingMode
)
from .consumer_group_coordinator import ConsumerGroupCoordinator
from .workflow_message_router import WorkflowMessageRouter
from .dead_letter_queue_handler import DeadLetterQueueHandler

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
    Enhanced communication service supporting both Pub/Sub and Consumer Groups.
    
    Provides unified interface for:
    - Traditional Pub/Sub for real-time notifications (VS 4.1)
    - Consumer Groups for reliable task distribution (VS 4.2)
    - Workflow-aware message routing with dependency management
    - Dead letter queue handling for poison messages
    - Seamless migration between communication patterns
    - Comprehensive monitoring and diagnostics
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        connection_pool_size: int = 10,
        enable_persistence: bool = True,
        message_ttl_seconds: int = 3600,
        ack_timeout_seconds: int = 30,
        enable_streams: bool = True,
        enable_consumer_groups: bool = True,
        enable_workflow_routing: bool = True,
        consumer_name: Optional[str] = None
    ):
        """
        Initialize the enhanced communication service.
        
        Args:
            redis_url: Redis connection URL
            connection_pool_size: Size of Redis connection pool
            enable_persistence: Whether to persist messages for reliability
            message_ttl_seconds: Default TTL for messages
            ack_timeout_seconds: Timeout for acknowledgments
            enable_streams: Whether to use Redis Streams for durable messaging
            enable_consumer_groups: Whether to enable consumer groups (VS 4.2)
            enable_workflow_routing: Whether to enable workflow-aware routing
            consumer_name: Unique consumer identifier for streams
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.connection_pool_size = connection_pool_size
        self.enable_persistence = enable_persistence
        self.message_ttl_seconds = message_ttl_seconds
        self.ack_timeout_seconds = ack_timeout_seconds
        self.enable_streams = enable_streams
        self.enable_consumer_groups = enable_consumer_groups
        self.enable_workflow_routing = enable_workflow_routing
        
        # Connection management
        self._redis: Optional[Redis] = None
        self._connection_pool = None
        self._connected = False
        
        # Redis Streams manager for durable messaging (VS 4.1)
        self._streams_manager: Optional[RedisPubSubManager] = None
        if enable_streams:
            self._streams_manager = RedisPubSubManager(
                redis_url=redis_url,
                connection_pool_size=connection_pool_size,
                consumer_name=consumer_name
            )
        
        # Enhanced components for VS 4.2
        self._enhanced_streams_manager: Optional[EnhancedRedisStreamsManager] = None
        self._consumer_group_coordinator: Optional[ConsumerGroupCoordinator] = None
        self._workflow_router: Optional[WorkflowMessageRouter] = None
        self._dlq_handler: Optional[DeadLetterQueueHandler] = None
        
        if enable_consumer_groups:
            self._enhanced_streams_manager = EnhancedRedisStreamsManager(
                redis_url=redis_url,
                connection_pool_size=connection_pool_size
            )
            
            self._consumer_group_coordinator = ConsumerGroupCoordinator(
                self._enhanced_streams_manager
            )
            
            self._dlq_handler = DeadLetterQueueHandler(
                self._enhanced_streams_manager
            )
            
            if enable_workflow_routing:
                self._workflow_router = WorkflowMessageRouter(
                    self._enhanced_streams_manager,
                    self._consumer_group_coordinator
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
            
            # Connect streams manager if enabled (VS 4.1)
            if self._streams_manager:
                await self._streams_manager.connect()
            
            # Connect enhanced components if enabled (VS 4.2)
            if self._enhanced_streams_manager:
                await self._enhanced_streams_manager.connect()
            
            if self._consumer_group_coordinator:
                await self._consumer_group_coordinator.start()
            
            if self._workflow_router:
                await self._workflow_router.start()
            
            if self._dlq_handler:
                await self._dlq_handler.start()
            
            logger.info(
                "Connected to Redis for enhanced agent communication",
                extra={
                    "redis_url": self.redis_url,
                    "pool_size": self.connection_pool_size,
                    "persistence_enabled": self.enable_persistence,
                    "streams_enabled": self.enable_streams,
                    "consumer_groups_enabled": self.enable_consumer_groups,
                    "workflow_routing_enabled": self.enable_workflow_routing
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
        
        # Disconnect enhanced components if enabled (VS 4.2)
        if self._dlq_handler:
            await self._dlq_handler.stop()
        
        if self._workflow_router:
            await self._workflow_router.stop()
        
        if self._consumer_group_coordinator:
            await self._consumer_group_coordinator.stop()
        
        if self._enhanced_streams_manager:
            await self._enhanced_streams_manager.disconnect()
        
        # Disconnect streams manager if enabled (VS 4.1)
        if self._streams_manager:
            await self._streams_manager.disconnect()
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        self._connected = False
        logger.info("Disconnected from enhanced Redis agent communication service")
    
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
    
    # Enhanced Communication Methods (VS 4.2)
    
    async def create_consumer_group(
        self,
        group_name: str,
        stream_name: str,
        agent_type: ConsumerGroupType,
        routing_mode: MessageRoutingMode = MessageRoutingMode.LOAD_BALANCED,
        **kwargs
    ) -> bool:
        """
        Create a consumer group for agent specialization.
        
        Args:
            group_name: Name of the consumer group
            stream_name: Target stream name
            agent_type: Type of agents in this group
            routing_mode: Message routing strategy
            **kwargs: Additional configuration options
            
        Returns:
            True if group was created successfully
        """
        if not self._consumer_group_coordinator:
            logger.warning("Consumer groups not enabled, falling back to streams")
            return False
        
        try:
            config = ConsumerGroupConfig(
                name=group_name,
                stream_name=stream_name,
                agent_type=agent_type,
                routing_mode=routing_mode,
                **kwargs
            )
            
            await self._enhanced_streams_manager.create_consumer_group(config)
            
            logger.info(f"Created consumer group {group_name} for {agent_type.value} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create consumer group {group_name}: {e}")
            return False
    
    async def join_consumer_group(
        self,
        agent_id: str,
        group_name: str,
        message_handler: Callable[[StreamMessage], Any]
    ) -> bool:
        """
        Add an agent to a consumer group.
        
        Args:
            agent_id: Unique agent identifier
            group_name: Consumer group to join
            message_handler: Function to handle received messages
            
        Returns:
            True if agent joined successfully
        """
        if not self._enhanced_streams_manager:
            logger.warning("Consumer groups not enabled")
            return False
        
        try:
            await self._enhanced_streams_manager.add_consumer_to_group(
                group_name, agent_id, message_handler
            )
            
            logger.info(f"Agent {agent_id} joined consumer group {group_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join consumer group {group_name}: {e}")
            return False
    
    async def leave_consumer_group(
        self,
        agent_id: str,
        group_name: str
    ) -> bool:
        """
        Remove an agent from a consumer group.
        
        Args:
            agent_id: Agent identifier
            group_name: Consumer group to leave
            
        Returns:
            True if agent left successfully
        """
        if not self._enhanced_streams_manager:
            return False
        
        try:
            await self._enhanced_streams_manager.remove_consumer_from_group(
                group_name, agent_id
            )
            
            logger.info(f"Agent {agent_id} left consumer group {group_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave consumer group {group_name}: {e}")
            return False
    
    async def send_to_consumer_group(
        self,
        group_name: str,
        message: AgentMessage,
        routing_mode: Optional[MessageRoutingMode] = None
    ) -> Optional[str]:
        """
        Send message to a consumer group with intelligent routing.
        
        Args:
            group_name: Target consumer group
            message: Message to send
            routing_mode: Optional routing mode override
            
        Returns:
            Message ID if successful, None otherwise
        """
        if not self._enhanced_streams_manager:
            logger.warning("Consumer groups not enabled, falling back to pub/sub")
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
            
            message_id = await self._enhanced_streams_manager.send_message_to_group(
                group_name, stream_message, routing_mode
            )
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send message to consumer group {group_name}: {e}")
            return None
    
    async def route_workflow_message(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Route workflow with intelligent task distribution.
        
        Args:
            workflow_id: Unique workflow identifier
            tasks: List of task definitions
            
        Returns:
            Routing results or None if workflow routing disabled
        """
        if not self._workflow_router:
            logger.warning("Workflow routing not enabled")
            return None
        
        try:
            result = await self._workflow_router.route_workflow(workflow_id, tasks)
            return result
            
        except Exception as e:
            logger.error(f"Failed to route workflow {workflow_id}: {e}")
            return None
    
    async def signal_task_completion(
        self,
        workflow_id: str,
        task_id: str,
        result: Dict[str, Any]
    ) -> List[str]:
        """
        Signal task completion and trigger dependent tasks.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Completed task identifier
            result: Task completion result
            
        Returns:
            List of newly triggered task IDs
        """
        if not self._workflow_router:
            return []
        
        try:
            triggered_tasks = await self._workflow_router.signal_task_completion(
                workflow_id, task_id, result
            )
            return triggered_tasks
            
        except Exception as e:
            logger.error(f"Failed to signal task completion: {e}")
            return []
    
    async def replay_failed_message(
        self,
        dlq_message_id: str,
        target_stream: Optional[str] = None,
        priority_boost: bool = False
    ) -> bool:
        """
        Replay a message from the Dead Letter Queue.
        
        Args:
            dlq_message_id: DLQ message identifier
            target_stream: Optional target stream override
            priority_boost: Whether to boost message priority
            
        Returns:
            True if replay was successful
        """
        if not self._dlq_handler:
            logger.warning("DLQ handler not enabled")
            return False
        
        try:
            success = await self._dlq_handler.replay_message(
                dlq_message_id, target_stream, priority_boost
            )
            return success
            
        except Exception as e:
            logger.error(f"Failed to replay DLQ message {dlq_message_id}: {e}")
            return False
    
    async def get_consumer_group_stats(self) -> Dict[str, Any]:
        """Get comprehensive consumer group statistics."""
        if not self._enhanced_streams_manager:
            return {}
        
        try:
            return await self._enhanced_streams_manager.get_all_group_stats()
        except Exception as e:
            logger.error(f"Failed to get consumer group stats: {e}")
            return {}
    
    async def get_dlq_statistics(self) -> Dict[str, Any]:
        """Get Dead Letter Queue statistics."""
        if not self._dlq_handler:
            return {}
        
        try:
            return await self._dlq_handler.get_dlq_statistics()
        except Exception as e:
            logger.error(f"Failed to get DLQ statistics: {e}")
            return {}
    
    async def get_enhanced_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including all VS 4.2 components."""
        base_metrics = await self.get_comprehensive_metrics()
        
        enhanced_metrics = {
            **base_metrics,
            "vs_4_2_enabled": {
                "consumer_groups": self.enable_consumer_groups,
                "workflow_routing": self.enable_workflow_routing
            }
        }
        
        # Add enhanced component metrics
        if self._enhanced_streams_manager:
            try:
                enhanced_metrics["enhanced_streams"] = await self._enhanced_streams_manager.get_performance_metrics()
            except Exception as e:
                logger.error(f"Failed to get enhanced streams metrics: {e}")
        
        if self._consumer_group_coordinator:
            try:
                enhanced_metrics["coordinator"] = await self._consumer_group_coordinator.get_coordinator_metrics()
            except Exception as e:
                logger.error(f"Failed to get coordinator metrics: {e}")
        
        if self._workflow_router:
            try:
                enhanced_metrics["workflow_routing"] = await self._workflow_router.get_routing_metrics()
            except Exception as e:
                logger.error(f"Failed to get workflow routing metrics: {e}")
        
        if self._dlq_handler:
            try:
                enhanced_metrics["dlq"] = await self._dlq_handler.get_dlq_statistics()
            except Exception as e:
                logger.error(f"Failed to get DLQ statistics: {e}")
        
        return enhanced_metrics
    
    async def enhanced_health_check(self) -> Dict[str, Any]:
        """Enhanced health check including all VS 4.2 components."""
        base_health = await self.health_check()
        
        enhanced_health = {
            **base_health,
            "vs_4_2_components": {}
        }
        
        # Check enhanced component health
        if self._enhanced_streams_manager:
            try:
                enhanced_health["vs_4_2_components"]["enhanced_streams"] = await self._enhanced_streams_manager.health_check()
            except Exception as e:
                enhanced_health["vs_4_2_components"]["enhanced_streams"] = {"status": "error", "error": str(e)}
        
        if self._consumer_group_coordinator:
            try:
                enhanced_health["vs_4_2_components"]["coordinator"] = await self._consumer_group_coordinator.health_check()
            except Exception as e:
                enhanced_health["vs_4_2_components"]["coordinator"] = {"status": "error", "error": str(e)}
        
        if self._workflow_router:
            try:
                enhanced_health["vs_4_2_components"]["workflow_router"] = await self._workflow_router.health_check()
            except Exception as e:
                enhanced_health["vs_4_2_components"]["workflow_router"] = {"status": "error", "error": str(e)}
        
        if self._dlq_handler:
            try:
                enhanced_health["vs_4_2_components"]["dlq_handler"] = await self._dlq_handler.health_check()
            except Exception as e:
                enhanced_health["vs_4_2_components"]["dlq_handler"] = {"status": "error", "error": str(e)}
        
        # Determine overall enhanced health
        component_healths = [
            comp.get("status", "unknown") 
            for comp in enhanced_health["vs_4_2_components"].values()
        ]
        
        if component_healths:
            all_healthy = all(status == "healthy" for status in component_healths)
            enhanced_health["enhanced_status"] = "healthy" if all_healthy else "degraded"
        else:
            enhanced_health["enhanced_status"] = base_health.get("status", "unknown")
        
        return enhanced_health