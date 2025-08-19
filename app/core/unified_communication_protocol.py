"""
Unified Communication Protocol Foundation - Phase 0 POC Week 1
LeanVibe Agent Hive 2.0 - Communication Protocol Unification

This module consolidates 554+ fragmented communication files into a single,
high-performance unified communication protocol with:
- Single Redis client pattern with connection pooling
- Standardized message formats across all components
- Unified event bus for system-wide coordination
- Protocol adapters for different CLI interfaces
- Performance monitoring and circuit breaker patterns

CONSOLIDATION TARGET:
- redis.py, optimized_redis.py, redis_integration.py → UnifiedRedisClient
- Multiple message models → StandardUniversalMessage
- communication/ and communication_hub/ → Single protocol layer
- Fragmented WebSocket implementations → Unified WebSocket adapter

PERFORMANCE REQUIREMENTS:
- Message routing: <10ms latency
- Throughput: 10,000+ messages/second
- Connection pooling: Auto-scaling based on load
- Circuit breaker: 5-failure threshold with exponential backoff
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator, Protocol

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

# Optional import of settings (graceful degradation if not available)
try:
    from .config import settings
    REDIS_HOST = getattr(settings, 'REDIS_HOST', 'localhost')
    REDIS_PORT = getattr(settings, 'REDIS_PORT', 6379)
except ImportError:
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
from .circuit_breaker import UnifiedCircuitBreaker, CircuitBreakerConfig

logger = structlog.get_logger("unified_communication")

# ================================================================================
# Core Data Models - Unified Message Format
# ================================================================================

class MessageType(str, Enum):
    """Standardized message types across all components."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    CONTEXT_HANDOFF = "context_handoff"
    AGENT_COORDINATION = "agent_coordination"
    HEALTH_CHECK = "health_check"
    ERROR_REPORT = "error_report"
    SYSTEM_EVENT = "system_event"
    WORKFLOW_CONTROL = "workflow_control"
    PERFORMANCE_METRICS = "performance_metrics"

class MessagePriority(str, Enum):
    """Message priority levels for routing optimization."""
    CRITICAL = "critical"      # System-critical messages (sub-second delivery)
    HIGH = "high"             # High-priority tasks (<5s delivery)
    NORMAL = "normal"         # Standard messages (<30s delivery)
    LOW = "low"              # Background messages (<5min delivery)
    BULK = "bulk"            # Bulk data transfer (best effort)

class DeliveryGuarantee(str, Enum):
    """Message delivery guarantees."""
    FIRE_AND_FORGET = "fire_and_forget"    # No acknowledgment required
    AT_LEAST_ONCE = "at_least_once"        # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"          # Guaranteed single delivery
    ORDERED = "ordered"                     # Maintain message order

class ProtocolType(str, Enum):
    """Supported protocol types."""
    REDIS_STREAM = "redis_stream"           # Redis Streams for reliable messaging
    REDIS_PUBSUB = "redis_pubsub"          # Redis Pub/Sub for real-time events
    WEBSOCKET = "websocket"                 # WebSocket for browser connections
    HTTP = "http"                           # HTTP for REST API
    TCP = "tcp"                            # Raw TCP for high-performance

@dataclass
class StandardUniversalMessage:
    """
    Unified message format that consolidates all previous message types.
    
    This replaces:
    - RedisStreamMessage
    - UniversalMessage from communication/
    - CLIMessage from protocol_models
    - BridgeConnection messages
    """
    
    # Core identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Message classification
    message_type: MessageType = MessageType.TASK_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    
    # Routing information
    from_agent: str = "system"
    to_agent: str = "broadcast"
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Message content
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery tracking
    retry_count: int = 0
    max_retries: int = 3
    ttl_seconds: int = 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "delivery_guarantee": self.delivery_guarantee.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "payload": self.payload,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "ttl_seconds": self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardUniversalMessage":
        """Create message from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            message_type=MessageType(data.get("message_type", "task_request")),
            priority=MessagePriority(data.get("priority", "normal")),
            delivery_guarantee=DeliveryGuarantee(data.get("delivery_guarantee", "at_least_once")),
            from_agent=data.get("from_agent", "system"),
            to_agent=data.get("to_agent", "broadcast"),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            ttl_seconds=data.get("ttl_seconds", 3600)
        )

# ================================================================================
# Unified Redis Client - Consolidates 6+ Redis Implementations
# ================================================================================

@dataclass
class RedisConfig:
    """Configuration for unified Redis client."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    retry_attempts: int = 3
    connection_timeout: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=lambda: {
        "TCP_KEEPINTVL": 1,
        "TCP_KEEPCNT": 3,
        "TCP_KEEPIDLE": 1
    })

class UnifiedRedisClient:
    """
    Unified Redis client that consolidates all Redis implementations.
    
    This replaces:
    - redis.py (base Redis integration)
    - optimized_redis.py (performance optimized)
    - redis_integration.py (integration layer)
    - redis_pubsub_manager.py (pub/sub management)
    - enhanced_redis_streams_manager.py (streams management)
    - team_coordination_redis.py (coordination)
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        self.config = config or RedisConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._circuit_breaker = UnifiedCircuitBreaker(
            CircuitBreakerConfig.for_redis("unified_redis_client")
        )
        self._connection_count = 0
        self._performance_metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_failures": 0,
            "avg_latency_ms": 0.0
        }
    
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                socket_connect_timeout=self.config.connection_timeout,
                retry_on_timeout=True
            )
            
            self._client = Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            logger.info("Unified Redis client initialized", 
                       host=self.config.host, 
                       port=self.config.port,
                       max_connections=self.config.max_connections)
            
        except Exception as e:
            logger.error("Failed to initialize Redis client", error=str(e))
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get Redis connection with circuit breaker protection."""
        if not self._client:
            await self.initialize()
        
        async with self._circuit_breaker.call():
            try:
                self._connection_count += 1
                yield self._client
            except Exception as e:
                self._performance_metrics["connection_failures"] += 1
                logger.error("Redis connection error", error=str(e))
                raise
            finally:
                self._connection_count -= 1
    
    async def publish_message(self, channel: str, message: StandardUniversalMessage) -> bool:
        """Publish message using Redis pub/sub."""
        start_time = time.time()
        
        async with self.get_connection() as client:
            try:
                message_data = json.dumps(message.to_dict())
                await client.publish(channel, message_data)
                
                self._performance_metrics["messages_sent"] += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency_metrics(latency)
                
                logger.debug("Message published", channel=channel, message_id=message.message_id)
                return True
                
            except Exception as e:
                logger.error("Failed to publish message", channel=channel, error=str(e))
                return False
    
    async def send_stream_message(self, stream: str, message: StandardUniversalMessage) -> Optional[str]:
        """Send message using Redis Streams for guaranteed delivery."""
        start_time = time.time()
        
        async with self.get_connection() as client:
            try:
                message_data = message.to_dict()
                
                # Add to stream with automatic ID generation
                stream_id = await client.xadd(stream, message_data)
                
                self._performance_metrics["messages_sent"] += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency_metrics(latency)
                
                logger.debug("Stream message sent", stream=stream, 
                           message_id=message.message_id, stream_id=stream_id)
                return stream_id
                
            except Exception as e:
                logger.error("Failed to send stream message", stream=stream, error=str(e))
                return None
    
    async def consume_stream_messages(
        self,
        streams: Dict[str, str],
        consumer_group: str,
        consumer_name: str,
        count: int = 10,
        block: int = 1000
    ) -> List[StandardUniversalMessage]:
        """Consume messages from Redis Streams with consumer group."""
        async with self.get_connection() as client:
            try:
                # Read from streams
                results = await client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    streams,
                    count=count,
                    block=block
                )
                
                messages = []
                for stream, stream_messages in results:
                    for message_id, fields in stream_messages:
                        try:
                            # Decode bytes to strings
                            decoded_fields = {
                                k.decode() if isinstance(k, bytes) else k: 
                                v.decode() if isinstance(v, bytes) else v
                                for k, v in fields.items()
                            }
                            
                            # Reconstruct message
                            message_data = json.loads(decoded_fields.get("payload", "{}"))
                            message = StandardUniversalMessage.from_dict({
                                **decoded_fields,
                                **message_data
                            })
                            
                            messages.append(message)
                            self._performance_metrics["messages_received"] += 1
                            
                        except Exception as e:
                            logger.error("Failed to decode stream message", 
                                       stream=stream, message_id=message_id, error=str(e))
                
                return messages
                
            except Exception as e:
                logger.error("Failed to consume stream messages", error=str(e))
                return []
    
    async def acknowledge_message(self, stream: str, consumer_group: str, message_id: str):
        """Acknowledge message processing completion."""
        async with self.get_connection() as client:
            try:
                await client.xack(stream, consumer_group, message_id)
                logger.debug("Message acknowledged", stream=stream, message_id=message_id)
            except Exception as e:
                logger.error("Failed to acknowledge message", 
                           stream=stream, message_id=message_id, error=str(e))
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update average latency metrics."""
        current_avg = self._performance_metrics["avg_latency_ms"]
        message_count = self._performance_metrics["messages_sent"] + self._performance_metrics["messages_received"]
        
        # Calculate running average
        new_avg = ((current_avg * (message_count - 1)) + latency_ms) / message_count
        self._performance_metrics["avg_latency_ms"] = new_avg
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self._performance_metrics,
            "active_connections": self._connection_count,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "pool_info": {
                "created_connections": self._pool.created_connections if self._pool else 0,
                "available_connections": self._pool.available_connections if self._pool else 0,
                "in_use_connections": self._pool.in_use_connections if self._pool else 0
            } if self._pool else {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        try:
            start_time = time.time()
            async with self.get_connection() as client:
                await client.ping()
            
            latency = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "latency_ms": latency,
                "active_connections": self._connection_count,
                "circuit_breaker_state": self._circuit_breaker.state.value
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self._circuit_breaker.state.value
            }
    
    async def cleanup(self):
        """Cleanup Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        
        logger.info("Unified Redis client cleaned up")

# ================================================================================
# Protocol Adapter Interface
# ================================================================================

class ProtocolAdapter(Protocol):
    """Interface for protocol adapters."""
    
    async def send_message(self, message: StandardUniversalMessage) -> bool:
        """Send message using this protocol."""
        ...
    
    async def receive_messages(self, timeout: Optional[float] = None) -> List[StandardUniversalMessage]:
        """Receive messages from this protocol."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check protocol health."""
        ...

# ================================================================================
# Unified Communication Manager
# ================================================================================

class UnifiedCommunicationManager:
    """
    Central communication manager that consolidates all communication components.
    
    This replaces:
    - communication_manager.py
    - communication.py
    - communication_analyzer.py
    - communication_hub/communication_hub.py
    - All protocol adapters and bridges
    """
    
    def __init__(self, redis_config: Optional[RedisConfig] = None):
        self.redis_client = UnifiedRedisClient(redis_config)
        self._adapters: Dict[ProtocolType, ProtocolAdapter] = {}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._performance_metrics = {
            "total_messages_routed": 0,
            "avg_routing_latency_ms": 0.0,
            "failed_deliveries": 0,
            "active_subscriptions": 0
        }
        
        # Message routing queues by priority
        self._priority_queues = {
            MessagePriority.CRITICAL: asyncio.Queue(),
            MessagePriority.HIGH: asyncio.Queue(),
            MessagePriority.NORMAL: asyncio.Queue(),
            MessagePriority.LOW: asyncio.Queue(),
            MessagePriority.BULK: asyncio.Queue()
        }
        
        self._running = False
        self._routing_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize communication manager."""
        await self.redis_client.initialize()
        
        # Start message routing task
        self._running = True
        self._routing_task = asyncio.create_task(self._route_messages())
        
        logger.info("Unified Communication Manager initialized")
    
    async def send_message(
        self,
        message: StandardUniversalMessage,
        protocol: Optional[ProtocolType] = None
    ) -> bool:
        """
        Send message using optimal protocol routing.
        
        If protocol is not specified, automatically selects based on:
        - Message priority (critical -> Redis Streams)
        - Delivery guarantee (exactly_once -> Redis Streams)
        - Target agent preferences
        """
        start_time = time.time()
        
        try:
            # Auto-select protocol if not specified
            if protocol is None:
                protocol = self._select_optimal_protocol(message)
            
            # Add to appropriate priority queue
            await self._priority_queues[message.priority].put((message, protocol))
            
            # Update metrics
            self._performance_metrics["total_messages_routed"] += 1
            routing_latency = (time.time() - start_time) * 1000
            self._update_routing_latency(routing_latency)
            
            return True
            
        except Exception as e:
            logger.error("Failed to queue message for sending", 
                        message_id=message.message_id, error=str(e))
            self._performance_metrics["failed_deliveries"] += 1
            return False
    
    def _select_optimal_protocol(self, message: StandardUniversalMessage) -> ProtocolType:
        """Select optimal protocol based on message characteristics."""
        # Critical messages always use Redis Streams for reliability
        if message.priority == MessagePriority.CRITICAL:
            return ProtocolType.REDIS_STREAM
        
        # Exactly-once delivery requires Redis Streams
        if message.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            return ProtocolType.REDIS_STREAM
        
        # Fire-and-forget messages can use pub/sub for speed
        if message.delivery_guarantee == DeliveryGuarantee.FIRE_AND_FORGET:
            return ProtocolType.REDIS_PUBSUB
        
        # Default to Redis Streams for reliable delivery
        return ProtocolType.REDIS_STREAM
    
    async def _route_messages(self):
        """Background task for message routing with priority handling."""
        while self._running:
            try:
                # Process queues in priority order
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW, MessagePriority.BULK]:
                    
                    queue = self._priority_queues[priority]
                    
                    if not queue.empty():
                        message, protocol = await queue.get()
                        await self._deliver_message(message, protocol)
                        
                        # Yield control after each message to prevent blocking
                        await asyncio.sleep(0.001)
                
                # Small delay when no messages to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error("Error in message routing task", error=str(e))
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _deliver_message(self, message: StandardUniversalMessage, protocol: ProtocolType):
        """Deliver message using specified protocol."""
        try:
            if protocol == ProtocolType.REDIS_STREAM:
                stream_name = f"agent_messages:{message.to_agent}"
                stream_id = await self.redis_client.send_stream_message(stream_name, message)
                return stream_id is not None
                
            elif protocol == ProtocolType.REDIS_PUBSUB:
                channel = f"agent_events:{message.to_agent}"
                return await self.redis_client.publish_message(channel, message)
                
            else:
                # Use protocol adapter if available
                adapter = self._adapters.get(protocol)
                if adapter:
                    return await adapter.send_message(message)
                else:
                    logger.error("No adapter available for protocol", protocol=protocol)
                    return False
                    
        except Exception as e:
            logger.error("Failed to deliver message", 
                        message_id=message.message_id, protocol=protocol, error=str(e))
            self._performance_metrics["failed_deliveries"] += 1
            return False
    
    async def subscribe_to_agent_messages(
        self,
        agent_id: str,
        callback: Callable[[StandardUniversalMessage], None]
    ):
        """Subscribe to messages for specific agent."""
        subscription_key = f"agent_messages:{agent_id}"
        self._subscribers[subscription_key].append(callback)
        self._performance_metrics["active_subscriptions"] += 1
        
        logger.info("Subscribed to agent messages", agent_id=agent_id)
    
    def _update_routing_latency(self, latency_ms: float):
        """Update routing latency metrics."""
        current_avg = self._performance_metrics["avg_routing_latency_ms"]
        total_routed = self._performance_metrics["total_messages_routed"]
        
        # Calculate running average
        new_avg = ((current_avg * (total_routed - 1)) + latency_ms) / total_routed
        self._performance_metrics["avg_routing_latency_ms"] = new_avg
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        redis_metrics = await self.redis_client.get_performance_metrics()
        
        return {
            **self._performance_metrics,
            "redis_metrics": redis_metrics,
            "queue_sizes": {
                priority.value: queue.qsize() 
                for priority, queue in self._priority_queues.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        redis_health = await self.redis_client.health_check()
        
        return {
            "status": "healthy" if redis_health["status"] == "healthy" else "degraded",
            "components": {
                "redis": redis_health,
                "message_router": {
                    "status": "healthy" if self._running else "stopped",
                    "queued_messages": sum(q.qsize() for q in self._priority_queues.values())
                }
            },
            "performance": await self.get_performance_metrics()
        }
    
    async def shutdown(self):
        """Graceful shutdown of communication manager."""
        self._running = False
        
        if self._routing_task:
            self._routing_task.cancel()
            try:
                await self._routing_task
            except asyncio.CancelledError:
                pass
        
        await self.redis_client.cleanup()
        logger.info("Unified Communication Manager shut down")

# ================================================================================
# Global Communication Manager Instance
# ================================================================================

# Singleton instance for system-wide use
_communication_manager: Optional[UnifiedCommunicationManager] = None

async def get_communication_manager() -> UnifiedCommunicationManager:
    """Get or create the global communication manager instance."""
    global _communication_manager
    
    if _communication_manager is None:
        _communication_manager = UnifiedCommunicationManager()
        await _communication_manager.initialize()
    
    return _communication_manager

# Convenience functions for common operations
async def send_agent_message(
    from_agent: str,
    to_agent: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL
) -> bool:
    """Convenience function to send agent-to-agent message."""
    manager = await get_communication_manager()
    
    message = StandardUniversalMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=message_type,
        priority=priority,
        payload=payload
    )
    
    return await manager.send_message(message)

async def broadcast_system_event(
    event_type: str,
    event_data: Dict[str, Any],
    priority: MessagePriority = MessagePriority.HIGH
) -> bool:
    """Convenience function to broadcast system events."""
    manager = await get_communication_manager()
    
    message = StandardUniversalMessage(
        from_agent="system",
        to_agent="broadcast",
        message_type=MessageType.SYSTEM_EVENT,
        priority=priority,
        payload={
            "event_type": event_type,
            "event_data": event_data
        }
    )
    
    return await manager.send_message(message)