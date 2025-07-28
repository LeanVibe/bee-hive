"""
Redis Pub/Sub Manager for LeanVibe Agent Hive 2.0

Provides Redis Streams and Pub/Sub integration for reliable, high-performance
agent communication with consumer groups, dead letter queues, and comprehensive monitoring.

Implements the Communication PRD requirements for >99.9% delivery and <200ms latency.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError, ResponseError

from ..models.message import StreamMessage, MessageType, MessagePriority, MessageStatus
from .config import settings

logger = logging.getLogger(__name__)


class StreamOperationError(Exception):
    """Error in Redis Stream operations."""
    pass


class ConsumerGroupError(Exception):
    """Error in consumer group management."""
    pass


@dataclass
class ConsumerInfo:
    """Information about a consumer in a consumer group."""
    name: str
    pending_count: int
    idle_time_ms: int
    last_delivery_id: str


@dataclass
class ConsumerGroupStats:
    """Statistics for a consumer group."""
    name: str
    consumer_count: int
    pending_count: int
    last_delivered_id: str
    consumers: List[ConsumerInfo]
    lag: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'consumers': [asdict(c) for c in self.consumers]
        }


@dataclass
class StreamStats:
    """Statistics for a Redis Stream."""
    name: str
    length: int
    groups: List[ConsumerGroupStats]
    first_entry_id: Optional[str]
    last_entry_id: Optional[str]
    radix_tree_keys: int
    radix_tree_nodes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'groups': [g.to_dict() for g in self.groups]
        }


@dataclass
class MessageProcessingResult:
    """Result of message processing operation."""
    success: bool
    message_id: str
    processing_time_ms: float
    error: Optional[str] = None
    retryable: bool = True


class RedisPubSubManager:
    """
    Advanced Redis Pub/Sub and Streams manager for agent communication.
    
    Features:
    - Redis Streams for durable messaging with consumer groups
    - Pub/Sub for fast notifications
    - Dead Letter Queue for failed messages
    - Automatic message claiming for stalled consumers
    - Comprehensive performance monitoring
    - Connection resilience and failover
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        connection_pool_size: int = 20,
        consumer_name: Optional[str] = None,
        max_claim_idle_time_ms: int = 30000,  # 30 seconds
        max_retries: int = 3,
        dead_letter_stream_suffix: str = ":dlq",
        stream_maxlen: int = 1000000,  # 1M messages
        batch_size: int = 10,
        claim_batch_size: int = 100
    ):
        """
        Initialize Redis Pub/Sub manager.
        
        Args:
            redis_url: Redis connection URL
            connection_pool_size: Size of connection pool
            consumer_name: Unique consumer identifier
            max_claim_idle_time_ms: Max idle time before claiming stalled messages
            max_retries: Maximum retry attempts before DLQ
            dead_letter_stream_suffix: Suffix for dead letter streams  
            stream_maxlen: Maximum stream length for trimming
            batch_size: Batch size for reading messages
            claim_batch_size: Batch size for claiming messages
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.connection_pool_size = connection_pool_size
        self.consumer_name = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        self.max_claim_idle_time_ms = max_claim_idle_time_ms
        self.max_retries = max_retries
        self.dead_letter_stream_suffix = dead_letter_stream_suffix
        self.stream_maxlen = stream_maxlen
        self.batch_size = batch_size
        self.claim_batch_size = claim_batch_size
        
        # Connection management
        self._redis: Optional[Redis] = None
        self._connection_pool = None
        self._connected = False
        
        # Consumer management
        self._active_consumers: Dict[str, asyncio.Task] = {}
        self._consumer_groups: Set[str] = set()
        self._message_handlers: Dict[str, Callable[[StreamMessage], Any]] = {}
        
        # Performance tracking
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'messages_claimed': 0,
            'messages_dlq': 0,
            'total_latency_ms': 0.0,
            'start_time': time.time()
        }
        self._latency_samples: deque = deque(maxlen=1000)
        
        # Circuit breaker for resilience
        self._circuit_breaker = {
            'failure_count': 0,
            'last_failure_time': 0,
            'circuit_open': False,
            'failure_threshold': 5,
            'recovery_timeout': 60  # seconds
        }
        
    async def connect(self) -> None:
        """Establish Redis connection with resilience."""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=self.connection_pool_size,
                retry_on_timeout=True,
                retry_on_error=[RedisConnectionError, ResponseError],
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            self._redis = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            self._circuit_breaker['circuit_open'] = False
            self._circuit_breaker['failure_count'] = 0
            
            logger.info(
                "Connected to Redis for Pub/Sub communication",
                extra={
                    "redis_url": self.redis_url,
                    "consumer_name": self.consumer_name,
                    "pool_size": self.connection_pool_size
                }
            )
            
        except RedisError as e:
            self._handle_connection_error(e)
            raise StreamOperationError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources."""
        # Stop all consumer tasks
        for task in self._active_consumers.values():
            if not task.done():
                task.cancel()
        
        if self._active_consumers:
            await asyncio.gather(*self._active_consumers.values(), return_exceptions=True)
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        self._connected = False
        logger.info("Disconnected from Redis Pub/Sub manager")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for Pub/Sub session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    def _ensure_connected(self) -> None:
        """Ensure Redis connection is active."""
        if not self._connected or not self._redis:
            raise StreamOperationError("Not connected to Redis. Call connect() first.")
        
        # Check circuit breaker
        if self._circuit_breaker['circuit_open']:
            if time.time() - self._circuit_breaker['last_failure_time'] > self._circuit_breaker['recovery_timeout']:
                self._circuit_breaker['circuit_open'] = False
                self._circuit_breaker['failure_count'] = 0
                logger.info("Circuit breaker recovered, reconnecting")
            else:
                raise StreamOperationError("Circuit breaker is open, connection unavailable")
    
    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors with circuit breaker pattern."""
        self._circuit_breaker['failure_count'] += 1
        self._circuit_breaker['last_failure_time'] = time.time()
        
        if self._circuit_breaker['failure_count'] >= self._circuit_breaker['failure_threshold']:
            self._circuit_breaker['circuit_open'] = True
            logger.error(f"Circuit breaker opened due to repeated failures: {error}")
        
        self._stats['messages_failed'] += 1
    
    async def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        consumer_id: str = "$",
        mkstream: bool = True
    ) -> None:
        """
        Create a consumer group for a stream.
        
        Args:
            stream_name: Name of the Redis stream
            group_name: Name of the consumer group
            consumer_id: Starting consumer ID ($ for latest)
            mkstream: Create stream if it doesn't exist
        """
        self._ensure_connected()
        
        try:
            await self._redis.xgroup_create(
                stream_name,
                group_name,
                id=consumer_id,
                mkstream=mkstream
            )
            
            self._consumer_groups.add(f"{stream_name}:{group_name}")
            
            logger.info(
                f"Created consumer group {group_name} for stream {stream_name}",
                extra={
                    "stream_name": stream_name,
                    "group_name": group_name,
                    "consumer_id": consumer_id
                }
            )
            
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists, that's OK
                self._consumer_groups.add(f"{stream_name}:{group_name}")
                logger.debug(f"Consumer group {group_name} already exists for {stream_name}")
            else:
                logger.error(f"Failed to create consumer group: {e}")
                raise ConsumerGroupError(f"Failed to create consumer group: {e}")
        except RedisError as e:
            self._handle_connection_error(e)
            raise StreamOperationError(f"Redis error creating consumer group: {e}")
    
    async def send_stream_message(
        self,
        stream_name: str,
        message: StreamMessage,
        maxlen: Optional[int] = None
    ) -> str:
        """
        Send message to Redis Stream.
        
        Args:
            stream_name: Target stream name
            message: StreamMessage to send
            maxlen: Maximum stream length (None uses default)
            
        Returns:
            Message ID assigned by Redis
        """
        self._ensure_connected()
        
        start_time = time.time()
        
        try:
            # Convert message to Redis format
            redis_data = message.to_redis_dict()
            
            # Send to stream with optional length limiting
            message_id = await self._redis.xadd(
                stream_name,
                redis_data,
                maxlen=maxlen or self.stream_maxlen,
                approximate=True  # Use ~ for better performance
            )
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self._update_send_metrics(latency_ms, True)
            
            logger.debug(
                f"Sent message to stream {stream_name}",
                extra={
                    "stream_name": stream_name,
                    "message_id": message_id,
                    "original_message_id": message.id,
                    "latency_ms": latency_ms
                }
            )
            
            return message_id
            
        except RedisError as e:
            self._handle_connection_error(e)
            self._update_send_metrics((time.time() - start_time) * 1000, False)
            logger.error(f"Failed to send message to stream {stream_name}: {e}")
            raise StreamOperationError(f"Failed to send stream message: {e}")
    
    async def consume_stream_messages(
        self,
        stream_name: str,
        group_name: str,
        handler: Callable[[StreamMessage], Any],
        auto_ack: bool = True,
        claim_stalled: bool = True
    ) -> None:
        """
        Start consuming messages from a stream with consumer group.
        
        Args:
            stream_name: Stream to consume from
            group_name: Consumer group name
            handler: Function to handle received messages
            auto_ack: Automatically acknowledge processed messages
            claim_stalled: Automatically claim stalled messages
        """
        self._ensure_connected()
        
        # Ensure consumer group exists
        await self.create_consumer_group(stream_name, group_name)
        
        # Register handler
        consumer_key = f"{stream_name}:{group_name}"
        self._message_handlers[consumer_key] = handler
        
        # Start consumer task
        if consumer_key not in self._active_consumers:
            task = asyncio.create_task(
                self._consumer_loop(
                    stream_name,
                    group_name,
                    handler,
                    auto_ack,
                    claim_stalled
                )
            )
            self._active_consumers[consumer_key] = task
            
            logger.info(
                f"Started consumer for stream {stream_name} group {group_name}",
                extra={
                    "stream_name": stream_name,
                    "group_name": group_name,
                    "consumer_name": self.consumer_name,
                    "auto_ack": auto_ack,
                    "claim_stalled": claim_stalled
                }
            )
    
    async def _consumer_loop(
        self,
        stream_name: str,
        group_name: str,
        handler: Callable[[StreamMessage], Any],
        auto_ack: bool,
        claim_stalled: bool
    ) -> None:
        """Main consumer loop for processing stream messages."""
        try:
            while True:
                try:
                    # Claim stalled messages first if enabled
                    if claim_stalled:
                        await self._claim_stalled_messages(stream_name, group_name, handler, auto_ack)
                    
                    # Read new messages
                    messages = await self._redis.xreadgroup(
                        group_name,
                        self.consumer_name,
                        {stream_name: ">"},
                        count=self.batch_size,
                        block=1000  # 1 second timeout
                    )
                    
                    # Process messages
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            await self._process_stream_message(
                                stream,
                                message_id,
                                fields,
                                group_name,
                                handler,
                                auto_ack
                            )
                
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                    
                except asyncio.CancelledError:
                    logger.info(f"Consumer loop cancelled for {stream_name}:{group_name}")
                    break
                    
                except RedisError as e:
                    self._handle_connection_error(e)
                    logger.error(f"Redis error in consumer loop: {e}")
                    await asyncio.sleep(5)  # Brief delay before retry
                    
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in consumer loop for {stream_name}:{group_name}: {e}")
    
    async def _claim_stalled_messages(
        self,
        stream_name: str,
        group_name: str,
        handler: Callable[[StreamMessage], Any],
        auto_ack: bool
    ) -> None:
        """Claim and process stalled messages from other consumers."""
        try:
            # Get pending messages info
            pending_info = await self._redis.xpending_range(
                stream_name,
                group_name,
                "-",
                "+",
                count=self.claim_batch_size
            )
            
            if not pending_info:
                return
            
            # Find messages that are idle too long
            stalled_message_ids = []
            for message_data in pending_info:
                message_id, consumer, idle_time_ms, delivery_count = message_data
                
                if idle_time_ms >= self.max_claim_idle_time_ms:
                    stalled_message_ids.append(message_id)
            
            if not stalled_message_ids:
                return
            
            # Claim stalled messages
            claimed_messages = await self._redis.xclaim(
                stream_name,
                group_name,
                self.consumer_name,
                min_idle_time=self.max_claim_idle_time_ms,
                message_ids=stalled_message_ids
            )
            
            # Process claimed messages
            for message_id, fields in claimed_messages:
                await self._process_stream_message(
                    stream_name,
                    message_id,
                    fields,
                    group_name,
                    handler,
                    auto_ack,
                    claimed=True
                )
                
            if claimed_messages:
                self._stats['messages_claimed'] += len(claimed_messages)
                logger.info(f"Claimed {len(claimed_messages)} stalled messages from {stream_name}")
                
        except RedisError as e:
            logger.error(f"Error claiming stalled messages: {e}")
    
    async def _process_stream_message(
        self,
        stream_name: str,
        message_id: str,
        fields: Dict[str, str],
        group_name: str,
        handler: Callable[[StreamMessage], Any],
        auto_ack: bool,
        claimed: bool = False
    ) -> None:
        """Process a single stream message."""
        start_time = time.time()
        
        try:
            # Convert Redis fields to StreamMessage
            message = StreamMessage.from_redis_dict(fields)
            
            # Check if message is expired
            if message.is_expired():
                logger.warning(f"Received expired message {message_id}")
                if auto_ack:
                    await self._redis.xack(stream_name, group_name, message_id)
                return
            
            # Execute handler
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, handler, message
                )
                
                # Acknowledge message if successful and auto_ack enabled
                if auto_ack:
                    await self._redis.xack(stream_name, group_name, message_id)
                
                # Track performance
                processing_time_ms = (time.time() - start_time) * 1000
                self._update_receive_metrics(processing_time_ms, True)
                
                logger.debug(
                    f"Processed message {message_id} from {stream_name}",
                    extra={
                        "message_id": message_id,
                        "stream_name": stream_name,
                        "processing_time_ms": processing_time_ms,
                        "claimed": claimed
                    }
                )
                
            except Exception as handler_error:
                logger.error(f"Handler error for message {message_id}: {handler_error}")
                await self._handle_message_failure(
                    stream_name, group_name, message_id, message, handler_error
                )
                
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            self._update_receive_metrics((time.time() - start_time) * 1000, False)
    
    async def _handle_message_failure(
        self,
        stream_name: str,
        group_name: str,
        message_id: str,
        message: StreamMessage,
        error: Exception
    ) -> None:
        """Handle failed message processing."""
        try:
            # Get delivery count for this message
            pending_info = await self._redis.xpending_range(
                stream_name,
                group_name,
                message_id,
                message_id,
                count=1
            )
            
            delivery_count = 1
            if pending_info:
                _, _, _, delivery_count = pending_info[0]
            
            if delivery_count >= self.max_retries:
                # Move to dead letter queue
                await self._send_to_dead_letter_queue(stream_name, message, error)
                
                # Acknowledge the failed message
                await self._redis.xack(stream_name, group_name, message_id)
                
                self._stats['messages_dlq'] += 1
                logger.warning(
                    f"Message {message_id} moved to DLQ after {delivery_count} attempts",
                    extra={
                        "message_id": message_id,
                        "stream_name": stream_name,
                        "delivery_count": delivery_count,
                        "error": str(error)
                    }
                )
            else:
                # Leave message unacknowledged for retry
                logger.info(f"Message {message_id} will be retried (attempt {delivery_count})")
                
        except Exception as dlq_error:
            logger.error(f"Error handling message failure: {dlq_error}")
    
    async def _send_to_dead_letter_queue(
        self,
        original_stream: str,
        message: StreamMessage,
        error: Exception
    ) -> None:
        """Send failed message to dead letter queue."""
        dlq_stream = f"{original_stream}{self.dead_letter_stream_suffix}"
        
        try:
            # Add error information to message
            dlq_message_data = message.to_redis_dict()
            dlq_message_data.update({
                "dlq_timestamp": str(time.time()),
                "dlq_error": str(error),
                "dlq_original_stream": original_stream
            })
            
            await self._redis.xadd(dlq_stream, dlq_message_data)
            
            logger.info(f"Message sent to DLQ: {dlq_stream}")
            
        except RedisError as e:
            logger.error(f"Failed to send message to DLQ: {e}")
    
    def _update_send_metrics(self, latency_ms: float, success: bool) -> None:
        """Update metrics for sent messages."""
        if success:
            self._stats['messages_sent'] += 1
        else:
            self._stats['messages_failed'] += 1
        
        self._latency_samples.append(latency_ms)
        self._stats['total_latency_ms'] += latency_ms
    
    def _update_receive_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update metrics for received messages."""
        if success:
            self._stats['messages_received'] += 1
        else:
            self._stats['messages_failed'] += 1
        
        self._latency_samples.append(processing_time_ms)
        self._stats['total_latency_ms'] += processing_time_ms
    
    async def get_stream_stats(self, stream_name: str) -> StreamStats:
        """Get comprehensive statistics for a stream."""
        self._ensure_connected()
        
        try:
            # Get stream info
            stream_info = await self._redis.xinfo_stream(stream_name)
            
            # Get groups info
            groups_info = await self._redis.xinfo_groups(stream_name)
            
            groups = []
            for group_info in groups_info:
                # Get consumers for this group
                consumers_info = await self._redis.xinfo_consumers(
                    stream_name, group_info['name']
                )
                
                consumers = [
                    ConsumerInfo(
                        name=c['name'],
                        pending_count=c['pending'],
                        idle_time_ms=c['idle'],
                        last_delivery_id=c.get('last-delivery-time', '0-0')
                    )
                    for c in consumers_info
                ]
                
                groups.append(ConsumerGroupStats(
                    name=group_info['name'],
                    consumer_count=group_info['consumers'],
                    pending_count=group_info['pending'],
                    last_delivered_id=group_info['last-delivered-id'],
                    consumers=consumers,
                    lag=group_info.get('lag', 0)
                ))
            
            return StreamStats(
                name=stream_name,
                length=stream_info['length'],
                groups=groups,
                first_entry_id=stream_info.get('first-entry'),
                last_entry_id=stream_info.get('last-entry'),
                radix_tree_keys=stream_info.get('radix-tree-keys', 0),
                radix_tree_nodes=stream_info.get('radix-tree-nodes', 0)
            )
            
        except RedisError as e:
            raise StreamOperationError(f"Failed to get stream stats: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        uptime = time.time() - self._stats['start_time']
        
        # Calculate latency percentiles
        latency_list = list(self._latency_samples)
        latency_list.sort()
        
        p95_latency = 0.0
        p99_latency = 0.0
        avg_latency = 0.0
        
        if latency_list:
            n = len(latency_list)
            avg_latency = sum(latency_list) / n
            p95_latency = latency_list[int(n * 0.95)] if n > 0 else 0
            p99_latency = latency_list[int(n * 0.99)] if n > 0 else 0
        
        return {
            "messages_sent": self._stats['messages_sent'],
            "messages_received": self._stats['messages_received'],
            "messages_failed": self._stats['messages_failed'],
            "messages_claimed": self._stats['messages_claimed'],
            "messages_dlq": self._stats['messages_dlq'],
            "uptime_seconds": uptime,
            "throughput_msg_per_sec": (
                (self._stats['messages_sent'] + self._stats['messages_received']) / uptime
                if uptime > 0 else 0
            ),
            "success_rate": (
                self._stats['messages_sent'] / 
                (self._stats['messages_sent'] + self._stats['messages_failed'])
                if (self._stats['messages_sent'] + self._stats['messages_failed']) > 0 else 1.0
            ),
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "active_consumers": len(self._active_consumers),
            "consumer_groups": len(self._consumer_groups),
            "circuit_breaker_status": {
                "open": self._circuit_breaker['circuit_open'],
                "failure_count": self._circuit_breaker['failure_count'],
                "last_failure_time": self._circuit_breaker['last_failure_time']
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            self._ensure_connected()
            
            # Test Redis connection
            start_time = time.time()
            await self._redis.ping()
            ping_latency_ms = (time.time() - start_time) * 1000
            
            metrics = await self.get_performance_metrics()
            
            # Determine health status
            is_healthy = (
                self._connected and
                not self._circuit_breaker['circuit_open'] and
                ping_latency_ms < 1000 and  # < 1 second
                metrics['success_rate'] >= 0.999  # 99.9% success rate
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "connected": self._connected,
                "ping_latency_ms": ping_latency_ms,
                "circuit_breaker_open": self._circuit_breaker['circuit_open'],
                "performance_metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def stop_consumer(self, stream_name: str, group_name: str) -> None:
        """Stop a specific consumer."""
        consumer_key = f"{stream_name}:{group_name}"
        
        if consumer_key in self._active_consumers:
            task = self._active_consumers[consumer_key]
            if not task.done():
                task.cancel()
                
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._active_consumers[consumer_key]
            if consumer_key in self._message_handlers:
                del self._message_handlers[consumer_key]
            
            logger.info(f"Stopped consumer for {stream_name}:{group_name}")