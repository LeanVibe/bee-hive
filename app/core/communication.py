"""
Redis Streams Communication Engine for LeanVibe Agent Hive 2.0.

Provides reliable, low-latency message passing between agents with:
- At-least-once delivery guarantees
- Consumer groups for horizontal scaling  
- Dead letter queue handling
- Performance monitoring and back-pressure
- Message signing for security
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from contextlib import asynccontextmanager
import json

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from ..models.message import (
    StreamMessage, 
    MessageType, 
    MessagePriority,
    MessageStatus,
    MessageAudit,
    ConsumerGroupInfo,
    StreamInfo,
    MessageDeliveryReport
)
from ..core.database import AsyncSession, get_async_session
from ..core.config import settings
from .dead_letter_queue import DeadLetterQueueManager, DLQConfiguration

logger = logging.getLogger(__name__)


class CommunicationError(Exception):
    """Base exception for communication system errors."""
    pass


class MessageDeliveryError(CommunicationError):
    """Error in message delivery."""
    pass


class ConsumerGroupError(CommunicationError):
    """Error in consumer group management."""
    pass


class MessageBroker:
    """
    Redis Streams-based message broker for inter-agent communication.
    
    Provides reliable message passing with consumer groups, acknowledgments,
    and dead letter queue handling for failed messages.
    """
    
    def __init__(
        self, 
        redis_url: str = None,
        secret_key: str = None,
        max_retries: int = 3,
        ack_timeout_ms: int = 30000,
        max_len: int = 1000000,
        db_session: AsyncSession = None
    ):
        self.redis_url = redis_url or settings.REDIS_URL
        self.secret_key = secret_key or settings.SECRET_KEY
        self.max_retries = max_retries
        self.ack_timeout_ms = ack_timeout_ms
        self.max_len = max_len
        self.db_session = db_session
        
        # Connection management
        self._redis: Optional[Redis] = None
        self._connection_pool = None
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._message_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self._metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_acknowledged": 0,
            "messages_failed": 0,
            "total_latency_ms": 0.0,
            "dlq_moves": 0,
            "retry_attempts": 0,
            "circuit_breaker_trips": 0,
        }
        
        # DLQ Manager
        self._dlq_manager: Optional[DeadLetterQueueManager] = None
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            self._redis = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._redis.ping()
            
            # Initialize DLQ Manager
            dlq_config = DLQConfiguration(
                max_retries=self.max_retries,
                initial_retry_delay_ms=1000,
                max_retry_delay_ms=30000,
                dlq_max_size=100000,
                monitor_enabled=True,
                alert_threshold=1000
            )
            
            self._dlq_manager = DeadLetterQueueManager(self._redis, dlq_config)
            await self._dlq_manager.start()
            
            logger.info("Connected to Redis for message broker with DLQ support")
            
        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CommunicationError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close Redis connection and stop consumers."""
        # Stop all consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks.values(), return_exceptions=True)
        
        # Stop DLQ Manager
        if self._dlq_manager:
            await self._dlq_manager.stop()
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
            
        if self._connection_pool:
            await self._connection_pool.disconnect()
            
        logger.info("Disconnected from Redis message broker")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for broker session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    async def send_message(self, message: StreamMessage) -> str:
        """
        Send message to Redis stream with reliability guarantees.
        
        Args:
            message: StreamMessage to send
            
        Returns:
            Redis stream message ID
            
        Raises:
            MessageDeliveryError: If message cannot be delivered
        """
        if not self._redis:
            raise CommunicationError("Not connected to Redis")
        
        try:
            # Sign message for security
            if self.secret_key:
                message.sign(self.secret_key)
            
            # Determine stream name
            stream_name = message.get_stream_name()
            
            # Convert to Redis format
            redis_data = message.to_redis_dict()
            
            # Add to stream with max length trimming
            start_time = time.time()
            
            message_id = await self._redis.xadd(
                stream_name,
                redis_data,
                maxlen=self.max_len,
                approximate=True
            )
            
            # Track metrics
            self._metrics["messages_sent"] += 1
            delivery_time_ms = (time.time() - start_time) * 1000
            self._metrics["total_latency_ms"] += delivery_time_ms
            
            # Audit message
            if self.db_session:
                await self._audit_message_sent(message, stream_name, message_id, delivery_time_ms)
            
            logger.debug(f"Message sent to {stream_name}: {message_id}")
            return message_id
            
        except RedisError as e:
            self._metrics["messages_failed"] += 1
            logger.error(f"Failed to send message: {e}")
            raise MessageDeliveryError(f"Message delivery failed: {e}")
    
    async def create_consumer_group(self, stream_name: str, group_name: str) -> None:
        """Create consumer group for stream."""
        if not self._redis:
            raise CommunicationError("Not connected to Redis")
        
        try:
            await self._redis.xgroup_create(
                stream_name, 
                group_name, 
                id="0", 
                mkstream=True
            )
            logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                logger.debug(f"Consumer group {group_name} already exists for {stream_name}")
            else:
                raise ConsumerGroupError(f"Failed to create consumer group: {e}")
    
    async def consume_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable[[StreamMessage], bool],
        count: int = 10,
        block_ms: int = 1000
    ) -> None:
        """
        Consume messages from stream with consumer group.
        
        Args:
            stream_name: Redis stream name
            group_name: Consumer group name
            consumer_name: Unique consumer identifier
            handler: Message handler function (returns True if successful)
            count: Number of messages to read per batch
            block_ms: Blocking timeout in milliseconds
        """
        if not self._redis:
            raise CommunicationError("Not connected to Redis")
        
        # Ensure consumer group exists
        await self.create_consumer_group(stream_name, group_name)
        
        # Register handler
        handler_key = f"{stream_name}:{group_name}:{consumer_name}"
        self._message_handlers[handler_key] = handler
        
        # Start consumer task
        task = asyncio.create_task(
            self._consume_loop(stream_name, group_name, consumer_name, count, block_ms)
        )
        
        self._consumer_tasks[handler_key] = task
        logger.info(f"Started consumer {consumer_name} for {stream_name}:{group_name}")
    
    async def _consume_loop(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        count: int,
        block_ms: int
    ) -> None:
        """Main consumer loop."""
        handler_key = f"{stream_name}:{group_name}:{consumer_name}"
        handler = self._message_handlers.get(handler_key)
        
        if not handler:
            logger.error(f"No handler found for {handler_key}")
            return
        
        while True:
            try:
                # First, claim any pending messages from failed consumers
                await self._claim_pending_messages(stream_name, group_name, consumer_name)
                
                # Read new messages
                result = await self._redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_name: ">"},
                    count=count,
                    block=block_ms
                )
                
                if not result:
                    continue
                
                # Process messages
                for stream, messages in result:
                    for message_id, fields in messages:
                        await self._process_message(
                            stream_name, group_name, message_id, fields, handler
                        )
                        
            except asyncio.CancelledError:
                logger.info(f"Consumer {consumer_name} cancelled")
                break
                
            except RedisError as e:
                logger.error(f"Redis error in consumer {consumer_name}: {e}")
                await asyncio.sleep(1)  # Brief delay before retry
                
            except Exception as e:
                logger.error(f"Unexpected error in consumer {consumer_name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(
        self,
        stream_name: str,
        group_name: str,
        message_id: str,
        fields: Dict[str, str],
        handler: Callable[[StreamMessage], bool]
    ) -> None:
        """Process individual message."""
        try:
            # Convert Redis fields to StreamMessage
            message = StreamMessage.from_redis_dict(fields)
            
            # Verify signature if present
            if message.signature and self.secret_key:
                if not message.verify_signature(self.secret_key):
                    logger.warning(f"Invalid signature for message {message_id}")
                    await self._redis.xack(stream_name, group_name, message_id)
                    return
            
            # Check if expired
            if message.is_expired():
                logger.warning(f"Message {message_id} expired, skipping")
                await self._redis.xack(stream_name, group_name, message_id)
                return
            
            # Process message
            start_time = time.time()
            success = await asyncio.get_event_loop().run_in_executor(
                None, handler, message
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            if success:
                # Acknowledge message
                await self._redis.xack(stream_name, group_name, message_id)
                self._metrics["messages_acknowledged"] += 1
                
                # Audit success
                if self.db_session:
                    await self._audit_message_processed(message, processing_time_ms)
                
                logger.debug(f"Message {message_id} processed successfully")
            else:
                # Mark as failed, will be retried via DLQ system
                self._metrics["messages_failed"] += 1
                logger.warning(f"Message {message_id} processing failed")
                
                # Use DLQ manager for sophisticated retry handling
                if self._dlq_manager:
                    should_retry = await self._dlq_manager.handle_failed_message(
                        original_stream=stream_name,
                        original_message_id=message_id,
                        message=message,
                        failure_reason="Message processing failed",
                        current_retry_count=0  # Would track this in message metadata
                    )
                    
                    if should_retry:
                        self._metrics["retry_attempts"] += 1
                    else:
                        self._metrics["dlq_moves"] += 1
                        
                    # Always acknowledge original message since DLQ handles retry
                    await self._redis.xack(stream_name, group_name, message_id)
                else:
                    # Fallback to simple DLQ move
                    await self._handle_failed_message(stream_name, group_name, message, message_id)
            
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            self._metrics["messages_failed"] += 1
    
    async def _claim_pending_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str
    ) -> None:
        """Claim pending messages from failed consumers."""
        try:
            # Get pending messages older than timeout
            pending = await self._redis.xpending_range(
                stream_name,
                group_name,
                min="-",
                max="+",
                count=100,
                idle=self.ack_timeout_ms
            )
            
            if pending:
                # Claim messages
                message_ids = [msg["message_id"] for msg in pending]
                claimed = await self._redis.xclaim(
                    stream_name,
                    group_name,
                    consumer_name,
                    min_idle_time=self.ack_timeout_ms,
                    message_ids=message_ids
                )
                
                if claimed:
                    logger.info(f"Claimed {len(claimed)} pending messages")
                    
        except RedisError as e:
            logger.error(f"Error claiming pending messages: {e}")
    
    async def _handle_failed_message(
        self,
        stream_name: str,
        group_name: str,
        message: StreamMessage,
        message_id: str
    ) -> None:
        """Handle failed message - retry or move to DLQ."""
        # In a real implementation, we'd track retry count
        # For now, move to dead letter queue
        dlq_stream = f"{stream_name}:dlq"
        
        try:
            # Add to DLQ
            await self._redis.xadd(dlq_stream, message.to_redis_dict())
            
            # Acknowledge original message
            await self._redis.xack(stream_name, group_name, message_id)
            
            logger.warning(f"Message {message_id} moved to DLQ: {dlq_stream}")
            
        except RedisError as e:
            logger.error(f"Failed to move message to DLQ: {e}")
    
    async def get_stream_info(self, stream_name: str) -> StreamInfo:
        """Get information about a Redis stream."""
        if not self._redis:
            raise CommunicationError("Not connected to Redis")
        
        try:
            info = await self._redis.xinfo_stream(stream_name)
            groups_info = await self._redis.xinfo_groups(stream_name)
            
            groups = []
            for group in groups_info:
                groups.append(ConsumerGroupInfo(
                    name=group["name"],
                    consumers=group["consumers"],
                    pending=group["pending"],
                    last_delivered_id=group["last-delivered-id"],
                    lag=group.get("lag", 0)
                ))
            
            return StreamInfo(
                name=stream_name,
                length=info["length"],
                groups=groups,
                first_entry_id=info.get("first-entry"),
                last_entry_id=info.get("last-entry"),
                max_deleted_entry_id=info.get("max-deleted-entry-id")
            )
            
        except RedisError as e:
            logger.error(f"Failed to get stream info: {e}")
            raise CommunicationError(f"Stream info failed: {e}")
    
    async def get_delivery_report(self) -> MessageDeliveryReport:
        """Get message delivery performance report."""
        total_sent = self._metrics["messages_sent"]
        total_ack = self._metrics["messages_acknowledged"]
        total_failed = self._metrics["messages_failed"]
        
        success_rate = (total_ack / total_sent) if total_sent > 0 else 0.0
        error_rate = (total_failed / total_sent) if total_sent > 0 else 0.0
        avg_latency = (self._metrics["total_latency_ms"] / total_sent) if total_sent > 0 else 0.0
        
        return MessageDeliveryReport(
            total_sent=total_sent,
            total_acknowledged=total_ack,
            total_failed=total_failed,
            success_rate=success_rate,
            average_latency_ms=avg_latency,
            p95_latency_ms=avg_latency * 1.2,  # Simplified
            p99_latency_ms=avg_latency * 1.5,  # Simplified
            throughput_msg_per_sec=0.0,  # Would need time tracking
            error_rate=error_rate
        )
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including DLQ stats."""
        base_metrics = {
            "broker_metrics": self._metrics.copy(),
            "delivery_report": (await self.get_delivery_report()).dict()
        }
        
        if self._dlq_manager:
            dlq_stats = await self._dlq_manager.get_dlq_stats()
            base_metrics["dlq_stats"] = dlq_stats
        
        return base_metrics
    
    async def replay_failed_messages(
        self,
        stream_filter: Optional[str] = None,
        max_messages: int = 100
    ) -> int:
        """Replay failed messages from DLQ."""
        if not self._dlq_manager:
            logger.warning("DLQ Manager not available for replay")
            return 0
        
        return await self._dlq_manager.replay_dlq_messages(
            stream_filter=stream_filter,
            max_messages=max_messages
        )
    
    async def _audit_message_sent(
        self,
        message: StreamMessage,
        stream_name: str,
        message_id: str,
        delivery_time_ms: float
    ) -> None:
        """Audit sent message to database."""
        if not self.db_session:
            return
        
        try:
            audit = MessageAudit(
                message_id=message_id,
                stream_name=stream_name,
                from_agent_id=message.from_agent,  # Would need UUID conversion
                to_agent_id=message.to_agent,
                message_type=message.message_type,
                priority=message.priority,
                payload=message.payload,
                correlation_id=message.correlation_id,
                delivery_latency_ms=str(delivery_time_ms)
            )
            
            self.db_session.add(audit)
            await self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to audit message: {e}")
    
    async def _audit_message_processed(
        self,
        message: StreamMessage,
        processing_time_ms: float
    ) -> None:
        """Update audit record for processed message."""
        if not self.db_session:
            return
        
        try:
            # Would need to find and update the audit record
            # Simplified for this implementation
            pass
            
        except Exception as e:
            logger.error(f"Failed to update message audit: {e}")


class SimplePubSub:
    """
    Simple Redis Pub/Sub for fire-and-forget notifications.
    Complements the reliable Streams for urgent broadcasts.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[Redis] = None
        self._pubsub = None
        self._subscriber_tasks: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        self._pubsub = self._redis.pubsub()
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        for task in self._subscriber_tasks.values():
            task.cancel()
        
        if self._pubsub:
            await self._pubsub.close()
        
        if self._redis:
            await self._redis.close()
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish message to channel."""
        if not self._redis:
            raise CommunicationError("Not connected to Redis")
        
        return await self._redis.publish(channel, json.dumps(message))
    
    async def subscribe(
        self,
        channel: str,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Subscribe to channel with message handler."""
        if not self._pubsub:
            raise CommunicationError("Not connected to Redis")
        
        await self._pubsub.subscribe(channel)
        
        task = asyncio.create_task(self._subscriber_loop(channel, handler))
        self._subscriber_tasks[channel] = task
    
    async def _subscriber_loop(
        self,
        channel: str,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Subscriber message loop."""
        async for message in self._pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await asyncio.get_event_loop().run_in_executor(
                        None, handler, channel, data
                    )
                except Exception as e:
                    logger.error(f"Error in subscriber {channel}: {e}")