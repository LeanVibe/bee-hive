"""
Unified Redis Adapter for CommunicationHub

This adapter consolidates all Redis communication patterns including:
- Redis Pub/Sub for real-time messaging
- Redis Streams for reliable message queuing
- Connection pooling and management
- Consumer groups and dead letter queues
- Performance optimization and monitoring

Replaces 8+ separate Redis implementations with a single, unified adapter.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass

import redis.asyncio as aioredis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError, ResponseError

from .base_adapter import BaseProtocolAdapter, AdapterStatus, HealthStatus, MessageHandler
from ..protocols import (
    UnifiedMessage, MessageResult, SubscriptionResult, 
    ProtocolType, ConnectionConfig, Priority, MessageStatus
)


@dataclass
class RedisStreamConfig:
    """Configuration for Redis Streams."""
    stream_prefix: str = "messages"
    consumer_group_prefix: str = "group"
    consumer_name_prefix: str = "consumer"
    max_length: int = 10000
    block_timeout_ms: int = 1000
    claim_min_idle_time: int = 60000  # 1 minute
    max_pending_messages: int = 100


@dataclass
class RedisPubSubConfig:
    """Configuration for Redis Pub/Sub."""
    channel_prefix: str = "channel"
    pattern_prefix: str = "pattern"
    max_subscriptions: int = 1000


class RedisAdapter(BaseProtocolAdapter):
    """
    Unified Redis adapter supporting both Pub/Sub and Streams.
    
    Features:
    - Intelligent routing between Pub/Sub and Streams based on delivery guarantees
    - Consumer group management for reliable message processing
    - Dead letter queue handling for failed messages
    - Connection pooling and health monitoring
    - Performance optimization with batching and compression
    """
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        
        # Redis-specific configuration
        self.stream_config = RedisStreamConfig()
        self.pubsub_config = RedisPubSubConfig()
        
        # Connection management
        self.redis_pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[Redis] = None
        self.pubsub_client: Optional[aioredis.client.PubSub] = None
        
        # Stream management
        self.consumer_groups: Dict[str, Set[str]] = {}  # stream -> consumer_groups
        self.active_consumers: Dict[str, str] = {}  # consumer_id -> stream
        self.stream_subscriptions: Dict[str, MessageHandler] = {}
        
        # Pub/Sub management
        self.pubsub_subscriptions: Dict[str, MessageHandler] = {}
        self.pubsub_patterns: Dict[str, MessageHandler] = {}
        
        # Performance optimization
        self.message_batch: List[UnifiedMessage] = []
        self.batch_size = 10
        self.batch_timeout = 100  # milliseconds
        self._batch_timer: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """Establish Redis connection with pooling."""
        try:
            # Create connection pool
            pool_kwargs = {
                "host": self.config.host,
                "port": self.config.port,
                "max_connections": self.config.pool_size,
                "retry_on_timeout": True,
                "socket_keepalive": True,
                "socket_keepalive_options": {},
                **self.config.connection_params
            }
            
            self.redis_pool = ConnectionPool(**pool_kwargs)
            self.redis_client = Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize pub/sub client
            self.pubsub_client = self.redis_client.pubsub()
            
            # Start message listeners
            await self._start_stream_listener()
            await self._start_pubsub_listener()
            
            # Record connection
            connection_info = self._create_connection_info(
                "redis_main",
                pool_size=self.config.pool_size,
                redis_version=await self._get_redis_version()
            )
            self.connections["redis_main"] = connection_info
            self.metrics.connection_count = 1
            
            return True
            
        except Exception as e:
            await self._record_error(f"Redis connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close Redis connections and clean up."""
        try:
            # Close pub/sub client
            if self.pubsub_client:
                await self.pubsub_client.close()
            
            # Close main Redis client
            if self.redis_client:
                await self.redis_client.close()
            
            # Close connection pool
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            # Cancel batch timer
            if self._batch_timer:
                self._batch_timer.cancel()
            
        except Exception as e:
            await self._record_error(f"Redis disconnect error: {e}")
    
    async def send_message(self, message: UnifiedMessage) -> MessageResult:
        """
        Send message using Redis Pub/Sub or Streams based on delivery guarantee.
        """
        start_time = time.time()
        
        try:
            if not self.is_connected():
                return MessageResult(
                    success=False,
                    message_id=message.id,
                    error="Redis adapter not connected"
                )
            
            # Choose routing strategy based on delivery guarantee
            if message.delivery_guarantee.value in ["exactly_once", "at_least_once", "ordered"]:
                result = await self._send_via_streams(message)
            else:
                result = await self._send_via_pubsub(message)
            
            # Record performance metrics
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(latency_ms)
            
            if result.success:
                self._record_message_sent(len(json.dumps(message.to_dict())))
                result.latency_ms = latency_ms
                result.protocol_used = ProtocolType.REDIS_STREAMS if message.delivery_guarantee.value != "best_effort" else ProtocolType.REDIS_PUBSUB
            
            return result
            
        except Exception as e:
            await self._record_error(f"Message send failed: {e}")
            return MessageResult(
                success=False,
                message_id=message.id,
                error=str(e)
            )
    
    async def _send_via_streams(self, message: UnifiedMessage) -> MessageResult:
        """Send message via Redis Streams for reliable delivery."""
        try:
            stream_key = f"{self.stream_config.stream_prefix}:{message.destination}"
            
            # Prepare message data
            message_data = message.to_dict()
            
            # Add to stream
            message_id = await self.redis_client.xadd(
                stream_key,
                message_data,
                maxlen=self.stream_config.max_length
            )
            
            # Update message status
            message.status = MessageStatus.DELIVERED
            
            return MessageResult(
                success=True,
                message_id=message.id,
                protocol_used=ProtocolType.REDIS_STREAMS,
                metadata={"stream_key": stream_key, "redis_message_id": message_id}
            )
            
        except Exception as e:
            await self._enqueue_for_retry(message)
            raise e
    
    async def _send_via_pubsub(self, message: UnifiedMessage) -> MessageResult:
        """Send message via Redis Pub/Sub for real-time delivery."""
        try:
            channel = f"{self.pubsub_config.channel_prefix}:{message.destination}"
            
            # Prepare message data
            message_data = json.dumps(message.to_dict())
            
            # Publish to channel
            subscribers = await self.redis_client.publish(channel, message_data)
            
            return MessageResult(
                success=True,
                message_id=message.id,
                protocol_used=ProtocolType.REDIS_PUBSUB,
                metadata={"channel": channel, "subscribers": subscribers}
            )
            
        except Exception as e:
            if message.delivery_attempts < message.max_retries:
                await self._enqueue_for_retry(message)
            raise e
    
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[UnifiedMessage], asyncio.coroutine],
        use_streams: bool = True,
        consumer_group: Optional[str] = None,
        **kwargs
    ) -> SubscriptionResult:
        """
        Subscribe to messages using Redis Streams or Pub/Sub.
        
        Args:
            pattern: Message pattern to match (routing key or channel pattern)
            handler: Async callback function for messages
            use_streams: Whether to use Streams (True) or Pub/Sub (False)
            consumer_group: Consumer group name for Streams
            **kwargs: Additional options
        """
        try:
            subscription_id = str(uuid.uuid4())
            message_handler = MessageHandler(subscription_id, handler, pattern)
            
            if use_streams:
                # Subscribe to streams
                stream_key = f"{self.stream_config.stream_prefix}:{pattern}"
                group_name = consumer_group or f"{self.stream_config.consumer_group_prefix}:{pattern}"
                
                # Create consumer group if it doesn't exist
                try:
                    await self.redis_client.xgroup_create(
                        stream_key, 
                        group_name, 
                        id="0", 
                        mkstream=True
                    )
                except ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        raise
                
                # Track consumer group
                if stream_key not in self.consumer_groups:
                    self.consumer_groups[stream_key] = set()
                self.consumer_groups[stream_key].add(group_name)
                
                # Register handler
                self.stream_subscriptions[subscription_id] = message_handler
                self.active_consumers[subscription_id] = stream_key
                
            else:
                # Subscribe to pub/sub
                if "*" in pattern or "?" in pattern:
                    # Pattern subscription
                    await self.pubsub_client.psubscribe(pattern)
                    self.pubsub_patterns[subscription_id] = message_handler
                else:
                    # Channel subscription
                    channel = f"{self.pubsub_config.channel_prefix}:{pattern}"
                    await self.pubsub_client.subscribe(channel)
                    self.pubsub_subscriptions[subscription_id] = message_handler
            
            # Update metrics
            self.metrics.active_subscriptions += 1
            
            return SubscriptionResult(
                success=True,
                subscription_id=subscription_id,
                pattern=pattern,
                metadata={
                    "use_streams": use_streams,
                    "consumer_group": consumer_group,
                    "stream_key": stream_key if use_streams else None
                }
            )
            
        except Exception as e:
            await self._record_error(f"Subscription failed: {e}")
            return SubscriptionResult(
                success=False,
                subscription_id="",
                pattern=pattern,
                error=str(e)
            )
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from messages."""
        try:
            # Check streams subscriptions
            if subscription_id in self.stream_subscriptions:
                del self.stream_subscriptions[subscription_id]
                if subscription_id in self.active_consumers:
                    del self.active_consumers[subscription_id]
                self.metrics.active_subscriptions -= 1
                return True
            
            # Check pub/sub subscriptions
            if subscription_id in self.pubsub_subscriptions:
                handler = self.pubsub_subscriptions[subscription_id]
                channel = f"{self.pubsub_config.channel_prefix}:{handler.pattern}"
                await self.pubsub_client.unsubscribe(channel)
                del self.pubsub_subscriptions[subscription_id]
                self.metrics.active_subscriptions -= 1
                return True
            
            # Check pub/sub patterns
            if subscription_id in self.pubsub_patterns:
                handler = self.pubsub_patterns[subscription_id]
                await self.pubsub_client.punsubscribe(handler.pattern)
                del self.pubsub_patterns[subscription_id]
                self.metrics.active_subscriptions -= 1
                return True
            
            return False
            
        except Exception as e:
            await self._record_error(f"Unsubscribe failed: {e}")
            return False
    
    async def health_check(self) -> HealthStatus:
        """Perform Redis health check."""
        try:
            if not self.redis_client:
                return HealthStatus.UNHEALTHY
            
            # Test Redis connection
            start_time = time.time()
            await self.redis_client.ping()
            ping_latency = (time.time() - start_time) * 1000
            
            # Check connection pool status
            pool_info = await self._get_pool_info()
            
            # Determine health status
            if ping_latency > 100:  # 100ms threshold
                return HealthStatus.DEGRADED
            elif pool_info["available_connections"] < 2:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
                
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _start_stream_listener(self) -> None:
        """Start background task to listen for stream messages."""
        async def stream_listener():
            while not self._shutdown_event.is_set():
                try:
                    for subscription_id, handler in list(self.stream_subscriptions.items()):
                        if subscription_id not in self.active_consumers:
                            continue
                        
                        stream_key = self.active_consumers[subscription_id]
                        consumer_name = f"{self.stream_config.consumer_name_prefix}:{subscription_id}"
                        group_name = f"{self.stream_config.consumer_group_prefix}:{handler.pattern}"
                        
                        try:
                            # Read messages from stream
                            messages = await self.redis_client.xreadgroup(
                                group_name,
                                consumer_name,
                                streams={stream_key: ">"},
                                count=10,
                                block=self.stream_config.block_timeout_ms
                            )
                            
                            # Process messages
                            for stream, message_list in messages:
                                for redis_msg_id, fields in message_list:
                                    try:
                                        # Parse message
                                        message_data = self._parse_redis_fields(fields)
                                        message = UnifiedMessage.from_dict(message_data)
                                        
                                        # Handle message
                                        success = await handler.handle_message(message)
                                        
                                        if success:
                                            # Acknowledge message
                                            await self.redis_client.xack(stream_key, group_name, redis_msg_id)
                                            self._record_message_received(len(json.dumps(message_data)))
                                        else:
                                            # Message will be retried by Redis Streams
                                            pass
                                    
                                    except Exception as e:
                                        await self._record_error(f"Stream message processing failed: {e}")
                        
                        except asyncio.TimeoutError:
                            continue  # Normal timeout, continue listening
                        except Exception as e:
                            await self._record_error(f"Stream read error: {e}")
                            await asyncio.sleep(1)
                    
                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await self._record_error(f"Stream listener error: {e}")
                    await asyncio.sleep(5)
        
        task = asyncio.create_task(stream_listener())
        self._background_tasks.append(task)
    
    async def _start_pubsub_listener(self) -> None:
        """Start background task to listen for pub/sub messages."""
        async def pubsub_listener():
            try:
                async for message in self.pubsub_client.listen():
                    if message["type"] in ["message", "pmessage"]:
                        try:
                            # Parse message
                            message_data = json.loads(message["data"])
                            unified_message = UnifiedMessage.from_dict(message_data)
                            
                            # Find appropriate handler
                            handler = None
                            if message["type"] == "message":
                                # Direct channel subscription
                                for sub_id, h in self.pubsub_subscriptions.items():
                                    if message["channel"].decode().endswith(h.pattern):
                                        handler = h
                                        break
                            elif message["type"] == "pmessage":
                                # Pattern subscription
                                pattern = message["pattern"].decode()
                                for sub_id, h in self.pubsub_patterns.items():
                                    if h.pattern == pattern:
                                        handler = h
                                        break
                            
                            # Handle message
                            if handler:
                                await handler.handle_message(unified_message)
                                self._record_message_received(len(message["data"]))
                        
                        except Exception as e:
                            await self._record_error(f"Pub/Sub message processing failed: {e}")
            
            except asyncio.CancelledError:
                pass
            except Exception as e:
                await self._record_error(f"Pub/Sub listener error: {e}")
        
        task = asyncio.create_task(pubsub_listener())
        self._background_tasks.append(task)
    
    def _parse_redis_fields(self, fields: Dict[bytes, bytes]) -> Dict[str, Any]:
        """Parse Redis stream fields to dictionary."""
        result = {}
        for key, value in fields.items():
            key_str = key.decode() if isinstance(key, bytes) else key
            value_str = value.decode() if isinstance(value, bytes) else value
            
            # Try to parse JSON fields
            if key_str in ["payload", "headers", "protocol_data"]:
                try:
                    result[key_str] = json.loads(value_str)
                except (json.JSONDecodeError, TypeError):
                    result[key_str] = value_str
            else:
                result[key_str] = value_str
        
        return result
    
    async def _get_redis_version(self) -> str:
        """Get Redis server version."""
        try:
            info = await self.redis_client.info("server")
            return info.get("redis_version", "unknown")
        except Exception:
            return "unknown"
    
    async def _get_pool_info(self) -> Dict[str, Any]:
        """Get connection pool information."""
        if not self.redis_pool:
            return {}
        
        return {
            "max_connections": self.redis_pool.max_connections,
            "created_connections": self.redis_pool.created_connections,
            "available_connections": len(self.redis_pool._available_connections),
            "in_use_connections": len(self.redis_pool._in_use_connections)
        }
    
    # === ADVANCED FEATURES ===
    
    async def claim_pending_messages(
        self,
        stream_key: str,
        consumer_group: str,
        min_idle_time: int = None
    ) -> List[UnifiedMessage]:
        """
        Claim pending messages that have been idle too long.
        Useful for handling failed consumers and ensuring message processing.
        """
        try:
            min_idle = min_idle_time or self.stream_config.claim_min_idle_time
            
            # Get pending messages
            pending = await self.redis_client.xpending_range(
                stream_key,
                consumer_group,
                min="-",
                max="+",
                count=100
            )
            
            claimed_messages = []
            for msg_info in pending:
                if msg_info["idle"] >= min_idle:
                    # Claim the message
                    claimed = await self.redis_client.xclaim(
                        stream_key,
                        consumer_group,
                        f"{self.stream_config.consumer_name_prefix}:claimer",
                        min_idle,
                        [msg_info["message_id"]]
                    )
                    
                    for redis_msg_id, fields in claimed:
                        message_data = self._parse_redis_fields(fields)
                        message = UnifiedMessage.from_dict(message_data)
                        claimed_messages.append(message)
            
            return claimed_messages
            
        except Exception as e:
            await self._record_error(f"Claim pending messages failed: {e}")
            return []
    
    async def get_stream_info(self, stream_key: str) -> Dict[str, Any]:
        """Get information about a Redis stream."""
        try:
            info = await self.redis_client.xinfo_stream(stream_key)
            return {
                "length": info["length"],
                "first_entry": info["first-entry"],
                "last_entry": info["last-entry"],
                "groups": info["groups"]
            }
        except Exception as e:
            await self._record_error(f"Get stream info failed: {e}")
            return {}
    
    async def get_consumer_group_info(self, stream_key: str) -> List[Dict[str, Any]]:
        """Get consumer group information for a stream."""
        try:
            groups = await self.redis_client.xinfo_groups(stream_key)
            return [
                {
                    "name": group["name"].decode(),
                    "consumers": group["consumers"],
                    "pending": group["pending"],
                    "last_delivered_id": group["last-delivered-id"].decode()
                }
                for group in groups
            ]
        except Exception as e:
            await self._record_error(f"Get consumer group info failed: {e}")
            return []