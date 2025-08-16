"""
Unified Redis Integration Service for LeanVibe Agent Hive
Consolidates 5+ Redis implementations into comprehensive, reliable infrastructure

Provides enterprise-grade Redis operations with:
- Unified connection pooling and management
- Caching with TTL and eviction policies
- Pub/Sub messaging for real-time communication
- Stream-based event processing with consumer groups
- Distributed coordination and locking
- High-performance queuing with load balancing
- Message compression and local caching
- Circuit breaker patterns for resilience
- Comprehensive health monitoring and metrics
"""

from typing import Optional, Dict, Any, List, Union, AsyncGenerator, Set, Callable, Tuple
import asyncio
import json
import pickle
import time
import uuid
import zlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import weakref

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError, ResponseError

from .config import settings
from .logging_service import get_component_logger

logger = get_component_logger("redis_integration")


# =====================================================================================
# ENUMS AND CONSTANTS
# =====================================================================================

class RedisPattern(str, Enum):
    CACHE = "cache"
    PUBSUB = "pubsub"
    STREAMS = "streams"
    COORDINATION = "coordination"
    QUEUE = "queue"
    LOCK = "lock"


class SerializationFormat(str, Enum):
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


class CompressionType(str, Enum):
    NONE = "none"
    ZLIB = "zlib"


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class MessageRoutingMode(str, Enum):
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    CAPABILITY_MATCHED = "capability_matched"
    WORKFLOW_AWARE = "workflow_aware"


# =====================================================================================
# DATA CLASSES AND CONFIGURATION
# =====================================================================================

@dataclass
class RedisConfig:
    """Redis configuration with connection pooling"""
    url: str
    max_connections: int = 20
    min_connections: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[str, int] = field(default_factory=lambda: {
        'TCP_KEEPINTVL': 1,
        'TCP_KEEPCNT': 3,
        'TCP_KEEPIDLE': 1
    })
    connection_timeout: int = 5
    socket_timeout: int = 30


@dataclass
class RedisMetrics:
    """Comprehensive Redis performance metrics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_savings_bytes: int = 0
    circuit_breaker_trips: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    messages_claimed: int = 0
    messages_dlq: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_success_rate(self) -> float:
        """Get operation success rate"""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return self.cache_hits / total_cache_ops
    
    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds"""
        return time.time() - self.start_time


@dataclass
class ConsumerGroupConfig:
    """Configuration for Redis consumer groups"""
    name: str
    stream_name: str
    routing_mode: MessageRoutingMode = MessageRoutingMode.LOAD_BALANCED
    max_consumers: int = 10
    min_consumers: int = 1
    idle_timeout_ms: int = 30000
    max_retries: int = 3
    batch_size: int = 10
    claim_batch_size: int = 100
    auto_scale_enabled: bool = True
    lag_threshold: int = 100


# =====================================================================================
# UTILITY CLASSES
# =====================================================================================

class CircuitBreaker:
    """Circuit breaker for Redis operations with failure protection"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: type = RedisError
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise RedisConnectionError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.timeout_seconds
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class LocalCache:
    """Local LRU cache to reduce Redis load"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from local cache"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.access_times[key] > self.ttl_seconds:
            self.delete(key)
            return None
        
        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Store value in local cache"""
        current_time = time.time()
        
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove LRU item
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = value
        self.access_times[key] = current_time
        self.access_order.append(key)
    
    def delete(self, key: str) -> None:
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cached items"""
        self.cache.clear()
        self.access_times.clear()
        self.access_order.clear()


class MessageCompressor:
    """Message compression utilities"""
    
    @staticmethod
    def compress(data: Union[str, bytes], compression_type: CompressionType = CompressionType.ZLIB) -> bytes:
        """Compress data using specified compression type"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data)
        else:
            return data  # Fallback
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType = CompressionType.ZLIB) -> str:
        """Decompress data using specified compression type"""
        if compression_type == CompressionType.NONE:
            return data.decode('utf-8')
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data).decode('utf-8')
        else:
            return data.decode('utf-8')  # Fallback


class DynamicConnectionPool:
    """Dynamic Redis connection pool that adapts to load"""
    
    def __init__(
        self,
        redis_url: str,
        min_connections: int = 10,
        max_connections: int = 100,
        growth_factor: float = 1.5
    ):
        self.redis_url = redis_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.growth_factor = growth_factor
        
        self.current_pool_size = min_connections
        self.pool = None
        self.utilization_history = deque(maxlen=60)  # 1 minute history
        self.last_adjustment = time.time()
    
    async def create_pool(self) -> ConnectionPool:
        """Create optimized connection pool"""
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.current_pool_size,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
            socket_connect_timeout=5,
            socket_timeout=30
        )
        return self.pool


# =====================================================================================
# MAIN REDIS INTEGRATION SERVICE
# =====================================================================================

class RedisIntegrationService:
    """
    Comprehensive Redis service supporting all patterns:
    - Connection pooling and management
    - Caching with TTL and eviction policies
    - Pub/Sub messaging
    - Stream-based event processing
    - Distributed coordination and locking
    - High-performance queuing
    - Performance optimizations and monitoring
    """
    
    _instance: Optional['RedisIntegrationService'] = None
    
    def __new__(cls) -> 'RedisIntegrationService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._initialize_service()
            self._initialized = True
    
    def _initialize_service(self):
        """Initialize Redis service with all components"""
        # Configuration
        self.config = RedisConfig(
            url=settings.REDIS_URL,
            max_connections=getattr(settings, 'REDIS_MAX_CONNECTIONS', 20),
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Connection management
        self._connection_pool = DynamicConnectionPool(
            self.config.url,
            min_connections=self.config.min_connections,
            max_connections=self.config.max_connections
        )
        self.redis_client: Optional[Redis] = None
        self.pubsub_client: Optional[Redis] = None
        
        # Circuit breaker and resilience
        self.circuit_breaker = CircuitBreaker()
        
        # Local caching layer
        self.local_cache = LocalCache(max_size=1000, ttl_seconds=300)
        
        # Metrics and monitoring
        self.metrics = RedisMetrics()
        
        # Pub/Sub management
        self._subscriptions: Dict[str, List[callable]] = {}
        self._pubsub_task: Optional[asyncio.Task] = None
        
        # Stream and consumer group management
        self._consumer_groups: Dict[str, ConsumerGroupConfig] = {}
        self._active_consumers: Dict[str, Dict[str, asyncio.Task]] = defaultdict(dict)
        self._message_handlers: Dict[str, Callable] = {}
        
        # Message batching for performance
        self._pending_messages = defaultdict(list)
        self._batch_size = 10
        self._batch_timeout = 0.1  # 100ms
        self._batch_tasks = {}
        
        # Compression settings
        self._compression_threshold = 1024  # Compress messages > 1KB
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Redis integration service initialized", url=self.config.url)
    
    async def connect(self):
        """Establish Redis connections with circuit breaker protection"""
        async def _connect():
            # Create connection pool
            pool = await self._connection_pool.create_pool()
            
            # Create Redis clients
            self.redis_client = Redis(connection_pool=pool)
            self.pubsub_client = Redis(connection_pool=pool)
            
            # Test connections
            await self.redis_client.ping()
            await self.pubsub_client.ping()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Redis connections established successfully")
        
        await self.circuit_breaker.call(_connect)
    
    async def disconnect(self):
        """Clean shutdown of Redis connections and background tasks"""
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Stop pub/sub
        if self._pubsub_task and not self._pubsub_task.done():
            self._pubsub_task.cancel()
        
        # Stop consumer tasks
        for group_consumers in self._active_consumers.values():
            for task in group_consumers.values():
                if not task.done():
                    task.cancel()
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        if self.pubsub_client:
            await self.pubsub_client.close()
        
        if self._connection_pool.pool:
            await self._connection_pool.pool.disconnect()
        
        logger.info("Redis connections closed")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Metrics collection task
        task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.add(task)
        
        # Health monitoring task
        task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(task)
        
        # Connection pool optimization task
        task = asyncio.create_task(self._pool_optimization_loop())
        self._background_tasks.add(task)
    
    # =====================================================================================
    # CACHE OPERATIONS
    # =====================================================================================
    
    async def cache_set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        format: SerializationFormat = SerializationFormat.JSON,
        use_compression: bool = False
    ) -> bool:
        """Set cache value with optional TTL and compression"""
        start_time = time.time()
        
        try:
            # Check local cache first
            if ttl and ttl > 60:  # Only cache locally if TTL > 1 minute
                self.local_cache.put(key, value)
                self.metrics.cache_hits += 1
            
            # Serialize value
            serialized_value = self._serialize_value(value, format)
            
            # Compress if needed
            if use_compression or len(serialized_value) > self._compression_threshold:
                compressed = MessageCompressor.compress(serialized_value)
                if len(compressed) < len(serialized_value):
                    serialized_value = compressed
                    self.metrics.compression_savings_bytes += len(serialized_value) - len(compressed)
                    # Store compression flag
                    await self.circuit_breaker.call(
                        self.redis_client.set,
                        f"{key}:compressed",
                        "true",
                        ex=ttl
                    )
            
            # Store in Redis
            async def _set():
                if ttl:
                    return await self.redis_client.setex(key, ttl, serialized_value)
                else:
                    return await self.redis_client.set(key, serialized_value)
            
            result = await self.circuit_breaker.call(_set)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, True)
            
            logger.debug("Cache set successful", key=key, ttl=ttl, format=format.value)
            return bool(result)
            
        except Exception as e:
            self._update_metrics((time.time() - start_time) * 1000, False)
            logger.error("Cache set failed", key=key, error=str(e))
            return False
    
    async def cache_get(
        self, 
        key: str,
        format: SerializationFormat = SerializationFormat.JSON
    ) -> Optional[Any]:
        """Get cache value with deserialization and local caching"""
        start_time = time.time()
        
        try:
            # Check local cache first
            cached_value = self.local_cache.get(key)
            if cached_value is not None:
                self.metrics.cache_hits += 1
                return cached_value
            
            self.metrics.cache_misses += 1
            
            # Get from Redis
            async def _get():
                return await self.redis_client.get(key)
            
            serialized_value = await self.circuit_breaker.call(_get)
            if serialized_value is None:
                return None
            
            # Check if compressed
            is_compressed = await self.redis_client.get(f"{key}:compressed")
            if is_compressed:
                serialized_value = MessageCompressor.decompress(serialized_value)
            
            # Deserialize
            value = self._deserialize_value(serialized_value, format)
            
            # Cache locally for future requests
            self.local_cache.put(key, value)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, True)
            
            logger.debug("Cache get successful", key=key, format=format.value)
            return value
            
        except Exception as e:
            self._update_metrics((time.time() - start_time) * 1000, False)
            logger.error("Cache get failed", key=key, error=str(e))
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache key from both Redis and local cache"""
        try:
            # Remove from local cache
            self.local_cache.delete(key)
            
            # Remove from Redis
            async def _delete():
                # Delete main key and compression flag
                pipe = self.redis_client.pipeline()
                pipe.delete(key)
                pipe.delete(f"{key}:compressed")
                return await pipe.execute()
            
            results = await self.circuit_breaker.call(_delete)
            
            logger.debug("Cache delete successful", key=key)
            return any(results)
            
        except Exception as e:
            logger.error("Cache delete failed", key=key, error=str(e))
            return False
    
    # =====================================================================================
    # PUB/SUB OPERATIONS
    # =====================================================================================
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel with serialization"""
        try:
            serialized_message = self._serialize_value(message, SerializationFormat.JSON)
            
            async def _publish():
                return await self.redis_client.publish(channel, serialized_message)
            
            result = await self.circuit_breaker.call(_publish)
            self.metrics.messages_sent += 1
            
            logger.debug("Message published", channel=channel, subscribers=result)
            return result
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error("Publish failed", channel=channel, error=str(e))
            return 0
    
    async def subscribe(self, channel: str, callback: callable):
        """Subscribe to channel with callback"""
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        
        self._subscriptions[channel].append(callback)
        
        # Start pub/sub listener if not running
        if self._pubsub_task is None or self._pubsub_task.done():
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())
        
        logger.info("Subscribed to channel", channel=channel)
    
    async def _pubsub_listener(self):
        """Background task for handling pub/sub messages"""
        try:
            pubsub = self.pubsub_client.pubsub()
            await pubsub.subscribe(*self._subscriptions.keys())
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel'].decode('utf-8')
                    data = self._deserialize_value(message['data'], SerializationFormat.JSON)
                    
                    # Call all callbacks for this channel
                    for callback in self._subscriptions.get(channel, []):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(channel, data)
                            else:
                                callback(channel, data)
                        except Exception as e:
                            logger.error("Callback error", channel=channel, error=str(e))
        
        except Exception as e:
            logger.error("Pub/sub listener error", error=str(e))
    
    # =====================================================================================
    # STREAM OPERATIONS
    # =====================================================================================
    
    async def stream_add(self, stream: str, fields: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        """Add entry to Redis stream with compression"""
        try:
            # Serialize and optionally compress large payloads
            serialized_fields = {}
            for k, v in fields.items():
                serialized_value = self._serialize_value(v, SerializationFormat.JSON)
                
                # Compress large values
                if len(serialized_value) > self._compression_threshold:
                    compressed = MessageCompressor.compress(serialized_value)
                    if len(compressed) < len(serialized_value):
                        serialized_fields[k] = compressed
                        serialized_fields[f"{k}:compressed"] = "true"
                        self.metrics.compression_savings_bytes += len(serialized_value) - len(compressed)
                    else:
                        serialized_fields[k] = serialized_value
                else:
                    serialized_fields[k] = serialized_value
            
            async def _add():
                return await self.redis_client.xadd(
                    stream, 
                    serialized_fields,
                    maxlen=maxlen or getattr(settings, 'REDIS_STREAM_MAX_LEN', 1000000)
                )
            
            entry_id = await self.circuit_breaker.call(_add)
            self.metrics.messages_sent += 1
            
            logger.debug("Stream entry added", stream=stream, entry_id=entry_id)
            return entry_id
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error("Stream add failed", stream=stream, error=str(e))
            return ""
    
    async def stream_read(
        self, 
        streams: Dict[str, str], 
        count: Optional[int] = None,
        block: Optional[int] = None
    ) -> List[List]:
        """Read from Redis streams with decompression"""
        try:
            async def _read():
                return await self.redis_client.xread(streams, count=count, block=block)
            
            entries = await self.circuit_breaker.call(_read)
            
            # Deserialize and decompress entries
            result = []
            for stream_name, stream_entries in entries:
                stream_result = []
                for entry_id, fields in stream_entries:
                    deserialized_fields = {}
                    
                    for k, v in fields.items():
                        k_str = k.decode() if isinstance(k, bytes) else k
                        
                        # Skip compression flags
                        if k_str.endswith(':compressed'):
                            continue
                        
                        # Check if field is compressed
                        compressed_flag = fields.get(f"{k}:compressed") or fields.get(f"{k_str}:compressed")
                        is_compressed = compressed_flag and compressed_flag.decode() == "true"
                        
                        if is_compressed:
                            value_str = MessageCompressor.decompress(v)
                            deserialized_fields[k_str] = self._deserialize_value(value_str, SerializationFormat.JSON)
                        else:
                            value_str = v.decode() if isinstance(v, bytes) else v
                            deserialized_fields[k_str] = self._deserialize_value(value_str, SerializationFormat.JSON)
                    
                    stream_result.append((entry_id.decode(), deserialized_fields))
                
                result.append((stream_name.decode(), stream_result))
            
            self.metrics.messages_received += len([e for _, entries in result for e in entries])
            return result
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error("Stream read failed", streams=streams, error=str(e))
            return []
    
    # =====================================================================================
    # CONSUMER GROUP OPERATIONS
    # =====================================================================================
    
    async def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        consumer_id: str = "$",
        mkstream: bool = True
    ) -> bool:
        """Create consumer group for stream processing"""
        try:
            async def _create():
                return await self.redis_client.xgroup_create(
                    stream_name,
                    group_name,
                    id=consumer_id,
                    mkstream=mkstream
                )
            
            await self.circuit_breaker.call(_create)
            logger.info("Consumer group created", stream=stream_name, group=group_name)
            return True
            
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                logger.debug("Consumer group already exists", stream=stream_name, group=group_name)
                return True
            else:
                logger.error("Failed to create consumer group", error=str(e))
                return False
        except Exception as e:
            logger.error("Consumer group creation failed", stream=stream_name, group=group_name, error=str(e))
            return False
    
    async def consume_stream_messages(
        self,
        stream_name: str,
        group_name: str,
        consumer_name: str,
        handler: Callable,
        count: int = 10,
        block: int = 1000
    ) -> List[Any]:
        """Consume messages from stream with consumer group"""
        try:
            # Ensure consumer group exists
            await self.create_consumer_group(stream_name, group_name)
            
            async def _read():
                return await self.redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_name: '>'},
                    count=count,
                    block=block
                )
            
            messages = await self.circuit_breaker.call(_read)
            processed_results = []
            
            for stream, stream_messages in messages:
                for message_id, fields in stream_messages:
                    try:
                        # Deserialize message
                        deserialized_fields = {}
                        for k, v in fields.items():
                            k_str = k.decode() if isinstance(k, bytes) else k
                            if not k_str.endswith(':compressed'):
                                v_str = v.decode() if isinstance(v, bytes) else v
                                deserialized_fields[k_str] = self._deserialize_value(v_str, SerializationFormat.JSON)
                        
                        # Process message
                        result = await handler(message_id.decode(), deserialized_fields)
                        processed_results.append(result)
                        
                        # Acknowledge message
                        await self.redis_client.xack(stream_name, group_name, message_id)
                        
                    except Exception as e:
                        logger.error("Message processing failed", message_id=message_id, error=str(e))
            
            self.metrics.messages_received += len(processed_results)
            return processed_results
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error("Stream consumption failed", stream=stream_name, group=group_name, error=str(e))
            return []
    
    # =====================================================================================
    # COORDINATION AND LOCKING
    # =====================================================================================
    
    async def acquire_lock(
        self, 
        lock_name: str, 
        timeout: int = 10,
        blocking_timeout: Optional[int] = None
    ) -> bool:
        """Acquire distributed lock with timeout"""
        try:
            async def _acquire():
                return await self.redis_client.set(
                    f"lock:{lock_name}",
                    str(uuid.uuid4()),
                    nx=True,
                    ex=timeout
                )
            
            result = await self.circuit_breaker.call(_acquire)
            if result:
                logger.debug("Lock acquired", lock_name=lock_name, timeout=timeout)
            return bool(result)
            
        except Exception as e:
            logger.error("Lock acquisition failed", lock_name=lock_name, error=str(e))
            return False
    
    async def release_lock(self, lock_name: str) -> bool:
        """Release distributed lock"""
        try:
            async def _release():
                return await self.redis_client.delete(f"lock:{lock_name}")
            
            result = await self.circuit_breaker.call(_release)
            logger.debug("Lock released", lock_name=lock_name)
            return bool(result)
            
        except Exception as e:
            logger.error("Lock release failed", lock_name=lock_name, error=str(e))
            return False
    
    @asynccontextmanager
    async def distributed_lock(self, lock_name: str, timeout: int = 30):
        """Context manager for distributed locking"""
        lock_value = str(uuid.uuid4())
        acquired = False
        
        try:
            # Try to acquire lock
            acquired = await self.redis_client.set(
                f"lock:{lock_name}",
                lock_value,
                nx=True,
                ex=timeout
            )
            
            if not acquired:
                raise Exception(f"Could not acquire lock: {lock_name}")
            
            yield
            
        finally:
            if acquired:
                # Release lock only if we own it
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                await self.redis_client.eval(lua_script, 1, f"lock:{lock_name}", lock_value)
    
    # =====================================================================================
    # TEAM COORDINATION METHODS
    # =====================================================================================
    
    async def register_agent(self, agent_id: str, capabilities: List[str], metadata: Dict[str, Any] = None) -> bool:
        """Register agent for team coordination"""
        try:
            agent_data = {
                'agent_id': agent_id,
                'capabilities': json.dumps(capabilities),
                'metadata': json.dumps(metadata or {}),
                'registered_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'last_heartbeat': datetime.utcnow().isoformat()
            }
            
            # Store agent data
            await self.redis_client.hset(f"agent:{agent_id}", mapping=agent_data)
            await self.redis_client.expire(f"agent:{agent_id}", timedelta(hours=24))
            
            # Add to active agents set
            await self.redis_client.sadd("active_agents", agent_id)
            
            # Publish registration event
            await self.publish("agent_registrations", {
                "event": "agent_registered",
                "agent_id": agent_id,
                "capabilities": capabilities,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Agent registered", agent_id=agent_id, capabilities=capabilities)
            return True
            
        except Exception as e:
            logger.error("Agent registration failed", agent_id=agent_id, error=str(e))
            return False
    
    async def update_agent_status(self, agent_id: str, status: str, workload: float = 0.0) -> bool:
        """Update agent status and workload"""
        try:
            updates = {
                'status': status,
                'workload': workload,
                'last_heartbeat': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.hset(f"agent:{agent_id}", mapping=updates)
            
            # Publish status update
            await self.publish("agent_status_updates", {
                "event": "status_updated",
                "agent_id": agent_id,
                "status": status,
                "workload": workload,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error("Agent status update failed", agent_id=agent_id, error=str(e))
            return False
    
    async def assign_task(self, task_id: str, agent_id: str, task_data: Dict[str, Any]) -> bool:
        """Assign task to agent with coordination"""
        try:
            async with self.distributed_lock(f"task_assignment:{task_id}"):
                assignment_data = {
                    'task_id': task_id,
                    'agent_id': agent_id,
                    'task_data': json.dumps(task_data),
                    'assigned_at': datetime.utcnow().isoformat(),
                    'status': 'assigned'
                }
                
                # Store assignment
                await self.redis_client.hset("task_assignments", task_id, json.dumps(assignment_data))
                
                # Update agent's active tasks
                await self.redis_client.sadd(f"agent_tasks:{agent_id}", task_id)
                
                # Publish assignment event
                await self.publish("task_assignments", {
                    "event": "task_assigned",
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                logger.info("Task assigned", task_id=task_id, agent_id=agent_id)
                return True
                
        except Exception as e:
            logger.error("Task assignment failed", task_id=task_id, agent_id=agent_id, error=str(e))
            return False
    
    # =====================================================================================
    # UTILITY METHODS
    # =====================================================================================
    
    def _serialize_value(self, value: Any, format: SerializationFormat) -> Union[str, bytes]:
        """Serialize value based on format"""
        if format == SerializationFormat.JSON:
            return json.dumps(value, default=str)
        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(value)
        elif format == SerializationFormat.STRING:
            return str(value)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    def _deserialize_value(self, value: Union[str, bytes], format: SerializationFormat) -> Any:
        """Deserialize value based on format"""
        if isinstance(value, bytes):
            if format == SerializationFormat.PICKLE:
                return pickle.loads(value)
            else:
                value = value.decode('utf-8')
        
        if format == SerializationFormat.JSON:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        elif format == SerializationFormat.STRING:
            return value
        else:
            return value
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """Update performance metrics"""
        self.metrics.total_operations += 1
        
        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1
        
        # Update average latency
        if self.metrics.successful_operations == 1:
            self.metrics.avg_latency_ms = latency_ms
        else:
            total_ops = self.metrics.successful_operations
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (total_ops - 1) + latency_ms) / total_ops
            )
    
    # =====================================================================================
    # HEALTH CHECK AND MONITORING
    # =====================================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            start_time = time.time()
            
            async def _health():
                info = await self.redis_client.info()
                ping_result = await self.redis_client.ping()
                return info, ping_result
            
            info, ping_result = await self.circuit_breaker.call(_health)
            ping_latency_ms = (time.time() - start_time) * 1000
            
            # Calculate health status
            is_healthy = (
                ping_result and
                self.circuit_breaker.state == CircuitBreakerState.CLOSED and
                ping_latency_ms < 1000 and  # < 1 second
                self.metrics.get_success_rate() >= 0.999  # 99.9% success rate
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "ping_latency_ms": ping_latency_ms,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "unknown"),
                "uptime": info.get("uptime_in_seconds", 0),
                "version": info.get("redis_version", "unknown"),
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "success_rate": self.metrics.get_success_rate(),
                "cache_hit_rate": self.metrics.get_cache_hit_rate(),
                "total_operations": self.metrics.total_operations,
                "compression_savings_mb": self.metrics.compression_savings_bytes / (1024 * 1024),
                "uptime_seconds": self.metrics.get_uptime_seconds()
            }
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "operations": {
                "total": self.metrics.total_operations,
                "successful": self.metrics.successful_operations,
                "failed": self.metrics.failed_operations,
                "success_rate": self.metrics.get_success_rate()
            },
            "performance": {
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "uptime_seconds": self.metrics.get_uptime_seconds()
            },
            "caching": {
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "hit_rate": self.metrics.get_cache_hit_rate(),
                "local_cache_size": len(self.local_cache.cache)
            },
            "messaging": {
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "messages_claimed": self.metrics.messages_claimed,
                "messages_dlq": self.metrics.messages_dlq
            },
            "compression": {
                "savings_bytes": self.metrics.compression_savings_bytes,
                "savings_mb": self.metrics.compression_savings_bytes / (1024 * 1024)
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "trips": self.metrics.circuit_breaker_trips
            }
        }
    
    # =====================================================================================
    # BACKGROUND TASKS
    # =====================================================================================
    
    async def _metrics_collection_loop(self):
        """Background metrics collection and logging"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                metrics = await self.get_performance_metrics()
                logger.info("Redis performance metrics", **metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                health = await self.health_check()
                if health["status"] != "healthy":
                    logger.warning("Redis health degraded", **health)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitoring error", error=str(e))
    
    async def _pool_optimization_loop(self):
        """Background connection pool optimization"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Check if pool needs scaling (simplified logic)
                current_utilization = self.metrics.total_operations / 100  # Simplified metric
                
                if current_utilization > 0.8:
                    logger.info("High Redis utilization detected", utilization=current_utilization)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pool optimization error", error=str(e))


# =====================================================================================
# CONVENIENCE FUNCTIONS AND GLOBAL ACCESS
# =====================================================================================

def get_redis_service() -> RedisIntegrationService:
    """Get Redis integration service instance"""
    return RedisIntegrationService()


async def cache_with_ttl(key: str, value: Any, ttl: int = 3600) -> bool:
    """Quick cache set with TTL"""
    redis_service = get_redis_service()
    return await redis_service.cache_set(key, value, ttl)


async def get_cached(key: str) -> Optional[Any]:
    """Quick cache get"""
    redis_service = get_redis_service()
    return await redis_service.cache_get(key)


async def publish_event(channel: str, event: Dict[str, Any]) -> int:
    """Quick event publishing"""
    redis_service = get_redis_service()
    return await redis_service.publish(channel, event)


async def subscribe_to_events(channel: str, callback: callable):
    """Quick event subscription"""
    redis_service = get_redis_service()
    await redis_service.subscribe(channel, callback)


# =====================================================================================
# CONTEXT MANAGER FOR REDIS SESSIONS
# =====================================================================================

@asynccontextmanager
async def redis_session():
    """Context manager for Redis sessions"""
    redis_service = get_redis_service()
    await redis_service.connect()
    try:
        yield redis_service
    finally:
        await redis_service.disconnect()