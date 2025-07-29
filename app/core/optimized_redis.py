"""
Optimized Redis Manager for LeanVibe Agent Hive 2.0

Performance-optimized Redis operations with:
- Dynamic connection pooling
- Message compression and batching
- Circuit breaker pattern
- Local caching layer
- Connection health monitoring
"""

import asyncio
import json
import uuid
import zlib
import pickle
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import weakref
import time

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .config import settings

logger = structlog.get_logger()


class CompressionType(Enum):
    """Compression types for message payloads."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RedisMetrics:
    """Redis performance metrics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_savings_bytes: int = 0
    circuit_breaker_trips: int = 0
    
    def get_success_rate(self) -> float:
        """Get operation success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return self.cache_hits / total_cache_ops


class CircuitBreaker:
    """Circuit breaker for Redis operations."""
    
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
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ConnectionError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.timeout_seconds
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class LocalCache:
    """Local LRU cache to reduce Redis load."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_order = []
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from local cache."""
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
        """Store value in local cache."""
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
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.access_order.remove(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
        self.access_order.clear()


class MessageCompressor:
    """Message compression utilities."""
    
    @staticmethod
    def compress(data: Union[str, bytes], compression_type: CompressionType = CompressionType.ZLIB) -> bytes:
        """Compress data using specified compression type."""
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
        """Decompress data using specified compression type."""
        if compression_type == CompressionType.NONE:
            return data.decode('utf-8')
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data).decode('utf-8')
        else:
            return data.decode('utf-8')  # Fallback


class OptimizedRedisStreamMessage:
    """Optimized Redis Stream Message with compression support."""
    
    def __init__(self, stream_id: str, fields: Dict[str, Any], compressed: bool = False):
        self.id = stream_id
        self.timestamp = datetime.fromtimestamp(int(stream_id.split('-')[0]) / 1000)
        self.fields = fields
        self.compressed = compressed
    
    @property
    def message_id(self) -> str:
        return self.fields.get('message_id', str(uuid.uuid4()))
    
    @property
    def from_agent(self) -> str:
        return self.fields.get('from_agent', 'unknown')
    
    @property
    def to_agent(self) -> str:
        return self.fields.get('to_agent', 'broadcast')
    
    @property
    def message_type(self) -> str:
        return self.fields.get('type', 'unknown')
    
    @property
    def payload(self) -> Dict[str, Any]:
        """Get decompressed message payload."""
        payload_data = self.fields.get('payload', '{}')
        
        if self.compressed and isinstance(payload_data, bytes):
            # Decompress payload
            try:
                payload_str = MessageCompressor.decompress(payload_data)
                return json.loads(payload_str)
            except Exception as e:
                logger.error(f"Failed to decompress payload: {e}")
                return {}
        
        try:
            return json.loads(payload_data) if isinstance(payload_data, str) else payload_data
        except json.JSONDecodeError:
            return {}
    
    @property
    def correlation_id(self) -> str:
        return self.fields.get('correlation_id', str(uuid.uuid4()))


class OptimizedAgentMessageBroker:
    """Optimized Redis-based message broker with compression and caching."""
    
    def __init__(
        self,
        redis_client: Redis,
        compression_threshold: int = 1024,  # Compress messages > 1KB
        enable_local_cache: bool = True
    ):
        self.redis = redis_client
        self.compression_threshold = compression_threshold
        self.circuit_breaker = CircuitBreaker()
        self.metrics = RedisMetrics()
        
        # Local caching
        self.local_cache = LocalCache() if enable_local_cache else None
        
        # Message batching
        self.pending_messages = defaultdict(list)
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms
        self.batch_tasks = {}
        
        # Consumer groups cache
        self.consumer_groups: Dict[str, str] = {}
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: bool = False
    ) -> str:
        """Send optimized message with compression and batching."""
        start_time = time.time()
        
        try:
            message_id = str(uuid.uuid4())
            correlation_id = correlation_id or str(uuid.uuid4())
            
            # Serialize payload
            payload_str = json.dumps(payload)
            payload_data = payload_str
            compressed = False
            
            # Compress large payloads
            if len(payload_str) > self.compression_threshold:
                compressed_data = MessageCompressor.compress(payload_str)
                if len(compressed_data) < len(payload_str):
                    payload_data = compressed_data
                    compressed = True
                    self.metrics.compression_savings_bytes += len(payload_str) - len(compressed_data)
            
            message_data = {
                'message_id': message_id,
                'from_agent': from_agent,
                'to_agent': to_agent,
                'type': message_type,
                'payload': payload_data,
                'correlation_id': correlation_id,
                'timestamp': datetime.utcnow().isoformat(),
                'compressed': str(compressed).lower()
            }
            
            # Send high-priority messages immediately
            if priority:
                stream_id = await self._send_message_immediate(to_agent, message_data)
            else:
                # Add to batch for regular messages
                stream_id = await self._add_to_batch(to_agent, message_data)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_operations += 1
            self.metrics.successful_operations += 1
            self._update_avg_latency(latency_ms)
            
            return stream_id
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Failed to send optimized message: {e}")
            raise
    
    async def _send_message_immediate(self, to_agent: str, message_data: Dict[str, Any]) -> str:
        """Send message immediately using circuit breaker."""
        stream_name = f"agent_messages:{to_agent}"
        
        return await self.circuit_breaker.call(
            self.redis.xadd,
            stream_name,
            message_data,
            maxlen=settings.REDIS_STREAM_MAX_LEN
        )
    
    async def _add_to_batch(self, to_agent: str, message_data: Dict[str, Any]) -> str:
        """Add message to batch for efficient sending."""
        self.pending_messages[to_agent].append(message_data)
        
        # Process batch if size threshold reached
        if len(self.pending_messages[to_agent]) >= self.batch_size:
            await self._process_message_batch(to_agent)
        
        # Set timeout for batch processing
        elif to_agent not in self.batch_tasks:
            self.batch_tasks[to_agent] = asyncio.create_task(
                self._batch_timeout_handler(to_agent)
            )
        
        return f"batch_{uuid.uuid4()}"  # Temporary ID for batched messages
    
    async def _batch_timeout_handler(self, to_agent: str) -> None:
        """Handle batch timeout."""
        await asyncio.sleep(self.batch_timeout)
        if to_agent in self.pending_messages and self.pending_messages[to_agent]:
            await self._process_message_batch(to_agent)
    
    async def _process_message_batch(self, to_agent: str) -> None:
        """Process accumulated message batch."""
        if not self.pending_messages[to_agent]:
            return
        
        try:
            stream_name = f"agent_messages:{to_agent}"
            pipe = self.redis.pipeline()
            
            # Add all messages in batch
            for message_data in self.pending_messages[to_agent]:
                pipe.xadd(stream_name, message_data, maxlen=settings.REDIS_STREAM_MAX_LEN)
            
            # Execute batch
            await self.circuit_breaker.call(pipe.execute)
            
            logger.debug(f"Processed batch of {len(self.pending_messages[to_agent])} messages for {to_agent}")
            
        except Exception as e:
            logger.error(f"Failed to process message batch for {to_agent}: {e}")
        
        finally:
            # Clear batch
            self.pending_messages[to_agent].clear()
            if to_agent in self.batch_tasks:
                self.batch_tasks[to_agent].cancel()
                del self.batch_tasks[to_agent]
    
    async def read_messages_optimized(
        self,
        agent_id: str,
        consumer_name: str,
        count: int = 10,
        block: int = 1000
    ) -> List[OptimizedRedisStreamMessage]:
        """Read messages with local caching and optimization."""
        start_time = time.time()
        
        try:
            stream_name = f"agent_messages:{agent_id}"
            group_name = f"group_{agent_id}"
            
            # Check local cache first
            cache_key = f"messages:{agent_id}:{consumer_name}"
            if self.local_cache:
                cached_messages = self.local_cache.get(cache_key)
                if cached_messages:
                    self.metrics.cache_hits += 1
                    return cached_messages
                self.metrics.cache_misses += 1
            
            # Ensure consumer group exists
            await self._ensure_consumer_group(stream_name, group_name, consumer_name)
            
            # Read messages with circuit breaker
            messages = await self.circuit_breaker.call(
                self.redis.xreadgroup,
                group_name,
                consumer_name,
                {stream_name: '>'},
                count=count,
                block=block
            )
            
            # Parse messages
            parsed_messages = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Convert bytes to strings
                    str_fields = {k.decode(): v.decode() if isinstance(v, bytes) else v for k, v in fields.items()}
                    
                    # Check if message is compressed
                    compressed = str_fields.get('compressed', 'false').lower() == 'true'
                    parsed_messages.append(OptimizedRedisStreamMessage(
                        msg_id.decode(), str_fields, compressed
                    ))
            
            # Cache results
            if self.local_cache and parsed_messages:
                self.local_cache.put(cache_key, parsed_messages)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_avg_latency(latency_ms)
            self.metrics.successful_operations += 1
            
            return parsed_messages
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Failed to read optimized messages for agent {agent_id}: {e}")
            return []
    
    async def _ensure_consumer_group(self, stream_name: str, group_name: str, consumer_name: str) -> None:
        """Ensure consumer group exists with caching."""
        cache_key = f"{stream_name}:{group_name}"
        if cache_key in self.consumer_groups:
            return
        
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id='0',
                mkstream=True
            )
            self.consumer_groups[cache_key] = group_name
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group: {e}")
            else:
                self.consumer_groups[cache_key] = group_name
    
    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update average latency metric."""
        total_ops = self.metrics.successful_operations
        if total_ops == 1:
            self.metrics.avg_latency_ms = latency_ms
        else:
            self.metrics.avg_latency_ms = (
                (self.metrics.avg_latency_ms * (total_ops - 1) + latency_ms) / total_ops
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive broker metrics."""
        return {
            'total_operations': self.metrics.total_operations,
            'success_rate': self.metrics.get_success_rate(),
            'avg_latency_ms': self.metrics.avg_latency_ms,
            'cache_hit_rate': self.metrics.get_cache_hit_rate(),
            'compression_savings_mb': self.metrics.compression_savings_bytes / (1024 * 1024),
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
            'pending_batches': len(self.pending_messages),
            'local_cache_size': len(self.local_cache.cache) if self.local_cache else 0
        }


class OptimizedSessionCache:
    """Optimized session cache with local caching and compression."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.circuit_breaker = CircuitBreaker()
        self.local_cache = LocalCache(max_size=500, ttl_seconds=60)  # 1-minute local cache
        self.metrics = RedisMetrics()
        self.default_ttl = 3600
    
    async def set_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store session state with local caching."""
        try:
            key = f"session_state:{session_id}"
            ttl = ttl or self.default_ttl
            
            # Serialize and optionally compress
            state_json = json.dumps(state, default=str)
            
            # Compress large states
            if len(state_json) > 1024:
                compressed_data = MessageCompressor.compress(state_json)
                if len(compressed_data) < len(state_json):
                    state_data = compressed_data
                    # Add compression marker
                    await self.circuit_breaker.call(
                        self.redis.setex,
                        f"{key}:compressed",
                        ttl,
                        "true"
                    )
                    self.metrics.compression_savings_bytes += len(state_json) - len(compressed_data)
                else:
                    state_data = state_json
            else:
                state_data = state_json
            
            # Store in Redis
            await self.circuit_breaker.call(
                self.redis.setex,
                key,
                ttl,
                state_data
            )
            
            # Cache locally for quick access
            self.local_cache.put(session_id, state)
            
            self.metrics.successful_operations += 1
            return True
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Failed to store optimized session state: {e}")
            return False
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session state with local caching."""
        try:
            # Check local cache first
            cached_state = self.local_cache.get(session_id)
            if cached_state:
                self.metrics.cache_hits += 1
                return cached_state
            
            self.metrics.cache_misses += 1
            
            # Retrieve from Redis
            key = f"session_state:{session_id}"
            
            # Check if compressed
            is_compressed = await self.circuit_breaker.call(
                self.redis.get,
                f"{key}:compressed"
            )
            
            state_data = await self.circuit_breaker.call(
                self.redis.get,
                key
            )
            
            if not state_data:
                return None
            
            # Decompress if needed
            if is_compressed:
                state_json = MessageCompressor.decompress(state_data)
            else:
                state_json = state_data.decode('utf-8') if isinstance(state_data, bytes) else state_data
            
            state = json.loads(state_json)
            
            # Cache locally
            self.local_cache.put(session_id, state)
            
            self.metrics.successful_operations += 1
            return state
            
        except Exception as e:
            self.metrics.failed_operations += 1
            logger.error(f"Failed to retrieve optimized session state: {e}")
            return None


class DynamicConnectionPool:
    """Dynamic Redis connection pool that adapts to load."""
    
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
        """Create optimized connection pool."""
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=self.current_pool_size,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30,
            # Optimized settings
            socket_connect_timeout=5,
            socket_timeout=30,
            connection_pool_class_kwargs={
                'max_connections': self.current_pool_size
            }
        )
        return self.pool
    
    def should_scale_pool(self) -> Tuple[bool, int]:
        """Determine if pool should be scaled."""
        if not self.pool or (time.time() - self.last_adjustment) < 30:
            return False, 0
        
        # Calculate current utilization
        try:
            active_connections = len([
                conn for conn in self.pool._created_connections
                if not self.pool._available_connections.count(conn)
            ])
            utilization = active_connections / self.current_pool_size
            self.utilization_history.append(utilization)
        except:
            return False, 0
        
        if len(self.utilization_history) < 5:
            return False, 0
        
        avg_utilization = sum(self.utilization_history) / len(self.utilization_history)
        
        # Scale up if high utilization
        if avg_utilization > 0.8 and self.current_pool_size < self.max_connections:
            new_size = min(
                self.max_connections,
                int(self.current_pool_size * self.growth_factor)
            )
            adjustment = new_size - self.current_pool_size
            self.current_pool_size = new_size
            self.last_adjustment = time.time()
            return True, adjustment
        
        # Scale down if low utilization
        elif avg_utilization < 0.3 and self.current_pool_size > self.min_connections:
            new_size = max(
                self.min_connections,
                int(self.current_pool_size * 0.8)
            )
            adjustment = new_size - self.current_pool_size  # Will be negative
            self.current_pool_size = new_size
            self.last_adjustment = time.time()
            return True, adjustment
        
        return False, 0


# Global optimized instances
_optimized_redis_pool: Optional[DynamicConnectionPool] = None
_optimized_redis_client: Optional[Redis] = None


async def init_optimized_redis() -> None:
    """Initialize optimized Redis connection."""
    global _optimized_redis_pool, _optimized_redis_client
    
    try:
        logger.info("🚀 Initializing optimized Redis connection...")
        
        _optimized_redis_pool = DynamicConnectionPool(settings.REDIS_URL)
        pool = await _optimized_redis_pool.create_pool()
        _optimized_redis_client = redis.Redis(connection_pool=pool)
        
        # Test connection
        await _optimized_redis_client.ping()
        
        logger.info("✅ Optimized Redis connection established")
        
    except Exception as e:
        logger.error("❌ Failed to initialize optimized Redis", error=str(e))
        raise


async def close_optimized_redis() -> None:
    """Close optimized Redis connections."""
    global _optimized_redis_client, _optimized_redis_pool
    
    if _optimized_redis_client:
        await _optimized_redis_client.close()
    
    if _optimized_redis_pool and _optimized_redis_pool.pool:
        await _optimized_redis_pool.pool.disconnect()
    
    logger.info("🔗 Optimized Redis connections closed")


def get_optimized_redis() -> Redis:
    """Get optimized Redis client instance."""
    if _optimized_redis_client is None:
        raise RuntimeError("Optimized Redis not initialized. Call init_optimized_redis() first.")
    return _optimized_redis_client


def get_optimized_message_broker() -> OptimizedAgentMessageBroker:
    """Get optimized message broker instance."""
    return OptimizedAgentMessageBroker(get_optimized_redis())


def get_optimized_session_cache() -> OptimizedSessionCache:
    """Get optimized session cache instance."""
    return OptimizedSessionCache(get_optimized_redis())