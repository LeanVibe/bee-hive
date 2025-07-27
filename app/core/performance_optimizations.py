"""
Performance Optimizations for Redis Streams Communication System.

Provides message batching, connection multiplexing, payload compression,
and other performance enhancements to achieve 10k+ msg/sec throughput.
"""

import asyncio
import gzip
import zlib
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import io

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError

from ..models.message import StreamMessage
from ..core.config import settings

logger = structlog.get_logger()


class CompressionAlgorithm(str, Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    # Could add LZ4, Snappy, etc. in the future


@dataclass
class BatchConfig:
    """Configuration for message batching."""
    
    max_batch_size: int = 100  # Maximum messages per batch
    max_batch_wait_ms: int = 50  # Maximum wait time before sending partial batch
    min_batch_size: int = 1  # Minimum messages to trigger a batch send
    
    # Adaptive batching parameters
    adaptive_batching: bool = True
    target_latency_ms: float = 100.0  # Target batch latency
    latency_adjustment_factor: float = 0.1  # How quickly to adjust batch size


@dataclass
class CompressionConfig:
    """Configuration for payload compression."""
    
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZLIB
    compression_level: int = 6  # 1-9, higher = better compression but slower
    min_payload_size: int = 1024  # Only compress payloads larger than this
    
    # Adaptive compression
    adaptive_compression: bool = True
    cpu_usage_threshold: float = 0.8  # Disable compression if CPU usage too high


@dataclass
class ConnectionConfig:
    """Configuration for connection multiplexing."""
    
    pool_size: int = 20  # Base connection pool size
    max_connections: int = 100  # Maximum connections
    connection_timeout: float = 5.0  # Connection timeout in seconds
    
    # Adaptive scaling
    adaptive_scaling: bool = True
    scale_up_threshold: float = 0.8  # Scale up when utilization > 80%
    scale_down_threshold: float = 0.3  # Scale down when utilization < 30%
    min_idle_connections: int = 5


class MessageBatch:
    """Container for batched messages."""
    
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.messages: List[StreamMessage] = []
        self.created_at = time.time()
        self.target_streams: Dict[str, List[int]] = defaultdict(list)  # stream -> message indices
    
    def add_message(self, message: StreamMessage) -> None:
        """Add a message to the batch."""
        index = len(self.messages)
        self.messages.append(message)
        
        stream_name = message.get_stream_name()
        self.target_streams[stream_name].append(index)
    
    def size(self) -> int:
        """Get number of messages in batch."""
        return len(self.messages)
    
    def age_ms(self) -> float:
        """Get age of batch in milliseconds."""
        return (time.time() - self.created_at) * 1000
    
    def is_ready(self, config: BatchConfig) -> bool:
        """Check if batch is ready to be sent."""
        return (
            self.size() >= config.max_batch_size or
            self.age_ms() >= config.max_batch_wait_ms or
            (self.size() >= config.min_batch_size and self.age_ms() >= 10)
        )


class PayloadCompressor:
    """Handles payload compression and decompression."""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self._compression_stats = {
            "compressed_payloads": 0,
            "uncompressed_payloads": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "compression_time_ms": 0.0
        }
    
    def compress_payload(self, payload: Dict[str, Any]) -> Tuple[bytes, bool]:
        """
        Compress payload if beneficial.
        
        Returns:
            (compressed_data, was_compressed)
        """
        # Serialize payload
        payload_json = json.dumps(payload, separators=(',', ':'))
        payload_bytes = payload_json.encode('utf-8')
        
        # Check if payload is large enough to compress
        if len(payload_bytes) < self.config.min_payload_size:
            self._compression_stats["uncompressed_payloads"] += 1
            return payload_bytes, False
        
        # Skip compression if disabled
        if self.config.algorithm == CompressionAlgorithm.NONE:
            self._compression_stats["uncompressed_payloads"] += 1
            return payload_bytes, False
        
        start_time = time.time()
        
        try:
            if self.config.algorithm == CompressionAlgorithm.GZIP:
                compressed_data = gzip.compress(payload_bytes, compresslevel=self.config.compression_level)
            elif self.config.algorithm == CompressionAlgorithm.ZLIB:
                compressed_data = zlib.compress(payload_bytes, level=self.config.compression_level)
            else:
                # Fallback to uncompressed
                self._compression_stats["uncompressed_payloads"] += 1
                return payload_bytes, False
            
            compression_time = (time.time() - start_time) * 1000
            
            # Only use compression if it actually saves space
            if len(compressed_data) < len(payload_bytes):
                self._compression_stats["compressed_payloads"] += 1
                self._compression_stats["total_original_size"] += len(payload_bytes)
                self._compression_stats["total_compressed_size"] += len(compressed_data)
                self._compression_stats["compression_time_ms"] += compression_time
                
                return compressed_data, True
            else:
                # Compression didn't help
                self._compression_stats["uncompressed_payloads"] += 1
                return payload_bytes, False
        
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            self._compression_stats["uncompressed_payloads"] += 1
            return payload_bytes, False
    
    def decompress_payload(self, data: bytes, was_compressed: bool, algorithm: str = None) -> Dict[str, Any]:
        """Decompress payload if it was compressed."""
        if not was_compressed:
            # Data is not compressed
            payload_json = data.decode('utf-8')
            return json.loads(payload_json)
        
        # Determine algorithm
        algo = algorithm or self.config.algorithm.value
        
        try:
            if algo == "gzip":
                decompressed_data = gzip.decompress(data)
            elif algo == "zlib":
                decompressed_data = zlib.decompress(data)
            else:
                raise ValueError(f"Unknown compression algorithm: {algo}")
            
            payload_json = decompressed_data.decode('utf-8')
            return json.loads(payload_json)
        
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            # Try as uncompressed
            payload_json = data.decode('utf-8')
            return json.loads(payload_json)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        stats = self._compression_stats.copy()
        
        if stats["total_original_size"] > 0:
            stats["compression_ratio"] = stats["total_compressed_size"] / stats["total_original_size"]
            stats["space_saved_bytes"] = stats["total_original_size"] - stats["total_compressed_size"]
        else:
            stats["compression_ratio"] = 1.0
            stats["space_saved_bytes"] = 0
        
        if stats["compressed_payloads"] > 0:
            stats["avg_compression_time_ms"] = stats["compression_time_ms"] / stats["compressed_payloads"]
        else:
            stats["avg_compression_time_ms"] = 0.0
        
        return stats


class ConnectionManager:
    """Manages Redis connection pool with adaptive scaling."""
    
    def __init__(self, redis_url: str, config: ConnectionConfig):
        self.redis_url = redis_url
        self.config = config
        
        # Connection pools for different purposes
        self._write_pool: Optional[ConnectionPool] = None
        self._read_pool: Optional[ConnectionPool] = None
        
        # Performance tracking
        self._connection_stats = {
            "active_connections": 0,
            "peak_connections": 0,
            "connection_errors": 0,
            "total_operations": 0,
            "avg_operation_time_ms": 0.0
        }
        
        # Adaptive scaling state
        self._last_scale_time = 0.0
        self._operation_times: deque = deque(maxlen=1000)  # Track recent operation times
    
    async def initialize(self) -> None:
        """Initialize connection pools."""
        try:
            # Create separate pools for read and write operations
            self._write_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.config.pool_size,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            self._read_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.config.pool_size // 2,  # Fewer read connections typically needed
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30,
                socket_connect_timeout=self.config.connection_timeout
            )
            
            # Test connections
            async with self.get_write_client() as client:
                await client.ping()
            
            async with self.get_read_client() as client:
                await client.ping()
            
            logger.info("Connection pools initialized", 
                       write_pool_size=self.config.pool_size,
                       read_pool_size=self.config.pool_size // 2)
        
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
    
    async def close(self) -> None:
        """Close all connection pools."""
        if self._write_pool:
            await self._write_pool.disconnect()
        
        if self._read_pool:
            await self._read_pool.disconnect()
        
        logger.info("Connection pools closed")
    
    @asynccontextmanager
    async def get_write_client(self):
        """Get a Redis client for write operations."""
        if not self._write_pool:
            raise RuntimeError("Connection pools not initialized")
        
        client = redis.Redis(connection_pool=self._write_pool)
        start_time = time.time()
        
        try:
            self._connection_stats["active_connections"] += 1
            self._connection_stats["peak_connections"] = max(
                self._connection_stats["peak_connections"],
                self._connection_stats["active_connections"]
            )
            
            yield client
            
            # Track operation time
            operation_time = (time.time() - start_time) * 1000
            self._operation_times.append(operation_time)
            self._connection_stats["total_operations"] += 1
            
        except Exception as e:
            self._connection_stats["connection_errors"] += 1
            logger.error(f"Error in write client: {e}")
            raise
        finally:
            self._connection_stats["active_connections"] -= 1
            await client.close()
    
    @asynccontextmanager
    async def get_read_client(self):
        """Get a Redis client for read operations."""
        if not self._read_pool:
            raise RuntimeError("Connection pools not initialized")
        
        client = redis.Redis(connection_pool=self._read_pool)
        start_time = time.time()
        
        try:
            self._connection_stats["active_connections"] += 1
            yield client
            
            # Track operation time
            operation_time = (time.time() - start_time) * 1000
            self._operation_times.append(operation_time)
            self._connection_stats["total_operations"] += 1
            
        except Exception as e:
            self._connection_stats["connection_errors"] += 1
            logger.error(f"Error in read client: {e}")
            raise
        finally:
            self._connection_stats["active_connections"] -= 1
            await client.close()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        stats = self._connection_stats.copy()
        
        if self._operation_times:
            stats["avg_operation_time_ms"] = sum(self._operation_times) / len(self._operation_times)
            stats["p95_operation_time_ms"] = sorted(self._operation_times)[int(len(self._operation_times) * 0.95)]
            stats["p99_operation_time_ms"] = sorted(self._operation_times)[int(len(self._operation_times) * 0.99)]
        
        return stats


class HighPerformanceMessageBroker:
    """
    High-performance message broker with batching, compression, and connection multiplexing.
    
    Designed to achieve 10k+ msg/sec throughput with sub-200ms latency.
    """
    
    def __init__(
        self,
        redis_url: str,
        batch_config: Optional[BatchConfig] = None,
        compression_config: Optional[CompressionConfig] = None,
        connection_config: Optional[ConnectionConfig] = None
    ):
        self.redis_url = redis_url
        self.batch_config = batch_config or BatchConfig()
        self.compression_config = compression_config or CompressionConfig()
        self.connection_config = connection_config or ConnectionConfig()
        
        # Components
        self.connection_manager = ConnectionManager(redis_url, self.connection_config)
        self.compressor = PayloadCompressor(self.compression_config)
        
        # Batching state
        self._pending_batches: Dict[str, MessageBatch] = {}  # stream -> batch
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._batch_sender_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self._performance_metrics = {
            "messages_sent": 0,
            "batches_sent": 0,
            "total_latency_ms": 0.0,
            "throughput_samples": deque(maxlen=100),
            "latency_samples": deque(maxlen=1000)
        }
        
        # Adaptive parameters
        self._current_batch_size = self.batch_config.max_batch_size
        self._last_adaptation_time = time.time()
    
    async def start(self) -> None:
        """Start the high-performance message broker."""
        try:
            # Initialize connection manager
            await self.connection_manager.initialize()
            
            # Start batch processing tasks
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self._batch_sender_task = asyncio.create_task(self._batch_sender_loop())
            
            logger.info("High-performance message broker started")
        
        except Exception as e:
            logger.error(f"Failed to start high-performance message broker: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the message broker."""
        # Cancel batch processing tasks
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        
        if self._batch_sender_task:
            self._batch_sender_task.cancel()
        
        # Wait for tasks to complete
        tasks = [self._batch_processor_task, self._batch_sender_task]
        completed_tasks = [t for t in tasks if t is not None]
        if completed_tasks:
            await asyncio.gather(*completed_tasks, return_exceptions=True)
        
        # Send any remaining batches
        await self._flush_pending_batches()
        
        # Close connection manager
        await self.connection_manager.close()
        
        logger.info("High-performance message broker stopped")
    
    async def send_message(self, message: StreamMessage) -> str:
        """Send a message with high-performance optimizations."""
        start_time = time.time()
        
        try:
            # Compress payload if beneficial
            compressed_payload, was_compressed = self.compressor.compress_payload(message.payload)
            
            # Create optimized message
            optimized_message = message.copy()
            if was_compressed:
                # Store compressed payload with metadata
                optimized_message.payload = {
                    "_compressed": True,
                    "_algorithm": self.compression_config.algorithm.value,
                    "_data": compressed_payload.hex()  # Store as hex string
                }
            
            # Add to batch
            await self._add_to_batch(optimized_message)
            
            # Track metrics
            send_time = (time.time() - start_time) * 1000
            self._performance_metrics["messages_sent"] += 1
            self._performance_metrics["total_latency_ms"] += send_time
            self._performance_metrics["latency_samples"].append(send_time)
            
            return f"batched_{message.id}"
        
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    async def _add_to_batch(self, message: StreamMessage) -> None:
        """Add message to appropriate batch."""
        stream_name = message.get_stream_name()
        
        # Get or create batch for this stream
        if stream_name not in self._pending_batches:
            batch_id = f"batch_{int(time.time() * 1000000)}_{stream_name}"
            self._pending_batches[stream_name] = MessageBatch(batch_id)
        
        batch = self._pending_batches[stream_name]
        batch.add_message(message)
        
        # Check if batch is ready
        if batch.is_ready(self.batch_config):
            await self._queue_batch_for_sending(stream_name, batch)
    
    async def _queue_batch_for_sending(self, stream_name: str, batch: MessageBatch) -> None:
        """Queue a batch for sending."""
        # Remove from pending batches
        if stream_name in self._pending_batches:
            del self._pending_batches[stream_name]
        
        # Queue for sending
        await self._batch_queue.put((stream_name, batch))
    
    async def _batch_processor_loop(self) -> None:
        """Process batches that are ready to send."""
        while True:
            try:
                # Check for ready batches every 10ms
                await asyncio.sleep(0.01)
                
                ready_streams = []
                for stream_name, batch in self._pending_batches.items():
                    if batch.is_ready(self.batch_config):
                        ready_streams.append(stream_name)
                
                # Queue ready batches
                for stream_name in ready_streams:
                    batch = self._pending_batches[stream_name]
                    await self._queue_batch_for_sending(stream_name, batch)
                
                # Adaptive batch size adjustment
                await self._adapt_batch_parameters()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor loop: {e}")
    
    async def _batch_sender_loop(self) -> None:
        """Send queued batches to Redis."""
        while True:
            try:
                # Get batch from queue
                stream_name, batch = await self._batch_queue.get()
                
                # Send batch
                await self._send_batch(stream_name, batch)
                
                # Mark task done
                self._batch_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch sender loop: {e}")
    
    async def _send_batch(self, stream_name: str, batch: MessageBatch) -> None:
        """Send a batch of messages to Redis."""
        start_time = time.time()
        
        try:
            async with self.connection_manager.get_write_client() as client:
                # Prepare pipeline for batch operations
                pipe = client.pipeline()
                
                # Add all messages in batch to pipeline
                for message in batch.messages:
                    redis_data = message.to_redis_dict()
                    pipe.xadd(
                        stream_name,
                        redis_data,
                        maxlen=settings.REDIS_STREAM_MAX_LEN,
                        approximate=True
                    )
                
                # Execute pipeline
                results = await pipe.execute()
                
                # Track metrics
                batch_time = (time.time() - start_time) * 1000
                self._performance_metrics["batches_sent"] += 1
                
                # Calculate throughput
                throughput = batch.size() / (batch_time / 1000.0) if batch_time > 0 else 0
                self._performance_metrics["throughput_samples"].append(throughput)
                
                logger.debug(
                    "Batch sent successfully",
                    stream=stream_name,
                    batch_size=batch.size(),
                    batch_time_ms=batch_time,
                    throughput_msg_per_sec=throughput
                )
        
        except Exception as e:
            logger.error(f"Error sending batch to {stream_name}: {e}")
            
            # Fallback: send messages individually
            await self._send_batch_individually(stream_name, batch)
    
    async def _send_batch_individually(self, stream_name: str, batch: MessageBatch) -> None:
        """Fallback: send batch messages individually."""
        async with self.connection_manager.get_write_client() as client:
            for message in batch.messages:
                try:
                    redis_data = message.to_redis_dict()
                    await client.xadd(
                        stream_name,
                        redis_data,
                        maxlen=settings.REDIS_STREAM_MAX_LEN,
                        approximate=True
                    )
                except Exception as e:
                    logger.error(f"Error sending individual message: {e}")
    
    async def _adapt_batch_parameters(self) -> None:
        """Adapt batch parameters based on performance."""
        if not self.batch_config.adaptive_batching:
            return
        
        current_time = time.time()
        if current_time - self._last_adaptation_time < 10.0:  # Adapt every 10 seconds
            return
        
        # Calculate recent latency
        if self._performance_metrics["latency_samples"]:
            recent_latency = sum(list(self._performance_metrics["latency_samples"])[-50:]) / min(50, len(self._performance_metrics["latency_samples"]))
        else:
            recent_latency = 0
        
        # Adjust batch size based on latency
        if recent_latency > self.batch_config.target_latency_ms:
            # Latency too high, reduce batch size
            new_batch_size = max(
                self.batch_config.min_batch_size,
                int(self._current_batch_size * (1 - self.batch_config.latency_adjustment_factor))
            )
        elif recent_latency < self.batch_config.target_latency_ms * 0.5:
            # Latency very low, can increase batch size
            new_batch_size = min(
                self.batch_config.max_batch_size,
                int(self._current_batch_size * (1 + self.batch_config.latency_adjustment_factor))
            )
        else:
            new_batch_size = self._current_batch_size
        
        if new_batch_size != self._current_batch_size:
            logger.info(
                "Adapted batch size",
                old_size=self._current_batch_size,
                new_size=new_batch_size,
                recent_latency_ms=recent_latency
            )
            self._current_batch_size = new_batch_size
        
        self._last_adaptation_time = current_time
    
    async def _flush_pending_batches(self) -> None:
        """Flush all pending batches."""
        for stream_name, batch in list(self._pending_batches.items()):
            if batch.size() > 0:
                await self._send_batch(stream_name, batch)
        
        self._pending_batches.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self._performance_metrics.copy()
        
        # Calculate throughput statistics
        if self._performance_metrics["throughput_samples"]:
            throughput_samples = list(self._performance_metrics["throughput_samples"])
            metrics["avg_throughput_msg_per_sec"] = sum(throughput_samples) / len(throughput_samples)
            metrics["peak_throughput_msg_per_sec"] = max(throughput_samples)
        else:
            metrics["avg_throughput_msg_per_sec"] = 0.0
            metrics["peak_throughput_msg_per_sec"] = 0.0
        
        # Calculate latency statistics
        if self._performance_metrics["latency_samples"]:
            latency_samples = sorted(list(self._performance_metrics["latency_samples"]))
            n = len(latency_samples)
            metrics["avg_latency_ms"] = sum(latency_samples) / n
            metrics["p50_latency_ms"] = latency_samples[n // 2]
            metrics["p95_latency_ms"] = latency_samples[int(n * 0.95)]
            metrics["p99_latency_ms"] = latency_samples[int(n * 0.99)]
        else:
            metrics["avg_latency_ms"] = 0.0
            metrics["p50_latency_ms"] = 0.0
            metrics["p95_latency_ms"] = 0.0
            metrics["p99_latency_ms"] = 0.0
        
        # Add compression stats
        metrics["compression_stats"] = self.compressor.get_compression_stats()
        
        # Add connection stats
        metrics["connection_stats"] = self.connection_manager.get_connection_stats()
        
        # Current configuration
        metrics["current_batch_size"] = self._current_batch_size
        metrics["pending_batches"] = len(self._pending_batches)
        metrics["queue_size"] = self._batch_queue.qsize()
        
        return metrics
    
    async def force_flush(self) -> int:
        """Force flush all pending batches and return count."""
        batch_count = len(self._pending_batches)
        await self._flush_pending_batches()
        return batch_count