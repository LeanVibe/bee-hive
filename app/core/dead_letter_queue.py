"""
Dead Letter Queue (DLQ) Management for Redis Streams Communication System.

Provides comprehensive retry handling, monitoring, and replay mechanisms for failed messages
with production-grade reliability and observability.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..models.message import StreamMessage, MessageStatus, MessageType, MessagePriority
from ..core.config import settings

logger = structlog.get_logger()


class DLQPolicy(str, Enum):
    """Dead letter queue handling policies."""
    IMMEDIATE = "immediate"  # Move to DLQ on first failure
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with exponential backoff
    LINEAR_BACKOFF = "linear_backoff"  # Retry with linear backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker pattern


@dataclass
class DLQConfiguration:
    """Configuration for dead letter queue behavior."""
    
    max_retries: int = 3
    initial_retry_delay_ms: int = 1000  # 1 second
    max_retry_delay_ms: int = 60000  # 1 minute
    dlq_max_size: int = 100000  # Maximum DLQ entries
    dlq_ttl_hours: int = 72  # DLQ entries TTL
    policy: DLQPolicy = DLQPolicy.EXPONENTIAL_BACKOFF
    circuit_breaker_threshold: float = 0.5  # 50% failure rate triggers circuit breaker
    circuit_breaker_window_minutes: int = 5
    
    # Monitoring and alerting
    monitor_enabled: bool = True
    alert_threshold: int = 1000  # Alert when DLQ size exceeds this
    alert_callback: Optional[Callable[[int, str], None]] = None


@dataclass
class DLQEntry:
    """Represents an entry in the dead letter queue."""
    
    original_stream: str
    original_message_id: str
    message: StreamMessage
    failure_reason: str
    retry_count: int
    first_failure_time: float
    last_failure_time: float
    next_retry_time: Optional[float] = None
    dlq_entry_id: Optional[str] = None
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert DLQ entry to Redis-compatible dictionary."""
        return {
            "original_stream": self.original_stream,
            "original_message_id": self.original_message_id,
            "message_data": self.message.json(),
            "failure_reason": self.failure_reason,
            "retry_count": str(self.retry_count),
            "first_failure_time": str(self.first_failure_time),
            "last_failure_time": str(self.last_failure_time),
            "next_retry_time": str(self.next_retry_time) if self.next_retry_time else "",
            "entry_timestamp": str(time.time())
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, str], entry_id: str) -> "DLQEntry":
        """Create DLQ entry from Redis dictionary."""
        message = StreamMessage.parse_raw(data["message_data"])
        
        return cls(
            original_stream=data["original_stream"],
            original_message_id=data["original_message_id"],
            message=message,
            failure_reason=data["failure_reason"],
            retry_count=int(data["retry_count"]),
            first_failure_time=float(data["first_failure_time"]),
            last_failure_time=float(data["last_failure_time"]),
            next_retry_time=float(data["next_retry_time"]) if data["next_retry_time"] else None,
            dlq_entry_id=entry_id
        )


class DeadLetterQueueManager:
    """
    Comprehensive Dead Letter Queue management for Redis Streams.
    
    Handles failed message retry logic, DLQ storage, monitoring, and replay mechanisms
    with production-grade reliability and observability features.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: Optional[DLQConfiguration] = None
    ):
        self.redis = redis_client
        self.config = config or DLQConfiguration()
        
        # DLQ stream names
        self.dlq_stream = "dead_letter_queue"
        self.retry_stream = "retry_queue"
        self.dlq_stats_key = "dlq:stats"
        
        # Circuit breaker state
        self._circuit_breaker_state: Dict[str, Dict] = {}
        
        # Performance metrics
        self._metrics = {
            "messages_retried": 0,
            "messages_moved_to_dlq": 0,
            "successful_replays": 0,
            "failed_replays": 0,
            "dlq_size": 0,
            "retry_queue_size": 0
        }
        
        # Background tasks
        self._retry_processor_task: Optional[asyncio.Task] = None
        self._dlq_monitor_task: Optional[asyncio.Task] = None
        self._circuit_breaker_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start DLQ background processing tasks."""
        try:
            # Start retry processor
            self._retry_processor_task = asyncio.create_task(
                self._retry_processor_loop()
            )
            
            # Start DLQ monitor
            if self.config.monitor_enabled:
                self._dlq_monitor_task = asyncio.create_task(
                    self._dlq_monitor_loop()
                )
            
            # Start circuit breaker monitor
            self._circuit_breaker_task = asyncio.create_task(
                self._circuit_breaker_monitor_loop()
            )
            
            logger.info("DLQ Manager started with background tasks")
            
        except Exception as e:
            logger.error("Failed to start DLQ Manager", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop DLQ background processing tasks."""
        tasks = [
            self._retry_processor_task,
            self._dlq_monitor_task,
            self._circuit_breaker_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        completed_tasks = [t for t in tasks if t is not None]
        if completed_tasks:
            await asyncio.gather(*completed_tasks, return_exceptions=True)
        
        logger.info("DLQ Manager stopped")
    
    async def handle_failed_message(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        current_retry_count: int = 0
    ) -> bool:
        """
        Handle a failed message according to DLQ policy.
        
        Returns True if message should be retried, False if moved to DLQ.
        """
        try:
            # Check circuit breaker state
            if self._is_circuit_breaker_open(original_stream):
                logger.warning(
                    "Circuit breaker open, moving message to DLQ",
                    stream=original_stream,
                    message_id=original_message_id
                )
                await self._move_to_dlq(
                    original_stream, original_message_id, message, 
                    f"Circuit breaker open: {failure_reason}", current_retry_count
                )
                return False
            
            # Check if message should be retried
            if current_retry_count < self.config.max_retries:
                # Calculate next retry time based on policy
                next_retry_time = self._calculate_next_retry_time(
                    current_retry_count, self.config.policy
                )
                
                # Add to retry queue
                await self._add_to_retry_queue(
                    original_stream, original_message_id, message,
                    failure_reason, current_retry_count, next_retry_time
                )
                
                self._metrics["messages_retried"] += 1
                
                logger.info(
                    "Message scheduled for retry",
                    stream=original_stream,
                    message_id=original_message_id,
                    retry_count=current_retry_count + 1,
                    next_retry=datetime.fromtimestamp(next_retry_time).isoformat()
                )
                
                return True
            else:
                # Max retries exceeded, move to DLQ
                await self._move_to_dlq(
                    original_stream, original_message_id, message,
                    f"Max retries exceeded: {failure_reason}", current_retry_count
                )
                
                self._metrics["messages_moved_to_dlq"] += 1
                
                logger.warning(
                    "Message moved to DLQ after max retries",
                    stream=original_stream,
                    message_id=original_message_id,
                    retry_count=current_retry_count
                )
                
                return False
                
        except Exception as e:
            logger.error(
                "Error handling failed message",
                stream=original_stream,
                message_id=original_message_id,
                error=str(e)
            )
            
            # Fallback: move to DLQ
            await self._move_to_dlq(
                original_stream, original_message_id, message,
                f"DLQ handler error: {str(e)}", current_retry_count
            )
            return False
    
    async def _add_to_retry_queue(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        retry_count: int,
        next_retry_time: float
    ) -> None:
        """Add message to retry queue with scheduled retry time."""
        current_time = time.time()
        
        dlq_entry = DLQEntry(
            original_stream=original_stream,
            original_message_id=original_message_id,
            message=message,
            failure_reason=failure_reason,
            retry_count=retry_count + 1,
            first_failure_time=current_time,
            last_failure_time=current_time,
            next_retry_time=next_retry_time
        )
        
        # Add to retry stream with score as retry time for sorted retrieval
        await self.redis.zadd(
            self.retry_stream,
            {json.dumps(dlq_entry.to_redis_dict()): next_retry_time}
        )
    
    async def _move_to_dlq(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        retry_count: int
    ) -> None:
        """Move message to dead letter queue."""
        current_time = time.time()
        
        dlq_entry = DLQEntry(
            original_stream=original_stream,
            original_message_id=original_message_id,
            message=message,
            failure_reason=failure_reason,
            retry_count=retry_count,
            first_failure_time=current_time,
            last_failure_time=current_time
        )
        
        # Add to DLQ stream
        dlq_id = await self.redis.xadd(
            self.dlq_stream,
            dlq_entry.to_redis_dict(),
            maxlen=self.config.dlq_max_size,
            approximate=True
        )
        
        # Update DLQ size metric
        self._metrics["dlq_size"] = await self.redis.xlen(self.dlq_stream)
        
        # Update failure stats for circuit breaker
        await self._update_failure_stats(original_stream)
    
    def _calculate_next_retry_time(self, retry_count: int, policy: DLQPolicy) -> float:
        """Calculate next retry time based on policy."""
        current_time = time.time()
        
        if policy == DLQPolicy.IMMEDIATE:
            return current_time
        elif policy == DLQPolicy.LINEAR_BACKOFF:
            delay_ms = self.config.initial_retry_delay_ms * (retry_count + 1)
        elif policy == DLQPolicy.EXPONENTIAL_BACKOFF:
            delay_ms = self.config.initial_retry_delay_ms * (2 ** retry_count)
        else:  # CIRCUIT_BREAKER
            delay_ms = self.config.initial_retry_delay_ms * (2 ** retry_count)
        
        # Cap at max delay
        delay_ms = min(delay_ms, self.config.max_retry_delay_ms)
        
        return current_time + (delay_ms / 1000.0)
    
    async def _retry_processor_loop(self) -> None:
        """Background loop to process retry queue."""
        while True:
            try:
                current_time = time.time()
                
                # Get messages ready for retry (score <= current_time)
                retry_entries = await self.redis.zrangebyscore(
                    self.retry_stream,
                    min=0,
                    max=current_time,
                    withscores=True,
                    start=0,
                    num=100  # Process up to 100 retries per batch
                )
                
                for entry_data, score in retry_entries:
                    try:
                        # Parse retry entry
                        entry_dict = json.loads(entry_data)
                        dlq_entry = DLQEntry.from_redis_dict(entry_dict, "")
                        
                        # Attempt to replay message
                        success = await self._replay_message(dlq_entry)
                        
                        # Remove from retry queue
                        await self.redis.zrem(self.retry_stream, entry_data)
                        
                        if success:
                            self._metrics["successful_replays"] += 1
                            logger.debug(
                                "Message replay successful",
                                stream=dlq_entry.original_stream,
                                message_id=dlq_entry.original_message_id
                            )
                        else:
                            # Failed again, handle according to policy
                            await self.handle_failed_message(
                                dlq_entry.original_stream,
                                dlq_entry.original_message_id,
                                dlq_entry.message,
                                f"Retry failed: {dlq_entry.failure_reason}",
                                dlq_entry.retry_count
                            )
                            self._metrics["failed_replays"] += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing retry entry: {e}")
                        # Remove malformed entry
                        await self.redis.zrem(self.retry_stream, entry_data)
                
                # Update retry queue size metric
                self._metrics["retry_queue_size"] = await self.redis.zcard(self.retry_stream)
                
                # Sleep before next iteration
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in retry processor loop: {e}")
                await asyncio.sleep(5)
    
    async def _replay_message(self, dlq_entry: DLQEntry) -> bool:
        """
        Attempt to replay a message to its original stream.
        
        Returns True if successful, False if failed.
        """
        try:
            # Re-add message to original stream
            await self.redis.xadd(
                dlq_entry.original_stream,
                dlq_entry.message.to_redis_dict(),
                maxlen=settings.REDIS_STREAM_MAX_LEN,
                approximate=True
            )
            
            logger.debug(
                "Message replayed to original stream",
                stream=dlq_entry.original_stream,
                message_id=dlq_entry.original_message_id
            )
            
            return True
            
        except RedisError as e:
            logger.error(
                "Failed to replay message",
                stream=dlq_entry.original_stream,
                message_id=dlq_entry.original_message_id,
                error=str(e)
            )
            return False
    
    async def _dlq_monitor_loop(self) -> None:
        """Background loop to monitor DLQ size and trigger alerts."""
        while True:
            try:
                # Update DLQ metrics
                self._metrics["dlq_size"] = await self.redis.xlen(self.dlq_stream)
                
                # Check alert threshold
                if (self._metrics["dlq_size"] > self.config.alert_threshold and 
                    self.config.alert_callback):
                    self.config.alert_callback(
                        self._metrics["dlq_size"],
                        f"DLQ size ({self._metrics['dlq_size']}) exceeded threshold ({self.config.alert_threshold})"
                    )
                
                # Clean up old DLQ entries
                await self._cleanup_old_dlq_entries()
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in DLQ monitor loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_dlq_entries(self) -> None:
        """Remove old DLQ entries based on TTL configuration."""
        try:
            cutoff_time = time.time() - (self.config.dlq_ttl_hours * 3600)
            
            # Get old entries
            old_entries = await self.redis.xrange(
                self.dlq_stream,
                min="-",
                max=f"{int(cutoff_time * 1000)}-0",
                count=1000  # Process up to 1000 entries per cleanup
            )
            
            if old_entries:
                # Delete old entries
                entry_ids = [entry_id for entry_id, _ in old_entries]
                await self.redis.xdel(self.dlq_stream, *entry_ids)
                
                logger.info(f"Cleaned up {len(entry_ids)} old DLQ entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up old DLQ entries: {e}")
    
    async def _circuit_breaker_monitor_loop(self) -> None:
        """Monitor circuit breaker state and recovery."""
        while True:
            try:
                current_time = time.time()
                window_start = current_time - (self.config.circuit_breaker_window_minutes * 60)
                
                # Check each stream's failure rate
                for stream_name in list(self._circuit_breaker_state.keys()):
                    cb_state = self._circuit_breaker_state[stream_name]
                    
                    if cb_state["state"] == "open":
                        # Check if circuit breaker should transition to half-open
                        if current_time >= cb_state["next_attempt_time"]:
                            cb_state["state"] = "half-open"
                            cb_state["half_open_attempts"] = 0
                            logger.info(f"Circuit breaker half-open for stream {stream_name}")
                    
                    elif cb_state["state"] == "half-open":
                        # Reset to closed if enough successful attempts
                        if cb_state["half_open_attempts"] >= 5:  # 5 successful attempts
                            cb_state["state"] = "closed"
                            cb_state["failure_count"] = 0
                            logger.info(f"Circuit breaker closed for stream {stream_name}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitor: {e}")
                await asyncio.sleep(30)
    
    async def _update_failure_stats(self, stream_name: str) -> None:
        """Update failure statistics for circuit breaker."""
        current_time = time.time()
        
        if stream_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[stream_name] = {
                "state": "closed",  # closed, open, half-open
                "failure_count": 0,
                "last_failure_time": current_time,
                "next_attempt_time": None,
                "half_open_attempts": 0
            }
        
        cb_state = self._circuit_breaker_state[stream_name]
        cb_state["failure_count"] += 1
        cb_state["last_failure_time"] = current_time
        
        # Check if circuit breaker should open
        if (cb_state["state"] == "closed" and 
            cb_state["failure_count"] >= 10):  # 10 failures trigger circuit breaker
            
            cb_state["state"] = "open"
            cb_state["next_attempt_time"] = current_time + 60  # Try again in 1 minute
            
            logger.warning(f"Circuit breaker opened for stream {stream_name}")
    
    def _is_circuit_breaker_open(self, stream_name: str) -> bool:
        """Check if circuit breaker is open for a stream."""
        if stream_name not in self._circuit_breaker_state:
            return False
        
        return self._circuit_breaker_state[stream_name]["state"] == "open"
    
    async def get_dlq_stats(self) -> Dict[str, Any]:
        """Get comprehensive DLQ statistics."""
        dlq_size = await self.redis.xlen(self.dlq_stream)
        retry_size = await self.redis.zcard(self.retry_stream)
        
        return {
            "dlq_size": dlq_size,
            "retry_queue_size": retry_size,
            "metrics": self._metrics.copy(),
            "circuit_breaker_states": {
                stream: state["state"] 
                for stream, state in self._circuit_breaker_state.items()
            },
            "configuration": {
                "max_retries": self.config.max_retries,
                "policy": self.config.policy.value,
                "dlq_max_size": self.config.dlq_max_size,
                "dlq_ttl_hours": self.config.dlq_ttl_hours
            }
        }
    
    async def replay_dlq_messages(
        self,
        stream_filter: Optional[str] = None,
        message_type_filter: Optional[MessageType] = None,
        max_messages: int = 100
    ) -> int:
        """
        Manually replay messages from DLQ back to their original streams.
        
        Returns number of messages successfully replayed.
        """
        replayed_count = 0
        
        try:
            # Get DLQ entries
            entries = await self.redis.xrange(
                self.dlq_stream,
                min="-",
                max="+",
                count=max_messages
            )
            
            for entry_id, fields in entries:
                try:
                    # Convert fields to DLQ entry
                    str_fields = {k.decode() if isinstance(k, bytes) else k: 
                                v.decode() if isinstance(v, bytes) else v 
                                for k, v in fields.items()}
                    
                    dlq_entry = DLQEntry.from_redis_dict(str_fields, entry_id.decode())
                    
                    # Apply filters
                    if stream_filter and stream_filter not in dlq_entry.original_stream:
                        continue
                    
                    if (message_type_filter and 
                        dlq_entry.message.message_type != message_type_filter):
                        continue
                    
                    # Attempt replay
                    if await self._replay_message(dlq_entry):
                        # Remove from DLQ on successful replay
                        await self.redis.xdel(self.dlq_stream, entry_id)
                        replayed_count += 1
                        
                        logger.info(
                            "Manual DLQ replay successful",
                            stream=dlq_entry.original_stream,
                            message_id=dlq_entry.original_message_id
                        )
                
                except Exception as e:
                    logger.error(f"Error replaying DLQ entry {entry_id}: {e}")
            
            logger.info(f"Manual DLQ replay completed: {replayed_count} messages replayed")
            return replayed_count
            
        except Exception as e:
            logger.error(f"Error in manual DLQ replay: {e}")
            return replayed_count
    
    async def get_dlq_entries(
        self,
        start: str = "-",
        end: str = "+",
        count: int = 100
    ) -> List[DLQEntry]:
        """Get DLQ entries for inspection."""
        entries = []
        
        try:
            redis_entries = await self.redis.xrange(
                self.dlq_stream,
                min=start,
                max=end,
                count=count
            )
            
            for entry_id, fields in redis_entries:
                str_fields = {k.decode() if isinstance(k, bytes) else k: 
                            v.decode() if isinstance(v, bytes) else v 
                            for k, v in fields.items()}
                
                dlq_entry = DLQEntry.from_redis_dict(str_fields, entry_id.decode())
                entries.append(dlq_entry)
            
        except Exception as e:
            logger.error(f"Error getting DLQ entries: {e}")
        
        return entries