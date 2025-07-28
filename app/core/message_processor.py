"""
Message Processor for LeanVibe Agent Hive 2.0

Handles priority queuing, TTL management, batch processing, and message routing
for optimal communication performance and reliability.

Implements priority-based message processing with TTL expiration and dead letter
handling to meet Communication PRD performance targets.
"""

import asyncio
import heapq
import logging
import time
from asyncio import Queue, PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum, IntEnum
import uuid
from collections import defaultdict, deque

from ..models.message import StreamMessage, MessageType, MessagePriority, MessageStatus
from .redis_pubsub_manager import RedisPubSubManager, MessageProcessingResult

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Message processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


class PriorityLevel(IntEnum):
    """Priority levels for message processing (lower number = higher priority)."""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class PriorityMessage:
    """Wrapper for messages in priority queue."""
    priority: int
    sequence: int  # For FIFO ordering within same priority
    timestamp: float
    message: StreamMessage
    retry_count: int = 0
    max_retries: int = 3
    next_retry_time: Optional[float] = None
    processing_started: Optional[float] = None
    
    def __lt__(self, other):
        """Compare for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.message.is_expired()
    
    def is_retryable(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries
    
    def calculate_retry_delay(self) -> float:
        """Calculate exponential backoff delay for retry."""
        base_delay = 1.0  # 1 second
        max_delay = 60.0  # 1 minute
        delay = min(base_delay * (2 ** self.retry_count), max_delay)
        return delay


@dataclass
class ProcessingMetrics:
    """Metrics for message processing performance."""
    total_processed: int = 0
    total_failed: int = 0
    total_expired: int = 0
    total_dead_letter: int = 0
    total_retries: int = 0
    average_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    throughput_msg_per_sec: float = 0.0
    queue_depth: int = 0
    backlog_age_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
            "total_expired": self.total_expired,
            "total_dead_letter": self.total_dead_letter,
            "total_retries": self.total_retries,
            "average_processing_time_ms": self.average_processing_time_ms,
            "p95_processing_time_ms": self.p95_processing_time_ms,
            "p99_processing_time_ms": self.p99_processing_time_ms,
            "throughput_msg_per_sec": self.throughput_msg_per_sec,
            "queue_depth": self.queue_depth,
            "backlog_age_seconds": self.backlog_age_seconds
        }


class MessageProcessor:
    """
    High-performance message processor with priority queuing and TTL management.
    
    Features:
    - Priority-based message processing
    - TTL expiration handling
    - Exponential backoff retry logic
    - Dead letter queue integration
    - Batch processing for efficiency
    - Comprehensive performance metrics
    - Circuit breaker pattern for resilience
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        batch_size: int = 10,
        max_processing_time_seconds: int = 30,
        ttl_check_interval_seconds: int = 5,
        metrics_collection_interval_seconds: int = 10,
        dead_letter_handler: Optional[Callable[[StreamMessage, Exception], None]] = None,
        max_concurrent_processors: int = 5
    ):
        """
        Initialize message processor.
        
        Args:
            max_queue_size: Maximum number of messages in queue
            batch_size: Number of messages to process in batch
            max_processing_time_seconds: Timeout for message processing
            ttl_check_interval_seconds: Interval for TTL expiration checks
            metrics_collection_interval_seconds: Interval for metrics updates
            dead_letter_handler: Handler for dead letter messages
            max_concurrent_processors: Maximum concurrent processing tasks
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.max_processing_time_seconds = max_processing_time_seconds
        self.ttl_check_interval_seconds = ttl_check_interval_seconds
        self.metrics_collection_interval_seconds = metrics_collection_interval_seconds
        self.dead_letter_handler = dead_letter_handler
        self.max_concurrent_processors = max_concurrent_processors
        
        # Priority queue for messages
        self._priority_queue: List[PriorityMessage] = []
        self._queue_lock = asyncio.Lock()
        self._sequence_counter = 0
        
        # Processing management
        self._processing_tasks: Set[asyncio.Task] = set()
        self._message_handlers: Dict[MessageType, Callable[[StreamMessage], Any]] = {}
        self._retry_queue: Queue = Queue()
        
        # Performance tracking
        self._metrics = ProcessingMetrics()
        self._processing_times: deque = deque(maxlen=1000)  # Last 1000 processing times
        self._start_time = time.time()
        
        # Status tracking
        self._running = False
        self._processor_tasks: List[asyncio.Task] = []
        self._ttl_cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._retry_processor_task: Optional[asyncio.Task] = None
        
        # Priority mapping
        self._priority_mapping = {
            MessagePriority.URGENT: PriorityLevel.URGENT,
            MessagePriority.HIGH: PriorityLevel.HIGH,
            MessagePriority.NORMAL: PriorityLevel.NORMAL,
            MessagePriority.LOW: PriorityLevel.LOW
        }
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[StreamMessage], Any]
    ) -> None:
        """
        Register handler for specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to process the message
        """
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type.value}")
    
    def unregister_handler(self, message_type: MessageType) -> None:
        """Unregister handler for message type."""
        if message_type in self._message_handlers:
            del self._message_handlers[message_type]
            logger.info(f"Unregistered handler for message type: {message_type.value}")
    
    async def start(self) -> None:
        """Start the message processor."""
        if self._running:
            logger.warning("Message processor is already running")
            return
        
        self._running = True
        
        # Start processor tasks
        for i in range(self.max_concurrent_processors):
            task = asyncio.create_task(self._processor_loop(f"processor-{i}"))
            self._processor_tasks.append(task)
        
        # Start TTL cleanup task
        self._ttl_cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())
        
        # Start metrics collection task
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        # Start retry processor task
        self._retry_processor_task = asyncio.create_task(self._retry_processor_loop())
        
        logger.info(
            f"Started message processor with {self.max_concurrent_processors} processors",
            extra={
                "max_queue_size": self.max_queue_size,
                "batch_size": self.batch_size,
                "max_processing_time": self.max_processing_time_seconds
            }
        )
    
    async def stop(self) -> None:
        """Stop the message processor."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all tasks
        all_tasks = (
            self._processor_tasks +
            ([self._ttl_cleanup_task] if self._ttl_cleanup_task else []) +
            ([self._metrics_task] if self._metrics_task else []) +
            ([self._retry_processor_task] if self._retry_processor_task else [])
        )
        
        for task in all_tasks:
            if task and not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        logger.info("Stopped message processor")
    
    async def enqueue_message(self, message: StreamMessage) -> bool:
        """
        Add message to processing queue.
        
        Args:
            message: StreamMessage to process
            
        Returns:
            True if message was queued successfully
        """
        async with self._queue_lock:
            # Check queue size limit
            if len(self._priority_queue) >= self.max_queue_size:
                logger.warning(
                    f"Queue full, dropping message {message.id}",
                    extra={"queue_size": len(self._priority_queue)}
                )
                return False
            
            # Check if message is already expired
            if message.is_expired():
                logger.warning(f"Message {message.id} expired before queuing")
                self._metrics.total_expired += 1
                return False
            
            # Create priority message
            priority_level = self._priority_mapping.get(
                message.priority, PriorityLevel.NORMAL
            )
            
            priority_message = PriorityMessage(
                priority=priority_level.value,
                sequence=self._sequence_counter,
                timestamp=time.time(),
                message=message
            )
            
            self._sequence_counter += 1
            
            # Add to priority queue
            heapq.heappush(self._priority_queue, priority_message)
            
            logger.debug(
                f"Queued message {message.id} with priority {priority_level.name}",
                extra={
                    "message_id": message.id,
                    "priority": priority_level.name,
                    "queue_size": len(self._priority_queue)
                }
            )
            
            return True
    
    async def _processor_loop(self, processor_name: str) -> None:
        """Main processing loop for a processor."""
        logger.info(f"Started processor loop: {processor_name}")
        
        try:
            while self._running:
                try:
                    # Get batch of messages
                    batch = await self._get_message_batch()
                    
                    if not batch:
                        # No messages, sleep briefly
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process batch
                    await self._process_message_batch(batch, processor_name)
                    
                except asyncio.CancelledError:
                    logger.info(f"Processor loop cancelled: {processor_name}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in processor loop {processor_name}: {e}")
                    await asyncio.sleep(1)  # Brief delay on error
                    
        except Exception as e:
            logger.error(f"Fatal error in processor loop {processor_name}: {e}")
        
        logger.info(f"Stopped processor loop: {processor_name}")
    
    async def _get_message_batch(self) -> List[PriorityMessage]:
        """Get a batch of messages from the priority queue."""
        async with self._queue_lock:
            batch = []
            
            for _ in range(min(self.batch_size, len(self._priority_queue))):
                if not self._priority_queue:
                    break
                
                priority_message = heapq.heappop(self._priority_queue)
                
                # Check if message expired
                if priority_message.is_expired():
                    self._metrics.total_expired += 1
                    logger.debug(f"Message {priority_message.message.id} expired in queue")
                    continue
                
                batch.append(priority_message)
            
            return batch
    
    async def _process_message_batch(
        self,
        batch: List[PriorityMessage],
        processor_name: str
    ) -> None:
        """Process a batch of messages."""
        for priority_message in batch:
            try:
                await self._process_single_message(priority_message, processor_name)
            except Exception as e:
                logger.error(f"Error processing message {priority_message.message.id}: {e}")
                await self._handle_processing_error(priority_message, e)
    
    async def _process_single_message(
        self,
        priority_message: PriorityMessage,
        processor_name: str
    ) -> None:
        """Process a single message."""
        message = priority_message.message
        start_time = time.time()
        priority_message.processing_started = start_time
        
        try:
            # Get handler for message type
            handler = self._message_handlers.get(message.message_type)
            if not handler:
                raise ValueError(f"No handler registered for message type: {message.message_type}")
            
            # Execute handler with timeout
            try:
                await asyncio.wait_for(
                    handler(message),
                    timeout=self.max_processing_time_seconds
                )
                
                # Track successful processing
                processing_time_ms = (time.time() - start_time) * 1000
                self._processing_times.append(processing_time_ms)
                self._metrics.total_processed += 1
                
                logger.debug(
                    f"Processed message {message.id} successfully",
                    extra={
                        "message_id": message.id,
                        "processor": processor_name,
                        "processing_time_ms": processing_time_ms,
                        "message_type": message.message_type.value
                    }
                )
                
            except asyncio.TimeoutError:
                raise Exception(f"Message processing timeout after {self.max_processing_time_seconds}s")
                
        except Exception as e:
            await self._handle_processing_error(priority_message, e)
    
    async def _handle_processing_error(
        self,
        priority_message: PriorityMessage,
        error: Exception
    ) -> None:
        """Handle message processing error."""
        message = priority_message.message
        
        logger.error(
            f"Failed to process message {message.id}: {error}",
            extra={
                "message_id": message.id,
                "retry_count": priority_message.retry_count,
                "error": str(error)
            }
        )
        
        self._metrics.total_failed += 1
        
        # Check if message can be retried
        if priority_message.is_retryable():
            priority_message.retry_count += 1
            priority_message.next_retry_time = (
                time.time() + priority_message.calculate_retry_delay()
            )
            
            # Add to retry queue
            await self._retry_queue.put(priority_message)
            self._metrics.total_retries += 1
            
            logger.info(
                f"Message {message.id} scheduled for retry {priority_message.retry_count}/{priority_message.max_retries}"
            )
        else:
            # Send to dead letter queue
            await self._send_to_dead_letter(priority_message, error)
            self._metrics.total_dead_letter += 1
            
            logger.warning(
                f"Message {message.id} sent to dead letter queue after {priority_message.retry_count} retries"
            )
    
    async def _send_to_dead_letter(
        self,
        priority_message: PriorityMessage,
        error: Exception
    ) -> None:
        """Send message to dead letter queue."""
        if self.dead_letter_handler:
            try:
                await self.dead_letter_handler(priority_message.message, error)
            except Exception as dlq_error:
                logger.error(f"Error sending message to dead letter queue: {dlq_error}")
    
    async def _retry_processor_loop(self) -> None:
        """Process retry queue for failed messages."""
        logger.info("Started retry processor loop")
        
        try:
            while self._running:
                try:
                    # Get message from retry queue
                    priority_message = await asyncio.wait_for(
                        self._retry_queue.get(), timeout=1.0
                    )
                    
                    # Check if it's time to retry
                    if (priority_message.next_retry_time and 
                        time.time() >= priority_message.next_retry_time):
                        
                        # Add back to priority queue
                        async with self._queue_lock:
                            heapq.heappush(self._priority_queue, priority_message)
                        
                        logger.debug(f"Retrying message {priority_message.message.id}")
                    else:
                        # Put back in retry queue
                        await self._retry_queue.put(priority_message)
                        await asyncio.sleep(0.1)  # Brief delay
                    
                except asyncio.TimeoutError:
                    # No messages in retry queue, continue
                    continue
                    
                except asyncio.CancelledError:
                    logger.info("Retry processor loop cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in retry processor loop: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in retry processor loop: {e}")
        
        logger.info("Stopped retry processor loop")
    
    async def _ttl_cleanup_loop(self) -> None:
        """Clean up expired messages from queue."""
        logger.info("Started TTL cleanup loop")
        
        try:
            while self._running:
                try:
                    async with self._queue_lock:
                        # Check for expired messages (they bubble to top due to priority)
                        expired_count = 0
                        
                        while self._priority_queue:
                            priority_message = self._priority_queue[0]
                            
                            if priority_message.is_expired():
                                heapq.heappop(self._priority_queue)
                                expired_count += 1
                                self._metrics.total_expired += 1
                            else:
                                break  # No more expired messages
                        
                        if expired_count > 0:
                            logger.info(f"Cleaned up {expired_count} expired messages")
                    
                    await asyncio.sleep(self.ttl_check_interval_seconds)
                    
                except asyncio.CancelledError:
                    logger.info("TTL cleanup loop cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in TTL cleanup loop: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in TTL cleanup loop: {e}")
        
        logger.info("Stopped TTL cleanup loop")
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and update performance metrics."""
        logger.info("Started metrics collection loop")
        
        try:
            while self._running:
                try:
                    await self._update_metrics()
                    await asyncio.sleep(self.metrics_collection_interval_seconds)
                    
                except asyncio.CancelledError:
                    logger.info("Metrics collection loop cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Fatal error in metrics collection loop: {e}")
        
        logger.info("Stopped metrics collection loop")
    
    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        async with self._queue_lock:
            self._metrics.queue_depth = len(self._priority_queue)
            
            # Calculate backlog age
            if self._priority_queue:
                oldest_message = min(self._priority_queue, key=lambda x: x.timestamp)
                self._metrics.backlog_age_seconds = time.time() - oldest_message.timestamp
            else:
                self._metrics.backlog_age_seconds = 0.0
        
        # Calculate processing time metrics
        if self._processing_times:
            processing_times = list(self._processing_times)
            processing_times.sort()
            
            self._metrics.average_processing_time_ms = sum(processing_times) / len(processing_times)
            
            n = len(processing_times)
            self._metrics.p95_processing_time_ms = processing_times[int(n * 0.95)] if n > 0 else 0
            self._metrics.p99_processing_time_ms = processing_times[int(n * 0.99)] if n > 0 else 0
        
        # Calculate throughput
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            self._metrics.throughput_msg_per_sec = self._metrics.total_processed / elapsed_time
    
    async def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        await self._update_metrics()
        return self._metrics
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status."""
        async with self._queue_lock:
            priority_counts = defaultdict(int)
            oldest_timestamp = None
            
            for priority_message in self._priority_queue:
                priority_level = PriorityLevel(priority_message.priority)
                priority_counts[priority_level.name] += 1
                
                if oldest_timestamp is None or priority_message.timestamp < oldest_timestamp:
                    oldest_timestamp = priority_message.timestamp
            
            return {
                "total_queued": len(self._priority_queue),
                "priority_distribution": dict(priority_counts),
                "oldest_message_age_seconds": (
                    time.time() - oldest_timestamp if oldest_timestamp else 0
                ),
                "retry_queue_size": self._retry_queue.qsize(),
                "running": self._running,
                "active_processors": len([t for t in self._processor_tasks if not t.done()])
            }
    
    def is_running(self) -> bool:
        """Check if processor is running."""
        return self._running