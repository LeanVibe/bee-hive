"""
DLQ Retry Scheduler for LeanVibe Agent Hive 2.0 - VS 4.3

Intelligent retry scheduling system with exponential backoff, circuit breaker integration,
and adaptive scheduling for optimal recovery strategies.

Performance targets:
- >99.9% eventual delivery rate
- <100ms message processing overhead
- Handle 10k+ poison messages without system impact
- Automatic recovery with <30s from DLQ processing failures
"""

import asyncio
import time
import json
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import settings
from .retry_policies import RetryConfig, RetryStrategy, JitterType, RetryPolicyFactory, RetryExecutor
from .error_handling_integration import get_error_handling_integration
from ..models.message import StreamMessage, MessageType, MessagePriority, MessageStatus

logger = structlog.get_logger()


class SchedulingStrategy(str, Enum):
    """Scheduling strategies for retry operations."""
    IMMEDIATE = "immediate"                    # Retry immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Standard exponential backoff
    LINEAR_BACKOFF = "linear_backoff"         # Linear increase in delay
    ADAPTIVE_BACKOFF = "adaptive_backoff"     # Adaptive based on success patterns
    FIBONACCI_BACKOFF = "fibonacci_backoff"   # Fibonacci sequence delays
    SMART_BATCHING = "smart_batching"         # Intelligent batching of retries
    PRIORITY_BASED = "priority_based"         # Priority-based scheduling


class RetryPriority(str, Enum):
    """Priority levels for retry operations."""
    CRITICAL = "critical"      # <1s scheduling
    HIGH = "high"             # <5s scheduling  
    MEDIUM = "medium"         # <30s scheduling
    LOW = "low"              # <300s scheduling
    BACKGROUND = "background" # Best effort


@dataclass
class ScheduledRetry:
    """Represents a scheduled retry operation."""
    
    retry_id: str
    original_stream: str
    original_message_id: str
    message: StreamMessage
    failure_reason: str
    retry_count: int
    max_retries: int
    
    # Scheduling details
    scheduled_time: float
    priority: RetryPriority = RetryPriority.MEDIUM
    strategy: SchedulingStrategy = SchedulingStrategy.EXPONENTIAL_BACKOFF
    
    # Tracking
    created_at: float = field(default_factory=time.time)
    last_attempt_time: Optional[float] = None
    circuit_breaker_context: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    processing_time_ms: float = 0.0
    success_probability: float = 0.5  # Estimated success probability
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert to Redis-compatible dictionary."""
        return {
            "retry_id": self.retry_id,
            "original_stream": self.original_stream,
            "original_message_id": self.original_message_id,
            "message_data": self.message.json(),
            "failure_reason": self.failure_reason,
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "scheduled_time": str(self.scheduled_time),
            "priority": self.priority.value,
            "strategy": self.strategy.value,
            "created_at": str(self.created_at),
            "last_attempt_time": str(self.last_attempt_time) if self.last_attempt_time else "",
            "circuit_breaker_context": json.dumps(self.circuit_breaker_context or {}),
            "processing_time_ms": str(self.processing_time_ms),
            "success_probability": str(self.success_probability)
        }
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, str]) -> "ScheduledRetry":
        """Create from Redis dictionary."""
        message = StreamMessage.parse_raw(data["message_data"])
        
        return cls(
            retry_id=data["retry_id"],
            original_stream=data["original_stream"],
            original_message_id=data["original_message_id"],
            message=message,
            failure_reason=data["failure_reason"],
            retry_count=int(data["retry_count"]),
            max_retries=int(data["max_retries"]),
            scheduled_time=float(data["scheduled_time"]),
            priority=RetryPriority(data["priority"]),
            strategy=SchedulingStrategy(data["strategy"]),
            created_at=float(data["created_at"]),
            last_attempt_time=float(data["last_attempt_time"]) if data["last_attempt_time"] else None,
            circuit_breaker_context=json.loads(data["circuit_breaker_context"]) if data["circuit_breaker_context"] else None,
            processing_time_ms=float(data["processing_time_ms"]),
            success_probability=float(data["success_probability"])
        )


@dataclass
class RetrySchedulerConfig:
    """Configuration for retry scheduler."""
    
    # Performance settings
    max_concurrent_retries: int = 100
    batch_processing_size: int = 50
    processing_timeout_ms: int = 5000
    
    # Redis settings
    retry_queue_key: str = "dlq:retry_queue"
    processing_queue_key: str = "dlq:processing_queue"
    completed_queue_key: str = "dlq:completed_queue"
    
    # Scheduling settings
    scheduler_interval_ms: int = 100  # 100ms scheduling intervals
    priority_boost_threshold: int = 5  # Boost priority after 5 failures
    success_tracking_window: int = 100  # Track success over 100 operations
    
    # Adaptive settings
    adaptive_learning_enabled: bool = True
    success_threshold_adjustment: float = 0.1
    performance_optimization_enabled: bool = True
    
    # Circuit breaker integration
    circuit_breaker_integration: bool = True
    circuit_breaker_timeout_multiplier: float = 2.0


class DLQRetryScheduler:
    """
    Intelligent retry scheduler for Dead Letter Queue with adaptive strategies.
    
    Provides:
    - High-performance retry scheduling (<100ms overhead)
    - Adaptive retry strategies based on success patterns
    - Circuit breaker integration for intelligent failure handling
    - Priority-based scheduling for critical operations
    - Smart batching for optimal throughput
    - Comprehensive monitoring and metrics
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: Optional[RetrySchedulerConfig] = None
    ):
        """Initialize the DLQ retry scheduler."""
        self.redis = redis_client
        self.config = config or RetrySchedulerConfig()
        self.error_integration = get_error_handling_integration()
        
        # Scheduler state
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._processor_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self._metrics = {
            "total_scheduled": 0,
            "total_processed": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "average_processing_time_ms": 0.0,
            "scheduler_overhead_ms": 0.0,
            "queue_sizes": {
                "retry": 0,
                "processing": 0,
                "completed": 0
            }
        }
        
        # Adaptive learning
        self._success_history: Dict[str, List[bool]] = {}  # Stream -> success history
        self._strategy_performance: Dict[SchedulingStrategy, float] = {
            strategy: 0.5 for strategy in SchedulingStrategy
        }
        
        # Priority queues for different retry priorities
        self._priority_queues = {
            priority: f"{self.config.retry_queue_key}:{priority.value}"
            for priority in RetryPriority
        }
        
        logger.info(
            "🔄 DLQ Retry Scheduler initialized",
            max_concurrent=self.config.max_concurrent_retries,
            batch_size=self.config.batch_processing_size,
            scheduler_interval_ms=self.config.scheduler_interval_ms
        )
    
    async def start(self) -> None:
        """Start the retry scheduler."""
        if self._running:
            logger.warning("DLQ Retry Scheduler already running")
            return
        
        self._running = True
        
        # Start main scheduler task
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start processor tasks for different priorities
        for priority in RetryPriority:
            processor_task = asyncio.create_task(
                self._priority_processor_loop(priority)
            )
            self._processor_tasks.append(processor_task)
        
        logger.info("✅ DLQ Retry Scheduler started")
    
    async def stop(self) -> None:
        """Stop the retry scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel scheduler task
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
        
        # Cancel processor tasks
        for task in self._processor_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = [self._scheduler_task] + self._processor_tasks
        active_tasks = [t for t in all_tasks if t and not t.done()]
        
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logger.info("🛑 DLQ Retry Scheduler stopped")
    
    async def schedule_retry(
        self,
        original_stream: str,
        original_message_id: str,
        message: StreamMessage,
        failure_reason: str,
        retry_count: int = 0,
        max_retries: int = 3,
        priority: RetryPriority = RetryPriority.MEDIUM,
        strategy: Optional[SchedulingStrategy] = None
    ) -> str:
        """
        Schedule a message for retry.
        
        Args:
            original_stream: Original stream name
            original_message_id: Original message ID
            message: Message to retry
            failure_reason: Reason for failure
            retry_count: Current retry count
            max_retries: Maximum retries allowed
            priority: Retry priority
            strategy: Optional specific strategy
            
        Returns:
            Retry ID for tracking
        """
        try:
            # Generate retry ID
            retry_id = f"retry_{original_message_id}_{int(time.time() * 1000)}"
            
            # Determine optimal strategy if not provided
            if strategy is None:
                strategy = await self._determine_optimal_strategy(
                    original_stream, failure_reason, retry_count
                )
            
            # Calculate scheduled time based on strategy
            scheduled_time = await self._calculate_scheduled_time(
                strategy, retry_count, priority, original_stream
            )
            
            # Create scheduled retry
            scheduled_retry = ScheduledRetry(
                retry_id=retry_id,
                original_stream=original_stream,
                original_message_id=original_message_id,
                message=message,
                failure_reason=failure_reason,
                retry_count=retry_count,
                max_retries=max_retries,
                scheduled_time=scheduled_time,
                priority=priority,
                strategy=strategy
            )
            
            # Add to appropriate priority queue with score as scheduled time
            priority_queue = self._priority_queues[priority]
            await self.redis.zadd(
                priority_queue,
                {json.dumps(scheduled_retry.to_redis_dict()): scheduled_time}
            )
            
            # Update metrics
            self._metrics["total_scheduled"] += 1
            
            logger.debug(
                "📅 Retry scheduled",
                retry_id=retry_id,
                original_stream=original_stream,
                priority=priority.value,
                strategy=strategy.value,
                scheduled_in_seconds=scheduled_time - time.time(),
                retry_count=retry_count
            )
            
            return retry_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to schedule retry",
                error=str(e),
                original_stream=original_stream,
                original_message_id=original_message_id
            )
            raise
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop for processing retry queues."""
        while self._running:
            try:
                start_time = time.time()
                
                # Update queue size metrics
                await self._update_queue_metrics()
                
                # Process expired retries from each priority queue
                for priority in RetryPriority:
                    await self._process_priority_queue(priority)
                
                # Calculate scheduler overhead
                processing_time = (time.time() - start_time) * 1000
                self._metrics["scheduler_overhead_ms"] = processing_time
                
                # Adaptive sleep based on queue sizes and performance
                sleep_time = await self._calculate_adaptive_sleep_time()
                await asyncio.sleep(sleep_time / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)  # Fallback sleep
    
    async def _priority_processor_loop(self, priority: RetryPriority) -> None:
        """Process retries for a specific priority level."""
        processing_queue = f"{self.config.processing_queue_key}:{priority.value}"
        
        while self._running:
            try:
                # Get batch of retries ready for processing
                retry_batch = await self._get_processing_batch(processing_queue)
                
                if not retry_batch:
                    # No work available, sleep based on priority
                    sleep_time = self._get_priority_sleep_time(priority)
                    await asyncio.sleep(sleep_time)
                    continue
                
                # Process batch concurrently
                await self._process_retry_batch(retry_batch, priority)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Error in priority processor loop for {priority.value}: {e}"
                )
                await asyncio.sleep(1.0)
    
    async def _process_priority_queue(self, priority: RetryPriority) -> None:
        """Process retries ready for execution from priority queue."""
        try:
            priority_queue = self._priority_queues[priority]
            processing_queue = f"{self.config.processing_queue_key}:{priority.value}"
            current_time = time.time()
            
            # Get retries ready for processing (score <= current_time)
            ready_retries = await self.redis.zrangebyscore(
                priority_queue,
                min=0,
                max=current_time,
                withscores=True,
                start=0,
                num=self.config.batch_processing_size
            )
            
            if not ready_retries:
                return
            
            # Move ready retries to processing queue
            pipe = self.redis.pipeline()
            for retry_data, score in ready_retries:
                # Remove from priority queue
                pipe.zrem(priority_queue, retry_data)
                # Add to processing queue
                pipe.lpush(processing_queue, retry_data)
            
            await pipe.execute()
            
            logger.debug(
                f"📋 Moved {len(ready_retries)} retries to processing for {priority.value}"
            )
            
        except Exception as e:
            logger.error(f"Error processing priority queue {priority.value}: {e}")
    
    async def _get_processing_batch(self, processing_queue: str) -> List[ScheduledRetry]:
        """Get a batch of retries from processing queue."""
        try:
            # Get batch from processing queue
            retry_data_list = await self.redis.lpop(
                processing_queue, self.config.batch_processing_size
            )
            
            if not retry_data_list:
                return []
            
            # Convert to ScheduledRetry objects
            retries = []
            for retry_data in retry_data_list:
                try:
                    if isinstance(retry_data, bytes):
                        retry_data = retry_data.decode('utf-8')
                    
                    retry_dict = json.loads(retry_data)
                    scheduled_retry = ScheduledRetry.from_redis_dict(retry_dict)
                    retries.append(scheduled_retry)
                    
                except Exception as e:
                    logger.error(f"Error parsing retry data: {e}")
            
            return retries
            
        except Exception as e:
            logger.error(f"Error getting processing batch: {e}")
            return []
    
    async def _process_retry_batch(
        self,
        retry_batch: List[ScheduledRetry],
        priority: RetryPriority
    ) -> None:
        """Process a batch of retries concurrently."""
        if not retry_batch:
            return
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_retries)
        
        # Process each retry
        tasks = []
        for scheduled_retry in retry_batch:
            task = asyncio.create_task(
                self._process_single_retry(scheduled_retry, semaphore)
            )
            tasks.append(task)
        
        # Wait for all retries to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update metrics and learning
        successful_count = sum(1 for result in results if result is True)
        failed_count = len(results) - successful_count
        
        self._metrics["total_processed"] += len(retry_batch)
        self._metrics["successful_retries"] += successful_count
        self._metrics["failed_retries"] += failed_count
        
        logger.debug(
            f"🔄 Processed retry batch for {priority.value}",
            batch_size=len(retry_batch),
            successful=successful_count,
            failed=failed_count
        )
    
    async def _process_single_retry(
        self,
        scheduled_retry: ScheduledRetry,
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Process a single retry operation."""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Create retry configuration based on strategy
                retry_config = await self._create_retry_config(scheduled_retry)
                retry_policy = RetryPolicyFactory.create_policy(retry_config)
                retry_executor = RetryExecutor(retry_policy)
                
                # Execute retry
                success = await retry_executor.execute(
                    self._replay_message_to_stream,
                    scheduled_retry.original_stream,
                    scheduled_retry.message
                )
                
                # Update processing time
                processing_time = (time.time() - start_time) * 1000
                scheduled_retry.processing_time_ms = processing_time
                scheduled_retry.last_attempt_time = time.time()
                
                # Update success tracking
                await self._update_success_tracking(
                    scheduled_retry.original_stream,
                    success,
                    scheduled_retry.strategy
                )
                
                # Handle result
                if success:
                    await self._handle_successful_retry(scheduled_retry)
                    logger.debug(
                        "✅ Retry successful",
                        retry_id=scheduled_retry.retry_id,
                        processing_time_ms=processing_time
                    )
                else:
                    await self._handle_failed_retry(scheduled_retry)
                    logger.warning(
                        "❌ Retry failed",
                        retry_id=scheduled_retry.retry_id,
                        retry_count=scheduled_retry.retry_count,
                        max_retries=scheduled_retry.max_retries
                    )
                
                return success
                
            except Exception as e:
                # Handle retry execution error
                processing_time = (time.time() - start_time) * 1000
                scheduled_retry.processing_time_ms = processing_time
                
                await self._handle_retry_error(scheduled_retry, str(e))
                
                logger.error(
                    "💥 Retry execution error",
                    retry_id=scheduled_retry.retry_id,
                    error=str(e),
                    processing_time_ms=processing_time
                )
                
                return False
    
    async def _replay_message_to_stream(
        self,
        stream_name: str,
        message: StreamMessage
    ) -> bool:
        """Replay message back to original stream."""
        try:
            # Add message back to stream
            message_id = await self.redis.xadd(
                stream_name,
                message.to_redis_dict(),
                maxlen=settings.REDIS_STREAM_MAX_LEN,
                approximate=True
            )
            
            logger.debug(
                f"📤 Message replayed to stream",
                stream=stream_name,
                message_id=message_id
            )
            
            return True
            
        except RedisError as e:
            logger.error(
                f"Redis error replaying message to {stream_name}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Error replaying message to {stream_name}: {e}"
            )
            return False
    
    async def _determine_optimal_strategy(
        self,
        stream_name: str,
        failure_reason: str,
        retry_count: int
    ) -> SchedulingStrategy:
        """Determine optimal retry strategy based on context and history."""
        
        # Check adaptive learning if enabled
        if self.config.adaptive_learning_enabled:
            # Use strategy with best historical performance for this stream
            stream_success_rate = await self._get_stream_success_rate(stream_name)
            
            if stream_success_rate > 0.8:
                return SchedulingStrategy.IMMEDIATE
            elif stream_success_rate > 0.6:
                return SchedulingStrategy.EXPONENTIAL_BACKOFF
            elif stream_success_rate > 0.4:
                return SchedulingStrategy.ADAPTIVE_BACKOFF
            else:
                return SchedulingStrategy.LINEAR_BACKOFF
        
        # Fallback strategy based on failure reason and retry count
        if "timeout" in failure_reason.lower():
            return SchedulingStrategy.EXPONENTIAL_BACKOFF
        elif "network" in failure_reason.lower():
            return SchedulingStrategy.FIBONACCI_BACKOFF
        elif retry_count > 2:
            return SchedulingStrategy.ADAPTIVE_BACKOFF
        else:
            return SchedulingStrategy.EXPONENTIAL_BACKOFF
    
    async def _calculate_scheduled_time(
        self,
        strategy: SchedulingStrategy,
        retry_count: int,
        priority: RetryPriority,
        stream_name: str
    ) -> float:
        """Calculate when retry should be scheduled."""
        current_time = time.time()
        
        # Base delay based on priority
        priority_delays = {
            RetryPriority.CRITICAL: 0.1,      # 100ms
            RetryPriority.HIGH: 1.0,          # 1s
            RetryPriority.MEDIUM: 5.0,        # 5s
            RetryPriority.LOW: 30.0,          # 30s
            RetryPriority.BACKGROUND: 300.0   # 5min
        }
        
        base_delay = priority_delays[priority]
        
        # Calculate delay based on strategy
        if strategy == SchedulingStrategy.IMMEDIATE:
            delay = 0.0
        elif strategy == SchedulingStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (2 ** retry_count)
        elif strategy == SchedulingStrategy.LINEAR_BACKOFF:
            delay = base_delay * (1 + retry_count)
        elif strategy == SchedulingStrategy.FIBONACCI_BACKOFF:
            fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            fib_multiplier = fib_sequence[min(retry_count, len(fib_sequence) - 1)]
            delay = base_delay * fib_multiplier
        elif strategy == SchedulingStrategy.ADAPTIVE_BACKOFF:
            # Adapt based on stream success rate
            success_rate = await self._get_stream_success_rate(stream_name)
            adaptive_factor = 2.0 if success_rate < 0.5 else 1.5
            delay = base_delay * (adaptive_factor ** retry_count)
        else:
            delay = base_delay * (2 ** retry_count)  # Default to exponential
        
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * random.random()
        final_delay = delay + jitter
        
        # Cap maximum delay
        max_delay = 3600.0  # 1 hour max
        final_delay = min(final_delay, max_delay)
        
        return current_time + final_delay
    
    async def _create_retry_config(self, scheduled_retry: ScheduledRetry) -> RetryConfig:
        """Create retry configuration for execution."""
        return RetryConfig(
            max_attempts=1,  # Single attempt since scheduling handles retries
            base_delay_ms=100,
            max_delay_ms=5000,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter_type=JitterType.EQUAL,
            enable_metrics=True,
            enable_logging=False  # Reduce logging overhead
        )
    
    async def _handle_successful_retry(self, scheduled_retry: ScheduledRetry) -> None:
        """Handle successful retry."""
        # Move to completed queue for metrics
        completed_queue = f"{self.config.completed_queue_key}:success"
        await self.redis.lpush(
            completed_queue,
            json.dumps({
                "retry_id": scheduled_retry.retry_id,
                "completed_at": time.time(),
                "processing_time_ms": scheduled_retry.processing_time_ms,
                "retry_count": scheduled_retry.retry_count,
                "strategy": scheduled_retry.strategy.value
            })
        )
        
        # Emit success event to observability
        await self.error_integration.emit_recovery_initiated(
            recovery_strategy="dlq_retry_successful",
            trigger_failure=scheduled_retry.failure_reason,
            recovery_steps=[f"retry_attempt_{scheduled_retry.retry_count}"],
            estimated_recovery_time_ms=scheduled_retry.processing_time_ms
        )
    
    async def _handle_failed_retry(self, scheduled_retry: ScheduledRetry) -> None:
        """Handle failed retry."""
        if scheduled_retry.retry_count < scheduled_retry.max_retries:
            # Reschedule with increased retry count
            await self.schedule_retry(
                original_stream=scheduled_retry.original_stream,
                original_message_id=scheduled_retry.original_message_id,
                message=scheduled_retry.message,
                failure_reason=scheduled_retry.failure_reason,
                retry_count=scheduled_retry.retry_count + 1,
                max_retries=scheduled_retry.max_retries,
                priority=scheduled_retry.priority,
                strategy=scheduled_retry.strategy
            )
        else:
            # Move to permanent failure queue
            failed_queue = f"{self.config.completed_queue_key}:failed"
            await self.redis.lpush(
                failed_queue,
                json.dumps({
                    "retry_id": scheduled_retry.retry_id,
                    "failed_at": time.time(),
                    "final_retry_count": scheduled_retry.retry_count,
                    "failure_reason": scheduled_retry.failure_reason
                })
            )
    
    async def _handle_retry_error(self, scheduled_retry: ScheduledRetry, error: str) -> None:
        """Handle retry execution error."""
        # Emit error event
        await self.error_integration.emit_error_handling_failure(
            error_type="retry_execution_error",
            error_message=error,
            component="dlq_retry_scheduler",
            context={
                "retry_id": scheduled_retry.retry_id,
                "original_stream": scheduled_retry.original_stream,
                "retry_count": scheduled_retry.retry_count,
                "strategy": scheduled_retry.strategy.value,
                "processing_time_ms": scheduled_retry.processing_time_ms
            }
        )
    
    async def _update_success_tracking(
        self,
        stream_name: str,
        success: bool,
        strategy: SchedulingStrategy
    ) -> None:
        """Update success tracking for adaptive learning."""
        # Update stream success history
        if stream_name not in self._success_history:
            self._success_history[stream_name] = []
        
        self._success_history[stream_name].append(success)
        
        # Keep only recent history
        if len(self._success_history[stream_name]) > self.config.success_tracking_window:
            self._success_history[stream_name] = self._success_history[stream_name][-self.config.success_tracking_window:]
        
        # Update strategy performance
        current_performance = self._strategy_performance[strategy]
        adjustment = self.config.success_threshold_adjustment
        
        if success:
            self._strategy_performance[strategy] = min(1.0, current_performance + adjustment)
        else:
            self._strategy_performance[strategy] = max(0.0, current_performance - adjustment)
    
    async def _get_stream_success_rate(self, stream_name: str) -> float:
        """Get success rate for a stream."""
        if stream_name not in self._success_history:
            return 0.5  # Default neutral rate
        
        history = self._success_history[stream_name]
        if not history:
            return 0.5
        
        return sum(history) / len(history)
    
    async def _update_queue_metrics(self) -> None:
        """Update queue size metrics."""
        for priority in RetryPriority:
            priority_queue = self._priority_queues[priority]
            processing_queue = f"{self.config.processing_queue_key}:{priority.value}"
            
            try:
                retry_size = await self.redis.zcard(priority_queue)
                processing_size = await self.redis.llen(processing_queue)
                
                self._metrics["queue_sizes"]["retry"] += retry_size
                self._metrics["queue_sizes"]["processing"] += processing_size
                
            except Exception as e:
                logger.error(f"Error updating queue metrics for {priority.value}: {e}")
    
    async def _calculate_adaptive_sleep_time(self) -> float:
        """Calculate adaptive sleep time based on system load."""
        base_interval = self.config.scheduler_interval_ms
        
        # Adjust based on queue sizes
        total_queued = sum(self._metrics["queue_sizes"].values())
        
        if total_queued > 1000:
            return base_interval * 0.5  # Process faster when queues are large
        elif total_queued > 100:
            return base_interval * 0.75
        elif total_queued < 10:
            return base_interval * 1.5  # Process slower when queues are small
        else:
            return base_interval
    
    def _get_priority_sleep_time(self, priority: RetryPriority) -> float:
        """Get sleep time for priority processor based on priority level."""
        sleep_times = {
            RetryPriority.CRITICAL: 0.1,      # 100ms
            RetryPriority.HIGH: 0.5,          # 500ms
            RetryPriority.MEDIUM: 1.0,        # 1s
            RetryPriority.LOW: 5.0,           # 5s
            RetryPriority.BACKGROUND: 30.0    # 30s
        }
        
        return sleep_times[priority]
    
    async def get_scheduler_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scheduler metrics."""
        # Update queue metrics
        await self._update_queue_metrics()
        
        # Calculate success rate
        total_processed = self._metrics["total_processed"]
        success_rate = (
            self._metrics["successful_retries"] / max(1, total_processed)
        )
        
        return {
            "performance_metrics": {
                "total_scheduled": self._metrics["total_scheduled"],
                "total_processed": self._metrics["total_processed"],
                "success_rate": success_rate,
                "scheduler_overhead_ms": self._metrics["scheduler_overhead_ms"],
                "average_processing_time_ms": self._metrics["average_processing_time_ms"]
            },
            "queue_metrics": self._metrics["queue_sizes"].copy(),
            "strategy_performance": self._strategy_performance.copy(),
            "stream_success_rates": {
                stream: self._get_stream_success_rate(stream)
                for stream in self._success_history.keys()
            } if self.config.adaptive_learning_enabled else {},
            "configuration": {
                "max_concurrent_retries": self.config.max_concurrent_retries,
                "batch_processing_size": self.config.batch_processing_size,
                "scheduler_interval_ms": self.config.scheduler_interval_ms,
                "adaptive_learning_enabled": self.config.adaptive_learning_enabled
            },
            "system_status": {
                "running": self._running,
                "scheduler_task_active": self._scheduler_task and not self._scheduler_task.done(),
                "processor_tasks_active": sum(
                    1 for task in self._processor_tasks if not task.done()
                )
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        metrics = await self.get_scheduler_metrics()
        
        # Determine health status
        is_healthy = (
            self._running and
            metrics["system_status"]["scheduler_task_active"] and
            metrics["performance_metrics"]["scheduler_overhead_ms"] < 100 and
            metrics["performance_metrics"]["success_rate"] > 0.5
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "running": self._running,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }