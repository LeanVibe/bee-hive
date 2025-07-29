"""
Semantic Memory Task Processor for LeanVibe Agent Hive 2.0

Handles semantic memory operations via Redis Streams with async processing,
message routing, retry logic, and performance monitoring.

Features:
- Async task processing with Redis Streams consumer groups
- Intelligent message routing and load balancing  
- Retry logic with exponential backoff
- Performance monitoring and SLA tracking
- Integration with semantic memory service API
- Dead letter queue handling for failed tasks
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import structlog
import httpx
from redis.asyncio import Redis

from .database import get_session
from .redis import get_redis, get_message_broker
from .enhanced_redis_streams_manager import EnhancedRedisStreamsManager
from .consumer_group_coordinator import ConsumerGroupCoordinator
from ..schemas.semantic_memory import (
    DocumentIngestRequest, SemanticSearchRequest, ContextCompressionRequest,
    ProcessingPriority
)

logger = structlog.get_logger()


# =============================================================================
# TASK DEFINITIONS AND TYPES
# =============================================================================

class SemanticTaskType(str, Enum):
    """Types of semantic memory tasks."""
    INGEST_DOCUMENT = "ingest_document"
    BATCH_INGEST = "batch_ingest"
    SEARCH_SEMANTIC = "search_semantic"
    FIND_SIMILAR = "find_similar"
    GET_RELATED = "get_related"
    COMPRESS_CONTEXT = "compress_context"
    CONTEXTUALIZE = "contextualize"
    GET_AGENT_KNOWLEDGE = "get_agent_knowledge"
    HEALTH_CHECK = "health_check"
    REBUILD_INDEX = "rebuild_index"


class TaskStatus(str, Enum):
    """Task processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class SemanticMemoryTask:
    """Semantic memory task with full lifecycle tracking."""
    task_id: str
    task_type: SemanticTaskType
    agent_id: str
    workflow_id: Optional[str] = None
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Retry and error handling
    retry_count: int = 0
    max_retries: int = 3
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    error_message: Optional[str] = None
    
    # Performance tracking
    processing_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    def to_redis_message(self) -> Dict[str, str]:
        """Convert task to Redis Stream message format."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id or "",
            "priority": self.priority.value,
            "status": self.status.value,
            "payload": json.dumps(self.payload),
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "error_message": self.error_message or ""
        }
    
    @classmethod
    def from_redis_message(cls, message_data: Dict[str, bytes]) -> 'SemanticMemoryTask':
        """Create task from Redis Stream message."""
        return cls(
            task_id=message_data[b"task_id"].decode(),
            task_type=SemanticTaskType(message_data[b"task_type"].decode()),
            agent_id=message_data[b"agent_id"].decode(),
            workflow_id=message_data[b"workflow_id"].decode() or None,
            priority=ProcessingPriority(message_data[b"priority"].decode()),
            status=TaskStatus(message_data[b"status"].decode()),
            payload=json.loads(message_data[b"payload"].decode()),
            metadata=json.loads(message_data[b"metadata"].decode()),
            created_at=datetime.fromisoformat(message_data[b"created_at"].decode()),
            retry_count=int(message_data[b"retry_count"].decode()),
            max_retries=int(message_data[b"max_retries"].decode()),
            error_message=message_data[b"error_message"].decode() or None
        )
    
    def mark_started(self) -> None:
        """Mark task as started processing."""
        self.status = TaskStatus.PROCESSING
        self.started_at = datetime.utcnow()
        if self.created_at:
            self.queue_time_ms = (self.started_at - self.created_at).total_seconds() * 1000
    
    def mark_completed(self, processing_time_ms: float) -> None:
        """Mark task as completed successfully."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.processing_time_ms = processing_time_ms
        if self.created_at:
            self.total_time_ms = (self.completed_at - self.created_at).total_seconds() * 1000
    
    def mark_failed(self, error_message: str, processing_time_ms: float = 0.0) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.processing_time_ms = processing_time_ms
        if self.created_at:
            self.total_time_ms = (self.completed_at - self.created_at).total_seconds() * 1000
    
    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.retry_count < self.max_retries
    
    def get_retry_delay(self) -> float:
        """Get delay before next retry attempt."""
        if self.retry_count < len(self.retry_delays):
            return self.retry_delays[self.retry_count]
        return self.retry_delays[-1] * (2 ** (self.retry_count - len(self.retry_delays)))


@dataclass
class TaskResult:
    """Result of semantic memory task execution."""
    task_id: str
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    service_response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PERFORMANCE MONITORING AND METRICS
# =============================================================================

@dataclass
class ProcessorMetrics:
    """Performance metrics for task processor."""
    # Task counts
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0
    dead_letter_tasks: int = 0
    
    # Performance metrics
    average_processing_time_ms: float = 0.0
    average_queue_time_ms: float = 0.0
    average_total_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    peak_tasks_per_second: float = 0.0
    current_queue_depth: int = 0
    
    # SLA compliance
    sla_target_processing_time_ms: float = 100.0
    sla_compliance_rate: float = 0.0
    sla_violations: int = 0
    
    # Resource utilization
    active_processors: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_counts": {
                "total_processed": self.total_tasks_processed,
                "successful": self.successful_tasks,
                "failed": self.failed_tasks,
                "retried": self.retried_tasks,
                "dead_letter": self.dead_letter_tasks
            },
            "performance": {
                "avg_processing_time_ms": self.average_processing_time_ms,
                "avg_queue_time_ms": self.average_queue_time_ms,
                "avg_total_time_ms": self.average_total_time_ms,
                "p95_processing_time_ms": self.p95_processing_time_ms,
                "p99_processing_time_ms": self.p99_processing_time_ms
            },
            "throughput": {
                "tasks_per_second": self.tasks_per_second,
                "peak_tasks_per_second": self.peak_tasks_per_second,
                "current_queue_depth": self.current_queue_depth
            },
            "sla": {
                "target_processing_time_ms": self.sla_target_processing_time_ms,
                "compliance_rate": self.sla_compliance_rate,
                "violations": self.sla_violations
            },
            "resources": {
                "active_processors": self.active_processors,
                "cpu_usage_percent": self.cpu_usage_percent,
                "memory_usage_mb": self.memory_usage_mb
            }
        }


class PerformanceTracker:
    """Tracks and calculates performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.queue_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        self.throughput_window = deque(maxlen=60)  # 1 minute window
        self.last_throughput_update = time.time()
        
    def record_task_completion(self, task: SemanticMemoryTask) -> None:
        """Record completed task metrics."""
        self.processing_times.append(task.processing_time_ms)
        self.queue_times.append(task.queue_time_ms)
        self.total_times.append(task.total_time_ms)
        
        # Update throughput
        current_time = time.time()
        if current_time - self.last_throughput_update >= 1.0:  # Update every second
            tasks_in_last_second = sum(1 for _ in range(len(self.throughput_window)))
            self.throughput_window.append(tasks_in_last_second)
            self.last_throughput_update = current_time
    
    def calculate_percentile(self, values: deque, percentile: float) -> float:
        """Calculate percentile from values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0,
            "avg_queue_time_ms": sum(self.queue_times) / len(self.queue_times) if self.queue_times else 0.0,
            "avg_total_time_ms": sum(self.total_times) / len(self.total_times) if self.total_times else 0.0,
            "p95_processing_time_ms": self.calculate_percentile(self.processing_times, 95),
            "p99_processing_time_ms": self.calculate_percentile(self.processing_times, 99),
            "current_throughput": sum(self.throughput_window) / len(self.throughput_window) if self.throughput_window else 0.0
        }


# =============================================================================
# SEMANTIC MEMORY TASK PROCESSOR
# =============================================================================

class SemanticMemoryTaskProcessor:
    """
    Async task processor for semantic memory operations.
    
    Features:
    - Redis Streams consumer group processing
    - Intelligent retry logic with exponential backoff
    - Performance monitoring and SLA tracking
    - Dead letter queue for failed tasks
    - Load balancing across multiple processors
    - Integration with semantic memory service API
    """
    
    # Stream and group names
    MEMORY_TASKS_STREAM = "semantic_memory_tasks"
    MEMORY_RESULTS_STREAM = "semantic_memory_results" 
    MEMORY_EVENTS_STREAM = "semantic_memory_events"
    DEAD_LETTER_STREAM = "semantic_memory_dead_letter"
    
    PROCESSORS_GROUP = "semantic_processors"
    LISTENERS_GROUP = "semantic_listeners"
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        memory_service_url: str = "http://semantic-memory-service:8001/api/v1",
        processor_id: Optional[str] = None,
        max_concurrent_tasks: int = 10,
        batch_size: int = 5,
        performance_targets: Optional[Dict[str, float]] = None
    ):
        """
        Initialize semantic memory task processor.
        
        Args:
            redis_client: Redis client instance
            memory_service_url: URL of semantic memory service
            processor_id: Unique processor identifier
            max_concurrent_tasks: Maximum concurrent tasks per processor
            batch_size: Number of tasks to read from stream at once
            performance_targets: Performance SLA targets
        """
        self.redis = redis_client or get_redis()
        self.memory_service_url = memory_service_url
        self.processor_id = processor_id or f"processor_{uuid.uuid4().hex[:8]}"
        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size
        
        # HTTP client for semantic memory service
        self.http_client = httpx.AsyncClient(
            base_url=memory_service_url,
            timeout=30.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        )
        
        # Performance tracking
        self.performance_targets = performance_targets or {
            "context_retrieval_ms": 50.0,
            "memory_task_processing_ms": 100.0,
            "workflow_overhead_ms": 10.0
        }
        self.performance_tracker = PerformanceTracker()
        self.metrics = ProcessorMetrics()
        self.metrics.sla_target_processing_time_ms = self.performance_targets.get(
            "memory_task_processing_ms", 100.0
        )
        
        # Processor state
        self.running = False
        self.processor_task: Optional[asyncio.Task] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Task routing
        self.task_handlers = {
            SemanticTaskType.INGEST_DOCUMENT: self._handle_ingest_document,
            SemanticTaskType.BATCH_INGEST: self._handle_batch_ingest,
            SemanticTaskType.SEARCH_SEMANTIC: self._handle_search_semantic,
            SemanticTaskType.FIND_SIMILAR: self._handle_find_similar,
            SemanticTaskType.GET_RELATED: self._handle_get_related,
            SemanticTaskType.COMPRESS_CONTEXT: self._handle_compress_context,
            SemanticTaskType.CONTEXTUALIZE: self._handle_contextualize,
            SemanticTaskType.GET_AGENT_KNOWLEDGE: self._handle_get_agent_knowledge,
            SemanticTaskType.HEALTH_CHECK: self._handle_health_check,
            SemanticTaskType.REBUILD_INDEX: self._handle_rebuild_index
        }
        
        logger.info(
            "Semantic Memory Task Processor initialized",
            processor_id=self.processor_id,
            memory_service_url=memory_service_url,
            max_concurrent_tasks=max_concurrent_tasks
        )
    
    async def start(self) -> None:
        """Start the task processor."""
        if self.running:
            logger.warning("Task processor already running")
            return
        
        # Initialize Redis streams and consumer groups
        await self._initialize_streams()
        
        # Start processor task
        self.running = True
        self.processor_task = asyncio.create_task(self._processor_loop())
        
        logger.info(f"✅ Semantic Memory Task Processor {self.processor_id} started")
    
    async def stop(self) -> None:
        """Stop the task processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel processor task
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            task.cancel()
            logger.debug(f"Cancelled active task {task_id}")
        
        # Wait for active tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Close HTTP client
        await self.http_client.aclose()
        
        logger.info(f"✅ Semantic Memory Task Processor {self.processor_id} stopped")
    
    async def submit_task(self, task: SemanticMemoryTask) -> str:
        """Submit a task for processing."""
        message_id = await self.redis.xadd(
            self.MEMORY_TASKS_STREAM,
            task.to_redis_message(),
            maxlen=10000  # Limit stream size
        )
        
        # Publish task event
        await self._publish_event("task_submitted", {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "agent_id": task.agent_id,
            "workflow_id": task.workflow_id,
            "priority": task.priority.value
        })
        
        logger.debug(f"Submitted task {task.task_id} to processing queue")
        return message_id.decode()
    
    async def get_metrics(self) -> ProcessorMetrics:
        """Get current processor metrics."""
        # Update current metrics from performance tracker
        current_perf = self.performance_tracker.get_current_metrics()
        
        self.metrics.average_processing_time_ms = current_perf["avg_processing_time_ms"]
        self.metrics.average_queue_time_ms = current_perf["avg_queue_time_ms"]
        self.metrics.average_total_time_ms = current_perf["avg_total_time_ms"]
        self.metrics.p95_processing_time_ms = current_perf["p95_processing_time_ms"]
        self.metrics.p99_processing_time_ms = current_perf["p99_processing_time_ms"]
        self.metrics.tasks_per_second = current_perf["current_throughput"]
        
        self.metrics.active_processors = 1 if self.running else 0
        self.metrics.current_queue_depth = len(self.active_tasks)
        
        # Calculate SLA compliance
        if self.metrics.total_tasks_processed > 0:
            compliant_tasks = self.metrics.total_tasks_processed - self.metrics.sla_violations
            self.metrics.sla_compliance_rate = compliant_tasks / self.metrics.total_tasks_processed
        
        return self.metrics
    
    # Private methods
    
    async def _initialize_streams(self) -> None:
        """Initialize Redis streams and consumer groups."""
        streams = [
            self.MEMORY_TASKS_STREAM,
            self.MEMORY_RESULTS_STREAM,
            self.MEMORY_EVENTS_STREAM,
            self.DEAD_LETTER_STREAM
        ]
        
        for stream_name in streams:
            try:
                await self.redis.xgroup_create(
                    stream_name,
                    self.PROCESSORS_GROUP,
                    id="0",
                    mkstream=True
                )
                logger.debug(f"Created consumer group {self.PROCESSORS_GROUP} for {stream_name}")
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Failed to create consumer group for {stream_name}: {e}")
    
    async def _processor_loop(self) -> None:
        """Main processor loop."""
        while self.running:
            try:
                # Read pending messages first
                messages = await self._read_pending_messages()
                
                if not messages:
                    # No pending messages, read new ones
                    messages = await self._read_new_messages()
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        if not self.running:
                            break
                        
                        # Create task and process it
                        try:
                            task = SemanticMemoryTask.from_redis_message(message_data)
                            await self._process_task_async(task, message_id)
                        except Exception as e:
                            logger.error(f"Error creating task from message {message_id}: {e}")
                            # Acknowledge the message to prevent infinite reprocessing
                            await self._acknowledge_message(message_id)
                
                # Brief sleep to prevent tight loop
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _read_pending_messages(self) -> List[tuple]:
        """Read pending messages from stream."""
        try:
            return await self.redis.xreadgroup(
                self.PROCESSORS_GROUP,
                self.processor_id,
                {self.MEMORY_TASKS_STREAM: "0"},
                count=self.batch_size,
                block=100
            )
        except Exception as e:
            logger.error(f"Error reading pending messages: {e}")
            return []
    
    async def _read_new_messages(self) -> List[tuple]:
        """Read new messages from stream."""
        try:
            return await self.redis.xreadgroup(
                self.PROCESSORS_GROUP,
                self.processor_id,
                {self.MEMORY_TASKS_STREAM: ">"},
                count=self.batch_size,
                block=1000
            )
        except Exception as e:
            logger.error(f"Error reading new messages: {e}")
            return []
    
    async def _process_task_async(self, task: SemanticMemoryTask, message_id: str) -> None:
        """Process task asynchronously with concurrency control."""
        # Check if we can process more tasks
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            logger.debug(f"Max concurrent tasks reached, queuing task {task.task_id}")
            return
        
        # Create task processing coroutine
        async def process_with_cleanup():
            async with self.task_semaphore:
                try:
                    await self._process_task(task, message_id)
                finally:
                    self.active_tasks.pop(task.task_id, None)
        
        # Start processing task
        processing_task = asyncio.create_task(process_with_cleanup())
        self.active_tasks[task.task_id] = processing_task
    
    async def _process_task(self, task: SemanticMemoryTask, message_id: str) -> None:
        """Process a single semantic memory task."""
        task.mark_started()
        start_time = time.time()
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type {task.task_type}")
            
            # Execute task
            result = await handler(task)
            
            # Mark task as completed
            processing_time_ms = (time.time() - start_time) * 1000
            task.mark_completed(processing_time_ms)
            
            # Update metrics
            self.metrics.total_tasks_processed += 1
            self.metrics.successful_tasks += 1
            
            # Check SLA compliance
            if processing_time_ms > self.metrics.sla_target_processing_time_ms:
                self.metrics.sla_violations += 1
            
            self.performance_tracker.record_task_completion(task)
            
            # Publish result
            await self._publish_result(result)
            
            # Acknowledge message
            await self._acknowledge_message(message_id)
            
            logger.debug(
                f"✅ Task {task.task_id} completed successfully",
                processing_time_ms=processing_time_ms,
                task_type=task.task_type.value
            )
            
        except Exception as e:
            # Mark task as failed
            processing_time_ms = (time.time() - start_time) * 1000
            task.mark_failed(str(e), processing_time_ms)
            
            # Update metrics
            self.metrics.total_tasks_processed += 1
            self.metrics.failed_tasks += 1
            
            # Handle retry logic
            if task.should_retry():
                await self._retry_task(task)
                self.metrics.retried_tasks += 1
            else:
                await self._send_to_dead_letter(task)
                self.metrics.dead_letter_tasks += 1
            
            # Acknowledge message (task moved to retry or dead letter)
            await self._acknowledge_message(message_id)
            
            logger.error(
                f"❌ Task {task.task_id} failed",
                error=str(e),
                retry_count=task.retry_count,
                will_retry=task.should_retry()
            )
    
    async def _retry_task(self, task: SemanticMemoryTask) -> None:
        """Retry a failed task with delay."""
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        
        # Schedule retry with delay
        retry_delay = task.get_retry_delay()
        
        async def delayed_retry():
            await asyncio.sleep(retry_delay)
            await self.submit_task(task)
        
        asyncio.create_task(delayed_retry())
        
        logger.info(
            f"Retrying task {task.task_id} in {retry_delay}s (attempt {task.retry_count})"
        )
    
    async def _send_to_dead_letter(self, task: SemanticMemoryTask) -> None:
        """Send task to dead letter queue."""
        task.status = TaskStatus.DEAD_LETTER
        
        await self.redis.xadd(
            self.DEAD_LETTER_STREAM,
            task.to_redis_message(),
            maxlen=1000
        )
        
        await self._publish_event("task_dead_letter", {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "error": task.error_message,
            "retry_count": task.retry_count
        })
        
        logger.warning(f"Task {task.task_id} sent to dead letter queue")
    
    async def _acknowledge_message(self, message_id: str) -> None:
        """Acknowledge message processing."""
        try:
            await self.redis.xack(
                self.MEMORY_TASKS_STREAM,
                self.PROCESSORS_GROUP,
                message_id
            )
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")
    
    async def _publish_result(self, result: TaskResult) -> None:
        """Publish task result."""
        message_data = {
            "task_id": result.task_id,
            "success": str(result.success),
            "result_data": json.dumps(result.result_data),
            "error_message": result.error_message or "",
            "processing_time_ms": str(result.processing_time_ms),
            "service_response_time_ms": str(result.service_response_time_ms),
            "timestamp": result.timestamp.isoformat(),
            "metadata": json.dumps(result.metadata)
        }
        
        await self.redis.xadd(
            self.MEMORY_RESULTS_STREAM,
            message_data,
            maxlen=5000
        )
        
        await self._publish_event("task_completed", {
            "task_id": result.task_id,
            "success": result.success,
            "processing_time_ms": result.processing_time_ms
        })
    
    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish event to events stream."""
        message_data = {
            "event_type": event_type,
            "event_data": json.dumps(event_data),
            "processor_id": self.processor_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.redis.xadd(
            self.MEMORY_EVENTS_STREAM,
            message_data,
            maxlen=1000
        )
    
    # Task handlers
    
    async def _handle_ingest_document(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle document ingestion task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/ingest", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_batch_ingest(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle batch document ingestion task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/batch-ingest", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_search_semantic(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle semantic search task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/search", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms,
                metadata={"results_count": len(result_data.get("results", []))}
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_find_similar(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle find similar documents task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/similarity", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_get_related(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle get related documents task."""
        start_time = time.time()
        
        try:
            document_id = task.payload.get("document_id")
            params = {k: v for k, v in task.payload.items() if k != "document_id"}
            
            response = await self.http_client.get(f"/memory/related/{document_id}", params=params)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_compress_context(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle context compression task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/compress", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms,
                metadata={"compression_ratio": result_data.get("compression_ratio", 0.0)}
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_contextualize(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle contextualization task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/contextualize", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_get_agent_knowledge(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle get agent knowledge task."""
        start_time = time.time()
        
        try:
            agent_id = task.payload.get("agent_id", task.agent_id)
            params = {k: v for k, v in task.payload.items() if k != "agent_id"}
            
            response = await self.http_client.get(f"/memory/agent-knowledge/{agent_id}", params=params)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_health_check(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle health check task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.get("/memory/health")
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )
    
    async def _handle_rebuild_index(self, task: SemanticMemoryTask) -> TaskResult:
        """Handle index rebuild task."""
        start_time = time.time()
        
        try:
            response = await self.http_client.post("/memory/rebuild-index", json=task.payload)
            response.raise_for_status()
            result_data = response.json()
            
            service_time_ms = (time.time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                service_response_time_ms=service_time_ms
            )
            
        except Exception as e:
            service_time_ms = (time.time() - start_time) * 1000
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                service_response_time_ms=service_time_ms
            )


# =============================================================================
# GLOBAL PROCESSOR MANAGER
# =============================================================================

class ProcessorManager:
    """Manages multiple semantic memory task processors."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client or get_redis()
        self.processors: Dict[str, SemanticMemoryTaskProcessor] = {}
        self.running = False
    
    async def start_processor(
        self,
        processor_id: str,
        memory_service_url: str = "http://semantic-memory-service:8001/api/v1",
        **kwargs
    ) -> SemanticMemoryTaskProcessor:
        """Start a new task processor."""
        if processor_id in self.processors:
            raise ValueError(f"Processor {processor_id} already exists")
        
        processor = SemanticMemoryTaskProcessor(
            redis_client=self.redis,
            memory_service_url=memory_service_url,
            processor_id=processor_id,
            **kwargs
        )
        
        await processor.start()
        self.processors[processor_id] = processor
        
        logger.info(f"Started processor {processor_id}")
        return processor
    
    async def stop_processor(self, processor_id: str) -> None:
        """Stop a specific processor."""
        if processor_id not in self.processors:
            logger.warning(f"Processor {processor_id} not found")
            return
        
        processor = self.processors[processor_id]
        await processor.stop()
        del self.processors[processor_id]
        
        logger.info(f"Stopped processor {processor_id}")
    
    async def stop_all_processors(self) -> None:
        """Stop all processors."""
        for processor_id in list(self.processors.keys()):
            await self.stop_processor(processor_id)
    
    async def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics from all processors."""
        if not self.processors:
            return {"error": "No active processors"}
        
        # Collect metrics from all processors
        all_metrics = []
        for processor in self.processors.values():
            metrics = await processor.get_metrics()
            all_metrics.append(metrics.to_dict())
        
        # Aggregate metrics
        aggregate = {
            "total_processors": len(self.processors),
            "aggregate_task_counts": {
                "total_processed": sum(m["task_counts"]["total_processed"] for m in all_metrics),
                "successful": sum(m["task_counts"]["successful"] for m in all_metrics),
                "failed": sum(m["task_counts"]["failed"] for m in all_metrics),
                "retried": sum(m["task_counts"]["retried"] for m in all_metrics),
                "dead_letter": sum(m["task_counts"]["dead_letter"] for m in all_metrics)
            },
            "aggregate_performance": {
                "avg_processing_time_ms": sum(m["performance"]["avg_processing_time_ms"] for m in all_metrics) / len(all_metrics),
                "max_p95_processing_time_ms": max(m["performance"]["p95_processing_time_ms"] for m in all_metrics),
                "total_throughput": sum(m["throughput"]["tasks_per_second"] for m in all_metrics)
            },
            "processor_details": all_metrics
        }
        
        return aggregate


# =============================================================================
# GLOBAL PROCESSOR INSTANCE
# =============================================================================

_processor_manager: Optional[ProcessorManager] = None


async def get_processor_manager() -> ProcessorManager:
    """Get global processor manager instance."""
    global _processor_manager
    
    if _processor_manager is None:
        _processor_manager = ProcessorManager()
    
    return _processor_manager


async def shutdown_processor_manager():
    """Shutdown global processor manager."""
    global _processor_manager
    
    if _processor_manager:
        await _processor_manager.stop_all_processors()
        _processor_manager = None