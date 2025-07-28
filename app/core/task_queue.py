"""
Task Queue Service for LeanVibe Agent Hive 2.0

Provides Redis-backed task queuing with persistent storage, priority handling,
intelligent scheduling, and comprehensive task lifecycle management.
Designed for high-throughput orchestration with fault tolerance.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import heapq

import structlog
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_

from .database import get_session
from .redis import get_redis
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class QueueStatus(Enum):
    """Task queue status values."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class QueuedTask:
    """Represents a task in the queue with metadata."""
    task_id: uuid.UUID
    priority_score: float
    queue_name: str
    required_capabilities: List[str]
    estimated_effort: Optional[int]
    timeout_seconds: Optional[int]
    queued_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'task_id': str(self.task_id),
            'queued_at': self.queued_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedTask':
        """Create from dictionary."""
        data['task_id'] = uuid.UUID(data['task_id'])
        data['queued_at'] = datetime.fromisoformat(data['queued_at'])
        return cls(**data)


@dataclass
class TaskAssignmentResult:
    """Result of task assignment operation."""
    success: bool
    task_id: uuid.UUID
    agent_id: Optional[uuid.UUID]
    assignment_time: Optional[datetime]
    queue_wait_time_seconds: float
    error_message: Optional[str]


class TaskQueue:
    """
    Redis-backed task queue with persistent storage and intelligent scheduling.
    
    Features:
    - Priority-based task ordering
    - Multiple queue support
    - Persistent task metadata in PostgreSQL
    - Redis streams for real-time task distribution
    - Automatic retry handling
    - Task timeout management
    - Performance metrics collection
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        self._queue_watchers: Dict[str, asyncio.Task] = {}
        
        # Queue configuration
        self.default_queue = "default"
        self.priority_queues = ["critical", "high", "normal", "low"]
        self.max_queue_size = 10000
        self.task_timeout_seconds = 3600  # 1 hour default
        
        # Performance tracking
        self._metrics = {
            "tasks_queued": 0,
            "tasks_assigned": 0,
            "tasks_expired": 0,
            "queue_depth": defaultdict(int),
            "average_wait_time": 0.0
        }
    
    async def start(self) -> None:
        """Start the task queue service."""
        if self._running:
            return
        
        self._running = True
        
        if not self.redis_client:
            self.redis_client = get_redis()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._expire_tasks_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._queue_cleanup_loop())
        ]
        
        # Start queue watchers for each priority queue
        for queue_name in self.priority_queues:
            self._queue_watchers[queue_name] = asyncio.create_task(
                self._queue_watcher(queue_name)
            )
        
        logger.info("TaskQueue started", queues=self.priority_queues)
    
    async def stop(self) -> None:
        """Stop the task queue and cleanup resources."""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Cancel queue watchers
        for watcher in self._queue_watchers.values():
            watcher.cancel()
        
        all_tasks = self._background_tasks + list(self._queue_watchers.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self._queue_watchers.clear()
        
        logger.info("TaskQueue stopped")
    
    async def enqueue_task(
        self,
        task_id: uuid.UUID,
        priority: TaskPriority = TaskPriority.MEDIUM,
        queue_name: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        estimated_effort: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enqueue a task for processing.
        
        Args:
            task_id: Task ID to enqueue
            priority: Task priority level
            queue_name: Specific queue name (auto-determined if None)
            required_capabilities: Required agent capabilities
            estimated_effort: Estimated effort in minutes
            timeout_seconds: Task timeout in seconds
            metadata: Additional task metadata
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        try:
            # Determine queue name based on priority if not specified
            if not queue_name:
                queue_name = self._get_queue_name_for_priority(priority)
            
            # Create queued task
            queued_task = QueuedTask(
                task_id=task_id,
                priority_score=priority.value,
                queue_name=queue_name,
                required_capabilities=required_capabilities or [],
                estimated_effort=estimated_effort,
                timeout_seconds=timeout_seconds or self.task_timeout_seconds,
                queued_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Store in PostgreSQL for persistence
            async with get_session() as db:
                await db.execute(
                    """
                    INSERT INTO task_queue 
                    (task_id, queue_name, priority_score, scheduling_metadata, 
                     estimated_wait_time, queue_status, queued_at, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $7, $7)
                    """,
                    (
                        task_id,
                        queue_name,
                        priority.value,
                        {
                            "required_capabilities": required_capabilities or [],
                            "estimated_effort": estimated_effort,
                            "timeout_seconds": timeout_seconds or self.task_timeout_seconds,
                            "retry_count": 0,
                            "max_retries": 3
                        },
                        await self._estimate_wait_time(queue_name, priority.value),
                        QueueStatus.QUEUED.value,
                        datetime.utcnow()
                    )
                )
                
                # Update task status
                await db.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .values(
                        status=TaskStatus.PENDING,
                        queued_at=datetime.utcnow(),
                        orchestrator_metadata={
                            "queue_name": queue_name,
                            "priority_score": priority.value,
                            "queued_at": datetime.utcnow().isoformat()
                        }
                    )
                )
                
                await db.commit()
            
            # Add to Redis queue with priority scoring
            queue_key = f"task_queue:{queue_name}"
            await self.redis_client.zadd(
                queue_key,
                {json.dumps(queued_task.to_dict()): priority.value}
            )
            
            # Publish task available event
            await self.redis_client.publish(
                f"task_available:{queue_name}",
                json.dumps({
                    "task_id": str(task_id),
                    "queue_name": queue_name,
                    "priority_score": priority.value,
                    "queued_at": datetime.utcnow().isoformat()
                })
            )
            
            # Update metrics
            self._metrics["tasks_queued"] += 1
            self._metrics["queue_depth"][queue_name] += 1
            
            logger.info(
                "Task enqueued successfully",
                task_id=str(task_id),
                queue_name=queue_name,
                priority_score=priority.value
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to enqueue task",
                task_id=str(task_id),
                queue_name=queue_name,
                error=str(e)
            )
            return False
    
    async def dequeue_task(
        self,
        queue_name: Optional[str] = None,
        agent_capabilities: Optional[List[str]] = None,
        timeout_seconds: float = 5.0
    ) -> Optional[QueuedTask]:
        """
        Dequeue the highest priority task from the specified queue.
        
        Args:
            queue_name: Queue to dequeue from (searches all if None)
            agent_capabilities: Agent capabilities for matching
            timeout_seconds: Timeout for waiting for tasks
            
        Returns:
            QueuedTask if available, None otherwise
        """
        try:
            queues_to_check = [queue_name] if queue_name else self.priority_queues
            
            for queue in queues_to_check:
                queue_key = f"task_queue:{queue}"
                
                # Get highest priority task
                result = await self.redis_client.zrevrange(
                    queue_key, 0, 0, withscores=True
                )
                
                if not result:
                    continue
                
                task_data, score = result[0]
                queued_task = QueuedTask.from_dict(json.loads(task_data))
                
                # Check capability match
                if agent_capabilities and not self._check_capability_match(
                    queued_task.required_capabilities, agent_capabilities
                ):
                    continue
                
                # Remove from Redis queue
                await self.redis_client.zrem(queue_key, task_data)
                
                # Update database status
                async with get_session() as db:
                    await db.execute(
                        update(db.table("task_queue"))
                        .where(db.table("task_queue").c.task_id == queued_task.task_id)
                        .values(
                            queue_status=QueueStatus.ASSIGNED.value,
                            dequeued_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                    )
                    await db.commit()
                
                # Update metrics
                self._metrics["tasks_assigned"] += 1
                self._metrics["queue_depth"][queue] = max(0, self._metrics["queue_depth"][queue] - 1)
                
                logger.info(
                    "Task dequeued successfully",
                    task_id=str(queued_task.task_id),
                    queue_name=queue,
                    wait_time_seconds=(datetime.utcnow() - queued_task.queued_at).total_seconds()
                )
                
                return queued_task
            
            return None
            
        except Exception as e:
            logger.error("Failed to dequeue task", queue_name=queue_name, error=str(e))
            return None
    
    async def get_queue_stats(self, queue_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        try:
            stats = {}
            
            queues_to_check = [queue_name] if queue_name else self.priority_queues
            
            for queue in queues_to_check:
                queue_key = f"task_queue:{queue}"
                
                # Get queue depth from Redis
                depth = await self.redis_client.zcard(queue_key)
                
                # Get average wait time from database
                async with get_session() as db:
                    result = await db.execute(
                        """
                        SELECT 
                            COUNT(*) as total_tasks,
                            AVG(EXTRACT(EPOCH FROM (dequeued_at - queued_at))) as avg_wait_time,
                            COUNT(*) FILTER (WHERE queue_status = 'queued') as queued_count,
                            COUNT(*) FILTER (WHERE queue_status = 'assigned') as assigned_count
                        FROM task_queue 
                        WHERE queue_name = $1 
                        AND queued_at > $2
                        """,
                        (queue, datetime.utcnow() - timedelta(hours=24))
                    )
                    
                    row = result.fetchone()
                    
                    stats[queue] = {
                        "current_depth": depth,
                        "total_tasks_24h": row.total_tasks if row else 0,
                        "average_wait_time_seconds": row.avg_wait_time if row and row.avg_wait_time else 0.0,
                        "queued_count": row.queued_count if row else 0,
                        "assigned_count": row.assigned_count if row else 0,
                        "throughput_per_hour": (row.assigned_count or 0) / 24.0 if row else 0.0
                    }
            
            # Add overall metrics
            if not queue_name:
                stats["overall"] = {
                    "total_queues": len(self.priority_queues),
                    "total_depth": sum(s["current_depth"] for s in stats.values()),
                    "tasks_queued_total": self._metrics["tasks_queued"],
                    "tasks_assigned_total": self._metrics["tasks_assigned"],
                    "tasks_expired_total": self._metrics["tasks_expired"]
                }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get queue stats", queue_name=queue_name, error=str(e))
            return {}
    
    async def cancel_task(self, task_id: uuid.UUID) -> bool:
        """Cancel a queued task."""
        try:
            # Find and remove from all Redis queues
            for queue_name in self.priority_queues:
                queue_key = f"task_queue:{queue_name}"
                
                # Get all tasks in queue
                tasks = await self.redis_client.zrange(queue_key, 0, -1)
                
                for task_data in tasks:
                    queued_task = QueuedTask.from_dict(json.loads(task_data))
                    if queued_task.task_id == task_id:
                        # Remove from Redis
                        await self.redis_client.zrem(queue_key, task_data)
                        
                        # Update database
                        async with get_session() as db:
                            await db.execute(
                                update(db.table("task_queue"))
                                .where(db.table("task_queue").c.task_id == task_id)
                                .values(
                                    queue_status=QueueStatus.CANCELLED.value,
                                    updated_at=datetime.utcnow()
                                )
                            )
                            
                            await db.execute(
                                update(Task)
                                .where(Task.id == task_id)
                                .values(
                                    status=TaskStatus.CANCELLED,
                                    updated_at=datetime.utcnow()
                                )
                            )
                            
                            await db.commit()
                        
                        logger.info("Task cancelled successfully", task_id=str(task_id))
                        return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to cancel task", task_id=str(task_id), error=str(e))
            return False
    
    async def retry_task(self, task_id: uuid.UUID) -> bool:
        """Retry a failed task."""
        try:
            async with get_session() as db:
                # Get task details
                result = await db.execute(
                    select(Task).where(Task.id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    return False
                
                # Check retry count
                retry_strategy = task.retry_strategy or {}
                retry_count = retry_strategy.get("retry_count", 0)
                max_retries = retry_strategy.get("max_retries", 3)
                
                if retry_count >= max_retries:
                    logger.warning("Task exceeded max retries", task_id=str(task_id))
                    return False
                
                # Update retry count
                retry_strategy["retry_count"] = retry_count + 1
                retry_strategy["last_retry_at"] = datetime.utcnow().isoformat()
                
                await db.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .values(
                        retry_strategy=retry_strategy,
                        status=TaskStatus.PENDING,
                        error_message=None,
                        updated_at=datetime.utcnow()
                    )
                )
                await db.commit()
                
                # Re-enqueue with higher priority
                priority = TaskPriority.HIGH if retry_count > 1 else task.priority
                return await self.enqueue_task(
                    task_id=task_id,
                    priority=priority,
                    required_capabilities=task.required_capabilities,
                    estimated_effort=task.estimated_effort,
                    metadata={"retry_attempt": retry_count + 1}
                )
                
        except Exception as e:
            logger.error("Failed to retry task", task_id=str(task_id), error=str(e))
            return False
    
    def _get_queue_name_for_priority(self, priority: TaskPriority) -> str:
        """Get queue name based on priority."""
        priority_mapping = {
            TaskPriority.CRITICAL: "critical",
            TaskPriority.HIGH: "high",
            TaskPriority.MEDIUM: "normal",
            TaskPriority.LOW: "low"
        }
        return priority_mapping.get(priority, "normal")
    
    def _check_capability_match(
        self,
        required_capabilities: List[str],
        agent_capabilities: List[str]
    ) -> bool:
        """Check if agent capabilities match task requirements."""
        if not required_capabilities:
            return True
        
        required_set = set(cap.lower() for cap in required_capabilities)
        agent_set = set(cap.lower() for cap in agent_capabilities)
        
        return required_set.issubset(agent_set)
    
    async def _estimate_wait_time(self, queue_name: str, priority_score: float) -> int:
        """Estimate wait time for a task in the queue."""
        try:
            queue_key = f"task_queue:{queue_name}"
            
            # Get tasks with higher priority
            higher_priority_count = await self.redis_client.zcount(
                queue_key, priority_score + 0.1, "+inf"
            )
            
            # Estimate based on average processing time and queue depth
            avg_processing_time = 60  # seconds
            return int(higher_priority_count * avg_processing_time)
            
        except Exception:
            return 300  # Default 5 minutes
    
    async def _expire_tasks_loop(self) -> None:
        """Background loop to expire timed-out tasks."""
        while self._running:
            try:
                await self._expire_timed_out_tasks()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Task expiration loop error", error=str(e))
                await asyncio.sleep(10)
    
    async def _expire_timed_out_tasks(self) -> None:
        """Expire tasks that have exceeded their timeout."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=self.task_timeout_seconds)
            
            async with get_session() as db:
                # Find expired tasks
                result = await db.execute(
                    select(db.table("task_queue").c.task_id, db.table("task_queue").c.queue_name)
                    .where(
                        and_(
                            db.table("task_queue").c.queued_at < cutoff_time,
                            db.table("task_queue").c.queue_status == QueueStatus.QUEUED.value
                        )
                    )
                )
                
                expired_tasks = result.fetchall()
                
                for task_id, queue_name in expired_tasks:
                    # Remove from Redis queue
                    queue_key = f"task_queue:{queue_name}"
                    tasks = await self.redis_client.zrange(queue_key, 0, -1)
                    
                    for task_data in tasks:
                        queued_task = QueuedTask.from_dict(json.loads(task_data))
                        if queued_task.task_id == task_id:
                            await self.redis_client.zrem(queue_key, task_data)
                            break
                    
                    # Update database status
                    await db.execute(
                        update(db.table("task_queue"))
                        .where(db.table("task_queue").c.task_id == task_id)
                        .values(
                            queue_status=QueueStatus.EXPIRED.value,
                            updated_at=datetime.utcnow()
                        )
                    )
                    
                    await db.execute(
                        update(Task)
                        .where(Task.id == task_id)
                        .values(
                            status=TaskStatus.FAILED,
                            error_message="Task expired in queue",
                            updated_at=datetime.utcnow()
                        )
                    )
                    
                    self._metrics["tasks_expired"] += 1
                
                if expired_tasks:
                    await db.commit()
                    logger.info("Expired tasks cleaned up", count=len(expired_tasks))
                    
        except Exception as e:
            logger.error("Failed to expire timed-out tasks", error=str(e))
    
    async def _metrics_collection_loop(self) -> None:
        """Background loop to collect and update metrics."""
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _update_metrics(self) -> None:
        """Update queue performance metrics."""
        try:
            # Update queue depths
            for queue_name in self.priority_queues:
                queue_key = f"task_queue:{queue_name}"
                depth = await self.redis_client.zcard(queue_key)
                self._metrics["queue_depth"][queue_name] = depth
            
            # Calculate average wait time
            async with get_session() as db:
                result = await db.execute(
                    """
                    SELECT AVG(EXTRACT(EPOCH FROM (dequeued_at - queued_at))) as avg_wait_time
                    FROM task_queue 
                    WHERE dequeued_at IS NOT NULL 
                    AND queued_at > $1
                    """,
                    (datetime.utcnow() - timedelta(hours=1),)
                )
                
                row = result.fetchone()
                if row and row.avg_wait_time:
                    self._metrics["average_wait_time"] = row.avg_wait_time
                    
        except Exception as e:
            logger.error("Failed to update metrics", error=str(e))
    
    async def _queue_cleanup_loop(self) -> None:
        """Background loop to clean up old queue records."""
        while self._running:
            try:
                await self._cleanup_old_records()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Queue cleanup loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _cleanup_old_records(self) -> None:
        """Clean up old task queue records."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            async with get_session() as db:
                result = await db.execute(
                    delete(db.table("task_queue"))
                    .where(
                        and_(
                            db.table("task_queue").c.updated_at < cutoff_time,
                            db.table("task_queue").c.queue_status.in_([
                                QueueStatus.ASSIGNED.value,
                                QueueStatus.EXPIRED.value,
                                QueueStatus.CANCELLED.value
                            ])
                        )
                    )
                )
                
                if result.rowcount > 0:
                    await db.commit()
                    logger.info("Cleaned up old queue records", count=result.rowcount)
                    
        except Exception as e:
            logger.error("Failed to cleanup old records", error=str(e))
    
    async def _queue_watcher(self, queue_name: str) -> None:
        """Watch a specific queue for real-time updates."""
        while self._running:
            try:
                # Subscribe to task available events
                pubsub = self.redis_client.pubsub()
                await pubsub.subscribe(f"task_available:{queue_name}")
                
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        data = json.loads(message["data"])
                        logger.debug(
                            "Task available notification",
                            queue_name=queue_name,
                            task_id=data.get("task_id")
                        )
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Queue watcher error", queue_name=queue_name, error=str(e))
                await asyncio.sleep(5)