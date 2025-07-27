"""
Consolidation Engine for context compression and performance optimization.

Provides automated consolidation capabilities during sleep cycles with:
- Context Engine API integration for compression
- Redis stream state preservation and optimization
- Database transaction management during consolidation
- Performance audit and metrics collection
- Multi-stage consolidation pipeline with job tracking
- Token reduction optimization and reporting
- Background processing optimization for compute efficiency
- Intelligent resource allocation and priority scheduling
- Integration with ContextConsolidator for semantic analysis
"""

import asyncio
import json
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import redis.asyncio as redis

from ..models.sleep_wake import (
    ConsolidationJob, ConsolidationStatus, SleepWakeCycle, SleepState
)
from ..models.context import Context
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.context_manager import ContextManager
from ..core.context_consolidator import get_context_consolidator
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class BackgroundTaskPriority:
    """Priority levels for background tasks."""
    CRITICAL = 1000
    HIGH = 800
    NORMAL = 500
    LOW = 200
    MAINTENANCE = 100


class ResourceMonitor:
    """Monitor system resources for intelligent scheduling."""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # Max CPU usage %
        self.memory_threshold = 85.0  # Max memory usage %
        self.disk_io_threshold = 50.0  # Max disk I/O %
        self.monitoring_interval = 5.0  # seconds
        
        # Resource history for trend analysis
        self.cpu_history = deque(maxlen=20)
        self.memory_history = deque(maxlen=20)
        self.disk_io_history = deque(maxlen=20)
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
    
    async def start_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._stop_monitoring = False
            self._monitoring_task = asyncio.create_task(self._monitor_resources())
            logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._stop_monitoring = True
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitor_resources(self):
        """Background task to monitor system resources."""
        while not self._stop_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_history.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_history.append(memory_percent)
                
                # Disk I/O (simplified)
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Calculate I/O rate based on previous measurement
                    current_io = disk_io.read_bytes + disk_io.write_bytes
                    io_percent = min(50.0, current_io / (1024 * 1024 * 100))  # Rough estimate
                    self.disk_io_history.append(io_percent)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_current_load(self) -> Dict[str, float]:
        """Get current system resource load."""
        return {
            "cpu_percent": self.cpu_history[-1] if self.cpu_history else 0.0,
            "memory_percent": self.memory_history[-1] if self.memory_history else 0.0,
            "disk_io_percent": self.disk_io_history[-1] if self.disk_io_history else 0.0
        }
    
    def is_system_under_load(self) -> bool:
        """Check if system is under high load."""
        if not self.cpu_history or not self.memory_history:
            return False
        
        current_cpu = self.cpu_history[-1]
        current_memory = self.memory_history[-1]
        current_disk = self.disk_io_history[-1] if self.disk_io_history else 0.0
        
        return (
            current_cpu > self.cpu_threshold or
            current_memory > self.memory_threshold or
            current_disk > self.disk_io_threshold
        )
    
    def get_resource_trend(self) -> Dict[str, str]:
        """Get resource usage trends."""
        trends = {}
        
        if len(self.cpu_history) >= 3:
            recent_cpu = sum(list(self.cpu_history)[-3:]) / 3
            older_cpu = sum(list(self.cpu_history)[-6:-3]) / 3 if len(self.cpu_history) >= 6 else recent_cpu
            trends["cpu"] = "increasing" if recent_cpu > older_cpu + 5 else "decreasing" if recent_cpu < older_cpu - 5 else "stable"
        else:
            trends["cpu"] = "unknown"
        
        if len(self.memory_history) >= 3:
            recent_memory = sum(list(self.memory_history)[-3:]) / 3
            older_memory = sum(list(self.memory_history)[-6:-3]) / 3 if len(self.memory_history) >= 6 else recent_memory
            trends["memory"] = "increasing" if recent_memory > older_memory + 5 else "decreasing" if recent_memory < older_memory - 5 else "stable"
        else:
            trends["memory"] = "unknown"
        
        return trends


class BackgroundTaskScheduler:
    """Intelligent background task scheduler with resource awareness."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.task_queue: Dict[int, deque] = defaultdict(deque)  # Priority -> tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Scheduling settings
        self.max_concurrent_tasks = 3
        self.low_priority_delay = 30  # seconds
        self.resource_check_interval = 10  # seconds
        
        # Task execution history for optimization
        self.execution_history: Dict[str, List[float]] = defaultdict(list)
        
        # Background scheduler task
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_scheduler = False
    
    async def start_scheduler(self):
        """Start the background task scheduler."""
        await self.resource_monitor.start_monitoring()
        
        if self._scheduler_task is None or self._scheduler_task.done():
            self._stop_scheduler = False
            self._scheduler_task = asyncio.create_task(self._run_scheduler())
            logger.info("Background task scheduler started")
    
    async def stop_scheduler(self):
        """Stop the background task scheduler."""
        self._stop_scheduler = True
        
        # Cancel active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled active task: {task_id}")
        
        # Stop scheduler task
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        await self.resource_monitor.stop_monitoring()
        logger.info("Background task scheduler stopped")
    
    def schedule_task(self, task_id: str, coro, priority: int = BackgroundTaskPriority.NORMAL, 
                     estimated_duration: Optional[float] = None):
        """Schedule a background task."""
        task_info = {
            "task_id": task_id,
            "coro": coro,
            "priority": priority,
            "estimated_duration": estimated_duration,
            "scheduled_at": datetime.utcnow(),
            "retries": 0,
            "max_retries": 2
        }
        
        self.task_queue[priority].append(task_info)
        logger.debug(f"Scheduled background task {task_id} with priority {priority}")
    
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while not self._stop_scheduler:
            try:
                # Check if we can run more tasks
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    # Check system resources
                    if not self.resource_monitor.is_system_under_load():
                        # Get next task to run
                        next_task = self._get_next_task()
                        if next_task:
                            await self._execute_task(next_task)
                    else:
                        logger.debug("System under load, delaying task execution")
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                await asyncio.sleep(self.resource_check_interval)
                
            except Exception as e:
                logger.error(f"Error in background scheduler: {e}")
                await asyncio.sleep(self.resource_check_interval)
    
    def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task to execute based on priority and resource constraints."""
        # Check high priority tasks first
        for priority in sorted(self.task_queue.keys(), reverse=True):
            if self.task_queue[priority]:
                # For low priority tasks, check if enough time has passed
                if priority <= BackgroundTaskPriority.LOW:
                    task = self.task_queue[priority][0]
                    scheduled_time = task["scheduled_at"]
                    if datetime.utcnow() - scheduled_time < timedelta(seconds=self.low_priority_delay):
                        continue
                
                return self.task_queue[priority].popleft()
        
        return None
    
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a background task."""
        task_id = task_info["task_id"]
        
        try:
            start_time = time.time()
            logger.info(f"Starting background task: {task_id}")
            
            # Create and start the task
            task = asyncio.create_task(task_info["coro"])
            self.active_tasks[task_id] = task
            
            # Wait for completion
            result = await task
            
            # Record execution time
            execution_time = time.time() - start_time
            self.execution_history[task_info.get("task_type", "unknown")].append(execution_time)
            
            # Record completion
            completion_info = {
                "task_id": task_id,
                "priority": task_info["priority"],
                "execution_time": execution_time,
                "completed_at": datetime.utcnow(),
                "success": True,
                "result": result
            }
            self.completed_tasks.append(completion_info)
            
            logger.info(f"Background task {task_id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")
            
            # Handle retry
            task_info["retries"] += 1
            if task_info["retries"] <= task_info["max_retries"]:
                # Reschedule with lower priority
                reduced_priority = max(BackgroundTaskPriority.MAINTENANCE, task_info["priority"] - 100)
                self.task_queue[reduced_priority].append(task_info)
                logger.info(f"Rescheduled failed task {task_id} with reduced priority {reduced_priority}")
            else:
                # Record failure
                completion_info = {
                    "task_id": task_id,
                    "priority": task_info["priority"],
                    "completed_at": datetime.utcnow(),
                    "success": False,
                    "error": str(e),
                    "retries": task_info["retries"]
                }
                self.completed_tasks.append(completion_info)
        
        finally:
            # Remove from active tasks
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed task records."""
        # Keep only last 100 completed tasks
        if len(self.completed_tasks) > 100:
            self.completed_tasks = self.completed_tasks[-100:]
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        total_queued = sum(len(queue) for queue in self.task_queue.values())
        
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": total_queued,
            "completed_tasks": len(self.completed_tasks),
            "system_load": self.resource_monitor.get_current_load(),
            "resource_trends": self.resource_monitor.get_resource_trend(),
            "queue_by_priority": {str(priority): len(queue) for priority, queue in self.task_queue.items()},
            "average_execution_times": {
                task_type: sum(times) / len(times) for task_type, times in self.execution_history.items()
            } if self.execution_history else {}
        }


class ContextAgingPolicies:
    """
    Policies for context aging and lifecycle management.
    """
    
    def __init__(self):
        # Age thresholds in hours
        self.fresh_context_threshold = 2       # < 2 hours: fresh
        self.active_context_threshold = 24     # < 24 hours: active  
        self.stale_context_threshold = 168     # < 1 week: stale
        self.archival_context_threshold = 720  # < 1 month: archival
        # > 1 month: candidate for deletion
        
        # Aging scoring weights
        self.recency_weight = 0.4
        self.access_frequency_weight = 0.3
        self.semantic_importance_weight = 0.2
        self.size_efficiency_weight = 0.1
    
    def calculate_context_age_score(self, context: Context) -> float:
        """Calculate a score indicating how urgently a context needs processing."""
        now = datetime.utcnow()
        age_hours = (now - context.created_at).total_seconds() / 3600
        last_access_hours = (now - (context.last_accessed_at or context.created_at)).total_seconds() / 3600
        
        # Recency score (newer = higher score)
        recency_score = max(0, 1 - (age_hours / (30 * 24)))  # 30 day decay
        
        # Access frequency score
        access_score = min(1.0, (context.access_count or 1) / 10)  # Cap at 10 accesses
        
        # Semantic importance (if available)
        semantic_score = getattr(context, 'importance_score', 0.5)
        
        # Size efficiency (larger contexts get higher priority for compression)
        size_score = min(1.0, len(context.content or "") / 10000)  # 10k chars = 1.0
        
        # Weighted final score
        final_score = (
            recency_score * self.recency_weight +
            access_score * self.access_frequency_weight +
            semantic_score * self.semantic_importance_weight +
            size_score * self.size_efficiency_weight
        )
        
        return final_score
    
    def get_aging_category(self, context: Context) -> str:
        """Categorize context based on age."""
        age_hours = (datetime.utcnow() - context.created_at).total_seconds() / 3600
        
        if age_hours < self.fresh_context_threshold:
            return "fresh"
        elif age_hours < self.active_context_threshold:
            return "active"
        elif age_hours < self.stale_context_threshold:
            return "stale"
        elif age_hours < self.archival_context_threshold:
            return "archival"
        else:
            return "deletion_candidate"
    
    def should_consolidate(self, context: Context) -> bool:
        """Determine if a context should be consolidated."""
        category = self.get_aging_category(context)
        
        # Don't consolidate fresh contexts
        if category == "fresh":
            return False
        
        # Always consolidate deletion candidates
        if category == "deletion_candidate":
            return True
        
        # For others, use the scoring system
        score = self.calculate_context_age_score(context)
        
        # Higher threshold for active contexts, lower for stale/archival
        thresholds = {
            "active": 0.7,
            "stale": 0.4,
            "archival": 0.2
        }
        
        return score >= thresholds.get(category, 0.5)


class ContextPrioritizationEngine:
    """
    Intelligent context prioritization for consolidation operations.
    """
    
    def __init__(self):
        self.aging_policies = ContextAgingPolicies()
        
        # Priority weights
        self.urgency_weight = 0.4
        self.impact_weight = 0.3
        self.efficiency_weight = 0.2
        self.resource_weight = 0.1
    
    async def prioritize_contexts_for_consolidation(
        self,
        contexts: List[Context],
        system_load: Dict[str, float],
        agent_id: Optional[UUID] = None
    ) -> List[Tuple[Context, float, Dict[str, Any]]]:
        """
        Prioritize contexts for consolidation based on multiple factors.
        
        Returns:
            List of (context, priority_score, metadata) tuples sorted by priority
        """
        prioritized = []
        
        for context in contexts:
            if not self.aging_policies.should_consolidate(context):
                continue
            
            priority_score, metadata = await self._calculate_priority_score(
                context, system_load, agent_id
            )
            
            prioritized.append((context, priority_score, metadata))
        
        # Sort by priority score (highest first)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        
        return prioritized
    
    async def _calculate_priority_score(
        self,
        context: Context,
        system_load: Dict[str, float],
        agent_id: Optional[UUID]
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate priority score for a context."""
        
        # Urgency score (age-based)
        urgency = self.aging_policies.calculate_context_age_score(context)
        aging_category = self.aging_policies.get_aging_category(context)
        
        # Impact score (potential token savings)
        content_size = len(context.content or "")
        estimated_compression = min(0.8, content_size / 5000)  # Estimate compression potential
        impact = estimated_compression
        
        # Efficiency score (processing cost vs benefit)
        processing_cost = content_size / 1000  # Simplified cost model
        benefit = estimated_compression * content_size
        efficiency = benefit / max(1, processing_cost)
        efficiency = min(1.0, efficiency / 100)  # Normalize
        
        # Resource score (current system load consideration)
        cpu_load = system_load.get("cpu_percent", 0) / 100
        memory_load = system_load.get("memory_percent", 0) / 100
        resource_availability = 1.0 - max(cpu_load, memory_load)
        
        # Calculate weighted final score
        final_score = (
            urgency * self.urgency_weight +
            impact * self.impact_weight +
            efficiency * self.efficiency_weight +
            resource_availability * self.resource_weight
        )
        
        metadata = {
            "urgency": urgency,
            "impact": impact,
            "efficiency": efficiency,
            "resource_availability": resource_availability,
            "aging_category": aging_category,
            "content_size": content_size,
            "estimated_compression": estimated_compression
        }
        
        return final_score, metadata


class ConsolidationEngine:
    """
    Enhanced automated consolidation engine with intelligent context management.
    
    Features:
    - Multi-stage context processing pipeline
    - Context aging and pruning policies
    - Intelligent context prioritization algorithms
    - Background processing optimization for compute efficiency
    - Resource-aware scheduling and execution
    - Performance metrics and optimization tracking
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.context_manager = ContextManager()
        self.context_consolidator = get_context_consolidator()
        
        # Enhanced context management
        self.aging_policies = ContextAgingPolicies()
        self.prioritization_engine = ContextPrioritizationEngine()
        
        # Background processing components
        self.background_scheduler = BackgroundTaskScheduler()
        self.resource_monitor = self.background_scheduler.resource_monitor
        
        # Consolidation settings
        self.max_concurrent_jobs = 3
        self.job_timeout_minutes = 30
        self.compression_target_ratio = 0.6  # Target 60% size reduction
        self.min_context_age_hours = 1  # Only consolidate contexts older than 1 hour
        
        # Performance thresholds
        self.token_reduction_target = 0.55  # 55% reduction goal
        self.max_processing_time_per_mb = 30000  # 30 seconds per MB
        
        # Enhanced processing settings
        self.enable_intelligent_prioritization = True
        self.enable_adaptive_aging = True
        self.max_contexts_per_batch = 50
        self.consolidation_batch_size = 10
        
        # Background processing optimization settings
        self.enable_background_optimization = True
        self.adaptive_scheduling = True
        self.resource_aware_processing = True
        
        # Active consolidation tracking
        self._active_jobs: Dict[UUID, ConsolidationJob] = {}
        self._consolidation_metrics: Dict[str, Any] = {}
        self._background_tasks: Dict[str, Any] = {}
        
        # Performance optimization state
        self._optimization_enabled = False
    
    async def enable_background_optimization(self):
        """Enable background processing optimization."""
        if not self._optimization_enabled:
            await self.background_scheduler.start_scheduler()
            self._optimization_enabled = True
            logger.info("Background processing optimization enabled")
    
    async def disable_background_optimization(self):
        """Disable background processing optimization."""
        if self._optimization_enabled:
            await self.background_scheduler.stop_scheduler()
            self._optimization_enabled = False
            logger.info("Background processing optimization disabled")
    
    async def schedule_background_consolidation(self, agent_id: UUID, priority: int = BackgroundTaskPriority.NORMAL):
        """Schedule background consolidation for an agent."""
        if not self._optimization_enabled:
            logger.warning("Background optimization not enabled, scheduling regular consolidation")
            return await self._schedule_regular_consolidation(agent_id)
        
        task_id = f"bg_consolidation_{agent_id}_{int(datetime.utcnow().timestamp())}"
        
        # Create consolidation coroutine
        consolidation_coro = self._background_consolidation_task(agent_id)
        
        # Schedule with background scheduler
        self.background_scheduler.schedule_task(
            task_id=task_id,
            coro=consolidation_coro,
            priority=priority,
            estimated_duration=300.0  # 5 minutes estimated
        )
        
        logger.info(f"Scheduled background consolidation for agent {agent_id} with priority {priority}")
        return task_id
    
    async def _background_consolidation_task(self, agent_id: UUID) -> Dict[str, Any]:
        """Background consolidation task implementation."""
        try:
            logger.info(f"Starting background consolidation for agent {agent_id}")
            
            # Use ContextConsolidator for intelligent consolidation
            consolidation_result = await self.context_consolidator.consolidate_during_sleep(agent_id)
            
            # Perform additional optimizations based on system resources
            if not self.resource_monitor.is_system_under_load():
                # System has resources available, do more aggressive consolidation
                await self._perform_aggressive_consolidation(agent_id)
            
            # Update metrics
            self._update_background_metrics(agent_id, consolidation_result)
            
            return {
                "agent_id": str(agent_id),
                "consolidation_result": consolidation_result.__dict__,
                "success": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Background consolidation failed for agent {agent_id}: {e}")
            return {
                "agent_id": str(agent_id),
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _perform_aggressive_consolidation(self, agent_id: UUID):
        """Perform more aggressive consolidation when system resources allow."""
        try:
            # Additional context compression
            compressed_count = await self.context_manager.consolidate_stale_contexts(agent_id, batch_size=20)
            
            # Vector index optimization
            if hasattr(self.context_manager, 'optimize_vector_indexes'):
                await self.context_manager.optimize_vector_indexes(agent_id)
            
            # Memory cleanup
            await self._background_memory_cleanup(agent_id)
            
            logger.info(f"Aggressive consolidation completed for agent {agent_id}: {compressed_count} contexts processed")
            
        except Exception as e:
            logger.warning(f"Aggressive consolidation failed for agent {agent_id}: {e}")
    
    async def _background_memory_cleanup(self, agent_id: UUID):
        """Background memory and cache cleanup."""
        try:
            redis_client = get_redis()
            
            # Clean temporary agent data
            temp_patterns = [
                f"temp:{agent_id}:*",
                f"cache:{agent_id}:*",
                f"session:{agent_id}:temp:*"
            ]
            
            cleaned_keys = 0
            for pattern in temp_patterns:
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
                    cleaned_keys += len(keys)
            
            logger.debug(f"Background memory cleanup for agent {agent_id}: {cleaned_keys} keys cleaned")
            
        except Exception as e:
            logger.warning(f"Background memory cleanup failed for agent {agent_id}: {e}")
    
    def _update_background_metrics(self, agent_id: UUID, consolidation_result):
        """Update background processing metrics."""
        agent_key = str(agent_id)
        
        if agent_key not in self._background_tasks:
            self._background_tasks[agent_key] = {
                "total_background_consolidations": 0,
                "total_tokens_saved_background": 0,
                "average_processing_time": 0.0,
                "last_consolidation": None
            }
        
        metrics = self._background_tasks[agent_key]
        metrics["total_background_consolidations"] += 1
        metrics["total_tokens_saved_background"] += consolidation_result.tokens_saved
        metrics["last_consolidation"] = datetime.utcnow().isoformat()
        
        # Update average processing time
        if hasattr(consolidation_result, 'processing_time_ms'):
            old_avg = metrics["average_processing_time"]
            count = metrics["total_background_consolidations"]
            new_time = consolidation_result.processing_time_ms
            metrics["average_processing_time"] = (old_avg * (count - 1) + new_time) / count
    
    async def get_background_optimization_status(self) -> Dict[str, Any]:
        """Get status of background optimization features."""
        scheduler_status = self.background_scheduler.get_scheduler_status() if self._optimization_enabled else {}
        
        return {
            "optimization_enabled": self._optimization_enabled,
            "adaptive_scheduling": self.adaptive_scheduling,
            "resource_aware_processing": self.resource_aware_processing,
            "scheduler_status": scheduler_status,
            "background_tasks_metrics": self._background_tasks,
            "resource_thresholds": {
                "cpu_threshold": self.resource_monitor.cpu_threshold,
                "memory_threshold": self.resource_monitor.memory_threshold,
                "disk_io_threshold": self.resource_monitor.disk_io_threshold
            }
        }
    
    async def optimize_consolidation_schedule(self, agent_id: UUID) -> Dict[str, Any]:
        """Optimize consolidation schedule based on agent activity patterns."""
        try:
            # Get consolidation opportunities analysis
            opportunities = await self._analyze_consolidation_opportunities(agent_id)
            
            # Calculate optimal scheduling
            optimal_schedule = await self._calculate_optimal_schedule(agent_id, opportunities)
            
            # Schedule background tasks based on optimization
            scheduled_tasks = []
            for schedule_item in optimal_schedule:
                task_id = await self.schedule_background_consolidation(
                    agent_id,
                    priority=schedule_item.get("priority", BackgroundTaskPriority.NORMAL)
                )
                scheduled_tasks.append(task_id)
            
            return {
                "agent_id": str(agent_id),
                "optimization_applied": True,
                "scheduled_tasks": scheduled_tasks,
                "opportunities": opportunities,
                "optimal_schedule": optimal_schedule,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize consolidation schedule for agent {agent_id}: {e}")
            return {
                "agent_id": str(agent_id),
                "optimization_applied": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_consolidation_opportunities(self, agent_id: UUID) -> Dict[str, Any]:
        """Analyze consolidation opportunities for an agent."""
        # Use the utility function from context_consolidator
        from ..core.context_consolidator import analyze_consolidation_opportunities
        return await analyze_consolidation_opportunities(agent_id)
    
    async def _calculate_optimal_schedule(self, agent_id: UUID, opportunities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate optimal consolidation schedule."""
        schedule = []
        
        # High priority if many consolidation opportunities
        consolidation_potential = opportunities.get("consolidation_potential", 0)
        
        if consolidation_potential > 50:
            # Immediate high-priority consolidation
            schedule.append({
                "type": "immediate_consolidation",
                "priority": BackgroundTaskPriority.HIGH,
                "reason": "high_consolidation_potential"
            })
        elif consolidation_potential > 20:
            # Normal priority consolidation
            schedule.append({
                "type": "scheduled_consolidation",
                "priority": BackgroundTaskPriority.NORMAL,
                "reason": "moderate_consolidation_potential"
            })
        else:
            # Low priority maintenance
            schedule.append({
                "type": "maintenance_consolidation",
                "priority": BackgroundTaskPriority.LOW,
                "reason": "low_consolidation_potential"
            })
        
        return schedule
    
    async def _schedule_regular_consolidation(self, agent_id: UUID) -> str:
        """Fallback to regular consolidation when background optimization is disabled."""
        try:
            # Create a simple sleep-wake cycle for consolidation
            async with get_async_session() as session:
                cycle = SleepWakeCycle(
                    agent_id=agent_id,
                    cycle_type="background_consolidation",
                    sleep_state=SleepState.SLEEPING,
                    sleep_time=datetime.utcnow()
                )
                session.add(cycle)
                await session.commit()
                await session.refresh(cycle)
                
                # Start consolidation cycle
                success = await self.start_consolidation_cycle(cycle.id, agent_id)
                
                return f"regular_consolidation_{cycle.id}"
                
        except Exception as e:
            logger.error(f"Failed to schedule regular consolidation for agent {agent_id}: {e}")
            return f"failed_consolidation_{agent_id}"
    
    async def start_consolidation_cycle(
        self,
        cycle_id: UUID,
        agent_id: Optional[UUID] = None
    ) -> bool:
        """
        Start a complete consolidation cycle for a sleep-wake cycle.
        
        Args:
            cycle_id: Sleep-wake cycle ID
            agent_id: Agent ID for agent-specific consolidation
            
        Returns:
            True if consolidation cycle started successfully
        """
        try:
            logger.info(f"Starting consolidation cycle {cycle_id} for agent {agent_id}")
            
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if not cycle:
                    logger.error(f"Sleep-wake cycle {cycle_id} not found")
                    return False
                
                # Update cycle state
                cycle.sleep_state = SleepState.CONSOLIDATING
                cycle.updated_at = datetime.utcnow()
                await session.commit()
            
            # Create consolidation jobs pipeline
            jobs = await self._create_consolidation_pipeline(cycle_id, agent_id)
            
            if not jobs:
                logger.warning(f"No consolidation jobs created for cycle {cycle_id}")
                return False
            
            # Execute jobs in priority order
            success = await self._execute_consolidation_pipeline(jobs)
            
            # Update cycle with results
            await self._finalize_consolidation_cycle(cycle_id, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting consolidation cycle {cycle_id}: {e}")
            await self._handle_consolidation_error(cycle_id, str(e))
            return False
    
    async def _create_consolidation_pipeline(
        self,
        cycle_id: UUID,
        agent_id: Optional[UUID]
    ) -> List[ConsolidationJob]:
        """Create the consolidation job pipeline."""
        jobs = []
        
        try:
            async with get_async_session() as session:
                # Job 1: Context compression (highest priority)
                context_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="context_compression",
                    status=ConsolidationStatus.PENDING,
                    priority=100,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=2
                )
                session.add(context_job)
                jobs.append(context_job)
                
                # Job 2: Vector index update (high priority)
                vector_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="vector_index_update",
                    status=ConsolidationStatus.PENDING,
                    priority=80,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=3
                )
                session.add(vector_job)
                jobs.append(vector_job)
                
                # Job 3: Redis stream cleanup (medium priority)
                redis_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="redis_stream_cleanup",
                    status=ConsolidationStatus.PENDING,
                    priority=60,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=2
                )
                session.add(redis_job)
                jobs.append(redis_job)
                
                # Job 4: Performance audit (low priority)
                audit_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="performance_audit",
                    status=ConsolidationStatus.PENDING,
                    priority=40,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=1
                )
                session.add(audit_job)
                jobs.append(audit_job)
                
                # Job 5: Database maintenance (lowest priority)
                db_job = ConsolidationJob(
                    cycle_id=cycle_id,
                    job_type="database_maintenance",
                    status=ConsolidationStatus.PENDING,
                    priority=20,
                    input_data={"agent_id": str(agent_id) if agent_id else None},
                    max_retries=1
                )
                session.add(db_job)
                jobs.append(db_job)
                
                await session.commit()
                
                # Sort by priority
                jobs.sort(key=lambda j: j.priority, reverse=True)
                
                logger.info(f"Created {len(jobs)} consolidation jobs for cycle {cycle_id}")
                return jobs
                
        except Exception as e:
            logger.error(f"Error creating consolidation pipeline: {e}")
            return []
    
    async def _execute_consolidation_pipeline(self, jobs: List[ConsolidationJob]) -> bool:
        """Execute consolidation jobs with concurrency control."""
        try:
            # Group jobs by priority for batch execution
            priority_groups = {}
            for job in jobs:
                if job.priority not in priority_groups:
                    priority_groups[job.priority] = []
                priority_groups[job.priority].append(job)
            
            overall_success = True
            
            # Execute groups in priority order
            for priority in sorted(priority_groups.keys(), reverse=True):
                group_jobs = priority_groups[priority]
                logger.info(f"Executing priority {priority} jobs: {[j.job_type for j in group_jobs]}")
                
                # Execute jobs in this priority group concurrently
                semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
                tasks = [
                    self._execute_consolidation_job(job, semaphore)
                    for job in group_jobs
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Job {group_jobs[i].job_type} failed with exception: {result}")
                        overall_success = False
                    elif not result:
                        logger.error(f"Job {group_jobs[i].job_type} failed")
                        overall_success = False
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Error executing consolidation pipeline: {e}")
            return False
    
    async def _execute_consolidation_job(
        self,
        job: ConsolidationJob,
        semaphore: asyncio.Semaphore
    ) -> bool:
        """Execute a single consolidation job."""
        async with semaphore:
            start_time = time.time()
            
            try:
                async with get_async_session() as session:
                    # Update job status
                    await session.refresh(job)
                    job.status = ConsolidationStatus.IN_PROGRESS
                    job.started_at = datetime.utcnow()
                    await session.commit()
                
                self._active_jobs[job.id] = job
                
                logger.info(f"Executing consolidation job: {job.job_type}")
                
                # Execute based on job type
                success = False
                if job.job_type == "context_compression":
                    success = await self._execute_context_compression(job)
                elif job.job_type == "vector_index_update":
                    success = await self._execute_vector_index_update(job)
                elif job.job_type == "redis_stream_cleanup":
                    success = await self._execute_redis_cleanup(job)
                elif job.job_type == "performance_audit":
                    success = await self._execute_performance_audit(job)
                elif job.job_type == "database_maintenance":
                    success = await self._execute_database_maintenance(job)
                else:
                    logger.error(f"Unknown job type: {job.job_type}")
                    success = False
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Update job status
                async with get_async_session() as session:
                    await session.refresh(job)
                    job.status = ConsolidationStatus.COMPLETED if success else ConsolidationStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.processing_time_ms = processing_time_ms
                    job.progress_percentage = 100.0 if success else job.progress_percentage
                    
                    if not success:
                        job.retry_count += 1
                        if job.can_retry:
                            job.status = ConsolidationStatus.PENDING
                            job.completed_at = None
                            logger.info(f"Job {job.job_type} will be retried (attempt {job.retry_count})")
                    
                    await session.commit()
                
                del self._active_jobs[job.id]
                
                if success:
                    logger.info(f"Job {job.job_type} completed successfully in {processing_time_ms:.0f}ms")
                else:
                    logger.error(f"Job {job.job_type} failed after {processing_time_ms:.0f}ms")
                
                return success
                
            except Exception as e:
                logger.error(f"Error executing job {job.job_type}: {e}")
                
                # Update job with error
                async with get_async_session() as session:
                    await session.refresh(job)
                    job.status = ConsolidationStatus.FAILED
                    job.error_message = str(e)
                    job.completed_at = datetime.utcnow()
                    job.processing_time_ms = (time.time() - start_time) * 1000
                    await session.commit()
                
                if job.id in self._active_jobs:
                    del self._active_jobs[job.id]
                
                return False
    
    async def _execute_context_compression(self, job: ConsolidationJob) -> bool:
        """Execute enhanced context compression with intelligent prioritization and aging policies."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            if not agent_id:
                logger.warning("No agent_id provided for context compression")
                return False
            
            logger.info(f"Starting enhanced context consolidation for agent {agent_id}")
            
            # Phase 1: Get current system load for intelligent prioritization
            system_load = self.resource_monitor.get_current_load()
            
            # Phase 2: Get contexts eligible for consolidation
            contexts_to_evaluate = await self._get_contexts_for_compression(agent_id)
            
            # Phase 3: Apply intelligent prioritization
            if self.enable_intelligent_prioritization and contexts_to_evaluate:
                prioritized_contexts = await self.prioritization_engine.prioritize_contexts_for_consolidation(
                    contexts_to_evaluate, system_load, agent_id
                )
                
                # Update job progress
                async with get_async_session() as session:
                    await session.refresh(job)
                    job.progress_percentage = 20.0  # Prioritization complete
                    await session.commit()
                
                logger.info(f"Prioritized {len(prioritized_contexts)} contexts for consolidation")
            else:
                # Fallback to simple ordering by age
                prioritized_contexts = [(ctx, 0.5, {}) for ctx in contexts_to_evaluate]
            
            # Phase 4: Use ContextConsolidator for intelligent compression on high-priority contexts
            high_priority_contexts = [
                ctx for ctx, score, metadata in prioritized_contexts 
                if score >= 0.6  # High priority threshold
            ][:self.consolidation_batch_size]
            
            consolidation_result = await self.context_consolidator.consolidate_during_sleep(agent_id)
            
            # Update job progress
            async with get_async_session() as session:
                await session.refresh(job)
                job.progress_percentage = 50.0  # Initial consolidation complete
                await session.commit()
            
            # Phase 5: Process remaining contexts with intelligent batching
            additional_tokens_saved = 0
            additional_compressed = 0
            contexts_by_category = {}
            
            remaining_contexts = [
                (ctx, score, metadata) for ctx, score, metadata in prioritized_contexts
                if not getattr(ctx, 'is_consolidated', False)
            ]
            
            # Group contexts by aging category for batch processing
            for context, score, metadata in remaining_contexts:
                category = metadata.get('aging_category', 'unknown')
                if category not in contexts_by_category:
                    contexts_by_category[category] = []
                contexts_by_category[category].append((context, score, metadata))
            
            # Process contexts by category priority
            category_priority = ["deletion_candidate", "archival", "stale", "active"]
            total_remaining = len(remaining_contexts)
            processed_count = 0
            
            for category in category_priority:
                if category not in contexts_by_category:
                    continue
                
                category_contexts = contexts_by_category[category]
                logger.info(f"Processing {len(category_contexts)} {category} contexts")
                
                for context, score, metadata in category_contexts:
                    try:
                        # Update progress
                        processed_count += 1
                        progress = 50.0 + (processed_count / total_remaining) * 45.0
                        async with get_async_session() as session:
                            await session.refresh(job)
                            job.progress_percentage = progress
                            await session.commit()
                        
                        # Apply aging-specific compression strategy
                        compression_level = self._get_compression_level_for_category(category)
                        
                        # Compress context using enhanced strategy
                        compression_result = await self._compress_context_with_aging_policy(
                            context, compression_level, metadata
                        )
                        
                        if compression_result:
                            additional_tokens_saved += compression_result.get("tokens_saved", 0)
                            additional_compressed += 1
                            
                            logger.debug(
                                f"Compressed {category} context {context.id}: "
                                f"saved {compression_result.get('tokens_saved', 0)} tokens "
                                f"(priority: {score:.2f})"
                            )
                        
                        # Check if we should pause due to high system load
                        if self.resource_aware_processing and processed_count % 5 == 0:
                            current_load = self.resource_monitor.get_current_load()
                            if self.resource_monitor.is_system_under_load():
                                logger.info("High system load detected, pausing context compression")
                                await asyncio.sleep(2)  # Brief pause
                        
                    except Exception as e:
                        logger.error(f"Error compressing {category} context {context.id}: {e}")
                        continue
            
            # Phase 6: Calculate comprehensive results
            total_tokens_saved = consolidation_result.tokens_saved + additional_tokens_saved
            total_contexts_processed = consolidation_result.contexts_processed + additional_compressed
            
            # Enhanced efficiency metrics
            compression_ratio = total_tokens_saved / max(1, total_contexts_processed * 1000)
            prioritization_efficiency = len(prioritized_contexts) / max(1, len(contexts_to_evaluate))
            
            job.output_data = {
                "intelligent_consolidation": {
                    "contexts_processed": consolidation_result.contexts_processed,
                    "contexts_merged": consolidation_result.contexts_merged,
                    "contexts_archived": consolidation_result.contexts_archived,
                    "redundant_removed": consolidation_result.redundant_contexts_removed,
                    "tokens_saved": consolidation_result.tokens_saved,
                    "consolidation_ratio": consolidation_result.consolidation_ratio,
                    "efficiency_score": consolidation_result.efficiency_score
                },
                "enhanced_compression": {
                    "contexts_evaluated": len(contexts_to_evaluate),
                    "contexts_prioritized": len(prioritized_contexts),
                    "contexts_compressed": additional_compressed,
                    "tokens_saved": additional_tokens_saved,
                    "prioritization_efficiency": prioritization_efficiency,
                    "contexts_by_category": {
                        category: len(contexts) for category, contexts in contexts_by_category.items()
                    }
                },
                "total_summary": {
                    "total_contexts_processed": total_contexts_processed,
                    "total_tokens_saved": total_tokens_saved,
                    "overall_compression_ratio": compression_ratio,
                    "system_load_during_processing": system_load
                }
            }
            job.tokens_processed = total_contexts_processed * 1000
            job.tokens_saved = total_tokens_saved
            
            logger.info(
                f"Enhanced context compression completed: {total_contexts_processed} contexts processed, "
                f"{total_tokens_saved} tokens saved ({compression_ratio:.2%} reduction), "
                f"efficiency score: {consolidation_result.efficiency_score:.2f}, "
                f"prioritization efficiency: {prioritization_efficiency:.2%}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced context compression: {e}")
            return False
    
    def _get_compression_level_for_category(self, category: str) -> str:
        """Get appropriate compression level based on aging category."""
        compression_levels = {
            "deletion_candidate": "maximum",
            "archival": "aggressive", 
            "stale": "normal",
            "active": "light"
        }
        return compression_levels.get(category, "normal")
    
    async def _compress_context_with_aging_policy(
        self,
        context: Context,
        compression_level: str,
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Compress a context using aging policy-informed strategies."""
        try:
            # Get aging category for strategy selection
            aging_category = metadata.get('aging_category', 'unknown')
            
            # Apply category-specific compression strategies
            if aging_category == "deletion_candidate":
                # Aggressive compression + archival preparation
                result = await self.context_manager.compress_context(
                    context.id,
                    compression_level="maximum",
                    preserve_references=False
                )
                # Mark for potential deletion after compression
                if result and result.get("compression_ratio", 0) > 0.8:
                    await self._mark_context_for_deletion(context.id)
                
            elif aging_category == "archival":
                # High compression + metadata preservation
                result = await self.context_manager.compress_context(
                    context.id,
                    compression_level="aggressive",
                    preserve_metadata=True
                )
                
            elif aging_category == "stale":
                # Standard compression + reference updates
                result = await self.context_manager.compress_context(
                    context.id,
                    compression_level="normal",
                    update_references=True
                )
                
            else:  # active or fresh
                # Light compression preserving full functionality
                result = await self.context_manager.compress_context(
                    context.id,
                    compression_level="light",
                    preserve_all=True
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in aging policy compression for context {context.id}: {e}")
            return None
    
    async def _mark_context_for_deletion(self, context_id: UUID) -> None:
        """Mark a context for future deletion after successful compression."""
        try:
            async with get_async_session() as session:
                context = await session.get(Context, context_id)
                if context:
                    # Add deletion marker to context metadata
                    if not context.metadata:
                        context.metadata = {}
                    context.metadata["marked_for_deletion"] = True
                    context.metadata["deletion_eligible_after"] = (
                        datetime.utcnow() + timedelta(days=7)
                    ).isoformat()
                    await session.commit()
                    logger.debug(f"Marked context {context_id} for deletion")
        except Exception as e:
            logger.warning(f"Failed to mark context {context_id} for deletion: {e}")
    
    async def _execute_vector_index_update(self, job: ConsolidationJob) -> bool:
        """Execute vector index update job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            # Update vector indexes for compressed contexts
            updated_count = await self.context_manager.rebuild_vector_indexes(agent_id)
            
            job.output_data = {
                "indexes_updated": updated_count,
                "agent_id": str(agent_id) if agent_id else "system"
            }
            
            logger.info(f"Vector index update completed: {updated_count} indexes updated")
            return True
            
        except Exception as e:
            logger.error(f"Error in vector index update: {e}")
            return False
    
    async def _execute_redis_cleanup(self, job: ConsolidationJob) -> bool:
        """Execute Redis stream cleanup job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            redis_client = get_redis()
            
            # Define cleanup patterns
            if agent_id:
                patterns = [
                    f"agent:{agent_id}:*",
                    f"tasks:{agent_id}:*",
                    f"temp:{agent_id}:*"
                ]
            else:
                patterns = ["temp:*", "cache:*"]
            
            cleaned_keys = 0
            cleaned_streams = 0
            
            for pattern in patterns:
                keys = await redis_client.keys(pattern)
                
                for key in keys:
                    try:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        
                        # Check if it's a stream
                        key_type = await redis_client.type(key)
                        
                        if key_type == b"stream":
                            # Trim old messages from streams
                            try:
                                info = await redis_client.xinfo_stream(key)
                                length = info.get("length", 0)
                                
                                if length > 1000:  # Keep only last 1000 messages
                                    await redis_client.xtrim(key, maxlen=1000, approximate=True)
                                    cleaned_streams += 1
                            except Exception as e:
                                logger.warning(f"Could not trim stream {key_str}: {e}")
                        
                        elif "temp:" in key_str or "cache:" in key_str:
                            # Delete temporary keys
                            await redis_client.delete(key)
                            cleaned_keys += 1
                            
                    except Exception as e:
                        logger.warning(f"Error cleaning Redis key {key}: {e}")
            
            job.output_data = {
                "cleaned_keys": cleaned_keys,
                "cleaned_streams": cleaned_streams,
                "agent_id": str(agent_id) if agent_id else "system"
            }
            
            logger.info(f"Redis cleanup completed: {cleaned_keys} keys deleted, {cleaned_streams} streams trimmed")
            return True
            
        except Exception as e:
            logger.error(f"Error in Redis cleanup: {e}")
            return False
    
    async def _execute_performance_audit(self, job: ConsolidationJob) -> bool:
        """Execute performance audit job."""
        try:
            agent_id = UUID(job.input_data["agent_id"]) if job.input_data.get("agent_id") else None
            
            # Collect performance metrics
            metrics = await self._collect_performance_metrics(agent_id)
            
            job.output_data = metrics
            
            # Store metrics for monitoring
            self._consolidation_metrics[str(job.cycle_id)] = metrics
            
            logger.info(f"Performance audit completed for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in performance audit: {e}")
            return False
    
    async def _execute_database_maintenance(self, job: ConsolidationJob) -> bool:
        """Execute database maintenance job."""
        try:
            async with get_async_session() as session:
                # Run cleanup functions
                result = await session.execute(
                    "SELECT cleanup_old_context_analytics(90)"
                )
                analytics_cleaned = result.scalar()
                
                # Vacuum analyze for performance
                await session.execute("VACUUM ANALYZE contexts")
                await session.execute("VACUUM ANALYZE sleep_wake_cycles")
                
                job.output_data = {
                    "analytics_records_cleaned": analytics_cleaned or 0,
                    "tables_vacuumed": ["contexts", "sleep_wake_cycles"]
                }
            
            logger.info("Database maintenance completed")
            return True
            
        except Exception as e:
            logger.error(f"Error in database maintenance: {e}")
            return False
    
    async def _get_contexts_for_compression(self, agent_id: Optional[UUID]) -> List[Context]:
        """Get contexts that are candidates for compression."""
        try:
            async with get_async_session() as session:
                # Get contexts older than threshold that haven't been compressed recently
                cutoff_time = datetime.utcnow() - timedelta(hours=self.min_context_age_hours)
                
                query = select(Context).where(
                    and_(
                        Context.created_at < cutoff_time,
                        or_(
                            Context.is_consolidated == False,
                            Context.is_consolidated.is_(None)
                        )
                    )
                )
                
                if agent_id:
                    query = query.where(Context.agent_id == agent_id)
                
                # Limit to avoid overwhelming the system
                query = query.limit(100)
                
                result = await session.execute(query)
                return list(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting contexts for compression: {e}")
            return []
    
    async def _collect_performance_metrics(self, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Collect performance metrics for audit."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(agent_id) if agent_id else None
        }
        
        try:
            async with get_async_session() as session:
                # Context metrics
                context_query = select(func.count(Context.id)).where(
                    Context.agent_id == agent_id if agent_id else True
                )
                total_contexts = await session.scalar(context_query)
                
                compressed_query = select(func.count(Context.id)).where(
                    and_(
                        Context.agent_id == agent_id if agent_id else True,
                        Context.is_consolidated == True
                    )
                )
                compressed_contexts = await session.scalar(compressed_query)
                
                metrics.update({
                    "total_contexts": total_contexts or 0,
                    "compressed_contexts": compressed_contexts or 0,
                    "compression_percentage": (compressed_contexts / total_contexts * 100) if total_contexts > 0 else 0
                })
                
                # Sleep cycle metrics
                cycle_query = select(func.count(SleepWakeCycle.id)).where(
                    SleepWakeCycle.agent_id == agent_id if agent_id else True
                )
                total_cycles = await session.scalar(cycle_query)
                
                metrics["total_sleep_cycles"] = total_cycles or 0
                
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    async def _finalize_consolidation_cycle(self, cycle_id: UUID, success: bool) -> None:
        """Finalize consolidation cycle with results."""
        try:
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if not cycle:
                    return
                
                # Get consolidation job results
                jobs_query = select(ConsolidationJob).where(ConsolidationJob.cycle_id == cycle_id)
                result = await session.execute(jobs_query)
                jobs = result.scalars().all()
                
                # Calculate overall metrics
                total_tokens_saved = sum(job.tokens_saved or 0 for job in jobs)
                total_tokens_processed = sum(job.tokens_processed or 0 for job in jobs)
                total_processing_time = sum(job.processing_time_ms or 0 for job in jobs)
                
                token_reduction = total_tokens_saved / total_tokens_processed if total_tokens_processed > 0 else 0
                
                # Update cycle with consolidation results
                cycle.token_reduction_achieved = token_reduction
                cycle.consolidation_time_ms = total_processing_time
                cycle.sleep_state = SleepState.SLEEPING  # Return to sleeping state
                cycle.performance_metrics = {
                    "tokens_saved": total_tokens_saved,
                    "tokens_processed": total_tokens_processed,
                    "processing_time_ms": total_processing_time,
                    "jobs_completed": len([j for j in jobs if j.status == ConsolidationStatus.COMPLETED]),
                    "jobs_failed": len([j for j in jobs if j.status == ConsolidationStatus.FAILED])
                }
                cycle.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info(
                    f"Consolidation cycle {cycle_id} finalized: "
                    f"{token_reduction:.2%} token reduction, "
                    f"{total_processing_time:.0f}ms processing time"
                )
                
        except Exception as e:
            logger.error(f"Error finalizing consolidation cycle {cycle_id}: {e}")
    
    async def _handle_consolidation_error(self, cycle_id: UUID, error_message: str) -> None:
        """Handle consolidation cycle errors."""
        try:
            async with get_async_session() as session:
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if cycle:
                    cycle.sleep_state = SleepState.ERROR
                    cycle.error_details = {"consolidation_error": error_message}
                    cycle.updated_at = datetime.utcnow()
                    await session.commit()
                    
        except Exception as e:
            logger.error(f"Error handling consolidation error for cycle {cycle_id}: {e}")
    
    async def get_consolidation_status(self, cycle_id: UUID) -> Dict[str, Any]:
        """Get status of consolidation for a cycle."""
        try:
            async with get_async_session() as session:
                jobs_query = select(ConsolidationJob).where(ConsolidationJob.cycle_id == cycle_id)
                result = await session.execute(jobs_query)
                jobs = result.scalars().all()
                
                status = {
                    "cycle_id": str(cycle_id),
                    "total_jobs": len(jobs),
                    "completed_jobs": len([j for j in jobs if j.status == ConsolidationStatus.COMPLETED]),
                    "failed_jobs": len([j for j in jobs if j.status == ConsolidationStatus.FAILED]),
                    "in_progress_jobs": len([j for j in jobs if j.status == ConsolidationStatus.IN_PROGRESS]),
                    "pending_jobs": len([j for j in jobs if j.status == ConsolidationStatus.PENDING]),
                    "jobs": [job.to_dict() for job in jobs]
                }
                
                return status
                
        except Exception as e:
            logger.error(f"Error getting consolidation status for cycle {cycle_id}: {e}")
            return {}


# Global consolidation engine instance
_consolidation_engine_instance: Optional[ConsolidationEngine] = None


def get_consolidation_engine() -> ConsolidationEngine:
    """Get the global consolidation engine instance."""
    global _consolidation_engine_instance
    if _consolidation_engine_instance is None:
        _consolidation_engine_instance = ConsolidationEngine()
    return _consolidation_engine_instance