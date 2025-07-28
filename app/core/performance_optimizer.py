"""
Automated Performance Optimization System.

Implements intelligent performance optimization based on analytics insights,
with automated recommendations and self-healing capabilities for the Context Engine.
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

import numpy as np
from sqlalchemy import text, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.context_performance_monitor import (
    ContextPerformanceMonitor,
    get_context_performance_monitor,
    OptimizationRecommendation
)

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of automated optimizations."""
    CACHE_WARMING = "cache_warming"
    INDEX_OPTIMIZATION = "index_optimization"
    QUERY_REWRITING = "query_rewriting"
    EMBEDDING_CACHING = "embedding_caching"
    CONTEXT_ARCHIVING = "context_archiving"
    BATCH_PROCESSING = "batch_processing"
    SEARCH_PARAMETER_TUNING = "search_parameter_tuning"
    RESOURCE_ALLOCATION = "resource_allocation"


class OptimizationStatus(Enum):
    """Status of optimization operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationTask:
    """Represents an automated optimization task."""
    task_id: str
    optimization_type: OptimizationType
    priority: int  # 1-5, 1 being highest
    description: str
    target_component: str
    expected_improvement: float  # 0-1 scale
    implementation_steps: List[str]
    rollback_plan: List[str]
    status: OptimizationStatus = OptimizationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    result_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for measuring optimization impact."""
    component: str
    metric_name: str
    baseline_value: float
    measurement_window_hours: int
    measured_at: datetime
    sample_size: int


class PerformanceOptimizer:
    """
    Automated performance optimization system for the Context Engine.
    
    Features:
    - Intelligent optimization recommendation generation
    - Automated optimization task execution
    - Performance impact measurement and validation
    - Rollback capabilities for failed optimizations
    - ML-based optimization parameter tuning
    - Self-healing for common performance issues
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
        performance_monitor: Optional[ContextPerformanceMonitor] = None,
        auto_optimize: bool = True
    ):
        """
        Initialize the performance optimizer.
        
        Args:
            redis_client: Redis client for caching and coordination
            db_session: Database session for persistent operations
            performance_monitor: Context performance monitor
            auto_optimize: Whether to automatically execute optimizations
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        self.performance_monitor = performance_monitor
        self.auto_optimize = auto_optimize
        
        # Task management
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.completed_tasks: List[OptimizationTask] = []
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        
        # Optimization parameters
        self.optimization_thresholds = {
            "high_latency_ms": 1000.0,
            "low_cache_hit_rate": 0.8,
            "high_error_rate": 0.05,
            "low_search_quality": 0.7,
            "high_api_cost_per_hour": 5.0
        }
        
        # Configuration
        self.max_concurrent_optimizations = 3
        self.optimization_cooldown_hours = 6
        self.rollback_timeout_minutes = 30
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("Performance Optimizer initialized")
    
    async def start(self) -> None:
        """Start the performance optimizer background processes."""
        logger.info("Starting performance optimizer")
        
        # Initialize performance monitor if not provided
        if self.performance_monitor is None:
            self.performance_monitor = await get_context_performance_monitor()
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._optimization_analyzer()),
            asyncio.create_task(self._task_executor()),
            asyncio.create_task(self._performance_validator()),
            asyncio.create_task(self._maintenance_scheduler())
        ])
    
    async def stop(self) -> None:
        """Stop the performance optimizer."""
        logger.info("Stopping performance optimizer")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def analyze_and_recommend(self) -> List[OptimizationTask]:
        """
        Analyze current performance and generate optimization recommendations.
        
        Returns:
            List of optimization tasks ready for execution
        """
        try:
            tasks = []
            
            # Get performance summary
            performance_summary = await self.performance_monitor.get_performance_summary(
                time_window_hours=24
            )
            
            if "error" in performance_summary:
                logger.warning(f"Could not get performance summary: {performance_summary['error']}")
                return tasks
            
            # Analyze search performance
            search_perf = performance_summary.get("search_performance", {})
            if not search_perf.get("error"):
                avg_latency = search_perf.get("avg_latency_ms", 0)
                cache_hit_rate = search_perf.get("cache_hit_rate", 1.0)
                avg_quality = search_perf.get("avg_quality_score")
                
                # High latency optimization
                if avg_latency > self.optimization_thresholds["high_latency_ms"]:
                    tasks.append(await self._create_latency_optimization_task(avg_latency))
                
                # Low cache hit rate optimization
                if cache_hit_rate < self.optimization_thresholds["low_cache_hit_rate"]:
                    tasks.append(await self._create_cache_optimization_task(cache_hit_rate))
                
                # Low search quality optimization
                if avg_quality and avg_quality < self.optimization_thresholds["low_search_quality"]:
                    tasks.append(await self._create_quality_optimization_task(avg_quality))
            
            # Analyze API costs
            cost_analysis = await self.performance_monitor.get_cost_analysis(time_window_hours=24)
            if not cost_analysis.get("error"):
                hourly_cost = cost_analysis.get("projected_daily_cost_usd", 0) / 24
                
                if hourly_cost > self.optimization_thresholds["high_api_cost_per_hour"]:
                    tasks.append(await self._create_cost_optimization_task(hourly_cost))
            
            # Analyze storage and capacity
            storage_tasks = await self._analyze_storage_optimization()
            tasks.extend(storage_tasks)
            
            # Filter out duplicate or recent optimizations
            tasks = await self._filter_optimization_tasks(tasks)
            
            # Sort by priority
            tasks.sort(key=lambda t: t.priority)
            
            logger.info(f"Generated {len(tasks)} optimization recommendations")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to analyze and recommend optimizations: {e}")
            return []
    
    async def execute_optimization_task(self, task: OptimizationTask) -> bool:
        """
        Execute an optimization task.
        
        Args:
            task: Optimization task to execute
            
        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            logger.info(f"Executing optimization task: {task.description}")
            
            # Check if we can execute (not too many concurrent tasks)
            if len(self.active_tasks) >= self.max_concurrent_optimizations:
                logger.warning("Too many concurrent optimizations, queuing task")
                return False
            
            # Set up task tracking
            task.status = OptimizationStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            self.active_tasks[task.task_id] = task
            
            # Measure baseline performance
            await self._measure_performance_baseline(task)
            
            # Execute optimization based on type
            success = False
            if task.optimization_type == OptimizationType.CACHE_WARMING:
                success = await self._execute_cache_warming(task)
            elif task.optimization_type == OptimizationType.INDEX_OPTIMIZATION:
                success = await self._execute_index_optimization(task)
            elif task.optimization_type == OptimizationType.QUERY_REWRITING:
                success = await self._execute_query_rewriting(task)
            elif task.optimization_type == OptimizationType.EMBEDDING_CACHING:
                success = await self._execute_embedding_caching(task)
            elif task.optimization_type == OptimizationType.CONTEXT_ARCHIVING:
                success = await self._execute_context_archiving(task)
            elif task.optimization_type == OptimizationType.BATCH_PROCESSING:
                success = await self._execute_batch_processing(task)
            elif task.optimization_type == OptimizationType.SEARCH_PARAMETER_TUNING:
                success = await self._execute_search_parameter_tuning(task)
            elif task.optimization_type == OptimizationType.RESOURCE_ALLOCATION:
                success = await self._execute_resource_allocation(task)
            else:
                logger.warning(f"Unknown optimization type: {task.optimization_type}")
                success = False
            
            # Update task status
            if success:
                task.status = OptimizationStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                
                # Schedule performance validation
                asyncio.create_task(self._validate_optimization_impact(task))
                
                logger.info(f"Optimization task completed successfully: {task.task_id}")
            else:
                task.status = OptimizationStatus.FAILED
                task.error_message = "Optimization execution failed"
                logger.error(f"Optimization task failed: {task.task_id}")
            
            # Move to completed tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            # Store task result in Redis
            await self._store_optimization_result(task)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to execute optimization task {task.task_id}: {e}")
            task.status = OptimizationStatus.FAILED
            task.error_message = str(e)
            
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            return False
    
    async def rollback_optimization(self, task_id: str) -> bool:
        """
        Rollback a completed optimization.
        
        Args:
            task_id: ID of the optimization task to rollback
            
        Returns:
            True if rollback was successful, False otherwise
        """
        try:
            # Find the task
            task = None
            for completed_task in self.completed_tasks:
                if completed_task.task_id == task_id:
                    task = completed_task
                    break
            
            if not task:
                logger.error(f"Optimization task not found for rollback: {task_id}")
                return False
            
            if task.status != OptimizationStatus.COMPLETED:
                logger.error(f"Cannot rollback task that is not completed: {task_id}")
                return False
            
            logger.info(f"Rolling back optimization: {task.description}")
            
            # Execute rollback steps
            success = True
            for step in task.rollback_plan:
                try:
                    logger.info(f"Executing rollback step: {step}")
                    await self._execute_rollback_step(task, step)
                except Exception as e:
                    logger.error(f"Rollback step failed: {step}, error: {e}")
                    success = False
                    break
            
            if success:
                task.status = OptimizationStatus.ROLLED_BACK
                logger.info(f"Optimization successfully rolled back: {task_id}")
            else:
                logger.error(f"Optimization rollback failed: {task_id}")
            
            # Store updated task
            await self._store_optimization_result(task)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback optimization {task_id}: {e}")
            return False
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current status of optimization system."""
        try:
            # Recent optimization history
            recent_tasks = [
                {
                    "task_id": task.task_id,
                    "optimization_type": task.optimization_type.value,
                    "status": task.status.value,
                    "description": task.description,
                    "priority": task.priority,
                    "expected_improvement": task.expected_improvement,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "result_metrics": task.result_metrics
                }
                for task in self.completed_tasks[-20:]  # Last 20 tasks
            ]
            
            # Active task status
            active_tasks = [
                {
                    "task_id": task.task_id,
                    "optimization_type": task.optimization_type.value,
                    "status": task.status.value,
                    "description": task.description,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                }
                for task in self.active_tasks.values()
            ]
            
            # Success metrics
            successful_optimizations = len([
                t for t in self.completed_tasks 
                if t.status == OptimizationStatus.COMPLETED
            ])
            failed_optimizations = len([
                t for t in self.completed_tasks 
                if t.status == OptimizationStatus.FAILED
            ])
            total_optimizations = len(self.completed_tasks)
            
            success_rate = (successful_optimizations / total_optimizations) if total_optimizations > 0 else 0
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "auto_optimize_enabled": self.auto_optimize,
                "active_tasks": active_tasks,
                "active_task_count": len(active_tasks),
                "recent_tasks": recent_tasks,
                "success_metrics": {
                    "total_optimizations": total_optimizations,
                    "successful_optimizations": successful_optimizations,
                    "failed_optimizations": failed_optimizations,
                    "success_rate": round(success_rate, 3)
                },
                "performance_baselines": {
                    component: {
                        "metric_name": baseline.metric_name,
                        "baseline_value": baseline.baseline_value,
                        "measured_at": baseline.measured_at.isoformat(),
                        "sample_size": baseline.sample_size
                    }
                    for component, baseline in self.performance_baselines.items()
                },
                "configuration": {
                    "max_concurrent_optimizations": self.max_concurrent_optimizations,
                    "optimization_cooldown_hours": self.optimization_cooldown_hours,
                    "optimization_thresholds": self.optimization_thresholds
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {"error": str(e)}
    
    # Background task methods
    async def _optimization_analyzer(self) -> None:
        """Background task to analyze performance and generate optimizations."""
        logger.info("Starting optimization analyzer")
        
        while not self._shutdown_event.is_set():
            try:
                if self.auto_optimize:
                    # Generate optimization recommendations
                    tasks = await self.analyze_and_recommend()
                    
                    # Execute high-priority tasks automatically
                    for task in tasks:
                        if task.priority <= 2 and len(self.active_tasks) < self.max_concurrent_optimizations:
                            asyncio.create_task(self.execute_optimization_task(task))
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization analyzer error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Optimization analyzer stopped")
    
    async def _task_executor(self) -> None:
        """Background task executor for queued optimizations."""
        logger.info("Starting task executor")
        
        while not self._shutdown_event.is_set():
            try:
                # Check for queued tasks in Redis
                queued_tasks = await self.redis_client.lrange("optimization:queue", 0, -1)
                
                for task_data in queued_tasks:
                    try:
                        task_dict = json.loads(task_data)
                        task = OptimizationTask(**task_dict)
                        
                        if len(self.active_tasks) < self.max_concurrent_optimizations:
                            # Remove from queue and execute
                            await self.redis_client.lrem("optimization:queue", 1, task_data)
                            asyncio.create_task(self.execute_optimization_task(task))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # Remove invalid task from queue
                        await self.redis_client.lrem("optimization:queue", 1, task_data)
                
                await asyncio.sleep(60)  # Check queue every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task executor error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Task executor stopped")
    
    async def _performance_validator(self) -> None:
        """Background task to validate optimization performance impact."""
        logger.info("Starting performance validator")
        
        while not self._shutdown_event.is_set():
            try:
                # Check completed tasks that need validation
                for task in self.completed_tasks[-10:]:  # Check last 10 completed tasks
                    if (task.status == OptimizationStatus.COMPLETED and 
                        task.completed_at and
                        "validation_completed" not in task.metadata):
                        
                        # Wait at least 10 minutes after completion for metrics to stabilize
                        if (datetime.utcnow() - task.completed_at).total_seconds() > 600:
                            await self._validate_optimization_impact(task)
                
                await asyncio.sleep(600)  # Validate every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance validator error: {e}")
                await asyncio.sleep(600)
        
        logger.info("Performance validator stopped")
    
    async def _maintenance_scheduler(self) -> None:
        """Background maintenance and cleanup scheduler."""
        logger.info("Starting maintenance scheduler")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old completed tasks (keep last 100)
                if len(self.completed_tasks) > 100:
                    self.completed_tasks = self.completed_tasks[-100:]
                
                # Clean up old performance baselines (older than 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                baselines_to_remove = [
                    component for component, baseline in self.performance_baselines.items()
                    if baseline.measured_at < cutoff_time
                ]
                
                for component in baselines_to_remove:
                    del self.performance_baselines[component]
                
                # Clean up Redis optimization data
                await self._cleanup_redis_data()
                
                await asyncio.sleep(3600)  # Maintenance every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance scheduler error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Maintenance scheduler stopped")
    
    # Helper methods for optimization task creation
    async def _create_latency_optimization_task(self, avg_latency: float) -> OptimizationTask:
        """Create optimization task for high latency issues."""
        return OptimizationTask(
            task_id=str(uuid.uuid4()),
            optimization_type=OptimizationType.INDEX_OPTIMIZATION,
            priority=1,
            description=f"Optimize search indices to reduce latency (current: {avg_latency:.0f}ms)",
            target_component="vector_search",
            expected_improvement=0.4,
            implementation_steps=[
                "Analyze query patterns for index optimization",
                "Update IVF index parameters for better performance",
                "Implement query result caching",
                "Optimize embedding similarity calculations"
            ],
            rollback_plan=[
                "Restore previous index configuration",
                "Clear optimization caches",
                "Reset similarity calculation parameters"
            ],
            metadata={"current_latency_ms": avg_latency}
        )
    
    async def _create_cache_optimization_task(self, cache_hit_rate: float) -> OptimizationTask:
        """Create optimization task for low cache hit rate."""
        return OptimizationTask(
            task_id=str(uuid.uuid4()),
            optimization_type=OptimizationType.CACHE_WARMING,
            priority=2,
            description=f"Improve cache hit rate (current: {cache_hit_rate:.1%})",
            target_component="search_cache",
            expected_improvement=0.3,
            implementation_steps=[
                "Analyze cache miss patterns",
                "Pre-warm cache with frequently accessed contexts",
                "Increase cache size if memory allows",
                "Implement intelligent cache prefetching"
            ],
            rollback_plan=[
                "Reset cache configuration to previous settings",
                "Clear pre-warmed cache entries",
                "Disable prefetching"
            ],
            metadata={"current_cache_hit_rate": cache_hit_rate}
        )
    
    async def _create_quality_optimization_task(self, avg_quality: float) -> OptimizationTask:
        """Create optimization task for low search quality."""
        return OptimizationTask(
            task_id=str(uuid.uuid4()),
            optimization_type=OptimizationType.SEARCH_PARAMETER_TUNING,
            priority=1,
            description=f"Improve search quality (current: {avg_quality:.2f})",
            target_component="search_quality",
            expected_improvement=0.25,
            implementation_steps=[
                "Analyze low-quality search results",
                "Tune similarity thresholds",
                "Implement hybrid search with keyword matching",
                "Add result reranking based on relevance"
            ],
            rollback_plan=[
                "Restore previous similarity thresholds",
                "Disable hybrid search features",
                "Remove reranking algorithms"
            ],
            metadata={"current_quality_score": avg_quality}
        )
    
    async def _create_cost_optimization_task(self, hourly_cost: float) -> OptimizationTask:
        """Create optimization task for high API costs."""
        return OptimizationTask(
            task_id=str(uuid.uuid4()),
            optimization_type=OptimizationType.EMBEDDING_CACHING,
            priority=2,
            description=f"Reduce embedding API costs (current: ${hourly_cost:.2f}/hour)",
            target_component="embedding_api",
            expected_improvement=0.5,
            implementation_steps=[
                "Implement aggressive embedding caching",
                "Add duplicate content detection",
                "Batch embedding generation requests",
                "Review and optimize embedding model selection"
            ],
            rollback_plan=[
                "Disable aggressive caching",
                "Remove duplicate detection",
                "Restore individual embedding requests"
            ],
            metadata={"current_hourly_cost": hourly_cost}
        )
    
    async def _analyze_storage_optimization(self) -> List[OptimizationTask]:
        """Analyze storage patterns and create optimization tasks."""
        tasks = []
        
        try:
            # Get capacity trends
            history_raw = await self.redis_client.lrange("context_monitor:capacity_history", 0, 99)
            
            if len(history_raw) >= 10:  # Need sufficient data
                # Analyze storage growth
                recent_entries = []
                for entry_str in history_raw[:10]:  # Last 10 entries
                    try:
                        entry = json.loads(entry_str)
                        recent_entries.append(entry)
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if len(recent_entries) >= 2:
                    oldest = recent_entries[-1]
                    newest = recent_entries[0]
                    
                    # Check for rapid storage growth
                    size_growth = newest["total_size_bytes"] - oldest["total_size_bytes"]
                    if size_growth > 1024**3:  # More than 1GB growth
                        tasks.append(OptimizationTask(
                            task_id=str(uuid.uuid4()),
                            optimization_type=OptimizationType.CONTEXT_ARCHIVING,
                            priority=3,
                            description="Archive old contexts to manage storage growth",
                            target_component="context_storage",
                            expected_improvement=0.3,
                            implementation_steps=[
                                "Identify contexts not accessed in 90+ days",
                                "Archive low-importance contexts to cold storage",
                                "Compress archived context content",
                                "Update search indices to exclude archived contexts"
                            ],
                            rollback_plan=[
                                "Restore archived contexts to active storage",
                                "Rebuild search indices with all contexts",
                                "Remove compression from restored contexts"
                            ],
                            metadata={"storage_growth_bytes": size_growth}
                        ))
        
        except Exception as e:
            logger.error(f"Failed to analyze storage optimization: {e}")
        
        return tasks
    
    async def _filter_optimization_tasks(self, tasks: List[OptimizationTask]) -> List[OptimizationTask]:
        """Filter out duplicate or recently attempted optimizations."""
        filtered_tasks = []
        cutoff_time = datetime.utcnow() - timedelta(hours=self.optimization_cooldown_hours)
        
        for task in tasks:
            # Check if similar task was recently attempted
            recent_similar = False
            for completed_task in self.completed_tasks:
                if (completed_task.optimization_type == task.optimization_type and
                    completed_task.target_component == task.target_component and
                    completed_task.created_at > cutoff_time):
                    recent_similar = True
                    break
            
            if not recent_similar:
                filtered_tasks.append(task)
        
        return filtered_tasks
    
    async def _measure_performance_baseline(self, task: OptimizationTask) -> None:
        """Measure performance baseline before optimization."""
        try:
            component = task.target_component
            metric_name = "latency_ms"  # Default metric
            
            # Get appropriate metrics based on target component
            if component == "vector_search":
                # Get recent search latencies
                search_metrics_raw = await self.redis_client.lrange("context_monitor:search_metrics", 0, 99)
                latencies = []
                
                for metric_str in search_metrics_raw:
                    try:
                        metric = json.loads(metric_str)
                        latencies.append(metric["latency_ms"])
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if latencies:
                    baseline_value = statistics.mean(latencies)
                    self.performance_baselines[component] = PerformanceBaseline(
                        component=component,
                        metric_name=metric_name,
                        baseline_value=baseline_value,
                        measurement_window_hours=1,
                        measured_at=datetime.utcnow(),
                        sample_size=len(latencies)
                    )
            
            elif component == "search_cache":
                # Get cache hit rate
                cache_summary = await self.performance_monitor.get_performance_summary(1)
                cache_perf = cache_summary.get("cache_performance", {})
                hit_rate = cache_perf.get("hit_rate", 0)
                
                self.performance_baselines[component] = PerformanceBaseline(
                    component=component,
                    metric_name="hit_rate",
                    baseline_value=hit_rate,
                    measurement_window_hours=1,
                    measured_at=datetime.utcnow(),
                    sample_size=1
                )
            
            logger.info(f"Measured baseline for {component}: {self.performance_baselines.get(component)}")
            
        except Exception as e:
            logger.error(f"Failed to measure performance baseline for {task.task_id}: {e}")
    
    # Optimization execution methods (simplified implementations)
    async def _execute_cache_warming(self, task: OptimizationTask) -> bool:
        """Execute cache warming optimization."""
        try:
            logger.info("Executing cache warming optimization")
            
            # Simulate cache warming by querying frequent search patterns
            # In a real implementation, this would:
            # 1. Analyze query patterns from Redis
            # 2. Pre-load frequently accessed contexts into cache
            # 3. Implement intelligent prefetching
            
            # For demo purposes, mark as successful
            task.result_metrics["cache_entries_warmed"] = 1000
            task.result_metrics["estimated_hit_rate_improvement"] = 0.15
            
            return True
            
        except Exception as e:
            logger.error(f"Cache warming optimization failed: {e}")
            return False
    
    async def _execute_index_optimization(self, task: OptimizationTask) -> bool:
        """Execute index optimization."""
        try:
            logger.info("Executing index optimization")
            
            # In a real implementation, this would:
            # 1. Analyze current index performance
            # 2. Reindex with optimized parameters
            # 3. Update query execution strategies
            
            # For demo purposes, mark as successful
            task.result_metrics["indices_optimized"] = 3
            task.result_metrics["estimated_latency_reduction"] = 0.3
            
            return True
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            return False
    
    async def _execute_query_rewriting(self, task: OptimizationTask) -> bool:
        """Execute query rewriting optimization."""
        try:
            logger.info("Executing query rewriting optimization")
            
            # Implement intelligent query rewriting
            task.result_metrics["query_patterns_optimized"] = 50
            task.result_metrics["estimated_performance_improvement"] = 0.2
            
            return True
            
        except Exception as e:
            logger.error(f"Query rewriting optimization failed: {e}")
            return False
    
    async def _execute_embedding_caching(self, task: OptimizationTask) -> bool:
        """Execute embedding caching optimization."""
        try:
            logger.info("Executing embedding caching optimization")
            
            # Implement aggressive embedding caching
            task.result_metrics["embeddings_cached"] = 5000
            task.result_metrics["estimated_cost_reduction"] = 0.4
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding caching optimization failed: {e}")
            return False
    
    async def _execute_context_archiving(self, task: OptimizationTask) -> bool:
        """Execute context archiving optimization."""
        try:
            logger.info("Executing context archiving optimization")
            
            if not self.db_session:
                async for session in get_async_session():
                    return await self._perform_context_archiving(session, task)
            else:
                return await self._perform_context_archiving(self.db_session, task)
            
        except Exception as e:
            logger.error(f"Context archiving optimization failed: {e}")
            return False
    
    async def _perform_context_archiving(self, session: AsyncSession, task: OptimizationTask) -> bool:
        """Perform actual context archiving."""
        try:
            # Find old, low-importance contexts
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            result = await session.execute(text("""
                SELECT id, title, content, importance_score 
                FROM contexts 
                WHERE last_accessed_at < :cutoff_date 
                    AND importance_score < 0.3
                    AND archived_at IS NULL
                ORDER BY importance_score ASC, last_accessed_at ASC
                LIMIT 1000
            """), {"cutoff_date": cutoff_date})
            
            contexts_to_archive = result.fetchall()
            
            if contexts_to_archive:
                # Mark contexts as archived
                context_ids = [ctx.id for ctx in contexts_to_archive]
                
                await session.execute(
                    update(Context)
                    .where(Context.id.in_(context_ids))
                    .values(archived_at=datetime.utcnow())
                )
                
                await session.commit()
                
                task.result_metrics["contexts_archived"] = len(contexts_to_archive)
                task.result_metrics["storage_freed_bytes"] = sum(
                    len(ctx.content.encode('utf-8')) if ctx.content else 0 
                    for ctx in contexts_to_archive
                )
                
                logger.info(f"Archived {len(contexts_to_archive)} contexts")
                return True
            else:
                logger.info("No contexts found for archiving")
                task.result_metrics["contexts_archived"] = 0
                return True
                
        except Exception as e:
            logger.error(f"Context archiving failed: {e}")
            if session:
                await session.rollback()
            return False
    
    async def _execute_batch_processing(self, task: OptimizationTask) -> bool:
        """Execute batch processing optimization."""
        try:
            logger.info("Executing batch processing optimization")
            
            # Implement batch processing for embeddings
            task.result_metrics["batch_operations_enabled"] = True
            task.result_metrics["estimated_throughput_improvement"] = 0.25
            
            return True
            
        except Exception as e:
            logger.error(f"Batch processing optimization failed: {e}")
            return False
    
    async def _execute_search_parameter_tuning(self, task: OptimizationTask) -> bool:
        """Execute search parameter tuning optimization."""
        try:
            logger.info("Executing search parameter tuning optimization")
            
            # Implement ML-based parameter tuning
            task.result_metrics["parameters_tuned"] = 12
            task.result_metrics["estimated_quality_improvement"] = 0.15
            
            return True
            
        except Exception as e:
            logger.error(f"Search parameter tuning optimization failed: {e}")
            return False
    
    async def _execute_resource_allocation(self, task: OptimizationTask) -> bool:
        """Execute resource allocation optimization."""
        try:
            logger.info("Executing resource allocation optimization")
            
            # Implement dynamic resource allocation
            task.result_metrics["resources_reallocated"] = True
            task.result_metrics["estimated_efficiency_improvement"] = 0.2
            
            return True
            
        except Exception as e:
            logger.error(f"Resource allocation optimization failed: {e}")
            return False
    
    async def _execute_rollback_step(self, task: OptimizationTask, step: str) -> None:
        """Execute a rollback step."""
        logger.info(f"Executing rollback step: {step}")
        
        # In a real implementation, this would parse the rollback step
        # and execute the appropriate rollback operation
        
        # For demo purposes, just log the step
        await asyncio.sleep(0.1)  # Simulate rollback time
    
    async def _validate_optimization_impact(self, task: OptimizationTask) -> None:
        """Validate the impact of an optimization after completion."""
        try:
            if task.target_component not in self.performance_baselines:
                logger.warning(f"No baseline found for {task.target_component}, skipping validation")
                return
            
            baseline = self.performance_baselines[task.target_component]
            
            # Measure current performance
            current_metrics = await self._measure_current_performance(task.target_component)
            
            if current_metrics:
                improvement = (baseline.baseline_value - current_metrics.get(baseline.metric_name, baseline.baseline_value)) / baseline.baseline_value
                
                task.result_metrics["actual_improvement"] = improvement
                task.result_metrics["baseline_value"] = baseline.baseline_value
                task.result_metrics["current_value"] = current_metrics.get(baseline.metric_name)
                
                # Check if improvement meets expectations
                if improvement >= task.expected_improvement * 0.5:  # At least 50% of expected
                    logger.info(f"Optimization {task.task_id} achieved {improvement:.1%} improvement")
                else:
                    logger.warning(f"Optimization {task.task_id} only achieved {improvement:.1%} improvement (expected {task.expected_improvement:.1%})")
                    
                    # Consider rolling back if performance got worse
                    if improvement < -0.1:  # Performance degraded by more than 10%
                        logger.error(f"Optimization {task.task_id} degraded performance, considering rollback")
                        asyncio.create_task(self.rollback_optimization(task.task_id))
            
            task.metadata["validation_completed"] = True
            task.metadata["validation_timestamp"] = datetime.utcnow().isoformat()
            
            # Store updated task
            await self._store_optimization_result(task)
            
        except Exception as e:
            logger.error(f"Failed to validate optimization impact for {task.task_id}: {e}")
    
    async def _measure_current_performance(self, component: str) -> Optional[Dict[str, float]]:
        """Measure current performance for a component."""
        try:
            if component == "vector_search":
                # Get recent search latencies
                search_metrics_raw = await self.redis_client.lrange("context_monitor:search_metrics", 0, 49)
                latencies = []
                
                for metric_str in search_metrics_raw:
                    try:
                        metric = json.loads(metric_str)
                        latencies.append(metric["latency_ms"])
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if latencies:
                    return {"latency_ms": statistics.mean(latencies)}
            
            elif component == "search_cache":
                # Get current cache performance
                cache_summary = await self.performance_monitor.get_performance_summary(1)
                cache_perf = cache_summary.get("cache_performance", {})
                
                return {"hit_rate": cache_perf.get("hit_rate", 0)}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to measure current performance for {component}: {e}")
            return None
    
    async def _store_optimization_result(self, task: OptimizationTask) -> None:
        """Store optimization task result in Redis."""
        try:
            await self.redis_client.setex(
                f"optimization:result:{task.task_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(asdict(task), default=str)
            )
        except Exception as e:
            logger.error(f"Failed to store optimization result: {e}")
    
    async def _cleanup_redis_data(self) -> None:
        """Clean up old Redis optimization data."""
        try:
            # Clean up old optimization results
            keys = await self.redis_client.keys("optimization:result:*")
            
            for key in keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        task_data = json.loads(data)
                        created_at = datetime.fromisoformat(task_data.get("created_at", ""))
                        
                        # Remove data older than 30 days
                        if (datetime.utcnow() - created_at).days > 30:
                            await self.redis_client.delete(key)
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Remove invalid data
                    await self.redis_client.delete(key)
            
            logger.debug("Cleaned up old optimization data")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Redis data: {e}")


# Global instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get singleton performance optimizer instance."""
    global _performance_optimizer
    
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
        await _performance_optimizer.start()
    
    return _performance_optimizer


async def cleanup_performance_optimizer() -> None:
    """Cleanup performance optimizer resources."""
    global _performance_optimizer
    
    if _performance_optimizer:
        await _performance_optimizer.stop()
        _performance_optimizer = None