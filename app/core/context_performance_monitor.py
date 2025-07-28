"""
Context Performance Monitoring System.

Comprehensive performance analytics and monitoring for the Context Engine
with real-time metrics, intelligent alerting, and optimization recommendations.
"""

import asyncio
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np
from sqlalchemy import select, text, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.context_analytics import ContextAnalyticsManager, ContextRetrieval
from ..core.search_analytics import SearchAnalytics, SearchEvent, SearchEventType
from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..observability.prometheus_exporter import metrics_exporter

logger = logging.getLogger(__name__)


class ContextOperation(Enum):
    """Context engine operations to monitor."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    EMBEDDING_GENERATION = "embedding_generation"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    ANALYTICS_QUERY = "analytics_query"


class PerformanceIssueType(Enum):
    """Types of performance issues."""
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CACHE_HIT_RATE = "low_cache_hit_rate"
    MEMORY_PRESSURE = "memory_pressure"
    EMBEDDING_API_LIMITS = "embedding_api_limits"
    SEARCH_QUALITY_DEGRADATION = "search_quality_degradation"
    CAPACITY_APPROACHING_LIMIT = "capacity_approaching_limit"


@dataclass
class ContextMetrics:
    """Context-specific performance metrics."""
    context_id: str
    operation_counts: Dict[str, int] = field(default_factory=dict)
    avg_latencies: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    last_accessed: Optional[datetime] = None
    embedding_api_calls: int = 0
    search_queries: int = 0
    avg_relevance_score: float = 0.0


@dataclass
class PerformanceAlert:
    """Performance alert configuration and data."""
    alert_id: str
    issue_type: PerformanceIssueType
    severity: str  # critical, warning, info
    component: str
    message: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    recommendation_id: str
    category: str  # indexing, caching, embedding, search_quality
    title: str
    description: str
    expected_improvement: float  # 0-1 scale
    implementation_difficulty: str  # low, medium, high
    priority: int  # 1-5
    estimated_impact: str
    implementation_steps: List[str]
    created_at: datetime


class ContextPerformanceMonitor:
    """
    Comprehensive context engine performance monitoring system.
    
    Features:
    - Real-time performance metrics collection
    - Intelligent alerting with ML-based anomaly detection
    - Cost monitoring for embedding API usage
    - Capacity planning with predictive analytics
    - Automated optimization recommendations
    - Health monitoring for all context components
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize context performance monitor.
        
        Args:
            redis_client: Redis client for real-time metrics
            db_session: Database session for persistent metrics
            alert_thresholds: Custom alert thresholds
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "high_latency_ms": 1000.0,
            "error_rate_threshold": 0.05,  # 5%
            "cache_hit_rate_threshold": 0.8,  # 80%
            "memory_usage_threshold": 0.85,  # 85%
            "api_cost_threshold_usd": 10.0,  # $10 per hour
            "search_quality_threshold": 0.7  # 70%
        }
        
        # Metrics storage
        self.context_metrics: Dict[str, ContextMetrics] = {}
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.recommendations: List[OptimizationRecommendation] = []
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        logger.info("Context Performance Monitor initialized")
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics for context engine."""
        # Context operation metrics
        self.context_operations_total = Counter(
            'leanvibe_context_operations_total',
            'Total context operations performed',
            ['operation_type', 'status', 'context_type'],
            registry=metrics_exporter.registry
        )
        
        self.context_operation_duration_seconds = Histogram(
            'leanvibe_context_operation_duration_seconds',
            'Context operation duration in seconds',
            ['operation_type', 'context_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=metrics_exporter.registry
        )
        
        # Search performance metrics
        self.search_latency_seconds = Histogram(
            'leanvibe_search_latency_seconds',
            'Search operation latency in seconds',
            ['search_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=metrics_exporter.registry
        )
        
        self.search_result_count = Histogram(
            'leanvibe_search_result_count',
            'Number of search results returned',
            ['search_type'],
            buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500],
            registry=metrics_exporter.registry
        )
        
        # Cache performance metrics
        self.cache_hit_rate = Gauge(
            'leanvibe_cache_hit_rate',
            'Cache hit rate (0-1)',
            ['cache_type'],
            registry=metrics_exporter.registry
        )
        
        self.cache_operations_total = Counter(
            'leanvibe_cache_operations_total',
            'Total cache operations',
            ['cache_type', 'operation', 'result'],
            registry=metrics_exporter.registry
        )
        
        # Embedding API metrics
        self.embedding_api_calls_total = Counter(
            'leanvibe_embedding_api_calls_total',
            'Total embedding API calls',
            ['provider', 'model', 'status'],
            registry=metrics_exporter.registry
        )
        
        self.embedding_api_cost_usd = Counter(
            'leanvibe_embedding_api_cost_usd_total',
            'Total embedding API cost in USD',
            ['provider', 'model'],
            registry=metrics_exporter.registry
        )
        
        self.embedding_generation_duration_seconds = Histogram(
            'leanvibe_embedding_generation_duration_seconds',
            'Embedding generation duration in seconds',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=metrics_exporter.registry
        )
        
        # Context storage metrics
        self.context_storage_size_bytes = Gauge(
            'leanvibe_context_storage_size_bytes',
            'Total context storage size in bytes',
            ['context_type'],
            registry=metrics_exporter.registry
        )
        
        self.context_count_total = Gauge(
            'leanvibe_context_count_total',
            'Total number of contexts',
            ['context_type', 'status'],
            registry=metrics_exporter.registry
        )
        
        # Performance alerts
        self.performance_alerts_active = Gauge(
            'leanvibe_performance_alerts_active',
            'Number of active performance alerts',
            ['severity', 'component'],
            registry=metrics_exporter.registry
        )
        
        # Search quality metrics
        self.search_quality_score = Gauge(
            'leanvibe_search_quality_score',
            'Search quality score (0-1)',
            ['metric_type'],
            registry=metrics_exporter.registry
        )
    
    async def start(self) -> None:
        """Start background monitoring tasks."""
        logger.info("Starting context performance monitor")
        
        # Start metrics collector
        self._background_tasks.append(
            asyncio.create_task(self._metrics_collector())
        )
        
        # Start alert manager
        self._background_tasks.append(
            asyncio.create_task(self._alert_manager())
        )
        
        # Start optimization analyzer
        self._background_tasks.append(
            asyncio.create_task(self._optimization_analyzer())
        )
        
        # Start capacity planner
        self._background_tasks.append(
            asyncio.create_task(self._capacity_planner())
        )
    
    async def stop(self) -> None:
        """Stop monitoring tasks."""
        logger.info("Stopping context performance monitor")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def record_operation(
        self,
        operation: ContextOperation,
        context_id: Optional[str] = None,
        context_type: Optional[ContextType] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a context operation for performance monitoring.
        
        Args:
            operation: Type of operation performed
            context_id: ID of the context (if applicable)
            context_type: Type of context
            duration_ms: Operation duration in milliseconds
            success: Whether operation was successful
            metadata: Additional operation metadata
        """
        try:
            # Record Prometheus metrics
            status = "success" if success else "error"
            ctx_type = context_type.value if context_type else "unknown"
            
            self.context_operations_total.labels(
                operation_type=operation.value,
                status=status,
                context_type=ctx_type
            ).inc()
            
            if duration_ms is not None:
                duration_seconds = duration_ms / 1000.0
                self.context_operation_duration_seconds.labels(
                    operation_type=operation.value,
                    context_type=ctx_type
                ).observe(duration_seconds)
                
                # Store for trend analysis
                self.operation_times[f"{operation.value}_{ctx_type}"].append(duration_ms)
            
            # Update context-specific metrics
            if context_id:
                if context_id not in self.context_metrics:
                    self.context_metrics[context_id] = ContextMetrics(context_id=context_id)
                
                metrics = self.context_metrics[context_id]
                metrics.operation_counts[operation.value] = metrics.operation_counts.get(operation.value, 0) + 1
                metrics.last_accessed = datetime.utcnow()
                
                if duration_ms is not None:
                    current_avg = metrics.avg_latencies.get(operation.value, 0.0)
                    count = metrics.operation_counts[operation.value]
                    metrics.avg_latencies[operation.value] = (current_avg * (count - 1) + duration_ms) / count
                
                if not success:
                    metrics.error_counts[operation.value] = metrics.error_counts.get(operation.value, 0) + 1
            
            # Store in Redis for real-time dashboard
            await self._store_realtime_metric(operation, duration_ms, success, metadata)
            
        except Exception as e:
            logger.error(f"Failed to record context operation: {e}")
    
    async def record_search_performance(
        self,
        search_type: str,
        query: str,
        result_count: int,
        latency_ms: float,
        quality_score: Optional[float] = None,
        cache_hit: bool = False
    ) -> None:
        """Record search performance metrics."""
        try:
            # Prometheus metrics
            self.search_latency_seconds.labels(search_type=search_type).observe(latency_ms / 1000.0)
            self.search_result_count.labels(search_type=search_type).observe(result_count)
            
            if quality_score is not None:
                self.search_quality_score.labels(metric_type="relevance").set(quality_score)
            
            # Cache metrics
            cache_result = "hit" if cache_hit else "miss"
            self.cache_operations_total.labels(
                cache_type="search",
                operation="read",
                result=cache_result
            ).inc()
            
            # Store for analysis
            await self.redis_client.lpush(
                "context_monitor:search_metrics",
                json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "search_type": search_type,
                    "query": query,
                    "result_count": result_count,
                    "latency_ms": latency_ms,
                    "quality_score": quality_score,
                    "cache_hit": cache_hit
                })
            )
            
            # Keep only recent metrics
            await self.redis_client.ltrim("context_monitor:search_metrics", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to record search performance: {e}")
    
    async def record_embedding_api_call(
        self,
        provider: str,
        model: str,
        tokens: int,
        duration_ms: float,
        cost_usd: float,
        success: bool = True
    ) -> None:
        """Record embedding API call metrics."""
        try:
            status = "success" if success else "error"
            
            self.embedding_api_calls_total.labels(
                provider=provider,
                model=model,
                status=status
            ).inc()
            
            if success:
                self.embedding_api_cost_usd.labels(
                    provider=provider,
                    model=model
                ).inc(cost_usd)
                
                self.embedding_generation_duration_seconds.labels(
                    provider=provider,
                    model=model
                ).observe(duration_ms / 1000.0)
            
            # Store cost tracking data
            await self.redis_client.lpush(
                "context_monitor:api_costs",
                json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "provider": provider,
                    "model": model,
                    "tokens": tokens,
                    "duration_ms": duration_ms,
                    "cost_usd": cost_usd,
                    "success": success
                })
            )
            
            # Keep only recent cost data
            await self.redis_client.ltrim("context_monitor:api_costs", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to record embedding API call: {e}")
    
    async def get_performance_summary(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Get recent metrics from Redis
            search_metrics_raw = await self.redis_client.lrange("context_monitor:search_metrics", 0, -1)
            api_costs_raw = await self.redis_client.lrange("context_monitor:api_costs", 0, -1)
            
            # Parse metrics
            search_metrics = []
            api_costs = []
            
            for metric_str in search_metrics_raw:
                try:
                    metric = json.loads(metric_str)
                    metric_time = datetime.fromisoformat(metric["timestamp"])
                    if metric_time >= cutoff_time:
                        search_metrics.append(metric)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            for cost_str in api_costs_raw:
                try:
                    cost = json.loads(cost_str)
                    cost_time = datetime.fromisoformat(cost["timestamp"])
                    if cost_time >= cutoff_time:
                        api_costs.append(cost)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            # Calculate summary metrics
            summary = {
                "time_window_hours": time_window_hours,
                "timestamp": datetime.utcnow().isoformat(),
                "search_performance": self._analyze_search_performance(search_metrics),
                "api_costs": self._analyze_api_costs(api_costs),
                "cache_performance": await self._get_cache_performance(),
                "active_alerts": len(self.active_alerts),
                "recommendations_count": len(self.recommendations),
                "context_metrics": await self._get_context_storage_summary()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations."""
        try:
            return [asdict(rec) for rec in self.recommendations]
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    async def get_cost_analysis(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get detailed cost analysis for embedding API usage."""
        try:
            # Get cost data from Redis
            api_costs_raw = await self.redis_client.lrange("context_monitor:api_costs", 0, -1)
            
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            costs = []
            
            for cost_str in api_costs_raw:
                try:
                    cost = json.loads(cost_str)
                    cost_time = datetime.fromisoformat(cost["timestamp"])
                    if cost_time >= cutoff_time and cost["success"]:
                        costs.append(cost)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            if not costs:
                return {"error": "No cost data available"}
            
            # Analyze costs by provider and model
            provider_costs = defaultdict(float)
            model_costs = defaultdict(float)
            hourly_costs = defaultdict(float)
            
            for cost in costs:
                provider_costs[cost["provider"]] += cost["cost_usd"]
                model_costs[f"{cost['provider']}:{cost['model']}"] += cost["cost_usd"]
                
                # Group by hour for trend analysis
                hour_key = datetime.fromisoformat(cost["timestamp"]).strftime("%Y-%m-%d %H:00")
                hourly_costs[hour_key] += cost["cost_usd"]
            
            total_cost = sum(provider_costs.values())
            
            analysis = {
                "time_window_hours": time_window_hours,
                "total_cost_usd": round(total_cost, 4),
                "projected_daily_cost_usd": round(total_cost * (24 / time_window_hours), 4),
                "projected_monthly_cost_usd": round(total_cost * (24 * 30 / time_window_hours), 4),
                "costs_by_provider": dict(provider_costs),
                "costs_by_model": dict(model_costs),
                "hourly_trend": dict(hourly_costs),
                "total_api_calls": len(costs),
                "avg_cost_per_call": round(total_cost / len(costs), 6) if costs else 0
            }
            
            # Add cost optimization recommendations
            if total_cost > self.alert_thresholds["api_cost_threshold_usd"]:
                analysis["cost_alert"] = {
                    "severity": "warning",
                    "message": f"API costs (${total_cost:.2f}) exceed threshold (${self.alert_thresholds['api_cost_threshold_usd']:.2f})",
                    "recommendations": [
                        "Consider caching embeddings more aggressively",
                        "Review embedding model selection for cost optimization",
                        "Implement batch embedding generation",
                        "Monitor for unnecessary duplicate embeddings"
                    ]
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to get cost analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_search_performance(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze search performance metrics."""
        if not metrics:
            return {"error": "No search metrics available"}
        
        latencies = [m["latency_ms"] for m in metrics]
        result_counts = [m["result_count"] for m in metrics]
        quality_scores = [m["quality_score"] for m in metrics if m.get("quality_score") is not None]
        cache_hits = sum(1 for m in metrics if m.get("cache_hit"))
        
        return {
            "total_searches": len(metrics),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "avg_result_count": round(statistics.mean(result_counts), 1),
            "cache_hit_rate": round(cache_hits / len(metrics), 3) if metrics else 0,
            "avg_quality_score": round(statistics.mean(quality_scores), 3) if quality_scores else None
        }
    
    def _analyze_api_costs(self, costs: List[Dict]) -> Dict[str, Any]:
        """Analyze API cost metrics."""
        if not costs:
            return {"error": "No cost data available"}
        
        total_cost = sum(c["cost_usd"] for c in costs)
        total_tokens = sum(c["tokens"] for c in costs)
        
        return {
            "total_calls": len(costs),
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_cost_per_call": round(total_cost / len(costs), 6),
            "avg_tokens_per_call": round(total_tokens / len(costs), 1)
        }
    
    async def _get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        try:
            # This would be implemented based on your caching system
            return {
                "hit_rate": 0.85,  # Placeholder
                "total_operations": 1000,  # Placeholder
                "cache_size_mb": 256  # Placeholder
            }
        except Exception as e:
            logger.error(f"Failed to get cache performance: {e}")
            return {"error": str(e)}
    
    async def _get_context_storage_summary(self) -> Dict[str, Any]:
        """Get context storage summary."""
        try:
            if not self.db_session:
                async for session in get_async_session():
                    return await self._query_context_storage(session)
            else:
                return await self._query_context_storage(self.db_session)
        except Exception as e:
            logger.error(f"Failed to get context storage summary: {e}")
            return {"error": str(e)}
    
    async def _query_context_storage(self, session: AsyncSession) -> Dict[str, Any]:
        """Query context storage metrics from database."""
        try:
            # Count contexts by type
            result = await session.execute(text("""
                SELECT 
                    context_type,
                    COUNT(*) as count,
                    AVG(importance_score) as avg_importance,
                    SUM(LENGTH(content)) as total_content_size
                FROM contexts 
                WHERE deleted_at IS NULL
                GROUP BY context_type
            """))
            
            storage_data = {}
            total_contexts = 0
            total_size = 0
            
            for row in result:
                storage_data[row.context_type] = {
                    "count": row.count,
                    "avg_importance": round(float(row.avg_importance or 0), 2),
                    "content_size_bytes": row.total_content_size or 0
                }
                total_contexts += row.count
                total_size += row.total_content_size or 0
            
            return {
                "total_contexts": total_contexts,
                "total_size_bytes": total_size,
                "by_type": storage_data
            }
            
        except Exception as e:
            logger.error(f"Failed to query context storage: {e}")
            return {"error": str(e)}
    
    async def _store_realtime_metric(
        self,
        operation: ContextOperation,
        duration_ms: Optional[float],
        success: bool,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Store real-time metric in Redis."""
        try:
            metric_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "operation": operation.value,
                "duration_ms": duration_ms,
                "success": success,
                "metadata": metadata or {}
            }
            
            await self.redis_client.lpush(
                "context_monitor:realtime_metrics",
                json.dumps(metric_data)
            )
            
            # Keep only recent metrics
            await self.redis_client.ltrim("context_monitor:realtime_metrics", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to store real-time metric: {e}")
    
    async def _metrics_collector(self) -> None:
        """Background metrics collection task."""
        logger.info("Starting context metrics collector")
        
        while not self._shutdown_event.is_set():
            try:
                # Update cache hit rates
                await self._update_cache_metrics()
                
                # Update context storage metrics
                await self._update_storage_metrics()
                
                # Check for performance issues
                await self._check_performance_issues()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(30)
        
        logger.info("Context metrics collector stopped")
    
    async def _alert_manager(self) -> None:
        """Background alert management task."""
        logger.info("Starting context alert manager")
        
        while not self._shutdown_event.is_set():
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert manager error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Context alert manager stopped")
    
    async def _optimization_analyzer(self) -> None:
        """Background optimization analysis task."""
        logger.info("Starting optimization analyzer")
        
        while not self._shutdown_event.is_set():
            try:
                await self._generate_optimization_recommendations()
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization analyzer error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Optimization analyzer stopped")
    
    async def _capacity_planner(self) -> None:
        """Background capacity planning task."""
        logger.info("Starting capacity planner")
        
        while not self._shutdown_event.is_set():
            try:
                await self._analyze_capacity_trends()
                await asyncio.sleep(900)  # Analyze every 15 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Capacity planner error: {e}")
                await asyncio.sleep(900)
        
        logger.info("Capacity planner stopped")
    
    async def _update_cache_metrics(self) -> None:
        """Update cache performance metrics."""
        try:
            # Calculate cache hit rates from context metrics
            total_hits = sum(m.cache_hits for m in self.context_metrics.values())
            total_misses = sum(m.cache_misses for m in self.context_metrics.values())
            total_requests = total_hits + total_misses
            
            if total_requests > 0:
                hit_rate = total_hits / total_requests
                self.cache_hit_rate.labels(cache_type="context").set(hit_rate)
                
                # Check for low cache hit rate alert
                if hit_rate < self.alert_thresholds["cache_hit_rate_threshold"]:
                    await self._trigger_alert(
                        PerformanceIssueType.LOW_CACHE_HIT_RATE,
                        "warning",
                        "context_cache",
                        f"Low cache hit rate: {hit_rate:.1%}",
                        self.alert_thresholds["cache_hit_rate_threshold"],
                        hit_rate
                    )
            
        except Exception as e:
            logger.error(f"Failed to update cache metrics: {e}")
    
    async def _update_storage_metrics(self) -> None:
        """Update context storage metrics."""
        try:
            storage_summary = await self._get_context_storage_summary()
            
            if "by_type" in storage_summary:
                for context_type, data in storage_summary["by_type"].items():
                    self.context_count_total.labels(
                        context_type=context_type,
                        status="active"
                    ).set(data["count"])
                    
                    self.context_storage_size_bytes.labels(
                        context_type=context_type
                    ).set(data["content_size_bytes"])
            
        except Exception as e:
            logger.error(f"Failed to update storage metrics: {e}")
    
    async def _check_performance_issues(self) -> None:
        """Check for performance issues and trigger alerts."""
        try:
            # Check operation latencies
            for operation_key, times in self.operation_times.items():
                if len(times) >= 10:  # Need enough samples
                    avg_latency = statistics.mean(times)
                    
                    if avg_latency > self.alert_thresholds["high_latency_ms"]:
                        operation_type = operation_key.split("_")[0]
                        await self._trigger_alert(
                            PerformanceIssueType.HIGH_LATENCY,
                            "warning",
                            f"context_{operation_type}",
                            f"High average latency for {operation_type}: {avg_latency:.0f}ms",
                            self.alert_thresholds["high_latency_ms"],
                            avg_latency
                        )
            
            # Check error rates
            for context_id, metrics in self.context_metrics.items():
                for operation, error_count in metrics.error_counts.items():
                    total_ops = metrics.operation_counts.get(operation, 0)
                    if total_ops > 0:
                        error_rate = error_count / total_ops
                        
                        if error_rate > self.alert_thresholds["error_rate_threshold"]:
                            await self._trigger_alert(
                                PerformanceIssueType.HIGH_ERROR_RATE,
                                "critical",
                                f"context_{operation}",
                                f"High error rate for {operation} on context {context_id}: {error_rate:.1%}",
                                self.alert_thresholds["error_rate_threshold"],
                                error_rate
                            )
            
        except Exception as e:
            logger.error(f"Failed to check performance issues: {e}")
    
    async def _trigger_alert(
        self,
        issue_type: PerformanceIssueType,
        severity: str,
        component: str,
        message: str,
        threshold_value: float,
        current_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Trigger a performance alert."""
        try:
            alert_id = f"{issue_type.value}_{component}_{int(time.time())}"
            
            # Check if similar alert is already active
            existing_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.issue_type == issue_type and alert.component == component
            ]
            
            if existing_alerts:
                # Update existing alert
                existing_alert = existing_alerts[0]
                existing_alert.current_value = current_value
                existing_alert.triggered_at = datetime.utcnow()
            else:
                # Create new alert
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    issue_type=issue_type,
                    severity=severity,
                    component=component,
                    message=message,
                    threshold_value=threshold_value,
                    current_value=current_value,
                    triggered_at=datetime.utcnow(),
                    metadata=metadata or {}
                )
                
                self.active_alerts[alert_id] = alert
                
                # Update Prometheus metrics
                self.performance_alerts_active.labels(
                    severity=severity,
                    component=component
                ).inc()
                
                # Store in Redis for dashboard
                await self.redis_client.setex(
                    f"context_monitor:alert:{alert_id}",
                    3600,  # 1 hour TTL
                    json.dumps(asdict(alert), default=str)
                )
                
                logger.warning(f"Performance alert triggered: {message}")
        
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    async def _process_alerts(self) -> None:
        """Process and manage active alerts."""
        try:
            current_time = datetime.utcnow()
            alerts_to_remove = []
            
            for alert_id, alert in self.active_alerts.items():
                # Auto-resolve old alerts
                if (current_time - alert.triggered_at).total_seconds() > 3600:  # 1 hour
                    alerts_to_remove.append(alert_id)
                    
                    # Update Prometheus metrics
                    self.performance_alerts_active.labels(
                        severity=alert.severity,
                        component=alert.component
                    ).dec()
            
            # Remove resolved alerts
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                await self.redis_client.delete(f"context_monitor:alert:{alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to process alerts: {e}")
    
    async def _generate_optimization_recommendations(self) -> None:
        """Generate optimization recommendations based on performance data."""
        try:
            recommendations = []
            
            # Analyze search performance for optimization opportunities
            search_metrics_raw = await self.redis_client.lrange("context_monitor:search_metrics", 0, 999)
            
            if search_metrics_raw:
                search_metrics = []
                for metric_str in search_metrics_raw:
                    try:
                        metric = json.loads(metric_str)
                        search_metrics.append(metric)
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if search_metrics:
                    avg_latency = statistics.mean([m["latency_ms"] for m in search_metrics])
                    avg_quality = statistics.mean([
                        m["quality_score"] for m in search_metrics 
                        if m.get("quality_score") is not None
                    ])
                    
                    # High latency recommendation
                    if avg_latency > 500:
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            category="indexing",
                            title="Optimize Vector Search Indexing",
                            description=f"Average search latency ({avg_latency:.0f}ms) is high. Consider optimizing vector indices.",
                            expected_improvement=0.4,
                            implementation_difficulty="medium",
                            priority=2,
                            estimated_impact="40% latency reduction",
                            implementation_steps=[
                                "Analyze query patterns for index optimization",
                                "Consider IVF index parameters tuning",
                                "Implement query result caching",
                                "Review embedding dimensionality"
                            ],
                            created_at=datetime.utcnow()
                        ))
                    
                    # Low quality recommendation
                    if avg_quality and avg_quality < 0.7:
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            category="search_quality",
                            title="Improve Search Result Quality",
                            description=f"Average search quality ({avg_quality:.2f}) is below threshold. Consider reranking strategies.",
                            expected_improvement=0.3,
                            implementation_difficulty="high",
                            priority=1,
                            estimated_impact="30% quality improvement",
                            implementation_steps=[
                                "Implement hybrid search with keyword matching",
                                "Add result reranking based on context relevance",
                                "Fine-tune similarity thresholds",
                                "Consider multi-stage retrieval pipeline"
                            ],
                            created_at=datetime.utcnow()
                        ))
            
            # Cost optimization recommendations
            cost_analysis = await self.get_cost_analysis(time_window_hours=24)
            if "total_cost_usd" in cost_analysis:
                daily_cost = cost_analysis.get("projected_daily_cost_usd", 0)
                
                if daily_cost > 5.0:  # $5 per day threshold
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        category="cost_optimization",
                        title="Optimize Embedding API Costs",
                        description=f"Daily API costs (${daily_cost:.2f}) are high. Consider caching and batching optimizations.",
                        expected_improvement=0.5,
                        implementation_difficulty="low",
                        priority=3,
                        estimated_impact="50% cost reduction",
                        implementation_steps=[
                            "Implement aggressive embedding caching",
                            "Batch embedding generation requests",
                            "Review embedding model selection",
                            "Add duplicate content detection"
                        ],
                        created_at=datetime.utcnow()
                    ))
            
            # Update recommendations list
            self.recommendations = recommendations[:10]  # Keep top 10
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
    
    async def _analyze_capacity_trends(self) -> None:
        """Analyze capacity trends for predictive planning."""
        try:
            # Analyze storage growth trends
            storage_summary = await self._get_context_storage_summary()
            
            if "total_contexts" in storage_summary:
                total_contexts = storage_summary["total_contexts"]
                total_size = storage_summary["total_size_bytes"]
                
                # Store historical data
                await self.redis_client.lpush(
                    "context_monitor:capacity_history",
                    json.dumps({
                        "timestamp": datetime.utcnow().isoformat(),
                        "total_contexts": total_contexts,
                        "total_size_bytes": total_size
                    })
                )
                
                # Keep only last 7 days of data (assuming 15-min intervals)
                await self.redis_client.ltrim("context_monitor:capacity_history", 0, 672)
                
                # Analyze growth trends (simplified - would use proper time series analysis in production)
                history_raw = await self.redis_client.lrange("context_monitor:capacity_history", 0, -1)
                
                if len(history_raw) >= 20:  # Need enough data points
                    history = []
                    for entry_str in history_raw:
                        try:
                            entry = json.loads(entry_str)
                            history.append(entry)
                        except (json.JSONDecodeError, KeyError):
                            continue
                    
                    # Simple linear growth calculation
                    if len(history) >= 2:
                        oldest = history[-1]
                        newest = history[0]
                        
                        time_diff_hours = (
                            datetime.fromisoformat(newest["timestamp"]) - 
                            datetime.fromisoformat(oldest["timestamp"])
                        ).total_seconds() / 3600
                        
                        if time_diff_hours > 0:
                            contexts_growth_per_hour = (
                                newest["total_contexts"] - oldest["total_contexts"]
                            ) / time_diff_hours
                            
                            size_growth_per_hour = (
                                newest["total_size_bytes"] - oldest["total_size_bytes"]
                            ) / time_diff_hours
                            
                            # Check if approaching capacity limits (example thresholds)
                            max_contexts = 1000000  # 1M contexts
                            max_size_gb = 100  # 100GB
                            
                            if contexts_growth_per_hour > 0:
                                hours_to_context_limit = (max_contexts - total_contexts) / contexts_growth_per_hour
                                
                                if hours_to_context_limit < 720:  # 30 days
                                    await self._trigger_alert(
                                        PerformanceIssueType.CAPACITY_APPROACHING_LIMIT,
                                        "warning",
                                        "context_storage",
                                        f"Context capacity limit will be reached in {hours_to_context_limit/24:.1f} days",
                                        max_contexts,
                                        total_contexts
                                    )
                            
                            if size_growth_per_hour > 0:
                                current_size_gb = total_size / (1024**3)
                                hours_to_size_limit = (max_size_gb - current_size_gb) / (size_growth_per_hour / (1024**3))
                                
                                if hours_to_size_limit < 720:  # 30 days
                                    await self._trigger_alert(
                                        PerformanceIssueType.CAPACITY_APPROACHING_LIMIT,
                                        "warning",
                                        "storage_size",
                                        f"Storage size limit will be reached in {hours_to_size_limit/24:.1f} days",
                                        max_size_gb,
                                        current_size_gb
                                    )
        
        except Exception as e:
            logger.error(f"Failed to analyze capacity trends: {e}")


# Global instance
_context_performance_monitor: Optional[ContextPerformanceMonitor] = None


async def get_context_performance_monitor() -> ContextPerformanceMonitor:
    """Get singleton context performance monitor instance."""
    global _context_performance_monitor
    
    if _context_performance_monitor is None:
        _context_performance_monitor = ContextPerformanceMonitor()
        await _context_performance_monitor.start()
    
    return _context_performance_monitor


async def cleanup_context_performance_monitor() -> None:
    """Cleanup context performance monitor resources."""
    global _context_performance_monitor
    
    if _context_performance_monitor:
        await _context_performance_monitor.stop()
        _context_performance_monitor = None