"""
Search Analytics System with Performance Monitoring and Query Optimization.

This module provides comprehensive analytics and monitoring for vector search
operations with:
- Real-time performance tracking and alerting
- Query pattern analysis and optimization recommendations
- Search quality metrics and A/B testing support
- Performance regression detection
- Intelligent query rewriting and optimization
- Search result ranking analysis
- User behavior tracking and personalization insights
"""

import asyncio
import time
import json
import logging
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import numpy as np

from sqlalchemy import select, and_, or_, desc, asc, func, text, insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.vector_search import ContextMatch, SearchFilters
from ..core.database import get_async_session
from ..core.redis import get_redis_client

logger = logging.getLogger(__name__)


class SearchEventType(Enum):
    """Types of search events to track."""
    QUERY_SUBMITTED = "query_submitted"
    RESULTS_RETURNED = "results_returned"
    RESULT_CLICKED = "result_clicked"
    RESULT_IGNORED = "result_ignored"
    QUERY_REFINED = "query_refined"
    NO_RESULTS_FOUND = "no_results_found"
    SEARCH_ABANDONED = "search_abandoned"
    FEEDBACK_PROVIDED = "feedback_provided"


class PerformanceAlert(Enum):
    """Performance alert types."""
    HIGH_LATENCY = "high_latency"
    LOW_RECALL = "low_recall"
    HIGH_ERROR_RATE = "high_error_rate"
    INDEX_DEGRADATION = "index_degradation"
    CACHE_MISSES = "cache_misses"
    QUALITY_DEGRADATION = "quality_degradation"


@dataclass
class SearchEvent:
    """Represents a search event for analytics."""
    event_id: str
    event_type: SearchEventType
    timestamp: datetime
    agent_id: Optional[str]
    session_id: str
    query: str
    results_count: int
    processing_time_ms: float
    similarity_algorithm: str = "cosine"
    search_method: str = "hybrid"
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    result_position: Optional[int] = None
    result_id: Optional[str] = None
    feedback_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for a query pattern."""
    query_pattern: str
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    avg_results_count: float
    click_through_rate: float
    bounce_rate: float
    refinement_rate: float
    satisfaction_score: float
    last_updated: datetime


@dataclass
class SearchQualityMetrics:
    """Quality metrics for search results."""
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]     # k -> recall
    ndcg_at_k: Dict[int, float]       # k -> NDCG
    mrr: float                        # Mean Reciprocal Rank
    map_score: float                  # Mean Average Precision
    diversity_score: float            # Result diversity
    freshness_score: float            # Result freshness
    relevance_score: float            # Average relevance


@dataclass
class QueryOptimizationSuggestion:
    """Optimization suggestion for a query pattern."""
    query_pattern: str
    suggestion_type: str
    description: str
    expected_improvement: float
    implementation_difficulty: str
    priority: int
    created_at: datetime


class QueryPatternAnalyzer:
    """Analyzes query patterns and suggests optimizations."""
    
    def __init__(self):
        self.pattern_metrics: Dict[str, QueryPerformanceMetrics] = {}
        self.slow_queries: deque = deque(maxlen=1000)
        self.failed_queries: deque = deque(maxlen=1000)
        self.optimization_suggestions: List[QueryOptimizationSuggestion] = []
    
    def analyze_query_pattern(self, query: str) -> str:
        """Extract pattern from query for analysis."""
        # Normalize query for pattern extraction
        normalized = query.lower().strip()
        
        # Extract patterns based on query characteristics
        words = normalized.split()
        word_count = len(words)
        
        # Basic pattern classification
        if word_count <= 2:
            pattern = "short_query"
        elif word_count <= 5:
            pattern = "medium_query"
        else:
            pattern = "long_query"
        
        # Add semantic patterns
        if any(word in normalized for word in ["error", "problem", "issue", "bug"]):
            pattern += "_error_related"
        elif any(word in normalized for word in ["how", "what", "when", "where", "why"]):
            pattern += "_question"
        elif any(word in normalized for word in ["documentation", "guide", "tutorial"]):
            pattern += "_documentation"
        
        return pattern
    
    def update_pattern_metrics(
        self, 
        query_pattern: str, 
        latency_ms: float, 
        results_count: int,
        had_error: bool = False
    ) -> None:
        """Update metrics for a query pattern."""
        if query_pattern not in self.pattern_metrics:
            self.pattern_metrics[query_pattern] = QueryPerformanceMetrics(
                query_pattern=query_pattern,
                total_queries=0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                error_rate=0.0,
                avg_results_count=0.0,
                click_through_rate=0.0,
                bounce_rate=0.0,
                refinement_rate=0.0,
                satisfaction_score=0.0,
                last_updated=datetime.utcnow()
            )
        
        metrics = self.pattern_metrics[query_pattern]
        
        # Update counters
        old_count = metrics.total_queries
        new_count = old_count + 1
        
        # Update averages
        metrics.avg_latency_ms = (metrics.avg_latency_ms * old_count + latency_ms) / new_count
        metrics.avg_results_count = (metrics.avg_results_count * old_count + results_count) / new_count
        
        # Update error rate
        old_errors = metrics.error_rate * old_count
        new_errors = old_errors + (1 if had_error else 0)
        metrics.error_rate = new_errors / new_count
        
        metrics.total_queries = new_count
        metrics.last_updated = datetime.utcnow()
        
        # Track slow queries
        if latency_ms > 1000:  # Over 1 second
            self.slow_queries.append({
                "pattern": query_pattern,
                "latency_ms": latency_ms,
                "timestamp": datetime.utcnow()
            })
        
        # Track failed queries
        if had_error:
            self.failed_queries.append({
                "pattern": query_pattern,
                "error": True,
                "timestamp": datetime.utcnow()
            })
    
    def generate_optimization_suggestions(self) -> List[QueryOptimizationSuggestion]:
        """Generate optimization suggestions based on patterns."""
        suggestions = []
        current_time = datetime.utcnow()
        
        for pattern, metrics in self.pattern_metrics.items():
            # High latency patterns
            if metrics.avg_latency_ms > 500:
                suggestions.append(QueryOptimizationSuggestion(
                    query_pattern=pattern,
                    suggestion_type="index_optimization",
                    description=f"Pattern '{pattern}' has high average latency ({metrics.avg_latency_ms:.0f}ms). Consider index optimization.",
                    expected_improvement=0.3,
                    implementation_difficulty="medium",
                    priority=1,
                    created_at=current_time
                ))
            
            # High error rate patterns
            if metrics.error_rate > 0.1:
                suggestions.append(QueryOptimizationSuggestion(
                    query_pattern=pattern,
                    suggestion_type="error_handling",
                    description=f"Pattern '{pattern}' has high error rate ({metrics.error_rate:.1%}). Improve error handling.",
                    expected_improvement=0.5,
                    implementation_difficulty="low",
                    priority=2,
                    created_at=current_time
                ))
            
            # Low result count patterns
            if metrics.avg_results_count < 2 and metrics.total_queries > 10:
                suggestions.append(QueryOptimizationSuggestion(
                    query_pattern=pattern,
                    suggestion_type="recall_improvement",
                    description=f"Pattern '{pattern}' returns few results ({metrics.avg_results_count:.1f}). Consider relaxing filters.",
                    expected_improvement=0.4,
                    implementation_difficulty="medium",
                    priority=3,
                    created_at=current_time
                ))
        
        # Sort by priority and expected improvement
        suggestions.sort(key=lambda s: (s.priority, -s.expected_improvement))
        
        self.optimization_suggestions = suggestions[:20]  # Keep top 20
        return self.optimization_suggestions


class SearchAnalytics:
    """
    Comprehensive search analytics system with performance monitoring.
    
    Features:
    - Real-time performance tracking and alerting
    - Query pattern analysis and optimization
    - Search quality assessment with multiple metrics
    - A/B testing support for search algorithms
    - Performance regression detection
    - User behavior analysis and personalization
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
        performance_threshold_ms: float = 100.0,
        quality_threshold: float = 0.8
    ):
        """
        Initialize search analytics system.
        
        Args:
            redis_client: Redis client for real-time metrics
            db_session: Database session for persistent analytics
            performance_threshold_ms: Performance alert threshold
            quality_threshold: Quality alert threshold
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        self.performance_threshold_ms = performance_threshold_ms
        self.quality_threshold = quality_threshold
        
        # Components
        self.query_analyzer = QueryPatternAnalyzer()
        
        # Real-time metrics
        self.event_buffer: deque[SearchEvent] = deque(maxlen=10000)
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.quality_metrics: Dict[str, SearchQualityMetrics] = {}
        
        # Alert system
        self.active_alerts: Set[str] = set()
        self.alert_thresholds = {
            PerformanceAlert.HIGH_LATENCY: 1000.0,  # ms
            PerformanceAlert.HIGH_ERROR_RATE: 0.05,  # 5%
            PerformanceAlert.LOW_RECALL: 0.5,        # 50%
            PerformanceAlert.CACHE_MISSES: 0.8       # 80%
        }
        
        # A/B testing
        self.ab_test_variants: Dict[str, Dict[str, Any]] = {}
        self.ab_test_results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start background analytics processing."""
        logger.info("Starting search analytics system")
        
        # Start event processor
        self._background_tasks.append(
            asyncio.create_task(self._event_processor())
        )
        
        # Start performance monitor
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitor())
        )
        
        # Start quality assessor
        self._background_tasks.append(
            asyncio.create_task(self._quality_assessor())
        )
        
        # Start alert manager
        self._background_tasks.append(
            asyncio.create_task(self._alert_manager())
        )
    
    async def stop(self) -> None:
        """Stop analytics processing."""
        logger.info("Stopping search analytics system")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def record_search_event(
        self,
        event_type: SearchEventType,
        query: str,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        results_count: int = 0,
        processing_time_ms: float = 0.0,
        **kwargs
    ) -> None:
        """
        Record a search event for analytics.
        
        Args:
            event_type: Type of search event
            query: Search query
            agent_id: Agent performing search
            session_id: Search session ID
            results_count: Number of results returned
            processing_time_ms: Processing time in milliseconds
            **kwargs: Additional event metadata
        """
        event = SearchEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            session_id=session_id or str(uuid.uuid4()),
            query=query,
            results_count=results_count,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
        
        # Add to buffer for real-time processing
        self.event_buffer.append(event)
        
        # Update real-time metrics
        await self._update_realtime_metrics(event)
        
        # Update query pattern analysis
        pattern = self.query_analyzer.analyze_query_pattern(query)
        self.query_analyzer.update_pattern_metrics(
            pattern, 
            processing_time_ms, 
            results_count,
            event_type == SearchEventType.NO_RESULTS_FOUND
        )
    
    async def record_search_results(
        self,
        query: str,
        results: List[ContextMatch],
        processing_time_ms: float,
        search_metadata: Dict[str, Any],
        agent_id: Optional[str] = None
    ) -> None:
        """
        Record search results for quality analysis.
        
        Args:
            query: Search query
            results: Search results
            processing_time_ms: Processing time
            search_metadata: Search metadata
            agent_id: Agent performing search
        """
        await self.record_search_event(
            event_type=SearchEventType.RESULTS_RETURNED,
            query=query,
            agent_id=agent_id,
            results_count=len(results),
            processing_time_ms=processing_time_ms,
            similarity_algorithm=search_metadata.get("similarity_algorithm", "cosine"),
            search_method=search_metadata.get("search_method", "hybrid"),
            cache_hit=search_metadata.get("cache_hit", False)
        )
        
        # Analyze result quality
        await self._analyze_result_quality(query, results, search_metadata)
    
    async def record_user_feedback(
        self,
        query: str,
        result_id: str,
        feedback_score: float,
        result_position: int,
        session_id: str,
        agent_id: Optional[str] = None
    ) -> None:
        """
        Record user feedback for result quality assessment.
        
        Args:
            query: Original search query
            result_id: ID of the result
            feedback_score: Feedback score (0-1)
            result_position: Position of result in ranking
            session_id: Search session ID
            agent_id: Agent providing feedback
        """
        await self.record_search_event(
            event_type=SearchEventType.FEEDBACK_PROVIDED,
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            result_id=result_id,
            feedback_score=feedback_score,
            result_position=result_position
        )
    
    async def get_performance_summary(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get performance summary for specified time window.
        
        Args:
            time_window_hours: Time window in hours
            
        Returns:
            Performance summary with key metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Filter recent events
        recent_events = [
            event for event in self.event_buffer
            if event.timestamp >= cutoff_time
        ]
        
        if not recent_events:
            return {"error": "No recent events found"}
        
        # Calculate metrics
        total_queries = len([e for e in recent_events if e.event_type == SearchEventType.QUERY_SUBMITTED])
        total_results = sum(e.results_count for e in recent_events if e.event_type == SearchEventType.RESULTS_RETURNED)
        
        latencies = [e.processing_time_ms for e in recent_events if e.processing_time_ms > 0]
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_queries": total_queries,
            "total_results_returned": total_results,
            "avg_results_per_query": total_results / max(1, total_queries),
            "performance": {
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "median_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0
            },
            "query_patterns": self._get_top_query_patterns(),
            "active_alerts": list(self.active_alerts),
            "optimization_suggestions": [
                asdict(suggestion) for suggestion in 
                self.query_analyzer.generate_optimization_suggestions()[:5]
            ]
        }
        
        return summary
    
    async def get_quality_metrics(
        self,
        query_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get search quality metrics.
        
        Args:
            query_pattern: Specific query pattern to analyze
            
        Returns:
            Quality metrics summary
        """
        if query_pattern and query_pattern in self.quality_metrics:
            metrics = self.quality_metrics[query_pattern]
            return {
                "query_pattern": query_pattern,
                "precision_at_k": metrics.precision_at_k,
                "recall_at_k": metrics.recall_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
                "mrr": metrics.mrr,
                "map_score": metrics.map_score,
                "diversity_score": metrics.diversity_score,
                "freshness_score": metrics.freshness_score,
                "relevance_score": metrics.relevance_score
            }
        
        # Return overall quality metrics
        all_metrics = list(self.quality_metrics.values())
        if not all_metrics:
            return {"error": "No quality metrics available"}
        
        # Aggregate metrics
        avg_precision = {}
        avg_recall = {}
        avg_ndcg = {}
        
        for k in [1, 3, 5, 10]:
            precision_values = [m.precision_at_k.get(k, 0) for m in all_metrics]
            recall_values = [m.recall_at_k.get(k, 0) for m in all_metrics]
            ndcg_values = [m.ndcg_at_k.get(k, 0) for m in all_metrics]
            
            avg_precision[k] = statistics.mean(precision_values) if precision_values else 0
            avg_recall[k] = statistics.mean(recall_values) if recall_values else 0
            avg_ndcg[k] = statistics.mean(ndcg_values) if ndcg_values else 0
        
        return {
            "overall_metrics": True,
            "patterns_analyzed": len(all_metrics),
            "avg_precision_at_k": avg_precision,
            "avg_recall_at_k": avg_recall,
            "avg_ndcg_at_k": avg_ndcg,
            "avg_mrr": statistics.mean([m.mrr for m in all_metrics]),
            "avg_map_score": statistics.mean([m.map_score for m in all_metrics]),
            "avg_diversity_score": statistics.mean([m.diversity_score for m in all_metrics]),
            "avg_relevance_score": statistics.mean([m.relevance_score for m in all_metrics])
        }
    
    def start_ab_test(
        self,
        test_name: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_split: Dict[str, float]
    ) -> None:
        """
        Start an A/B test for search algorithms.
        
        Args:
            test_name: Name of the test
            variants: Variant configurations
            traffic_split: Traffic split percentages
        """
        self.ab_test_variants[test_name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "start_time": datetime.utcnow(),
            "active": True
        }
        
        logger.info(f"Started A/B test '{test_name}' with variants: {list(variants.keys())}")
    
    def record_ab_test_result(
        self,
        test_name: str,
        variant: str,
        metric_name: str,
        value: float
    ) -> None:
        """Record A/B test result."""
        if test_name in self.ab_test_variants:
            self.ab_test_results[test_name][f"{variant}_{metric_name}"].append(value)
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results with statistical significance."""
        if test_name not in self.ab_test_variants:
            return {"error": "Test not found"}
        
        test_config = self.ab_test_variants[test_name]
        results = self.ab_test_results[test_name]
        
        # Calculate statistics for each variant and metric
        variant_stats = {}
        for key, values in results.items():
            if values:
                variant_stats[key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return {
            "test_name": test_name,
            "start_time": test_config["start_time"].isoformat(),
            "variants": list(test_config["variants"].keys()),
            "statistics": variant_stats,
            "active": test_config["active"]
        }
    
    async def _update_realtime_metrics(self, event: SearchEvent) -> None:
        """Update real-time metrics in Redis."""
        try:
            # Update counters
            await self.redis_client.incr(f"search_analytics:queries_total")
            await self.redis_client.incr(f"search_analytics:queries:{event.event_type.value}")
            
            # Update latency metrics
            if event.processing_time_ms > 0:
                await self.redis_client.lpush(
                    "search_analytics:latencies",
                    json.dumps({
                        "timestamp": event.timestamp.isoformat(),
                        "latency_ms": event.processing_time_ms,
                        "query_pattern": self.query_analyzer.analyze_query_pattern(event.query)
                    })
                )
                # Keep only recent latencies
                await self.redis_client.ltrim("search_analytics:latencies", 0, 999)
            
        except Exception as e:
            logger.warning(f"Failed to update real-time metrics: {e}")
    
    def _get_top_query_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top query patterns by frequency."""
        patterns = list(self.query_analyzer.pattern_metrics.values())
        patterns.sort(key=lambda p: p.total_queries, reverse=True)
        
        return [
            {
                "pattern": p.query_pattern,
                "total_queries": p.total_queries,
                "avg_latency_ms": p.avg_latency_ms,
                "error_rate": p.error_rate,
                "avg_results_count": p.avg_results_count
            }
            for p in patterns[:limit]
        ]
    
    async def _analyze_result_quality(
        self,
        query: str,
        results: List[ContextMatch],
        metadata: Dict[str, Any]
    ) -> None:
        """Analyze quality of search results."""
        if not results:
            return
        
        pattern = self.query_analyzer.analyze_query_pattern(query)
        
        # Calculate quality metrics
        quality_metrics = SearchQualityMetrics(
            precision_at_k={},
            recall_at_k={},
            ndcg_at_k={},
            mrr=0.0,
            map_score=0.0,
            diversity_score=0.0,
            freshness_score=0.0,
            relevance_score=0.0
        )
        
        # Precision and Recall at K
        for k in [1, 3, 5, 10]:
            if len(results) >= k:
                top_k_results = results[:k]
                relevant_count = sum(1 for r in top_k_results if r.relevance_score > 0.7)
                quality_metrics.precision_at_k[k] = relevant_count / k
                quality_metrics.recall_at_k[k] = relevant_count / max(1, len(results))
        
        # Average relevance score
        quality_metrics.relevance_score = statistics.mean([r.relevance_score for r in results])
        
        # Diversity score (based on different context types)
        unique_types = len(set(r.context.context_type for r in results if r.context.context_type))
        quality_metrics.diversity_score = min(1.0, unique_types / max(1, len(results)))
        
        # Freshness score (based on recent contexts)
        now = datetime.utcnow()
        fresh_results = sum(
            1 for r in results 
            if r.context.created_at and (now - r.context.created_at).days < 30
        )
        quality_metrics.freshness_score = fresh_results / len(results)
        
        # Store quality metrics
        self.quality_metrics[pattern] = quality_metrics
    
    async def _event_processor(self) -> None:
        """Background event processor."""
        logger.info("Starting search analytics event processor")
        
        while not self._shutdown_event.is_set():
            try:
                # Process events in buffer
                events_to_process = list(self.event_buffer)
                self.event_buffer.clear()
                
                if events_to_process:
                    # Batch process events
                    await self._process_event_batch(events_to_process)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processor error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Search analytics event processor stopped")
    
    async def _process_event_batch(self, events: List[SearchEvent]) -> None:
        """Process a batch of events."""
        if not self.db_session:
            return
        
        try:
            # Store events in database for long-term analysis
            for event in events:
                # In a real implementation, you'd have a search_events table
                # This is simplified for the example
                pass
                
        except Exception as e:
            logger.error(f"Failed to process event batch: {e}")
    
    async def _performance_monitor(self) -> None:
        """Background performance monitor."""
        logger.info("Starting performance monitor")
        
        while not self._shutdown_event.is_set():
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Performance monitor stopped")
    
    async def _check_performance_alerts(self) -> None:
        """Check for performance alerts."""
        try:
            # Check recent latencies
            recent_events = [
                e for e in self.event_buffer
                if e.timestamp >= datetime.utcnow() - timedelta(minutes=5)
                and e.processing_time_ms > 0
            ]
            
            if recent_events:
                avg_latency = statistics.mean([e.processing_time_ms for e in recent_events])
                
                if avg_latency > self.alert_thresholds[PerformanceAlert.HIGH_LATENCY]:
                    alert_key = f"high_latency_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
                    if alert_key not in self.active_alerts:
                        self.active_alerts.add(alert_key)
                        logger.warning(f"Performance alert: High latency detected ({avg_latency:.0f}ms)")
                
                # Check error rate
                error_events = [e for e in recent_events if e.event_type == SearchEventType.NO_RESULTS_FOUND]
                error_rate = len(error_events) / len(recent_events)
                
                if error_rate > self.alert_thresholds[PerformanceAlert.HIGH_ERROR_RATE]:
                    alert_key = f"high_error_rate_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
                    if alert_key not in self.active_alerts:
                        self.active_alerts.add(alert_key)
                        logger.warning(f"Performance alert: High error rate detected ({error_rate:.1%})")
            
            # Clean old alerts
            current_hour = datetime.utcnow().strftime('%Y%m%d_%H%M')
            self.active_alerts = {
                alert for alert in self.active_alerts
                if current_hour in alert or 
                (datetime.utcnow() - datetime.strptime(alert.split('_')[-1], '%Y%m%d_%H%M')).seconds < 3600
            }
            
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
    
    async def _quality_assessor(self) -> None:
        """Background quality assessor."""
        logger.info("Starting quality assessor")
        
        while not self._shutdown_event.is_set():
            try:
                # Assess search quality trends
                await self._assess_quality_trends()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quality assessor error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Quality assessor stopped")
    
    async def _assess_quality_trends(self) -> None:
        """Assess quality trends and generate alerts."""
        try:
            # Check if quality metrics are degrading
            for pattern, metrics in self.quality_metrics.items():
                if metrics.relevance_score < self.quality_threshold:
                    alert_key = f"quality_degradation_{pattern}"
                    if alert_key not in self.active_alerts:
                        self.active_alerts.add(alert_key)
                        logger.warning(f"Quality alert: Low relevance for pattern '{pattern}' ({metrics.relevance_score:.2f})")
                        
        except Exception as e:
            logger.error(f"Quality trend assessment failed: {e}")
    
    async def _alert_manager(self) -> None:
        """Background alert manager."""
        logger.info("Starting alert manager")
        
        while not self._shutdown_event.is_set():
            try:
                # Manage alerts and send notifications
                await self._process_alerts()
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert manager error: {e}")
                await asyncio.sleep(120)
        
        logger.info("Alert manager stopped")
    
    async def _process_alerts(self) -> None:
        """Process and handle active alerts."""
        try:
            if self.active_alerts:
                # In a real implementation, you'd send notifications
                # via email, Slack, etc.
                logger.info(f"Active alerts: {len(self.active_alerts)}")
                
                # Store alerts in Redis for dashboard
                alert_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "alerts": list(self.active_alerts),
                    "count": len(self.active_alerts)
                }
                
                await self.redis_client.set(
                    "search_analytics:active_alerts",
                    json.dumps(alert_data),
                    expire=3600
                )
                
        except Exception as e:
            logger.error(f"Alert processing failed: {e}")


# Global instance for application use
_search_analytics: Optional[SearchAnalytics] = None


async def get_search_analytics() -> SearchAnalytics:
    """
    Get singleton search analytics instance.
    
    Returns:
        SearchAnalytics instance
    """
    global _search_analytics
    
    if _search_analytics is None:
        _search_analytics = SearchAnalytics()
        await _search_analytics.start()
    
    return _search_analytics


async def cleanup_search_analytics() -> None:
    """Cleanup search analytics resources."""
    global _search_analytics
    
    if _search_analytics:
        await _search_analytics.stop()
        _search_analytics = None