"""
Advanced Performance Storage Engine for LeanVibe Agent Hive 2.0

High-performance storage and indexing system for performance metrics with
time-series optimization, intelligent data retention, and fast query capabilities.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
import redis.asyncio as redis
import numpy as np
from sqlalchemy import (
    select, func, and_, or_, desc, asc, 
    Index, Column, String, Float, DateTime, JSON, Integer, Boolean
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from .config import settings
from .database import get_session, Base
from .redis import get_redis_client
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class DataRetentionPolicy(Enum):
    """Data retention policies for different metric categories."""
    REAL_TIME = "real_time"      # Keep for 2 hours in high resolution
    SHORT_TERM = "short_term"    # Keep for 24 hours in medium resolution
    MEDIUM_TERM = "medium_term"  # Keep for 7 days in reduced resolution
    LONG_TERM = "long_term"      # Keep for 90 days in aggregated form
    ARCHIVE = "archive"          # Keep for 1 year in summary form


class AggregationType(Enum):
    """Types of metric aggregations."""
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"
    STDDEV = "stddev"


@dataclass
class RetentionRule:
    """Retention rule configuration."""
    policy: DataRetentionPolicy
    duration_hours: int
    resolution_seconds: int
    aggregation_types: List[AggregationType]
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricIndex:
    """Metric index configuration for fast queries."""
    index_name: str
    columns: List[str]
    index_type: str  # 'btree', 'hash', 'gin', 'gist'
    partial_condition: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class QueryOptimization:
    """Query optimization statistics and suggestions."""
    query_pattern: str
    execution_time_ms: float
    rows_examined: int
    index_usage: Dict[str, Any]
    optimization_suggestions: List[str]
    performance_score: float


class PerformanceMetricAggregated(Base):
    """Aggregated performance metrics table for long-term storage."""
    
    __tablename__ = "performance_metrics_aggregated"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(255), nullable=False, index=True)
    aggregation_type = Column(String(50), nullable=False)
    aggregated_value = Column(Float, nullable=False)
    aggregation_period_start = Column(DateTime(timezone=True), nullable=False, index=True)
    aggregation_period_end = Column(DateTime(timezone=True), nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    tags = Column(JSON, nullable=True, default=dict)
    retention_policy = Column(String(50), nullable=False, index=True)
    
    # Add composite indexes for common query patterns
    __table_args__ = (
        Index('idx_metric_time_policy', 'metric_name', 'aggregation_period_start', 'retention_policy'),
        Index('idx_metric_aggregation', 'metric_name', 'aggregation_type'),
        Index('idx_time_policy', 'aggregation_period_start', 'retention_policy'),
    )


class TimeSeriesBuffer:
    """High-performance in-memory buffer for time series data."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.last_flush = datetime.utcnow()
        self.lock = threading.RLock()
    
    def add_metric(self, metric_name: str, value: float, timestamp: datetime, tags: Dict[str, Any] = None) -> None:
        """Add metric to buffer."""
        with self.lock:
            metric_point = {
                'value': value,
                'timestamp': timestamp,
                'tags': tags or {}
            }
            self.data[metric_name].append(metric_point)
            
            # Update metadata
            if metric_name not in self.metadata:
                self.metadata[metric_name] = {
                    'first_timestamp': timestamp,
                    'last_timestamp': timestamp,
                    'count': 0,
                    'min_value': value,
                    'max_value': value
                }
            
            meta = self.metadata[metric_name]
            meta['last_timestamp'] = timestamp
            meta['count'] += 1
            meta['min_value'] = min(meta['min_value'], value)
            meta['max_value'] = max(meta['max_value'], value)
    
    def get_metrics(self, metric_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics from buffer."""
        with self.lock:
            data = list(self.data.get(metric_name, []))
            if limit:
                data = data[-limit:]
            return data
    
    def get_recent_metrics(self, metric_name: str, minutes: int) -> List[Dict[str, Any]]:
        """Get recent metrics within time window."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        with self.lock:
            return [
                point for point in self.data.get(metric_name, [])
                if point['timestamp'] >= cutoff
            ]
    
    def flush_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Flush metrics from buffer and return data."""
        with self.lock:
            if metric_name:
                data = {metric_name: list(self.data.get(metric_name, []))}
                self.data[metric_name].clear()
                if metric_name in self.metadata:
                    self.metadata[metric_name]['count'] = 0
            else:
                data = {name: list(series) for name, series in self.data.items()}
                self.data.clear()
                self.metadata.clear()
            
            self.last_flush = datetime.utcnow()
            return data
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            total_points = sum(len(series) for series in self.data.values())
            return {
                'total_metrics': len(self.data),
                'total_points': total_points,
                'buffer_utilization': total_points / (len(self.data) * self.max_size) if self.data else 0,
                'last_flush': self.last_flush,
                'metrics': {name: meta.copy() for name, meta in self.metadata.items()}
            }


class PerformanceStorageEngine:
    """
    Advanced Performance Storage Engine with time-series optimization.
    
    Features:
    - High-performance time-series data storage
    - Intelligent data retention and aggregation
    - Advanced indexing and query optimization
    - Real-time and batch data processing
    - Compression and archival strategies
    - Performance monitoring and diagnostics
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[callable] = None,
        buffer_size: int = 10000,
        flush_interval_seconds: int = 60
    ):
        """Initialize the performance storage engine."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds
        
        # Storage buffers
        self.time_series_buffer = TimeSeriesBuffer(buffer_size)
        self.batch_write_queue = deque()
        
        # Retention policies
        self.retention_rules = self._initialize_retention_rules()
        self.metric_indexes = self._initialize_metric_indexes()
        
        # Configuration
        self.config = {
            "batch_size": 1000,
            "max_query_time_seconds": 30,
            "compression_enabled": True,
            "archival_enabled": True,
            "auto_indexing_enabled": True,
            "query_optimization_enabled": True,
            "retention_cleanup_interval_hours": 6,
            "aggregation_batch_size": 5000,
            "index_maintenance_interval_hours": 24
        }
        
        # Performance tracking
        self.query_stats = defaultdict(list)
        self.storage_stats = {
            'writes_per_second': 0,
            'reads_per_second': 0,
            'average_query_time_ms': 0,
            'buffer_utilization': 0,
            'compression_ratio': 0
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="storage")
        
        logger.info("Performance Storage Engine initialized")
    
    async def start(self) -> None:
        """Start the storage engine background processes."""
        if self.is_running:
            logger.warning("Performance Storage Engine already running")
            return
        
        logger.info("Starting Performance Storage Engine")
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._buffer_flush_loop()),
            asyncio.create_task(self._data_aggregation_loop()),
            asyncio.create_task(self._retention_cleanup_loop()),
            asyncio.create_task(self._index_maintenance_loop()),
            asyncio.create_task(self._query_optimization_loop()),
            asyncio.create_task(self._stats_collection_loop())
        ]
        
        logger.info("Performance Storage Engine started successfully")
    
    async def stop(self) -> None:
        """Stop the storage engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping Performance Storage Engine")
        self.is_running = False
        self.shutdown_event.set()
        
        # Flush remaining data
        await self._flush_all_buffers()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Performance Storage Engine stopped")
    
    async def store_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a single metric with high performance."""
        try:
            timestamp = timestamp or datetime.utcnow()
            
            # Add to buffer for real-time access
            self.time_series_buffer.add_metric(metric_name, value, timestamp, tags)
            
            # Store in Redis for immediate access
            await self._store_to_redis(metric_name, value, timestamp, tags)
            
            # Queue for batch database write
            self.batch_write_queue.append({
                'metric_name': metric_name,
                'value': value,
                'timestamp': timestamp,
                'tags': tags or {}
            })
            
            return True
            
        except Exception as e:
            logger.error("Failed to store metric", 
                        metric_name=metric_name, 
                        error=str(e))
            return False
    
    async def store_metrics_batch(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store multiple metrics in batch for better performance."""
        try:
            start_time = time.time()
            success_count = 0
            error_count = 0
            
            # Process metrics in batches
            batch_size = self.config["batch_size"]
            for i in range(0, len(metrics), batch_size):
                batch = metrics[i:i + batch_size]
                
                # Store to buffer and Redis
                for metric in batch:
                    try:
                        timestamp = metric.get('timestamp', datetime.utcnow())
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        
                        self.time_series_buffer.add_metric(
                            metric['metric_name'],
                            metric['value'],
                            timestamp,
                            metric.get('tags')
                        )
                        
                        await self._store_to_redis(
                            metric['metric_name'],
                            metric['value'],
                            timestamp,
                            metric.get('tags')
                        )
                        
                        self.batch_write_queue.append(metric)
                        success_count += 1
                        
                    except Exception as e:
                        logger.error("Failed to store metric in batch", 
                                   metric=metric, 
                                   error=str(e))
                        error_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'success_count': success_count,
                'error_count': error_count,
                'processing_time_ms': processing_time,
                'throughput_per_second': success_count / max(processing_time / 1000, 0.001)
            }
            
        except Exception as e:
            logger.error("Failed to store metrics batch", error=str(e))
            return {'success_count': 0, 'error_count': len(metrics), 'error': str(e)}
    
    async def query_metrics(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        aggregation: Optional[AggregationType] = None,
        resolution_seconds: Optional[int] = None,
        tags_filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Query metrics with advanced filtering and aggregation."""
        try:
            query_start = time.time()
            end_time = end_time or datetime.utcnow()
            
            # Check if data is available in buffer for recent queries
            if (datetime.utcnow() - start_time).total_seconds() < 3600:  # Last hour
                buffer_data = await self._query_from_buffer(
                    metric_names, start_time, end_time, limit
                )
                if buffer_data['data_points'] > 0:
                    buffer_data['query_time_ms'] = (time.time() - query_start) * 1000
                    buffer_data['data_source'] = 'buffer'
                    return buffer_data
            
            # Query from Redis for short-term data
            redis_data = await self._query_from_redis(
                metric_names, start_time, end_time, aggregation, limit
            )
            if redis_data['data_points'] > 0:
                redis_data['query_time_ms'] = (time.time() - query_start) * 1000
                redis_data['data_source'] = 'redis'
                return redis_data
            
            # Query from database for historical data
            db_data = await self._query_from_database(
                metric_names, start_time, end_time, aggregation, resolution_seconds, tags_filter, limit
            )
            
            query_time = (time.time() - query_start) * 1000
            db_data['query_time_ms'] = query_time
            db_data['data_source'] = 'database'
            
            # Record query statistics
            self._record_query_stats(metric_names, query_time, db_data['data_points'])
            
            return db_data
            
        except Exception as e:
            logger.error("Failed to query metrics", 
                        metric_names=metric_names, 
                        error=str(e))
            return {
                'error': str(e),
                'data_points': 0,
                'query_time_ms': (time.time() - query_start) * 1000 if 'query_start' in locals() else 0
            }
    
    async def get_metric_statistics(
        self,
        metric_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive statistics for a metric."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Query recent data
            data = await self.query_metrics([metric_name], start_time, end_time)
            
            if not data.get('metrics', {}).get(metric_name):
                return {'error': f'No data found for metric {metric_name}'}
            
            values = [point['value'] for point in data['metrics'][metric_name]]
            
            if not values:
                return {'error': f'No values found for metric {metric_name}'}
            
            # Calculate statistics
            stats = {
                'metric_name': metric_name,
                'time_window_hours': time_window_hours,
                'sample_count': len(values),
                'min_value': min(values),
                'max_value': max(values),
                'avg_value': sum(values) / len(values),
                'median_value': np.median(values),
                'std_deviation': np.std(values),
                'percentiles': {
                    'p50': np.percentile(values, 50),
                    'p90': np.percentile(values, 90),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                },
                'trend_analysis': self._analyze_trend(values),
                'anomalies_detected': self._detect_simple_anomalies(values),
                'data_quality': {
                    'completeness': len(values) / max(time_window_hours * 60, 1),  # Assuming 1 point per minute
                    'consistency': self._calculate_consistency_score(values)
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get metric statistics", 
                        metric_name=metric_name, 
                        error=str(e))
            return {'error': str(e)}
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage performance and maintenance."""
        try:
            optimization_start = time.time()
            
            results = {
                'started_at': datetime.utcnow().isoformat(),
                'operations': []
            }
            
            # Aggregate old data
            aggregation_result = await self._aggregate_old_data()
            results['operations'].append({
                'operation': 'data_aggregation',
                'result': aggregation_result
            })
            
            # Clean up expired data
            cleanup_result = await self._cleanup_expired_data()
            results['operations'].append({
                'operation': 'expired_data_cleanup',
                'result': cleanup_result
            })
            
            # Update indexes
            index_result = await self._update_indexes()
            results['operations'].append({
                'operation': 'index_maintenance',
                'result': index_result
            })
            
            # Optimize queries
            query_optimization_result = await self._optimize_slow_queries()
            results['operations'].append({
                'operation': 'query_optimization',
                'result': query_optimization_result
            })
            
            # Calculate optimization impact
            optimization_time = (time.time() - optimization_start) * 1000
            results['total_time_ms'] = optimization_time
            results['completed_at'] = datetime.utcnow().isoformat()
            
            logger.info("Storage optimization completed", 
                       duration_ms=optimization_time)
            
            return results
            
        except Exception as e:
            logger.error("Storage optimization failed", error=str(e))
            return {'error': str(e)}
    
    # Background task methods
    async def _buffer_flush_loop(self) -> None:
        """Background task to flush buffers to persistent storage."""
        logger.info("Starting buffer flush loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._flush_all_buffers()
                await asyncio.sleep(self.flush_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Buffer flush loop error", error=str(e))
                await asyncio.sleep(self.flush_interval_seconds)
        
        logger.info("Buffer flush loop stopped")
    
    async def _data_aggregation_loop(self) -> None:
        """Background task for data aggregation."""
        logger.info("Starting data aggregation loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._aggregate_old_data()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Data aggregation loop error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.info("Data aggregation loop stopped")
    
    async def _retention_cleanup_loop(self) -> None:
        """Background task for data retention cleanup."""
        logger.info("Starting retention cleanup loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(self.config["retention_cleanup_interval_hours"] * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Retention cleanup loop error", error=str(e))
                await asyncio.sleep(self.config["retention_cleanup_interval_hours"] * 3600)
        
        logger.info("Retention cleanup loop stopped")
    
    async def _index_maintenance_loop(self) -> None:
        """Background task for index maintenance."""
        logger.info("Starting index maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._update_indexes()
                await asyncio.sleep(self.config["index_maintenance_interval_hours"] * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Index maintenance loop error", error=str(e))
                await asyncio.sleep(self.config["index_maintenance_interval_hours"] * 3600)
        
        logger.info("Index maintenance loop stopped")
    
    async def _query_optimization_loop(self) -> None:
        """Background task for query optimization."""
        logger.info("Starting query optimization loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._optimize_slow_queries()
                await asyncio.sleep(14400)  # Run every 4 hours
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Query optimization loop error", error=str(e))
                await asyncio.sleep(14400)
        
        logger.info("Query optimization loop stopped")
    
    async def _stats_collection_loop(self) -> None:
        """Background task for collecting storage statistics."""
        logger.info("Starting stats collection loop")
        
        while not self.shutdown_event.is_set():
            try:
                await self._update_storage_stats()
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Stats collection loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("Stats collection loop stopped")
    
    # Helper methods (placeholder implementations)
    def _initialize_retention_rules(self) -> Dict[str, RetentionRule]:
        """Initialize retention rules."""
        return {
            'real_time': RetentionRule(
                policy=DataRetentionPolicy.REAL_TIME,
                duration_hours=2,
                resolution_seconds=5,
                aggregation_types=[AggregationType.AVG, AggregationType.MAX]
            ),
            'short_term': RetentionRule(
                policy=DataRetentionPolicy.SHORT_TERM,
                duration_hours=24,
                resolution_seconds=60,
                aggregation_types=[AggregationType.AVG, AggregationType.MIN, AggregationType.MAX]
            ),
            'medium_term': RetentionRule(
                policy=DataRetentionPolicy.MEDIUM_TERM,
                duration_hours=168,  # 7 days
                resolution_seconds=300,  # 5 minutes
                aggregation_types=[AggregationType.AVG, AggregationType.P95]
            ),
            'long_term': RetentionRule(
                policy=DataRetentionPolicy.LONG_TERM,
                duration_hours=2160,  # 90 days
                resolution_seconds=3600,  # 1 hour
                aggregation_types=[AggregationType.AVG]
            )
        }
    
    def _initialize_metric_indexes(self) -> Dict[str, MetricIndex]:
        """Initialize metric indexes for performance."""
        return {
            'metric_time': MetricIndex(
                index_name='idx_performance_metric_time',
                columns=['metric_name', 'timestamp'],
                index_type='btree'
            ),
            'metric_tags': MetricIndex(
                index_name='idx_performance_metric_tags',
                columns=['tags'],
                index_type='gin'
            ),
            'time_range': MetricIndex(
                index_name='idx_performance_time_range',
                columns=['timestamp'],
                index_type='btree'
            )
        }
    
    async def _store_to_redis(self, metric_name: str, value: float, timestamp: datetime, tags: Dict[str, Any]) -> None:
        """Store metric to Redis for fast access."""
        # Implementation placeholder
        pass
    
    async def _query_from_buffer(self, metric_names: List[str], start_time: datetime, end_time: datetime, limit: Optional[int]) -> Dict[str, Any]:
        """Query metrics from in-memory buffer."""
        # Implementation placeholder
        return {'metrics': {}, 'data_points': 0}
    
    async def _query_from_redis(self, metric_names: List[str], start_time: datetime, end_time: datetime, aggregation: Optional[AggregationType], limit: Optional[int]) -> Dict[str, Any]:
        """Query metrics from Redis."""
        # Implementation placeholder
        return {'metrics': {}, 'data_points': 0}
    
    async def _query_from_database(self, metric_names: List[str], start_time: datetime, end_time: datetime, aggregation: Optional[AggregationType], resolution_seconds: Optional[int], tags_filter: Optional[Dict[str, Any]], limit: Optional[int]) -> Dict[str, Any]:
        """Query metrics from database with optimization."""
        # Implementation placeholder
        return {'metrics': {}, 'data_points': 0}
    
    def _record_query_stats(self, metric_names: List[str], query_time_ms: float, data_points: int) -> None:
        """Record query statistics."""
        query_pattern = f"{len(metric_names)}_metrics"
        self.query_stats[query_pattern].append({
            'query_time_ms': query_time_ms,
            'data_points': data_points,
            'timestamp': datetime.utcnow()
        })
        
        # Keep only recent stats
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.query_stats[query_pattern] = [
            stat for stat in self.query_stats[query_pattern]
            if stat['timestamp'] > cutoff
        ]
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in metric values."""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        return {
            'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
            'slope': slope,
            'correlation': np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
        }
    
    def _detect_simple_anomalies(self, values: List[float]) -> List[int]:
        """Detect simple statistical anomalies."""
        if len(values) < 5:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        threshold = 2.0  # 2 standard deviations
        anomalies = []
        
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val
            if z_score > threshold:
                anomalies.append(i)
        
        return anomalies
    
    def _calculate_consistency_score(self, values: List[float]) -> float:
        """Calculate data consistency score."""
        if len(values) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        
        cv = np.std(values) / abs(mean_val)
        
        # Convert to consistency score (lower CV = higher consistency)
        return max(0.0, 1.0 - min(cv, 1.0))
    
    async def _flush_all_buffers(self) -> None:
        """Flush all data buffers to persistent storage."""
        # Implementation placeholder
        pass
    
    async def _aggregate_old_data(self) -> Dict[str, Any]:
        """Aggregate old data according to retention policies."""
        # Implementation placeholder
        return {'aggregated_metrics': 0, 'time_saved_ms': 0}
    
    async def _cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up expired data."""
        # Implementation placeholder
        return {'deleted_records': 0, 'space_freed_mb': 0}
    
    async def _update_indexes(self) -> Dict[str, Any]:
        """Update and optimize database indexes."""
        # Implementation placeholder
        return {'indexes_updated': 0, 'performance_improvement': 0}
    
    async def _optimize_slow_queries(self) -> Dict[str, Any]:
        """Optimize slow queries."""
        # Implementation placeholder
        return {'queries_optimized': 0, 'average_improvement_ms': 0}
    
    async def _update_storage_stats(self) -> None:
        """Update storage performance statistics."""
        buffer_stats = self.time_series_buffer.get_buffer_stats()
        self.storage_stats['buffer_utilization'] = buffer_stats['buffer_utilization']
        # Add more stats collection here


# Global instance
_performance_storage_engine: Optional[PerformanceStorageEngine] = None


async def get_performance_storage_engine() -> PerformanceStorageEngine:
    """Get singleton performance storage engine instance."""
    global _performance_storage_engine
    
    if _performance_storage_engine is None:
        _performance_storage_engine = PerformanceStorageEngine()
        await _performance_storage_engine.start()
    
    return _performance_storage_engine


async def cleanup_performance_storage_engine() -> None:
    """Cleanup performance storage engine resources."""
    global _performance_storage_engine
    
    if _performance_storage_engine:
        await _performance_storage_engine.stop()
        _performance_storage_engine = None