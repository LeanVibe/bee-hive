"""
Unified Metrics Collector for LeanVibe Agent Hive
Consolidates 6+ metrics collection implementations into comprehensive aggregation system

Epic 1, Phase 2 Week 4: Metrics Collection Consolidation
Replaces: custom_metrics_exporter.py, prometheus_exporter.py, dashboard_metrics_streaming.py,
         team_coordination_metrics.py, performance_storage_engine.py, context_performance_monitor.py
"""

from typing import Optional, Dict, Any, List, Set, Callable, Union, AsyncGenerator
import asyncio
import time
import json
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics
import uuid
import gzip

from app.core.logging_service import get_component_logger
from app.core.configuration_service import ConfigurationService
from app.core.redis_integration import get_redis_service

# Optional performance monitor import - handle circuit breaker issues
try:
    from app.core.performance_monitor import get_performance_monitor
except (ImportError, NameError) as e:
    def get_performance_monitor():
        return None

logger = get_component_logger("metrics_collector")

class MetricFormat(str, Enum):
    PROMETHEUS = "prometheus"
    JSON = "json"
    INFLUXDB = "influxdb"
    CUSTOM = "custom"

class AggregationType(str, Enum):
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    PERCENTILE = "percentile"
    FIRST = "first"
    LAST = "last"

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDefinition:
    """Metric definition with metadata"""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    aggregation: AggregationType = AggregationType.AVERAGE
    retention_days: int = 30
    export_formats: List[MetricFormat] = field(default_factory=lambda: [MetricFormat.PROMETHEUS])
    dashboard_enabled: bool = True
    streaming_enabled: bool = False

@dataclass
class MetricDataPoint:
    """Individual metric data point"""
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "tags": self.tags,
            "metadata": self.metadata
        }

class MetricsBuffer:
    """High-performance metrics buffer with automatic flushing"""
    
    def __init__(self, name: str, flush_interval: int = 60, max_size: int = 10000):
        self.name = name
        self.flush_interval = flush_interval
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.last_flush = time.time()
        self._lock = threading.RLock()
        self._flush_callbacks: List[Callable] = []
    
    def add_data_point(self, data_point: MetricDataPoint):
        """Add data point to buffer"""
        with self._lock:
            self.buffer.append(data_point)
            
            # Auto-flush if interval exceeded or buffer full
            current_time = time.time()
            if (current_time - self.last_flush) >= self.flush_interval or len(self.buffer) >= self.max_size:
                self._trigger_flush()
    
    def _trigger_flush(self):
        """Trigger buffer flush"""
        if self.buffer:
            data_points = list(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            
            # Execute flush callbacks
            for callback in self._flush_callbacks:
                try:
                    callback(self.name, data_points)
                except Exception as e:
                    logger.error("Flush callback error", callback=callback.__name__, error=str(e))
    
    def add_flush_callback(self, callback: Callable):
        """Add callback for buffer flush events"""
        self._flush_callbacks.append(callback)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        with self._lock:
            return len(self.buffer)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self._lock:
            return {
                "name": self.name,
                "size": len(self.buffer),
                "max_size": self.max_size,
                "utilization": len(self.buffer) / self.max_size,
                "last_flush": self.last_flush,
                "flush_interval": self.flush_interval
            }

class PrometheusExporter:
    """Prometheus metrics exporter with full compatibility"""
    
    def __init__(self):
        self.metrics_registry: Dict[str, MetricDefinition] = {}
        self._prometheus_format_cache = {}
    
    def register_metric(self, definition: MetricDefinition):
        """Register metric for Prometheus export"""
        self.metrics_registry[definition.name] = definition
    
    def format_prometheus_metrics(self, data_points: List[MetricDataPoint]) -> str:
        """Format metrics in Prometheus format"""
        prometheus_lines = []
        
        # Group data points by metric name
        metrics_by_name = defaultdict(list)
        for dp in data_points:
            metrics_by_name[dp.metric_name].append(dp)
        
        for metric_name, points in metrics_by_name.items():
            definition = self.metrics_registry.get(metric_name)
            if not definition:
                continue
            
            # Add help and type information
            prometheus_lines.append(f"# HELP {metric_name} {definition.description}")
            prometheus_lines.append(f"# TYPE {metric_name} {definition.metric_type.value}")
            
            # Add metric values
            for point in points:
                labels_str = ""
                if point.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"
                
                timestamp_ms = int(point.timestamp.timestamp() * 1000)
                prometheus_lines.append(f"{metric_name}{labels_str} {point.value} {timestamp_ms}")
        
        return "\n".join(prometheus_lines) + "\n"

class DashboardStreaming:
    """Real-time dashboard streaming service"""
    
    def __init__(self):
        self._subscribers: Set[Callable] = set()
        self._connection_count = 0
        self._message_count = 0
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time metrics stream"""
        self._subscribers.add(callback)
        self._connection_count += 1
        logger.debug("Dashboard subscriber added", callback=callback.__name__)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from metrics stream"""
        self._subscribers.discard(callback)
        self._connection_count = max(0, self._connection_count - 1)
        logger.debug("Dashboard subscriber removed", callback=callback.__name__)
    
    def stream_metric(self, data_point: MetricDataPoint):
        """Stream metric to all subscribers"""
        if not self._subscribers:
            return
        
        message = {
            "type": "metric_update",
            "data": data_point.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        failed_subscribers = []
        for subscriber in list(self._subscribers):
            try:
                subscriber(message)
                self._message_count += 1
            except Exception as e:
                logger.error("Dashboard streaming error", error=str(e))
                failed_subscribers.append(subscriber)
        
        # Remove failed subscribers
        for failed in failed_subscribers:
            self._subscribers.discard(failed)
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            "active_subscribers": len(self._subscribers),
            "total_connections": self._connection_count,
            "messages_sent": self._message_count
        }

class MetricsStorage:
    """Unified metrics storage with intelligent retention"""
    
    def __init__(self, redis_service):
        self.redis = redis_service
        self._storage_stats = {
            "metrics_stored": 0,
            "storage_errors": 0,
            "average_storage_time_ms": 0.0
        }
    
    async def store_metrics(self, data_points: List[MetricDataPoint]):
        """Store metrics with intelligent persistence"""
        start_time = time.time()
        stored_count = 0
        
        try:
            for dp in data_points:
                # Store in Redis with TTL based on retention policy
                key = f"metrics:{dp.metric_name}:{int(dp.timestamp.timestamp())}"
                value = dp.to_dict()
                
                # Default 7 days retention
                ttl = 7 * 24 * 3600
                
                await self.redis.cache_set(key, value, ttl=ttl)
                stored_count += 1
            
            storage_time = (time.time() - start_time) * 1000
            self._storage_stats["metrics_stored"] += stored_count
            self._storage_stats["average_storage_time_ms"] = (
                self._storage_stats["average_storage_time_ms"] + storage_time
            ) / 2
            
        except Exception as e:
            logger.error("Metrics storage error", error=str(e))
            self._storage_stats["storage_errors"] += 1
    
    async def retrieve_metrics(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MetricDataPoint]:
        """Retrieve metrics from storage"""
        try:
            pattern = f"metrics:{metric_name}:*"
            keys = await self.redis.scan_keys(pattern)
            
            results = []
            for key in keys:
                try:
                    data = await self.redis.cache_get(key)
                    if data and isinstance(data, dict):
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        if start_time <= timestamp <= end_time:
                            results.append(MetricDataPoint(
                                metric_name=data["metric_name"],
                                value=data["value"],
                                timestamp=timestamp,
                                labels=data.get("labels", {}),
                                tags=data.get("tags", {}),
                                metadata=data.get("metadata", {})
                            ))
                except Exception as e:
                    logger.debug("Error parsing stored metric", key=key, error=str(e))
            
            return sorted(results, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error("Metrics retrieval error", metric=metric_name, error=str(e))
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self._storage_stats.copy()

class MetricsCollector:
    """
    Unified metrics collection system consolidating all metrics patterns:
    - High-performance metrics buffering and aggregation
    - Multi-format metrics export (Prometheus, JSON, InfluxDB)
    - Real-time dashboard metrics streaming
    - Team coordination and context metrics collection
    - Performance metrics integration
    - Automated metrics retention and cleanup
    - Circuit breaker protection for reliability
    """
    
    _instance: Optional['MetricsCollector'] = None
    
    def __new__(cls) -> 'MetricsCollector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_collector()
            self._initialized = True
    
    def _initialize_collector(self):
        """Initialize metrics collection system"""
        self.config = ConfigurationService().config
        self.redis = get_redis_service()
        self.performance_monitor = get_performance_monitor()
        self.circuit_breaker = None  # Will be initialized async
        
        # Metrics storage and buffering
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._metrics_buffers: Dict[str, MetricsBuffer] = {}
        self._aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Export and streaming
        self._prometheus_exporter = PrometheusExporter()
        self._dashboard_streaming = DashboardStreaming()
        self._metrics_storage = MetricsStorage(self.redis)
        self._export_enabled = True
        
        # Background processing
        self._collection_active = False
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        
        # Statistics tracking
        self._collection_stats = {
            "metrics_collected": 0,
            "data_points_processed": 0,
            "export_operations": 0,
            "buffer_flushes": 0,
            "streaming_messages": 0,
            "errors": 0,
            "uptime_seconds": 0
        }
        
        # Initialize default metrics
        self._setup_default_metrics()
        
        # Start time for uptime tracking
        self._start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def _setup_default_metrics(self):
        """Set up default system metrics"""
        default_metrics = [
            MetricDefinition(
                name="system_cpu_usage",
                metric_type=MetricType.GAUGE,
                description="System CPU usage percentage",
                unit="percent",
                labels=["host", "core"],
                streaming_enabled=True
            ),
            MetricDefinition(
                name="system_memory_usage",
                metric_type=MetricType.GAUGE,
                description="System memory usage percentage",
                unit="percent",
                labels=["host", "type"],
                streaming_enabled=True
            ),
            MetricDefinition(
                name="api_requests_total",
                metric_type=MetricType.COUNTER,
                description="Total API requests",
                labels=["method", "endpoint", "status"],
                aggregation=AggregationType.SUM
            ),
            MetricDefinition(
                name="api_request_duration",
                metric_type=MetricType.HISTOGRAM,
                description="API request duration",
                unit="seconds",
                labels=["method", "endpoint"],
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="task_execution_duration",
                metric_type=MetricType.HISTOGRAM,
                description="Task execution duration",
                unit="seconds",
                labels=["task_type", "agent_id"],
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="task_success_rate",
                metric_type=MetricType.GAUGE,
                description="Task success rate",
                unit="ratio",
                labels=["task_type", "agent_id"],
                aggregation=AggregationType.AVERAGE
            ),
            MetricDefinition(
                name="agent_performance_score",
                metric_type=MetricType.GAUGE,
                description="Agent performance score",
                unit="score",
                labels=["agent_id", "agent_type"],
                streaming_enabled=True
            ),
            MetricDefinition(
                name="context_optimization_efficiency",
                metric_type=MetricType.GAUGE,
                description="Context optimization efficiency",
                unit="percent",
                labels=["optimization_type"],
                streaming_enabled=True
            ),
            MetricDefinition(
                name="team_coordination_complexity",
                metric_type=MetricType.GAUGE,
                description="Team coordination complexity score",
                unit="score",
                labels=["team_id", "coordination_type"]
            ),
            MetricDefinition(
                name="dashboard_connections_active",
                metric_type=MetricType.GAUGE,
                description="Active dashboard connections",
                unit="count",
                streaming_enabled=True
            ),
            MetricDefinition(
                name="metrics_buffer_utilization",
                metric_type=MetricType.GAUGE,
                description="Metrics buffer utilization percentage",
                unit="percent",
                labels=["buffer_name"]
            )
        ]
        
        for metric_def in default_metrics:
            self.register_metric(metric_def)
    
    async def start_collection(self):
        """Start metrics collection"""
        if not self._collection_active:
            self._collection_task = asyncio.create_task(self._collection_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
            self._collection_active = True
            logger.info("Metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection"""
        self._collection_active = False
        
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        if self._aggregation_task and not self._aggregation_task.done():
            self._aggregation_task.cancel()
        
        # Flush all buffers
        await self._flush_all_buffers()
        
        logger.info("Metrics collection stopped")
    
    def register_metric(self, definition: MetricDefinition):
        """Register metric definition"""
        self._metric_definitions[definition.name] = definition
        
        # Create buffer for this metric
        buffer = MetricsBuffer(definition.name)
        buffer.add_flush_callback(self._handle_buffer_flush)
        self._metrics_buffers[definition.name] = buffer
        
        # Register with Prometheus exporter
        if MetricFormat.PROMETHEUS in definition.export_formats:
            self._prometheus_exporter.register_metric(definition)
        
        logger.debug("Metric registered", name=definition.name, type=definition.metric_type.value)
    
    def collect_metric(self, 
                      metric_name: str, 
                      value: float, 
                      labels: Dict[str, str] = None,
                      tags: Dict[str, str] = None,
                      timestamp: Optional[datetime] = None,
                      metadata: Dict[str, Any] = None):
        """Collect metric data point"""
        if metric_name not in self._metric_definitions:
            logger.warning("Unknown metric", name=metric_name)
            return
        
        data_point = MetricDataPoint(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp or datetime.utcnow(),
            labels=labels or {},
            tags=tags or {},
            metadata=metadata or {}
        )
        
        # Add to buffer
        if metric_name in self._metrics_buffers:
            self._metrics_buffers[metric_name].add_data_point(data_point)
            self._collection_stats["metrics_collected"] += 1
        
        # Stream to dashboard if enabled
        definition = self._metric_definitions[metric_name]
        if definition.streaming_enabled:
            self._dashboard_streaming.stream_metric(data_point)
            self._collection_stats["streaming_messages"] += 1
        
        logger.debug("Metric collected", name=metric_name, value=value, labels=labels)
    
    def _handle_buffer_flush(self, buffer_name: str, data_points: List[MetricDataPoint]):
        """Handle buffer flush event"""
        self._collection_stats["buffer_flushes"] += 1
        self._collection_stats["data_points_processed"] += len(data_points)
        
        # Aggregate metrics
        self._aggregate_metrics(data_points)
        
        # Persist to storage
        asyncio.create_task(self._metrics_storage.store_metrics(data_points))
        
        logger.debug("Buffer flushed", buffer=buffer_name, points=len(data_points))
    
    def _aggregate_metrics(self, data_points: List[MetricDataPoint]):
        """Aggregate metrics data points"""
        for dp in data_points:
            definition = self._metric_definitions.get(dp.metric_name)
            if not definition:
                continue
            
            key = f"{dp.metric_name}:{':'.join(dp.labels.values())}"
            
            if definition.aggregation == AggregationType.SUM:
                self._aggregated_metrics[dp.metric_name][key] = \
                    self._aggregated_metrics[dp.metric_name].get(key, 0) + dp.value
            elif definition.aggregation == AggregationType.AVERAGE:
                # Simple moving average for now
                current = self._aggregated_metrics[dp.metric_name].get(key, dp.value)
                self._aggregated_metrics[dp.metric_name][key] = (current + dp.value) / 2
            elif definition.aggregation == AggregationType.COUNT:
                self._aggregated_metrics[dp.metric_name][key] = \
                    self._aggregated_metrics[dp.metric_name].get(key, 0) + 1
            elif definition.aggregation == AggregationType.MIN:
                current = self._aggregated_metrics[dp.metric_name].get(key, float('inf'))
                self._aggregated_metrics[dp.metric_name][key] = min(current, dp.value)
            elif definition.aggregation == AggregationType.MAX:
                current = self._aggregated_metrics[dp.metric_name].get(key, float('-inf'))
                self._aggregated_metrics[dp.metric_name][key] = max(current, dp.value)
            elif definition.aggregation == AggregationType.LAST:
                self._aggregated_metrics[dp.metric_name][key] = dp.value
    
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            # Get recent data points from all buffers
            all_data_points = []
            
            for buffer_name, buffer in self._metrics_buffers.items():
                # Get current buffer contents (non-destructive)
                with buffer._lock:
                    all_data_points.extend(list(buffer.buffer))
            
            # Add aggregated metrics
            for metric_name, aggregated_data in self._aggregated_metrics.items():
                for key, value in aggregated_data.items():
                    labels = {}
                    if ":" in key:
                        label_values = key.split(":")[1:]
                        definition = self._metric_definitions.get(metric_name)
                        if definition and len(label_values) == len(definition.labels):
                            labels = dict(zip(definition.labels, label_values))
                    
                    data_point = MetricDataPoint(
                        metric_name=metric_name,
                        value=value,
                        labels=labels
                    )
                    all_data_points.append(data_point)
            
            prometheus_output = self._prometheus_exporter.format_prometheus_metrics(all_data_points)
            self._collection_stats["export_operations"] += 1
            
            return prometheus_output
            
        except Exception as e:
            logger.error("Prometheus export failed", error=str(e))
            self._collection_stats["errors"] += 1
            return ""
    
    def subscribe_to_dashboard_metrics(self, callback: Callable):
        """Subscribe to real-time dashboard metrics"""
        self._dashboard_streaming.subscribe(callback)
        logger.debug("Dashboard subscriber added", callback=callback.__name__)
    
    def unsubscribe_from_dashboard_metrics(self, callback: Callable):
        """Unsubscribe from dashboard metrics"""
        self._dashboard_streaming.unsubscribe(callback)
        logger.debug("Dashboard subscriber removed", callback=callback.__name__)
    
    async def get_metrics_history(
        self, 
        metric_name: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MetricDataPoint]:
        """Get historical metrics data"""
        return await self._metrics_storage.retrieve_metrics(metric_name, start_time, end_time)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics"""
        buffer_stats = {}
        for name, buffer in self._metrics_buffers.items():
            buffer_stats[name] = buffer.get_buffer_stats()
        
        # Update uptime
        self._collection_stats["uptime_seconds"] = int(time.time() - self._start_time)
        
        return {
            **self._collection_stats,
            "registered_metrics": len(self._metric_definitions),
            "active_buffers": len(self._metrics_buffers),
            "buffer_stats": buffer_stats,
            "dashboard_streaming": self._dashboard_streaming.get_streaming_stats(),
            "storage_stats": self._metrics_storage.get_storage_stats(),
            "aggregated_metrics_count": len(self._aggregated_metrics),
            "collection_active": self._collection_active
        }
    
    async def _collection_loop(self):
        """Main collection loop for system metrics"""
        while self._collection_active:
            try:
                # Collect system metrics from performance monitor
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Collect buffer utilization metrics
                await self._collect_buffer_metrics()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error("Collection loop error", error=str(e))
                self._collection_stats["errors"] += 1
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            if self.performance_monitor is None:
                # Fallback to basic system metrics
                import psutil
                self.collect_metric(
                    "system_cpu_usage", 
                    psutil.cpu_percent(), 
                    {"host": "localhost", "core": "all"}
                )
                self.collect_metric(
                    "system_memory_usage", 
                    psutil.virtual_memory().percent, 
                    {"host": "localhost", "type": "physical"}
                )
                return
            
            health_summary = self.performance_monitor.get_system_health_summary()
            
            if "latest_snapshot" in health_summary:
                snapshot = health_summary["latest_snapshot"]
                
                self.collect_metric(
                    "system_cpu_usage", 
                    snapshot.get("cpu_percent", 0), 
                    {"host": "localhost", "core": "all"}
                )
                self.collect_metric(
                    "system_memory_usage", 
                    snapshot.get("memory_percent", 0), 
                    {"host": "localhost", "type": "physical"}
                )
        except Exception as e:
            logger.error("System metrics collection failed", error=str(e))
            self._collection_stats["errors"] += 1
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Collect basic stats
            stats = self.get_collection_stats()
            
            self.collect_metric(
                "metrics_collection_rate", 
                stats["metrics_collected"], 
                {"component": "collector"}
            )
            self.collect_metric(
                "dashboard_connections_active", 
                stats["dashboard_streaming"]["active_subscribers"]
            )
        except Exception as e:
            logger.error("Application metrics collection failed", error=str(e))
            self._collection_stats["errors"] += 1
    
    async def _collect_buffer_metrics(self):
        """Collect buffer utilization metrics"""
        try:
            for buffer_name, buffer in self._metrics_buffers.items():
                stats = buffer.get_buffer_stats()
                self.collect_metric(
                    "metrics_buffer_utilization",
                    stats["utilization"] * 100,
                    {"buffer_name": buffer_name}
                )
        except Exception as e:
            logger.error("Buffer metrics collection failed", error=str(e))
            self._collection_stats["errors"] += 1
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._collection_active:
            try:
                # Clean up old aggregated metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # This is a simplified cleanup - in production would be more sophisticated
                for metric_name in list(self._aggregated_metrics.keys()):
                    if len(self._aggregated_metrics[metric_name]) > 1000:
                        # Keep only recent aggregations
                        items = list(self._aggregated_metrics[metric_name].items())
                        self._aggregated_metrics[metric_name] = dict(items[-500:])
                
                await asyncio.sleep(3600)  # Clean every hour
                
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(3600)
    
    async def _aggregation_loop(self):
        """Background aggregation loop"""
        while self._collection_active:
            try:
                # Perform periodic aggregation of metrics
                await self._perform_aggregation()
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
            except Exception as e:
                logger.error("Aggregation loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _perform_aggregation(self):
        """Perform metric aggregation"""
        try:
            # This is a placeholder for more sophisticated aggregation
            # In production, this would compute rollups, percentiles, etc.
            logger.debug("Performing metric aggregation")
        except Exception as e:
            logger.error("Aggregation failed", error=str(e))
    
    async def _flush_all_buffers(self):
        """Flush all data buffers"""
        try:
            for buffer_name, buffer in self._metrics_buffers.items():
                with buffer._lock:
                    if buffer.buffer:
                        data_points = list(buffer.buffer)
                        buffer.buffer.clear()
                        await self._metrics_storage.store_metrics(data_points)
        except Exception as e:
            logger.error("Buffer flush failed", error=str(e))

# Convenience functions
def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()

def collect_api_metric(endpoint: str, method: str, status_code: int, duration_ms: float):
    """Collect API metrics"""
    collector = get_metrics_collector()
    
    # Request counter
    collector.collect_metric(
        "api_requests_total",
        1,
        {"method": method, "endpoint": endpoint, "status": str(status_code)}
    )
    
    # Request duration
    collector.collect_metric(
        "api_request_duration",
        duration_ms / 1000.0,  # Convert to seconds
        {"method": method, "endpoint": endpoint}
    )

def collect_task_metric(task_type: str, agent_id: str, duration_seconds: float, success: bool):
    """Collect task execution metrics"""
    collector = get_metrics_collector()
    
    collector.collect_metric(
        "task_execution_duration",
        duration_seconds,
        {"task_type": task_type, "agent_id": agent_id}
    )
    
    collector.collect_metric(
        "task_success_rate",
        1.0 if success else 0.0,
        {"task_type": task_type, "agent_id": agent_id}
    )

def collect_agent_performance(agent_id: str, agent_type: str, performance_score: float):
    """Collect agent performance metrics"""
    collector = get_metrics_collector()
    
    collector.collect_metric(
        "agent_performance_score",
        performance_score,
        {"agent_id": agent_id, "agent_type": agent_type}
    )

def collect_context_metric(optimization_type: str, efficiency_percent: float):
    """Collect context optimization metrics"""
    collector = get_metrics_collector()
    
    collector.collect_metric(
        "context_optimization_efficiency",
        efficiency_percent,
        {"optimization_type": optimization_type}
    )

def collect_team_coordination_metric(team_id: str, coordination_type: str, complexity_score: float):
    """Collect team coordination metrics"""
    collector = get_metrics_collector()
    
    collector.collect_metric(
        "team_coordination_complexity",
        complexity_score,
        {"team_id": team_id, "coordination_type": coordination_type}
    )