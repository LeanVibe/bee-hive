#!/usr/bin/env python3
"""
PerformanceManager - Metrics, Monitoring, and Performance Consolidation
Phase 2.1 Implementation of Technical Debt Remediation Plan

This manager consolidates all performance monitoring, metrics collection, performance
optimization, and system observability into a unified, high-performance system
built on the BaseManager framework.

TARGET CONSOLIDATION: 14+ performance-related manager classes → 1 unified PerformanceManager
- Metrics collection and aggregation
- Performance monitoring and alerting
- Resource utilization tracking
- Benchmarking and profiling
- Performance optimization
- SLA monitoring and reporting
- System health monitoring
- Load balancing and scaling decisions
- Performance analytics and insights
"""

import asyncio
import psutil
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager
from collections import deque, defaultdict
import json
import gc

import structlog

# Import BaseManager framework
from .base_manager import (
    BaseManager, ManagerConfig, ManagerDomain, ManagerStatus, ManagerMetrics,
    PluginInterface, PluginType
)

# Import shared patterns from Phase 1
from ...common.utilities.shared_patterns import (
    standard_logging_setup, standard_error_handling
)

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    DISTRIBUTION = "distribution"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceThreshold(str, Enum):
    """Performance threshold types."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    AVAILABILITY = "availability"


class ResourceType(str, Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[float, int]
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "labels": self.labels
        }


@dataclass
class PerformanceMetric:
    """Aggregated performance metric with statistics."""
    name: str
    type: MetricType
    current_value: Union[float, int]
    min_value: Union[float, int]
    max_value: Union[float, int]
    avg_value: float
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def update(self, value: Union[float, int]) -> None:
        """Update metric with new value."""
        self.current_value = value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.count += 1
        
        # Update running average
        if self.count == 1:
            self.avg_value = float(value)
        else:
            self.avg_value = ((self.avg_value * (self.count - 1)) + value) / self.count
        
        self.last_updated = datetime.utcnow()


@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    process_count: int = 0
    thread_count: int = 0
    file_descriptors: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition and results."""
    name: str
    description: str
    target_latency_ms: float
    target_throughput: float
    target_error_rate: float
    actual_latency_ms: Optional[float] = None
    actual_throughput: Optional[float] = None
    actual_error_rate: Optional[float] = None
    passed: Optional[bool] = None
    last_run: Optional[datetime] = None
    run_count: int = 0


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    id: str
    name: str
    metric_name: str
    threshold_type: PerformanceThreshold
    threshold_value: float
    severity: AlertSeverity
    active: bool = True
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    trigger_count: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceManagerMetrics:
    """PerformanceManager-specific metrics."""
    total_metrics_collected: int = 0
    metrics_per_second: float = 0.0
    alerts_triggered: int = 0
    alerts_resolved: int = 0
    benchmarks_run: int = 0
    benchmarks_passed: int = 0
    system_resource_checks: int = 0
    performance_optimizations_applied: int = 0
    avg_metric_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gc_collections: int = 0


class PerformancePlugin(PluginInterface):
    """Base class for performance plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.PERFORMANCE
    
    async def pre_metric_collection_hook(self, metric_name: str) -> Dict[str, Any]:
        """Hook called before metric collection."""
        return {}
    
    async def post_metric_collection_hook(self, metric: MetricPoint) -> None:
        """Hook called after metric collection."""
        pass
    
    async def performance_alert_hook(self, alert: PerformanceAlert, triggered: bool) -> None:
        """Hook called when performance alert is triggered or resolved."""
        pass


class PerformanceManager(BaseManager):
    """
    Unified manager for all performance monitoring and optimization operations.
    
    CONSOLIDATION TARGET: Replaces 14+ specialized performance managers:
    - MetricsCollector
    - PerformanceMonitor
    - ResourceMonitor
    - BenchmarkManager
    - AlertManager
    - ProfilingManager
    - OptimizationManager
    - SLAMonitor
    - HealthChecker
    - LoadBalancer
    - ScalingManager
    - AnalyticsEngine
    - ReportingManager
    - ObservabilityManager
    
    Built on BaseManager framework with Phase 2 enhancements.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        # Create default config if none provided
        if config is None:
            config = ManagerConfig(
                name="PerformanceManager",
                domain=ManagerDomain.PERFORMANCE,
                max_concurrent_operations=1000,
                health_check_interval=5,  # Very frequent for performance monitoring
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=3
            )
        
        super().__init__(config)
        
        # Performance-specific state
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxsize=1000))
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.resource_metrics: SystemResourceMetrics = SystemResourceMetrics()
        self.performance_metrics = PerformanceManagerMetrics()
        
        # Performance tracking
        self.operation_timers: Dict[str, List[float]] = defaultdict(list)
        self.throughput_counters: Dict[str, int] = defaultdict(int)
        self.error_counters: Dict[str, int] = defaultdict(int)
        
        # Optimization state
        self.optimization_rules: List[Callable] = []
        self.performance_baselines: Dict[str, float] = {}
        self.sla_definitions: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._metrics_collector_task: Optional[asyncio.Task] = None
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._alert_monitor_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._metrics_lock = threading.RLock()
        self._alerts_lock = threading.RLock()
        self._benchmarks_lock = threading.RLock()
        
        # Performance optimization
        self._last_gc_time = time.time()
        self._metric_batch_size = 100
        self._metric_buffer: List[MetricPoint] = []
        
        self.logger = standard_logging_setup(
            name="PerformanceManager",
            level="INFO"
        )
    
    # BaseManager Implementation
    
    async def _setup(self) -> None:
        """Initialize performance monitoring systems."""
        self.logger.info("Setting up PerformanceManager")
        
        # Load default benchmarks and alerts
        await self._load_default_benchmarks()
        await self._load_default_alerts()
        
        # Start background tasks
        self._metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())
        self._resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        self._alert_monitor_task = asyncio.create_task(self._alert_monitor_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("PerformanceManager setup completed")
    
    async def _cleanup(self) -> None:
        """Clean up performance monitoring systems."""
        self.logger.info("Cleaning up PerformanceManager")
        
        # Cancel background tasks
        tasks = [
            self._metrics_collector_task,
            self._resource_monitor_task,
            self._alert_monitor_task,
            self._optimization_task,
            self._cleanup_task
        ]
        
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush remaining metrics
        if self._metric_buffer:
            await self._process_metric_batch(self._metric_buffer)
            self._metric_buffer.clear()
        
        self.logger.info("PerformanceManager cleanup completed")
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Performance-specific health check."""
        with self._metrics_lock:
            total_metrics = len(self.metrics)
            recent_metrics = sum(
                1 for metric in self.metrics.values()
                if (datetime.utcnow() - metric.last_updated).total_seconds() < 60
            )
        
        with self._alerts_lock:
            active_alerts = sum(1 for alert in self.alerts.values() if alert.triggered_at and not alert.resolved_at)
            total_alerts = len(self.alerts)
        
        return {
            "total_metrics": total_metrics,
            "recent_metrics": recent_metrics,
            "active_alerts": active_alerts,
            "total_alerts": total_alerts,
            "system_resources": {
                "cpu_percent": self.resource_metrics.cpu_percent,
                "memory_percent": self.resource_metrics.memory_percent,
                "disk_usage_percent": self.resource_metrics.disk_usage_percent
            },
            "performance_metrics": {
                "metrics_collected": self.performance_metrics.total_metrics_collected,
                "metrics_per_second": self.performance_metrics.metrics_per_second,
                "alerts_triggered": self.performance_metrics.alerts_triggered,
                "benchmarks_run": self.performance_metrics.benchmarks_run
            }
        }
    
    # Core Performance Operations
    
    async def record_metric(
        self,
        name: str,
        value: Union[float, int],
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a performance metric.
        
        CONSOLIDATES: MetricsCollector, GaugeRecorder, CounterIncrementer patterns
        """
        async with self.execute_with_monitoring("record_metric"):
            start_time = time.time()
            
            try:
                # Pre-metric collection hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, PerformancePlugin):
                        await plugin.pre_metric_collection_hook(name)
                
                # Create metric point
                metric_point = MetricPoint(
                    name=name,
                    value=value,
                    type=metric_type,
                    tags=tags or {},
                    labels=labels or {}
                )
                
                # Add to buffer for batch processing
                self._metric_buffer.append(metric_point)
                
                # Process batch if full
                if len(self._metric_buffer) >= self._metric_batch_size:
                    await self._process_metric_batch(self._metric_buffer)
                    self._metric_buffer.clear()
                
                # Post-metric collection hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, PerformancePlugin):
                        await plugin.post_metric_collection_hook(metric_point)
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                self.performance_metrics.total_metrics_collected += 1
                self._update_processing_time_metrics(processing_time)
                
            except Exception as e:
                self.logger.error(f"Failed to record metric: {e}", metric_name=name)
                raise
    
    @asynccontextmanager
    async def timer(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        CONSOLIDATES: TimerManager, LatencyTracker patterns
        """
        start_time = time.time()
        exception_occurred = False
        
        try:
            yield
        except Exception as e:
            exception_occurred = True
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            # Record timing metric
            await self.record_metric(
                name=f"{operation_name}.duration",
                value=duration_ms,
                metric_type=MetricType.TIMER,
                tags=tags
            )
            
            # Record success/error count
            if exception_occurred:
                await self.record_metric(
                    name=f"{operation_name}.errors",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags=tags
                )
            else:
                await self.record_metric(
                    name=f"{operation_name}.success",
                    value=1,
                    metric_type=MetricType.COUNTER,
                    tags=tags
                )
    
    async def increment_counter(
        self,
        name: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        await self.record_metric(name, value, MetricType.COUNTER, tags)
    
    async def set_gauge(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""
        await self.record_metric(name, value, MetricType.GAUGE, tags)
    
    async def record_histogram(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram metric value."""
        await self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    # Alert Management
    
    async def create_alert(
        self,
        name: str,
        metric_name: str,
        threshold_type: PerformanceThreshold,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a performance alert.
        
        CONSOLIDATES: AlertManager, ThresholdMonitor patterns
        """
        async with self.execute_with_monitoring("create_alert"):
            alert_id = str(time.time()).replace('.', '')  # Simple ID generation
            
            alert = PerformanceAlert(
                id=alert_id,
                name=name,
                metric_name=metric_name,
                threshold_type=threshold_type,
                threshold_value=threshold_value,
                severity=severity,
                conditions=conditions or {}
            )
            
            with self._alerts_lock:
                self.alerts[alert_id] = alert
            
            self.logger.info(
                f"Performance alert created",
                alert_id=alert_id,
                name=name,
                metric=metric_name,
                threshold=threshold_value
            )
            
            return alert_id
    
    async def trigger_alert(self, alert_id: str) -> None:
        """Trigger a performance alert."""
        with self._alerts_lock:
            alert = self.alerts.get(alert_id)
            if alert and not alert.triggered_at:
                alert.triggered_at = datetime.utcnow()
                alert.trigger_count += 1
                
                # Notify plugins
                for plugin in self.plugins.values():
                    if isinstance(plugin, PerformancePlugin):
                        await plugin.performance_alert_hook(alert, True)
                
                self.performance_metrics.alerts_triggered += 1
                
                self.logger.warning(
                    f"Performance alert triggered",
                    alert_id=alert_id,
                    name=alert.name,
                    severity=alert.severity.value
                )
    
    async def resolve_alert(self, alert_id: str) -> None:
        """Resolve a performance alert."""
        with self._alerts_lock:
            alert = self.alerts.get(alert_id)
            if alert and alert.triggered_at and not alert.resolved_at:
                alert.resolved_at = datetime.utcnow()
                
                # Notify plugins
                for plugin in self.plugins.values():
                    if isinstance(plugin, PerformancePlugin):
                        await plugin.performance_alert_hook(alert, False)
                
                self.performance_metrics.alerts_resolved += 1
                
                self.logger.info(
                    f"Performance alert resolved",
                    alert_id=alert_id,
                    name=alert.name,
                    duration_minutes=(alert.resolved_at - alert.triggered_at).total_seconds() / 60
                )
    
    # Benchmark Management
    
    async def create_benchmark(
        self,
        name: str,
        description: str,
        target_latency_ms: float,
        target_throughput: float,
        target_error_rate: float = 0.01
    ) -> None:
        """
        Create a performance benchmark.
        
        CONSOLIDATES: BenchmarkRunner, PerformanceValidator patterns
        """
        async with self.execute_with_monitoring("create_benchmark"):
            benchmark = PerformanceBenchmark(
                name=name,
                description=description,
                target_latency_ms=target_latency_ms,
                target_throughput=target_throughput,
                target_error_rate=target_error_rate
            )
            
            with self._benchmarks_lock:
                self.benchmarks[name] = benchmark
            
            self.logger.info(f"Performance benchmark created", name=name)
    
    async def run_benchmark(self, name: str) -> bool:
        """
        Run a performance benchmark.
        
        CONSOLIDATES: BenchmarkExecutor, PerformanceTester patterns
        """
        async with self.execute_with_monitoring("run_benchmark"):
            with self._benchmarks_lock:
                benchmark = self.benchmarks.get(name)
                if not benchmark:
                    return False
            
            try:
                # Get current metrics for the benchmark
                latency_metric = self.metrics.get(f"{name}.latency")
                throughput_metric = self.metrics.get(f"{name}.throughput")
                error_rate_metric = self.metrics.get(f"{name}.error_rate")
                
                if latency_metric:
                    benchmark.actual_latency_ms = latency_metric.current_value
                
                if throughput_metric:
                    benchmark.actual_throughput = throughput_metric.current_value
                
                if error_rate_metric:
                    benchmark.actual_error_rate = error_rate_metric.current_value
                
                # Check if benchmark passed
                benchmark.passed = (
                    (benchmark.actual_latency_ms is None or 
                     benchmark.actual_latency_ms <= benchmark.target_latency_ms) and
                    (benchmark.actual_throughput is None or 
                     benchmark.actual_throughput >= benchmark.target_throughput) and
                    (benchmark.actual_error_rate is None or 
                     benchmark.actual_error_rate <= benchmark.target_error_rate)
                )
                
                benchmark.last_run = datetime.utcnow()
                benchmark.run_count += 1
                
                self.performance_metrics.benchmarks_run += 1
                if benchmark.passed:
                    self.performance_metrics.benchmarks_passed += 1
                
                self.logger.info(
                    f"Benchmark completed",
                    name=name,
                    passed=benchmark.passed,
                    latency_actual=benchmark.actual_latency_ms,
                    latency_target=benchmark.target_latency_ms,
                    throughput_actual=benchmark.actual_throughput,
                    throughput_target=benchmark.target_throughput
                )
                
                return benchmark.passed
                
            except Exception as e:
                self.logger.error(f"Benchmark failed: {e}", name=name)
                return False
    
    # System Resource Monitoring
    
    async def get_system_metrics(self) -> SystemResourceMetrics:
        """
        Get current system resource metrics.
        
        CONSOLIDATES: ResourceMonitor, SystemProfiler patterns
        """
        async with self.execute_with_monitoring("get_system_metrics"):
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                
                # Disk metrics
                disk_usage = psutil.disk_usage('/')
                disk_usage_percent = disk_usage.percent
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                disk_io_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
                disk_io_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
                
                # Network metrics
                network_io = psutil.net_io_counters()
                network_bytes_sent = network_io.bytes_sent if network_io else 0
                network_bytes_recv = network_io.bytes_recv if network_io else 0
                network_packets_sent = network_io.packets_sent if network_io else 0
                network_packets_recv = network_io.packets_recv if network_io else 0
                
                # Load average (Unix-like systems)
                try:
                    load_average = psutil.getloadavg()
                except (AttributeError, OSError):
                    load_average = (0.0, 0.0, 0.0)
                
                # Process metrics
                process_count = len(psutil.pids())
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
                
                try:
                    file_descriptors = current_process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    file_descriptors = 0
                
                # Update resource metrics
                self.resource_metrics = SystemResourceMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_gb=memory_used_gb,
                    memory_available_gb=memory_available_gb,
                    disk_usage_percent=disk_usage_percent,
                    disk_io_read_mb=disk_io_read_mb,
                    disk_io_write_mb=disk_io_write_mb,
                    network_bytes_sent=network_bytes_sent,
                    network_bytes_recv=network_bytes_recv,
                    network_packets_sent=network_packets_sent,
                    network_packets_recv=network_packets_recv,
                    load_average=load_average,
                    process_count=process_count,
                    thread_count=thread_count,
                    file_descriptors=file_descriptors
                )
                
                self.performance_metrics.system_resource_checks += 1
                
                return self.resource_metrics
                
            except Exception as e:
                self.logger.error(f"Failed to get system metrics: {e}")
                return self.resource_metrics
    
    # Private Implementation Methods
    
    async def _process_metric_batch(self, metrics: List[MetricPoint]) -> None:
        """Process a batch of metrics efficiently."""
        start_time = time.time()
        
        try:
            with self._metrics_lock:
                for metric_point in metrics:
                    # Update or create performance metric
                    if metric_point.name in self.metrics:
                        self.metrics[metric_point.name].update(metric_point.value)
                    else:
                        self.metrics[metric_point.name] = PerformanceMetric(
                            name=metric_point.name,
                            type=metric_point.type,
                            current_value=metric_point.value,
                            min_value=metric_point.value,
                            max_value=metric_point.value,
                            avg_value=float(metric_point.value),
                            count=1,
                            tags=metric_point.tags
                        )
                    
                    # Add to history for percentile calculations
                    history = self.metric_history[metric_point.name]
                    history.append((metric_point.value, metric_point.timestamp))
                    
                    # Calculate percentiles for histograms and timers
                    if metric_point.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                        values = [v[0] for v in history]
                        if len(values) >= 10:  # Only calculate with sufficient data
                            metric = self.metrics[metric_point.name]
                            metric.p50 = statistics.median(values)
                            metric.p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else None
                            metric.p99 = statistics.quantiles(values, n=100)[98] if len(values) >= 100 else None
            
            # Update metrics per second calculation
            processing_time = time.time() - start_time
            if processing_time > 0:
                self.performance_metrics.metrics_per_second = len(metrics) / processing_time
            
        except Exception as e:
            self.logger.error(f"Failed to process metric batch: {e}")
    
    async def _check_alerts(self) -> None:
        """Check all alerts against current metrics."""
        with self._alerts_lock:
            alerts_to_check = [alert for alert in self.alerts.values() if alert.active]
        
        for alert in alerts_to_check:
            try:
                metric = self.metrics.get(alert.metric_name)
                if not metric:
                    continue
                
                # Check threshold
                threshold_exceeded = False
                
                if alert.threshold_type == PerformanceThreshold.LATENCY:
                    threshold_exceeded = metric.current_value > alert.threshold_value
                elif alert.threshold_type == PerformanceThreshold.THROUGHPUT:
                    threshold_exceeded = metric.current_value < alert.threshold_value
                elif alert.threshold_type == PerformanceThreshold.ERROR_RATE:
                    threshold_exceeded = metric.current_value > alert.threshold_value
                elif alert.threshold_type == PerformanceThreshold.RESOURCE_USAGE:
                    threshold_exceeded = metric.current_value > alert.threshold_value
                
                # Handle alert state changes
                if threshold_exceeded and not alert.triggered_at:
                    await self.trigger_alert(alert.id)
                elif not threshold_exceeded and alert.triggered_at and not alert.resolved_at:
                    await self.resolve_alert(alert.id)
                
            except Exception as e:
                self.logger.error(f"Failed to check alert: {e}", alert_id=alert.id)
    
    async def _apply_optimizations(self) -> None:
        """Apply performance optimizations based on current metrics."""
        try:
            # Garbage collection optimization
            current_time = time.time()
            if current_time - self._last_gc_time > 300:  # Every 5 minutes
                memory_before = self.resource_metrics.memory_used_gb
                
                # Force garbage collection
                collected = gc.collect()
                
                memory_after = (await self.get_system_metrics()).memory_used_gb
                memory_freed = memory_before - memory_after
                
                if memory_freed > 0:
                    self.performance_metrics.performance_optimizations_applied += 1
                    self.logger.info(
                        f"Garbage collection freed memory",
                        collected_objects=collected,
                        memory_freed_mb=memory_freed * 1024
                    )
                
                self.performance_metrics.gc_collections += 1
                self._last_gc_time = current_time
            
            # Apply custom optimization rules
            for optimization_rule in self.optimization_rules:
                try:
                    applied = await optimization_rule(self.metrics, self.resource_metrics)
                    if applied:
                        self.performance_metrics.performance_optimizations_applied += 1
                except Exception as e:
                    self.logger.warning(f"Optimization rule failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
    
    async def _load_default_benchmarks(self) -> None:
        """Load default performance benchmarks."""
        default_benchmarks = [
            ("api_response_time", "API Response Time", 100.0, 1000.0, 0.01),
            ("database_query", "Database Query Performance", 50.0, 500.0, 0.005),
            ("cache_hit_rate", "Cache Hit Rate", 10.0, 0.95, 0.0),
        ]
        
        for name, desc, latency, throughput, error_rate in default_benchmarks:
            await self.create_benchmark(name, desc, latency, throughput, error_rate)
        
        self.logger.info("Default performance benchmarks loaded")
    
    async def _load_default_alerts(self) -> None:
        """Load default performance alerts."""
        default_alerts = [
            ("high_cpu_usage", "system.cpu_percent", PerformanceThreshold.RESOURCE_USAGE, 80.0, AlertSeverity.WARNING),
            ("high_memory_usage", "system.memory_percent", PerformanceThreshold.RESOURCE_USAGE, 85.0, AlertSeverity.WARNING),
            ("high_disk_usage", "system.disk_usage_percent", PerformanceThreshold.RESOURCE_USAGE, 90.0, AlertSeverity.ERROR),
        ]
        
        for name, metric, threshold_type, value, severity in default_alerts:
            await self.create_alert(name, metric, threshold_type, value, severity)
        
        self.logger.info("Default performance alerts loaded")
    
    # Background Tasks
    
    async def _metrics_collector_loop(self) -> None:
        """Collect system metrics periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # Collect every 5 seconds
                if self._shutdown_event.is_set():
                    break
                
                # Collect system metrics
                system_metrics = await self.get_system_metrics()
                
                # Record as performance metrics
                await self.set_gauge("system.cpu_percent", system_metrics.cpu_percent)
                await self.set_gauge("system.memory_percent", system_metrics.memory_percent)
                await self.set_gauge("system.memory_used_gb", system_metrics.memory_used_gb)
                await self.set_gauge("system.disk_usage_percent", system_metrics.disk_usage_percent)
                await self.set_gauge("system.load_average_1m", system_metrics.load_average[0])
                await self.set_gauge("system.process_count", system_metrics.process_count)
                await self.set_gauge("system.thread_count", system_metrics.thread_count)
                
                # Process any buffered metrics
                if self._metric_buffer:
                    await self._process_metric_batch(self._metric_buffer)
                    self._metric_buffer.clear()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collector loop error: {e}")
    
    async def _resource_monitor_loop(self) -> None:
        """Monitor system resources and performance."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                if self._shutdown_event.is_set():
                    break
                
                # Update performance manager memory usage
                current_process = psutil.Process()
                memory_info = current_process.memory_info()
                self.performance_metrics.memory_usage_mb = memory_info.rss / (1024**2)
                
                await self.set_gauge("performance_manager.memory_mb", self.performance_metrics.memory_usage_mb)
                await self.set_gauge("performance_manager.metrics_count", len(self.metrics))
                await self.set_gauge("performance_manager.alerts_count", len(self.alerts))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource monitor loop error: {e}")
    
    async def _alert_monitor_loop(self) -> None:
        """Monitor performance alerts."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(15)  # Check every 15 seconds
                if self._shutdown_event.is_set():
                    break
                
                await self._check_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert monitor loop error: {e}")
    
    async def _optimization_loop(self) -> None:
        """Apply performance optimizations periodically."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                if self._shutdown_event.is_set():
                    break
                
                await self._apply_optimizations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Clean up old metrics and performance data."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self._shutdown_event.is_set():
                    break
                
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean up metric history
                for metric_name, history in self.metric_history.items():
                    # Remove old entries
                    while history and history[0][1] < cutoff_time:
                        history.popleft()
                
                # Clean up resolved alerts older than 7 days
                old_alert_cutoff = datetime.utcnow() - timedelta(days=7)
                with self._alerts_lock:
                    alerts_to_remove = []
                    for alert_id, alert in self.alerts.items():
                        if (alert.resolved_at and 
                            alert.resolved_at < old_alert_cutoff):
                            alerts_to_remove.append(alert_id)
                    
                    for alert_id in alerts_to_remove:
                        del self.alerts[alert_id]
                
                if alerts_to_remove:
                    self.logger.debug(f"Cleaned up {len(alerts_to_remove)} old alerts")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    # Metrics Helpers
    
    def _update_processing_time_metrics(self, processing_time_ms: float) -> None:
        """Update metric processing time metrics."""
        current_avg = self.performance_metrics.avg_metric_processing_time_ms
        total_collected = self.performance_metrics.total_metrics_collected
        
        if total_collected == 1:
            self.performance_metrics.avg_metric_processing_time_ms = processing_time_ms
        else:
            self.performance_metrics.avg_metric_processing_time_ms = (
                (current_avg * (total_collected - 1) + processing_time_ms) / total_collected
            )
    
    # Public API Extensions
    
    def get_performance_metrics(self) -> PerformanceManagerMetrics:
        """Get current performance manager metrics."""
        return self.performance_metrics
    
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a specific metric by name."""
        with self._metrics_lock:
            return self.metrics.get(name)
    
    def list_metrics(self, pattern: Optional[str] = None) -> List[PerformanceMetric]:
        """List all metrics, optionally filtered by pattern."""
        with self._metrics_lock:
            metrics = list(self.metrics.values())
        
        if pattern:
            metrics = [m for m in metrics if pattern in m.name]
        
        return metrics
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all currently active alerts."""
        with self._alerts_lock:
            return [
                alert for alert in self.alerts.values()
                if alert.triggered_at and not alert.resolved_at
            ]
    
    async def add_optimization_rule(self, rule: Callable) -> None:
        """Add a custom optimization rule."""
        self.optimization_rules.append(rule)
        self.logger.info("Custom optimization rule added")
    
    async def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        if format.lower() == "json":
            with self._metrics_lock:
                metrics_data = {
                    name: {
                        "current_value": metric.current_value,
                        "min_value": metric.min_value,
                        "max_value": metric.max_value,
                        "avg_value": metric.avg_value,
                        "count": metric.count,
                        "p50": metric.p50,
                        "p95": metric.p95,
                        "p99": metric.p99,
                        "last_updated": metric.last_updated.isoformat(),
                        "tags": metric.tags
                    }
                    for name, metric in self.metrics.items()
                }
            
            return json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics_data,
                "system_resources": {
                    "cpu_percent": self.resource_metrics.cpu_percent,
                    "memory_percent": self.resource_metrics.memory_percent,
                    "disk_usage_percent": self.resource_metrics.disk_usage_percent
                }
            }, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Plugin Examples

class PrometheusExporterPlugin(PerformancePlugin):
    """Plugin for exporting metrics to Prometheus format."""
    
    @property
    def name(self) -> str:
        return "PrometheusExporter"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.exported_metrics: Dict[str, Any] = {}
    
    async def cleanup(self) -> None:
        pass
    
    async def post_metric_collection_hook(self, metric: MetricPoint) -> None:
        # Convert to Prometheus format
        prometheus_line = f"{metric.name.replace('.', '_')} {metric.value} {int(metric.timestamp.timestamp() * 1000)}"
        self.exported_metrics[metric.name] = prometheus_line


class PerformanceAlertsPlugin(PerformancePlugin):
    """Plugin for advanced performance alerting."""
    
    @property
    def name(self) -> str:
        return "PerformanceAlerts"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.alert_history: List[Dict[str, Any]] = []
    
    async def cleanup(self) -> None:
        pass
    
    async def performance_alert_hook(self, alert: PerformanceAlert, triggered: bool) -> None:
        alert_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_id": alert.id,
            "alert_name": alert.name,
            "metric_name": alert.metric_name,
            "severity": alert.severity.value,
            "triggered": triggered,
            "threshold_value": alert.threshold_value
        }
        
        self.alert_history.append(alert_event)
        
        # Keep only last 1000 events
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]