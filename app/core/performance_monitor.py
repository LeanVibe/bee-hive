"""
Unified Performance Monitor for LeanVibe Agent Hive
Consolidates 8+ performance monitoring implementations into comprehensive tracking system

This system provides:
- System resource monitoring (CPU, memory, disk, network)
- Application performance tracking (response times, throughput, errors)
- Database and Redis performance monitoring
- Orchestrator and task engine performance tracking
- Real-time alerting and threshold monitoring
- Performance benchmarking and validation
- Historical trend analysis and reporting
- Intelligent optimization recommendations

Replaces and consolidates:
- performance_monitoring.py
- performance_metrics_collector.py
- performance_evaluator.py
- performance_validator.py
- performance_benchmarks.py
- vs_2_1_performance_validator.py
- database_performance_validator.py
- performance_metrics_publisher.py
"""

from typing import Optional, Dict, Any, List, Set, Callable, Union, Tuple
import asyncio
import time
import psutil
import resource
import statistics
import threading
import weakref
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from .database import get_session
except ImportError:
    def get_session():
        return None

try:
    from .redis_integration import get_redis_service
except ImportError:
    def get_redis_service():
        return None

try:
    from .circuit_breaker import CircuitBreakerService
except ImportError:
    class CircuitBreakerService:
        @staticmethod
        def get_circuit_breaker(name):
            def dummy_breaker(func):
                return func
            return dummy_breaker

try:
    from .configuration_service import ConfigurationService
except ImportError:
    class ConfigurationService:
        def __init__(self):
            self.config = {}

try:
    from ..models.performance_metric import PerformanceMetric
except ImportError:
    class PerformanceMetric:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory
except ImportError:
    class WorkloadSnapshot:
        pass
    class AgentPerformanceHistory:
        pass

logger = structlog.get_logger()


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class PerformanceLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_connections: int = 0
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0


@dataclass
class PerformanceBenchmark:
    """Performance benchmark definition"""
    name: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    higher_is_better: bool = True
    description: str = ""


@dataclass
class PerformanceMetricValue:
    """Individual performance metric value"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False


class PerformanceTracker:
    """High-performance metric tracking with circular buffers"""
    
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self.values = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # seconds
    
    def record(self, value: float, timestamp: Optional[datetime] = None):
        """Record performance metric value"""
        with self._lock:
            ts = timestamp or datetime.utcnow()
            self.values.append(value)
            self.timestamps.append(ts)
            self._invalidate_cache()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of tracked values"""
        with self._lock:
            current_time = time.time()
            
            # Return cached stats if still valid
            if (current_time - self._cache_timestamp) < self._cache_ttl and self._stats_cache:
                return self._stats_cache.copy()
            
            if not self.values:
                return {}
            
            values_list = list(self.values)
            
            stats = {
                "count": len(values_list),
                "latest": values_list[-1],
                "min": min(values_list),
                "max": max(values_list),
                "mean": statistics.mean(values_list),
                "median": statistics.median(values_list),
            }
            
            if len(values_list) > 1:
                stats["stdev"] = statistics.stdev(values_list)
                stats["p95"] = self._percentile(values_list, 95)
                stats["p99"] = self._percentile(values_list, 99)
            
            # Cache results
            self._stats_cache = stats
            self._cache_timestamp = current_time
            
            return stats.copy()
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _invalidate_cache(self):
        """Invalidate statistics cache"""
        self._stats_cache.clear()
        self._cache_timestamp = 0


class PerformanceValidator:
    """Performance validation and benchmarking system"""
    
    def __init__(self, benchmarks: List[PerformanceBenchmark]):
        self.benchmarks = {b.name: b for b in benchmarks}
        self.validation_results = {}
    
    async def validate_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate performance against benchmarks"""
        validation_results = {}
        
        for benchmark_name, benchmark in self.benchmarks.items():
            if benchmark_name in metrics:
                current_value = metrics[benchmark_name]
                
                # Determine performance level
                if benchmark.higher_is_better:
                    if current_value >= benchmark.target_value:
                        level = PerformanceLevel.EXCELLENT
                    elif current_value >= benchmark.warning_threshold:
                        level = PerformanceLevel.GOOD
                    elif current_value >= benchmark.critical_threshold:
                        level = PerformanceLevel.WARNING
                    else:
                        level = PerformanceLevel.CRITICAL
                else:
                    if current_value <= benchmark.target_value:
                        level = PerformanceLevel.EXCELLENT
                    elif current_value <= benchmark.warning_threshold:
                        level = PerformanceLevel.GOOD
                    elif current_value <= benchmark.critical_threshold:
                        level = PerformanceLevel.WARNING
                    else:
                        level = PerformanceLevel.CRITICAL
                
                validation_results[benchmark_name] = {
                    "current_value": current_value,
                    "target_value": benchmark.target_value,
                    "performance_level": level.value,
                    "within_target": level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD],
                    "benchmark": benchmark
                }
        
        return validation_results
    
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add a new performance benchmark"""
        self.benchmarks[benchmark.name] = benchmark
    
    def remove_benchmark(self, name: str):
        """Remove a performance benchmark"""
        self.benchmarks.pop(name, None)


class PerformanceMonitor:
    """
    Unified performance monitoring system consolidating all monitoring patterns:
    - System resource monitoring (CPU, memory, disk, network)
    - Application performance tracking (response times, throughput, errors)
    - Database and Redis performance monitoring
    - Orchestrator and task engine performance tracking
    - Real-time alerting and threshold monitoring
    - Performance benchmarking and validation
    - Historical trend analysis and reporting
    """
    
    _instance: Optional['PerformanceMonitor'] = None
    
    def __new__(cls) -> 'PerformanceMonitor':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_monitor()
            self._initialized = True
    
    def _initialize_monitor(self):
        """Initialize performance monitoring system"""
        try:
            self.config_service = ConfigurationService()
        except:
            self.config_service = type('MockConfig', (), {'config': {}})()
        
        try:
            self.redis = get_redis_service()
        except:
            self.redis = None
        
        try:
            self.circuit_breaker = CircuitBreakerService().get_circuit_breaker("performance_monitor")
        except:
            def dummy_breaker(func):
                return func
            self.circuit_breaker = dummy_breaker
        
        # Performance trackers
        self._trackers: Dict[str, PerformanceTracker] = {}
        
        # System monitoring
        self._system_snapshots: deque = deque(maxlen=1440)  # 24 hours of minute snapshots
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Performance validation
        self._validator = PerformanceValidator(self._get_default_benchmarks())
        
        # Alerting system
        self._alerts: deque = deque(maxlen=1000)
        self._alert_callbacks: List[Callable] = []
        
        # Performance baselines
        self._baselines: Dict[str, float] = {}
        
        # Thread pool for heavy computations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="perf-monitor")
        
        logger.info("Unified Performance Monitor initialized")
    
    def _get_default_benchmarks(self) -> List[PerformanceBenchmark]:
        """Get default performance benchmarks"""
        return [
            PerformanceBenchmark(
                name="api_response_time",
                target_value=200.0,
                warning_threshold=500.0,
                critical_threshold=1000.0,
                unit="ms",
                higher_is_better=False,
                description="API response time"
            ),
            PerformanceBenchmark(
                name="task_execution_time",
                target_value=60.0,
                warning_threshold=300.0,
                critical_threshold=600.0,
                unit="seconds",
                higher_is_better=False,
                description="Task execution time"
            ),
            PerformanceBenchmark(
                name="memory_usage",
                target_value=70.0,
                warning_threshold=80.0,
                critical_threshold=90.0,
                unit="percent",
                higher_is_better=False,
                description="Memory usage percentage"
            ),
            PerformanceBenchmark(
                name="cpu_usage",
                target_value=70.0,
                warning_threshold=80.0,
                critical_threshold=90.0,
                unit="percent",
                higher_is_better=False,
                description="CPU usage percentage"
            ),
            PerformanceBenchmark(
                name="error_rate",
                target_value=1.0,
                warning_threshold=5.0,
                critical_threshold=10.0,
                unit="percent",
                higher_is_better=False,
                description="Error rate percentage"
            ),
            PerformanceBenchmark(
                name="agent_spawn_time",
                target_value=10.0,
                warning_threshold=15.0,
                critical_threshold=30.0,
                unit="seconds",
                higher_is_better=False,
                description="Agent spawn time"
            ),
            PerformanceBenchmark(
                name="context_retrieval_time",
                target_value=0.05,
                warning_threshold=0.1,
                critical_threshold=0.5,
                unit="seconds",
                higher_is_better=False,
                description="Context retrieval time"
            )
        ]
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if not self._monitoring_active:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._monitoring_active = True
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring_active = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                snapshot = await self._collect_system_snapshot()
                self._system_snapshots.append(snapshot)
                
                # Evaluate performance against benchmarks
                await self._evaluate_performance(snapshot)
                
                # Check for alerts
                await self._check_alerts()
                
                # Persist metrics to Redis
                await self._persist_snapshot(snapshot)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(10)
    
    async def _collect_system_snapshot(self) -> PerformanceSnapshot:
        """Collect comprehensive system performance snapshot"""
        try:
            @self.circuit_breaker
            async def _collect():
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Network I/O
                network_io = psutil.net_io_counters()
                
                # Process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                
                snapshot = PerformanceSnapshot(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=process_memory.rss / 1024 / 1024,
                    disk_io_read_mb=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
                    disk_io_write_mb=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
                    network_sent_mb=network_io.bytes_sent / 1024 / 1024 if network_io else 0,
                    network_recv_mb=network_io.bytes_recv / 1024 / 1024 if network_io else 0,
                    active_connections=len(process.connections()) if hasattr(process, 'connections') else 0
                )
                
                return snapshot
            
            return await _collect()
            
        except Exception as e:
            logger.error("System snapshot collection failed", error=str(e))
            return PerformanceSnapshot()
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None):
        """Record performance metric"""
        if name not in self._trackers:
            self._trackers[name] = PerformanceTracker(name)
        
        self._trackers[name].record(value)
        
        # Update baselines if this is a new metric
        if name not in self._baselines:
            self._baselines[name] = value
        
        logger.debug("Metric recorded", name=name, value=value, type=metric_type.value)
    
    def record_timing(self, operation_name: str, duration_ms: float):
        """Record operation timing"""
        self.record_metric(f"{operation_name}_duration", duration_ms, MetricType.TIMER)
    
    def record_counter(self, counter_name: str, increment: int = 1):
        """Record counter increment"""
        current_value = self.get_latest_metric(counter_name) or 0
        self.record_metric(counter_name, current_value + increment, MetricType.COUNTER)
    
    def get_metric_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistical summary for metric"""
        if name in self._trackers:
            return self._trackers[name].get_statistics()
        return None
    
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get latest metric value"""
        stats = self.get_metric_statistics(name)
        return stats.get("latest") if stats else None
    
    async def validate_performance(self) -> Dict[str, Any]:
        """Validate system performance against benchmarks"""
        # Collect current metrics
        current_metrics = {}
        for name, tracker in self._trackers.items():
            stats = tracker.get_statistics()
            if stats and "latest" in stats:
                current_metrics[name] = stats["latest"]
        
        # Add system metrics
        if self._system_snapshots:
            latest_snapshot = self._system_snapshots[-1]
            current_metrics.update({
                "cpu_usage": latest_snapshot.cpu_percent,
                "memory_usage": latest_snapshot.memory_percent,
                "response_time": latest_snapshot.response_time_ms,
                "error_rate": latest_snapshot.error_rate
            })
        
        # Validate against benchmarks
        return await self._validator.validate_performance(current_metrics)
    
    async def _evaluate_performance(self, snapshot: PerformanceSnapshot):
        """Evaluate performance and generate alerts if needed"""
        # Check system metrics against thresholds
        alerts = []
        
        if snapshot.cpu_percent > 90:
            alerts.append(PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                metric_name="cpu_usage",
                current_value=snapshot.cpu_percent,
                threshold_value=90.0,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical CPU usage: {snapshot.cpu_percent:.1f}%"
            ))
        elif snapshot.cpu_percent > 80:
            alerts.append(PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                metric_name="cpu_usage",
                current_value=snapshot.cpu_percent,
                threshold_value=80.0,
                severity=AlertSeverity.HIGH,
                message=f"High CPU usage: {snapshot.cpu_percent:.1f}%"
            ))
        
        if snapshot.memory_percent > 90:
            alerts.append(PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                metric_name="memory_usage",
                current_value=snapshot.memory_percent,
                threshold_value=90.0,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical memory usage: {snapshot.memory_percent:.1f}%"
            ))
        elif snapshot.memory_percent > 80:
            alerts.append(PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                metric_name="memory_usage",
                current_value=snapshot.memory_percent,
                threshold_value=80.0,
                severity=AlertSeverity.HIGH,
                message=f"High memory usage: {snapshot.memory_percent:.1f}%"
            ))
        
        # Store alerts
        for alert in alerts:
            self._alerts.append(alert)
            await self._trigger_alert(alert)
    
    async def _check_alerts(self):
        """Check all metrics for alert conditions"""
        current_time = datetime.utcnow()
        
        # Check tracked metrics
        for name, tracker in self._trackers.items():
            stats = tracker.get_statistics()
            if not stats or "latest" not in stats:
                continue
            
            latest_value = stats["latest"]
            
            # Check if metric has associated benchmark
            if name in self._validator.benchmarks:
                benchmark = self._validator.benchmarks[name]
                
                alert_needed = False
                severity = AlertSeverity.LOW
                
                if not benchmark.higher_is_better:
                    if latest_value > benchmark.critical_threshold:
                        alert_needed = True
                        severity = AlertSeverity.CRITICAL
                    elif latest_value > benchmark.warning_threshold:
                        alert_needed = True
                        severity = AlertSeverity.HIGH
                else:
                    if latest_value < benchmark.critical_threshold:
                        alert_needed = True
                        severity = AlertSeverity.CRITICAL
                    elif latest_value < benchmark.warning_threshold:
                        alert_needed = True
                        severity = AlertSeverity.HIGH
                
                if alert_needed:
                    alert = PerformanceAlert(
                        alert_id=str(uuid.uuid4()),
                        metric_name=name,
                        current_value=latest_value,
                        threshold_value=benchmark.warning_threshold if severity == AlertSeverity.HIGH else benchmark.critical_threshold,
                        severity=severity,
                        message=f"Performance threshold exceeded for {name}: {latest_value:.2f} {benchmark.unit}"
                    )
                    
                    # Check if this is a new alert (not duplicate)
                    if not self._is_duplicate_alert(alert):
                        self._alerts.append(alert)
                        await self._trigger_alert(alert)
    
    def _is_duplicate_alert(self, alert: PerformanceAlert) -> bool:
        """Check if this alert is a duplicate of recent alerts"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        
        recent_alerts = [a for a in self._alerts if a.timestamp >= cutoff_time and not a.resolved]
        
        for existing_alert in recent_alerts:
            if (existing_alert.metric_name == alert.metric_name and
                existing_alert.severity == alert.severity):
                return True
        
        return False
    
    async def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger alert callbacks"""
        logger.warning("Performance alert triggered",
                      metric=alert.metric_name,
                      severity=alert.severity.value,
                      message=alert.message)
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self._alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remove alert callback function"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)
    
    async def _persist_snapshot(self, snapshot: PerformanceSnapshot):
        """Persist snapshot to Redis and database"""
        try:
            # Store to Redis for real-time access
            if self.redis:
                snapshot_data = {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "cpu_percent": snapshot.cpu_percent,
                    "memory_percent": snapshot.memory_percent,
                    "memory_used_mb": snapshot.memory_used_mb,
                    "response_time_ms": snapshot.response_time_ms,
                    "error_rate": snapshot.error_rate
                }
                
                await self.redis.hset(
                    "performance:latest", 
                    mapping={k: json.dumps(v) for k, v in snapshot_data.items()}
                )
                await self.redis.expire("performance:latest", 3600)
            
            # Store to database every 5 minutes
            if len(self._system_snapshots) % 5 == 0:
                async with get_session() as session:
                    metric = PerformanceMetric(
                        metric_name="system_snapshot",
                        metric_value=snapshot.cpu_percent,  # Use CPU as primary metric
                        tags={
                            "memory_percent": str(snapshot.memory_percent),
                            "memory_used_mb": str(snapshot.memory_used_mb),
                            "error_rate": str(snapshot.error_rate)
                        }
                    )
                    session.add(metric)
                    await session.commit()
        
        except Exception as e:
            logger.error("Failed to persist snapshot", error=str(e))
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        if not self._system_snapshots:
            return {"status": "no_data"}
        
        latest_snapshot = self._system_snapshots[-1]
        
        # Calculate health score based on latest metrics
        health_factors = []
        
        if latest_snapshot.cpu_percent < 80:
            health_factors.append(("cpu", 1.0))
        elif latest_snapshot.cpu_percent < 90:
            health_factors.append(("cpu", 0.7))
        else:
            health_factors.append(("cpu", 0.3))
        
        if latest_snapshot.memory_percent < 80:
            health_factors.append(("memory", 1.0))
        elif latest_snapshot.memory_percent < 90:
            health_factors.append(("memory", 0.7))
        else:
            health_factors.append(("memory", 0.3))
        
        # Factor in recent alerts
        recent_alerts = [a for a in self._alerts if not a.resolved and 
                        (datetime.utcnow() - a.timestamp).seconds < 3600]
        
        alert_factor = 1.0
        if recent_alerts:
            critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
            high_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.HIGH]
            
            if critical_alerts:
                alert_factor = 0.3
            elif high_alerts:
                alert_factor = 0.6
            else:
                alert_factor = 0.8
        
        health_factors.append(("alerts", alert_factor))
        
        overall_health = sum(score for _, score in health_factors) / len(health_factors)
        
        if overall_health >= 0.8:
            status = "healthy"
        elif overall_health >= 0.6:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": overall_health,
            "latest_snapshot": {
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "cpu_percent": latest_snapshot.cpu_percent,
                "memory_percent": latest_snapshot.memory_percent,
                "memory_used_mb": latest_snapshot.memory_used_mb,
                "response_time_ms": latest_snapshot.response_time_ms,
                "error_rate": latest_snapshot.error_rate
            },
            "active_trackers": len(self._trackers),
            "snapshots_collected": len(self._system_snapshots),
            "recent_alerts": len(recent_alerts),
            "health_factors": dict(health_factors)
        }
    
    def get_performance_recommendations(self) -> List[str]:
        """Get AI-powered performance optimization recommendations"""
        recommendations = []
        
        if not self._system_snapshots:
            return ["Insufficient data - monitor system for at least 5 minutes"]
        
        latest_snapshot = self._system_snapshots[-1]
        
        # CPU recommendations
        if latest_snapshot.cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider scaling horizontally or optimizing CPU-intensive tasks")
        
        # Memory recommendations
        if latest_snapshot.memory_percent > 80:
            recommendations.append("High memory usage detected - implement memory cleanup routines or add more memory")
        
        # Response time recommendations
        if latest_snapshot.response_time_ms > 1000:
            recommendations.append("High response times detected - optimize database queries and add caching")
        
        # Error rate recommendations
        if latest_snapshot.error_rate > 5:
            recommendations.append("High error rate detected - review error logs and implement better error handling")
        
        # Alert-based recommendations
        recent_alerts = [a for a in self._alerts if not a.resolved and 
                        (datetime.utcnow() - a.timestamp).seconds < 3600]
        
        if len(recent_alerts) > 5:
            recommendations.append("Multiple performance alerts active - investigate system health immediately")
        
        # Baseline comparison recommendations
        for name, tracker in self._trackers.items():
            stats = tracker.get_statistics()
            if stats and name in self._baselines:
                current = stats.get("latest", 0)
                baseline = self._baselines[name]
                
                if current > baseline * 1.5:  # 50% degradation
                    recommendations.append(f"Performance degradation in {name} - current value {current:.2f} vs baseline {baseline:.2f}")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        benchmark_start = time.time()
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmarks": {},
            "system_performance": {},
            "recommendations": []
        }
        
        try:
            # Test basic system operations
            await self._benchmark_cpu_performance(results)
            await self._benchmark_memory_performance(results)
            await self._benchmark_disk_performance(results)
            await self._benchmark_network_performance(results)
            
            # Test application-specific operations
            await self._benchmark_database_performance(results)
            await self._benchmark_redis_performance(results)
            
            # Calculate overall benchmark score
            benchmark_scores = []
            for benchmark_name, benchmark_data in results["benchmarks"].items():
                if "score" in benchmark_data:
                    benchmark_scores.append(benchmark_data["score"])
            
            if benchmark_scores:
                results["overall_score"] = statistics.mean(benchmark_scores)
                results["overall_grade"] = self._calculate_grade(results["overall_score"])
            
            benchmark_duration = time.time() - benchmark_start
            results["benchmark_duration_seconds"] = benchmark_duration
            
            # Generate recommendations based on benchmark results
            results["recommendations"] = self._generate_benchmark_recommendations(results)
            
            logger.info("Performance benchmark completed", 
                       duration=benchmark_duration,
                       overall_score=results.get("overall_score", 0))
            
        except Exception as e:
            logger.error("Performance benchmark failed", error=str(e))
            results["error"] = str(e)
        
        return results
    
    async def _benchmark_cpu_performance(self, results: Dict[str, Any]):
        """Benchmark CPU performance"""
        start_time = time.time()
        
        # CPU-intensive operation
        total = 0
        for i in range(1000000):
            total += i * i
        
        duration = time.time() - start_time
        
        # Score based on duration (lower is better)
        score = max(0, 100 - (duration * 100))  # Normalize to 0-100 scale
        
        results["benchmarks"]["cpu_performance"] = {
            "duration_seconds": duration,
            "operations": 1000000,
            "ops_per_second": 1000000 / duration,
            "score": score
        }
    
    async def _benchmark_memory_performance(self, results: Dict[str, Any]):
        """Benchmark memory performance"""
        start_time = time.time()
        
        # Memory allocation test
        data = []
        for i in range(100000):
            data.append({"id": i, "data": f"test_data_{i}"})
        
        duration = time.time() - start_time
        
        # Measure memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Score based on speed and memory efficiency
        speed_score = max(0, 100 - (duration * 50))
        memory_score = max(0, 100 - (memory_mb / 10))
        score = (speed_score + memory_score) / 2
        
        results["benchmarks"]["memory_performance"] = {
            "allocation_duration_seconds": duration,
            "memory_used_mb": memory_mb,
            "allocations": 100000,
            "allocations_per_second": 100000 / duration,
            "score": score
        }
        
        # Cleanup
        del data
    
    async def _benchmark_disk_performance(self, results: Dict[str, Any]):
        """Benchmark disk I/O performance"""
        import tempfile
        import os
        
        start_time = time.time()
        
        # Write test
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_data = b"x" * 1024 * 1024  # 1MB
            for _ in range(10):
                tmp_file.write(test_data)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            
            write_duration = time.time() - start_time
            
            # Read test
            read_start = time.time()
            tmp_file.seek(0)
            while tmp_file.read(1024 * 1024):
                pass
            read_duration = time.time() - read_start
            
            # Cleanup
            os.unlink(tmp_file.name)
        
        # Calculate throughput and score
        data_size_mb = 10
        write_throughput = data_size_mb / write_duration
        read_throughput = data_size_mb / read_duration
        
        # Score based on throughput (higher is better)
        write_score = min(100, write_throughput * 10)
        read_score = min(100, read_throughput * 10)
        score = (write_score + read_score) / 2
        
        results["benchmarks"]["disk_performance"] = {
            "write_duration_seconds": write_duration,
            "read_duration_seconds": read_duration,
            "write_throughput_mbps": write_throughput,
            "read_throughput_mbps": read_throughput,
            "data_size_mb": data_size_mb,
            "score": score
        }
    
    async def _benchmark_network_performance(self, results: Dict[str, Any]):
        """Benchmark network performance (simulated)"""
        # Simulate network operations since we can't make real external calls
        start_time = time.time()
        
        # Simulate network latency
        await asyncio.sleep(0.01)  # 10ms simulated latency
        
        duration = time.time() - start_time
        
        # Simulate network stats
        simulated_latency_ms = 10
        simulated_throughput_mbps = 100
        
        # Score based on latency (lower is better) and throughput (higher is better)
        latency_score = max(0, 100 - simulated_latency_ms)
        throughput_score = min(100, simulated_throughput_mbps)
        score = (latency_score + throughput_score) / 2
        
        results["benchmarks"]["network_performance"] = {
            "simulated_latency_ms": simulated_latency_ms,
            "simulated_throughput_mbps": simulated_throughput_mbps,
            "test_duration_seconds": duration,
            "score": score
        }
    
    async def _benchmark_database_performance(self, results: Dict[str, Any]):
        """Benchmark database performance"""
        start_time = time.time()
        
        try:
            async with get_session() as session:
                # Simple query test
                query = select(PerformanceMetric).limit(10)
                result = await session.execute(query)
                metrics = result.scalars().all()
                
                query_duration = time.time() - start_time
                
                # Score based on query speed
                score = max(0, 100 - (query_duration * 1000))  # Penalize slow queries
                
                results["benchmarks"]["database_performance"] = {
                    "query_duration_seconds": query_duration,
                    "records_retrieved": len(metrics),
                    "queries_per_second": 1 / query_duration if query_duration > 0 else 0,
                    "score": score
                }
        
        except Exception as e:
            results["benchmarks"]["database_performance"] = {
                "error": str(e),
                "score": 0
            }
    
    async def _benchmark_redis_performance(self, results: Dict[str, Any]):
        """Benchmark Redis performance"""
        start_time = time.time()
        
        try:
            if self.redis:
                # Set operation
                await self.redis.set("benchmark_test", "test_value")
                
                # Get operation  
                value = await self.redis.get("benchmark_test")
                
                operation_duration = time.time() - start_time
                
                # Cleanup
                await self.redis.delete("benchmark_test")
                
                # Score based on operation speed
                score = max(0, 100 - (operation_duration * 10000))  # High penalty for slow Redis
                
                results["benchmarks"]["redis_performance"] = {
                    "operation_duration_seconds": operation_duration,
                    "operations_per_second": 2 / operation_duration if operation_duration > 0 else 0,
                    "score": score
                }
            else:
                results["benchmarks"]["redis_performance"] = {
                    "error": "Redis not available",
                    "score": 0
                }
        
        except Exception as e:
            results["benchmarks"]["redis_performance"] = {
                "error": str(e),
                "score": 0
            }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from numeric score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_benchmark_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        for benchmark_name, benchmark_data in results["benchmarks"].items():
            score = benchmark_data.get("score", 0)
            
            if score < 60:
                if benchmark_name == "cpu_performance":
                    recommendations.append("CPU performance is below acceptable levels - consider upgrading hardware or optimizing algorithms")
                elif benchmark_name == "memory_performance":
                    recommendations.append("Memory performance is suboptimal - implement better memory management or add more RAM")
                elif benchmark_name == "disk_performance":
                    recommendations.append("Disk I/O performance is slow - consider SSD upgrade or I/O optimization")
                elif benchmark_name == "database_performance":
                    recommendations.append("Database performance is poor - optimize queries, add indexes, or scale database")
                elif benchmark_name == "redis_performance":
                    recommendations.append("Redis performance is suboptimal - check Redis configuration and network latency")
        
        overall_score = results.get("overall_score", 0)
        if overall_score >= 90:
            recommendations.append("Excellent performance across all benchmarks - system is well optimized")
        elif overall_score >= 70:
            recommendations.append("Good overall performance with some areas for improvement")
        else:
            recommendations.append("Performance issues detected - prioritize optimization efforts")
        
        return recommendations


# Performance monitoring decorators
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_timing(name, duration_ms)
                return result
            except Exception as e:
                monitor.record_counter(f"{name}_errors")
                raise
        
        async def async_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_timing(name, duration_ms)
                return result
            except Exception as e:
                monitor.record_counter(f"{name}_errors")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Convenience functions
def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance"""
    return PerformanceMonitor()


def record_api_response_time(endpoint: str, duration_ms: float):
    """Record API response time"""
    monitor = get_performance_monitor()
    monitor.record_timing(f"api_{endpoint}", duration_ms)
    monitor.record_metric("api_response_time", duration_ms)


def record_task_execution_time(task_type: str, duration_seconds: float):
    """Record task execution time"""
    monitor = get_performance_monitor()
    monitor.record_timing(f"task_{task_type}", duration_seconds * 1000)
    monitor.record_metric("task_execution_time", duration_seconds)


def record_agent_spawn_time(duration_seconds: float):
    """Record agent spawn time"""
    monitor = get_performance_monitor()
    monitor.record_metric("agent_spawn_time", duration_seconds)


def record_context_retrieval_time(duration_seconds: float):
    """Record context retrieval time"""
    monitor = get_performance_monitor()
    monitor.record_metric("context_retrieval_time", duration_seconds)


def record_orchestrator_metrics(agent_count: int, task_queue_size: int, load_factor: float):
    """Record orchestrator performance metrics"""
    monitor = get_performance_monitor()
    monitor.record_metric("orchestrator_active_agents", agent_count)
    monitor.record_metric("orchestrator_task_queue_size", task_queue_size)
    monitor.record_metric("orchestrator_load_factor", load_factor)


def record_task_engine_metrics(tasks_executed: int, avg_execution_time: float, active_executions: int):
    """Record task engine performance metrics"""
    monitor = get_performance_monitor()
    monitor.record_metric("task_engine_tasks_executed", tasks_executed)
    monitor.record_metric("task_engine_avg_execution_time", avg_execution_time)
    monitor.record_metric("task_engine_active_executions", active_executions)