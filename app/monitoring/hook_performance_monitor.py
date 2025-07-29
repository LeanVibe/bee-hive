"""
Hook Performance Monitoring System for LeanVibe Agent Hive 2.0 - VS 6.1

Comprehensive performance monitoring for observability hooks with <5ms targets:
- Real-time performance metrics collection and analysis
- Hook execution time tracking and alerting
- System impact assessment and optimization recommendations
- Performance regression detection and reporting
- Automatic sampling adjustment based on performance
- Integration with Prometheus metrics and alerting
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import structlog

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY

logger = structlog.get_logger()


class PerformanceStatus(Enum):
    """Performance status levels."""
    EXCELLENT = "excellent"    # < 1ms average
    GOOD = "good"             # 1-3ms average
    ACCEPTABLE = "acceptable"  # 3-5ms average
    DEGRADED = "degraded"     # 5-10ms average
    POOR = "poor"             # > 10ms average


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetrics:
    """Performance metrics for hook operations."""
    total_operations: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    p50_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    
    # Time windows
    last_minute_ops: int = 0
    last_minute_avg_ms: float = 0.0
    last_hour_ops: int = 0
    last_hour_avg_ms: float = 0.0
    
    # Performance status
    status: PerformanceStatus = PerformanceStatus.GOOD
    
    # Regression detection
    is_regressing: bool = False
    regression_factor: float = 1.0
    
    def update_status(self):
        """Update performance status based on current metrics."""
        if self.avg_time_ms < 1.0:
            self.status = PerformanceStatus.EXCELLENT
        elif self.avg_time_ms < 3.0:
            self.status = PerformanceStatus.GOOD
        elif self.avg_time_ms < 5.0:
            self.status = PerformanceStatus.ACCEPTABLE
        elif self.avg_time_ms < 10.0:
            self.status = PerformanceStatus.DEGRADED
        else:
            self.status = PerformanceStatus.POOR


class PerformanceAlert(NamedTuple):
    """Performance alert information."""
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    hook_category: Optional[str] = None


@dataclass
class HookOperationRecord:
    """Record of a single hook operation."""
    timestamp: datetime
    hook_name: str
    hook_category: str
    execution_time_ms: float
    success: bool
    error: Optional[str] = None
    
    @property
    def age_seconds(self) -> float:
        """Age of this record in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()


class HookPerformanceMonitor:
    """
    Advanced performance monitoring for observability hooks.
    
    Tracks execution times, detects regressions, and provides
    optimization recommendations while maintaining <5ms targets.
    """
    
    def __init__(
        self,
        target_p95_ms: float = 5.0,
        target_avg_ms: float = 2.0,
        alert_threshold_multiplier: float = 1.5,
        regression_detection_window: int = 1000,
        metrics_retention_hours: int = 24
    ):
        """Initialize hook performance monitor."""
        self.target_p95_ms = target_p95_ms
        self.target_avg_ms = target_avg_ms
        self.alert_threshold_multiplier = alert_threshold_multiplier
        self.regression_window = regression_detection_window
        self.retention_hours = metrics_retention_hours
        
        # Performance tracking
        self._operation_records: List[HookOperationRecord] = []
        self._metrics_by_category: Dict[str, PerformanceMetrics] = {}
        self._metrics_by_hook: Dict[str, PerformanceMetrics] = {}
        self._global_metrics = PerformanceMetrics()
        
        # Alert management
        self._active_alerts: List[PerformanceAlert] = []
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info(
            "📊 HookPerformanceMonitor initialized",
            target_p95_ms=target_p95_ms,
            target_avg_ms=target_avg_ms,
            retention_hours=metrics_retention_hours
        )
    
    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics for hook performance."""
        self.hook_execution_time = Histogram(
            'hook_execution_duration_seconds',
            'Time spent executing observability hooks',
            ['hook_name', 'hook_category', 'status'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        )
        
        self.hook_operations_total = Counter(
            'hook_operations_total',
            'Total number of hook operations',
            ['hook_name', 'hook_category', 'status']
        )
        
        self.hook_performance_status = Gauge(
            'hook_performance_status',
            'Current performance status (0=excellent, 1=good, 2=acceptable, 3=degraded, 4=poor)',
            ['hook_category']
        )
        
        self.hook_p95_latency = Gauge(
            'hook_p95_latency_seconds',
            'P95 latency for hook operations',
            ['hook_category']
        )
        
        self.hook_regression_factor = Gauge(
            'hook_regression_factor',
            'Performance regression factor (1.0 = no regression)',
            ['hook_category']
        )
    
    def record_hook_execution(
        self,
        hook_name: str,
        hook_category: str,
        execution_time_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Record a hook execution for performance analysis.
        
        Args:
            hook_name: Name of the hook that was executed
            hook_category: Category of the hook (workflow, agent, tool, etc.)
            execution_time_ms: Execution time in milliseconds
            success: Whether the hook execution succeeded
            error: Error message if execution failed
        """
        # Create operation record
        record = HookOperationRecord(
            timestamp=datetime.utcnow(),
            hook_name=hook_name,
            hook_category=hook_category,
            execution_time_ms=execution_time_ms,
            success=success,
            error=error
        )
        
        # Store record
        self._operation_records.append(record)
        
        # Update Prometheus metrics
        status = "success" if success else "error"
        self.hook_execution_time.labels(
            hook_name=hook_name,
            hook_category=hook_category,
            status=status
        ).observe(execution_time_ms / 1000.0)  # Convert to seconds
        
        self.hook_operations_total.labels(
            hook_name=hook_name,
            hook_category=hook_category,
            status=status
        ).inc()
        
        # Update internal metrics (asynchronously to avoid blocking)
        asyncio.create_task(self._update_metrics_async())
        
        # Check for immediate alerts
        if execution_time_ms > self.target_p95_ms * self.alert_threshold_multiplier:
            alert = PerformanceAlert(
                severity=AlertSeverity.WARNING,
                message=f"Hook {hook_name} exceeded target time: {execution_time_ms:.2f}ms > {self.target_p95_ms * self.alert_threshold_multiplier:.2f}ms",
                metric_name="execution_time",
                current_value=execution_time_ms,
                threshold=self.target_p95_ms * self.alert_threshold_multiplier,
                timestamp=datetime.utcnow(),
                hook_category=hook_category
            )
            asyncio.create_task(self._handle_alert_async(alert))
    
    async def _update_metrics_async(self) -> None:
        """Update performance metrics asynchronously."""
        try:
            # Clean up old records first
            await self._cleanup_old_records()
            
            # Update global metrics
            self._update_global_metrics()
            
            # Update category metrics
            self._update_category_metrics()
            
            # Update hook-specific metrics
            self._update_hook_metrics()
            
            # Update Prometheus gauges
            self._update_prometheus_gauges()
            
        except Exception as e:
            logger.error("❌ Failed to update performance metrics", error=str(e))
    
    def _update_global_metrics(self) -> None:
        """Update global performance metrics."""
        if not self._operation_records:
            return
        
        execution_times = [r.execution_time_ms for r in self._operation_records if r.success]
        
        if execution_times:
            self._global_metrics.total_operations = len(self._operation_records)
            self._global_metrics.total_time_ms = sum(execution_times)
            self._global_metrics.min_time_ms = min(execution_times)
            self._global_metrics.max_time_ms = max(execution_times)
            self._global_metrics.avg_time_ms = statistics.mean(execution_times)
            
            # Percentiles
            sorted_times = sorted(execution_times)
            n = len(sorted_times)
            self._global_metrics.p50_time_ms = sorted_times[n // 2] if n > 0 else 0
            self._global_metrics.p95_time_ms = sorted_times[int(0.95 * n)] if n > 0 else 0
            self._global_metrics.p99_time_ms = sorted_times[int(0.99 * n)] if n > 0 else 0
            
            # Time window metrics
            now = datetime.utcnow()
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            recent_minute = [r for r in self._operation_records if r.timestamp >= minute_ago and r.success]
            recent_hour = [r for r in self._operation_records if r.timestamp >= hour_ago and r.success]
            
            self._global_metrics.last_minute_ops = len(recent_minute)
            self._global_metrics.last_minute_avg_ms = (
                statistics.mean([r.execution_time_ms for r in recent_minute])
                if recent_minute else 0
            )
            
            self._global_metrics.last_hour_ops = len(recent_hour)
            self._global_metrics.last_hour_avg_ms = (
                statistics.mean([r.execution_time_ms for r in recent_hour])
                if recent_hour else 0
            )
            
            # Update status
            self._global_metrics.update_status()
            
            # Regression detection
            self._detect_regression(self._global_metrics, execution_times)
    
    def _update_category_metrics(self) -> None:
        """Update performance metrics by hook category."""
        categories = set(r.hook_category for r in self._operation_records)
        
        for category in categories:
            category_records = [r for r in self._operation_records if r.hook_category == category]
            execution_times = [r.execution_time_ms for r in category_records if r.success]
            
            if not execution_times:
                continue
            
            metrics = PerformanceMetrics()
            metrics.total_operations = len(category_records)
            metrics.total_time_ms = sum(execution_times)
            metrics.min_time_ms = min(execution_times)
            metrics.max_time_ms = max(execution_times)
            metrics.avg_time_ms = statistics.mean(execution_times)
            
            # Percentiles
            sorted_times = sorted(execution_times)
            n = len(sorted_times)
            metrics.p50_time_ms = sorted_times[n // 2] if n > 0 else 0
            metrics.p95_time_ms = sorted_times[int(0.95 * n)] if n > 0 else 0
            metrics.p99_time_ms = sorted_times[int(0.99 * n)] if n > 0 else 0
            
            # Update status
            metrics.update_status()
            
            # Regression detection
            self._detect_regression(metrics, execution_times)
            
            self._metrics_by_category[category] = metrics
    
    def _update_hook_metrics(self) -> None:
        """Update performance metrics by individual hook."""
        hooks = set(r.hook_name for r in self._operation_records)
        
        for hook_name in hooks:
            hook_records = [r for r in self._operation_records if r.hook_name == hook_name]
            execution_times = [r.execution_time_ms for r in hook_records if r.success]
            
            if not execution_times:
                continue
            
            metrics = PerformanceMetrics()
            metrics.total_operations = len(hook_records)
            metrics.avg_time_ms = statistics.mean(execution_times)
            metrics.p95_time_ms = sorted(execution_times)[int(0.95 * len(execution_times))] if execution_times else 0
            
            # Update status
            metrics.update_status()
            
            self._metrics_by_hook[hook_name] = metrics
    
    def _detect_regression(self, metrics: PerformanceMetrics, execution_times: List[float]) -> None:
        """Detect performance regression."""
        if len(execution_times) < self.regression_window:
            return
        
        # Compare recent performance to historical baseline
        recent_window = execution_times[-self.regression_window // 2:]
        historical_window = execution_times[-self.regression_window:-self.regression_window // 2]
        
        if historical_window and recent_window:
            recent_avg = statistics.mean(recent_window)
            historical_avg = statistics.mean(historical_window)
            
            regression_factor = recent_avg / historical_avg if historical_avg > 0 else 1.0
            metrics.regression_factor = regression_factor
            
            # Consider it a regression if recent performance is 20% worse
            metrics.is_regressing = regression_factor > 1.2
            
            if metrics.is_regressing:
                alert = PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"Performance regression detected: {regression_factor:.2f}x slower than baseline",
                    metric_name="regression_factor",
                    current_value=regression_factor,
                    threshold=1.2,
                    timestamp=datetime.utcnow()
                )
                asyncio.create_task(self._handle_alert_async(alert))
    
    def _update_prometheus_gauges(self) -> None:
        """Update Prometheus gauge metrics."""
        # Update global performance status
        status_value = {
            PerformanceStatus.EXCELLENT: 0,
            PerformanceStatus.GOOD: 1,
            PerformanceStatus.ACCEPTABLE: 2,
            PerformanceStatus.DEGRADED: 3,
            PerformanceStatus.POOR: 4
        }
        
        # Update category metrics
        for category, metrics in self._metrics_by_category.items():
            self.hook_performance_status.labels(hook_category=category).set(
                status_value[metrics.status]
            )
            self.hook_p95_latency.labels(hook_category=category).set(
                metrics.p95_time_ms / 1000.0  # Convert to seconds
            )
            self.hook_regression_factor.labels(hook_category=category).set(
                metrics.regression_factor
            )
    
    async def _cleanup_old_records(self) -> None:
        """Clean up old operation records to manage memory."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        before_count = len(self._operation_records)
        self._operation_records = [
            r for r in self._operation_records 
            if r.timestamp >= cutoff_time
        ]
        after_count = len(self._operation_records)
        
        if before_count != after_count:
            logger.debug(
                "🗑️ Cleaned up old performance records",
                removed=before_count - after_count,
                remaining=after_count
            )
    
    async def _handle_alert_async(self, alert: PerformanceAlert) -> None:
        """Handle performance alert asynchronously."""
        self._active_alerts.append(alert)
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("❌ Alert callback failed", error=str(e))
        
        logger.warning(
            f"⚠️ Performance Alert: {alert.message}",
            severity=alert.severity.value,
            metric=alert.metric_name,
            value=alert.current_value,
            threshold=alert.threshold
        )
    
    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Register a callback for performance alerts."""
        self._alert_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self._is_monitoring = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Start analysis task
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("🔍 Started performance monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Stop tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Stopped performance monitoring")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        try:
            while self._is_monitoring:
                await self._cleanup_old_records()
                await asyncio.sleep(300)  # Clean up every 5 minutes
        except asyncio.CancelledError:
            pass
    
    async def _analysis_loop(self) -> None:
        """Background analysis loop."""
        try:
            while self._is_monitoring:
                await self._update_metrics_async()
                await asyncio.sleep(10)  # Update metrics every 10 seconds
        except asyncio.CancelledError:
            pass
    
    def get_global_metrics(self) -> PerformanceMetrics:
        """Get global performance metrics."""
        return self._global_metrics
    
    def get_category_metrics(self, category: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific category."""
        return self._metrics_by_category.get(category)
    
    def get_hook_metrics(self, hook_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific hook."""
        return self._metrics_by_hook.get(hook_name)
    
    def get_all_category_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for all categories."""
        return self._metrics_by_category.copy()
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active performance alerts."""
        # Remove old alerts (older than 1 hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        self._active_alerts = [
            alert for alert in self._active_alerts
            if alert.timestamp >= cutoff_time
        ]
        
        return self._active_alerts.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "global_metrics": {
                "status": self._global_metrics.status.value,
                "avg_time_ms": self._global_metrics.avg_time_ms,
                "p95_time_ms": self._global_metrics.p95_time_ms,
                "p99_time_ms": self._global_metrics.p99_time_ms,
                "total_operations": self._global_metrics.total_operations,
                "is_regressing": self._global_metrics.is_regressing,
                "regression_factor": self._global_metrics.regression_factor,
                "within_target": self._global_metrics.p95_time_ms <= self.target_p95_ms
            },
            "category_metrics": {
                category: {
                    "status": metrics.status.value,
                    "avg_time_ms": metrics.avg_time_ms,
                    "p95_time_ms": metrics.p95_time_ms,
                    "total_operations": metrics.total_operations,
                    "is_regressing": metrics.is_regressing
                }
                for category, metrics in self._metrics_by_category.items()
            },
            "active_alerts": [
                {
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric": alert.metric_name,
                    "value": alert.current_value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                    "category": alert.hook_category
                }
                for alert in self.get_active_alerts()
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check global performance
        if self._global_metrics.status == PerformanceStatus.POOR:
            recommendations.append("Overall hook performance is poor - consider reducing hook complexity or increasing sampling")
        elif self._global_metrics.status == PerformanceStatus.DEGRADED:
            recommendations.append("Hook performance is degraded - monitor for potential issues")
        
        # Check regression
        if self._global_metrics.is_regressing:
            recommendations.append(f"Performance regression detected ({self._global_metrics.regression_factor:.2f}x) - investigate recent changes")
        
        # Check category-specific issues
        for category, metrics in self._metrics_by_category.items():
            if metrics.p95_time_ms > self.target_p95_ms * 2:
                recommendations.append(f"Category '{category}' has very high latency - consider optimization")
            elif metrics.is_regressing:
                recommendations.append(f"Category '{category}' showing performance regression")
        
        # Check operation volume
        if self._global_metrics.last_minute_ops > 1000:
            recommendations.append("High hook operation volume - consider increasing sampling rate")
        
        return recommendations
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics in text format."""
        return generate_latest(REGISTRY)


# Global performance monitor instance
_performance_monitor: Optional[HookPerformanceMonitor] = None


def get_performance_monitor() -> Optional[HookPerformanceMonitor]:
    """Get the global performance monitor instance."""
    return _performance_monitor


def initialize_performance_monitor(**kwargs) -> HookPerformanceMonitor:
    """Initialize and set the global performance monitor."""
    global _performance_monitor
    
    _performance_monitor = HookPerformanceMonitor(**kwargs)
    
    logger.info("✅ Global hook performance monitor initialized")
    return _performance_monitor


async def start_performance_monitoring() -> None:
    """Start performance monitoring if initialized."""
    if _performance_monitor:
        await _performance_monitor.start_monitoring()


async def stop_performance_monitoring() -> None:
    """Stop performance monitoring if running."""
    if _performance_monitor:
        await _performance_monitor.stop_monitoring()