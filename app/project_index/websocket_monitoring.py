"""
WebSocket Monitoring and Performance Metrics for Project Index Events

This module provides comprehensive monitoring, alerting, and performance metrics
for the project index WebSocket system, including Prometheus integration.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry

from .websocket_events import ProjectIndexEventType
from .websocket_integration import ProjectIndexWebSocketManager

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricAlert:
    """Alert configuration for metrics."""
    metric_name: str
    threshold: float
    operator: str  # "gt", "lt", "eq", "gte", "lte"
    severity: AlertSeverity
    message: str
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    acceptable_variance: float  # Percentage
    measurement_window_minutes: int = 60
    samples: deque = field(default_factory=lambda: deque(maxlen=1000))


class PrometheusMetrics:
    """Prometheus metrics for WebSocket monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Event metrics
        self.events_published_total = Counter(
            'project_index_websocket_events_published_total',
            'Total number of WebSocket events published',
            ['event_type', 'project_id'],
            registry=self.registry
        )
        
        self.events_delivered_total = Counter(
            'project_index_websocket_events_delivered_total',
            'Total number of WebSocket events delivered',
            ['event_type', 'connection_id'],
            registry=self.registry
        )
        
        self.events_filtered_total = Counter(
            'project_index_websocket_events_filtered_total',
            'Total number of WebSocket events filtered',
            ['event_type', 'filter_reason'],
            registry=self.registry
        )
        
        self.event_delivery_duration = Histogram(
            'project_index_websocket_event_delivery_duration_seconds',
            'Time taken to deliver WebSocket events',
            ['event_type'],
            registry=self.registry
        )
        
        # Connection metrics
        self.active_connections = Gauge(
            'project_index_websocket_active_connections',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.connection_duration = Histogram(
            'project_index_websocket_connection_duration_seconds',
            'Duration of WebSocket connections',
            registry=self.registry
        )
        
        self.messages_sent_total = Counter(
            'project_index_websocket_messages_sent_total',
            'Total number of WebSocket messages sent',
            ['connection_id', 'status'],
            registry=self.registry
        )
        
        # Performance metrics
        self.event_processing_duration = Histogram(
            'project_index_websocket_event_processing_duration_seconds',
            'Time taken to process WebSocket events',
            ['event_type'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'project_index_websocket_memory_usage_bytes',
            'Memory usage of WebSocket system',
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'project_index_websocket_queue_size',
            'Size of WebSocket event queue',
            ['queue_type'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_violations_total = Counter(
            'project_index_websocket_rate_limit_violations_total',
            'Total number of rate limit violations',
            ['connection_id'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'project_index_websocket_errors_total',
            'Total number of WebSocket errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Health metrics
        self.health_score = Gauge(
            'project_index_websocket_health_score',
            'Overall health score of WebSocket system',
            registry=self.registry
        )


class PerformanceMonitor:
    """Performance monitoring and analysis."""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.alerts: List[MetricAlert] = []
        self.alert_handlers: List[Callable] = []
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.last_analysis = datetime.utcnow()
        
        # Default baselines
        self._setup_default_baselines()
        
        # Default alerts
        self._setup_default_alerts()
    
    def _setup_default_baselines(self):
        """Set up default performance baselines."""
        self.baselines = {
            "event_delivery_latency_ms": PerformanceBaseline(
                "event_delivery_latency_ms", 50.0, 20.0  # 50ms ±20%
            ),
            "events_per_second": PerformanceBaseline(
                "events_per_second", 100.0, 30.0  # 100 events/sec ±30%
            ),
            "connection_success_rate": PerformanceBaseline(
                "connection_success_rate", 0.99, 5.0  # 99% ±5%
            ),
            "memory_usage_mb": PerformanceBaseline(
                "memory_usage_mb", 100.0, 50.0  # 100MB ±50%
            ),
            "error_rate": PerformanceBaseline(
                "error_rate", 0.01, 100.0  # 1% ±100% (very tolerant)
            )
        }
    
    def _setup_default_alerts(self):
        """Set up default alert configurations."""
        self.alerts = [
            MetricAlert(
                "event_delivery_latency_ms", 500.0, "gt",
                AlertSeverity.WARNING,
                "High event delivery latency detected"
            ),
            MetricAlert(
                "event_delivery_latency_ms", 1000.0, "gt",
                AlertSeverity.ERROR,
                "Critical event delivery latency detected"
            ),
            MetricAlert(
                "connection_success_rate", 0.90, "lt",
                AlertSeverity.WARNING,
                "Low connection success rate detected"
            ),
            MetricAlert(
                "connection_success_rate", 0.80, "lt",
                AlertSeverity.ERROR,
                "Critical connection success rate detected"
            ),
            MetricAlert(
                "memory_usage_mb", 500.0, "gt",
                AlertSeverity.WARNING,
                "High memory usage detected"
            ),
            MetricAlert(
                "memory_usage_mb", 1000.0, "gt",
                AlertSeverity.CRITICAL,
                "Critical memory usage detected"
            ),
            MetricAlert(
                "error_rate", 0.05, "gt",
                AlertSeverity.WARNING,
                "High error rate detected"
            ),
            MetricAlert(
                "active_connections", 800, "gt",
                AlertSeverity.WARNING,
                "High number of active connections"
            ),
            MetricAlert(
                "queue_size", 1000, "gt",
                AlertSeverity.ERROR,
                "Event queue size is critically high"
            )
        ]
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value."""
        timestamp = timestamp or datetime.utcnow()
        
        self.metrics_history[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Update baseline if configured
        if metric_name in self.baselines:
            self.baselines[metric_name].samples.append(value)
        
        # Check alerts
        self._check_alerts(metric_name, value, timestamp)
    
    def _check_alerts(self, metric_name: str, value: float, timestamp: datetime):
        """Check if metric value triggers any alerts."""
        for alert in self.alerts:
            if alert.metric_name != metric_name:
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                timestamp - alert.last_triggered < timedelta(minutes=alert.cooldown_minutes)):
                continue
            
            # Evaluate condition
            triggered = False
            if alert.operator == "gt" and value > alert.threshold:
                triggered = True
            elif alert.operator == "lt" and value < alert.threshold:
                triggered = True
            elif alert.operator == "gte" and value >= alert.threshold:
                triggered = True
            elif alert.operator == "lte" and value <= alert.threshold:
                triggered = True
            elif alert.operator == "eq" and value == alert.threshold:
                triggered = True
            
            if triggered:
                alert.last_triggered = timestamp
                self._trigger_alert(alert, value, timestamp)
    
    def _trigger_alert(self, alert: MetricAlert, value: float, timestamp: datetime):
        """Trigger an alert."""
        alert_data = {
            "alert": alert,
            "metric_value": value,
            "timestamp": timestamp,
            "severity": alert.severity.value,
            "message": alert.message
        }
        
        logger.warning(
            f"WebSocket Alert: {alert.message}",
            metric_name=alert.metric_name,
            metric_value=value,
            threshold=alert.threshold,
            severity=alert.severity.value
        )
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert_data))
                else:
                    handler(alert_data)
            except Exception as e:
                logger.error("Alert handler failed", error=str(e))
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            # Filter recent data
            recent_data = [
                entry for entry in history
                if entry["timestamp"] >= cutoff_time
            ]
            
            if not recent_data:
                continue
            
            values = [entry["value"] for entry in recent_data]
            
            # Calculate statistics
            summary[metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest": values[-1] if values else None
            }
            
            # Add baseline comparison if available
            if metric_name in self.baselines:
                baseline = self.baselines[metric_name]
                if baseline.samples:
                    baseline_mean = statistics.mean(baseline.samples)
                    current_mean = summary[metric_name]["mean"]
                    variance_percent = abs(current_mean - baseline_mean) / baseline_mean * 100
                    
                    summary[metric_name]["baseline_comparison"] = {
                        "baseline_value": baseline.baseline_value,
                        "current_vs_baseline": current_mean - baseline.baseline_value,
                        "variance_percent": variance_percent,
                        "within_acceptable_range": variance_percent <= baseline.acceptable_variance
                    }
        
        return summary
    
    def detect_anomalies(self, metric_name: str, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        if metric_name not in self.metrics_history:
            return []
        
        # Get recent data
        recent_data = [
            entry for entry in self.metrics_history[metric_name]
            if entry["timestamp"] >= cutoff_time
        ]
        
        if len(recent_data) < 10:  # Need minimum data points
            return []
        
        values = [entry["value"] for entry in recent_data]
        mean_val = statistics.mean(values)
        stddev_val = statistics.stdev(values)
        
        # Detect outliers (values beyond 2 standard deviations)
        anomalies = []
        for entry in recent_data:
            z_score = abs(entry["value"] - mean_val) / stddev_val if stddev_val > 0 else 0
            if z_score > 2.0:  # 2 sigma threshold
                anomalies.append({
                    "timestamp": entry["timestamp"],
                    "value": entry["value"],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3.0 else "medium"
                })
        
        return anomalies


class WebSocketHealthMonitor:
    """Health monitoring for WebSocket system."""
    
    def __init__(self, websocket_manager: ProjectIndexWebSocketManager):
        self.websocket_manager = websocket_manager
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        self.health_checks = {
            "connection_pool": self._check_connection_pool_health,
            "event_publisher": self._check_event_publisher_health,
            "event_filter": self._check_event_filter_health,
            "redis_connectivity": self._check_redis_connectivity,
            "memory_usage": self._check_memory_usage,
            "error_rates": self._check_error_rates
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        results = {}
        overall_health = True
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                results[check_name] = result
                
                if not result.get("healthy", False):
                    overall_health = False
                    
            except Exception as e:
                results[check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_health = False
        
        results["overall_health"] = overall_health
        results["timestamp"] = datetime.utcnow().isoformat()
        
        return results
    
    def _check_connection_pool_health(self) -> Dict[str, Any]:
        """Check connection pool health."""
        pool_stats = self.websocket_manager.performance_manager.connection_pool.get_pool_stats()
        
        healthy = (
            pool_stats.get("active_connections", 0) >= 0 and
            pool_stats.get("message_success_rate", 0) > 0.8 and
            pool_stats.get("average_latency_ms", 1000) < 500
        )
        
        return {
            "healthy": healthy,
            "stats": pool_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_event_publisher_health(self) -> Dict[str, Any]:
        """Check event publisher health."""
        publisher_metrics = self.websocket_manager.event_publisher.get_metrics()
        
        total_events = publisher_metrics.get("events_published", 0)
        failed_events = publisher_metrics.get("events_failed", 0)
        
        success_rate = 1.0
        if total_events > 0:
            success_rate = (total_events - failed_events) / total_events
        
        healthy = success_rate > 0.9
        
        return {
            "healthy": healthy,
            "success_rate": success_rate,
            "metrics": publisher_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_event_filter_health(self) -> Dict[str, Any]:
        """Check event filter health."""
        filter_metrics = self.websocket_manager.event_filter.get_metrics()
        
        # Basic health check - filter should be processing events
        healthy = filter_metrics.get("filter_pass_rate", 0) >= 0
        
        return {
            "healthy": healthy,
            "metrics": filter_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_redis_connectivity(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            # Simple ping test
            await self.websocket_manager.redis_client.ping()
            
            return {
                "healthy": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Alert if memory usage is very high
            healthy = memory_mb < 1000  # Less than 1GB
            
            return {
                "healthy": healthy,
                "memory_mb": memory_mb,
                "timestamp": datetime.utcnow().isoformat()
            }
        except ImportError:
            return {
                "healthy": True,
                "message": "psutil not available, skipping memory check",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _check_error_rates(self) -> Dict[str, Any]:
        """Check error rates across the system."""
        manager_metrics = self.websocket_manager.get_metrics()
        
        total_operations = manager_metrics["websocket_manager"].get("connections_handled", 0)
        total_errors = manager_metrics["websocket_manager"].get("errors", 0)
        
        error_rate = 0.0
        if total_operations > 0:
            error_rate = total_errors / total_operations
        
        # Healthy if error rate is below 5%
        healthy = error_rate < 0.05
        
        return {
            "healthy": healthy,
            "error_rate": error_rate,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "timestamp": datetime.utcnow().isoformat()
        }


class WebSocketMonitoringSystem:
    """Complete monitoring system for WebSocket infrastructure."""
    
    def __init__(self, websocket_manager: ProjectIndexWebSocketManager):
        self.websocket_manager = websocket_manager
        self.prometheus_metrics = PrometheusMetrics()
        self.performance_monitor = PerformanceMonitor()
        self.health_monitor = WebSocketHealthMonitor(websocket_manager)
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.monitoring_interval = 30  # seconds
        
        # Setup alert handlers
        self.performance_monitor.add_alert_handler(self._handle_performance_alert)
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        logger.info("Starting WebSocket monitoring system")
        
        # Start periodic monitoring
        self.monitoring_tasks = [
            asyncio.create_task(self._periodic_metrics_collection()),
            asyncio.create_task(self._periodic_health_checks()),
            asyncio.create_task(self._periodic_performance_analysis())
        ]
        
        logger.info("WebSocket monitoring system started")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        logger.info("Stopping WebSocket monitoring system")
        
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.monitoring_tasks.clear()
        logger.info("WebSocket monitoring system stopped")
    
    async def _periodic_metrics_collection(self):
        """Collect metrics periodically."""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection failed", error=str(e))
                await asyncio.sleep(self.monitoring_interval)
    
    async def _periodic_health_checks(self):
        """Run health checks periodically."""
        while True:
            try:
                health_results = await self.health_monitor.run_health_checks()
                
                # Update Prometheus health metric
                health_score = 1.0 if health_results.get("overall_health", False) else 0.0
                self.prometheus_metrics.health_score.set(health_score)
                
                # Record performance metric
                self.performance_monitor.record_metric("health_score", health_score)
                
                await asyncio.sleep(self.monitoring_interval * 2)  # Less frequent than metrics
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _periodic_performance_analysis(self):
        """Analyze performance periodically."""
        while True:
            try:
                # Detect anomalies
                for metric_name in ["event_delivery_latency_ms", "events_per_second", "error_rate"]:
                    anomalies = self.performance_monitor.detect_anomalies(metric_name)
                    if anomalies:
                        logger.warning(
                            f"Performance anomalies detected in {metric_name}",
                            anomaly_count=len(anomalies),
                            anomalies=anomalies
                        )
                
                await asyncio.sleep(self.monitoring_interval * 4)  # Less frequent analysis
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance analysis failed", error=str(e))
                await asyncio.sleep(self.monitoring_interval * 4)
    
    async def _collect_metrics(self):
        """Collect and record current metrics."""
        try:
            # Get WebSocket manager metrics
            manager_metrics = self.websocket_manager.get_metrics()
            
            # Update Prometheus metrics
            self.prometheus_metrics.active_connections.set(
                manager_metrics.get("active_connections", 0)
            )
            
            # Update performance monitor
            self.performance_monitor.record_metric(
                "active_connections",
                float(manager_metrics.get("active_connections", 0))
            )
            
            # Get performance metrics from subsystems
            perf_metrics = self.websocket_manager.performance_manager.get_performance_summary()
            
            # Record key performance metrics
            if "global_metrics" in perf_metrics:
                global_metrics = perf_metrics["global_metrics"]
                
                if global_metrics.get("average_processing_time_ms", 0) > 0:
                    self.performance_monitor.record_metric(
                        "event_delivery_latency_ms",
                        global_metrics["average_processing_time_ms"]
                    )
                
                total_events = global_metrics.get("total_events_processed", 0)
                failed_events = global_metrics.get("total_events_failed", 0)
                
                if total_events > 0:
                    error_rate = failed_events / total_events
                    self.performance_monitor.record_metric("error_rate", error_rate)
            
            # Update memory usage
            try:
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                self.prometheus_metrics.memory_usage_bytes.set(memory_mb * 1024 * 1024)
                self.performance_monitor.record_metric("memory_usage_mb", memory_mb)
                
            except ImportError:
                pass  # psutil not available
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
    
    async def _handle_performance_alert(self, alert_data: Dict[str, Any]):
        """Handle performance alerts."""
        alert = alert_data["alert"]
        
        # Update Prometheus error counter
        self.prometheus_metrics.errors_total.labels(
            error_type="performance_alert",
            component="monitoring"
        ).inc()
        
        # Log structured alert
        logger.warning(
            "Performance alert triggered",
            alert_metric=alert.metric_name,
            alert_threshold=alert.threshold,
            alert_value=alert_data["metric_value"],
            alert_severity=alert.severity.value,
            alert_message=alert.message
        )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "websocket_manager": self.websocket_manager.get_metrics(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "health_status": asyncio.create_task(self.health_monitor.run_health_checks()),
            "monitoring_active": len(self.monitoring_tasks) > 0,
            "uptime_seconds": (datetime.utcnow() - self.performance_monitor.start_time).total_seconds()
        }
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        from prometheus_client import generate_latest
        return generate_latest(self.prometheus_metrics.registry).decode('utf-8')


# Global monitoring system instance
_monitoring_system: Optional[WebSocketMonitoringSystem] = None


async def get_monitoring_system(
    websocket_manager: ProjectIndexWebSocketManager
) -> WebSocketMonitoringSystem:
    """Get or create the global monitoring system."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = WebSocketMonitoringSystem(websocket_manager)
    return _monitoring_system


# Utility functions for external monitoring integration
async def export_prometheus_metrics(websocket_manager: ProjectIndexWebSocketManager) -> str:
    """Export Prometheus metrics."""
    monitoring_system = await get_monitoring_system(websocket_manager)
    return monitoring_system.export_metrics_for_prometheus()


async def get_health_status(websocket_manager: ProjectIndexWebSocketManager) -> Dict[str, Any]:
    """Get current health status."""
    monitoring_system = await get_monitoring_system(websocket_manager)
    return await monitoring_system.health_monitor.run_health_checks()


async def get_performance_dashboard_data(
    websocket_manager: ProjectIndexWebSocketManager,
    window_minutes: int = 60
) -> Dict[str, Any]:
    """Get data for performance dashboard."""
    monitoring_system = await get_monitoring_system(websocket_manager)
    
    return {
        "performance_summary": monitoring_system.performance_monitor.get_performance_summary(window_minutes),
        "health_status": await monitoring_system.health_monitor.run_health_checks(),
        "websocket_metrics": websocket_manager.get_metrics(),
        "anomalies": {
            metric_name: monitoring_system.performance_monitor.detect_anomalies(metric_name, window_minutes)
            for metric_name in ["event_delivery_latency_ms", "events_per_second", "error_rate"]
        }
    }