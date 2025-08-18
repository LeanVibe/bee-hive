"""
Comprehensive Monitoring and Logging System for Real-Time Communication

This module provides advanced monitoring, logging, and observability for the
LeanVibe Agent Hive 2.0 real-time communication system.

Features:
- Structured logging with correlation IDs and context
- Performance metrics collection and analysis
- Real-time monitoring dashboards and alerts
- Event tracing and audit trails
- Resource usage monitoring
- Anomaly detection and alerting
- Integration with external monitoring systems
- Custom metrics and KPI tracking
"""

import asyncio
import logging
import json
import time
import uuid
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
import statistics
import threading
from collections import defaultdict, deque
import weakref

from .realtime_communication_hub import (
    RealTimeCommunicationHub,
    HealthMonitoringUpdate,
    MessagePriority,
    NotificationType
)

# ================================================================================
# Monitoring Models and Enums
# ================================================================================

class LogLevel(Enum):
    """Enhanced log levels for communication monitoring."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"          # Incrementing values
    GAUGE = "gauge"             # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"             # Duration measurements
    RATE = "rate"               # Rate of change

class AlertSeverity(IntEnum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class LogEntry:
    """Structured log entry with context and correlation."""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    duration_ms: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics for communication components."""
    component_name: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert with context and severity."""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    component: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

# ================================================================================
# Enhanced Structured Logger
# ================================================================================

class StructuredLogger:
    """
    Enhanced structured logger with correlation tracking and context.
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize structured logger."""
        self.name = name
        self.level = level
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.context = context or {}
        self.trace_stack: List[str] = []
        
        # Create standard Python logger
        self._python_logger = logging.getLogger(name)
        
        # Custom handlers for structured logging
        self._handlers: List[Callable[[LogEntry], None]] = []
        self._filters: List[Callable[[LogEntry], bool]] = []
        
        # Metrics integration
        self._metrics_collector: Optional['MetricsCollector'] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for tracking related operations."""
        self.correlation_id = correlation_id
    
    def add_context(self, key: str, value: Any):
        """Add context data to all log entries."""
        self.context[key] = value
    
    def push_trace(self, operation: str):
        """Push operation onto trace stack."""
        self.trace_stack.append(operation)
    
    def pop_trace(self):
        """Pop operation from trace stack."""
        if self.trace_stack:
            return self.trace_stack.pop()
        return None
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, exc_info=None, **kwargs):
        """Log exception with stack trace."""
        if exc_info is None:
            exc_info = sys.exc_info()
        
        exception_str = None
        stack_trace = None
        
        if exc_info and exc_info[0] is not None:
            exception_str = f"{exc_info[0].__name__}: {exc_info[1]}"
            stack_trace = ''.join(traceback.format_exception(*exc_info))
        
        kwargs.update({
            'exception': exception_str,
            'stack_trace': stack_trace
        })
        
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        if level.value < self.level.value:
            return
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            logger_name=self.name,
            message=message,
            correlation_id=self.correlation_id,
            trace_id='.'.join(self.trace_stack) if self.trace_stack else None,
            component=kwargs.get('component'),
            operation=kwargs.get('operation'),
            duration_ms=kwargs.get('duration_ms'),
            context={**self.context, **kwargs.get('context', {})},
            exception=kwargs.get('exception'),
            stack_trace=kwargs.get('stack_trace'),
            tags=kwargs.get('tags', {})
        )
        
        # Apply filters
        if not all(f(entry) for f in self._filters):
            return
        
        # Send to handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception as e:
                # Fallback to standard logging for handler errors
                self._python_logger.error(f"Log handler error: {e}")
        
        # Also log to standard Python logger
        self._log_to_python_logger(entry)
        
        # Update metrics if available
        if self._metrics_collector:
            self._metrics_collector.increment_counter(
                f"log_entries_{level.value.lower()}",
                tags={"component": self.name, "level": level.value}
            )
    
    def _log_to_python_logger(self, entry: LogEntry):
        """Log to standard Python logger for compatibility."""
        # Create formatted message
        msg_parts = [entry.message]
        
        if entry.correlation_id:
            msg_parts.append(f"[{entry.correlation_id}]")
        
        if entry.trace_id:
            msg_parts.append(f"({entry.trace_id})")
        
        if entry.duration_ms is not None:
            msg_parts.append(f"{entry.duration_ms:.1f}ms")
        
        formatted_message = " ".join(msg_parts)
        
        # Map to Python logging level
        python_level = {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }[entry.level]
        
        # Log with extra context
        extra = {
            'correlation_id': entry.correlation_id,
            'trace_id': entry.trace_id,
            'component': entry.component,
            'operation': entry.operation
        }
        
        self._python_logger.log(python_level, formatted_message, extra=extra)
    
    def add_handler(self, handler: Callable[[LogEntry], None]):
        """Add custom log handler."""
        self._handlers.append(handler)
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]):
        """Add log filter."""
        self._filters.append(filter_func)
    
    def with_context(self, **context) -> 'StructuredLogger':
        """Create logger with additional context."""
        new_context = {**self.context, **context}
        return StructuredLogger(
            name=self.name,
            level=self.level,
            correlation_id=self.correlation_id,
            context=new_context
        )

# ================================================================================
# Performance Timer Context Manager
# ================================================================================

class PerformanceTimer:
    """Context manager for measuring operation performance."""
    
    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        component: str,
        log_on_completion: bool = True,
        collect_metrics: bool = True
    ):
        self.logger = logger
        self.operation = operation
        self.component = component
        self.log_on_completion = log_on_completion
        self.collect_metrics = collect_metrics
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.success = True
        self.error_message: Optional[str] = None
    
    def __enter__(self) -> 'PerformanceTimer':
        self.start_time = datetime.utcnow()
        self.logger.push_trace(self.operation)
        
        if self.log_on_completion:
            self.logger.trace(f"Starting {self.operation}", 
                            component=self.component, 
                            operation=self.operation)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.utcnow()
        self.logger.pop_trace()
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
        
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if self.log_on_completion:
            if self.success:
                self.logger.info(f"Completed {self.operation}",
                               component=self.component,
                               operation=self.operation,
                               duration_ms=duration_ms)
            else:
                self.logger.error(f"Failed {self.operation}: {self.error_message}",
                                component=self.component,
                                operation=self.operation,
                                duration_ms=duration_ms)
        
        # Collect metrics if enabled
        if self.collect_metrics and hasattr(self.logger, '_metrics_collector') and self.logger._metrics_collector:
            self.logger._metrics_collector.record_timer(
                f"{self.component}.{self.operation}",
                duration_ms,
                tags={
                    "component": self.component,
                    "operation": self.operation,
                    "success": str(self.success)
                }
            )
    
    def add_custom_metric(self, name: str, value: Any):
        """Add custom metric to this operation."""
        # This would be stored and reported with the operation
        pass

# ================================================================================
# Metrics Collector
# ================================================================================

class MetricsCollector:
    """
    Comprehensive metrics collection system.
    """
    
    def __init__(self, realtime_hub: Optional[RealTimeCommunicationHub] = None):
        self.realtime_hub = realtime_hub
        
        # Metric storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Metric metadata
        self._metric_metadata: Dict[str, Dict[str, Any]] = {}
        self._metric_tags: Dict[str, Dict[str, str]] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._collection_task: Optional[asyncio.Task] = None
        
        # Export handlers
        self._export_handlers: List[Callable[[List[MetricPoint]], None]] = []
        
        # Configuration
        self._config = {
            "collection_interval_seconds": 60,
            "histogram_max_samples": 10000,
            "timer_max_samples": 10000,
            "enable_real_time_export": True,
            "enable_aggregation": True
        }
    
    async def initialize(self) -> bool:
        """Initialize metrics collector."""
        try:
            # Start background collection
            self._collection_task = asyncio.create_task(self._collection_loop())
            self._background_tasks.add(self._collection_task)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize metrics collector: {e}")
            return False
    
    def increment_counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._get_metric_key(name, tags)
        self._counters[key] += value
        
        if tags:
            self._metric_tags[key] = tags
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        key = self._get_metric_key(name, tags)
        self._gauges[key] = value
        
        if tags:
            self._metric_tags[key] = tags
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        key = self._get_metric_key(name, tags)
        self._histograms[key].append(value)
        
        # Limit histogram size
        if len(self._histograms[key]) > self._config["histogram_max_samples"]:
            self._histograms[key] = self._histograms[key][-self._config["histogram_max_samples"]:]
        
        if tags:
            self._metric_tags[key] = tags
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        key = self._get_metric_key(name, tags)
        self._timers[key].append(duration_ms)
        
        # Limit timer samples
        if len(self._timers[key]) > self._config["timer_max_samples"]:
            self._timers[key] = self._timers[key][-self._config["timer_max_samples"]:]
        
        if tags:
            self._metric_tags[key] = tags
    
    def record_rate(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Record an event for rate calculation."""
        key = self._get_metric_key(name, tags)
        self._rates[key].append(time.time())
        
        if tags:
            self._metric_tags[key] = tags
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._get_metric_key(name, tags)
        return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._get_metric_key(name, tags)
        return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._get_metric_key(name, tags)
        values = self._histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0,
            "p99": sorted_values[int(n * 0.99)] if n > 0 else 0,
            "stddev": statistics.stdev(values) if n > 1 else 0
        }
    
    def get_timer_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        return self.get_histogram_stats(name, tags)  # Same calculation
    
    def get_rate(self, name: str, window_seconds: int = 60, tags: Optional[Dict[str, str]] = None) -> float:
        """Get rate per second over time window."""
        key = self._get_metric_key(name, tags)
        timestamps = self._rates.get(key, deque())
        
        if not timestamps:
            return 0.0
        
        # Filter to time window
        cutoff_time = time.time() - window_seconds
        recent_events = [t for t in timestamps if t > cutoff_time]
        
        return len(recent_events) / window_seconds
    
    def get_all_metrics(self) -> List[MetricPoint]:
        """Get all current metrics as MetricPoint objects."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # Counters
        for key, value in self._counters.items():
            name, tags = self._parse_metric_key(key)
            metrics.append(MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                timestamp=timestamp,
                tags=tags
            ))
        
        # Gauges
        for key, value in self._gauges.items():
            name, tags = self._parse_metric_key(key)
            metrics.append(MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                tags=tags
            ))
        
        # Histogram aggregates
        for key, values in self._histograms.items():
            if values:
                name, tags = self._parse_metric_key(key)
                stats = self.get_histogram_stats(name, tags)
                
                for stat_name, stat_value in stats.items():
                    metrics.append(MetricPoint(
                        name=f"{name}_{stat_name}",
                        value=stat_value,
                        metric_type=MetricType.HISTOGRAM,
                        timestamp=timestamp,
                        tags=tags
                    ))
        
        return metrics
    
    def _get_metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create unique metric key from name and tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
    
    def _parse_metric_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse metric key back to name and tags."""
        if "|" not in key:
            return key, {}
        
        name, tag_str = key.split("|", 1)
        tags = {}
        
        if tag_str:
            for tag_pair in tag_str.split(","):
                if "=" in tag_pair:
                    k, v = tag_pair.split("=", 1)
                    tags[k] = v
        
        return name, tags
    
    async def _collection_loop(self):
        """Background loop for metric collection and export."""
        while True:
            try:
                await asyncio.sleep(self._config["collection_interval_seconds"])
                
                # Get current metrics
                metrics = self.get_all_metrics()
                
                # Export to handlers
                for handler in self._export_handlers:
                    try:
                        handler(metrics)
                    except Exception as e:
                        logging.error(f"Metric export handler error: {e}")
                
                # Export to real-time hub
                if self._config["enable_real_time_export"] and self.realtime_hub:
                    await self._export_to_realtime_hub(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in metrics collection loop: {e}")
    
    async def _export_to_realtime_hub(self, metrics: List[MetricPoint]):
        """Export metrics to real-time hub."""
        try:
            # Convert metrics to dashboard data
            dashboard_data = {
                "metrics": [asdict(metric) for metric in metrics],
                "timestamp": datetime.utcnow().isoformat(),
                "collection_interval": self._config["collection_interval_seconds"]
            }
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="metrics_update",
                data=dashboard_data
            )
            
        except Exception as e:
            logging.error(f"Failed to export metrics to real-time hub: {e}")
    
    def add_export_handler(self, handler: Callable[[List[MetricPoint]], None]):
        """Add custom metrics export handler."""
        self._export_handlers.append(handler)
    
    async def shutdown(self):
        """Shutdown metrics collector."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        except Exception as e:
            logging.error(f"Error during metrics collector shutdown: {e}")

# ================================================================================
# Alert Manager
# ================================================================================

class AlertManager:
    """
    Alert management system for monitoring and notifications.
    """
    
    def __init__(self, realtime_hub: Optional[RealTimeCommunicationHub] = None):
        self.realtime_hub = realtime_hub
        
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        
        # Alert rules
        self._alert_rules: List[Callable[[Dict[str, Any]], Optional[Alert]]] = []
        
        # Alert handlers
        self._alert_handlers: Dict[AlertSeverity, List[Callable[[Alert], None]]] = {
            severity: [] for severity in AlertSeverity
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._evaluation_task: Optional[asyncio.Task] = None
        
        # Configuration
        self._config = {
            "evaluation_interval_seconds": 30,
            "max_alert_history": 10000,
            "auto_resolve_timeout_minutes": 60,
            "enable_real_time_alerts": True
        }
    
    async def initialize(self) -> bool:
        """Initialize alert manager."""
        try:
            # Start background evaluation
            self._evaluation_task = asyncio.create_task(self._evaluation_loop())
            self._background_tasks.add(self._evaluation_task)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize alert manager: {e}")
            return False
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add alert evaluation rule."""
        self._alert_rules.append(rule)
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Add alert handler for specific severity."""
        self._alert_handlers[severity].append(handler)
    
    async def trigger_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        component: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Manually trigger an alert."""
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            description=description,
            component=component,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        await self._process_alert(alert)
        return alert_id
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """Manually resolve an alert."""
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        
        if resolution_note:
            alert.metadata["resolution_note"] = resolution_note
        
        # Move to history
        self._alert_history.append(alert)
        del self._active_alerts[alert_id]
        
        # Limit history size
        if len(self._alert_history) > self._config["max_alert_history"]:
            self._alert_history = self._alert_history[-self._config["max_alert_history"]:]
        
        # Broadcast resolution
        if self.realtime_hub and self._config["enable_real_time_alerts"]:
            await self._broadcast_alert_resolution(alert)
        
        return True
    
    async def _process_alert(self, alert: Alert):
        """Process and handle a new alert."""
        try:
            # Check for duplicate alerts
            duplicate_key = f"{alert.component}_{alert.title}"
            existing_alert = None
            
            for existing in self._active_alerts.values():
                if f"{existing.component}_{existing.title}" == duplicate_key:
                    existing_alert = existing
                    break
            
            if existing_alert:
                # Update existing alert timestamp
                existing_alert.timestamp = alert.timestamp
                existing_alert.metadata.update(alert.metadata)
                return
            
            # Add to active alerts
            self._active_alerts[alert.alert_id] = alert
            
            # Trigger handlers
            for handler in self._alert_handlers[alert.severity]:
                try:
                    handler(alert)
                except Exception as e:
                    logging.error(f"Alert handler error: {e}")
            
            # Broadcast to real-time hub
            if self.realtime_hub and self._config["enable_real_time_alerts"]:
                await self._broadcast_alert(alert)
            
        except Exception as e:
            logging.error(f"Error processing alert: {e}")
    
    async def _broadcast_alert(self, alert: Alert):
        """Broadcast alert to real-time hub."""
        try:
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="alert_triggered",
                data={
                    "alert": asdict(alert),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logging.error(f"Failed to broadcast alert: {e}")
    
    async def _broadcast_alert_resolution(self, alert: Alert):
        """Broadcast alert resolution to real-time hub."""
        try:
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="alert_resolved",
                data={
                    "alert": asdict(alert),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logging.error(f"Failed to broadcast alert resolution: {e}")
    
    async def _evaluation_loop(self):
        """Background loop for alert rule evaluation."""
        while True:
            try:
                await asyncio.sleep(self._config["evaluation_interval_seconds"])
                
                # Evaluate alert rules (would need metrics data)
                # This is a placeholder for rule evaluation
                
                # Auto-resolve old alerts
                await self._auto_resolve_alerts()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in alert evaluation loop: {e}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that haven't been updated."""
        cutoff_time = datetime.utcnow() - timedelta(
            minutes=self._config["auto_resolve_timeout_minutes"]
        )
        
        expired_alerts = [
            alert_id for alert_id, alert in self._active_alerts.items()
            if alert.timestamp < cutoff_time
        ]
        
        for alert_id in expired_alerts:
            await self.resolve_alert(alert_id, "Auto-resolved due to timeout")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_by_severity = defaultdict(int)
        for alert in self._active_alerts.values():
            active_by_severity[alert.severity.name] += 1
        
        return {
            "total_active": len(self._active_alerts),
            "active_by_severity": dict(active_by_severity),
            "total_resolved": len(self._alert_history),
            "evaluation_rules": len(self._alert_rules)
        }
    
    async def shutdown(self):
        """Shutdown alert manager."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        except Exception as e:
            logging.error(f"Error during alert manager shutdown: {e}")

# ================================================================================
# Comprehensive Monitoring System
# ================================================================================

class ComprehensiveMonitoringSystem:
    """
    Integrated monitoring system combining logging, metrics, and alerting.
    """
    
    def __init__(self, realtime_hub: Optional[RealTimeCommunicationHub] = None):
        self.realtime_hub = realtime_hub
        
        # Core components
        self.metrics_collector = MetricsCollector(realtime_hub)
        self.alert_manager = AlertManager(realtime_hub)
        
        # Logger registry
        self._loggers: Dict[str, StructuredLogger] = {}
        
        # Log handlers
        self._log_handlers: List[Callable[[LogEntry], None]] = []
        
        # Integration hooks
        self._monitoring_hooks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    async def initialize(self) -> bool:
        """Initialize comprehensive monitoring system."""
        try:
            # Initialize components
            await self.metrics_collector.initialize()
            await self.alert_manager.initialize()
            
            # Set up default alert rules
            self._setup_default_alert_rules()
            
            # Set up default log handlers
            self._setup_default_log_handlers()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize monitoring system: {e}")
            return False
    
    def get_logger(
        self,
        name: str,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> StructuredLogger:
        """Get or create a structured logger."""
        logger_key = f"{name}_{correlation_id or 'default'}"
        
        if logger_key not in self._loggers:
            logger = StructuredLogger(name, correlation_id=correlation_id, context=context)
            logger._metrics_collector = self.metrics_collector
            
            # Add global log handlers
            for handler in self._log_handlers:
                logger.add_handler(handler)
            
            self._loggers[logger_key] = logger
        
        return self._loggers[logger_key]
    
    def create_performance_timer(
        self,
        operation: str,
        component: str,
        logger: Optional[StructuredLogger] = None
    ) -> PerformanceTimer:
        """Create a performance timer for an operation."""
        if logger is None:
            logger = self.get_logger(component)
        
        return PerformanceTimer(logger, operation, component)
    
    def _setup_default_alert_rules(self):
        """Set up default alerting rules."""
        # High error rate alert
        def high_error_rate_rule(metrics_data: Dict[str, Any]) -> Optional[Alert]:
            # This would analyze metrics for high error rates
            return None
        
        # High latency alert
        def high_latency_rule(metrics_data: Dict[str, Any]) -> Optional[Alert]:
            # This would analyze metrics for high latency
            return None
        
        self.alert_manager.add_alert_rule(high_error_rate_rule)
        self.alert_manager.add_alert_rule(high_latency_rule)
    
    def _setup_default_log_handlers(self):
        """Set up default log handlers."""
        # Console handler
        def console_handler(entry: LogEntry):
            if entry.level.value >= LogLevel.INFO.value:
                print(f"[{entry.timestamp.isoformat()}] {entry.level.value} {entry.logger_name}: {entry.message}")
        
        # Metrics handler
        def metrics_handler(entry: LogEntry):
            self.metrics_collector.increment_counter(
                "log_entries_total",
                tags={
                    "logger": entry.logger_name,
                    "level": entry.level.value,
                    "component": entry.component or "unknown"
                }
            )
        
        self._log_handlers.extend([console_handler, metrics_handler])
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            # Get metrics summary
            metrics = self.metrics_collector.get_all_metrics()
            
            # Get alert summary
            alert_summary = self.alert_manager.get_alert_summary()
            
            # Calculate overall health score
            critical_alerts = alert_summary["active_by_severity"].get("CRITICAL", 0)
            error_alerts = alert_summary["active_by_severity"].get("ERROR", 0)
            
            if critical_alerts > 0:
                health_status = "critical"
                health_score = 0.0
            elif error_alerts > 5:
                health_status = "degraded"
                health_score = 0.3
            elif error_alerts > 0:
                health_status = "warning"
                health_score = 0.7
            else:
                health_status = "healthy"
                health_score = 1.0
            
            return {
                "status": health_status,
                "health_score": health_score,
                "metrics_count": len(metrics),
                "alerts": alert_summary,
                "active_loggers": len(self._loggers),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to get system health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self):
        """Shutdown monitoring system."""
        try:
            await self.metrics_collector.shutdown()
            await self.alert_manager.shutdown()
            
        except Exception as e:
            logging.error(f"Error during monitoring system shutdown: {e}")

# ================================================================================
# Factory Functions
# ================================================================================

def create_monitoring_system(
    realtime_hub: Optional[RealTimeCommunicationHub] = None
) -> ComprehensiveMonitoringSystem:
    """Create a comprehensive monitoring system."""
    return ComprehensiveMonitoringSystem(realtime_hub)

# Example usage
async def example_usage():
    """Example of how to use the monitoring system."""
    # Create monitoring system
    monitoring = create_monitoring_system()
    await monitoring.initialize()
    
    # Get logger
    logger = monitoring.get_logger("example_component")
    
    # Use performance timer
    with monitoring.create_performance_timer("example_operation", "example_component"):
        logger.info("Performing example operation")
        await asyncio.sleep(0.1)  # Simulate work
    
    # Record metrics
    monitoring.metrics_collector.increment_counter("example_counter", tags={"type": "demo"})
    monitoring.metrics_collector.set_gauge("example_gauge", 42.0)
    
    # Trigger alert
    await monitoring.alert_manager.trigger_alert(
        title="Example Alert",
        description="This is an example alert",
        severity=AlertSeverity.WARNING,
        component="example_component"
    )
    
    # Get system health
    health = await monitoring.get_system_health()
    print(f"System health: {health}")
    
    # Cleanup
    await monitoring.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())