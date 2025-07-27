"""
Redis Streams Monitoring System with Prometheus Integration.

Provides comprehensive monitoring, metrics collection, and alerting for Redis Streams
communication system with production-grade observability features.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

# Prometheus metrics (optional import)
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST, Info
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus is not available
    class Counter:
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def info(self, *args, **kwargs): pass

from ..core.config import settings

logger = structlog.get_logger()


@dataclass
class StreamHealthMetrics:
    """Health metrics for a Redis stream."""
    
    stream_name: str
    length: int
    consumer_groups: int
    total_consumers: int
    total_pending: int
    total_lag: int
    messages_per_second: float
    error_rate: float
    avg_processing_latency_ms: float
    p95_processing_latency_ms: float
    p99_processing_latency_ms: float
    oldest_pending_age_seconds: float
    health_score: float  # 0.0 to 1.0
    status: str  # "healthy", "warning", "critical", "error"
    last_updated: float


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    
    name: str
    condition: str  # Python expression to evaluate
    severity: str  # "warning", "critical"
    cooldown_seconds: int = 300  # 5 minutes default
    message_template: str = ""
    enabled: bool = True


class StreamMonitor:
    """
    Comprehensive Redis Streams monitoring with Prometheus integration.
    
    Provides real-time metrics collection, health scoring, alerting,
    and performance tracking for production Redis Streams deployments.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        enable_prometheus: bool = True,
        custom_registry: Optional[Any] = None  # CollectorRegistry
    ):
        self.redis = redis_client
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Prometheus registry
        if self.enable_prometheus:
            self.registry = custom_registry or CollectorRegistry()
        
        # Stream monitoring state
        self._stream_metrics: Dict[str, StreamHealthMetrics] = {}
        self._historical_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._alert_states: Dict[str, Dict[str, float]] = {}  # Alert name -> last fired time
        
        # Performance tracking
        self._latency_samples: Dict[str, List[float]] = defaultdict(list)
        self._throughput_samples: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._metrics_cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.monitoring_interval = 5  # seconds
        self.history_retention_hours = 24
        self.latency_sample_size = 1000
        self.throughput_window_minutes = 5
        
        # Alert rules
        self._alert_rules: List[AlertRule] = self._default_alert_rules()
        
        # Initialize Prometheus metrics
        if self.enable_prometheus:
            self._init_prometheus_metrics()
    
    def _default_alert_rules(self) -> List[AlertRule]:
        """Define default alerting rules."""
        return [
            AlertRule(
                name="high_consumer_lag",
                condition="stream_metrics.total_lag > 1000",
                severity="warning",
                message_template="High consumer lag on {stream_name}: {total_lag} messages"
            ),
            AlertRule(
                name="critical_consumer_lag", 
                condition="stream_metrics.total_lag > 5000",
                severity="critical",
                message_template="CRITICAL: Consumer lag on {stream_name}: {total_lag} messages"
            ),
            AlertRule(
                name="high_error_rate",
                condition="stream_metrics.error_rate > 0.05",  # 5%
                severity="warning",
                message_template="High error rate on {stream_name}: {error_rate:.2%}"
            ),
            AlertRule(
                name="no_consumers",
                condition="stream_metrics.total_consumers == 0 and stream_metrics.length > 0",
                severity="critical",
                message_template="No active consumers on {stream_name} with {length} messages"
            ),
            AlertRule(
                name="high_latency",
                condition="stream_metrics.p95_processing_latency_ms > 1000",  # 1 second
                severity="warning",
                message_template="High processing latency on {stream_name}: P95 {p95_processing_latency_ms:.1f}ms"
            ),
            AlertRule(
                name="low_throughput",
                condition="stream_metrics.messages_per_second < 0.1 and stream_metrics.length > 100",
                severity="warning",
                message_template="Low throughput on {stream_name}: {messages_per_second:.2f} msg/sec"
            ),
            AlertRule(
                name="unhealthy_stream",
                condition="stream_metrics.health_score < 0.5",
                severity="critical",
                message_template="Stream {stream_name} unhealthy: score {health_score:.2f}"
            )
        ]
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        # Stream metrics
        self.stream_length = Gauge(
            'redis_stream_length_total',
            'Total number of messages in stream',
            ['stream_name'],
            registry=self.registry
        )
        
        self.stream_consumer_groups = Gauge(
            'redis_stream_consumer_groups_total',
            'Number of consumer groups per stream',
            ['stream_name'],
            registry=self.registry
        )
        
        self.stream_consumers = Gauge(
            'redis_stream_consumers_total',
            'Number of active consumers per stream',
            ['stream_name'],
            registry=self.registry
        )
        
        self.stream_pending = Gauge(
            'redis_stream_pending_messages_total',
            'Number of pending messages per stream',
            ['stream_name'],
            registry=self.registry
        )
        
        self.stream_lag = Gauge(
            'redis_stream_consumer_lag_total',
            'Consumer lag per stream',
            ['stream_name'],
            registry=self.registry
        )
        
        # Throughput metrics
        self.throughput_messages_per_second = Gauge(
            'redis_stream_throughput_messages_per_second',
            'Messages processed per second',
            ['stream_name'],
            registry=self.registry
        )
        
        # Latency metrics
        self.processing_latency = Histogram(
            'redis_stream_processing_latency_seconds',
            'Message processing latency',
            ['stream_name'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Error metrics
        self.error_rate = Gauge(
            'redis_stream_error_rate',
            'Error rate for stream processing',
            ['stream_name'],
            registry=self.registry
        )
        
        # Health metrics
        self.health_score = Gauge(
            'redis_stream_health_score',
            'Stream health score (0.0 to 1.0)',
            ['stream_name'],
            registry=self.registry
        )
        
        # Alert metrics
        self.active_alerts = Gauge(
            'redis_stream_active_alerts_total',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'redis_stream_monitor_info',
            'Stream monitor system information',
            registry=self.registry
        )
        
        self.system_info.info({
            'version': '1.0.0',
            'monitoring_interval': str(self.monitoring_interval),
            'prometheus_enabled': 'true'
        })
    
    async def start(self) -> None:
        """Start stream monitoring."""
        try:
            # Start monitoring task
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop()
            )
            
            # Start cleanup task
            self._metrics_cleanup_task = asyncio.create_task(
                self._cleanup_loop()
            )
            
            logger.info("Stream Monitor started", prometheus_enabled=self.enable_prometheus)
            
        except Exception as e:
            logger.error("Failed to start Stream Monitor", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop stream monitoring."""
        tasks = [self._monitoring_task, self._metrics_cleanup_task]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        completed_tasks = [t for t in tasks if t is not None]
        if completed_tasks:
            await asyncio.gather(*completed_tasks, return_exceptions=True)
        
        logger.info("Stream Monitor stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_metrics(self) -> None:
        """Collect metrics from all monitored streams."""
        current_time = time.time()
        
        try:
            # Discover active streams
            stream_keys = await self.redis.keys("agent_messages:*")
            
            for key in stream_keys:
                if isinstance(key, bytes):
                    key = key.decode()
                
                try:
                    await self._collect_stream_metrics(key, current_time)
                except Exception as e:
                    logger.error(f"Error collecting metrics for {key}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering streams: {e}")
    
    async def _collect_stream_metrics(self, stream_name: str, current_time: float) -> None:
        """Collect comprehensive metrics for a single stream."""
        try:
            # Get basic stream info
            stream_info = await self.redis.xinfo_stream(stream_name)
            stream_length = stream_info["length"]
            
            # Get consumer group info
            try:
                groups_info = await self.redis.xinfo_groups(stream_name)
            except RedisError:
                groups_info = []
            
            # Calculate aggregate metrics
            total_consumers = 0
            total_pending = 0
            total_lag = 0
            oldest_pending_age = 0.0
            
            for group in groups_info:
                # Get consumers for this group
                try:
                    consumers = await self.redis.xinfo_consumers(stream_name, group["name"])
                    total_consumers += len(consumers)
                    
                    for consumer in consumers:
                        total_pending += consumer["pending"]
                        if consumer["idle"] > oldest_pending_age:
                            oldest_pending_age = consumer["idle"] / 1000.0  # Convert to seconds
                
                except RedisError:
                    continue
                
                total_lag += group.get("lag", 0)
            
            # Calculate throughput
            messages_per_second = self._calculate_throughput(stream_name, stream_length, current_time)
            
            # Calculate latency metrics
            latency_metrics = self._calculate_latency_metrics(stream_name)
            
            # Calculate error rate (simplified)
            error_rate = 0.0  # TODO: Implement actual error tracking
            
            # Calculate health score
            health_score = self._calculate_health_score(
                stream_length, total_consumers, total_pending, total_lag,
                messages_per_second, error_rate, latency_metrics["p95"]
            )
            
            # Determine status
            status = self._determine_status(health_score, total_lag, error_rate)
            
            # Create metrics object
            metrics = StreamHealthMetrics(
                stream_name=stream_name,
                length=stream_length,
                consumer_groups=len(groups_info),
                total_consumers=total_consumers,
                total_pending=total_pending,
                total_lag=total_lag,
                messages_per_second=messages_per_second,
                error_rate=error_rate,
                avg_processing_latency_ms=latency_metrics["avg"],
                p95_processing_latency_ms=latency_metrics["p95"],
                p99_processing_latency_ms=latency_metrics["p99"],
                oldest_pending_age_seconds=oldest_pending_age,
                health_score=health_score,
                status=status,
                last_updated=current_time
            )
            
            # Store metrics
            self._stream_metrics[stream_name] = metrics
            
            # Update historical data
            self._update_historical_data(stream_name, metrics, current_time)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self._update_prometheus_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {stream_name}: {e}")
    
    def _calculate_throughput(self, stream_name: str, current_length: int, current_time: float) -> float:
        """Calculate messages per second throughput."""
        
        # Add current sample
        self._throughput_samples[stream_name].append((current_time, current_length))
        
        # Keep only recent samples
        cutoff_time = current_time - (self.throughput_window_minutes * 60)
        self._throughput_samples[stream_name] = [
            (t, length) for t, length in self._throughput_samples[stream_name]
            if t > cutoff_time
        ]
        
        samples = self._throughput_samples[stream_name]
        if len(samples) < 2:
            return 0.0
        
        # Calculate rate based on oldest and newest samples
        oldest_time, oldest_length = samples[0]
        newest_time, newest_length = samples[-1]
        
        time_diff = newest_time - oldest_time
        if time_diff <= 0:
            return 0.0
        
        # Note: This is simplified - in a real system you'd track actual processed messages
        # For now, estimate based on length changes
        length_diff = max(0, newest_length - oldest_length)
        return length_diff / time_diff
    
    def _calculate_latency_metrics(self, stream_name: str) -> Dict[str, float]:
        """Calculate latency percentiles."""
        samples = self._latency_samples.get(stream_name, [])
        
        if not samples:
            return {"avg": 0.0, "p95": 0.0, "p99": 0.0}
        
        samples_sorted = sorted(samples)
        n = len(samples_sorted)
        
        avg = sum(samples_sorted) / n
        p95 = samples_sorted[int(n * 0.95)] if n > 0 else 0.0
        p99 = samples_sorted[int(n * 0.99)] if n > 0 else 0.0
        
        return {"avg": avg, "p95": p95, "p99": p99}
    
    def _calculate_health_score(
        self,
        length: int,
        consumers: int,
        pending: int,
        lag: int,
        throughput: float,
        error_rate: float,
        latency_p95: float
    ) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        
        score = 1.0
        
        # Penalize based on consumer lag
        if lag > 0:
            lag_penalty = min(0.5, lag / 10000.0)  # Max 50% penalty for 10k+ lag
            score -= lag_penalty
        
        # Penalize based on error rate
        error_penalty = min(0.3, error_rate * 6)  # Max 30% penalty at 5% error rate
        score -= error_penalty
        
        # Penalize based on high latency
        if latency_p95 > 100:  # 100ms threshold
            latency_penalty = min(0.2, (latency_p95 - 100) / 5000.0)  # Max 20% penalty
            score -= latency_penalty
        
        # Penalize if no consumers for non-empty stream
        if length > 0 and consumers == 0:
            score -= 0.4
        
        # Penalize very low throughput for active streams
        if length > 100 and throughput < 0.1:
            score -= 0.2
        
        return max(0.0, score)
    
    def _determine_status(self, health_score: float, lag: int, error_rate: float) -> str:
        """Determine stream status based on metrics."""
        if health_score < 0.3 or lag > 10000 or error_rate > 0.1:
            return "critical"
        elif health_score < 0.6 or lag > 1000 or error_rate > 0.05:
            return "warning"
        elif health_score >= 0.8:
            return "healthy"
        else:
            return "ok"
    
    def _update_historical_data(
        self,
        stream_name: str,
        metrics: StreamHealthMetrics,
        current_time: float
    ) -> None:
        """Update historical data for trend analysis."""
        
        history_entry = {
            "timestamp": current_time,
            "length": metrics.length,
            "total_consumers": metrics.total_consumers,
            "total_lag": metrics.total_lag,
            "messages_per_second": metrics.messages_per_second,
            "health_score": metrics.health_score,
            "status": metrics.status
        }
        
        self._historical_data[stream_name].append(history_entry)
        
        # Keep only recent history
        cutoff_time = current_time - (self.history_retention_hours * 3600)
        self._historical_data[stream_name] = [
            entry for entry in self._historical_data[stream_name]
            if entry["timestamp"] > cutoff_time
        ]
    
    def _update_prometheus_metrics(self, metrics: StreamHealthMetrics) -> None:
        """Update Prometheus metrics."""
        if not self.enable_prometheus:
            return
        
        stream_name = metrics.stream_name
        
        self.stream_length.labels(stream_name=stream_name).set(metrics.length)
        self.stream_consumer_groups.labels(stream_name=stream_name).set(metrics.consumer_groups)
        self.stream_consumers.labels(stream_name=stream_name).set(metrics.total_consumers)
        self.stream_pending.labels(stream_name=stream_name).set(metrics.total_pending)
        self.stream_lag.labels(stream_name=stream_name).set(metrics.total_lag)
        self.throughput_messages_per_second.labels(stream_name=stream_name).set(metrics.messages_per_second)
        self.error_rate.labels(stream_name=stream_name).set(metrics.error_rate)
        self.health_score.labels(stream_name=stream_name).set(metrics.health_score)
    
    async def _check_alerts(self) -> None:
        """Check alert conditions and fire alerts."""
        current_time = time.time()
        active_alerts_by_severity = defaultdict(int)
        
        for stream_name, stream_metrics in self._stream_metrics.items():
            for rule in self._alert_rules:
                if not rule.enabled:
                    continue
                
                # Check cooldown
                alert_key = f"{stream_name}:{rule.name}"
                last_fired = self._alert_states.get(alert_key, 0)
                
                if current_time - last_fired < rule.cooldown_seconds:
                    continue
                
                # Evaluate condition
                try:
                    # Create evaluation context
                    eval_context = {
                        "stream_metrics": stream_metrics,
                        "stream_name": stream_name
                    }
                    
                    # Add individual metric fields for easy access
                    for field_name, field_value in asdict(stream_metrics).items():
                        eval_context[field_name] = field_value
                    
                    # Evaluate condition
                    if eval(rule.condition, {"__builtins__": {}}, eval_context):
                        # Fire alert
                        await self._fire_alert(rule, stream_name, stream_metrics)
                        self._alert_states[alert_key] = current_time
                        active_alerts_by_severity[rule.severity] += 1
                
                except Exception as e:
                    logger.error(f"Error evaluating alert rule {rule.name}: {e}")
        
        # Update Prometheus alert metrics
        if self.enable_prometheus:
            for severity in ["warning", "critical"]:
                self.active_alerts.labels(severity=severity).set(
                    active_alerts_by_severity[severity]
                )
    
    async def _fire_alert(
        self,
        rule: AlertRule,
        stream_name: str,
        metrics: StreamHealthMetrics
    ) -> None:
        """Fire an alert."""
        try:
            # Format message
            message_vars = asdict(metrics)
            message_vars["stream_name"] = stream_name
            
            message = rule.message_template.format(**message_vars)
            
            logger.warning(
                f"ALERT [{rule.severity.upper()}]: {rule.name}",
                stream=stream_name,
                message=message,
                rule=rule.name,
                severity=rule.severity
            )
            
        except Exception as e:
            logger.error(f"Error firing alert {rule.name}: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old data periodically."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old historical data and samples."""
        current_time = time.time()
        cutoff_time = current_time - (self.history_retention_hours * 3600)
        
        # Clean up latency samples
        for stream_name in list(self._latency_samples.keys()):
            if len(self._latency_samples[stream_name]) > self.latency_sample_size:
                self._latency_samples[stream_name] = self._latency_samples[stream_name][-self.latency_sample_size:]
        
        # Clean up old alert states
        old_alerts = [
            key for key, timestamp in self._alert_states.items()
            if current_time - timestamp > 3600  # 1 hour
        ]
        
        for key in old_alerts:
            del self._alert_states[key]
    
    def record_processing_latency(self, stream_name: str, latency_ms: float) -> None:
        """Record a processing latency sample."""
        self._latency_samples[stream_name].append(latency_ms)
        
        # Keep only recent samples
        if len(self._latency_samples[stream_name]) > self.latency_sample_size:
            self._latency_samples[stream_name] = self._latency_samples[stream_name][-self.latency_sample_size:]
        
        # Update Prometheus histogram
        if self.enable_prometheus:
            self.processing_latency.labels(stream_name=stream_name).observe(latency_ms / 1000.0)
    
    def get_metrics(self, stream_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current metrics for streams."""
        if stream_name:
            metrics = self._stream_metrics.get(stream_name)
            return asdict(metrics) if metrics else {}
        
        return {
            name: asdict(metrics)
            for name, metrics in self._stream_metrics.items()
        }
    
    def get_historical_data(
        self,
        stream_name: str,
        hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Get historical data for a stream."""
        if stream_name not in self._historical_data:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        return [
            entry for entry in self._historical_data[stream_name]
            if entry["timestamp"] > cutoff_time
        ]
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not self.enable_prometheus:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self._alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        original_count = len(self._alert_rules)
        self._alert_rules = [rule for rule in self._alert_rules if rule.name != rule_name]
        return len(self._alert_rules) < original_count
    
    def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return self._alert_rules.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        if not self._stream_metrics:
            return {
                "status": "no_data",
                "total_streams": 0,
                "healthy_streams": 0,
                "warning_streams": 0,
                "critical_streams": 0,
                "overall_health_score": 0.0
            }
        
        status_counts = defaultdict(int)
        total_health = 0.0
        
        for metrics in self._stream_metrics.values():
            status_counts[metrics.status] += 1
            total_health += metrics.health_score
        
        avg_health = total_health / len(self._stream_metrics)
        
        # Determine overall status
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        elif avg_health >= 0.8:
            overall_status = "healthy"
        else:
            overall_status = "ok"
        
        return {
            "status": overall_status,
            "total_streams": len(self._stream_metrics),
            "healthy_streams": status_counts["healthy"],
            "warning_streams": status_counts["warning"],
            "critical_streams": status_counts["critical"],
            "overall_health_score": avg_health
        }