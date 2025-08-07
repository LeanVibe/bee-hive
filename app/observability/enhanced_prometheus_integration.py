"""
Enhanced Prometheus Integration for Real-Time Observability
==========================================================

Extends the existing Prometheus metrics with real-time event processing metrics,
WebSocket streaming performance, and comprehensive observability KPIs that align
with the PRD requirements for enterprise-grade monitoring.

Performance Targets Integration:
- Event processing latency: <150ms P95 monitoring
- WebSocket streaming: <1s update rate tracking  
- System overhead: <3% CPU monitoring per agent
- Event coverage: 100% lifecycle event capture validation
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, Info

from app.observability.prometheus_exporter import get_metrics_exporter
from app.observability.real_time_hooks import get_real_time_processor
from app.observability.enhanced_websocket_streaming import get_enhanced_websocket_streaming

logger = structlog.get_logger()


class EnhancedPrometheusMetrics:
    """
    Enhanced Prometheus metrics specifically for real-time observability system.
    
    Extends existing metrics with:
    - Real-time event processing performance
    - WebSocket streaming metrics  
    - Observability system health indicators
    - Enterprise-grade KPIs and SLIs
    """
    
    def __init__(self, base_exporter):
        self.base_exporter = base_exporter
        
        # === REAL-TIME EVENT PROCESSING METRICS ===
        self.realtime_events_processed_total = Counter(
            'leanvibe_realtime_events_processed_total',
            'Total real-time events processed by type and status',
            ['event_type', 'status', 'priority'],
            registry=base_exporter.registry
        )
        
        self.realtime_event_processing_latency_seconds = Histogram(
            'leanvibe_realtime_event_processing_latency_seconds',
            'Real-time event processing latency in seconds',
            ['event_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0],
            registry=base_exporter.registry
        )
        
        self.realtime_event_buffer_size = Gauge(
            'leanvibe_realtime_event_buffer_size',
            'Current size of real-time event buffer',
            registry=base_exporter.registry
        )
        
        self.realtime_event_buffer_overflows_total = Counter(
            'leanvibe_realtime_event_buffer_overflows_total',
            'Total event buffer overflow occurrences',
            registry=base_exporter.registry
        )
        
        self.realtime_event_retry_count = Counter(
            'leanvibe_realtime_event_retry_count',
            'Total event processing retries by reason',
            ['retry_reason'],
            registry=base_exporter.registry
        )
        
        self.realtime_events_per_second = Gauge(
            'leanvibe_realtime_events_per_second',
            'Current real-time event processing rate per second',
            registry=base_exporter.registry
        )
        
        # === WEBSOCKET STREAMING METRICS ===
        self.websocket_stream_connections_total = Gauge(
            'leanvibe_websocket_stream_connections_total',
            'Total active WebSocket streaming connections',
            ['connection_type', 'filter_active'],
            registry=base_exporter.registry
        )
        
        self.websocket_stream_events_broadcast_total = Counter(
            'leanvibe_websocket_stream_events_broadcast_total',
            'Total events broadcast to WebSocket clients',
            ['event_type', 'result'],  # result: sent, filtered, rate_limited
            registry=base_exporter.registry
        )
        
        self.websocket_stream_latency_seconds = Histogram(
            'leanvibe_websocket_stream_latency_seconds',
            'WebSocket streaming latency from event to client delivery',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=base_exporter.registry
        )
        
        self.websocket_stream_message_size_bytes = Histogram(
            'leanvibe_websocket_stream_message_size_bytes',
            'WebSocket message size distribution',
            ['event_type'],
            buckets=[128, 512, 1024, 4096, 16384, 65536, 262144],
            registry=base_exporter.registry
        )
        
        self.websocket_stream_rate_limited_events_total = Counter(
            'leanvibe_websocket_stream_rate_limited_events_total',
            'Total events that were rate limited',
            ['connection_type'],
            registry=base_exporter.registry
        )
        
        self.websocket_stream_connection_duration_seconds = Histogram(
            'leanvibe_websocket_stream_connection_duration_seconds',
            'WebSocket streaming connection duration',
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
            registry=base_exporter.registry
        )
        
        # === OBSERVABILITY SYSTEM HEALTH ===
        self.observability_component_health = Gauge(
            'leanvibe_observability_component_health',
            'Observability component health score (0-1)',
            ['component'],  # real_time_processor, websocket_streaming, event_hooks
            registry=base_exporter.registry
        )
        
        self.observability_sli_compliance = Gauge(
            'leanvibe_observability_sli_compliance',
            'Observability SLI compliance score (0-1)',
            ['sli_name'],  # latency_p95, event_coverage, update_rate, cpu_overhead
            registry=base_exporter.registry
        )
        
        self.observability_alerts_triggered_total = Counter(
            'leanvibe_observability_alerts_triggered_total',
            'Total observability system alerts triggered',
            ['alert_type', 'severity', 'component'],
            registry=base_exporter.registry
        )
        
        # === PERFORMANCE TARGET MONITORING ===
        self.performance_target_compliance = Gauge(
            'leanvibe_performance_target_compliance',
            'Performance target compliance status (0=failing, 1=meeting)',
            ['target_name'],  # p95_latency_150ms, coverage_100pct, cpu_overhead_3pct
            registry=base_exporter.registry
        )
        
        self.event_coverage_percentage = Gauge(
            'leanvibe_event_coverage_percentage',
            'Percentage of lifecycle events captured (0-100)',
            registry=base_exporter.registry
        )
        
        self.cpu_overhead_percentage = Gauge(
            'leanvibe_cpu_overhead_percentage',
            'CPU overhead percentage from observability system',
            ['component'],
            registry=base_exporter.registry
        )
        
        # === DASHBOARD INTEGRATION METRICS ===
        self.dashboard_updates_per_second = Gauge(
            'leanvibe_dashboard_updates_per_second',
            'Dashboard update rate per second',
            ['dashboard_type'],
            registry=base_exporter.registry
        )
        
        self.dashboard_client_errors_total = Counter(
            'leanvibe_dashboard_client_errors_total',
            'Total dashboard client errors',
            ['error_type', 'dashboard_type'],
            registry=base_exporter.registry
        )
        
        self.dashboard_data_freshness_seconds = Gauge(
            'leanvibe_dashboard_data_freshness_seconds',
            'Dashboard data freshness (age in seconds)',
            ['dashboard_type'],
            registry=base_exporter.registry
        )
        
        # === AGENT LIFECYCLE HOOK METRICS ===
        self.agent_lifecycle_events_total = Counter(
            'leanvibe_agent_lifecycle_events_total',
            'Total agent lifecycle events by hook type',
            ['hook_type', 'agent_id', 'status'],  # hook_type: PreToolUse, PostToolUse, etc.
            registry=base_exporter.registry
        )
        
        self.agent_tool_execution_latency_seconds = Histogram(
            'leanvibe_agent_tool_execution_latency_seconds',
            'Agent tool execution latency measured by hooks',
            ['tool_name', 'agent_id'],
            buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=base_exporter.registry
        )
        
        self.agent_tool_success_rate = Gauge(
            'leanvibe_agent_tool_success_rate',
            'Agent tool success rate by tool and agent',
            ['tool_name', 'agent_id'],
            registry=base_exporter.registry
        )
        
        self.agent_session_active_duration_seconds = Gauge(
            'leanvibe_agent_session_active_duration_seconds',
            'Current active session duration',
            ['session_id', 'agent_id'],
            registry=base_exporter.registry
        )
        
        # === ENTERPRISE KPIs ===
        self.enterprise_availability_percentage = Gauge(
            'leanvibe_enterprise_availability_percentage',
            'Enterprise availability percentage',
            ['service_component'],
            registry=base_exporter.registry
        )
        
        self.enterprise_mttr_seconds = Gauge(
            'leanvibe_enterprise_mttr_seconds',
            'Mean Time To Recovery in seconds',
            ['incident_type'],
            registry=base_exporter.registry
        )
        
        self.enterprise_mttd_seconds = Gauge(
            'leanvibe_enterprise_mttd_seconds',
            'Mean Time To Detection in seconds',
            ['alert_type'],
            registry=base_exporter.registry
        )
        
        logger.info("Enhanced Prometheus metrics initialized")
    
    async def collect_real_time_processor_metrics(self) -> None:
        """Collect metrics from the real-time event processor."""
        try:
            processor = await get_real_time_processor()
            metrics = processor.get_performance_metrics()
            
            # Event processing metrics
            self.realtime_events_per_second.set(metrics.get("events_per_second", 0))
            
            # Buffer metrics
            buffer_metrics = metrics.get("buffer_metrics", {})
            self.realtime_event_buffer_size.set(
                buffer_metrics.get("events_buffered", 0) - buffer_metrics.get("events_flushed", 0)
            )
            
            # Buffer overflow tracking
            overflow_count = buffer_metrics.get("buffer_overflows", 0)
            self.realtime_event_buffer_overflows_total._value._value = overflow_count
            
            # Performance target compliance
            p95_latency = metrics.get("p95_processing_latency_ms", 0)
            meets_p95_target = p95_latency <= 150
            self.performance_target_compliance.labels(target_name="p95_latency_150ms").set(
                1.0 if meets_p95_target else 0.0
            )
            
            # Success rate compliance
            success_rate = metrics["performance_targets"].get("current_success_rate", 0)
            self.event_coverage_percentage.set(success_rate * 100)
            
            meets_coverage = success_rate >= 0.999  # 99.9% coverage target
            self.performance_target_compliance.labels(target_name="coverage_100pct").set(
                1.0 if meets_coverage else 0.0
            )
            
            # Component health score
            health_score = min(1.0, success_rate) if meets_p95_target else 0.7
            self.observability_component_health.labels(component="real_time_processor").set(health_score)
            
            # SLI compliance scores
            self.observability_sli_compliance.labels(sli_name="latency_p95").set(
                1.0 if meets_p95_target else max(0.0, (200 - p95_latency) / 200)
            )
            self.observability_sli_compliance.labels(sli_name="event_coverage").set(success_rate)
            
        except Exception as e:
            logger.error("Failed to collect real-time processor metrics", error=str(e))
            self.observability_component_health.labels(component="real_time_processor").set(0.0)
    
    async def collect_websocket_streaming_metrics(self) -> None:
        """Collect metrics from the WebSocket streaming system."""
        try:
            streaming = await get_enhanced_websocket_streaming()
            metrics = streaming.get_metrics()
            
            # Connection metrics
            active_connections = metrics.get("active_connections", 0)
            self.websocket_stream_connections_total.labels(
                connection_type="dashboard", 
                filter_active="true"
            ).set(active_connections)
            
            # Streaming performance metrics
            avg_latency_ms = metrics.get("average_stream_latency_ms", 0)
            events_per_second = metrics.get("events_per_second", 0)
            
            self.dashboard_updates_per_second.labels(dashboard_type="real_time").set(events_per_second)
            
            # Rate limiting metrics
            rate_limited = metrics.get("rate_limited_events", 0)
            self.websocket_stream_rate_limited_events_total._value._value = rate_limited
            
            # Performance target compliance for <1s updates
            meets_update_rate = avg_latency_ms <= 1000
            self.observability_sli_compliance.labels(sli_name="update_rate").set(
                1.0 if meets_update_rate else max(0.0, (2000 - avg_latency_ms) / 2000)
            )
            
            # Component health score
            error_rate = metrics.get("websocket_errors", 0) / max(metrics.get("total_events_streamed", 1), 1)
            connection_capacity = 1.0 - (active_connections / streaming.config["max_connections"])
            latency_score = 1.0 if meets_update_rate else 0.7
            
            health_score = (latency_score + connection_capacity + (1.0 - error_rate)) / 3
            self.observability_component_health.labels(component="websocket_streaming").set(health_score)
            
        except Exception as e:
            logger.error("Failed to collect WebSocket streaming metrics", error=str(e))
            self.observability_component_health.labels(component="websocket_streaming").set(0.0)
    
    async def collect_cpu_overhead_metrics(self) -> None:
        """Collect CPU overhead metrics for observability system."""
        try:
            import psutil
            
            # Get current process CPU usage (this is a simplified approach)
            current_process = psutil.Process()
            cpu_percent = current_process.cpu_percent(interval=1.0)
            
            # Component-specific CPU overhead (estimated)
            self.cpu_overhead_percentage.labels(component="real_time_processor").set(cpu_percent * 0.4)
            self.cpu_overhead_percentage.labels(component="websocket_streaming").set(cpu_percent * 0.3)
            self.cpu_overhead_percentage.labels(component="event_hooks").set(cpu_percent * 0.3)
            
            # Overall CPU overhead target compliance (<3% per agent)
            total_overhead = cpu_percent
            meets_cpu_target = total_overhead <= 3.0
            
            self.performance_target_compliance.labels(target_name="cpu_overhead_3pct").set(
                1.0 if meets_cpu_target else 0.0
            )
            
            self.observability_sli_compliance.labels(sli_name="cpu_overhead").set(
                1.0 if meets_cpu_target else max(0.0, (6.0 - total_overhead) / 6.0)
            )
            
        except Exception as e:
            logger.error("Failed to collect CPU overhead metrics", error=str(e))
    
    async def collect_enterprise_kpi_metrics(self) -> None:
        """Collect enterprise-grade KPI metrics."""
        try:
            # Calculate system availability (simplified)
            components = ["real_time_processor", "websocket_streaming", "event_hooks"]
            availability_scores = []
            
            for component in components:
                health = self.observability_component_health.labels(component=component)._value._value
                availability = 99.9 if health > 0.8 else (95.0 if health > 0.5 else 90.0)
                availability_scores.append(availability)
                
                self.enterprise_availability_percentage.labels(
                    service_component=component
                ).set(availability)
            
            # Overall system availability
            overall_availability = min(availability_scores)
            self.enterprise_availability_percentage.labels(
                service_component="observability_system"
            ).set(overall_availability)
            
            # MTTR metrics (example values - would be calculated from actual incident data)
            self.enterprise_mttr_seconds.labels(incident_type="performance_degradation").set(300)  # 5 minutes
            self.enterprise_mttr_seconds.labels(incident_type="component_failure").set(600)  # 10 minutes
            
            # MTTD metrics (example values - would be calculated from actual alert data)
            self.enterprise_mttd_seconds.labels(alert_type="performance_threshold").set(60)  # 1 minute
            self.enterprise_mttd_seconds.labels(alert_type="component_health").set(30)  # 30 seconds
            
        except Exception as e:
            logger.error("Failed to collect enterprise KPI metrics", error=str(e))
    
    def record_agent_lifecycle_event(
        self, 
        hook_type: str, 
        agent_id: str, 
        status: str = "success",
        tool_name: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> None:
        """Record agent lifecycle event metrics."""
        self.agent_lifecycle_events_total.labels(
            hook_type=hook_type,
            agent_id=agent_id,
            status=status
        ).inc()
        
        if tool_name and execution_time_ms is not None:
            execution_time_seconds = execution_time_ms / 1000.0
            self.agent_tool_execution_latency_seconds.labels(
                tool_name=tool_name,
                agent_id=agent_id
            ).observe(execution_time_seconds)
    
    def record_websocket_event_broadcast(
        self, 
        event_type: str, 
        result: str,  # sent, filtered, rate_limited
        latency_ms: float,
        message_size: int
    ) -> None:
        """Record WebSocket event broadcast metrics."""
        self.websocket_stream_events_broadcast_total.labels(
            event_type=event_type,
            result=result
        ).inc()
        
        if result == "sent":
            self.websocket_stream_latency_seconds.observe(latency_ms / 1000.0)
            self.websocket_stream_message_size_bytes.labels(event_type=event_type).observe(message_size)
    
    def record_real_time_event_processing(
        self,
        event_type: str,
        status: str,
        priority: str,
        processing_latency_ms: float
    ) -> None:
        """Record real-time event processing metrics."""
        self.realtime_events_processed_total.labels(
            event_type=event_type,
            status=status,
            priority=priority
        ).inc()
        
        self.realtime_event_processing_latency_seconds.labels(
            event_type=event_type
        ).observe(processing_latency_ms / 1000.0)
    
    def trigger_observability_alert(
        self,
        alert_type: str,
        severity: str,
        component: str
    ) -> None:
        """Record observability alert triggers."""
        self.observability_alerts_triggered_total.labels(
            alert_type=alert_type,
            severity=severity,
            component=component
        ).inc()
    
    async def collect_all_enhanced_metrics(self) -> None:
        """Collect all enhanced observability metrics."""
        await asyncio.gather(
            self.collect_real_time_processor_metrics(),
            self.collect_websocket_streaming_metrics(),
            self.collect_cpu_overhead_metrics(),
            self.collect_enterprise_kpi_metrics(),
            return_exceptions=True
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring dashboards."""
        try:
            return {
                "targets": {
                    "p95_latency_150ms": bool(self.performance_target_compliance.labels(
                        target_name="p95_latency_150ms"
                    )._value._value),
                    "coverage_100pct": bool(self.performance_target_compliance.labels(
                        target_name="coverage_100pct"
                    )._value._value),
                    "cpu_overhead_3pct": bool(self.performance_target_compliance.labels(
                        target_name="cpu_overhead_3pct"
                    )._value._value),
                },
                "sli_scores": {
                    "latency_p95": self.observability_sli_compliance.labels(sli_name="latency_p95")._value._value,
                    "event_coverage": self.observability_sli_compliance.labels(sli_name="event_coverage")._value._value,
                    "update_rate": self.observability_sli_compliance.labels(sli_name="update_rate")._value._value,
                    "cpu_overhead": self.observability_sli_compliance.labels(sli_name="cpu_overhead")._value._value,
                },
                "component_health": {
                    "real_time_processor": self.observability_component_health.labels(
                        component="real_time_processor"
                    )._value._value,
                    "websocket_streaming": self.observability_component_health.labels(
                        component="websocket_streaming"
                    )._value._value,
                },
                "enterprise_kpis": {
                    "availability": self.enterprise_availability_percentage.labels(
                        service_component="observability_system"
                    )._value._value,
                    "mttr_performance": self.enterprise_mttr_seconds.labels(
                        incident_type="performance_degradation"
                    )._value._value,
                    "mttd_performance": self.enterprise_mttd_seconds.labels(
                        alert_type="performance_threshold"
                    )._value._value,
                }
            }
        except Exception as e:
            logger.error("Failed to get performance summary", error=str(e))
            return {"error": str(e)}


# Global enhanced metrics instance
_enhanced_metrics: Optional[EnhancedPrometheusMetrics] = None


def get_enhanced_prometheus_metrics() -> EnhancedPrometheusMetrics:
    """Get global enhanced Prometheus metrics instance."""
    global _enhanced_metrics
    
    if _enhanced_metrics is None:
        base_exporter = get_metrics_exporter()
        _enhanced_metrics = EnhancedPrometheusMetrics(base_exporter)
    
    return _enhanced_metrics


async def update_event_metrics(
    event_type: str,
    success: Optional[bool],
    latency_ms: Optional[int],
    tool_name: Optional[str] = None
) -> None:
    """
    Update event metrics from observability middleware.
    
    This function is called by the observability middleware to update
    Prometheus metrics when events are processed.
    """
    try:
        metrics = get_enhanced_prometheus_metrics()
        
        if success is not None:
            status = "success" if success else "error"
            priority = "high" if not success else "normal"
            
            if latency_ms is not None:
                metrics.record_real_time_event_processing(
                    event_type=event_type,
                    status=status,
                    priority=priority,
                    processing_latency_ms=float(latency_ms)
                )
        
        # Record tool-specific metrics if available
        if tool_name and latency_ms is not None:
            base_metrics = get_metrics_exporter()
            base_metrics.record_tool_execution(
                tool_name=tool_name,
                status=status,
                duration=latency_ms / 1000.0
            )
            
    except Exception as e:
        logger.error("Failed to update event metrics", error=str(e))


# Convenience functions for integration

async def record_pre_tool_use_metrics(
    agent_id: str, 
    tool_name: str, 
    start_time: datetime
) -> None:
    """Record PreToolUse metrics."""
    try:
        metrics = get_enhanced_prometheus_metrics()
        metrics.record_agent_lifecycle_event(
            hook_type="PreToolUse",
            agent_id=agent_id,
            tool_name=tool_name
        )
    except Exception as e:
        logger.error("Failed to record PreToolUse metrics", error=str(e))


async def record_post_tool_use_metrics(
    agent_id: str,
    tool_name: str,
    success: bool,
    execution_time_ms: float
) -> None:
    """Record PostToolUse metrics."""
    try:
        metrics = get_enhanced_prometheus_metrics()
        status = "success" if success else "error"
        
        metrics.record_agent_lifecycle_event(
            hook_type="PostToolUse",
            agent_id=agent_id,
            status=status,
            tool_name=tool_name,
            execution_time_ms=execution_time_ms
        )
        
        # Update base metrics for compatibility
        base_metrics = get_metrics_exporter()
        base_metrics.record_tool_execution(
            tool_name=tool_name,
            status=status,
            duration=execution_time_ms / 1000.0
        )
        
    except Exception as e:
        logger.error("Failed to record PostToolUse metrics", error=str(e))


async def record_websocket_broadcast_metrics(
    event_type: str,
    sent_count: int,
    filtered_count: int,
    rate_limited_count: int,
    avg_latency_ms: float,
    avg_message_size: int
) -> None:
    """Record WebSocket broadcast metrics."""
    try:
        metrics = get_enhanced_prometheus_metrics()
        
        if sent_count > 0:
            metrics.record_websocket_event_broadcast(
                event_type=event_type,
                result="sent",
                latency_ms=avg_latency_ms,
                message_size=avg_message_size
            )
        
        if filtered_count > 0:
            for _ in range(filtered_count):
                metrics.record_websocket_event_broadcast(
                    event_type=event_type,
                    result="filtered", 
                    latency_ms=0,
                    message_size=0
                )
        
        if rate_limited_count > 0:
            for _ in range(rate_limited_count):
                metrics.record_websocket_event_broadcast(
                    event_type=event_type,
                    result="rate_limited",
                    latency_ms=0,
                    message_size=0
                )
                
    except Exception as e:
        logger.error("Failed to record WebSocket broadcast metrics", error=str(e))


# Background task for continuous metrics collection
async def start_enhanced_metrics_collection():
    """Start background task for enhanced metrics collection."""
    async def metrics_collection_loop():
        while True:
            try:
                metrics = get_enhanced_prometheus_metrics()
                await metrics.collect_all_enhanced_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(60)  # Longer sleep on error
    
    # Start background task
    asyncio.create_task(metrics_collection_loop())
    logger.info("Enhanced metrics collection started")