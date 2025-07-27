"""
Prometheus metrics exporter for LeanVibe Agent Hive 2.0

Comprehensive metrics collection and exposition for production monitoring,
including system health, performance, agent activities, and business metrics.
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from sqlalchemy import text
from fastapi import Response

from app.core.database import get_async_session
from app.core.redis import get_redis
from app.core.event_processor import get_event_processor

logger = structlog.get_logger()

class PrometheusMetricsExporter:
    """
    Comprehensive Prometheus metrics exporter for Agent Hive observability.
    
    Collects and exposes metrics across all system components including:
    - HTTP request/response metrics
    - Agent and session metrics  
    - Database and Redis performance
    - Event processing metrics
    - System resource utilization
    - Business logic metrics
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize the metrics exporter with comprehensive metrics."""
        self.registry = registry or CollectorRegistry()
        
        # === HTTP REQUEST METRICS ===
        self.http_requests_total = Counter(
            'leanvibe_http_requests_total',
            'Total HTTP requests processed',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'leanvibe_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.http_request_size_bytes = Histogram(
            'leanvibe_http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            buckets=[64, 256, 1024, 4096, 16384, 65536, 262144, 1048576],
            registry=self.registry
        )
        
        self.http_response_size_bytes = Histogram(
            'leanvibe_http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        # === AGENT METRICS ===
        self.active_agents_total = Gauge(
            'leanvibe_active_agents_total',
            'Number of currently active agents',
            registry=self.registry
        )
        
        self.agent_operations_total = Counter(
            'leanvibe_agent_operations_total',
            'Total agent operations performed',
            ['agent_id', 'operation_type', 'status'],
            registry=self.registry
        )
        
        self.agent_execution_duration_seconds = Histogram(
            'leanvibe_agent_execution_duration_seconds',
            'Agent operation execution time in seconds',
            ['agent_id', 'operation_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )
        
        # === SESSION METRICS ===
        self.active_sessions_total = Gauge(
            'leanvibe_active_sessions_total',
            'Number of currently active sessions',
            registry=self.registry
        )
        
        self.session_duration_seconds = Histogram(
            'leanvibe_session_duration_seconds',
            'Session duration in seconds',
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800],
            registry=self.registry
        )
        
        self.session_events_total = Counter(
            'leanvibe_session_events_total',
            'Total events per session',
            ['session_id', 'event_type'],
            registry=self.registry
        )
        
        # === EVENT PROCESSING METRICS ===
        self.events_processed_total = Counter(
            'leanvibe_events_processed_total',
            'Total events processed',
            ['event_type', 'status'],
            registry=self.registry
        )
        
        self.event_processing_duration_seconds = Histogram(
            'leanvibe_event_processing_duration_seconds',
            'Event processing duration in seconds',
            ['event_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.event_queue_size = Gauge(
            'leanvibe_event_queue_size',
            'Number of events waiting in queue',
            registry=self.registry
        )
        
        self.event_processing_rate_per_second = Gauge(
            'leanvibe_event_processing_rate_per_second',
            'Current event processing rate per second',
            registry=self.registry
        )
        
        # === TASK METRICS ===
        self.tasks_total = Counter(
            'leanvibe_tasks_total',
            'Total tasks created',
            ['task_type', 'status'],
            registry=self.registry
        )
        
        self.tasks_in_progress = Gauge(
            'leanvibe_tasks_in_progress',
            'Number of tasks currently in progress',
            ['task_type'],
            registry=self.registry
        )
        
        self.task_execution_duration_seconds = Histogram(
            'leanvibe_task_execution_duration_seconds',
            'Task execution duration in seconds',
            ['task_type'],
            buckets=[1, 5, 10, 30, 60, 300, 900, 1800, 3600],
            registry=self.registry
        )
        
        # === TOOL EXECUTION METRICS ===
        self.tool_executions_total = Counter(
            'leanvibe_tool_executions_total',
            'Total tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        self.tool_execution_duration_seconds = Histogram(
            'leanvibe_tool_execution_duration_seconds',
            'Tool execution duration in seconds',
            ['tool_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.tool_success_rate = Gauge(
            'leanvibe_tool_success_rate',
            'Tool execution success rate (0-1)',
            ['tool_name'],
            registry=self.registry
        )
        
        # === DATABASE METRICS ===
        self.database_connections_active = Gauge(
            'leanvibe_database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'leanvibe_database_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.database_queries_total = Counter(
            'leanvibe_database_queries_total',
            'Total database queries executed',
            ['query_type', 'status'],
            registry=self.registry
        )
        
        # === REDIS METRICS ===
        self.redis_connections_active = Gauge(
            'leanvibe_redis_connections_active',
            'Number of active Redis connections',
            registry=self.registry
        )
        
        self.redis_memory_used_bytes = Gauge(
            'leanvibe_redis_memory_used_bytes',
            'Redis memory usage in bytes',
            registry=self.registry
        )
        
        self.redis_operations_total = Counter(
            'leanvibe_redis_operations_total',
            'Total Redis operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.redis_operation_duration_seconds = Histogram(
            'leanvibe_redis_operation_duration_seconds',
            'Redis operation duration in seconds',
            ['operation_type'],
            registry=self.registry
        )
        
        # === WEBSOCKET METRICS ===
        self.websocket_connections_active = Gauge(
            'leanvibe_websocket_connections_active',
            'Number of active WebSocket connections',
            ['connection_type'],
            registry=self.registry
        )
        
        self.websocket_messages_total = Counter(
            'leanvibe_websocket_messages_total',
            'Total WebSocket messages',
            ['connection_type', 'message_type', 'direction'],
            registry=self.registry
        )
        
        self.websocket_connection_duration_seconds = Histogram(
            'leanvibe_websocket_connection_duration_seconds',
            'WebSocket connection duration in seconds',
            ['connection_type'],
            registry=self.registry
        )
        
        # === SYSTEM RESOURCE METRICS ===
        self.system_cpu_usage_percent = Gauge(
            'leanvibe_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage_bytes = Gauge(
            'leanvibe_system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],  # total, available, used, free
            registry=self.registry
        )
        
        self.system_disk_usage_bytes = Gauge(
            'leanvibe_system_disk_usage_bytes',
            'System disk usage in bytes',
            ['device', 'type'],  # total, used, free
            registry=self.registry
        )
        
        self.system_network_bytes_total = Counter(
            'leanvibe_system_network_bytes_total',
            'Total network bytes',
            ['interface', 'direction'],  # sent, received
            registry=self.registry
        )
        
        # === APPLICATION HEALTH METRICS ===
        self.application_info = Info(
            'leanvibe_application',
            'Application information',
            registry=self.registry
        )
        
        self.component_health_status = Enum(
            'leanvibe_component_health_status',
            'Component health status',
            ['component'],
            states=['healthy', 'degraded', 'unhealthy', 'unknown'],
            registry=self.registry
        )
        
        self.application_uptime_seconds = Gauge(
            'leanvibe_application_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        self.application_start_time_seconds = Gauge(
            'leanvibe_application_start_time_seconds',
            'Application start time as Unix timestamp',
            registry=self.registry
        )
        
        # === BUSINESS METRICS ===
        self.workflow_executions_total = Counter(
            'leanvibe_workflow_executions_total',
            'Total workflow executions',
            ['workflow_type', 'status'],
            registry=self.registry
        )
        
        self.workflow_step_duration_seconds = Histogram(
            'leanvibe_workflow_step_duration_seconds',
            'Workflow step execution time in seconds',
            ['workflow_type', 'step_name'],
            registry=self.registry
        )
        
        self.context_operations_total = Counter(
            'leanvibe_context_operations_total',
            'Total context operations',
            ['operation_type', 'context_type'],
            registry=self.registry
        )
        
        # === ERROR AND ALERTING METRICS ===
        self.errors_total = Counter(
            'leanvibe_errors_total',
            'Total errors by component and type',
            ['component', 'error_type', 'severity'],
            registry=self.registry
        )
        
        self.alerts_triggered_total = Counter(
            'leanvibe_alerts_triggered_total',
            'Total alerts triggered',
            ['alert_name', 'severity', 'component'],
            registry=self.registry
        )
        
        # === PERFORMANCE TRACKING ===
        self.performance_percentiles = Summary(
            'leanvibe_performance_percentiles_seconds',
            'Performance percentiles for operations',
            ['operation_type'],
            registry=self.registry
        )
        
        # Initialize application info
        self.application_info.info({
            'version': '2.0.0',
            'build_date': datetime.utcnow().isoformat(),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            'environment': 'development'  # Should be configurable
        })
        
        # Set application start time
        self.application_start_time_seconds.set(time.time())
        
        logger.info("📊 Prometheus metrics exporter initialized with comprehensive metrics")
    
    async def collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage_percent.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage_bytes.labels(type='total').set(memory.total)
            self.system_memory_usage_bytes.labels(type='available').set(memory.available)
            self.system_memory_usage_bytes.labels(type='used').set(memory.used)
            self.system_memory_usage_bytes.labels(type='free').set(memory.free)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    disk_usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device
                    self.system_disk_usage_bytes.labels(device=device, type='total').set(disk_usage.total)
                    self.system_disk_usage_bytes.labels(device=device, type='used').set(disk_usage.used)
                    self.system_disk_usage_bytes.labels(device=device, type='free').set(disk_usage.free)
                except PermissionError:
                    # Skip partitions we can't access
                    pass
            
            # Network usage
            network = psutil.net_io_counters(pernic=True)
            for interface, stats in network.items():
                self.system_network_bytes_total.labels(interface=interface, direction='sent').inc(stats.bytes_sent)
                self.system_network_bytes_total.labels(interface=interface, direction='received').inc(stats.bytes_recv)
            
            # Application uptime
            current_time = time.time()
            start_time = self.application_start_time_seconds._value._value
            uptime = current_time - start_time
            self.application_uptime_seconds.set(uptime)
            
        except Exception as e:
            logger.error("📊 Failed to collect system metrics", error=str(e))
    
    async def collect_database_metrics(self):
        """Collect database performance metrics."""
        try:
            async for session in get_async_session():
                # Database connection count
                result = await session.execute(text("""
                    SELECT count(*) as active_connections
                    FROM pg_stat_activity 
                    WHERE state = 'active'
                """))
                active_connections = result.scalar()
                self.database_connections_active.set(active_connections)
                
                # Query performance stats
                result = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        seq_scan + idx_scan as total_scans,
                        n_tup_ins + n_tup_upd + n_tup_del as total_modifications
                    FROM pg_stat_user_tables
                    ORDER BY total_scans DESC
                    LIMIT 10
                """))
                
                # Count total operations by table (could be expanded)
                for row in result.fetchall():
                    # This is a simplified example - in production you'd want more detailed metrics
                    pass
                
                break  # Only need one session for metrics
                
        except Exception as e:
            logger.error("📊 Failed to collect database metrics", error=str(e))
    
    async def collect_redis_metrics(self):
        """Collect Redis performance metrics."""
        try:
            redis_client = get_redis()
            
            # Redis info
            info = await redis_client.info()
            
            # Memory usage
            used_memory = info.get('used_memory', 0)
            self.redis_memory_used_bytes.set(used_memory)
            
            # Connection count
            connected_clients = info.get('connected_clients', 0)
            self.redis_connections_active.set(connected_clients)
            
            # Additional Redis metrics could be added here
            
        except Exception as e:
            logger.error("📊 Failed to collect Redis metrics", error=str(e))
    
    async def collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Collect metrics from event processor
            event_processor = get_event_processor()
            if event_processor:
                health = await event_processor.health_check()
                
                self.events_processed_total.labels(event_type='all', status='success').inc(
                    health.get('events_processed', 0)
                )
                self.events_processed_total.labels(event_type='all', status='failed').inc(
                    health.get('events_failed', 0)
                )
                self.event_processing_rate_per_second.set(
                    health.get('processing_rate_per_second', 0)
                )
            
            # Component health status
            components = {
                'database': 'healthy',  # Should check actual health
                'redis': 'healthy',     # Should check actual health
                'event_processor': 'healthy' if event_processor else 'unhealthy',
                'websocket': 'healthy'  # Should check actual health
            }
            
            for component, status in components.items():
                self.component_health_status.labels(component=component).state(status)
            
        except Exception as e:
            logger.error("📊 Failed to collect application metrics", error=str(e))
    
    async def collect_all_metrics(self):
        """Collect all metrics from all sources."""
        await asyncio.gather(
            self.collect_system_metrics(),
            self.collect_database_metrics(),
            self.collect_redis_metrics(),
            self.collect_application_metrics(),
            return_exceptions=True
        )
    
    def generate_metrics_response(self) -> Response:
        """Generate Prometheus metrics response."""
        try:
            metrics_data = generate_latest(self.registry)
            return Response(
                content=metrics_data,
                media_type=CONTENT_TYPE_LATEST,
                headers={
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
        except Exception as e:
            logger.error("📊 Failed to generate metrics response", error=str(e))
            return Response(
                content="# Error generating metrics\n",
                media_type=CONTENT_TYPE_LATEST,
                status_code=500
            )
    
    # Convenience methods for recording metrics
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float, 
                          request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
        
        if request_size > 0:
            self.http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        if response_size > 0:
            self.http_response_size_bytes.labels(method=method, endpoint=endpoint, status_code=status_code).observe(response_size)
    
    def record_agent_operation(self, agent_id: str, operation_type: str, status: str, duration: float):
        """Record agent operation metrics."""
        self.agent_operations_total.labels(agent_id=agent_id, operation_type=operation_type, status=status).inc()
        self.agent_execution_duration_seconds.labels(agent_id=agent_id, operation_type=operation_type).observe(duration)
    
    def record_tool_execution(self, tool_name: str, status: str, duration: float):
        """Record tool execution metrics."""
        self.tool_executions_total.labels(tool_name=tool_name, status=status).inc()
        self.tool_execution_duration_seconds.labels(tool_name=tool_name).observe(duration)
        
        # Update success rate (simplified - in production you'd calculate this properly)
        if status == 'success':
            self.tool_success_rate.labels(tool_name=tool_name).set(1.0)  # This is overly simplified
    
    def record_error(self, component: str, error_type: str, severity: str = 'error'):
        """Record error metrics."""
        self.errors_total.labels(component=component, error_type=error_type, severity=severity).inc()
    
    def record_alert(self, alert_name: str, severity: str, component: str):
        """Record alert metrics."""
        self.alerts_triggered_total.labels(alert_name=alert_name, severity=severity, component=component).inc()
    
    def set_websocket_connections(self, connection_type: str, count: int):
        """Set WebSocket connection count."""
        self.websocket_connections_active.labels(connection_type=connection_type).set(count)
    
    def record_websocket_message(self, connection_type: str, message_type: str, direction: str):
        """Record WebSocket message."""
        self.websocket_messages_total.labels(
            connection_type=connection_type, 
            message_type=message_type, 
            direction=direction
        ).inc()

# Global metrics exporter instance
metrics_exporter = PrometheusMetricsExporter()

def get_metrics_exporter() -> PrometheusMetricsExporter:
    """Get the global metrics exporter instance."""
    return metrics_exporter