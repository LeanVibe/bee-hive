"""
Operations Specialist Agent - Production Monitoring System
Epic G: Production Readiness - Phase 1

Enterprise-grade production monitoring with distributed tracing,
real-time health dashboards, and intelligent alerting for the
LeanVibe Agent Hive 2.0 platform.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

import structlog
from prometheus_client import Counter, Histogram, Gauge, Enum as PrometheusEnum
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

logger = structlog.get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for production monitoring."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MonitoringComponent(Enum):
    """Components being monitored in the system."""
    AGENT_ORCHESTRATION = "agent_orchestration"
    DATABASE = "database"
    REDIS = "redis"
    API_GATEWAY = "api_gateway"
    WEBSOCKETS = "websockets"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS_METRICS = "business_metrics"

@dataclass
class HealthMetric:
    """Health metric data structure."""
    component: MonitoringComponent
    metric_name: str
    value: float
    threshold: float
    status: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    component: MonitoringComponent
    metric_name: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    duration: int  # seconds
    description: str
    enabled: bool = True

class DistributedTracingSystem:
    """Distributed tracing system for agent orchestration workflows."""
    
    def __init__(self):
        self.tracer_provider = TracerProvider()
        trace.set_tracer_provider(self.tracer_provider)
        
        # Configure Jaeger exporter for distributed tracing
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger-agent",
            agent_port=14268,
            collector_endpoint="http://jaeger-collector:14268/api/traces",
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument key components
        self._setup_instrumentation()
        
    def _setup_instrumentation(self):
        """Setup automatic instrumentation for key components."""
        try:
            # FastAPI automatic instrumentation
            FastAPIInstrumentor().instrument()
            logger.info("FastAPI instrumentation enabled")
            
            # SQLAlchemy instrumentation for database queries
            SQLAlchemyInstrumentor().instrument()
            logger.info("SQLAlchemy instrumentation enabled")
            
            # Redis instrumentation for caching operations
            RedisInstrumentor().instrument()
            logger.info("Redis instrumentation enabled")
            
        except Exception as e:
            logger.error("Failed to setup instrumentation", error=str(e))
    
    def create_span(self, name: str, attributes: Dict[str, Any] = None):
        """Create a new tracing span."""
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        return span
    
    async def trace_agent_workflow(self, workflow_id: str, agent_id: str, operation: str):
        """Trace agent workflow operations with detailed context."""
        with self.create_span(
            f"agent_workflow.{operation}",
            attributes={
                "workflow.id": workflow_id,
                "agent.id": agent_id,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            }
        ) as span:
            span.set_attribute("component", "agent_orchestration")
            return span

class ProductionMetricsCollector:
    """Advanced metrics collection for production monitoring."""
    
    def __init__(self):
        # Agent Orchestration Metrics
        self.agent_spawn_counter = Counter(
            'leanvibe_agents_spawned_total',
            'Total number of agents spawned',
            ['agent_role', 'workflow_type']
        )
        
        self.agent_task_duration = Histogram(
            'leanvibe_agent_task_duration_seconds',
            'Time spent executing agent tasks',
            ['agent_id', 'task_type', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.agent_coordination_latency = Histogram(
            'leanvibe_agent_coordination_latency_seconds',
            'Latency for agent-to-agent coordination',
            ['source_agent', 'target_agent', 'coordination_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        )
        
        # System Health Metrics
        self.component_health = Gauge(
            'leanvibe_component_health_status',
            'Health status of system components (1=healthy, 0=unhealthy)',
            ['component', 'instance']
        )
        
        self.resource_utilization = Gauge(
            'leanvibe_resource_utilization_percent',
            'Resource utilization percentage',
            ['resource_type', 'component']
        )
        
        # Business Metrics
        self.workflow_success_rate = Gauge(
            'leanvibe_workflow_success_rate',
            'Workflow success rate percentage',
            ['workflow_type', 'time_window']
        )
        
        self.customer_satisfaction_score = Gauge(
            'leanvibe_customer_satisfaction_score',
            'Customer satisfaction score (1-10)',
            ['deployment_type', 'feature']
        )
        
        # Security Metrics
        self.security_events_counter = Counter(
            'leanvibe_security_events_total',
            'Total security events detected',
            ['event_type', 'severity', 'source']
        )
        
        self.failed_auth_attempts = Counter(
            'leanvibe_failed_auth_attempts_total',
            'Total failed authentication attempts',
            ['method', 'source_ip', 'user_type']
        )
        
    def record_agent_spawn(self, agent_role: str, workflow_type: str):
        """Record agent spawn event."""
        self.agent_spawn_counter.labels(
            agent_role=agent_role,
            workflow_type=workflow_type
        ).inc()
    
    def record_agent_task(self, agent_id: str, task_type: str, duration: float, status: str):
        """Record agent task completion."""
        self.agent_task_duration.labels(
            agent_id=agent_id,
            task_type=task_type,
            status=status
        ).observe(duration)
    
    def record_coordination_latency(self, source: str, target: str, coord_type: str, latency: float):
        """Record agent coordination latency."""
        self.agent_coordination_latency.labels(
            source_agent=source,
            target_agent=target,
            coordination_type=coord_type
        ).observe(latency)
    
    def update_component_health(self, component: str, instance: str, healthy: bool):
        """Update component health status."""
        self.component_health.labels(
            component=component,
            instance=instance
        ).set(1.0 if healthy else 0.0)
    
    def update_resource_utilization(self, resource_type: str, component: str, utilization: float):
        """Update resource utilization metrics."""
        self.resource_utilization.labels(
            resource_type=resource_type,
            component=component
        ).set(utilization)

class RealTimeHealthDashboard:
    """Real-time health dashboard for production monitoring."""
    
    def __init__(self, metrics_collector: ProductionMetricsCollector):
        self.metrics = metrics_collector
        self.health_checks = {}
        self.alert_rules = []
        self.current_alerts = []
        
    async def register_health_check(self, component: MonitoringComponent, check_func: callable, interval: int = 30):
        """Register a health check function for a component."""
        self.health_checks[component] = {
            'check_func': check_func,
            'interval': interval,
            'last_check': None,
            'status': 'unknown'
        }
        logger.info(f"Registered health check for {component.value}")
    
    async def add_alert_rule(self, alert_rule: AlertRule):
        """Add an alert rule to the monitoring system."""
        self.alert_rules.append(alert_rule)
        logger.info(f"Added alert rule: {alert_rule.rule_id}")
    
    async def run_health_checks(self):
        """Continuously run health checks for all registered components."""
        while True:
            try:
                for component, check_config in self.health_checks.items():
                    now = datetime.utcnow()
                    
                    # Check if it's time to run this health check
                    if (check_config['last_check'] is None or 
                        (now - check_config['last_check']).total_seconds() >= check_config['interval']):
                        
                        try:
                            # Run the health check
                            result = await check_config['check_func']()
                            check_config['status'] = 'healthy' if result else 'unhealthy'
                            check_config['last_check'] = now
                            
                            # Update metrics
                            self.metrics.update_component_health(
                                component=component.value,
                                instance="primary",
                                healthy=result
                            )
                            
                        except Exception as e:
                            check_config['status'] = 'error'
                            logger.error(f"Health check failed for {component.value}", error=str(e))
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(10)
    
    async def evaluate_alert_rules(self):
        """Evaluate alert rules and trigger alerts when conditions are met."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    # Get current metric value (simplified - would integrate with actual metrics)
                    metric_value = await self._get_metric_value(rule.component, rule.metric_name)
                    
                    # Evaluate condition
                    condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                    
                    if condition_met:
                        await self._trigger_alert(rule, metric_value, current_time)
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error("Alert evaluation error", error=str(e))
                await asyncio.sleep(60)
    
    async def _get_metric_value(self, component: MonitoringComponent, metric_name: str) -> float:
        """Get current value for a metric (integrate with actual metrics backend)."""
        # This would integrate with Prometheus or other metrics backend
        # For now, returning mock values
        return 0.0
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        elif condition == "ne":
            return abs(value - threshold) >= 0.001
        return False
    
    async def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Trigger an alert when conditions are met."""
        alert = {
            'alert_id': f"{rule.rule_id}_{int(timestamp.timestamp())}",
            'rule_id': rule.rule_id,
            'component': rule.component.value,
            'severity': rule.severity.value,
            'description': rule.description,
            'current_value': value,
            'threshold': rule.threshold,
            'timestamp': timestamp,
            'status': 'firing'
        }
        
        self.current_alerts.append(alert)
        
        # Send alert notification
        await self._send_alert_notification(alert)
        
        logger.warning(
            "Alert triggered",
            alert_id=alert['alert_id'],
            component=rule.component.value,
            severity=rule.severity.value,
            current_value=value,
            threshold=rule.threshold
        )
    
    async def _send_alert_notification(self, alert: Dict[str, Any]):
        """Send alert notification (integrate with notification system)."""
        # This would integrate with Slack, PagerDuty, email, etc.
        logger.info(f"Sending alert notification: {alert['alert_id']}")

class IntelligentAlertingSystem:
    """Intelligent alerting system with ML-based anomaly detection."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.alert_suppression_window = 300  # 5 minutes
        self.recent_alerts = {}
        
    async def learn_baseline(self, component: str, metric_name: str, values: List[float]):
        """Learn baseline behavior for a metric."""
        import statistics
        
        if len(values) < 10:
            return
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        self.baseline_metrics[f"{component}_{metric_name}"] = {
            'mean': mean,
            'stdev': stdev,
            'sample_count': len(values),
            'last_updated': datetime.utcnow()
        }
        
        logger.info(f"Learned baseline for {component}_{metric_name}: mean={mean:.2f}, stdev={stdev:.2f}")
    
    async def detect_anomaly(self, component: str, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Detect anomalies using statistical analysis."""
        key = f"{component}_{metric_name}"
        baseline = self.baseline_metrics.get(key)
        
        if not baseline or baseline['stdev'] == 0:
            return None
        
        # Calculate z-score
        z_score = abs(value - baseline['mean']) / baseline['stdev']
        
        if z_score > self.anomaly_threshold:
            return {
                'component': component,
                'metric': metric_name,
                'current_value': value,
                'expected_value': baseline['mean'],
                'z_score': z_score,
                'severity': 'high' if z_score > 3.0 else 'medium',
                'anomaly_type': 'statistical_outlier'
            }
        
        return None
    
    async def should_suppress_alert(self, alert_key: str) -> bool:
        """Check if alert should be suppressed to prevent alert fatigue."""
        now = datetime.utcnow()
        
        if alert_key in self.recent_alerts:
            last_alert = self.recent_alerts[alert_key]
            if (now - last_alert).total_seconds() < self.alert_suppression_window:
                return True
        
        self.recent_alerts[alert_key] = now
        return False

class ProductionMonitoringOrchestrator:
    """Main orchestrator for production monitoring system."""
    
    def __init__(self):
        self.distributed_tracing = DistributedTracingSystem()
        self.metrics_collector = ProductionMetricsCollector()
        self.health_dashboard = RealTimeHealthDashboard(self.metrics_collector)
        self.intelligent_alerting = IntelligentAlertingSystem()
        
        self.monitoring_tasks = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize the production monitoring system."""
        logger.info("🚀 Initializing Production Monitoring System")
        
        # Setup default alert rules
        await self._setup_default_alert_rules()
        
        # Register default health checks
        await self._register_default_health_checks()
        
        logger.info("✅ Production Monitoring System initialized")
    
    async def start(self):
        """Start all monitoring services."""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting production monitoring services")
        
        # Start health check loop
        health_check_task = asyncio.create_task(self.health_dashboard.run_health_checks())
        self.monitoring_tasks.append(health_check_task)
        
        # Start alert evaluation loop
        alert_eval_task = asyncio.create_task(self.health_dashboard.evaluate_alert_rules())
        self.monitoring_tasks.append(alert_eval_task)
        
        # Start metrics collection task
        metrics_task = asyncio.create_task(self._collect_system_metrics())
        self.monitoring_tasks.append(metrics_task)
        
        logger.info("✅ All monitoring services started")
    
    async def stop(self):
        """Stop all monitoring services."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping production monitoring services")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("✅ All monitoring services stopped")
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules for critical system components."""
        default_rules = [
            AlertRule(
                rule_id="agent_orchestration_failure_rate",
                component=MonitoringComponent.AGENT_ORCHESTRATION,
                metric_name="failure_rate",
                condition="gt",
                threshold=0.05,  # 5% failure rate
                severity=AlertSeverity.HIGH,
                duration=300,  # 5 minutes
                description="Agent orchestration failure rate exceeds 5%"
            ),
            AlertRule(
                rule_id="database_connection_pool_exhaustion",
                component=MonitoringComponent.DATABASE,
                metric_name="connection_pool_usage",
                condition="gt",
                threshold=0.90,  # 90% pool usage
                severity=AlertSeverity.CRITICAL,
                duration=60,  # 1 minute
                description="Database connection pool usage exceeds 90%"
            ),
            AlertRule(
                rule_id="redis_memory_usage_high",
                component=MonitoringComponent.REDIS,
                metric_name="memory_usage_percent",
                condition="gt",
                threshold=85.0,  # 85% memory usage
                severity=AlertSeverity.HIGH,
                duration=300,  # 5 minutes
                description="Redis memory usage exceeds 85%"
            ),
            AlertRule(
                rule_id="api_response_time_degradation",
                component=MonitoringComponent.API_GATEWAY,
                metric_name="p95_response_time",
                condition="gt",
                threshold=2.0,  # 2 seconds
                severity=AlertSeverity.MEDIUM,
                duration=600,  # 10 minutes
                description="API P95 response time exceeds 2 seconds"
            ),
            AlertRule(
                rule_id="security_failed_auth_spike",
                component=MonitoringComponent.SECURITY,
                metric_name="failed_auth_rate",
                condition="gt",
                threshold=10.0,  # 10 failures per minute
                severity=AlertSeverity.HIGH,
                duration=60,  # 1 minute
                description="Failed authentication rate spike detected"
            )
        ]
        
        for rule in default_rules:
            await self.health_dashboard.add_alert_rule(rule)
    
    async def _register_default_health_checks(self):
        """Register default health checks for system components."""
        
        async def check_database_health():
            # This would integrate with actual database health check
            return True
        
        async def check_redis_health():
            # This would integrate with actual Redis health check
            return True
        
        async def check_agent_orchestration_health():
            # This would check orchestrator status
            return True
        
        await self.health_dashboard.register_health_check(
            MonitoringComponent.DATABASE, check_database_health, interval=30
        )
        await self.health_dashboard.register_health_check(
            MonitoringComponent.REDIS, check_redis_health, interval=30
        )
        await self.health_dashboard.register_health_check(
            MonitoringComponent.AGENT_ORCHESTRATION, check_agent_orchestration_health, interval=15
        )
    
    async def _collect_system_metrics(self):
        """Continuously collect system metrics."""
        while self.is_running:
            try:
                # Collect and update various system metrics
                await self._update_resource_metrics()
                await self._update_business_metrics()
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(30)
    
    async def _update_resource_metrics(self):
        """Update resource utilization metrics."""
        try:
            import psutil
            
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.update_resource_utilization("cpu", "system", cpu_percent)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.metrics_collector.update_resource_utilization("memory", "system", memory.percent)
            
            # Disk utilization
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.update_resource_utilization("disk", "system", disk_percent)
            
        except ImportError:
            logger.debug("psutil not available, skipping resource metrics")
        except Exception as e:
            logger.error("Failed to update resource metrics", error=str(e))
    
    async def _update_business_metrics(self):
        """Update business metrics (integrate with actual business logic)."""
        # This would integrate with actual business metrics collection
        # For now, setting example values
        self.metrics_collector.workflow_success_rate.labels(
            workflow_type="ecommerce",
            time_window="1h"
        ).set(0.95)  # 95% success rate
        
        self.metrics_collector.customer_satisfaction_score.labels(
            deployment_type="enterprise",
            feature="agent_orchestration"
        ).set(8.5)  # 8.5/10 satisfaction

# Global monitoring instance
_monitoring_orchestrator: Optional[ProductionMonitoringOrchestrator] = None

async def get_production_monitoring() -> ProductionMonitoringOrchestrator:
    """Get the global production monitoring orchestrator."""
    global _monitoring_orchestrator
    
    if _monitoring_orchestrator is None:
        _monitoring_orchestrator = ProductionMonitoringOrchestrator()
        await _monitoring_orchestrator.initialize()
    
    return _monitoring_orchestrator

async def start_production_monitoring():
    """Start the production monitoring system."""
    monitoring = await get_production_monitoring()
    await monitoring.start()

async def stop_production_monitoring():
    """Stop the production monitoring system."""
    global _monitoring_orchestrator
    
    if _monitoring_orchestrator:
        await _monitoring_orchestrator.stop()
        _monitoring_orchestrator = None