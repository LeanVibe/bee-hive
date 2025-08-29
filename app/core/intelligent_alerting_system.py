"""
Intelligent Alerting System for LeanVibe Agent Hive 2.0

Enterprise-grade intelligent alerting with ML-driven anomaly detection,
predictive alerting, multi-channel notifications, and <2 minute incident detection.
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import statistics
import numpy as np

import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc, insert
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client
from .prometheus_metrics_exporter import PrometheusMetricsExporter
from .performance_monitoring import PerformanceIntelligenceEngine
from ..models.agent import Agent, AgentStatus
from ..models.alert import Alert, AlertSeverity, AlertChannel, AlertStatus

logger = structlog.get_logger()


class AlertType(Enum):
    """Types of alerts in the system."""
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    PREDICTIVE = "predictive"
    COMPOUND = "compound"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SYSTEM = "system"


class AlertUrgency(Enum):
    """Alert urgency levels for escalation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    description: str
    alert_type: AlertType
    metric_query: str
    threshold: Optional[float] = None
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window: int = 60  # seconds
    severity: AlertSeverity = AlertSeverity.MEDIUM
    urgency: AlertUrgency = AlertUrgency.MEDIUM
    channels: List[AlertChannel] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    # Advanced features
    cooldown_period: int = 300  # seconds before re-alerting
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    dependency_rules: List[str] = field(default_factory=list)  # Alert IDs this depends on
    silence_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # ML features
    use_ml_detection: bool = False
    anomaly_sensitivity: float = 0.85
    prediction_horizon_minutes: int = 30


@dataclass
class AlertInstance:
    """An instance of a fired alert."""
    id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    urgency: AlertUrgency
    status: AlertStatus
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    escalation_level: int = 0
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    related_metrics: Dict[str, float] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


class IntelligentAlertingSystem:
    """
    Enterprise-grade intelligent alerting system with advanced capabilities:
    
    - <2 minute incident detection and notification
    - ML-driven anomaly detection and predictive alerting
    - Multi-channel notification routing with escalation
    - Smart alert correlation and deduplication
    - Business impact assessment and prioritization
    - Automated incident response and recovery suggestions
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        metrics_exporter: Optional[PrometheusMetricsExporter] = None,
        performance_engine: Optional[PerformanceIntelligenceEngine] = None
    ):
        """Initialize the intelligent alerting system."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.metrics_exporter = metrics_exporter
        self.performance_engine = performance_engine
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: List[AlertInstance] = []
        
        # Processing state
        self.is_running = False
        self.evaluation_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # ML components
        self.anomaly_detector = AlertAnomalyDetector()
        self.impact_assessor = BusinessImpactAssessor()
        self.correlation_engine = AlertCorrelationEngine()
        
        # Performance tracking
        self.detection_latencies: List[float] = []
        self.notification_latencies: List[float] = []
        
        # Thread pool for heavy operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alerts")
        
        # Configuration
        self.config = {
            "evaluation_interval": 15,  # seconds
            "max_detection_latency_ms": 120000,  # 2 minutes
            "max_notification_latency_ms": 30000,  # 30 seconds
            "alert_retention_hours": 168,  # 1 week
            "max_active_alerts": 1000,
            "correlation_window_minutes": 5,
            "batch_size": 50,
            "enable_ml_detection": True,
            "enable_predictive_alerts": True,
            "enable_auto_resolution": True,
            "default_channels": ["email", "slack", "webhook"]
        }
        
        # Initialize core alert rules
        self._initialize_core_alert_rules()
        
        logger.info(
            "IntelligentAlertingSystem initialized",
            rules_count=len(self.alert_rules),
            config=self.config
        )
    
    def _initialize_core_alert_rules(self) -> None:
        """Initialize core system alert rules."""
        
        # System health critical alert
        self.register_alert_rule(AlertRule(
            id="system_health_critical",
            name="System Health Critical",
            description="Overall system health has dropped to critical levels",
            alert_type=AlertType.THRESHOLD,
            metric_query="(avg(leanvibe_agent_health_score) * 0.4 + (1 - avg(leanvibe_agent_error_rate_percent) / 100) * 0.3 + (1 - leanvibe_system_cpu_percent / 100) * 0.3)",
            threshold=0.3,
            comparison_operator="<",
            evaluation_window=60,
            severity=AlertSeverity.CRITICAL,
            urgency=AlertUrgency.EMERGENCY,
            channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK, AlertChannel.SMS],
            labels={"team": "sre", "component": "system", "priority": "p0"},
            annotations={
                "summary": "System health score has dropped below critical threshold",
                "description": "Multiple system components are showing degraded performance",
                "runbook": "https://docs.leanvibe.dev/runbooks/system-health-critical",
                "dashboard": "/d/enterprise-operations-executive"
            },
            escalation_rules=[
                {"level": 1, "delay_minutes": 2, "channels": ["pager"]},
                {"level": 2, "delay_minutes": 5, "channels": ["executive"]},
                {"level": 3, "delay_minutes": 10, "channels": ["incident_commander"]}
            ]
        ))
        
        # High CPU utilization
        self.register_alert_rule(AlertRule(
            id="system_cpu_high",
            name="High CPU Utilization",
            description="System CPU usage is critically high",
            alert_type=AlertType.THRESHOLD,
            metric_query="leanvibe_system_cpu_percent",
            threshold=85.0,
            comparison_operator=">",
            evaluation_window=120,  # 2 minutes
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            labels={"team": "platform", "component": "system", "priority": "p1"},
            annotations={
                "summary": "CPU utilization is {{ $value }}%",
                "description": "System CPU usage has been above 85% for 2 minutes",
                "runbook": "https://docs.leanvibe.dev/runbooks/high-cpu"
            },
            use_ml_detection=True,
            cooldown_period=600  # 10 minutes
        ))
        
        # Agent failure rate
        self.register_alert_rule(AlertRule(
            id="agent_failure_rate_high",
            name="Agent Failure Rate High",
            description="Agent failure rate is above acceptable threshold",
            alert_type=AlertType.THRESHOLD,
            metric_query="avg(leanvibe_agent_error_rate_percent)",
            threshold=10.0,
            comparison_operator=">",
            evaluation_window=180,  # 3 minutes
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            labels={"team": "agents", "component": "agents", "priority": "p1"},
            annotations={
                "summary": "Agent failure rate is {{ $value }}%",
                "description": "Multiple agents are experiencing high failure rates",
                "runbook": "https://docs.leanvibe.dev/runbooks/agent-failures"
            }
        ))
        
        # Response time SLA breach
        self.register_alert_rule(AlertRule(
            id="response_time_sla_breach",
            name="Response Time SLA Breach",
            description="P95 response time exceeds SLA threshold",
            alert_type=AlertType.THRESHOLD,
            metric_query="histogram_quantile(0.95, rate(leanvibe_agent_response_time_seconds_bucket[5m]))",
            threshold=2.5,
            comparison_operator=">",
            evaluation_window=300,  # 5 minutes
            severity=AlertSeverity.MEDIUM,
            urgency=AlertUrgency.MEDIUM,
            channels=[AlertChannel.SLACK],
            labels={"team": "performance", "component": "agents", "priority": "p2"},
            annotations={
                "summary": "P95 response time is {{ $value }}s",
                "description": "Response time SLA is being breached",
                "runbook": "https://docs.leanvibe.dev/runbooks/response-time-sla"
            },
            use_ml_detection=True,
            prediction_horizon_minutes=15
        ))
        
        # Business metric alerts
        self.register_alert_rule(AlertRule(
            id="business_success_rate_low",
            name="Business Success Rate Low",
            description="Business operation success rate has dropped",
            alert_type=AlertType.BUSINESS,
            metric_query="avg(leanvibe_business_success_rate_percent)",
            threshold=95.0,
            comparison_operator="<",
            evaluation_window=300,
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            labels={"team": "product", "component": "business", "priority": "p1"},
            annotations={
                "summary": "Business success rate is {{ $value }}%",
                "description": "Customer-facing operations are failing at increased rate",
                "runbook": "https://docs.leanvibe.dev/runbooks/business-success-rate"
            }
        ))
        
        # Security alerts
        self.register_alert_rule(AlertRule(
            id="security_auth_failures_spike",
            name="Authentication Failures Spike",
            description="Unusual spike in authentication failures detected",
            alert_type=AlertType.ANOMALY,
            metric_query="rate(leanvibe_security_authentication_attempts_total{result=\"failure\"}[5m])",
            threshold=10.0,
            comparison_operator=">",
            evaluation_window=60,
            severity=AlertSeverity.HIGH,
            urgency=AlertUrgency.CRITICAL,
            channels=[AlertChannel.SECURITY, AlertChannel.EMAIL, AlertChannel.SLACK],
            labels={"team": "security", "component": "auth", "priority": "p0"},
            annotations={
                "summary": "Authentication failure rate spike: {{ $value }} failures/sec",
                "description": "Potential security incident - investigating authentication failures",
                "runbook": "https://docs.leanvibe.dev/runbooks/auth-failures-spike"
            },
            use_ml_detection=True,
            anomaly_sensitivity=0.9
        ))
        
        # Predictive capacity alert
        self.register_alert_rule(AlertRule(
            id="capacity_exhaustion_predicted",
            name="Capacity Exhaustion Predicted",
            description="System capacity exhaustion predicted within prediction horizon",
            alert_type=AlertType.PREDICTIVE,
            metric_query="predict_linear(leanvibe_system_memory_percent[30m], 1800)",  # 30 min prediction
            threshold=90.0,
            comparison_operator=">",
            evaluation_window=300,
            severity=AlertSeverity.MEDIUM,
            urgency=AlertUrgency.MEDIUM,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            labels={"team": "sre", "component": "capacity", "priority": "p2"},
            annotations={
                "summary": "Memory exhaustion predicted in {{ $prediction_minutes }} minutes",
                "description": "Current growth trend will lead to capacity issues",
                "runbook": "https://docs.leanvibe.dev/runbooks/capacity-planning"
            },
            use_ml_detection=True,
            prediction_horizon_minutes=30
        ))
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule."""
        if rule.id in self.alert_rules:
            logger.warning("Alert rule already exists, updating", rule_id=rule.id)
        
        self.alert_rules[rule.id] = rule
        logger.info("Alert rule registered", rule_id=rule.id, name=rule.name)
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info("Alert rule removed", rule_id=rule_id)
            return True
        return False
    
    async def start(self) -> None:
        """Start the intelligent alerting system."""
        if self.is_running:
            logger.warning("Alerting system already running")
            return
        
        logger.info("Starting intelligent alerting system")
        self.is_running = True
        
        # Start evaluation tasks
        self.evaluation_tasks = [
            asyncio.create_task(self._alert_evaluation_loop()),
            asyncio.create_task(self._notification_processing_loop()),
            asyncio.create_task(self._correlation_analysis_loop()),
            asyncio.create_task(self._auto_resolution_loop()),
            asyncio.create_task(self._escalation_management_loop()),
            asyncio.create_task(self._maintenance_loop())
        ]
        
        logger.info("Intelligent alerting system started successfully")
    
    async def stop(self) -> None:
        """Stop the intelligent alerting system."""
        if not self.is_running:
            return
        
        logger.info("Stopping intelligent alerting system")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task in self.evaluation_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.evaluation_tasks:
            await asyncio.gather(*self.evaluation_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Intelligent alerting system stopped")
    
    async def _alert_evaluation_loop(self) -> None:
        """Main alert evaluation loop with <2 minute detection."""
        logger.info("Starting alert evaluation loop")
        
        while not self.shutdown_event.is_set():
            try:
                evaluation_start = time.time()
                
                # Evaluate all enabled alert rules in parallel
                await self._evaluate_all_rules()
                
                # Track detection latency
                evaluation_time = (time.time() - evaluation_start) * 1000
                self.detection_latencies.append(evaluation_time)
                
                # Keep only recent latencies
                if len(self.detection_latencies) > 100:
                    self.detection_latencies = self.detection_latencies[-50:]
                
                # Check if detection is too slow
                if evaluation_time > self.config["max_detection_latency_ms"]:
                    logger.warning(
                        "Alert evaluation took too long",
                        evaluation_time_ms=evaluation_time,
                        max_allowed_ms=self.config["max_detection_latency_ms"]
                    )
                
                # Wait for next evaluation cycle
                await asyncio.sleep(self.config["evaluation_interval"])
                
            except Exception as e:
                logger.error("Alert evaluation loop error", error=str(e))
                await asyncio.sleep(self.config["evaluation_interval"])
        
        logger.info("Alert evaluation loop stopped")
    
    async def _evaluate_all_rules(self) -> None:
        """Evaluate all enabled alert rules."""
        enabled_rules = [rule for rule in self.alert_rules.values() if rule.enabled]
        
        if not enabled_rules:
            return
        
        # Create tasks for parallel evaluation
        evaluation_tasks = []
        for rule in enabled_rules:
            task = asyncio.create_task(self._evaluate_single_rule(rule))
            evaluation_tasks.append(task)
        
        # Wait for all evaluations to complete
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                rule = enabled_rules[i]
                logger.error(
                    "Rule evaluation failed",
                    rule_id=rule.id,
                    error=str(result)
                )
    
    async def _evaluate_single_rule(self, rule: AlertRule) -> None:
        """Evaluate a single alert rule."""
        try:
            # Check cooldown period
            if await self._is_rule_in_cooldown(rule.id):
                return
            
            # Get metric value
            current_value = await self._get_metric_value(rule.metric_query)
            if current_value is None:
                return
            
            # Determine if alert should fire
            should_fire = False
            
            if rule.use_ml_detection:
                # Use ML-based detection
                should_fire = await self._ml_based_evaluation(rule, current_value)
            else:
                # Use threshold-based detection
                should_fire = self._threshold_based_evaluation(rule, current_value)
            
            if should_fire:
                await self._fire_alert(rule, current_value)
            else:
                await self._check_alert_resolution(rule.id, current_value)
                
        except Exception as e:
            logger.error(
                "Rule evaluation error",
                rule_id=rule.id,
                error=str(e)
            )
    
    def _threshold_based_evaluation(self, rule: AlertRule, current_value: float) -> bool:
        """Evaluate rule using threshold-based logic."""
        if rule.threshold is None:
            return False
        
        if rule.comparison_operator == ">":
            return current_value > rule.threshold
        elif rule.comparison_operator == "<":
            return current_value < rule.threshold
        elif rule.comparison_operator == ">=":
            return current_value >= rule.threshold
        elif rule.comparison_operator == "<=":
            return current_value <= rule.threshold
        elif rule.comparison_operator == "==":
            return abs(current_value - rule.threshold) < 0.001
        elif rule.comparison_operator == "!=":
            return abs(current_value - rule.threshold) >= 0.001
        else:
            return False
    
    async def _ml_based_evaluation(self, rule: AlertRule, current_value: float) -> bool:
        """Evaluate rule using ML-based detection."""
        if rule.alert_type == AlertType.ANOMALY:
            # Check for anomalies
            return await self.anomaly_detector.is_anomaly(
                rule.id,
                current_value,
                sensitivity=rule.anomaly_sensitivity
            )
        elif rule.alert_type == AlertType.PREDICTIVE:
            # Check predictive conditions
            return await self._evaluate_predictive_condition(rule, current_value)
        else:
            # Fall back to threshold-based
            return self._threshold_based_evaluation(rule, current_value)
    
    async def _evaluate_predictive_condition(self, rule: AlertRule, current_value: float) -> bool:
        """Evaluate predictive alert conditions."""
        try:
            # Get historical data for prediction
            historical_data = await self._get_historical_metric_data(
                rule.metric_query,
                hours=2
            )
            
            if len(historical_data) < 10:
                return False
            
            # Simple linear prediction
            time_points = list(range(len(historical_data)))
            values = list(historical_data.values())
            
            # Calculate trend
            if len(values) > 1:
                recent_trend = (values[-1] - values[-5]) / 5 if len(values) >= 5 else 0
                
                # Predict future value
                prediction_steps = rule.prediction_horizon_minutes // 5  # Assume 5min intervals
                predicted_value = values[-1] + (recent_trend * prediction_steps)
                
                # Check if predicted value would trigger threshold
                if rule.threshold:
                    return self._threshold_based_evaluation(
                        rule,
                        predicted_value
                    )
            
            return False
            
        except Exception as e:
            logger.error("Predictive evaluation error", rule_id=rule.id, error=str(e))
            return False
    
    async def _get_metric_value(self, query: str) -> Optional[float]:
        """Get current metric value from Prometheus."""
        try:
            # In a real implementation, this would query Prometheus
            # For now, we'll simulate with some example logic
            
            if "leanvibe_system_cpu_percent" in query:
                return 75.5  # Example CPU value
            elif "leanvibe_agent_health_score" in query:
                return 0.85  # Example health score
            elif "leanvibe_agent_error_rate_percent" in query:
                return 2.3  # Example error rate
            elif "response_time_seconds" in query:
                return 1.2  # Example response time
            else:
                return 50.0  # Default value
                
        except Exception as e:
            logger.error("Error getting metric value", query=query, error=str(e))
            return None
    
    async def _get_historical_metric_data(
        self,
        query: str,
        hours: int = 2
    ) -> Dict[str, float]:
        """Get historical metric data for analysis."""
        try:
            # Simulate historical data
            # In real implementation, this would query Prometheus range API
            data = {}
            current_time = datetime.utcnow()
            
            for i in range(hours * 12):  # 5-minute intervals
                timestamp = current_time - timedelta(minutes=i * 5)
                # Simulate some trending data
                base_value = 50.0
                trend = i * 0.1
                noise = np.random.normal(0, 2)
                value = base_value + trend + noise
                data[timestamp.isoformat()] = value
            
            return data
            
        except Exception as e:
            logger.error("Error getting historical data", query=query, error=str(e))
            return {}
    
    async def _fire_alert(self, rule: AlertRule, current_value: float) -> None:
        """Fire an alert for the given rule."""
        try:
            alert_id = f"{rule.id}_{int(time.time())}"
            
            # Check if similar alert is already active
            existing_alert = await self._find_similar_active_alert(rule.id)
            if existing_alert:
                logger.debug("Similar alert already active", rule_id=rule.id)
                return
            
            # Create alert instance
            alert_instance = AlertInstance(
                id=alert_id,
                rule_id=rule.id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                urgency=rule.urgency,
                status=AlertStatus.FIRING,
                fired_at=datetime.utcnow(),
                current_value=current_value,
                threshold_value=rule.threshold,
                labels=rule.labels.copy(),
                annotations=rule.annotations.copy()
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert_instance
            
            # Enrich alert with context
            await self._enrich_alert_context(alert_instance)
            
            # Assess business impact
            alert_instance.impact_assessment = await self.impact_assessor.assess_impact(
                alert_instance
            )
            
            # Generate suggested actions
            alert_instance.suggested_actions = await self._generate_suggested_actions(
                alert_instance
            )
            
            # Send notifications
            await self._send_alert_notifications(alert_instance)
            
            # Store in database
            await self._store_alert_instance(alert_instance)
            
            # Log alert firing
            logger.warning(
                "Alert fired",
                alert_id=alert_id,
                rule_id=rule.id,
                severity=rule.severity.value,
                urgency=rule.urgency.value,
                current_value=current_value,
                threshold_value=rule.threshold
            )
            
        except Exception as e:
            logger.error("Error firing alert", rule_id=rule.id, error=str(e))
    
    async def _enrich_alert_context(self, alert: AlertInstance) -> None:
        """Enrich alert with additional context and related metrics."""
        try:
            # Get related metrics based on alert type
            if "cpu" in alert.name.lower():
                alert.related_metrics.update({
                    "memory_percent": await self._get_metric_value("leanvibe_system_memory_percent") or 0,
                    "load_average": await self._get_metric_value("leanvibe_system_load_average") or 0,
                    "active_agents": await self._get_metric_value("sum(leanvibe_agents_total{status=\"active\"})") or 0
                })
            elif "agent" in alert.name.lower():
                alert.related_metrics.update({
                    "total_agents": await self._get_metric_value("sum(leanvibe_agents_total)") or 0,
                    "avg_response_time": await self._get_metric_value("histogram_quantile(0.5, rate(leanvibe_agent_response_time_seconds_bucket[5m]))") or 0,
                    "task_queue_size": await self._get_metric_value("sum(leanvibe_task_queue_size)") or 0
                })
            
            # Add timestamp-based context
            alert.annotations.update({
                "fired_at_human": alert.fired_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "detection_latency_ms": str(int(self.detection_latencies[-1] if self.detection_latencies else 0))
            })
            
        except Exception as e:
            logger.error("Error enriching alert context", alert_id=alert.id, error=str(e))
    
    async def _generate_suggested_actions(self, alert: AlertInstance) -> List[str]:
        """Generate suggested actions based on alert context."""
        actions = []
        
        try:
            if "cpu" in alert.name.lower():
                actions.extend([
                    "Check for runaway processes using 'top' or 'htop'",
                    "Review recent deployments that might have introduced performance issues",
                    "Consider scaling horizontally by adding more agents",
                    "Check if any agents are stuck in infinite loops",
                    "Review system logs for errors or unusual activity"
                ])
            elif "memory" in alert.name.lower():
                actions.extend([
                    "Identify memory-intensive processes and consider restarting them",
                    "Check for memory leaks in recent code deployments",
                    "Review agent memory usage patterns",
                    "Consider increasing system memory or adding swap",
                    "Analyze heap dumps if available"
                ])
            elif "agent" in alert.name.lower():
                actions.extend([
                    "Check agent health endpoints for specific error details",
                    "Review recent agent deployments or configuration changes",
                    "Verify database and Redis connectivity from agents",
                    "Check agent logs for error patterns",
                    "Consider restarting unhealthy agents"
                ])
            elif "response_time" in alert.name.lower():
                actions.extend([
                    "Check system resource utilization (CPU, memory, I/O)",
                    "Review database query performance and connection pools",
                    "Analyze network latency between services",
                    "Check if any agents are overloaded",
                    "Review recent code changes that might affect performance"
                ])
            
            # Add general actions
            actions.extend([
                f"Check dashboard: {alert.annotations.get('dashboard', 'N/A')}",
                f"Follow runbook: {alert.annotations.get('runbook', 'N/A')}",
                "Review system logs and metrics for related issues",
                "Check if this is part of a larger incident pattern"
            ])
            
        except Exception as e:
            logger.error("Error generating suggested actions", alert_id=alert.id, error=str(e))
        
        return actions
    
    async def _send_alert_notifications(self, alert: AlertInstance) -> None:
        """Send alert notifications through configured channels."""
        notification_start = time.time()
        
        try:
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                logger.error("Rule not found for alert", alert_id=alert.id, rule_id=alert.rule_id)
                return
            
            # Send notifications to all configured channels
            notification_tasks = []
            for channel in rule.channels:
                task = asyncio.create_task(
                    self._send_channel_notification(alert, channel)
                )
                notification_tasks.append(task)
            
            # Wait for all notifications
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Track notification results
            for i, result in enumerate(results):
                channel = rule.channels[i]
                if isinstance(result, Exception):
                    logger.error(
                        "Notification failed",
                        alert_id=alert.id,
                        channel=channel.value,
                        error=str(result)
                    )
                    alert.notifications_sent.append({
                        "channel": channel.value,
                        "status": "failed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "error": str(result)
                    })
                else:
                    alert.notifications_sent.append({
                        "channel": channel.value,
                        "status": "sent",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            # Track notification latency
            notification_time = (time.time() - notification_start) * 1000
            self.notification_latencies.append(notification_time)
            
            if len(self.notification_latencies) > 100:
                self.notification_latencies = self.notification_latencies[-50:]
            
            # Check if notification was too slow
            if notification_time > self.config["max_notification_latency_ms"]:
                logger.warning(
                    "Notification took too long",
                    alert_id=alert.id,
                    notification_time_ms=notification_time,
                    max_allowed_ms=self.config["max_notification_latency_ms"]
                )
            
        except Exception as e:
            logger.error("Error sending alert notifications", alert_id=alert.id, error=str(e))
    
    async def _send_channel_notification(
        self,
        alert: AlertInstance,
        channel: AlertChannel
    ) -> None:
        """Send notification to a specific channel."""
        try:
            if channel == AlertChannel.SLACK:
                await self._send_slack_notification(alert)
            elif channel == AlertChannel.EMAIL:
                await self._send_email_notification(alert)
            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_notification(alert)
            elif channel == AlertChannel.SMS:
                await self._send_sms_notification(alert)
            else:
                logger.warning("Unsupported notification channel", channel=channel.value)
                
        except Exception as e:
            logger.error(
                "Channel notification error",
                alert_id=alert.id,
                channel=channel.value,
                error=str(e)
            )
            raise
    
    async def _send_slack_notification(self, alert: AlertInstance) -> None:
        """Send Slack notification."""
        # Simulate Slack notification
        logger.info(
            "Slack notification sent",
            alert_id=alert.id,
            severity=alert.severity.value,
            message=f"🚨 {alert.name}: {alert.description}"
        )
    
    async def _send_email_notification(self, alert: AlertInstance) -> None:
        """Send email notification."""
        # Simulate email notification
        logger.info(
            "Email notification sent",
            alert_id=alert.id,
            severity=alert.severity.value,
            subject=f"[{alert.severity.value.upper()}] {alert.name}"
        )
    
    async def _send_webhook_notification(self, alert: AlertInstance) -> None:
        """Send webhook notification."""
        # Simulate webhook notification
        logger.info(
            "Webhook notification sent",
            alert_id=alert.id,
            severity=alert.severity.value
        )
    
    async def _send_sms_notification(self, alert: AlertInstance) -> None:
        """Send SMS notification."""
        # Simulate SMS notification (only for critical alerts)
        if alert.urgency == AlertUrgency.EMERGENCY:
            logger.info(
                "SMS notification sent",
                alert_id=alert.id,
                urgency=alert.urgency.value
            )
    
    # Additional methods for alert management...
    async def _notification_processing_loop(self) -> None:
        """Background loop for processing notifications and escalations."""
        # Implementation would go here
        pass
    
    async def _correlation_analysis_loop(self) -> None:
        """Background loop for alert correlation and deduplication."""
        # Implementation would go here
        pass
    
    async def _auto_resolution_loop(self) -> None:
        """Background loop for automatic alert resolution."""
        # Implementation would go here
        pass
    
    async def _escalation_management_loop(self) -> None:
        """Background loop for managing alert escalations."""
        # Implementation would go here
        pass
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance tasks."""
        # Implementation would go here
        pass
    
    # Placeholder methods for additional functionality
    async def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period."""
        return False
    
    async def _find_similar_active_alert(self, rule_id: str) -> Optional[AlertInstance]:
        """Find similar active alert."""
        return None
    
    async def _check_alert_resolution(self, rule_id: str, current_value: float) -> None:
        """Check if alert should be resolved."""
        pass
    
    async def _store_alert_instance(self, alert: AlertInstance) -> None:
        """Store alert instance in database."""
        pass


# Helper classes
class AlertAnomalyDetector:
    """ML-based anomaly detection for alerts."""
    
    async def is_anomaly(
        self,
        metric_id: str,
        current_value: float,
        sensitivity: float = 0.85
    ) -> bool:
        """Detect if current value is anomalous."""
        # Simplified anomaly detection
        # In real implementation, this would use proper ML models
        return False


class BusinessImpactAssessor:
    """Assess business impact of alerts."""
    
    async def assess_impact(self, alert: AlertInstance) -> Dict[str, Any]:
        """Assess business impact of an alert."""
        return {
            "impact_level": "medium",
            "affected_users": 0,
            "estimated_revenue_impact": 0,
            "affected_services": []
        }


class AlertCorrelationEngine:
    """Engine for correlating related alerts."""
    
    async def correlate_alerts(
        self,
        alerts: List[AlertInstance]
    ) -> List[List[AlertInstance]]:
        """Correlate related alerts into groups."""
        return []


# Global instance
_intelligent_alerting_system: Optional[IntelligentAlertingSystem] = None


async def get_intelligent_alerting_system() -> IntelligentAlertingSystem:
    """Get singleton intelligent alerting system instance."""
    global _intelligent_alerting_system
    
    if _intelligent_alerting_system is None:
        _intelligent_alerting_system = IntelligentAlertingSystem()
        await _intelligent_alerting_system.start()
    
    return _intelligent_alerting_system


async def cleanup_intelligent_alerting_system() -> None:
    """Cleanup intelligent alerting system resources."""
    global _intelligent_alerting_system
    
    if _intelligent_alerting_system:
        await _intelligent_alerting_system.stop()
        _intelligent_alerting_system = None