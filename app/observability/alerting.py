"""
Real-time Alerting System for LeanVibe Agent Hive 2.0

Provides intelligent alerting based on metrics thresholds, event patterns,
and system health indicators. Integrates with external notification systems
and includes alert correlation, deduplication, and escalation.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from app.observability.external_hooks import (
    ExternalHookManager, HookEvent, ExternalHookConfig, HookType
)
from app.observability.prometheus_exporter import get_metrics_exporter

logger = structlog.get_logger()

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertStatus(str, Enum):
    """Alert status values."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"
    ACKNOWLEDGED = "acknowledged"

class ThresholdOperator(str, Enum):
    """Threshold comparison operators."""
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    EQUAL = "eq"
    NOT_EQUAL = "ne"

@dataclass
class MetricThreshold:
    """Configuration for metric-based alerting."""
    metric_name: str
    operator: ThresholdOperator
    value: float
    duration_seconds: int = 60  # How long threshold must be exceeded
    labels: Dict[str, str] = field(default_factory=dict)  # Metric label filters

@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    threshold: MetricThreshold
    enabled: bool = True
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    escalation_delay_minutes: int = 15
    max_escalations: int = 3
    
    # Deduplication settings
    grouping_labels: List[str] = field(default_factory=list)
    deduplication_window_minutes: int = 5
    
    # Custom metadata
    runbook_url: Optional[str] = None
    dashboard_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class Alert:
    """An active or resolved alert."""
    id: str
    rule_id: str
    rule_name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    
    # Timing
    started_at: datetime
    resolved_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Context
    metric_name: str = ""
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Escalation tracking
    escalation_level: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    # Notification tracking
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "description": self.description,
            "severity": self.severity,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "last_updated": self.last_updated.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "labels": self.labels,
            "escalation_level": self.escalation_level,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "notifications_sent": self.notifications_sent
        }

class AlertingEngine:
    """
    Real-time alerting engine with intelligent correlation and notification.
    
    Features:
    - Metric threshold monitoring
    - Event pattern detection  
    - Alert deduplication and correlation
    - Escalation management
    - Integration with external notification systems
    """
    
    def __init__(self, hook_manager: Optional[ExternalHookManager] = None):
        """Initialize the alerting engine."""
        self.hook_manager = hook_manager
        
        # Alert rules and active alerts
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.max_resolved_alerts = 1000
        
        # Metric tracking for threshold evaluation
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.threshold_violations: Dict[str, datetime] = {}
        
        # Deduplication tracking
        self.alert_fingerprints: Dict[str, str] = {}  # fingerprint -> alert_id
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info("🚨 Alerting engine initialized")
    
    async def start(self) -> None:
        """Start the alerting engine."""
        if self.running:
            return
        
        self.running = True
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.escalation_task = asyncio.create_task(self._escalation_loop())
        
        logger.info("🚨 Alerting engine started")
    
    async def stop(self) -> None:
        """Stop the alerting engine."""
        self.running = False
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.escalation_task:
            self.escalation_task.cancel()
            try:
                await self.escalation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🚨 Alerting engine stopped")
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules[rule.id] = rule
        logger.info(
            "📋 Alert rule added",
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            metric=rule.threshold.metric_name
        )
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self.rules:
            rule = self.rules.pop(rule_id)
            
            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]
            
            for alert in alerts_to_resolve:
                self._resolve_alert(alert.id, "Rule removed")
            
            logger.info("🗑️ Alert rule removed", rule_id=rule_id, rule_name=rule.name)
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        return list(self.rules.values())
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_resolved_alerts(self, limit: int = 100) -> List[Alert]:
        """Get resolved alerts."""
        return self.resolved_alerts[-limit:]
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID (active or resolved)."""
        # Check active alerts first
        if alert_id in self.active_alerts:
            return self.active_alerts[alert_id]
        
        # Check resolved alerts
        for alert in self.resolved_alerts:
            if alert.id == alert_id:
                return alert
        
        return None
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        alert.last_updated = datetime.utcnow()
        
        logger.info(
            "✅ Alert acknowledged",
            alert_id=alert_id,
            acknowledged_by=acknowledged_by
        )
        
        # Send notification about acknowledgment
        asyncio.create_task(self._send_alert_notification(alert, "acknowledged"))
        
        return True
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60) -> bool:
        """Silence an alert for a specified duration."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.SILENCED
        alert.last_updated = datetime.utcnow()
        
        # Schedule automatic unsilencing
        asyncio.create_task(self._unsilence_after_delay(alert_id, duration_minutes))
        
        logger.info(
            "🔇 Alert silenced",
            alert_id=alert_id,
            duration_minutes=duration_minutes
        )
        
        return True
    
    async def record_metric_value(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a metric value for threshold evaluation."""
        timestamp = datetime.utcnow()
        
        # Create metric key including labels
        metric_key = self._create_metric_key(metric_name, labels or {})
        
        # Store metric value with timestamp
        self.metric_history[metric_key].append((timestamp, value))
        
        # Evaluate thresholds for this metric
        await self._evaluate_metric_thresholds(metric_name, value, labels or {})
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for collecting metrics and evaluating thresholds."""
        while self.running:
            try:
                # Collect current metrics from Prometheus exporter
                await self._collect_system_metrics()
                
                # Clean up old metric history
                self._cleanup_metric_history()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error("❌ Error in monitoring loop", error=str(e), exc_info=True)
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _escalation_loop(self) -> None:
        """Handle alert escalation and notifications."""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for alert in list(self.active_alerts.values()):
                    if alert.status in [AlertStatus.SILENCED, AlertStatus.ACKNOWLEDGED]:
                        continue
                    
                    rule = self.rules.get(alert.rule_id)
                    if not rule:
                        continue
                    
                    # Check if escalation is needed
                    time_since_last = (current_time - alert.last_updated).total_seconds()
                    escalation_threshold = rule.escalation_delay_minutes * 60
                    
                    if (time_since_last >= escalation_threshold and 
                        alert.escalation_level < rule.max_escalations):
                        
                        await self._escalate_alert(alert)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("❌ Error in escalation loop", error=str(e), exc_info=True)
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics for threshold evaluation."""
        try:
            metrics_exporter = get_metrics_exporter()
            if not metrics_exporter:
                return
            
            # Collect key metrics for alerting
            # This is a simplified example - in production you'd collect from Prometheus
            
            # CPU usage
            cpu_value = 75.0  # Placeholder - would get from metrics_exporter
            await self.record_metric_value("system_cpu_usage_percent", cpu_value)
            
            # Memory usage
            memory_value = 80.0  # Placeholder
            await self.record_metric_value("system_memory_usage_percent", memory_value)
            
            # HTTP error rate
            error_rate = 0.02  # Placeholder
            await self.record_metric_value("http_error_rate", error_rate)
            
            # Response time
            response_time = 0.5  # Placeholder
            await self.record_metric_value("http_response_time_p95", response_time)
            
        except Exception as e:
            logger.error("❌ Failed to collect system metrics", error=str(e))
    
    async def _evaluate_metric_thresholds(self, metric_name: str, value: float, labels: Dict[str, str]) -> None:
        """Evaluate metric against alert rule thresholds."""
        current_time = datetime.utcnow()
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            threshold = rule.threshold
            if threshold.metric_name != metric_name:
                continue
            
            # Check label filters
            if not self._labels_match(threshold.labels, labels):
                continue
            
            # Evaluate threshold
            violation = self._evaluate_threshold(threshold.operator, value, threshold.value)
            violation_key = f"{rule.id}:{self._create_metric_key(metric_name, labels)}"
            
            if violation:
                # Track when violation started
                if violation_key not in self.threshold_violations:
                    self.threshold_violations[violation_key] = current_time
                
                # Check if violation duration exceeds threshold
                violation_duration = (current_time - self.threshold_violations[violation_key]).total_seconds()
                
                if violation_duration >= threshold.duration_seconds:
                    await self._trigger_alert(rule, metric_name, value, labels)
            else:
                # Clear violation tracking
                if violation_key in self.threshold_violations:
                    del self.threshold_violations[violation_key]
                
                # Resolve alert if it exists
                await self._check_alert_resolution(rule, metric_name, labels)
    
    def _evaluate_threshold(self, operator: ThresholdOperator, value: float, threshold: float) -> bool:
        """Evaluate threshold condition."""
        if operator == ThresholdOperator.GREATER_THAN:
            return value > threshold
        elif operator == ThresholdOperator.GREATER_EQUAL:
            return value >= threshold
        elif operator == ThresholdOperator.LESS_THAN:
            return value < threshold
        elif operator == ThresholdOperator.LESS_EQUAL:
            return value <= threshold
        elif operator == ThresholdOperator.EQUAL:
            return value == threshold
        elif operator == ThresholdOperator.NOT_EQUAL:
            return value != threshold
        return False
    
    def _labels_match(self, filter_labels: Dict[str, str], metric_labels: Dict[str, str]) -> bool:
        """Check if metric labels match filter criteria."""
        for key, value in filter_labels.items():
            if metric_labels.get(key) != value:
                return False
        return True
    
    async def _trigger_alert(self, rule: AlertRule, metric_name: str, value: float, labels: Dict[str, str]) -> None:
        """Trigger a new alert."""
        # Create alert fingerprint for deduplication
        fingerprint = self._create_alert_fingerprint(rule, metric_name, labels)
        
        # Check for existing alert (deduplication)
        if fingerprint in self.alert_fingerprints:
            existing_alert_id = self.alert_fingerprints[fingerprint]
            if existing_alert_id in self.active_alerts:
                # Update existing alert
                alert = self.active_alerts[existing_alert_id]
                alert.metric_value = value
                alert.last_updated = datetime.utcnow()
                return
        
        # Create new alert
        alert_id = f"alert_{int(time.time())}_{rule.id}"
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            started_at=datetime.utcnow(),
            metric_name=metric_name,
            metric_value=value,
            threshold_value=rule.threshold.value,
            labels=labels
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_fingerprints[fingerprint] = alert_id
        
        logger.warning(
            "🚨 Alert triggered",
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            metric_name=metric_name,
            value=value,
            threshold=rule.threshold.value
        )
        
        # Send notifications
        await self._send_alert_notification(alert, "triggered")
        
        # Record metrics
        metrics_exporter = get_metrics_exporter()
        if metrics_exporter:
            metrics_exporter.record_alert(rule.name, rule.severity, "alerting")
    
    async def _check_alert_resolution(self, rule: AlertRule, metric_name: str, labels: Dict[str, str]) -> None:
        """Check if an alert should be resolved."""
        fingerprint = self._create_alert_fingerprint(rule, metric_name, labels)
        
        if fingerprint in self.alert_fingerprints:
            alert_id = self.alert_fingerprints[fingerprint]
            if alert_id in self.active_alerts:
                self._resolve_alert(alert_id, "Threshold no longer exceeded")
    
    def _resolve_alert(self, alert_id: str, reason: str) -> None:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return
        
        alert = self.active_alerts.pop(alert_id)
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.last_updated = datetime.utcnow()
        
        # Remove from fingerprint tracking
        fingerprint_to_remove = None
        for fp, aid in self.alert_fingerprints.items():
            if aid == alert_id:
                fingerprint_to_remove = fp
                break
        
        if fingerprint_to_remove:
            del self.alert_fingerprints[fingerprint_to_remove]
        
        # Add to resolved alerts
        self.resolved_alerts.append(alert)
        if len(self.resolved_alerts) > self.max_resolved_alerts:
            self.resolved_alerts = self.resolved_alerts[-self.max_resolved_alerts:]
        
        logger.info(
            "✅ Alert resolved",
            alert_id=alert_id,
            reason=reason,
            duration_seconds=(alert.resolved_at - alert.started_at).total_seconds()
        )
        
        # Send resolution notification
        asyncio.create_task(self._send_alert_notification(alert, "resolved"))
    
    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to the next level."""
        alert.escalation_level += 1
        alert.last_updated = datetime.utcnow()
        
        logger.warning(
            "📈 Alert escalated",
            alert_id=alert.id,
            escalation_level=alert.escalation_level
        )
        
        # Send escalation notification
        await self._send_alert_notification(alert, "escalated")
    
    async def _send_alert_notification(self, alert: Alert, action: str) -> None:
        """Send alert notification through configured channels."""
        if not self.hook_manager:
            return
        
        rule = self.rules.get(alert.rule_id)
        if not rule:
            return
        
        # Prepare notification payload
        notification_payload = {
            "action": action,
            "alert": alert.to_dict(),
            "rule": {
                "name": rule.name,
                "description": rule.description,
                "runbook_url": rule.runbook_url,
                "dashboard_url": rule.dashboard_url,
                "tags": rule.tags
            }
        }
        
        # Send to configured notification channels
        for channel in rule.notification_channels:
            try:
                await self.hook_manager.trigger_hooks(
                    event_type=HookEvent.SYSTEM_ALERT,
                    payload=notification_payload,
                    session_id=None,
                    agent_id=None
                )
                
                # Track notification
                alert.notifications_sent.append({
                    "channel": channel,
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True
                })
                
            except Exception as e:
                logger.error(
                    "❌ Failed to send alert notification",
                    alert_id=alert.id,
                    channel=channel,
                    error=str(e)
                )
                
                alert.notifications_sent.append({
                    "channel": channel,
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": False,
                    "error": str(e)
                })
    
    async def _unsilence_after_delay(self, alert_id: str, delay_minutes: int) -> None:
        """Unsilence an alert after specified delay."""
        await asyncio.sleep(delay_minutes * 60)
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            if alert.status == AlertStatus.SILENCED:
                alert.status = AlertStatus.FIRING
                alert.last_updated = datetime.utcnow()
                
                logger.info("🔊 Alert unsilenced", alert_id=alert_id)
    
    def _create_metric_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return metric_name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}{{{label_str}}}"
    
    def _create_alert_fingerprint(self, rule: AlertRule, metric_name: str, labels: Dict[str, str]) -> str:
        """Create a fingerprint for alert deduplication."""
        # Include rule ID and grouping labels
        fingerprint_parts = [rule.id, metric_name]
        
        for label_key in rule.grouping_labels:
            if label_key in labels:
                fingerprint_parts.append(f"{label_key}={labels[label_key]}")
        
        return ":".join(fingerprint_parts)
    
    def _cleanup_metric_history(self) -> None:
        """Clean up old metric history data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for metric_key in list(self.metric_history.keys()):
            history = self.metric_history[metric_key]
            
            # Remove old entries
            while history and history[0][0] < cutoff_time:
                history.popleft()
            
            # Remove empty histories
            if not history:
                del self.metric_history[metric_key]

# Pre-configured alert rules for common scenarios
def create_default_alert_rules() -> List[AlertRule]:
    """Create default alert rules for common monitoring scenarios."""
    return [
        AlertRule(
            id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 80% for more than 2 minutes",
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="system_cpu_usage_percent",
                operator=ThresholdOperator.GREATER_THAN,
                value=80.0,
                duration_seconds=120
            ),
            notification_channels=["slack", "email"]
        ),
        
        AlertRule(
            id="critical_cpu_usage", 
            name="Critical CPU Usage",
            description="CPU usage is above 95% for more than 1 minute",
            severity=AlertSeverity.CRITICAL,
            threshold=MetricThreshold(
                metric_name="system_cpu_usage_percent",
                operator=ThresholdOperator.GREATER_THAN,
                value=95.0,
                duration_seconds=60
            ),
            notification_channels=["slack", "email", "pagerduty"]
        ),
        
        AlertRule(
            id="high_memory_usage",
            name="High Memory Usage", 
            description="Memory usage is above 85% for more than 2 minutes",
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="system_memory_usage_percent",
                operator=ThresholdOperator.GREATER_THAN,
                value=85.0,
                duration_seconds=120
            ),
            notification_channels=["slack"]
        ),
        
        AlertRule(
            id="high_error_rate",
            name="High HTTP Error Rate",
            description="HTTP error rate is above 5% for more than 2 minutes", 
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="http_error_rate",
                operator=ThresholdOperator.GREATER_THAN,
                value=0.05,
                duration_seconds=120
            ),
            notification_channels=["slack", "email"]
        ),
        
        AlertRule(
            id="slow_response_time",
            name="Slow Response Time",
            description="95th percentile response time is above 2 seconds",
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="http_response_time_p95",
                operator=ThresholdOperator.GREATER_THAN,
                value=2.0,
                duration_seconds=180
            ),
            notification_channels=["slack"]
        )
    ]

# Global alerting engine
_alerting_engine: Optional[AlertingEngine] = None

def get_alerting_engine() -> Optional[AlertingEngine]:
    """Get the global alerting engine."""
    return _alerting_engine

def set_alerting_engine(engine: AlertingEngine) -> None:
    """Set the global alerting engine."""
    global _alerting_engine
    _alerting_engine = engine
    logger.info("🚨 Global alerting engine set")