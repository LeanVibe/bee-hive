"""
Intelligent Alerting System for Context Performance Issues.

Advanced alerting system with ML-based anomaly detection, smart alert routing,
noise reduction, and escalation management for the Context Engine.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis

from ..core.redis import get_redis_client
from ..core.context_performance_monitor import (
    ContextPerformanceMonitor,
    get_context_performance_monitor,
    PerformanceIssueType
)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels for alerts."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DASHBOARD = "dashboard"
    LOG = "log"


@dataclass
class AlertRule:
    """Defines an alert rule with conditions and actions."""
    rule_id: str
    name: str
    description: str
    component: str
    metric: str
    condition: str  # "greater_than", "less_than", "percentage_change", "anomaly"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 15
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents an active alert."""
    alert_id: str
    rule_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertStatus
    component: str
    metric: str
    current_value: float
    threshold_value: float
    message: str
    description: str
    first_detected: datetime
    last_updated: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notification_history: List[Dict[str, Any]] = field(default_factory=list)
    escalation_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection."""
    model_type: str = "isolation_forest"
    contamination: float = 0.1
    window_size: int = 100
    min_samples: int = 20
    sensitivity: float = 0.8


class MetricAnomalyDetector:
    """ML-based anomaly detector for metrics."""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.window_size))
        self.last_trained: Dict[str, datetime] = {}
    
    def add_metric_value(self, metric_key: str, value: float, timestamp: datetime) -> None:
        """Add a metric value for anomaly detection."""
        self.metric_history[metric_key].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Retrain model if enough new data
        if (len(self.metric_history[metric_key]) >= self.config.min_samples and
            (metric_key not in self.last_trained or 
             (datetime.utcnow() - self.last_trained[metric_key]).total_seconds() > 3600)):
            self._train_model(metric_key)
    
    def detect_anomaly(self, metric_key: str, value: float) -> Tuple[bool, float]:
        """
        Detect if a metric value is anomalous.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if metric_key not in self.models:
            return False, 0.0
        
        try:
            model = self.models[metric_key]
            scaler = self.scalers[metric_key]
            
            # Scale the value
            scaled_value = scaler.transform([[value]])
            
            # Get anomaly score
            anomaly_score = model.decision_function(scaled_value)[0]
            is_anomaly = model.predict(scaled_value)[0] == -1
            
            # Adjust based on sensitivity
            if not is_anomaly and abs(anomaly_score) > (1 - self.config.sensitivity):
                is_anomaly = True
            
            return is_anomaly, abs(anomaly_score)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_key}: {e}")
            return False, 0.0
    
    def _train_model(self, metric_key: str) -> None:
        """Train anomaly detection model for a metric."""
        try:
            history = list(self.metric_history[metric_key])
            if len(history) < self.config.min_samples:
                return
            
            # Prepare training data
            values = np.array([entry["value"] for entry in history]).reshape(-1, 1)
            
            # Scale the data
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values)
            
            # Train isolation forest
            if self.config.model_type == "isolation_forest":
                model = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42
                )
                model.fit(scaled_values)
                
                self.models[metric_key] = model
                self.scalers[metric_key] = scaler
                self.last_trained[metric_key] = datetime.utcnow()
                
                logger.info(f"Trained anomaly detection model for {metric_key}")
        
        except Exception as e:
            logger.error(f"Failed to train anomaly detection model for {metric_key}: {e}")


class AlertManager:
    """
    Intelligent alert management system with ML-based anomaly detection.
    
    Features:
    - Smart alert rules with flexible conditions
    - ML-based anomaly detection for metric trends
    - Alert deduplication and noise reduction
    - Intelligent escalation and routing
    - Multiple notification channels
    - Alert correlation and grouping
    - Performance impact analysis
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        performance_monitor: Optional[ContextPerformanceMonitor] = None
    ):
        """
        Initialize the alert manager.
        
        Args:
            redis_client: Redis client for persistent storage
            performance_monitor: Context performance monitor
        """
        self.redis_client = redis_client or get_redis_client()
        self.performance_monitor = performance_monitor
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Anomaly detection
        self.anomaly_detector = MetricAnomalyDetector(AnomalyDetectionConfig())
        
        # Alert correlation
        self.alert_correlations: Dict[str, Set[str]] = defaultdict(set)
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        
        # Notification tracking
        self.notification_queue: deque = deque()
        self.notification_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Initialize default alert rules
        self._initialize_default_rules()
        
        logger.info("Intelligent Alert Manager initialized")
    
    async def start(self) -> None:
        """Start the alert manager background processes."""
        logger.info("Starting intelligent alert manager")
        
        # Initialize performance monitor if not provided
        if self.performance_monitor is None:
            self.performance_monitor = await get_context_performance_monitor()
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._metric_collector()),
            asyncio.create_task(self._alert_evaluator()),
            asyncio.create_task(self._notification_dispatcher()),
            asyncio.create_task(self._alert_correlator()),
            asyncio.create_task(self._maintenance_scheduler())
        ])
    
    async def stop(self) -> None:
        """Stop the alert manager."""
        logger.info("Stopping intelligent alert manager")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False
    
    def enable_alert_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
            return True
        return False
    
    def disable_alert_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
            return True
        return False
    
    async def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        Evaluate metrics against alert rules and generate alerts.
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            List of new alerts generated
        """
        new_alerts = []
        current_time = datetime.utcnow()
        
        try:
            for metric_name, value in metrics.items():
                # Add to anomaly detector
                self.anomaly_detector.add_metric_value(metric_name, value, current_time)
                
                # Check anomaly detection
                is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(metric_name, value)
                
                if is_anomaly:
                    # Create anomaly alert
                    alert = await self._create_anomaly_alert(metric_name, value, anomaly_score)
                    if alert:
                        new_alerts.append(alert)
                
                # Evaluate rule-based alerts
                for rule in self.alert_rules.values():
                    if not rule.enabled or rule.metric != metric_name:
                        continue
                    
                    # Check if alert should be triggered
                    should_alert = await self._evaluate_alert_condition(rule, value, current_time)
                    
                    if should_alert:
                        alert = await self._create_rule_alert(rule, value, current_time)
                        if alert:
                            new_alerts.append(alert)
            
            # Store new alerts
            for alert in new_alerts:
                await self._store_alert(alert)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Failed to evaluate metrics: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.utcnow()
                alert.last_updated = datetime.utcnow()
                
                # Store updated alert
                await self._store_alert(alert)
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, user: str, resolution_note: str = "") -> bool:
        """Resolve an active alert."""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                alert.last_updated = datetime.utcnow()
                alert.metadata["resolved_by"] = user
                alert.metadata["resolution_note"] = resolution_note
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                # Store updated alert
                await self._store_alert(alert)
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def suppress_alerts(
        self,
        component: str,
        duration_minutes: int,
        reason: str,
        user: str
    ) -> bool:
        """Suppress alerts for a component for a specified duration."""
        try:
            suppression_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + timedelta(minutes=duration_minutes)
            
            self.suppression_rules[suppression_id] = {
                "component": component,
                "expires_at": expires_at,
                "reason": reason,
                "user": user,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Suppressed alerts for {component} until {expires_at} (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suppress alerts for {component}: {e}")
            return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status."""
        try:
            active_by_severity = defaultdict(int)
            active_by_component = defaultdict(int)
            
            for alert in self.active_alerts.values():
                active_by_severity[alert.severity.value] += 1
                active_by_component[alert.component] += 1
            
            # Recent alert trends
            recent_alerts = [
                alert for alert in self.alert_history
                if (datetime.utcnow() - alert.first_detected).total_seconds() < 86400  # Last 24h
            ]
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "active_alerts": {
                    "total": len(self.active_alerts),
                    "by_severity": dict(active_by_severity),
                    "by_component": dict(active_by_component)
                },
                "recent_activity": {
                    "alerts_last_24h": len(recent_alerts),
                    "resolved_last_24h": len([a for a in recent_alerts if a.status == AlertStatus.RESOLVED])
                },
                "alert_rules": {
                    "total": len(self.alert_rules),
                    "enabled": len([r for r in self.alert_rules.values() if r.enabled]),
                    "disabled": len([r for r in self.alert_rules.values() if not r.enabled])
                },
                "suppression_rules": {
                    "active": len([
                        s for s in self.suppression_rules.values()
                        if s["expires_at"] > datetime.utcnow()
                    ])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert summary: {e}")
            return {"error": str(e)}
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of active alerts with optional severity filter."""
        try:
            alerts = list(self.active_alerts.values())
            
            if severity_filter:
                try:
                    severity = AlertSeverity(severity_filter.lower())
                    alerts = [a for a in alerts if a.severity == severity]
                except ValueError:
                    pass
            
            # Sort by severity and creation time
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3,
                AlertSeverity.INFO: 4
            }
            
            alerts.sort(key=lambda a: (severity_order[a.severity], a.first_detected), reverse=True)
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "rule_id": alert.rule_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "component": alert.component,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "message": alert.message,
                    "description": alert.description,
                    "first_detected": alert.first_detected.isoformat(),
                    "last_updated": alert.last_updated.isoformat(),
                    "acknowledged_by": alert.acknowledged_by,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "escalation_level": alert.escalation_level,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    # Background task methods
    async def _metric_collector(self) -> None:
        """Background task to collect metrics for alert evaluation."""
        logger.info("Starting metric collector for alerting")
        
        while not self._shutdown_event.is_set():
            try:
                if self.performance_monitor:
                    # Get performance summary
                    summary = await self.performance_monitor.get_performance_summary(1)  # 1 hour window
                    
                    if "error" not in summary:
                        # Extract metrics for evaluation
                        metrics = {}
                        
                        # Search performance metrics
                        search_perf = summary.get("search_performance", {})
                        if not search_perf.get("error"):
                            metrics["search_avg_latency_ms"] = search_perf.get("avg_latency_ms", 0)
                            metrics["search_cache_hit_rate"] = search_perf.get("cache_hit_rate", 1.0)
                            if search_perf.get("avg_quality_score"):
                                metrics["search_quality_score"] = search_perf.get("avg_quality_score")
                        
                        # API cost metrics
                        api_costs = summary.get("api_costs", {})
                        if not api_costs.get("error"):
                            metrics["api_cost_per_hour"] = api_costs.get("avg_cost_per_call", 0) * api_costs.get("total_calls", 0)
                        
                        # Context metrics
                        context_metrics = summary.get("context_metrics", {})
                        if not context_metrics.get("error"):
                            metrics["total_contexts"] = context_metrics.get("total_contexts", 0)
                            metrics["storage_size_bytes"] = context_metrics.get("total_size_bytes", 0)
                        
                        # Evaluate metrics
                        if metrics:
                            new_alerts = await self.evaluate_metrics(metrics)
                            
                            if new_alerts:
                                logger.info(f"Generated {len(new_alerts)} new alerts")
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metric collector error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Metric collector stopped")
    
    async def _alert_evaluator(self) -> None:
        """Background task to evaluate alert conditions and escalations."""
        logger.info("Starting alert evaluator")
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Check for alert escalations
                for alert in list(self.active_alerts.values()):
                    # Check if alert needs escalation
                    if (alert.status == AlertStatus.OPEN and
                        (current_time - alert.first_detected).total_seconds() > 1800):  # 30 minutes
                        
                        await self._escalate_alert(alert)
                    
                    # Check if alert should auto-resolve
                    if await self._should_auto_resolve(alert):
                        await self.resolve_alert(alert.alert_id, "system", "Auto-resolved: conditions no longer met")
                
                await asyncio.sleep(300)  # Evaluate every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluator error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Alert evaluator stopped")
    
    async def _notification_dispatcher(self) -> None:
        """Background task to dispatch notifications."""
        logger.info("Starting notification dispatcher")
        
        while not self._shutdown_event.is_set():
            try:
                while self.notification_queue:
                    notification = self.notification_queue.popleft()
                    await self._send_notification(notification)
                
                await asyncio.sleep(10)  # Check queue every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Notification dispatcher error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Notification dispatcher stopped")
    
    async def _alert_correlator(self) -> None:
        """Background task to correlate related alerts."""
        logger.info("Starting alert correlator")
        
        while not self._shutdown_event.is_set():
            try:
                # Find related alerts based on component and timing
                await self._correlate_alerts()
                
                await asyncio.sleep(300)  # Correlate every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert correlator error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Alert correlator stopped")
    
    async def _maintenance_scheduler(self) -> None:
        """Background maintenance and cleanup scheduler."""
        logger.info("Starting alert maintenance scheduler")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old alert history
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.alert_history = deque([
                    alert for alert in self.alert_history
                    if alert.first_detected > cutoff_time
                ], maxlen=10000)
                
                # Clean up expired suppression rules
                expired_suppressions = [
                    suppression_id for suppression_id, rule in self.suppression_rules.items()
                    if rule["expires_at"] < datetime.utcnow()
                ]
                
                for suppression_id in expired_suppressions:
                    del self.suppression_rules[suppression_id]
                
                # Clean up notification history
                for alert_id in list(self.notification_history.keys()):
                    if alert_id not in self.active_alerts:
                        # Keep only recent notifications for closed alerts
                        notifications = self.notification_history[alert_id]
                        recent_notifications = [
                            n for n in notifications
                            if datetime.fromisoformat(n["sent_at"]) > cutoff_time
                        ]
                        
                        if recent_notifications:
                            self.notification_history[alert_id] = recent_notifications
                        else:
                            del self.notification_history[alert_id]
                
                await asyncio.sleep(3600)  # Maintenance every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert maintenance error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Alert maintenance scheduler stopped")
    
    # Helper methods
    def _initialize_default_rules(self) -> None:
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_search_latency",
                name="High Search Latency",
                description="Alert when search latency exceeds threshold",
                component="search",
                metric="search_avg_latency_ms",
                condition="greater_than",
                threshold=1000.0,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.DASHBOARD, NotificationChannel.LOG],
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                description="Alert when cache hit rate drops below threshold",
                component="cache",
                metric="search_cache_hit_rate",
                condition="less_than",
                threshold=0.8,
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.DASHBOARD, NotificationChannel.LOG],
                cooldown_minutes=30
            ),
            AlertRule(
                rule_id="high_api_costs",
                name="High API Costs",
                description="Alert when API costs exceed hourly threshold",
                component="embedding_api",
                metric="api_cost_per_hour",
                condition="greater_than",
                threshold=10.0,
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.DASHBOARD, NotificationChannel.LOG],
                cooldown_minutes=60
            ),
            AlertRule(
                rule_id="low_search_quality",
                name="Low Search Quality",
                description="Alert when search quality drops below threshold",
                component="search",
                metric="search_quality_score",
                condition="less_than",
                threshold=0.7,
                severity=AlertSeverity.HIGH,
                notification_channels=[NotificationChannel.DASHBOARD, NotificationChannel.LOG],
                cooldown_minutes=20
            ),
            AlertRule(
                rule_id="storage_growth_anomaly",
                name="Storage Growth Anomaly",
                description="Alert on unusual storage growth patterns",
                component="storage",
                metric="storage_size_bytes",
                condition="anomaly",
                threshold=0.8,  # Anomaly confidence threshold
                severity=AlertSeverity.MEDIUM,
                notification_channels=[NotificationChannel.DASHBOARD, NotificationChannel.LOG],
                cooldown_minutes=120
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        logger.info(f"Initialized {len(default_rules)} default alert rules")
    
    async def _evaluate_alert_condition(self, rule: AlertRule, value: float, current_time: datetime) -> bool:
        """Evaluate if an alert condition is met."""
        try:
            # Check cooldown period
            cooldown_key = f"alert_cooldown:{rule.rule_id}"
            last_alert = await self.redis_client.get(cooldown_key)
            
            if last_alert:
                last_alert_time = datetime.fromisoformat(last_alert.decode())
                if (current_time - last_alert_time).total_seconds() < (rule.cooldown_minutes * 60):
                    return False
            
            # Check if component is suppressed
            for suppression in self.suppression_rules.values():
                if (suppression["component"] == rule.component and
                    suppression["expires_at"] > current_time):
                    return False
            
            # Evaluate condition
            condition_met = False
            
            if rule.condition == "greater_than":
                condition_met = value > rule.threshold
            elif rule.condition == "less_than":
                condition_met = value < rule.threshold
            elif rule.condition == "percentage_change":
                # Get historical value for comparison
                historical_value = await self._get_historical_metric_value(rule.metric, hours_back=1)
                if historical_value:
                    change_percentage = abs(value - historical_value) / historical_value
                    condition_met = change_percentage > rule.threshold
            elif rule.condition == "anomaly":
                is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(rule.metric, value)
                condition_met = is_anomaly and anomaly_score > rule.threshold
            
            return condition_met
            
        except Exception as e:
            logger.error(f"Failed to evaluate alert condition for rule {rule.rule_id}: {e}")
            return False
    
    async def _create_rule_alert(self, rule: AlertRule, value: float, current_time: datetime) -> Optional[Alert]:
        """Create an alert from a rule evaluation."""
        try:
            alert_id = str(uuid.uuid4())
            
            # Check if similar alert already exists
            existing_alert = None
            for alert in self.active_alerts.values():
                if (alert.rule_id == rule.rule_id and
                    alert.status in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]):
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.current_value = value
                existing_alert.last_updated = current_time
                return None
            
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                alert_type="rule_based",
                severity=rule.severity,
                status=AlertStatus.OPEN,
                component=rule.component,
                metric=rule.metric,
                current_value=value,
                threshold_value=rule.threshold,
                message=f"{rule.name}: {rule.metric} = {value:.2f} (threshold: {rule.threshold:.2f})",
                description=rule.description,
                first_detected=current_time,
                last_updated=current_time,
                metadata=rule.metadata.copy()
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            # Set cooldown
            cooldown_key = f"alert_cooldown:{rule.rule_id}"
            await self.redis_client.setex(cooldown_key, rule.cooldown_minutes * 60, current_time.isoformat())
            
            # Queue notifications
            await self._queue_notifications(alert, rule.notification_channels)
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create rule alert: {e}")
            return None
    
    async def _create_anomaly_alert(self, metric_name: str, value: float, anomaly_score: float) -> Optional[Alert]:
        """Create an alert for detected anomaly."""
        try:
            alert_id = str(uuid.uuid4())
            
            # Check if similar anomaly alert already exists
            existing_alert = None
            for alert in self.active_alerts.values():
                if (alert.alert_type == "anomaly" and
                    alert.metric == metric_name and
                    alert.status in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]):
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert if anomaly score is higher
                if anomaly_score > existing_alert.metadata.get("anomaly_score", 0):
                    existing_alert.current_value = value
                    existing_alert.last_updated = datetime.utcnow()
                    existing_alert.metadata["anomaly_score"] = anomaly_score
                return None
            
            # Determine severity based on anomaly score
            if anomaly_score > 2.0:
                severity = AlertSeverity.CRITICAL
            elif anomaly_score > 1.5:
                severity = AlertSeverity.HIGH
            elif anomaly_score > 1.0:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alert = Alert(
                alert_id=alert_id,
                rule_id="anomaly_detection",
                alert_type="anomaly",
                severity=severity,
                status=AlertStatus.OPEN,
                component=metric_name.split("_")[0],  # Extract component from metric name
                metric=metric_name,
                current_value=value,
                threshold_value=anomaly_score,
                message=f"Anomaly detected in {metric_name}: value = {value:.2f} (anomaly score: {anomaly_score:.2f})",
                description=f"Machine learning detected unusual behavior in {metric_name}",
                first_detected=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                metadata={"anomaly_score": anomaly_score, "detection_method": "isolation_forest"}
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            # Queue notifications
            await self._queue_notifications(alert, [NotificationChannel.DASHBOARD, NotificationChannel.LOG])
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create anomaly alert: {e}")
            return None
    
    async def _queue_notifications(self, alert: Alert, channels: List[NotificationChannel]) -> None:
        """Queue notifications for an alert."""
        for channel in channels:
            notification = {
                "alert_id": alert.alert_id,
                "channel": channel.value,
                "alert_data": asdict(alert),
                "queued_at": datetime.utcnow().isoformat()
            }
            self.notification_queue.append(notification)
    
    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a notification through the specified channel."""
        try:
            channel = notification["channel"]
            alert_data = notification["alert_data"]
            
            if channel == "dashboard":
                # Store in Redis for dashboard display
                await self.redis_client.setex(
                    f"alert_notification:{alert_data['alert_id']}",
                    3600,  # 1 hour TTL
                    json.dumps(notification)
                )
            
            elif channel == "log":
                # Log the alert
                severity = alert_data["severity"]
                message = alert_data["message"]
                
                if severity == "critical":
                    logger.critical(f"ALERT: {message}")
                elif severity == "high":
                    logger.error(f"ALERT: {message}")
                elif severity == "medium":
                    logger.warning(f"ALERT: {message}")
                else:
                    logger.info(f"ALERT: {message}")
            
            elif channel == "webhook":
                # In a real implementation, send HTTP webhook
                logger.info(f"Would send webhook for alert: {alert_data['message']}")
            
            elif channel == "email":
                # In a real implementation, send email
                logger.info(f"Would send email for alert: {alert_data['message']}")
            
            elif channel == "slack":
                # In a real implementation, send Slack message
                logger.info(f"Would send Slack message for alert: {alert_data['message']}")
            
            # Record notification history
            notification_record = {
                "channel": channel,
                "sent_at": datetime.utcnow().isoformat(),
                "status": "sent"
            }
            
            self.notification_history[alert_data["alert_id"]].append(notification_record)
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to higher severity or additional channels."""
        try:
            alert.escalation_level += 1
            
            # Increase severity if not already critical
            if alert.severity != AlertSeverity.CRITICAL:
                if alert.severity == AlertSeverity.HIGH:
                    alert.severity = AlertSeverity.CRITICAL
                elif alert.severity == AlertSeverity.MEDIUM:
                    alert.severity = AlertSeverity.HIGH
                elif alert.severity == AlertSeverity.LOW:
                    alert.severity = AlertSeverity.MEDIUM
            
            alert.last_updated = datetime.utcnow()
            alert.metadata[f"escalated_at_level_{alert.escalation_level}"] = datetime.utcnow().isoformat()
            
            # Add additional notification channels for escalation
            escalation_channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
            await self._queue_notifications(alert, escalation_channels)
            
            logger.warning(f"Escalated alert {alert.alert_id} to level {alert.escalation_level}")
            
        except Exception as e:
            logger.error(f"Failed to escalate alert {alert.alert_id}: {e}")
    
    async def _should_auto_resolve(self, alert: Alert) -> bool:
        """Check if an alert should be auto-resolved."""
        try:
            # Only auto-resolve rule-based alerts
            if alert.alert_type != "rule_based":
                return False
            
            # Get current metric value
            current_metrics = await self._get_current_metrics([alert.metric])
            
            if alert.metric not in current_metrics:
                return False
            
            current_value = current_metrics[alert.metric]
            
            # Check if condition is no longer met
            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                return False
            
            condition_still_met = False
            
            if rule.condition == "greater_than":
                condition_still_met = current_value > rule.threshold
            elif rule.condition == "less_than":
                condition_still_met = current_value < rule.threshold
            
            # Auto-resolve if condition not met for 10 minutes
            if not condition_still_met:
                resolution_threshold = datetime.utcnow() - timedelta(minutes=10)
                if alert.last_updated < resolution_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check auto-resolve for alert {alert.alert_id}: {e}")
            return False
    
    async def _correlate_alerts(self) -> None:
        """Correlate related alerts to reduce noise."""
        try:
            # Group alerts by component and time window
            component_groups = defaultdict(list)
            
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.OPEN:
                    component_groups[alert.component].append(alert)
            
            # Find correlations within each component
            for component, alerts in component_groups.items():
                if len(alerts) > 1:
                    # Sort by creation time
                    alerts.sort(key=lambda a: a.first_detected)
                    
                    # Group alerts within 5-minute windows
                    time_groups = []
                    current_group = [alerts[0]]
                    
                    for alert in alerts[1:]:
                        time_diff = (alert.first_detected - current_group[-1].first_detected).total_seconds()
                        
                        if time_diff <= 300:  # 5 minutes
                            current_group.append(alert)
                        else:
                            if len(current_group) > 1:
                                time_groups.append(current_group)
                            current_group = [alert]
                    
                    if len(current_group) > 1:
                        time_groups.append(current_group)
                    
                    # Mark correlated alerts
                    for group in time_groups:
                        if len(group) > 1:
                            primary_alert = group[0]  # First alert becomes primary
                            
                            for alert in group[1:]:
                                self.alert_correlations[primary_alert.alert_id].add(alert.alert_id)
                                alert.metadata["correlated_with"] = primary_alert.alert_id
                                alert.metadata["correlation_reason"] = f"Related to {component} issues"
            
        except Exception as e:
            logger.error(f"Failed to correlate alerts: {e}")
    
    async def _get_historical_metric_value(self, metric: str, hours_back: int) -> Optional[float]:
        """Get historical metric value for comparison."""
        try:
            # This would typically query a time series database
            # For now, return None to indicate no historical data
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical metric value: {e}")
            return None
    
    async def _get_current_metrics(self, metric_names: List[str]) -> Dict[str, float]:
        """Get current values for specified metrics."""
        try:
            metrics = {}
            
            if self.performance_monitor:
                summary = await self.performance_monitor.get_performance_summary(1)
                
                if "error" not in summary:
                    search_perf = summary.get("search_performance", {})
                    api_costs = summary.get("api_costs", {})
                    context_metrics = summary.get("context_metrics", {})
                    
                    metric_mapping = {
                        "search_avg_latency_ms": search_perf.get("avg_latency_ms", 0),
                        "search_cache_hit_rate": search_perf.get("cache_hit_rate", 1.0),
                        "search_quality_score": search_perf.get("avg_quality_score", 1.0),
                        "api_cost_per_hour": api_costs.get("avg_cost_per_call", 0) * api_costs.get("total_calls", 0),
                        "total_contexts": context_metrics.get("total_contexts", 0),
                        "storage_size_bytes": context_metrics.get("total_size_bytes", 0)
                    }
                    
                    for metric_name in metric_names:
                        if metric_name in metric_mapping:
                            metrics[metric_name] = metric_mapping[metric_name]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    async def _store_alert(self, alert: Alert) -> None:
        """Store alert in Redis for persistence."""
        try:
            await self.redis_client.setex(
                f"alert:active:{alert.alert_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(asdict(alert), default=str)
            )
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")


# Global instance
_alert_manager: Optional[AlertManager] = None


async def get_alert_manager() -> AlertManager:
    """Get singleton alert manager instance."""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
        await _alert_manager.start()
    
    return _alert_manager


async def cleanup_alert_manager() -> None:
    """Cleanup alert manager resources."""
    global _alert_manager
    
    if _alert_manager:
        await _alert_manager.stop()
        _alert_manager = None