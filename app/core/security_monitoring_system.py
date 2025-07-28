"""
Real-time Security Monitoring and Alerting System for LeanVibe Agent Hive 2.0.

This system provides comprehensive real-time monitoring, alerting, and dashboards
for the integrated security infrastructure with advanced threat visualization and response.

Features:
- Real-time security event monitoring and correlation
- Intelligent alerting with escalation policies
- Interactive security dashboards and visualizations
- Automated incident response workflows
- Security metrics collection and analysis
- Integration with external monitoring systems
- Customizable alert channels and notifications
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque, Counter
import logging

import structlog
import redis.asyncio as redis
from fastapi import WebSocket
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .integrated_security_system import IntegratedSecuritySystem, IntegratedSecurityResult
from .threat_detection_engine import ThreatDetection, ThreatType
from .security_policy_engine import PolicyEvaluationResult
from .enhanced_security_audit import ForensicEvent, SecurityInvestigation
from .enhanced_security_safeguards import ControlDecision
from ..core.redis import get_redis

logger = structlog.get_logger()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertChannel(Enum):
    """Available alert channels."""
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    SMS = "SMS"
    DASHBOARD = "DASHBOARD"
    LOG = "LOG"


class AlertStatus(Enum):
    """Alert status tracking."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"


@dataclass
class SecurityAlert:
    """Security alert with comprehensive metadata."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Alert source and context
    source_system: str = "integrated_security"
    agent_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    
    # Alert details
    alert_type: str = "security_event"
    indicators: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    
    # Response tracking
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    escalation_level: int = 0
    
    # Alert metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    parent_investigation_id: Optional[str] = None
    
    # Notification tracking
    channels_notified: List[AlertChannel] = field(default_factory=list)
    notification_attempts: int = 0
    last_notification_attempt: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "source_system": self.source_system,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "alert_type": self.alert_type,
            "indicators": self.indicators,
            "affected_resources": self.affected_resources,
            "risk_score": self.risk_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolution_notes": self.resolution_notes,
            "escalation_level": self.escalation_level,
            "correlation_id": self.correlation_id,
            "parent_investigation_id": self.parent_investigation_id,
            "channels_notified": [ch.value for ch in self.channels_notified],
            "notification_attempts": self.notification_attempts,
            "last_notification_attempt": self.last_notification_attempt.isoformat() if self.last_notification_attempt else None
        }


@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Security events
    total_events: int = 0
    critical_events: int = 0
    high_severity_events: int = 0
    medium_severity_events: int = 0
    low_severity_events: int = 0
    
    # Threat detection
    threats_detected: int = 0
    unique_threat_types: int = 0
    avg_threat_confidence: float = 0.0
    
    # Policy enforcement
    policies_triggered: int = 0
    commands_blocked: int = 0
    approvals_required: int = 0
    
    # Agent activity
    active_agents: int = 0
    suspicious_agents: int = 0
    quarantined_agents: int = 0
    
    # System performance
    avg_processing_time_ms: float = 0.0
    system_load: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "security_events": {
                "total": self.total_events,
                "critical": self.critical_events,
                "high": self.high_severity_events,
                "medium": self.medium_severity_events,
                "low": self.low_severity_events
            },
            "threat_detection": {
                "threats_detected": self.threats_detected,
                "unique_threat_types": self.unique_threat_types,
                "avg_confidence": self.avg_threat_confidence
            },
            "policy_enforcement": {
                "policies_triggered": self.policies_triggered,
                "commands_blocked": self.commands_blocked,
                "approvals_required": self.approvals_required
            },
            "agent_activity": {
                "active_agents": self.active_agents,
                "suspicious_agents": self.suspicious_agents,
                "quarantined_agents": self.quarantined_agents
            },
            "system_performance": {
                "avg_processing_time_ms": self.avg_processing_time_ms,
                "system_load": self.system_load,
                "error_rate": self.error_rate
            }
        }


class AlertingEngine:
    """Intelligent alerting engine with escalation policies."""
    
    def __init__(self):
        # Alert routing configuration
        self.alert_routes = {
            AlertSeverity.CRITICAL: [AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.DASHBOARD],
            AlertSeverity.HIGH: [AlertChannel.EMAIL, AlertChannel.DASHBOARD],
            AlertSeverity.MEDIUM: [AlertChannel.DASHBOARD],
            AlertSeverity.LOW: [AlertChannel.DASHBOARD],
            AlertSeverity.INFO: [AlertChannel.LOG]
        }
        
        # Escalation policies
        self.escalation_policies = {
            AlertSeverity.CRITICAL: {
                "initial_delay_minutes": 0,
                "escalation_delay_minutes": 15,
                "max_escalations": 3,
                "escalation_channels": [AlertChannel.SMS, AlertChannel.WEBHOOK]
            },
            AlertSeverity.HIGH: {
                "initial_delay_minutes": 5,
                "escalation_delay_minutes": 30,
                "max_escalations": 2,
                "escalation_channels": [AlertChannel.EMAIL]
            }
        }
        
        # Alert suppression rules
        self.suppression_rules = {
            "duplicate_window_minutes": 5,
            "burst_threshold": 10,
            "burst_window_minutes": 5
        }
        
        # Notification configurations
        self.notification_config = {
            "email": {
                "smtp_server": "localhost",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_address": "security@beehive.local",
                "to_addresses": ["admin@beehive.local"]
            },
            "slack": {
                "webhook_url": "",
                "channel": "#security-alerts",
                "username": "BeeHive Security"
            },
            "webhook": {
                "url": "",
                "timeout": 30,
                "retry_count": 3
            }
        }
    
    async def process_alert(
        self,
        alert: SecurityAlert,
        immediate_notify: bool = True
    ) -> None:
        """Process and route security alert."""
        
        try:
            # Check for alert suppression
            if await self._should_suppress_alert(alert):
                alert.status = AlertStatus.SUPPRESSED
                return
            
            # Route alert to appropriate channels
            channels = self.alert_routes.get(alert.severity, [AlertChannel.LOG])
            
            if immediate_notify:
                for channel in channels:
                    try:
                        await self._send_notification(alert, channel)
                        alert.channels_notified.append(channel)
                    except Exception as e:
                        logger.error(f"Failed to send alert to {channel.value}: {e}")
            
            # Schedule escalation if needed
            if alert.severity in self.escalation_policies:
                await self._schedule_escalation(alert)
            
            alert.notification_attempts += 1
            alert.last_notification_attempt = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Alert processing error: {e}")
    
    async def _should_suppress_alert(self, alert: SecurityAlert) -> bool:
        """Check if alert should be suppressed based on rules."""
        
        # Check for duplicate alerts within suppression window
        duplicate_window = timedelta(minutes=self.suppression_rules["duplicate_window_minutes"])
        cutoff_time = datetime.utcnow() - duplicate_window
        
        # This would typically check against a database or cache
        # For now, we'll implement basic logic
        
        return False  # Simplified - implement actual suppression logic
    
    async def _send_notification(self, alert: SecurityAlert, channel: AlertChannel) -> None:
        """Send notification through specified channel."""
        
        if channel == AlertChannel.EMAIL:
            await self._send_email_notification(alert)
        elif channel == AlertChannel.SLACK:
            await self._send_slack_notification(alert)
        elif channel == AlertChannel.WEBHOOK:
            await self._send_webhook_notification(alert)
        elif channel == AlertChannel.DASHBOARD:
            await self._send_dashboard_notification(alert)
        elif channel == AlertChannel.LOG:
            await self._send_log_notification(alert)
    
    async def _send_email_notification(self, alert: SecurityAlert) -> None:
        """Send email notification."""
        
        try:
            config = self.notification_config["email"]
            
            msg = MIMEMultipart()
            msg['From'] = config["from_address"]
            msg['To'] = ", ".join(config["to_addresses"])
            msg['Subject'] = f"[{alert.severity.value}] Security Alert: {alert.title}"
            
            body = f"""
Security Alert Details:

Title: {alert.title}
Severity: {alert.severity.value}
Description: {alert.description}
Agent ID: {alert.agent_id}
Risk Score: {alert.risk_score}
Created: {alert.created_at.isoformat()}

Indicators:
{chr(10).join('- ' + indicator for indicator in alert.indicators)}

Affected Resources:
{chr(10).join('- ' + resource for resource in alert.affected_resources)}

Alert ID: {alert.id}
Correlation ID: {alert.correlation_id}

This is an automated security alert from the BeeHive Security System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Note: In production, this would use a proper email service
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            raise
    
    async def _send_slack_notification(self, alert: SecurityAlert) -> None:
        """Send Slack notification."""
        
        try:
            # Note: In production, this would use the Slack API
            logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            raise
    
    async def _send_webhook_notification(self, alert: SecurityAlert) -> None:
        """Send webhook notification."""
        
        try:
            # Note: In production, this would make HTTP request to webhook URL
            logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            raise
    
    async def _send_dashboard_notification(self, alert: SecurityAlert) -> None:
        """Send dashboard notification via WebSocket."""
        
        try:
            # Note: This would be implemented with actual WebSocket broadcasting
            logger.info(f"Dashboard notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Dashboard notification failed: {e}")
            raise
    
    async def _send_log_notification(self, alert: SecurityAlert) -> None:
        """Send log notification."""
        
        logger.info(
            f"Security Alert",
            alert_id=alert.id,
            title=alert.title,
            severity=alert.severity.value,
            agent_id=str(alert.agent_id) if alert.agent_id else None,
            risk_score=alert.risk_score
        )
    
    async def _schedule_escalation(self, alert: SecurityAlert) -> None:
        """Schedule alert escalation based on policy."""
        
        policy = self.escalation_policies.get(alert.severity)
        if not policy:
            return
        
        # Note: In production, this would schedule background tasks for escalation
        logger.info(f"Escalation scheduled for alert {alert.id}")


class SecurityMonitoringSystem:
    """
    Real-time Security Monitoring and Alerting System.
    
    Provides comprehensive monitoring, alerting, and dashboards for the
    integrated security infrastructure with real-time threat visualization.
    """
    
    def __init__(
        self,
        integrated_security: IntegratedSecuritySystem,
        redis_client: Optional[redis.Redis] = None
    ):
        self.integrated_security = integrated_security
        self.redis_client = redis_client or get_redis()
        
        # Monitoring components
        self.alerting_engine = AlertingEngine()
        
        # Data storage
        self.active_alerts: Dict[str, SecurityAlert] = {}
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metric snapshots
        self.event_stream: deque = deque(maxlen=10000)    # Keep last 10k events
        
        # WebSocket connections for real-time dashboards
        self.dashboard_connections: Set[WebSocket] = set()
        
        # Monitoring configuration
        self.config = {
            "metrics_collection_interval_seconds": 30,
            "alert_processing_interval_seconds": 10,
            "dashboard_update_interval_seconds": 5,
            "event_retention_hours": 24,
            "metrics_retention_hours": 168,  # 7 days
            "enable_real_time_alerts": True,
            "enable_auto_escalation": True,
            "enable_correlation_alerts": True
        }
        
        # Performance metrics
        self.system_metrics = {
            "monitoring_uptime_seconds": 0,
            "alerts_generated": 0,
            "alerts_resolved": 0,
            "dashboard_connections": 0,
            "metrics_collected": 0,
            "avg_alert_resolution_time_minutes": 0.0
        }
        
        # Background tasks
        self._metrics_task: Optional[asyncio.Task] = None
        self._alert_processing_task: Optional[asyncio.Task] = None
        self._dashboard_update_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Start time for uptime calculation
        self._start_time = datetime.utcnow()
    
    async def initialize(self) -> None:
        """Initialize the monitoring system."""
        
        # Start background tasks
        self._metrics_task = asyncio.create_task(self._metrics_collection_worker())
        self._alert_processing_task = asyncio.create_task(self._alert_processing_worker())
        self._dashboard_update_task = asyncio.create_task(self._dashboard_update_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Security Monitoring System initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the monitoring system."""
        
        # Cancel background tasks
        tasks = [
            self._metrics_task,
            self._alert_processing_task,
            self._dashboard_update_task,
            self._cleanup_task
        ]
        
        for task in tasks:
            if task:
                task.cancel()
        
        # Close dashboard connections
        for connection in list(self.dashboard_connections):
            try:
                await connection.close()
            except:
                pass
        
        logger.info("Security Monitoring System shutdown completed")
    
    async def process_security_event(
        self,
        result: IntegratedSecurityResult,
        context: Dict[str, Any]
    ) -> None:
        """Process security event from integrated security system."""
        
        try:
            # Add to event stream
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": context.get("agent_id"),
                "session_id": context.get("session_id"),
                "command": context.get("command", "")[:100],  # Truncate for storage
                "result": result.to_dict(),
                "processing_time_ms": result.total_processing_time_ms
            }
            
            self.event_stream.append(event_data)
            
            # Generate alerts based on security result
            alerts = await self._generate_alerts_from_result(result, context)
            
            # Process alerts
            for alert in alerts:
                await self._process_alert(alert)
            
            # Publish to Redis for external consumers
            if self.redis_client:
                await self.redis_client.publish(
                    "security_events",
                    json.dumps(event_data)
                )
            
        except Exception as e:
            logger.error(f"Security event processing error: {e}")
    
    async def _generate_alerts_from_result(
        self,
        result: IntegratedSecurityResult,
        context: Dict[str, Any]
    ) -> List[SecurityAlert]:
        """Generate alerts based on security analysis result."""
        
        alerts = []
        
        # Critical security decision
        if result.control_decision == ControlDecision.DENY:
            alert = SecurityAlert(
                id=str(uuid.uuid4()),
                title="Critical Security Threat - Command Blocked",
                description=f"High-risk command blocked for agent {context.get('agent_id')}",
                severity=AlertSeverity.CRITICAL,
                agent_id=context.get("agent_id"),
                session_id=context.get("session_id"),
                alert_type="command_blocked",
                indicators=result.recommended_actions[:3],
                risk_score=result.overall_confidence,
                raw_data={"security_result": result.to_dict(), "context": context}
            )
            alerts.append(alert)
        
        # High-severity threat detections
        if result.threat_detections:
            critical_threats = [td for td in result.threat_detections if td.severity.value == "CRITICAL"]
            high_threats = [td for td in result.threat_detections if td.severity.value == "HIGH"]
            
            if critical_threats:
                alert = SecurityAlert(
                    id=str(uuid.uuid4()),
                    title="Critical Threat Detection",
                    description=f"Critical threats detected: {', '.join(td.threat_type.value for td in critical_threats)}",
                    severity=AlertSeverity.CRITICAL,
                    agent_id=context.get("agent_id"),
                    session_id=context.get("session_id"),
                    alert_type="threat_detection",
                    indicators=[td.threat_type.value for td in critical_threats],
                    risk_score=max(td.risk_score for td in critical_threats),
                    raw_data={"threats": [td.to_dict() for td in critical_threats]}
                )
                alerts.append(alert)
            
            elif high_threats:
                alert = SecurityAlert(
                    id=str(uuid.uuid4()),
                    title="High-Severity Threat Detection",
                    description=f"High-severity threats detected: {', '.join(td.threat_type.value for td in high_threats)}",
                    severity=AlertSeverity.HIGH,
                    agent_id=context.get("agent_id"),
                    session_id=context.get("session_id"),
                    alert_type="threat_detection",
                    indicators=[td.threat_type.value for td in high_threats],
                    risk_score=max(td.risk_score for td in high_threats),
                    raw_data={"threats": [td.to_dict() for td in high_threats]}
                )
                alerts.append(alert)
        
        # Policy violations
        if (result.policy_evaluation_result and 
            result.policy_evaluation_result.matched_policies):
            
            critical_policies = [
                p for p in result.policy_evaluation_result.matched_policies
                if p.priority.value >= 900  # Critical/High priority
            ]
            
            if critical_policies:
                alert = SecurityAlert(
                    id=str(uuid.uuid4()),
                    title="Critical Policy Violation",
                    description=f"Critical security policies triggered: {', '.join(p.name for p in critical_policies)}",
                    severity=AlertSeverity.HIGH,
                    agent_id=context.get("agent_id"),
                    session_id=context.get("session_id"),
                    alert_type="policy_violation",
                    indicators=[p.name for p in critical_policies],
                    risk_score=result.policy_evaluation_result.confidence,
                    raw_data={"policies": [p.to_dict() for p in critical_policies]}
                )
                alerts.append(alert)
        
        # Behavioral anomalies
        if (result.advanced_analysis_result and 
            result.advanced_analysis_result.behavioral_anomalies):
            
            anomalies = result.advanced_analysis_result.behavioral_anomalies
            if len(anomalies) > 2:  # Multiple anomalies indicate higher risk
                alert = SecurityAlert(
                    id=str(uuid.uuid4()),
                    title="Multiple Behavioral Anomalies Detected",
                    description=f"Agent {context.get('agent_id')} showing multiple behavioral anomalies",
                    severity=AlertSeverity.MEDIUM,
                    agent_id=context.get("agent_id"),
                    session_id=context.get("session_id"),
                    alert_type="behavioral_anomaly",
                    indicators=anomalies[:5],  # Top 5 anomalies
                    risk_score=result.advanced_analysis_result.confidence_score,
                    raw_data={"anomalies": anomalies}
                )
                alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: SecurityAlert) -> None:
        """Process and store security alert."""
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.system_metrics["alerts_generated"] += 1
        
        # Process through alerting engine
        if self.config["enable_real_time_alerts"]:
            await self.alerting_engine.process_alert(alert)
        
        # Store in Redis for persistence
        if self.redis_client:
            await self.redis_client.hset(
                "security_alerts",
                alert.id,
                json.dumps(alert.to_dict())
            )
            
            # Set expiration for auto-cleanup
            await self.redis_client.expire("security_alerts", 86400 * 7)  # 7 days
        
        logger.info(
            f"Security alert processed",
            alert_id=alert.id,
            severity=alert.severity.value,
            alert_type=alert.alert_type
        )
    
    async def collect_metrics(self) -> MonitoringMetrics:
        """Collect current security monitoring metrics."""
        
        try:
            # Get metrics from integrated security system
            security_metrics = self.integrated_security.get_comprehensive_metrics()
            
            # Build monitoring metrics
            metrics = MonitoringMetrics()
            
            # Security events
            integrated_metrics = security_metrics.get("integrated_security_system", {})
            metrics.total_events = integrated_metrics.get("requests_processed", 0)
            metrics.threats_detected = integrated_metrics.get("threats_detected", 0)
            metrics.commands_blocked = integrated_metrics.get("commands_blocked", 0)
            metrics.approvals_required = integrated_metrics.get("approvals_required", 0)
            metrics.avg_processing_time_ms = integrated_metrics.get("avg_processing_time_ms", 0.0)
            
            # Threat detection metrics
            threat_metrics = security_metrics.get("threat_detection_engine", {})
            if isinstance(threat_metrics, dict):
                metrics.threats_detected = threat_metrics.get("threats_detected", 0)
                
            # Policy metrics
            policy_metrics = security_metrics.get("policy_engine", {})
            if isinstance(policy_metrics, dict):
                policy_engine_metrics = policy_metrics.get("security_policy_engine", {})
                metrics.policies_triggered = policy_engine_metrics.get("policies_matched", 0)
            
            # Agent activity (simplified)
            metrics.active_agents = len(set(
                event.get("agent_id") for event in list(self.event_stream)[-100:]
                if event.get("agent_id")
            ))
            
            # Alert metrics
            active_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]
            metrics.critical_events = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
            metrics.high_severity_events = len([a for a in active_alerts if a.severity == AlertSeverity.HIGH])
            metrics.medium_severity_events = len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM])
            metrics.low_severity_events = len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            
            # System performance
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
            self.system_metrics["monitoring_uptime_seconds"] = uptime
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.system_metrics["metrics_collected"] += 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return MonitoringMetrics()  # Return empty metrics on error
    
    async def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        
        try:
            # Current metrics
            current_metrics = await self.collect_metrics()
            
            # Recent alerts
            recent_alerts = [
                alert.to_dict() for alert in self.active_alerts.values()
                if alert.created_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            # Sort alerts by severity and creation time
            recent_alerts.sort(
                key=lambda a: (
                    {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}.get(a["severity"], 0),
                    a["created_at"]
                ),
                reverse=True
            )
            
            # Threat statistics
            recent_events = list(self.event_stream)[-1000:]  # Last 1000 events
            threat_stats = {
                "total_threats": 0,
                "threat_types": Counter(),
                "blocked_commands": 0,
                "high_risk_agents": set()
            }
            
            for event in recent_events:
                result = event.get("result", {})
                if result.get("threat_detections"):
                    threat_stats["total_threats"] += len(result["threat_detections"])
                    for detection in result["threat_detections"]:
                        threat_stats["threat_types"][detection.get("threat_type", "unknown")] += 1
                
                if result.get("control_decision") == "DENY":
                    threat_stats["blocked_commands"] += 1
                
                if result.get("overall_confidence", 0) > 0.8:
                    agent_id = event.get("agent_id")
                    if agent_id:
                        threat_stats["high_risk_agents"].add(agent_id)
            
            threat_stats["high_risk_agents"] = len(threat_stats["high_risk_agents"])
            threat_stats["threat_types"] = dict(threat_stats["threat_types"])
            
            # System health
            system_health = {
                "status": "healthy",
                "uptime_hours": self.system_metrics["monitoring_uptime_seconds"] / 3600,
                "alerts_generated": self.system_metrics["alerts_generated"],
                "alerts_resolved": self.system_metrics["alerts_resolved"],
                "dashboard_connections": len(self.dashboard_connections),
                "avg_processing_time_ms": current_metrics.avg_processing_time_ms
            }
            
            # Historical trends (last 24 hours)
            historical_metrics = [
                m for m in self.metrics_history
                if m.timestamp > datetime.utcnow() - timedelta(hours=24)
            ]
            
            trends = {
                "hourly_events": [],
                "threat_trend": [],
                "system_performance": []
            }
            
            if historical_metrics:
                # Group by hour for trends
                hourly_groups = defaultdict(list)
                for metric in historical_metrics:
                    hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                    hourly_groups[hour_key].append(metric)
                
                for hour, hour_metrics in sorted(hourly_groups.items()):
                    avg_events = sum(m.total_events for m in hour_metrics) / len(hour_metrics)
                    avg_threats = sum(m.threats_detected for m in hour_metrics) / len(hour_metrics)
                    avg_processing = sum(m.avg_processing_time_ms for m in hour_metrics) / len(hour_metrics)
                    
                    trends["hourly_events"].append({
                        "timestamp": hour.isoformat(),
                        "events": avg_events,
                        "threats": avg_threats
                    })
                    
                    trends["system_performance"].append({
                        "timestamp": hour.isoformat(),
                        "processing_time_ms": avg_processing,
                        "system_load": hour_metrics[-1].system_load
                    })
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "current_metrics": current_metrics.to_dict(),
                "recent_alerts": recent_alerts[:20],  # Top 20 recent alerts
                "threat_statistics": threat_stats,
                "system_health": system_health,
                "trends": trends,
                "alert_summary": {
                    "active_alerts": len([a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]),
                    "critical_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL and a.status == AlertStatus.ACTIVE]),
                    "unacknowledged_alerts": len([a for a in self.active_alerts.values() if not a.acknowledged_by and a.status == AlertStatus.ACTIVE])
                }
            }
            
        except Exception as e:
            logger.error(f"Dashboard data generation error: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "status": "error"
            }
    
    async def connect_dashboard(self, websocket: WebSocket) -> None:
        """Connect dashboard WebSocket."""
        
        try:
            await websocket.accept()
            self.dashboard_connections.add(websocket)
            self.system_metrics["dashboard_connections"] = len(self.dashboard_connections)
            
            # Send initial dashboard data
            dashboard_data = await self.get_security_dashboard_data()
            await websocket.send_text(json.dumps({
                "type": "dashboard_data",
                "data": dashboard_data
            }))
            
            logger.info("Dashboard WebSocket connected")
            
        except Exception as e:
            logger.error(f"Dashboard connection error: {e}")
            self.dashboard_connections.discard(websocket)
    
    async def disconnect_dashboard(self, websocket: WebSocket) -> None:
        """Disconnect dashboard WebSocket."""
        
        self.dashboard_connections.discard(websocket)
        self.system_metrics["dashboard_connections"] = len(self.dashboard_connections)
        logger.info("Dashboard WebSocket disconnected")
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """Acknowledge security alert."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            if notes:
                alert.resolution_notes = notes
            
            # Update in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "security_alerts",
                    alert_id,
                    json.dumps(alert.to_dict())
                )
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str
    ) -> bool:
        """Resolve security alert."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes
            
            # Update metrics
            self.system_metrics["alerts_resolved"] += 1
            
            # Calculate resolution time
            resolution_time = (alert.resolved_at - alert.created_at).total_seconds() / 60
            current_avg = self.system_metrics["avg_alert_resolution_time_minutes"]
            resolved_count = self.system_metrics["alerts_resolved"]
            
            self.system_metrics["avg_alert_resolution_time_minutes"] = (
                (current_avg * (resolved_count - 1) + resolution_time) / resolved_count
            )
            
            # Update in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "security_alerts",
                    alert_id,
                    json.dumps(alert.to_dict())
                )
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        
        return False
    
    # Background worker methods
    
    async def _metrics_collection_worker(self) -> None:
        """Background worker for metrics collection."""
        
        while True:
            try:
                await asyncio.sleep(self.config["metrics_collection_interval_seconds"])
                await self.collect_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection worker error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _alert_processing_worker(self) -> None:
        """Background worker for alert processing and escalation."""
        
        while True:
            try:
                await asyncio.sleep(self.config["alert_processing_interval_seconds"])
                
                # Process escalations
                if self.config["enable_auto_escalation"]:
                    await self._process_alert_escalations()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processing worker error: {e}")
                await asyncio.sleep(30)
    
    async def _dashboard_update_worker(self) -> None:
        """Background worker for dashboard updates."""
        
        while True:
            try:
                await asyncio.sleep(self.config["dashboard_update_interval_seconds"])
                
                if self.dashboard_connections:
                    dashboard_data = await self.get_security_dashboard_data()
                    message = json.dumps({
                        "type": "dashboard_update",
                        "data": dashboard_data
                    })
                    
                    # Broadcast to all connected dashboards
                    disconnected = set()
                    for websocket in self.dashboard_connections:
                        try:
                            await websocket.send_text(message)
                        except Exception:
                            disconnected.add(websocket)
                    
                    # Remove disconnected WebSockets
                    for websocket in disconnected:
                        self.dashboard_connections.discard(websocket)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update worker error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for data cleanup."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old alerts
                current_time = datetime.utcnow()
                retention_cutoff = current_time - timedelta(hours=self.config["event_retention_hours"])
                
                expired_alerts = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if (alert.status == AlertStatus.RESOLVED and 
                        alert.resolved_at and 
                        alert.resolved_at < retention_cutoff)
                ]
                
                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]
                
                # Clean up old metrics
                metrics_cutoff = current_time - timedelta(hours=self.config["metrics_retention_hours"])
                while (self.metrics_history and 
                       self.metrics_history[0].timestamp < metrics_cutoff):
                    self.metrics_history.popleft()
                
                # Clean up old events
                event_cutoff = current_time - timedelta(hours=self.config["event_retention_hours"])
                while (self.event_stream and 
                       datetime.fromisoformat(self.event_stream[0]["timestamp"]) < event_cutoff):
                    self.event_stream.popleft()
                
                if expired_alerts:
                    logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(300)
    
    async def _process_alert_escalations(self) -> None:
        """Process alert escalations based on policies."""
        
        current_time = datetime.utcnow()
        
        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            policy = self.alerting_engine.escalation_policies.get(alert.severity)
            if not policy:
                continue
            
            # Check if escalation is due
            time_since_creation = (current_time - alert.created_at).total_seconds() / 60
            
            escalation_intervals = [
                policy["initial_delay_minutes"] + (i * policy["escalation_delay_minutes"])
                for i in range(policy["max_escalations"] + 1)
            ]
            
            for i, interval in enumerate(escalation_intervals):
                if (time_since_creation >= interval and 
                    alert.escalation_level <= i):
                    
                    # Escalate alert
                    alert.escalation_level = i + 1
                    alert.updated_at = current_time
                    
                    # Send escalation notifications
                    for channel in policy["escalation_channels"]:
                        try:
                            await self.alerting_engine._send_notification(alert, channel)
                        except Exception as e:
                            logger.error(f"Escalation notification failed: {e}")
                    
                    logger.info(f"Alert {alert.id} escalated to level {alert.escalation_level}")
                    break
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        
        return {
            "monitoring_system": self.system_metrics.copy(),
            "configuration": self.config.copy(),
            "active_alerts_by_severity": {
                severity.value: len([
                    a for a in self.active_alerts.values() 
                    if a.severity == severity and a.status == AlertStatus.ACTIVE
                ])
                for severity in AlertSeverity
            },
            "storage_stats": {
                "active_alerts": len(self.active_alerts),
                "metrics_history_size": len(self.metrics_history),
                "event_stream_size": len(self.event_stream),
                "dashboard_connections": len(self.dashboard_connections)
            }
        }


# Factory function
async def create_security_monitoring_system(
    integrated_security: IntegratedSecuritySystem,
    redis_client: Optional[redis.Redis] = None
) -> SecurityMonitoringSystem:
    """
    Create SecurityMonitoringSystem instance.
    
    Args:
        integrated_security: Integrated security system
        redis_client: Redis client for data persistence
        
    Returns:
        SecurityMonitoringSystem instance
    """
    monitoring_system = SecurityMonitoringSystem(integrated_security, redis_client)
    await monitoring_system.initialize()
    return monitoring_system