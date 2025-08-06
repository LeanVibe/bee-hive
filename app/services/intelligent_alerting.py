"""
Advanced Intelligent Alerting Service for LeanVibe Agent Hive 2.0

Enhanced alerting service with predictive capabilities, smart notifications,
and advanced escalation management integrated with the performance monitoring system.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

import structlog
import redis.asyncio as redis
import numpy as np
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.database import get_session
from ..core.redis import get_redis_client
from ..core.intelligent_alerting import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus, 
    NotificationChannel, get_alert_manager
)
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class PredictiveAlertType(Enum):
    """Types of predictive alerts."""
    THRESHOLD_APPROACHING = "threshold_approaching"
    TREND_DEGRADATION = "trend_degradation"
    CAPACITY_EXHAUSTION = "capacity_exhaustion"
    PERFORMANCE_REGRESSION = "performance_regression"
    ANOMALY_PATTERN = "anomaly_pattern"
    CORRELATION_BREAK = "correlation_break"


class EscalationLevel(Enum):
    """Alert escalation levels."""
    L1_MONITORING = "l1_monitoring"
    L2_TECHNICAL = "l2_technical"
    L3_ENGINEERING = "l3_engineering"
    L4_MANAGEMENT = "l4_management"
    L5_EXECUTIVE = "l5_executive"


@dataclass
class PredictiveAlert:
    """Predictive alert with forecasting information."""
    alert_id: str
    prediction_type: PredictiveAlertType
    metric_name: str
    component: str
    current_value: float
    predicted_value: float
    threshold_value: float
    time_to_threshold_hours: float
    confidence_score: float
    severity: AlertSeverity
    description: str
    recommendations: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationTemplate:
    """Template for alert notifications."""
    template_id: str
    channel: NotificationChannel
    severity: AlertSeverity
    subject_template: str
    body_template: str
    html_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts."""
    policy_id: str
    component: str
    severity: AlertSeverity
    escalation_steps: List[Dict[str, Any]]
    max_escalation_time_hours: int
    auto_escalation_enabled: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationManager:
    """Advanced notification manager with multiple channels and templates."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.notification_templates = {}
        self.notification_history = defaultdict(list)
        self.notification_stats = defaultdict(int)
        
        # Initialize default templates
        self._initialize_notification_templates()
        
        logger.info("Notification Manager initialized")
    
    def _initialize_notification_templates(self) -> None:
        """Initialize default notification templates."""
        templates = [
            NotificationTemplate(
                template_id="critical_system_alert",
                channel=NotificationChannel.EMAIL,
                severity=AlertSeverity.CRITICAL,
                subject_template="🚨 CRITICAL: {alert.component} - {alert.message}",
                body_template="""
CRITICAL ALERT DETECTED

Component: {alert.component}
Metric: {alert.metric}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Time Detected: {alert.first_detected}

Description: {alert.description}

This alert requires immediate attention. Please investigate and take corrective action.

Dashboard: {dashboard_url}
Runbook: {runbook_url}
                """,
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<div style="border-left: 5px solid #dc3545; padding: 20px; background-color: #f8f9fa;">
    <h2 style="color: #dc3545;">🚨 CRITICAL SYSTEM ALERT</h2>
    
    <table style="width: 100%; margin: 20px 0;">
        <tr><td><strong>Component:</strong></td><td>{alert.component}</td></tr>
        <tr><td><strong>Metric:</strong></td><td>{alert.metric}</td></tr>
        <tr><td><strong>Current Value:</strong></td><td>{alert.current_value}</td></tr>
        <tr><td><strong>Threshold:</strong></td><td>{alert.threshold_value}</td></tr>
        <tr><td><strong>Time Detected:</strong></td><td>{alert.first_detected}</td></tr>
    </table>
    
    <div style="background-color: white; padding: 15px; margin: 20px 0; border-radius: 5px;">
        <h3>Description</h3>
        <p>{alert.description}</p>
    </div>
    
    <div style="margin: 20px 0;">
        <a href="{dashboard_url}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 10px;">View Dashboard</a>
        <a href="{runbook_url}" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Runbook</a>
    </div>
</div>
</body>
</html>
                """
            ),
            NotificationTemplate(
                template_id="predictive_threshold_alert",
                channel=NotificationChannel.EMAIL,
                severity=AlertSeverity.MEDIUM,
                subject_template="⚠️  PREDICTIVE ALERT: {alert.component} threshold approaching",
                body_template="""
PREDICTIVE THRESHOLD ALERT

Component: {alert.component}
Metric: {alert.metric}
Current Value: {alert.current_value}
Predicted Value: {alert.predicted_value}
Threshold: {alert.threshold_value}
Time to Threshold: {alert.time_to_threshold_hours} hours
Confidence: {alert.confidence_score:.1%}

Based on current trends, this metric is predicted to exceed its threshold.

Recommendations:
{recommendations}

Dashboard: {dashboard_url}
                """,
                html_template="""
<html>
<body style="font-family: Arial, sans-serif;">
<div style="border-left: 5px solid #ffc107; padding: 20px; background-color: #fffbf0;">
    <h2 style="color: #856404;">⚠️ PREDICTIVE THRESHOLD ALERT</h2>
    
    <div style="background-color: white; padding: 15px; margin: 20px 0; border-radius: 5px;">
        <h3>Prediction Details</h3>
        <table style="width: 100%;">
            <tr><td><strong>Component:</strong></td><td>{alert.component}</td></tr>
            <tr><td><strong>Metric:</strong></td><td>{alert.metric}</td></tr>
            <tr><td><strong>Current Value:</strong></td><td>{alert.current_value}</td></tr>
            <tr><td><strong>Predicted Value:</strong></td><td>{alert.predicted_value}</td></tr>
            <tr><td><strong>Threshold:</strong></td><td>{alert.threshold_value}</td></tr>
            <tr><td><strong>Time to Threshold:</strong></td><td>{alert.time_to_threshold_hours} hours</td></tr>
            <tr><td><strong>Confidence:</strong></td><td>{alert.confidence_score:.1%}</td></tr>
        </table>
    </div>
    
    <div style="background-color: white; padding: 15px; margin: 20px 0; border-radius: 5px;">
        <h3>Recommendations</h3>
        <p>{recommendations}</p>
    </div>
    
    <div style="margin: 20px 0;">
        <a href="{dashboard_url}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Dashboard</a>
    </div>
</div>
</body>
</html>
                """
            ),
            NotificationTemplate(
                template_id="slack_critical_alert",
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.CRITICAL,
                subject_template="CRITICAL SYSTEM ALERT",
                body_template="""
{
    "text": "🚨 CRITICAL SYSTEM ALERT",
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 Critical System Alert"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*Component:*\\n{alert.component}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Severity:*\\n{alert.severity}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Metric:*\\n{alert.metric}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Value:*\\n{alert.current_value}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Description:*\\n{alert.description}"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Dashboard"
                    },
                    "url": "{dashboard_url}",
                    "style": "primary"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Acknowledge"
                    },
                    "action_id": "acknowledge_alert",
                    "value": "{alert.alert_id}"
                }
            ]
        }
    ]
}
                """
            )
        ]
        
        for template in templates:
            self.notification_templates[template.template_id] = template
    
    async def send_notification(
        self, 
        alert: Union[Alert, PredictiveAlert], 
        channel: NotificationChannel,
        escalation_level: Optional[EscalationLevel] = None
    ) -> bool:
        """Send notification through specified channel."""
        try:
            # Find appropriate template
            template = self._get_notification_template(alert.severity, channel)
            
            if not template:
                logger.warning(f"No template found for {alert.severity.value} {channel.value}")
                return False
            
            # Prepare notification content
            content = self._prepare_notification_content(alert, template, escalation_level)
            
            # Send through appropriate channel
            success = False
            if channel == NotificationChannel.EMAIL:
                success = await self._send_email_notification(alert, content)
            elif channel == NotificationChannel.SLACK:
                success = await self._send_slack_notification(alert, content)
            elif channel == NotificationChannel.SMS:
                success = await self._send_sms_notification(alert, content)
            elif channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook_notification(alert, content)
            elif channel == NotificationChannel.DASHBOARD:
                success = await self._send_dashboard_notification(alert, content)
            elif channel == NotificationChannel.LOG:
                success = await self._send_log_notification(alert, content)
            
            # Record notification attempt
            notification_record = {
                "alert_id": alert.alert_id,
                "channel": channel.value,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                "escalation_level": escalation_level.value if escalation_level else None
            }
            
            self.notification_history[alert.alert_id].append(notification_record)
            self.notification_stats[f"{channel.value}_{'success' if success else 'failure'}"] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send notification", 
                        alert_id=alert.alert_id, 
                        channel=channel.value, 
                        error=str(e))
            return False
    
    def _get_notification_template(
        self, 
        severity: AlertSeverity, 
        channel: NotificationChannel
    ) -> Optional[NotificationTemplate]:
        """Get appropriate notification template."""
        # Look for exact match first
        for template in self.notification_templates.values():
            if template.severity == severity and template.channel == channel:
                return template
        
        # Look for channel match with compatible severity
        compatible_templates = [
            template for template in self.notification_templates.values()
            if template.channel == channel
        ]
        
        if compatible_templates:
            return compatible_templates[0]  # Return first compatible template
        
        return None
    
    def _prepare_notification_content(
        self, 
        alert: Union[Alert, PredictiveAlert], 
        template: NotificationTemplate,
        escalation_level: Optional[EscalationLevel]
    ) -> Dict[str, str]:
        """Prepare notification content from template."""
        # Base context variables
        context = {
            "alert": alert,
            "dashboard_url": f"{settings.BASE_URL}/dashboard/alerts/{alert.alert_id}",
            "runbook_url": f"{settings.BASE_URL}/docs/runbooks/{alert.component}",
            "escalation_level": escalation_level.value if escalation_level else "none"
        }
        
        # Format templates
        subject = template.subject_template.format(**context)
        body = template.body_template.format(**context)
        
        content = {
            "subject": subject,
            "body": body
        }
        
        if template.html_template:
            content["html"] = template.html_template.format(**context)
        
        # Add recommendations for predictive alerts
        if isinstance(alert, PredictiveAlert):
            context["recommendations"] = "\n".join([f"• {rec}" for rec in alert.recommendations])
            content["body"] = content["body"].format(**context)
            if "html" in content:
                content["html"] = content["html"].format(**context)
        
        return content
    
    async def _send_email_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send email notification."""
        try:
            # Email configuration from settings
            smtp_server = getattr(settings, 'SMTP_SERVER', 'localhost')
            smtp_port = getattr(settings, 'SMTP_PORT', 587)
            smtp_user = getattr(settings, 'SMTP_USER', None)
            smtp_password = getattr(settings, 'SMTP_PASSWORD', None)
            from_email = getattr(settings, 'FROM_EMAIL', 'alerts@leanhive.com')
            to_emails = getattr(settings, 'ALERT_EMAIL_RECIPIENTS', ['admin@leanhive.com'])
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = content['subject']
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            
            # Add text part
            text_part = MimeText(content['body'], 'plain')
            msg.attach(text_part)
            
            # Add HTML part if available
            if 'html' in content:
                html_part = MimeText(content['html'], 'html')
                msg.attach(html_part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent successfully", alert_id=alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False
    
    async def _send_slack_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send Slack notification."""
        try:
            # In production, this would use the Slack Web API
            logger.info(f"Would send Slack notification: {content['subject']}", alert_id=alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False
    
    async def _send_sms_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send SMS notification."""
        try:
            # In production, this would use an SMS service like Twilio
            logger.info(f"Would send SMS notification: {content['subject']}", alert_id=alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False
    
    async def _send_webhook_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send webhook notification."""
        try:
            # In production, this would send HTTP POST to configured webhooks
            logger.info(f"Would send webhook notification", alert_id=alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False
    
    async def _send_dashboard_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send dashboard notification."""
        try:
            # Store in Redis for dashboard display
            notification_data = {
                "alert_id": alert.alert_id,
                "subject": content["subject"],
                "body": content["body"],
                "timestamp": datetime.utcnow().isoformat(),
                "severity": alert.severity.value
            }
            
            await self.redis_client.setex(
                f"dashboard_notification:{alert.alert_id}",
                3600,  # 1 hour TTL
                json.dumps(notification_data)
            )
            
            # Also add to notification stream
            await self.redis_client.xadd(
                "dashboard_notifications",
                notification_data,
                maxlen=1000
            )
            
            logger.info(f"Dashboard notification sent", alert_id=alert.alert_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send dashboard notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False
    
    async def _send_log_notification(self, alert: Union[Alert, PredictiveAlert], content: Dict[str, str]) -> bool:
        """Send log notification."""
        try:
            severity_map = {
                AlertSeverity.CRITICAL: logger.critical,
                AlertSeverity.HIGH: logger.error,
                AlertSeverity.MEDIUM: logger.warning,
                AlertSeverity.LOW: logger.info,
                AlertSeverity.INFO: logger.info
            }
            
            log_func = severity_map.get(alert.severity, logger.info)
            log_func(f"ALERT: {content['subject']}", 
                    alert_id=alert.alert_id, 
                    component=alert.component if hasattr(alert, 'component') else 'unknown')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send log notification", 
                        alert_id=alert.alert_id, 
                        error=str(e))
            return False


class PredictiveAlertEngine:
    """Advanced predictive alerting engine."""
    
    def __init__(
        self, 
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[callable] = None
    ):
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        
        self.predictive_alerts = {}
        self.prediction_models = {}
        self.threshold_predictions = defaultdict(dict)
        
        logger.info("Predictive Alert Engine initialized")
    
    async def generate_predictive_alerts(
        self, 
        time_horizon_hours: int = 4
    ) -> List[PredictiveAlert]:
        """Generate predictive alerts based on trend analysis."""
        try:
            predictive_alerts = []
            
            # Get recent performance metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)  # Look at last 24 hours for prediction
            
            async with self.session_factory() as session:
                query = select(PerformanceMetric).where(
                    PerformanceMetric.timestamp >= start_time
                ).order_by(PerformanceMetric.timestamp.asc())
                
                result = await session.execute(query)
                metrics = result.scalars().all()
            
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in metrics:
                metrics_by_name[metric.metric_name].append(metric)
            
            # Analyze each metric for predictive alerts
            for metric_name, metric_list in metrics_by_name.items():
                if len(metric_list) < 10:  # Need sufficient data for prediction
                    continue
                
                # Check for threshold approaching
                threshold_alert = await self._check_threshold_approaching(
                    metric_name, 
                    metric_list, 
                    time_horizon_hours
                )
                if threshold_alert:
                    predictive_alerts.append(threshold_alert)
                
                # Check for trend degradation
                trend_alert = await self._check_trend_degradation(
                    metric_name, 
                    metric_list, 
                    time_horizon_hours
                )
                if trend_alert:
                    predictive_alerts.append(trend_alert)
                
                # Check for capacity exhaustion
                capacity_alert = await self._check_capacity_exhaustion(
                    metric_name, 
                    metric_list, 
                    time_horizon_hours
                )
                if capacity_alert:
                    predictive_alerts.append(capacity_alert)
            
            return predictive_alerts
            
        except Exception as e:
            logger.error("Failed to generate predictive alerts", error=str(e))
            return []
    
    async def _check_threshold_approaching(
        self, 
        metric_name: str, 
        metric_list: List[PerformanceMetric], 
        time_horizon_hours: int
    ) -> Optional[PredictiveAlert]:
        """Check if metric is approaching a threshold."""
        try:
            # Define thresholds for different metrics
            thresholds = {
                "system.cpu.percent": 85.0,
                "system.memory.percent": 90.0,
                "search_avg_latency_ms": 2000.0,
                "api_cost_per_hour": 50.0,
                "error_rate.percent": 5.0
            }
            
            threshold = thresholds.get(metric_name)
            if not threshold:
                return None
            
            # Get recent values and calculate trend
            values = [m.metric_value for m in metric_list[-20:]]  # Last 20 points
            timestamps = [m.timestamp for m in metric_list[-20:]]
            
            if len(values) < 5:
                return None
            
            current_value = values[-1]
            
            # Simple linear regression for trend prediction
            x_values = np.arange(len(values))
            coeffs = np.polyfit(x_values, values, 1)
            slope = coeffs[0]
            
            # Project future value
            future_steps = time_horizon_hours * 2  # Assuming 30-minute intervals
            predicted_value = current_value + (slope * future_steps)
            
            # Check if predicted value will exceed threshold
            if ((slope > 0 and predicted_value > threshold and current_value < threshold) or
                (slope < 0 and predicted_value < threshold and current_value > threshold)):
                
                time_to_threshold = abs((threshold - current_value) / slope) * 0.5  # Convert to hours
                confidence_score = min(0.9, abs(slope) / max(abs(current_value), 1) * len(values) / 20)
                
                severity = AlertSeverity.HIGH if time_to_threshold < 2 else AlertSeverity.MEDIUM
                
                recommendations = self._get_threshold_recommendations(metric_name, slope > 0)
                
                return PredictiveAlert(
                    alert_id=str(uuid.uuid4()),
                    prediction_type=PredictiveAlertType.THRESHOLD_APPROACHING,
                    metric_name=metric_name,
                    component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                    current_value=current_value,
                    predicted_value=predicted_value,
                    threshold_value=threshold,
                    time_to_threshold_hours=time_to_threshold,
                    confidence_score=confidence_score,
                    severity=severity,
                    description=f"{metric_name} is predicted to {'exceed' if slope > 0 else 'fall below'} threshold of {threshold} in {time_to_threshold:.1f} hours",
                    recommendations=recommendations,
                    created_at=datetime.utcnow(),
                    metadata={
                        "trend_slope": slope,
                        "data_points": len(values),
                        "prediction_method": "linear_regression"
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check threshold approaching for {metric_name}", error=str(e))
            return None
    
    async def _check_trend_degradation(
        self, 
        metric_name: str, 
        metric_list: List[PerformanceMetric], 
        time_horizon_hours: int
    ) -> Optional[PredictiveAlert]:
        """Check for degrading performance trends."""
        try:
            # Only check metrics where higher values are worse
            degradation_metrics = [
                "search_avg_latency_ms", 
                "error_rate.percent", 
                "system.cpu.percent",
                "api_cost_per_hour"
            ]
            
            if metric_name not in degradation_metrics:
                return None
            
            values = [m.metric_value for m in metric_list[-30:]]  # Last 30 points
            
            if len(values) < 10:
                return None
            
            # Calculate trend over different windows
            recent_trend = np.polyfit(range(len(values[-10:])), values[-10:], 1)[0]
            overall_trend = np.polyfit(range(len(values)), values, 1)[0]
            
            # Check if recent trend is significantly worse than overall trend
            if recent_trend > 0 and recent_trend > overall_trend * 2:
                current_value = values[-1]
                predicted_value = current_value + (recent_trend * time_horizon_hours * 2)
                
                confidence_score = min(0.85, recent_trend / max(current_value, 1) * 10)
                
                recommendations = self._get_degradation_recommendations(metric_name)
                
                return PredictiveAlert(
                    alert_id=str(uuid.uuid4()),
                    prediction_type=PredictiveAlertType.TREND_DEGRADATION,
                    metric_name=metric_name,
                    component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                    current_value=current_value,
                    predicted_value=predicted_value,
                    threshold_value=current_value * 1.5,  # 50% increase threshold
                    time_to_threshold_hours=time_horizon_hours,
                    confidence_score=confidence_score,
                    severity=AlertSeverity.MEDIUM,
                    description=f"{metric_name} shows accelerating degradation trend (recent: {recent_trend:.2f}/hr vs overall: {overall_trend:.2f}/hr)",
                    recommendations=recommendations,
                    created_at=datetime.utcnow(),
                    metadata={
                        "recent_trend": recent_trend,
                        "overall_trend": overall_trend,
                        "degradation_ratio": recent_trend / max(overall_trend, 0.001)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check trend degradation for {metric_name}", error=str(e))
            return None
    
    async def _check_capacity_exhaustion(
        self, 
        metric_name: str, 
        metric_list: List[PerformanceMetric], 
        time_horizon_hours: int
    ) -> Optional[PredictiveAlert]:
        """Check for capacity exhaustion patterns."""
        try:
            capacity_metrics = {
                "system.memory.percent": 95.0,
                "storage_size_bytes": 1e12,  # 1TB
                "total_contexts": 100000,
                "agents.active": 50
            }
            
            max_capacity = capacity_metrics.get(metric_name)
            if not max_capacity:
                return None
            
            values = [m.metric_value for m in metric_list[-15:]]  # Last 15 points
            
            if len(values) < 8:
                return None
            
            current_value = values[-1]
            
            # Check if we're approaching capacity with accelerating growth
            growth_rate = np.polyfit(range(len(values)), values, 1)[0]
            
            if growth_rate > 0:
                time_to_capacity = (max_capacity - current_value) / growth_rate * 0.5  # Convert to hours
                
                if time_to_capacity > 0 and time_to_capacity <= time_horizon_hours * 2:  # Within 2x horizon
                    predicted_value = current_value + (growth_rate * time_horizon_hours * 2)
                    utilization_percent = (current_value / max_capacity) * 100
                    
                    if utilization_percent > 70:  # Only alert if already high utilization
                        confidence_score = min(0.8, growth_rate / current_value * len(values) / 15)
                        
                        recommendations = self._get_capacity_recommendations(metric_name)
                        
                        return PredictiveAlert(
                            alert_id=str(uuid.uuid4()),
                            prediction_type=PredictiveAlertType.CAPACITY_EXHAUSTION,
                            metric_name=metric_name,
                            component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                            current_value=current_value,
                            predicted_value=predicted_value,
                            threshold_value=max_capacity * 0.9,  # 90% of capacity
                            time_to_threshold_hours=time_to_capacity,
                            confidence_score=confidence_score,
                            severity=AlertSeverity.HIGH if time_to_capacity < 24 else AlertSeverity.MEDIUM,
                            description=f"{metric_name} is approaching capacity limit. Current: {utilization_percent:.1f}%, capacity exhaustion predicted in {time_to_capacity:.1f} hours",
                            recommendations=recommendations,
                            created_at=datetime.utcnow(),
                            metadata={
                                "max_capacity": max_capacity,
                                "current_utilization": utilization_percent,
                                "growth_rate": growth_rate
                            }
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check capacity exhaustion for {metric_name}", error=str(e))
            return None
    
    def _get_threshold_recommendations(self, metric_name: str, increasing: bool) -> List[str]:
        """Get recommendations for threshold approaching alerts."""
        recommendations_map = {
            "system.cpu.percent": [
                "Scale horizontally by adding more instances",
                "Optimize CPU-intensive processes",
                "Implement load balancing",
                "Review and optimize database queries"
            ] if increasing else [
                "Monitor for potential system issues causing low CPU usage",
                "Verify system health and workload distribution"
            ],
            "system.memory.percent": [
                "Increase memory allocation",
                "Implement memory optimization strategies",
                "Review memory leaks in applications",
                "Consider memory caching optimization"
            ] if increasing else [
                "Investigate potential memory management issues",
                "Verify application memory release patterns"
            ],
            "search_avg_latency_ms": [
                "Optimize search algorithms and indexes",
                "Implement caching for frequent queries",
                "Scale search infrastructure",
                "Review database performance"
            ] if increasing else [
                "Monitor for potential data quality issues",
                "Verify search result accuracy"
            ]
        }
        
        return recommendations_map.get(metric_name, [
            "Monitor metric closely",
            "Review system performance",
            "Consider scaling if needed"
        ])
    
    def _get_degradation_recommendations(self, metric_name: str) -> List[str]:
        """Get recommendations for trend degradation alerts."""
        return [
            f"Investigate root cause of {metric_name} degradation",
            "Review recent system changes or deployments",
            "Check for resource constraints or bottlenecks",
            "Consider implementing performance optimizations",
            "Monitor related metrics for correlation"
        ]
    
    def _get_capacity_recommendations(self, metric_name: str) -> List[str]:
        """Get recommendations for capacity exhaustion alerts."""
        capacity_recommendations = {
            "system.memory.percent": [
                "Increase memory allocation immediately",
                "Implement memory cleanup routines",
                "Review memory usage patterns",
                "Consider upgrading hardware"
            ],
            "storage_size_bytes": [
                "Implement data archival strategy",
                "Add additional storage capacity",
                "Review data retention policies",
                "Optimize storage usage"
            ],
            "total_contexts": [
                "Implement context cleanup routines",
                "Review context retention policies",
                "Optimize context storage efficiency",
                "Consider context archival"
            ]
        }
        
        return capacity_recommendations.get(metric_name, [
            "Scale capacity immediately",
            "Implement resource optimization",
            "Review usage patterns",
            "Plan for capacity expansion"
        ])


class EnhancedAlertingService:
    """Enhanced alerting service integrating predictive capabilities."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[callable] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.alert_manager = alert_manager
        
        self.notification_manager = NotificationManager(redis_client)
        self.predictive_engine = PredictiveAlertEngine(redis_client, session_factory)
        
        self.escalation_policies = {}
        self.active_escalations = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        self._initialize_escalation_policies()
        
        logger.info("Enhanced Alerting Service initialized")
    
    async def start(self) -> None:
        """Start the enhanced alerting service."""
        if self.is_running:
            logger.warning("Enhanced Alerting Service already running")
            return
        
        logger.info("Starting Enhanced Alerting Service")
        self.is_running = True
        
        # Initialize alert manager if not provided
        if self.alert_manager is None:
            self.alert_manager = await get_alert_manager()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._predictive_alerting_loop()),
            asyncio.create_task(self._escalation_management_loop()),
            asyncio.create_task(self._notification_dispatcher_loop())
        ]
        
        logger.info("Enhanced Alerting Service started successfully")
    
    async def stop(self) -> None:
        """Stop the enhanced alerting service."""
        if not self.is_running:
            return
        
        logger.info("Stopping Enhanced Alerting Service")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Enhanced Alerting Service stopped")
    
    def _initialize_escalation_policies(self) -> None:
        """Initialize default escalation policies."""
        policies = [
            EscalationPolicy(
                policy_id="critical_system_escalation",
                component="system",
                severity=AlertSeverity.CRITICAL,
                escalation_steps=[
                    {"level": EscalationLevel.L1_MONITORING, "delay_minutes": 0, "channels": [NotificationChannel.DASHBOARD, NotificationChannel.LOG]},
                    {"level": EscalationLevel.L2_TECHNICAL, "delay_minutes": 5, "channels": [NotificationChannel.EMAIL, NotificationChannel.SLACK]},
                    {"level": EscalationLevel.L3_ENGINEERING, "delay_minutes": 15, "channels": [NotificationChannel.EMAIL, NotificationChannel.SMS]},
                    {"level": EscalationLevel.L4_MANAGEMENT, "delay_minutes": 30, "channels": [NotificationChannel.EMAIL, NotificationChannel.SMS]}
                ],
                max_escalation_time_hours=2,
                auto_escalation_enabled=True
            ),
            EscalationPolicy(
                policy_id="high_priority_escalation",
                component="general",
                severity=AlertSeverity.HIGH,
                escalation_steps=[
                    {"level": EscalationLevel.L1_MONITORING, "delay_minutes": 0, "channels": [NotificationChannel.DASHBOARD, NotificationChannel.LOG]},
                    {"level": EscalationLevel.L2_TECHNICAL, "delay_minutes": 15, "channels": [NotificationChannel.EMAIL, NotificationChannel.SLACK]},
                    {"level": EscalationLevel.L3_ENGINEERING, "delay_minutes": 60, "channels": [NotificationChannel.EMAIL]}
                ],
                max_escalation_time_hours=4,
                auto_escalation_enabled=True
            )
        ]
        
        for policy in policies:
            self.escalation_policies[policy.policy_id] = policy
    
    async def _predictive_alerting_loop(self) -> None:
        """Background task for predictive alerting."""
        logger.info("Starting predictive alerting loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Generate predictive alerts
                predictive_alerts = await self.predictive_engine.generate_predictive_alerts()
                
                # Process each predictive alert
                for alert in predictive_alerts:
                    await self._handle_predictive_alert(alert)
                
                # Wait for next cycle (run every 30 minutes)
                await asyncio.sleep(1800)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Predictive alerting loop error", error=str(e))
                await asyncio.sleep(1800)
        
        logger.info("Predictive alerting loop stopped")
    
    async def _escalation_management_loop(self) -> None:
        """Background task for managing alert escalations."""
        logger.info("Starting escalation management loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for alerts that need escalation
                await self._process_escalations()
                
                # Wait for next cycle (run every 5 minutes)
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Escalation management loop error", error=str(e))
                await asyncio.sleep(300)
        
        logger.info("Escalation management loop stopped")
    
    async def _notification_dispatcher_loop(self) -> None:
        """Background task for dispatching notifications."""
        logger.info("Starting notification dispatcher loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Process pending notifications from Redis queue
                await self._process_notification_queue()
                
                # Wait for next cycle (run every 10 seconds)
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Notification dispatcher loop error", error=str(e))
                await asyncio.sleep(10)
        
        logger.info("Notification dispatcher loop stopped")
    
    async def _handle_predictive_alert(self, alert: PredictiveAlert) -> None:
        """Handle a predictive alert."""
        try:
            # Store predictive alert
            await self._store_predictive_alert(alert)
            
            # Send notifications
            channels = [NotificationChannel.DASHBOARD, NotificationChannel.EMAIL]
            
            for channel in channels:
                await self.notification_manager.send_notification(alert, channel)
            
            logger.info("Predictive alert handled", 
                       alert_id=alert.alert_id, 
                       type=alert.prediction_type.value)
            
        except Exception as e:
            logger.error("Failed to handle predictive alert", 
                        alert_id=alert.alert_id, 
                        error=str(e))
    
    async def _store_predictive_alert(self, alert: PredictiveAlert) -> None:
        """Store predictive alert in Redis."""
        try:
            alert_data = asdict(alert)
            # Convert datetime to ISO string for JSON serialization
            alert_data["created_at"] = alert.created_at.isoformat()
            
            await self.redis_client.setex(
                f"predictive_alert:{alert.alert_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(alert_data, default=str)
            )
            
            # Add to predictive alerts stream
            await self.redis_client.xadd(
                "predictive_alerts_stream",
                alert_data,
                maxlen=1000
            )
            
        except Exception as e:
            logger.error("Failed to store predictive alert", 
                        alert_id=alert.alert_id, 
                        error=str(e))
    
    async def _process_escalations(self) -> None:
        """Process alert escalations."""
        # This would integrate with the existing alert manager
        # to handle escalation logic
        pass
    
    async def _process_notification_queue(self) -> None:
        """Process pending notifications from Redis queue."""
        try:
            # Read from notification queue stream
            messages = await self.redis_client.xread(
                {"notification_queue": "$"},
                count=10,
                block=1000
            )
            
            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    await self._process_notification_message(fields)
                    
                    # Acknowledge processing
                    await self.redis_client.xdel("notification_queue", message_id)
            
        except Exception as e:
            logger.error("Failed to process notification queue", error=str(e))
    
    async def _process_notification_message(self, message_data: Dict[str, Any]) -> None:
        """Process individual notification message."""
        # Implementation for processing notification messages
        pass


# Global instance
_enhanced_alerting_service: Optional[EnhancedAlertingService] = None


async def get_enhanced_alerting_service() -> EnhancedAlertingService:
    """Get singleton enhanced alerting service instance."""
    global _enhanced_alerting_service
    
    if _enhanced_alerting_service is None:
        _enhanced_alerting_service = EnhancedAlertingService()
        await _enhanced_alerting_service.start()
    
    return _enhanced_alerting_service


async def cleanup_enhanced_alerting_service() -> None:
    """Cleanup enhanced alerting service resources."""
    global _enhanced_alerting_service
    
    if _enhanced_alerting_service:
        await _enhanced_alerting_service.stop()
        _enhanced_alerting_service = None