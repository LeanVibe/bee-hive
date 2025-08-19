"""
Comprehensive Monitoring and Analytics System
Production-grade monitoring, alerting, and analytics for customer projects.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from decimal import Decimal
import uuid
import hashlib
from collections import defaultdict, deque
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert, text
import redis.asyncio as redis
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import grafana_api
from elasticsearch import AsyncElasticsearch
import structlog

from app.core.database import get_async_session
from app.core.redis import get_redis_client


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataRetentionPeriod(Enum):
    """Data retention periods."""
    REAL_TIME = "real_time"  # 1 hour
    SHORT_TERM = "short_term"  # 7 days
    MEDIUM_TERM = "medium_term"  # 30 days
    LONG_TERM = "long_term"  # 1 year
    ARCHIVAL = "archival"  # 5 years


@dataclass
class MetricDefinition:
    """Definition of a metric to collect."""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    unit: str
    collection_interval_seconds: int
    retention_period: DataRetentionPeriod
    alert_thresholds: Dict[AlertSeverity, float] = field(default_factory=dict)
    aggregation_functions: List[str] = field(default_factory=lambda: ["avg", "min", "max", "sum"])


@dataclass
class MetricDataPoint:
    """Individual metric data point."""
    metric_id: str
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    tenant_id: str
    project_id: Optional[str] = None


@dataclass
class Alert:
    """System alert definition."""
    alert_id: str
    metric_id: str
    tenant_id: str
    project_id: Optional[str]
    severity: AlertSeverity
    threshold_value: float
    current_value: float
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledgment_required: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    tenant_id: str
    name: str
    description: str
    panels: List[Dict[str, Any]]
    refresh_interval_seconds: int
    access_permissions: Dict[str, List[str]]
    created_at: datetime
    last_updated: datetime


@dataclass
class AnalyticsReport:
    """Generated analytics report."""
    report_id: str
    tenant_id: str
    report_type: str
    title: str
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    generated_at: datetime
    period_start: datetime
    period_end: datetime


class MetricsCollector:
    """Collects metrics from various sources."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.prometheus_registry = CollectorRegistry()
        self.metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.elasticsearch_client: Optional[AsyncElasticsearch] = None
        
    async def initialize(self):
        """Initialize metrics collection system."""
        
        # Initialize Elasticsearch for metric storage
        try:
            self.elasticsearch_client = AsyncElasticsearch(
                ["http://elasticsearch:9200"],  # Configure based on deployment
                retry_on_timeout=True,
                max_retries=3
            )
            
            # Create indices for different retention periods
            await self._create_elasticsearch_indices()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Elasticsearch: {e}")
        
        self.logger.info("Metrics collector initialized successfully")
    
    async def collect_system_metrics(self, tenant_id: str, project_id: Optional[str] = None) -> List[MetricDataPoint]:
        """Collect system-level metrics."""
        
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Collect agent performance metrics
            agent_metrics = await self._collect_agent_metrics(tenant_id, project_id)
            for metric_name, value in agent_metrics.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"agent.{metric_name}",
                    timestamp=timestamp,
                    value=value,
                    labels={"tenant_id": tenant_id, "type": "agent"},
                    tenant_id=tenant_id,
                    project_id=project_id
                ))
            
            # Collect API performance metrics
            api_metrics = await self._collect_api_metrics(tenant_id, project_id)
            for metric_name, value in api_metrics.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"api.{metric_name}",
                    timestamp=timestamp,
                    value=value,
                    labels={"tenant_id": tenant_id, "type": "api"},
                    tenant_id=tenant_id,
                    project_id=project_id
                ))
            
            # Collect resource utilization metrics
            resource_metrics = await self._collect_resource_metrics(tenant_id, project_id)
            for metric_name, value in resource_metrics.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"resource.{metric_name}",
                    timestamp=timestamp,
                    value=value,
                    labels={"tenant_id": tenant_id, "type": "resource"},
                    tenant_id=tenant_id,
                    project_id=project_id
                ))
            
            # Collect business metrics
            business_metrics = await self._collect_business_metrics(tenant_id, project_id)
            for metric_name, value in business_metrics.items():
                metrics.append(MetricDataPoint(
                    metric_id=f"business.{metric_name}",
                    timestamp=timestamp,
                    value=value,
                    labels={"tenant_id": tenant_id, "type": "business"},
                    tenant_id=tenant_id,
                    project_id=project_id
                ))
            
            # Store metrics
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return []
    
    async def _collect_agent_metrics(self, tenant_id: str, project_id: Optional[str]) -> Dict[str, float]:
        """Collect agent-specific metrics."""
        
        metrics = {}
        
        try:
            # Get active agents for tenant
            active_agents_key = f"active_agents:{tenant_id}"
            if project_id:
                active_agents_key += f":{project_id}"
            
            active_agents = await self.redis.smembers(active_agents_key)
            metrics["agents.active_count"] = len(active_agents)
            
            # Collect per-agent metrics
            total_tasks_completed = 0
            total_tasks_failed = 0
            total_response_time = 0
            total_memory_usage = 0
            
            for agent_id in active_agents:
                agent_stats = await self.redis.hgetall(f"agent_stats:{agent_id}")
                
                if agent_stats:
                    total_tasks_completed += int(agent_stats.get("tasks_completed", 0))
                    total_tasks_failed += int(agent_stats.get("tasks_failed", 0))
                    total_response_time += float(agent_stats.get("avg_response_time", 0))
                    total_memory_usage += float(agent_stats.get("memory_usage_mb", 0))
            
            metrics["agents.tasks_completed_total"] = total_tasks_completed
            metrics["agents.tasks_failed_total"] = total_tasks_failed
            metrics["agents.avg_response_time_ms"] = total_response_time / max(len(active_agents), 1)
            metrics["agents.memory_usage_mb"] = total_memory_usage
            
            # Calculate success rate
            total_tasks = total_tasks_completed + total_tasks_failed
            metrics["agents.success_rate"] = (
                total_tasks_completed / max(total_tasks, 1)
            ) * 100
            
        except Exception as e:
            self.logger.warning(f"Failed to collect agent metrics: {e}")
        
        return metrics
    
    async def _collect_api_metrics(self, tenant_id: str, project_id: Optional[str]) -> Dict[str, float]:
        """Collect API performance metrics."""
        
        metrics = {}
        
        try:
            # Get API metrics from Redis
            api_key = f"api_metrics:{tenant_id}"
            if project_id:
                api_key += f":{project_id}"
            
            api_stats = await self.redis.hgetall(api_key)
            
            if api_stats:
                metrics["api.requests_per_minute"] = float(api_stats.get("requests_per_minute", 0))
                metrics["api.avg_response_time_ms"] = float(api_stats.get("avg_response_time_ms", 0))
                metrics["api.error_rate_percentage"] = float(api_stats.get("error_rate_percentage", 0))
                metrics["api.p95_response_time_ms"] = float(api_stats.get("p95_response_time_ms", 0))
                metrics["api.active_connections"] = float(api_stats.get("active_connections", 0))
            
        except Exception as e:
            self.logger.warning(f"Failed to collect API metrics: {e}")
        
        return metrics
    
    async def _collect_resource_metrics(self, tenant_id: str, project_id: Optional[str]) -> Dict[str, float]:
        """Collect resource utilization metrics."""
        
        metrics = {}
        
        try:
            # Get resource metrics from monitoring system
            resource_key = f"resource_metrics:{tenant_id}"
            if project_id:
                resource_key += f":{project_id}"
            
            resource_stats = await self.redis.hgetall(resource_key)
            
            if resource_stats:
                metrics["resource.cpu_usage_percentage"] = float(resource_stats.get("cpu_usage_percentage", 0))
                metrics["resource.memory_usage_percentage"] = float(resource_stats.get("memory_usage_percentage", 0))
                metrics["resource.disk_usage_percentage"] = float(resource_stats.get("disk_usage_percentage", 0))
                metrics["resource.network_io_mbps"] = float(resource_stats.get("network_io_mbps", 0))
                metrics["resource.database_connections"] = float(resource_stats.get("database_connections", 0))
                metrics["resource.redis_connections"] = float(resource_stats.get("redis_connections", 0))
            
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")
        
        return metrics
    
    async def _collect_business_metrics(self, tenant_id: str, project_id: Optional[str]) -> Dict[str, float]:
        """Collect business-level metrics."""
        
        metrics = {}
        
        try:
            # Get business metrics
            business_key = f"business_metrics:{tenant_id}"
            if project_id:
                business_key += f":{project_id}"
            
            business_stats = await self.redis.hgetall(business_key)
            
            if business_stats:
                metrics["business.features_delivered"] = float(business_stats.get("features_delivered", 0))
                metrics["business.velocity_story_points"] = float(business_stats.get("velocity_story_points", 0))
                metrics["business.deployment_frequency"] = float(business_stats.get("deployment_frequency", 0))
                metrics["business.lead_time_hours"] = float(business_stats.get("lead_time_hours", 0))
                metrics["business.mttr_hours"] = float(business_stats.get("mttr_hours", 0))
                metrics["business.change_failure_rate"] = float(business_stats.get("change_failure_rate", 0))
            
        except Exception as e:
            self.logger.warning(f"Failed to collect business metrics: {e}")
        
        return metrics
    
    async def _store_metrics(self, metrics: List[MetricDataPoint]):
        """Store metrics in multiple storage systems."""
        
        # Store in Redis for real-time access
        for metric in metrics:
            key = f"metric:{metric.tenant_id}:{metric.metric_id}"
            await self.redis.lpush(key, json.dumps({
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.value,
                "labels": metric.labels
            }))
            await self.redis.ltrim(key, 0, 999)  # Keep last 1000 points
            
            # Cache in memory for fast access
            self.metrics_cache[key].append(metric)
        
        # Store in Elasticsearch for long-term analysis
        if self.elasticsearch_client:
            try:
                es_docs = []
                for metric in metrics:
                    doc = {
                        "metric_id": metric.metric_id,
                        "timestamp": metric.timestamp,
                        "value": metric.value,
                        "labels": metric.labels,
                        "tenant_id": metric.tenant_id,
                        "project_id": metric.project_id
                    }
                    es_docs.append({
                        "_index": f"metrics-{datetime.now().strftime('%Y-%m')}",
                        "_source": doc
                    })
                
                if es_docs:
                    from elasticsearch.helpers import async_bulk
                    await async_bulk(self.elasticsearch_client, es_docs)
                    
            except Exception as e:
                self.logger.warning(f"Failed to store metrics in Elasticsearch: {e}")


class AlertingEngine:
    """Engine for monitoring metrics and generating alerts."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, Callable] = {}
    
    async def initialize(self):
        """Initialize the alerting engine."""
        
        # Load alert rules from configuration
        await self._load_alert_rules()
        
        # Initialize notification channels
        await self._initialize_notification_channels()
        
        # Start alert monitoring loop
        asyncio.create_task(self._alert_monitoring_loop())
        
        self.logger.info("Alerting engine initialized successfully")
    
    async def evaluate_alert_rules(
        self, 
        metrics: List[MetricDataPoint]
    ) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        
        triggered_alerts = []
        
        for metric in metrics:
            # Check if there are alert rules for this metric
            rules = self.alert_rules.get(metric.metric_id, [])
            
            for rule in rules:
                alert = await self._evaluate_single_rule(metric, rule)
                if alert:
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    async def _evaluate_single_rule(
        self, 
        metric: MetricDataPoint, 
        rule: Dict[str, Any]
    ) -> Optional[Alert]:
        """Evaluate a single alert rule."""
        
        try:
            # Check if metric matches rule conditions
            if not self._matches_rule_conditions(metric, rule):
                return None
            
            # Evaluate threshold
            threshold = rule["threshold"]
            operator = rule.get("operator", "gt")  # greater than
            
            triggered = False
            if operator == "gt" and metric.value > threshold:
                triggered = True
            elif operator == "lt" and metric.value < threshold:
                triggered = True
            elif operator == "eq" and metric.value == threshold:
                triggered = True
            elif operator == "ne" and metric.value != threshold:
                triggered = True
            
            if not triggered:
                return None
                
            # Check if alert is already active
            alert_key = f"{metric.tenant_id}:{metric.metric_id}:{rule['rule_id']}"
            if alert_key in self.active_alerts:
                # Update existing alert
                existing_alert = self.active_alerts[alert_key]
                existing_alert.current_value = metric.value
                return None  # Don't create duplicate
            
            # Create new alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                metric_id=metric.metric_id,
                tenant_id=metric.tenant_id,
                project_id=metric.project_id,
                severity=AlertSeverity(rule["severity"]),
                threshold_value=threshold,
                current_value=metric.value,
                message=rule["message"].format(
                    metric_name=metric.metric_id,
                    current_value=metric.value,
                    threshold=threshold
                ),
                triggered_at=datetime.now(),
                acknowledgment_required=rule.get("acknowledgment_required", False),
                escalation_rules=rule.get("escalation_rules", [])
            )
            
            # Store active alert
            self.active_alerts[alert_key] = alert
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate alert rule: {e}")
            return None
    
    async def _load_alert_rules(self):
        """Load alert rules from configuration."""
        
        # Default alert rules for common metrics
        default_rules = {
            "agents.success_rate": [
                {
                    "rule_id": "agent_success_rate_low",
                    "threshold": 95.0,
                    "operator": "lt",
                    "severity": "warning",
                    "message": "Agent success rate ({current_value}%) is below threshold ({threshold}%)",
                    "acknowledgment_required": False
                },
                {
                    "rule_id": "agent_success_rate_critical",
                    "threshold": 85.0,
                    "operator": "lt",
                    "severity": "critical",
                    "message": "Agent success rate ({current_value}%) is critically low",
                    "acknowledgment_required": True,
                    "escalation_rules": [
                        {"escalate_after_minutes": 15, "notify": ["management"]},
                        {"escalate_after_minutes": 30, "notify": ["executive"]}
                    ]
                }
            ],
            "api.error_rate_percentage": [
                {
                    "rule_id": "api_error_rate_high",
                    "threshold": 5.0,
                    "operator": "gt",
                    "severity": "error",
                    "message": "API error rate ({current_value}%) is above threshold ({threshold}%)",
                    "acknowledgment_required": True
                }
            ],
            "resource.cpu_usage_percentage": [
                {
                    "rule_id": "cpu_usage_high",
                    "threshold": 80.0,
                    "operator": "gt",
                    "severity": "warning",
                    "message": "CPU usage ({current_value}%) is high",
                    "acknowledgment_required": False
                },
                {
                    "rule_id": "cpu_usage_critical",
                    "threshold": 95.0,
                    "operator": "gt",
                    "severity": "critical",
                    "message": "CPU usage ({current_value}%) is critical",
                    "acknowledgment_required": True
                }
            ],
            "resource.memory_usage_percentage": [
                {
                    "rule_id": "memory_usage_high",
                    "threshold": 85.0,
                    "operator": "gt",
                    "severity": "warning",
                    "message": "Memory usage ({current_value}%) is high",
                    "acknowledgment_required": False
                }
            ]
        }
        
        self.alert_rules = default_rules
        
        # Load custom rules from Redis
        try:
            custom_rules = await self.redis.hgetall("alert_rules")
            for rule_key, rule_data in custom_rules.items():
                rule = json.loads(rule_data)
                metric_id = rule["metric_id"]
                if metric_id not in self.alert_rules:
                    self.alert_rules[metric_id] = []
                self.alert_rules[metric_id].append(rule)
                
        except Exception as e:
            self.logger.warning(f"Failed to load custom alert rules: {e}")
    
    async def _alert_monitoring_loop(self):
        """Continuous loop for monitoring alerts."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for alert escalations
                await self._check_alert_escalations()
                
                # Check for alert auto-resolution
                await self._check_alert_auto_resolution()
                
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(60)


class AnalyticsEngine:
    """Engine for generating analytics and insights."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.ml_models: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the analytics engine."""
        
        # Initialize ML models for predictive analytics
        await self._initialize_ml_models()
        
        self.logger.info("Analytics engine initialized successfully")
    
    async def generate_performance_report(
        self,
        tenant_id: str,
        project_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> AnalyticsReport:
        """Generate comprehensive performance report."""
        
        try:
            # Collect metrics for the period
            metrics_data = await self._collect_metrics_for_period(
                tenant_id, project_id, start_date, end_date
            )
            
            # Analyze performance trends
            performance_analysis = await self._analyze_performance_trends(metrics_data)
            
            # Generate insights
            insights = await self._generate_performance_insights(performance_analysis)
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                performance_analysis, insights
            )
            
            report = AnalyticsReport(
                report_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                report_type="performance",
                title=f"Performance Report - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                data={
                    "metrics_summary": performance_analysis["summary"],
                    "trends": performance_analysis["trends"],
                    "anomalies": performance_analysis["anomalies"],
                    "benchmarks": performance_analysis["benchmarks"]
                },
                insights=insights,
                recommendations=recommendations,
                generated_at=datetime.now(),
                period_start=start_date,
                period_end=end_date
            )
            
            # Store report
            await self.redis.setex(
                f"analytics_report:{report.report_id}",
                86400 * 30,  # 30 days TTL
                json.dumps(report.__dict__, default=str)
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            raise
    
    async def _analyze_performance_trends(self, metrics_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze performance trends in the metrics data."""
        
        analysis = {
            "summary": {},
            "trends": {},
            "anomalies": [],
            "benchmarks": {}
        }
        
        try:
            for metric_id, data_points in metrics_data.items():
                if not data_points:
                    continue
                
                values = [point["value"] for point in data_points]
                timestamps = [datetime.fromisoformat(point["timestamp"]) for point in data_points]
                
                # Calculate summary statistics
                analysis["summary"][metric_id] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
                }
                
                # Analyze trends using linear regression
                if len(values) >= 3:
                    X = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps]).reshape(-1, 1)
                    y = np.array(values)
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    slope = model.coef_[0]
                    r_squared = model.score(X, y)
                    
                    trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                    trend_strength = "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.3 else "weak"
                    
                    analysis["trends"][metric_id] = {
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "slope": slope,
                        "r_squared": r_squared
                    }
                
                # Detect anomalies using statistical methods
                if len(values) >= 10:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    threshold = 2.5 * std_val
                    
                    for i, (value, timestamp) in enumerate(zip(values, timestamps)):
                        if abs(value - mean_val) > threshold:
                            analysis["anomalies"].append({
                                "metric_id": metric_id,
                                "timestamp": timestamp.isoformat(),
                                "value": value,
                                "expected_range": [mean_val - threshold, mean_val + threshold],
                                "severity": "high" if abs(value - mean_val) > 3 * std_val else "medium"
                            })
                
                # Benchmark against historical data
                analysis["benchmarks"][metric_id] = await self._get_historical_benchmark(metric_id, values)
                
        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
        
        return analysis
    
    async def _generate_performance_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from performance analysis."""
        
        insights = []
        
        try:
            # Analyze overall system health
            summary = analysis["summary"]
            trends = analysis["trends"]
            anomalies = analysis["anomalies"]
            
            # Success rate insights
            if "agents.success_rate" in summary:
                success_rate = summary["agents.success_rate"]["mean"]
                if success_rate >= 98:
                    insights.append("🟢 Agent success rate is excellent, indicating highly reliable autonomous operations")
                elif success_rate >= 95:
                    insights.append("🟡 Agent success rate is good but has room for improvement")
                else:
                    insights.append("🔴 Agent success rate is below expectations and requires immediate attention")
            
            # Performance trend insights
            performance_metrics = ["api.avg_response_time_ms", "agents.avg_response_time_ms"]
            for metric in performance_metrics:
                if metric in trends:
                    trend = trends[metric]
                    if trend["direction"] == "increasing" and trend["strength"] in ["strong", "moderate"]:
                        insights.append(f"⚠️ {metric} shows {trend['strength']} upward trend - performance degradation detected")
                    elif trend["direction"] == "decreasing" and trend["strength"] in ["strong", "moderate"]:
                        insights.append(f"✅ {metric} shows {trend['strength']} improvement trend")
            
            # Resource utilization insights
            resource_metrics = ["resource.cpu_usage_percentage", "resource.memory_usage_percentage"]
            for metric in resource_metrics:
                if metric in summary:
                    avg_usage = summary[metric]["mean"]
                    max_usage = summary[metric]["max"]
                    
                    if avg_usage > 80:
                        insights.append(f"🔴 High average {metric.split('.')[-1]}: {avg_usage:.1f}% - consider scaling up")
                    elif max_usage > 95:
                        insights.append(f"⚠️ Peak {metric.split('.')[-1]} reached {max_usage:.1f}% - monitor for capacity issues")
            
            # Anomaly insights
            high_severity_anomalies = [a for a in anomalies if a["severity"] == "high"]
            if high_severity_anomalies:
                insights.append(f"🔍 Detected {len(high_severity_anomalies)} high-severity anomalies requiring investigation")
            
            # Business metric insights
            if "business.velocity_story_points" in summary:
                velocity = summary["business.velocity_story_points"]["mean"]
                if "business.velocity_story_points" in trends:
                    velocity_trend = trends["business.velocity_story_points"]
                    if velocity_trend["direction"] == "increasing":
                        insights.append(f"📈 Development velocity is improving: {velocity:.1f} story points average")
                    elif velocity_trend["direction"] == "decreasing":
                        insights.append(f"📉 Development velocity is declining: {velocity:.1f} story points average")
            
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {e}")
        
        return insights
    
    async def _generate_performance_recommendations(
        self, 
        analysis: Dict[str, Any], 
        insights: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        try:
            summary = analysis["summary"]
            trends = analysis["trends"]
            
            # Agent performance recommendations
            if "agents.success_rate" in summary:
                success_rate = summary["agents.success_rate"]["mean"]
                if success_rate < 95:
                    recommendations.append("Investigate failed agent tasks and improve error handling mechanisms")
                if success_rate < 90:
                    recommendations.append("Consider implementing circuit breakers and retry mechanisms for critical agent operations")
            
            # API performance recommendations
            if "api.avg_response_time_ms" in summary:
                avg_response_time = summary["api.avg_response_time_ms"]["mean"]
                if avg_response_time > 500:
                    recommendations.append("Optimize API response times by implementing caching and database query optimization")
                if "api.avg_response_time_ms" in trends and trends["api.avg_response_time_ms"]["direction"] == "increasing":
                    recommendations.append("Monitor API performance degradation trend and implement performance testing")
            
            # Resource optimization recommendations
            resource_recommendations = []
            
            if "resource.cpu_usage_percentage" in summary:
                cpu_usage = summary["resource.cpu_usage_percentage"]["mean"]
                if cpu_usage > 80:
                    resource_recommendations.append("Scale up CPU resources or optimize CPU-intensive operations")
                elif cpu_usage < 30:
                    resource_recommendations.append("Consider scaling down CPU resources to optimize costs")
            
            if "resource.memory_usage_percentage" in summary:
                memory_usage = summary["resource.memory_usage_percentage"]["mean"]
                if memory_usage > 85:
                    resource_recommendations.append("Increase memory allocation or investigate memory leaks")
                elif memory_usage < 40:
                    resource_recommendations.append("Consider reducing memory allocation to optimize costs")
            
            if resource_recommendations:
                recommendations.extend(resource_recommendations)
            
            # Business process recommendations
            if "business.lead_time_hours" in summary:
                lead_time = summary["business.lead_time_hours"]["mean"]
                if lead_time > 48:
                    recommendations.append("Reduce lead time by streamlining development processes and removing bottlenecks")
            
            if "business.deployment_frequency" in summary:
                deploy_freq = summary["business.deployment_frequency"]["mean"]
                if deploy_freq < 1:  # Less than once per day
                    recommendations.append("Increase deployment frequency by implementing continuous deployment practices")
            
            # Alert and monitoring recommendations
            anomaly_count = len(analysis.get("anomalies", []))
            if anomaly_count > 10:
                recommendations.append("High number of anomalies detected - review alert thresholds and implement better baseline monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations


class ComprehensiveMonitoringService:
    """Main service for comprehensive monitoring and analytics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.alerting_engine: Optional[AlertingEngine] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None
        self.dashboards: Dict[str, Dashboard] = {}
    
    async def initialize(self):
        """Initialize the comprehensive monitoring service."""
        self.redis_client = await get_redis_client()
        self.metrics_collector = MetricsCollector(self.redis_client, self.logger)
        self.alerting_engine = AlertingEngine(self.redis_client, self.logger)
        self.analytics_engine = AnalyticsEngine(self.redis_client, self.logger)
        
        await self.metrics_collector.initialize()
        await self.alerting_engine.initialize()
        await self.analytics_engine.initialize()
        
        # Start monitoring loops
        asyncio.create_task(self._continuous_monitoring_loop())
        
        self.logger.info("Comprehensive monitoring service initialized successfully")
    
    async def start_monitoring(
        self, 
        tenant_id: str, 
        project_id: Optional[str] = None,
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start comprehensive monitoring for a tenant/project."""
        
        try:
            # Create default dashboard
            dashboard = await self._create_default_dashboard(tenant_id, project_id)
            
            # Setup custom alert rules if provided
            if monitoring_config and "alert_rules" in monitoring_config:
                await self._setup_custom_alert_rules(tenant_id, monitoring_config["alert_rules"])
            
            # Initialize metrics collection
            initial_metrics = await self.metrics_collector.collect_system_metrics(tenant_id, project_id)
            
            return {
                "status": "success",
                "tenant_id": tenant_id,
                "project_id": project_id,
                "dashboard_id": dashboard.dashboard_id,
                "monitoring_started": datetime.now().isoformat(),
                "initial_metrics_count": len(initial_metrics),
                "dashboard_url": f"/dashboard/{dashboard.dashboard_id}",
                "alerts_configured": len(self.alerting_engine.alert_rules),
                "next_steps": [
                    "Access real-time dashboard at provided URL",
                    "Configure notification preferences",
                    "Review and customize alert thresholds",
                    "Schedule regular analytics reports"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return {
                "status": "error",
                "tenant_id": tenant_id,
                "error_message": str(e)
            }
    
    async def get_real_time_metrics(
        self, 
        tenant_id: str, 
        project_id: Optional[str] = None,
        metric_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get real-time metrics for a tenant/project."""
        
        try:
            # Collect current metrics
            current_metrics = await self.metrics_collector.collect_system_metrics(tenant_id, project_id)
            
            # Filter by requested metric IDs if specified
            if metric_ids:
                current_metrics = [m for m in current_metrics if m.metric_id in metric_ids]
            
            # Format for API response
            metrics_data = {}
            for metric in current_metrics:
                if metric.metric_id not in metrics_data:
                    metrics_data[metric.metric_id] = []
                
                metrics_data[metric.metric_id].append({
                    "timestamp": metric.timestamp.isoformat(),
                    "value": metric.value,
                    "labels": metric.labels
                })
            
            return {
                "status": "success",
                "tenant_id": tenant_id,
                "project_id": project_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics_data,
                "metrics_count": len(current_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time metrics: {e}")
            return {
                "status": "error",
                "tenant_id": tenant_id,
                "error_message": str(e)
            }
    
    async def generate_analytics_report(
        self,
        tenant_id: str,
        report_type: str,
        project_id: Optional[str] = None,
        period_days: int = 7
    ) -> Dict[str, Any]:
        """Generate analytics report for a tenant/project."""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            if report_type == "performance":
                report = await self.analytics_engine.generate_performance_report(
                    tenant_id, project_id, start_date, end_date
                )
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            return {
                "status": "success",
                "report": report.__dict__,
                "download_url": f"/reports/{report.report_id}/download"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate analytics report: {e}")
            return {
                "status": "error",
                "tenant_id": tenant_id,
                "error_message": str(e)
            }
    
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for all tenants."""
        
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Get all active tenants
                active_tenants = await self.redis_client.smembers("active_tenants")
                
                for tenant_id in active_tenants:
                    try:
                        # Collect metrics
                        metrics = await self.metrics_collector.collect_system_metrics(tenant_id)
                        
                        # Evaluate alerts
                        alerts = await self.alerting_engine.evaluate_alert_rules(metrics)
                        
                        # Process triggered alerts
                        for alert in alerts:
                            await self._process_alert(alert)
                            
                    except Exception as e:
                        self.logger.error(f"Error monitoring tenant {tenant_id}: {e}")
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)


# Global service instance
_monitoring_service: Optional[ComprehensiveMonitoringService] = None


async def get_monitoring_service() -> ComprehensiveMonitoringService:
    """Get the global monitoring service instance."""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = ComprehensiveMonitoringService()
        await _monitoring_service.initialize()
    
    return _monitoring_service


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class MonitoringServiceTest(BaseScript):
        """Test the comprehensive monitoring service."""
        
        async def execute(self):
            """Execute monitoring service test."""
            service = await get_monitoring_service()
            
            # Start monitoring for a tenant
            monitoring_result = await service.start_monitoring(
                "tenant_techcorp",
                "project_mvp_001",
                {
                    "alert_rules": [
                        {
                            "metric_id": "custom.deployment_success_rate",
                            "threshold": 95.0,
                            "operator": "lt",
                            "severity": "warning",
                            "message": "Deployment success rate is below 95%"
                        }
                    ]
                }
            )
            self.logger.info("Monitoring start result", result=monitoring_result)
            
            # Get real-time metrics
            metrics_result = await service.get_real_time_metrics("tenant_techcorp", "project_mvp_001")
            self.logger.info("Real-time metrics", metrics=metrics_result)
            
            # Generate analytics report
            report_result = await service.generate_analytics_report(
                "tenant_techcorp", "performance", "project_mvp_001", 7
            )
            self.logger.info("Analytics report", report=report_result)
            
            return {
                "monitoring_status": monitoring_result.get("status"),
                "metrics_count": len(metrics_result.get("metrics", [])),
                "report_generated": bool(report_result.get("report_id"))
            }
    
    script_main(MonitoringServiceTest)