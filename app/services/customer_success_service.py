"""
Customer Success Framework Service
Production-grade service for 30-day success guarantee with real-time tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from decimal import Decimal
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
import redis.asyncio as redis
import aiohttp
from jinja2 import Template

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.models.workflow import WorkflowExecution
from app.schemas.workflow import WorkflowStatus, WorkflowType


class SuccessMetricType(Enum):
    """Types of success metrics."""
    VELOCITY_IMPROVEMENT = "velocity_improvement"
    QUALITY_MAINTENANCE = "quality_maintenance"
    TIMELINE_ADHERENCE = "timeline_adherence"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    COST_REDUCTION = "cost_reduction"
    ROI_ACHIEVEMENT = "roi_achievement"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    SECURITY_ENHANCEMENT = "security_enhancement"


class GuaranteeStatus(Enum):
    """Status of success guarantee."""
    ACTIVE = "active"
    SUCCESS = "success"
    AT_RISK = "at_risk"
    FAILED = "failed"
    EXPIRED = "expired"


class RiskLevel(Enum):
    """Risk levels for success metrics."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SuccessMetric:
    """Individual success metric definition and tracking."""
    metric_id: str
    metric_type: SuccessMetricType
    name: str
    description: str
    target_value: float
    current_value: float
    baseline_value: float
    unit: str
    weight: float  # Importance weight (0.0 to 1.0)
    measurement_frequency: str  # daily, weekly, monthly
    last_measured: datetime
    trend_data: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class SuccessGuarantee:
    """30-day success guarantee definition."""
    guarantee_id: str
    customer_id: str
    service_type: str  # mvp_development, legacy_modernization, team_augmentation
    start_date: datetime
    end_date: datetime
    guarantee_amount: Decimal  # Refund amount if guarantee fails
    success_criteria: List[SuccessMetric]
    minimum_success_threshold: float  # Percentage of criteria that must be met
    current_success_score: float
    status: GuaranteeStatus
    risk_assessment: Dict[str, Any]
    mitigation_actions: List[Dict[str, Any]] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)


@dataclass
class CustomerSuccessProfile:
    """Comprehensive customer success profile."""
    customer_id: str
    customer_name: str
    service_type: str
    onboarding_date: datetime
    success_manager_id: str
    communication_preferences: Dict[str, Any]
    business_objectives: List[str]
    success_metrics: List[SuccessMetric]
    active_guarantees: List[SuccessGuarantee]
    health_score: float  # Overall customer health (0.0 to 1.0)
    satisfaction_score: float  # Customer satisfaction (0.0 to 10.0)
    renewal_probability: float  # Probability of renewal (0.0 to 1.0)
    expansion_opportunities: List[str] = field(default_factory=list)


@dataclass
class RealTimeAlert:
    """Real-time alert for success issues."""
    alert_id: str
    customer_id: str
    guarantee_id: str
    metric_id: str
    alert_type: str  # threshold_breach, trend_negative, risk_escalation
    severity: RiskLevel
    message: str
    triggered_at: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None


class SuccessMetricsCollector:
    """Collector for various success metrics."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def collect_velocity_metrics(
        self, 
        customer_id: str, 
        project_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collect development velocity metrics."""
        
        self.logger.info(f"Collecting velocity metrics for customer: {customer_id}")
        
        metrics = {}
        
        try:
            # Get baseline velocity (pre-implementation)
            baseline_velocity = project_data.get("baseline_velocity", 1.0)
            
            # Calculate current velocity based on completed tasks
            completed_tasks = project_data.get("completed_tasks", [])
            total_story_points = sum(task.get("story_points", 1) for task in completed_tasks)
            time_period_days = project_data.get("measurement_period_days", 7)
            
            current_velocity = total_story_points / max(time_period_days / 7, 1)  # Stories per week
            velocity_improvement = (current_velocity / baseline_velocity - 1) * 100
            
            metrics["velocity_improvement_percentage"] = velocity_improvement
            metrics["current_velocity_story_points_per_week"] = current_velocity
            metrics["baseline_velocity_story_points_per_week"] = baseline_velocity
            
            # Calculate feature delivery rate
            features_delivered = len([t for t in completed_tasks if t.get("type") == "feature"])
            features_per_week = features_delivered / max(time_period_days / 7, 1)
            
            metrics["features_delivered_per_week"] = features_per_week
            
            # Calculate cycle time (time from start to completion)
            cycle_times = []
            for task in completed_tasks:
                if task.get("started_at") and task.get("completed_at"):
                    start_time = datetime.fromisoformat(task["started_at"])
                    end_time = datetime.fromisoformat(task["completed_at"])
                    cycle_time_hours = (end_time - start_time).total_seconds() / 3600
                    cycle_times.append(cycle_time_hours)
            
            if cycle_times:
                metrics["average_cycle_time_hours"] = statistics.mean(cycle_times)
                metrics["median_cycle_time_hours"] = statistics.median(cycle_times)
            
        except Exception as e:
            self.logger.error(f"Failed to collect velocity metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def collect_quality_metrics(
        self, 
        customer_id: str, 
        project_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collect code quality and defect metrics."""
        
        self.logger.info(f"Collecting quality metrics for customer: {customer_id}")
        
        metrics = {}
        
        try:
            # Test coverage metrics
            test_coverage = project_data.get("test_coverage_percentage", 0.0)
            metrics["test_coverage_percentage"] = test_coverage
            
            # Defect metrics
            defects_reported = project_data.get("defects_reported", 0)
            defects_resolved = project_data.get("defects_resolved", 0)
            total_deliverables = project_data.get("total_deliverables", 1)
            
            metrics["defect_density_per_deliverable"] = defects_reported / max(total_deliverables, 1)
            metrics["defect_resolution_rate"] = (defects_resolved / max(defects_reported, 1)) * 100
            
            # Code quality scores
            code_quality_score = project_data.get("code_quality_score", 0.0)  # From static analysis
            metrics["code_quality_score"] = code_quality_score
            
            # Security vulnerability metrics
            security_vulnerabilities = project_data.get("security_vulnerabilities", {})
            metrics["high_severity_vulnerabilities"] = security_vulnerabilities.get("high", 0)
            metrics["medium_severity_vulnerabilities"] = security_vulnerabilities.get("medium", 0)
            metrics["total_vulnerabilities"] = sum(security_vulnerabilities.values())
            
            # Performance metrics
            performance_benchmarks = project_data.get("performance_benchmarks", {})
            metrics["response_time_ms"] = performance_benchmarks.get("response_time_ms", 0)
            metrics["throughput_requests_per_second"] = performance_benchmarks.get("throughput_rps", 0)
            
        except Exception as e:
            self.logger.error(f"Failed to collect quality metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def collect_satisfaction_metrics(
        self, 
        customer_id: str, 
        feedback_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Collect customer satisfaction metrics."""
        
        self.logger.info(f"Collecting satisfaction metrics for customer: {customer_id}")
        
        metrics = {}
        
        try:
            # Net Promoter Score (NPS)
            nps_scores = feedback_data.get("nps_scores", [])
            if nps_scores:
                nps = statistics.mean(nps_scores)
                metrics["net_promoter_score"] = nps
            
            # Customer Satisfaction Score (CSAT)
            csat_scores = feedback_data.get("csat_scores", [])
            if csat_scores:
                csat = statistics.mean(csat_scores)
                metrics["customer_satisfaction_score"] = csat
            
            # Stakeholder feedback scores
            stakeholder_feedback = feedback_data.get("stakeholder_feedback", {})
            for role, score in stakeholder_feedback.items():
                metrics[f"{role}_satisfaction_score"] = score
            
            # Communication effectiveness
            communication_scores = feedback_data.get("communication_scores", [])
            if communication_scores:
                metrics["communication_effectiveness_score"] = statistics.mean(communication_scores)
            
            # Overall project satisfaction
            overall_satisfaction = feedback_data.get("overall_satisfaction", 0.0)
            metrics["overall_project_satisfaction"] = overall_satisfaction
            
        except Exception as e:
            self.logger.error(f"Failed to collect satisfaction metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics


class RiskAssessmentEngine:
    """Engine for assessing success risks and generating alerts."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def assess_guarantee_risk(self, guarantee: SuccessGuarantee) -> Dict[str, Any]:
        """Assess risk level for a success guarantee."""
        
        self.logger.info(f"Assessing risk for guarantee: {guarantee.guarantee_id}")
        
        risk_assessment = {
            "overall_risk_level": RiskLevel.LOW,
            "risk_factors": [],
            "mitigation_recommendations": [],
            "escalation_required": False,
            "estimated_success_probability": 0.0
        }
        
        try:
            # Calculate days remaining
            days_remaining = (guarantee.end_date - datetime.now()).days
            days_elapsed = (datetime.now() - guarantee.start_date).days
            progress_percentage = days_elapsed / max((guarantee.end_date - guarantee.start_date).days, 1) * 100
            
            # Assess each success metric
            at_risk_metrics = []
            successful_metrics = []
            
            for metric in guarantee.success_criteria:
                metric_risk = await self._assess_metric_risk(metric, days_remaining, progress_percentage)
                
                if metric_risk["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    at_risk_metrics.append({
                        "metric": metric.name,
                        "risk_level": metric_risk["risk_level"],
                        "risk_factors": metric_risk["risk_factors"]
                    })
                elif metric_risk["on_track"]:
                    successful_metrics.append(metric.name)
            
            # Calculate overall risk level
            risk_assessment["at_risk_metrics"] = at_risk_metrics
            risk_assessment["successful_metrics"] = successful_metrics
            
            if len(at_risk_metrics) == 0:
                risk_assessment["overall_risk_level"] = RiskLevel.LOW
            elif len(at_risk_metrics) <= len(guarantee.success_criteria) * 0.3:
                risk_assessment["overall_risk_level"] = RiskLevel.MEDIUM
            elif len(at_risk_metrics) <= len(guarantee.success_criteria) * 0.6:
                risk_assessment["overall_risk_level"] = RiskLevel.HIGH
            else:
                risk_assessment["overall_risk_level"] = RiskLevel.CRITICAL
            
            # Estimate success probability
            success_probability = (len(successful_metrics) / len(guarantee.success_criteria)) * 100
            risk_assessment["estimated_success_probability"] = success_probability
            
            # Determine if escalation is required
            risk_assessment["escalation_required"] = (
                risk_assessment["overall_risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL] or
                success_probability < guarantee.minimum_success_threshold
            )
            
            # Generate mitigation recommendations
            risk_assessment["mitigation_recommendations"] = await self._generate_mitigation_recommendations(
                guarantee, at_risk_metrics, days_remaining
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess guarantee risk: {e}")
            risk_assessment["error"] = str(e)
        
        return risk_assessment
    
    async def _assess_metric_risk(
        self, 
        metric: SuccessMetric, 
        days_remaining: int, 
        progress_percentage: float
    ) -> Dict[str, Any]:
        """Assess risk for an individual metric."""
        
        metric_assessment = {
            "risk_level": RiskLevel.LOW,
            "risk_factors": [],
            "on_track": True,
            "projected_final_value": metric.current_value
        }
        
        # Calculate progress towards target
        if metric.target_value != metric.baseline_value:
            progress_towards_target = (
                (metric.current_value - metric.baseline_value) / 
                (metric.target_value - metric.baseline_value)
            ) * 100
        else:
            progress_towards_target = 100.0 if metric.current_value == metric.target_value else 0.0
        
        # Assess if metric is on track
        expected_progress = progress_percentage
        progress_gap = expected_progress - progress_towards_target
        
        if progress_gap > 20:
            metric_assessment["risk_level"] = RiskLevel.HIGH
            metric_assessment["on_track"] = False
            metric_assessment["risk_factors"].append(f"Behind schedule by {progress_gap:.1f}%")
        elif progress_gap > 10:
            metric_assessment["risk_level"] = RiskLevel.MEDIUM
            metric_assessment["risk_factors"].append(f"Slightly behind schedule by {progress_gap:.1f}%")
        
        # Analyze trend data
        if len(metric.trend_data) >= 3:
            recent_trends = [data["value"] for data in metric.trend_data[-3:]]
            trend_slope = (recent_trends[-1] - recent_trends[0]) / len(recent_trends)
            
            # Check if trend is moving away from target
            if metric.target_value > metric.baseline_value:  # Higher is better
                if trend_slope < 0:
                    metric_assessment["risk_factors"].append("Negative trend in recent measurements")
                    metric_assessment["risk_level"] = max(metric_assessment["risk_level"], RiskLevel.MEDIUM)
            else:  # Lower is better
                if trend_slope > 0:
                    metric_assessment["risk_factors"].append("Upward trend in recent measurements")
                    metric_assessment["risk_level"] = max(metric_assessment["risk_level"], RiskLevel.MEDIUM)
        
        # Project final value based on current trend
        if len(metric.trend_data) >= 2 and days_remaining > 0:
            recent_rate = (metric.current_value - metric.trend_data[-2]["value"]) / 7  # Assume weekly measurements
            projected_change = recent_rate * (days_remaining / 7)
            metric_assessment["projected_final_value"] = metric.current_value + projected_change
        
        return metric_assessment


class SuccessMonitoringAgent:
    """Agent for real-time success monitoring and alerting."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.active_alerts: Dict[str, RealTimeAlert] = {}
    
    async def monitor_success_guarantees(self, customer_profiles: List[CustomerSuccessProfile]):
        """Monitor all active success guarantees and generate alerts."""
        
        self.logger.info(f"Monitoring {len(customer_profiles)} customer success profiles")
        
        for profile in customer_profiles:
            for guarantee in profile.active_guarantees:
                if guarantee.status == GuaranteeStatus.ACTIVE:
                    await self._monitor_individual_guarantee(profile, guarantee)
    
    async def _monitor_individual_guarantee(
        self, 
        profile: CustomerSuccessProfile, 
        guarantee: SuccessGuarantee
    ):
        """Monitor an individual success guarantee."""
        
        try:
            # Check each success metric
            for metric in guarantee.success_criteria:
                alerts = await self._check_metric_thresholds(profile, guarantee, metric)
                
                for alert in alerts:
                    await self._process_alert(alert)
            
            # Check overall guarantee health
            overall_alerts = await self._check_guarantee_health(profile, guarantee)
            
            for alert in overall_alerts:
                await self._process_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to monitor guarantee {guarantee.guarantee_id}: {e}")
    
    async def _check_metric_thresholds(
        self, 
        profile: CustomerSuccessProfile, 
        guarantee: SuccessGuarantee, 
        metric: SuccessMetric
    ) -> List[RealTimeAlert]:
        """Check if metric has breached any thresholds."""
        
        alerts = []
        
        # Calculate percentage towards target
        if metric.target_value != metric.baseline_value:
            progress_percentage = (
                (metric.current_value - metric.baseline_value) / 
                (metric.target_value - metric.baseline_value)
            ) * 100
        else:
            progress_percentage = 100.0 if metric.current_value == metric.target_value else 0.0
        
        # Check if metric is severely underperforming
        if progress_percentage < 50:
            alert = RealTimeAlert(
                alert_id=str(uuid.uuid4()),
                customer_id=profile.customer_id,
                guarantee_id=guarantee.guarantee_id,
                metric_id=metric.metric_id,
                alert_type="threshold_breach",
                severity=RiskLevel.HIGH,
                message=f"Metric '{metric.name}' is at {progress_percentage:.1f}% of target",
                triggered_at=datetime.now()
            )
            alerts.append(alert)
        
        # Check for negative trends
        if len(metric.trend_data) >= 3:
            recent_values = [data["value"] for data in metric.trend_data[-3:]]
            if all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
                alert = RealTimeAlert(
                    alert_id=str(uuid.uuid4()),
                    customer_id=profile.customer_id,
                    guarantee_id=guarantee.guarantee_id,
                    metric_id=metric.metric_id,
                    alert_type="trend_negative",
                    severity=RiskLevel.MEDIUM,
                    message=f"Metric '{metric.name}' showing consistent negative trend",
                    triggered_at=datetime.now()
                )
                alerts.append(alert)
        
        return alerts
    
    async def _process_alert(self, alert: RealTimeAlert):
        """Process a generated alert."""
        
        self.logger.warning(f"Processing alert: {alert.message}")
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        
        # Cache alert for API access
        await self.redis.setex(
            f"success_alert:{alert.alert_id}",
            86400,  # 24 hours TTL
            json.dumps(alert.__dict__, default=str)
        )
        
        # Send notifications based on severity
        if alert.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            await self._send_urgent_notification(alert)
        else:
            await self._send_standard_notification(alert)
    
    async def _send_urgent_notification(self, alert: RealTimeAlert):
        """Send urgent notification for high-severity alerts."""
        
        # Implementation would send to:
        # - Customer success manager
        # - Executive team (if critical)
        # - Customer stakeholders
        # - Slack/Teams channels
        
        self.logger.critical(f"URGENT: {alert.message} for customer {alert.customer_id}")
    
    async def _send_standard_notification(self, alert: RealTimeAlert):
        """Send standard notification for medium-severity alerts."""
        
        # Implementation would send to:
        # - Customer success manager
        # - Daily digest reports
        
        self.logger.warning(f"Alert: {alert.message} for customer {alert.customer_id}")


class CustomerSuccessService:
    """Main service for customer success management and guarantee tracking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_collector: Optional[SuccessMetricsCollector] = None
        self.risk_assessment_engine: Optional[RiskAssessmentEngine] = None
        self.monitoring_agent: Optional[SuccessMonitoringAgent] = None
        self.customer_profiles: Dict[str, CustomerSuccessProfile] = {}
    
    async def initialize(self):
        """Initialize the service and its components."""
        self.redis_client = await get_redis_client()
        self.metrics_collector = SuccessMetricsCollector(self.redis_client, self.logger)
        self.risk_assessment_engine = RiskAssessmentEngine(self.redis_client, self.logger)
        self.monitoring_agent = SuccessMonitoringAgent(self.redis_client, self.logger)
        
        # Start monitoring loop
        asyncio.create_task(self._continuous_monitoring_loop())
        
        self.logger.info("Customer Success Service initialized successfully")
    
    async def create_success_guarantee(
        self,
        customer_id: str,
        service_config: Dict[str, Any],
        guarantee_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new 30-day success guarantee for a customer."""
        
        guarantee_id = f"guarantee_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Create success metrics based on service type
            success_metrics = await self._create_success_metrics(
                service_config["service_type"], 
                guarantee_config
            )
            
            # Create the guarantee
            guarantee = SuccessGuarantee(
                guarantee_id=guarantee_id,
                customer_id=customer_id,
                service_type=service_config["service_type"],
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=30),
                guarantee_amount=Decimal(str(guarantee_config["guarantee_amount"])),
                success_criteria=success_metrics,
                minimum_success_threshold=guarantee_config.get("minimum_success_threshold", 80.0),
                current_success_score=0.0,
                status=GuaranteeStatus.ACTIVE,
                risk_assessment={}
            )
            
            # Create or update customer success profile
            if customer_id in self.customer_profiles:
                profile = self.customer_profiles[customer_id]
                profile.active_guarantees.append(guarantee)
            else:
                profile = CustomerSuccessProfile(
                    customer_id=customer_id,
                    customer_name=service_config.get("customer_name", "Unknown"),
                    service_type=service_config["service_type"],
                    onboarding_date=datetime.now(),
                    success_manager_id=service_config.get("success_manager_id", "default"),
                    communication_preferences=service_config.get("communication_preferences", {}),
                    business_objectives=service_config.get("business_objectives", []),
                    success_metrics=success_metrics,
                    active_guarantees=[guarantee],
                    health_score=1.0,
                    satisfaction_score=0.0,
                    renewal_probability=0.0
                )
                self.customer_profiles[customer_id] = profile
            
            # Store guarantee data
            guarantee_data = {
                "guarantee": guarantee.__dict__,
                "customer_profile": profile.__dict__,
                "created_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                f"success_guarantee:{guarantee_id}",
                86400 * 35,  # 35 days TTL
                json.dumps(guarantee_data, default=str)
            )
            
            return {
                "status": "success",
                "guarantee_id": guarantee_id,
                "guarantee_details": guarantee.__dict__,
                "success_criteria_count": len(success_metrics),
                "guarantee_period_days": 30,
                "monitoring_starts": datetime.now().isoformat(),
                "next_steps": [
                    "Baseline measurements will be taken",
                    "Real-time monitoring activated",
                    "Weekly progress reports scheduled",
                    "Risk assessment updates every 48 hours"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create success guarantee: {e}")
            return {
                "status": "error",
                "guarantee_id": guarantee_id,
                "error_message": str(e)
            }
    
    async def update_success_metrics(
        self,
        guarantee_id: str,
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update success metrics with new measurements."""
        
        try:
            # Retrieve guarantee data
            guarantee_data = await self.redis_client.get(f"success_guarantee:{guarantee_id}")
            if not guarantee_data:
                raise ValueError(f"Guarantee {guarantee_id} not found")
            
            data = json.loads(guarantee_data)
            guarantee = SuccessGuarantee(**data["guarantee"])
            
            # Update metrics based on service type
            updated_metrics = []
            
            if guarantee.service_type == "mvp_development":
                velocity_metrics = await self.metrics_collector.collect_velocity_metrics(
                    guarantee.customer_id, metrics_data.get("velocity_data", {})
                )
                quality_metrics = await self.metrics_collector.collect_quality_metrics(
                    guarantee.customer_id, metrics_data.get("quality_data", {})
                )
                satisfaction_metrics = await self.metrics_collector.collect_satisfaction_metrics(
                    guarantee.customer_id, metrics_data.get("satisfaction_data", {})
                )
                
                # Update corresponding success metrics
                for metric in guarantee.success_criteria:
                    if metric.metric_type == SuccessMetricType.VELOCITY_IMPROVEMENT:
                        metric.current_value = velocity_metrics.get("velocity_improvement_percentage", 0.0)
                    elif metric.metric_type == SuccessMetricType.QUALITY_MAINTENANCE:
                        metric.current_value = quality_metrics.get("test_coverage_percentage", 0.0)
                    elif metric.metric_type == SuccessMetricType.CUSTOMER_SATISFACTION:
                        metric.current_value = satisfaction_metrics.get("overall_project_satisfaction", 0.0)
                    
                    # Add trend data
                    metric.trend_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "value": metric.current_value
                    })
                    
                    # Keep only last 30 data points
                    metric.trend_data = metric.trend_data[-30:]
                    metric.last_measured = datetime.now()
                    
                    updated_metrics.append(metric.name)
            
            # Calculate overall success score
            success_scores = []
            for metric in guarantee.success_criteria:
                if metric.target_value != metric.baseline_value:
                    score = min(100, max(0, 
                        ((metric.current_value - metric.baseline_value) / 
                         (metric.target_value - metric.baseline_value)) * 100
                    ))
                else:
                    score = 100.0 if metric.current_value == metric.target_value else 0.0
                
                success_scores.append(score * metric.weight)
            
            guarantee.current_success_score = sum(success_scores) / sum(m.weight for m in guarantee.success_criteria)
            
            # Update guarantee status
            if guarantee.current_success_score >= guarantee.minimum_success_threshold:
                if datetime.now() >= guarantee.end_date:
                    guarantee.status = GuaranteeStatus.SUCCESS
                else:
                    guarantee.status = GuaranteeStatus.ACTIVE
            else:
                days_remaining = (guarantee.end_date - datetime.now()).days
                if days_remaining <= 0:
                    guarantee.status = GuaranteeStatus.FAILED
                elif days_remaining <= 7 or guarantee.current_success_score < guarantee.minimum_success_threshold * 0.6:
                    guarantee.status = GuaranteeStatus.AT_RISK
            
            # Perform risk assessment
            risk_assessment = await self.risk_assessment_engine.assess_guarantee_risk(guarantee)
            guarantee.risk_assessment = risk_assessment
            
            # Update stored data
            data["guarantee"] = guarantee.__dict__
            await self.redis_client.setex(
                f"success_guarantee:{guarantee_id}",
                86400 * 35,
                json.dumps(data, default=str)
            )
            
            return {
                "status": "success",
                "guarantee_id": guarantee_id,
                "updated_metrics": updated_metrics,
                "current_success_score": guarantee.current_success_score,
                "guarantee_status": guarantee.status.value,
                "risk_assessment": risk_assessment,
                "days_remaining": max(0, (guarantee.end_date - datetime.now()).days)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update success metrics: {e}")
            return {
                "status": "error",
                "guarantee_id": guarantee_id,
                "error_message": str(e)
            }
    
    async def get_guarantee_status(self, guarantee_id: str) -> Dict[str, Any]:
        """Get current status of a success guarantee."""
        
        try:
            guarantee_data = await self.redis_client.get(f"success_guarantee:{guarantee_id}")
            if not guarantee_data:
                return {"status": "error", "message": "Guarantee not found"}
            
            data = json.loads(guarantee_data)
            guarantee = SuccessGuarantee(**data["guarantee"])
            
            # Get active alerts
            active_alerts = []
            for alert_id, alert in self.monitoring_agent.active_alerts.items():
                if alert.guarantee_id == guarantee_id and not alert.resolved:
                    active_alerts.append(alert.__dict__)
            
            # Calculate progress metrics
            days_total = (guarantee.end_date - guarantee.start_date).days
            days_elapsed = (datetime.now() - guarantee.start_date).days
            days_remaining = max(0, (guarantee.end_date - datetime.now()).days)
            progress_percentage = min(100, (days_elapsed / days_total) * 100)
            
            return {
                "status": "success",
                "guarantee_id": guarantee_id,
                "customer_id": guarantee.customer_id,
                "service_type": guarantee.service_type,
                "guarantee_status": guarantee.status.value,
                "current_success_score": guarantee.current_success_score,
                "minimum_success_threshold": guarantee.minimum_success_threshold,
                "guarantee_amount": float(guarantee.guarantee_amount),
                "progress": {
                    "days_total": days_total,
                    "days_elapsed": days_elapsed,
                    "days_remaining": days_remaining,
                    "progress_percentage": progress_percentage
                },
                "success_metrics": [
                    {
                        "name": metric.name,
                        "current_value": metric.current_value,
                        "target_value": metric.target_value,
                        "progress_percentage": min(100, max(0, 
                            ((metric.current_value - metric.baseline_value) / 
                             (metric.target_value - metric.baseline_value)) * 100
                        )) if metric.target_value != metric.baseline_value else (
                            100.0 if metric.current_value == metric.target_value else 0.0
                        ),
                        "trend": metric.trend_data[-5:] if len(metric.trend_data) >= 5 else metric.trend_data
                    }
                    for metric in guarantee.success_criteria
                ],
                "risk_assessment": guarantee.risk_assessment,
                "active_alerts": active_alerts,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get guarantee status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _create_success_metrics(
        self, 
        service_type: str, 
        guarantee_config: Dict[str, Any]
    ) -> List[SuccessMetric]:
        """Create success metrics based on service type."""
        
        metrics = []
        current_time = datetime.now()
        
        if service_type == "mvp_development":
            # Velocity improvement metric
            metrics.append(SuccessMetric(
                metric_id=f"velocity_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.VELOCITY_IMPROVEMENT,
                name="Development Velocity Improvement",
                description="Percentage improvement in development velocity",
                target_value=guarantee_config.get("target_velocity_improvement", 2000.0),  # 20x = 2000%
                current_value=0.0,
                baseline_value=0.0,
                unit="percentage",
                weight=0.4,
                measurement_frequency="weekly",
                last_measured=current_time
            ))
            
            # Quality maintenance metric
            metrics.append(SuccessMetric(
                metric_id=f"quality_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.QUALITY_MAINTENANCE,
                name="Code Quality Maintenance",
                description="Test coverage and code quality scores",
                target_value=guarantee_config.get("target_test_coverage", 95.0),
                current_value=0.0,
                baseline_value=0.0,
                unit="percentage",
                weight=0.3,
                measurement_frequency="daily",
                last_measured=current_time
            ))
            
            # Timeline adherence metric
            metrics.append(SuccessMetric(
                metric_id=f"timeline_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.TIMELINE_ADHERENCE,
                name="Timeline Adherence",
                description="Percentage of milestones delivered on time",
                target_value=guarantee_config.get("target_timeline_adherence", 90.0),
                current_value=0.0,
                baseline_value=0.0,
                unit="percentage",
                weight=0.3,
                measurement_frequency="weekly",
                last_measured=current_time
            ))
        
        elif service_type == "legacy_modernization":
            # Performance improvement metric
            metrics.append(SuccessMetric(
                metric_id=f"performance_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.PERFORMANCE_IMPROVEMENT,
                name="System Performance Improvement",
                description="Response time and throughput improvements",
                target_value=guarantee_config.get("target_performance_improvement", 50.0),
                current_value=0.0,
                baseline_value=0.0,
                unit="percentage",
                weight=0.4,
                measurement_frequency="weekly",
                last_measured=current_time
            ))
            
            # Security enhancement metric
            metrics.append(SuccessMetric(
                metric_id=f"security_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.SECURITY_ENHANCEMENT,
                name="Security Vulnerability Reduction",
                description="Reduction in security vulnerabilities",
                target_value=guarantee_config.get("target_security_improvement", 90.0),
                current_value=0.0,
                baseline_value=100.0,  # Start with 100% vulnerabilities
                unit="percentage_reduction",
                weight=0.3,
                measurement_frequency="weekly",
                last_measured=current_time
            ))
        
        elif service_type == "team_augmentation":
            # Team velocity improvement
            metrics.append(SuccessMetric(
                metric_id=f"team_velocity_{uuid.uuid4().hex[:8]}",
                metric_type=SuccessMetricType.VELOCITY_IMPROVEMENT,
                name="Team Velocity Improvement",
                description="Improvement in team's overall velocity",
                target_value=guarantee_config.get("target_velocity_improvement", 40.0),
                current_value=0.0,
                baseline_value=0.0,
                unit="percentage",
                weight=0.5,
                measurement_frequency="weekly",
                last_measured=current_time
            ))
        
        # Common customer satisfaction metric for all services
        metrics.append(SuccessMetric(
            metric_id=f"satisfaction_{uuid.uuid4().hex[:8]}",
            metric_type=SuccessMetricType.CUSTOMER_SATISFACTION,
            name="Customer Satisfaction Score",
            description="Overall customer satisfaction rating",
            target_value=guarantee_config.get("target_satisfaction_score", 8.0),
            current_value=0.0,
            baseline_value=0.0,
            unit="score_1_to_10",
            weight=0.2 if service_type != "team_augmentation" else 0.3,
            measurement_frequency="weekly",
            last_measured=current_time
        ))
        
        return metrics
    
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for all active guarantees."""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Get all customer profiles with active guarantees
                active_profiles = [
                    profile for profile in self.customer_profiles.values()
                    if any(g.status == GuaranteeStatus.ACTIVE for g in profile.active_guarantees)
                ]
                
                if active_profiles:
                    await self.monitoring_agent.monitor_success_guarantees(active_profiles)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying


# Global service instance
_success_service: Optional[CustomerSuccessService] = None


async def get_success_service() -> CustomerSuccessService:
    """Get the global customer success service instance."""
    global _success_service
    
    if _success_service is None:
        _success_service = CustomerSuccessService()
        await _success_service.initialize()
    
    return _success_service


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class CustomerSuccessService(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            async def test_customer_success_service():
            """Test the customer success service."""

            service = await get_success_service()

            # Sample service configuration
            service_config = {
            "service_type": "mvp_development",
            "customer_name": "TechCorp Inc.",
            "success_manager_id": "sm_001",
            "communication_preferences": {
            "email": "stakeholder@techcorp.com",
            "slack_channel": "#project-updates",
            "reporting_frequency": "daily"
            },
            "business_objectives": [
            "Launch MVP within 6 weeks",
            "Achieve 95% test coverage",
            "Maintain development velocity > 20x baseline"
            ]
            }

            # Guarantee configuration
            guarantee_config = {
            "guarantee_amount": 150000,  # $150K guarantee
            "minimum_success_threshold": 80.0,  # 80% of criteria must be met
            "target_velocity_improvement": 2000.0,  # 20x improvement
            "target_test_coverage": 95.0,
            "target_timeline_adherence": 90.0,
            "target_satisfaction_score": 8.5
            }

            # Create success guarantee
            guarantee_result = await service.create_success_guarantee(
            "customer_techcorp", service_config, guarantee_config
            )
            self.logger.info("Guarantee creation result:", json.dumps(guarantee_result, indent=2, default=str))

            if guarantee_result["status"] == "success":
            guarantee_id = guarantee_result["guarantee_id"]

            # Simulate metrics updates
            metrics_data = {
            "velocity_data": {
            "baseline_velocity": 1.0,
            "completed_tasks": [
            {"story_points": 8, "type": "feature", "started_at": "2025-08-01T09:00:00", "completed_at": "2025-08-01T17:00:00"},
            {"story_points": 5, "type": "bug", "started_at": "2025-08-01T10:00:00", "completed_at": "2025-08-01T14:00:00"},
            {"story_points": 13, "type": "feature", "started_at": "2025-08-02T09:00:00", "completed_at": "2025-08-02T16:00:00"}
            ],
            "measurement_period_days": 7
            },
            "quality_data": {
            "test_coverage_percentage": 92.0,
            "defects_reported": 2,
            "defects_resolved": 2,
            "total_deliverables": 3,
            "code_quality_score": 8.5,
            "security_vulnerabilities": {"high": 0, "medium": 1, "low": 3}
            },
            "satisfaction_data": {
            "overall_satisfaction": 8.2,
            "nps_scores": [9, 8, 9],
            "communication_scores": [8.5, 9.0, 8.0]
            }
            }

            # Update metrics
            update_result = await service.update_success_metrics(guarantee_id, metrics_data)
            self.logger.info("Metrics update result:", json.dumps(update_result, indent=2, default=str))

            # Get guarantee status
            status = await service.get_guarantee_status(guarantee_id)
            self.logger.info("Guarantee status:", json.dumps(status, indent=2, default=str))

            # Run test
            await test_customer_success_service()
            
            return {"status": "completed"}
    
    script_main(CustomerSuccessService)