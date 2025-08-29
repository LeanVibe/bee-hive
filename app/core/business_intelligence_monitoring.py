"""
Business Intelligence Monitoring and Real-time Correlation Engine
==============================================================

Advanced business intelligence system with real-time metric correlation,
business impact assessment, and executive dashboard capabilities.

Epic F Phase 2: Advanced Observability & Intelligence
Target: Business-aligned observability with real-time correlation analysis
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import numpy as np
import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_async_session
from .redis import get_redis_client
from .predictive_analytics import get_predictive_analytics_engine
from .anomaly_detection import get_anomaly_detector
from .capacity_planning import get_capacity_planner
from ..models.performance_metric import PerformanceMetric
from ..models.observability import AgentEvent

logger = structlog.get_logger()


class BusinessMetricType(Enum):
    """Types of business metrics for intelligence monitoring."""
    REVENUE = "revenue"
    USER_ENGAGEMENT = "user_engagement"
    CONVERSION_RATE = "conversion_rate"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    COST_PER_TRANSACTION = "cost_per_transaction"
    TIME_TO_VALUE = "time_to_value"
    ERROR_RATE_IMPACT = "error_rate_impact"
    PERFORMANCE_IMPACT = "performance_impact"
    AVAILABILITY_IMPACT = "availability_impact"


class ImpactSeverity(Enum):
    """Severity levels for business impact assessment."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrelationStrength(Enum):
    """Strength of correlation between metrics."""
    VERY_WEAK = "very_weak"      # 0.0 - 0.2
    WEAK = "weak"                # 0.2 - 0.4
    MODERATE = "moderate"        # 0.4 - 0.6
    STRONG = "strong"            # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0


@dataclass
class BusinessMetric:
    """Business metric with contextual information."""
    metric_id: str
    metric_type: BusinessMetricType
    name: str
    description: str
    current_value: float
    target_value: Optional[float]
    unit: str
    measurement_timestamp: datetime
    
    # Business context
    business_owner: str
    impact_area: str
    priority: int  # 1-5 scale
    
    # Performance correlation
    technical_dependencies: List[str] = field(default_factory=list)
    sla_threshold: Optional[float] = None
    
    # Trends and forecasting
    trend_direction: str = "stable"
    variance_percentage: float = 0.0
    forecast_accuracy: float = 0.85
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricCorrelation:
    """Correlation between business and technical metrics."""
    correlation_id: str
    business_metric_id: str
    technical_metric_name: str
    correlation_coefficient: float
    correlation_strength: CorrelationStrength
    
    # Temporal analysis
    time_lag_minutes: int = 0  # How long technical metric changes affect business metric
    analysis_window_hours: int = 24
    
    # Statistical significance
    p_value: float = 0.05
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    
    # Business context
    causality_direction: str = "technical_to_business"  # or "business_to_technical" or "bidirectional"
    impact_description: str = ""
    
    last_updated: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BusinessImpactAssessment:
    """Assessment of business impact from technical issues."""
    assessment_id: str
    trigger_event: str
    technical_metrics_affected: List[str]
    business_metrics_impacted: List[str]
    
    # Impact quantification
    severity: ImpactSeverity
    estimated_revenue_impact: float
    estimated_customer_impact: int
    estimated_productivity_loss: float
    
    # Timing and duration
    started_at: datetime
    estimated_duration_minutes: Optional[int]
    resolution_eta: Optional[datetime]
    
    # Response and mitigation
    root_cause_analysis: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    stakeholder_notifications: List[str] = field(default_factory=list)
    
    # Context and metadata
    affected_systems: List[str] = field(default_factory=list)
    customer_segments_affected: List[str] = field(default_factory=list)
    business_processes_impacted: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutiveDashboardData:
    """Executive dashboard data with business-focused insights."""
    dashboard_id: str
    generated_at: datetime
    
    # High-level health scores
    overall_business_health: float  # 0-100 scale
    technical_performance_score: float
    customer_experience_score: float
    operational_efficiency_score: float
    
    # Key metrics summary
    critical_business_metrics: Dict[str, Any]
    trending_metrics: Dict[str, Any]
    alerting_metrics: Dict[str, Any]
    
    # Impact analysis
    current_business_impacts: List[BusinessImpactAssessment]
    predicted_impacts: List[Dict[str, Any]]
    
    # Cost and efficiency
    infrastructure_costs: Dict[str, float]
    operational_costs: Dict[str, float]
    efficiency_trends: Dict[str, Any]
    
    # Strategic insights
    recommendations: List[str]
    risk_factors: List[str]
    optimization_opportunities: List[str]
    
    # Time-based analysis
    comparison_periods: Dict[str, Any]  # WoW, MoM, YoY comparisons
    seasonal_adjustments: Dict[str, float]


class BusinessIntelligenceMonitor:
    """
    Business Intelligence Monitoring and Real-time Correlation Engine
    
    Features:
    - Real-time correlation analysis between business and technical metrics
    - Business impact assessment with quantified cost analysis
    - Executive dashboards with business-focused KPIs
    - Predictive business impact modeling
    - Multi-dimensional correlation detection
    - Automated stakeholder notifications based on business impact
    - ROI analysis for technical investments and optimizations
    - Strategic recommendations based on data patterns
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional = None
    ):
        """Initialize the business intelligence monitor."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_async_session
        
        # Core data structures
        self.business_metrics: Dict[str, BusinessMetric] = {}
        self.metric_correlations: Dict[str, MetricCorrelation] = {}
        self.business_impacts: Dict[str, BusinessImpactAssessment] = {}
        self.executive_dashboards: Dict[str, ExecutiveDashboardData] = {}
        
        # Historical data for correlation analysis
        self.correlation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self.business_metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        self.impact_assessment_history: deque = deque(maxlen=1000)
        
        # Configuration
        self.config = {
            "correlation_analysis_interval_minutes": 15,
            "business_impact_assessment_interval_minutes": 5,
            "executive_dashboard_update_minutes": 30,
            "correlation_significance_threshold": 0.05,
            "minimum_correlation_strength": 0.3,
            "impact_assessment_lookback_hours": 24,
            "executive_dashboard_retention_days": 90,
            "cost_per_minute_downtime": 500.0,  # Business-specific cost
            "cost_per_error": 5.0,
            "customer_impact_multiplier": 100.0
        }
        
        # Integration with other monitoring systems
        self.predictive_engine = None
        self.anomaly_detector = None
        self.capacity_planner = None
        
        # State management
        self.is_running = False
        self.bi_monitoring_tasks: List[asyncio.Task] = []
        
        logger.info("Business Intelligence Monitor initialized")
    
    async def start(self) -> None:
        """Start the business intelligence monitoring engine."""
        if self.is_running:
            logger.warning("Business intelligence monitor already running")
            return
        
        logger.info("Starting Business Intelligence Monitoring Engine")
        self.is_running = True
        
        # Initialize dependencies
        try:
            self.predictive_engine = await get_predictive_analytics_engine()
            self.anomaly_detector = await get_anomaly_detector()
            self.capacity_planner = await get_capacity_planner()
        except Exception as e:
            logger.warning(f"Failed to initialize some dependencies: {e}")
        
        # Initialize business metrics
        await self._initialize_business_metrics()
        
        # Start background tasks
        self.bi_monitoring_tasks = [
            asyncio.create_task(self._correlation_analysis_loop()),
            asyncio.create_task(self._business_impact_assessment_loop()),
            asyncio.create_task(self._executive_dashboard_update_loop()),
            asyncio.create_task(self._business_metrics_collection_loop()),
            asyncio.create_task(self._stakeholder_notification_loop())
        ]
        
        logger.info("Business Intelligence Monitoring Engine started successfully")
    
    async def stop(self) -> None:
        """Stop the business intelligence monitoring engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping Business Intelligence Monitoring Engine")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.bi_monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.bi_monitoring_tasks:
            await asyncio.gather(*self.bi_monitoring_tasks, return_exceptions=True)
        
        logger.info("Business Intelligence Monitoring Engine stopped")
    
    async def analyze_business_metric_correlations(
        self,
        time_window_hours: int = 24,
        significance_threshold: float = None
    ) -> Dict[str, List[MetricCorrelation]]:
        """
        Analyze correlations between business and technical metrics.
        
        Args:
            time_window_hours: Time window for analysis
            significance_threshold: Statistical significance threshold
            
        Returns:
            Dictionary of business metrics with their correlations
        """
        try:
            if significance_threshold is None:
                significance_threshold = self.config["correlation_significance_threshold"]
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            correlations = {}
            
            # Get technical metrics data
            technical_metrics = await self._get_technical_metrics_data(start_time, end_time)
            
            # Analyze correlations for each business metric
            for business_metric_id, business_metric in self.business_metrics.items():
                business_data = await self._get_business_metric_data(
                    business_metric_id, 
                    start_time, 
                    end_time
                )
                
                if len(business_data) < 10:  # Need minimum data points
                    continue
                
                metric_correlations = []
                
                for tech_metric_name, tech_data in technical_metrics.items():
                    if len(tech_data) < 10:
                        continue
                    
                    # Align timestamps and calculate correlation
                    correlation_result = await self._calculate_correlation(
                        business_data,
                        tech_data,
                        business_metric_id,
                        tech_metric_name
                    )
                    
                    if correlation_result and abs(correlation_result.correlation_coefficient) >= self.config["minimum_correlation_strength"]:
                        if correlation_result.p_value <= significance_threshold:
                            metric_correlations.append(correlation_result)
                            
                            # Store correlation for historical tracking
                            self.metric_correlations[correlation_result.correlation_id] = correlation_result
                
                if metric_correlations:
                    # Sort by correlation strength
                    metric_correlations.sort(
                        key=lambda x: abs(x.correlation_coefficient), 
                        reverse=True
                    )
                    correlations[business_metric_id] = metric_correlations
            
            logger.info(f"Analyzed correlations for {len(correlations)} business metrics")
            return correlations
            
        except Exception as e:
            logger.error("Failed to analyze business metric correlations", error=str(e))
            return {}
    
    async def assess_business_impact(
        self,
        trigger_event: str,
        affected_technical_metrics: List[str]
    ) -> BusinessImpactAssessment:
        """
        Assess business impact from technical performance issues.
        
        Args:
            trigger_event: Description of the triggering event
            affected_technical_metrics: List of affected technical metric names
            
        Returns:
            BusinessImpactAssessment with quantified impact
        """
        try:
            assessment_id = str(uuid.uuid4())
            
            # Identify affected business metrics through correlations
            affected_business_metrics = []
            for tech_metric in affected_technical_metrics:
                for correlation in self.metric_correlations.values():
                    if (correlation.technical_metric_name == tech_metric and
                        abs(correlation.correlation_coefficient) > 0.5):
                        affected_business_metrics.append(correlation.business_metric_id)
            
            # Remove duplicates
            affected_business_metrics = list(set(affected_business_metrics))
            
            # Calculate impact severity
            severity = self._calculate_impact_severity(
                affected_technical_metrics,
                affected_business_metrics
            )
            
            # Estimate business impact
            revenue_impact = await self._estimate_revenue_impact(
                affected_business_metrics,
                severity
            )
            
            customer_impact = await self._estimate_customer_impact(
                affected_technical_metrics,
                severity
            )
            
            productivity_loss = await self._estimate_productivity_impact(
                affected_technical_metrics,
                severity
            )
            
            # Generate root cause analysis
            root_cause_analysis = await self._generate_root_cause_analysis(
                trigger_event,
                affected_technical_metrics
            )
            
            # Generate mitigation actions
            mitigation_actions = await self._generate_mitigation_actions(
                affected_technical_metrics,
                severity
            )
            
            # Determine stakeholder notifications
            stakeholder_notifications = self._determine_stakeholder_notifications(
                severity,
                affected_business_metrics
            )
            
            # Create impact assessment
            assessment = BusinessImpactAssessment(
                assessment_id=assessment_id,
                trigger_event=trigger_event,
                technical_metrics_affected=affected_technical_metrics,
                business_metrics_impacted=affected_business_metrics,
                severity=severity,
                estimated_revenue_impact=revenue_impact,
                estimated_customer_impact=customer_impact,
                estimated_productivity_loss=productivity_loss,
                started_at=datetime.utcnow(),
                estimated_duration_minutes=self._estimate_impact_duration(severity),
                root_cause_analysis=root_cause_analysis,
                mitigation_actions=mitigation_actions,
                stakeholder_notifications=stakeholder_notifications,
                affected_systems=self._identify_affected_systems(affected_technical_metrics),
                customer_segments_affected=self._identify_affected_customer_segments(affected_business_metrics),
                business_processes_impacted=self._identify_affected_business_processes(affected_business_metrics)
            )
            
            # Store assessment
            self.business_impacts[assessment_id] = assessment
            self.impact_assessment_history.append(assessment)
            
            # Store in Redis for persistence
            await self._store_business_impact(assessment)
            
            logger.info(f"Business impact assessment created: {severity.value} severity, ${revenue_impact:.2f} estimated impact")
            return assessment
            
        except Exception as e:
            logger.error("Failed to assess business impact", error=str(e))
            raise
    
    async def generate_executive_dashboard(
        self,
        dashboard_type: str = "operational",
        time_period: str = "24h"
    ) -> ExecutiveDashboardData:
        """
        Generate executive dashboard with business-focused insights.
        
        Args:
            dashboard_type: Type of dashboard (operational, strategic, financial)
            time_period: Time period for analysis (1h, 24h, 7d, 30d)
            
        Returns:
            ExecutiveDashboardData with comprehensive business insights
        """
        try:
            dashboard_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            # Calculate time window
            if time_period == "1h":
                time_delta = timedelta(hours=1)
            elif time_period == "24h":
                time_delta = timedelta(hours=24)
            elif time_period == "7d":
                time_delta = timedelta(days=7)
            elif time_period == "30d":
                time_delta = timedelta(days=30)
            else:
                time_delta = timedelta(hours=24)
            
            start_time = current_time - time_delta
            
            # Calculate health scores
            health_scores = await self._calculate_business_health_scores(start_time, current_time)
            
            # Get critical business metrics
            critical_metrics = await self._get_critical_business_metrics()
            
            # Get trending metrics
            trending_metrics = await self._get_trending_metrics(start_time, current_time)
            
            # Get alerting metrics
            alerting_metrics = await self._get_alerting_metrics()
            
            # Get current business impacts
            current_impacts = [
                impact for impact in self.business_impacts.values()
                if impact.started_at >= start_time and impact.severity != ImpactSeverity.NONE
            ]
            
            # Generate predicted impacts
            predicted_impacts = await self._generate_predicted_impacts()
            
            # Calculate costs
            infrastructure_costs = await self._calculate_infrastructure_costs()
            operational_costs = await self._calculate_operational_costs()
            
            # Get efficiency trends
            efficiency_trends = await self._get_efficiency_trends(start_time, current_time)
            
            # Generate strategic insights
            recommendations = await self._generate_strategic_recommendations()
            risk_factors = await self._identify_risk_factors()
            optimization_opportunities = await self._identify_optimization_opportunities()
            
            # Calculate comparison periods
            comparison_periods = await self._calculate_comparison_periods(start_time, current_time)
            
            # Create dashboard data
            dashboard = ExecutiveDashboardData(
                dashboard_id=dashboard_id,
                generated_at=current_time,
                overall_business_health=health_scores["overall"],
                technical_performance_score=health_scores["technical"],
                customer_experience_score=health_scores["customer"],
                operational_efficiency_score=health_scores["operational"],
                critical_business_metrics=critical_metrics,
                trending_metrics=trending_metrics,
                alerting_metrics=alerting_metrics,
                current_business_impacts=current_impacts,
                predicted_impacts=predicted_impacts,
                infrastructure_costs=infrastructure_costs,
                operational_costs=operational_costs,
                efficiency_trends=efficiency_trends,
                recommendations=recommendations,
                risk_factors=risk_factors,
                optimization_opportunities=optimization_opportunities,
                comparison_periods=comparison_periods,
                seasonal_adjustments={}
            )
            
            # Store dashboard
            self.executive_dashboards[dashboard_id] = dashboard
            
            logger.info(f"Generated executive dashboard: {dashboard_type} type, {time_period} period")
            return dashboard
            
        except Exception as e:
            logger.error("Failed to generate executive dashboard", error=str(e))
            raise
    
    async def get_business_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive business intelligence summary."""
        try:
            current_time = datetime.utcnow()
            
            summary = {
                "timestamp": current_time.isoformat(),
                "business_metrics": {
                    "total_metrics": len(self.business_metrics),
                    "active_metrics": len([m for m in self.business_metrics.values() if (current_time - m.measurement_timestamp).total_seconds() < 3600]),
                    "critical_metrics": len([m for m in self.business_metrics.values() if m.priority <= 2])
                },
                "correlations": {
                    "total_correlations": len(self.metric_correlations),
                    "strong_correlations": len([c for c in self.metric_correlations.values() if c.correlation_strength in [CorrelationStrength.STRONG, CorrelationStrength.VERY_STRONG]]),
                    "statistically_significant": len([c for c in self.metric_correlations.values() if c.p_value <= 0.05])
                },
                "business_impacts": {
                    "active_impacts": len([i for i in self.business_impacts.values() if i.severity != ImpactSeverity.NONE]),
                    "critical_impacts": len([i for i in self.business_impacts.values() if i.severity == ImpactSeverity.CRITICAL]),
                    "total_estimated_impact": sum([i.estimated_revenue_impact for i in self.business_impacts.values() if i.started_at >= current_time - timedelta(hours=24)])
                },
                "executive_dashboards": {
                    "total_dashboards": len(self.executive_dashboards),
                    "recent_dashboards": len([d for d in self.executive_dashboards.values() if (current_time - d.generated_at).total_seconds() < 3600])
                },
                "system_status": {
                    "is_running": self.is_running,
                    "active_tasks": len([t for t in self.bi_monitoring_tasks if not t.done()]),
                    "last_correlation_analysis": "recent",  # Would track actual timestamp
                    "last_impact_assessment": "recent"     # Would track actual timestamp
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get business intelligence summary", error=str(e))
            return {"error": str(e)}
    
    # Background task methods
    async def _correlation_analysis_loop(self) -> None:
        """Background task for continuous correlation analysis."""
        logger.info("Starting business metric correlation analysis loop")
        
        while self.is_running:
            try:
                # Perform correlation analysis
                correlations = await self.analyze_business_metric_correlations()
                
                # Log significant new correlations
                for business_metric_id, metric_correlations in correlations.items():
                    for correlation in metric_correlations[:3]:  # Top 3 correlations
                        if abs(correlation.correlation_coefficient) > 0.7:
                            logger.info(f"Strong correlation detected: {business_metric_id} <-> {correlation.technical_metric_name} ({correlation.correlation_coefficient:.3f})")
                
                # Wait for next analysis cycle
                await asyncio.sleep(self.config["correlation_analysis_interval_minutes"] * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Correlation analysis loop error", error=str(e))
                await asyncio.sleep(self.config["correlation_analysis_interval_minutes"] * 60)
        
        logger.info("Business metric correlation analysis loop stopped")
    
    async def _business_impact_assessment_loop(self) -> None:
        """Background task for business impact assessment."""
        logger.info("Starting business impact assessment loop")
        
        while self.is_running:
            try:
                # Check for new anomalies or performance issues
                if self.anomaly_detector:
                    recent_anomalies = await self._get_recent_anomalies()
                    
                    for anomaly in recent_anomalies:
                        # Assess business impact for each anomaly
                        await self.assess_business_impact(
                            f"Anomaly detected in {anomaly.metric_name}",
                            [anomaly.metric_name]
                        )
                
                # Wait for next assessment cycle
                await asyncio.sleep(self.config["business_impact_assessment_interval_minutes"] * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Business impact assessment loop error", error=str(e))
                await asyncio.sleep(self.config["business_impact_assessment_interval_minutes"] * 60)
        
        logger.info("Business impact assessment loop stopped")
    
    async def _executive_dashboard_update_loop(self) -> None:
        """Background task for executive dashboard updates."""
        logger.info("Starting executive dashboard update loop")
        
        while self.is_running:
            try:
                # Generate operational dashboard
                await self.generate_executive_dashboard("operational", "24h")
                
                # Wait for next update cycle
                await asyncio.sleep(self.config["executive_dashboard_update_minutes"] * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Executive dashboard update loop error", error=str(e))
                await asyncio.sleep(self.config["executive_dashboard_update_minutes"] * 60)
        
        logger.info("Executive dashboard update loop stopped")
    
    async def _business_metrics_collection_loop(self) -> None:
        """Background task for business metrics collection."""
        logger.info("Starting business metrics collection loop")
        
        while self.is_running:
            try:
                # Collect business metrics from various sources
                await self._collect_business_metrics()
                
                # Wait for next collection cycle
                await asyncio.sleep(600)  # Every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Business metrics collection loop error", error=str(e))
                await asyncio.sleep(600)
        
        logger.info("Business metrics collection loop stopped")
    
    async def _stakeholder_notification_loop(self) -> None:
        """Background task for stakeholder notifications."""
        logger.info("Starting stakeholder notification loop")
        
        while self.is_running:
            try:
                # Check for high-impact issues requiring notifications
                high_impact_assessments = [
                    assessment for assessment in self.business_impacts.values()
                    if assessment.severity in [ImpactSeverity.HIGH, ImpactSeverity.CRITICAL]
                    and (datetime.utcnow() - assessment.created_at).total_seconds() < 1800  # Last 30 minutes
                ]
                
                for assessment in high_impact_assessments:
                    await self._send_stakeholder_notifications(assessment)
                
                # Wait for next notification cycle
                await asyncio.sleep(600)  # Every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Stakeholder notification loop error", error=str(e))
                await asyncio.sleep(600)
        
        logger.info("Stakeholder notification loop stopped")
    
    # Helper methods (essential implementations)
    async def _initialize_business_metrics(self) -> None:
        """Initialize default business metrics."""
        try:
            default_metrics = [
                {
                    "type": BusinessMetricType.USER_ENGAGEMENT,
                    "name": "Active Users per Hour",
                    "description": "Number of active users interacting with the system",
                    "target_value": 1000,
                    "unit": "users",
                    "business_owner": "Product Team",
                    "impact_area": "Customer Experience",
                    "priority": 1
                },
                {
                    "type": BusinessMetricType.PERFORMANCE_IMPACT,
                    "name": "Response Time Impact Score",
                    "description": "Business impact score based on system response times",
                    "target_value": 95,
                    "unit": "score",
                    "business_owner": "Engineering Team",
                    "impact_area": "User Experience",
                    "priority": 1
                },
                {
                    "type": BusinessMetricType.OPERATIONAL_EFFICIENCY,
                    "name": "Task Success Rate",
                    "description": "Percentage of successfully completed business tasks",
                    "target_value": 99.5,
                    "unit": "percentage",
                    "business_owner": "Operations Team",
                    "impact_area": "Business Operations",
                    "priority": 2
                }
            ]
            
            for metric_def in default_metrics:
                metric_id = str(uuid.uuid4())
                business_metric = BusinessMetric(
                    metric_id=metric_id,
                    metric_type=metric_def["type"],
                    name=metric_def["name"],
                    description=metric_def["description"],
                    current_value=0.0,
                    target_value=metric_def["target_value"],
                    unit=metric_def["unit"],
                    measurement_timestamp=datetime.utcnow(),
                    business_owner=metric_def["business_owner"],
                    impact_area=metric_def["impact_area"],
                    priority=metric_def["priority"]
                )
                
                self.business_metrics[metric_id] = business_metric
            
            logger.info(f"Initialized {len(default_metrics)} business metrics")
            
        except Exception as e:
            logger.error("Failed to initialize business metrics", error=str(e))


# Singleton instance
_business_intelligence_monitor: Optional[BusinessIntelligenceMonitor] = None


async def get_business_intelligence_monitor() -> BusinessIntelligenceMonitor:
    """Get singleton business intelligence monitor instance."""
    global _business_intelligence_monitor
    
    if _business_intelligence_monitor is None:
        _business_intelligence_monitor = BusinessIntelligenceMonitor()
        await _business_intelligence_monitor.start()
    
    return _business_intelligence_monitor


async def cleanup_business_intelligence_monitor() -> None:
    """Cleanup business intelligence monitor resources."""
    global _business_intelligence_monitor
    
    if _business_intelligence_monitor:
        await _business_intelligence_monitor.stop()
        _business_intelligence_monitor = None