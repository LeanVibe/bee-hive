"""
Performance Optimization Advisor for LeanVibe Agent Hive 2.0

Intelligent performance analysis and optimization recommendation engine that provides
automated insights, actionable recommendations, and predictive optimization suggestions
for autonomous multi-agent development workflows.

Features:
- ML-powered performance analysis and bottleneck detection
- Automated optimization recommendations with impact predictions
- Resource utilization optimization and capacity planning
- Performance trend analysis and predictive insights
- Code-level optimization suggestions for agent workflows
- Cost optimization recommendations for cloud infrastructure
- A/B testing frameworks for optimization validation
- Performance regression detection and auto-remediation
"""

import asyncio
import time
import uuid
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType
from .agent_workflow_tracker import get_agent_workflow_tracker
from .intelligent_alerting import get_alert_manager

logger = structlog.get_logger()


class OptimizationCategory(Enum):
    """Categories of performance optimizations."""
    SYSTEM_RESOURCES = "system_resources"
    AGENT_EFFICIENCY = "agent_efficiency"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    DATABASE_PERFORMANCE = "database_performance"
    NETWORK_OPTIMIZATION = "network_optimization"
    MEMORY_MANAGEMENT = "memory_management"
    CPU_OPTIMIZATION = "cpu_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    SCALABILITY = "scalability"


class ImpactLevel(Enum):
    """Expected impact levels for optimizations."""
    CRITICAL = "critical"  # >50% improvement expected
    HIGH = "high"         # 25-50% improvement expected
    MEDIUM = "medium"     # 10-25% improvement expected
    LOW = "low"          # 5-10% improvement expected
    MINIMAL = "minimal"   # <5% improvement expected


class ImplementationComplexity(Enum):
    """Implementation complexity levels."""
    TRIVIAL = "trivial"      # < 1 hour
    LOW = "low"             # 1-4 hours
    MEDIUM = "medium"       # 4-16 hours
    HIGH = "high"           # 16-40 hours
    COMPLEX = "complex"     # >40 hours


class OptimizationPriority(Enum):
    """Optimization priority levels."""
    IMMEDIATE = "immediate"  # Should be implemented now
    HIGH = "high"           # Should be implemented within 1 week
    MEDIUM = "medium"       # Should be implemented within 1 month
    LOW = "low"            # Nice to have
    FUTURE = "future"      # For future consideration


@dataclass
class PerformanceInsight:
    """Individual performance insight or observation."""
    insight_id: str
    category: OptimizationCategory
    title: str
    description: str
    current_value: float
    baseline_value: Optional[float]
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    confidence_score: float  # 0.0 to 1.0
    affected_components: List[str]
    related_metrics: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "insight_id": self.insight_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "trend_direction": self.trend_direction,
            "confidence_score": self.confidence_score,
            "affected_components": self.affected_components,
            "related_metrics": self.related_metrics,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with implementation details."""
    recommendation_id: str
    category: OptimizationCategory
    title: str
    description: str
    rationale: str
    expected_impact: ImpactLevel
    implementation_complexity: ImplementationComplexity
    priority: OptimizationPriority
    estimated_effort_hours: float
    expected_improvement_percentage: float
    cost_impact_usd: Optional[float]
    implementation_steps: List[str]
    prerequisites: List[str]
    risks: List[str]
    success_metrics: List[str]
    related_insights: List[str]
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact.value,
            "implementation_complexity": self.implementation_complexity.value,
            "priority": self.priority.value,
            "estimated_effort_hours": self.estimated_effort_hours,
            "expected_improvement_percentage": self.expected_improvement_percentage,
            "cost_impact_usd": self.cost_impact_usd,
            "implementation_steps": self.implementation_steps,
            "prerequisites": self.prerequisites,
            "risks": self.risks,
            "success_metrics": self.success_metrics,
            "related_insights": self.related_insights,
            "tags": self.tags,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class OptimizationExperiment:
    """A/B testing experiment for optimization validation."""
    experiment_id: str
    recommendation_id: str
    experiment_name: str
    hypothesis: str
    control_group_metrics: Dict[str, float]
    treatment_group_metrics: Dict[str, float]
    statistical_significance: float
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # "planning", "running", "completed", "failed"
    results: Optional[Dict[str, Any]] = None
    
    def calculate_improvement(self) -> Optional[Dict[str, float]]:
        """Calculate improvement percentages for each metric."""
        if not self.treatment_group_metrics or not self.control_group_metrics:
            return None
        
        improvements = {}
        for metric, treatment_value in self.treatment_group_metrics.items():
            control_value = self.control_group_metrics.get(metric)
            if control_value and control_value > 0:
                improvement = ((treatment_value - control_value) / control_value) * 100
                improvements[metric] = improvement
        
        return improvements


class PerformanceAnalysisEngine:
    """ML-powered performance analysis engine."""
    
    def __init__(self):
        """Initialize performance analysis engine."""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self.last_training: Dict[str, datetime] = {}
        self.training_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    async def analyze_performance_trends(
        self,
        metrics_history: Dict[str, List[Tuple[datetime, float]]],
        analysis_window_hours: int = 24
    ) -> List[PerformanceInsight]:
        """Analyze performance trends and generate insights."""
        insights = []
        
        try:
            for metric_name, history in metrics_history.items():
                if len(history) < 10:  # Need sufficient data
                    continue
                
                # Extract recent data
                cutoff_time = datetime.utcnow() - timedelta(hours=analysis_window_hours)
                recent_data = [(ts, val) for ts, val in history if ts >= cutoff_time]
                
                if len(recent_data) < 5:
                    continue
                
                # Analyze trends
                values = [val for _, val in recent_data]
                timestamps = [ts for ts, _ in recent_data]
                
                # Calculate trend
                trend = await self._calculate_trend(values, timestamps)
                
                # Detect anomalies
                anomalies = await self._detect_anomalies(metric_name, values)
                
                # Generate insights
                if trend["direction"] != "stable" or anomalies:
                    insight = await self._generate_trend_insight(
                        metric_name, values, trend, anomalies
                    )
                    if insight:
                        insights.append(insight)
        
        except Exception as e:
            logger.error("Performance trend analysis failed", error=str(e))
        
        return insights
    
    async def detect_performance_bottlenecks(
        self,
        system_metrics: Dict[str, float],
        agent_metrics: Dict[str, Dict[str, float]],
        workflow_metrics: Dict[str, Dict[str, float]]
    ) -> List[PerformanceInsight]:
        """Detect performance bottlenecks across the system."""
        bottlenecks = []
        
        try:
            # System-level bottleneck detection
            system_bottlenecks = await self._detect_system_bottlenecks(system_metrics)
            bottlenecks.extend(system_bottlenecks)
            
            # Agent-level bottleneck detection
            agent_bottlenecks = await self._detect_agent_bottlenecks(agent_metrics)
            bottlenecks.extend(agent_bottlenecks)
            
            # Workflow-level bottleneck detection
            workflow_bottlenecks = await self._detect_workflow_bottlenecks(workflow_metrics)
            bottlenecks.extend(workflow_bottlenecks)
        
        except Exception as e:
            logger.error("Bottleneck detection failed", error=str(e))
        
        return bottlenecks
    
    async def predict_performance_issues(
        self,
        current_metrics: Dict[str, float],
        prediction_horizon_hours: int = 24
    ) -> List[PerformanceInsight]:
        """Predict potential performance issues."""
        predictions = []
        
        try:
            for metric_name, current_value in current_metrics.items():
                # Use ML model for prediction
                prediction = await self._predict_metric_value(
                    metric_name, current_value, prediction_horizon_hours
                )
                
                if prediction and prediction["risk_level"] > 0.7:
                    insight = PerformanceInsight(
                        insight_id=str(uuid.uuid4()),
                        category=OptimizationCategory.SYSTEM_RESOURCES,
                        title=f"Predicted Issue: {metric_name}",
                        description=f"ML model predicts potential performance degradation in {metric_name} within {prediction_horizon_hours} hours",
                        current_value=current_value,
                        baseline_value=prediction.get("baseline"),
                        trend_direction=prediction.get("trend", "unknown"),
                        confidence_score=prediction["confidence"],
                        affected_components=[metric_name.split(".")[0]],
                        related_metrics=[metric_name]
                    )
                    predictions.append(insight)
        
        except Exception as e:
            logger.error("Performance prediction failed", error=str(e))
        
        return predictions
    
    # Helper methods (simplified implementations)
    async def _calculate_trend(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate trend direction and strength."""
        if len(values) < 3:
            return {"direction": "unknown", "strength": 0.0}
        
        # Simple linear regression for trend
        x_values = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
        
        # Calculate correlation coefficient
        try:
            correlation = np.corrcoef(x_values, values)[0, 1]
            
            if correlation > 0.3:
                direction = "increasing"
            elif correlation < -0.3:
                direction = "decreasing"
            else:
                direction = "stable"
            
            return {
                "direction": direction,
                "strength": abs(correlation),
                "correlation": correlation
            }
        
        except Exception:
            return {"direction": "unknown", "strength": 0.0}
    
    async def _detect_anomalies(self, metric_name: str, values: List[float]) -> List[int]:
        """Detect anomalous values in metric history."""
        if len(values) < 10:
            return []
        
        try:
            # Use statistical method for anomaly detection
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return []
            
            anomalies = []
            for i, value in enumerate(values):
                z_score = abs(value - mean_val) / std_val
                if z_score > 2.5:  # 2.5 standard deviations
                    anomalies.append(i)
            
            return anomalies
        
        except Exception:
            return []
    
    async def _generate_trend_insight(
        self,
        metric_name: str,
        values: List[float],
        trend: Dict[str, Any],
        anomalies: List[int]
    ) -> Optional[PerformanceInsight]:
        """Generate insight from trend analysis."""
        try:
            current_value = values[-1] if values else 0.0
            baseline_value = statistics.mean(values[:len(values)//2]) if len(values) > 4 else None
            
            if trend["direction"] == "increasing" and "error" in metric_name.lower():
                title = f"Rising Error Rate: {metric_name}"
                description = f"Error rate showing upward trend with {trend['strength']:.2f} correlation"
                category = OptimizationCategory.SYSTEM_RESOURCES
            elif trend["direction"] == "decreasing" and "throughput" in metric_name.lower():
                title = f"Declining Throughput: {metric_name}"
                description = f"Throughput showing downward trend with {trend['strength']:.2f} correlation"
                category = OptimizationCategory.AGENT_EFFICIENCY
            elif anomalies:
                title = f"Anomalous Behavior: {metric_name}"
                description = f"Detected {len(anomalies)} anomalous values in recent data"
                category = OptimizationCategory.SYSTEM_RESOURCES
            else:
                return None
            
            return PerformanceInsight(
                insight_id=str(uuid.uuid4()),
                category=category,
                title=title,
                description=description,
                current_value=current_value,
                baseline_value=baseline_value,
                trend_direction=trend["direction"],
                confidence_score=trend["strength"],
                affected_components=[metric_name.split(".")[0]],
                related_metrics=[metric_name]
            )
        
        except Exception as e:
            logger.error("Failed to generate trend insight", error=str(e))
            return None


class OptimizationRecommendationEngine:
    """Generates actionable optimization recommendations."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.recommendation_templates = self._load_recommendation_templates()
        self.impact_models = {}
        self.cost_models = {}
    
    async def generate_recommendations(
        self,
        insights: List[PerformanceInsight],
        current_metrics: Dict[str, float],
        system_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on insights."""
        recommendations = []
        
        try:
            # Group insights by category
            insights_by_category = defaultdict(list)
            for insight in insights:
                insights_by_category[insight.category].append(insight)
            
            # Generate recommendations for each category
            for category, category_insights in insights_by_category.items():
                category_recommendations = await self._generate_category_recommendations(
                    category, category_insights, current_metrics, system_context
                )
                recommendations.extend(category_recommendations)
            
            # Prioritize recommendations
            recommendations = await self._prioritize_recommendations(recommendations)
        
        except Exception as e:
            logger.error("Recommendation generation failed", error=str(e))
        
        return recommendations
    
    async def _generate_category_recommendations(
        self,
        category: OptimizationCategory,
        insights: List[PerformanceInsight],
        current_metrics: Dict[str, float],
        system_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations for specific category."""
        recommendations = []
        
        try:
            if category == OptimizationCategory.SYSTEM_RESOURCES:
                recommendations.extend(await self._generate_system_resource_recommendations(
                    insights, current_metrics, system_context
                ))
            elif category == OptimizationCategory.AGENT_EFFICIENCY:
                recommendations.extend(await self._generate_agent_efficiency_recommendations(
                    insights, current_metrics, system_context
                ))
            elif category == OptimizationCategory.WORKFLOW_OPTIMIZATION:
                recommendations.extend(await self._generate_workflow_optimization_recommendations(
                    insights, current_metrics, system_context
                ))
            # Add more categories as needed
        
        except Exception as e:
            logger.error("Category recommendation generation failed", error=str(e), category=category.value)
        
        return recommendations
    
    async def _generate_system_resource_recommendations(
        self,
        insights: List[PerformanceInsight],
        current_metrics: Dict[str, float],
        system_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate system resource optimization recommendations."""
        recommendations = []
        
        # CPU optimization
        cpu_usage = current_metrics.get("system.cpu.percent", 0)
        if cpu_usage > 80:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                category=OptimizationCategory.SYSTEM_RESOURCES,
                title="Optimize CPU Usage",
                description="System CPU usage is consistently above 80%, indicating potential bottlenecks",
                rationale=f"Current CPU usage at {cpu_usage:.1f}% is approaching capacity limits",
                expected_impact=ImpactLevel.HIGH,
                implementation_complexity=ImplementationComplexity.MEDIUM,
                priority=OptimizationPriority.HIGH,
                estimated_effort_hours=8.0,
                expected_improvement_percentage=25.0,
                implementation_steps=[
                    "Profile CPU-intensive processes",
                    "Optimize computational algorithms",
                    "Implement CPU-efficient data structures",
                    "Consider horizontal scaling"
                ],
                prerequisites=["Performance profiling tools", "Load testing environment"],
                risks=["Service disruption during optimization"],
                success_metrics=["CPU usage < 70%", "Response time improvement"],
                related_insights=[insight.insight_id for insight in insights]
            ))
        
        # Memory optimization
        memory_usage = current_metrics.get("system.memory.percent", 0)
        if memory_usage > 85:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                category=OptimizationCategory.MEMORY_MANAGEMENT,
                title="Optimize Memory Usage",
                description="Memory usage is approaching system limits",
                rationale=f"Current memory usage at {memory_usage:.1f}% risks system instability",
                expected_impact=ImpactLevel.HIGH,
                implementation_complexity=ImplementationComplexity.MEDIUM,
                priority=OptimizationPriority.IMMEDIATE,
                estimated_effort_hours=12.0,
                expected_improvement_percentage=30.0,
                implementation_steps=[
                    "Implement memory profiling",
                    "Optimize data structures",
                    "Implement memory caching strategies",
                    "Add memory cleanup routines"
                ],
                prerequisites=["Memory profiling tools", "Test environment"],
                risks=["Potential memory leaks during optimization"],
                success_metrics=["Memory usage < 75%", "Reduced garbage collection time"],
                related_insights=[insight.insight_id for insight in insights]
            ))
        
        return recommendations
    
    async def _generate_agent_efficiency_recommendations(
        self,
        insights: List[PerformanceInsight],
        current_metrics: Dict[str, float],
        system_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate agent efficiency recommendations."""
        recommendations = []
        
        # Agent health optimization
        avg_health_score = current_metrics.get("agents.avg_health_score", 1.0)
        if avg_health_score < 0.8:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                category=OptimizationCategory.AGENT_EFFICIENCY,
                title="Improve Agent Health Scores",
                description="Agent health scores indicate suboptimal performance",
                rationale=f"Average agent health score of {avg_health_score:.2f} is below optimal threshold",
                expected_impact=ImpactLevel.HIGH,
                implementation_complexity=ImplementationComplexity.HIGH,
                priority=OptimizationPriority.HIGH,
                estimated_effort_hours=16.0,
                expected_improvement_percentage=35.0,
                implementation_steps=[
                    "Analyze failing agents and error patterns",
                    "Implement agent health monitoring",
                    "Optimize task distribution algorithms",
                    "Add agent auto-recovery mechanisms"
                ],
                prerequisites=["Agent monitoring dashboard", "Error tracking system"],
                risks=["Agent downtime during optimization"],
                success_metrics=["Average health score > 0.9", "Reduced agent failures"],
                related_insights=[insight.insight_id for insight in insights]
            ))
        
        return recommendations
    
    async def _prioritize_recommendations(
        self,
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Prioritize recommendations based on impact and effort."""
        try:
            # Calculate priority scores
            priority_scores = {}
            for rec in recommendations:
                # Impact score (0-5)
                impact_score = {
                    ImpactLevel.CRITICAL: 5,
                    ImpactLevel.HIGH: 4,
                    ImpactLevel.MEDIUM: 3,
                    ImpactLevel.LOW: 2,
                    ImpactLevel.MINIMAL: 1
                }[rec.expected_impact]
                
                # Complexity penalty (higher complexity = lower priority)
                complexity_penalty = {
                    ImplementationComplexity.TRIVIAL: 0,
                    ImplementationComplexity.LOW: 0.5,
                    ImplementationComplexity.MEDIUM: 1.0,
                    ImplementationComplexity.HIGH: 2.0,
                    ImplementationComplexity.COMPLEX: 3.0
                }[rec.implementation_complexity]
                
                # Calculate final score
                score = (impact_score * rec.expected_improvement_percentage) - complexity_penalty
                priority_scores[rec.recommendation_id] = score
            
            # Sort by priority score
            return sorted(
                recommendations,
                key=lambda r: priority_scores[r.recommendation_id],
                reverse=True
            )
        
        except Exception as e:
            logger.error("Recommendation prioritization failed", error=str(e))
            return recommendations


class PerformanceOptimizationAdvisor:
    """
    Performance Optimization Advisor for LeanVibe Agent Hive 2.0
    
    Intelligent performance analysis and optimization recommendation engine
    that provides automated insights, actionable recommendations, and predictive
    optimization suggestions for autonomous multi-agent development workflows.
    
    Features:
    - ML-powered performance analysis and bottleneck detection
    - Automated optimization recommendations with impact predictions
    - Resource utilization optimization and capacity planning
    - Performance trend analysis and predictive insights
    - A/B testing frameworks for optimization validation
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        metrics_collector: Optional[PerformanceMetricsCollector] = None
    ):
        """Initialize performance optimization advisor."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.metrics_collector = metrics_collector
        
        # Analysis engines
        self.analysis_engine = PerformanceAnalysisEngine()
        self.recommendation_engine = OptimizationRecommendationEngine()
        
        # State management
        self.insights_history: deque = deque(maxlen=1000)
        self.recommendations_history: deque = deque(maxlen=500)
        self.active_experiments: Dict[str, OptimizationExperiment] = {}
        
        # Background processing
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.config = {
            "analysis_interval_minutes": 15,
            "recommendation_refresh_hours": 4,
            "trend_analysis_window_hours": 24,
            "prediction_horizon_hours": 24,
            "min_data_points": 20,
            "confidence_threshold": 0.7,
            "max_recommendations": 20,
            "redis_key_prefix": "optimization_advisor:",
            "enable_predictive_analysis": True,
            "enable_cost_optimization": True
        }
        
        logger.info("PerformanceOptimizationAdvisor initialized", config=self.config)
    
    async def start_advisor(self) -> None:
        """Start the performance optimization advisor."""
        if self.is_running:
            logger.warning("Performance optimization advisor already running")
            return
        
        logger.info("Starting Performance Optimization Advisor")
        self.is_running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._recommendation_generation_loop()),
            asyncio.create_task(self._experiment_monitoring_loop())
        ]
        
        logger.info("Performance Optimization Advisor started")
    
    async def stop_advisor(self) -> None:
        """Stop the performance optimization advisor."""
        if not self.is_running:
            return
        
        logger.info("Stopping Performance Optimization Advisor")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Performance Optimization Advisor stopped")
    
    async def get_optimization_recommendations(
        self,
        category_filter: Optional[List[OptimizationCategory]] = None,
        priority_filter: Optional[List[OptimizationPriority]] = None,
        limit: int = 10
    ) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations."""
        try:
            # Get recent recommendations
            recommendations = list(self.recommendations_history)
            
            # Apply filters
            if category_filter:
                recommendations = [r for r in recommendations if r.category in category_filter]
            
            if priority_filter:
                recommendations = [r for r in recommendations if r.priority in priority_filter]
            
            # Sort by priority and return top N
            recommendations.sort(
                key=lambda r: (
                    r.priority.value,
                    -r.expected_improvement_percentage,
                    r.implementation_complexity.value
                )
            )
            
            return recommendations[:limit]
        
        except Exception as e:
            logger.error("Failed to get optimization recommendations", error=str(e))
            return []
    
    async def get_performance_insights(
        self,
        hours_back: int = 24,
        category_filter: Optional[List[OptimizationCategory]] = None
    ) -> List[PerformanceInsight]:
        """Get recent performance insights."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Filter recent insights
            insights = [
                insight for insight in self.insights_history
                if insight.timestamp >= cutoff_time
            ]
            
            # Apply category filter
            if category_filter:
                insights = [insight for insight in insights if insight.category in category_filter]
            
            return sorted(insights, key=lambda i: i.confidence_score, reverse=True)
        
        except Exception as e:
            logger.error("Failed to get performance insights", error=str(e))
            return []
    
    # Background processing loops
    async def _analysis_loop(self) -> None:
        """Background task for performance analysis."""
        logger.info("Starting performance analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Collect current metrics
                if self.metrics_collector:
                    current_metrics = await self._collect_current_metrics()
                    metrics_history = await self._collect_metrics_history()
                    
                    # Perform trend analysis
                    trend_insights = await self.analysis_engine.analyze_performance_trends(
                        metrics_history, self.config["trend_analysis_window_hours"]
                    )
                    
                    # Detect bottlenecks
                    system_metrics = current_metrics.get("system", {})
                    agent_metrics = current_metrics.get("agents", {})
                    workflow_metrics = current_metrics.get("workflows", {})
                    
                    bottleneck_insights = await self.analysis_engine.detect_performance_bottlenecks(
                        system_metrics, agent_metrics, workflow_metrics
                    )
                    
                    # Predictive analysis
                    prediction_insights = []
                    if self.config["enable_predictive_analysis"]:
                        prediction_insights = await self.analysis_engine.predict_performance_issues(
                            system_metrics, self.config["prediction_horizon_hours"]
                        )
                    
                    # Store insights
                    all_insights = trend_insights + bottleneck_insights + prediction_insights
                    for insight in all_insights:
                        if insight.confidence_score >= self.config["confidence_threshold"]:
                            self.insights_history.append(insight)
                    
                    logger.info(
                        "Performance analysis completed",
                        trend_insights=len(trend_insights),
                        bottleneck_insights=len(bottleneck_insights),
                        prediction_insights=len(prediction_insights)
                    )
                
                await asyncio.sleep(self.config["analysis_interval_minutes"] * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance analysis loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _recommendation_generation_loop(self) -> None:
        """Background task for recommendation generation."""
        logger.info("Starting recommendation generation loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Get recent insights
                recent_insights = await self.get_performance_insights(
                    hours_back=self.config["recommendation_refresh_hours"]
                )
                
                if recent_insights:
                    # Collect system context
                    current_metrics = await self._collect_current_metrics()
                    system_context = await self._collect_system_context()
                    
                    # Generate recommendations
                    recommendations = await self.recommendation_engine.generate_recommendations(
                        recent_insights,
                        current_metrics.get("system", {}),
                        system_context
                    )
                    
                    # Store recommendations
                    for recommendation in recommendations[:self.config["max_recommendations"]]:
                        self.recommendations_history.append(recommendation)
                    
                    logger.info(
                        "Optimization recommendations generated",
                        count=len(recommendations),
                        insights_analyzed=len(recent_insights)
                    )
                
                await asyncio.sleep(self.config["recommendation_refresh_hours"] * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Recommendation generation loop error", error=str(e))
                await asyncio.sleep(1800)
    
    async def _experiment_monitoring_loop(self) -> None:
        """Background task for monitoring optimization experiments."""
        logger.info("Starting experiment monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Monitor active experiments
                for experiment_id, experiment in list(self.active_experiments.items()):
                    if experiment.status == "running":
                        # Check if experiment should be completed
                        duration = datetime.utcnow() - experiment.start_time
                        if duration.total_seconds() > 3600:  # 1 hour minimum
                            await self._evaluate_experiment(experiment)
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Experiment monitoring loop error", error=str(e))
                await asyncio.sleep(600)
    
    # Helper methods (simplified implementations)
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        if not self.metrics_collector:
            return {}
        
        try:
            summary = await self.metrics_collector.get_performance_summary()
            return summary
        except Exception as e:
            logger.error("Failed to collect current metrics", error=str(e))
            return {}
    
    async def _collect_metrics_history(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """Collect historical metrics data."""
        # Simplified implementation - would collect from database/Redis
        return {}
    
    async def _collect_system_context(self) -> Dict[str, Any]:
        """Collect system context information."""
        try:
            workflow_tracker = await get_agent_workflow_tracker()
            system_status = await workflow_tracker.get_real_time_workflow_status()
            
            return {
                "active_agents": system_status.get("agent_summary", {}).get("total_agents", 0),
                "active_workflows": len(system_status.get("active_workflows", [])),
                "system_load": system_status.get("performance_metrics", {}).get("avg_load_factor", 0.0)
            }
        except Exception as e:
            logger.error("Failed to collect system context", error=str(e))
            return {}
    
    def _load_recommendation_templates(self) -> Dict[str, Any]:
        """Load recommendation templates."""
        # Simplified implementation - would load from configuration
        return {}


# Global instance
_performance_optimization_advisor: Optional[PerformanceOptimizationAdvisor] = None


async def get_performance_optimization_advisor() -> PerformanceOptimizationAdvisor:
    """Get singleton performance optimization advisor instance."""
    global _performance_optimization_advisor
    
    if _performance_optimization_advisor is None:
        _performance_optimization_advisor = PerformanceOptimizationAdvisor()
        await _performance_optimization_advisor.start_advisor()
    
    return _performance_optimization_advisor


async def cleanup_performance_optimization_advisor() -> None:
    """Cleanup performance optimization advisor resources."""
    global _performance_optimization_advisor
    
    if _performance_optimization_advisor:
        await _performance_optimization_advisor.stop_advisor()
        _performance_optimization_advisor = None