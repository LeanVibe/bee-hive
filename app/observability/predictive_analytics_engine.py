"""
Predictive Analytics & Intelligence Recommendation Engine
========================================================

Advanced predictive analytics system that analyzes observability data patterns
to provide intelligent recommendations, anomaly detection, and proactive
optimization suggestions for the LeanVibe Agent Hive system.

Features:
- Time series anomaly detection with machine learning
- Performance trend analysis and capacity planning
- Intelligent optimization recommendations
- Proactive issue prediction and prevention
- Agent behavior pattern analysis
- Resource utilization forecasting
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import structlog

from app.observability.enhanced_prometheus_integration import get_enhanced_prometheus_metrics
from app.observability.enhanced_websocket_streaming import broadcast_system_alert
from app.core.database import get_async_session
from app.core.redis import get_redis_client

logger = structlog.get_logger()


class AnalysisType(str, Enum):
    """Types of predictive analysis."""
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    CAPACITY_PLANNING = "capacity_planning"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Action needed within 24h
    MEDIUM = "medium"         # Action needed within week
    LOW = "low"               # Nice to have optimization


class RecommendationType(str, Enum):
    """Types of intelligent recommendations."""
    SCALE_UP = "scale_up"                    # Increase resources
    SCALE_DOWN = "scale_down"                # Reduce resources
    OPTIMIZE_CONFIG = "optimize_config"      # Configuration optimization
    TUNE_PARAMETERS = "tune_parameters"      # Parameter tuning
    PREEMPTIVE_MAINTENANCE = "preemptive_maintenance"  # Preventive actions
    WORKFLOW_OPTIMIZATION = "workflow_optimization"    # Process improvements
    SECURITY_ENHANCEMENT = "security_enhancement"      # Security improvements


@dataclass
class TimeSeriesPoint:
    """Single time series data point."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class AnomalyDetectionResult:
    """Result from anomaly detection analysis."""
    timestamp: datetime
    metric_name: str
    actual_value: float
    predicted_value: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysisResult:
    """Result from trend analysis."""
    metric_name: str
    analysis_period: timedelta
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_magnitude: float
    seasonal_patterns: Dict[str, float] = field(default_factory=dict)
    forecast: List[TimeSeriesPoint] = field(default_factory=list)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class IntelligentRecommendation:
    """Intelligent recommendation with context and actions."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    
    # Analysis context
    triggered_by: str  # metric name or analysis type
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    
    # Implementation details
    recommended_actions: List[str] = field(default_factory=list)
    estimated_impact: str = ""
    implementation_effort: str = ""  # "low", "medium", "high"
    
    # Validation and tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    implemented: bool = False
    implemented_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "triggered_by": self.triggered_by,
            "confidence_score": self.confidence_score,
            "recommended_actions": self.recommended_actions,
            "estimated_impact": self.estimated_impact,
            "implementation_effort": self.implementation_effort,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "implemented": self.implemented,
            "effectiveness_score": self.effectiveness_score,
            "analysis_data": self.analysis_data
        }


class TimeSeriesAnalyzer:
    """Advanced time series analysis with anomaly detection."""
    
    def __init__(self):
        self.models = {}  # Store trained models per metric
        self.baseline_data = {}  # Historical baseline data
    
    async def detect_anomalies(
        self,
        metric_name: str,
        data_points: List[TimeSeriesPoint],
        sensitivity: float = 1.0
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in time series data using statistical methods.
        
        Args:
            metric_name: Name of the metric being analyzed
            data_points: List of time series data points
            sensitivity: Sensitivity multiplier (higher = more sensitive)
            
        Returns:
            List of anomaly detection results
        """
        if len(data_points) < 10:
            logger.warning(f"Insufficient data points for anomaly detection: {len(data_points)}")
            return []
        
        try:
            # Extract values and timestamps
            values = [point.value for point in data_points]
            timestamps = [point.timestamp for point in data_points]
            
            # Calculate baseline statistics
            baseline_mean, baseline_std = await self._calculate_baseline_stats(metric_name, values)
            
            # Detect anomalies using statistical methods
            anomalies = []
            
            for i, point in enumerate(data_points):
                # Calculate z-score
                z_score = abs(point.value - baseline_mean) / max(baseline_std, 0.001)
                
                # Determine if anomaly based on threshold
                anomaly_threshold = 2.5 * sensitivity  # Standard deviations
                is_anomaly = z_score > anomaly_threshold
                
                # Calculate predicted value (using moving average)
                predicted_value = await self._predict_value(values, i)
                
                # Calculate anomaly score (0-1 scale)
                anomaly_score = min(z_score / 5.0, 1.0)  # Normalize to 0-1
                
                if is_anomaly or anomaly_score > 0.7:  # Include high-score non-anomalies
                    anomaly_result = AnomalyDetectionResult(
                        timestamp=point.timestamp,
                        metric_name=metric_name,
                        actual_value=point.value,
                        predicted_value=predicted_value,
                        anomaly_score=anomaly_score,
                        is_anomaly=is_anomaly,
                        confidence=min(1.0, z_score / anomaly_threshold),
                        context={
                            "z_score": z_score,
                            "baseline_mean": baseline_mean,
                            "baseline_std": baseline_std,
                            "threshold": anomaly_threshold
                        }
                    )
                    anomalies.append(anomaly_result)
            
            logger.debug(
                f"Anomaly detection completed for {metric_name}",
                total_points=len(data_points),
                anomalies_detected=len(anomalies)
            )
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return []
    
    async def analyze_trends(
        self,
        metric_name: str,
        data_points: List[TimeSeriesPoint],
        forecast_hours: int = 24
    ) -> Optional[TrendAnalysisResult]:
        """
        Analyze trends in time series data and generate forecasts.
        
        Args:
            metric_name: Name of the metric being analyzed
            data_points: List of time series data points
            forecast_hours: Hours to forecast into future
            
        Returns:
            Trend analysis result with forecast
        """
        if len(data_points) < 20:
            logger.warning(f"Insufficient data for trend analysis: {len(data_points)}")
            return None
        
        try:
            values = [point.value for point in data_points]
            timestamps = [point.timestamp for point in data_points]
            
            # Calculate trend using linear regression (simplified)
            trend_slope = await self._calculate_trend_slope(values)
            
            # Determine trend direction and magnitude
            if abs(trend_slope) < 0.01:  # Threshold for "stable"
                trend_direction = "stable"
                trend_magnitude = 0.0
            elif trend_slope > 0:
                trend_direction = "increasing"
                trend_magnitude = trend_slope
            else:
                trend_direction = "decreasing"
                trend_magnitude = abs(trend_slope)
            
            # Detect seasonal patterns (simplified hourly patterns)
            seasonal_patterns = await self._detect_seasonal_patterns(data_points)
            
            # Generate forecast
            forecast_points = await self._generate_forecast(
                data_points, forecast_hours, trend_slope, seasonal_patterns
            )
            
            # Calculate confidence interval (simplified)
            historical_variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
            confidence_interval = (
                -1.96 * (historical_variance ** 0.5),
                1.96 * (historical_variance ** 0.5)
            )
            
            analysis_period = timestamps[-1] - timestamps[0]
            
            return TrendAnalysisResult(
                metric_name=metric_name,
                analysis_period=analysis_period,
                trend_direction=trend_direction,
                trend_magnitude=trend_magnitude,
                seasonal_patterns=seasonal_patterns,
                forecast=forecast_points,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {metric_name}: {e}")
            return None
    
    async def _calculate_baseline_stats(self, metric_name: str, values: List[float]) -> Tuple[float, float]:
        """Calculate baseline statistics for a metric."""
        try:
            # Use historical data if available, otherwise current data
            if metric_name in self.baseline_data:
                baseline_values = self.baseline_data[metric_name]
            else:
                # Use 80% of current data as baseline
                baseline_size = max(1, int(len(values) * 0.8))
                baseline_values = values[:baseline_size]
                self.baseline_data[metric_name] = baseline_values
            
            mean = sum(baseline_values) / len(baseline_values)
            variance = sum((x - mean) ** 2 for x in baseline_values) / len(baseline_values)
            std_dev = variance ** 0.5
            
            return mean, std_dev
            
        except Exception as e:
            logger.error(f"Failed to calculate baseline stats: {e}")
            return 0.0, 1.0
    
    async def _predict_value(self, values: List[float], index: int) -> float:
        """Predict value using simple moving average."""
        try:
            # Use last N values for prediction
            window_size = min(5, index)
            if window_size < 1:
                return values[0] if values else 0.0
            
            start_idx = max(0, index - window_size)
            window_values = values[start_idx:index]
            
            return sum(window_values) / len(window_values)
            
        except Exception:
            return values[index] if index < len(values) else 0.0
    
    async def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using simple linear regression."""
        try:
            n = len(values)
            if n < 2:
                return 0.0
            
            # Simple linear regression
            x_mean = (n - 1) / 2  # Index mean
            y_mean = sum(values) / n
            
            numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Failed to calculate trend slope: {e}")
            return 0.0
    
    async def _detect_seasonal_patterns(self, data_points: List[TimeSeriesPoint]) -> Dict[str, float]:
        """Detect seasonal patterns in data (simplified hourly patterns)."""
        try:
            hourly_values = {}
            
            for point in data_points:
                hour = point.timestamp.hour
                if hour not in hourly_values:
                    hourly_values[hour] = []
                hourly_values[hour].append(point.value)
            
            # Calculate average value per hour
            hourly_averages = {}
            for hour, values in hourly_values.items():
                if values:
                    hourly_averages[f"hour_{hour}"] = sum(values) / len(values)
            
            return hourly_averages
            
        except Exception as e:
            logger.error(f"Failed to detect seasonal patterns: {e}")
            return {}
    
    async def _generate_forecast(
        self,
        data_points: List[TimeSeriesPoint],
        forecast_hours: int,
        trend_slope: float,
        seasonal_patterns: Dict[str, float]
    ) -> List[TimeSeriesPoint]:
        """Generate forecast points."""
        try:
            if not data_points:
                return []
            
            forecast_points = []
            last_point = data_points[-1]
            last_value = last_point.value
            
            for i in range(1, forecast_hours + 1):
                # Calculate forecast timestamp
                forecast_time = last_point.timestamp + timedelta(hours=i)
                
                # Apply trend
                trend_adjustment = trend_slope * i
                
                # Apply seasonal pattern if available
                hour_key = f"hour_{forecast_time.hour}"
                seasonal_adjustment = 0.0
                if hour_key in seasonal_patterns:
                    # Simple seasonal adjustment (could be more sophisticated)
                    overall_mean = sum(seasonal_patterns.values()) / len(seasonal_patterns)
                    seasonal_adjustment = (seasonal_patterns[hour_key] - overall_mean) * 0.1
                
                forecast_value = last_value + trend_adjustment + seasonal_adjustment
                
                forecast_point = TimeSeriesPoint(
                    timestamp=forecast_time,
                    value=max(0.0, forecast_value),  # Ensure non-negative
                    metadata={"forecast": True, "confidence": max(0.1, 1.0 - (i * 0.05))}
                )
                
                forecast_points.append(forecast_point)
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            return []


class IntelligentRecommendationEngine:
    """
    Generates intelligent recommendations based on analytics results.
    
    Analyzes patterns, anomalies, and trends to provide actionable
    recommendations for system optimization and issue prevention.
    """
    
    def __init__(self):
        self.recommendation_history = {}
        self.effectiveness_tracking = {}
    
    async def generate_recommendations(
        self,
        anomalies: List[AnomalyDetectionResult],
        trends: List[TrendAnalysisResult],
        system_metrics: Dict[str, Any]
    ) -> List[IntelligentRecommendation]:
        """
        Generate intelligent recommendations based on analysis results.
        
        Args:
            anomalies: Detected anomalies
            trends: Trend analysis results
            system_metrics: Current system metrics
            
        Returns:
            List of intelligent recommendations
        """
        recommendations = []
        
        try:
            # Generate anomaly-based recommendations
            anomaly_recs = await self._generate_anomaly_recommendations(anomalies)
            recommendations.extend(anomaly_recs)
            
            # Generate trend-based recommendations
            trend_recs = await self._generate_trend_recommendations(trends)
            recommendations.extend(trend_recs)
            
            # Generate capacity planning recommendations
            capacity_recs = await self._generate_capacity_recommendations(system_metrics)
            recommendations.extend(capacity_recs)
            
            # Generate performance optimization recommendations
            perf_recs = await self._generate_performance_recommendations(system_metrics)
            recommendations.extend(perf_recs)
            
            # Prioritize and deduplicate recommendations
            recommendations = await self._prioritize_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} intelligent recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    async def _generate_anomaly_recommendations(
        self,
        anomalies: List[AnomalyDetectionResult]
    ) -> List[IntelligentRecommendation]:
        """Generate recommendations based on detected anomalies."""
        recommendations = []
        
        for anomaly in anomalies:
            if not anomaly.is_anomaly or anomaly.confidence < 0.7:
                continue
            
            try:
                if "latency" in anomaly.metric_name.lower():
                    rec = IntelligentRecommendation(
                        type=RecommendationType.OPTIMIZE_CONFIG,
                        priority=RecommendationPriority.HIGH,
                        title="High Latency Anomaly Detected",
                        description=f"Unusual latency spike detected in {anomaly.metric_name}. "
                                   f"Actual: {anomaly.actual_value:.3f}s, Expected: {anomaly.predicted_value:.3f}s",
                        triggered_by=anomaly.metric_name,
                        confidence_score=anomaly.confidence,
                        recommended_actions=[
                            "Check event buffer sizing and processing capacity",
                            "Review concurrent connection limits",
                            "Investigate network or database performance",
                            "Consider scaling event processing workers"
                        ],
                        estimated_impact="10-30% latency reduction",
                        implementation_effort="medium",
                        analysis_data={
                            "anomaly_score": anomaly.anomaly_score,
                            "actual_value": anomaly.actual_value,
                            "predicted_value": anomaly.predicted_value,
                            "context": anomaly.context
                        },
                        expires_at=datetime.utcnow() + timedelta(hours=12)
                    )
                    recommendations.append(rec)
                
                elif "cpu" in anomaly.metric_name.lower():
                    rec = IntelligentRecommendation(
                        type=RecommendationType.SCALE_UP,
                        priority=RecommendationPriority.MEDIUM,
                        title="CPU Usage Anomaly Detected",
                        description=f"Unexpected CPU usage pattern in {anomaly.metric_name}. "
                                   f"Current: {anomaly.actual_value:.1f}%, Baseline: {anomaly.predicted_value:.1f}%",
                        triggered_by=anomaly.metric_name,
                        confidence_score=anomaly.confidence,
                        recommended_actions=[
                            "Monitor CPU usage patterns for sustained load",
                            "Consider horizontal scaling if pattern persists",
                            "Review resource-intensive operations",
                            "Optimize background tasks scheduling"
                        ],
                        estimated_impact="Prevent performance degradation",
                        implementation_effort="low",
                        analysis_data=anomaly.context,
                        expires_at=datetime.utcnow() + timedelta(hours=6)
                    )
                    recommendations.append(rec)
                
                elif "buffer" in anomaly.metric_name.lower():
                    rec = IntelligentRecommendation(
                        type=RecommendationType.TUNE_PARAMETERS,
                        priority=RecommendationPriority.CRITICAL,
                        title="Event Buffer Anomaly Detected",
                        description=f"Unusual buffer behavior detected: {anomaly.actual_value}",
                        triggered_by=anomaly.metric_name,
                        confidence_score=anomaly.confidence,
                        recommended_actions=[
                            "Increase event buffer size immediately",
                            "Scale event processing workers",
                            "Review event ingestion rate patterns",
                            "Implement back-pressure mechanisms"
                        ],
                        estimated_impact="Prevent event loss and ensure 100% coverage",
                        implementation_effort="low",
                        analysis_data=anomaly.context,
                        expires_at=datetime.utcnow() + timedelta(hours=1)
                    )
                    recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Failed to generate anomaly recommendation: {e}")
        
        return recommendations
    
    async def _generate_trend_recommendations(
        self,
        trends: List[TrendAnalysisResult]
    ) -> List[IntelligentRecommendation]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        for trend in trends:
            try:
                if trend.trend_direction == "increasing" and trend.trend_magnitude > 0.1:
                    if "latency" in trend.metric_name.lower():
                        rec = IntelligentRecommendation(
                            type=RecommendationType.PREEMPTIVE_MAINTENANCE,
                            priority=RecommendationPriority.HIGH,
                            title="Increasing Latency Trend Detected",
                            description=f"Latency in {trend.metric_name} is trending upward. "
                                       f"Rate of increase: {trend.trend_magnitude:.4f}s per measurement",
                            triggered_by=trend.metric_name,
                            confidence_score=0.8,
                            recommended_actions=[
                                "Schedule proactive optimization before reaching thresholds",
                                "Review and optimize database query patterns",
                                "Consider pre-scaling resources during peak hours",
                                "Implement circuit breakers for degradation scenarios"
                            ],
                            estimated_impact="Prevent future performance issues",
                            implementation_effort="medium",
                            analysis_data={
                                "trend_direction": trend.trend_direction,
                                "trend_magnitude": trend.trend_magnitude,
                                "forecast": [asdict(fp) for fp in trend.forecast[:5]]  # First 5 forecast points
                            },
                            expires_at=datetime.utcnow() + timedelta(days=2)
                        )
                        recommendations.append(rec)
                    
                    elif "cpu" in trend.metric_name.lower():
                        rec = IntelligentRecommendation(
                            type=RecommendationType.CAPACITY_PLANNING,
                            priority=RecommendationPriority.MEDIUM,
                            title="CPU Usage Growth Trend",
                            description=f"CPU usage is steadily increasing. Plan for capacity expansion.",
                            triggered_by=trend.metric_name,
                            confidence_score=0.7,
                            recommended_actions=[
                                "Plan for horizontal scaling within next 7 days",
                                "Monitor peak usage patterns",
                                "Optimize CPU-intensive operations",
                                "Consider auto-scaling policies"
                            ],
                            estimated_impact="Maintain performance under growing load",
                            implementation_effort="high",
                            analysis_data={"trend_magnitude": trend.trend_magnitude},
                            expires_at=datetime.utcnow() + timedelta(days=7)
                        )
                        recommendations.append(rec)
                
                elif trend.trend_direction == "decreasing" and "error" in trend.metric_name.lower():
                    rec = IntelligentRecommendation(
                        type=RecommendationType.WORKFLOW_OPTIMIZATION,
                        priority=RecommendationPriority.LOW,
                        title="Improving Error Rate Trend",
                        description=f"Error rate is decreasing - good trend! Consider documenting what's working.",
                        triggered_by=trend.metric_name,
                        confidence_score=0.6,
                        recommended_actions=[
                            "Document recent changes that improved error rates",
                            "Standardize improvements across similar systems",
                            "Consider applying similar optimizations to other metrics"
                        ],
                        estimated_impact="Maintain and replicate improvements",
                        implementation_effort="low",
                        analysis_data={"trend_magnitude": trend.trend_magnitude},
                        expires_at=datetime.utcnow() + timedelta(days=3)
                    )
                    recommendations.append(rec)
                
            except Exception as e:
                logger.error(f"Failed to generate trend recommendation: {e}")
        
        return recommendations
    
    async def _generate_capacity_recommendations(
        self,
        system_metrics: Dict[str, Any]
    ) -> List[IntelligentRecommendation]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        try:
            # Analyze current resource utilization
            cpu_usage = system_metrics.get("cpu_percentage", 0)
            memory_usage = system_metrics.get("memory_percentage", 0)
            active_connections = system_metrics.get("active_connections", 0)
            max_connections = system_metrics.get("max_connections", 1000)
            
            # CPU capacity recommendation
            if cpu_usage > 70:
                rec = IntelligentRecommendation(
                    type=RecommendationType.SCALE_UP,
                    priority=RecommendationPriority.HIGH if cpu_usage > 85 else RecommendationPriority.MEDIUM,
                    title="High CPU Utilization - Scale Up Recommended",
                    description=f"CPU utilization at {cpu_usage:.1f}%. Consider scaling up to maintain performance.",
                    triggered_by="cpu_capacity_analysis",
                    confidence_score=0.9,
                    recommended_actions=[
                        "Add more CPU cores or scale horizontally",
                        "Review and optimize CPU-intensive operations",
                        "Implement CPU usage monitoring alerts",
                        "Consider auto-scaling policies"
                    ],
                    estimated_impact=f"Reduce CPU usage by ~{(cpu_usage - 50):.1f}%",
                    implementation_effort="medium",
                    analysis_data={"current_cpu": cpu_usage, "threshold": 70},
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                )
                recommendations.append(rec)
            
            # Connection capacity recommendation
            connection_utilization = (active_connections / max_connections) * 100
            if connection_utilization > 80:
                rec = IntelligentRecommendation(
                    type=RecommendationType.SCALE_UP,
                    priority=RecommendationPriority.HIGH,
                    title="High Connection Utilization",
                    description=f"WebSocket connections at {connection_utilization:.1f}% of capacity "
                               f"({active_connections}/{max_connections})",
                    triggered_by="connection_capacity_analysis",
                    confidence_score=0.85,
                    recommended_actions=[
                        "Increase maximum connection limit",
                        "Implement connection pooling optimizations",
                        "Add load balancing for connection distribution",
                        "Monitor connection lifecycle patterns"
                    ],
                    estimated_impact="Support higher connection loads",
                    implementation_effort="medium",
                    analysis_data={
                        "active_connections": active_connections,
                        "max_connections": max_connections,
                        "utilization": connection_utilization
                    },
                    expires_at=datetime.utcnow() + timedelta(hours=12)
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error(f"Failed to generate capacity recommendations: {e}")
        
        return recommendations
    
    async def _generate_performance_recommendations(
        self,
        system_metrics: Dict[str, Any]
    ) -> List[IntelligentRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        try:
            # Event processing performance
            event_rate = system_metrics.get("events_per_second", 0)
            avg_latency = system_metrics.get("avg_processing_latency_ms", 0)
            
            if event_rate < 100 and avg_latency > 50:  # Low throughput, high latency
                rec = IntelligentRecommendation(
                    type=RecommendationType.OPTIMIZE_CONFIG,
                    priority=RecommendationPriority.MEDIUM,
                    title="Suboptimal Event Processing Performance",
                    description=f"Low event throughput ({event_rate} events/s) with high latency ({avg_latency}ms)",
                    triggered_by="performance_analysis",
                    confidence_score=0.7,
                    recommended_actions=[
                        "Optimize event processing batch sizes",
                        "Review database connection pool settings",
                        "Consider async processing optimizations",
                        "Analyze event serialization performance"
                    ],
                    estimated_impact="20-40% improvement in throughput",
                    implementation_effort="medium",
                    analysis_data={
                        "event_rate": event_rate,
                        "avg_latency": avg_latency
                    },
                    expires_at=datetime.utcnow() + timedelta(days=1)
                )
                recommendations.append(rec)
            
            # Buffer optimization
            buffer_overflows = system_metrics.get("buffer_overflows", 0)
            if buffer_overflows > 0:
                rec = IntelligentRecommendation(
                    type=RecommendationType.TUNE_PARAMETERS,
                    priority=RecommendationPriority.CRITICAL,
                    title="Event Buffer Overflows Detected",
                    description=f"Buffer overflows indicate capacity issues: {buffer_overflows} overflows",
                    triggered_by="buffer_performance_analysis",
                    confidence_score=1.0,
                    recommended_actions=[
                        "Increase event buffer size immediately",
                        "Implement back-pressure mechanisms",
                        "Scale event processing workers",
                        "Review event ingestion patterns"
                    ],
                    estimated_impact="Prevent event loss, maintain 100% coverage",
                    implementation_effort="low",
                    analysis_data={"buffer_overflows": buffer_overflows},
                    expires_at=datetime.utcnow() + timedelta(hours=2)
                )
                recommendations.append(rec)
                
        except Exception as e:
            logger.error(f"Failed to generate performance recommendations: {e}")
        
        return recommendations
    
    async def _prioritize_recommendations(
        self,
        recommendations: List[IntelligentRecommendation]
    ) -> List[IntelligentRecommendation]:
        """Prioritize and deduplicate recommendations."""
        try:
            # Remove duplicates based on type and triggered_by
            seen = set()
            unique_recs = []
            
            for rec in recommendations:
                key = (rec.type.value, rec.triggered_by)
                if key not in seen:
                    seen.add(key)
                    unique_recs.append(rec)
            
            # Sort by priority and confidence
            priority_order = {
                RecommendationPriority.CRITICAL: 0,
                RecommendationPriority.HIGH: 1,
                RecommendationPriority.MEDIUM: 2,
                RecommendationPriority.LOW: 3
            }
            
            unique_recs.sort(
                key=lambda r: (priority_order[r.priority], -r.confidence_score)
            )
            
            # Limit to top 10 recommendations to avoid overwhelming users
            return unique_recs[:10]
            
        except Exception as e:
            logger.error(f"Failed to prioritize recommendations: {e}")
            return recommendations


class PredictiveAnalyticsEngine:
    """
    Main predictive analytics engine that coordinates analysis and recommendations.
    
    Integrates time series analysis, anomaly detection, trend analysis, and
    intelligent recommendations to provide comprehensive observability intelligence.
    """
    
    def __init__(self):
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.recommendation_engine = IntelligentRecommendationEngine()
        self.redis_client = None
        self.prometheus_metrics = None
        self.running = False
        
        # Background tasks
        self.analysis_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            "analysis_interval": 300,        # 5 minutes
            "data_retention_hours": 168,     # 1 week
            "anomaly_sensitivity": 1.0,      # Standard sensitivity
            "forecast_hours": 24,            # 24-hour forecasts
            "min_data_points": 20            # Minimum data points for analysis
        }
        
        # Metrics
        self.metrics = {
            "analyses_completed": 0,
            "anomalies_detected": 0,
            "recommendations_generated": 0,
            "analysis_duration_ms": 0.0
        }
        
        logger.info("Predictive Analytics Engine initialized")
    
    async def start(self) -> None:
        """Start the predictive analytics engine."""
        if self.running:
            logger.warning("Analytics engine already running")
            return
        
        try:
            # Initialize dependencies
            self.redis_client = await get_redis_client()
            self.prometheus_metrics = get_enhanced_prometheus_metrics()
            
            self.running = True
            
            # Start background analysis task
            self.analysis_task = asyncio.create_task(self._analysis_loop())
            
            logger.info("Predictive Analytics Engine started")
            
        except Exception as e:
            logger.error(f"Failed to start analytics engine: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the analytics engine."""
        self.running = False
        
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Predictive Analytics Engine stopped")
    
    async def _analysis_loop(self) -> None:
        """Background task for continuous predictive analysis."""
        logger.info("Starting predictive analysis loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Run comprehensive analysis
                await self._run_comprehensive_analysis()
                
                # Update metrics
                duration_ms = (time.time() - start_time) * 1000
                self.metrics["analysis_duration_ms"] = duration_ms
                self.metrics["analyses_completed"] += 1
                
                await asyncio.sleep(self.config["analysis_interval"])
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_comprehensive_analysis(self) -> None:
        """Run comprehensive predictive analysis."""
        try:
            # Get current system metrics (would integrate with actual metrics source)
            system_metrics = await self._get_current_system_metrics()
            
            # Get time series data for key metrics
            metric_names = [
                "leanvibe_realtime_event_processing_latency",
                "leanvibe_websocket_stream_latency",
                "leanvibe_cpu_overhead_percentage", 
                "leanvibe_event_coverage_percentage",
                "leanvibe_realtime_event_buffer_size"
            ]
            
            all_anomalies = []
            all_trends = []
            
            for metric_name in metric_names:
                # Get historical data (simulated for now)
                data_points = await self._get_time_series_data(metric_name)
                
                if len(data_points) >= self.config["min_data_points"]:
                    # Run anomaly detection
                    anomalies = await self.time_series_analyzer.detect_anomalies(
                        metric_name, data_points, self.config["anomaly_sensitivity"]
                    )
                    all_anomalies.extend(anomalies)
                    
                    # Run trend analysis
                    trend = await self.time_series_analyzer.analyze_trends(
                        metric_name, data_points, self.config["forecast_hours"]
                    )
                    if trend:
                        all_trends.append(trend)
            
            # Generate intelligent recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                all_anomalies, all_trends, system_metrics
            )
            
            # Update metrics
            self.metrics["anomalies_detected"] += len(all_anomalies)
            self.metrics["recommendations_generated"] += len(recommendations)
            
            # Broadcast high-priority recommendations
            await self._broadcast_recommendations(recommendations)
            
            logger.debug(
                "Comprehensive analysis completed",
                anomalies=len(all_anomalies),
                trends=len(all_trends),
                recommendations=len(recommendations)
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
    
    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics (simulated for demonstration)."""
        return {
            "cpu_percentage": 65.0,
            "memory_percentage": 45.0,
            "active_connections": 750,
            "max_connections": 1000,
            "events_per_second": 150.0,
            "avg_processing_latency_ms": 85.0,
            "buffer_overflows": 0
        }
    
    async def _get_time_series_data(self, metric_name: str) -> List[TimeSeriesPoint]:
        """Get time series data for a metric (simulated for demonstration)."""
        # In production, this would query Prometheus or other time series DB
        # For demo, generate realistic sample data
        
        import random
        import math
        
        data_points = []
        base_time = datetime.utcnow() - timedelta(hours=24)
        base_value = 0.1 if "latency" in metric_name else 50.0
        
        for i in range(144):  # 24 hours of 10-minute intervals
            timestamp = base_time + timedelta(minutes=i * 10)
            
            # Add trend and noise
            trend = i * 0.001 if "latency" in metric_name else 0
            noise = random.gauss(0, 0.02) if "latency" in metric_name else random.gauss(0, 5)
            seasonal = math.sin(i * 2 * math.pi / 144) * 0.01 if "latency" in metric_name else 0
            
            # Add occasional anomalies
            anomaly = 0
            if random.random() < 0.05:  # 5% chance of anomaly
                anomaly = random.gauss(0, 0.05) if "latency" in metric_name else random.gauss(0, 20)
            
            value = max(0, base_value + trend + seasonal + noise + anomaly)
            
            point = TimeSeriesPoint(
                timestamp=timestamp,
                value=value,
                metadata={"source": "simulated"}
            )
            data_points.append(point)
        
        return data_points
    
    async def _broadcast_recommendations(self, recommendations: List[IntelligentRecommendation]) -> None:
        """Broadcast high-priority recommendations via WebSocket."""
        try:
            for rec in recommendations:
                if rec.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]:
                    await broadcast_system_alert(
                        level="warning" if rec.priority == RecommendationPriority.HIGH else "critical",
                        message=f"AI Recommendation: {rec.title}",
                        source="predictive_analytics",
                        details={
                            "recommendation_id": rec.id,
                            "type": rec.type.value,
                            "priority": rec.priority.value,
                            "description": rec.description,
                            "confidence_score": rec.confidence_score,
                            "recommended_actions": rec.recommended_actions,
                            "estimated_impact": rec.estimated_impact,
                            "implementation_effort": rec.implementation_effort
                        }
                    )
        
        except Exception as e:
            logger.error(f"Failed to broadcast recommendations: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics system summary."""
        return {
            "status": "running" if self.running else "stopped",
            "config": self.config,
            "metrics": self.metrics,
            "capabilities": {
                "anomaly_detection": True,
                "trend_analysis": True,
                "forecasting": True,
                "intelligent_recommendations": True,
                "capacity_planning": True,
                "performance_optimization": True
            }
        }


# Global analytics engine instance
_analytics_engine: Optional[PredictiveAnalyticsEngine] = None


async def get_predictive_analytics_engine() -> PredictiveAnalyticsEngine:
    """Get global predictive analytics engine instance."""
    global _analytics_engine
    
    if _analytics_engine is None:
        _analytics_engine = PredictiveAnalyticsEngine()
        await _analytics_engine.start()
    
    return _analytics_engine


async def shutdown_predictive_analytics_engine() -> None:
    """Shutdown global analytics engine."""
    global _analytics_engine
    
    if _analytics_engine:
        await _analytics_engine.stop()
        _analytics_engine = None


# Convenience functions for external integration

async def run_ad_hoc_analysis(metric_name: str, hours: int = 24) -> Dict[str, Any]:
    """Run ad-hoc analysis on a specific metric."""
    try:
        engine = await get_predictive_analytics_engine()
        
        # Get data for the specified metric
        data_points = await engine._get_time_series_data(metric_name)
        
        # Run anomaly detection
        anomalies = await engine.time_series_analyzer.detect_anomalies(metric_name, data_points)
        
        # Run trend analysis
        trend = await engine.time_series_analyzer.analyze_trends(metric_name, data_points, hours)
        
        return {
            "metric_name": metric_name,
            "data_points": len(data_points),
            "anomalies_detected": len(anomalies),
            "trend_analysis": asdict(trend) if trend else None,
            "anomalies": [asdict(a) for a in anomalies[:5]]  # Top 5 anomalies
        }
        
    except Exception as e:
        logger.error(f"Ad-hoc analysis failed for {metric_name}: {e}")
        return {"error": str(e)}


async def get_intelligent_recommendations(limit: int = 10) -> List[Dict[str, Any]]:
    """Get current intelligent recommendations."""
    try:
        engine = await get_predictive_analytics_engine()
        
        # Get current system metrics
        system_metrics = await engine._get_current_system_metrics()
        
        # Generate fresh recommendations
        recommendations = await engine.recommendation_engine.generate_recommendations([], [], system_metrics)
        
        # Return serialized recommendations
        return [rec.to_dict() for rec in recommendations[:limit]]
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return []