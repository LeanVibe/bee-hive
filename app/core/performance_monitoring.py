"""
Advanced Performance Monitoring System for LeanVibe Agent Hive 2.0

Comprehensive performance intelligence platform that integrates real-time monitoring,
predictive analytics, intelligent alerting, and capacity planning for autonomous
development operations.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

import structlog
import redis.asyncio as redis
import numpy as np
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_session
from .redis import get_redis_client
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType, MetricAggregation
from .intelligent_alerting import AlertManager, AlertSeverity, AlertRule, get_alert_manager
from ..models.performance_metric import PerformanceMetric
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory

logger = structlog.get_logger()


class PerformanceCategory(Enum):
    """Categories of performance metrics."""
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


class MetricTrend(Enum):
    """Metric trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class PerformancePrediction:
    """Performance prediction result."""
    metric_name: str
    current_value: float
    predicted_value: float
    prediction_confidence: float
    time_horizon_hours: int
    trend: MetricTrend
    risk_level: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection result."""
    anomaly_id: str
    metric_name: str
    component: str
    current_value: float
    expected_value: float
    deviation_percentage: float
    severity: AlertSeverity
    detected_at: datetime
    description: str
    root_cause_analysis: Dict[str, Any] = field(default_factory=dict)
    suggested_actions: List[str] = field(default_factory=list)


@dataclass
class CapacityPlanningResult:
    """Capacity planning analysis result."""
    resource_type: str
    current_utilization: float
    projected_utilization: float
    capacity_threshold: float
    time_to_threshold_days: Optional[int]
    recommended_actions: List[str]
    cost_projections: Dict[str, float] = field(default_factory=dict)
    scaling_recommendations: Dict[str, Any] = field(default_factory=dict)


class PerformanceIntelligenceEngine:
    """
    Advanced Performance Intelligence Engine for LeanVibe Agent Hive 2.0
    
    Provides comprehensive performance monitoring, predictive analytics, 
    intelligent alerting, and capacity planning capabilities.
    
    Features:
    - Real-time performance metrics collection and analysis
    - Machine learning-based anomaly detection and prediction
    - Intelligent alerting with contextual recommendations
    - Automated capacity planning and scaling recommendations
    - Performance trend analysis and forecasting
    - Multi-dimensional performance correlation analysis
    - Resource optimization recommendations
    - Performance impact assessment for system changes
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional[Callable] = None,
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize the performance intelligence engine."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_session
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # Performance data storage
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.prediction_cache: Dict[str, PerformancePrediction] = {}
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # Analysis engines
        self.trend_analyzer = PerformanceTrendAnalyzer()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.capacity_planner = CapacityPlanningEngine()
        self.correlation_analyzer = PerformanceCorrelationAnalyzer()
        
        # Configuration
        self.config = {
            "collection_interval": 30,  # seconds
            "prediction_horizons": [1, 4, 12, 24],  # hours
            "anomaly_sensitivity": 0.85,
            "capacity_threshold": 0.8,
            "trend_analysis_window": 168,  # hours (1 week)
            "prediction_cache_ttl": 300,  # seconds (5 minutes)
            "max_historical_points": 10000,
            "alert_cooldown_minutes": 15,
            "batch_size": 100
        }
        
        # State management
        self.is_running = False
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        # Thread pool for heavy computations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="perf-intel")
        
        logger.info("Performance Intelligence Engine initialized")
    
    async def start(self) -> None:
        """Start the performance intelligence engine."""
        if self.is_running:
            logger.warning("Performance Intelligence Engine already running")
            return
        
        logger.info("Starting Performance Intelligence Engine")
        self.is_running = True
        
        # Initialize components
        if self.metrics_collector is None:
            self.metrics_collector = PerformanceMetricsCollector(
                redis_client=self.redis_client,
                session_factory=self.session_factory,
                collection_interval=self.config["collection_interval"]
            )
        
        if self.alert_manager is None:
            self.alert_manager = await get_alert_manager()
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._prediction_engine_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._capacity_planning_loop()),
            asyncio.create_task(self._correlation_analysis_loop()),
            asyncio.create_task(self._maintenance_loop())
        ]
        
        logger.info("Performance Intelligence Engine started successfully")
    
    async def stop(self) -> None:
        """Stop the performance intelligence engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping Performance Intelligence Engine")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop metrics collection
        if self.metrics_collector:
            await self.metrics_collector.stop_collection()
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=False)
        
        logger.info("Performance Intelligence Engine stopped")
    
    async def get_real_time_performance_dashboard(
        self, 
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get comprehensive real-time performance dashboard data."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            # Get system performance summary
            system_metrics = await self.metrics_collector.get_performance_summary()
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts() if self.alert_manager else []
            
            # Get performance predictions
            predictions = await self._get_cached_predictions()
            
            # Get recent anomalies
            recent_anomalies = [
                anomaly for anomaly in self.anomaly_history
                if anomaly.detected_at >= start_time
            ]
            
            # Calculate performance scores
            performance_scores = await self._calculate_performance_scores()
            
            # Get capacity utilization
            capacity_status = await self._get_capacity_status()
            
            # Compile dashboard data
            dashboard_data = {
                "timestamp": end_time.isoformat(),
                "time_window_minutes": time_window_minutes,
                "system_health": {
                    "overall_score": performance_scores.get("overall", 0.0),
                    "component_scores": performance_scores,
                    "status": self._determine_system_health_status(performance_scores)
                },
                "real_time_metrics": system_metrics,
                "alerts_summary": {
                    "total_active": len(active_alerts),
                    "critical_count": len([a for a in active_alerts if a.get("severity") == "critical"]),
                    "high_count": len([a for a in active_alerts if a.get("severity") == "high"]),
                    "recent_alerts": active_alerts[:5]
                },
                "performance_predictions": {
                    "next_hour": [p for p in predictions if p.time_horizon_hours == 1],
                    "next_day": [p for p in predictions if p.time_horizon_hours == 24]
                },
                "anomalies": {
                    "count_last_hour": len(recent_anomalies),
                    "recent_anomalies": [asdict(a) for a in recent_anomalies[-5:]]
                },
                "capacity_status": capacity_status,
                "performance_trends": await self._get_performance_trends(time_window_minutes)
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to get real-time performance dashboard", error=str(e))
            return {"error": str(e)}
    
    async def predict_performance_metrics(
        self, 
        metric_names: List[str], 
        horizon_hours: int = 1
    ) -> List[PerformancePrediction]:
        """Predict future values for specified metrics."""
        try:
            predictions = []
            
            for metric_name in metric_names:
                # Get historical data
                historical_data = await self._get_metric_history(metric_name, hours=168)  # 1 week
                
                if len(historical_data) < 10:
                    logger.warning(f"Insufficient data for prediction of {metric_name}")
                    continue
                
                # Perform prediction
                prediction = await self._predict_metric_value(
                    metric_name, 
                    historical_data, 
                    horizon_hours
                )
                
                if prediction:
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error("Failed to predict performance metrics", error=str(e))
            return []
    
    async def detect_performance_anomalies(
        self, 
        time_window_hours: int = 1
    ) -> List[PerformanceAnomaly]:
        """Detect performance anomalies in recent metrics."""
        try:
            anomalies = []
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Get recent performance metrics
            async with self.session_factory() as session:
                query = select(PerformanceMetric).where(
                    PerformanceMetric.timestamp >= start_time
                ).order_by(PerformanceMetric.timestamp.desc())
                
                result = await session.execute(query)
                metrics = result.scalars().all()
            
            # Group metrics by name for anomaly detection
            metrics_by_name = defaultdict(list)
            for metric in metrics:
                metrics_by_name[metric.metric_name].append(metric)
            
            # Detect anomalies for each metric
            for metric_name, metric_list in metrics_by_name.items():
                if len(metric_list) < 5:  # Need minimum data points
                    continue
                
                values = [m.metric_value for m in metric_list]
                timestamps = [m.timestamp for m in metric_list]
                
                anomaly_indices = await self.anomaly_detector.detect_anomalies(values)
                
                for idx in anomaly_indices:
                    if idx < len(metric_list):
                        anomalous_metric = metric_list[idx]
                        
                        # Calculate expected value
                        expected_value = np.median([v for i, v in enumerate(values) if i not in anomaly_indices])
                        deviation = abs(anomalous_metric.metric_value - expected_value) / expected_value * 100
                        
                        # Determine severity based on deviation
                        if deviation > 50:
                            severity = AlertSeverity.CRITICAL
                        elif deviation > 25:
                            severity = AlertSeverity.HIGH
                        elif deviation > 10:
                            severity = AlertSeverity.MEDIUM
                        else:
                            severity = AlertSeverity.LOW
                        
                        anomaly = PerformanceAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            metric_name=metric_name,
                            component=metric_name.split('.')[0] if '.' in metric_name else 'system',
                            current_value=anomalous_metric.metric_value,
                            expected_value=expected_value,
                            deviation_percentage=deviation,
                            severity=severity,
                            detected_at=anomalous_metric.timestamp,
                            description=f"Anomalous value detected for {metric_name}: {anomalous_metric.metric_value:.2f} (expected: {expected_value:.2f})",
                            root_cause_analysis=await self._analyze_anomaly_root_cause(anomalous_metric, metric_list),
                            suggested_actions=await self._generate_anomaly_recommendations(anomaly)
                        )
                        
                        anomalies.append(anomaly)
                        self.anomaly_history.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error("Failed to detect performance anomalies", error=str(e))
            return []
    
    async def generate_capacity_plan(
        self, 
        resource_types: List[str] = None,
        planning_horizon_days: int = 30
    ) -> List[CapacityPlanningResult]:
        """Generate capacity planning recommendations."""
        try:
            if resource_types is None:
                resource_types = ["cpu", "memory", "storage", "network", "agents", "tasks"]
            
            results = []
            
            for resource_type in resource_types:
                # Get resource utilization history
                utilization_history = await self._get_resource_utilization_history(
                    resource_type, 
                    days=30
                )
                
                if len(utilization_history) < 7:  # Need at least a week of data
                    logger.warning(f"Insufficient data for capacity planning of {resource_type}")
                    continue
                
                # Analyze current utilization
                current_utilization = utilization_history[-1]["utilization"] if utilization_history else 0.0
                
                # Project future utilization
                projected_utilization = await self.capacity_planner.project_utilization(
                    utilization_history, 
                    planning_horizon_days
                )
                
                # Calculate time to capacity threshold
                time_to_threshold = await self.capacity_planner.calculate_time_to_threshold(
                    utilization_history, 
                    self.config["capacity_threshold"]
                )
                
                # Generate recommendations
                recommendations = await self._generate_capacity_recommendations(
                    resource_type, 
                    current_utilization, 
                    projected_utilization
                )
                
                # Calculate cost projections
                cost_projections = await self._calculate_capacity_costs(
                    resource_type, 
                    current_utilization, 
                    projected_utilization
                )
                
                result = CapacityPlanningResult(
                    resource_type=resource_type,
                    current_utilization=current_utilization,
                    projected_utilization=projected_utilization,
                    capacity_threshold=self.config["capacity_threshold"],
                    time_to_threshold_days=time_to_threshold,
                    recommended_actions=recommendations,
                    cost_projections=cost_projections,
                    scaling_recommendations=await self._generate_scaling_recommendations(
                        resource_type, 
                        projected_utilization
                    )
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error("Failed to generate capacity plan", error=str(e))
            return []
    
    async def analyze_performance_correlations(
        self, 
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze correlations between different performance metrics."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Get performance metrics for correlation analysis
            async with self.session_factory() as session:
                query = select(PerformanceMetric).where(
                    PerformanceMetric.timestamp >= start_time
                ).order_by(PerformanceMetric.timestamp.asc())
                
                result = await session.execute(query)
                metrics = result.scalars().all()
            
            # Group metrics by name and create time series
            metric_series = defaultdict(list)
            for metric in metrics:
                metric_series[metric.metric_name].append({
                    "timestamp": metric.timestamp,
                    "value": metric.metric_value
                })
            
            # Calculate correlations
            correlations = await self.correlation_analyzer.calculate_correlations(metric_series)
            
            # Identify significant correlations
            significant_correlations = [
                corr for corr in correlations 
                if abs(corr["correlation_coefficient"]) > 0.7
            ]
            
            # Find causal relationships
            causal_relationships = await self.correlation_analyzer.identify_causal_relationships(
                significant_correlations, 
                metric_series
            )
            
            return {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": time_window_hours
                },
                "metrics_analyzed": list(metric_series.keys()),
                "total_correlations": len(correlations),
                "significant_correlations": significant_correlations,
                "causal_relationships": causal_relationships,
                "correlation_summary": {
                    "strong_positive": len([c for c in correlations if c["correlation_coefficient"] > 0.7]),
                    "strong_negative": len([c for c in correlations if c["correlation_coefficient"] < -0.7]),
                    "moderate": len([c for c in correlations if 0.3 < abs(c["correlation_coefficient"]) < 0.7]),
                    "weak": len([c for c in correlations if abs(c["correlation_coefficient"]) < 0.3])
                }
            }
            
        except Exception as e:
            logger.error("Failed to analyze performance correlations", error=str(e))
            return {"error": str(e)}
    
    async def get_performance_optimization_recommendations(
        self, 
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get AI-powered performance optimization recommendations."""
        try:
            recommendations = []
            
            # Get recent performance data
            performance_summary = await self.metrics_collector.get_performance_summary()
            
            if "error" in performance_summary:
                return []
            
            # Analyze system performance
            system_metrics = performance_summary.get("system_metrics", {})
            agent_summary = performance_summary.get("agent_summary", {})
            
            # CPU optimization recommendations
            if "system.cpu.percent" in system_metrics:
                cpu_usage = system_metrics["system.cpu.percent"]
                if cpu_usage > 80:
                    recommendations.append({
                        "category": "cpu_optimization",
                        "priority": "high",
                        "title": "High CPU Usage Detected",
                        "description": f"System CPU usage is at {cpu_usage:.1f}%",
                        "recommendations": [
                            "Consider scaling horizontally by adding more agents",
                            "Review and optimize CPU-intensive tasks",
                            "Implement task load balancing",
                            "Consider upgrading to higher CPU capacity"
                        ],
                        "expected_impact": "15-30% reduction in response times",
                        "implementation_effort": "medium"
                    })
            
            # Memory optimization recommendations
            if "system.memory.rss_mb" in system_metrics:
                memory_usage = system_metrics["system.memory.rss_mb"]
                if memory_usage > 2048:  # 2GB threshold
                    recommendations.append({
                        "category": "memory_optimization",
                        "priority": "medium",
                        "title": "Memory Usage Optimization",
                        "description": f"Memory usage is at {memory_usage:.0f}MB",
                        "recommendations": [
                            "Implement memory-efficient data structures",
                            "Add memory caching strategies",
                            "Consider memory cleanup routines",
                            "Optimize context storage and retrieval"
                        ],
                        "expected_impact": "20-40% reduction in memory usage",
                        "implementation_effort": "medium"
                    })
            
            # Agent performance recommendations
            avg_health_score = agent_summary.get("avg_health_score", 1.0)
            if avg_health_score < 0.8:
                recommendations.append({
                    "category": "agent_optimization",
                    "priority": "high",
                    "title": "Agent Health Optimization",
                    "description": f"Average agent health score is {avg_health_score:.2f}",
                    "recommendations": [
                        "Review failing agents and common error patterns",
                        "Implement agent health monitoring and auto-recovery",
                        "Optimize task distribution algorithms",
                        "Consider agent capacity adjustments"
                    ],
                    "expected_impact": "25-45% improvement in task success rates",
                    "implementation_effort": "high"
                })
            
            # Database performance recommendations
            # This would be based on actual database metrics
            recommendations.append({
                "category": "database_optimization",
                "priority": "low",
                "title": "Database Performance Tuning",
                "description": "Proactive database optimization opportunities",
                "recommendations": [
                    "Review and optimize database indexes",
                    "Implement connection pooling optimization",
                    "Consider database query caching",
                    "Monitor and optimize slow queries"
                ],
                "expected_impact": "10-25% improvement in data access speeds",
                "implementation_effort": "low"
            })
            
            # Filter by component if specified
            if component:
                recommendations = [
                    rec for rec in recommendations 
                    if component.lower() in rec["category"].lower()
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error("Failed to get optimization recommendations", error=str(e))
            return []
    
    # Background task methods
    async def _performance_analysis_loop(self) -> None:
        """Background task for continuous performance analysis."""
        logger.info("Starting performance analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Perform performance analysis
                await self._analyze_current_performance()
                
                # Wait for next cycle
                await asyncio.sleep(self.config["collection_interval"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance analysis loop error", error=str(e))
                await asyncio.sleep(self.config["collection_interval"])
        
        logger.info("Performance analysis loop stopped")
    
    async def _prediction_engine_loop(self) -> None:
        """Background task for performance prediction."""
        logger.info("Starting prediction engine loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update performance predictions
                await self._update_performance_predictions()
                
                # Wait for next cycle (less frequent than monitoring)
                await asyncio.sleep(self.config["prediction_cache_ttl"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Prediction engine loop error", error=str(e))
                await asyncio.sleep(self.config["prediction_cache_ttl"])
        
        logger.info("Prediction engine loop stopped")
    
    async def _anomaly_detection_loop(self) -> None:
        """Background task for anomaly detection."""
        logger.info("Starting anomaly detection loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Detect anomalies in recent performance data
                anomalies = await self.detect_performance_anomalies(time_window_hours=1)
                
                # Generate alerts for critical anomalies
                for anomaly in anomalies:
                    if anomaly.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                        await self._create_anomaly_alert(anomaly)
                
                # Wait for next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Anomaly detection loop error", error=str(e))
                await asyncio.sleep(300)
        
        logger.info("Anomaly detection loop stopped")
    
    async def _capacity_planning_loop(self) -> None:
        """Background task for capacity planning analysis."""
        logger.info("Starting capacity planning loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Generate capacity planning analysis
                capacity_results = await self.generate_capacity_plan()
                
                # Create alerts for capacity issues
                for result in capacity_results:
                    if (result.time_to_threshold_days is not None and 
                        result.time_to_threshold_days < 7):
                        await self._create_capacity_alert(result)
                
                # Store results for dashboard
                await self._store_capacity_analysis(capacity_results)
                
                # Wait for next cycle (run hourly)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Capacity planning loop error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.info("Capacity planning loop stopped")
    
    async def _correlation_analysis_loop(self) -> None:
        """Background task for correlation analysis."""
        logger.info("Starting correlation analysis loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Analyze performance correlations
                correlation_results = await self.analyze_performance_correlations(time_window_hours=24)
                
                # Store results for API access
                await self._store_correlation_analysis(correlation_results)
                
                # Wait for next cycle (run every 4 hours)
                await asyncio.sleep(14400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Correlation analysis loop error", error=str(e))
                await asyncio.sleep(14400)
        
        logger.info("Correlation analysis loop stopped")
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance and cleanup tasks."""
        logger.info("Starting performance intelligence maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Clean old performance data
                await self._cleanup_old_performance_data()
                
                # Update cached predictions
                await self._refresh_prediction_cache()
                
                # Optimize storage and indexes
                await self._optimize_performance_storage()
                
                # Wait for next cycle (run daily)
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Maintenance loop error", error=str(e))
                await asyncio.sleep(86400)
        
        logger.info("Performance intelligence maintenance loop stopped")
    
    # Helper methods (placeholder implementations)
    async def _analyze_current_performance(self) -> None:
        """Analyze current system performance."""
        pass  # Implementation would go here
    
    async def _update_performance_predictions(self) -> None:
        """Update performance predictions cache."""
        pass  # Implementation would go here
    
    async def _get_cached_predictions(self) -> List[PerformancePrediction]:
        """Get cached performance predictions."""
        return list(self.prediction_cache.values())
    
    async def _calculate_performance_scores(self) -> Dict[str, float]:
        """Calculate component performance scores."""
        return {
            "overall": 0.85,
            "system": 0.90,
            "agents": 0.80,
            "database": 0.88,
            "network": 0.92
        }
    
    def _determine_system_health_status(self, scores: Dict[str, float]) -> str:
        """Determine overall system health status."""
        overall_score = scores.get("overall", 0.0)
        
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
            return "fair"
        elif overall_score >= 0.6:
            return "poor"
        else:
            return "critical"
    
    async def _get_capacity_status(self) -> Dict[str, Any]:
        """Get current capacity utilization status."""
        return {
            "cpu": {"utilization": 0.65, "available_capacity": 0.35},
            "memory": {"utilization": 0.45, "available_capacity": 0.55},
            "storage": {"utilization": 0.30, "available_capacity": 0.70},
            "agents": {"active": 5, "capacity": 10, "utilization": 0.50}
        }
    
    async def _get_performance_trends(self, time_window_minutes: int) -> Dict[str, Any]:
        """Get performance trends for dashboard."""
        return {
            "cpu_trend": "stable",
            "memory_trend": "increasing",
            "response_time_trend": "improving",
            "throughput_trend": "stable"
        }
    
    async def _get_metric_history(self, metric_name: str, hours: int) -> List[Dict[str, Any]]:
        """Get historical data for a metric."""
        return []  # Placeholder implementation
    
    async def _predict_metric_value(
        self, 
        metric_name: str, 
        historical_data: List[Dict[str, Any]], 
        horizon_hours: int
    ) -> Optional[PerformancePrediction]:
        """Predict future metric value."""
        # Placeholder implementation
        return None
    
    async def _cleanup_old_performance_data(self) -> None:
        """Clean up old performance data."""
        pass  # Implementation would go here
    
    async def _refresh_prediction_cache(self) -> None:
        """Refresh prediction cache."""
        pass  # Implementation would go here
    
    async def _optimize_performance_storage(self) -> None:
        """Optimize performance data storage."""
        pass  # Implementation would go here


# Placeholder classes for advanced analytics
class PerformanceTrendAnalyzer:
    """Analyzes performance trends and patterns."""
    pass


class AdvancedAnomalyDetector:
    """Advanced machine learning-based anomaly detection."""
    
    async def detect_anomalies(self, values: List[float]) -> List[int]:
        """Detect anomalies in a series of values."""
        # Simple statistical anomaly detection
        if len(values) < 5:
            return []
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_val == 0:
            return []
        
        threshold = 2.0  # 2 standard deviations
        anomaly_indices = []
        
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val
            if z_score > threshold:
                anomaly_indices.append(i)
        
        return anomaly_indices


class CapacityPlanningEngine:
    """Advanced capacity planning and forecasting."""
    
    async def project_utilization(
        self, 
        utilization_history: List[Dict[str, Any]], 
        days: int
    ) -> float:
        """Project future utilization based on historical trends."""
        if not utilization_history:
            return 0.0
        
        # Simple linear projection
        current = utilization_history[-1]["utilization"]
        if len(utilization_history) > 1:
            previous = utilization_history[-2]["utilization"]
            growth_rate = (current - previous) / max(previous, 0.01)
            return min(1.0, current * (1 + growth_rate * days))
        
        return current
    
    async def calculate_time_to_threshold(
        self, 
        utilization_history: List[Dict[str, Any]], 
        threshold: float
    ) -> Optional[int]:
        """Calculate days until capacity threshold is reached."""
        if not utilization_history or len(utilization_history) < 2:
            return None
        
        current = utilization_history[-1]["utilization"]
        if current >= threshold:
            return 0
        
        # Simple linear projection
        previous = utilization_history[-2]["utilization"]
        daily_growth = current - previous
        
        if daily_growth <= 0:
            return None  # Not growing towards threshold
        
        days_to_threshold = (threshold - current) / daily_growth
        return max(0, int(days_to_threshold))


class PerformanceCorrelationAnalyzer:
    """Analyzes correlations between performance metrics."""
    
    async def calculate_correlations(
        self, 
        metric_series: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Calculate correlations between metric series."""
        correlations = []
        metric_names = list(metric_series.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                # Calculate correlation coefficient
                corr_coeff = self._calculate_correlation_coefficient(
                    metric_series[metric1], 
                    metric_series[metric2]
                )
                
                if corr_coeff is not None:
                    correlations.append({
                        "metric1": metric1,
                        "metric2": metric2,
                        "correlation_coefficient": corr_coeff,
                        "strength": self._classify_correlation_strength(corr_coeff)
                    })
        
        return correlations
    
    def _calculate_correlation_coefficient(
        self, 
        series1: List[Dict[str, Any]], 
        series2: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        # Simplified implementation - would need proper time alignment
        if len(series1) < 2 or len(series2) < 2:
            return None
        
        values1 = [point["value"] for point in series1]
        values2 = [point["value"] for point in series2]
        
        # Take minimum length
        min_len = min(len(values1), len(values2))
        values1 = values1[:min_len]
        values2 = values2[:min_len]
        
        try:
            correlation_matrix = np.corrcoef(values1, values2)
            return float(correlation_matrix[0, 1])
        except Exception:
            return None
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    async def identify_causal_relationships(
        self, 
        correlations: List[Dict[str, Any]], 
        metric_series: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Identify potential causal relationships."""
        # Simplified causal analysis
        causal_relationships = []
        
        for correlation in correlations:
            if correlation["strength"] in ["strong", "very_strong"]:
                causal_relationships.append({
                    "cause_metric": correlation["metric1"],
                    "effect_metric": correlation["metric2"],
                    "confidence": abs(correlation["correlation_coefficient"]),
                    "relationship_type": "positive" if correlation["correlation_coefficient"] > 0 else "negative"
                })
        
        return causal_relationships


# Global instance
_performance_intelligence_engine: Optional[PerformanceIntelligenceEngine] = None


async def get_performance_intelligence_engine() -> PerformanceIntelligenceEngine:
    """Get singleton performance intelligence engine instance."""
    global _performance_intelligence_engine
    
    if _performance_intelligence_engine is None:
        _performance_intelligence_engine = PerformanceIntelligenceEngine()
        await _performance_intelligence_engine.start()
    
    return _performance_intelligence_engine


async def cleanup_performance_intelligence_engine() -> None:
    """Cleanup performance intelligence engine resources."""
    global _performance_intelligence_engine
    
    if _performance_intelligence_engine:
        await _performance_intelligence_engine.stop()
        _performance_intelligence_engine = None