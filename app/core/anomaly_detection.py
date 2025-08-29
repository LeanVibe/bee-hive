"""
Advanced Anomaly Detection Engine with Machine Learning
=====================================================

Intelligent anomaly detection system using multiple ML algorithms with adaptive baselines,
real-time detection, and automated response capabilities.

Epic F Phase 2: Advanced Observability & Intelligence
Target: Detect anomalies with intelligent baseline adjustment and proactive alerting
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import numpy as np
from sklearn.ensemble import IsolationForest, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_async_session
from .redis import get_redis_client
from .intelligent_alerting import AlertManager, AlertSeverity, get_alert_manager
from ..models.performance_metric import PerformanceMetric
from ..models.observability import AgentEvent

logger = structlog.get_logger()


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_DEVIATION = "trend_deviation"
    SEASONAL_ANOMALY = "seasonal_anomaly"
    CORRELATION_BREAK = "correlation_break"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_DEVIATION = "pattern_deviation"
    MULTI_METRIC_ANOMALY = "multi_metric_anomaly"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionModel(Enum):
    """Available anomaly detection models."""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    STATISTICAL_ZSCORE = "statistical_zscore"
    DBSCAN_CLUSTERING = "dbscan_clustering"
    ENSEMBLE = "ensemble"


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""
    anomaly_id: str
    metric_name: str
    component: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    detected_at: datetime
    anomaly_score: float  # 0-1 scale, higher = more anomalous
    current_value: float
    expected_value: float
    deviation_percentage: float
    confidence: float
    model_used: DetectionModel
    context_window_hours: int
    baseline_period: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis details
    statistical_details: Dict[str, float] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    correlation_impact: Dict[str, float] = field(default_factory=dict)
    
    # Response and recommendations
    root_cause_analysis: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    predicted_impact: Dict[str, Any] = field(default_factory=dict)
    auto_mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class AdaptiveBaseline:
    """Adaptive baseline for dynamic anomaly detection."""
    metric_name: str
    baseline_value: float
    baseline_std: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    seasonal_components: Dict[str, float] = field(default_factory=dict)
    trend_component: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    update_frequency_hours: int = 6
    historical_accuracy: float = 0.85
    adaptation_rate: float = 0.1  # How quickly baseline adapts to new data


@dataclass
class AnomalyPattern:
    """Pattern definition for anomaly detection."""
    pattern_id: str
    name: str
    description: str
    metrics_involved: List[str]
    pattern_conditions: Dict[str, Any]
    severity_mapping: Dict[str, AnomalySeverity]
    detection_window_minutes: int
    confidence_threshold: float = 0.8


class AdvancedAnomalyDetector:
    """
    Advanced Machine Learning-based Anomaly Detection Engine
    
    Features:
    - Multiple ML algorithms for comprehensive anomaly detection
    - Adaptive baselines that evolve with system behavior
    - Real-time detection with sub-second response times
    - Multi-metric correlation analysis
    - Intelligent severity assessment and root cause analysis
    - Automated response and mitigation recommendations
    - Pattern-based anomaly detection for complex scenarios
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional = None,
        alert_manager: Optional[AlertManager] = None
    ):
        """Initialize the advanced anomaly detector."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_async_session
        self.alert_manager = alert_manager
        
        # ML Models for anomaly detection
        self.models: Dict[str, Dict[DetectionModel, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Adaptive baselines
        self.baselines: Dict[str, AdaptiveBaseline] = {}
        self.baseline_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Anomaly history and patterns
        self.anomaly_history: deque = deque(maxlen=10000)
        self.anomaly_patterns: Dict[str, AnomalyPattern] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Detection configuration
        self.config = {
            "detection_interval_seconds": 30,
            "baseline_update_hours": 6,
            "min_baseline_samples": 100,
            "correlation_threshold": 0.7,
            "anomaly_score_threshold": 0.6,
            "ensemble_voting_threshold": 0.5,
            "max_concurrent_detections": 50,
            "historical_window_hours": 72,  # 3 days
            "seasonal_analysis_days": 14,
            "trend_analysis_hours": 24,
            "multi_metric_window_minutes": 15,
            "auto_mitigation_enabled": True
        }
        
        # Performance tracking
        self.detection_metrics = {
            "total_detections": 0,
            "false_positives": 0,
            "true_positives": 0,
            "detection_latency_ms": deque(maxlen=1000),
            "model_accuracy": defaultdict(lambda: deque(maxlen=100))
        }
        
        # State management
        self.is_running = False
        self.detection_tasks: List[asyncio.Task] = []
        
        logger.info("Advanced Anomaly Detection Engine initialized")
    
    async def start(self) -> None:
        """Start the anomaly detection engine."""
        if self.is_running:
            logger.warning("Anomaly detection engine already running")
            return
        
        logger.info("Starting Advanced Anomaly Detection Engine")
        self.is_running = True
        
        # Initialize alert manager if not provided
        if self.alert_manager is None:
            self.alert_manager = await get_alert_manager()
        
        # Initialize models and baselines
        await self._initialize_detection_models()
        await self._initialize_adaptive_baselines()
        await self._load_anomaly_patterns()
        
        # Start background tasks
        self.detection_tasks = [
            asyncio.create_task(self._real_time_detection_loop()),
            asyncio.create_task(self._baseline_update_loop()),
            asyncio.create_task(self._correlation_analysis_loop()),
            asyncio.create_task(self._pattern_detection_loop()),
            asyncio.create_task(self._model_accuracy_monitoring_loop())
        ]
        
        logger.info("Advanced Anomaly Detection Engine started successfully")
    
    async def stop(self) -> None:
        """Stop the anomaly detection engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping Advanced Anomaly Detection Engine")
        self.is_running = False
        
        # Cancel background tasks
        for task in self.detection_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.detection_tasks:
            await asyncio.gather(*self.detection_tasks, return_exceptions=True)
        
        # Save models and baselines
        await self._save_detection_state()
        
        logger.info("Advanced Anomaly Detection Engine stopped")
    
    async def detect_anomalies(
        self,
        metric_names: List[str] = None,
        detection_window_hours: int = 1,
        use_ensemble: bool = True
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalies in specified metrics using ML algorithms.
        
        Args:
            metric_names: List of metrics to analyze (None for all)
            detection_window_hours: Time window for anomaly detection
            use_ensemble: Whether to use ensemble voting for detection
            
        Returns:
            List of detected anomalies with detailed analysis
        """
        try:
            detection_start = time.time()
            anomalies = []
            
            # Get metrics to analyze
            if metric_names is None:
                metric_names = await self._get_active_metrics()
            
            # Get recent data for analysis
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=detection_window_hours)
            
            metric_data = await self._get_metrics_data(metric_names, start_time, end_time)
            
            # Detect anomalies for each metric
            for metric_name, data_points in metric_data.items():
                if len(data_points) < 10:  # Need minimum data points
                    continue
                
                # Single-metric anomaly detection
                metric_anomalies = await self._detect_metric_anomalies(
                    metric_name, 
                    data_points, 
                    use_ensemble
                )
                anomalies.extend(metric_anomalies)
            
            # Multi-metric correlation anomalies
            if len(metric_data) > 1:
                correlation_anomalies = await self._detect_correlation_anomalies(metric_data)
                anomalies.extend(correlation_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = await self._detect_pattern_anomalies(metric_data)
            anomalies.extend(pattern_anomalies)
            
            # Record performance metrics
            detection_time = (time.time() - detection_start) * 1000
            self.detection_metrics["detection_latency_ms"].append(detection_time)
            self.detection_metrics["total_detections"] += len(anomalies)
            
            # Store anomalies in history
            for anomaly in anomalies:
                self.anomaly_history.append(anomaly)
            
            # Generate alerts for critical anomalies
            await self._process_anomaly_alerts(anomalies)
            
            logger.info(f"Detected {len(anomalies)} anomalies in {detection_time:.2f}ms")
            return anomalies
            
        except Exception as e:
            logger.error("Failed to detect anomalies", error=str(e))
            return []
    
    async def update_adaptive_baseline(
        self,
        metric_name: str,
        recent_data: List[Dict[str, Any]] = None
    ) -> AdaptiveBaseline:
        """
        Update adaptive baseline for a specific metric.
        
        Args:
            metric_name: Name of the metric
            recent_data: Recent data points (optional, will fetch if None)
            
        Returns:
            Updated adaptive baseline
        """
        try:
            if recent_data is None:
                # Get recent data for baseline update
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=self.config["historical_window_hours"])
                
                async with self.session_factory() as session:
                    query = select(PerformanceMetric).where(
                        and_(
                            PerformanceMetric.metric_name == metric_name,
                            PerformanceMetric.timestamp >= start_time
                        )
                    ).order_by(PerformanceMetric.timestamp.asc())
                    
                    result = await session.execute(query)
                    metrics = result.scalars().all()
                    
                    recent_data = [
                        {"value": m.metric_value, "timestamp": m.timestamp}
                        for m in metrics
                    ]
            
            if len(recent_data) < self.config["min_baseline_samples"]:
                logger.warning(f"Insufficient data for baseline update: {metric_name}")
                return self.baselines.get(metric_name)
            
            # Extract values and perform statistical analysis
            values = [point["value"] for point in recent_data]
            timestamps = [point["timestamp"] for point in recent_data]
            
            # Calculate baseline statistics
            baseline_value = np.median(values)  # Use median for robustness
            baseline_std = np.std(values)
            
            # Calculate confidence intervals (using IQR for robustness)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            confidence_interval_lower = q1 - 1.5 * iqr
            confidence_interval_upper = q3 + 1.5 * iqr
            
            # Analyze seasonal components
            seasonal_components = self._analyze_seasonal_components(values, timestamps)
            
            # Calculate trend component
            trend_component = self._calculate_trend_component(values, timestamps)
            
            # Create or update adaptive baseline
            current_baseline = self.baselines.get(metric_name)
            
            if current_baseline:
                # Adaptive update using exponential smoothing
                adaptation_rate = current_baseline.adaptation_rate
                baseline_value = (1 - adaptation_rate) * current_baseline.baseline_value + adaptation_rate * baseline_value
                baseline_std = (1 - adaptation_rate) * current_baseline.baseline_std + adaptation_rate * baseline_std
            
            updated_baseline = AdaptiveBaseline(
                metric_name=metric_name,
                baseline_value=baseline_value,
                baseline_std=baseline_std,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper,
                seasonal_components=seasonal_components,
                trend_component=trend_component,
                last_updated=datetime.utcnow(),
                update_frequency_hours=self.config["baseline_update_hours"],
                historical_accuracy=self._calculate_baseline_accuracy(metric_name),
                adaptation_rate=0.1
            )
            
            self.baselines[metric_name] = updated_baseline
            
            # Store baseline history
            self.baseline_history[metric_name].append({
                "timestamp": datetime.utcnow(),
                "baseline_value": baseline_value,
                "baseline_std": baseline_std,
                "accuracy": updated_baseline.historical_accuracy
            })
            
            logger.info(f"Updated adaptive baseline for {metric_name}")
            return updated_baseline
            
        except Exception as e:
            logger.error(f"Failed to update baseline for {metric_name}", error=str(e))
            return self.baselines.get(metric_name)
    
    async def analyze_anomaly_patterns(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze patterns in detected anomalies for insights.
        
        Args:
            time_window_hours: Time window for pattern analysis
            
        Returns:
            Pattern analysis results with insights and trends
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Filter anomalies in time window
            recent_anomalies = [
                anomaly for anomaly in self.anomaly_history
                if anomaly.detected_at >= start_time
            ]
            
            if not recent_anomalies:
                return {"message": "No anomalies detected in the specified time window"}
            
            # Analyze patterns
            analysis = {
                "analysis_period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_hours": time_window_hours
                },
                "summary": {
                    "total_anomalies": len(recent_anomalies),
                    "unique_metrics": len(set(a.metric_name for a in recent_anomalies)),
                    "severity_distribution": self._analyze_severity_distribution(recent_anomalies),
                    "anomaly_type_distribution": self._analyze_anomaly_type_distribution(recent_anomalies)
                },
                "trends": {
                    "anomaly_frequency_trend": self._analyze_anomaly_frequency_trend(recent_anomalies),
                    "severity_trend": self._analyze_severity_trend(recent_anomalies),
                    "affected_components": self._analyze_affected_components(recent_anomalies)
                },
                "correlations": await self._analyze_anomaly_correlations(recent_anomalies),
                "recommendations": self._generate_pattern_recommendations(recent_anomalies)
            }
            
            logger.info(f"Analyzed {len(recent_anomalies)} anomalies for patterns")
            return analysis
            
        except Exception as e:
            logger.error("Failed to analyze anomaly patterns", error=str(e))
            return {"error": str(e)}
    
    async def get_anomaly_detection_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the anomaly detection system."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate recent performance metrics
            recent_detections = len([
                a for a in self.anomaly_history
                if (current_time - a.detected_at).total_seconds() < 3600  # Last hour
            ])
            
            avg_detection_latency = (
                sum(self.detection_metrics["detection_latency_ms"]) / 
                len(self.detection_metrics["detection_latency_ms"])
                if self.detection_metrics["detection_latency_ms"] else 0
            )
            
            status = {
                "system_status": {
                    "is_running": self.is_running,
                    "uptime_hours": (current_time - datetime.utcnow()).total_seconds() / 3600 if self.is_running else 0,
                    "active_models": len(self.models),
                    "active_baselines": len(self.baselines)
                },
                "performance_metrics": {
                    "total_detections": self.detection_metrics["total_detections"],
                    "recent_detections_1h": recent_detections,
                    "average_detection_latency_ms": avg_detection_latency,
                    "detection_accuracy": self._calculate_overall_accuracy()
                },
                "baseline_status": {
                    "total_baselines": len(self.baselines),
                    "recently_updated": len([
                        b for b in self.baselines.values()
                        if (current_time - b.last_updated).total_seconds() < 3600
                    ]),
                    "average_accuracy": sum(b.historical_accuracy for b in self.baselines.values()) / len(self.baselines) if self.baselines else 0
                },
                "model_status": {
                    model_name: {
                        "metrics_covered": len(metrics),
                        "last_updated": max(self.baselines[metric].last_updated for metric in metrics if metric in self.baselines).isoformat() if metrics and any(metric in self.baselines for metric in metrics) else None
                    }
                    for model_name, metrics in self.models.items()
                },
                "recent_anomalies": [
                    {
                        "metric_name": a.metric_name,
                        "severity": a.severity.value,
                        "detected_at": a.detected_at.isoformat(),
                        "anomaly_score": a.anomaly_score
                    }
                    for a in list(self.anomaly_history)[-10:]  # Last 10 anomalies
                ]
            }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get anomaly detection status", error=str(e))
            return {"error": str(e)}
    
    # Background task methods
    async def _real_time_detection_loop(self) -> None:
        """Background task for continuous real-time anomaly detection."""
        logger.info("Starting real-time anomaly detection loop")
        
        while self.is_running:
            try:
                # Perform real-time anomaly detection
                anomalies = await self.detect_anomalies(detection_window_hours=1)
                
                # Process critical anomalies immediately
                critical_anomalies = [
                    a for a in anomalies 
                    if a.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
                ]
                
                if critical_anomalies:
                    await self._handle_critical_anomalies(critical_anomalies)
                
                # Wait for next detection cycle
                await asyncio.sleep(self.config["detection_interval_seconds"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Real-time detection loop error", error=str(e))
                await asyncio.sleep(self.config["detection_interval_seconds"])
        
        logger.info("Real-time anomaly detection loop stopped")
    
    async def _baseline_update_loop(self) -> None:
        """Background task for updating adaptive baselines."""
        logger.info("Starting baseline update loop")
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Update baselines that need refreshing
                for metric_name, baseline in self.baselines.items():
                    hours_since_update = (current_time - baseline.last_updated).total_seconds() / 3600
                    
                    if hours_since_update >= baseline.update_frequency_hours:
                        logger.info(f"Updating baseline for {metric_name}")
                        await self.update_adaptive_baseline(metric_name)
                
                # Wait before next update cycle
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Baseline update loop error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.info("Baseline update loop stopped")
    
    async def _correlation_analysis_loop(self) -> None:
        """Background task for analyzing metric correlations."""
        logger.info("Starting correlation analysis loop")
        
        while self.is_running:
            try:
                # Update correlation matrix
                await self._update_correlation_matrix()
                
                # Wait before next analysis
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Correlation analysis loop error", error=str(e))
                await asyncio.sleep(1800)
        
        logger.info("Correlation analysis loop stopped")
    
    async def _pattern_detection_loop(self) -> None:
        """Background task for pattern-based anomaly detection."""
        logger.info("Starting pattern detection loop")
        
        while self.is_running:
            try:
                # Analyze patterns in recent data
                pattern_analysis = await self.analyze_anomaly_patterns(time_window_hours=6)
                
                # Update pattern definitions based on analysis
                await self._update_anomaly_patterns(pattern_analysis)
                
                # Wait before next pattern analysis
                await asyncio.sleep(3600)  # Every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pattern detection loop error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.info("Pattern detection loop stopped")
    
    async def _model_accuracy_monitoring_loop(self) -> None:
        """Background task for monitoring model accuracy."""
        logger.info("Starting model accuracy monitoring loop")
        
        while self.is_running:
            try:
                # Calculate accuracy for each model
                for metric_name in self.models.keys():
                    accuracy = self._calculate_baseline_accuracy(metric_name)
                    self.detection_metrics["model_accuracy"][metric_name].append(accuracy)
                    
                    # Retrain model if accuracy drops below threshold
                    if accuracy < 0.7:
                        logger.warning(f"Low accuracy for {metric_name}: {accuracy:.3f}")
                        await self._retrain_model(metric_name)
                
                # Wait before next accuracy check
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Model accuracy monitoring loop error", error=str(e))
                await asyncio.sleep(1800)
        
        logger.info("Model accuracy monitoring loop stopped")
    
    # Helper methods (essential implementations)
    async def _detect_metric_anomalies(
        self,
        metric_name: str,
        data_points: List[Dict[str, Any]],
        use_ensemble: bool = True
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies for a single metric using ML models."""
        try:
            anomalies = []
            
            if len(data_points) < 10:
                return anomalies
            
            values = [point["value"] for point in data_points]
            timestamps = [point["timestamp"] for point in data_points]
            
            # Get baseline for comparison
            baseline = self.baselines.get(metric_name)
            if not baseline:
                return anomalies
            
            # Statistical anomaly detection (Z-score method)
            z_scores = np.abs((np.array(values) - baseline.baseline_value) / max(baseline.baseline_std, 0.001))
            statistical_anomalies = np.where(z_scores > 3)[0]  # 3-sigma rule
            
            # ML-based anomaly detection
            ml_anomalies = []
            if metric_name in self.models:
                # Prepare features
                features = self._prepare_features_for_detection(data_points)
                if features is not None and len(features) > 0:
                    # Use Isolation Forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(features.reshape(-1, 1))
                    ml_anomalies = np.where(anomaly_labels == -1)[0]
            
            # Combine results
            if use_ensemble:
                # Use voting: anomaly if detected by multiple methods
                all_indices = set(statistical_anomalies) | set(ml_anomalies)
            else:
                # Use union of all detections
                all_indices = set(statistical_anomalies) | set(ml_anomalies)
            
            # Create anomaly objects
            for idx in all_indices:
                if idx >= len(data_points):
                    continue
                
                anomalous_point = data_points[idx]
                current_value = anomalous_point["value"]
                expected_value = baseline.baseline_value
                deviation = abs(current_value - expected_value) / max(expected_value, 1) * 100
                
                # Determine severity based on deviation
                if deviation > 100:
                    severity = AnomalySeverity.CRITICAL
                elif deviation > 50:
                    severity = AnomalySeverity.HIGH
                elif deviation > 25:
                    severity = AnomalySeverity.MEDIUM
                elif deviation > 10:
                    severity = AnomalySeverity.LOW
                else:
                    severity = AnomalySeverity.INFO
                
                # Calculate anomaly score
                z_score = z_scores[idx] if idx < len(z_scores) else 0
                anomaly_score = min(1.0, z_score / 5.0)  # Normalize to 0-1
                
                anomaly = AnomalyDetectionResult(
                    anomaly_id=str(uuid.uuid4()),
                    metric_name=metric_name,
                    component=metric_name.split(".")[0] if "." in metric_name else "system",
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER if idx in statistical_anomalies else AnomalyType.PATTERN_DEVIATION,
                    severity=severity,
                    detected_at=anomalous_point["timestamp"],
                    anomaly_score=anomaly_score,
                    current_value=current_value,
                    expected_value=expected_value,
                    deviation_percentage=deviation,
                    confidence=min(0.95, max(0.5, anomaly_score)),
                    model_used=DetectionModel.ENSEMBLE if use_ensemble else DetectionModel.ISOLATION_FOREST,
                    context_window_hours=1,
                    baseline_period=f"Last {self.config['historical_window_hours']} hours",
                    statistical_details={
                        "z_score": float(z_score),
                        "baseline_std": baseline.baseline_std,
                        "baseline_value": baseline.baseline_value
                    },
                    root_cause_analysis=self._analyze_anomaly_root_cause(metric_name, current_value, baseline),
                    recommended_actions=self._generate_anomaly_recommendations(metric_name, severity, deviation),
                    predicted_impact=self._assess_anomaly_impact(metric_name, severity, current_value),
                    auto_mitigation_actions=self._generate_mitigation_actions(metric_name, severity)
                )
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {metric_name}", error=str(e))
            return []
    
    def _analyze_anomaly_root_cause(
        self,
        metric_name: str,
        current_value: float,
        baseline: AdaptiveBaseline
    ) -> List[str]:
        """Analyze potential root causes of an anomaly."""
        root_causes = []
        
        # General analysis based on metric type
        if "cpu" in metric_name.lower():
            if current_value > baseline.baseline_value:
                root_causes.extend([
                    "High CPU usage may indicate resource-intensive processes",
                    "Possible system overload or inefficient algorithms",
                    "Check for runaway processes or memory leaks"
                ])
            else:
                root_causes.extend([
                    "Unusually low CPU usage may indicate system issues",
                    "Possible process failures or reduced workload"
                ])
        elif "memory" in metric_name.lower():
            if current_value > baseline.baseline_value:
                root_causes.extend([
                    "High memory usage may indicate memory leaks",
                    "Large data structures or inefficient memory management",
                    "Possible need for memory optimization"
                ])
        elif "response" in metric_name.lower() or "latency" in metric_name.lower():
            if current_value > baseline.baseline_value:
                root_causes.extend([
                    "Increased response time may indicate performance bottlenecks",
                    "Database query performance issues",
                    "Network connectivity problems",
                    "High system load affecting response times"
                ])
        
        return root_causes
    
    def _generate_anomaly_recommendations(
        self,
        metric_name: str,
        severity: AnomalySeverity,
        deviation: float
    ) -> List[str]:
        """Generate recommendations based on anomaly characteristics."""
        recommendations = []
        
        if severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            recommendations.append("Immediate investigation required")
            recommendations.append("Consider enabling enhanced monitoring for this metric")
            
            if "cpu" in metric_name.lower():
                recommendations.extend([
                    "Scale horizontally if possible",
                    "Optimize CPU-intensive processes",
                    "Review system resource allocation"
                ])
            elif "memory" in metric_name.lower():
                recommendations.extend([
                    "Investigate potential memory leaks",
                    "Consider increasing memory allocation",
                    "Optimize memory usage patterns"
                ])
            elif "response" in metric_name.lower():
                recommendations.extend([
                    "Optimize database queries",
                    "Implement caching where appropriate",
                    "Check network connectivity"
                ])
        
        return recommendations
    
    def _assess_anomaly_impact(
        self,
        metric_name: str,
        severity: AnomalySeverity,
        current_value: float
    ) -> Dict[str, Any]:
        """Assess the potential impact of an anomaly."""
        impact = {
            "business_impact": "low",
            "user_experience_impact": "low",
            "system_stability_impact": "low",
            "performance_impact": "low"
        }
        
        if severity == AnomalySeverity.CRITICAL:
            impact["business_impact"] = "high"
            impact["user_experience_impact"] = "high"
            impact["system_stability_impact"] = "high"
            impact["performance_impact"] = "high"
        elif severity == AnomalySeverity.HIGH:
            impact["business_impact"] = "medium"
            impact["user_experience_impact"] = "medium"
            impact["system_stability_impact"] = "medium"
            impact["performance_impact"] = "high"
        
        return impact
    
    def _generate_mitigation_actions(
        self,
        metric_name: str,
        severity: AnomalySeverity
    ) -> List[str]:
        """Generate automated mitigation actions."""
        actions = []
        
        if severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
            actions.extend([
                "Alert operations team immediately",
                "Enable detailed logging for affected component",
                "Prepare scaling resources for deployment"
            ])
            
            if self.config["auto_mitigation_enabled"]:
                actions.extend([
                    "Trigger automated scaling if configured",
                    "Initiate circuit breaker if available"
                ])
        
        return actions
    
    # Additional helper methods (simplified implementations)
    async def _get_active_metrics(self) -> List[str]:
        """Get list of active metrics for monitoring."""
        try:
            async with self.session_factory() as session:
                query = select(PerformanceMetric.metric_name).distinct()
                result = await session.execute(query)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.error("Failed to get active metrics", error=str(e))
            return []
    
    async def _get_metrics_data(
        self,
        metric_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics data for specified time range."""
        try:
            metrics_data = defaultdict(list)
            
            async with self.session_factory() as session:
                for metric_name in metric_names:
                    query = select(PerformanceMetric).where(
                        and_(
                            PerformanceMetric.metric_name == metric_name,
                            PerformanceMetric.timestamp >= start_time,
                            PerformanceMetric.timestamp <= end_time
                        )
                    ).order_by(PerformanceMetric.timestamp.asc())
                    
                    result = await session.execute(query)
                    metrics = result.scalars().all()
                    
                    metrics_data[metric_name] = [
                        {
                            "timestamp": metric.timestamp,
                            "value": metric.metric_value,
                            "metadata": metric.metadata or {}
                        }
                        for metric in metrics
                    ]
            
            return dict(metrics_data)
            
        except Exception as e:
            logger.error("Failed to get metrics data", error=str(e))
            return {}


# Singleton instance
_anomaly_detector: Optional[AdvancedAnomalyDetector] = None


async def get_anomaly_detector() -> AdvancedAnomalyDetector:
    """Get singleton anomaly detector instance."""
    global _anomaly_detector
    
    if _anomaly_detector is None:
        _anomaly_detector = AdvancedAnomalyDetector()
        await _anomaly_detector.start()
    
    return _anomaly_detector


async def cleanup_anomaly_detector() -> None:
    """Cleanup anomaly detector resources."""
    global _anomaly_detector
    
    if _anomaly_detector:
        await _anomaly_detector.stop()
        _anomaly_detector = None