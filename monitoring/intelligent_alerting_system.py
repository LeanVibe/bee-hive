"""
IntelligentAlertingSystem - AI-Powered Anomaly Detection and Alerting

Provides intelligent alerting with AI-powered anomaly detection, noise reduction,
and escalation management for maintaining extraordinary performance while
minimizing alert fatigue.

Key Features:
- Machine learning-based anomaly detection using multiple algorithms
- Statistical anomaly detection with seasonal trend analysis
- Intelligent alert correlation and noise reduction
- Dynamic threshold adjustment based on historical patterns
- Smart escalation with context-aware severity assessment
- Performance regression detection with automatic baseline updates
"""

import asyncio
import time
import json
import statistics
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Machine learning for anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Time series analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """Types of detected anomalies."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    SEASONAL_DEVIATION = "seasonal_deviation"
    TREND_BREAK = "trend_break"
    PATTERN_ANOMALY = "pattern_anomaly"
    PERFORMANCE_REGRESSION = "performance_regression"
    CLUSTER_ANOMALY = "cluster_anomaly"


@dataclass
class AnomalyDetection:
    """Detected anomaly information."""
    anomaly_id: str
    timestamp: datetime
    metric_name: str
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float  # 0-1
    
    # Anomaly details
    current_value: float
    expected_value: Optional[float] = None
    deviation_score: Optional[float] = None
    pattern_description: str = ""
    
    # Context
    contributing_factors: List[str] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    business_impact: str = "unknown"
    
    # Detection algorithm info
    detection_algorithm: str = ""
    algorithm_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertAction:
    """Action to be taken for an alert."""
    action_type: str  # 'notify', 'escalate', 'auto_remediate', 'log'
    target: str
    message: str
    priority: int = 1
    delay_seconds: int = 0
    conditions: List[str] = field(default_factory=list)


@dataclass
class SmartAlert:
    """Intelligent alert with context and correlation."""
    alert_id: str
    created_at: datetime
    updated_at: datetime
    
    # Core alert info
    title: str
    description: str
    severity: AlertSeverity
    source_anomalies: List[AnomalyDetection]
    
    # Intelligence features
    correlation_score: float = 0.0
    noise_reduction_applied: bool = False
    escalation_level: int = 0
    suppression_reason: Optional[str] = None
    
    # Actions and responses
    actions: List[AlertAction] = field(default_factory=list)
    acknowledgment: Optional[datetime] = None
    resolution: Optional[datetime] = None
    
    # Business context
    affected_systems: List[str] = field(default_factory=list)
    estimated_impact: str = "low"
    recommendation: str = ""


class StatisticalAnomalyDetector:
    """Statistical anomaly detection using multiple statistical methods."""
    
    def __init__(self):
        self.detection_methods = [
            self._z_score_detection,
            self._iqr_detection,
            self._modified_z_score_detection,
            self._isolation_forest_detection
        ]
        
        # Detection parameters
        self.z_score_threshold = 3.0
        self.modified_z_score_threshold = 3.5
        self.iqr_multiplier = 1.5
        
        # Statistical models
        self.isolation_forest = None
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    async def detect_anomalies(self, metric_name: str, values: List[float], 
                              timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect anomalies using multiple statistical methods."""
        if len(values) < 10:  # Need minimum data points
            return []
        
        anomalies = []
        
        # Apply each detection method
        for method in self.detection_methods:
            try:
                method_anomalies = await method(metric_name, values, timestamps)
                anomalies.extend(method_anomalies)
            except Exception as e:
                logging.error(f"Error in anomaly detection method {method.__name__}: {e}")
        
        # Remove duplicates and merge similar anomalies
        return await self._merge_anomalies(anomalies)
    
    async def _z_score_detection(self, metric_name: str, values: List[float], 
                                timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect anomalies using Z-score method."""
        anomalies = []
        
        if len(values) < 3:
            return anomalies
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        if std_val == 0:  # No variation
            return anomalies
        
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val)
            
            if z_score > self.z_score_threshold:
                timestamp = timestamps[i] if timestamps else datetime.utcnow()
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"z_score_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.WARNING if z_score < 4.0 else AlertSeverity.CRITICAL,
                    confidence=min(1.0, z_score / 5.0),
                    current_value=value,
                    expected_value=mean_val,
                    deviation_score=z_score,
                    pattern_description=f"Value deviates {z_score:.2f} standard deviations from mean",
                    detection_algorithm="z_score",
                    algorithm_parameters={'threshold': self.z_score_threshold, 'z_score': z_score}
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _modified_z_score_detection(self, metric_name: str, values: List[float], 
                                         timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect anomalies using Modified Z-score (median-based)."""
        anomalies = []
        
        if len(values) < 3:
            return anomalies
        
        median_val = statistics.median(values)
        mad = statistics.median([abs(x - median_val) for x in values])
        
        if mad == 0:
            return anomalies
        
        for i, value in enumerate(values):
            modified_z_score = 0.6745 * (value - median_val) / mad
            
            if abs(modified_z_score) > self.modified_z_score_threshold:
                timestamp = timestamps[i] if timestamps else datetime.utcnow()
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"mod_z_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.WARNING,
                    confidence=min(1.0, abs(modified_z_score) / 5.0),
                    current_value=value,
                    expected_value=median_val,
                    deviation_score=abs(modified_z_score),
                    pattern_description=f"Value deviates {abs(modified_z_score):.2f} MAD from median",
                    detection_algorithm="modified_z_score",
                    algorithm_parameters={'threshold': self.modified_z_score_threshold}
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _iqr_detection(self, metric_name: str, values: List[float], 
                            timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect anomalies using Interquartile Range method."""
        anomalies = []
        
        if len(values) < 4:
            return anomalies
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return anomalies
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                timestamp = timestamps[i] if timestamps else datetime.utcnow()
                
                deviation = min(abs(value - lower_bound), abs(value - upper_bound))
                deviation_score = deviation / iqr if iqr > 0 else 0
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"iqr_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=AlertSeverity.INFO if deviation_score < 2.0 else AlertSeverity.WARNING,
                    confidence=min(1.0, deviation_score / 3.0),
                    current_value=value,
                    expected_value=(q1 + q3) / 2,
                    deviation_score=deviation_score,
                    pattern_description=f"Value outside IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}]",
                    detection_algorithm="iqr",
                    algorithm_parameters={'iqr_multiplier': self.iqr_multiplier}
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _isolation_forest_detection(self, metric_name: str, values: List[float], 
                                         timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect anomalies using Isolation Forest algorithm."""
        anomalies = []
        
        if not SKLEARN_AVAILABLE or len(values) < 10:
            return anomalies
        
        try:
            # Prepare data
            X = np.array(values).reshape(-1, 1)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit and predict
            self.isolation_forest.fit(X_scaled)
            outlier_labels = self.isolation_forest.predict(X_scaled)
            anomaly_scores = self.isolation_forest.decision_function(X_scaled)
            
            for i, (label, score) in enumerate(zip(outlier_labels, anomaly_scores)):
                if label == -1:  # Anomaly detected
                    timestamp = timestamps[i] if timestamps else datetime.utcnow()
                    
                    # Convert score to confidence (more negative = more anomalous)
                    confidence = min(1.0, abs(score) * 2)
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"isolation_{metric_name}_{int(timestamp.timestamp())}",
                        timestamp=timestamp,
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.PATTERN_ANOMALY,
                        severity=AlertSeverity.WARNING if confidence < 0.8 else AlertSeverity.CRITICAL,
                        confidence=confidence,
                        current_value=values[i],
                        deviation_score=abs(score),
                        pattern_description=f"Isolation Forest detected pattern anomaly (score: {score:.3f})",
                        detection_algorithm="isolation_forest",
                        algorithm_parameters={'contamination': 0.1, 'score': score}
                    )
                    
                    anomalies.append(anomaly)
        
        except Exception as e:
            logging.error(f"Error in Isolation Forest detection: {e}")
        
        return anomalies
    
    async def _merge_anomalies(self, anomalies: List[AnomalyDetection]) -> List[AnomalyDetection]:
        """Merge similar anomalies to reduce noise."""
        if len(anomalies) <= 1:
            return anomalies
        
        # Group by metric name and timestamp (within 5 minute window)
        groups = defaultdict(list)
        
        for anomaly in anomalies:
            # Create grouping key based on metric and time window
            time_window = int(anomaly.timestamp.timestamp() / 300) * 300  # 5-minute windows
            key = (anomaly.metric_name, time_window)
            groups[key].append(anomaly)
        
        merged = []
        
        for group in groups.values():
            if len(group) == 1:
                merged.extend(group)
            else:
                # Merge multiple detections into single high-confidence anomaly
                merged_anomaly = await self._create_merged_anomaly(group)
                merged.append(merged_anomaly)
        
        return merged
    
    async def _create_merged_anomaly(self, anomalies: List[AnomalyDetection]) -> AnomalyDetection:
        """Create merged anomaly from multiple detections."""
        # Take the highest severity and confidence
        max_severity = max(anomalies, key=lambda x: x.severity.value).severity
        avg_confidence = sum(a.confidence for a in anomalies) / len(anomalies)
        
        # Combine algorithm names
        algorithms = list(set(a.detection_algorithm for a in anomalies))
        
        # Use the first anomaly as base and enhance with merged info
        base_anomaly = anomalies[0]
        
        merged = AnomalyDetection(
            anomaly_id=f"merged_{base_anomaly.metric_name}_{int(base_anomaly.timestamp.timestamp())}",
            timestamp=base_anomaly.timestamp,
            metric_name=base_anomaly.metric_name,
            anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
            severity=max_severity,
            confidence=min(1.0, avg_confidence * 1.2),  # Boost confidence for multiple detections
            current_value=base_anomaly.current_value,
            expected_value=base_anomaly.expected_value,
            deviation_score=max(a.deviation_score for a in anomalies if a.deviation_score),
            pattern_description=f"Multiple algorithms detected anomaly: {', '.join(algorithms)}",
            detection_algorithm=f"merged_{'+'.join(algorithms)}",
            algorithm_parameters={
                'merged_from': len(anomalies),
                'detection_methods': algorithms
            }
        )
        
        return merged


class SeasonalAnomalyDetector:
    """Seasonal pattern anomaly detection using time series analysis."""
    
    def __init__(self):
        self.seasonal_models = {}
        self.trend_models = {}
        self.baseline_windows = defaultdict(list)  # Store baseline data for each metric
        
        # Seasonal detection parameters
        self.seasonal_periods = {
            'daily': 24 * 60,      # Daily pattern (minutes)
            'weekly': 7 * 24 * 60,  # Weekly pattern (minutes)
            'monthly': 30 * 24 * 60 # Monthly pattern (minutes)
        }
        
        self.seasonal_deviation_threshold = 2.0  # Standard deviations
        self.trend_change_threshold = 0.15      # 15% change in trend
    
    async def detect_seasonal_anomalies(self, metric_name: str, values: List[float], 
                                       timestamps: List[datetime]) -> List[AnomalyDetection]:
        """Detect anomalies based on seasonal patterns and trends."""
        anomalies = []
        
        if len(values) < 50 or not STATSMODELS_AVAILABLE:  # Need enough data for seasonal analysis
            return anomalies
        
        try:
            # Convert to time series
            time_series_data = await self._prepare_time_series_data(values, timestamps)
            
            # Perform seasonal decomposition
            seasonal_anomalies = await self._detect_seasonal_deviations(
                metric_name, time_series_data, timestamps
            )
            anomalies.extend(seasonal_anomalies)
            
            # Detect trend breaks
            trend_anomalies = await self._detect_trend_breaks(
                metric_name, time_series_data, timestamps
            )
            anomalies.extend(trend_anomalies)
            
        except Exception as e:
            logging.error(f"Error in seasonal anomaly detection: {e}")
        
        return anomalies
    
    async def _prepare_time_series_data(self, values: List[float], 
                                       timestamps: List[datetime]) -> np.ndarray:
        """Prepare time series data for analysis."""
        # For now, return values as numpy array
        # In practice, you'd create proper time series with pandas
        return np.array(values)
    
    async def _detect_seasonal_deviations(self, metric_name: str, time_series_data: np.ndarray,
                                         timestamps: List[datetime]) -> List[AnomalyDetection]:
        """Detect deviations from seasonal patterns."""
        anomalies = []
        
        if not STATSMODELS_AVAILABLE or len(time_series_data) < 100:
            return anomalies
        
        try:
            # Perform seasonal decomposition
            # Note: This is simplified - real implementation would use pandas DataFrame
            period = min(24, len(time_series_data) // 4)  # Adaptive period
            
            # Calculate rolling mean and seasonal component
            window_size = max(3, period // 4)
            rolling_mean = np.convolve(time_series_data, np.ones(window_size)/window_size, mode='same')
            seasonal_component = time_series_data - rolling_mean
            
            # Calculate seasonal deviation threshold
            seasonal_std = np.std(seasonal_component)
            threshold = self.seasonal_deviation_threshold * seasonal_std
            
            # Detect anomalies
            for i, (original, seasonal) in enumerate(zip(time_series_data, seasonal_component)):
                if abs(seasonal) > threshold:
                    timestamp = timestamps[i] if i < len(timestamps) else datetime.utcnow()
                    
                    severity = AlertSeverity.WARNING
                    if abs(seasonal) > threshold * 2:
                        severity = AlertSeverity.CRITICAL
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"seasonal_{metric_name}_{int(timestamp.timestamp())}",
                        timestamp=timestamp,
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.SEASONAL_DEVIATION,
                        severity=severity,
                        confidence=min(1.0, abs(seasonal) / (threshold * 2)),
                        current_value=original,
                        expected_value=rolling_mean[i],
                        deviation_score=abs(seasonal) / seasonal_std,
                        pattern_description=f"Seasonal deviation: {abs(seasonal):.3f} (threshold: {threshold:.3f})",
                        detection_algorithm="seasonal_decomposition",
                        algorithm_parameters={
                            'period': period,
                            'threshold': threshold,
                            'seasonal_component': seasonal
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        except Exception as e:
            logging.error(f"Error in seasonal deviation detection: {e}")
        
        return anomalies
    
    async def _detect_trend_breaks(self, metric_name: str, time_series_data: np.ndarray,
                                  timestamps: List[datetime]) -> List[AnomalyDetection]:
        """Detect sudden changes in trend."""
        anomalies = []
        
        if len(time_series_data) < 20:
            return anomalies
        
        try:
            # Calculate local trends using linear regression on windows
            window_size = max(5, len(time_series_data) // 10)
            trends = []
            
            for i in range(len(time_series_data) - window_size + 1):
                window_data = time_series_data[i:i + window_size]
                x = np.arange(window_size)
                
                # Calculate trend (slope of linear regression)
                if len(window_data) > 1:
                    slope, _ = np.polyfit(x, window_data, 1)
                    trends.append(slope)
            
            if len(trends) < 5:
                return anomalies
            
            # Detect sudden trend changes
            trend_changes = np.diff(trends)
            trend_std = np.std(trend_changes)
            
            if trend_std == 0:
                return anomalies
            
            change_threshold = self.trend_change_threshold * trend_std
            
            for i, change in enumerate(trend_changes):
                if abs(change) > change_threshold:
                    # Find corresponding timestamp
                    data_index = i + window_size
                    timestamp = timestamps[data_index] if data_index < len(timestamps) else datetime.utcnow()
                    
                    severity = AlertSeverity.WARNING
                    if abs(change) > change_threshold * 2:
                        severity = AlertSeverity.CRITICAL
                    
                    anomaly = AnomalyDetection(
                        anomaly_id=f"trend_break_{metric_name}_{int(timestamp.timestamp())}",
                        timestamp=timestamp,
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.TREND_BREAK,
                        severity=severity,
                        confidence=min(1.0, abs(change) / (change_threshold * 2)),
                        current_value=time_series_data[data_index],
                        deviation_score=abs(change) / trend_std,
                        pattern_description=f"Trend break detected: {change:.3f} change in slope",
                        detection_algorithm="trend_analysis",
                        algorithm_parameters={
                            'window_size': window_size,
                            'threshold': change_threshold,
                            'trend_change': change
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        except Exception as e:
            logging.error(f"Error in trend break detection: {e}")
        
        return anomalies


class PerformanceRegressionDetector:
    """Detect performance regressions against established baselines."""
    
    def __init__(self):
        # Performance baselines (from LeanVibe Agent Hive 2.0 achievements)
        self.performance_baselines = {
            'task_assignment_latency_ms': {
                'exceptional': 0.01,    # Current exceptional performance
                'acceptable': 0.02,     # 2x exceptional (warning threshold)
                'critical': 0.1,        # 10x exceptional (critical threshold)
                'trend_window': 100     # Number of samples for trend analysis
            },
            'message_throughput_per_sec': {
                'exceptional': 50000,   # Target performance
                'acceptable': 40000,    # Warning threshold
                'critical': 25000,      # Critical threshold
                'trend_window': 50
            },
            'memory_usage_mb': {
                'exceptional': 285,     # Current exceptional performance
                'acceptable': 400,      # Warning threshold
                'critical': 500,        # Critical threshold
                'trend_window': 200
            },
            'error_rate_percent': {
                'exceptional': 0.005,   # Current exceptional performance
                'acceptable': 0.1,      # Warning threshold
                'critical': 1.0,        # Critical threshold
                'trend_window': 100
            }
        }
        
        # Regression detection parameters
        self.regression_threshold = 0.05  # 5% degradation
        self.trend_samples = 20           # Samples for trend calculation
    
    async def detect_performance_regressions(self, metric_name: str, values: List[float],
                                           timestamps: List[datetime] = None) -> List[AnomalyDetection]:
        """Detect performance regressions against baselines."""
        anomalies = []
        
        if metric_name not in self.performance_baselines or len(values) < 10:
            return anomalies
        
        baseline_config = self.performance_baselines[metric_name]
        
        try:
            # Detect threshold violations
            threshold_anomalies = await self._detect_threshold_violations(
                metric_name, values, timestamps, baseline_config
            )
            anomalies.extend(threshold_anomalies)
            
            # Detect performance degradation trends
            trend_anomalies = await self._detect_performance_trends(
                metric_name, values, timestamps, baseline_config
            )
            anomalies.extend(trend_anomalies)
            
        except Exception as e:
            logging.error(f"Error in performance regression detection: {e}")
        
        return anomalies
    
    async def _detect_threshold_violations(self, metric_name: str, values: List[float],
                                          timestamps: List[datetime], 
                                          baseline_config: Dict[str, Any]) -> List[AnomalyDetection]:
        """Detect violations of performance thresholds."""
        anomalies = []
        
        exceptional = baseline_config['exceptional']
        acceptable = baseline_config['acceptable']
        critical = baseline_config['critical']
        
        for i, value in enumerate(values):
            timestamp = timestamps[i] if timestamps and i < len(timestamps) else datetime.utcnow()
            violation_detected = False
            severity = AlertSeverity.INFO
            
            # Determine violation severity based on metric type
            if metric_name in ['task_assignment_latency_ms', 'memory_usage_mb', 'error_rate_percent']:
                # Lower is better metrics
                if value > critical:
                    violation_detected = True
                    severity = AlertSeverity.CRITICAL
                elif value > acceptable:
                    violation_detected = True
                    severity = AlertSeverity.WARNING
                
                expected_value = exceptional
                degradation_factor = value / exceptional if exceptional > 0 else 0
                
            else:
                # Higher is better metrics (e.g., throughput)
                if value < critical:
                    violation_detected = True
                    severity = AlertSeverity.CRITICAL
                elif value < acceptable:
                    violation_detected = True
                    severity = AlertSeverity.WARNING
                
                expected_value = exceptional
                degradation_factor = exceptional / value if value > 0 else float('inf')
            
            if violation_detected:
                confidence = min(1.0, abs(degradation_factor - 1.0))
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"regression_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.PERFORMANCE_REGRESSION,
                    severity=severity,
                    confidence=confidence,
                    current_value=value,
                    expected_value=expected_value,
                    deviation_score=degradation_factor,
                    pattern_description=f"Performance regression: {degradation_factor:.2f}x degradation from exceptional baseline",
                    business_impact="high" if severity == AlertSeverity.CRITICAL else "medium",
                    detection_algorithm="threshold_violation",
                    algorithm_parameters={
                        'exceptional_baseline': exceptional,
                        'acceptable_threshold': acceptable,
                        'critical_threshold': critical,
                        'degradation_factor': degradation_factor
                    }
                )
                
                # Add contributing factors based on metric type
                if metric_name == 'task_assignment_latency_ms':
                    anomaly.contributing_factors = [
                        'memory_pressure', 'cpu_contention', 'gc_activity', 'lock_contention'
                    ]
                elif metric_name == 'message_throughput_per_sec':
                    anomaly.contributing_factors = [
                        'network_congestion', 'connection_pool_exhaustion', 'cpu_bottleneck'
                    ]
                elif metric_name == 'memory_usage_mb':
                    anomaly.contributing_factors = [
                        'memory_leaks', 'cache_overflow', 'object_accumulation'
                    ]
                
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_performance_trends(self, metric_name: str, values: List[float],
                                        timestamps: List[datetime],
                                        baseline_config: Dict[str, Any]) -> List[AnomalyDetection]:
        """Detect concerning performance trends."""
        anomalies = []
        
        if len(values) < self.trend_samples:
            return anomalies
        
        try:
            # Analyze recent trend
            recent_values = values[-self.trend_samples:]
            x = np.arange(len(recent_values))
            
            # Calculate trend slope
            slope, intercept = np.polyfit(x, recent_values, 1)
            
            # Determine if trend is concerning
            exceptional = baseline_config['exceptional']
            trend_direction_concerning = False
            
            if metric_name in ['task_assignment_latency_ms', 'memory_usage_mb', 'error_rate_percent']:
                # Increasing trend is concerning for "lower is better" metrics
                trend_direction_concerning = slope > 0
                relative_trend = slope / exceptional if exceptional > 0 else 0
            else:
                # Decreasing trend is concerning for "higher is better" metrics
                trend_direction_concerning = slope < 0
                relative_trend = abs(slope) / exceptional if exceptional > 0 else 0
            
            # Check if trend magnitude is significant
            if trend_direction_concerning and abs(relative_trend) > self.regression_threshold:
                timestamp = timestamps[-1] if timestamps else datetime.utcnow()
                
                # Predict future value based on trend
                future_steps = 10  # Predict 10 steps ahead
                predicted_value = slope * (len(recent_values) + future_steps) + intercept
                
                severity = AlertSeverity.WARNING
                if abs(relative_trend) > self.regression_threshold * 3:
                    severity = AlertSeverity.CRITICAL
                
                anomaly = AnomalyDetection(
                    anomaly_id=f"trend_regression_{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.PERFORMANCE_REGRESSION,
                    severity=severity,
                    confidence=min(1.0, abs(relative_trend) * 10),
                    current_value=recent_values[-1],
                    expected_value=exceptional,
                    deviation_score=abs(relative_trend),
                    pattern_description=f"Concerning performance trend: {slope:.4f} per sample, predicting {predicted_value:.3f}",
                    business_impact="medium",
                    detection_algorithm="trend_analysis",
                    algorithm_parameters={
                        'trend_slope': slope,
                        'trend_samples': self.trend_samples,
                        'predicted_value': predicted_value,
                        'relative_trend': relative_trend
                    }
                )
                
                # Add specific recommendations based on trend
                if slope > 0 and metric_name == 'task_assignment_latency_ms':
                    anomaly.recommendation = "Consider optimization review: memory allocation, CPU usage, or system load"
                elif slope < 0 and metric_name == 'message_throughput_per_sec':
                    anomaly.recommendation = "Investigate throughput degradation: connection pools, network, or processing bottlenecks"
                
                anomalies.append(anomaly)
        
        except Exception as e:
            logging.error(f"Error in performance trend analysis: {e}")
        
        return anomalies


class AlertCorrelationEngine:
    """Correlate related alerts to reduce noise and provide context."""
    
    def __init__(self):
        # Correlation rules - metrics that often correlate
        self.correlation_rules = {
            'task_assignment_latency_ms': {
                'related_metrics': ['cpu_percent', 'memory_percent', 'gc_collections'],
                'correlation_window': 300,  # 5 minutes
                'correlation_strength': 0.7
            },
            'message_throughput_per_sec': {
                'related_metrics': ['network_bytes_sent_per_sec', 'cpu_percent', 'connection_count'],
                'correlation_window': 180,  # 3 minutes
                'correlation_strength': 0.6
            },
            'memory_usage_mb': {
                'related_metrics': ['gc_collections', 'task_assignment_latency_ms'],
                'correlation_window': 600,  # 10 minutes
                'correlation_strength': 0.8
            }
        }
        
        # Alert suppression rules
        self.suppression_rules = {
            'duplicate_window': 300,      # Don't send duplicate alerts within 5 minutes
            'escalation_delay': 600,      # Wait 10 minutes before escalating
            'noise_threshold': 5          # Suppress if more than 5 similar alerts in window
        }
    
    async def correlate_alerts(self, new_anomalies: List[AnomalyDetection],
                              recent_anomalies: List[AnomalyDetection]) -> List[SmartAlert]:
        """Correlate anomalies and create intelligent alerts."""
        smart_alerts = []
        
        # Group anomalies by correlation
        correlation_groups = await self._group_correlated_anomalies(
            new_anomalies + recent_anomalies
        )
        
        for group in correlation_groups:
            # Create smart alert for each group
            smart_alert = await self._create_smart_alert(group, recent_anomalies)
            if smart_alert:
                smart_alerts.append(smart_alert)
        
        return smart_alerts
    
    async def _group_correlated_anomalies(self, anomalies: List[AnomalyDetection]) -> List[List[AnomalyDetection]]:
        """Group anomalies that are likely correlated."""
        if not anomalies:
            return []
        
        # Sort anomalies by timestamp
        sorted_anomalies = sorted(anomalies, key=lambda x: x.timestamp)
        
        groups = []
        current_group = [sorted_anomalies[0]]
        
        for i in range(1, len(sorted_anomalies)):
            current_anomaly = sorted_anomalies[i]
            last_in_group = current_group[-1]
            
            # Check if anomalies are correlated
            is_correlated = await self._are_anomalies_correlated(last_in_group, current_anomaly)
            
            if is_correlated:
                current_group.append(current_anomaly)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [current_anomaly]
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _are_anomalies_correlated(self, anomaly1: AnomalyDetection, 
                                       anomaly2: AnomalyDetection) -> bool:
        """Determine if two anomalies are correlated."""
        # Time correlation - within correlation window
        time_diff = abs((anomaly1.timestamp - anomaly2.timestamp).total_seconds())
        
        metric1 = anomaly1.metric_name
        metric2 = anomaly2.metric_name
        
        # Check correlation rules
        for rule_metric, rule_config in self.correlation_rules.items():
            if rule_metric == metric1 and metric2 in rule_config['related_metrics']:
                return time_diff <= rule_config['correlation_window']
            elif rule_metric == metric2 and metric1 in rule_config['related_metrics']:
                return time_diff <= rule_config['correlation_window']
        
        # Same metric correlation
        if metric1 == metric2:
            return time_diff <= 300  # 5 minutes for same metric
        
        return False
    
    async def _create_smart_alert(self, anomalies: List[AnomalyDetection],
                                 recent_anomalies: List[AnomalyDetection]) -> Optional[SmartAlert]:
        """Create intelligent alert from correlated anomalies."""
        if not anomalies:
            return None
        
        # Determine alert properties
        max_severity = max(anomaly.severity for anomaly in anomalies)
        avg_confidence = sum(anomaly.confidence for anomaly in anomalies) / len(anomalies)
        
        # Check for noise/suppression
        suppression_reason = await self._check_suppression(anomalies, recent_anomalies)
        
        # Create alert
        primary_anomaly = anomalies[0]  # Use first anomaly as primary
        
        # Generate intelligent title and description
        if len(anomalies) == 1:
            title = f"Performance Anomaly: {primary_anomaly.metric_name}"
            description = primary_anomaly.pattern_description
        else:
            metrics = list(set(a.metric_name for a in anomalies))
            title = f"Correlated Performance Anomalies ({len(anomalies)} metrics)"
            description = f"Multiple related performance issues detected across: {', '.join(metrics)}"
        
        # Calculate correlation score
        correlation_score = len(anomalies) * avg_confidence / 10.0  # Normalize to 0-1
        
        smart_alert = SmartAlert(
            alert_id=f"smart_alert_{int(primary_anomaly.timestamp.timestamp())}_{len(anomalies)}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            title=title,
            description=description,
            severity=max_severity,
            source_anomalies=anomalies,
            correlation_score=min(1.0, correlation_score),
            noise_reduction_applied=suppression_reason is not None,
            suppression_reason=suppression_reason
        )
        
        # Add intelligent actions
        smart_alert.actions = await self._generate_alert_actions(smart_alert)
        
        # Determine business impact
        smart_alert.estimated_impact = await self._assess_business_impact(anomalies)
        
        # Generate recommendations
        smart_alert.recommendation = await self._generate_recommendations(anomalies)
        
        # Identify affected systems
        smart_alert.affected_systems = list(set(
            a.metric_name.split('_')[0] for a in anomalies  # Extract system from metric name
        ))
        
        return smart_alert
    
    async def _check_suppression(self, anomalies: List[AnomalyDetection],
                                recent_anomalies: List[AnomalyDetection]) -> Optional[str]:
        """Check if alert should be suppressed to reduce noise."""
        # Check for duplicate alerts
        duplicate_count = 0
        current_time = datetime.utcnow()
        
        for recent in recent_anomalies:
            if (current_time - recent.timestamp).total_seconds() <= self.suppression_rules['duplicate_window']:
                # Check if it's a similar anomaly
                for current in anomalies:
                    if (recent.metric_name == current.metric_name and
                        recent.anomaly_type == current.anomaly_type):
                        duplicate_count += 1
        
        # Suppress if too many duplicates
        if duplicate_count >= self.suppression_rules['noise_threshold']:
            return f"Suppressed due to {duplicate_count} similar alerts in {self.suppression_rules['duplicate_window']}s window"
        
        return None
    
    async def _generate_alert_actions(self, alert: SmartAlert) -> List[AlertAction]:
        """Generate intelligent actions for the alert."""
        actions = []
        
        # Always log the alert
        actions.append(AlertAction(
            action_type='log',
            target='performance_log',
            message=f"Smart alert generated: {alert.title}",
            priority=1
        ))
        
        # Notification based on severity
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            actions.append(AlertAction(
                action_type='notify',
                target='ops_team',
                message=f"CRITICAL: {alert.title} - {alert.description}",
                priority=1
            ))
            
            # Auto-escalation for critical alerts
            actions.append(AlertAction(
                action_type='escalate',
                target='engineering_lead',
                message=f"Escalated critical alert: {alert.title}",
                priority=2,
                delay_seconds=300,  # Escalate after 5 minutes
                conditions=['unacknowledged']
            ))
        
        elif alert.severity == AlertSeverity.WARNING:
            actions.append(AlertAction(
                action_type='notify',
                target='monitoring_team',
                message=f"WARNING: {alert.title} - {alert.description}",
                priority=2
            ))
        
        # Auto-remediation for specific issues
        performance_metrics = [a.metric_name for a in alert.source_anomalies]
        
        if 'memory_usage_mb' in performance_metrics:
            actions.append(AlertAction(
                action_type='auto_remediate',
                target='memory_optimizer',
                message='Trigger memory optimization procedures',
                priority=3,
                delay_seconds=60
            ))
        
        return actions
    
    async def _assess_business_impact(self, anomalies: List[AnomalyDetection]) -> str:
        """Assess business impact of anomalies."""
        # Impact assessment based on metrics affected
        critical_metrics = [
            'task_assignment_latency_ms',
            'message_throughput_per_sec',
            'error_rate_percent',
            'system_availability_percent'
        ]
        
        affected_critical = sum(1 for a in anomalies if a.metric_name in critical_metrics)
        total_anomalies = len(anomalies)
        
        # Severity-based impact
        critical_count = sum(1 for a in anomalies if a.severity == AlertSeverity.CRITICAL)
        
        if critical_count >= 2 or affected_critical >= 3:
            return "high"
        elif critical_count >= 1 or affected_critical >= 2:
            return "medium"
        else:
            return "low"
    
    async def _generate_recommendations(self, anomalies: List[AnomalyDetection]) -> str:
        """Generate intelligent recommendations based on anomalies."""
        recommendations = []
        
        # Metric-specific recommendations
        metrics_affected = [a.metric_name for a in anomalies]
        
        if 'task_assignment_latency_ms' in metrics_affected:
            recommendations.append("Review task assignment optimization settings and memory allocation patterns")
        
        if 'message_throughput_per_sec' in metrics_affected:
            recommendations.append("Check connection pools, network configuration, and message batching settings")
        
        if 'memory_usage_mb' in metrics_affected:
            recommendations.append("Investigate memory leaks, optimize garbage collection, and review object pooling")
        
        if 'error_rate_percent' in metrics_affected:
            recommendations.append("Review error logs, check system dependencies, and validate input handling")
        
        # Correlation-based recommendations
        if len(set(metrics_affected)) > 1:
            recommendations.append("Multiple metrics affected - consider system-wide performance review")
        
        return ". ".join(recommendations) if recommendations else "Monitor situation and investigate if pattern continues"


class IntelligentAlertingSystem:
    """
    AI-powered intelligent alerting system.
    
    Combines multiple anomaly detection algorithms with intelligent correlation,
    noise reduction, and escalation management to maintain extraordinary
    performance while minimizing alert fatigue.
    """
    
    def __init__(self):
        # Anomaly detection components
        self.statistical_detector = StatisticalAnomalyDetector()
        self.seasonal_detector = SeasonalAnomalyDetector()
        self.regression_detector = PerformanceRegressionDetector()
        
        # Alert management components
        self.correlation_engine = AlertCorrelationEngine()
        
        # Alert storage and tracking
        self.recent_anomalies = deque(maxlen=1000)
        self.active_alerts = {}
        self.suppressed_alerts = {}
        
        # System state
        self.alerting_active = False
        self.alert_handlers = []
        
        # Configuration
        self.detection_interval = 30  # seconds
        self.correlation_window = 600  # 10 minutes
        
    async def initialize(self) -> bool:
        """Initialize intelligent alerting system."""
        try:
            self.alerting_active = True
            logging.info("Intelligent alerting system initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize intelligent alerting system: {e}")
            return False
    
    async def start_intelligent_alerting(self) -> bool:
        """Start intelligent alerting with all detection algorithms."""
        if not self.alerting_active:
            return False
        
        try:
            # Start alerting loop
            self.alerting_task = asyncio.create_task(self._alerting_loop())
            logging.info("Intelligent alerting started")
            return True
        except Exception as e:
            logging.error(f"Failed to start intelligent alerting: {e}")
            return False
    
    async def stop_intelligent_alerting(self) -> None:
        """Stop intelligent alerting."""
        self.alerting_active = False
        
        if hasattr(self, 'alerting_task'):
            self.alerting_task.cancel()
            try:
                await self.alerting_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Intelligent alerting stopped")
    
    async def _alerting_loop(self) -> None:
        """Main intelligent alerting loop."""
        while self.alerting_active:
            try:
                await self._process_intelligent_alerting_cycle()
                await asyncio.sleep(self.detection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in intelligent alerting loop: {e}")
                await asyncio.sleep(self.detection_interval)
    
    async def _process_intelligent_alerting_cycle(self) -> None:
        """Process one cycle of intelligent alerting."""
        # This would integrate with the PerformanceMonitoringSystem to get current metrics
        # For now, simulate the process
        
        # Get current metrics (this would come from monitoring system)
        current_metrics = await self._get_current_metrics()
        
        # Apply all anomaly detection algorithms
        all_anomalies = []
        
        for metric_name, metric_data in current_metrics.items():
            values = metric_data.get('values', [])
            timestamps = metric_data.get('timestamps', [])
            
            if len(values) < 5:  # Need minimum data
                continue
            
            try:
                # Statistical anomaly detection
                statistical_anomalies = await self.statistical_detector.detect_anomalies(
                    metric_name, values, timestamps
                )
                all_anomalies.extend(statistical_anomalies)
                
                # Seasonal anomaly detection
                if len(values) >= 50:  # Need more data for seasonal
                    seasonal_anomalies = await self.seasonal_detector.detect_seasonal_anomalies(
                        metric_name, values, timestamps
                    )
                    all_anomalies.extend(seasonal_anomalies)
                
                # Performance regression detection
                regression_anomalies = await self.regression_detector.detect_performance_regressions(
                    metric_name, values, timestamps
                )
                all_anomalies.extend(regression_anomalies)
                
            except Exception as e:
                logging.error(f"Error detecting anomalies for {metric_name}: {e}")
        
        # Store new anomalies
        self.recent_anomalies.extend(all_anomalies)
        
        # Correlate and create intelligent alerts
        if all_anomalies:
            smart_alerts = await self.correlation_engine.correlate_alerts(
                all_anomalies, list(self.recent_anomalies)
            )
            
            # Process smart alerts
            for alert in smart_alerts:
                await self._process_smart_alert(alert)
    
    async def _get_current_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics data (simulated for now)."""
        # This would integrate with the actual monitoring system
        # For now, return simulated data
        import random
        
        current_time = datetime.utcnow()
        timestamps = [current_time - timedelta(minutes=i) for i in range(30, 0, -1)]
        
        return {
            'task_assignment_latency_ms': {
                'values': [0.01 + random.uniform(-0.005, 0.02) for _ in range(30)],
                'timestamps': timestamps
            },
            'message_throughput_per_sec': {
                'values': [50000 + random.uniform(-5000, 5000) for _ in range(30)],
                'timestamps': timestamps
            },
            'memory_usage_mb': {
                'values': [285 + random.uniform(-20, 100) for _ in range(30)],
                'timestamps': timestamps
            }
        }
    
    async def _process_smart_alert(self, alert: SmartAlert) -> None:
        """Process a smart alert."""
        # Check if alert should be suppressed
        if alert.suppression_reason:
            self.suppressed_alerts[alert.alert_id] = alert
            logging.info(f"Alert suppressed: {alert.title} - {alert.suppression_reason}")
            return
        
        # Store active alert
        self.active_alerts[alert.alert_id] = alert
        
        # Execute alert actions
        for action in alert.actions:
            try:
                await self._execute_alert_action(alert, action)
            except Exception as e:
                logging.error(f"Error executing alert action {action.action_type}: {e}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Error in alert handler: {e}")
        
        logging.info(f"Smart alert processed: {alert.title} (severity: {alert.severity.value})")
    
    async def _execute_alert_action(self, alert: SmartAlert, action: AlertAction) -> None:
        """Execute a specific alert action."""
        if action.delay_seconds > 0:
            await asyncio.sleep(action.delay_seconds)
        
        # Check action conditions
        if action.conditions:
            for condition in action.conditions:
                if condition == 'unacknowledged' and alert.acknowledgment is not None:
                    return  # Skip action if alert is acknowledged
        
        # Execute action based on type
        if action.action_type == 'log':
            logging.info(f"ALERT LOG: {action.message}")
        
        elif action.action_type == 'notify':
            # In practice, this would send notifications via email, Slack, PagerDuty, etc.
            logging.warning(f"ALERT NOTIFICATION to {action.target}: {action.message}")
        
        elif action.action_type == 'escalate':
            # Escalate alert
            alert.escalation_level += 1
            logging.critical(f"ALERT ESCALATION (level {alert.escalation_level}) to {action.target}: {action.message}")
        
        elif action.action_type == 'auto_remediate':
            # Trigger auto-remediation
            logging.info(f"AUTO-REMEDIATION triggered for {action.target}: {action.message}")
            # This would trigger actual remediation procedures
    
    def add_alert_handler(self, handler: Callable[[SmartAlert], None]) -> None:
        """Add custom alert handler."""
        self.alert_handlers.append(handler)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgment = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            logging.info(f"Alert acknowledged: {alert.title} by {acknowledged_by}")
            return True
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolution = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            # Move from active to resolved
            del self.active_alerts[alert_id]
            
            logging.info(f"Alert resolved: {alert.title} by {resolved_by}")
            return True
        return False
    
    def get_alerting_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive alerting dashboard data."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'alerting_status': {
                'active': self.alerting_active,
                'detection_algorithms': [
                    'statistical_detection',
                    'seasonal_analysis',
                    'performance_regression',
                    'correlation_engine'
                ],
                'ml_algorithms_available': SKLEARN_AVAILABLE,
                'time_series_analysis_available': STATSMODELS_AVAILABLE
            },
            'alert_summary': {
                'active_alerts': len(self.active_alerts),
                'suppressed_alerts': len(self.suppressed_alerts),
                'recent_anomalies': len(self.recent_anomalies)
            },
            'active_alerts': [
                {
                    'alert_id': alert.alert_id,
                    'title': alert.title,
                    'severity': alert.severity.value,
                    'created_at': alert.created_at.isoformat(),
                    'correlation_score': alert.correlation_score,
                    'anomaly_count': len(alert.source_anomalies),
                    'business_impact': alert.estimated_impact,
                    'acknowledged': alert.acknowledgment is not None
                }
                for alert in self.active_alerts.values()
            ],
            'anomaly_statistics': {
                'statistical_outliers': len([a for a in self.recent_anomalies 
                                           if a.anomaly_type == AnomalyType.STATISTICAL_OUTLIER]),
                'seasonal_deviations': len([a for a in self.recent_anomalies 
                                          if a.anomaly_type == AnomalyType.SEASONAL_DEVIATION]),
                'performance_regressions': len([a for a in self.recent_anomalies 
                                              if a.anomaly_type == AnomalyType.PERFORMANCE_REGRESSION]),
                'trend_breaks': len([a for a in self.recent_anomalies 
                                   if a.anomaly_type == AnomalyType.TREND_BREAK])
            }
        }