"""
Performance Regression Detection and Alerting System.

This module provides comprehensive regression detection for semantic memory
performance metrics with automated alerting and trend analysis.

Features:
- Statistical regression detection using multiple methods
- Performance baseline management and tracking
- Automated alerting for performance degradation
- Trend analysis and forecasting
- Integration with CI/CD pipelines
- Historical performance data storage
- Regression severity classification
- Automated remediation suggestions
"""

import asyncio
import logging
import json
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pickle
import numpy as np
from scipy import stats
import sqlite3
from collections import defaultdict, deque

from .semantic_memory_benchmarks import BenchmarkResult, PerformanceReport, PerformanceTarget

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for performance regression."""
    NONE = "none"
    MINOR = "minor"          # 5-15% degradation
    MODERATE = "moderate"    # 15-30% degradation  
    MAJOR = "major"          # 30-50% degradation
    CRITICAL = "critical"    # >50% degradation


class DetectionMethod(Enum):
    """Statistical methods for regression detection."""
    THRESHOLD_BASED = "threshold_based"
    STATISTICAL_TEST = "statistical_test"
    TREND_ANALYSIS = "trend_analysis"
    CHANGE_POINT = "change_point"
    ENSEMBLE = "ensemble"


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    metric_name: str
    baseline_value: float
    baseline_std: float
    baseline_samples: int
    confidence_interval: Tuple[float, float]
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    
    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if baseline is stale and needs updating."""
        age = datetime.utcnow() - self.updated_at
        return age.days > max_age_days


@dataclass
class RegressionAlert:
    """Regression detection alert."""
    alert_id: str
    metric_name: str
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    degradation_percent: float
    detection_method: DetectionMethod
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestions: List[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Performance trend analysis results."""
    metric_name: str
    trend_direction: str  # "improving", "degrading", "stable"
    trend_slope: float
    trend_confidence: float
    forecast_7d: float
    forecast_30d: float
    analysis_timestamp: datetime
    samples_analyzed: int


class PerformanceDatabase:
    """SQLite database for storing performance history."""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    test_context TEXT,
                    git_commit TEXT,
                    environment TEXT DEFAULT 'test',
                    metadata TEXT
                )
            """)
            
            # Performance baselines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_baselines (
                    metric_name TEXT PRIMARY KEY,
                    baseline_value REAL NOT NULL,
                    baseline_std REAL NOT NULL,
                    baseline_samples INTEGER NOT NULL,
                    confidence_lower REAL NOT NULL,
                    confidence_upper REAL NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    version TEXT DEFAULT '1.0'
                )
            """)
            
            # Regression alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regression_alerts (
                    alert_id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    degradation_percent REAL NOT NULL,
                    detection_method TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    context TEXT,
                    remediation_suggestions TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON performance_metrics(metric_name, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_metric ON regression_alerts(metric_name)")
            
            conn.commit()
    
    def store_performance_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime = None,
        test_context: str = None,
        git_commit: str = None,
        environment: str = "test",
        metadata: Dict[str, Any] = None
    ):
        """Store performance metric."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics 
                (metric_name, value, timestamp, test_context, git_commit, environment, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric_name, value, timestamp, test_context, git_commit, environment,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
    
    def get_performance_history(
        self,
        metric_name: str,
        days: int = 30,
        limit: int = None
    ) -> List[Tuple[datetime, float]]:
        """Get performance history for metric."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, value FROM performance_metrics 
                WHERE metric_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (metric_name, since_date))
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
    
    def store_baseline(self, baseline: PerformanceBaseline):
        """Store or update performance baseline."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO performance_baselines 
                (metric_name, baseline_value, baseline_std, baseline_samples,
                 confidence_lower, confidence_upper, created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                baseline.metric_name, baseline.baseline_value, baseline.baseline_std,
                baseline.baseline_samples, baseline.confidence_interval[0],
                baseline.confidence_interval[1], baseline.created_at,
                baseline.updated_at, baseline.version
            ))
            conn.commit()
    
    def get_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get performance baseline for metric."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM performance_baselines WHERE metric_name = ?
            """, (metric_name,))
            
            row = cursor.fetchone()
            if row:
                return PerformanceBaseline(
                    metric_name=row[0],
                    baseline_value=row[1],
                    baseline_std=row[2],
                    baseline_samples=row[3],
                    confidence_interval=(row[4], row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    version=row[8]
                )
            return None
    
    def store_alert(self, alert: RegressionAlert):
        """Store regression alert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO regression_alerts 
                (alert_id, metric_name, severity, current_value, baseline_value,
                 degradation_percent, detection_method, confidence, timestamp,
                 context, remediation_suggestions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.metric_name, alert.severity.value,
                alert.current_value, alert.baseline_value, alert.degradation_percent,
                alert.detection_method.value, alert.confidence, alert.timestamp,
                json.dumps(alert.context), json.dumps(alert.remediation_suggestions)
            ))
            conn.commit()


class StatisticalDetectors:
    """Statistical methods for regression detection."""
    
    @staticmethod
    def threshold_based_detection(
        current_value: float,
        baseline: PerformanceBaseline,
        threshold_percent: float = 15.0
    ) -> Tuple[bool, float, RegressionSeverity]:
        """Threshold-based regression detection."""
        # Calculate percentage change
        change_percent = abs((current_value - baseline.baseline_value) / baseline.baseline_value) * 100
        
        # Determine if this is a regression (assuming higher values are worse for latency metrics)
        is_regression = current_value > baseline.baseline_value
        
        if not is_regression:
            return False, change_percent, RegressionSeverity.NONE
        
        # Classify severity
        if change_percent >= 50:
            severity = RegressionSeverity.CRITICAL
        elif change_percent >= 30:
            severity = RegressionSeverity.MAJOR
        elif change_percent >= 15:
            severity = RegressionSeverity.MODERATE
        elif change_percent >= 5:
            severity = RegressionSeverity.MINOR
        else:
            severity = RegressionSeverity.NONE
            is_regression = False
        
        return is_regression and change_percent >= threshold_percent, change_percent, severity
    
    @staticmethod
    def statistical_test_detection(
        recent_values: List[float],
        baseline_values: List[float],
        alpha: float = 0.05
    ) -> Tuple[bool, float, float]:
        """Statistical test-based regression detection using t-test."""
        if len(recent_values) < 3 or len(baseline_values) < 3:
            return False, 0.0, 0.0
        
        try:
            # Perform independent t-test
            t_stat, p_value = stats.ttest_ind(recent_values, baseline_values)
            
            # Check if means are significantly different and recent values are higher
            is_regression = (p_value < alpha and 
                           statistics.mean(recent_values) > statistics.mean(baseline_values))
            
            confidence = 1.0 - p_value
            effect_size = abs(t_stat) / np.sqrt(len(recent_values) + len(baseline_values))
            
            return is_regression, confidence, effect_size
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return False, 0.0, 0.0
    
    @staticmethod
    def trend_analysis_detection(
        values: List[Tuple[datetime, float]],
        min_samples: int = 10
    ) -> TrendAnalysis:
        """Trend analysis using linear regression."""
        if len(values) < min_samples:
            return TrendAnalysis(
                metric_name="unknown",
                trend_direction="insufficient_data",
                trend_slope=0.0,
                trend_confidence=0.0,
                forecast_7d=0.0,
                forecast_30d=0.0,
                analysis_timestamp=datetime.utcnow(),
                samples_analyzed=len(values)
            )
        
        # Convert to numerical arrays
        timestamps = np.array([(ts - values[0][0]).total_seconds() / 86400 for ts, _ in values])  # Days
        performance_values = np.array([val for _, val in values])
        
        try:
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, performance_values)
            
            # Determine trend direction
            if abs(slope) < std_err:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "degrading"  # Assuming higher values are worse
            else:
                trend_direction = "improving"
            
            # Calculate confidence
            confidence = abs(r_value)
            
            # Forecast future values
            max_days = max(timestamps)
            forecast_7d = intercept + slope * (max_days + 7)
            forecast_30d = intercept + slope * (max_days + 30)
            
            return TrendAnalysis(
                metric_name="unknown",
                trend_direction=trend_direction,
                trend_slope=slope,
                trend_confidence=confidence,
                forecast_7d=forecast_7d,
                forecast_30d=forecast_30d,
                analysis_timestamp=datetime.utcnow(),
                samples_analyzed=len(values)
            )
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return TrendAnalysis(
                metric_name="unknown",
                trend_direction="error",
                trend_slope=0.0,
                trend_confidence=0.0,
                forecast_7d=0.0,
                forecast_30d=0.0,
                analysis_timestamp=datetime.utcnow(),
                samples_analyzed=len(values)
            )
    
    @staticmethod
    def change_point_detection(
        values: List[float],
        window_size: int = 10
    ) -> Tuple[bool, int, float]:
        """Simple change point detection using moving averages."""
        if len(values) < window_size * 2:
            return False, -1, 0.0
        
        max_change = 0.0
        change_point = -1
        
        for i in range(window_size, len(values) - window_size):
            # Calculate averages before and after point
            before_avg = statistics.mean(values[i-window_size:i])
            after_avg = statistics.mean(values[i:i+window_size])
            
            # Calculate relative change
            if before_avg > 0:
                change = abs((after_avg - before_avg) / before_avg)
                
                if change > max_change:
                    max_change = change
                    change_point = i
        
        # Significant change threshold
        is_change_point = max_change > 0.2  # 20% change
        
        return is_change_point, change_point, max_change


class PerformanceRegressionDetector:
    """Main regression detection system."""
    
    def __init__(self, db_path: str = "performance_history.db"):
        self.db = PerformanceDatabase(db_path)
        self.detectors = StatisticalDetectors()
        self.active_alerts: Dict[str, RegressionAlert] = {}
        
        # Detection thresholds
        self.thresholds = {
            RegressionSeverity.MINOR: 5.0,
            RegressionSeverity.MODERATE: 15.0,
            RegressionSeverity.MAJOR: 30.0,
            RegressionSeverity.CRITICAL: 50.0
        }
    
    def update_baseline(
        self,
        metric_name: str,
        historical_values: List[float] = None,
        days_back: int = 30,
        min_samples: int = 20
    ) -> PerformanceBaseline:
        """Update performance baseline for metric."""
        if historical_values is None:
            # Get historical data from database
            history = self.db.get_performance_history(metric_name, days=days_back)
            historical_values = [value for _, value in history]
        
        if len(historical_values) < min_samples:
            logger.warning(f"Insufficient data for baseline: {len(historical_values)} samples < {min_samples}")
            # Use default baseline if available
            existing = self.db.get_baseline(metric_name)
            if existing:
                return existing
            raise ValueError(f"Insufficient data for baseline: {metric_name}")
        
        # Calculate baseline statistics
        baseline_value = statistics.median(historical_values)  # Use median for robustness
        baseline_std = statistics.stdev(historical_values)
        
        # Calculate confidence interval (95%)
        confidence_margin = 1.96 * baseline_std / np.sqrt(len(historical_values))
        confidence_interval = (
            baseline_value - confidence_margin,
            baseline_value + confidence_margin
        )
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            baseline_std=baseline_std,
            baseline_samples=len(historical_values),
            confidence_interval=confidence_interval,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store in database
        self.db.store_baseline(baseline)
        
        logger.info(f"Updated baseline for {metric_name}: {baseline_value:.2f} ± {baseline_std:.2f}")
        
        return baseline
    
    def detect_regression(
        self,
        metric_name: str,
        current_value: float,
        detection_method: DetectionMethod = DetectionMethod.ENSEMBLE
    ) -> Optional[RegressionAlert]:
        """Detect performance regression for metric."""
        # Get baseline
        baseline = self.db.get_baseline(metric_name)
        if not baseline:
            logger.warning(f"No baseline found for {metric_name}, creating one...")
            try:
                baseline = self.update_baseline(metric_name)
            except ValueError:
                logger.error(f"Cannot create baseline for {metric_name}")
                return None
        
        # Check if baseline is stale
        if baseline.is_stale():
            logger.info(f"Baseline for {metric_name} is stale, updating...")
            baseline = self.update_baseline(metric_name)
        
        # Store current metric
        self.db.store_performance_metric(metric_name, current_value)
        
        alert = None
        
        if detection_method == DetectionMethod.THRESHOLD_BASED:
            alert = self._threshold_based_detection(metric_name, current_value, baseline)
        
        elif detection_method == DetectionMethod.STATISTICAL_TEST:
            alert = self._statistical_test_detection(metric_name, current_value, baseline)
        
        elif detection_method == DetectionMethod.TREND_ANALYSIS:
            alert = self._trend_analysis_detection(metric_name, current_value)
        
        elif detection_method == DetectionMethod.CHANGE_POINT:
            alert = self._change_point_detection(metric_name, current_value)
        
        elif detection_method == DetectionMethod.ENSEMBLE:
            alert = self._ensemble_detection(metric_name, current_value, baseline)
        
        if alert:
            # Generate remediation suggestions
            alert.remediation_suggestions = self._generate_remediation_suggestions(alert)
            
            # Store alert
            self.db.store_alert(alert)
            self.active_alerts[alert.alert_id] = alert
            
            logger.warning(f"Regression detected: {alert.metric_name} - {alert.severity.value} "
                          f"({alert.degradation_percent:.1f}% degradation)")
        
        return alert
    
    def _threshold_based_detection(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline
    ) -> Optional[RegressionAlert]:
        """Threshold-based regression detection."""
        is_regression, degradation_percent, severity = self.detectors.threshold_based_detection(
            current_value, baseline, threshold_percent=self.thresholds[RegressionSeverity.MINOR]
        )
        
        if is_regression:
            return RegressionAlert(
                alert_id=f"{metric_name}_{int(time.time())}",
                metric_name=metric_name,
                severity=severity,
                current_value=current_value,
                baseline_value=baseline.baseline_value,
                degradation_percent=degradation_percent,
                detection_method=DetectionMethod.THRESHOLD_BASED,
                confidence=0.8,  # Fixed confidence for threshold method
                timestamp=datetime.utcnow(),
                context={
                    'baseline_std': baseline.baseline_std,
                    'baseline_samples': baseline.baseline_samples
                }
            )
        
        return None
    
    def _statistical_test_detection(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline
    ) -> Optional[RegressionAlert]:
        """Statistical test-based regression detection."""
        # Get recent values for comparison
        recent_history = self.db.get_performance_history(metric_name, days=7, limit=10)
        recent_values = [value for _, value in recent_history]
        
        if len(recent_values) < 3:
            return None
        
        # Get baseline values
        baseline_history = self.db.get_performance_history(metric_name, days=30, limit=30)
        baseline_values = [value for _, value in baseline_history[-20:]]  # Use older data
        
        is_regression, confidence, effect_size = self.detectors.statistical_test_detection(
            recent_values, baseline_values
        )
        
        if is_regression and confidence > 0.95:
            # Calculate degradation percentage
            recent_mean = statistics.mean(recent_values)
            baseline_mean = statistics.mean(baseline_values)
            degradation_percent = ((recent_mean - baseline_mean) / baseline_mean) * 100
            
            # Determine severity based on degradation
            if degradation_percent >= 50:
                severity = RegressionSeverity.CRITICAL
            elif degradation_percent >= 30:
                severity = RegressionSeverity.MAJOR
            elif degradation_percent >= 15:
                severity = RegressionSeverity.MODERATE
            else:
                severity = RegressionSeverity.MINOR
            
            return RegressionAlert(
                alert_id=f"{metric_name}_stat_{int(time.time())}",
                metric_name=metric_name,
                severity=severity,
                current_value=current_value,
                baseline_value=baseline.baseline_value,
                degradation_percent=degradation_percent,
                detection_method=DetectionMethod.STATISTICAL_TEST,
                confidence=confidence,
                timestamp=datetime.utcnow(),
                context={
                    'effect_size': effect_size,
                    'recent_samples': len(recent_values),
                    'baseline_samples': len(baseline_values)
                }
            )
        
        return None
    
    def _trend_analysis_detection(
        self,
        metric_name: str,
        current_value: float
    ) -> Optional[RegressionAlert]:
        """Trend analysis-based regression detection."""
        history = self.db.get_performance_history(metric_name, days=14)
        
        if len(history) < 10:
            return None
        
        trend = self.detectors.trend_analysis_detection(history)
        
        # Alert if trend is degrading with high confidence
        if (trend.trend_direction == "degrading" and 
            trend.trend_confidence > 0.7 and
            trend.trend_slope > 0):
            
            # Estimate degradation based on trend
            baseline_value = history[-1][1] - (trend.trend_slope * 7)  # 7 days ago
            degradation_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # Determine severity
            if degradation_percent >= 30:
                severity = RegressionSeverity.MAJOR
            elif degradation_percent >= 15:
                severity = RegressionSeverity.MODERATE
            else:
                severity = RegressionSeverity.MINOR
            
            return RegressionAlert(
                alert_id=f"{metric_name}_trend_{int(time.time())}",
                metric_name=metric_name,
                severity=severity,
                current_value=current_value,
                baseline_value=baseline_value,
                degradation_percent=degradation_percent,
                detection_method=DetectionMethod.TREND_ANALYSIS,
                confidence=trend.trend_confidence,
                timestamp=datetime.utcnow(),
                context={
                    'trend_slope': trend.trend_slope,
                    'forecast_7d': trend.forecast_7d,
                    'forecast_30d': trend.forecast_30d
                }
            )
        
        return None
    
    def _change_point_detection(
        self,
        metric_name: str,
        current_value: float
    ) -> Optional[RegressionAlert]:
        """Change point-based regression detection."""
        history = self.db.get_performance_history(metric_name, days=30, limit=50)
        values = [value for _, value in history]
        
        if len(values) < 20:
            return None
        
        is_change_point, change_index, change_magnitude = self.detectors.change_point_detection(values)
        
        if is_change_point and change_magnitude > 0.15:  # 15% change
            # Calculate degradation from change point
            before_values = values[:change_index]
            after_values = values[change_index:]
            
            before_avg = statistics.mean(before_values)
            after_avg = statistics.mean(after_values)
            
            degradation_percent = ((after_avg - before_avg) / before_avg) * 100
            
            if degradation_percent > 5:  # Only alert for degradation
                severity = RegressionSeverity.MODERATE
                if degradation_percent >= 30:
                    severity = RegressionSeverity.MAJOR
                
                return RegressionAlert(
                    alert_id=f"{metric_name}_change_{int(time.time())}",
                    metric_name=metric_name,
                    severity=severity,
                    current_value=current_value,
                    baseline_value=before_avg,
                    degradation_percent=degradation_percent,
                    detection_method=DetectionMethod.CHANGE_POINT,
                    confidence=change_magnitude,
                    timestamp=datetime.utcnow(),
                    context={
                        'change_point_index': change_index,
                        'change_magnitude': change_magnitude,
                        'samples_before': len(before_values),
                        'samples_after': len(after_values)
                    }
                )
        
        return None
    
    def _ensemble_detection(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline
    ) -> Optional[RegressionAlert]:
        """Ensemble method combining multiple detection approaches."""
        alerts = []
        
        # Try each detection method
        threshold_alert = self._threshold_based_detection(metric_name, current_value, baseline)
        if threshold_alert:
            alerts.append(threshold_alert)
        
        stat_alert = self._statistical_test_detection(metric_name, current_value, baseline)
        if stat_alert:
            alerts.append(stat_alert)
        
        trend_alert = self._trend_analysis_detection(metric_name, current_value)
        if trend_alert:
            alerts.append(trend_alert)
        
        # If multiple methods agree, create ensemble alert
        if len(alerts) >= 2:
            # Use the most severe alert as base
            base_alert = max(alerts, key=lambda a: list(RegressionSeverity).index(a.severity))
            
            # Calculate ensemble confidence
            ensemble_confidence = sum(a.confidence for a in alerts) / len(alerts)
            
            ensemble_alert = RegressionAlert(
                alert_id=f"{metric_name}_ensemble_{int(time.time())}",
                metric_name=base_alert.metric_name,
                severity=base_alert.severity,
                current_value=base_alert.current_value,
                baseline_value=base_alert.baseline_value,
                degradation_percent=base_alert.degradation_percent,
                detection_method=DetectionMethod.ENSEMBLE,
                confidence=ensemble_confidence,
                timestamp=datetime.utcnow(),
                context={
                    'methods_agreed': len(alerts),
                    'detection_methods': [a.detection_method.value for a in alerts],
                    'individual_confidences': [a.confidence for a in alerts]
                }
            )
            
            return ensemble_alert
        
        # If only one method detected, use it if confidence is high
        elif len(alerts) == 1 and alerts[0].confidence > 0.8:
            return alerts[0]
        
        return None
    
    def _generate_remediation_suggestions(self, alert: RegressionAlert) -> List[str]:
        """Generate remediation suggestions based on alert context."""
        suggestions = []
        
        metric = alert.metric_name.lower()
        severity = alert.severity
        
        # General suggestions based on metric type
        if "search" in metric and "latency" in metric:
            suggestions.extend([
                "🔍 Review search index configuration and optimization",
                "📊 Check query complexity and filtering efficiency",
                "💾 Verify vector index (HNSW/IVFFlat) performance",
                "🔧 Consider increasing search timeout thresholds",
                "📈 Monitor database connection pool utilization"
            ])
        
        elif "ingestion" in metric and "throughput" in metric:
            suggestions.extend([
                "📥 Review batch processing configuration",
                "⚡ Check embedding service performance",
                "🔄 Verify database write performance",
                "🏗️ Consider increasing ingestion parallelism",
                "💾 Monitor disk I/O and storage performance"
            ])
        
        elif "compression" in metric:
            suggestions.extend([
                "🗜️ Review compression algorithm efficiency",
                "🧮 Check semantic clustering parameters",
                "📊 Verify context importance scoring",
                "⚙️ Consider compression timeout adjustments",
                "🔧 Monitor compression memory usage"
            ])
        
        elif "knowledge" in metric:
            suggestions.extend([
                "🤝 Review agent knowledge caching strategy",
                "📊 Check knowledge graph query optimization",
                "🔄 Verify cross-agent communication efficiency",
                "💾 Monitor knowledge database performance",
                "⚡ Consider knowledge prefetching"
            ])
        
        # Severity-specific suggestions
        if severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
            suggestions.extend([
                "🚨 Immediate investigation required",
                "📱 Alert on-call team",
                "🔧 Consider emergency performance tuning",
                "📊 Enable detailed performance profiling",
                "🔄 Consider rollback if recent deployment"
            ])
        
        # Detection method specific suggestions
        if alert.detection_method == DetectionMethod.TREND_ANALYSIS:
            suggestions.append("📈 Monitor trend forecast and plan capacity scaling")
        
        elif alert.detection_method == DetectionMethod.CHANGE_POINT:
            suggestions.append("🔍 Investigate changes around detected change point")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def analyze_performance_trends(self, metric_name: str, days: int = 30) -> TrendAnalysis:
        """Analyze performance trends for metric."""
        history = self.db.get_performance_history(metric_name, days=days)
        
        trend = self.detectors.trend_analysis_detection(history)
        trend.metric_name = metric_name
        
        return trend
    
    def get_active_alerts(self, severity_filter: RegressionSeverity = None) -> List[RegressionAlert]:
        """Get active regression alerts."""
        if severity_filter:
            return [alert for alert in self.active_alerts.values() 
                   if alert.severity == severity_filter]
        return list(self.active_alerts.values())
    
    def generate_regression_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive regression analysis report."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all metrics with recent data
        metrics_analyzed = set()
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.active_alerts.values() 
                        if alert.timestamp >= since_date]
        
        # Analyze trends for key metrics
        key_metrics = [
            "search_latency_p95_ms",
            "ingestion_throughput_docs_per_sec",
            "context_compression_time_ms",
            "knowledge_sharing_latency_p95_ms"
        ]
        
        trend_analyses = {}
        for metric in key_metrics:
            try:
                trend = self.analyze_performance_trends(metric, days=days)
                trend_analyses[metric] = trend
                metrics_analyzed.add(metric)
            except Exception as e:
                logger.warning(f"Failed to analyze trends for {metric}: {e}")
        
        # Calculate summary statistics
        alert_by_severity = defaultdict(int)
        alert_by_method = defaultdict(int)
        
        for alert in recent_alerts:
            alert_by_severity[alert.severity.value] += 1
            alert_by_method[alert.detection_method.value] += 1
        
        # Overall system health assessment
        critical_alerts = len([a for a in recent_alerts if a.severity == RegressionSeverity.CRITICAL])
        major_alerts = len([a for a in recent_alerts if a.severity == RegressionSeverity.MAJOR])
        
        if critical_alerts > 0:
            health_status = "CRITICAL"
        elif major_alerts > 2:
            health_status = "DEGRADED"
        elif len(recent_alerts) > 5:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
        
        return {
            'report_timestamp': datetime.utcnow().isoformat(),
            'analysis_period_days': days,
            'health_status': health_status,
            'metrics_analyzed': len(metrics_analyzed),
            'alerts_summary': {
                'total_alerts': len(recent_alerts),
                'by_severity': dict(alert_by_severity),
                'by_detection_method': dict(alert_by_method)
            },
            'trend_analyses': {
                metric: {
                    'trend_direction': trend.trend_direction,
                    'trend_confidence': trend.trend_confidence,
                    'forecast_7d': trend.forecast_7d,
                    'forecast_30d': trend.forecast_30d
                }
                for metric, trend in trend_analyses.items()
            },
            'recent_alerts': [
                {
                    'metric_name': alert.metric_name,
                    'severity': alert.severity.value,
                    'degradation_percent': alert.degradation_percent,
                    'detection_method': alert.detection_method.value,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in recent_alerts
            ],
            'recommendations': self._generate_system_recommendations(
                recent_alerts, trend_analyses, health_status
            )
        }
    
    def _generate_system_recommendations(
        self,
        recent_alerts: List[RegressionAlert],
        trend_analyses: Dict[str, TrendAnalysis],
        health_status: str
    ) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        if health_status == "CRITICAL":
            recommendations.extend([
                "🚨 CRITICAL: Immediate performance investigation required",
                "📱 Escalate to senior engineering team",
                "🔄 Consider emergency rollback procedures",
                "📊 Enable maximum performance monitoring"
            ])
        
        elif health_status == "DEGRADED":
            recommendations.extend([
                "⚠️ Performance degradation detected across multiple metrics",
                "🔍 Investigate recent system changes",
                "📈 Increase monitoring frequency",
                "🔧 Plan performance optimization sprint"
            ])
        
        # Trend-based recommendations
        degrading_trends = [metric for metric, trend in trend_analyses.items() 
                           if trend.trend_direction == "degrading"]
        
        if len(degrading_trends) > 2:
            recommendations.append(
                f"📉 Multiple degrading trends detected: {', '.join(degrading_trends)}"
            )
        
        # Alert pattern recommendations
        if len(recent_alerts) > 10:
            recommendations.append(
                "🔔 High alert volume - consider adjusting detection thresholds"
            )
        
        return recommendations