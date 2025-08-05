"""
Alert Analysis Engine - Phase 3 Intelligence Layer Implementation

This module implements pattern recognition and intelligent alert analysis
to improve signal-to-noise ratio and developer experience.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from statistics import mean, stdev

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from app.core.database import get_db
from app.models.alert import Alert, AlertPattern, AlertFrequency
from app.core.redis import get_redis


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for intelligent classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PatternType(Enum):
    """Types of alert patterns that can be detected"""
    RECURRING = "recurring"
    ESCALATING = "escalating"
    BURST = "burst"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"


@dataclass
class AlertMetrics:
    """Metrics for alert analysis and pattern detection"""
    frequency: float = 0.0
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    response_time_avg: float = 0.0
    resolution_rate: float = 0.0
    peak_hours: List[int] = field(default_factory=list)
    correlation_score: float = 0.0


@dataclass
class AlertPattern:
    """Detected alert pattern with metadata"""
    pattern_id: str
    pattern_type: PatternType
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    related_alerts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTrend:
    """Performance trend analysis data"""
    metric_name: str
    baseline_value: float
    current_value: float
    trend_direction: str  # "improving", "degrading", "stable"
    change_percentage: float
    confidence: float
    anomaly_detected: bool = False


class AlertAnalysisEngine:
    """
    Intelligence layer for alert analysis and pattern recognition.
    
    Provides:
    - Historical alert pattern analysis
    - Basic pattern recognition for similar alerts
    - Performance trend detection
    - Alert frequency tracking
    """
    
    def __init__(self, redis_client: redis.Redis, db_session: AsyncSession):
        self.redis = redis_client
        self.db = db_session
        self.alert_history = deque(maxlen=10000)  # Rolling window of alerts
        self.patterns: Dict[str, AlertPattern] = {}
        self.performance_baselines: Dict[str, float] = {}
        self.frequency_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.pattern_detection_window = timedelta(hours=24)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.min_pattern_occurrences = 3
        
    async def initialize(self) -> None:
        """Initialize the alert analysis engine"""
        try:
            # Load recent alert history
            await self._load_alert_history()
            
            # Load existing patterns
            await self._load_patterns()
            
            # Establish performance baselines
            await self._establish_baselines()
            
            logger.info("Alert Analysis Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alert Analysis Engine: {e}")
            raise
    
    async def analyze_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze incoming alert and return enriched data with patterns
        
        Args:
            alert_data: Raw alert data
            
        Returns:
            Enriched alert data with analysis results
        """
        try:
            # Add to history
            alert_timestamp = datetime.fromisoformat(alert_data.get('timestamp', datetime.now().isoformat()))
            self.alert_history.append({
                'timestamp': alert_timestamp,
                'type': alert_data.get('type', 'unknown'),
                'severity': alert_data.get('severity', 'medium'),
                'message': alert_data.get('message', ''),
                'metadata': alert_data.get('metadata', {})
            })
            
            # Update frequency tracking
            alert_type = alert_data.get('type', 'unknown')
            self.frequency_tracker[alert_type].append(alert_timestamp)
            
            # Analyze patterns
            patterns = await self._detect_patterns(alert_data)
            
            # Check for anomalies
            anomaly_score = await self._calculate_anomaly_score(alert_data)
            
            # Calculate priority based on patterns and frequency
            priority_score = await self._calculate_priority(alert_data, patterns)
            
            # Enrich alert data
            enriched_data = {
                **alert_data,
                'analysis': {
                    'patterns': [p.__dict__ for p in patterns],
                    'anomaly_score': anomaly_score,
                    'priority_score': priority_score,
                    'frequency_rank': await self._get_frequency_rank(alert_type),
                    'similar_alerts': await self._find_similar_alerts(alert_data),
                    'recommended_action': await self._recommend_action(alert_data, patterns)
                }
            }
            
            # Store analysis results
            await self._store_analysis(enriched_data)
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error analyzing alert: {e}")
            return {**alert_data, 'analysis_error': str(e)}
    
    async def get_alert_metrics(self, time_window: timedelta = None) -> AlertMetrics:
        """Get comprehensive alert metrics for the specified time window"""
        if time_window is None:
            time_window = timedelta(hours=24)
            
        cutoff_time = datetime.now() - time_window
        relevant_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
        
        if not relevant_alerts:
            return AlertMetrics()
        
        # Calculate metrics
        frequency = len(relevant_alerts) / time_window.total_seconds() * 3600  # per hour
        
        severity_dist = defaultdict(int)
        response_times = []
        
        for alert in relevant_alerts:
            severity_dist[alert['severity']] += 1
            # Mock response time calculation (in real system, track actual response times)
            response_times.append(alert.get('response_time', 300))  # 5 min default
        
        peak_hours = await self._identify_peak_hours(relevant_alerts)
        
        return AlertMetrics(
            frequency=frequency,
            severity_distribution=dict(severity_dist),
            response_time_avg=mean(response_times) if response_times else 0,
            resolution_rate=0.85,  # Mock value, calculate from actual data
            peak_hours=peak_hours,
            correlation_score=await self._calculate_correlation_score(relevant_alerts)
        )
    
    async def detect_performance_trends(self, metrics: Dict[str, float]) -> List[PerformanceTrend]:
        """Detect performance trends and anomalies"""
        trends = []
        
        for metric_name, current_value in metrics.items():
            baseline = self.performance_baselines.get(metric_name)
            
            if baseline is None:
                # Establish baseline
                self.performance_baselines[metric_name] = current_value
                continue
            
            # Calculate trend
            change_pct = ((current_value - baseline) / baseline) * 100 if baseline != 0 else 0
            
            # Determine trend direction
            if abs(change_pct) < 5:
                direction = "stable"
            elif change_pct > 0:
                direction = "degrading"  # Assuming higher metrics are worse
            else:
                direction = "improving"
            
            # Detect anomalies
            anomaly_detected = abs(change_pct) > 20  # 20% change threshold
            
            # Calculate confidence based on data history
            confidence = min(0.95, len(self.alert_history) / 1000)
            
            trends.append(PerformanceTrend(
                metric_name=metric_name,
                baseline_value=baseline,
                current_value=current_value,
                trend_direction=direction,
                change_percentage=change_pct,
                confidence=confidence,
                anomaly_detected=anomaly_detected
            ))
            
            # Update baseline with exponential moving average
            alpha = 0.1  # Smoothing factor
            self.performance_baselines[metric_name] = alpha * current_value + (1 - alpha) * baseline
        
        return trends
    
    async def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about detected patterns"""
        insights = {
            'total_patterns': len(self.patterns),
            'pattern_types': defaultdict(int),
            'high_confidence_patterns': [],
            'recurring_issues': [],
            'recommendations': []
        }
        
        for pattern in self.patterns.values():
            insights['pattern_types'][pattern.pattern_type.value] += 1
            
            if pattern.confidence > 0.8:
                insights['high_confidence_patterns'].append({
                    'id': pattern.pattern_id,
                    'type': pattern.pattern_type.value,
                    'confidence': pattern.confidence,
                    'frequency': pattern.frequency
                })
            
            if pattern.pattern_type == PatternType.RECURRING and pattern.frequency > 10:
                insights['recurring_issues'].append({
                    'pattern_id': pattern.pattern_id,
                    'frequency': pattern.frequency,
                    'impact': 'high' if pattern.frequency > 50 else 'medium'
                })
        
        # Generate recommendations
        insights['recommendations'] = await self._generate_recommendations(insights)
        
        return insights
    
    # Private methods
    
    async def _load_alert_history(self) -> None:
        """Load recent alert history from database"""
        try:
            # Mock implementation - in real system, load from database
            cutoff_time = datetime.now() - timedelta(days=7)
            
            # For now, initialize with empty history
            # In production, query actual alert table
            pass
            
        except Exception as e:
            logger.error(f"Error loading alert history: {e}")
    
    async def _load_patterns(self) -> None:
        """Load existing patterns from storage"""
        try:
            # Load patterns from Redis
            pattern_keys = await self.redis.keys("alert_pattern:*")
            
            for key in pattern_keys:
                pattern_data = await self.redis.hgetall(key)
                if pattern_data:
                    pattern_id = key.decode().split(":")[-1]
                    self.patterns[pattern_id] = AlertPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType(pattern_data.get(b'pattern_type', b'recurring').decode()),
                        confidence=float(pattern_data.get(b'confidence', b'0.5')),
                        frequency=int(pattern_data.get(b'frequency', b'1')),
                        first_seen=datetime.fromisoformat(pattern_data.get(b'first_seen', datetime.now().isoformat()).decode()),
                        last_seen=datetime.fromisoformat(pattern_data.get(b'last_seen', datetime.now().isoformat()).decode()),
                        related_alerts=json.loads(pattern_data.get(b'related_alerts', b'[]')),
                        metadata=json.loads(pattern_data.get(b'metadata', b'{}'))
                    )
                    
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    async def _establish_baselines(self) -> None:
        """Establish performance baselines from historical data"""
        try:
            # Mock baseline establishment
            # In production, calculate from historical performance data
            self.performance_baselines = {
                'response_time': 150.0,  # ms
                'error_rate': 0.02,      # 2%
                'cpu_usage': 45.0,       # %
                'memory_usage': 60.0,    # %
                'throughput': 1000.0     # req/s
            }
            
        except Exception as e:
            logger.error(f"Error establishing baselines: {e}")
    
    async def _detect_patterns(self, alert_data: Dict[str, Any]) -> List[AlertPattern]:
        """Detect patterns in alert data"""
        patterns = []
        alert_type = alert_data.get('type', 'unknown')
        
        # Check for recurring patterns
        if alert_type in self.frequency_tracker:
            recent_occurrences = [
                ts for ts in self.frequency_tracker[alert_type]
                if datetime.now() - ts < self.pattern_detection_window
            ]
            
            if len(recent_occurrences) >= self.min_pattern_occurrences:
                pattern_id = f"recurring_{alert_type}_{len(recent_occurrences)}"
                
                if pattern_id not in self.patterns:
                    pattern = AlertPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.RECURRING,
                        confidence=min(0.95, len(recent_occurrences) / 10),
                        frequency=len(recent_occurrences),
                        first_seen=min(recent_occurrences),
                        last_seen=max(recent_occurrences),
                        related_alerts=[alert_type],
                        metadata={'window_hours': self.pattern_detection_window.total_seconds() / 3600}
                    )
                    
                    self.patterns[pattern_id] = pattern
                    patterns.append(pattern)
        
        return patterns
    
    async def _calculate_anomaly_score(self, alert_data: Dict[str, Any]) -> float:
        """Calculate anomaly score for alert"""
        # Simple anomaly detection based on frequency
        alert_type = alert_data.get('type', 'unknown')
        
        if alert_type not in self.frequency_tracker:
            return 0.5  # Unknown alert type
        
        recent_count = len([
            ts for ts in self.frequency_tracker[alert_type]
            if datetime.now() - ts < timedelta(hours=1)
        ])
        
        # Calculate z-score based on historical frequency
        historical_counts = []
        for i in range(24):  # Last 24 hours
            hour_start = datetime.now() - timedelta(hours=i+1)
            hour_end = datetime.now() - timedelta(hours=i)
            
            count = len([
                ts for ts in self.frequency_tracker[alert_type]
                if hour_start <= ts < hour_end
            ])
            historical_counts.append(count)
        
        if len(historical_counts) < 2:
            return 0.5
        
        mean_count = mean(historical_counts)
        std_count = stdev(historical_counts) if len(historical_counts) > 1 else 1
        
        if std_count == 0:
            return 0.5
        
        z_score = abs(recent_count - mean_count) / std_count
        return min(1.0, z_score / self.anomaly_threshold)
    
    async def _calculate_priority(self, alert_data: Dict[str, Any], patterns: List[AlertPattern]) -> float:
        """Calculate priority score for alert"""
        base_priority = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }.get(alert_data.get('severity', 'medium'), 0.6)
        
        # Adjust based on patterns
        pattern_adjustment = 0.0
        for pattern in patterns:
            if pattern.pattern_type == PatternType.RECURRING:
                pattern_adjustment += 0.2 * pattern.confidence
            elif pattern.pattern_type == PatternType.ESCALATING:
                pattern_adjustment += 0.3 * pattern.confidence
        
        return min(1.0, base_priority + pattern_adjustment)
    
    async def _get_frequency_rank(self, alert_type: str) -> str:
        """Get frequency ranking for alert type"""
        if alert_type not in self.frequency_tracker:
            return "rare"
        
        recent_count = len([
            ts for ts in self.frequency_tracker[alert_type]
            if datetime.now() - ts < timedelta(hours=24)
        ])
        
        if recent_count > 50:
            return "very_frequent"
        elif recent_count > 20:
            return "frequent"
        elif recent_count > 5:
            return "common"
        else:
            return "rare"
    
    async def _find_similar_alerts(self, alert_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar alerts in history"""
        similar = []
        alert_message = alert_data.get('message', '').lower()
        alert_type = alert_data.get('type', 'unknown')
        
        for alert in list(self.alert_history)[-100:]:  # Check last 100 alerts
            if alert['type'] == alert_type:
                # Simple similarity check
                message_similarity = len(set(alert_message.split()) & set(alert['message'].lower().split()))
                if message_similarity > 2:  # At least 2 common words
                    similar.append({
                        'timestamp': alert['timestamp'].isoformat(),
                        'message': alert['message'][:100],  # Truncate
                        'similarity_score': message_similarity / max(len(alert_message.split()), len(alert['message'].split()))
                    })
        
        return sorted(similar, key=lambda x: x['similarity_score'], reverse=True)[:5]
    
    async def _recommend_action(self, alert_data: Dict[str, Any], patterns: List[AlertPattern]) -> str:
        """Recommend action based on alert and patterns"""
        severity = alert_data.get('severity', 'medium')
        alert_type = alert_data.get('type', 'unknown')
        
        # Check for recurring patterns
        recurring_patterns = [p for p in patterns if p.pattern_type == PatternType.RECURRING]
        
        if recurring_patterns:
            return f"Investigate recurring {alert_type} pattern - consider automated resolution"
        
        if severity in ['critical', 'high']:
            return "Immediate attention required - escalate to on-call engineer"
        elif severity == 'medium':
            return "Monitor closely - investigate if pattern emerges"
        else:
            return "Log for trend analysis - no immediate action required"
    
    async def _identify_peak_hours(self, alerts: List[Dict[str, Any]]) -> List[int]:
        """Identify peak alert hours"""
        hour_counts = defaultdict(int)
        
        for alert in alerts:
            hour = alert['timestamp'].hour
            hour_counts[hour] += 1
        
        if not hour_counts:
            return []
        
        max_count = max(hour_counts.values())
        threshold = max_count * 0.8  # 80% of peak
        
        return [hour for hour, count in hour_counts.items() if count >= threshold]
    
    async def _calculate_correlation_score(self, alerts: List[Dict[str, Any]]) -> float:
        """Calculate correlation score between different alert types"""
        if len(alerts) < 2:
            return 0.0
        
        # Simple correlation based on temporal proximity
        correlation_events = 0
        total_pairs = 0
        
        for i, alert1 in enumerate(alerts[:-1]):
            for alert2 in alerts[i+1:i+6]:  # Check next 5 alerts
                total_pairs += 1
                time_diff = abs((alert2['timestamp'] - alert1['timestamp']).total_seconds())
                
                if time_diff < 300:  # Within 5 minutes
                    correlation_events += 1
        
        return correlation_events / total_pairs if total_pairs > 0 else 0.0
    
    async def _store_analysis(self, enriched_data: Dict[str, Any]) -> None:
        """Store analysis results"""
        try:
            # Store in Redis for quick access
            alert_id = enriched_data.get('id', 'unknown')
            await self.redis.setex(
                f"alert_analysis:{alert_id}",
                3600,  # 1 hour TTL
                json.dumps(enriched_data['analysis'], default=str)
            )
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")
    
    async def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        if insights['total_patterns'] > 50:
            recommendations.append("Consider implementing automated responses for recurring patterns")
        
        recurring_count = insights['pattern_types'].get('recurring', 0)
        if recurring_count > 10:
            recommendations.append(f"High number of recurring patterns ({recurring_count}) - investigate root causes")
        
        if len(insights['high_confidence_patterns']) > 5:
            recommendations.append("Multiple high-confidence patterns detected - prioritize pattern-based alerting")
        
        if not recommendations:
            recommendations.append("Alert patterns are within normal ranges - continue monitoring")
        
        return recommendations


# Factory function for dependency injection
async def create_alert_analysis_engine() -> AlertAnalysisEngine:
    """Create and initialize Alert Analysis Engine"""
    redis_client = await get_redis()
    # Note: In production, use proper dependency injection for db session
    # For now, this is a placeholder
    engine = AlertAnalysisEngine(redis_client, None)
    await engine.initialize()
    return engine