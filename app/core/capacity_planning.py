"""
Capacity Planning System with Predictive Analytics.

Advanced capacity planning for the Context Engine with ML-based forecasting,
automated scaling recommendations, and resource optimization.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import redis.asyncio as redis
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.context_performance_monitor import (
    get_context_performance_monitor,
    ContextPerformanceMonitor
)

logger = logging.getLogger(__name__)


class GrowthTrend(Enum):
    """Growth trend classifications."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    SEASONAL = "seasonal"
    STABLE = "stable"
    DECLINING = "declining"


class CapacityMetric(Enum):
    """Metrics to track for capacity planning."""
    CONTEXT_COUNT = "context_count"
    STORAGE_SIZE = "storage_size"
    SEARCH_QUERIES = "search_queries"
    API_CALLS = "api_calls"
    ACTIVE_USERS = "active_users"
    EMBEDDING_REQUESTS = "embedding_requests"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_USAGE = "cache_usage"


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    ARCHIVE = "archive"


@dataclass
class CapacityDataPoint:
    """Single capacity measurement point."""
    timestamp: datetime
    metric: CapacityMetric
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GrowthForecast:
    """Growth forecast for a specific metric."""
    metric: CapacityMetric
    forecast_horizon_days: int
    trend_type: GrowthTrend
    current_value: float
    predicted_values: List[Tuple[datetime, float]]  # (timestamp, predicted_value)
    growth_rate: float  # per day
    confidence_score: float  # 0-1
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    model_accuracy: float = 0.0  # R² score
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CapacityThreshold:
    """Capacity threshold configuration."""
    threshold_id: str
    metric: CapacityMetric
    warning_threshold: float
    critical_threshold: float
    max_capacity: float
    unit: str
    alert_enabled: bool = True
    auto_scaling_enabled: bool = False
    scaling_actions: List[ScalingAction] = field(default_factory=list)


@dataclass
class ScalingRecommendation:
    """Automated scaling recommendation."""
    recommendation_id: str
    metric: CapacityMetric
    current_value: float
    predicted_peak: float
    days_to_threshold: int
    recommended_actions: List[ScalingAction]
    priority: int  # 1-5
    estimated_cost_impact: float  # USD
    implementation_timeline: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PredictiveModel:
    """ML model for capacity forecasting."""
    
    def __init__(self, metric: CapacityMetric):
        self.metric = metric
        self.model = None
        self.feature_scaler = None
        self.last_trained = None
        self.training_data: List[CapacityDataPoint] = []
        self.accuracy_score = 0.0
        
    def add_data_point(self, data_point: CapacityDataPoint) -> None:
        """Add a data point to training data."""
        if data_point.metric == self.metric:
            self.training_data.append(data_point)
            
            # Keep only recent data (last 90 days)
            cutoff_time = datetime.utcnow() - timedelta(days=90)
            self.training_data = [
                dp for dp in self.training_data
                if dp.timestamp > cutoff_time
            ]
    
    def train(self) -> bool:
        """Train the forecasting model."""
        try:
            if len(self.training_data) < 20:  # Need minimum data points
                return False
            
            # Prepare training data
            self.training_data.sort(key=lambda x: x.timestamp)
            
            # Create features (time-based)
            start_time = self.training_data[0].timestamp
            X = []
            y = []
            
            for data_point in self.training_data:
                # Time since start (days)
                days_since_start = (data_point.timestamp - start_time).total_seconds() / 86400
                
                # Day of week (0-6)
                day_of_week = data_point.timestamp.weekday()
                
                # Hour of day (0-23)
                hour_of_day = data_point.timestamp.hour
                
                # Features: [days_since_start, day_of_week, hour_of_day]
                X.append([days_since_start, day_of_week, hour_of_day])
                y.append(data_point.value)
            
            X = np.array(X)
            y = np.array(y)
            
            # Create polynomial features for better fitting
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            
            # Train the model
            self.model.fit(X, y)
            
            # Calculate accuracy
            predictions = self.model.predict(X)
            self.accuracy_score = r2_score(y, predictions)
            
            self.last_trained = datetime.utcnow()
            
            logger.info(f"Trained model for {self.metric.value} with R² = {self.accuracy_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model for {self.metric.value}: {e}")
            return False
    
    def predict(self, target_datetime: datetime) -> Optional[float]:
        """Predict value for a specific datetime."""
        try:
            if not self.model or not self.training_data:
                return None
            
            # Prepare features
            start_time = self.training_data[0].timestamp
            days_since_start = (target_datetime - start_time).total_seconds() / 86400
            day_of_week = target_datetime.weekday()
            hour_of_day = target_datetime.hour
            
            features = np.array([[days_since_start, day_of_week, hour_of_day]])
            prediction = self.model.predict(features)[0]
            
            # Ensure non-negative prediction
            return max(0, prediction)
            
        except Exception as e:
            logger.error(f"Prediction failed for {self.metric.value}: {e}")
            return None
    
    def get_trend_type(self) -> GrowthTrend:
        """Classify the growth trend."""
        try:
            if not self.training_data or len(self.training_data) < 10:
                return GrowthTrend.STABLE
            
            # Calculate growth rate over recent data
            recent_data = self.training_data[-30:]  # Last 30 points
            if len(recent_data) < 2:
                return GrowthTrend.STABLE
            
            values = [dp.value for dp in recent_data]
            time_diffs = [(dp.timestamp - recent_data[0].timestamp).total_seconds() / 86400 for dp in recent_data]
            
            # Simple linear regression to get growth rate
            if len(set(time_diffs)) > 1:  # Avoid division by zero
                correlation = np.corrcoef(time_diffs, values)[0, 1]
                
                # Calculate growth rate
                growth_rate = (values[-1] - values[0]) / max(1, time_diffs[-1])
                relative_growth = growth_rate / max(1, values[0])
                
                # Classify trend
                if abs(relative_growth) < 0.01:  # Less than 1% change per day
                    return GrowthTrend.STABLE
                elif relative_growth > 0.1:  # More than 10% growth per day
                    return GrowthTrend.EXPONENTIAL
                elif relative_growth > 0.01:  # 1-10% growth per day
                    return GrowthTrend.LINEAR
                elif relative_growth < -0.01:  # Declining
                    return GrowthTrend.DECLINING
                else:
                    return GrowthTrend.STABLE
            
            return GrowthTrend.STABLE
            
        except Exception as e:
            logger.error(f"Failed to classify trend for {self.metric.value}: {e}")
            return GrowthTrend.STABLE


class CapacityPlanner:
    """
    Advanced capacity planning system with predictive analytics.
    
    Features:
    - ML-based growth forecasting for multiple metrics
    - Automated threshold monitoring and alerting
    - Intelligent scaling recommendations
    - Seasonal pattern detection
    - Resource optimization suggestions
    - Cost-aware capacity planning
    - Multi-dimensional capacity analysis
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
        performance_monitor: Optional[ContextPerformanceMonitor] = None
    ):
        """
        Initialize the capacity planner.
        
        Args:
            redis_client: Redis client for data storage
            db_session: Database session
            performance_monitor: Context performance monitor
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        self.performance_monitor = performance_monitor
        
        # Predictive models for each metric
        self.models: Dict[CapacityMetric, PredictiveModel] = {
            metric: PredictiveModel(metric) for metric in CapacityMetric
        }
        
        # Capacity thresholds
        self.thresholds: Dict[CapacityMetric, CapacityThreshold] = {}
        
        # Historical data storage
        self.capacity_data: Dict[CapacityMetric, deque] = {
            metric: deque(maxlen=10000) for metric in CapacityMetric
        }
        
        # Forecasts and recommendations
        self.current_forecasts: Dict[CapacityMetric, GrowthForecast] = {}
        self.scaling_recommendations: List[ScalingRecommendation] = []
        
        # Configuration
        self.forecast_horizons = {
            CapacityMetric.CONTEXT_COUNT: 30,      # 30 days
            CapacityMetric.STORAGE_SIZE: 60,       # 60 days
            CapacityMetric.SEARCH_QUERIES: 14,     # 14 days
            CapacityMetric.API_CALLS: 7,           # 7 days
            CapacityMetric.ACTIVE_USERS: 30,       # 30 days
            CapacityMetric.EMBEDDING_REQUESTS: 14, # 14 days
            CapacityMetric.DATABASE_CONNECTIONS: 7, # 7 days
            CapacityMetric.CACHE_USAGE: 14         # 14 days
        }
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        logger.info("Capacity Planner initialized")
    
    async def start(self) -> None:
        """Start the capacity planner background processes."""
        logger.info("Starting capacity planner")
        
        # Initialize performance monitor if not provided
        if self.performance_monitor is None:
            self.performance_monitor = await get_context_performance_monitor()
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._data_collector()),
            asyncio.create_task(self._model_trainer()),
            asyncio.create_task(self._forecast_generator()),
            asyncio.create_task(self._threshold_monitor()),
            asyncio.create_task(self._recommendation_engine()),
            asyncio.create_task(self._data_maintenance())
        ])
    
    async def stop(self) -> None:
        """Stop the capacity planner."""
        logger.info("Stopping capacity planner")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def record_capacity_metric(
        self,
        metric: CapacityMetric,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a capacity metric measurement.
        
        Args:
            metric: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        try:
            data_point = CapacityDataPoint(
                timestamp=datetime.utcnow(),
                metric=metric,
                value=value,
                metadata=metadata or {}
            )
            
            # Store in memory
            self.capacity_data[metric].append(data_point)
            
            # Add to ML model
            self.models[metric].add_data_point(data_point)
            
            # Store in Redis for persistence
            await self._store_capacity_data_redis(data_point)
            
            # Check thresholds
            await self._check_capacity_thresholds(metric, value)
            
        except Exception as e:
            logger.error(f"Failed to record capacity metric: {e}")
    
    async def get_capacity_forecast(
        self,
        metric: CapacityMetric,
        days_ahead: Optional[int] = None
    ) -> Optional[GrowthForecast]:
        """
        Get capacity forecast for a specific metric.
        
        Args:
            metric: Metric to forecast
            days_ahead: Days to forecast (uses default if None)
            
        Returns:
            GrowthForecast if available, None otherwise
        """
        try:
            days_ahead = days_ahead or self.forecast_horizons.get(metric, 30)
            
            model = self.models[metric]
            
            # Ensure model is trained
            if not model.model or not model.last_trained:
                if not model.train():
                    return None
            
            # Get current value
            if not self.capacity_data[metric]:
                return None
                
            current_value = self.capacity_data[metric][-1].value
            
            # Generate predictions
            predictions = []
            current_time = datetime.utcnow()
            
            for day in range(1, days_ahead + 1):
                target_time = current_time + timedelta(days=day)
                predicted_value = model.predict(target_time)
                
                if predicted_value is not None:
                    predictions.append((target_time, predicted_value))
            
            if not predictions:
                return None
            
            # Calculate growth rate
            if len(predictions) >= 2:
                growth_rate = (predictions[-1][1] - current_value) / days_ahead
            else:
                growth_rate = 0.0
            
            # Create forecast
            forecast = GrowthForecast(
                metric=metric,
                forecast_horizon_days=days_ahead,
                trend_type=model.get_trend_type(),
                current_value=current_value,
                predicted_values=predictions,
                growth_rate=growth_rate,
                confidence_score=min(1.0, max(0.0, model.accuracy_score)),
                model_accuracy=model.accuracy_score
            )
            
            # Store current forecast
            self.current_forecasts[metric] = forecast
            
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to get capacity forecast for {metric.value}: {e}")
            return None
    
    async def get_scaling_recommendations(
        self,
        priority_filter: Optional[int] = None
    ) -> List[ScalingRecommendation]:
        """
        Get current scaling recommendations.
        
        Args:
            priority_filter: Optional priority filter (1-5)
            
        Returns:
            List of scaling recommendations
        """
        try:
            recommendations = self.scaling_recommendations.copy()
            
            if priority_filter is not None:
                recommendations = [
                    rec for rec in recommendations
                    if rec.priority <= priority_filter
                ]
            
            # Sort by priority and days to threshold
            recommendations.sort(key=lambda r: (r.priority, r.days_to_threshold))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get scaling recommendations: {e}")
            return []
    
    async def get_capacity_summary(self) -> Dict[str, Any]:
        """Get comprehensive capacity planning summary."""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {},
                "forecasts": {},
                "thresholds": {},
                "recommendations": len(self.scaling_recommendations),
                "model_status": {}
            }
            
            # Current metrics
            for metric in CapacityMetric:
                if self.capacity_data[metric]:
                    latest = self.capacity_data[metric][-1]
                    summary["metrics"][metric.value] = {
                        "current_value": latest.value,
                        "last_updated": latest.timestamp.isoformat(),
                        "data_points": len(self.capacity_data[metric])
                    }
                
                # Model status
                model = self.models[metric]
                summary["model_status"][metric.value] = {
                    "trained": model.model is not None,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None,
                    "accuracy_score": model.accuracy_score,
                    "training_data_points": len(model.training_data)
                }
            
            # Current forecasts
            for metric, forecast in self.current_forecasts.items():
                if forecast.predicted_values:
                    # Get forecast for next 7, 30 days
                    forecast_7d = None
                    forecast_30d = None
                    
                    for timestamp, value in forecast.predicted_values:
                        days_ahead = (timestamp - datetime.utcnow()).days
                        
                        if 6 <= days_ahead <= 8 and forecast_7d is None:
                            forecast_7d = value
                        elif 29 <= days_ahead <= 31 and forecast_30d is None:
                            forecast_30d = value
                    
                    summary["forecasts"][metric.value] = {
                        "trend_type": forecast.trend_type.value,
                        "current_value": forecast.current_value,
                        "forecast_7d": forecast_7d,
                        "forecast_30d": forecast_30d,
                        "growth_rate_per_day": forecast.growth_rate,
                        "confidence_score": forecast.confidence_score
                    }
            
            # Threshold status
            for metric, threshold in self.thresholds.items():
                current_value = summary["metrics"].get(metric.value, {}).get("current_value", 0)
                
                warning_pct = (current_value / threshold.warning_threshold * 100) if threshold.warning_threshold > 0 else 0
                critical_pct = (current_value / threshold.critical_threshold * 100) if threshold.critical_threshold > 0 else 0
                
                summary["thresholds"][metric.value] = {
                    "warning_threshold": threshold.warning_threshold,
                    "critical_threshold": threshold.critical_threshold,
                    "max_capacity": threshold.max_capacity,
                    "warning_percentage": min(100, warning_pct),
                    "critical_percentage": min(100, critical_pct),
                    "unit": threshold.unit
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get capacity summary: {e}")
            return {"error": str(e)}
    
    async def add_capacity_threshold(
        self,
        metric: CapacityMetric,
        warning_threshold: float,
        critical_threshold: float,
        max_capacity: float,
        unit: str,
        auto_scaling_enabled: bool = False
    ) -> CapacityThreshold:
        """Add or update a capacity threshold."""
        try:
            threshold = CapacityThreshold(
                threshold_id=str(uuid.uuid4()),
                metric=metric,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                max_capacity=max_capacity,
                unit=unit,
                auto_scaling_enabled=auto_scaling_enabled
            )
            
            self.thresholds[metric] = threshold
            
            # Store in Redis
            await self.redis_client.setex(
                f"capacity_threshold:{metric.value}",
                86400 * 365,  # 1 year TTL
                json.dumps(asdict(threshold), default=str)
            )
            
            logger.info(f"Added capacity threshold for {metric.value}: {warning_threshold}/{critical_threshold}")
            
            return threshold
            
        except Exception as e:
            logger.error(f"Failed to add capacity threshold: {e}")
            raise
    
    # Background task methods
    async def _data_collector(self) -> None:
        """Background task to collect capacity metrics."""
        logger.info("Starting capacity data collector")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect context count
                await self._collect_context_metrics()
                
                # Collect storage metrics
                await self._collect_storage_metrics()
                
                # Collect search metrics
                await self._collect_search_metrics()
                
                # Collect API metrics
                await self._collect_api_metrics()
                
                await asyncio.sleep(300)  # Collect every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data collector error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Capacity data collector stopped")
    
    async def _model_trainer(self) -> None:
        """Background task to train ML models."""
        logger.info("Starting model trainer")
        
        while not self._shutdown_event.is_set():
            try:
                for metric, model in self.models.items():
                    # Retrain if we have new data and it's been a while
                    if (len(model.training_data) >= 20 and
                        (model.last_trained is None or 
                         (datetime.utcnow() - model.last_trained).total_seconds() > 21600)):  # 6 hours
                        
                        model.train()
                
                await asyncio.sleep(3600)  # Train every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Model trainer error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Model trainer stopped")
    
    async def _forecast_generator(self) -> None:
        """Background task to generate forecasts."""
        logger.info("Starting forecast generator")
        
        while not self._shutdown_event.is_set():
            try:
                for metric in CapacityMetric:
                    await self.get_capacity_forecast(metric)
                
                await asyncio.sleep(1800)  # Generate forecasts every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Forecast generator error: {e}")
                await asyncio.sleep(1800)
        
        logger.info("Forecast generator stopped")
    
    async def _threshold_monitor(self) -> None:
        """Background task to monitor capacity thresholds."""
        logger.info("Starting threshold monitor")
        
        while not self._shutdown_event.is_set():
            try:
                for metric, threshold in self.thresholds.items():
                    if self.capacity_data[metric]:
                        current_value = self.capacity_data[metric][-1].value
                        await self._check_capacity_thresholds(metric, current_value)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Threshold monitor error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Threshold monitor stopped")
    
    async def _recommendation_engine(self) -> None:
        """Background task to generate scaling recommendations."""
        logger.info("Starting recommendation engine")
        
        while not self._shutdown_event.is_set():
            try:
                await self._generate_scaling_recommendations()
                
                await asyncio.sleep(1800)  # Generate recommendations every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recommendation engine error: {e}")
                await asyncio.sleep(1800)
        
        logger.info("Recommendation engine stopped")
    
    async def _data_maintenance(self) -> None:
        """Background task for data cleanup and maintenance."""
        logger.info("Starting data maintenance")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old recommendations
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.scaling_recommendations = [
                    rec for rec in self.scaling_recommendations
                    if rec.created_at > cutoff_time
                ]
                
                # Clean up old forecasts
                old_forecasts = []
                for metric, forecast in self.current_forecasts.items():
                    if (datetime.utcnow() - forecast.created_at).total_seconds() > 3600:  # 1 hour
                        old_forecasts.append(metric)
                
                for metric in old_forecasts:
                    del self.current_forecasts[metric]
                
                await asyncio.sleep(3600)  # Maintenance every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data maintenance error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Data maintenance stopped")
    
    # Helper methods
    def _initialize_default_thresholds(self) -> None:
        """Initialize default capacity thresholds."""
        default_thresholds = [
            (CapacityMetric.CONTEXT_COUNT, 800000, 1000000, 1200000, "contexts"),
            (CapacityMetric.STORAGE_SIZE, 80 * 1024**3, 100 * 1024**3, 120 * 1024**3, "bytes"),  # 80GB, 100GB, 120GB
            (CapacityMetric.SEARCH_QUERIES, 50000, 75000, 100000, "queries/day"),
            (CapacityMetric.API_CALLS, 100000, 150000, 200000, "calls/day"),
            (CapacityMetric.DATABASE_CONNECTIONS, 80, 100, 120, "connections"),
            (CapacityMetric.CACHE_USAGE, 80, 95, 100, "percent")
        ]
        
        for metric, warning, critical, max_cap, unit in default_thresholds:
            self.thresholds[metric] = CapacityThreshold(
                threshold_id=str(uuid.uuid4()),
                metric=metric,
                warning_threshold=warning,
                critical_threshold=critical,
                max_capacity=max_cap,
                unit=unit
            )
        
        logger.info(f"Initialized {len(default_thresholds)} default capacity thresholds")
    
    async def _collect_context_metrics(self) -> None:
        """Collect context-related capacity metrics."""
        try:
            if not self.db_session:
                async for session in get_async_session():
                    await self._query_context_metrics(session)
                    break
            else:
                await self._query_context_metrics(self.db_session)
                
        except Exception as e:
            logger.error(f"Failed to collect context metrics: {e}")
    
    async def _query_context_metrics(self, session: AsyncSession) -> None:
        """Query context metrics from database."""
        try:
            # Context count
            result = await session.execute(text("SELECT COUNT(*) FROM contexts WHERE deleted_at IS NULL"))
            context_count = result.scalar() or 0
            
            await self.record_capacity_metric(CapacityMetric.CONTEXT_COUNT, float(context_count))
            
            # Storage size
            result = await session.execute(text("""
                SELECT COALESCE(SUM(LENGTH(content)), 0) as total_size 
                FROM contexts 
                WHERE deleted_at IS NULL
            """))
            storage_size = result.scalar() or 0
            
            await self.record_capacity_metric(CapacityMetric.STORAGE_SIZE, float(storage_size))
            
        except Exception as e:
            logger.error(f"Failed to query context metrics: {e}")
    
    async def _collect_storage_metrics(self) -> None:
        """Collect storage capacity metrics."""
        try:
            # Get storage data from Redis
            capacity_history = await self.redis_client.lrange("context_monitor:capacity_history", 0, 0)
            
            if capacity_history:
                try:
                    latest_data = json.loads(capacity_history[0])
                    storage_size = latest_data.get("total_size_bytes", 0)
                    
                    await self.record_capacity_metric(CapacityMetric.STORAGE_SIZE, float(storage_size))
                    
                except (json.JSONDecodeError, KeyError):
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to collect storage metrics: {e}")
    
    async def _collect_search_metrics(self) -> None:
        """Collect search-related capacity metrics."""
        try:
            # Get search metrics from Redis
            search_metrics = await self.redis_client.lrange("context_monitor:search_metrics", 0, 999)
            
            # Count queries in the last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            daily_queries = 0
            
            for metric_str in search_metrics:
                try:
                    metric = json.loads(metric_str)
                    metric_time = datetime.fromisoformat(metric["timestamp"])
                    
                    if metric_time >= cutoff_time:
                        daily_queries += 1
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            await self.record_capacity_metric(CapacityMetric.SEARCH_QUERIES, float(daily_queries))
            
        except Exception as e:
            logger.error(f"Failed to collect search metrics: {e}")
    
    async def _collect_api_metrics(self) -> None:
        """Collect API usage capacity metrics."""
        try:
            # Get API cost data from Redis
            api_costs = await self.redis_client.lrange("context_monitor:api_costs", 0, 999)
            
            # Count API calls in the last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            daily_api_calls = 0
            
            for cost_str in api_costs:
                try:
                    cost = json.loads(cost_str)
                    cost_time = datetime.fromisoformat(cost["timestamp"])
                    
                    if cost_time >= cutoff_time and cost["success"]:
                        daily_api_calls += 1
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            await self.record_capacity_metric(CapacityMetric.API_CALLS, float(daily_api_calls))
            await self.record_capacity_metric(CapacityMetric.EMBEDDING_REQUESTS, float(daily_api_calls))
            
        except Exception as e:
            logger.error(f"Failed to collect API metrics: {e}")
    
    async def _store_capacity_data_redis(self, data_point: CapacityDataPoint) -> None:
        """Store capacity data point in Redis."""
        try:
            await self.redis_client.lpush(
                f"capacity_data:{data_point.metric.value}",
                json.dumps(asdict(data_point), default=str)
            )
            
            # Keep only recent data
            await self.redis_client.ltrim(f"capacity_data:{data_point.metric.value}", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to store capacity data in Redis: {e}")
    
    async def _check_capacity_thresholds(self, metric: CapacityMetric, current_value: float) -> None:
        """Check if current value exceeds capacity thresholds."""
        try:
            if metric not in self.thresholds:
                return
            
            threshold = self.thresholds[metric]
            
            if current_value >= threshold.critical_threshold:
                logger.critical(f"CRITICAL: {metric.value} at {current_value} exceeds critical threshold {threshold.critical_threshold}")
                
                # Trigger auto-scaling if enabled
                if threshold.auto_scaling_enabled:
                    await self._trigger_auto_scaling(metric, current_value, "critical")
                    
            elif current_value >= threshold.warning_threshold:
                logger.warning(f"WARNING: {metric.value} at {current_value} exceeds warning threshold {threshold.warning_threshold}")
                
                # Trigger auto-scaling if enabled
                if threshold.auto_scaling_enabled:
                    await self._trigger_auto_scaling(metric, current_value, "warning")
            
        except Exception as e:
            logger.error(f"Failed to check capacity thresholds: {e}")
    
    async def _trigger_auto_scaling(self, metric: CapacityMetric, current_value: float, severity: str) -> None:
        """Trigger auto-scaling actions."""
        try:
            logger.info(f"Triggering auto-scaling for {metric.value} (severity: {severity})")
            
            # This would integrate with actual scaling systems
            # For now, just log the action
            
            scaling_action = ScalingAction.SCALE_OUT if severity == "critical" else ScalingAction.OPTIMIZE
            
            # Store scaling event
            scaling_event = {
                "metric": metric.value,
                "current_value": current_value,
                "severity": severity,
                "action": scaling_action.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.lpush("capacity_scaling_events", json.dumps(scaling_event))
            
        except Exception as e:
            logger.error(f"Failed to trigger auto-scaling: {e}")
    
    async def _generate_scaling_recommendations(self) -> None:
        """Generate intelligent scaling recommendations."""
        try:
            new_recommendations = []
            
            for metric in CapacityMetric:
                forecast = self.current_forecasts.get(metric)
                threshold = self.thresholds.get(metric)
                
                if not forecast or not threshold:
                    continue
                
                # Check if forecast indicates approaching thresholds
                days_to_warning = None
                days_to_critical = None
                predicted_peak = forecast.current_value
                
                for timestamp, predicted_value in forecast.predicted_values:
                    days_ahead = (timestamp - datetime.utcnow()).days
                    
                    if predicted_value > predicted_peak:
                        predicted_peak = predicted_value
                    
                    if (predicted_value >= threshold.warning_threshold and days_to_warning is None):
                        days_to_warning = days_ahead
                    
                    if (predicted_value >= threshold.critical_threshold and days_to_critical is None):
                        days_to_critical = days_ahead
                
                # Generate recommendations based on forecast
                if days_to_critical is not None and days_to_critical <= 30:
                    # Critical threshold will be reached soon
                    recommendation = ScalingRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        metric=metric,
                        current_value=forecast.current_value,
                        predicted_peak=predicted_peak,
                        days_to_threshold=days_to_critical,
                        recommended_actions=[ScalingAction.SCALE_OUT, ScalingAction.OPTIMIZE],
                        priority=1,
                        estimated_cost_impact=1000.0,  # Estimated
                        implementation_timeline="immediate",
                        description=f"{metric.value} will reach critical threshold in {days_to_critical} days. Immediate scaling required."
                    )
                    
                    new_recommendations.append(recommendation)
                    
                elif days_to_warning is not None and days_to_warning <= 14:
                    # Warning threshold will be reached soon
                    recommendation = ScalingRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        metric=metric,
                        current_value=forecast.current_value,
                        predicted_peak=predicted_peak,
                        days_to_threshold=days_to_warning,
                        recommended_actions=[ScalingAction.OPTIMIZE, ScalingAction.SCALE_UP],
                        priority=2,
                        estimated_cost_impact=500.0,  # Estimated
                        implementation_timeline="within_week",
                        description=f"{metric.value} will reach warning threshold in {days_to_warning} days. Plan scaling activities."
                    )
                    
                    new_recommendations.append(recommendation)
                
                # Check for over-provisioning
                elif (forecast.trend_type == GrowthTrend.STABLE and 
                      forecast.current_value < threshold.warning_threshold * 0.5):
                    
                    recommendation = ScalingRecommendation(
                        recommendation_id=str(uuid.uuid4()),
                        metric=metric,
                        current_value=forecast.current_value,
                        predicted_peak=predicted_peak,
                        days_to_threshold=999,  # No threshold breach expected
                        recommended_actions=[ScalingAction.SCALE_DOWN, ScalingAction.OPTIMIZE],
                        priority=4,
                        estimated_cost_impact=-200.0,  # Cost savings
                        implementation_timeline="next_month",
                        description=f"{metric.value} appears over-provisioned. Consider scaling down to reduce costs."
                    )
                    
                    new_recommendations.append(recommendation)
            
            # Update recommendations list
            self.scaling_recommendations = new_recommendations
            
            logger.info(f"Generated {len(new_recommendations)} scaling recommendations")
            
        except Exception as e:
            logger.error(f"Failed to generate scaling recommendations: {e}")


# Global instance
_capacity_planner: Optional[CapacityPlanner] = None


async def get_capacity_planner() -> CapacityPlanner:
    """Get singleton capacity planner instance."""
    global _capacity_planner
    
    if _capacity_planner is None:
        _capacity_planner = CapacityPlanner()
        await _capacity_planner.start()
    
    return _capacity_planner


async def cleanup_capacity_planner() -> None:
    """Cleanup capacity planner resources."""
    global _capacity_planner
    
    if _capacity_planner:
        await _capacity_planner.stop()
        _capacity_planner = None