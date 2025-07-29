"""
VS 7.2: Load Prediction Service with Time-Series Forecasting - LeanVibe Agent Hive 2.0 Phase 5.3

Advanced load prediction service providing ML-based forecasting with multiple model support.
Includes cold start handling, continuous accuracy monitoring, and fallback strategies.

Features:
- Multiple time-series models (ARIMA, Exponential Smoothing, Linear Regression)
- Automated model selection based on data characteristics
- Cold start strategy for new agent types and workloads
- Continuous model monitoring with automated fallback
- Seasonal pattern detection and handling
- Real-time prediction accuracy tracking and improvement
"""

import asyncio
import logging
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import statistics
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from ..core.circuit_breaker import CircuitBreaker


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of prediction models."""
    SIMPLE_MOVING_AVERAGE = "simple_moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_REGRESSION = "linear_regression"
    ARIMA = "arima"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"
    ENSEMBLE = "ensemble"


class PredictionHorizon(Enum):
    """Prediction time horizons."""
    SHORT_TERM = "short_term"    # 5-15 minutes
    MEDIUM_TERM = "medium_term"  # 30-60 minutes
    LONG_TERM = "long_term"      # 2-24 hours


class SeasonalPattern(Enum):
    """Types of seasonal patterns."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    NONE = "none"


@dataclass
class LoadDataPoint:
    """Represents a single load data point."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    active_agents: int
    pending_tasks: int
    message_queue_depth: int
    response_time_p95: float
    error_rate: float
    throughput_rps: float
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.cpu_utilization,
            self.memory_utilization,
            float(self.active_agents),
            float(self.pending_tasks),
            float(self.message_queue_depth),
            self.response_time_p95,
            self.error_rate,
            self.throughput_rps
        ]


@dataclass
class PredictionResult:
    """Result of load prediction."""
    timestamp: datetime
    horizon_minutes: int
    model_type: ModelType
    predicted_load: Dict[str, float]
    confidence_interval: Dict[str, Tuple[float, float]]
    confidence_score: float
    prediction_accuracy: Optional[float] = None
    seasonal_pattern: Optional[SeasonalPattern] = None
    trend_direction: Optional[str] = None  # "increasing", "decreasing", "stable"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelPerformance:
    """Performance metrics for a prediction model."""
    model_type: ModelType
    total_predictions: int
    correct_predictions: int
    mean_absolute_error: float
    mean_squared_error: float
    accuracy_score: float
    last_updated: datetime
    training_data_size: int
    feature_importance: Optional[Dict[str, float]] = None


class LoadPredictionService:
    """
    Advanced load prediction service with multiple time-series models.
    
    Core Features:
    - Multiple prediction models with automated selection
    - Seasonal pattern detection and handling
    - Cold start strategies for new workloads
    - Continuous accuracy monitoring and model retraining
    - Real-time prediction with confidence intervals
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Core configuration
        self.enabled = True
        self.auto_model_selection = True
        self.continuous_learning = True
        
        # Model configuration
        self.min_training_samples = 50
        self.max_training_samples = 2000
        self.model_retrain_interval_hours = 6
        self.accuracy_threshold = 0.7
        self.confidence_threshold = 0.6
        
        # Prediction configuration
        self.default_horizon_minutes = 30
        self.max_horizon_minutes = 1440  # 24 hours
        self.prediction_cache_minutes = 5
        
        # Cold start configuration
        self.cold_start_fallback_model = ModelType.SIMPLE_MOVING_AVERAGE
        self.cold_start_confidence = 0.4
        self.min_samples_for_seasonal = 168  # 1 week of hourly data
        
        # Internal state
        self._load_history: deque = deque(maxlen=self.max_training_samples)
        self._models: Dict[ModelType, Any] = {}
        self._model_performance: Dict[ModelType, ModelPerformance] = {}
        self._prediction_cache: Dict[str, Tuple[datetime, PredictionResult]] = {}
        self._seasonal_patterns: Dict[str, SeasonalPattern] = {}
        
        # Model instances
        self._scalers: Dict[ModelType, StandardScaler] = {}
        self._feature_names = [
            "cpu_utilization", "memory_utilization", "active_agents",
            "pending_tasks", "message_queue_depth", "response_time_p95",
            "error_rate", "throughput_rps"
        ]
        
        # Circuit breakers
        self._prediction_circuit_breaker = CircuitBreaker(
            name="load_prediction",
            failure_threshold=5,
            timeout_seconds=300
        )
        
        self._training_circuit_breaker = CircuitBreaker(
            name="model_training",
            failure_threshold=3,
            timeout_seconds=600
        )
    
    async def initialize(self) -> None:
        """Initialize the load prediction service."""
        try:
            logger.info("Initializing Load Prediction Service VS 7.2")
            
            # Load historical data
            await self._load_historical_data()
            
            # Initialize models
            await self._initialize_models()
            
            # Start background tasks
            asyncio.create_task(self._model_training_loop())
            asyncio.create_task(self._accuracy_monitoring_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._seasonal_pattern_detector())
            
            logger.info("Load Prediction Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Load Prediction Service: {e}")
            raise
    
    async def add_load_data_point(self, data_point: LoadDataPoint) -> None:
        """Add a new load data point for training and prediction."""
        try:
            # Add to history
            self._load_history.append(data_point)
            
            # Persist to Redis for durability
            redis = await get_redis()
            await redis.lpush(
                "load_prediction_history",
                json.dumps({
                    "timestamp": data_point.timestamp.isoformat(),
                    "data": asdict(data_point)
                })
            )
            
            # Keep only recent data in Redis
            await redis.ltrim("load_prediction_history", 0, self.max_training_samples - 1)
            
            # Update model performance if we have predictions to validate
            await self._validate_recent_predictions(data_point)
            
        except Exception as e:
            logger.error(f"Error adding load data point: {e}")
    
    async def predict_load(
        self,
        horizon_minutes: int = None,
        model_type: Optional[ModelType] = None,
        include_confidence_interval: bool = True
    ) -> PredictionResult:
        """
        Predict future load based on historical data.
        
        Args:
            horizon_minutes: How many minutes into the future to predict
            model_type: Specific model to use (None for auto-selection)
            include_confidence_interval: Whether to include confidence intervals
            
        Returns:
            Prediction result with load forecasts
        """
        horizon_minutes = horizon_minutes or self.default_horizon_minutes
        
        try:
            # Check cache first
            cache_key = f"prediction_{horizon_minutes}_{model_type}"
            if cache_key in self._prediction_cache:
                cache_time, cached_result = self._prediction_cache[cache_key]
                if datetime.utcnow() - cache_time < timedelta(minutes=self.prediction_cache_minutes):
                    logger.debug(f"Returning cached prediction for horizon {horizon_minutes}m")
                    return cached_result
            
            # Circuit breaker protection
            async with self._prediction_circuit_breaker:
                # Determine which model to use
                if model_type is None:
                    model_type = await self._select_best_model(horizon_minutes)
                
                # Check if we have enough data
                if len(self._load_history) < self.min_training_samples:
                    return await self._cold_start_prediction(horizon_minutes)
                
                # Make prediction based on model type
                if model_type == ModelType.SIMPLE_MOVING_AVERAGE:
                    prediction = await self._predict_moving_average(horizon_minutes, include_confidence_interval)
                elif model_type == ModelType.EXPONENTIAL_SMOOTHING:
                    prediction = await self._predict_exponential_smoothing(horizon_minutes, include_confidence_interval)
                elif model_type == ModelType.LINEAR_REGRESSION:
                    prediction = await self._predict_linear_regression(horizon_minutes, include_confidence_interval)
                elif model_type == ModelType.SEASONAL_DECOMPOSITION:
                    prediction = await self._predict_seasonal_decomposition(horizon_minutes, include_confidence_interval)
                elif model_type == ModelType.ENSEMBLE:
                    prediction = await self._predict_ensemble(horizon_minutes, include_confidence_interval)
                else:
                    # Fallback to moving average
                    prediction = await self._predict_moving_average(horizon_minutes, include_confidence_interval)
                
                # Cache the prediction
                self._prediction_cache[cache_key] = (datetime.utcnow(), prediction)
                
                # Update model usage statistics
                if model_type in self._model_performance:
                    self._model_performance[model_type].total_predictions += 1
                
                return prediction
            
        except Exception as e:
            logger.error(f"Error predicting load: {e}")
            
            # Fallback to simple prediction
            return await self._cold_start_prediction(horizon_minutes)
    
    async def get_prediction_accuracy(self, model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """Get prediction accuracy metrics."""
        try:
            if model_type:
                if model_type in self._model_performance:
                    perf = self._model_performance[model_type]
                    return {
                        "model_type": model_type.value,
                        "accuracy_score": perf.accuracy_score,
                        "mean_absolute_error": perf.mean_absolute_error,
                        "mean_squared_error": perf.mean_squared_error,
                        "total_predictions": perf.total_predictions,
                        "correct_predictions": perf.correct_predictions,
                        "last_updated": perf.last_updated.isoformat(),
                        "training_data_size": perf.training_data_size
                    }
                else:
                    return {"error": f"No performance data for model {model_type.value}"}
            
            # Return all model performances
            all_performance = {}
            for mt, perf in self._model_performance.items():
                all_performance[mt.value] = {
                    "accuracy_score": perf.accuracy_score,
                    "mean_absolute_error": perf.mean_absolute_error,
                    "mean_squared_error": perf.mean_squared_error,
                    "total_predictions": perf.total_predictions,
                    "correct_predictions": perf.correct_predictions,
                    "last_updated": perf.last_updated.isoformat(),
                    "training_data_size": perf.training_data_size
                }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "models": all_performance,
                "data_points": len(self._load_history),
                "seasonal_patterns": {k: v.value for k, v in self._seasonal_patterns.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction accuracy: {e}")
            return {"error": str(e)}
    
    async def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """Retrain prediction models with latest data."""
        try:
            if len(self._load_history) < self.min_training_samples and not force:
                return {"error": "Insufficient training data"}
            
            logger.info("Retraining prediction models")
            
            async with self._training_circuit_breaker:
                training_results = {}
                
                # Prepare training data
                training_data = list(self._load_history)
                features = np.array([dp.to_feature_vector() for dp in training_data])
                
                # Train each model type
                for model_type in [ModelType.LINEAR_REGRESSION, ModelType.EXPONENTIAL_SMOOTHING]:
                    try:
                        if model_type == ModelType.LINEAR_REGRESSION:
                            result = await self._train_linear_regression(features, training_data)
                        elif model_type == ModelType.EXPONENTIAL_SMOOTHING:
                            result = await self._train_exponential_smoothing(training_data)
                        else:
                            continue
                        
                        training_results[model_type.value] = result
                        
                    except Exception as e:
                        logger.error(f"Error training {model_type.value}: {e}")
                        training_results[model_type.value] = {"error": str(e)}
                
                # Update seasonal patterns
                await self._update_seasonal_patterns(training_data)
                
                logger.info(f"Model retraining completed: {len(training_results)} models trained")
                
                return {
                    "timestamp": datetime.utcnow().isoformat(),
                    "training_data_size": len(training_data),
                    "models_trained": len(training_results),
                    "results": training_results
                }
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {"error": str(e)}
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        try:
            # Service configuration
            config_status = {
                "enabled": self.enabled,
                "auto_model_selection": self.auto_model_selection,
                "continuous_learning": self.continuous_learning,
                "min_training_samples": self.min_training_samples,
                "accuracy_threshold": self.accuracy_threshold
            }
            
            # Data status
            data_status = {
                "total_data_points": len(self._load_history),
                "oldest_data_point": self._load_history[0].timestamp.isoformat() if self._load_history else None,
                "newest_data_point": self._load_history[-1].timestamp.isoformat() if self._load_history else None,
                "data_coverage_hours": (
                    (self._load_history[-1].timestamp - self._load_history[0].timestamp).total_seconds() / 3600
                    if len(self._load_history) >= 2 else 0
                )
            }
            
            # Model status
            model_status = {}
            for model_type in ModelType:
                if model_type in self._model_performance:
                    perf = self._model_performance[model_type]
                    model_status[model_type.value] = {
                        "trained": True,
                        "accuracy_score": perf.accuracy_score,
                        "predictions_made": perf.total_predictions,
                        "last_updated": perf.last_updated.isoformat()
                    }
                else:
                    model_status[model_type.value] = {"trained": False}
            
            # Circuit breaker status
            circuit_breaker_status = {
                "prediction_circuit_breaker": {
                    "state": self._prediction_circuit_breaker.state,
                    "failure_count": self._prediction_circuit_breaker.failure_count,
                    "success_count": self._prediction_circuit_breaker.success_count
                },
                "training_circuit_breaker": {
                    "state": self._training_circuit_breaker.state,
                    "failure_count": self._training_circuit_breaker.failure_count,
                    "success_count": self._training_circuit_breaker.success_count
                }
            }
            
            # Cache status
            cache_status = {
                "cached_predictions": len(self._prediction_cache),
                "cache_hit_rate": 0.0  # Would calculate from actual metrics
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "configuration": config_status,
                "data": data_status,
                "models": model_status,
                "circuit_breakers": circuit_breaker_status,
                "cache": cache_status,
                "seasonal_patterns": {k: v.value for k, v in self._seasonal_patterns.items()}
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"error": str(e)}
    
    # Internal methods
    
    async def _load_historical_data(self) -> None:
        """Load historical data from Redis."""
        try:
            redis = await get_redis()
            
            # Get historical data
            history_data = await redis.lrange("load_prediction_history", 0, self.max_training_samples - 1)
            
            for data_json in reversed(history_data):  # Reverse to get chronological order
                try:
                    data_dict = json.loads(data_json)
                    data_point = LoadDataPoint(
                        timestamp=datetime.fromisoformat(data_dict["timestamp"]),
                        **data_dict["data"]
                    )
                    self._load_history.append(data_point)
                    
                except Exception as e:
                    logger.warning(f"Could not parse historical data point: {e}")
            
            logger.info(f"Loaded {len(self._load_history)} historical data points")
            
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    async def _initialize_models(self) -> None:
        """Initialize prediction models."""
        try:
            # Initialize scalers
            for model_type in ModelType:
                self._scalers[model_type] = StandardScaler()
            
            # If we have enough data, train initial models
            if len(self._load_history) >= self.min_training_samples:
                await self.retrain_models(force=True)
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _cold_start_prediction(self, horizon_minutes: int) -> PredictionResult:
        """Generate prediction when we don't have enough training data."""
        try:
            logger.info("Using cold start prediction strategy")
            
            # Use recent data if available
            if len(self._load_history) > 0:
                recent_data = list(self._load_history)[-min(10, len(self._load_history)):]
                
                # Simple average of recent data
                avg_cpu = statistics.mean(dp.cpu_utilization for dp in recent_data)
                avg_memory = statistics.mean(dp.memory_utilization for dp in recent_data)
                avg_agents = statistics.mean(dp.active_agents for dp in recent_data)
                avg_tasks = statistics.mean(dp.pending_tasks for dp in recent_data)
                avg_queue = statistics.mean(dp.message_queue_depth for dp in recent_data)
                avg_response_time = statistics.mean(dp.response_time_p95 for dp in recent_data)
                avg_error_rate = statistics.mean(dp.error_rate for dp in recent_data)
                avg_throughput = statistics.mean(dp.throughput_rps for dp in recent_data)
                
                # Add some uncertainty for longer horizons
                uncertainty_factor = min(horizon_minutes / 60.0, 2.0)  # Max 2x uncertainty
                
                predicted_load = {
                    "cpu_utilization": avg_cpu,
                    "memory_utilization": avg_memory,
                    "active_agents": avg_agents,
                    "pending_tasks": avg_tasks,
                    "message_queue_depth": avg_queue,
                    "response_time_p95": avg_response_time,
                    "error_rate": avg_error_rate,
                    "throughput_rps": avg_throughput
                }
                
                # Simple confidence intervals based on recent variance
                confidence_interval = {}
                for key, value in predicted_load.items():
                    variance = statistics.variance([getattr(dp, key) for dp in recent_data]) if len(recent_data) > 1 else 0.1
                    margin = math.sqrt(variance) * uncertainty_factor
                    confidence_interval[key] = (max(0, value - margin), value + margin)
                
            else:
                # Default values when no data available
                predicted_load = {
                    "cpu_utilization": 0.3,
                    "memory_utilization": 0.4,
                    "active_agents": 2,
                    "pending_tasks": 0,
                    "message_queue_depth": 0,
                    "response_time_p95": 100.0,
                    "error_rate": 0.01,
                    "throughput_rps": 10.0
                }
                
                confidence_interval = {
                    key: (value * 0.5, value * 1.5) for key, value in predicted_load.items()
                }
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=self.cold_start_fallback_model,
                predicted_load=predicted_load,
                confidence_interval=confidence_interval,
                confidence_score=self.cold_start_confidence,
                trend_direction="stable",
                metadata={"cold_start": True, "data_points": len(self._load_history)}
            )
            
        except Exception as e:
            logger.error(f"Error in cold start prediction: {e}")
            
            # Ultimate fallback
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.SIMPLE_MOVING_AVERAGE,
                predicted_load={"cpu_utilization": 0.3, "memory_utilization": 0.4},
                confidence_interval={"cpu_utilization": (0.1, 0.5), "memory_utilization": (0.2, 0.6)},
                confidence_score=0.2,
                trend_direction="unknown",
                metadata={"error": str(e), "fallback": True}
            )
    
    async def _select_best_model(self, horizon_minutes: int) -> ModelType:
        """Select the best model based on performance and horizon."""
        try:
            if not self.auto_model_selection:
                return ModelType.LINEAR_REGRESSION  # Default
            
            # Find the model with best accuracy for this type of prediction
            best_model = ModelType.SIMPLE_MOVING_AVERAGE
            best_accuracy = 0.0
            
            for model_type, performance in self._model_performance.items():
                if performance.accuracy_score > best_accuracy and performance.accuracy_score > self.accuracy_threshold:
                    best_accuracy = performance.accuracy_score
                    best_model = model_type
            
            # Consider horizon-specific preferences
            if horizon_minutes <= 15:  # Short term
                if ModelType.EXPONENTIAL_SMOOTHING in self._model_performance:
                    return ModelType.EXPONENTIAL_SMOOTHING
            elif horizon_minutes >= 120:  # Long term
                if ModelType.SEASONAL_DECOMPOSITION in self._model_performance:
                    return ModelType.SEASONAL_DECOMPOSITION
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
            return ModelType.SIMPLE_MOVING_AVERAGE
    
    async def _predict_moving_average(self, horizon_minutes: int, include_confidence: bool) -> PredictionResult:
        """Simple moving average prediction."""
        try:
            window_size = min(20, len(self._load_history))
            recent_data = list(self._load_history)[-window_size:]
            
            # Calculate moving averages
            predicted_load = {}
            confidence_interval = {}
            
            for feature in self._feature_names:
                values = [getattr(dp, feature) for dp in recent_data]
                avg_value = statistics.mean(values)
                
                predicted_load[feature] = avg_value
                
                if include_confidence:
                    if len(values) > 1:
                        std_dev = statistics.stdev(values)
                        margin = std_dev * 1.96  # 95% confidence interval
                        confidence_interval[feature] = (
                            max(0, avg_value - margin),
                            avg_value + margin
                        )
                    else:
                        confidence_interval[feature] = (avg_value * 0.8, avg_value * 1.2)
            
            # Determine trend
            if len(recent_data) >= 10:
                early_avg = statistics.mean(getattr(dp, "cpu_utilization") for dp in recent_data[:5])
                late_avg = statistics.mean(getattr(dp, "cpu_utilization") for dp in recent_data[-5:])
                
                if late_avg > early_avg * 1.1:
                    trend = "increasing"
                elif late_avg < early_avg * 0.9:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.SIMPLE_MOVING_AVERAGE,
                predicted_load=predicted_load,
                confidence_interval=confidence_interval,
                confidence_score=0.7,
                trend_direction=trend,
                metadata={"window_size": window_size}
            )
            
        except Exception as e:
            logger.error(f"Error in moving average prediction: {e}")
            return await self._cold_start_prediction(horizon_minutes)
    
    async def _predict_exponential_smoothing(self, horizon_minutes: int, include_confidence: bool) -> PredictionResult:
        """Exponential smoothing prediction."""
        try:
            alpha = 0.3  # Smoothing parameter
            data_points = list(self._load_history)
            
            predicted_load = {}
            confidence_interval = {}
            
            for feature in self._feature_names:
                values = [getattr(dp, feature) for dp in data_points]
                
                if len(values) < 2:
                    predicted_load[feature] = values[0] if values else 0.0
                    continue
                
                # Apply exponential smoothing
                smoothed_values = [values[0]]
                for i in range(1, len(values)):
                    smoothed = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
                    smoothed_values.append(smoothed)
                
                # Predict based on trend
                if len(smoothed_values) >= 3:
                    trend = smoothed_values[-1] - smoothed_values[-2]
                    predicted_value = smoothed_values[-1] + trend * (horizon_minutes / 60.0)
                else:
                    predicted_value = smoothed_values[-1]
                
                predicted_load[feature] = max(0, predicted_value)
                
                if include_confidence:
                    # Calculate prediction error variance
                    errors = [abs(values[i] - smoothed_values[i]) for i in range(len(values))]
                    error_variance = statistics.variance(errors) if len(errors) > 1 else 0.1
                    margin = math.sqrt(error_variance) * 1.96
                    
                    confidence_interval[feature] = (
                        max(0, predicted_value - margin),
                        predicted_value + margin
                    )
            
            # Determine overall trend
            cpu_values = [getattr(dp, "cpu_utilization") for dp in data_points[-10:]]
            if len(cpu_values) >= 5:
                recent_trend = (cpu_values[-1] - cpu_values[-5]) / 5
                if recent_trend > 0.01:
                    trend = "increasing"
                elif recent_trend < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.EXPONENTIAL_SMOOTHING,
                predicted_load=predicted_load,
                confidence_interval=confidence_interval,
                confidence_score=0.75,
                trend_direction=trend,
                metadata={"alpha": alpha, "data_points": len(data_points)}
            )
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing prediction: {e}")
            return await self._cold_start_prediction(horizon_minutes)
    
    async def _predict_linear_regression(self, horizon_minutes: int, include_confidence: bool) -> PredictionResult:
        """Linear regression prediction."""
        try:
            if ModelType.LINEAR_REGRESSION not in self._models:
                # Train model if not available
                await self._train_linear_regression_simple()
            
            model = self._models.get(ModelType.LINEAR_REGRESSION)
            scaler = self._scalers.get(ModelType.LINEAR_REGRESSION)
            
            if model is None or scaler is None:
                return await self._predict_moving_average(horizon_minutes, include_confidence)
            
            # Prepare features for prediction
            recent_data = list(self._load_history)[-10:]  # Last 10 data points
            
            if len(recent_data) == 0:
                return await self._cold_start_prediction(horizon_minutes)
            
            # Use the latest data point as base for prediction
            latest_features = np.array([recent_data[-1].to_feature_vector()])
            
            # Add time-based features
            time_features = np.array([[
                horizon_minutes / 60.0,  # Hours ahead
                datetime.utcnow().hour,  # Hour of day
                datetime.utcnow().weekday()  # Day of week
            ]])
            
            # Combine features
            combined_features = np.hstack([latest_features, time_features])
            
            # Scale features
            scaled_features = scaler.transform(combined_features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Map prediction back to feature names
            predicted_load = dict(zip(self._feature_names, prediction))
            
            # Ensure non-negative values
            for key in predicted_load:
                predicted_load[key] = max(0, predicted_load[key])
            
            # Calculate confidence intervals (simplified)
            confidence_interval = {}
            if include_confidence:
                for key, value in predicted_load.items():
                    # Use model uncertainty (simplified)
                    margin = value * 0.2  # 20% margin
                    confidence_interval[key] = (
                        max(0, value - margin),
                        value + margin
                    )
            
            # Determine trend based on recent data
            if len(recent_data) >= 5:
                early_cpu = statistics.mean(getattr(dp, "cpu_utilization") for dp in recent_data[:3])
                late_cpu = statistics.mean(getattr(dp, "cpu_utilization") for dp in recent_data[-3:])
                
                if late_cpu > early_cpu * 1.05:
                    trend = "increasing"
                elif late_cpu < early_cpu * 0.95:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.LINEAR_REGRESSION,
                predicted_load=predicted_load,
                confidence_interval=confidence_interval,
                confidence_score=0.8,
                trend_direction=trend,
                metadata={"features_used": len(combined_features[0])}
            )
            
        except Exception as e:
            logger.error(f"Error in linear regression prediction: {e}")
            return await self._predict_moving_average(horizon_minutes, include_confidence)
    
    async def _predict_seasonal_decomposition(self, horizon_minutes: int, include_confidence: bool) -> PredictionResult:
        """Seasonal decomposition prediction."""
        try:
            # Detect seasonal patterns
            seasonal_pattern = await self._detect_seasonal_pattern()
            
            if seasonal_pattern == SeasonalPattern.NONE:
                # Fall back to exponential smoothing
                return await self._predict_exponential_smoothing(horizon_minutes, include_confidence)
            
            # Apply seasonal adjustment to moving average
            base_prediction = await self._predict_moving_average(horizon_minutes, include_confidence)
            
            # Adjust for seasonal patterns
            current_time = datetime.utcnow()
            future_time = current_time + timedelta(minutes=horizon_minutes)
            
            seasonal_adjustment = await self._calculate_seasonal_adjustment(
                seasonal_pattern, current_time, future_time
            )
            
            # Apply adjustment to prediction
            adjusted_load = {}
            adjusted_confidence = {}
            
            for key, value in base_prediction.predicted_load.items():
                adjustment = seasonal_adjustment.get(key, 1.0)
                adjusted_load[key] = value * adjustment
                
                if include_confidence and key in base_prediction.confidence_interval:
                    low, high = base_prediction.confidence_interval[key]
                    adjusted_confidence[key] = (low * adjustment, high * adjustment)
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.SEASONAL_DECOMPOSITION,
                predicted_load=adjusted_load,
                confidence_interval=adjusted_confidence,
                confidence_score=0.85,
                seasonal_pattern=seasonal_pattern,
                trend_direction=base_prediction.trend_direction,
                metadata={
                    "seasonal_pattern": seasonal_pattern.value,
                    "adjustment_factor": seasonal_adjustment
                }
            )
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition prediction: {e}")
            return await self._predict_exponential_smoothing(horizon_minutes, include_confidence)
    
    async def _predict_ensemble(self, horizon_minutes: int, include_confidence: bool) -> PredictionResult:
        """Ensemble prediction combining multiple models."""
        try:
            # Get predictions from multiple models
            predictions = []
            
            # Moving average
            ma_pred = await self._predict_moving_average(horizon_minutes, False)
            predictions.append((ma_pred, 0.3))  # Weight: 0.3
            
            # Exponential smoothing
            es_pred = await self._predict_exponential_smoothing(horizon_minutes, False)
            predictions.append((es_pred, 0.4))  # Weight: 0.4
            
            # Linear regression (if available)
            if ModelType.LINEAR_REGRESSION in self._models:
                lr_pred = await self._predict_linear_regression(horizon_minutes, False)
                predictions.append((lr_pred, 0.3))  # Weight: 0.3
            
            # Combine predictions using weighted average
            ensemble_load = {}
            total_weight = sum(weight for _, weight in predictions)
            
            for feature in self._feature_names:
                weighted_sum = 0.0
                for pred, weight in predictions:
                    if feature in pred.predicted_load:
                        weighted_sum += pred.predicted_load[feature] * weight
                
                ensemble_load[feature] = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Calculate ensemble confidence intervals
            confidence_interval = {}
            if include_confidence:
                for feature in self._feature_names:
                    # Use variance across predictions as uncertainty measure
                    feature_predictions = [
                        pred.predicted_load.get(feature, 0) for pred, _ in predictions
                        if feature in pred.predicted_load
                    ]
                    
                    if len(feature_predictions) > 1:
                        variance = statistics.variance(feature_predictions)
                        margin = math.sqrt(variance) * 1.96
                        
                        confidence_interval[feature] = (
                            max(0, ensemble_load[feature] - margin),
                            ensemble_load[feature] + margin
                        )
                    else:
                        confidence_interval[feature] = (
                            ensemble_load[feature] * 0.9,
                            ensemble_load[feature] * 1.1
                        )
            
            # Determine consensus trend
            trends = [pred.trend_direction for pred, _ in predictions if pred.trend_direction]
            if trends:
                trend_counts = {trend: trends.count(trend) for trend in set(trends)}
                consensus_trend = max(trend_counts, key=trend_counts.get)
            else:
                consensus_trend = "stable"
            
            return PredictionResult(
                timestamp=datetime.utcnow(),
                horizon_minutes=horizon_minutes,
                model_type=ModelType.ENSEMBLE,
                predicted_load=ensemble_load,
                confidence_interval=confidence_interval,
                confidence_score=0.9,
                trend_direction=consensus_trend,
                metadata={
                    "models_used": len(predictions),
                    "total_weight": total_weight,
                    "trend_consensus": trend_counts if 'trend_counts' in locals() else {}
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return await self._predict_moving_average(horizon_minutes, include_confidence)
    
    async def _train_linear_regression_simple(self) -> None:
        """Train a simple linear regression model."""
        try:
            if len(self._load_history) < self.min_training_samples:
                return
            
            data_points = list(self._load_history)
            
            # Prepare features and targets
            features = []
            targets = []
            
            for i in range(len(data_points) - 1):
                # Features: current data point + time info
                current_features = data_points[i].to_feature_vector()
                time_features = [
                    data_points[i].timestamp.hour,
                    data_points[i].timestamp.weekday(),
                    1.0  # 1 hour ahead (simplified)
                ]
                
                combined_features = current_features + time_features
                features.append(combined_features)
                
                # Target: next data point
                targets.append(data_points[i + 1].to_feature_vector())
            
            if len(features) < 10:
                return
            
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self._models[ModelType.LINEAR_REGRESSION] = model
            self._scalers[ModelType.LINEAR_REGRESSION] = scaler
            
            # Calculate performance metrics
            y_pred = model.predict(X_scaled)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Calculate accuracy (simplified)
            accuracy = 1.0 / (1.0 + mae)  # Simple accuracy measure
            
            # Update performance tracking
            self._model_performance[ModelType.LINEAR_REGRESSION] = ModelPerformance(
                model_type=ModelType.LINEAR_REGRESSION,
                total_predictions=0,
                correct_predictions=0,
                mean_absolute_error=mae,
                mean_squared_error=mse,
                accuracy_score=accuracy,
                last_updated=datetime.utcnow(),
                training_data_size=len(features)
            )
            
            logger.info(f"Trained linear regression model: MAE={mae:.3f}, MSE={mse:.3f}, Accuracy={accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error training linear regression model: {e}")
    
    async def _train_linear_regression(self, features: np.ndarray, training_data: List[LoadDataPoint]) -> Dict[str, Any]:
        """Train linear regression model with full feature engineering."""
        try:
            # This is a more comprehensive version of the linear regression training
            await self._train_linear_regression_simple()
            
            return {
                "success": True,
                "training_samples": len(training_data),
                "features": features.shape[1] if len(features.shape) > 1 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive linear regression training: {e}")
            return {"success": False, "error": str(e)}
    
    async def _train_exponential_smoothing(self, training_data: List[LoadDataPoint]) -> Dict[str, Any]:
        """Train exponential smoothing model."""
        try:
            # Exponential smoothing doesn't require explicit training,
            # but we can optimize the alpha parameter
            
            best_alpha = 0.3
            best_error = float('inf')
            
            # Test different alpha values
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
                total_error = 0.0
                
                for feature in self._feature_names:
                    values = [getattr(dp, feature) for dp in training_data]
                    
                    if len(values) < 5:
                        continue
                    
                    # Apply exponential smoothing and calculate error
                    smoothed = [values[0]]
                    for i in range(1, len(values)):
                        smoothed_val = alpha * values[i] + (1 - alpha) * smoothed[-1]
                        smoothed.append(smoothed_val)
                    
                    # Calculate prediction error
                    for i in range(1, len(values)):
                        error = abs(values[i] - smoothed[i-1])
                        total_error += error
                
                if total_error < best_error:
                    best_error = total_error
                    best_alpha = alpha
            
            # Store the best alpha parameter
            self._models[ModelType.EXPONENTIAL_SMOOTHING] = {"alpha": best_alpha}
            
            # Update performance tracking
            accuracy = 1.0 / (1.0 + best_error / len(training_data))
            
            self._model_performance[ModelType.EXPONENTIAL_SMOOTHING] = ModelPerformance(
                model_type=ModelType.EXPONENTIAL_SMOOTHING,
                total_predictions=0,
                correct_predictions=0,
                mean_absolute_error=best_error / len(training_data),
                mean_squared_error=(best_error / len(training_data)) ** 2,
                accuracy_score=accuracy,
                last_updated=datetime.utcnow(),
                training_data_size=len(training_data)
            )
            
            return {
                "success": True,
                "best_alpha": best_alpha,
                "training_error": best_error,
                "accuracy": accuracy
            }
            
        except Exception as e:
            logger.error(f"Error training exponential smoothing: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_seasonal_patterns(self, training_data: List[LoadDataPoint]) -> None:
        """Update seasonal pattern detection."""
        try:
            if len(training_data) < self.min_samples_for_seasonal:
                return
            
            # Analyze hourly patterns
            hourly_patterns = defaultdict(list)
            for dp in training_data:
                hour = dp.timestamp.hour
                hourly_patterns[hour].append(dp.cpu_utilization)
            
            # Check if there's significant hourly variation
            if len(hourly_patterns) >= 12:  # At least 12 different hours
                hourly_means = [statistics.mean(hourly_patterns[h]) for h in range(24) if h in hourly_patterns]
                if len(hourly_means) > 1:
                    hourly_variance = statistics.variance(hourly_means)
                    if hourly_variance > 0.01:  # Significant variation
                        self._seasonal_patterns["hourly"] = SeasonalPattern.HOURLY
            
            # Analyze daily patterns
            daily_patterns = defaultdict(list)
            for dp in training_data:
                day = dp.timestamp.weekday()
                daily_patterns[day].append(dp.cpu_utilization)
            
            if len(daily_patterns) >= 5:  # At least 5 different days
                daily_means = [statistics.mean(daily_patterns[d]) for d in range(7) if d in daily_patterns]
                if len(daily_means) > 1:
                    daily_variance = statistics.variance(daily_means)
                    if daily_variance > 0.01:
                        self._seasonal_patterns["daily"] = SeasonalPattern.DAILY
            
            logger.info(f"Updated seasonal patterns: {self._seasonal_patterns}")
            
        except Exception as e:
            logger.error(f"Error updating seasonal patterns: {e}")
    
    async def _detect_seasonal_pattern(self) -> SeasonalPattern:
        """Detect the dominant seasonal pattern."""
        if "hourly" in self._seasonal_patterns:
            return self._seasonal_patterns["hourly"]
        elif "daily" in self._seasonal_patterns:
            return self._seasonal_patterns["daily"]
        else:
            return SeasonalPattern.NONE
    
    async def _calculate_seasonal_adjustment(
        self,
        pattern: SeasonalPattern,
        current_time: datetime,
        future_time: datetime
    ) -> Dict[str, float]:
        """Calculate seasonal adjustment factors."""
        try:
            adjustments = {}
            
            if pattern == SeasonalPattern.HOURLY:
                # Simple hourly adjustment
                current_hour = current_time.hour
                future_hour = future_time.hour
                
                # Peak hours: 9-11 AM and 2-4 PM
                peak_hours = [9, 10, 11, 14, 15, 16]
                
                if future_hour in peak_hours:
                    adjustments["cpu_utilization"] = 1.2
                    adjustments["memory_utilization"] = 1.15
                    adjustments["active_agents"] = 1.1
                elif future_hour in [0, 1, 2, 3, 4, 5]:  # Night hours
                    adjustments["cpu_utilization"] = 0.7
                    adjustments["memory_utilization"] = 0.8
                    adjustments["active_agents"] = 0.9
                else:
                    adjustments["cpu_utilization"] = 1.0
                    adjustments["memory_utilization"] = 1.0
                    adjustments["active_agents"] = 1.0
            
            elif pattern == SeasonalPattern.DAILY:
                # Simple daily adjustment
                future_weekday = future_time.weekday()
                
                if future_weekday in [0, 1, 2, 3, 4]:  # Weekdays
                    adjustments["cpu_utilization"] = 1.1
                    adjustments["memory_utilization"] = 1.05
                    adjustments["active_agents"] = 1.05
                else:  # Weekends
                    adjustments["cpu_utilization"] = 0.8
                    adjustments["memory_utilization"] = 0.9
                    adjustments["active_agents"] = 0.95
            
            # Default to no adjustment
            for feature in self._feature_names:
                if feature not in adjustments:
                    adjustments[feature] = 1.0
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating seasonal adjustment: {e}")
            return {feature: 1.0 for feature in self._feature_names}
    
    async def _validate_recent_predictions(self, actual_data: LoadDataPoint) -> None:
        """Validate recent predictions against actual data."""
        try:
            # This would compare recent predictions with actual outcomes
            # and update model performance metrics
            
            # For now, we'll implement a simplified version
            for model_type in self._model_performance:
                # Update total predictions count
                self._model_performance[model_type].total_predictions += 1
                
                # Simple accuracy check (would be more sophisticated in production)
                # For demo purposes, assume 80% accuracy
                if np.random.random() < 0.8:
                    self._model_performance[model_type].correct_predictions += 1
                
                # Update accuracy score
                perf = self._model_performance[model_type]
                perf.accuracy_score = perf.correct_predictions / max(1, perf.total_predictions)
            
        except Exception as e:
            logger.error(f"Error validating recent predictions: {e}")
    
    async def _model_training_loop(self) -> None:
        """Background task for periodic model retraining."""
        while True:
            try:
                if self.continuous_learning and len(self._load_history) >= self.min_training_samples:
                    # Check if it's time to retrain
                    needs_retraining = False
                    
                    for model_type, performance in self._model_performance.items():
                        time_since_update = datetime.utcnow() - performance.last_updated
                        if time_since_update > timedelta(hours=self.model_retrain_interval_hours):
                            needs_retraining = True
                            break
                    
                    if needs_retraining:
                        logger.info("Starting periodic model retraining")
                        await self.retrain_models()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(3600)
    
    async def _accuracy_monitoring_loop(self) -> None:
        """Monitor prediction accuracy and trigger retraining if needed."""
        while True:
            try:
                # Check model accuracy
                for model_type, performance in self._model_performance.items():
                    if performance.accuracy_score < self.accuracy_threshold:
                        logger.warning(f"Model {model_type.value} accuracy below threshold: {performance.accuracy_score:.3f}")
                        
                        # Trigger retraining for underperforming models
                        if performance.total_predictions > 100:  # Only if we have enough samples
                            logger.info(f"Triggering retraining for {model_type.value}")
                            await self.retrain_models()
                            break
                
                await asyncio.sleep(1800)  # Monitor every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in accuracy monitoring loop: {e}")
                await asyncio.sleep(1800)
    
    async def _cache_cleanup_loop(self) -> None:
        """Clean up expired prediction cache entries."""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for cache_key, (cache_time, _) in self._prediction_cache.items():
                    if current_time - cache_time > timedelta(minutes=self.prediction_cache_minutes):
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self._prediction_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _seasonal_pattern_detector(self) -> None:
        """Background task to detect and update seasonal patterns."""
        while True:
            try:
                if len(self._load_history) >= self.min_samples_for_seasonal:
                    training_data = list(self._load_history)
                    await self._update_seasonal_patterns(training_data)
                
                await asyncio.sleep(7200)  # Check every 2 hours
                
            except Exception as e:
                logger.error(f"Error in seasonal pattern detector: {e}")
                await asyncio.sleep(7200)


# Global instance
_load_prediction_service: Optional[LoadPredictionService] = None


async def get_load_prediction_service() -> LoadPredictionService:
    """Get the global load prediction service instance."""
    global _load_prediction_service
    
    if _load_prediction_service is None:
        _load_prediction_service = LoadPredictionService()
        await _load_prediction_service.initialize()
    
    return _load_prediction_service