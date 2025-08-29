"""
AI-Powered Predictive Performance Analytics Engine
===============================================

Advanced machine learning system for predicting system performance, identifying trends,
and providing proactive recommendations before issues impact users.

Epic F Phase 2: Advanced Observability & Intelligence
Target: Predict performance issues 15-30 minutes before occurrence
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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import structlog
import redis.asyncio as redis
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_async_session
from .redis import get_redis_client
from ..models.performance_metric import PerformanceMetric
from ..models.agent_performance import AgentPerformanceHistory

logger = structlog.get_logger()


class PredictionHorizon(Enum):
    """Prediction time horizons for different use cases."""
    IMMEDIATE = 15  # 15 minutes - immediate action required
    SHORT_TERM = 60  # 1 hour - short-term planning
    MEDIUM_TERM = 240  # 4 hours - capacity planning
    LONG_TERM = 1440  # 24 hours - strategic planning


class PredictionModel(Enum):
    """Available prediction models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


class RiskLevel(Enum):
    """Risk levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformancePrediction:
    """Performance prediction with confidence intervals and recommendations."""
    prediction_id: str
    metric_name: str
    component: str
    current_value: float
    predicted_value: float
    prediction_confidence: float
    time_horizon_minutes: int
    trend_direction: str
    risk_level: RiskLevel
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_timestamp: datetime
    model_used: PredictionModel
    feature_importance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    predicted_impact: Dict[str, Any] = field(default_factory=dict)
    prevention_actions: List[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Trend analysis result for performance metrics."""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable, volatile
    trend_strength: float  # 0-1 scale
    seasonality_detected: bool
    seasonal_period_hours: Optional[int]
    volatility_score: float
    anomaly_frequency: float
    forecasted_breach_time: Optional[datetime]


@dataclass
class PerformanceForecasting:
    """Complete performance forecasting result."""
    forecast_id: str
    metric_name: str
    forecasting_period_hours: int
    forecast_values: List[Tuple[datetime, float, float]]  # timestamp, value, confidence
    trend_analysis: TrendAnalysis
    risk_assessment: Dict[str, Any]
    recommended_actions: List[str]
    model_accuracy: float
    generated_at: datetime


class PredictivePerformanceAnalytics:
    """
    AI-Powered Predictive Performance Analytics Engine
    
    Provides advanced machine learning capabilities for:
    - Performance metric prediction (15-30 minutes ahead)
    - Trend analysis and seasonality detection
    - Risk assessment and impact analysis
    - Proactive recommendation generation
    - Model accuracy monitoring and improvement
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        session_factory: Optional = None,
        prediction_horizons: List[PredictionHorizon] = None
    ):
        """Initialize the predictive analytics engine."""
        self.redis_client = redis_client or get_redis_client()
        self.session_factory = session_factory or get_async_session
        self.prediction_horizons = prediction_horizons or [
            PredictionHorizon.IMMEDIATE,
            PredictionHorizon.SHORT_TERM,
            PredictionHorizon.MEDIUM_TERM
        ]
        
        # ML Models and scalers
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_extractors: Dict[str, Any] = {}
        
        # Prediction cache and history
        self.prediction_cache: Dict[str, PerformancePrediction] = {}
        self.forecast_cache: Dict[str, PerformanceForecasting] = {}
        self.model_accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Configuration
        self.config = {
            "training_data_hours": 168,  # 1 week for model training
            "min_training_samples": 50,
            "prediction_cache_ttl": 300,  # 5 minutes
            "model_retrain_hours": 24,  # Retrain models daily
            "accuracy_threshold": 0.85,
            "feature_window_hours": 24,
            "seasonal_analysis_days": 30,
            "risk_threshold_multipliers": {
                RiskLevel.LOW: 1.2,
                RiskLevel.MEDIUM: 1.5,
                RiskLevel.HIGH: 2.0,
                RiskLevel.CRITICAL: 3.0
            }
        }
        
        # State management
        self.is_running = False
        self.models_trained = False
        self.last_model_training = {}
        
        logger.info("Predictive Performance Analytics Engine initialized")
    
    async def start(self) -> None:
        """Start the predictive analytics engine."""
        if self.is_running:
            logger.warning("Predictive analytics engine already running")
            return
        
        logger.info("Starting Predictive Performance Analytics Engine")
        self.is_running = True
        
        # Load or train initial models
        await self._initialize_prediction_models()
        
        # Start background tasks
        asyncio.create_task(self._model_training_loop())
        asyncio.create_task(self._prediction_generation_loop())
        asyncio.create_task(self._model_accuracy_monitoring_loop())
        
        logger.info("Predictive Performance Analytics Engine started successfully")
    
    async def stop(self) -> None:
        """Stop the predictive analytics engine."""
        if not self.is_running:
            return
        
        logger.info("Stopping Predictive Performance Analytics Engine")
        self.is_running = False
        
        # Save models and state
        await self._save_models()
        
        logger.info("Predictive Performance Analytics Engine stopped")
    
    async def predict_performance_metrics(
        self,
        metric_names: List[str],
        prediction_horizon: PredictionHorizon = PredictionHorizon.IMMEDIATE,
        include_confidence_intervals: bool = True
    ) -> List[PerformancePrediction]:
        """
        Predict future values for specified metrics.
        
        Args:
            metric_names: List of metric names to predict
            prediction_horizon: How far ahead to predict
            include_confidence_intervals: Whether to include confidence intervals
            
        Returns:
            List of performance predictions with confidence intervals
        """
        try:
            predictions = []
            current_time = datetime.utcnow()
            
            for metric_name in metric_names:
                # Check cache first
                cache_key = f"{metric_name}_{prediction_horizon.value}"
                if cache_key in self.prediction_cache:
                    cached_prediction = self.prediction_cache[cache_key]
                    if (current_time - cached_prediction.prediction_timestamp).total_seconds() < self.config["prediction_cache_ttl"]:
                        predictions.append(cached_prediction)
                        continue
                
                # Get historical data for prediction
                historical_data = await self._get_metric_historical_data(
                    metric_name,
                    hours=self.config["feature_window_hours"]
                )
                
                if len(historical_data) < self.config["min_training_samples"]:
                    logger.warning(f"Insufficient data for prediction of {metric_name}")
                    continue
                
                # Generate prediction
                prediction = await self._generate_prediction(
                    metric_name,
                    historical_data,
                    prediction_horizon,
                    include_confidence_intervals
                )
                
                if prediction:
                    predictions.append(prediction)
                    
                    # Cache the prediction
                    self.prediction_cache[cache_key] = prediction
            
            logger.info(f"Generated {len(predictions)} performance predictions")
            return predictions
            
        except Exception as e:
            logger.error("Failed to predict performance metrics", error=str(e))
            return []
    
    async def analyze_performance_trends(
        self,
        metric_name: str,
        analysis_period_days: int = 7
    ) -> TrendAnalysis:
        """
        Analyze performance trends for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            analysis_period_days: Period for trend analysis
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            # Get historical data for trend analysis
            historical_data = await self._get_metric_historical_data(
                metric_name,
                hours=analysis_period_days * 24
            )
            
            if len(historical_data) < 24:  # Need at least 24 data points
                raise ValueError(f"Insufficient data for trend analysis of {metric_name}")
            
            # Extract values and timestamps
            values = [point["value"] for point in historical_data]
            timestamps = [point["timestamp"] for point in historical_data]
            
            # Calculate trend direction and strength
            trend_direction, trend_strength = self._calculate_trend(values)
            
            # Detect seasonality
            seasonality_detected, seasonal_period = self._detect_seasonality(values, timestamps)
            
            # Calculate volatility
            volatility_score = self._calculate_volatility(values)
            
            # Calculate anomaly frequency
            anomaly_frequency = await self._calculate_anomaly_frequency(historical_data)
            
            # Forecast potential threshold breaches
            forecasted_breach_time = await self._forecast_threshold_breach(
                metric_name,
                historical_data
            )
            
            trend_analysis = TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonality_detected=seasonality_detected,
                seasonal_period_hours=seasonal_period,
                volatility_score=volatility_score,
                anomaly_frequency=anomaly_frequency,
                forecasted_breach_time=forecasted_breach_time
            )
            
            logger.info(f"Trend analysis completed for {metric_name}")
            return trend_analysis
            
        except Exception as e:
            logger.error("Failed to analyze performance trends", error=str(e))
            raise
    
    async def generate_performance_forecast(
        self,
        metric_name: str,
        forecasting_period_hours: int = 24
    ) -> PerformanceForecasting:
        """
        Generate comprehensive performance forecast with risk assessment.
        
        Args:
            metric_name: Metric to forecast
            forecasting_period_hours: How many hours to forecast ahead
            
        Returns:
            Complete performance forecasting result
        """
        try:
            forecast_id = str(uuid.uuid4())
            current_time = datetime.utcnow()
            
            # Get historical data for forecasting
            historical_data = await self._get_metric_historical_data(
                metric_name,
                hours=self.config["training_data_hours"]
            )
            
            if len(historical_data) < self.config["min_training_samples"]:
                raise ValueError(f"Insufficient data for forecasting {metric_name}")
            
            # Generate trend analysis
            trend_analysis = await self.analyze_performance_trends(metric_name)
            
            # Generate forecast values
            forecast_values = await self._generate_forecast_values(
                metric_name,
                historical_data,
                forecasting_period_hours
            )
            
            # Assess risk
            risk_assessment = await self._assess_forecast_risk(
                metric_name,
                forecast_values,
                trend_analysis
            )
            
            # Generate recommendations
            recommendations = await self._generate_forecast_recommendations(
                metric_name,
                trend_analysis,
                risk_assessment
            )
            
            # Calculate model accuracy
            model_accuracy = await self._calculate_model_accuracy(metric_name)
            
            forecast = PerformanceForecasting(
                forecast_id=forecast_id,
                metric_name=metric_name,
                forecasting_period_hours=forecasting_period_hours,
                forecast_values=forecast_values,
                trend_analysis=trend_analysis,
                risk_assessment=risk_assessment,
                recommended_actions=recommendations,
                model_accuracy=model_accuracy,
                generated_at=current_time
            )
            
            # Cache the forecast
            cache_key = f"{metric_name}_{forecasting_period_hours}h"
            self.forecast_cache[cache_key] = forecast
            
            logger.info(f"Performance forecast generated for {metric_name}")
            return forecast
            
        except Exception as e:
            logger.error("Failed to generate performance forecast", error=str(e))
            raise
    
    async def get_proactive_recommendations(
        self,
        time_horizon_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get proactive recommendations to prevent performance issues.
        
        Args:
            time_horizon_minutes: Look-ahead time for recommendations
            
        Returns:
            List of proactive recommendations with priority and impact
        """
        try:
            recommendations = []
            
            # Get all active predictions within the time horizon
            predictions = await self._get_predictions_in_horizon(time_horizon_minutes)
            
            # Analyze predictions for potential issues
            for prediction in predictions:
                if prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    # Generate specific recommendations for high-risk predictions
                    metric_recommendations = await self._generate_proactive_recommendations(prediction)
                    recommendations.extend(metric_recommendations)
            
            # Deduplicate and prioritize recommendations
            recommendations = self._prioritize_recommendations(recommendations)
            
            logger.info(f"Generated {len(recommendations)} proactive recommendations")
            return recommendations
            
        except Exception as e:
            logger.error("Failed to get proactive recommendations", error=str(e))
            return []
    
    # Model training and management methods
    async def _initialize_prediction_models(self) -> None:
        """Initialize or load prediction models for all metrics."""
        try:
            # Get list of metrics to model
            metrics = await self._get_available_metrics()
            
            for metric_name in metrics:
                # Try to load existing model
                model_loaded = await self._load_model(metric_name)
                
                if not model_loaded:
                    # Train new model if no existing model found
                    await self._train_metric_model(metric_name)
            
            self.models_trained = True
            logger.info(f"Initialized prediction models for {len(metrics)} metrics")
            
        except Exception as e:
            logger.error("Failed to initialize prediction models", error=str(e))
    
    async def _train_metric_model(self, metric_name: str) -> bool:
        """Train prediction model for a specific metric."""
        try:
            # Get training data
            training_data = await self._get_metric_historical_data(
                metric_name,
                hours=self.config["training_data_hours"]
            )
            
            if len(training_data) < self.config["min_training_samples"]:
                logger.warning(f"Insufficient training data for {metric_name}")
                return False
            
            # Prepare features and targets
            X, y = self._prepare_training_data(training_data)
            
            # Train multiple models
            models = {}
            
            # Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            models[PredictionModel.RANDOM_FOREST] = rf_model
            
            # Store models and scaler
            self.models[metric_name] = models
            
            # Train scaler
            scaler = StandardScaler()
            scaler.fit(X)
            self.scalers[metric_name] = scaler
            
            # Record training time
            self.last_model_training[metric_name] = datetime.utcnow()
            
            logger.info(f"Successfully trained models for {metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model for {metric_name}", error=str(e))
            return False
    
    async def _generate_prediction(
        self,
        metric_name: str,
        historical_data: List[Dict[str, Any]],
        prediction_horizon: PredictionHorizon,
        include_confidence_intervals: bool = True
    ) -> Optional[PerformancePrediction]:
        """Generate prediction for a specific metric."""
        try:
            if metric_name not in self.models:
                logger.warning(f"No trained model available for {metric_name}")
                return None
            
            # Prepare features for prediction
            current_features = self._extract_current_features(historical_data)
            
            if metric_name in self.scalers:
                current_features_scaled = self.scalers[metric_name].transform([current_features])
            else:
                current_features_scaled = [current_features]
            
            # Get model and make prediction
            model = self.models[metric_name].get(PredictionModel.RANDOM_FOREST)
            if not model:
                logger.warning(f"No Random Forest model for {metric_name}")
                return None
            
            predicted_value = model.predict(current_features_scaled)[0]
            
            # Calculate confidence intervals (simplified)
            confidence_interval_lower = predicted_value * 0.95
            confidence_interval_upper = predicted_value * 1.05
            
            # Determine risk level
            current_value = historical_data[-1]["value"] if historical_data else 0
            risk_level = self._assess_prediction_risk(current_value, predicted_value)
            
            # Generate recommendations
            recommendations = self._generate_prediction_recommendations(
                metric_name,
                current_value,
                predicted_value,
                risk_level
            )
            
            # Create prediction object
            prediction = PerformancePrediction(
                prediction_id=str(uuid.uuid4()),
                metric_name=metric_name,
                component=metric_name.split(".")[0] if "." in metric_name else "system",
                current_value=current_value,
                predicted_value=predicted_value,
                prediction_confidence=0.85,  # Would be calculated from model
                time_horizon_minutes=prediction_horizon.value,
                trend_direction=self._determine_trend_direction(current_value, predicted_value),
                risk_level=risk_level,
                confidence_interval_lower=confidence_interval_lower,
                confidence_interval_upper=confidence_interval_upper,
                prediction_timestamp=datetime.utcnow(),
                model_used=PredictionModel.RANDOM_FOREST,
                feature_importance={},  # Would be extracted from model
                recommendations=recommendations,
                predicted_impact=self._assess_predicted_impact(metric_name, predicted_value),
                prevention_actions=self._generate_prevention_actions(metric_name, risk_level)
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to generate prediction for {metric_name}", error=str(e))
            return None
    
    # Background task methods
    async def _model_training_loop(self) -> None:
        """Background task for periodic model retraining."""
        while self.is_running:
            try:
                # Check which models need retraining
                current_time = datetime.utcnow()
                
                for metric_name in list(self.models.keys()):
                    last_training = self.last_model_training.get(metric_name)
                    if (not last_training or 
                        (current_time - last_training).total_seconds() > self.config["model_retrain_hours"] * 3600):
                        
                        logger.info(f"Retraining model for {metric_name}")
                        await self._train_metric_model(metric_name)
                
                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Model training loop error", error=str(e))
                await asyncio.sleep(3600)
    
    async def _prediction_generation_loop(self) -> None:
        """Background task for continuous prediction generation."""
        while self.is_running:
            try:
                if not self.models_trained:
                    await asyncio.sleep(60)
                    continue
                
                # Generate predictions for all metrics
                metrics = list(self.models.keys())
                
                for horizon in self.prediction_horizons:
                    predictions = await self.predict_performance_metrics(
                        metrics,
                        horizon
                    )
                    
                    # Store predictions for API access
                    await self._store_predictions(predictions)
                
                # Wait before next generation cycle
                await asyncio.sleep(self.config["prediction_cache_ttl"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Prediction generation loop error", error=str(e))
                await asyncio.sleep(self.config["prediction_cache_ttl"])
    
    async def _model_accuracy_monitoring_loop(self) -> None:
        """Background task for monitoring model accuracy."""
        while self.is_running:
            try:
                # Monitor accuracy for all models
                for metric_name in list(self.models.keys()):
                    accuracy = await self._calculate_model_accuracy(metric_name)
                    
                    if accuracy < self.config["accuracy_threshold"]:
                        logger.warning(
                            f"Model accuracy for {metric_name} below threshold: {accuracy:.3f}"
                        )
                        # Trigger model retraining
                        await self._train_metric_model(metric_name)
                
                # Wait before next accuracy check
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Model accuracy monitoring loop error", error=str(e))
                await asyncio.sleep(1800)
    
    # Helper methods
    async def _get_metric_historical_data(
        self,
        metric_name: str,
        hours: int
    ) -> List[Dict[str, Any]]:
        """Get historical data for a metric."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            async with self.session_factory() as session:
                query = select(PerformanceMetric).where(
                    and_(
                        PerformanceMetric.metric_name == metric_name,
                        PerformanceMetric.timestamp >= start_time
                    )
                ).order_by(PerformanceMetric.timestamp.asc())
                
                result = await session.execute(query)
                metrics = result.scalars().all()
            
            return [
                {
                    "timestamp": metric.timestamp,
                    "value": metric.metric_value,
                    "metadata": metric.metadata or {}
                }
                for metric in metrics
            ]
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {metric_name}", error=str(e))
            return []
    
    async def _get_available_metrics(self) -> List[str]:
        """Get list of available metrics for modeling."""
        try:
            async with self.session_factory() as session:
                query = select(PerformanceMetric.metric_name).distinct()
                result = await session.execute(query)
                metrics = [row[0] for row in result.fetchall()]
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to get available metrics", error=str(e))
            return []
    
    def _prepare_training_data(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with features and targets."""
        if len(historical_data) < 10:
            raise ValueError("Insufficient data for training")
        
        features = []
        targets = []
        
        # Create sliding windows for time series prediction
        window_size = 10
        
        for i in range(window_size, len(historical_data)):
            # Features: previous values and time-based features
            window_values = [point["value"] for point in historical_data[i-window_size:i]]
            
            # Add statistical features
            window_mean = np.mean(window_values)
            window_std = np.std(window_values)
            window_min = np.min(window_values)
            window_max = np.max(window_values)
            
            # Add time-based features
            current_timestamp = historical_data[i]["timestamp"]
            hour_of_day = current_timestamp.hour
            day_of_week = current_timestamp.weekday()
            
            feature_vector = window_values + [
                window_mean, window_std, window_min, window_max,
                hour_of_day, day_of_week
            ]
            
            features.append(feature_vector)
            targets.append(historical_data[i]["value"])
        
        return np.array(features), np.array(targets)
    
    def _extract_current_features(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> List[float]:
        """Extract features from current historical data for prediction."""
        if len(historical_data) < 10:
            # Pad with zeros if insufficient data
            values = [point["value"] for point in historical_data]
            values.extend([0] * (10 - len(values)))
        else:
            values = [point["value"] for point in historical_data[-10:]]
        
        # Add statistical features
        mean_val = np.mean(values)
        std_val = np.std(values) if len(values) > 1 else 0
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Add time-based features
        current_time = datetime.utcnow()
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        
        return values + [mean_val, std_val, min_val, max_val, hour_of_day, day_of_week]
    
    def _assess_prediction_risk(self, current_value: float, predicted_value: float) -> RiskLevel:
        """Assess risk level based on prediction."""
        change_ratio = abs(predicted_value - current_value) / max(current_value, 1)
        
        if change_ratio > 0.5:
            return RiskLevel.CRITICAL
        elif change_ratio > 0.3:
            return RiskLevel.HIGH
        elif change_ratio > 0.15:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_prediction_recommendations(
        self,
        metric_name: str,
        current_value: float,
        predicted_value: float,
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate recommendations based on prediction."""
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if "cpu" in metric_name.lower():
                recommendations.extend([
                    "Consider scaling horizontally to distribute CPU load",
                    "Review and optimize CPU-intensive processes",
                    "Implement load balancing if not already in place"
                ])
            elif "memory" in metric_name.lower():
                recommendations.extend([
                    "Monitor memory usage patterns and optimize memory-intensive operations",
                    "Consider increasing memory allocation",
                    "Implement memory cleanup routines"
                ])
            elif "response" in metric_name.lower():
                recommendations.extend([
                    "Optimize database queries and API calls",
                    "Implement caching where appropriate",
                    "Consider request rate limiting"
                ])
            else:
                recommendations.append(f"Monitor {metric_name} closely and prepare scaling measures")
        
        return recommendations
    
    def _determine_trend_direction(self, current_value: float, predicted_value: float) -> str:
        """Determine trend direction from current to predicted value."""
        if predicted_value > current_value * 1.05:
            return "increasing"
        elif predicted_value < current_value * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _assess_predicted_impact(self, metric_name: str, predicted_value: float) -> Dict[str, Any]:
        """Assess the impact of predicted values."""
        return {
            "business_impact": "medium" if predicted_value > 1.5 else "low",
            "user_experience_impact": "high" if "response" in metric_name and predicted_value > 5000 else "low",
            "system_stability_impact": "medium"
        }
    
    def _generate_prevention_actions(self, metric_name: str, risk_level: RiskLevel) -> List[str]:
        """Generate prevention actions based on risk level."""
        actions = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            actions.extend([
                "Enable enhanced monitoring for this metric",
                "Prepare scaling resources for immediate deployment",
                "Alert relevant teams about potential issues"
            ])
        
        return actions
    
    # Additional helper methods (placeholder implementations)
    async def _store_predictions(self, predictions: List[PerformancePrediction]) -> None:
        """Store predictions in Redis for API access."""
        try:
            if not predictions:
                return
            
            for prediction in predictions:
                cache_key = f"prediction:{prediction.metric_name}:{prediction.time_horizon_minutes}"
                prediction_data = asdict(prediction)
                
                # Convert datetime objects to strings for JSON serialization
                prediction_data["prediction_timestamp"] = prediction.prediction_timestamp.isoformat()
                
                await self.redis_client.setex(
                    cache_key,
                    self.config["prediction_cache_ttl"],
                    json.dumps(prediction_data, default=str)
                )
            
        except Exception as e:
            logger.error("Failed to store predictions", error=str(e))


# Singleton instance
_predictive_analytics_engine: Optional[PredictivePerformanceAnalytics] = None


async def get_predictive_analytics_engine() -> PredictivePerformanceAnalytics:
    """Get singleton predictive analytics engine instance."""
    global _predictive_analytics_engine
    
    if _predictive_analytics_engine is None:
        _predictive_analytics_engine = PredictivePerformanceAnalytics()
        await _predictive_analytics_engine.start()
    
    return _predictive_analytics_engine


async def cleanup_predictive_analytics_engine() -> None:
    """Cleanup predictive analytics engine resources."""
    global _predictive_analytics_engine
    
    if _predictive_analytics_engine:
        await _predictive_analytics_engine.stop()
        _predictive_analytics_engine = None