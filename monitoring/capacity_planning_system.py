"""
CapacityPlanningSystem - Intelligent Capacity Planning and Growth Projection

Provides intelligent capacity planning with growth projections, trend analysis,
and automated scaling recommendations to maintain extraordinary performance
as the LeanVibe Agent Hive 2.0 system scales.

Key Features:
- ML-based growth trend analysis and forecasting
- Resource utilization projections with confidence intervals
- Performance threshold breach prediction
- Automated scaling recommendations with cost analysis
- Seasonal usage pattern recognition
- Business impact assessment for capacity decisions
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

# Machine learning for forecasting
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Time series forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ForecastHorizon(Enum):
    """Forecast time horizons."""
    SHORT_TERM = "1_week"      # 1 week
    MEDIUM_TERM = "1_month"    # 1 month
    LONG_TERM = "6_months"     # 6 months
    STRATEGIC = "1_year"       # 1 year


class ScalingRecommendation(Enum):
    """Types of scaling recommendations."""
    SCALE_UP = "scale_up"
    SCALE_OUT = "scale_out"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"
    SCALE_DOWN = "scale_down"


class ResourceType(Enum):
    """Types of resources for capacity planning."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    AGENTS = "agents"
    THROUGHPUT = "throughput"


@dataclass
class CapacityForecast:
    """Capacity forecast with confidence intervals."""
    resource_type: ResourceType
    forecast_horizon: ForecastHorizon
    timestamp: datetime
    
    # Forecast data
    current_usage: float
    forecasted_values: List[float]
    forecast_timestamps: List[datetime]
    
    # Confidence intervals
    lower_confidence: List[float]
    upper_confidence: List[float]
    confidence_level: float = 0.95
    
    # Model performance
    model_accuracy: float = 0.0
    model_type: str = ""
    forecast_error: float = 0.0
    
    # Trend analysis
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    trend_strength: float = 0.0
    seasonal_component: bool = False
    
    # Business context
    growth_rate_percent: float = 0.0
    capacity_breach_date: Optional[datetime] = None
    recommended_action: ScalingRecommendation = ScalingRecommendation.MAINTAIN


@dataclass
class ScalingRecommendationDetails:
    """Detailed scaling recommendation."""
    recommendation_id: str
    resource_type: ResourceType
    recommendation: ScalingRecommendation
    urgency: str  # "low", "medium", "high", "critical"
    
    # Current state
    current_usage: float
    current_capacity: float
    utilization_percent: float
    
    # Projected state
    projected_usage: float
    projected_breach_date: Optional[datetime] = None
    time_to_breach_days: Optional[int] = None
    
    # Recommendations
    recommended_capacity: float
    scaling_factor: float
    estimated_cost_impact: str = "unknown"
    implementation_complexity: str = "medium"
    
    # Business justification
    business_justification: str = ""
    performance_impact: str = ""
    risk_assessment: str = ""
    
    # Implementation details
    implementation_steps: List[str] = field(default_factory=list)
    rollback_plan: str = ""
    testing_requirements: List[str] = field(default_factory=list)


@dataclass
class CapacityAnalysis:
    """Comprehensive capacity analysis results."""
    analysis_timestamp: datetime
    analysis_period_days: int
    
    # Resource forecasts
    forecasts: Dict[ResourceType, List[CapacityForecast]] = field(default_factory=dict)
    
    # Scaling recommendations
    recommendations: List[ScalingRecommendationDetails] = field(default_factory=list)
    
    # Performance projections
    performance_projections: Dict[str, Any] = field(default_factory=dict)
    
    # Summary metrics
    overall_health_score: float = 100.0
    capacity_pressure_score: float = 0.0
    growth_acceleration: float = 0.0


class TimeSeriesForecaster:
    """Advanced time series forecasting using multiple algorithms."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
        # Model configurations
        self.model_configs = {
            'linear_regression': {
                'model': LinearRegression,
                'params': {},
                'preprocessing': 'standard'
            },
            'polynomial_regression': {
                'model': LinearRegression,
                'params': {},
                'preprocessing': 'polynomial'
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'preprocessing': 'standard'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'preprocessing': 'standard'
            }
        }
        
        # Feature engineering parameters
        self.feature_window = 10  # Use last 10 points as features
        self.polynomial_degree = 2
    
    async def create_forecast(self, resource_name: str, values: List[float],
                             timestamps: List[datetime], 
                             horizon: ForecastHorizon) -> CapacityForecast:
        """Create forecast using ensemble of models."""
        if len(values) < 20:  # Need minimum historical data
            return self._create_fallback_forecast(resource_name, values, timestamps, horizon)
        
        try:
            # Prepare data for forecasting
            X, y = await self._prepare_forecasting_data(values, timestamps)
            
            if len(X) < 10:  # Not enough prepared data
                return self._create_fallback_forecast(resource_name, values, timestamps, horizon)
            
            # Train multiple models and select best
            best_model, best_score, model_type = await self._train_ensemble_models(X, y)
            
            # Generate forecast
            forecast_points = await self._generate_forecast_points(horizon)
            forecasted_values, confidence_intervals = await self._generate_predictions(
                best_model, X, forecast_points, model_type
            )
            
            # Create forecast timestamps
            last_timestamp = timestamps[-1]
            forecast_timestamps = [
                last_timestamp + timedelta(minutes=i * self._get_interval_minutes(horizon))
                for i in range(1, len(forecasted_values) + 1)
            ]
            
            # Analyze trends
            trend_direction, trend_strength = await self._analyze_trend(values)
            seasonal_component = await self._detect_seasonality(values, timestamps)
            
            # Calculate growth rate
            growth_rate = await self._calculate_growth_rate(values, timestamps)
            
            # Determine capacity breach date
            breach_date = await self._predict_capacity_breach(
                forecasted_values, forecast_timestamps, resource_name
            )
            
            # Generate recommendation
            recommendation = await self._generate_scaling_recommendation(
                resource_name, values[-1], forecasted_values, trend_direction, growth_rate
            )
            
            resource_type = self._map_resource_name_to_type(resource_name)
            
            forecast = CapacityForecast(
                resource_type=resource_type,
                forecast_horizon=horizon,
                timestamp=datetime.utcnow(),
                current_usage=values[-1],
                forecasted_values=forecasted_values,
                forecast_timestamps=forecast_timestamps,
                lower_confidence=confidence_intervals['lower'],
                upper_confidence=confidence_intervals['upper'],
                model_accuracy=best_score,
                model_type=model_type,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_component=seasonal_component,
                growth_rate_percent=growth_rate * 100,
                capacity_breach_date=breach_date,
                recommended_action=recommendation
            )
            
            return forecast
            
        except Exception as e:
            logging.error(f"Error creating forecast for {resource_name}: {e}")
            return self._create_fallback_forecast(resource_name, values, timestamps, horizon)
    
    async def _prepare_forecasting_data(self, values: List[float], 
                                       timestamps: List[datetime]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for time series forecasting."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for forecasting")
        
        # Create feature matrix using sliding window
        X = []
        y = []
        
        for i in range(self.feature_window, len(values)):
            # Features: previous values in window
            feature_vector = values[i - self.feature_window:i]
            
            # Add time-based features
            timestamp = timestamps[i]
            time_features = [
                timestamp.hour,
                timestamp.day_of_week,
                timestamp.day,
                (timestamp - timestamps[0]).total_seconds() / 3600  # Hours since start
            ]
            
            # Combine value-based and time-based features
            feature_vector.extend(time_features)
            X.append(feature_vector)
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    async def _train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float, str]:
        """Train ensemble of models and return best performer."""
        best_model = None
        best_score = -float('inf')
        best_model_type = ""
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3) if SKLEARN_AVAILABLE else None
        
        for model_name, config in self.model_configs.items():
            try:
                # Prepare data based on preprocessing requirement
                X_processed = await self._preprocess_features(X, model_name, config['preprocessing'])
                
                # Create and train model
                model_class = config['model']
                model = model_class(**config['params'])
                
                # Cross-validation for time series
                cv_scores = []
                if tscv:
                    for train_idx, val_idx in tscv.split(X_processed):
                        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        
                        # Calculate R² score
                        score = r2_score(y_val, y_pred)
                        cv_scores.append(score)
                    
                    avg_score = np.mean(cv_scores)
                else:
                    # Fallback: simple train/test split
                    split_idx = int(0.8 * len(X_processed))
                    X_train, X_test = X_processed[:split_idx], X_processed[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    avg_score = r2_score(y_test, y_pred)
                
                # Update best model
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_model_type = model_name
                    
                    # Retrain on full dataset
                    best_model.fit(X_processed, y)
                    
                    # Store preprocessing info
                    if model_name not in self.scalers and config['preprocessing'] == 'standard':
                        self.scalers[model_name] = StandardScaler().fit(X)
            
            except Exception as e:
                logging.error(f"Error training model {model_name}: {e}")
                continue
        
        return best_model, max(0, best_score), best_model_type
    
    async def _preprocess_features(self, X: np.ndarray, model_name: str, preprocessing: str) -> np.ndarray:
        """Preprocess features based on model requirements."""
        if preprocessing == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        
        elif preprocessing == 'polynomial':
            # Polynomial features for polynomial regression
            poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            
            # Scale polynomial features
            scaler = StandardScaler()
            return scaler.fit_transform(X_poly)
        
        else:
            return X
    
    async def _generate_forecast_points(self, horizon: ForecastHorizon) -> int:
        """Generate number of forecast points based on horizon."""
        if horizon == ForecastHorizon.SHORT_TERM:
            return 7 * 24  # 7 days, hourly
        elif horizon == ForecastHorizon.MEDIUM_TERM:
            return 30 * 4  # 30 days, 6-hourly
        elif horizon == ForecastHorizon.LONG_TERM:
            return 26 * 7  # 26 weeks, weekly
        else:  # STRATEGIC
            return 12  # 12 months, monthly
    
    def _get_interval_minutes(self, horizon: ForecastHorizon) -> int:
        """Get interval in minutes between forecast points."""
        if horizon == ForecastHorizon.SHORT_TERM:
            return 60  # Hourly
        elif horizon == ForecastHorizon.MEDIUM_TERM:
            return 360  # 6-hourly
        elif horizon == ForecastHorizon.LONG_TERM:
            return 10080  # Weekly
        else:  # STRATEGIC
            return 43200  # Monthly
    
    async def _generate_predictions(self, model: Any, X_historical: np.ndarray, 
                                   forecast_points: int, model_type: str) -> Tuple[List[float], Dict[str, List[float]]]:
        """Generate predictions with confidence intervals."""
        predictions = []
        prediction_errors = []
        
        # Use last window as starting point
        last_window = X_historical[-1].copy()
        
        for _ in range(forecast_points):
            # Predict next value
            pred = model.predict(last_window.reshape(1, -1))[0]
            predictions.append(pred)
            
            # Update window for next prediction (sliding window approach)
            # Remove first value and add prediction
            last_window = np.roll(last_window, -1)
            last_window[-5] = pred  # Update the last value-based feature
            
            # Update time-based features (simplified)
            last_window[-1] += 1  # Increment time
        
        # Generate confidence intervals (simplified approach)
        # In practice, you'd use model-specific methods or bootstrap
        if len(predictions) > 0:
            # Estimate uncertainty based on historical model performance
            historical_predictions = model.predict(X_historical)
            historical_actual = X_historical[:, 0]  # First feature was previous value
            
            if len(historical_predictions) > 0:
                residuals = historical_actual[1:] - historical_predictions[:-1]  # Align arrays
                std_error = np.std(residuals) if len(residuals) > 0 else np.std(predictions) * 0.1
            else:
                std_error = np.std(predictions) * 0.1
            
            # 95% confidence interval
            z_score = 1.96
            confidence_margin = z_score * std_error
            
            lower_confidence = [max(0, pred - confidence_margin) for pred in predictions]
            upper_confidence = [pred + confidence_margin for pred in predictions]
        else:
            lower_confidence = []
            upper_confidence = []
        
        confidence_intervals = {
            'lower': lower_confidence,
            'upper': upper_confidence
        }
        
        return predictions, confidence_intervals
    
    async def _analyze_trend(self, values: List[float]) -> Tuple[str, float]:
        """Analyze trend direction and strength."""
        if len(values) < 3:
            return "stable", 0.0
        
        # Linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        
        slope, _ = np.polyfit(x, y, 1) if SCIPY_AVAILABLE else (0, 0)
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for stable trend
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Calculate trend strength (R-squared)
        if SCIPY_AVAILABLE and len(values) > 1:
            correlation, _ = stats.pearsonr(x, y)
            trend_strength = correlation ** 2
        else:
            trend_strength = 0.0
        
        return trend_direction, trend_strength
    
    async def _detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> bool:
        """Detect seasonal patterns in the data."""
        if len(values) < 48 or not SCIPY_AVAILABLE:  # Need at least 2 days of hourly data
            return False
        
        try:
            # Simple seasonality detection using autocorrelation
            # Look for patterns at 24-hour intervals
            if len(values) >= 48:  # At least 2 days
                values_array = np.array(values)
                
                # Calculate autocorrelation at lag 24 (daily pattern)
                if len(values_array) > 24:
                    lag_24_corr = np.corrcoef(values_array[:-24], values_array[24:])[0, 1]
                    
                    # Consider seasonal if correlation > 0.3
                    return not np.isnan(lag_24_corr) and lag_24_corr > 0.3
        
        except Exception:
            pass
        
        return False
    
    async def _calculate_growth_rate(self, values: List[float], timestamps: List[datetime]) -> float:
        """Calculate growth rate as percentage per time unit."""
        if len(values) < 2:
            return 0.0
        
        # Calculate compound growth rate
        start_value = values[0]
        end_value = values[-1]
        
        if start_value <= 0:
            return 0.0
        
        time_span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        
        if time_span_hours <= 0:
            return 0.0
        
        # Compound growth rate per hour
        growth_rate = ((end_value / start_value) ** (1 / time_span_hours)) - 1
        
        return growth_rate
    
    async def _predict_capacity_breach(self, forecasted_values: List[float],
                                     forecast_timestamps: List[datetime],
                                     resource_name: str) -> Optional[datetime]:
        """Predict when capacity will be breached."""
        # Define capacity thresholds based on resource type
        capacity_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'memory_usage_mb': 450.0,  # 450MB warning, 500MB critical
            'task_assignment_latency_ms': 0.02,  # 2x baseline
            'message_throughput_per_sec': 40000,  # Warning threshold
            'error_rate_percent': 0.5
        }
        
        threshold = capacity_thresholds.get(resource_name, float('inf'))
        
        # Check if any forecasted value exceeds threshold
        for value, timestamp in zip(forecasted_values, forecast_timestamps):
            if resource_name in ['message_throughput_per_sec']:
                # Higher is better - breach when below threshold
                if value < threshold:
                    return timestamp
            else:
                # Lower is better - breach when above threshold
                if value > threshold:
                    return timestamp
        
        return None
    
    async def _generate_scaling_recommendation(self, resource_name: str, current_value: float,
                                             forecasted_values: List[float], trend_direction: str,
                                             growth_rate: float) -> ScalingRecommendation:
        """Generate scaling recommendation based on forecast."""
        if not forecasted_values:
            return ScalingRecommendation.MAINTAIN
        
        max_forecast = max(forecasted_values)
        avg_forecast = sum(forecasted_values) / len(forecasted_values)
        
        # Growth-based recommendations
        if abs(growth_rate) > 0.1:  # 10% growth rate threshold
            if growth_rate > 0:  # Growing
                if trend_direction == "increasing":
                    return ScalingRecommendation.SCALE_UP
                else:
                    return ScalingRecommendation.OPTIMIZE
            else:  # Shrinking
                return ScalingRecommendation.SCALE_DOWN
        
        # Threshold-based recommendations
        capacity_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'memory_usage_mb': 400.0,
            'task_assignment_latency_ms': 0.015
        }
        
        threshold = capacity_thresholds.get(resource_name, float('inf'))
        
        if resource_name in capacity_thresholds:
            if max_forecast > threshold:
                return ScalingRecommendation.SCALE_UP
            elif avg_forecast > threshold * 0.8:
                return ScalingRecommendation.OPTIMIZE
        
        return ScalingRecommendation.MAINTAIN
    
    def _create_fallback_forecast(self, resource_name: str, values: List[float],
                                 timestamps: List[datetime], horizon: ForecastHorizon) -> CapacityForecast:
        """Create simple fallback forecast when ML models fail."""
        if not values:
            current_value = 0.0
        else:
            current_value = values[-1]
        
        # Simple linear extrapolation
        forecast_points = self._generate_forecast_points(horizon)
        
        if len(values) >= 2:
            # Calculate simple trend
            recent_values = values[-min(10, len(values)):]
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        else:
            trend = 0.0
        
        # Generate forecast
        forecasted_values = []
        for i in range(1, forecast_points + 1):
            forecasted_value = max(0, current_value + (trend * i))
            forecasted_values.append(forecasted_value)
        
        # Generate timestamps
        last_timestamp = timestamps[-1] if timestamps else datetime.utcnow()
        forecast_timestamps = [
            last_timestamp + timedelta(minutes=i * self._get_interval_minutes(horizon))
            for i in range(1, len(forecasted_values) + 1)
        ]
        
        # Simple confidence intervals (±10%)
        lower_confidence = [max(0, v * 0.9) for v in forecasted_values]
        upper_confidence = [v * 1.1 for v in forecasted_values]
        
        resource_type = self._map_resource_name_to_type(resource_name)
        
        return CapacityForecast(
            resource_type=resource_type,
            forecast_horizon=horizon,
            timestamp=datetime.utcnow(),
            current_usage=current_value,
            forecasted_values=forecasted_values,
            forecast_timestamps=forecast_timestamps,
            lower_confidence=lower_confidence,
            upper_confidence=upper_confidence,
            confidence_level=0.8,  # Lower confidence for fallback
            model_accuracy=0.5,
            model_type="linear_extrapolation",
            trend_direction="increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
            trend_strength=0.5,
            seasonal_component=False,
            growth_rate_percent=trend * 100,
            recommended_action=ScalingRecommendation.MAINTAIN
        )
    
    def _map_resource_name_to_type(self, resource_name: str) -> ResourceType:
        """Map resource name to ResourceType enum."""
        if 'cpu' in resource_name.lower():
            return ResourceType.CPU
        elif 'memory' in resource_name.lower():
            return ResourceType.MEMORY
        elif 'network' in resource_name.lower() or 'throughput' in resource_name.lower():
            return ResourceType.NETWORK
        elif 'agent' in resource_name.lower():
            return ResourceType.AGENTS
        else:
            return ResourceType.CPU  # Default


class CapacityPlanningSystem:
    """
    Intelligent capacity planning system for LeanVibe Agent Hive 2.0.
    
    Provides ML-based forecasting, trend analysis, and automated scaling
    recommendations to maintain extraordinary performance under growth.
    """
    
    def __init__(self):
        self.forecaster = TimeSeriesForecaster()
        
        # Historical data storage
        self.historical_metrics = defaultdict(lambda: {
            'values': deque(maxlen=10000),
            'timestamps': deque(maxlen=10000)
        })
        
        # Capacity planning configuration
        self.planning_config = {
            'forecast_horizons': [
                ForecastHorizon.SHORT_TERM,
                ForecastHorizon.MEDIUM_TERM,
                ForecastHorizon.LONG_TERM
            ],
            'critical_metrics': [
                'task_assignment_latency_ms',
                'message_throughput_per_sec',
                'memory_usage_mb',
                'cpu_percent',
                'error_rate_percent'
            ],
            'capacity_thresholds': {
                'cpu_percent': {'warning': 75, 'critical': 85},
                'memory_percent': {'warning': 80, 'critical': 90},
                'memory_usage_mb': {'warning': 400, 'critical': 450},
                'task_assignment_latency_ms': {'warning': 0.02, 'critical': 0.1},
                'message_throughput_per_sec': {'warning': 40000, 'critical': 25000}
            }
        }
        
        # Analysis state
        self.last_analysis = None
        self.planning_active = False
        
        # Scaling recommendations cache
        self.active_recommendations = {}
    
    async def initialize(self) -> bool:
        """Initialize capacity planning system."""
        try:
            self.planning_active = True
            logging.info("Capacity planning system initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize capacity planning system: {e}")
            return False
    
    async def add_metric_data(self, metric_name: str, value: float, timestamp: datetime = None) -> None:
        """Add new metric data point for analysis."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.historical_metrics[metric_name]['values'].append(value)
        self.historical_metrics[metric_name]['timestamps'].append(timestamp)
    
    async def run_capacity_analysis(self) -> CapacityAnalysis:
        """Run comprehensive capacity planning analysis."""
        analysis = CapacityAnalysis(
            analysis_timestamp=datetime.utcnow(),
            analysis_period_days=30  # Analyze last 30 days
        )
        
        try:
            # Generate forecasts for all critical metrics
            await self._generate_forecasts(analysis)
            
            # Create scaling recommendations
            await self._generate_scaling_recommendations(analysis)
            
            # Calculate performance projections
            await self._calculate_performance_projections(analysis)
            
            # Calculate summary metrics
            await self._calculate_summary_metrics(analysis)
            
            self.last_analysis = analysis
            logging.info("Capacity analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Error in capacity analysis: {e}")
        
        return analysis
    
    async def _generate_forecasts(self, analysis: CapacityAnalysis) -> None:
        """Generate forecasts for all critical metrics."""
        for metric_name in self.planning_config['critical_metrics']:
            if metric_name in self.historical_metrics:
                metric_data = self.historical_metrics[metric_name]
                values = list(metric_data['values'])
                timestamps = list(metric_data['timestamps'])
                
                if len(values) >= 10:  # Need minimum data for forecasting
                    forecasts = []
                    
                    for horizon in self.planning_config['forecast_horizons']:
                        try:
                            forecast = await self.forecaster.create_forecast(
                                metric_name, values, timestamps, horizon
                            )
                            forecasts.append(forecast)
                        except Exception as e:
                            logging.error(f"Error creating forecast for {metric_name} at {horizon}: {e}")
                    
                    if forecasts:
                        resource_type = self._get_resource_type_from_metric(metric_name)
                        analysis.forecasts[resource_type] = forecasts
    
    async def _generate_scaling_recommendations(self, analysis: CapacityAnalysis) -> None:
        """Generate detailed scaling recommendations."""
        for resource_type, forecasts in analysis.forecasts.items():
            for forecast in forecasts:
                if forecast.forecast_horizon == ForecastHorizon.MEDIUM_TERM:  # Focus on medium-term
                    recommendation = await self._create_scaling_recommendation_details(
                        resource_type, forecast
                    )
                    
                    if recommendation:
                        analysis.recommendations.append(recommendation)
    
    async def _create_scaling_recommendation_details(self, resource_type: ResourceType,
                                                   forecast: CapacityForecast) -> Optional[ScalingRecommendationDetails]:
        """Create detailed scaling recommendation."""
        if not forecast.forecasted_values:
            return None
        
        try:
            # Get current capacity and usage
            current_usage = forecast.current_usage
            max_forecasted = max(forecast.forecasted_values)
            
            # Determine current capacity based on resource type
            current_capacity = await self._get_current_capacity(resource_type)
            utilization_percent = (current_usage / current_capacity * 100) if current_capacity > 0 else 0
            
            # Calculate projected utilization
            projected_utilization = (max_forecasted / current_capacity * 100) if current_capacity > 0 else 0
            
            # Determine urgency
            urgency = "low"
            if forecast.capacity_breach_date:
                days_to_breach = (forecast.capacity_breach_date - datetime.utcnow()).days
                if days_to_breach <= 7:
                    urgency = "critical"
                elif days_to_breach <= 30:
                    urgency = "high"
                elif days_to_breach <= 90:
                    urgency = "medium"
            elif projected_utilization > 85:
                urgency = "high"
            elif projected_utilization > 75:
                urgency = "medium"
            
            # Calculate recommended capacity
            if forecast.recommended_action == ScalingRecommendation.SCALE_UP:
                # Scale to handle peak with 20% buffer
                recommended_capacity = max_forecasted * 1.2
                scaling_factor = recommended_capacity / current_capacity if current_capacity > 0 else 2.0
            elif forecast.recommended_action == ScalingRecommendation.SCALE_DOWN:
                # Scale down but maintain 30% buffer
                recommended_capacity = max_forecasted * 1.3
                scaling_factor = recommended_capacity / current_capacity if current_capacity > 0 else 0.8
            else:
                recommended_capacity = current_capacity
                scaling_factor = 1.0
            
            # Generate recommendation details
            recommendation = ScalingRecommendationDetails(
                recommendation_id=f"rec_{resource_type.value}_{int(datetime.utcnow().timestamp())}",
                resource_type=resource_type,
                recommendation=forecast.recommended_action,
                urgency=urgency,
                current_usage=current_usage,
                current_capacity=current_capacity,
                utilization_percent=utilization_percent,
                projected_usage=max_forecasted,
                projected_breach_date=forecast.capacity_breach_date,
                time_to_breach_days=(forecast.capacity_breach_date - datetime.utcnow()).days if forecast.capacity_breach_date else None,
                recommended_capacity=recommended_capacity,
                scaling_factor=scaling_factor
            )
            
            # Add business context
            await self._add_business_context(recommendation, forecast)
            
            return recommendation
            
        except Exception as e:
            logging.error(f"Error creating scaling recommendation for {resource_type}: {e}")
            return None
    
    async def _add_business_context(self, recommendation: ScalingRecommendationDetails,
                                   forecast: CapacityForecast) -> None:
        """Add business context to scaling recommendation."""
        # Business justification
        if recommendation.urgency == "critical":
            recommendation.business_justification = (
                "Critical capacity shortage predicted within 7 days. "
                "Immediate action required to prevent service degradation."
            )
        elif recommendation.recommendation == ScalingRecommendation.SCALE_UP:
            recommendation.business_justification = (
                f"Resource utilization trending upward with {forecast.growth_rate_percent:.1f}% growth rate. "
                "Scaling up will ensure continued performance excellence."
            )
        elif recommendation.recommendation == ScalingRecommendation.OPTIMIZE:
            recommendation.business_justification = (
                "Current resources sufficient but optimization opportunities identified. "
                "Improvements will enhance efficiency and reduce costs."
            )
        
        # Performance impact
        if recommendation.resource_type == ResourceType.CPU:
            recommendation.performance_impact = (
                "CPU scaling directly impacts task assignment latency and overall system responsiveness."
            )
        elif recommendation.resource_type == ResourceType.MEMORY:
            recommendation.performance_impact = (
                "Memory scaling prevents GC pressure and maintains sub-millisecond task assignment."
            )
        elif recommendation.resource_type == ResourceType.NETWORK:
            recommendation.performance_impact = (
                "Network capacity directly affects message throughput and routing latency."
            )
        
        # Risk assessment
        if recommendation.urgency in ["critical", "high"]:
            recommendation.risk_assessment = (
                "High risk of performance degradation if action not taken. "
                "May impact SLA compliance and user experience."
            )
        else:
            recommendation.risk_assessment = (
                "Low risk of immediate impact but monitoring recommended. "
                "Proactive scaling preferred over reactive measures."
            )
        
        # Implementation steps
        await self._generate_implementation_steps(recommendation)
    
    async def _generate_implementation_steps(self, recommendation: ScalingRecommendationDetails) -> None:
        """Generate implementation steps for scaling recommendation."""
        if recommendation.recommendation == ScalingRecommendation.SCALE_UP:
            if recommendation.resource_type == ResourceType.CPU:
                recommendation.implementation_steps = [
                    "1. Analyze current CPU bottlenecks and hotspots",
                    "2. Increase CPU allocation or add additional compute nodes",
                    "3. Update load balancing configuration",
                    "4. Monitor CPU utilization and performance metrics",
                    "5. Validate task assignment latency improvements"
                ]
            elif recommendation.resource_type == ResourceType.MEMORY:
                recommendation.implementation_steps = [
                    "1. Review memory usage patterns and allocation",
                    "2. Increase memory allocation for application",
                    "3. Optimize garbage collection settings",
                    "4. Monitor memory usage and GC performance",
                    "5. Validate memory-related performance improvements"
                ]
            elif recommendation.resource_type == ResourceType.NETWORK:
                recommendation.implementation_steps = [
                    "1. Assess network bandwidth and connection limits",
                    "2. Scale connection pool sizes and network capacity",
                    "3. Optimize message batching and compression",
                    "4. Monitor throughput and latency metrics",
                    "5. Validate communication performance improvements"
                ]
        
        elif recommendation.recommendation == ScalingRecommendation.OPTIMIZE:
            recommendation.implementation_steps = [
                "1. Conduct performance profiling and analysis",
                "2. Identify optimization opportunities",
                "3. Implement performance improvements",
                "4. Test optimizations in staging environment",
                "5. Deploy optimizations with monitoring"
            ]
        
        # Common final steps
        recommendation.implementation_steps.extend([
            f"{len(recommendation.implementation_steps) + 1}. Document changes and update capacity planning",
            f"{len(recommendation.implementation_steps) + 2}. Schedule follow-up capacity review"
        ])
        
        # Rollback plan
        recommendation.rollback_plan = (
            "1. Monitor key performance metrics during implementation. "
            "2. If degradation detected, immediately revert configuration changes. "
            "3. Restore previous resource allocation if necessary. "
            "4. Investigate issues and plan alternative approach."
        )
        
        # Testing requirements
        recommendation.testing_requirements = [
            "Load testing with increased capacity",
            "Performance regression testing",
            "Monitoring and alerting validation",
            "Rollback procedure testing"
        ]
    
    async def _calculate_performance_projections(self, analysis: CapacityAnalysis) -> None:
        """Calculate performance projections based on forecasts."""
        projections = {}
        
        # Project key performance metrics
        key_metrics = ['task_assignment_latency_ms', 'message_throughput_per_sec', 'memory_usage_mb']
        
        for resource_type, forecasts in analysis.forecasts.items():
            for forecast in forecasts:
                metric_name = self._get_metric_name_from_resource_type(resource_type)
                
                if metric_name in key_metrics and forecast.forecast_horizon == ForecastHorizon.MEDIUM_TERM:
                    projections[metric_name] = {
                        'current': forecast.current_usage,
                        'projected_peak': max(forecast.forecasted_values) if forecast.forecasted_values else forecast.current_usage,
                        'projected_average': sum(forecast.forecasted_values) / len(forecast.forecasted_values) if forecast.forecasted_values else forecast.current_usage,
                        'trend_direction': forecast.trend_direction,
                        'growth_rate_percent': forecast.growth_rate_percent,
                        'breach_prediction': forecast.capacity_breach_date.isoformat() if forecast.capacity_breach_date else None
                    }
        
        analysis.performance_projections = projections
    
    async def _calculate_summary_metrics(self, analysis: CapacityAnalysis) -> None:
        """Calculate summary metrics for capacity analysis."""
        # Overall health score (0-100)
        health_factors = []
        
        # Check current utilization levels
        for resource_type, forecasts in analysis.forecasts.items():
            for forecast in forecasts:
                if forecast.forecast_horizon == ForecastHorizon.SHORT_TERM:
                    max_forecast = max(forecast.forecasted_values) if forecast.forecasted_values else forecast.current_usage
                    current_capacity = await self._get_current_capacity(resource_type)
                    
                    if current_capacity > 0:
                        utilization = max_forecast / current_capacity
                        
                        if utilization > 0.9:
                            health_factors.append(60)  # Critical utilization
                        elif utilization > 0.8:
                            health_factors.append(75)  # High utilization
                        elif utilization > 0.7:
                            health_factors.append(85)  # Moderate utilization
                        else:
                            health_factors.append(95)  # Good utilization
        
        analysis.overall_health_score = sum(health_factors) / len(health_factors) if health_factors else 100.0
        
        # Capacity pressure score (0-100, higher = more pressure)
        pressure_factors = []
        
        critical_recommendations = [r for r in analysis.recommendations if r.urgency in ["critical", "high"]]
        pressure_factors.append(len(critical_recommendations) * 20)  # Each critical rec adds 20 points
        
        # Growth acceleration factor
        growth_rates = []
        for forecasts in analysis.forecasts.values():
            for forecast in forecasts:
                if forecast.forecast_horizon == ForecastHorizon.SHORT_TERM:
                    growth_rates.append(abs(forecast.growth_rate_percent))
        
        analysis.growth_acceleration = sum(growth_rates) / len(growth_rates) if growth_rates else 0.0
        analysis.capacity_pressure_score = min(100, sum(pressure_factors))
    
    async def _get_current_capacity(self, resource_type: ResourceType) -> float:
        """Get current capacity for resource type."""
        # These would be configured based on actual system capacity
        capacity_map = {
            ResourceType.CPU: 100.0,          # 100% CPU
            ResourceType.MEMORY: 500.0,       # 500MB memory capacity
            ResourceType.NETWORK: 100000.0,   # 100K msg/sec network capacity
            ResourceType.AGENTS: 100.0,       # 100 agents capacity
            ResourceType.THROUGHPUT: 60000.0  # 60K throughput capacity
        }
        
        return capacity_map.get(resource_type, 100.0)
    
    def _get_resource_type_from_metric(self, metric_name: str) -> ResourceType:
        """Map metric name to resource type."""
        if 'cpu' in metric_name.lower():
            return ResourceType.CPU
        elif 'memory' in metric_name.lower():
            return ResourceType.MEMORY
        elif 'throughput' in metric_name.lower() or 'network' in metric_name.lower():
            return ResourceType.NETWORK
        elif 'agent' in metric_name.lower():
            return ResourceType.AGENTS
        else:
            return ResourceType.CPU
    
    def _get_metric_name_from_resource_type(self, resource_type: ResourceType) -> str:
        """Map resource type back to metric name."""
        type_map = {
            ResourceType.CPU: 'cpu_percent',
            ResourceType.MEMORY: 'memory_usage_mb',
            ResourceType.NETWORK: 'message_throughput_per_sec',
            ResourceType.AGENTS: 'active_agents_count'
        }
        
        return type_map.get(resource_type, 'unknown_metric')
    
    def get_capacity_dashboard_data(self) -> Dict[str, Any]:
        """Get capacity planning dashboard data."""
        dashboard_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_status': {
                'planning_active': self.planning_active,
                'ml_available': SKLEARN_AVAILABLE,
                'time_series_analysis_available': STATSMODELS_AVAILABLE,
                'statistical_analysis_available': SCIPY_AVAILABLE
            },
            'last_analysis': None,
            'active_recommendations': len(self.active_recommendations),
            'metrics_tracked': len(self.historical_metrics)
        }
        
        if self.last_analysis:
            dashboard_data['last_analysis'] = {
                'timestamp': self.last_analysis.analysis_timestamp.isoformat(),
                'overall_health_score': self.last_analysis.overall_health_score,
                'capacity_pressure_score': self.last_analysis.capacity_pressure_score,
                'growth_acceleration': self.last_analysis.growth_acceleration,
                'recommendations_count': len(self.last_analysis.recommendations),
                'critical_recommendations': len([
                    r for r in self.last_analysis.recommendations 
                    if r.urgency == "critical"
                ]),
                'forecasts_generated': sum(len(f) for f in self.last_analysis.forecasts.values())
            }
        
        return dashboard_data
    
    async def get_detailed_analysis_report(self) -> Dict[str, Any]:
        """Get detailed capacity analysis report."""
        if not self.last_analysis:
            return {'error': 'No analysis available'}
        
        return {
            'analysis_summary': {
                'timestamp': self.last_analysis.analysis_timestamp.isoformat(),
                'period_days': self.last_analysis.analysis_period_days,
                'overall_health_score': self.last_analysis.overall_health_score,
                'capacity_pressure_score': self.last_analysis.capacity_pressure_score,
                'growth_acceleration': self.last_analysis.growth_acceleration
            },
            'forecasts': {
                resource_type.value: [
                    {
                        'horizon': forecast.forecast_horizon.value,
                        'current_usage': forecast.current_usage,
                        'trend_direction': forecast.trend_direction,
                        'growth_rate_percent': forecast.growth_rate_percent,
                        'model_accuracy': forecast.model_accuracy,
                        'capacity_breach_date': forecast.capacity_breach_date.isoformat() if forecast.capacity_breach_date else None,
                        'recommended_action': forecast.recommended_action.value
                    }
                    for forecast in forecasts
                ]
                for resource_type, forecasts in self.last_analysis.forecasts.items()
            },
            'recommendations': [
                {
                    'resource_type': rec.resource_type.value,
                    'recommendation': rec.recommendation.value,
                    'urgency': rec.urgency,
                    'current_utilization': rec.utilization_percent,
                    'scaling_factor': rec.scaling_factor,
                    'business_justification': rec.business_justification,
                    'time_to_breach_days': rec.time_to_breach_days,
                    'implementation_steps': rec.implementation_steps[:3]  # First 3 steps
                }
                for rec in self.last_analysis.recommendations
            ],
            'performance_projections': self.last_analysis.performance_projections
        }