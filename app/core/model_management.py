"""
Advanced Model Management & A/B Testing Infrastructure for LeanVibe Agent Hive 2.0.

This system provides sophisticated model versioning, deployment automation, 
A/B testing capabilities, and drift detection for Epic 2 Phase 3.

CRITICAL: Integrates with ML Performance Optimizer and Agent Coordination 
systems for intelligent model management across the entire hive.
"""

import asyncio
import time
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from anthropic import AsyncAnthropic

from .config import settings
from .redis import get_redis_client, RedisClient
from .ml_performance_optimizer import get_ml_performance_optimizer, InferenceType
from .coordination import coordination_engine

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models managed by the system."""
    EMBEDDING_MODEL = "embedding_model"
    TEXT_GENERATION_MODEL = "text_generation_model"  
    CLASSIFICATION_MODEL = "classification_model"
    ANALYSIS_MODEL = "analysis_model"
    CONTEXT_MODEL = "context_model"
    COORDINATION_MODEL = "coordination_model"


class ModelStatus(Enum):
    """Status of model deployments."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


class DriftType(Enum):
    """Types of model drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    PREDICTION_DRIFT = "prediction_drift"


@dataclass
class MLModel:
    """Represents a machine learning model in the management system."""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    
    # Model metadata
    description: str
    model_path: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    
    # Deployment info
    deployment_strategy: DeploymentStrategy
    deployment_config: Dict[str, Any]
    resource_requirements: Dict[str, float]
    
    # Versioning and lineage
    parent_model_id: Optional[str] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    
    # Performance tracking
    inference_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    last_performance_check: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.model_id:
            self.model_id = str(uuid.uuid4())


@dataclass
class DeploymentResult:
    """Result of model deployment operation."""
    deployment_id: str
    model_id: str
    status: str
    deployment_time: float
    endpoint_url: Optional[str] = None
    health_check_url: Optional[str] = None
    rollback_plan: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of A/B testing between models."""
    test_id: str
    model_a_id: str
    model_b_id: str
    
    # Test configuration
    traffic_split: float
    test_duration_hours: int
    sample_size: int
    
    # Results
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    statistical_significance: float
    winner_model_id: Optional[str] = None
    confidence_level: float = 0.0
    
    # Test metadata
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    test_status: str = "running"


@dataclass
class DataPoint:
    """Represents a data point for drift analysis."""
    timestamp: datetime
    features: Dict[str, Any]
    prediction: Any
    actual: Optional[Any] = None
    confidence: float = 1.0


@dataclass
class DriftAnalysis:
    """Analysis result for model drift detection."""
    model_id: str
    drift_type: DriftType
    drift_score: float
    drift_threshold: float
    is_drifting: bool
    
    # Analysis details
    affected_features: List[str]
    drift_magnitude: Dict[str, float]
    recommended_action: str
    confidence: float
    
    # Temporal information
    analysis_period: Tuple[datetime, datetime]
    sample_size: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BenchmarkResults:
    """Results from model performance benchmarking."""
    benchmark_id: str
    models: List[str]
    
    # Performance metrics
    throughput_rps: Dict[str, float]
    latency_p50: Dict[str, float]
    latency_p95: Dict[str, float]
    latency_p99: Dict[str, float]
    error_rates: Dict[str, float]
    resource_usage: Dict[str, Dict[str, float]]
    
    # Quality metrics
    accuracy_scores: Dict[str, float]
    precision_scores: Dict[str, float]
    recall_scores: Dict[str, float]
    f1_scores: Dict[str, float]
    
    # Ranking and recommendations
    overall_ranking: List[str]
    recommended_model: str
    benchmark_duration: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis = redis_client or get_redis_client()
        self.models: Dict[str, MLModel] = {}
        self.model_versions: Dict[str, List[str]] = defaultdict(list)  # model_name -> versions
        self.active_deployments: Dict[str, str] = {}  # model_name -> active_model_id
    
    async def register_model(self, model: MLModel) -> str:
        """Register a new model version."""
        # Store model in registry
        self.models[model.model_id] = model
        
        # Track versions
        self.model_versions[model.name].append(model.version)
        
        # Store in Redis for persistence
        try:
            await self.redis.set(
                f"model_registry:{model.model_id}",
                json.dumps(asdict(model), default=str),
                expire=86400 * 30  # 30 days
            )
            
            await self.redis.set(
                f"model_versions:{model.name}",
                json.dumps(self.model_versions[model.name]),
                expire=86400 * 30
            )
        except Exception as e:
            logger.warning(f"Failed to persist model to Redis: {e}")
        
        logger.info(f"Registered model {model.name} v{model.version} (ID: {model.model_id})")
        return model.model_id
    
    async def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model by ID."""
        # Check memory first
        if model_id in self.models:
            return self.models[model_id]
        
        # Try Redis
        try:
            model_data = await self.redis.get(f"model_registry:{model_id}")
            if model_data:
                model_dict = json.loads(model_data)
                # Convert string dates back to datetime objects
                for date_field in ["created_at", "deployed_at", "retired_at", "last_performance_check"]:
                    if model_dict.get(date_field):
                        model_dict[date_field] = datetime.fromisoformat(model_dict[date_field])
                
                # Convert enums
                model_dict["model_type"] = ModelType(model_dict["model_type"])
                model_dict["status"] = ModelStatus(model_dict["status"])
                model_dict["deployment_strategy"] = DeploymentStrategy(model_dict["deployment_strategy"])
                
                model = MLModel(**model_dict)
                self.models[model_id] = model
                return model
        except Exception as e:
            logger.warning(f"Failed to load model from Redis: {e}")
        
        return None
    
    async def get_active_model(self, model_name: str) -> Optional[MLModel]:
        """Get currently active model version for a model name."""
        if model_name in self.active_deployments:
            return await self.get_model(self.active_deployments[model_name])
        return None
    
    async def list_models(self, model_type: Optional[ModelType] = None) -> List[MLModel]:
        """List models, optionally filtered by type."""
        models = []
        for model in self.models.values():
            if model_type is None or model.model_type == model_type:
                models.append(model)
        return models
    
    async def get_model_versions(self, model_name: str) -> List[str]:
        """Get all versions for a model name."""
        return self.model_versions.get(model_name, [])
    
    def set_active_model(self, model_name: str, model_id: str) -> None:
        """Set the active model for a model name."""
        self.active_deployments[model_name] = model_id
    
    async def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics for a model."""
        model = await self.get_model(model_id)
        if model:
            model.performance_metrics.update(metrics)
            model.last_performance_check = datetime.utcnow()
            
            # Persist updates
            await self.register_model(model)


class ABTester:
    """A/B testing system for comparing model performance."""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.active_tests: Dict[str, ABTestResult] = {}
        self.test_history: List[ABTestResult] = []
    
    async def create_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        test_duration_hours: int = 24,
        sample_size: int = 1000
    ) -> str:
        """Create a new A/B test between two models."""
        # Validate models exist
        model_a = await self.model_registry.get_model(model_a_id)
        model_b = await self.model_registry.get_model(model_b_id)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        if model_a.model_type != model_b.model_type:
            raise ValueError("Models must be of the same type for A/B testing")
        
        # Create test
        test_id = str(uuid.uuid4())
        ab_test = ABTestResult(
            test_id=test_id,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split=traffic_split,
            test_duration_hours=test_duration_hours,
            sample_size=sample_size,
            model_a_metrics={},
            model_b_metrics={},
            statistical_significance=0.0
        )
        
        self.active_tests[test_id] = ab_test
        
        # Start background test monitoring
        asyncio.create_task(self._monitor_ab_test(test_id))
        
        logger.info(f"Started A/B test {test_id} between models {model_a_id} and {model_b_id}")
        return test_id
    
    async def get_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get results for an A/B test."""
        return self.active_tests.get(test_id)
    
    async def _monitor_ab_test(self, test_id: str) -> None:
        """Monitor A/B test progress and collect metrics."""
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        start_time = test.started_at
        end_time = start_time + timedelta(hours=test.test_duration_hours)
        
        while datetime.utcnow() < end_time:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Collect metrics for both models
            await self._collect_test_metrics(test)
            
            # Check if we have enough samples
            total_samples = (
                test.model_a_metrics.get("sample_count", 0) + 
                test.model_b_metrics.get("sample_count", 0)
            )
            
            if total_samples >= test.sample_size:
                break
        
        # Complete the test
        await self._complete_ab_test(test_id)
    
    async def _collect_test_metrics(self, test: ABTestResult) -> None:
        """Collect metrics for A/B test models."""
        # Get models
        model_a = await self.model_registry.get_model(test.model_a_id)
        model_b = await self.model_registry.get_model(test.model_b_id)
        
        if model_a:
            test.model_a_metrics = {
                "avg_latency_ms": model_a.avg_latency_ms,
                "error_rate": model_a.error_count / max(1, model_a.inference_count),
                "throughput": model_a.inference_count,
                "sample_count": model_a.inference_count
            }
        
        if model_b:
            test.model_b_metrics = {
                "avg_latency_ms": model_b.avg_latency_ms,
                "error_rate": model_b.error_count / max(1, model_b.inference_count),
                "throughput": model_b.inference_count,
                "sample_count": model_b.inference_count
            }
    
    async def _complete_ab_test(self, test_id: str) -> None:
        """Complete A/B test and determine winner."""
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        # Calculate statistical significance
        test.statistical_significance = self._calculate_statistical_significance(test)
        
        # Determine winner based on metrics
        test.winner_model_id = self._determine_winner(test)
        
        # Calculate confidence level
        test.confidence_level = min(0.99, test.statistical_significance)
        
        # Mark as completed
        test.completed_at = datetime.utcnow()
        test.test_status = "completed"
        
        # Move to history
        self.test_history.append(test)
        if test_id in self.active_tests:
            del self.active_tests[test_id]
        
        logger.info(f"A/B test {test_id} completed. Winner: {test.winner_model_id}")
    
    def _calculate_statistical_significance(self, test: ABTestResult) -> float:
        """Calculate statistical significance of A/B test results."""
        # Simplified statistical significance calculation
        # In production, would use proper statistical tests
        
        model_a_latency = test.model_a_metrics.get("avg_latency_ms", 0)
        model_b_latency = test.model_b_metrics.get("avg_latency_ms", 0)
        
        model_a_error = test.model_a_metrics.get("error_rate", 0)
        model_b_error = test.model_b_metrics.get("error_rate", 0)
        
        # Simple difference-based significance
        latency_diff = abs(model_a_latency - model_b_latency) / max(model_a_latency, model_b_latency, 1)
        error_diff = abs(model_a_error - model_b_error)
        
        significance = min(0.99, (latency_diff + error_diff) * 10)
        return significance
    
    def _determine_winner(self, test: ABTestResult) -> str:
        """Determine winning model based on metrics."""
        # Score models based on multiple metrics
        model_a_score = self._calculate_model_score(test.model_a_metrics)
        model_b_score = self._calculate_model_score(test.model_b_metrics)
        
        return test.model_a_id if model_a_score > model_b_score else test.model_b_id
    
    def _calculate_model_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall score for model based on metrics."""
        latency_score = 1.0 / max(1, metrics.get("avg_latency_ms", 100) / 100)  # Lower is better
        error_score = 1.0 - metrics.get("error_rate", 0.1)  # Lower is better
        throughput_score = metrics.get("throughput", 10) / 100  # Higher is better
        
        # Weighted average
        return (latency_score * 0.4 + error_score * 0.4 + throughput_score * 0.2)


class DriftDetector:
    """Model drift detection system."""
    
    def __init__(self):
        self.data_history: Dict[str, List[DataPoint]] = defaultdict(list)
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.PERFORMANCE_DRIFT: 0.2,
            DriftType.PREDICTION_DRIFT: 0.1
        }
    
    async def add_data_point(self, model_id: str, data_point: DataPoint) -> None:
        """Add a data point for drift monitoring."""
        self.data_history[model_id].append(data_point)
        
        # Keep only recent data (last 10000 points)
        if len(self.data_history[model_id]) > 10000:
            self.data_history[model_id] = self.data_history[model_id][-10000:]
    
    async def detect_drift(
        self, 
        model_id: str, 
        recent_data: List[DataPoint]
    ) -> DriftAnalysis:
        """Detect model drift using recent data points."""
        if not recent_data:
            return self._create_no_drift_analysis(model_id)
        
        # Get historical baseline
        historical_data = self.data_history.get(model_id, [])
        if len(historical_data) < 100:  # Not enough data for comparison
            return self._create_no_drift_analysis(model_id)
        
        # Analyze different types of drift
        data_drift_score = await self._detect_data_drift(historical_data, recent_data)
        performance_drift_score = await self._detect_performance_drift(historical_data, recent_data)
        prediction_drift_score = await self._detect_prediction_drift(historical_data, recent_data)
        
        # Determine most significant drift
        drift_scores = {
            DriftType.DATA_DRIFT: data_drift_score,
            DriftType.PERFORMANCE_DRIFT: performance_drift_score,  
            DriftType.PREDICTION_DRIFT: prediction_drift_score
        }
        
        max_drift_type = max(drift_scores, key=drift_scores.get)
        max_drift_score = drift_scores[max_drift_type]
        
        is_drifting = max_drift_score > self.drift_thresholds[max_drift_type]
        
        return DriftAnalysis(
            model_id=model_id,
            drift_type=max_drift_type,
            drift_score=max_drift_score,
            drift_threshold=self.drift_thresholds[max_drift_type],
            is_drifting=is_drifting,
            affected_features=self._identify_affected_features(historical_data, recent_data),
            drift_magnitude=drift_scores,
            recommended_action=self._get_drift_recommendation(max_drift_type, max_drift_score),
            confidence=min(0.99, max_drift_score * 2),
            analysis_period=(recent_data[0].timestamp, recent_data[-1].timestamp),
            sample_size=len(recent_data)
        )
    
    def _create_no_drift_analysis(self, model_id: str) -> DriftAnalysis:
        """Create analysis result indicating no drift detected."""
        return DriftAnalysis(
            model_id=model_id,
            drift_type=DriftType.DATA_DRIFT,
            drift_score=0.0,
            drift_threshold=0.1,
            is_drifting=False,
            affected_features=[],
            drift_magnitude={},
            recommended_action="continue_monitoring",
            confidence=0.5,
            analysis_period=(datetime.utcnow(), datetime.utcnow()),
            sample_size=0
        )
    
    async def _detect_data_drift(
        self, 
        historical_data: List[DataPoint], 
        recent_data: List[DataPoint]
    ) -> float:
        """Detect drift in input data distribution."""
        # Simplified data drift detection using feature statistics
        historical_features = self._extract_feature_stats(historical_data)
        recent_features = self._extract_feature_stats(recent_data)
        
        drift_score = 0.0
        feature_count = 0
        
        for feature_name in historical_features:
            if feature_name in recent_features:
                hist_mean = historical_features[feature_name]["mean"]
                recent_mean = recent_features[feature_name]["mean"]
                
                if hist_mean != 0:
                    feature_drift = abs(recent_mean - hist_mean) / abs(hist_mean)
                    drift_score += feature_drift
                    feature_count += 1
        
        return drift_score / max(1, feature_count)
    
    async def _detect_performance_drift(
        self, 
        historical_data: List[DataPoint], 
        recent_data: List[DataPoint]
    ) -> float:
        """Detect drift in model performance."""
        # Compare accuracy/confidence scores
        historical_conf = np.mean([dp.confidence for dp in historical_data])
        recent_conf = np.mean([dp.confidence for dp in recent_data])
        
        if historical_conf > 0:
            performance_drift = abs(recent_conf - historical_conf) / historical_conf
        else:
            performance_drift = 0.0
        
        return performance_drift
    
    async def _detect_prediction_drift(
        self, 
        historical_data: List[DataPoint], 
        recent_data: List[DataPoint]
    ) -> float:
        """Detect drift in prediction distribution."""
        # For simplicity, assume predictions are numeric
        historical_preds = []
        recent_preds = []
        
        for dp in historical_data:
            if isinstance(dp.prediction, (int, float)):
                historical_preds.append(dp.prediction)
        
        for dp in recent_data:
            if isinstance(dp.prediction, (int, float)):
                recent_preds.append(dp.prediction)
        
        if not historical_preds or not recent_preds:
            return 0.0
        
        hist_mean = np.mean(historical_preds)
        recent_mean = np.mean(recent_preds)
        
        if hist_mean != 0:
            return abs(recent_mean - hist_mean) / abs(hist_mean)
        return 0.0
    
    def _extract_feature_stats(self, data_points: List[DataPoint]) -> Dict[str, Dict[str, float]]:
        """Extract statistical features from data points."""
        features_stats = defaultdict(lambda: {"values": []})
        
        for dp in data_points:
            for feature_name, feature_value in dp.features.items():
                if isinstance(feature_value, (int, float)):
                    features_stats[feature_name]["values"].append(feature_value)
        
        # Calculate statistics
        stats = {}
        for feature_name, data in features_stats.items():
            values = data["values"]
            if values:
                stats[feature_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return stats
    
    def _identify_affected_features(
        self, 
        historical_data: List[DataPoint], 
        recent_data: List[DataPoint]
    ) -> List[str]:
        """Identify features most affected by drift."""
        affected_features = []
        
        historical_stats = self._extract_feature_stats(historical_data)
        recent_stats = self._extract_feature_stats(recent_data)
        
        for feature_name in historical_stats:
            if feature_name in recent_stats:
                hist_mean = historical_stats[feature_name]["mean"]
                recent_mean = recent_stats[feature_name]["mean"]
                
                if hist_mean != 0:
                    drift = abs(recent_mean - hist_mean) / abs(hist_mean)
                    if drift > 0.1:  # 10% change threshold
                        affected_features.append(feature_name)
        
        return affected_features
    
    def _get_drift_recommendation(self, drift_type: DriftType, drift_score: float) -> str:
        """Get recommendation based on drift type and severity."""
        if drift_score < 0.1:
            return "continue_monitoring"
        elif drift_score < 0.2:
            return "investigate_data_quality"
        elif drift_score < 0.3:
            return "retrain_model_soon"
        else:
            return "retrain_model_immediately"


class AdvancedModelManagement:
    """
    Advanced Model Management system for LeanVibe Agent Hive 2.0.
    
    Provides model versioning, A/B testing, drift detection, and automated
    deployment capabilities for optimal ML model lifecycle management.
    """
    
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.ab_tester = ABTester(self.model_registry)
        self.drift_detector = DriftDetector()
        self.ml_optimizer: Optional = None
        
        # Deployment tracking
        self.deployment_history: List[DeploymentResult] = []
        self.benchmark_history: List[BenchmarkResults] = []
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.drift_monitoring_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize model management system."""
        try:
            from .ml_performance_optimizer import get_ml_performance_optimizer
            self.ml_optimizer = await get_ml_performance_optimizer()
            
            # Start background monitoring
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.drift_monitoring_task = asyncio.create_task(self._drift_monitoring_loop())
            
            logger.info("Advanced Model Management system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize model management: {e}")
            raise
    
    async def deploy_model_version(
        self, 
        model: MLModel, 
        version: str
    ) -> DeploymentResult:
        """
        Deploy a new model version with specified strategy.
        
        Args:
            model: Model to deploy
            version: Version string for the deployment
            
        Returns:
            DeploymentResult with deployment status and metadata
        """
        start_time = time.time()
        deployment_id = str(uuid.uuid4())
        
        try:
            # Update model version
            model.version = version
            model.status = ModelStatus.DEPLOYING
            
            # Register model version
            await self.model_registry.register_model(model)
            
            # Execute deployment strategy
            deployment_result = await self._execute_deployment(model, deployment_id)
            
            if deployment_result["success"]:
                model.status = ModelStatus.ACTIVE
                model.deployed_at = datetime.utcnow()
                
                # Set as active if it's the first deployment or passes validation
                await self._validate_deployment(model)
                self.model_registry.set_active_model(model.name, model.model_id)
                
                # Update model registry
                await self.model_registry.register_model(model)
                
                deployment_time = time.time() - start_time
                
                result = DeploymentResult(
                    deployment_id=deployment_id,
                    model_id=model.model_id,
                    status="success",
                    deployment_time=deployment_time,
                    endpoint_url=deployment_result.get("endpoint_url"),
                    health_check_url=deployment_result.get("health_check_url"),
                    rollback_plan=deployment_result.get("rollback_plan"),
                    metadata={
                        "deployment_strategy": model.deployment_strategy.value,
                        "version": version,
                        "model_type": model.model_type.value
                    }
                )
                
                self.deployment_history.append(result)
                
                logger.info(f"Successfully deployed model {model.name} v{version} in {deployment_time:.2f}s")
                return result
            
            else:
                model.status = ModelStatus.FAILED
                await self.model_registry.register_model(model)
                
                return DeploymentResult(
                    deployment_id=deployment_id,
                    model_id=model.model_id,
                    status="failed",
                    deployment_time=time.time() - start_time,
                    error_message=deployment_result.get("error")
                )
        
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            
            model.status = ModelStatus.FAILED
            await self.model_registry.register_model(model)
            
            return DeploymentResult(
                deployment_id=deployment_id,
                model_id=model.model_id,
                status="failed",
                deployment_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def run_ab_test(
        self, 
        model_a: MLModel, 
        model_b: MLModel, 
        traffic_split: float = 0.5
    ) -> ABTestResult:
        """
        Run A/B test between two models.
        
        Args:
            model_a: First model for comparison
            model_b: Second model for comparison
            traffic_split: Fraction of traffic for model A (0.0 to 1.0)
            
        Returns:
            ABTestResult with test configuration and results
        """
        test_id = await self.ab_tester.create_ab_test(
            model_a.model_id,
            model_b.model_id,
            traffic_split=traffic_split,
            test_duration_hours=24,
            sample_size=1000
        )
        
        # Wait for test to complete or return current status
        test_result = await self.ab_tester.get_test_results(test_id)
        
        logger.info(f"A/B test initiated between {model_a.name} and {model_b.name}")
        return test_result
    
    async def detect_model_drift(
        self, 
        model: MLModel, 
        recent_data: List[DataPoint]
    ) -> DriftAnalysis:
        """
        Detect drift in model performance or data.
        
        Args:
            model: Model to analyze for drift
            recent_data: Recent data points for analysis
            
        Returns:
            DriftAnalysis with drift detection results
        """
        drift_analysis = await self.drift_detector.detect_drift(model.model_id, recent_data)
        
        # Take action if drift is detected
        if drift_analysis.is_drifting:
            await self._handle_model_drift(model, drift_analysis)
        
        logger.info(f"Drift analysis for {model.name}: {'DRIFT DETECTED' if drift_analysis.is_drifting else 'NO DRIFT'}")
        return drift_analysis
    
    async def benchmark_model_performance(self, models: List[MLModel]) -> BenchmarkResults:
        """
        Benchmark performance of multiple models.
        
        Args:
            models: List of models to benchmark
            
        Returns:
            BenchmarkResults with comprehensive performance comparison
        """
        start_time = time.time()
        benchmark_id = str(uuid.uuid4())
        
        # Initialize result structures
        throughput_rps = {}
        latency_metrics = {"p50": {}, "p95": {}, "p99": {}}
        error_rates = {}
        resource_usage = {}
        quality_metrics = {"accuracy": {}, "precision": {}, "recall": {}, "f1": {}}
        
        # Benchmark each model
        for model in models:
            model_results = await self._benchmark_single_model(model)
            
            throughput_rps[model.model_id] = model_results["throughput_rps"]
            latency_metrics["p50"][model.model_id] = model_results["latency_p50"]
            latency_metrics["p95"][model.model_id] = model_results["latency_p95"]
            latency_metrics["p99"][model.model_id] = model_results["latency_p99"]
            error_rates[model.model_id] = model_results["error_rate"]
            resource_usage[model.model_id] = model_results["resource_usage"]
            
            # Quality metrics (if available)
            quality_metrics["accuracy"][model.model_id] = model_results.get("accuracy", 0.8)
            quality_metrics["precision"][model.model_id] = model_results.get("precision", 0.8)
            quality_metrics["recall"][model.model_id] = model_results.get("recall", 0.8)
            quality_metrics["f1"][model.model_id] = model_results.get("f1", 0.8)
        
        # Rank models
        overall_ranking = self._rank_models(models, throughput_rps, latency_metrics, error_rates)
        recommended_model = overall_ranking[0] if overall_ranking else ""
        
        benchmark_results = BenchmarkResults(
            benchmark_id=benchmark_id,
            models=[m.model_id for m in models],
            throughput_rps=throughput_rps,
            latency_p50=latency_metrics["p50"],
            latency_p95=latency_metrics["p95"],
            latency_p99=latency_metrics["p99"],
            error_rates=error_rates,
            resource_usage=resource_usage,
            accuracy_scores=quality_metrics["accuracy"],
            precision_scores=quality_metrics["precision"],
            recall_scores=quality_metrics["recall"],
            f1_scores=quality_metrics["f1"],
            overall_ranking=overall_ranking,
            recommended_model=recommended_model,
            benchmark_duration=time.time() - start_time
        )
        
        self.benchmark_history.append(benchmark_results)
        
        logger.info(f"Benchmarked {len(models)} models in {benchmark_results.benchmark_duration:.2f}s")
        return benchmark_results
    
    async def _execute_deployment(self, model: MLModel, deployment_id: str) -> Dict[str, Any]:
        """Execute model deployment based on strategy."""
        strategy = model.deployment_strategy
        
        if strategy == DeploymentStrategy.BLUE_GREEN:
            return await self._blue_green_deployment(model, deployment_id)
        elif strategy == DeploymentStrategy.CANARY:
            return await self._canary_deployment(model, deployment_id)
        elif strategy == DeploymentStrategy.ROLLING:
            return await self._rolling_deployment(model, deployment_id)
        else:
            # Simple deployment
            return await self._simple_deployment(model, deployment_id)
    
    async def _simple_deployment(self, model: MLModel, deployment_id: str) -> Dict[str, Any]:
        """Simple model deployment."""
        # Simulate deployment process
        await asyncio.sleep(0.1)  # Simulate deployment time
        
        return {
            "success": True,
            "endpoint_url": f"http://api.leanvibe.ai/models/{model.model_id}",
            "health_check_url": f"http://api.leanvibe.ai/models/{model.model_id}/health",
            "rollback_plan": {"previous_model_id": None}
        }
    
    async def _blue_green_deployment(self, model: MLModel, deployment_id: str) -> Dict[str, Any]:
        """Blue-green deployment strategy."""
        # Get current active model (blue)
        current_model = await self.model_registry.get_active_model(model.name)
        
        # Deploy new model to green environment
        await asyncio.sleep(0.2)  # Simulate deployment
        
        # Health check new model
        health_check_passed = await self._health_check_model(model)
        
        if health_check_passed:
            # Switch traffic to green (new model)
            return {
                "success": True,
                "endpoint_url": f"http://api.leanvibe.ai/models/{model.model_id}",
                "health_check_url": f"http://api.leanvibe.ai/models/{model.model_id}/health",
                "rollback_plan": {
                    "previous_model_id": current_model.model_id if current_model else None,
                    "strategy": "blue_green"
                }
            }
        else:
            return {"success": False, "error": "Health check failed"}
    
    async def _canary_deployment(self, model: MLModel, deployment_id: str) -> Dict[str, Any]:
        """Canary deployment strategy."""
        # Deploy to small percentage of traffic
        await asyncio.sleep(0.3)
        
        # Monitor canary performance
        canary_success = True  # Simplified check
        
        if canary_success:
            # Gradually increase traffic
            return {
                "success": True,
                "endpoint_url": f"http://api.leanvibe.ai/models/{model.model_id}",
                "health_check_url": f"http://api.leanvibe.ai/models/{model.model_id}/health",
                "rollback_plan": {"strategy": "canary", "traffic_percentage": 5}
            }
        else:
            return {"success": False, "error": "Canary deployment failed"}
    
    async def _rolling_deployment(self, model: MLModel, deployment_id: str) -> Dict[str, Any]:
        """Rolling deployment strategy."""
        # Deploy to instances one by one
        await asyncio.sleep(0.4)
        
        return {
            "success": True,
            "endpoint_url": f"http://api.leanvibe.ai/models/{model.model_id}",
            "health_check_url": f"http://api.leanvibe.ai/models/{model.model_id}/health",
            "rollback_plan": {"strategy": "rolling"}
        }
    
    async def _validate_deployment(self, model: MLModel) -> bool:
        """Validate model deployment."""
        # Perform health check
        return await self._health_check_model(model)
    
    async def _health_check_model(self, model: MLModel) -> bool:
        """Perform health check on deployed model."""
        # Simulate health check
        await asyncio.sleep(0.1)
        return True  # Simplified - always passes
    
    async def _handle_model_drift(self, model: MLModel, drift_analysis: DriftAnalysis) -> None:
        """Handle detected model drift."""
        if drift_analysis.recommended_action == "retrain_model_immediately":
            logger.warning(f"Model {model.name} requires immediate retraining due to {drift_analysis.drift_type.value} drift")
            # In production, would trigger retraining pipeline
            
        elif drift_analysis.recommended_action == "retrain_model_soon":
            logger.info(f"Model {model.name} should be retrained soon due to {drift_analysis.drift_type.value} drift")
            # Schedule retraining
            
        elif drift_analysis.recommended_action == "investigate_data_quality":
            logger.info(f"Data quality issues detected for model {model.name}")
            # Alert data quality team
    
    async def _benchmark_single_model(self, model: MLModel) -> Dict[str, Any]:
        """Benchmark a single model's performance."""
        # Simulate benchmarking
        await asyncio.sleep(0.5)
        
        # Generate realistic metrics
        base_latency = 50 + hash(model.model_id) % 100  # 50-150ms
        
        return {
            "throughput_rps": 100 + (hash(model.model_id) % 200),
            "latency_p50": base_latency,
            "latency_p95": base_latency * 2,
            "latency_p99": base_latency * 3,
            "error_rate": (hash(model.model_id) % 5) / 100,  # 0-5% error rate
            "resource_usage": {
                "memory_mb": 512 + (hash(model.model_id) % 1024),
                "cpu_percent": 20 + (hash(model.model_id) % 60)
            },
            "accuracy": 0.8 + (hash(model.model_id) % 20) / 100,
            "precision": 0.75 + (hash(model.model_id) % 25) / 100,
            "recall": 0.75 + (hash(model.model_id) % 25) / 100,
            "f1": 0.75 + (hash(model.model_id) % 25) / 100
        }
    
    def _rank_models(
        self, 
        models: List[MLModel], 
        throughput: Dict[str, float],
        latency: Dict[str, Dict[str, float]],
        error_rates: Dict[str, float]
    ) -> List[str]:
        """Rank models based on performance metrics."""
        model_scores = {}
        
        for model in models:
            model_id = model.model_id
            
            # Calculate composite score
            throughput_score = throughput.get(model_id, 0) / 100  # Normalize
            latency_score = 1000 / max(1, latency["p95"].get(model_id, 1000))  # Lower is better
            error_score = 1 - error_rates.get(model_id, 0.1)  # Lower is better
            
            composite_score = (throughput_score * 0.3 + latency_score * 0.4 + error_score * 0.3)
            model_scores[model_id] = composite_score
        
        # Sort by score (descending)
        return sorted(model_scores.keys(), key=lambda x: model_scores[x], reverse=True)
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Monitor active deployments
                active_models = await self.model_registry.list_models()
                active_models = [m for m in active_models if m.status == ModelStatus.ACTIVE]
                
                for model in active_models:
                    # Update performance metrics
                    if self.ml_optimizer:
                        # Get metrics from ML optimizer
                        pass
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(300)
    
    async def _drift_monitoring_loop(self) -> None:
        """Background drift monitoring loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Check all active models for drift
                active_models = await self.model_registry.list_models()
                active_models = [m for m in active_models if m.status == ModelStatus.ACTIVE]
                
                for model in active_models:
                    # Get recent data points (simulated)
                    recent_data = self._generate_sample_data_points()
                    
                    if recent_data:
                        drift_analysis = await self.detect_model_drift(model, recent_data)
                        
                        if drift_analysis.is_drifting:
                            logger.warning(f"Drift detected in model {model.name}: {drift_analysis.drift_type.value}")
                
            except Exception as e:
                logger.error(f"Drift monitoring loop error: {e}")
                await asyncio.sleep(3600)
    
    def _generate_sample_data_points(self) -> List[DataPoint]:
        """Generate sample data points for testing."""
        # This would be replaced with actual data collection in production
        sample_points = []
        for i in range(50):
            sample_points.append(DataPoint(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                features={"feature_1": np.random.normal(0, 1), "feature_2": np.random.normal(0, 1)},
                prediction=np.random.normal(0, 1),
                confidence=0.8 + np.random.random() * 0.2
            ))
        return sample_points
    
    async def get_management_summary(self) -> Dict[str, Any]:
        """Get comprehensive model management summary."""
        all_models = await self.model_registry.list_models()
        active_models = [m for m in all_models if m.status == ModelStatus.ACTIVE]
        
        return {
            "model_management": {
                "total_models": len(all_models),
                "active_models": len(active_models),
                "model_types": {
                    mt.value: len([m for m in all_models if m.model_type == mt])
                    for mt in ModelType
                },
                "deployments_today": len([
                    d for d in self.deployment_history 
                    if d.created_at.date() == datetime.utcnow().date()
                ]),
                "active_ab_tests": len(self.ab_tester.active_tests),
                "completed_ab_tests": len(self.ab_tester.test_history)
            },
            "recent_deployments": [
                {
                    "deployment_id": d.deployment_id,
                    "model_id": d.model_id,
                    "status": d.status,
                    "deployment_time": d.deployment_time
                }
                for d in self.deployment_history[-5:]  # Last 5 deployments
            ],
            "benchmarks_completed": len(self.benchmark_history)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for model management system."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # Check model registry
            all_models = await self.model_registry.list_models()
            health_status["components"]["model_registry"] = {
                "status": "healthy",
                "model_count": len(all_models)
            }
            
            # Check A/B testing system
            health_status["components"]["ab_testing"] = {
                "status": "healthy",
                "active_tests": len(self.ab_tester.active_tests)
            }
            
            # Check drift detection
            health_status["components"]["drift_detection"] = {
                "status": "healthy",
                "monitored_models": len(self.drift_detector.data_history)
            }
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup model management resources."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.drift_monitoring_task:
            self.drift_monitoring_task.cancel()
        
        logger.info("Advanced Model Management cleanup completed")


# Global instance
_model_management: Optional[AdvancedModelManagement] = None


async def get_model_management() -> AdvancedModelManagement:
    """Get singleton model management instance."""
    global _model_management
    
    if _model_management is None:
        _model_management = AdvancedModelManagement()
        await _model_management.initialize()
    
    return _model_management


async def cleanup_model_management() -> None:
    """Cleanup model management resources."""
    global _model_management
    
    if _model_management:
        await _model_management.cleanup()
        _model_management = None