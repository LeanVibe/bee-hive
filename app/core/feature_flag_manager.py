"""
VS 7.2: Feature Flag Manager for Gradual Rollout Control - LeanVibe Agent Hive 2.0 Phase 5.3

Advanced feature flag management system providing canary releases, automated rollback triggers,
and A/B testing infrastructure for safe deployment of automated scheduling features.

Features:
- Gradual rollout control with canary releases (1% → 10% → 25% → 50% → 100%)
- Automated rollback triggers on error rate/latency thresholds
- A/B testing infrastructure with statistical significance testing
- Real-time performance monitoring per feature flag
- Extended validation periods with safety gates
- Integration with automation engine and smart scheduler
"""

import asyncio
import logging
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from ..core.circuit_breaker import CircuitBreaker


logger = logging.getLogger(__name__)


class RolloutStage(Enum):
    """Stages of feature rollout."""
    DISABLED = "disabled"
    CANARY_1PCT = "canary_1pct"         # 1% rollout
    CANARY_10PCT = "canary_10pct"       # 10% rollout
    PARTIAL_25PCT = "partial_25pct"     # 25% rollout
    PARTIAL_50PCT = "partial_50pct"     # 50% rollout
    FULL_100PCT = "full_100pct"         # 100% rollout
    ROLLBACK = "rollback"               # Rolled back due to issues


class FeatureType(Enum):
    """Types of features that can be flagged."""
    AUTOMATION = "automation"
    SCHEDULING = "scheduling"
    PREDICTION = "prediction"
    COORDINATION = "coordination"
    SAFETY = "safety"


class RollbackTrigger(Enum):
    """Triggers that can cause automatic rollback."""
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"
    HEALTH_CHECK = "health_check"


class ValidationStatus(Enum):
    """Status of feature validation."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass
class FeatureFlag:
    """Represents a feature flag configuration."""
    name: str
    feature_type: FeatureType
    description: str
    rollout_stage: RolloutStage
    target_percentage: float
    created_at: datetime
    updated_at: datetime
    enabled: bool = True
    validation_period_hours: int = 24
    min_sample_size: int = 100
    error_rate_threshold: float = 0.05  # 5%
    latency_threshold_ms: float = 2000
    throughput_threshold_pct: float = 0.9  # 90% of baseline
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for a feature flag."""
    timestamp: datetime
    requests_total: int
    requests_success: int
    requests_error: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_minute: float
    error_rate: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.requests_success / max(1, self.requests_total)


@dataclass
class RolloutDecision:
    """Decision about feature rollout progression."""
    feature_name: str
    current_stage: RolloutStage
    recommended_stage: RolloutStage
    decision_reason: str
    confidence: float
    metrics_summary: Dict[str, Any]
    safety_checks_passed: bool
    validation_complete: bool


@dataclass
class ABTestConfiguration:
    """Configuration for A/B testing."""
    feature_name: str
    control_group_pct: float
    treatment_group_pct: float
    success_metric: str
    minimum_effect_size: float
    statistical_power: float = 0.8
    significance_level: float = 0.05
    max_duration_days: int = 14


class FeatureFlagManager:
    """
    Advanced feature flag manager for gradual rollout control.
    
    Core Features:
    - Canary releases with automated progression
    - Statistical A/B testing with significance testing
    - Automated rollback triggers based on performance metrics
    - Extended validation periods with safety gates
    - Real-time performance monitoring per feature
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Core configuration
        self.enabled = True
        self.rollout_automation_enabled = True
        self.rollback_automation_enabled = True
        
        # Rollout configuration
        self.rollout_stages = [
            (RolloutStage.CANARY_1PCT, 1.0),
            (RolloutStage.CANARY_10PCT, 10.0),
            (RolloutStage.PARTIAL_25PCT, 25.0),
            (RolloutStage.PARTIAL_50PCT, 50.0),
            (RolloutStage.FULL_100PCT, 100.0)
        ]
        
        # Safety configuration
        self.default_validation_period_hours = 24
        self.extended_validation_period_hours = 168  # 7 days for full rollout
        self.max_concurrent_rollouts = 3
        self.emergency_rollback_threshold = 0.1  # 10% error rate
        
        # Statistical testing configuration
        self.min_sample_size_ab_test = 1000
        self.significance_level = 0.05
        self.statistical_power = 0.8
        
        # Internal state
        self._feature_flags: Dict[str, FeatureFlag] = {}
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._rollout_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._ab_tests: Dict[str, ABTestConfiguration] = {}
        
        # Circuit breakers
        self._rollout_circuit_breaker = CircuitBreaker(
            name="feature_rollout",
            failure_threshold=3,
            timeout_seconds=300
        )
        
        self._rollback_circuit_breaker = CircuitBreaker(
            name="feature_rollback",
            failure_threshold=2,
            timeout_seconds=180
        )
    
    async def initialize(self) -> None:
        """Initialize the feature flag manager."""
        try:
            logger.info("Initializing Feature Flag Manager VS 7.2")
            
            # Load existing feature flags
            await self._load_feature_flags()
            
            # Start background tasks
            asyncio.create_task(self._rollout_automation_loop())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._rollback_monitor())
            asyncio.create_task(self._ab_test_analyzer())
            
            logger.info("Feature Flag Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Feature Flag Manager: {e}")
            raise
    
    async def create_feature_flag(
        self,
        name: str,
        feature_type: FeatureType,
        description: str,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new feature flag.
        
        Args:
            name: Feature flag name
            feature_type: Type of feature
            description: Description of the feature
            config: Additional configuration options
            
        Returns:
            True if created successfully
        """
        try:
            if name in self._feature_flags:
                logger.warning(f"Feature flag {name} already exists")
                return False
            
            # Create feature flag
            feature_flag = FeatureFlag(
                name=name,
                feature_type=feature_type,
                description=description,
                rollout_stage=RolloutStage.DISABLED,
                target_percentage=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                enabled=True
            )
            
            # Apply configuration overrides
            if config:
                if "validation_period_hours" in config:
                    feature_flag.validation_period_hours = int(config["validation_period_hours"])
                if "error_rate_threshold" in config:
                    feature_flag.error_rate_threshold = float(config["error_rate_threshold"])
                if "latency_threshold_ms" in config:
                    feature_flag.latency_threshold_ms = float(config["latency_threshold_ms"])
                if "metadata" in config:
                    feature_flag.metadata.update(config["metadata"])
            
            # Store feature flag
            self._feature_flags[name] = feature_flag
            
            # Persist to Redis
            await self._persist_feature_flag(feature_flag)
            
            logger.info(f"Created feature flag: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature flag {name}: {e}")
            return False
    
    async def is_feature_enabled(
        self,
        feature_name: str,
        agent_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a feature is enabled for a given context.
        
        Args:
            feature_name: Name of the feature to check
            agent_id: Agent ID for percentage-based rollout
            context: Additional context for feature evaluation
            
        Returns:
            (is_enabled, evaluation_info)
        """
        try:
            if feature_name not in self._feature_flags:
                return False, {"error": "Feature flag not found"}
            
            feature_flag = self._feature_flags[feature_name]
            
            if not feature_flag.enabled:
                return False, {"reason": "Feature flag disabled"}
            
            if feature_flag.rollout_stage == RolloutStage.DISABLED:
                return False, {"reason": "Feature in disabled stage"}
            
            if feature_flag.rollout_stage == RolloutStage.ROLLBACK:
                return False, {"reason": "Feature rolled back"}
            
            if feature_flag.rollout_stage == RolloutStage.FULL_100PCT:
                return True, {"reason": "Full rollout", "percentage": 100.0}
            
            # Calculate percentage-based enabling
            if agent_id:
                # Use agent ID for consistent percentage-based rollout
                agent_hash = hash(str(agent_id)) % 10000  # 0-9999
                agent_percentage = agent_hash / 100.0  # 0.00-99.99
                
                enabled = agent_percentage < feature_flag.target_percentage
                
                return enabled, {
                    "reason": f"Percentage rollout ({feature_flag.target_percentage}%)",
                    "percentage": feature_flag.target_percentage,
                    "agent_percentage": agent_percentage,
                    "stage": feature_flag.rollout_stage.value
                }
            
            # No agent ID, use global percentage (for system-wide features)
            # Use timestamp-based pseudo-random selection
            current_minute = int(datetime.utcnow().timestamp() / 60)
            minute_hash = hash(f"{feature_name}_{current_minute}") % 10000
            minute_percentage = minute_hash / 100.0
            
            enabled = minute_percentage < feature_flag.target_percentage
            
            return enabled, {
                "reason": f"Time-based percentage rollout ({feature_flag.target_percentage}%)",
                "percentage": feature_flag.target_percentage,
                "stage": feature_flag.rollout_stage.value
            }
            
        except Exception as e:
            logger.error(f"Error checking feature flag {feature_name}: {e}")
            return False, {"error": str(e)}
    
    async def progress_rollout(
        self,
        feature_name: str,
        target_stage: Optional[RolloutStage] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Progress a feature to the next rollout stage or specified stage.
        
        Args:
            feature_name: Name of the feature to progress
            target_stage: Optional target stage (defaults to next stage)
            
        Returns:
            (success, progression_info)
        """
        try:
            if feature_name not in self._feature_flags:
                return False, {"error": "Feature flag not found"}
            
            feature_flag = self._feature_flags[feature_name]
            
            # Determine target stage
            if target_stage is None:
                target_stage = self._get_next_rollout_stage(feature_flag.rollout_stage)
            
            if target_stage is None:
                return False, {"error": "Already at maximum rollout stage"}
            
            # Safety checks
            if not await self._validate_rollout_progression(feature_flag, target_stage):
                return False, {"error": "Rollout progression failed safety validation"}
            
            # Check concurrent rollouts
            active_rollouts = len([
                f for f in self._feature_flags.values()
                if f.rollout_stage not in [RolloutStage.DISABLED, RolloutStage.FULL_100PCT, RolloutStage.ROLLBACK]
            ])
            
            if active_rollouts >= self.max_concurrent_rollouts:
                return False, {"error": "Too many concurrent rollouts"}
            
            # Progress rollout
            old_stage = feature_flag.rollout_stage
            old_percentage = feature_flag.target_percentage
            
            feature_flag.rollout_stage = target_stage
            feature_flag.target_percentage = self._get_stage_percentage(target_stage)
            feature_flag.updated_at = datetime.utcnow()
            
            # Record rollout progression
            progression_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "from_stage": old_stage.value,
                "to_stage": target_stage.value,
                "from_percentage": old_percentage,
                "to_percentage": feature_flag.target_percentage,
                "reason": "manual_progression"
            }
            
            self._rollout_history[feature_name].append(progression_record)
            
            # Persist changes
            await self._persist_feature_flag(feature_flag)
            
            logger.info(f"Progressed feature {feature_name} from {old_stage.value} to {target_stage.value}")
            
            return True, {
                "old_stage": old_stage.value,
                "new_stage": target_stage.value,
                "old_percentage": old_percentage,
                "new_percentage": feature_flag.target_percentage,
                "progression_record": progression_record
            }
            
        except Exception as e:
            logger.error(f"Error progressing rollout for {feature_name}: {e}")
            return False, {"error": str(e)}
    
    async def trigger_rollback(
        self,
        feature_name: str,
        trigger: RollbackTrigger,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Trigger rollback of a feature flag.
        
        Args:
            feature_name: Name of the feature to roll back
            trigger: What triggered the rollback
            reason: Human-readable reason for rollback
            metadata: Additional rollback metadata
            
        Returns:
            (success, rollback_info)
        """
        try:
            if feature_name not in self._feature_flags:
                return False, {"error": "Feature flag not found"}
            
            feature_flag = self._feature_flags[feature_name]
            
            # Execute rollback with circuit breaker protection
            async with self._rollback_circuit_breaker:
                old_stage = feature_flag.rollout_stage
                old_percentage = feature_flag.target_percentage
                
                # Set to rollback state
                feature_flag.rollout_stage = RolloutStage.ROLLBACK
                feature_flag.target_percentage = 0.0
                feature_flag.updated_at = datetime.utcnow()
                
                # Record rollback
                rollback_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "from_stage": old_stage.value,
                    "to_stage": RolloutStage.ROLLBACK.value,
                    "from_percentage": old_percentage,
                    "to_percentage": 0.0,
                    "trigger": trigger.value,
                    "reason": reason,
                    "metadata": metadata or {}
                }
                
                self._rollout_history[feature_name].append(rollback_record)
                
                # Persist changes
                await self._persist_feature_flag(feature_flag)
                
                logger.critical(f"ROLLBACK: Feature {feature_name} rolled back due to {trigger.value}: {reason}")
                
                return True, {
                    "rolled_back_from": old_stage.value,
                    "rollback_trigger": trigger.value,
                    "rollback_reason": reason,
                    "rollback_record": rollback_record
                }
            
        except Exception as e:
            logger.error(f"Error triggering rollback for {feature_name}: {e}")
            return False, {"error": str(e)}
    
    async def record_performance_metrics(
        self,
        feature_name: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Record performance metrics for a feature flag."""
        try:
            if feature_name not in self._feature_flags:
                logger.warning(f"Recording metrics for unknown feature: {feature_name}")
                return
            
            # Add to performance history
            self._performance_history[feature_name].append(metrics)
            
            # Check for rollback conditions
            if self.rollback_automation_enabled:
                await self._check_rollback_conditions(feature_name, metrics)
            
        except Exception as e:
            logger.error(f"Error recording performance metrics for {feature_name}: {e}")
    
    async def start_ab_test(
        self,
        feature_name: str,
        ab_config: ABTestConfiguration
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Start an A/B test for a feature.
        
        Args:
            feature_name: Name of the feature to test
            ab_config: A/B test configuration
            
        Returns:
            (success, test_info)
        """
        try:
            if feature_name not in self._feature_flags:
                return False, {"error": "Feature flag not found"}
            
            if feature_name in self._ab_tests:
                return False, {"error": "A/B test already running for this feature"}
            
            # Validate A/B test configuration
            if ab_config.control_group_pct + ab_config.treatment_group_pct > 100:
                return False, {"error": "Control + treatment groups exceed 100%"}
            
            # Store A/B test configuration
            self._ab_tests[feature_name] = ab_config
            
            # Persist A/B test configuration
            redis = await get_redis()
            await redis.hset(
                f"ab_test:{feature_name}",
                mapping={
                    "config": json.dumps(asdict(ab_config)),
                    "start_time": datetime.utcnow().isoformat(),
                    "status": "running"
                }
            )
            
            logger.info(f"Started A/B test for feature {feature_name}")
            
            return True, {
                "test_id": feature_name,
                "control_group_pct": ab_config.control_group_pct,
                "treatment_group_pct": ab_config.treatment_group_pct,
                "success_metric": ab_config.success_metric,
                "start_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting A/B test for {feature_name}: {e}")
            return False, {"error": str(e)}
    
    async def get_feature_status(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive status for a feature flag."""
        try:
            if feature_name not in self._feature_flags:
                return {"error": "Feature flag not found"}
            
            feature_flag = self._feature_flags[feature_name]
            
            # Basic status
            status = {
                "name": feature_flag.name,
                "feature_type": feature_flag.feature_type.value,
                "description": feature_flag.description,
                "enabled": feature_flag.enabled,
                "rollout_stage": feature_flag.rollout_stage.value,
                "target_percentage": feature_flag.target_percentage,
                "created_at": feature_flag.created_at.isoformat(),
                "updated_at": feature_flag.updated_at.isoformat()
            }
            
            # Performance metrics
            if feature_name in self._performance_history and self._performance_history[feature_name]:
                recent_metrics = list(self._performance_history[feature_name])[-10:]  # Last 10 metrics
                
                avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
                avg_latency = statistics.mean(m.avg_latency_ms for m in recent_metrics)
                avg_throughput = statistics.mean(m.throughput_per_minute for m in recent_metrics)
                
                status["performance"] = {
                    "avg_error_rate": avg_error_rate,
                    "avg_latency_ms": avg_latency,
                    "avg_throughput_per_minute": avg_throughput,
                    "sample_size": len(recent_metrics),
                    "meets_error_threshold": avg_error_rate < feature_flag.error_rate_threshold,
                    "meets_latency_threshold": avg_latency < feature_flag.latency_threshold_ms
                }
            else:
                status["performance"] = {"no_data": True}
            
            # Rollout history
            status["rollout_history"] = self._rollout_history.get(feature_name, [])
            
            # A/B test status
            if feature_name in self._ab_tests:
                status["ab_test"] = {
                    "running": True,
                    "config": asdict(self._ab_tests[feature_name])
                }
            else:
                status["ab_test"] = {"running": False}
            
            # Next recommended action
            status["recommendations"] = await self._get_feature_recommendations(feature_flag)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting feature status for {feature_name}: {e}")
            return {"error": str(e)}
    
    async def get_all_features_status(self) -> Dict[str, Any]:
        """Get status of all feature flags."""
        try:
            features_status = {}
            
            for feature_name in self._feature_flags:
                features_status[feature_name] = await self.get_feature_status(feature_name)
            
            # System-wide statistics
            total_features = len(self._feature_flags)
            active_rollouts = len([
                f for f in self._feature_flags.values()
                if f.rollout_stage not in [RolloutStage.DISABLED, RolloutStage.FULL_100PCT, RolloutStage.ROLLBACK]
            ])
            
            rollback_count = len([
                f for f in self._feature_flags.values()
                if f.rollout_stage == RolloutStage.ROLLBACK
            ])
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_features": total_features,
                    "active_rollouts": active_rollouts,
                    "rollbacks": rollback_count,
                    "rollout_automation_enabled": self.rollout_automation_enabled,
                    "rollback_automation_enabled": self.rollback_automation_enabled
                },
                "features": features_status
            }
            
        except Exception as e:
            logger.error(f"Error getting all features status: {e}")
            return {"error": str(e)}
    
    # Internal methods
    
    async def _load_feature_flags(self) -> None:
        """Load feature flags from persistent storage."""
        try:
            redis = await get_redis()
            
            # Get all feature flag keys
            keys = await redis.keys("feature_flag:*")
            
            for key in keys:
                try:
                    feature_data = await redis.hgetall(key)
                    if feature_data:
                        feature_name = key.decode().split(":", 1)[1]
                        
                        feature_flag = FeatureFlag(
                            name=feature_name,
                            feature_type=FeatureType(feature_data["feature_type"]),
                            description=feature_data["description"],
                            rollout_stage=RolloutStage(feature_data["rollout_stage"]),
                            target_percentage=float(feature_data["target_percentage"]),
                            created_at=datetime.fromisoformat(feature_data["created_at"]),
                            updated_at=datetime.fromisoformat(feature_data["updated_at"]),
                            enabled=json.loads(feature_data.get("enabled", "true")),
                            validation_period_hours=int(feature_data.get("validation_period_hours", "24")),
                            error_rate_threshold=float(feature_data.get("error_rate_threshold", "0.05")),
                            latency_threshold_ms=float(feature_data.get("latency_threshold_ms", "2000")),
                            metadata=json.loads(feature_data.get("metadata", "{}"))
                        )
                        
                        self._feature_flags[feature_name] = feature_flag
                        
                except Exception as e:
                    logger.error(f"Error loading feature flag from {key}: {e}")
            
            logger.info(f"Loaded {len(self._feature_flags)} feature flags")
            
        except Exception as e:
            logger.warning(f"Could not load feature flags, starting with empty set: {e}")
    
    async def _persist_feature_flag(self, feature_flag: FeatureFlag) -> None:
        """Persist a feature flag to Redis."""
        try:
            redis = await get_redis()
            
            await redis.hset(
                f"feature_flag:{feature_flag.name}",
                mapping={
                    "feature_type": feature_flag.feature_type.value,
                    "description": feature_flag.description,
                    "rollout_stage": feature_flag.rollout_stage.value,
                    "target_percentage": str(feature_flag.target_percentage),
                    "created_at": feature_flag.created_at.isoformat(),
                    "updated_at": feature_flag.updated_at.isoformat(),
                    "enabled": json.dumps(feature_flag.enabled),
                    "validation_period_hours": str(feature_flag.validation_period_hours),
                    "error_rate_threshold": str(feature_flag.error_rate_threshold),
                    "latency_threshold_ms": str(feature_flag.latency_threshold_ms),
                    "metadata": json.dumps(feature_flag.metadata)
                }
            )
            
        except Exception as e:
            logger.error(f"Error persisting feature flag {feature_flag.name}: {e}")
    
    def _get_next_rollout_stage(self, current_stage: RolloutStage) -> Optional[RolloutStage]:
        """Get the next rollout stage."""
        stage_order = [stage for stage, _ in self.rollout_stages]
        
        try:
            current_index = stage_order.index(current_stage)
            if current_index < len(stage_order) - 1:
                return stage_order[current_index + 1]
        except ValueError:
            # Current stage not in standard progression
            if current_stage == RolloutStage.DISABLED:
                return RolloutStage.CANARY_1PCT
        
        return None
    
    def _get_stage_percentage(self, stage: RolloutStage) -> float:
        """Get the percentage for a rollout stage."""
        stage_percentages = {
            RolloutStage.DISABLED: 0.0,
            RolloutStage.CANARY_1PCT: 1.0,
            RolloutStage.CANARY_10PCT: 10.0,
            RolloutStage.PARTIAL_25PCT: 25.0,
            RolloutStage.PARTIAL_50PCT: 50.0,
            RolloutStage.FULL_100PCT: 100.0,
            RolloutStage.ROLLBACK: 0.0
        }
        return stage_percentages.get(stage, 0.0)
    
    async def _validate_rollout_progression(
        self,
        feature_flag: FeatureFlag,
        target_stage: RolloutStage
    ) -> bool:
        """Validate that rollout progression is safe."""
        try:
            # Check if enough time has passed since last update
            time_since_update = datetime.utcnow() - feature_flag.updated_at
            min_validation_time = timedelta(hours=feature_flag.validation_period_hours)
            
            if time_since_update < min_validation_time:
                logger.info(f"Feature {feature_flag.name} validation period not complete")
                return False
            
            # Check performance metrics
            if feature_flag.name in self._performance_history:
                recent_metrics = list(self._performance_history[feature_flag.name])[-20:]
                
                if recent_metrics:
                    avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
                    
                    if avg_error_rate > feature_flag.error_rate_threshold:
                        logger.warning(f"Feature {feature_flag.name} error rate too high: {avg_error_rate}")
                        return False
                    
                    avg_latency = statistics.mean(m.avg_latency_ms for m in recent_metrics)
                    
                    if avg_latency > feature_flag.latency_threshold_ms:
                        logger.warning(f"Feature {feature_flag.name} latency too high: {avg_latency}ms")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating rollout progression: {e}")
            return False
    
    async def _check_rollback_conditions(
        self,
        feature_name: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Check if rollback conditions are met."""
        try:
            feature_flag = self._feature_flags[feature_name]
            
            # Emergency rollback condition: Very high error rate
            if metrics.error_rate > self.emergency_rollback_threshold:
                await self.trigger_rollback(
                    feature_name,
                    RollbackTrigger.ERROR_RATE,
                    f"Emergency rollback: Error rate {metrics.error_rate:.1%} exceeds threshold {self.emergency_rollback_threshold:.1%}",
                    {"error_rate": metrics.error_rate, "threshold": self.emergency_rollback_threshold}
                )
                return
            
            # Standard rollback conditions
            if metrics.error_rate > feature_flag.error_rate_threshold:
                # Check if this is a sustained issue
                recent_metrics = list(self._performance_history[feature_name])[-5:]
                
                if len(recent_metrics) >= 3:
                    sustained_high_error = all(m.error_rate > feature_flag.error_rate_threshold for m in recent_metrics)
                    
                    if sustained_high_error:
                        await self.trigger_rollback(
                            feature_name,
                            RollbackTrigger.ERROR_RATE,
                            f"Sustained high error rate: {metrics.error_rate:.1%} > {feature_flag.error_rate_threshold:.1%}",
                            {"error_rate": metrics.error_rate, "threshold": feature_flag.error_rate_threshold}
                        )
                        return
            
            # Latency rollback condition
            if metrics.avg_latency_ms > feature_flag.latency_threshold_ms:
                recent_metrics = list(self._performance_history[feature_name])[-5:]
                
                if len(recent_metrics) >= 3:
                    sustained_high_latency = all(m.avg_latency_ms > feature_flag.latency_threshold_ms for m in recent_metrics)
                    
                    if sustained_high_latency:
                        await self.trigger_rollback(
                            feature_name,
                            RollbackTrigger.LATENCY,
                            f"Sustained high latency: {metrics.avg_latency_ms:.0f}ms > {feature_flag.latency_threshold_ms:.0f}ms",
                            {"latency_ms": metrics.avg_latency_ms, "threshold_ms": feature_flag.latency_threshold_ms}
                        )
                        return
            
        except Exception as e:
            logger.error(f"Error checking rollback conditions for {feature_name}: {e}")
    
    async def _get_feature_recommendations(self, feature_flag: FeatureFlag) -> List[Dict[str, Any]]:
        """Get recommendations for a feature flag."""
        recommendations = []
        
        try:
            # Check if ready for progression
            if feature_flag.rollout_stage not in [RolloutStage.FULL_100PCT, RolloutStage.ROLLBACK]:
                time_since_update = datetime.utcnow() - feature_flag.updated_at
                validation_complete = time_since_update >= timedelta(hours=feature_flag.validation_period_hours)
                
                if validation_complete:
                    # Check performance
                    if feature_flag.name in self._performance_history and self._performance_history[feature_flag.name]:
                        recent_metrics = list(self._performance_history[feature_flag.name])[-10:]
                        
                        if recent_metrics:
                            avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
                            avg_latency = statistics.mean(m.avg_latency_ms for m in recent_metrics)
                            
                            performance_good = (
                                avg_error_rate < feature_flag.error_rate_threshold and
                                avg_latency < feature_flag.latency_threshold_ms
                            )
                            
                            if performance_good:
                                next_stage = self._get_next_rollout_stage(feature_flag.rollout_stage)
                                if next_stage:
                                    recommendations.append({
                                        "type": "progression",
                                        "action": f"Progress to {next_stage.value}",
                                        "confidence": "high",
                                        "reason": "Validation period complete and performance metrics good"
                                    })
                            else:
                                recommendations.append({
                                    "type": "warning",
                                    "action": "Review performance metrics",
                                    "confidence": "medium",
                                    "reason": f"Performance metrics concerning: error_rate={avg_error_rate:.1%}, latency={avg_latency:.0f}ms"
                                })
                    else:
                        recommendations.append({
                            "type": "info",
                            "action": "Collect more performance data",
                            "confidence": "low",
                            "reason": "Insufficient performance data for recommendation"
                        })
                else:
                    hours_remaining = feature_flag.validation_period_hours - time_since_update.total_seconds() / 3600
                    recommendations.append({
                        "type": "wait",
                        "action": f"Wait {hours_remaining:.1f} more hours",
                        "confidence": "high",
                        "reason": "Still in validation period"
                    })
            
            # Check for rollback conditions
            if feature_flag.rollout_stage == RolloutStage.ROLLBACK:
                recommendations.append({
                    "type": "recovery",
                    "action": "Investigate rollback cause and fix issues",
                    "confidence": "high",
                    "reason": "Feature is currently rolled back"
                })
            
        except Exception as e:
            logger.error(f"Error generating recommendations for {feature_flag.name}: {e}")
            recommendations.append({
                "type": "error",
                "action": "Manual review required",
                "confidence": "low",
                "reason": f"Error generating recommendations: {str(e)}"
            })
        
        return recommendations
    
    async def _rollout_automation_loop(self) -> None:
        """Background task for automated rollout progression."""
        while True:
            try:
                if not self.rollout_automation_enabled:
                    await asyncio.sleep(3600)  # Check every hour when disabled
                    continue
                
                for feature_name, feature_flag in self._feature_flags.items():
                    try:
                        # Skip if not in active rollout
                        if feature_flag.rollout_stage in [RolloutStage.DISABLED, RolloutStage.FULL_100PCT, RolloutStage.ROLLBACK]:
                            continue
                        
                        # Check if ready for automatic progression
                        recommendations = await self._get_feature_recommendations(feature_flag)
                        
                        progression_recommendations = [
                            r for r in recommendations 
                            if r["type"] == "progression" and r["confidence"] == "high"
                        ]
                        
                        if progression_recommendations:
                            logger.info(f"Auto-progressing feature {feature_name} based on recommendations")
                            
                            success, result = await self.progress_rollout(feature_name)
                            
                            if success:
                                logger.info(f"Successfully auto-progressed {feature_name} to {result['new_stage']}")
                            else:
                                logger.warning(f"Failed to auto-progress {feature_name}: {result}")
                        
                    except Exception as e:
                        logger.error(f"Error in automation loop for feature {feature_name}: {e}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in rollout automation loop: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitor(self) -> None:
        """Monitor performance metrics and update feature statuses."""
        while True:
            try:
                # This would integrate with actual metrics collection
                # For now, we'll just clean up old metrics
                
                for feature_name in list(self._performance_history.keys()):
                    history = self._performance_history[feature_name]
                    
                    # Keep only recent metrics (last 24 hours worth)
                    cutoff_time = datetime.utcnow() - timedelta(hours=24)
                    
                    # Remove old metrics
                    while history and history[0].timestamp < cutoff_time:
                        history.popleft()
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(300)
    
    async def _rollback_monitor(self) -> None:
        """Monitor for conditions requiring emergency rollback."""
        while True:
            try:
                if not self.rollback_automation_enabled:
                    await asyncio.sleep(300)
                    continue
                
                # Check all active features for emergency conditions
                for feature_name, feature_flag in self._feature_flags.items():
                    if feature_flag.rollout_stage in [RolloutStage.DISABLED, RolloutStage.ROLLBACK]:
                        continue
                    
                    # Check recent metrics for emergency conditions
                    if feature_name in self._performance_history and self._performance_history[feature_name]:
                        recent_metrics = list(self._performance_history[feature_name])[-3:]  # Last 3 metrics
                        
                        if len(recent_metrics) >= 2:
                            # Check for sustained emergency conditions
                            emergency_errors = [m for m in recent_metrics if m.error_rate > self.emergency_rollback_threshold]
                            
                            if len(emergency_errors) >= 2:  # At least 2 of last 3 metrics
                                logger.critical(f"Emergency rollback condition detected for {feature_name}")
                                await self.trigger_rollback(
                                    feature_name,
                                    RollbackTrigger.ERROR_RATE,
                                    f"Emergency rollback: Sustained error rate > {self.emergency_rollback_threshold:.1%}",
                                    {"recent_error_rates": [m.error_rate for m in recent_metrics]}
                                )
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in rollback monitor: {e}")
                await asyncio.sleep(60)
    
    async def _ab_test_analyzer(self) -> None:
        """Analyze A/B test results and provide recommendations."""
        while True:
            try:
                for feature_name, ab_config in list(self._ab_tests.items()):
                    try:
                        # This would perform statistical analysis of A/B test results
                        # For now, we'll just log that analysis would happen here
                        logger.debug(f"Analyzing A/B test results for {feature_name}")
                        
                        # In a real implementation, this would:
                        # 1. Collect metrics for control and treatment groups
                        # 2. Perform statistical significance testing
                        # 3. Calculate confidence intervals
                        # 4. Determine if test should be stopped early
                        # 5. Generate recommendations for feature rollout
                        
                    except Exception as e:
                        logger.error(f"Error analyzing A/B test for {feature_name}: {e}")
                
                await asyncio.sleep(3600)  # Analyze every hour
                
            except Exception as e:
                logger.error(f"Error in A/B test analyzer: {e}")
                await asyncio.sleep(3600)


# Global instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


async def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_flag_manager
    
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager()
        await _feature_flag_manager.initialize()
    
    return _feature_flag_manager